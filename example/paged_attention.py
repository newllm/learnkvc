"""PagedAttention: serve one KV cache from a pool of fixed-size blocks.

``kv_cache.py`` appends K/V to one contiguous tensor per layer.  Real serving
engines cannot do that: requests arrive with unknown lengths, finish at
different times, and their KV cache would fragment the GPU memory.  vLLM's
PagedAttention instead keeps a pool of fixed-size blocks and gives every
sequence a **block table** that maps logical token positions to physical
blocks.  This file builds that mapping in miniature and then checks the
attention output against a dense, contiguous reference.

The three pieces:

* ``BlockPool``   -- ``num_blocks`` blocks of ``block_size`` tokens; each block
                     holds K/V for ``block_size`` positions.
* ``Sequence``    -- ``block_table`` (physical block ids) + number of tokens.
* ``PagedKVCache``-- appends K/V into the block table and gathers them back in
                     logical order for the attention kernel.

Run ``python3 example/paged_attention.py`` (requires PyTorch).  The demo
interleaves two sequences, frees one, reuses its blocks for a third, and
asserts that paged attention equals the dense reference at every step.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

BLOCK_SIZE = 4
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
NUM_BLOCKS = 8
SEED = 7
SCALE = 0.5
GENERATOR = torch.Generator().manual_seed(SEED)


def _select_device() -> str:
    requested = os.environ.get("PAGED_DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _select_device()


# --------------------------------------------------------------------------- #
# 1. Physical memory: a pool of KV blocks
# --------------------------------------------------------------------------- #


class BlockPool:
    """Fixed-size KV blocks, as on a GPU: ``(blocks, block_size, kv_heads, dim)``.

    Allocation is just popping from a free list, so physical block ids come back
    in arbitrary order -- exactly what makes a block table necessary.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = BLOCK_SIZE,
        kv_heads: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        # One tensor per layer in a real engine; one layer here keeps the math
        # readable.  Both are allocated on the GPU when DEVICE is cuda.
        self.k = torch.zeros(num_blocks, block_size, kv_heads, head_dim, device=DEVICE)
        self.v = torch.zeros(num_blocks, block_size, kv_heads, head_dim, device=DEVICE)
        self.free_blocks = list(range(num_blocks))
        # Shuffle so the demo's block tables are deliberately non-contiguous.
        random.Random(SEED).shuffle(self.free_blocks)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("KV block pool is full (production would preempt)")
        return self.free_blocks.pop()

    def release(self, block_id: int) -> None:
        self.free_blocks.append(block_id)

    def write(self, block_id: int, slot: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write one token's K/V into ``slot`` of a physical block."""
        # copy_ is a device copy kernel; on GPU this is a small H2D/D2D memcpy.
        self.k[block_id, slot].copy_(key)
        self.v[block_id, slot].copy_(value)


# --------------------------------------------------------------------------- #
# 2. Logical memory: per-sequence block table
# --------------------------------------------------------------------------- #


@dataclass
class Sequence:
    seq_id: str
    block_table: list[int] = field(default_factory=list)
    length: int = 0

    @property
    def internal_waste(self) -> int:
        """Slots allocated but not yet used (block granularity waste)."""
        return len(self.block_table) * BLOCK_SIZE - self.length


class PagedKVCache:
    """Writes K/V into blocks and gathers them back in logical order."""

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool

    def append(self, sequence: Sequence, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("K/V must be (tokens, kv_heads, head_dim)")
        if key.size(1) != self.pool.kv_heads or key.size(2) != self.pool.head_dim:
            raise ValueError("K/V shape does not match the pool")
        key = key.to(DEVICE)
        value = value.to(DEVICE)
        for token in range(key.size(0)):
            slot = sequence.length % self.pool.block_size
            if slot == 0:  # a new logical block starts here
                sequence.block_table.append(self.pool.allocate())
            block_id = sequence.block_table[-1]
            self.pool.write(block_id, slot, key[token], value[token])
            sequence.length += 1

    def gather(self, sequence: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect the blocks in table order, like a paged attention kernel."""
        if sequence.length == 0:
            raise ValueError("sequence has no cached tokens")
        blocks = torch.tensor(
            sequence.block_table, dtype=torch.long, device=self.pool.k.device
        )
        # (blocks, block_size, kv_heads, head_dim) -> (tokens, kv_heads, head_dim).
        # Advanced indexing is a gather kernel on the GPU; the block table lives
        # on the device in a real engine (rebuilt per call only for clarity here).
        key = self.pool.k[blocks].reshape(-1, self.pool.kv_heads, self.pool.head_dim)
        value = self.pool.v[blocks].reshape(-1, self.pool.kv_heads, self.pool.head_dim)
        return key[: sequence.length], value[: sequence.length]

    def free(self, sequence: Sequence) -> None:
        for block_id in sequence.block_table:
            self.pool.release(block_id)
        sequence.block_table.clear()
        sequence.length = 0


# --------------------------------------------------------------------------- #
# 3. Attention over a gathered prefix
# --------------------------------------------------------------------------- #


def _expand_kv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.size(-1) != key.size(-1) or query.size(1) % key.size(1):
        raise ValueError("query heads must be a multiple of KV heads")
    repeat = query.size(1) // key.size(1)
    return key.repeat_interleave(repeat, dim=1), value.repeat_interleave(repeat, dim=1)


def _attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Causal attention for the last ``query.size(0)`` positions of ``key``.

    ``query`` is ``(q_len, q_heads, head_dim)``; ``key``/``value`` are the whole
    prefix ``(k_len, kv_heads, head_dim)``.  Same math as ``kv_cache.py``'s
    ``cached_attention``, but the prefix arrives through a block table.
    """
    key, value = _expand_kv(query, key, value)
    # (1, heads, tokens, dim) for batched GEMM / softmax.
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)

    q_len, k_len = q.size(-2), k.size(-2)
    # Batched GEMM -> cuBLAS on CUDA; softmax -> ATen CUDA kernel.
    scores = q @ k.transpose(-1, -2) / math.sqrt(query.size(-1))
    # Query i sits at absolute position k_len - q_len + i.
    causal = torch.arange(k_len, device=query.device)[None, :] <= (
        k_len - q_len + torch.arange(q_len, device=query.device)
    )[:, None]
    scores = scores.masked_fill(~causal, float("-inf"))
    out = F.softmax(scores, dim=-1) @ v
    return out.squeeze(0).transpose(0, 1)


def paged_attention(
    query: torch.Tensor, sequence: Sequence, cache: PagedKVCache
) -> torch.Tensor:
    key, value = cache.gather(sequence)
    return _attention(query, key, value)


def dense_attention(
    query: torch.Tensor, history: dict[str, list[torch.Tensor]]
) -> torch.Tensor:
    key = torch.cat(history["k"], dim=0)
    value = torch.cat(history["v"], dim=0)
    return _attention(query, key, value)


# --------------------------------------------------------------------------- #
# 4. Demo helpers
# --------------------------------------------------------------------------- #


def random_kv(tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (tokens, KV_HEADS, HEAD_DIM)
    key = (torch.randn(*shape, generator=GENERATOR) * SCALE).to(DEVICE)
    value = (torch.randn(*shape, generator=GENERATOR) * SCALE).to(DEVICE)
    return key, value


def random_query(tokens: int) -> torch.Tensor:
    shape = (tokens, Q_HEADS, HEAD_DIM)
    return (torch.randn(*shape, generator=GENERATOR) * SCALE).to(DEVICE)


def feed(
    cache: PagedKVCache,
    sequence: Sequence,
    key: torch.Tensor,
    value: torch.Tensor,
    history: dict[str, list[torch.Tensor]],
) -> torch.Tensor:
    """Append K/V, run paged attention, and compare it with the dense prefix."""
    cache.append(sequence, key, value)
    history["k"].append(key)
    history["v"].append(value)

    query = random_query(key.size(0))
    paged = paged_attention(query, sequence, cache)
    dense = dense_attention(query, history)
    assert torch.allclose(paged, dense, atol=0, rtol=0), "paged attention diverged"
    return paged


def print_block_table(sequence: Sequence, pool: BlockPool) -> None:
    print(f"  {sequence.seq_id}: 长度 {sequence.length}，逻辑块 → 物理块")
    for logical, physical in enumerate(sequence.block_table):
        start = logical * pool.block_size
        end = min(start + pool.block_size, sequence.length)
        used = end - start
        suffix = f"（{used}/{pool.block_size}）" if used < pool.block_size else ""
        print(f"    逻辑块 {logical}: token {start}-{end - 1} → 物理块 b{physical}{suffix}")


def print_pool(pool: BlockPool) -> None:
    print(
        f"  块池: {pool.num_blocks} 块 × {pool.block_size} token = "
        f"{pool.capacity_tokens} token；已用 {pool.used_blocks} 块，"
        f"空闲 {len(pool.free_blocks)} 块"
    )


# --------------------------------------------------------------------------- #
# 5. Demo
# --------------------------------------------------------------------------- #


def main() -> None:
    print("=" * 78)
    print("PagedAttention：把 KV cache 放进固定大小的块池，用 block table 索引")
    print(f"  block_size={BLOCK_SIZE}，q_heads={Q_HEADS}，kv_heads={KV_HEADS}，head_dim={HEAD_DIM}")
    print(f"  device={DEVICE}")
    print("=" * 78)
    print()

    pool = BlockPool(NUM_BLOCKS)
    cache = PagedKVCache(pool)
    print_pool(pool)
    print()

    seq_a = Sequence("seq-a")
    seq_b = Sequence("seq-b")
    history = {
        "seq-a": {"k": [], "v": []},
        "seq-b": {"k": [], "v": []},
    }

    print("1) 两个请求交错 prefill，然后用 decode 继续追加")
    print("-" * 78)

    key, value = random_kv(10)
    feed(cache, seq_a, key, value, history["seq-a"])
    print_block_table(seq_a, pool)

    key, value = random_kv(6)
    feed(cache, seq_b, key, value, history["seq-b"])
    print_block_table(seq_b, pool)

    # A 的第三块还有 2 个空位：第一批 2 个 token 正好填满它，第二批才会分配第 4 块。
    for tokens in (2, 2):
        key, value = random_kv(tokens)
        feed(cache, seq_a, key, value, history["seq-a"])
    key, value = random_kv(1)
    feed(cache, seq_b, key, value, history["seq-b"])

    print()
    print_block_table(seq_a, pool)
    print_block_table(seq_b, pool)
    print()
    print("  每个 feed 都断言了：paged attention == 连续缓存的 attention（差异 0）")
    print()

    print("2) 释放 seq-a，物理块回到池里，seq-c 直接复用")
    print("-" * 78)
    released = list(seq_a.block_table)
    cache.free(seq_a)
    print(f"  释放的物理块: {['b%d' % block for block in released]}")
    print_pool(pool)

    seq_c = Sequence("seq-c")
    key, value = random_kv(5)
    history["seq-c"] = {"k": [], "v": []}
    feed(cache, seq_c, key, value, history["seq-c"])
    print_block_table(seq_c, pool)
    reused = sorted(set(released) & set(seq_c.block_table))
    print(f"  seq-c 复用了其中 {len(reused)} 块: {['b%d' % block for block in reused]}")
    print_pool(pool)
    print()

    print("3) 内部碎片：块按固定大小分配，最后一块常常用不满")
    print("-" * 78)
    total_waste = 0
    for sequence in (seq_b, seq_c):
        print(
            f"  {sequence.seq_id}: {sequence.length} token 占用 "
            f"{len(sequence.block_table)} 块，浪费 {sequence.internal_waste} 个 slot"
        )
        total_waste += sequence.internal_waste
    print(f"  合计浪费 {total_waste} 个 slot（容量 {pool.capacity_tokens}）")
    print()
    print("连续性检查：所有 attention 逐步与连续缓存对比，全部一致。")
    print("注意 block table 里的物理块号并不连续（如 b5、b1、b3）——这正是 paged 的含义。")


if __name__ == "__main__":
    main()
