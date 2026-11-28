"""Request-level KV cache reuse, simulated with a block-based prefix cache.

``kv_cache.py`` keeps the K/V tensors of **one** generation so that token
``t+1`` does not recompute the prefix ``t``.  This file looks one level higher:
an inference server that serves **many** requests.  When a later request starts
with the same tokens as an earlier one (a shared system prompt, few-shot
examples, a long document, a multi-turn chat history), the K/V blocks computed
for that prefix can be reused instead of being prefilled again.  That is what
vLLM calls *automatic prefix caching* and what SGLang calls *RadixAttention*.

The scheme implemented here mirrors paged attention:

* KV memory is a pool of fixed-size blocks of ``BLOCK_SIZE`` tokens.
* A block's identity is ``hash(parent_hash, tokens)``, so the same token prefix
  always maps to the same block, and a prefix match is a walk down that chain.
* Blocks are shared by reference count.  A block is only evicted -- least
  recently used first, leaves before parents -- when no request holds it.

Only prompt blocks are modelled.  Real engines also allocate blocks for the
decoding tokens; that would double the bookkeeping without teaching anything
new about reuse.

This file is a pure-Python *simulation* of the block manager: no tensor and no
GPU API is involved.  In a real engine the blocks hold K/V in GPU HBM, and a
paged-attention kernel (CUDA/Triton) reads them through a per-request block
table; this file only models the bookkeeping around that memory.

Run ``python3 example/prefix_cache_demo.py`` (no dependencies beyond the
standard library) to execute the three scenarios.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

BLOCK_SIZE = 4
MASK64 = (1 << 64) - 1
HASH_SEED = b"whykvc/prefix-cache/v1"
VOCAB_SIZE = 1000


# --------------------------------------------------------------------------- #
# Toy "model": a deterministic per-token state update.
# --------------------------------------------------------------------------- #
# A real engine caches K/V tensors.  We cannot afford real weights here, but we
# still want to *prove* that reuse is exact, not approximate.  So each request
# keeps a 64-bit state that is a pure function of its token prefix.  Reusing a
# cached prefix must produce exactly the state a full recompute would produce.


def block_hash(parent_hash: int, tokens: Sequence[int]) -> int:
    """Content-addressed identity of one block in the prefix chain."""
    payload = HASH_SEED + parent_hash.to_bytes(8, "little")
    payload += b"".join(int(token).to_bytes(4, "little") for token in tokens)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def mix(state: int, token: int) -> int:
    """Advance the toy hidden state by one token."""
    x = ((state ^ (token + 0x9E3779B97F4A7C15)) * 0x100000001B3) & MASK64
    x ^= x >> 29
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= x >> 32
    return x & MASK64


def run_state(state: int, tokens: Iterable[int]) -> int:
    for token in tokens:
        state = mix(state, token)
    return state


def next_token(state: int) -> int:
    """Stand-in for the LM head: deterministic, so runs are reproducible."""
    return (state >> 11) % VOCAB_SIZE


# --------------------------------------------------------------------------- #
# Block pool
# --------------------------------------------------------------------------- #


@dataclass
class Block:
    """One physical KV block. ``tokens`` is what makes it reusable."""

    block_id: int
    block_hash: int
    tokens: tuple[int, ...]
    parent_hash: Optional[int]
    state: int  # toy hidden state after the last token of this block
    ref_count: int = 0
    last_used: int = 0
    children: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f"b{self.block_id:02d}"


@dataclass
class CacheStats:
    prompt_tokens: int = 0
    reused_tokens: int = 0
    prefilled_tokens: int = 0
    decoded_tokens: int = 0
    blocks_created: int = 0
    blocks_evicted: int = 0
    blocks_reused: int = 0
    peak_blocks: int = 0
    peak_blocks_in_use: int = 0

    @property
    def hit_rate(self) -> float:
        return self.reused_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def prefill_ratio(self) -> float:
        return self.prefilled_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class PrefixMatch:
    """Result of looking a prompt up in the block pool."""

    block_hashes: list[int]
    cached_blocks: int
    cached_tokens: int
    prefill_tokens: int
    state: int  # hidden state after the reused prefix


class CacheFullError(RuntimeError):
    """Every block is pinned by an active request; a real engine would preempt."""


class PrefixCache:
    """Block pool with reference counting, LRU eviction and prefix matching."""

    def __init__(self, capacity_blocks: int, block_size: int = BLOCK_SIZE) -> None:
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self.blocks: dict[int, Block] = {}
        self.stats = CacheStats()
        self.events: list[tuple[str, str]] = []
        self._next_id = 0
        self._clock = 0

    # -- introspection ----------------------------------------------------- #

    @property
    def size(self) -> int:
        return len(self.blocks)

    @property
    def in_use(self) -> int:
        return sum(1 for block in self.blocks.values() if block.ref_count > 0)

    def block_labels(self, hashes: Iterable[int]) -> list[str]:
        return [self.blocks[h].label for h in hashes if h in self.blocks]

    # -- request lifetime -------------------------------------------------- #

    def acquire(self, token_ids: Sequence[int]) -> PrefixMatch:
        """Match a prompt against cached blocks, then reserve blocks for it.

        Returns how many leading tokens were reused and how many still have to
        be prefilled.  Only whole blocks are reused: a trailing partial block
        is recomputed, which is the granularity trade-off of paged caches.
        """
        self._clock += 1
        block_size = self.block_size
        matched: list[int] = []
        cached_blocks = 0
        prefix_state = 0
        parent: Optional[int] = None

        # 1. Walk the hash chain while blocks exist.
        for start in range(0, len(token_ids) - block_size + 1, block_size):
            chunk = tuple(token_ids[start : start + block_size])
            digest = block_hash(parent or 0, chunk)
            block = self.blocks.get(digest)
            if block is None:
                break
            block.ref_count += 1
            block.last_used = self._clock
            matched.append(digest)
            cached_blocks += 1
            prefix_state = block.state
            parent = digest

        cached_tokens = cached_blocks * block_size

        # 2. Allocate fresh blocks for the rest.  Pinned blocks are released if
        #    allocation fails, so a failed admission never leaks references.
        allocated: list[int] = []
        chain_state = prefix_state
        try:
            for start in range(cached_tokens, len(token_ids) - block_size + 1, block_size):
                chunk = tuple(token_ids[start : start + block_size])
                chain_state = run_state(chain_state, chunk)
                allocated.append(self._create_block(parent, chunk, chain_state))
                parent = allocated[-1]
        except CacheFullError:
            self._release_hashes(allocated + matched)
            raise

        self.stats.blocks_reused += len(matched)
        return PrefixMatch(
            block_hashes=matched + allocated,
            cached_blocks=cached_blocks,
            cached_tokens=cached_tokens,
            prefill_tokens=len(token_ids) - cached_tokens,
            state=prefix_state,
        )

    def release(self, block_hashes: Iterable[int]) -> None:
        """Drop a request's references; blocks stay cached until eviction."""
        hashes = list(block_hashes)
        self._release_hashes(hashes)
        self._clock += 1
        for digest in hashes:
            block = self.blocks.get(digest)
            if block is not None:
                block.last_used = self._clock

    def _release_hashes(self, hashes: Iterable[int]) -> None:
        for digest in hashes:
            block = self.blocks.get(digest)
            if block is None:
                continue
            block.ref_count = max(0, block.ref_count - 1)
            block.last_used = self._clock

    # -- allocation and eviction ------------------------------------------- #

    def _create_block(
        self, parent: Optional[int], tokens: tuple[int, ...], state: int
    ) -> int:
        self._make_room()
        digest = block_hash(parent or 0, tokens)
        block = Block(
            block_id=self._next_id,
            block_hash=digest,
            tokens=tokens,
            parent_hash=parent,
            state=state,
            ref_count=1,
            last_used=self._clock,
        )
        self._next_id += 1
        self.blocks[digest] = block
        if parent is not None and parent in self.blocks:
            self.blocks[parent].children.add(digest)
        self.stats.blocks_created += 1
        self.stats.peak_blocks = max(self.stats.peak_blocks, self.size)
        self.stats.peak_blocks_in_use = max(self.stats.peak_blocks_in_use, self.in_use)
        return digest

    def _make_room(self) -> None:
        """Evict idle leaf blocks until one slot is free."""
        while self.size >= self.capacity_blocks:
            victim = self._pick_victim()
            if victim is None:
                raise CacheFullError(
                    f"all {self.size} blocks are pinned by active requests"
                )
            self._evict(victim)

    def _pick_victim(self) -> Optional[Block]:
        # Leaves first: evicting an ancestor while a descendant is still cached
        # would strand the descendant, because it is only reachable through its
        # parent's hash.
        candidates = [
            block
            for block in self.blocks.values()
            if block.ref_count == 0 and not block.children
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda block: (block.last_used, block.block_id))

    def _evict(self, block: Block) -> None:
        del self.blocks[block.block_hash]
        if block.parent_hash is not None and block.parent_hash in self.blocks:
            self.blocks[block.parent_hash].children.discard(block.block_hash)
        self.stats.blocks_evicted += 1
        self.events.append(("evict", block.label))


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival: int
    prompt: tuple[int, ...]
    max_new_tokens: int


@dataclass
class RequestRecord:
    request: Request
    cached_blocks: int
    cached_tokens: int
    prefill_tokens: int
    block_hashes: list[int]
    block_labels: list[str]
    prefill_state: int
    decoded_tokens: int = 0
    decode_state: int = 0

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def prompt_tokens(self) -> int:
        return len(self.request.prompt)

    @property
    def new_blocks(self) -> int:
        return len(self.block_labels) - self.cached_blocks

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens


class Simulator:
    """Runs requests step by step: admit -> decode one token -> release."""

    def __init__(self, cache: PrefixCache, requests: Sequence[Request]) -> None:
        self.cache = cache
        self.requests = sorted(requests, key=lambda request: (request.arrival, request.request_id))
        self.records: dict[str, RequestRecord] = {}
        self.trace: list[str] = []
        self._active: list[str] = []
        self._pending = list(self.requests)
        self._event_cursor = 0

    def run(self) -> dict[str, RequestRecord]:
        step = 0
        while self._pending or self._active:
            self._admit(step)
            self._decode(step)
            step += 1
            if step > 10_000:  # pragma: no cover - guards against a stuck loop
                raise RuntimeError("simulation did not terminate")
        return self.records

    # -- phases ------------------------------------------------------------ #

    def _admit(self, step: int) -> None:
        while self._pending and self._pending[0].arrival <= step:
            request = self._pending.pop(0)
            match = self.cache.acquire(request.prompt)
            state = run_state(match.state, request.prompt[match.cached_tokens :])
            record = RequestRecord(
                request=request,
                cached_blocks=match.cached_blocks,
                cached_tokens=match.cached_tokens,
                prefill_tokens=match.prefill_tokens,
                block_hashes=match.block_hashes,
                block_labels=self.cache.block_labels(match.block_hashes),
                prefill_state=state,
                decode_state=state,
            )
            self.records[request.request_id] = record
            self._active.append(request.request_id)

            self.cache.stats.prompt_tokens += record.prompt_tokens
            self.cache.stats.reused_tokens += match.cached_tokens
            self.cache.stats.prefilled_tokens += match.prefill_tokens

            hit = (
                f"复用 {match.cached_blocks} 块={match.cached_tokens} tok"
                if match.cached_blocks
                else "无命中"
            )
            self.trace.append(
                f"[t={step}] admit  {_pad(request.request_id, 14)} "
                f"prompt={record.prompt_tokens:>3} {_pad(hit, 22)} "
                f"prefill={match.prefill_tokens:>3} tok  "
                f"持有块={','.join(record.block_labels) or '-'}"
            )
            self._drain_events(step)

    def _decode(self, step: int) -> None:
        for request_id in list(self._active):
            record = self.records[request_id]
            record.decode_state = mix(record.decode_state, next_token(record.decode_state))
            record.decoded_tokens += 1
            self.cache.stats.decoded_tokens += 1
            if record.decoded_tokens < record.request.max_new_tokens:
                self.trace.append(
                    f"[t={step}] decode {_pad(request_id, 14)} "
                    f"token {record.decoded_tokens}/{record.request.max_new_tokens}"
                )
            else:
                self._finish(record, step)

    def _finish(self, record: RequestRecord, step: int) -> None:
        self.cache.release(record.block_hashes)
        self._active.remove(record.request_id)
        self.trace.append(
            f"[t={step}] free   {_pad(record.request_id, 14)} "
            f"释放 {len(record.block_labels)} 块（仍在池中，可被后续请求复用）"
        )

    def _drain_events(self, step: int) -> None:
        new_events = self.cache.events[self._event_cursor :]
        self._event_cursor = len(self.cache.events)
        labels = [payload for kind, payload in new_events if kind == "evict"]
        if labels:
            self.trace.append(
                f"[t={step}] cache  淘汰 {len(labels)} 块（LRU，无人占用）：{','.join(labels)}"
            )


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def display_width(text: str) -> int:
    """Terminal width of ``text``: CJK characters occupy two columns."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    """Pad to a display width so Chinese and ASCII columns line up."""
    padding = " " * max(0, width - display_width(text))
    return padding + text if right else text + padding


def print_trace(simulator: Simulator) -> None:
    print("请求时间线")
    print("-" * 108)
    for line in simulator.trace:
        print(line)


def print_request_table(records: dict[str, RequestRecord]) -> None:
    print()
    print("每个请求的命中情况")
    print("-" * 108)
    print(
        _pad("请求", 16)
        + _pad("prompt", 8, right=True)
        + _pad("复用", 8, right=True)
        + _pad("重算", 8, right=True)
        + _pad("命中率", 10, right=True)
        + _pad("新建块", 8, right=True)
        + "  持有块"
    )
    for record in records.values():
        print(
            _pad(record.request_id, 16)
            + _pad(str(record.prompt_tokens), 8, right=True)
            + _pad(str(record.cached_tokens), 8, right=True)
            + _pad(str(record.prefill_tokens), 8, right=True)
            + _pad(f"{record.hit_rate:.0%}", 10, right=True)
            + _pad(str(record.new_blocks), 8, right=True)
            + f"  {','.join(record.block_labels) or '-'}"
        )


def print_summary(cache: PrefixCache, title: str) -> None:
    stats = cache.stats
    print()
    print(f"汇总：{title}")
    print("-" * 108)
    print(f"  prompt tokens       : {stats.prompt_tokens}")
    print(f"  复用（命中）tokens  : {stats.reused_tokens}  ({stats.hit_rate:.1%})")
    print(
        f"  实际 prefill tokens : {stats.prefilled_tokens}  "
        f"(占 {stats.prefill_ratio:.1%})"
    )
    print(
        f"  相当于省下的 prefill: {stats.reused_tokens} 个 token"
        f"（不提速 decode，只减少首 token 前的计算量）"
    )
    print(f"  decode tokens       : {stats.decoded_tokens}")
    print(
        f"  块：新建 {stats.blocks_created}，命中复用 {stats.blocks_reused} 次，"
        f"淘汰 {stats.blocks_evicted}，峰值 {stats.peak_blocks}/{cache.capacity_blocks}"
        f"（同时被占用峰值 {stats.peak_blocks_in_use}）"
    )
    print(f"  结束后池中块        : {cache.size}/{cache.capacity_blocks}（仍被占用 {cache.in_use}）")


def verify_reuse_exact(records: dict[str, RequestRecord]) -> None:
    """The reused prefix must give the same state as a full recompute."""
    for record in records.values():
        recomputed = run_state(0, record.request.prompt)
        assert record.prefill_state == recomputed, (
            f"reuse diverged from full recompute for {record.request_id}"
        )


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def req(
    request_id: str, arrival: int, prompt: Sequence[int], max_new_tokens: int = 2
) -> Request:
    """Shorthand that keeps the scenario tables readable."""
    return Request(request_id, arrival, tuple(prompt), max_new_tokens)


SYSTEM_PROMPT = tuple(range(1, 25))          # 24 tokens = 6 blocks
FEW_SHOT = tuple(range(101, 121))            # 20 tokens = 5 blocks
ALT_SYSTEM_PROMPT = tuple(range(301, 313))   # 12 tokens = 3 blocks


def scenario_shared_prefix() -> dict[str, RequestRecord]:
    print("=" * 108)
    print("场景 1：一个客服机器人，多条请求共享 system prompt 与 few-shot 示例")
    print("  block_size = 4，块池容量 = 64 块（256 tokens）")
    print("=" * 108)
    print()

    cache = PrefixCache(capacity_blocks=64)
    requests = [
        # 冷启动：池中没有任何块可以复用。
        req("req-refund-1", 0, SYSTEM_PROMPT + tuple(range(201, 209))),
        # 只有 system prompt（6 块）是共享的。
        req("req-shipping-2", 0, SYSTEM_PROMPT + tuple(range(211, 217))),
        # system prompt + few-shot 都被共享，命中更长。
        req("req-tech-3", 1, SYSTEM_PROMPT + FEW_SHOT + tuple(range(221, 226))),
        # 与上一个请求完全相同：只有末尾那个不满 4 token 的块需要重算。
        req("req-tech-4", 2, SYSTEM_PROMPT + FEW_SHOT + tuple(range(221, 226))),
        # 完全不同的 system prompt：无命中。
        req("req-billing-5", 3, ALT_SYSTEM_PROMPT + tuple(range(231, 243))),
    ]

    simulator = Simulator(cache, requests)
    records = simulator.run()
    print_trace(simulator)
    print_request_table(records)
    print_summary(cache, "共享前缀场景")

    verify_reuse_exact(records)
    refund, shipping, tech3, tech4, billing = (
        records["req-refund-1"],
        records["req-shipping-2"],
        records["req-tech-3"],
        records["req-tech-4"],
        records["req-billing-5"],
    )
    assert refund.cached_tokens == 0
    assert shipping.cached_tokens == 24, "system prompt should be shared"
    assert shipping.prefill_tokens == 6
    assert tech3.cached_tokens == 24, "only the system prompt existed before it arrived"
    assert tech3.prefill_tokens == 25
    assert tech4.cached_tokens == 48, "12 whole blocks, including the few-shot examples"
    assert tech4.prefill_tokens == 1, "the trailing partial token is prefilled again"
    assert billing.cached_tokens == 0, "a different system prompt must not hit"
    assert cache.stats.blocks_evicted == 0
    assert cache.stats.peak_blocks <= cache.capacity_blocks
    return records


def scenario_eviction() -> dict[str, RequestRecord]:
    print()
    print("=" * 108)
    print("场景 2：块池变小，空闲前缀被 LRU 淘汰，之后的请求只能全量重算")
    print("  block_size = 4，块池容量 = 10 块（40 tokens）")
    print("=" * 108)
    print()

    prefix_a = tuple(range(10, 34))   # 24 tokens = 6 blocks
    prefix_c = tuple(range(40, 64))   # 24 tokens = 6 blocks

    cache = PrefixCache(capacity_blocks=10)
    requests = [
        # 8 块，正好占满大半个池子。
        req("req-a", 0, prefix_a + tuple(range(70, 78)), max_new_tokens=1),
        # 前 24 个 token 与 req-a 相同，命中 6 块，只需要新算自己的尾巴。
        req("req-b", 1, prefix_a + tuple(range(80, 88)), max_new_tokens=1),
        # 前缀完全不同、且要 10 块：只能先把 req-a / req-b 的空闲块全部淘汰。
        req("req-c", 2, prefix_c + tuple(range(90, 106))),
        # req-a 的块早已被淘汰，这次只能重新 prefill。
        req("req-d", 4, prefix_a + tuple(range(70, 78))),
    ]

    simulator = Simulator(cache, requests)
    records = simulator.run()
    print_trace(simulator)
    print_request_table(records)
    print_summary(cache, "块池不足 + LRU 淘汰场景")

    verify_reuse_exact(records)
    a, b, c, d = (records[k] for k in ("req-a", "req-b", "req-c", "req-d"))
    assert a.cached_tokens == 0
    assert b.cached_tokens == 24, "shared prefix of req-a should still be cached"
    assert b.prefill_tokens == 8
    assert c.cached_tokens == 0
    assert c.new_blocks == 10
    assert d.cached_tokens == 0, "req-a's blocks were all evicted by req-c"
    assert d.prefill_tokens == d.prompt_tokens
    assert cache.stats.blocks_evicted >= 10, "scenario is meant to thrash the pool"
    assert cache.stats.peak_blocks <= cache.capacity_blocks
    return records


def scenario_repeated_phrase() -> dict[str, RequestRecord]:
    print()
    print("=" * 108)
    print("场景 3：同一段 4 个 token 在 prompt 里出现两次 —— 相同的 token 不等于相同的 K/V")
    print("  block_size = 4，块池容量 = 32 块")
    print("=" * 108)
    print()

    phrase = (9, 9, 9, 9)              # 4 tokens = 1 block
    prefix = tuple(range(600, 612))    # 12 tokens = 3 blocks
    middle = tuple(range(620, 628))    # 8 tokens = 2 blocks

    cache = PrefixCache(capacity_blocks=32)
    requests = [
        # phrase 出现在两个不同的上下文里：位置 12 和位置 24。
        req("req-x", 0, prefix + phrase + middle + phrase, max_new_tokens=1),
        # 与 req-x 共享到第一个 phrase 为止；第二个 phrase 的块绝不能被借用。
        req("req-y", 1, prefix + phrase + tuple(range(640, 644)), max_new_tokens=1),
    ]

    simulator = Simulator(cache, requests)
    records = simulator.run()
    print_trace(simulator)
    print_request_table(records)
    print_summary(cache, "重复片段场景")

    verify_reuse_exact(records)
    x, y = records["req-x"], records["req-y"]
    assert x.cached_tokens == 0
    assert y.cached_tokens == 16, "prefix + first phrase = 4 blocks"
    assert y.prefill_tokens == 4
    return records


def main() -> None:
    shared = scenario_shared_prefix()
    evicted = scenario_eviction()
    repeated = scenario_repeated_phrase()

    def stats(records: dict[str, RequestRecord]) -> tuple[int, int]:
        return (
            sum(record.prompt_tokens for record in records.values()),
            sum(record.cached_tokens for record in records.values()),
        )

    shared_prompt, shared_reused = stats(shared)
    evicted_prompt, evicted_reused = stats(evicted)
    repeated_prompt, repeated_reused = stats(repeated)
    assert shared_reused * 2 > shared_prompt, "shared-prefix scenario should mostly hit"

    print()
    print("三个场景都通过了断言：")
    print("  * 复用得到的隐藏状态与全量重算逐位一致（verify_reuse_exact）")
    print("  * 共享前缀命中、块对齐截断、LRU 淘汰后的回退都符合预期")
    print(
        f"  * 场景 1：{shared_prompt} 个 prompt token 中 {shared_reused} 个直接复用"
        f"（{shared_reused / shared_prompt:.0%}）"
    )
    print(
        f"  * 场景 2：{evicted_prompt} 个 prompt token 中只有 {evicted_reused} 个能复用"
        f"（{evicted_reused / evicted_prompt:.0%}），淘汰把命中率拉低了"
    )
    print(
        f"  * 场景 3：{repeated_prompt} 个 prompt token 中 {repeated_reused} 个复用，"
        f"第二个 phrase 没有被误当成前缀"
    )


if __name__ == "__main__":
    main()
