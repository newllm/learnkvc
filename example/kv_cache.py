"""One-request KV cache, from "recompute everything" to "append K/V".

Two generation flows, one model, one file:

* **no cache** -- every decode step re-feeds the whole prefix.  Each layer
  re-projects Q/K/V for every previous token, runs causal attention over all of
  them, and keeps only the last position's logits.
* **cache** -- prefill the prompt once, then project one new token per step and
  read the past K/V from ``KVCache``.  ``cached_attention`` appends the new K/V
  and computes attention against the whole (possibly sliding) window.

The file is organised as:

1. cache containers  -- ``LayerKVCache`` / ``KVCache`` (tensor layout, append,
   sliding window, beam reorder, fork)
2. attention         -- ``cached_attention`` (cache path) and ``full_attention``
   (naive path), written with the same math so both flows agree
3. tiny model        -- ``TinyModel`` (2 layers, GQA, absolute positions) plus
   ``generate`` for both flows and counters for the work performed
4. demo + checks     -- run both flows on the same prompt, compare tokens and
   logits, then exercise sliding window / reorder / fork

Run ``python3 example/kv_cache.py`` (requires PyTorch).

GPU / API notes
---------------
Every operation is device-agnostic: the demo runs on CUDA when one is
available and on CPU otherwise (``KVCACHE_DEVICE=cpu|cuda|mps`` overrides it).
With tensors on the GPU, PyTorch dispatches each call to a CUDA implementation:

* ``embed[tokens]``             -> ATen ``index_select`` CUDA kernel (a gather)
* ``hidden @ w``                -> cuBLAS GEMM (``cublasSgemm`` / ``cublasGemmEx``
                                   for fp16/bf16; cuBLASLt for some shapes)
* ``q @ k.transpose(-1, -2)``   -> batched GEMM (``cublasGemmStridedBatched``)
* ``softmax @ value``           -> second batched GEMM
* ``cat`` / ``repeat_interleave`` / ``masked_fill`` / ``+`` / ``relu``
                                -> ATen elementwise/copy CUDA kernels
* ``softmax``                   -> ATen CUDA softmax kernel (a fused
                                   FlashAttention kernel in production)
* ``argmax``                    -> ATen CUDA reduction kernel
* KV storage                    -> GPU HBM from PyTorch's caching allocator
                                   (``cudaMalloc``/``cudaFree`` under the hood)
* token upload and ``int(logits.argmax())``
                                -> H2D/D2H ``cudaMemcpyAsync`` + stream sync

On CPU the same calls go to MKL/oneDNN (x86) or Accelerate (macOS); on Apple
MPS they become Metal kernels.  No NCCL / multi-GPU collectives are used here.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _select_device() -> str:
    """CUDA when available, unless KVCACHE_DEVICE overrides it."""
    requested = os.environ.get("KVCACHE_DEVICE")
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


#: Where the model and the caches live.  All kernels below dispatch on this.
DEVICE = _select_device()


# --------------------------------------------------------------------------- #
# 1. Cache containers
# --------------------------------------------------------------------------- #
# K/V layout everywhere: (batch, kv_heads, sequence_length, head_dim).


@dataclass
class LayerKVCache:
    """KV cache for one transformer layer.

    ``max_length=None`` retains the full prefix.  Setting it implements a
    sliding-window cache: after every append only the latest ``max_length``
    key/value positions are retained.
    """

    max_length: Optional[int] = None
    key: Optional[torch.Tensor] = field(default=None, init=False)
    value: Optional[torch.Tensor] = field(default=None, init=False)

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.size(-2)

    def append(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append newly projected K/V and return the complete usable cache."""
        self._validate(key, value)
        if self.key is not None:
            if self.key.shape[:2] != key.shape[:2] or self.key.size(-1) != key.size(-1):
                raise ValueError("new K/V shape is incompatible with this cache")
            key = torch.cat((self.key, key), dim=-2)
            value = torch.cat((self.value, value), dim=-2)

        if self.max_length is not None:
            if self.max_length <= 0:
                raise ValueError("max_length must be positive")
            key, value = key[..., -self.max_length :, :], value[..., -self.max_length :, :]

        self.key, self.value = key, value
        return key, value

    def reorder(self, batch_indices: torch.LongTensor) -> None:
        """Reorder/duplicate batch rows, e.g. after beam-search selection."""
        if self.key is not None:
            self.key = self.key.index_select(0, batch_indices)
            self.value = self.value.index_select(0, batch_indices)

    def fork(self) -> "LayerKVCache":
        """Make an independent branch; clone because later appends must not alias."""
        branch = LayerKVCache(self.max_length)
        if self.key is not None:
            branch.key, branch.value = self.key.clone(), self.value.clone()
        return branch

    @staticmethod
    def _validate(key: torch.Tensor, value: torch.Tensor) -> None:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("K and V must have shape (batch, kv_heads, seq, head_dim)")
        if key.shape != value.shape:
            raise ValueError("K and V must have identical shapes")


@dataclass
class KVCache:
    """One ``LayerKVCache`` per decoder layer."""

    num_layers: int
    max_length: Optional[int] = None
    layers: list[LayerKVCache] = field(init=False)

    def __post_init__(self) -> None:
        self.layers = [LayerKVCache(self.max_length) for _ in range(self.num_layers)]

    @property
    def length(self) -> int:
        """Cached sequence length (every layer advances in lockstep)."""
        return self.layers[0].length if self.layers else 0

    def update(
        self, layer: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer].append(key, value)

    def reorder(self, batch_indices: torch.LongTensor) -> None:
        for layer in self.layers:
            layer.reorder(batch_indices)

    def fork(self) -> "KVCache":
        branch = KVCache(self.num_layers, self.max_length)
        branch.layers = [layer.fork() for layer in self.layers]
        return branch


# --------------------------------------------------------------------------- #
# 2. Attention: with cache and without
# --------------------------------------------------------------------------- #


def _expand_kv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat K/V heads to match GQA/MQA query heads."""
    if query.ndim != 4 or query.size(0) != key.size(0) or query.size(-1) != key.size(-1):
        raise ValueError("Q must match K/V batch size and head dimension")
    if query.size(1) % key.size(1):
        raise ValueError("number of query heads must be a multiple of KV heads")
    repeat = query.size(1) // key.size(1)
    # Gather kernel on GPU.  Production attention (FlashAttention) reads GQA
    # heads directly instead of materialising the repeated K/V.
    return key.repeat_interleave(repeat, dim=1), value.repeat_interleave(repeat, dim=1)


def cached_attention(
    query: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    cache: KVCache,
    layer: int,
) -> torch.Tensor:
    """Run causal attention and update a layer cache.

    This is the incremental path: append the new K/V, then attend against the
    whole cache.  The causal mask is built from absolute positions, because a
    sliding window may have dropped older keys.
    """
    past = cache.layers[layer].length
    # ``torch.cat`` allocates a new GPU buffer and copies old+new K/V: a CUDA
    # copy kernel plus an allocator call every step.  A paged cache writes into
    # preallocated blocks instead (see PREFIX_CACHE_READING.md).
    key, value = cache.update(layer, new_key, new_value)
    key, value = _expand_kv(query, key, value)

    q_len, k_len = query.size(-2), key.size(-2)
    # (1,H,T,D) @ (1,H,D,T): batched GEMM -> cuBLAS strided-batched.
    scores = query @ key.transpose(-1, -2) / (query.size(-1) ** 0.5)
    # Query i belongs at absolute position ``past + i``.  A window may have
    # discarded older keys, so key 0 is not necessarily absolute position 0.
    first_key_position = past + q_len - k_len
    causal = (first_key_position + torch.arange(k_len, device=query.device)[None, :]) <= (
        past + torch.arange(q_len, device=query.device)[:, None]
    )
    # masked_fill is an elementwise kernel; softmax is an ATen reduction
    # kernel.  A real engine fuses both into the attention kernel.
    scores = scores.masked_fill(~causal, float("-inf"))
    # (1,H,T,T) @ (1,H,T,D): the second batched GEMM.
    return F.softmax(scores, dim=-1) @ value


def full_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Causal attention over a fresh, complete prefix (the no-cache path).

    Same math as ``cached_attention`` with ``past=0``, so the two flows can be
    compared frame by frame.
    """
    key, value = _expand_kv(query, key, value)
    q_len, k_len = query.size(-2), key.size(-2)
    # Same two batched GEMMs as ``cached_attention``; only the K/V source differs.
    scores = query @ key.transpose(-1, -2) / (query.size(-1) ** 0.5)
    causal = torch.arange(k_len, device=query.device)[None, :] <= torch.arange(
        q_len, device=query.device
    )[:, None]
    scores = scores.masked_fill(~causal, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


# --------------------------------------------------------------------------- #
# 3. A tiny model, plus the two generation loops
# --------------------------------------------------------------------------- #

D_MODEL = 32
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
N_LAYERS = 2
VOCAB = 64
MAX_POS = 256
SEED = 13

# Weight shapes used by TinyModel, kept in one place so the dimensions are
# visible at a glance.
PROJ_SHAPES = (
    (D_MODEL, Q_HEADS * HEAD_DIM),  # Wq
    (D_MODEL, KV_HEADS * HEAD_DIM),  # Wk
    (D_MODEL, KV_HEADS * HEAD_DIM),  # Wv
    (D_MODEL, D_MODEL),  # Wo
    (D_MODEL, 2 * D_MODEL),  # W1
    (2 * D_MODEL, D_MODEL),  # W2
)


class TinyModel:
    """A 2-layer decoder with GQA, absolute position embeddings and tied head.

    No LayerNorm and no RoPE, so the math stays readable; neither omission
    changes the point that a token's K/V is a function of its prefix and its
    absolute position.
    """

    def __init__(self, seed: int = SEED) -> None:
        generator = torch.Generator().manual_seed(seed)

        def rand(*shape: int) -> torch.Tensor:
            return (torch.rand(*shape, generator=generator) - 0.5) * 0.4

        self.num_layers = N_LAYERS
        # Weights live wherever ``DEVICE`` points; on CUDA this is one H2D copy
        # per weight before generation.
        self.embed = rand(VOCAB, D_MODEL).to(DEVICE)
        self.pos = rand(MAX_POS, D_MODEL).to(DEVICE)
        self.layers = [
            tuple(rand(*shape).to(DEVICE) for shape in PROJ_SHAPES)
            for _ in range(N_LAYERS)
        ]

    @property
    def device(self) -> torch.device:
        return self.embed.device

    def forward(
        self,
        tokens: torch.Tensor,
        cache: Optional[KVCache] = None,
        stats: Optional["Stats"] = None,
    ) -> torch.Tensor:
        """Return the logits of the last position.

        ``tokens`` is a 1-D tensor of *new* tokens.  With a cache, their
        absolute positions continue from ``cache.length``; without one they
        start at 0 and the whole prefix is recomputed.
        """
        if tokens.ndim != 1:
            raise ValueError("tokens must be 1-D")
        # CPU -> GPU copy (cudaMemcpyAsync) when the caller built a CPU tensor.
        tokens = tokens.to(self.device)
        token_count = tokens.numel()
        past = cache.length if cache is not None else 0
        if past + token_count > MAX_POS:
            raise ValueError("sequence exceeds MAX_POS")

        if stats is not None:
            stats.positions += token_count

        # Gather (index_select kernel) + elementwise add, both ATen CUDA kernels.
        hidden = self.embed[tokens] + self.pos[past : past + token_count]
        hidden = hidden.unsqueeze(0)  # (1, tokens, d_model)

        for layer, (wq, wk, wv, wo, w1, w2) in enumerate(self.layers):
            # (1,T,D) @ (D,D): GEMM -> cuBLAS (cublasSgemm/GemmEx).
            # view/transpose below only change strides; they launch no kernel.
            query = (hidden @ wq).view(1, token_count, Q_HEADS, HEAD_DIM).transpose(1, 2)
            key = (hidden @ wk).view(1, token_count, KV_HEADS, HEAD_DIM).transpose(1, 2)
            value = (hidden @ wv).view(1, token_count, KV_HEADS, HEAD_DIM).transpose(1, 2)

            if stats is not None:
                # (query, key) pairs for this layer.  Works for both flows:
                # the naive one has past=0, the cached one adds the history.
                stats.attention_pairs += token_count * past + token_count * (token_count + 1) // 2

            if cache is not None:
                attended = cached_attention(query, key, value, cache, layer)
            else:
                attended = full_attention(query, key, value)

            attended = attended.transpose(1, 2).reshape(1, token_count, D_MODEL)
            # Output projection and the two MLP matrices: three more cuBLAS GEMMs.
            hidden = hidden + attended @ wo
            hidden = hidden + F.relu(hidden @ w1) @ w2

        # Weight tying: logits = last hidden state · embeddingᵀ.
        # (D,) @ (D,V): GEMV -> cuBLAS.
        return hidden[0, -1] @ self.embed.T


@dataclass
class Stats:
    """Counted work for one generation run."""

    positions: int = 0        # token passes through all layers
    attention_pairs: int = 0  # (query, key) pairs, summed over layers


@dataclass
class Step:
    step: int
    input_tokens: int
    token: int


def generate(
    model: TinyModel,
    prompt: Sequence[int],
    n_new: int,
    use_cache: bool,
    stats: Stats,
) -> tuple[list[int], list[Step]]:
    """Greedy generation; the only difference between the flows is the input."""
    sequence = list(prompt)
    steps: list[Step] = []

    if use_cache:
        cache = KVCache(model.num_layers)
        # H2D copy of the prompt (cudaMemcpyAsync) when running on GPU.
        logits = model.forward(torch.tensor(sequence, device=model.device), cache=cache, stats=stats)
        token = int(logits.argmax())
        steps.append(Step(0, len(sequence), token))
        sequence.append(token)
        while len(steps) < n_new:
            logits = model.forward(
                torch.tensor([sequence[-1]], device=model.device), cache=cache, stats=stats
            )
            # argmax is a CUDA reduction kernel; int(...) copies one value back
            # to the host and synchronises the stream.  Fine for a demo, but a
            # real engine samples on the GPU and keeps the loop device-side.
            token = int(logits.argmax())
            steps.append(Step(len(steps), 1, token))
            sequence.append(token)
        return sequence, steps

    while len(steps) < n_new:
        # The whole prefix is fed again; only the last logits are used.
        # H2D copy of the prefix on GPU.
        logits = model.forward(torch.tensor(sequence, device=model.device), stats=stats)
        token = int(logits.argmax())
        steps.append(Step(len(steps), len(sequence), token))
        sequence.append(token)
    return sequence, steps


# --------------------------------------------------------------------------- #
# 4. Reporting helpers
# --------------------------------------------------------------------------- #


def display_width(text: str) -> int:
    """Terminal width of ``text``: CJK/full-width characters occupy two columns."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int, right: bool = False) -> str:
    padding = " " * max(0, width - display_width(text))
    return padding + text if right else text + padding


def print_steps(no_cache: list[Step], cached: list[Step]) -> None:
    print(
        pad("步骤", 6, right=True) + " "
        + pad("新 token", 10, right=True) + " "
        + pad("无 cache 输入", 16, right=True) + " "
        + pad("有 cache 输入", 16, right=True)
    )
    for plain, cached_step in zip(no_cache, cached):
        assert plain.step == cached_step.step and plain.token == cached_step.token
        print(
            pad(str(plain.step), 6, right=True) + " "
            + pad(str(plain.token), 10, right=True) + " "
            + pad(f"{plain.input_tokens} 个", 16, right=True) + " "
            + pad(f"{cached_step.input_tokens} 个", 16, right=True)
        )


def print_summary(
    plain: Stats, cached: Stats, prompt_len: int, n_new: int
) -> None:
    # What the flows must compute, independent of the toy weights.
    assert plain.positions == sum(range(prompt_len, prompt_len + n_new))
    assert cached.positions == prompt_len + (n_new - 1)
    assert plain.attention_pairs == N_LAYERS * sum(
        count * (count + 1) // 2 for count in range(prompt_len, prompt_len + n_new)
    )
    assert cached.attention_pairs == N_LAYERS * (
        prompt_len * (prompt_len + 1) // 2
        + sum(range(prompt_len + 1, prompt_len + n_new))
    )

    rows = [
        ("位置前向（token 穿过全部层）", plain.positions, cached.positions),
        ("attention (query,key) 对数（各层合计）", plain.attention_pairs, cached.attention_pairs),
    ]
    label_width = max(display_width(label) for label, _, _ in rows) + 2

    print()
    print("汇总")
    print("-" * 90)
    for label, plain_value, cached_value in rows:
        print(
            "  " + pad(label, label_width)
            + f"无 cache {plain_value:>8}    有 cache {cached_value:>8}"
            + f"   {plain_value / cached_value:.1f}x"
        )
    print()
    print(
        f"  白白重算的位置：{plain.positions - cached.positions} 个"
        f"（算出了和之前完全相同的 K/V）"
    )
    print("  attention 对数也在下降，但 decode 每步仍要读全部历史；")
    print("  前缀越长，有 cache 的路径里 attention 占比越高——cache 省的是重算子，不是读历史。")


def check_numerics(model: TinyModel, prompt: Sequence[int]) -> None:
    """One full forward and prefill + decode must produce the same logits."""
    token = 33
    plain = model.forward(torch.tensor(list(prompt) + [token]))

    cache = KVCache(model.num_layers)
    model.forward(torch.tensor(list(prompt)), cache=cache)
    cached = model.forward(torch.tensor([token]), cache=cache)

    difference = (plain - cached).abs().max().item()
    assert difference < 1e-5, f"cache path diverged from full forward: {difference}"
    assert int(plain.argmax()) == int(cached.argmax()), "cache changed the argmax"
    print()
    print("数值层面：同一段 [prefix + token]，一次性 forward 与 prefill + decode 的")
    print(f"logits 最大差异 {difference:.2e}（浮点排序差异），argmax 完全一致。")


def check_cache_features(model: TinyModel) -> None:
    """Exercise the parts of ``KVCache`` that the generation loop does not."""
    # 1. GQA: 4 query heads read 2 KV heads; prefill writes both layers.
    cache = KVCache(model.num_layers)
    logits = model.forward(torch.tensor([1, 2, 3]), cache=cache)
    assert cache.length == 3 and logits.shape == (VOCAB,)
    assert cache.layers[0].key.shape == (1, KV_HEADS, 3, HEAD_DIM)

    # 2. Sliding window: only the latest max_length positions survive, and they
    #    are exactly the tail of what a full cache would hold.  (This checks
    #    storage; a windowed cache also needs absolute positions for real
    #    generation, which this toy model does not model.)
    window = KVCache(model.num_layers, max_length=4)
    model.forward(torch.tensor(list(range(6))), cache=window)
    full = KVCache(model.num_layers)
    model.forward(torch.tensor(list(range(6))), cache=full)
    assert window.length == 4
    assert torch.equal(window.layers[0].key, full.layers[0].key[..., -4:, :])

    # 3. Beam search: reorder/duplicate every layer consistently.
    key = torch.randn(2, KV_HEADS, 3, HEAD_DIM)
    reorder_cache = KVCache(2)
    for layer in range(2):
        reorder_cache.update(layer, key, key.clone())
    reorder_cache.reorder(torch.tensor([1, 0]))
    assert torch.equal(reorder_cache.layers[0].key, key.flip(0))

    # 4. Fork: branch appends must not leak into the parent.
    parent = KVCache(1)
    k = torch.randn(1, KV_HEADS, 2, HEAD_DIM)
    parent.update(0, k, k.clone())
    branch = parent.fork()
    extra = torch.randn(1, KV_HEADS, 1, HEAD_DIM)
    branch.update(0, extra, extra)
    assert parent.length == 2 and branch.length == 3


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    model = TinyModel(seed=SEED)
    prompt = (7, 12, 3, 25, 9, 18, 1, 30)
    n_new = 8

    print("=" * 78)
    print("同一个 LLM、同一段 prompt，跑两遍：不带 cache 与带 cache")
    print(f"  LLM：{N_LAYERS} 层、{Q_HEADS}/{KV_HEADS} 头（GQA）、d_model={D_MODEL}、绝对位置嵌入")
    device_type = torch.device(DEVICE).type
    if device_type == "cuda":
        print(f"  device = {DEVICE}：matmul -> cuBLAS，其余算子 -> ATen CUDA kernel")
    elif device_type == "mps":
        print(f"  device = {DEVICE}：Apple Metal Performance Shaders（无 cuBLAS）")
    else:
        print(f"  device = {DEVICE}：CPU（MKL/oneDNN/Accelerate），本机没有 GPU 参与")
    print(f"  prompt = {list(prompt)}（{len(prompt)} tokens），greedy 生成 {n_new} 个 token")
    print("=" * 78)
    print()

    plain_stats = Stats()
    plain_seq, plain_steps = generate(model, prompt, n_new, use_cache=False, stats=plain_stats)

    cached_stats = Stats()
    cached_seq, cached_steps = generate(model, prompt, n_new, use_cache=True, stats=cached_stats)

    assert plain_seq == cached_seq, "cache must produce exactly the same tokens"
    print_steps(plain_steps, cached_steps)
    print_summary(plain_stats, cached_stats, len(prompt), n_new)
    print(f"\n  生成结果：{plain_seq[len(prompt):]}")

    check_numerics(model, prompt)
    check_cache_features(model)
    print()
    print("全部自测通过：两条路径 token 一致、logits 一致；GQA / 滑动窗口 / reorder / fork 正常。")


if __name__ == "__main__":
    main()
