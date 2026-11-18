"""A small, real KV cache for PyTorch decoder attention.

The cache stores the key/value tensors produced by each transformer layer, not
token IDs or a synthetic summary.  Tensor layout is:

    (batch, number_of_kv_heads, sequence_length, head_dimension)

Run this file after installing PyTorch to execute the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


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
            key, value = key[..., -self.max_length:, :], value[..., -self.max_length:, :]

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

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer].append(key, value)

    def reorder(self, batch_indices: torch.LongTensor) -> None:
        for layer in self.layers:
            layer.reorder(batch_indices)

    def fork(self) -> "KVCache":
        branch = KVCache(self.num_layers, self.max_length)
        branch.layers = [layer.fork() for layer in self.layers]
        return branch


def cached_attention(
    query: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    cache: KVCache,
    layer: int,
) -> torch.Tensor:
    """Run causal attention and update a layer cache.

    Query has ``q_heads`` heads, while K/V may have fewer heads (GQA/MQA).
    This intentionally uses explicit scores so the causal mask remains correct
    when a multi-token prompt is appended to an existing prefix.
    """
    past = cache.layers[layer].length
    key, value = cache.update(layer, new_key, new_value)
    if query.ndim != 4 or query.size(0) != key.size(0) or query.size(-1) != key.size(-1):
        raise ValueError("Q must match K/V batch size and head dimension")
    if query.size(1) % key.size(1):
        raise ValueError("number of query heads must be a multiple of KV heads")
    repeat = query.size(1) // key.size(1)
    key = key.repeat_interleave(repeat, dim=1)
    value = value.repeat_interleave(repeat, dim=1)

    q_len, k_len = query.size(-2), key.size(-2)
    scores = query @ key.transpose(-1, -2) / (query.size(-1) ** 0.5)
    # Query i belongs at absolute position ``past + i``.  A window may have
    # discarded older keys, so key 0 is not necessarily absolute position 0.
    first_key_position = past + q_len - k_len
    causal = (first_key_position + torch.arange(k_len, device=query.device)[None, :]) <= (
        past + torch.arange(q_len, device=query.device)[:, None]
    )
    scores = scores.masked_fill(~causal, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


if __name__ == "__main__":
    torch.manual_seed(0)
    cache = KVCache(num_layers=2, max_length=128)
    # A prompt prefill: batch=1, q_heads=4, kv_heads=2, tokens=3, dim=8.
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    output = cached_attention(q, k, v, cache, layer=0)
    assert output.shape == (1, 4, 3, 8)

    # Decode one more token.  K/V length grows from 3 to 4.
    output = cached_attention(
        torch.randn(1, 4, 1, 8), torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), cache, layer=0
    )
    assert output.shape == (1, 4, 1, 8) and cache.layers[0].length == 4
    print("KV cache self-test passed")
