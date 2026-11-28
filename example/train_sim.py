#!/usr/bin/env python3
"""Simulate per-GPU training memory for one step.

Run:
    python3 example/train_sim.py

Pure standard library. It does not allocate tensors; it keeps the ledger:

    weights + grads + optimizer state + activations + overhead

and applies DP / ZeRO-1/2/3 / TP / PP / activation checkpointing as sharding
and recomputation factors. The numbers are estimates, but the accounting is
explicit and the assertions at the bottom pin down the important ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GB = 1e9


@dataclass(frozen=True)
class Model:
    name: str = "7B"
    params: float = 7.62e9
    layers: int = 28
    hidden: int = 4096
    seq: int = 2048
    micro_batch: int = 1
    weight_bytes: int = 2  # BF16
    grad_bytes: int = 2  # BF16
    opt_bytes: int = 12  # AdamW m/v fp32 (8) + fp32 master (4)


@dataclass(frozen=True)
class Parallel:
    dp: int = 8
    tp: int = 1
    pp: int = 1
    zero: int = 0  # 0, 1, 2, 3
    ckpt: bool = False
    overhead_gb: float = 2.0
    gpu_gb: float = 80.0


def ledger(model: Model, par: Parallel) -> dict:
    shard = par.tp * par.pp
    weights = model.params * model.weight_bytes / shard
    if par.zero >= 3:
        weights /= par.dp

    grads = model.params * model.grad_bytes / shard
    if par.zero >= 2:
        grads /= par.dp

    opt = model.params * model.opt_bytes / shard
    if par.zero >= 1:
        opt /= par.dp

    per_layer = model.micro_batch * model.seq * model.hidden * 12 * model.weight_bytes / par.tp
    if par.ckpt:
        per_layer *= 0.35  # store layer inputs, recompute the rest
    act = per_layer * math.ceil(model.layers / par.pp)

    overhead = par.overhead_gb * GB
    base = weights + grads + opt + overhead
    peak = base + act
    usable = par.gpu_gb * 0.9 * GB
    return {
        "weights": weights,
        "grads": grads,
        "opt": opt,
        "act": act,
        "peak": peak,
        "usable": usable,
        "fit": peak <= usable,
    }


def step_trace(model: Model, par: Parallel) -> list[tuple[str, float]]:
    """A coarse forward/backward memory trace, in GB."""
    row = ledger(model, par)
    base = row["weights"] + row["grads"] + row["opt"] + par.overhead_gb * GB
    per_layer = row["act"] / max(1, math.ceil(model.layers / par.pp))
    live = base
    trace = [("start (weights+grads+opt)", live / GB)]
    for i in range(math.ceil(model.layers / par.pp)):
        live += per_layer
        if i in (0, math.ceil(model.layers / par.pp) - 1):
            trace.append((f"forward layer {i + 1}", live / GB))
    trace.append(("backward (peak, before frees)", live / GB))
    trace.append(("optimizer (after frees)", base / GB))
    return trace


def scenario(name: str, par: Parallel) -> tuple[str, dict]:
    return name, ledger(Model(), par)


def main() -> None:
    model = Model()
    print(f"model={model.name} params={model.params/1e9:.2f}B layers={model.layers} hidden={model.hidden} seq={model.seq}")
    print("one step: forward -> loss -> backward -> AdamW; per-GPU estimates in GB")
    print(f"{'scenario':<22} {'weights':>8} {'grads':>7} {'opt':>7} {'act':>7} {'peak':>7} {'fit 72GB':>10}")

    scenarios = [
        ("DP=8  ZeRO-0", Parallel(zero=0)),
        ("DP=8  ZeRO-1", Parallel(zero=1)),
        ("DP=8  ZeRO-2", Parallel(zero=2)),
        ("DP=8  ZeRO-3", Parallel(zero=3)),
        ("ZeRO-3 + ckpt", Parallel(zero=3, ckpt=True)),
        ("TP=8  ZeRO-0", Parallel(dp=1, tp=8, zero=0)),
        ("PP=8  ZeRO-0", Parallel(dp=1, pp=8, zero=0)),
    ]
    rows = {}
    for name, par in scenarios:
        r = ledger(model, par)
        rows[name] = r
        print(
            f"{name:<22} {r['weights']/GB:>8.1f} {r['grads']/GB:>7.1f} {r['opt']/GB:>7.1f} "
            f"{r['act']/GB:>7.1f} {r['peak']/GB:>7.1f} {('yes' if r['fit'] else 'no'):>10}"
        )

    print("\nstep trace for ZeRO-2 (DP=8):")
    for label, gib in step_trace(model, Parallel(zero=2)):
        print(f"  {label:<28} {gib:>7.1f} GB")

    print("\nwhy:")
    print("  ZeRO-1 shards optimizer state, ZeRO-2 also shards grads, ZeRO-3 also shards weights.")
    print("  TP/PP shard the model, but TP adds per-layer all-reduce and PP has pipeline bubbles.")
    print("  ckpt stores layer inputs and recomputes activations: ~30% more compute for less memory.")

    # Assertions: the ordering that the teaching depends on.
    assert rows["DP=8  ZeRO-1"]["opt"] < rows["DP=8  ZeRO-0"]["opt"]
    assert rows["DP=8  ZeRO-2"]["grads"] < rows["DP=8  ZeRO-1"]["grads"]
    assert rows["DP=8  ZeRO-3"]["weights"] < rows["DP=8  ZeRO-2"]["weights"]
    assert rows["ZeRO-3 + ckpt"]["peak"] < rows["DP=8  ZeRO-3"]["peak"]
    assert rows["DP=8  ZeRO-3"]["fit"] and not rows["DP=8  ZeRO-0"]["fit"]
    print("\nOK: sharding reduces the right term, and checkpointing lowers the peak.")


if __name__ == "__main__":
    main()
