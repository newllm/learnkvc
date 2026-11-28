#!/usr/bin/env python3
"""Simulate NCCL collectives: ring/tree all-reduce, alpha-beta crossover, ledger.

Run:
    python3 example/nccl_sim.py

Pure standard library: no GPU, no NCCL, no torch. Three experiments:
1. ring all-reduce schedule (reduce-scatter + all-gather), step by step;
2. tree all-reduce / broadcast: levels, steps, alpha-beta time vs ring;
3. message-size sweep, crossover, collective ledger, and the TP/decode link.

Every printed number is produced by the functions below and checked by the
assertions at the end, so the web page can quote this as a single source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30

RANKS = 8          # example cluster size
MSG = 128 * MIB    # example all-reduce message size
LATENCY_US = 2.0   # per-hop latency
BW_GBPS = 400.0    # per-port bandwidth: 400 Gb/s = 50 GB/s


@dataclass(frozen=True)
class Link:
    """One network hop: alpha = latency, beta = 1/bandwidth."""

    gbps: float = BW_GBPS
    latency_us: float = LATENCY_US

    @property
    def bytes_per_s(self) -> float:
        return self.gbps / 8 * 1e9

    @property
    def gbs(self) -> float:
        return self.gbps / 8

    @property
    def latency_s(self) -> float:
        return self.latency_us / 1e6


def hop_time(link: Link, nbytes: float) -> float:
    """alpha-beta model for one step: fixed latency + bytes / bandwidth."""
    return link.latency_s + nbytes / link.bytes_per_s


def fmt_bytes(b: float) -> str:
    if b >= GIB:
        return f"{b / GIB:.2f} GiB"
    if b >= MIB:
        return f"{b / MIB:.1f} MiB"
    if b >= KIB:
        return f"{b / KIB:.1f} KiB"
    return f"{b:.0f} B"


def fmt_time(s: float) -> str:
    if s >= 1:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.2f} us"


# --------------------------------------------------------------------------- #
# ring all-reduce
# --------------------------------------------------------------------------- #

def ring_schedule(n: int, size: float, link: Link) -> list[dict]:
    """One row per ring step: which chunk each rank sends to its right neighbour."""
    chunk = size / n
    rows = []
    for s in range(2 * (n - 1)):
        if s < n - 1:
            phase, local = "reduce-scatter", s
            chunk_id = lambda r, local=local: (r - local) % n
        else:
            phase, local = "all-gather", s - (n - 1)
            chunk_id = lambda r, local=local: (r - local + 1) % n
        rows.append(
            {
                "step": s + 1,
                "phase": phase,
                "phase_step": local + 1,
                "sends": [(r, chunk_id(r), (r + 1) % n) for r in range(n)],
                "bytes": chunk,
                "cum_bytes": (s + 1) * chunk,
                "time": hop_time(link, chunk),
                "cum_time": (s + 1) * hop_time(link, chunk),
            }
        )
    return rows


def ring_stats(n: int, size: float, link: Link) -> dict:
    steps = 2 * (n - 1)
    chunk = size / n
    seconds = steps * hop_time(link, chunk)
    per_rank = size * 2 * (n - 1) / n
    return {
        "steps": steps,
        "chunk": chunk,
        "per_rank": per_rank,
        "network": per_rank * n,
        "seconds": seconds,
        "algbw": size / seconds,
        "busbw": per_rank / seconds,
    }


# --------------------------------------------------------------------------- #
# tree all-reduce: reduce up, broadcast down
# --------------------------------------------------------------------------- #

def tree_schedule(n: int) -> list[dict]:
    """Binary-tree levels: reduce phases, then broadcast phases (whole S per step)."""
    levels = int(math.log2(n))
    rows = []
    for k in range(1, levels + 1):
        sends = [(r, r - 2 ** (k - 1)) for r in range(n) if r % 2**k == 2 ** (k - 1)]
        rows.append({"phase": "reduce", "level": k, "sends": sends})
    for k in range(1, levels + 1):
        sends = [
            (r, r + 2 ** (levels - k))
            for r in range(n)
            if r % 2 ** (levels - k + 1) == 0
        ]
        rows.append({"phase": "broadcast", "level": k, "sends": sends})
    return rows


def tree_stats(n: int, size: float, link: Link) -> dict:
    levels = int(math.log2(n))
    steps = 2 * levels
    seconds = steps * hop_time(link, size)
    per_rank = size * 2 * (n - 1) / n  # same ledger convention as ring
    return {
        "levels": levels,
        "steps": steps,
        "seconds": seconds,
        "per_rank": per_rank,
        "algbw": size / seconds,
        "busbw": per_rank / seconds,
    }


def tree_broadcast_stats(n: int, size: float, link: Link) -> dict:
    levels = int(math.log2(n))
    return {"steps": levels, "seconds": levels * hop_time(link, size)}


def crossover_bytes(n: int, link: Link) -> float:
    """S where ring and tree all-reduce cost the same; 0 means tree never wins."""
    levels = int(math.log2(n))
    denom = levels - (n - 1) / n
    if denom <= 0:
        return 0.0
    return max(0.0, link.bytes_per_s * link.latency_s * ((n - 1) - levels) / denom)


# --------------------------------------------------------------------------- #
# collectives ledger + inference link
# --------------------------------------------------------------------------- #

def collective_ledger(n: int, size: float) -> list[dict]:
    """Per-rank bytes and ring steps for the six collectives this chapter covers."""
    return [
        {"name": "AllReduce", "bytes": size * 2 * (n - 1) / n, "steps": 2 * (n - 1),
         "formula": "2(N-1)/N*S", "use": "TP per layer; gradient sync"},
        {"name": "AllGather", "bytes": size * (n - 1) / n, "steps": n - 1,
         "formula": "(N-1)/N*S", "use": "FSDP params; expert weights"},
        {"name": "ReduceScatter", "bytes": size * (n - 1) / n, "steps": n - 1,
         "formula": "(N-1)/N*S", "use": "ZeRO grads; ring all-reduce phase 1"},
        {"name": "AllToAll", "bytes": size * (n - 1) / n, "steps": n - 1,
         "formula": "(N-1)/N*S", "use": "MoE token dispatch/combine (EP)"},
        {"name": "Broadcast", "bytes": size / n, "steps": int(math.log2(n)),
         "formula": "S/N avg (root sends S)", "use": "weights, config, seeds"},
        {"name": "SendRecv", "bytes": size, "steps": 1,
         "formula": "S", "use": "PP activations; PD KV transfer"},
    ]


def tp_message_bytes(hidden: int = 2048, batch: int = 1, elem_bytes: int = 2) -> int:
    """TP all-reduce message in decode: batch x hidden x dtype bytes."""
    return batch * hidden * elem_bytes


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def main() -> None:
    link = Link(gbps=BW_GBPS, latency_us=LATENCY_US)
    n, size = RANKS, MSG

    print(
        f"NCCL collectives: N={n}, S={size / MIB:.0f} MiB, "
        f"link {link.gbps:.0f} Gb/s ({link.gbs:.0f} GB/s), {link.latency_us:.0f} us/hop"
    )

    # ---- 1. ring all-reduce walkthrough ---------------------------------- #
    st = ring_stats(n, size, link)
    rows = ring_schedule(n, size, link)
    print(f"\n[1] ring all-reduce: N={n}, S={size / MIB:.0f} MiB, chunk = S/N = {fmt_bytes(st['chunk'])}")
    print(f"{'step':>4} {'phase':<15} {'bytes':>9} {'cum/rank':>9} {'cum time':>9}  sends (rank:chunk->next)")
    for row in rows:
        sends = " ".join(f"{r}:c{c}->{d}" for r, c, d in row["sends"])
        print(
            f"{row['step']:>4} {row['phase']:<15} {fmt_bytes(row['bytes']):>9} "
            f"{fmt_bytes(row['cum_bytes']):>9} {fmt_time(row['cum_time']):>9}  {sends}"
        )
    print(
        f"ring total: {st['steps']} steps, {fmt_time(st['seconds'])}; "
        f"per rank {fmt_bytes(st['per_rank'])}; algbw {st['algbw'] / 1e9:.2f} GB/s; "
        f"busbw {st['busbw'] / 1e9:.2f} GB/s"
    )
    print(
        f"assert per-rank = 2(N-1)/N*S = {fmt_bytes(st['per_rank'])}; "
        f"network = N*per-rank = {fmt_bytes(st['network'])}"
    )

    # ---- 2. tree all-reduce / broadcast ---------------------------------- #
    tst = tree_stats(n, size, link)
    print(f"\n[2] tree all-reduce: reduce up {tst['levels']} levels + broadcast down {tst['levels']} levels")
    print(f"{'step':>4} {'phase':<11} {'level':>5}  senders -> receivers")
    tree_rows = tree_schedule(n)
    for i, row in enumerate(tree_rows, 1):
        pairs = ", ".join(f"{s}->{d}" for s, d in row["sends"])
        print(f"{i:>4} {row['phase']:<11} {row['level']:>5}  {pairs}")
    bst = tree_broadcast_stats(n, size, link)
    print(
        f"tree all-reduce: {tst['steps']} steps, {fmt_time(tst['seconds'])}; "
        f"tree broadcast only: {bst['steps']} steps, {fmt_time(bst['seconds'])}"
    )
    print(
        f"same message: ring {fmt_time(st['seconds'])} vs tree {fmt_time(tst['seconds'])} "
        f"-> ring {tst['seconds'] / st['seconds']:.2f}x faster at this size"
    )

    # ---- 3. alpha-beta sweep --------------------------------------------- #
    sizes = [1 * KIB, 64 * KIB, 1 * MIB, 16 * MIB, 1 * GIB]
    print(f"\n[3] alpha-beta sweep: N={n}, {link.latency_us:.0f} us/hop, {link.gbs:.0f} GB/s")
    print(f"{'size':>9} {'ring':>10} {'tree':>10} {'ring busbw':>12} {'tree busbw':>12}  winner")
    sweep = []
    for s in sizes:
        r = ring_stats(n, s, link)
        t = tree_stats(n, s, link)
        winner = "tree" if t["seconds"] < r["seconds"] else "ring"
        sweep.append({"size": s, "ring": r, "tree": t, "winner": winner})
        print(
            f"{fmt_bytes(s):>9} {fmt_time(r['seconds']):>10} {fmt_time(t['seconds']):>10} "
            f"{r['busbw'] / 1e9:>9.2f} GB/s {t['busbw'] / 1e9:>9.2f} GB/s  {winner}"
        )
    cross = crossover_bytes(n, link)
    print(
        f"crossover (N={n}): {fmt_bytes(cross)} -> tree wins below, ring above; "
        f"ring steps={st['steps']}, tree steps={tst['steps']}"
    )

    # ---- 4. collective ledger -------------------------------------------- #
    ledger = collective_ledger(n, size)
    print(f"\n[4] collective ledger (S={size / MIB:.0f} MiB, N={n})")
    lr = ledger[0]
    print(
        f"AllReduce: per rank {lr['formula']} = {fmt_bytes(lr['bytes'])}, {lr['steps']} steps (ring); "
        f"AllGather/ReduceScatter/AllToAll: (N-1)/N*S = {fmt_bytes(ledger[1]['bytes'])}, "
        f"{ledger[1]['steps']} steps"
    )
    print(
        f"Broadcast: root sends S, ~{ledger[4]['steps']} tree steps; "
        f"SendRecv: {fmt_bytes(ledger[5]['bytes'])}, 1 step | "
        f"one ring step moves chunk = S/N = {fmt_bytes(st['chunk'])}"
    )
    print(
        f"TP usage: AllReduce x2/layer; EP usage: AllToAll x2/layer; "
        f"PD usage: SendRecv once per request"
    )

    # ---- 5. inference link ----------------------------------------------- #
    tp_bytes = tp_message_bytes()
    r4 = ring_stats(n, tp_bytes, link)
    t4 = tree_stats(n, tp_bytes, link)
    big = tp_message_bytes(hidden=2048, batch=2048)  # prefill: batch x seq = 2048
    print(f"\n[5] inference link: TP all-reduce, batch=1, hidden=2048, FP16")
    print(
        f"decode message = batch*hidden*2 B = {tp_bytes} B = {fmt_bytes(tp_bytes)} per layer; "
        f"ring {fmt_time(r4['seconds'])}, tree {fmt_time(t4['seconds'])} per layer"
    )
    print(
        f"prefill seq=2048 message = {fmt_bytes(big)} per layer; "
        f"ring {fmt_time(ring_stats(n, big, link)['seconds'])}, tree {fmt_time(tree_stats(n, big, link)['seconds'])}"
    )
    print(
        "conclusion: decode TP all-reduce is latency-bound (tree/LL/NVLS, same-node), "
        "prefill and DP gradients are bandwidth-bound (ring/Simple)"
    )

    # ---- assertions: the printed ledger must be self-consistent ---------- #
    assert st["steps"] == 14
    assert abs(st["per_rank"] - size * 2 * (n - 1) / n) < 1e-6
    assert abs(st["network"] - 2 * (n - 1) * size) < 1e-6
    assert 0.9 * link.bytes_per_s < st["busbw"] < link.bytes_per_s
    assert abs(st["algbw"] - size / st["seconds"]) < 1e-9
    assert tst["steps"] == 6 and tst["levels"] == 3
    assert tst["seconds"] > st["seconds"]
    assert sweep[0]["winner"] == "tree" and sweep[-1]["winner"] == "ring"
    for prev, cur in zip(sweep, sweep[1:]):
        assert cur["ring"]["seconds"] > prev["ring"]["seconds"]
        assert cur["tree"]["seconds"] > prev["tree"]["seconds"]
        assert cur["ring"]["busbw"] > prev["ring"]["busbw"]
        assert cur["tree"]["busbw"] > prev["tree"]["busbw"]
    assert 64 * KIB < cross < 1 * MIB
    assert tree_stats(n, tp_bytes, link)["seconds"] < ring_stats(n, tp_bytes, link)["seconds"]
    assert abs(ledger[0]["bytes"] - st["per_rank"]) < 1e-6

    print(
        f"\nOK: ring {st['steps']} steps / {fmt_time(st['seconds'])}; "
        f"tree {tst['steps']} steps; crossover {fmt_bytes(cross)}; ledger self-consistent."
    )


if __name__ == "__main__":
    main()
