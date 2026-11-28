#!/usr/bin/env python3
"""Simulate an LLM optimization ledger: FlashAttention HBM traffic, quantization,
speculative decoding speedup, and one-line notes for the other levers.

Run:
    python3 example/opt_sim.py

Pure standard library: no GPU, no CUDA, no torch. Numbers are order of magnitude
(量级/示例): H100-class 3.35 TB/s HBM, 7B dense BF16 weights, seq=4096, one head
of d=128 for the attention traffic model. Every printed number is computed below
and checked by the assertions at the end.
"""

from __future__ import annotations

BITS_PER_BYTE = 8.0
HBM_GBPS = 3350.0            # GB/s, H100-class (量级)
SEQ = 4096
BATCH = 1
PARAMS_7B = 7e9
KV_PER_TOKEN_FP16 = 56 * 1024  # 57344 B: 2 * 28 layers * 4 kv_heads * 128 d * 2 B
HEAD_DIM = 128
ELEM_BYTES = 2                 # BF16 (attention traffic)
ATTN_NS = (1024, 4096, 16384, 65536)


# --------------------------------------------------------------------------- #
# 1. attention HBM traffic: naive materialize vs FlashAttention tiling
# --------------------------------------------------------------------------- #

def attn_prefill_traffic(n: int, d: int = HEAD_DIM, b: int = ELEM_BYTES) -> dict:
    """Per layer / per head.

    naive: Q/K/V/O reads+writes (4·N·d·b) + write/read the N×N score matrix and
    the N×N probability matrix (4·N²·b).
    flash: ideal tiling keeps the N×N blocks in SRAM, so HBM only moves
    Q/K/V/O once: 4·N·d·b. (简化：假定 K/V tile 在 SRAM 内被复用。)
    """
    qkvo = 4 * n * d * b
    scores = 4 * n * n * b
    return {"n": n, "tiled": qkvo, "naive": qkvo + scores,
            "ratio": (qkvo + scores) / qkvo}


def attn_decode_traffic(kv_len: int, d: int = HEAD_DIM, b: int = ELEM_BYTES) -> dict:
    """Decode: one query, KV length L. No N×N matrix exists, so naive ≈ tiled."""
    q = 1
    qkvo = (2 * q * d + 2 * kv_len * d) * b      # Q read + K/V read + O write (量级)
    scores = 2 * q * kv_len * b                   # scores/probs traffic (1×L)
    return {"kv_len": kv_len, "tiled": qkvo, "naive": qkvo + scores,
            "ratio": (qkvo + scores) / qkvo}


# --------------------------------------------------------------------------- #
# 2. quantization ledger: weights + KV, decode memory-bound time
# --------------------------------------------------------------------------- #

def quant_row(weight_bits: int, kv_bits: int) -> dict:
    weight_bytes = PARAMS_7B * weight_bits / BITS_PER_BYTE
    kv_bytes = SEQ * KV_PER_TOKEN_FP16 * kv_bits / 16
    total = weight_bytes + kv_bytes
    ms = total / (HBM_GBPS * 1e9) * 1e3
    return {"weight_bits": weight_bits, "kv_bits": kv_bits,
            "weight_gb": weight_bytes / 1e9, "kv_mb": kv_bytes / 1e6,
            "total_gb": total / 1e9, "ms": ms}


QUANT_CASES = (
    ("BF16 + FP16", 16, 16),
    ("BF16 + FP8", 16, 8),
    ("FP8 + FP8", 8, 8),
    ("INT8 + INT8", 8, 8),
    ("INT4 + FP8", 4, 8),
)


def quant_rows() -> list[dict]:
    base = quant_row(16, 16)["ms"]
    rows = []
    for name, wb, kb in QUANT_CASES:
        r = quant_row(wb, kb)
        r["name"] = name
        r["speedup"] = base / r["ms"]
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# 3. speculative decoding speedup
# --------------------------------------------------------------------------- #

def spec_ledger(alpha: float, gamma: int = 4, c: float = 0.1) -> dict:
    """Expected accepted tokens per verification step, and speedup.

    accepted = (1 - alpha^(gamma+1)) / (1 - alpha)  [gamma draft + 1 bonus]
    time    = 1 (target verify pass) + c*gamma (draft cost in target-pass units)
    speedup = accepted / time
    Assumptions: independent acceptance, no tree verification, overhead ignored.
    """
    if alpha >= 1.0:
        accepted = gamma + 1.0
    else:
        accepted = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    time_units = 1 + c * gamma
    return {"alpha": alpha, "gamma": gamma, "c": c,
            "accepted": accepted, "time": time_units, "speedup": accepted / time_units}


SPEC_ALPHAS = (0.5, 0.7, 0.9)
SPEC_COSTS = (0.1, 0.2, 0.3)


# --------------------------------------------------------------------------- #
# 4. other levers: what bottleneck does each one attack?
# --------------------------------------------------------------------------- #

OTHER_LEVERS = (
    ("scheduling", "continuous batching -> throughput；chunked prefill -> latency/TPOT"),
    ("cache", "prefix cache -> TTFT/cost；PD 分离 -> latency/隔离（KV 迁移）"),
    ("kernel/launch", "CUDA Graph -> launch；kernel fusion -> launch/HBM 流量"),
    ("parallel", "TP/PP/EP -> compute/memory/communication；overlap/offload -> 通信/容量"),
    ("memory", "KV/weight quantization -> memory bandwidth；FlashAttention -> HBM traffic"),
)


# --------------------------------------------------------------------------- #

def fmt_bytes_mb(b: float) -> str:
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    return f"{b / 1e6:.2f} MB"


def main() -> None:
    print("LLM optimization ledger (量级/示例；H100 3.35 TB/s, 7B BF16 baseline, 1 head d=128)")

    # ---- 1. attention HBM traffic --------------------------------------- #
    print("\n[1] attention HBM traffic: naive materialize vs FlashAttention tiling")
    print(f"{'seq N':>7} {'naive':>12} {'tiled':>10} {'naive/tiled':>12}")
    attn_rows = [attn_prefill_traffic(n) for n in ATTN_NS]
    for r in attn_rows:
        print(f"{r['n']:>7} {fmt_bytes_mb(r['naive']):>12} {fmt_bytes_mb(r['tiled']):>10} "
              f"{r['ratio']:>11.1f}x")
    dec = attn_decode_traffic(4096)
    print(f"decode: q=1, KV=4096 -> naive/tiled {dec['ratio']:.2f}x；没有 N×N 可省，"
          f"收益靠 FlashDecoding/split-K 提并行度。")

    # ---- 2. quantization ------------------------------------------------- #
    qrows = quant_rows()
    print(f"\n[2] quantization ledger (7B dense, seq={SEQ}, decode batch={BATCH}, "
          f"{HBM_GBPS:.0f} GB/s)")
    print(f"{'weights':<12}{'KV/tok':>8}{'W GB':>8}{'KV MB':>9}{'total GB':>10}"
          f"{'decode ms':>11}{'speedup':>9}")
    kv_bits_labels = {16: "FP16", 8: "FP8/INT8"}
    for r in qrows:
        print(f"{r['name']:<12}{kv_bits_labels[r['kv_bits']]:>8}{r['weight_gb']:>8.2f}"
              f"{r['kv_mb']:>9.1f}{r['total_gb']:>10.3f}{r['ms']:>11.2f}{r['speedup']:>8.2f}x")
    print("KV/token: FP16 56 KiB, FP8 28 KiB, INT8 28 KiB (1 B/elem；FP8/INT8 同宽，故两行相同); "
          "7B/4K decode 以权重为主，长上下文/大 batch 时 KV 占比才上升。")

    # ---- 3. speculative decoding ---------------------------------------- #
    print(f"\n[3] speculative decoding: speedup = accepted/(1+c*gamma), "
          f"accepted=(1-alpha^(gamma+1))/(1-alpha), gamma=4")
    print(f"{'alpha':>6}" + "".join(f"{'c=' + str(c):>10}" for c in SPEC_COSTS))
    spec_table = []
    for alpha in SPEC_ALPHAS:
        cells = [spec_ledger(alpha, gamma=4, c=c) for c in SPEC_COSTS]
        spec_table.append((alpha, cells))
        print(f"{alpha:>6.2f}" + "".join(f"{r['speedup']:>10.2f}" for r in cells))
    print(f"break-even: alpha=0 -> {spec_ledger(0.0, gamma=4, c=0.1)['speedup']:.2f}x；"
          "接受率低或草稿太贵时投机解码会亏。")

    # ---- 4. other levers ------------------------------------------------- #
    print("\n[4] other levers（3-5 行账本：攻击什么瓶颈）")
    for group, text in OTHER_LEVERS:
        print(f"  {group:<16} {text}")

    # ---- assertions ------------------------------------------------------ #
    assert attn_rows[-1]["ratio"] > 100, "64K prefill should be two orders faster under tiling"
    for lo, hi in zip(attn_rows, attn_rows[1:]):
        assert hi["ratio"] > lo["ratio"], "longer prefill -> bigger FlashAttention benefit"
        assert abs(hi["tiled"] / lo["tiled"] - 4) < 1e-6, "tiled traffic grows linearly with N"
        growth = hi["naive"] / lo["naive"]
        assert 14 < growth < 16, "naive traffic grows ~N^2 (x4 in N -> ~16x bytes)"
    assert dec["ratio"] < 1.1, "decode q=1 has no N×N materialization to save"
    for lo, hi in zip(qrows, qrows[1:]):
        assert hi["total_gb"] <= lo["total_gb"] + 1e-9, "lower precision must not read more bytes"
        assert hi["ms"] <= lo["ms"] + 1e-12, "fewer bytes -> lower memory-bound time"
    assert qrows[-1]["total_gb"] < qrows[0]["total_gb"] / 3
    assert qrows[-1]["speedup"] > 3.5
    for r in qrows:
        assert abs(r["ms"] - r["total_gb"] / HBM_GBPS * 1e3) < 1e-6
    for c in SPEC_COSTS:
        speeds = [spec_ledger(a, gamma=4, c=c)["speedup"] for a in SPEC_ALPHAS]
        assert speeds[0] < speeds[1] < speeds[2], "higher acceptance -> faster"
    for alpha in SPEC_ALPHAS:
        speeds = [spec_ledger(alpha, gamma=4, c=c)["speedup"] for c in SPEC_COSTS]
        assert speeds[0] > speeds[1] > speeds[2], "higher draft cost -> slower"
    assert spec_ledger(0.0, gamma=4, c=0.1)["speedup"] < 1.0

    print("\nOK: FlashAttention traffic, quantization time, speculative speedup and lever map "
          "are self-consistent.")


if __name__ == "__main__":
    main()
