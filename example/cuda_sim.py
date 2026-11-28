#!/usr/bin/env python3
"""Simulate the CUDA execution ledger: occupancy, coalescing, roofline,
CUDA Graph launch overhead, and the decode byte account.

Run:
    python3 example/cuda_sim.py

Pure standard library: no GPU, no CUDA, no torch. The point is to make the
GPU execution arithmetic visible on an H100-class device with a 30B-class
MoE model: how many blocks/warps fit per SM, how many sectors a warp request
touches, which roofline slope decode lives on, and what CUDA Graph saves.
"""

from __future__ import annotations

import math

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30

# H100 SXM class. 数量级，非基准 (order of magnitude, not a benchmark).
SMS = 132
MAX_WARPS_PER_SM = 64
MAX_BLOCKS_PER_SM = 32
REGS_PER_SM = 65536
SMEM_PER_SM = 228 * KIB          # 228 KiB
L2_BYTES = 50 * MIB
HBM_BYTES = 80 * GIB
HBM_GBPS = 3350.0                # GB/s (HBM3, H100 SXM)
BF16_TFLOPS = 990.0              # dense BF16 tensor core

# 30B-class MoE: 3.35B active params, 48 layers, 96 KiB KV per token (FP16).
ACTIVE_PARAMS = 3.35e9
LAYERS = 48
KV_PER_TOKEN = 96 * KIB
BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)


# --------------------------------------------------------------------------- #
# 1. occupancy: how many blocks / warps fit on one SM
# --------------------------------------------------------------------------- #

def occupancy(block: int, regs_per_thread: int = 32, smem_per_block: int = 16 * KIB) -> dict:
    """Occupancy of one block shape under the four SM limits."""
    warps = math.ceil(block / 32)
    # Registers are allocated per warp in 256-register units (8 regs/thread).
    regs_per_warp = math.ceil(regs_per_thread * 32 / 256) * 256
    regs_per_block = regs_per_warp * warps
    limits = {
        "regs": REGS_PER_SM // regs_per_block,
        "smem": SMEM_PER_SM // smem_per_block if smem_per_block else MAX_BLOCKS_PER_SM,
        "blocks": MAX_BLOCKS_PER_SM,
        "warps": MAX_WARPS_PER_SM // warps,
    }
    binding = min(limits, key=limits.get)
    resident = max(0, limits[binding])
    active_warps = resident * warps
    return {
        "block": block,
        "warps": warps,
        "regs_per_block": regs_per_block,
        "limits": limits,
        "binding": binding,
        "resident": resident,
        "active_warps": active_warps,
        "occupancy": active_warps / MAX_WARPS_PER_SM,
    }


# --------------------------------------------------------------------------- #
# 2. coalescing: sectors touched by one 32-thread load
# --------------------------------------------------------------------------- #

SECTOR_BYTES = 32          # L1/L2 的 sector 粒度
THREADS_PER_WARP = 32
ELEMENT_BYTES = 4          # FP32 / int32 元素


def coalesce_sectors(stride: int) -> int:
    """Distinct 32 B sectors touched by 32 threads spaced ``stride`` elements apart."""
    sectors = {(t * stride * ELEMENT_BYTES) // SECTOR_BYTES for t in range(THREADS_PER_WARP)}
    return len(sectors)


def coalesce_row(stride: int) -> dict:
    sectors = coalesce_sectors(stride)
    useful = THREADS_PER_WARP * ELEMENT_BYTES
    moved = sectors * SECTOR_BYTES
    return {"stride": stride, "sectors": sectors, "moved": moved, "useful": useful,
            "util": useful / moved}


# --------------------------------------------------------------------------- #
# 3. roofline: arithmetic intensity and the bound
# --------------------------------------------------------------------------- #

RIDGE = BF16_TFLOPS * 1e12 / (HBM_GBPS * 1e9)   # FLOP per byte


def roofline(m_tokens: int, kv_bytes: int = 4096 * KV_PER_TOKEN) -> dict:
    """Decode (M=1) vs prefill (M<=seq) on the roofline model.

    Only the weight matmuls are counted (2 FLOP per active parameter per
    token); the attention O(M^2) term is ignored on purpose.
    """
    flops = 2 * ACTIVE_PARAMS * m_tokens
    weight_bytes = ACTIVE_PARAMS * 2              # BF16
    bytes_moved = weight_bytes + kv_bytes
    ai = flops / bytes_moved
    t_mem = bytes_moved / (HBM_GBPS * 1e9)
    t_compute = flops / (BF16_TFLOPS * 1e12)
    return {
        "m": m_tokens,
        "flops": flops,
        "bytes": bytes_moved,
        "ai": ai,
        "bound": "compute" if ai >= RIDGE else "memory",
        "t_mem": t_mem,
        "t_compute": t_compute,
    }


# --------------------------------------------------------------------------- #
# 4. CUDA Graph: CPU launch overhead vs one graph replay
# --------------------------------------------------------------------------- #

KERNELS_PER_LAYER = 8
LAUNCH_US = 4.0            # eager per-kernel CPU launch cost (3-8 us 区间)
KERNEL_US = 3.0            # tiny decode kernel GPU time
GRAPH_LAUNCH_US = 6.0      # one cudaGraphLaunch for the whole step


def graph_ledger(layers: int = LAYERS, per_layer: int = KERNELS_PER_LAYER) -> dict:
    kernels = layers * per_layer
    eager_cpu = kernels * LAUNCH_US
    gpu_busy = kernels * KERNEL_US
    eager_step = max(eager_cpu, gpu_busy)
    graph_step = GRAPH_LAUNCH_US + gpu_busy
    return {
        "kernels": kernels,
        "eager_cpu_us": eager_cpu,
        "gpu_busy_us": gpu_busy,
        "eager_step_us": eager_step,
        "graph_cpu_us": GRAPH_LAUNCH_US,
        "graph_step_us": graph_step,
        "saving": 1 - GRAPH_LAUNCH_US / eager_cpu,
        "speedup": eager_step / graph_step,
    }


# --------------------------------------------------------------------------- #
# 5. decode byte account (weights + KV), FP16 vs FP8 KV
# --------------------------------------------------------------------------- #

def decode_step(batch: int, seq: int = 4096, kv_bytes_per_token: int = KV_PER_TOKEN) -> dict:
    weight_bytes = ACTIVE_PARAMS * 2                        # BF16 weights
    kv_bytes = batch * seq * kv_bytes_per_token
    total = weight_bytes + kv_bytes
    ms = total / (HBM_GBPS * 1e9) * 1e3
    return {
        "batch": batch,
        "weight_bytes": weight_bytes,
        "kv_bytes": kv_bytes,
        "total": total,
        "ms": ms,
        "ms_per_token": ms / batch,
    }


# --------------------------------------------------------------------------- #

def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> None:
    print("CUDA execution ledger (H100-class; 数量级，非基准 order of magnitude, not a benchmark)")
    print(f"{SMS} SM | {MAX_WARPS_PER_SM} warps/SM (warp = {THREADS_PER_WARP} threads) | "
          f"{MAX_BLOCKS_PER_SM} blocks/SM | {REGS_PER_SM} regs/SM | {SMEM_PER_SM // KIB} KiB smem/SM")
    print(f"model: 30B-class MoE, {ACTIVE_PARAMS / 1e9:.2f}B active params, "
          f"{KV_PER_TOKEN // KIB} KiB KV/token, BF16 weights")

    # 1) occupancy scan
    print(f"\n[1] occupancy: block-size scan (regs/thread=32, shared/block=16 KiB)")
    print(f"{'block':>6} {'warps':>5} {'regs/blk':>8} {'(reg,smem,blk)':>15} "
          f"{'resident':>8} {'warps':>5} {'occ':>6}  binding")
    rows = []
    for block in BLOCK_SIZES:
        r = occupancy(block)
        rows.append(r)
        lim = r["limits"]
        print(f"{block:>6} {r['warps']:>5} {r['regs_per_block']:>8} "
              f"({lim['regs']:>3},{lim['smem']:>3},{lim['blocks']:>2}) "
              f"{r['resident']:>8} {r['active_warps']:>5} {r['occupancy'] * 100:>5.1f}%  "
              f"{r['binding']}")

    # 2) memory hierarchy
    print("\n[2] memory hierarchy (order of magnitude, not a benchmark)")
    print(f"{'level':<10} {'capacity':<14} {'latency':<16} bandwidth")
    print(f"{'register':<10} {'256 KiB/SM':<14} {'~1 cycle':<16} per-SM private")
    print(f"{'shared':<10} {'228 KiB/SM':<14} {'~20-30 cycles':<16} ~33 TiB/s aggregate")
    print(f"{'L2':<10} {'50 MiB':<14} {'~200-300 cycles':<16} ~10 TiB/s")
    print(f"{'HBM3':<10} {'80 GiB':<14} {'~500-800 cycles':<16} "
          f"{HBM_GBPS / 1000:.2f} TiB/s ({HBM_GBPS:.0f} GB/s)")

    # 3) coalescing
    print(f"\n[3] coalescing: 32 threads x {ELEMENT_BYTES} B, sector = {SECTOR_BYTES} B")
    print(f"{'stride':>6} {'sectors':>7} {'moved':>8} {'useful':>8} {'util':>7}")
    for stride in (1, 2, 4, 8, 32):
        c = coalesce_row(stride)
        print(f"{c['stride']:>6} {c['sectors']:>7} {c['moved']:>7} B {c['useful']:>7} B "
              f"{c['util'] * 100:>6.1f}%")

    # 4) roofline
    print(f"\n[4] roofline: ridge = {BF16_TFLOPS:.0f} TFLOPS / {HBM_GBPS / 1000:.2f} TB/s "
          f"= {RIDGE:.1f} FLOP/B")
    print(f"{'workload':<9} {'M':>5} {'AI FLOP/B':>10} {'bound':>9} {'t_mem':>9} {'t_compute':>10}")
    decode = roofline(1)
    prefill = roofline(4096)
    for name, r in (("decode", decode), ("prefill", prefill)):
        print(f"{name:<9} {r['m']:>5} {r['ai']:>10.2f} {r['bound']:>9} "
              f"{r['t_mem'] * 1e3:>8.2f} ms {r['t_compute'] * 1e3:>8.2f} ms")

    # 5) CUDA Graph
    g = graph_ledger()
    print(f"\n[5] CUDA Graph: {LAYERS} layers x {KERNELS_PER_LAYER} kernels = "
          f"{g['kernels']} launches per decode step")
    print(f"eager: {g['kernels']} x {LAUNCH_US:.1f} us = {g['eager_cpu_us']:.0f} us CPU, "
          f"GPU busy {g['gpu_busy_us']:.0f} us -> step {g['eager_step_us']:.0f} us (CPU-bound)")
    print(f"graph replay: 1 launch ~= {g['graph_cpu_us']:.0f} us CPU -> "
          f"step {g['graph_step_us']:.0f} us (GPU-bound)")
    print(f"CPU launch overhead saved {g['saving'] * 100:.1f}%; "
          f"step speedup {g['speedup']:.2f}x")

    # 6) inference link: decode bytes = weights + batch x seq x KV/token
    print("\n[6] decode byte account: bytes = active weights + batch x seq x KV/token")
    kv1 = decode_step(1)
    kv32 = decode_step(32)
    print(f"batch  1, seq 4096, FP16 KV: {gb(kv1['weight_bytes'])} + {gb(kv1['kv_bytes'])} "
          f"= {gb(kv1['total'])} -> {kv1['ms']:.2f} ms/step")
    print(f"batch 32, seq 4096, FP16 KV: {gb(kv32['weight_bytes'])} + {gb(kv32['kv_bytes'])} "
          f"= {gb(kv32['total'])} -> {kv32['ms']:.2f} ms/step "
          f"({kv32['ms_per_token']:.2f} ms/token)")
    kv32_fp8 = decode_step(32, kv_bytes_per_token=KV_PER_TOKEN // 2)
    print(f"KV FP16 -> FP8 halves KV bytes: {gb(kv32['kv_bytes'])} -> "
          f"{gb(kv32_fp8['kv_bytes'])}, total {gb(kv32_fp8['total'])}")
    print("paged gather: block_table entries are scattered, so one logical read "
          "can touch ~8x sectors (see [3])")

    # Assertions make this a ledger, not a pile of prints.
    for r in rows:
        assert r["warps"] == math.ceil(r["block"] / 32)
        assert 0 < r["occupancy"] <= 1, r
    occ = {r["block"]: r["occupancy"] for r in rows}
    assert abs(max(occ.values()) - 1.0) < 1e-9
    assert occ[32] < 0.25                       # 32-thread blocks: smem caps at 14 blocks -> 22%
    assert rows[-1]["warps"] == 32              # 1024 threads = 32 warps
    assert coalesce_sectors(1) == 4
    assert coalesce_sectors(2) == 8
    assert coalesce_sectors(32) == 32
    assert coalesce_row(1)["util"] == 1.0
    assert coalesce_row(2)["util"] < coalesce_row(1)["util"]
    assert decode["ai"] < RIDGE < prefill["ai"]
    assert decode["bound"] == "memory" and prefill["bound"] == "compute"
    assert g["kernels"] == LAYERS * KERNELS_PER_LAYER
    assert g["saving"] > 0.9 and g["graph_step_us"] < g["eager_step_us"]
    assert kv32_fp8["kv_bytes"] * 2 == kv32["kv_bytes"]
    assert kv32["total"] > kv1["total"]

    print("\nOK: occupancy, sectors, roofline bounds, graph saving and decode bytes are self-consistent.")


if __name__ == "__main__":
    main()
