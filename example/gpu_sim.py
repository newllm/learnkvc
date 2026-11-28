#!/usr/bin/env python3
"""Simulate the CPU-vs-GPU ledger: design point, memory hierarchy, LLM
roofline, launch/Amdahl limits, and the accelerator software stacks.

Run:
    python3 example/gpu_sim.py

Pure standard library: no GPU, no CUDA, no torch. Every printed number is an
order of magnitude produced by the functions below (量级，非基准) and checked
by the assertions at the end, so the web page can quote this as a source.
"""

from __future__ import annotations

from dataclasses import dataclass

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30

# --------------------------------------------------------------------------- #
# two design points (量级，非基准; dense, no sparsity)
# --------------------------------------------------------------------------- #

# CPU: 32-core server with AVX-512, dual-channel DDR5.
CPU = {
    "cores": 32,
    "lanes_per_core": 16,          # AVX-512 FP32
    "clock_ghz": 3.5,
    "fp32_tflops": 4.0,
    "fp16_tflops": 8.0,            # AVX512-FP16 / BF16, still vector units
    "bw_gbps": 80.0,               # DDR5 dual channel ~ 80 GB/s
    "l1_kib": 48,
    "l2_mib": 1.25,
    "l3_mib": 128,
    "ddr_gib": 512,
    "power_w": "150-300",
}

# GPU: H100 SXM class.
GPU = {
    "name": "H100 SXM",
    "sms": 132,
    "lanes_per_sm": 128,
    "clock_ghz": 1.8,
    "fp32_tflops": 67.0,
    "fp16_tflops": 990.0,
    "bw_gbps": 3350.0,
    "reg_kib": 256,                # register file per SM
    "shared_kib": 228,             # max shared per SM, carveout of the 256 KiB L1
    "l1_kib": "128-256",           # unified L1/shared per SM
    "l2_mib": 50,
    "hbm_gib": 80,
    "power_w": "~700",
}

CPU["lanes"] = CPU["cores"] * CPU["lanes_per_core"]
GPU["lanes"] = GPU["sms"] * GPU["lanes_per_sm"]

# GPU catalog: dense FP16/BF16 TFLOPS, HBM GB/s, VRAM GiB, L2 MiB (量级).
@dataclass(frozen=True)
class Accelerator:
    name: str
    tflops: float
    hbm_gbps: float
    vram_gib: float
    l2_mib: float | None


ACCELERATORS = (
    Accelerator("H100 SXM", 990, 3350, 80, 50),
    Accelerator("A100 SXM", 312, 2039, 80, 40),
    Accelerator("RTX 4090", 165, 1008, 24, 72),
    Accelerator("Ascend 910B", 376, 1600, 64, None),
    Accelerator("HCU (Hygon DCU)", 200, 1000, 32, None),
)

# LLM workload assumptions: 7B dense BF16, 32 layers, 56 KiB KV/token.
LLM = {
    "params": 7e9,
    "weight_bytes": 14e9,          # BF16
    "kv_per_token": 56 * KIB,
    "seq": 4096,
    "layers": 32,
    "heads": 32,
    "head_dim": 128,
}
LLM["kv_bytes"] = LLM["seq"] * LLM["kv_per_token"]

CACHE_LINE_B = 128
LAUNCH_US = 4.0                    # GPU kernel launch, 3-8 us range
CPU_OP_US = 0.2                    # small CPU task / function call


# --------------------------------------------------------------------------- #
# 1. memory hierarchy: CPU caches vs GPU caches (all 量级)
# --------------------------------------------------------------------------- #

def hierarchy_rows() -> list[dict]:
    return [
        {"level": "CPU L1", "cap": "48 KiB/core", "lat": "~4-5 cycles",
         "bw": "~100 GB/s per core", "cache": "yes", "scope": "per-core"},
        {"level": "CPU L2", "cap": "1.25 MiB/core", "lat": "~10-15 cycles",
         "bw": "~50-100 GB/s per core", "cache": "yes", "scope": "per-core"},
        {"level": "CPU L3", "cap": "128 MiB", "lat": "~30-50 cycles",
         "bw": "~300 GB/s", "cache": "yes (LLC)", "scope": "socket-wide"},
        {"level": "CPU DDR5", "cap": "512 GiB", "lat": "~80-120 ns",
         "bw": "~80 GB/s", "cache": "no", "scope": "system RAM"},
        {"level": "GPU registers", "cap": "256 KiB/SM", "lat": "~1 cycle",
         "bw": "per-SM private", "cache": "register file", "scope": "per-SM"},
        {"level": "GPU shared", "cap": "228 KiB/SM", "lat": "~20-30 cycles",
         "bw": "~33 TiB/s aggregate", "cache": "software-managed", "scope": "per-block"},
        {"level": "GPU L1", "cap": "128-256 KiB/SM", "lat": "~30-40 cycles",
         "bw": "~10-20 TiB/s", "cache": "yes, shares L1 with shared", "scope": "per-SM"},
        {"level": "GPU L2", "cap": "50 MiB", "lat": "~200-300 cycles",
         "bw": "~10 TiB/s", "cache": "yes, chip-wide", "scope": "whole GPU"},
        {"level": "GPU HBM3", "cap": "80 GiB", "lat": "~500-800 cycles",
         "bw": "3.35 TiB/s", "cache": "no", "scope": "device memory"},
    ]


# --------------------------------------------------------------------------- #
# 2. LLM workloads: roofline on CPU and GPU
# --------------------------------------------------------------------------- #

def gpu_peak_tflops() -> float:
    return GPU["fp16_tflops"]


def cpu_peak_tflops() -> float:
    """Conservative sustained BF16 work on a CPU: about half of AVX-512 peak."""
    return CPU["fp16_tflops"] / 2


def ridge_point(tflops: float, bw_gbps: float) -> float:
    """Roofline knee: FLOPs per byte where the two roofs cross."""
    return tflops * 1e12 / (bw_gbps * 1e9)


def llm_workloads() -> list[dict]:
    p, wb = LLM["params"], LLM["weight_bytes"]
    decode_flops = 2 * p * 1
    decode_bytes = wb + LLM["kv_bytes"]
    prefill_flops = 2 * p * 2048
    prefill_bytes = wb + 2048 * LLM["kv_per_token"]
    # attention QK^T + AV only: 4 * layers * heads * head_dim * seq^2
    attn_flops = 4 * LLM["layers"] * LLM["heads"] * LLM["head_dim"] * LLM["seq"] ** 2
    attn_bytes = 2 * LLM["seq"] * LLM["heads"] * LLM["head_dim"] * 2  # Q/K/V/O tiles (量级)
    cpu_tflops, cpu_bw = cpu_peak_tflops() * 1e12, CPU["bw_gbps"] * 1e9
    gpu_tflops, gpu_bw = gpu_peak_tflops() * 1e12 * 0.4, GPU["bw_gbps"] * 1e9  # 40% MFU for matmul
    rows = []
    for name, flops, nbytes in (
        ("decode 7B batch=1", decode_flops, decode_bytes),
        ("prefill 7B M=2048", prefill_flops, prefill_bytes),
        ("attention S=4096", attn_flops, attn_bytes),
    ):
        cpu_t = max(flops / cpu_tflops, nbytes / cpu_bw)
        gpu_t = max(flops / gpu_tflops, nbytes / gpu_bw)
        rows.append({"name": name, "flops": flops, "bytes": nbytes,
                     "cpu_s": cpu_t, "gpu_s": gpu_t, "ratio": cpu_t / gpu_t,
                     "ai": flops / nbytes})
    return rows


# --------------------------------------------------------------------------- #
# 3. why the GPU cannot replace the CPU
# --------------------------------------------------------------------------- #

def small_task_ledger() -> dict:
    """One tiny task: CPU function call vs GPU launch + sync."""
    return {"cpu_us": CPU_OP_US, "gpu_us": LAUNCH_US, "ratio": LAUNCH_US / CPU_OP_US}


def amdahl(serial_ms: float, parallel_ms: float, gpu_speedup: float) -> dict:
    """Serial CPU part + parallel GPU part; speed up only the parallel part."""
    old = serial_ms + parallel_ms
    new = serial_ms + parallel_ms / gpu_speedup
    return {"old_ms": old, "new_ms": new, "speedup": old / new,
            "max_speedup": old / serial_ms}


# --------------------------------------------------------------------------- #
# 4. software stacks: CUDA is a stack, not a chip
# --------------------------------------------------------------------------- #

ECOSYSTEM = (
    ("CUDA", "NVIDIA", "software stack + toolchain (not a chip)", "default backend"),
    ("ROCm + HIP", "AMD", "open stack; HIP is close to CUDA source", "ROCm backend"),
    ("CANN + AscendCL", "Huawei", "NPU stack for Ascend 910B/910C", "vLLM-Ascend / MindIE"),
    ("HCU = DTK + HIP", "Hygon", "DCU device name in HAMi; DTK/HIP stack", "DCU backend"),
    ("OpenCL / oneAPI", "Khronos/Intel", "cross-vendor compute API", "oneAPI backend"),
    ("Triton", "OpenAI (open)", "Python kernel DSL -> PTX/LLVM", "fused/custom kernels"),
)


# --------------------------------------------------------------------------- #

def fmt_time(s: float) -> str:
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.1f} ms"
    return f"{s * 1e6:.1f} us"


def fmt_bytes(b: float) -> str:
    if b >= GIB:
        return f"{b / GIB:.1f} GiB"
    if b >= MIB:
        return f"{b / MIB:.1f} MiB"
    if b >= KIB:
        return f"{b / KIB:.1f} KiB"
    return f"{b:.0f} B"


def main() -> None:
    print("CPU vs GPU ledger (量级，非基准 order of magnitude, not a benchmark)")

    # ---- 1. design point ------------------------------------------------ #
    print("\n[1] design point: CPU = latency, GPU = throughput")
    print(f"{'item':<18} {'CPU (32-core server)':<32} GPU ({GPU['name']})")
    print(f"{'cores / lanes':<18} {str(CPU['cores']) + ' cores x ' + str(CPU['lanes_per_core']) + ' FP32 lanes':<32} "
          f"{GPU['sms']} SM x {GPU['lanes_per_sm']} lanes = {GPU['lanes']}")
    print(f"{'clock / vector':<18} {str(CPU['clock_ghz']) + ' GHz / 16 FP32 (AVX-512)':<32} "
          f"~{GPU['clock_ghz']} GHz / 32 lanes per warp instr")
    print(f"{'FP32 TFLOPS':<18} {('~' + str(CPU['fp32_tflops'])).replace('.0', ''):<32} "
          f"~{GPU['fp32_tflops']:.0f} (CUDA cores)")
    print(f"{'FP16/BF16 TFLOPS':<18} {'~' + str(CPU['fp16_tflops']).replace('.0', '') + ' (AVX512-FP16)':<32} "
          f"~{GPU['fp16_tflops']:.0f} (Tensor Core, dense)")
    print(f"{'memory BW GB/s':<18} {'~' + str(CPU['bw_gbps']).replace('.0', '') + ' (DDR5 2ch)':<32} "
          f"{GPU['bw_gbps']:.0f} (HBM3)")
    print(f"{'cache':<18} {'L1/L2/L3 + 128 MiB LLC':<32} per-SM L1/shared + 50 MiB L2")
    print(f"{'power / package':<18} {CPU['power_w'] + ' W (socket)':<32} {GPU['power_w'] + ' W (module)'}")

    # ---- 2. accelerator catalog ----------------------------------------- #
    print("\n[2] accelerators at a glance (量级; dense, no sparsity; 910B/HCU 是 NPU/DCU 口径，"
          "L1/L2 命名不同，SKU 不同)")
    print(f"{'chip':<20} {'FP16/BF16 TFLOPS':>16} {'mem GB/s':>9} {'VRAM GiB':>9} {'L2 MiB':>7}")
    for a in ACCELERATORS:
        l2 = "n/p" if a.l2_mib is None else f"{a.l2_mib:.0f}"
        print(f"{a.name:<20} {a.tflops:>16.0f} {a.hbm_gbps:>9.0f} {a.vram_gib:>9.0f} {l2:>7}")

    # ---- 3. cache hierarchy --------------------------------------------- #
    print("\n[3] memory hierarchy: CPU cache vs GPU cache")
    print(f"{'level':<14} {'capacity':<16} {'latency':<16} {'bandwidth':<20} cache?")
    for r in hierarchy_rows():
        print(f"{r['level']:<14} {r['cap']:<16} {r['lat']:<16} {r['bw']:<20} {r['cache']}")
    print(f"GPU answers: L1=yes (unified with shared, carveout); L2=yes (chip-wide: H100 50, "
          f"A100 40, 4090 72 MiB; {CACHE_LINE_B} B line); HBM=no; coalescing decides line usage.")

    # ---- 4. LLM roofline ------------------------------------------------- #
    rows = llm_workloads()
    cpu_ridge = ridge_point(cpu_peak_tflops(), CPU["bw_gbps"])
    gpu_ridge = ridge_point(gpu_peak_tflops(), GPU["bw_gbps"])
    print(f"\n[4] LLM workloads: CPU vs GPU (7B BF16; CPU {cpu_peak_tflops():.0f}T/{CPU['bw_gbps']:.0f}GB/s, "
          f"GPU {gpu_peak_tflops():.0f}T/{GPU['bw_gbps']:.0f}GB/s @40% MFU)")
    print(f"{'workload':<22} {'FLOPs':>9} {'bytes':>9} {'CPU':>9} {'GPU':>9} {'ratio':>6}")
    for r in rows:
        print(f"{r['name']:<22} {r['flops'] / 1e9:>7.1f} G {fmt_bytes(r['bytes']):>9} "
              f"{fmt_time(r['cpu_s']):>9} {fmt_time(r['gpu_s']):>9} {r['ratio']:>5.0f}x")
    print(f"ridge point: CPU {cpu_ridge:.1f}, GPU {gpu_ridge:.1f} FLOP/B; decode AI={rows[0]['ai']:.2f} "
          f"(memory-bound) vs prefill AI={rows[1]['ai']:.0f} (compute-bound).")

    # ---- 5. why the GPU cannot replace the CPU --------------------------- #
    small = small_task_ledger()
    amd = amdahl(serial_ms=40, parallel_ms=100, gpu_speedup=10)
    print("\n[5] why the GPU cannot replace the CPU")
    print("single-thread / control flow: CPU ~1 ns/op with OoO + branch prediction; GPU warp waits for "
          "its slowest lane, SIMT divergence serializes both branch paths.")
    print(f"small task: CPU call ~{small['cpu_us']:.1f} us vs GPU launch {small['gpu_us']:.1f} us "
          f"(3-8 us) -> {small['ratio']:.0f}x slower; GPU cannot boot, take interrupts, do IO/page faults.")
    print(f"Amdahl: serial 40 ms + parallel 100 ms, GPU 10x -> {amd['new_ms']:.0f} ms, "
          f"speedup {amd['speedup']:.1f}x (max {amd['max_speedup']:.1f}x); CPU stays the host.")

    # ---- 6. ecosystem ---------------------------------------------------- #
    print("\n[6] software stacks: CUDA is a stack, not a chip; 昇腾=Ascend（非「升腾」）；"
          "vLLM/SGLang 用 attention backend + custom ops 对接")
    print(f"{'stack':<18} {'vendor':<14} {'what it is':<46} framework path")
    for name, vendor, what, path in ECOSYSTEM:
        print(f"{name:<18} {vendor:<14} {what:<46} {path}")

    # ---- assertions ------------------------------------------------------ #
    assert CPU["lanes"] < GPU["lanes"] / 20
    assert CPU["bw_gbps"] * 20 < GPU["bw_gbps"]
    assert CPU["fp16_tflops"] * 50 < GPU["fp16_tflops"]
    assert CPU["bw_gbps"] < GPU["bw_gbps"] and CPU["bw_gbps"] < ACCELERATORS[3].hbm_gbps / 10
    by_name = {a.name: a for a in ACCELERATORS}
    assert by_name["H100 SXM"].l2_mib == 50
    assert by_name["A100 SXM"].l2_mib == 40
    assert by_name["RTX 4090"].l2_mib == 72
    assert all(r["cache"] == "no" for r in hierarchy_rows() if r["level"] in ("CPU DDR5", "GPU HBM3"))
    assert all(r["cache"].startswith("yes") for r in hierarchy_rows()
               if r["level"] in ("CPU L1", "CPU L2", "CPU L3", "GPU L1", "GPU L2"))
    assert rows[0]["ai"] < cpu_ridge < rows[1]["ai"]      # decode memory, prefill compute
    assert rows[0]["ratio"] > 20                       # DDR vs HBM bandwidth
    assert rows[1]["ratio"] > 50 and rows[2]["ratio"] > 50
    assert small["ratio"] > 10 and small["gpu_us"] >= 3 and small["gpu_us"] <= 8
    assert amd["speedup"] < amd["max_speedup"] < 10
    assert [e[0] for e in ECOSYSTEM] == ["CUDA", "ROCm + HIP", "CANN + AscendCL",
                                         "HCU = DTK + HIP", "OpenCL / oneAPI", "Triton"]

    print("\nOK: design point, cache hierarchy, LLM ratios, launch limits and stacks are self-consistent.")


if __name__ == "__main__":
    main()
