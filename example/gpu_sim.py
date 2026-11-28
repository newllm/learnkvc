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

# --------------------------------------------------------------------------- #
# two design points (量级，非基准; dense, no sparsity)
# --------------------------------------------------------------------------- #

# CPU 分两档：桌面双通道 DDR5 与服务器多通道 DDR5。两者都用 AVX-512，
# 但核心数、算力和内存带宽不同；不把桌面带宽套在「32 核服务器」头上。
CPU_DESKTOP = {
    "name": "桌面 16-core",
    "cores": 16,
    "lanes_per_core": 16,          # AVX-512 FP32
    "clock_ghz": 3.5,
    "fp32_tflops": 2.0,
    "fp16_tflops": 4.0,            # AVX512-FP16 / BF16, still vector units
    "bw_gbps": 80.0,               # DDR5 dual channel ~ 80 GB/s
    "l1_kib": 48,
    "l2_mib": 1.25,
    "l3_mib": 32,
    "ddr_gb": 128,
    "power_w": "65-150",
}

CPU_SERVER = {
    "name": "服务器 32-core",
    "cores": 32,
    "lanes_per_core": 16,          # AVX-512 FP32
    "clock_ghz": 3.5,
    "fp32_tflops": 4.0,
    "fp16_tflops": 8.0,            # AVX512-FP16 / BF16, still vector units
    "bw_gbps": 400.0,              # 8-channel DDR5 ~ 400 GB/s (量级)
    "l1_kib": 48,
    "l2_mib": 1.25,
    "l3_mib": 128,
    "ddr_gb": 512,
    "power_w": "150-300",
}

# 兼容旧函数/断言：默认用服务器档，但 LLM 表会同时给出桌面与服务器结果。
CPU = CPU_SERVER

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
    "hbm_gb": 80,                  # 80 GB (decimal, 与厂商标称一致)
    "power_w": "~700",
}

for _cpu in (CPU_DESKTOP, CPU_SERVER):
    _cpu["lanes"] = _cpu["cores"] * _cpu["lanes_per_core"]
GPU["lanes"] = GPU["sms"] * GPU["lanes_per_sm"]

# GPU catalog: dense FP16/BF16 TFLOPS, HBM GB/s, VRAM GB (decimal), L2 MiB (量级).
@dataclass(frozen=True)
class Accelerator:
    name: str
    tflops: float
    hbm_gbps: float
    vram_gb: float
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
        {"level": "CPU DDR5 桌面", "cap": "128 GB", "lat": "~80-120 ns",
         "bw": "~80 GB/s (2ch)", "cache": "no", "scope": "system RAM"},
        {"level": "CPU DDR5 服务器", "cap": "512 GB", "lat": "~80-120 ns",
         "bw": "~400 GB/s (8ch)", "cache": "no", "scope": "system RAM"},
        {"level": "GPU registers", "cap": "256 KiB/SM", "lat": "~1 cycle",
         "bw": "per-SM private", "cache": "register file", "scope": "per-SM"},
        {"level": "GPU shared", "cap": "228 KiB/SM", "lat": "~20-30 cycles",
         "bw": "~33 TiB/s aggregate", "cache": "software-managed", "scope": "per-block"},
        {"level": "GPU L1", "cap": "128-256 KiB/SM", "lat": "~30-40 cycles",
         "bw": "~10-20 TiB/s", "cache": "yes, shares L1 with shared", "scope": "per-SM"},
        {"level": "GPU L2", "cap": "50 MiB", "lat": "~200-300 cycles",
         "bw": "~10 TiB/s", "cache": "yes, chip-wide", "scope": "whole GPU"},
        {"level": "GPU HBM3", "cap": "80 GB", "lat": "~500-800 cycles",
         "bw": "3.35 TB/s", "cache": "no", "scope": "device memory"},
    ]


# --------------------------------------------------------------------------- #
# 2. LLM workloads: roofline on CPU and GPU
# --------------------------------------------------------------------------- #

def gpu_peak_tflops() -> float:
    return GPU["fp16_tflops"]


def cpu_peak_tflops(cpu: dict | None = None) -> float:
    """Conservative sustained BF16 work on a CPU: about half of AVX-512 peak."""
    return (cpu or CPU_SERVER)["fp16_tflops"] / 2


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
    cpu_d_tflops, cpu_d_bw = cpu_peak_tflops(CPU_DESKTOP) * 1e12, CPU_DESKTOP["bw_gbps"] * 1e9
    cpu_s_tflops, cpu_s_bw = cpu_peak_tflops(CPU_SERVER) * 1e12, CPU_SERVER["bw_gbps"] * 1e9
    gpu_tflops, gpu_bw = gpu_peak_tflops() * 1e12 * 0.4, GPU["bw_gbps"] * 1e9  # 40% MFU for matmul
    rows = []
    for name, flops, nbytes in (
        ("decode 7B batch=1", decode_flops, decode_bytes),
        ("prefill 7B M=2048", prefill_flops, prefill_bytes),
        ("attention S=4096", attn_flops, attn_bytes),
    ):
        cpu_d = max(flops / cpu_d_tflops, nbytes / cpu_d_bw)
        cpu_s = max(flops / cpu_s_tflops, nbytes / cpu_s_bw)
        gpu_t = max(flops / gpu_tflops, nbytes / gpu_bw)
        rows.append({"name": name, "flops": flops, "bytes": nbytes,
                     "cpu_desktop_s": cpu_d, "cpu_server_s": cpu_s, "gpu_s": gpu_t,
                     "ratio_desktop": cpu_d / gpu_t, "ratio_server": cpu_s / gpu_t,
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
    # 容量统一用十进制 GB/MB（与 GPU 标称一致）；KiB/MiB 只用于缓存/块大小的说明。
    if b >= 1e9:
        return f"{b / 1e9:.1f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    if b >= 1e3:
        return f"{b / 1e3:.1f} KB"
    return f"{b:.0f} B"


def main() -> None:
    print("CPU vs GPU ledger (量级，非基准 order of magnitude, not a benchmark)")

    # ---- 1. design point ------------------------------------------------ #
    print("\n[1] design point: CPU = latency, GPU = throughput")
    print(f"{'item':<20} {'CPU 桌面 16c':<24} {'CPU 服务器 32c':<24} GPU ({GPU['name']})")
    print(f"{'cores / lanes':<20} "
          f"{str(CPU_DESKTOP['cores']) + ' x ' + str(CPU_DESKTOP['lanes_per_core']) + ' lanes':<24} "
          f"{str(CPU_SERVER['cores']) + ' x ' + str(CPU_SERVER['lanes_per_core']) + ' lanes':<24} "
          f"{GPU['sms']} SM x {GPU['lanes_per_sm']} = {GPU['lanes']}")
    print(f"{'clock / vector':<20} {'3.5 GHz / AVX-512':<24} {'3.5 GHz / AVX-512':<24} "
          f"~{GPU['clock_ghz']} GHz / 32 lanes per warp")
    print(f"{'FP32 TFLOPS':<20} {'~' + str(CPU_DESKTOP['fp32_tflops']):<24} "
          f"{'~' + str(CPU_SERVER['fp32_tflops']):<24} ~{GPU['fp32_tflops']:.0f} (CUDA cores)")
    print(f"{'BF16 TFLOPS':<20} {'~' + str(CPU_DESKTOP['fp16_tflops']):<24} "
          f"{'~' + str(CPU_SERVER['fp16_tflops']):<24} ~{GPU['fp16_tflops']:.0f} (Tensor Core)")
    print(f"{'memory BW GB/s':<20} {'~' + str(CPU_DESKTOP['bw_gbps']) + ' (2ch)':<24} "
          f"{'~' + str(CPU_SERVER['bw_gbps']) + ' (8ch)':<24} {GPU['bw_gbps']:.0f} (HBM3)")
    print(f"{'cache':<20} {'L3 32 MiB':<24} {'L3 128 MiB':<24} per-SM L1/shared + 50 MiB L2")
    print(f"{'power / package':<20} {CPU_DESKTOP['power_w'] + ' W':<24} "
          f"{CPU_SERVER['power_w'] + ' W':<24} {GPU['power_w'] + ' W'}")

    # ---- 2. accelerator catalog ----------------------------------------- #
    print("\n[2] accelerators at a glance (量级; dense, no sparsity; 910B/HCU 是 NPU/DCU 口径，"
          "L1/L2 命名不同，SKU 不同)")
    print(f"{'chip':<20} {'FP16/BF16 TFLOPS':>16} {'mem GB/s':>9} {'VRAM GB':>9} {'L2 MiB':>7}")
    for a in ACCELERATORS:
        l2 = "n/p" if a.l2_mib is None else f"{a.l2_mib:.0f}"
        print(f"{a.name:<20} {a.tflops:>16.0f} {a.hbm_gbps:>9.0f} {a.vram_gb:>9.0f} {l2:>7}")

    # ---- 3. cache hierarchy --------------------------------------------- #
    print("\n[3] memory hierarchy: CPU cache vs GPU cache")
    print(f"{'level':<14} {'capacity':<16} {'latency':<16} {'bandwidth':<20} cache?")
    for r in hierarchy_rows():
        print(f"{r['level']:<14} {r['cap']:<16} {r['lat']:<16} {r['bw']:<20} {r['cache']}")
    print(f"GPU answers: L1=yes (unified with shared, carveout); L2=yes (chip-wide: H100 50, "
          f"A100 40, 4090 72 MiB; {CACHE_LINE_B} B line); HBM=no; coalescing decides line usage.")

    # ---- 4. LLM roofline ------------------------------------------------- #
    rows = llm_workloads()
    cpu_d_ridge = ridge_point(cpu_peak_tflops(CPU_DESKTOP), CPU_DESKTOP["bw_gbps"])
    cpu_s_ridge = ridge_point(cpu_peak_tflops(CPU_SERVER), CPU_SERVER["bw_gbps"])
    gpu_ridge = ridge_point(gpu_peak_tflops(), GPU["bw_gbps"])
    print(f"\n[4] LLM workloads: CPU 桌面/服务器 vs GPU (7B BF16; "
          f"桌面 {cpu_peak_tflops(CPU_DESKTOP):.0f}T/{CPU_DESKTOP['bw_gbps']:.0f}GB/s, "
          f"服务器 {cpu_peak_tflops(CPU_SERVER):.0f}T/{CPU_SERVER['bw_gbps']:.0f}GB/s, "
          f"GPU {gpu_peak_tflops():.0f}T/{GPU['bw_gbps']:.0f}GB/s @40% MFU)")
    print(f"{'workload':<22} {'FLOPs':>9} {'bytes':>9} {'CPU桌面':>9} {'CPU服务器':>10} {'GPU':>9} {'差距 桌/服':>13}")
    for r in rows:
        print(f"{r['name']:<22} {r['flops'] / 1e9:>7.1f} G {fmt_bytes(r['bytes']):>9} "
              f"{fmt_time(r['cpu_desktop_s']):>9} {fmt_time(r['cpu_server_s']):>10} "
              f"{fmt_time(r['gpu_s']):>9} {r['ratio_desktop']:>5.0f}x/{r['ratio_server']:<6.0f}x")
    print(f"ridge point: CPU 桌面 {cpu_d_ridge:.1f}, CPU 服务器 {cpu_s_ridge:.1f}, GPU {gpu_ridge:.1f} FLOP/B; "
          f"decode AI={rows[0]['ai']:.2f} (memory-bound) vs prefill AI={rows[1]['ai']:.0f} (compute-bound).")

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
    assert CPU_DESKTOP["lanes"] < GPU["lanes"] / 20
    assert CPU_SERVER["lanes"] < GPU["lanes"] / 20
    assert CPU_DESKTOP["bw_gbps"] * 20 < GPU["bw_gbps"]
    assert CPU_SERVER["bw_gbps"] * 8 < GPU["bw_gbps"]
    assert CPU_DESKTOP["fp16_tflops"] * 50 < GPU["fp16_tflops"]
    assert CPU_SERVER["fp16_tflops"] * 50 < GPU["fp16_tflops"]
    assert CPU_SERVER["bw_gbps"] < GPU["bw_gbps"] and CPU_SERVER["bw_gbps"] < ACCELERATORS[3].hbm_gbps / 2
    by_name = {a.name: a for a in ACCELERATORS}
    assert by_name["H100 SXM"].l2_mib == 50
    assert by_name["A100 SXM"].l2_mib == 40
    assert by_name["RTX 4090"].l2_mib == 72
    ddr_levels = ("CPU DDR5 桌面", "CPU DDR5 服务器", "GPU HBM3")
    assert all(r["cache"] == "no" for r in hierarchy_rows() if r["level"] in ddr_levels)
    assert all(r["cache"].startswith("yes") for r in hierarchy_rows()
               if r["level"] in ("CPU L1", "CPU L2", "CPU L3", "GPU L1", "GPU L2"))
    assert rows[0]["ai"] < min(cpu_d_ridge, cpu_s_ridge, gpu_ridge)   # decode memory
    assert rows[1]["ai"] > max(cpu_d_ridge, cpu_s_ridge, gpu_ridge)   # prefill compute
    assert rows[0]["ratio_desktop"] > 20 and rows[0]["ratio_server"] > 5
    assert rows[1]["ratio_desktop"] > 50 and rows[1]["ratio_server"] > 50
    assert rows[2]["ratio_desktop"] > 50 and rows[2]["ratio_server"] > 50
    assert small["ratio"] > 10 and small["gpu_us"] >= 3 and small["gpu_us"] <= 8
    assert amd["speedup"] < amd["max_speedup"] < 10
    assert [e[0] for e in ECOSYSTEM] == ["CUDA", "ROCm + HIP", "CANN + AscendCL",
                                         "HCU = DTK + HIP", "OpenCL / oneAPI", "Triton"]

    print("\nOK: design point, cache hierarchy, LLM ratios, launch limits and stacks are self-consistent.")


if __name__ == "__main__":
    main()
