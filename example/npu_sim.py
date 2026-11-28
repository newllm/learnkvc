"""GPU / NPU / 国产算力的量级账本（纯标准库，输出确定性）。

数字是公开资料的量级示意，不是 benchmark；目的是把「算力 / 带宽 /
decode 时间 / 软件栈」放进同一张表里比较。
"""

# name, kind, peak BF16 TFLOPS, HBM GB/s, software stack
ACCELERATORS = [
    ("NVIDIA H100", "GPU", 990, 3350, "CUDA"),
    ("NVIDIA A100", "GPU", 312, 2039, "CUDA"),
    ("AMD MI300X", "GPU", 1300, 5300, "ROCm / HIP"),
    ("Ascend 910B", "NPU", 376, 1600, "CANN"),
    ("Hygon DCU", "GPGPU", 200, 1000, "DTK / HIP"),
    ("Cambricon MLU590", "NPU", 256, 1024, "Neuware"),
    ("Biren BR100", "GPGPU", 512, 1600, "BIRENSUPA"),
    ("Moore Threads S5000", "GPGPU", 128, 1024, "MUSA"),
]

# 7B BF16 权重 ≈ 14 GB；4K 上下文 KV + 运行时按 0.5 GB 估。
WEIGHT_GB = 14.0
KV_GB = 0.5
SEQ = 4096
PARAMS = 7e9


def decode_tps(bw_gbs: float) -> float:
    """decode batch=1：每步读权重 + KV，时间 ≈ 字节 / 带宽。"""
    return bw_gbs / (WEIGHT_GB + KV_GB)


def prefill_ms(tflops: float) -> float:
    """prefill 4K：FLOPs ≈ 2 × 参数 × token 数；忽略 attention 的平方项。"""
    flops = 2 * PARAMS * SEQ
    return flops / (tflops * 1e12) * 1000


def ridge_point(tflops: float, bw_gbs: float) -> float:
    """roofline 拐点：FLOPs / Byte。"""
    return tflops * 1e12 / (bw_gbs * 1e9)


def main() -> None:
    print("== 7B 模型：同一条 roofline，不同加速器 ==")
    print(f"{'加速器':<22}{'类型':<8}{'BF16':>7}{'HBM':>8}{'decode':>9}{'prefill':>10}{'拐点':>8}  软件栈")
    print(f"{'':22}{'':8}{'TFLOPS':>7}{'GB/s':>8}{'tok/s':>9}{'4K ms':>10}{'F/B':>8}")
    for name, kind, peak, bw, stack in ACCELERATORS:
        print(
            f"{name:<22}{kind:<8}{peak:>7}{bw:>8}"
            f"{decode_tps(bw):>9.1f}{prefill_ms(peak):>10.1f}"
            f"{ridge_point(peak, bw):>8.0f}  {stack}"
        )

    # 核心断言：量级关系必须成立，避免把数字改坏。
    h100 = decode_tps(3350)
    a100 = decode_tps(2039)
    assert h100 > a100, "H100 decode 应快于 A100"
    assert prefill_ms(990) < prefill_ms(312), "H100 prefill 应快于 A100"
    assert 200 < h100 < 300, "7B decode 应在几百 tok/s 量级"
    assert decode_tps(1600) > decode_tps(1024), "带宽越高 decode 越快"

    print("\n== 选型不是只看峰值 ==")
    rows = [
        ("大模型训练", "NVIDIA GPU / 昇腾集群", "先看集合通信、框架支持、集群规模"),
        ("通用推理", "NVIDIA GPU / AMD GPU / 国产 GPU", "先看引擎与算子覆盖，再看带宽"),
        ("信创 / 国产替代", "昇腾 NPU / 海光 DCU / 国产 GPGPU", "先验证模型与算子，再谈性价比"),
        ("小团队快速上线", "NVIDIA GPU 或成熟云服务", "生态与排障成本最低"),
    ]
    for scene, pick, why in rows:
        print(f"- {scene:<14} → {pick:<32} {why}")

    print("\nOK: GPU/NPU 账本自洽；峰值之外，软件栈和算子覆盖决定能不能跑起来。")


if __name__ == "__main__":
    main()
