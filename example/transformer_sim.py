"""Transformer / LLM 的量级账本（纯标准库，输出确定性）。

回答三个问题：
1. 自注意力的计算量为什么随序列长度平方增长？
2. KV cache 为什么随上下文线性增长？
3. 为什么 Transformer 比 RNN 更适合用 GPU 训练？
"""

LAYERS = 32
HEADS = 32
HEAD_DIM = 128
KV_HEADS = 8
BYTES = 2  # BF16

# 参数量、训练 token 数（Chinchilla 经验：~20 tokens/param）
SCALING = [
    ("1B", 1e9, 20e9),
    ("7B", 7e9, 140e9),
    ("70B", 70e9, 1.4e12),
    ("400B", 400e9, 8e12),
]


def attn_score_elems(n: int) -> int:
    """每层注意力分数矩阵的元素数（heads × N × N）。"""
    return HEADS * n * n


def attn_tflops(n: int) -> float:
    """QK^T 与 softmax·V 的量级：约 4 × N² × d（每层），再乘层数。"""
    return 4 * LAYERS * HEADS * n * n * HEAD_DIM / 1e12


def kv_bytes(n: int) -> int:
    """KV cache：2（K/V）× layers × kv_heads × head_dim × tokens × bytes。"""
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * n * BYTES


def rnn_steps(n: int) -> int:
    """RNN 必须按时间步串行；Transformer 训练可在序列维并行。"""
    return n


def main() -> None:
    print("== 1. 自注意力：N² 的计算，换 O(1) 的路径长度 ==")
    print(f"{'seq':>8}{'score 元素':>15}{'attention TFLOPs':>19}{'KV cache':>13}")
    for n in (1024, 4096, 16384, 65536, 131072):
        print(f"{n:>8}{attn_score_elems(n):>15,}{attn_tflops(n):>19.1f}{kv_bytes(n)/2**30:>12.1f}G")

    assert attn_score_elems(4096) == 16 * attn_score_elems(1024)
    assert attn_score_elems(16384) == 256 * attn_score_elems(1024)
    assert kv_bytes(8192) == 2 * kv_bytes(4096), "KV cache 应随上下文线性增长"

    print("\n== 2. 为什么 GPU 喜欢 Transformer ==")
    n = 4096
    print(f"RNN：训练时 {rnn_steps(n):,} 个串行时间步（batch 之外很难再并行）")
    print(f"Transformer：训练时序列维可并行；串行深度只有 {LAYERS} 层（每层内部是 GEMM）")
    print(f"decode 时：每生成 1 个 token 仍要跑 {LAYERS} 层，但 KV cache 让历史不再重算")
    assert LAYERS < n

    print("\n== 3. 缩放账本：参数量 × 训练 token ==")
    print(f"{'params':>8}{'tokens':>14}{'训练 FLOPs≈6ND':>20}")
    for name, params, tokens in SCALING:
        flops = 6 * params * tokens
        print(f"{name:>8}{tokens/1e9:>12.0f}B{flops:>20.1e}")
        assert flops > 0

    print("\n结论：Transformer 把「序列建模」变成了大矩阵乘 + 注意力；")
    print("它既能吃 GPU 的并行，又能随数据/参数/算力一起放大——这是 LLM 胜出的工程基础。")
    print("OK: Transformer / LLM 账本自洽。")


if __name__ == "__main__":
    main()
