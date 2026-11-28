"""LLM 结构的量级账本（纯标准库，输出确定性）。

参数 = 层数 × 每层参数 + embedding + lm_head；
激活参数 = 每步真正读到的部分（MoE 只读 top-k expert）；
KV/token = 2 × layers × kv_heads × head_dim × dtype_bytes。
"""
PRESETS = [
    ("Tiny", 2, 32, 4, 2, 8, 64, 0, 0, 0, 64, True, 256, True),
    ("Qwen2.5-7B", 28, 3584, 28, 4, 128, 18944, 0, 0, 0, 152064, False, 32768, False),
    ("Qwen2.5-32B", 64, 5120, 40, 8, 128, 27648, 0, 0, 0, 152064, False, 32768, False),
    ("Qwen3-30B-A3B", 48, 2048, 32, 4, 128, 6144, 128, 768, 8, 151936, False, 262144, False),
]
DTYPE = 2  # BF16 bytes


def math(preset):
    name, layers, h, qh, kh, hd, ffn, experts, einter, topk, vocab, tie, max_pos, pos_embed = preset
    q_dim, kv_dim = qh * hd, kh * hd
    attn = h * q_dim + 2 * (h * kv_dim) + q_dim * h + 2 * hd
    router = h * experts if experts else 0
    ffn_p = experts * 3 * h * einter if experts else 3 * h * ffn
    per_layer = attn + ffn_p + router
    embed = vocab * h
    head = 0 if tie else vocab * h
    pos = max_pos * h if pos_embed else 0
    total = layers * per_layer + embed + head + pos
    active_layer = attn + router + (topk * 3 * h * einter if experts else ffn_p)
    active = layers * active_layer + embed + head
    kv = 2 * layers * kh * hd * DTYPE
    return total, active, kv, per_layer, q_dim, kv_dim


def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def main():
    print("== 1. 参数账本：总参数 / 每 token 激活参数 / KV ==")
    print(f"{'preset':>14}{'layers':>7}{'hidden':>8}{'total':>10}{'active':>10}{'KV/token':>11}")
    rows = {}
    for p in PRESETS:
        name, layers, h = p[0], p[1], p[2]
        total, active, kv, *_ = math(p)
        rows[name] = (total, active, kv)
        print(f"{name:>14}{layers:>7}{h:>8}{fmt(total):>10}{fmt(active):>10}{kv/1024:>10.0f}KiB")

    assert abs(rows["Qwen3-30B-A3B"][0] / 1e9 - 30.5) < 0.2
    assert abs(rows["Qwen3-30B-A3B"][1] / 1e9 - 3.3) < 0.2
    assert rows["Qwen3-30B-A3B"][2] == 96 * 1024
    assert rows["Qwen2.5-7B"][2] == 56 * 1024
    assert rows["Qwen2.5-32B"][2] == 256 * 1024

    print("\n== 2. 同样的 48 层，MoE 与 dense 的激活账本 ==")
    p = PRESETS[3]
    total, active, kv, per_layer, q_dim, kv_dim = math(p)
    print(f"Qwen3-30B-A3B：{p[1]} 层 × 每层 {fmt(per_layer)} 参数 = 总参数 {fmt(total)}")
    print(f"Qwen 头维度：Q {q_dim} / KV {kv_dim}；attention 输出回到 hidden {p[2]}")
    print(f"每 token 只读 top-{p[9]} expert：激活参数 {fmt(active)}，约为总参数的 {active/total:.0%}")
    print(f"KV/token = 2 × {p[1]} × {p[4]} × {p[5]} × {DTYPE} B = {kv/1024:.0f} KiB")
    print("结论：参数决定显存；激活参数决定每步读多少；KV 只随层数 × KV 头 × 精度增长。")
    print("OK: LLM 结构账本自洽。")


if __name__ == "__main__":
    main()
