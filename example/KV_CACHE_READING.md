# `kv_cache.py` 阅读导图

这份文件把两件事放在了一起，用同一个模型跑给你看：

1. **没有 cache 的生成流程长什么样**：每步重算整个前缀，只取最后一个位置的 logits；
2. **KV cache 怎么实现这层复用**：每层缓存 K/V，decode 时只算新 token，历史直接读。

代码顺序就是阅读顺序：`LayerKVCache` / `KVCache` → `cached_attention` → `full_attention` → `TinyModel` → `generate` → 自测。文件末尾会跑出两条路径的逐步对照、计算量统计，并断言两者结果一致。

> 这里讲的是**单个请求内部**的复用。多个请求之间共享同一段前缀（system prompt、few-shot、长文档）的做法见 [`PREFIX_CACHE_READING.md`](PREFIX_CACHE_READING.md)，那份代码把 KV 切成块，用哈希匹配 + 引用计数 + LRU 实现跨请求复用。

运行方式（需要 PyTorch）：

```bash
pip install torch
python3 example/kv_cache.py

# 有 CUDA 会自动用 GPU；也可以显式指定（cpu / cuda / mps）
KVCACHE_DEVICE=cuda python3 example/kv_cache.py
```

## 目录

- [先建立整体图像](#先建立整体图像)
- [第 1 部分：没有 cache 的流程](#第-1-部分没有-cache-的流程)
- [第 2 部分：缓存的数据结构](#第-2-部分缓存的数据结构)
  - [`LayerKVCache`：一层的历史状态](#layerkvcache一层的历史状态)
  - [`append()`：Cache 的核心操作](#appendcache-的核心操作)
  - [`KVCache`：把多层装在一起](#kvcache把多层装在一起)
- [第 3 部分：`cached_attention`：缓存如何接上注意力](#第-3-部分cached_attention缓存如何接上注意力)
  - [为什么不能只写 `is_causal=True`](#为什么不能只写-is_causaltrue)
  - [`full_attention`：无 cache 的对照实现](#full_attention无-cache-的对照实现)
- [第 4 部分：`TinyModel` 与两条生成循环](#第-4-部分tinymodel-与两条生成循环)
- [第 5 部分：自测都检查了什么](#第-5-部分自测都检查了什么)
- [第 6 部分：GPU 上具体发生了什么](#第-6-部分gpu-上具体发生了什么)
- [代码阅读顺序](#代码阅读顺序)
- [阅读时可以检查的七个问题](#阅读时可以检查的七个问题)
- [这份示例刻意没有处理的生产问题](#这份示例刻意没有处理的生产问题)

## 先建立整体图像

prompt `[A, B, C]`，要生成 `D, E, F`：

```text
无 cache（每步重算整个前缀）
step 0: [A B C]     → 算 3 个位置 → D
step 1: [A B C D]   → 算 4 个位置 → E    ← A B C 又算了一遍
step 2: [A B C D E] → 算 5 个位置 → F    ← A B C D 又算了一遍

有 cache
step 0: [A B C] → 算 3 个位置，K/V 写进 cache → D
step 1: [D]     → 算 1 个位置，读 3 个历史 K/V → E
step 2: [E]     → 算 1 个位置，读 4 个历史 K/V → F
```

无 cache 时，每一步的输入都包含整段前缀，所以每一层都要重新做 `QKV 投影 + attention + MLP`；而一个 step 真正有用的只有**最后一个位置**的输出，其余位置的结果全部丢弃。

关键事实：**重复算出来的 K/V 和上一次完全相同**（同模型、同精度、同位置）。所以缓存不是近似，而是省掉纯浪费。文件里的 `check_numerics()` 会把这一点变成断言。

## 第 1 部分：没有 cache 的流程

入口 `generate(..., use_cache=False)` 就是一个循环：

```python
while len(steps) < n_new:
    logits = model.forward(torch.tensor(sequence))   # 整段前缀
    token = logits.argmax()
    sequence.append(token)
```

`TinyModel.forward` 收到 `token_count` 个 token，就会对**每一个**位置做每层的投影和 attention，最后只返回 `hidden[0, -1] @ embed.T`。前面位置的输出没人要，但计算一份不少。

演示里 prompt 8 个 token、生成 8 个 token，于是：

```text
无 cache 输入长度：8, 9, 10, ..., 15
位置前向（token 穿过全部层）：92 次
白白重算的位置：92 - 15 = 77 个
```

## 第 2 部分：缓存的数据结构

### `LayerKVCache`：一层的历史状态

一个 Transformer layer 有自己独立的一组 K/V，所以类里只有 `key`、`value` 两个张量，布局是全文件统一的约定：

```text
(batch, kv_heads, sequence_length, head_dim)
```

初始为 `None`。`length` 读倒数第二维，即当前缓存了多少个位置。

### `append()`：Cache 的核心操作

```text
空缓存：              cache = new_kv
已有历史：            cache = cat(old_kv, new_kv, seq 维)
开启窗口：            cache = cache 的最后 max_length 个位置
```

拼接维度是 `dim=-2`（sequence 维），不要沿 `head_dim` 或 head 数拼。方法既更新内部张量，也返回完整缓存供当前 attention 直接使用。

`max_length=4` 时，喂 6 个 token 后只剩最后 4 个；自测会把它和一个全量缓存的后 4 个位置逐位比较。

### `KVCache`：把多层装在一起

```python
cache = KVCache(num_layers=2, max_length=None)
cache.update(layer, key, value)   # 写某一层
cache.length                     # 各层同步推进，取第 0 层即可
```

另外两个方法：

- `reorder(batch_indices)`：beam search 每步会筛选、复制候选序列，所有层必须用同一组下标重排；
- `fork()`：从同一前缀分叉出另一条生成路径。内部用 `clone()`，所以分支的 append 不会改到父缓存。自测里验证了 `parent.length == 2` 而 `branch.length == 3`。

## 第 3 部分：`cached_attention`：缓存如何接上注意力

这是全文件最值得逐行读的函数：

1. **先读 `past`**：`past = cache.layers[layer].length`，必须在 `update` 之前读，否则拿到的是拼接后的长度；
2. **写入并取回**：`cache.update()` 把新 K/V 拼到历史后面，返回完整 K/V；
3. **GQA/MQA 展开**：`repeat_interleave` 把 KV 头复制到和 Q 头一样多；
4. **显式算分数**：`Q @ Kᵀ / sqrt(head_dim)`，施加 causal mask，softmax 后乘 V。

对应数学：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

其中的 `K`、`V` 已经包含历史部分。

### 为什么不能只写 `is_causal=True`

纯 prefill 时 Q 和 K 都从 0 开始，普通下三角 mask 够用。但 cache 场景可能是：已有 100 个位置，本次再处理 3 个新 token。此时 Q 的绝对位置是 100、101、102，而 K 是 0 到 102；第一条新 Q 只能看到 0 到 100。

代码因此手工建立 mask：

```python
first_key_position = past + q_len - k_len
causal = (first_key_position + arange(k_len)) <= (past + arange(q_len)[:, None])
```

`first_key_position` 同时处理了滑动窗口：窗口淘汰旧 key 后，缓存第 0 项不再对应绝对位置 0，不能把缓存下标直接当 token 位置。

### `full_attention`：无 cache 的对照实现

它和不带 cache 的路径一一对应：没有 `past`、没有拼接，直接对完整前缀做同一套 GQA + causal mask，等价于 `cached_attention` 里 `past=0` 的情况。两条路径用同一套公式，才使得“结果一致”可以被逐帧比较。

## 第 4 部分：`TinyModel` 与两条生成循环

`TinyModel` 是一个 2 层、4 个 Q 头 / 2 个 KV 头（GQA）、d_model=32 的玩具模型，带绝对位置嵌入、残差和 MLP，没有 LayerNorm 和 RoPE——数学保持可读，但“K/V 是前缀和位置的函数”这一点不变。

`forward(tokens, cache=None)` 的关键几行：

```python
past = cache.length if cache is not None else 0
hidden = embed[tokens] + pos[past:past + token_count]   # 位置必须接着历史
# 每层：QKV 投影 → cached_attention / full_attention → Wo + 残差 → MLP + 残差
return hidden[0, -1] @ embed.T                            # 只返回最后位置的 logits
```

`generate()` 只有输入不同：无 cache 每步传整段序列，有 cache 首步 prefill、之后每步只传 `[sequence[-1]]`。两条循环各自用 `Stats` 记录：

- `positions`：token 穿过全部层的次数；
- `attention_pairs`：attention 里 `(query, key)` 对的个数（各层合计）。

演示输出：

```text
  步骤   新 token    无 cache 输入    有 cache 输入
     0         30             8 个             8 个
     1         15             9 个             1 个
   ...
汇总
  位置前向（token 穿过全部层）            无 cache       92    有 cache       15   6.1x
  attention (query,key) 对数（各层合计）  无 cache     1192    有 cache      240   5.0x
```

`print_summary` 里的断言会先用公式算出这两组数，确保统计口径没有被改坏：

```text
无 cache positions        = 8 + 9 + ... + 15
无 cache attention pairs  = 层数 × Σ T(T+1)/2
有 cache positions        = prefill 8 + decode 7×1
有 cache attention pairs  = 层数 × (prefill 36 + decode 9+...+15)
```

两个容易误解的地方：

1. **decode 每步仍要读全部历史。** attention 对数总量会下降，但每步的读取量随前缀增长。上下文越长，decode 越像“读显存”而不是“算矩阵”，这也是生产引擎要做连续批处理的原因。
2. **位置 id 必须对齐。** 有 cache 时新 token 的绝对位置是 `past`，不是 0；`forward` 用 `pos[past:past+token_count]` 取嵌入，`cached_attention` 用 `past` 建 mask。写错位置，logits 立刻改变。

## 第 5 部分：自测都检查了什么

文件末尾的真实输出：

```text
生成结果：[30, 15, 15, 15, 41, 41, 41, 41]

数值层面：同一段 [prefix + token]，一次性 forward 与 prefill + decode 的
logits 最大差异 4.47e-08（浮点排序差异），argmax 完全一致。

全部自测通过：两条路径 token 一致、logits 一致；GQA / 滑动窗口 / reorder / fork 正常。
```

对应的断言：

- `generate` 两条路径的 token 列表必须完全相同；
- `check_numerics()`：同一段 `[prefix + token]`，一次性 forward 与 prefill+decode 的 logits 最大差异小于 `1e-5`（实测 `4.47e-08`，来自 `cat`/matmul 的浮点排序），且 argmax 相同；
- `check_cache_features()`：GQA 下 K 的形状是 `(1, 2, 3, 8)`；滑动窗口保留的正好是全量缓存的后 4 个位置；`reorder([1,0])` 等价于翻转；`fork()` 后父缓存长度不变。

`1e-08` 级别的差异是正常的：两条路径的数学等价，但矩阵乘法的累加顺序可能不同。token 完全一致才是对生成有意义的结论。

## 第 6 部分：GPU 上具体发生了什么

代码里所有张量运算都是 device-agnostic 的：默认 `DEVICE` 在 CUDA 可用时取 `cuda`，否则取 `cpu`，也可以用 `KVCACHE_DEVICE=cuda`（或 `mps` / `cpu`）强制指定。切到 GPU 不需要改任何代码，PyTorch 会把每个调用分派到对应的实现：

| 代码 | 运算 | GPU 上使用的 API |
| --- | --- | --- |
| `embed[tokens]` | gather | ATen `index_select` CUDA kernel |
| `hidden @ wq/wk/wv/wo/w1/w2` | GEMM | cuBLAS `cublasSgemm` / `cublasGemmEx`（fp16/bf16；部分形状走 cuBLASLt） |
| `query @ key.transpose(-1,-2)` | batched GEMM | cuBLAS `cublasGemmStridedBatched` |
| `softmax @ value` | batched GEMM | 同上 |
| `torch.cat`（`cache.update`） | 拷贝 | CUDA copy kernel + caching allocator 分配新显存 |
| `repeat_interleave` | gather | CUDA gather kernel（生产实现由 FlashAttention 直接读 GQA 头，不做复制） |
| `masked_fill` / `+` / `relu` | elementwise | ATen elementwise CUDA kernel |
| `softmax` | reduction | ATen CUDA softmax kernel（生产里和 attention 融合） |
| `argmax` | reduction | ATen CUDA `argmax` kernel |
| K/V 张量 | 存储 | GPU HBM，由 caching allocator 管理（底层 `cudaMalloc` / `cudaFree`） |
| `torch.tensor(sequence)` | H2D | `cudaMemcpyAsync` |
| `int(logits.argmax())` | D2H | `cudaMemcpyAsync` + stream 同步 |

由此可以回答“哪些计算有 GPU 参与”：

- **参与的是这些张量算子**；`generate()` 的 Python 循环、`Stats` 计数，以及 `prefix_cache_demo.py` 的块管理都在 CPU 上，只负责构造请求、发 kernel。
- decode 每步取一次 `argmax` 结果会触发 D2H 同步，打断 GPU 流水线；真实引擎在 GPU 上采样，避免每个 token 同步。
- `torch.cat` 每步“分配新显存 + 拷贝”；生产引擎用预分配的 paged cache，attention kernel 按 block table 取数（见 [`PREFIX_CACHE_READING.md`](PREFIX_CACHE_READING.md)）。
- 本示例的 attention 是三个独立算子：QKᵀ → softmax → PV。生产里 FlashAttention / `scaled_dot_product_attention` 会把它们融合成一个 CUDA/Triton kernel，减少显存读写。
- 多卡并行不在本文件范围，那会用到 NCCL 的 all-reduce / all-gather 等集合通信。
- 在 Apple Silicon 上设置 `KVCACHE_DEVICE=mps` 会走 Metal kernel（没有 cuBLAS）；CPU 上则走 MKL/oneDNN（x86）或 Accelerate（macOS）。

## 代码阅读顺序

1. `LayerKVCache.append()`：看 `dim=-2` 和窗口裁剪，这是“缓存”两个字的全部内容。
2. `KVCache`：只是多层容器，顺手看 `reorder` / `fork`。
3. `cached_attention()`：`past` 的读取时机、GQA 展开、mask 的绝对位置对齐。
4. `full_attention()`：和无 cache 路径对照，注意它只是 `past=0` 的特例。
5. `TinyModel.forward()`：位置嵌入为什么从 `past` 开始，logits 为什么只取最后一位。
6. `generate()`：两个循环唯一的区别是每步喂什么。
7. `check_numerics()` / `check_cache_features()`：把“应该对”变成“跑得过”。

## 阅读时可以检查的七个问题

1. `past` 为什么必须在 `cache.update()` 之前读取？
2. 为什么 K/V 沿 `dim=-2` 拼接，而不是 `head_dim` 或 head 数？
3. 有 cache 时，新 token 的位置嵌入为什么是 `past` 而不是 0？
4. 为什么不能直接把 `cached_attention` 换成 `is_causal=True`？
5. GQA 中为什么 Q 头数必须是 KV 头数的整数倍？
6. 开启滑动窗口后，为什么不能把缓存下标当成绝对 token 位置？
7. 为什么 `positions` 能降到 15，而 attention 对数只降到 240（而不是同一个比例）？

## 这份示例刻意没有处理的生产问题

- 预分配固定大小的 GPU buffer，避免每次 `torch.cat` 分配和拷贝（真实引擎用 paged KV cache）。
- 不同请求长度的 padding、attention mask 和 continuous batching。
- RoPE 的 position ids，以及滑动窗口后的位置策略（本文件只用绝对位置嵌入演示“位置要对齐”）。
- 显存分页、KV 量化、跨请求共享前缀（见 [`PREFIX_CACHE_READING.md`](PREFIX_CACHE_READING.md)）。
- 使用 FlashAttention / `scaled_dot_product_attention` 等高性能 kernel。

因此它适合理解正确性和数据流；生产推理引擎会在同一套语义上换掉存储和 kernel。
