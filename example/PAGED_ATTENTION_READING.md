# `paged_attention.py` 阅读导图

前两份示例里，KV cache 要么是一整块连续张量（`kv_cache.py`），要么是“块”只做记账（`prefix_cache_demo.py`）。这份代码把两者接起来：**KV 真的存在固定大小的块里，attention 通过 block table 找到它们**。这就是 vLLM 的 PagedAttention 的核心。

一句话概括：

```text
连续 KV cache：      token 位置 → 内存偏移（必须预留 max_len，且不能共享）
PagedAttention：     token 位置 → 逻辑块 + 块内偏移 → block_table → 物理块
```

运行（需要 PyTorch）：

```bash
python3 example/paged_attention.py
PAGED_DEVICE=cuda python3 example/paged_attention.py   # 有 CUDA 时也可以显式指定
```

## 为什么需要 PagedAttention

如果按最大长度给每个请求预留连续显存：

```text
seq-a 预留 4096 token 的 KV： [########################################]
实际只用了 300 token：        [###.....................................]
                              ↑ 其余空间被占着，别人不能用
```

问题有三个：

1. **内部碎片**：实际长度远小于预留长度；
2. **外部碎片**：不同长度的请求交替分配/释放，连续大块越来越难找；
3. **无法共享**：两个请求共享一段前缀时，连续布局没法让它们指向同一段显存。

PagedAttention 把 KV 显存切成固定大小的块（vLLM 默认 16 个 token 一块），请求按需拿块，块之间不需要物理连续。每个请求维护一张 **block table**，记录“我的第 i 个逻辑块放在哪个物理块”。

## 整体图像

```text
seq-a 的逻辑 token： 0 1 2 3 | 4 5 6 7 | 8 9
                     └─逻辑块0─┘└─逻辑块1─┘└逻辑块2┘
                          │         │         │
block_table[0]=5 ─────────┘         │         │
block_table[1]=1 ───────────────────┘         │
block_table[2]=3 ─────────────────────────────┘

物理块池：
  b0: seq-b 的 token 0-3
  b1: seq-a 的 token 4-7
  b2: 空闲 / 之后给 seq-a 的第 4 块
  b3: seq-a 的 token 8-11（只用了 2 个 slot）
  b4: seq-b 的 token 4-7
  b5: seq-a 的 token 0-3
  ...
```

映射公式只有两行：

```python
logical_block = token_position // block_size
slot          = token_position %  block_size
physical      = block_table[logical_block]
```

## 数据结构

### `BlockPool`：物理显存

```python
k: (num_blocks, block_size, kv_heads, head_dim)
v: (num_blocks, block_size, kv_heads, head_dim)
free_blocks: 空闲物理块编号的列表（本示例用随机顺序，故意让表不连续）
```

真实引擎里这两块张量在 GPU 显存上，用显存分配器管理；这里就是一个 `torch.zeros`，但形状和访问方式与真实引擎一致。

`allocate()` 从空闲列表弹出一个块，`release()` 放回去。注意：**物理块号本身没有顺序含义**，b5 完全可能排在 b1 前面。

### `Sequence`：block table

```python
block_table: [5, 1, 3]   # 逻辑块 0/1/2 的物理位置
length: 10               # 已经写了多少 token
```

`block_table` 是每个请求私有的，长度随请求增长；块池是所有请求共享的。

### `PagedKVCache`：写入与读取

- `append(sequence, key, value)`：把新 token 的 K/V 写进当前最后一块；如果最后一块已满（`length % block_size == 0`），先分配一个新块。
- `gather(sequence)`：按 `block_table` 的顺序取块，拼成逻辑顺序的 `(length, kv_heads, head_dim)`，再截掉最后一块里没用的 slot。
- `free(sequence)`：把块还给池子，清空 block table。

## 写入：`append()` 怎么落块

```python
for token in range(key.size(0)):
    slot = sequence.length % block_size
    if slot == 0:                                  # 新逻辑块
        sequence.block_table.append(pool.allocate())
    block = sequence.block_table[-1]
    pool.write(block, slot, key[token], value[token])
    sequence.length += 1
```

关键点：

- 只有 `slot == 0` 时才分配，所以一个 4-token 的块可以分多次 decode 填满；
- 块内偏移由 `length % block_size` 决定，与物理块号无关；
- 最后一块用不满时会留下空 slot（内部碎片），但不会污染有效 token —— `gather()` 最后会按 `length` 截断。

## 读取：`gather()` 怎么还原逻辑顺序

```python
blocks = tensor(sequence.block_table)             # [5, 1, 3]
key = pool.k[blocks]                              # (3, 4, kv_heads, dim)
key = key.reshape(-1, kv_heads, dim)              # 3*4 个 token
return key[:sequence.length]                      # 截掉最后一块的空 slot
```

- `pool.k[blocks]` 是一次 gather（GPU 上是 gather kernel），按 block table 的顺序取块；
- `reshape` 把“块 × 槽位”压平成 token 顺序，顺序天然正确，因为 block table 就是按逻辑块顺序排的；
- 末尾的 `[:length]` 处理部分填充的块。

这一步就是 PagedAttention kernel 在 GPU 上做的事。真实 kernel 不会先把整段 K/V 拷成一个连续张量（那会多一次显存读写），而是在计算每个 query 的分数时按 block table 逐块读取；这里为了可读性，先 gather 成连续张量再算 attention。

## attention：和连续布局的公式完全一样

```text
scores = Q·Kᵀ / √d，causal mask，softmax，乘 V
```

`_attention()` 拿到的是 `gather()` 之后的 `(length, heads, dim)`，所以 mask 逻辑和 `kv_cache.py` 一样：

```python
q_len, k_len = ...
first_query_position = k_len - q_len
causal = arange(k_len)[None, :] <= (first_query_position + arange(q_len))[:, None]
```

query 的绝对位置仍然是 `k_len - q_len + i`，与物理块号无关。块表只影响“K/V 从哪里取”，不影响“谁是第几个 token”。

## 为什么结果一定正确

block table 只是一次**置换**：逻辑块 i 的 token 被写到某个物理块，读的时候又按同一张表取回来，顺序不变。因此：

```python
paged  = attention(query, gather(sequence))
dense  = attention(query, cat(history_k), cat(history_v))
assert torch.allclose(paged, dense, atol=0, rtol=0)
```

示例里每次 `feed()` 都会对当前这一步做这个断言；运行结果是最大差异 `0`。

演示的真实输出（`block_size=4`，8 个块）：

```text
seq-a: 长度 10
  逻辑块 0: token 0-3 → 物理块 b5
  逻辑块 1: token 4-7 → 物理块 b1
  逻辑块 2: token 8-9 → 物理块 b3（2/4）
seq-b: 长度 6
  逻辑块 0: token 0-3 → 物理块 b0
  逻辑块 1: token 4-5 → 物理块 b4（2/4）
```

物理块是 `b5, b1, b3`——完全不连续，但 attention 结果与连续缓存一致。

## 释放与复用

```python
cache.free(seq_a)          # 4 块回到空闲列表
cache.feed(seq_c, ...)     # 立刻复用其中两块
```

输出里 `seq-c` 拿到 `b2, b3`，说明块是按需分配、按释放顺序回收的。真实引擎还会做：

- 块表与引用计数：共享前缀/并行采样时多个序列指向同一物理块（copy-on-write 写时复制）；
- 池满时的抢占（preempt）：换出或重算某些请求的 KV；
- 按 token 数而不是块数计费。

## 与前面两份示例的关系

| 示例 | 关注点 |
| --- | --- |
| `kv_cache.py` | 单请求，K/V 追加到一个连续张量；cache 的数学与位置对齐 |
| `prefix_cache_demo.py` | 多请求间复用“前缀块”，重点是哈希、引用计数、LRU 淘汰（纯记账） |
| `paged_attention.py` | 块真的存 K/V，attention 通过 block table 取数；块分配、释放、碎片 |

`prefix_cache_demo.py` 里的“块”只有 id 和引用计数；这里的块有真实的 K/V 张量，`gather()` 就是 prefix cache 命中之后 attention 真正读数据的方式。vLLM 把两者合在一起：PagedAttention 负责块表与 kernel，prefix caching 负责让不同请求的块表指向相同的物理块。

## 和 vLLM 的对应关系

| 本文件 | vLLM |
| --- | --- |
| `BlockPool` | vLLM v1 的 `BlockPool` + `KVCacheBlock`（GPU 上的 block 张量） |
| `block_table` | 每个序列的 `block_table`，存在 GPU 上 |
| `PagedKVCache.gather()` | `csrc/attention/attention_kernels.cu` 里的按块读取（不 materialize 整段 K/V） |
| `append()` | KV cache manager（v0 的 `BlockSpaceManager`，v1 的 `KVCacheManager`）+ `reshape_and_cache` |
| `free()` | 调度器释放块、回收进 free queue |
| `block_size=4` | 默认 16（`--block-size` 可调） |
| 内部碎片统计 | vLLM 日志里的 “GPU KV cache size / Maximum concurrency” |

结合 vLLM 的参数看：

- `--gpu-memory-utilization`：允许 vLLM 用多少显存，启动时按剩余显存算出能放多少块，日志会打印 `GPU KV cache size: N tokens`；
- `--max-model-len`：单请求最大上下文，决定一个请求最多需要多少块，也影响并发数；
- `--block-size`：块越大，block table 越短、kernel 越高效，但最后一块的内部碎片越多；
- `/metrics` 里的 `vllm:gpu_cache_usage_perc`：块池使用率，接近 1 时调度器会开始抢占。

## 代码阅读顺序

1. `BlockPool`：块张量的形状和 allocate/release；
2. `PagedKVCache.append()`：`slot == 0` 才分配，块内偏移怎么算；
3. `PagedKVCache.gather()`：index → reshape → 截断，这三步是 paged 的核心；
4. `_attention()`：和 `kv_cache.py` 的 `cached_attention` 逐行对照；
5. `feed()` 与 `main()`：交错请求、释放复用、内部碎片统计，以及每一步的 `allclose` 断言。

## 阅读时可以检查的七个问题

1. 物理块号不连续时，怎么根据 token 位置找到它的 K/V？
2. 为什么 `append()` 只在 `slot == 0` 时分配新块？
3. `gather()` 里为什么先 `reshape` 再 `[:length]`？顺序由什么保证？
4. 最后一块只用了 2 个 slot，attention 会不会读到 0 初始化留下的垃圾？
5. paged 布局下 causal mask 为什么和连续布局完全一样？
6. 内部碎片从哪来？`block_size` 是不是越小越好？
7. 两个序列的 block table 能指向同一个物理块吗？写的时候要注意什么？

## 这份示例刻意没有处理的生产问题

- **块共享 / copy-on-write**：共享前缀时多个 block table 指向同一块，写入前要复制；本示例每个序列独占块。
- **kernel 融合**：真实 kernel 按块读 K/V 并直接算 attention，不会先 gather 成连续张量。
- **抢占与换出**：池满时 vLLM 会重算或换出请求，而不是像这里直接抛 `RuntimeError`。
- **GPU 上的 block table**、多序列 batch、chunked prefill、KV 量化、多卡并行与显存池化。
- **调度策略**：块分配与请求调度（FCFS、优先级）耦合在一起。

因此它适合理解“逻辑位置 → 物理块”的映射和 attention 的读取方式；生产系统会在同样的映射上换上 CUDA kernel、共享与调度策略。
