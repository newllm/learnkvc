# `prefix_cache_demo.py` 阅读导图

上一篇 `kv_cache.py` 解决的是**一个请求内部**的重复计算：生成第 `t` 个 token 时，不要重算前 `t-1` 个 token 的 K/V。

这一篇再往上一层：一个推理服务同时（或先后）接到很多请求，它们的 prompt 往往共享一大段前缀——同一个 system prompt、同一批 few-shot 示例、同一篇长文档、同一段多轮对话历史。既然这段前缀的 K/V 已经在显存里算过一次，后来的请求就不该再 prefill 一遍。

这就是生产引擎里的 **prefix caching**：vLLM 叫 automatic prefix caching，SGLang 叫 RadixAttention。这份实现用 `block` + `hash` + `ref_count` + `LRU` 四个概念，把它压缩成一份可以逐行读完、能跑出数字的模拟。

## 先建立整体图像

```text
请求 A: [ system prompt ][ few-shot ][ 问题 A ]
请求 B: [ system prompt ][ few-shot ][ 问题 B ]
              └──────┬──────┘
                共享的 K/V 块
                     │
                只 prefill 问题 B 的部分
```

块的画法（`block_size = 4`，每块 4 个 token）：

```text
块池:  [b00][b01][b02][b03][b04][b05]  [b06][b07]  [b08][b09]
        └──── system prompt ────────┘  └ A 的尾巴 ┘ └ B 的尾巴 ┘
        └──────── A 和 B 都指向这里 ────────┘
```

`req-b` 的 block table 只需要写成 `[b00,b01,b02,b03,b04,b05,b08,b09]`，不需要复制任何 K/V 张量。**共享发生在指针层，不发生在数据层。**

两个示例的分工：

| | `kv_cache.py` | `prefix_cache_demo.py` |
| --- | --- | --- |
| 复用粒度 | token（张量拼接） | 块（固定 4 token） |
| 服务对象 | 一个请求的整个生成过程 | 多个请求之间 |
| 核心操作 | `cat(old, new)` | 哈希匹配 + 引用计数 |
| 解决的问题 | decode 不重算前缀 | prefill 不重算共同前缀 |
| 依赖 | PyTorch | 仅标准库 |

两者不冲突：真实引擎里，**每个请求有自己的 block table（本篇）**，**每个请求的每个 token 又有自己的 K/V 位置（上篇）**。

## 数据结构：三层

### 1. `Block`：一块物理 KV 显存

```python
block_id    # 物理编号，打印时显示为 b03
block_hash  # 内容指纹，块池的 key
tokens      # 这 4 个 token 是什么
parent_hash # 上一块的指纹，形成链
state       # 本块的"K/V 内容"（本篇用 64 位状态代替真实张量）
ref_count   # 多少活跃请求正持有它
last_used   # LRU 时间戳
children    # 有哪些块以它为父块
```

`state` 是这份代码唯一的"假"东西。真实引擎里一块显存装的是某层 K/V 的一段，形状约为 `2 × (kv_heads, block_size, head_dim)`；这里换成一个 64 位整数，好处是**可以在最后断言"复用结果 == 全量重算结果"**，而不是只说一句"应该没问题"。

### 2. `PrefixCache`：块池 + 匹配 + 淘汰

对外只有两个动作：

```python
match = cache.acquire(prompt_tokens)   # 进请求：匹配前缀、占用/新建块
cache.release(match.block_hashes)      # 出请求：归还引用（块仍留在池里）
```

内部的 `_make_room` / `_pick_victim` / `_evict` 负责在池满时腾地方。

### 3. `Simulator`：把请求放进时间线

`admit -> decode 1 token -> finish` 的循环，模拟连续批处理服务的调度。它本身不是重点，重点看 `_admit` 里那 20 行：拿到 `match` 后先 `run_state` 补齐没命中的 token，再把请求塞进 active 列表。

## 最关键的一行：块的身份

```python
# 伪代码：真实实现把 parent_hash 和 tokens 编码成字节，再做 blake2b
def block_hash(parent_hash, tokens):
    return hash(parent_hash + tokens)
```

为什么哈希要带上父块？

- 只哈希这 4 个 token，`[system 的第 3 块]` 和 `[某段回答里的同样 4 个 token]` 会撞成同一个块，但它们前面的上下文不同，K/V 完全不同，共享就是错的。
- 带上父块哈希后，`hash(父, tokens)` 唯一确定"从根到这里"的整条 token 前缀。相同前缀 → 相同块 → 相同 K/V。

这条链还带来一个免费的结论：**前缀匹配就是沿链走**。走不动的第一个块，就是命中的边界；不用做字符串匹配，也不用比较 token 数组。

> 真实引擎还会在哈希里加模型名、LoRA 名、adapter、多模态 hash 等"salt"，避免不同模型/不同任务误共享。见文末"刻意没处理的问题"。

## `acquire()`：一次请求如何命中

```text
1. 按 block_size 把 prompt 切成 4 个 token 一组
2. 从第一组开始，算 hash(父块, 本组)，去块池里找
3. 找到：ref_count += 1，继续找下一组
4. 找不到：从这个位置开始，为后面的完整块新建 Block
5. 末尾不足 4 个 token 的部分：不建块，留给本轮 prefill
```

第 5 步是这个设计里最容易被忽略的取舍：**只复用整块**。所以场景 1 里 `req-tech-4` 和 `req-tech-3` 的 prompt 完全相同（49 个 token），命中却是 48 个 token、还要 prefill 1 个——因为 49 = 12 个整块 + 1 个零头。

如果为了多省这 1 个 token 而允许"半块复用"，就得让最后一个块同时承载两种前缀，块表和引用计数都会立刻变复杂。生产系统同样选择"半块重算"。

## `ref_count` 与 LRU：什么时候能淘汰

三个状态要分清：

```text
ref_count > 0   正在被某个活跃请求使用，绝不能淘汰
ref_count == 0  没人用了，但内容还在池里，随时可能被复用   ← 缓存的意义就在这里
被淘汰          从池里删除；下次同名前缀来了只能重新 prefill
```

`release()` 只做减引用，**不删块**。删块只发生在 `_make_room()`：池子满了，按 `last_used` 找最久没被碰过、且 `ref_count == 0` 的块淘汰。

还有一个容易写错的细节——**只能淘汰叶子块**：

```python
candidates = [b for b in blocks if b.ref_count == 0 and not b.children]
```

因为子块只能通过父块的哈希被找到、被复用。如果父块先被淘汰，子块还留在池里，它就变成永远无法命中的垃圾。所以淘汰顺序必须从链尾往前推，父块在子块消失后才变成候选。

## 为什么复用一定是对的

初学者最容易担心：复用别人的 K/V，结果会不会和重新算不一样？

不会，因为 **K/V 只取决于"从位置 0 到该 token 的整条前缀"**（同模型、同精度、同 RoPE 位置）。两个请求的 token 前缀逐位相同，那一段的 K/V 就逐位相同，只是它们恰好属于不同的序列。

代码没有让你"相信"，而是让你"验证"：

```python
def verify_reuse_exact(records):
    for record in records.values():
        recomputed = run_state(0, record.request.prompt)   # 从零全量算
        assert record.prefill_state == recomputed          # 复用路径的结果
```

`prefill_state` 是"复用前缀的 state + 补齐剩余 token"走出来的；`run_state(0, prompt)` 是把整个 prompt 从头算一遍。两个 64 位值必须完全相等，否则断言失败。把 `state` 传递写错，或者把 `block_hash` 改成只哈希 `tokens`（见场景 3），这个断言就会立刻抓到。

## 场景 1 的时间线怎么读

下面只保留每行开头的关键信息，实际输出还会在行尾列出请求持有的块（`持有块=b00,...`）。

```text
[t=0] admit  req-refund-1   无命中              prefill=32   ← 冷启动
[t=0] admit  req-shipping-2 复用 6 块=24 tok    prefill= 6   ← 命中 system prompt
[t=1] admit  req-tech-3     复用 6 块=24 tok    prefill=25   ← few-shot 还没人算过
[t=2] admit  req-tech-4     复用 12 块=48 tok   prefill= 1   ← 同上一个请求，只差零头
[t=3] admit  req-billing-5  无命中              prefill=24   ← 换了 system prompt
```

三个值得注意的点：

1. `req-tech-3` 没有命中 few-shot，因为它是第一个带 few-shot 的请求——**缓存只能复用"已经算过"的东西**。但它的 prefill 顺手把这 6 块写进了池子，于是下一个请求直接吃满。
2. `req-shipping-2` 命中 24 个 token、prefill 6 个；它实际持有 7 块（6 块共享 + 1 块自己的尾巴），但统计按"复用了多少 token"算。
3. 请求结束时 `free` 只减引用。所以 `req-refund-1` 释放后，它的块仍在池里，等待下一个相同前缀的请求。

汇总行里：

```text
prompt tokens       : 184
复用（命中）tokens  : 96  (52.2%)
实际 prefill tokens : 88  (占 47.8%)
```

这就是 prefix cache 的收益口径：**省的是 prefill 计算量（直接影响首 token 延迟 TTFT），不改变 decode 每步的速度**。decode 阶段每生成一个 token 仍要走完所有层，只是不再重算提示词。

## 场景 2：池子小的时候会发生什么

块池只有 10 块（40 token），`req-c` 需要 10 块但前缀与前面完全不同：

```text
[t=2] admit  req-c  无命中  prefill=40
[t=2] cache  淘汰 10 块（LRU，无人占用）：b07,b06,b09,b08,b05,b04,b03,b02,b01,b00
[t=4] admit  req-d  无命中  prefill=32   ← req-a 的块已经全没了
```

读懂淘汰顺序就懂了整个机制：

- `b07` 是 `req-a` 尾巴的叶子块，最久没用，先走；
- `b07` 走后 `b06` 变成叶子，跟着走；
- 然后轮到 `req-b` 的尾巴 `b09`、`b08`，最后才是共享的 `b05…b00`。

结果：`req-b` 还享受了 75% 命中，`req-a` 的块却已经被清空，`req-d` 再来时只能全量重算。**命中率不是一个静态数字，它取决于池子大小、请求顺序和淘汰策略。**

## 场景 3：相同的 4 个 token，不同的位置

prompt 里重复出现同一段内容（模板、表头、few-shot 里重复的示例）时，"块内容相同"和"K/V 相同"就会脱钩：

```text
req-x: [ prefix 12 tok ][ phrase ][ middle 8 tok ][ phrase ]
req-y: [ prefix 12 tok ][ phrase ][ tail ]
```

`phrase` 在 `req-x` 里出现了两次：位置 12 和位置 24。位置 24 的 K/V 里已经混进了 `middle` 的信息，和位置 12 的 K/V 并不相同。如果块的身份只由这 4 个 token 决定，两个 `phrase` 会映射到同一个块（后建的那个还会覆盖先建的），`req-y` 走到位置 12 时就会拿到"位置 24 的 K/V"。

时间线里 `req-y` 命中 4 块（prefix + 第一个 phrase = 16 token）、prefill 4 个 token（同样省略行尾的持有块列表）：

```text
[t=0] admit  req-x  prompt=28 无命中              prefill=28
[t=1] admit  req-y  prompt=20 复用 4 块=16 tok    prefill= 4
```

把 `block_hash` 改成只哈希 `tokens`，`verify_reuse_exact` 会立刻报 `req-y` 与全量重算不一致。这就是块哈希必须包含 `parent_hash` 的原因：**它编码的是"整条前缀"，不是"这 4 个 token"**。生产实现还会把模型名、LoRA、多模态哈希一起混进去，避免不同来源的块被误认为同一个。

## 代码阅读顺序

1. `mix()` / `run_state()` / `block_hash()`：三个纯函数，先看懂"state 是前缀的函数"和"块哈希是链式"的。
2. `Block` / `CacheStats` / `PrefixMatch`：只是数据，扫一眼字段。
3. `PrefixCache.acquire()`：这是最核心的函数，重点看"匹配到哪停"和"从哪开始建块"。
4. `PrefixCache._make_room()` / `_pick_victim()` / `_evict()`：淘汰的三段式，注意叶子约束。
5. `Simulator._admit()` / `_decode()` / `_finish()`：请求生命周期，`_admit` 里补齐剩余 token 的那两行是关键。
6. `verify_reuse_exact()` 与三个 `scenario_*` 里的断言：把"应该对"变成"跑得过"。

## 和真实引擎的对应关系

| 本文件 | vLLM | SGLang |
| --- | --- | --- |
| `Block` | `KVCacheBlock` + 一段 GPU 显存 | radix tree 节点 + token 段 |
| `block_hash` | block 的 token id 元组哈希（含 prefix） | radix tree 的路径 |
| `ref_count` | `ref_cnt` | `lock_ref` |
| `last_used` | free queue 的 LRU 顺序 | LRU 链表 |
| `children` / 叶子优先 | 空闲队列的 LRU 顺序 | LRU + 叶子优先 |
| block table（本文件是 `RequestRecord.block_hashes`） | `block_tables` | `req_to_token` |

本文件是纯 Python 模拟：块表、引用计数和淘汰都跑在 CPU 上，不调用任何 GPU API。真实引擎里每块对应 GPU HBM 上的一段 K/V，由 paged attention kernel（CUDA/Triton）按 block table 取数。

真实实现还要多做三件事，本文件省略了：

1. **decode 也要占块**：每生成一个 token 也要写 K/V，池子压力比这里大得多；
2. **attention kernel 要按块取数**：paged attention 用 block table 把逻辑位置映射到物理块号，kernel 才能跳着读显存；
3. **块大小通常是 16**（vLLM 默认），更省 block table 内存，但前缀命中粒度更粗。

## 这份示例刻意没有处理的生产问题

- 真实 K/V 张量、显存分配、paged attention kernel 和 block table 的索引计算。
- decode 阶段的块分配、chunked prefill（把长 prompt 分几次算）与调度。
- 更精细的淘汰：近似 LRU、按 token 数而非块数计费、radix tree 的节点分裂与合并。
- 前缀共享的隔离与安全：不同租户/不同采样参数的请求能否共享、哈希 salt、命中率侧信道。
- 分布式场景：多副本之间的前缀缓存、KV 传输（如 prefill-decode 分离）、跨节点共享。
- KV 量化、压缩、offload；以及命中率的上限分析（理论收益 = 命中 token 数 / 总 prefill token 数）。

它适合用来理解**数据结构与生命周期**；生产引擎是在同一套骨架里换上真实显存和 kernel。

## 阅读时可以检查的八个问题

1. 为什么块的哈希要包含父块哈希？只用本块的 4 个 token 会错在哪？
2. 为什么两个请求可以安全地读同一块 K/V？什么机制保证它不会被提前释放？
3. `release()` 之后块为什么还在池里？这和"释放显存"有什么不同？
4. 为什么 `req-tech-4` 与 `req-tech-3` 的 prompt 完全相同，`prefill` 却不是 0？
5. 为什么淘汰必须叶子优先？如果允许摘掉中间的块会发生什么？
6. `req-b` 只创建了 2 块，为什么它还"持有" 8 块？
7. 场景 1 的 `命中率 52.2%` 和 `省下 96 个 token` 分别刻画什么？为什么两者都要看？
8. 如果 `block_size` 从 4 改成 16，三个场景的命中率会怎么变？

## 可以动手改的实验

1. 把 `block_size` 改成 8，重跑，解释统计变化。
2. 加一个只共享前 3 个 token 的请求，确认它命中 0 块，并说明原因。
3. 让 `req-tech-3` 和 `req-tech-4` 同时到达（`arrival` 相同），观察 `ref_count` 变成 2 时淘汰是否仍会避开它们。
4. 去掉 `_pick_victim` 里的 `not block.children` 约束，跑场景 2，看看子块会不会变成永远命不中的孤儿。
5. 在 `print_summary` 里加一项"平均每请求 prefill token"，对比开/关前缀缓存。

如果你能不看代码说清"一个请求从 `acquire` 到 `release` 期间，块池里每个块经历了什么"，这份实现就吃透了。
