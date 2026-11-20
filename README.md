# learnkvc

用三份最小可运行的示例代码，把 LLM 推理里的 **KV Cache** 从「单请求的重算与复用」讲到「多请求的共享块」，再到「PagedAttention 的块表与显存管理」；另附用 Docker 跑 Ollama / vLLM 的实操示例。

没有模型权重：`kv_cache.py` 用一个 2 层玩具模型把「无 cache 重算」和「KV cache 增量」跑在同一份代码里（需要 PyTorch），`prefix_cache_demo.py` 用纯标准库模拟服务端的块级前缀共享。每个结论都有可执行的代码和断言。

## 为什么需要 KV Cache

自回归生成时，第 `t` 个 token 的 attention 需要前 `t-1` 个 token 的 K/V。不做缓存，每生成一个 token 都要把整个前缀的 K/V 重算一遍，总计算量随生成长度平方增长；缓存后每步只算新 token，总计算量线性增长。

再往上一层：多个请求常常共享同一段 prompt 前缀。如果每个请求都重新 prefill 这段前缀，同样的计算会被重复做很多次——这正是 prefix cache 要消除的浪费。

## 目录结构

```text
.
├── docker/
│   ├── ollama/                     # 用 Docker 跑 Ollama + Llama 的示例
│   │   ├── run.sh                  # CPU/GPU 启动 + 拉模型 + 调用一次 API
│   │   └── README.md               # GPU 前提、API、上下文与 KV cache 参数
│   └── vllm/                       # 用 Docker 跑 vLLM（CPU / GPU）的示例
│       ├── run.sh                  # GPU 启动 + 等待就绪 + 调用 OpenAI 接口
│       └── README.md               # 镜像/模型来源、参数、块池指标
├── example/
│   ├── kv_cache.py                 # 单请求：无 cache 重算 vs KV cache 增量 + 完整缓存实现
│   ├── KV_CACHE_READING.md         # 逐段阅读导图：两条流程 + 缓存实现
│   ├── prefix_cache_demo.py        # 多请求共享前缀：块池 + 哈希匹配 + 引用计数 + LRU
│   ├── PREFIX_CACHE_READING.md     # 逐段阅读导图
│   ├── paged_attention.py          # 块表 + PagedAttention：逻辑位置 → 物理块
│   └── PAGED_ATTENTION_READING.md  # 逐段阅读导图
├── LICENSE
└── README.md
```

## KV Cache 的三个视角

### 1. 单请求：从「每步重算」到「只算新 token」

`example/kv_cache.py` 一个文件里做了四件事：

- **实现缓存**：`LayerKVCache` / `KVCache` 缓存每层 K/V，`cached_attention` 负责追加、GQA 展开、按绝对位置对齐 causal mask，还支持滑动窗口、beam search 的 `reorder()` 和分支 `fork()`；
- **跑两条路径**：同一个 `TinyModel`，一条每步重喂整段前缀，一条 prefill 后每步只喂 1 个 token；
- **断言一致性**：两条路径生成的 token 必须相同，同一段 `[prefix + token]` 的 logits 差异小于 `1e-5`（实测 `4.47e-08`，来自浮点累加顺序），并顺带自测滑动窗口、reorder、fork；
- **标注 GPU 路径**：每个张量算子旁注明了 CUDA 上走 cuBLAS 还是 ATen CUDA kernel（以及 `cat`/H2D/D2H 这类隐式开销），`KVCACHE_DEVICE=cuda` 即可整套搬到 GPU。

prompt 8 个 token、生成 8 个 token 的实际对比：

```text
  步骤   新 token    无 cache 输入    有 cache 输入
     0         30             8 个             8 个
     1         15             9 个             1 个
     2         15            10 个             1 个
...
  位置前向（token 穿过全部层）            无 cache       92    有 cache       15   6.1x
  attention (query,key) 对数（各层合计）  无 cache     1192    有 cache      240   5.0x

  白白重算的位置：77 个（算出了和之前完全相同的 K/V）
  生成结果：[30, 15, 15, 15, 41, 41, 41, 41]
```

代码顺序就是阅读顺序；逐段讲解见 [`example/KV_CACHE_READING.md`](example/KV_CACHE_READING.md)。

### 2. 多请求：prefill 不重算共同前缀

`example/prefix_cache_demo.py` 把视角抬高一层，模拟一个服务多个请求的推理引擎：

```text
请求 A: [ system prompt ][ few-shot ][ 问题 A ]
请求 B: [ system prompt ][ few-shot ][ 问题 B ]
              └──────┬──────┘
                共享的 K/V 块，只 prefill 各自的问题
```

KV 显存被切成固定大小的块（`block_size = 4`），块的身份是 `hash(父块, 本块 token)`，于是「相同前缀 → 相同块」，前缀匹配就是沿哈希链走。块用引用计数共享，池满时按 LRU 淘汰空闲叶子块；请求结束后块仍留在池里等待被复用。这份示例只模拟块管理（纯 Python，跑在 CPU 上）；真实引擎里块是 GPU 显存，由 paged attention kernel 按 block table 读取。

示例跑三个场景：

- **场景 1**：客服机器人的 5 条请求共享 system prompt / few-shot。输出命中率、每请求 prefill 量、块的新建/复用情况；结果与全量重算逐位一致（`verify_reuse_exact`）。
- **场景 2**：把块池缩小到 10 块，让 LRU 真的发生淘汰，观察「命中 → 淘汰 → 全量重算」的完整过程。
- **场景 3**：同一段 4 个 token 在 prompt 里出现两次，说明块哈希为什么必须带上父块——相同的 token 不等于相同的 K/V。

### 3. 显存管理：PagedAttention

`example/paged_attention.py` 把前两份示例接起来：KV 真的存放在固定大小的块里，每个请求用一张 **block table** 把逻辑位置映射到物理块，attention 通过块表取数。

```text
token 位置 → 逻辑块 + 槽位 → block_table → 物理块
```

示例让两个请求交错 prefill/decode，物理块是 `b5, b1, b3` 这种不连续编号，但每一步 attention 都与连续缓存逐位一致（差异 0）；随后释放一个请求、让第三个请求复用它的块，并统计最后一块产生的内部碎片。

```text
逻辑块 0: token 0-3 → 物理块 b5
逻辑块 1: token 4-7 → 物理块 b1
逻辑块 2: token 8-9 → 物理块 b3（2/4）
```

逐段讲解见 [`example/PAGED_ATTENTION_READING.md`](example/PAGED_ATTENTION_READING.md)；vLLM 的 `--block-size`、`--gpu-memory-utilization` 就是这套机制的真实参数。

## 快速开始

只需要 Python 3.10+。

```bash
# 单请求：无 cache vs KV cache（需要 PyTorch）
pip install torch
python3 example/kv_cache.py

# 有 CUDA 会自动用 GPU；也可以显式指定（cpu / cuda / mps）
KVCACHE_DEVICE=cuda python3 example/kv_cache.py
```

```bash
# PagedAttention：块表 + 物理块（需要 PyTorch）
python3 example/paged_attention.py

# 有 CUDA 会自动用 GPU；也可以显式指定
PAGED_DEVICE=cuda python3 example/paged_attention.py
```

```bash
# 多请求前缀共享：零依赖，直接跑
python3 example/prefix_cache_demo.py
```

`prefix_cache_demo.py` 的输出是请求时间线 + 命中统计，例如：

```text
[t=0] admit  req-shipping-2 prompt= 30 复用 6 块=24 tok       prefill=  6 tok  持有块=b00,b01,b02,b03,b04,b05,b08
[t=2] admit  req-tech-4     prompt= 49 复用 12 块=48 tok      prefill=  1 tok  持有块=b00,b01,b02,b03,b04,b05,b09,b10,b11,b12,b13,b14
...
汇总：共享前缀场景
  prompt tokens       : 184
  复用（命中）tokens  : 96  (52.2%)
  实际 prefill tokens : 88  (占 47.8%)
  相当于省下的 prefill: 96 个 token（不提速 decode，只减少首 token 前的计算量）
```

`example/` 里的三份代码都是确定性输出（固定种子、greedy 解码），重复运行结果一致；末尾都会跑断言，失败说明实现被改坏了。

## Docker 示例：Ollama 与 vLLM

想先跑一个真实模型、再回头看 KV cache 的实现，可以用 [`docker/ollama/`](docker/ollama/) 里的示例：

```bash
bash docker/ollama/run.sh                 # 默认 llama3.2:3b，自动判断 CPU/GPU
MODELS="llama3.2:3b qwen2.5:3b" bash docker/ollama/run.sh   # 一次拉多个
bash docker/ollama/run.sh --model llama3.1:8b               # 也可重复 --model
bash docker/ollama/run.sh --dry-run       # 只打印将执行的 docker 命令
```

模型列表由环境变量 `MODELS`（空格/逗号分隔）或可重复的 `--model` 指定，`MODEL` 保留为单模型别名。镜像是 Docker Hub 的 `ollama/ollama`，模型默认来自 [Ollama 官方模型库](https://ollama.com/library)（`llama3.2:3b` 的完整坐标是 `registry.ollama.ai/library/llama3.2:3b`，也支持 Hugging Face 上的 GGUF）。脚本会启动容器、等待就绪、逐个拉取模型、调用一次 `/api/generate`，并打印 `ollama ps` 的内存占用；README 里还有 GPU 前提、API 用法，以及 `num_ctx` / `OLLAMA_KV_CACHE_TYPE` 和 KV cache 大小的对应关系。

### vLLM（CPU / GPU，PagedAttention）

```bash
bash docker/vllm/run.sh                              # 有 GPU 用 GPU，否则用官方 CPU 镜像
bash docker/vllm/run.sh --cpu                        # 强制 CPU（vllm/vllm-openai-cpu）
bash docker/vllm/run.sh --gpu                        # 强制 GPU（vllm/vllm-openai）
VLLM_CPU_KVCACHE_SPACE=8 DTYPE=bfloat16 bash docker/vllm/run.sh --cpu
HF_ENDPOINT=https://hf-mirror.com bash docker/vllm/run.sh --cpu   # HF 下载慢时
```

脚本会按 `nvidia-smi` 自动选择镜像，等待 `/health`，从日志里打印 KV cache 块规模，并调用一次 `/v1/chat/completions`。README 里还有 CPU/GPU 前提、模型来源、`--block-size` / `--gpu-memory-utilization` / `VLLM_CPU_KVCACHE_SPACE` 与块池大小的关系，以及 `/metrics` 的用法。

## 建议的阅读路线

1. 先读 [`example/KV_CACHE_READING.md`](example/KV_CACHE_READING.md)，对着 `kv_cache.py` 理解「无 cache 为什么在重复算、`past` 为什么要先读、mask 为什么要按绝对位置对齐、位置嵌入为什么要接着历史」。
2. 再读 [`example/PAGED_ATTENTION_READING.md`](example/PAGED_ATTENTION_READING.md)，对着 `paged_attention.py` 理解「逻辑位置怎么映射到物理块、gather 为什么能还原顺序、内部碎片从哪来」。
3. 然后读 [`example/PREFIX_CACHE_READING.md`](example/PREFIX_CACHE_READING.md)，对着 `prefix_cache_demo.py` 理解「块哈希为什么带父块、引用计数和 LRU 怎么配合、为什么只能淘汰叶子块」。
4. 用文档末尾的问题自测，再动手改参数（生成长度、模型层数、`block_size`、块池容量、请求到达顺序），观察计算量和命中率变化。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [`example/KV_CACHE_READING.md`](example/KV_CACHE_READING.md) | 无 cache 流程、`append()`、`cached_attention`、mask 与位置对齐、GQA、滑动窗口、`reorder()`/`fork()`、两条生成循环的统计口径 |
| [`example/PREFIX_CACHE_READING.md`](example/PREFIX_CACHE_READING.md) | 块池、链式哈希、前缀匹配、引用计数、LRU + 叶子优先淘汰、与 vLLM / SGLang 的对应关系 |
| [`example/PAGED_ATTENTION_READING.md`](example/PAGED_ATTENTION_READING.md) | PagedAttention：块池、block table、gather、内部碎片、与 vLLM 的对应关系 |
| [`docker/ollama/README.md`](docker/ollama/README.md) | Docker 跑 Ollama/Llama：CPU/GPU、API、上下文与 KV cache 参数 |
| [`docker/vllm/README.md`](docker/vllm/README.md) | Docker 跑 vLLM：PagedAttention 参数、块池日志与指标 |

## 这个仓库是什么、不是什么

**是**：理解 KV cache 数据结构与生命周期的教科书式代码，逻辑完整、可断言、可修改。

**不是**：可以直接上线的高性能实现。示例刻意省略了真实显存分配、paged attention kernel、decode 阶段的块管理、chunked prefill、多租户隔离、KV 量化与分布式共享等生产问题；每份文档末尾都列出了这些边界。

## License

[Apache License 2.0](LICENSE)
