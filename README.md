# learnkvc

用三份最小可运行的示例代码，把 LLM 推理里的 **KV Cache** 从「单请求的重算与复用」讲到「多请求的共享块」，再到「PagedAttention 的块表与显存管理」；另附用 Docker 跑 Ollama / vLLM、在 macOS 上用 Ollama / llama.cpp 原生跑 Llama 的实操示例。

没有模型权重：`kv_cache.py` 用一个 2 层玩具模型把「无 cache 重算」和「KV cache 增量」跑在同一份代码里（需要 PyTorch），`prefix_cache_demo.py` 用纯标准库模拟服务端的块级前缀共享。每个结论都有可执行的代码和断言。

## 目录

- [为什么需要 KV Cache](#为什么需要-kv-cache)
- [目录结构](#目录结构)
- [KV Cache 的三个视角](#kv-cache-的三个视角)
  - [1. 单请求：从「每步重算」到「只算新 token」](#1-单请求从每步重算到只算新-token)
  - [2. 多请求：prefill 不重算共同前缀](#2-多请求prefill-不重算共同前缀)
  - [3. 显存管理：PagedAttention](#3-显存管理pagedattention)
- [快速开始](#快速开始)
- [Docker 示例：Ollama 与 vLLM](#docker-示例ollama-与-vllm)
  - [vLLM（CPU / GPU，PagedAttention）](#vllmcpu--gpupagedattention)
- [macOS 原生跑 Llama（Ollama / llama.cpp，Apple GPU）](#macos-原生跑-llamaollama--llamacppapple-gpu)
  - [Ollama（模型库 + 菜单栏 App）](#ollama模型库--菜单栏-app)
  - [llama.cpp（手动参数、Metal GPU）](#llamacpp手动参数metal-gpu)
  - [怎么选：Ollama 还是 llama.cpp](#怎么选ollama-还是-llamacpp)
  - [性能：谁更快](#性能谁更快)
  - [写代码：Qwen 还是 Llama？](#写代码qwen-还是-llama)
- [建议的阅读路线](#建议的阅读路线)
- [文档索引](#文档索引)
- [这个仓库是什么、不是什么](#这个仓库是什么不是什么)
- [License](#license)

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
├── macos/
│   ├── ollama/                     # 在 macOS 上原生跑 Ollama（Metal GPU）的示例
│   │   ├── run.sh                  # 检查/启动 serve + 拉模型 + 调用一次 API
│   │   └── README.md               # 三种安装、四种启动、Llama 选型、环境变量、与 Docker 版的差异
│   └── llama-cpp/                  # 用 llama.cpp 直接在 macOS 上跑 Llama 的示例
│       ├── run.sh                  # 启动 llama serve + 等 /health + 调用一次 API
│       └── README.md               # 安装、模型来源、参数表、API、KV cache 与调优
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

## macOS 原生跑 Llama（Ollama / llama.cpp，Apple GPU）

Docker 在 macOS 上跑的是 Linux 容器，拿不到 Apple GPU；想用 Metal 直接在 Mac 上跑 Llama，有两条路：省心用 Ollama，想调底层参数用 llama.cpp。

### Ollama（模型库 + 菜单栏 App）

看 [`macos/ollama/`](macos/ollama/)：

```bash
bash macos/ollama/run.sh                 # 后台起 serve，默认 llama3.2:3b
bash macos/ollama/run.sh --ctx 8192      # 显式设 8K（默认按显存取 4k/32k/256k）
bash macos/ollama/run.sh --app           # 用菜单栏 App 启动
bash macos/ollama/run.sh --install       # 没装时先跑官方安装脚本
bash macos/ollama/run.sh --dry-run       # 只打印将执行的命令
```

文档里写了三种安装方式（官方 App / 一键脚本 / Homebrew）、三种原生启动方式（菜单栏 App / `ollama serve` / `brew services`，另有脚本封装）、macOS 上 `launchctl setenv` 与 shell 环境变量的区别；还有一节专门讲 Llama 家族在 Mac 上的选型：1B/3B/8B/11B/70B/405B 与 Llama 4 的下载体积、长上下文 KV 开销、按内存挑模型的建议，以及 `num_ctx`、`OLLAMA_KV_CACHE_TYPE`、`OLLAMA_NUM_PARALLEL` 与 `ollama ps` 的对应关系——和 Docker 版是同一套 Ollama 参数，只是改用 Apple 统一内存。

### llama.cpp（手动参数、Metal GPU）

想看底层引擎、自己调 `-c` / `-ngl` / `-fa` / `-ctk/-ctv`，用 [`macos/llama-cpp/`](macos/llama-cpp/)：

```bash
# 官方脚本安装（Apple Silicon 预编译 Metal 版，装到 ~/.local/bin/llama）
curl -LsSf https://llama.app/install.sh | sh

# 起 OpenAI 兼容服务（默认端口 9931，注意别用 -c 0）
llama serve -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M -c 4096 -ngl auto

# 或本仓库脚本：自动等 /health、调一次 API、打印上下文与吞吐
bash macos/llama-cpp/run.sh --ctx 8192
bash macos/llama-cpp/run.sh --kv-type q8_0      # KV cache 量化，省一半

# 写代码请换 Coder 模型（默认的 Llama 3.2 3B 只用于演示）
bash macos/llama-cpp/run.sh \
  --hf unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M --ctx 8192
```

文档里写了三种安装方式（官方脚本 / Homebrew / 源码）、模型来源（`-hf` / 本地 GGUF）、编程模型选型（Llama 3.2 3B 不适合写代码，16 GB Mac 上 7B / 14B Coder 怎么挑）、OpenCode 对接（custom OpenAI-compatible provider + 样例配置）、常用参数表、OpenAI 兼容 API，以及 `-c` / `-ctk/-ctv` 与 KV cache 大小的对应关系。

### 怎么选：Ollama 还是 llama.cpp

两者在 Apple Silicon 上都走 Metal + 统一内存，跑同一个 7B Q4 模型的速度和内存占用很接近；差别主要在模型管理、参数控制和接入成本。以本仓库这台测试机（Apple M2 / 16 GB / macOS 14.6 Sonoma）为例：

| 维度 | Ollama | llama.cpp |
| --- | --- | --- |
| 安装（Sonoma） | 官方 App / `install.sh` 直接可用；**不要用 `brew install ollama`**（14 没有 bottle，会源码编译） | `llama.app` 的 `install.sh` 直接可用（M2 有预编译 Metal 包）；**`brew install llama.cpp` 同样会源码编译** |
| Metal GPU | ✅ | ✅ |
| 模型管理 | 内置模型库、`ollama pull/rm`、自动更新 | `-hf` 从 Hugging Face 拉 GGUF，或本地 `.gguf`；模型缓存在 `~/Library/Caches/llama.cpp` |
| 参数控制 | `OLLAMA_*` 环境变量 + `num_ctx` | 全部底层参数：`-c`、`-ngl`、`-fa`、`-ctk/-ctv`、`-b/-ub`、`-np` |
| 默认上下文 | 按显存档位自动选（16 GB 机器通常是 4k，以启动日志 `vram-based default context` 为准），Agent 场景要显式 `OLLAMA_CONTEXT_LENGTH=32768` | 默认 `-c 0` = 模型训练长度（Llama 3.2 是 128K，容易吃爆内存），必须显式 `-c` |
| OpenCode | 内置 provider + 自动发现，`ollama launch opencode` 一键可跑 | 需要 custom provider 配置（本文档附了 `opencode.jsonc` 样例） |
| 更新 | App 自动更新 / 重跑安装脚本 | 重跑 `llama.app` 安装脚本或 `llama update` |
| 适合 | 日常使用、省心、写代码接 OpenCode | 调参、对照 KV cache、需要最新特性/单一二进制 |

### 性能：谁更快

同一模型、同一量化、同一上下文、同一 prompt 下，Ollama 和 llama.cpp 的差距通常只有个位数到 ~10%：Ollama 底层就是 llama.cpp（Apple Silicon 上还可能走 MLX runner），两边用的都是 Metal 内核，decode 阶段都受 M2 约 100 GB/s 内存带宽的限制——量级由硬件决定，不是框架。

- **Ollama 的优势是“默认就调得差不多”**：新版默认自动 FlashAttention、按显存调度，Apple Silicon 上还有 MLX runner，不用自己试参数。
- **llama.cpp 的优势是“能调”**：`-fa on`、`-ngl all`、`-ctk/-ctv q8_0`（减少 KV 读取带宽，长上下文下可能更快）、`-b/-ub`、`-lm`。Ollama 也能设 `OLLAMA_FLASH_ATTENTION`、`OLLAMA_KV_CACHE_TYPE`，两边都调好后性能会趋同。
- **公平对比要固定条件**：同一模型/量化、同一 `-c`/`num_ctx`、同一 prompt 与输出长度、模型已预热，并且把 prefill（pp）和 decode（tg）分开看；不要拿默认 4k 上下文的 Ollama 和 32k 的 llama.cpp 直接比。
- 本机（M2 / 16 GB）跑 7B Q4：decode 大致是十几 tok/s 的量级，具体随上下文长度、KV 量化、内存压力和散热波动；16 GB 内存和上下文长度对体验的影响，比“选哪个框架”更大。

```bash
# Ollama：/api/generate 返回 eval_count / eval_duration（纳秒）
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen2.5-coder:7b","prompt":"用一句话解释 KV cache","stream":false,"options":{"num_ctx":32768}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("decode", round(d["eval_count"]/d["eval_duration"]*1e9, 2), "tok/s"); print("prefill", round(d["prompt_eval_count"]/d["prompt_eval_duration"]*1e9, 2), "tok/s")'

# llama.cpp：启动时加 --metrics，看最近窗口的吞吐
curl -s http://127.0.0.1:9931/metrics | grep -E 'prompt_tokens_seconds|predicted_tokens_seconds'

# 或直接跑离线基准（prompt 512 / 生成 128）
llama bench -m model.gguf -p 512 -n 128
```

### 写代码：Qwen 还是 Llama？

**写代码选 Qwen 的 Coder 系列，Llama 更适合通用对话/推理。**

- Qwen 有专门的代码模型：Qwen2.5-Coder 覆盖 0.5B/1.5B/3B/7B/14B/32B，是按代码生成、修复、补全（FIM）训练的；Qwen3-Coder 进一步面向 Agent 场景（256K 上下文、专门的 function call 格式，支持 Qwen Code / CLINE / Claude Code 这类工具）。官方模型卡和公开代码基准上，Qwen Coder 系列在同尺寸里通常明显强于通用模型。
- Llama 3.x / Llama 4 是通用模型，强项在通用对话、推理、多语言和安全对齐，代码不是主攻方向。文档里默认的 Llama 3.2 3B 更是“能跑 demo”的尺寸，写代码只能回答简单问题。

这台 16 GB M2 上的现实选择：

| 模型 | 代码能力 | 16 GB 可行性 | 用途 |
| --- | --- | --- | --- |
| Qwen2.5-Coder-7B-Instruct Q4（4.36 GB，32K） | 强（同尺寸里第一梯队） | ✅ 32K 上下文很稳 | 补全/FIM、单文件、小项目；**Agent 工具调用不稳定** |
| Qwen3.5-4B（4.0 GB，256K，MLX/tools/thinking） | 中上，偏 Agent | ✅ 32K 很稳；64K 也放得下 | Agent 的**速度优先**选择 |
| Qwen3.5-9B（6.6–7.6 GB，256K，tools/thinking） | 更强，偏 Agent | ✅ 32K 可 100% GPU；64K 接近 10.7 GiB 上限 | Agent 的**质量优先**选择 |
| Qwen3-Coder-30B-A3B Q4（17.3 GB，256K，MoE） | 最强本地 Agent 级 | ❌ 24 GB+ | 仓库级/Agent 任务，大内存机器 |
| Llama 3.1 8B Q4（4.9 GB，128K） | 中等，通用型 | ⚠️ 能跑但写代码一般 | 通用对话/推理，不建议专用于编程 |
| Llama 3.2 3B（2.0 GB） | 弱 | ✅ | 演示、轻量问答 |
| Llama 3.3 70B / Llama 4 | 强但通用 | ❌ 43–245 GB | 大内存/服务器 |

- 接 OpenCode / Cline 时还要看**工具调用**：Qwen2.5-Coder-7B 虽然代码强，但实测常把工具调用输出成裸 JSON、不产生 `tool_calls`，Agent 会“只聊天不改文件”；Agent 场景优先 `qwen3.5:4b`（快）或 `qwen3.5:9b`（质量更好、32K 仍能 100% GPU），复杂任务再上云端/更大模型。
- **64K 上下文**：OpenCode 自身没有全局 64K 默认值（catalog 外模型 fallback 是 200K/32K），64K 是 Ollama 接入 OpenCode 时的建议值。Ollama 会把 `num_ctx` 压回模型训练长度，`qwen2.5-coder:7b` 设 64K 实际仍是 32K；llama.cpp 可以用 `--rope-scaling yarn --rope-scale 2` 把它外推到 64K（16 GB 能跑，但质量有损）。细节见 [`macos/llama-cpp/README.md`](macos/llama-cpp/README.md) 的「64K 上下文」。
- 结论：**补全/直接写代码用 Qwen2.5-Coder-7B；Agent 写代码用 Qwen3.5-4B；Llama 留给通用对话，或等你有 24 GB+ 内存再考虑更大的 Qwen3-Coder。** 参考：[Qwen2.5-Coder 发布博客](https://qwenlm.github.io/blog/qwen2.5-coder-family/)、[Qwen3-Coder 仓库](https://github.com/QwenLM/Qwen3-Coder)、[Ollama qwen2.5-coder](https://ollama.com/library/qwen2.5-coder)、[Ollama qwen3-coder](https://ollama.com/library/qwen3-coder)。

**这台 16 GB M2 上的建议：**

- **主力用 Ollama**：装官方 App 或 `curl -fsSL https://ollama.com/install.sh | sh`，拉 `qwen2.5-coder:7b`，接 OpenCode 几乎零配置；把上下文设到 32K（16 GB 上 64K 会和系统抢内存，7B 原生窗口也只有 32K）。
- **llama.cpp 留作调参/实验**：仓库里的 `macos/llama-cpp/` 用来看 `-c`、`-ctk/-ctv`、`-ngl` 对 KV cache 和 Metal 卸载的影响；OpenCode 也能接，但要自己维护 provider 配置。
- 两者可以并存（Ollama 11434、llama.cpp 9931），但**不要同时各加载一个 7B**，16 GB 会吃紧。
- 真正吃力的不是框架，是 16 GB 统一内存和 M2 的内存带宽：Agent 长任务/读大仓库建议用云端模型，本地 7B + 32K 适合补全、单文件修改和小项目。

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
| [`macos/ollama/README.md`](macos/ollama/README.md) | macOS 原生跑 Ollama/Llama（Metal GPU）：安装、App/brew/serve 启动、Llama 选型与内存、Qwen3-Coder-30B 的显存与 prefill/decode 账、OpenCode / Cline 对接、环境变量、卸载 |
| [`macos/llama-cpp/README.md`](macos/llama-cpp/README.md) | macOS 上用 llama.cpp 跑 Llama：安装、`-hf` 模型、编程模型选型、OpenCode 对接、参数表、API、KV cache 与调优 |
| [`docker/vllm/README.md`](docker/vllm/README.md) | Docker 跑 vLLM：PagedAttention 参数、块池日志与指标 |

## 这个仓库是什么、不是什么

**是**：理解 KV cache 数据结构与生命周期的教科书式代码，逻辑完整、可断言、可修改。

**不是**：可以直接上线的高性能实现。示例刻意省略了真实显存分配、paged attention kernel、decode 阶段的块管理、chunked prefill、多租户隔离、KV 量化与分布式共享等生产问题；每份文档末尾都列出了这些边界。

## License

[Apache License 2.0](LICENSE)
