# 在 macOS 上跑 Ollama / Llama（Apple GPU / Metal）

在 Mac 上直接跑 [Ollama](https://ollama.com/)：装官方 App 或 CLI，拉一个 Llama 模型，用 HTTP API 对话。文档既讲 Ollama 本身的安装/启动，也讲 Llama 家族在 Mac 上怎么选模型、内存怎么算，另有一节把 Qwen3-Coder-30B（30B MoE 代码模型）的显存账、10K prompt 的 prefill 与每步 decode 占用算清楚。和 [`docker/ollama/`](../../docker/ollama/README.md) 是同一件事的两种跑法——那边是 Docker 容器（在 macOS 上只能用 CPU，Linux + NVIDIA 才用得到 GPU）；这边是 macOS 原生版，Apple Silicon 上会走 **Metal GPU**，模型权重和 KV cache 都放在统一内存里。

目录内容：

```text
macos/ollama/
├── run.sh     # 检查/安装 + 启动 serve + 拉模型 + 调一次 API 的脚本
└── README.md  # 本文件
```

## 目录

- [前置条件](#前置条件)
- [安装（三种方式，选一种）](#安装三种方式选一种)
  - [1. 官方 App（推荐）](#1-官方-app推荐)
  - [2. 官方一键脚本](#2-官方一键脚本)
  - [3. Homebrew（CLI + brew services）](#3-homebrewcli--brew-services)
- [启动（三种原生方式 + 本目录脚本）](#启动三种原生方式--本目录脚本)
  - [方式 A：菜单栏 App](#方式-a菜单栏-app)
  - [方式 B：终端前台 `ollama serve`](#方式-b终端前台-ollama-serve)
  - [方式 C：Homebrew service](#方式-chomebrew-service)
  - [方式 D：本目录的脚本（推荐先跑这个）](#方式-d本目录的脚本推荐先跑这个)
- [环境变量：macOS 上最容易踩的坑](#环境变量macos-上最容易踩的坑)
  - [App 模式：用 launchctl](#app-模式用-launchctl)
  - [终端 / 脚本模式：直接前缀](#终端--脚本模式直接前缀)
  - [Homebrew service 模式](#homebrew-service-模式)
  - [常用变量](#常用变量)
- [模型、目录与 API](#模型目录与-api)
  - [模型从哪来](#模型从哪来)
  - [文件都在哪](#文件都在哪)
  - [HTTP API](#http-api)
- [在 Mac 上跑 Llama：选型、内存与速度](#在-mac-上跑-llama选型内存与速度)
  - [按内存挑模型](#按内存挑模型)
  - [量化怎么选](#量化怎么选)
  - [速度由什么决定](#速度由什么决定)
  - [和 llama.cpp / MLX 的关系](#和-llamacpp--mlx-的关系)
- [Qwen3-Coder-30B：复杂代码开发的显存要求](#qwen3-coder-30b复杂代码开发的显存要求)
  - [结论：24 GB 独显起，统一内存按约 2/3 折算](#结论24-gb-独显起统一内存按约-23-折算)
  - [权重与 KV：每 token 96 KiB](#权重与-kv每-token-96-kib)
  - [Prefill：10K prompt 的额外占用](#prefill10k-prompt-的额外占用)
  - [Decode：每步只涨 96 KiB](#decode每步只涨-96-kib)
  - [Ollama 上怎么配](#ollama-上怎么配)
- [对接 OpenCode（编程 Agent）](#对接-opencode编程-agent)
- [对接 Cline（VS Code 编程 Agent）](#对接-clinevs-code-编程-agent)
  - [1. 启动 Ollama 并准备模型](#1-启动-ollama-并准备模型)
  - [2. Cline 里的设置](#2-cline-里的设置)
  - [3. 几个容易踩的坑](#3-几个容易踩的坑)
  - [4. OpenAI Compatible 兜底](#4-openai-compatible-兜底)
  - [5. Cline Desktop](#5-cline-desktop)
  - [常见问题：OpenCode Desktop 2.x 与 CLI 1.x 共库冲突](#常见问题opencode-desktop-2x-与-cli-1x-共库冲突)
- [上下文与 KV cache：和本仓库的关系](#上下文与-kv-cache和本仓库的关系)
- [和 Docker 版的差异](#和-docker-版的差异)
- [怎么自己测模型](#怎么自己测模型)
- [token 输出慢怎么定位](#token-输出慢怎么定位)
  - [看懂 MLX runner 的日志](#看懂-mlx-runner-的日志)
  - [一组 API 数字怎么算](#一组-api-数字怎么算)
  - [先关掉 thinking 再比速度](#先关掉-thinking-再比速度)
  - [让模型常驻（避免每次请求都等加载）](#让模型常驻避免每次请求都等加载)
  - [怎么判断有没有触发 offload](#怎么判断有没有触发-offload)
  - [prefill 到底占多少内存、为什么慢](#prefill-到底占多少内存为什么慢)
  - [每层在做什么：一层的数据流](#每层在做什么一层的数据流)
- [日常开发要多大上下文](#日常开发要多大上下文)
  - [一次 7m45s 请求的拆解](#一次-7m45s-请求的拆解)
- [能不能看算子级耗时（类似 Linux perf）](#能不能看算子级耗时类似-linux-perf)
  - [1. CPU 侧：`sample`（无需 Xcode）](#1-cpu-侧sample无需-xcode)
  - [2. GPU 侧：`powermetrics`（需要 sudo）](#2-gpu-侧powermetrics需要-sudo)
  - [3. 真正到 kernel：Xcode / Metal GPU Trace](#3-真正到-kernelxcode--metal-gpu-trace)
- [常见问题](#常见问题)
- [清理 / 卸载](#清理--卸载)
- [参考](#参考)

## 前置条件

| 项 | 要求 |
| --- | --- |
| 系统 | macOS Sonoma（14）或更新 |
| 芯片 | Apple Silicon（M 系列）：CPU + GPU（Metal）；Intel Mac：只有 CPU |
| 内存 | 模型大小 + 上下文开销；量化后的 8B 模型约占 5–6 GB，KV cache 另算（见下文） |
| 磁盘 | 模型存在 `~/.ollama/models`，一个 3B/8B 模型通常 2–5 GB |
| 网络 | 拉模型走 HTTPS（`registry.ollama.ai`），公司网络需要代理时只设 `HTTPS_PROXY` |

> Apple Silicon 是**统一内存**：模型和 KV cache 都从同一块内存里分配，和 CPU 共享。所以“显存够不够”在 Mac 上等价于“内存够不够”，同时开一堆大内存应用会直接影响推理。

## 安装（三种方式，选一种）

### 1. 官方 App（推荐）

从 <https://ollama.com/download/mac> 下载 `Ollama.dmg`，拖进 `/Applications`。第一次启动时 App 会检查 PATH 里有没有 `ollama` CLI，没有的话会请求权限在 `/usr/local/bin` 建一个符号链接（指向 `Ollama.app/Contents/Resources/ollama`）。

App 是菜单栏常驻程序，默认**登录自启**，并会自己检查更新（菜单栏图标 → Restart to update）。不想开机启动：系统设置 → 通用 → 登录项，把 Ollama 关掉。

### 2. 官方一键脚本

和 Linux 同一个脚本，macOS 分支会下载 `Ollama-darwin.zip`、解压到 `/Applications/Ollama.app`、建 `/usr/local/bin/ollama` 符号链接（可能需要输入密码），然后后台启动 App：

```bash
curl -fsSL https://ollama.com/install.sh | sh

# 只装不启动（之后自己决定用 App 还是 serve）
curl -fsSL https://ollama.com/install.sh | OLLAMA_NO_START=1 sh

# 装指定版本
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.16 sh
```

### 3. Homebrew（CLI + brew services）

```bash
brew install ollama
brew services start ollama     # 后台运行，登录自启；日志见下文
```

Homebrew 装的是 CLI 版，Apple Silicon 上同样带 Metal/MLX 后端。两个细节：

- formula 的 service 定义默认带 `OLLAMA_FLASH_ATTENTION=1` 和 `OLLAMA_KV_CACHE_TYPE=q8_0`，也就是**默认开了 FlashAttention、KV cache 量化成 q8_0**（比默认 f16 省一半 KV 内存）。想改就覆盖环境变量，见下文。
- 目前 macOS 14（Sonoma）没有对应的预编译 bottle，`brew install ollama` 会从源码编译（需要 cmake / go / ccache 和 Xcode CLT，耗时较长）；macOS 15+ 才有现成 bottle。在 Sonoma 上直接用方式 1 的 App 更省事。

装完确认：

```bash
command -v ollama        # /usr/local/bin/ollama 或 /opt/homebrew/bin/ollama
ollama --version
curl -s http://127.0.0.1:11434/api/version
```

> ⚠️ App 和 `brew services` 都会监听 `11434`，**只保留一种启动方式**，否则第二个会因为端口被占用起不来。

## 启动（三种原生方式 + 本目录脚本）

### 方式 A：菜单栏 App

装好即启动，之后每次登录自动运行。手动控制：

```bash
open -a Ollama                          # 启动（--args hidden 可后台启动，不弹窗口）
osascript -e 'quit app "Ollama"'        # 退出，服务随之停止
```

App 的日志在 `~/.ollama/logs/`：`app.log` 是 GUI 日志，`server.log` 是推理服务日志。

### 方式 B：终端前台 `ollama serve`

```bash
ollama serve
# 或者换端口/监听地址（默认 127.0.0.1:11434）
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

前台运行，日志直接打在终端，`Ctrl+C` 停止。适合调试参数、看启动日志。注意它和 App 抢同一个端口，先退出 App 再跑。

### 方式 C：Homebrew service

```bash
brew services start ollama     # 启动 + 登录自启
brew services restart ollama   # 改完环境变量后重启
brew services stop ollama      # 停止
brew services info ollama      # 看状态
tail -f "$(brew --prefix)/var/log/ollama.log"
```

### 方式 D：本目录的脚本（推荐先跑这个）

脚本把上面这些事串起来：找安装位置 → 后台启动 `ollama serve`（或交给 App）→ 等待就绪 → 拉模型 → 调一次 `/api/generate` → 打印 `ollama ps`。

```bash
bash macos/ollama/run.sh                     # 后台 serve，默认拉 llama3.2:3b

MODELS="llama3.2:3b qwen2.5:3b" bash macos/ollama/run.sh   # 一次拉多个
MODEL=llama3.1:8b bash macos/ollama/run.sh                 # 单模型别名
bash macos/ollama/run.sh --model qwen2.5:3b --ctx 8192     # 显式设默认上下文（8K）
bash macos/ollama/run.sh --app                             # 用菜单栏 App 启动
bash macos/ollama/run.sh --install                         # 没装时先跑官方安装脚本
bash macos/ollama/run.sh --dry-run                         # 只打印命令
bash macos/ollama/run.sh --status                          # 只查看状态
bash macos/ollama/run.sh --stop                            # 停掉本脚本起的后台 serve
```

默认行为是 `ollama serve` 后台运行（日志写到 `~/.ollama/logs/ollama-serve.log`，pid 记在 `~/.ollama/ollama-serve.pid`），因为这样 `--ctx`、`--host`、`--port` 这些参数才真正生效。**App 的环境变量不能这样传**，见下一节。

完整参数与默认值见 `bash macos/ollama/run.sh --help`（还有 `--timeout`、`--no-pull`、`--no-demo`，以及 `PULL` / `DEMO` / `HEALTH_TIMEOUT` / `OLLAMA_SERVE_LOG` / `OLLAMA_SERVE_PID_FILE` / `OLLAMA_BIN` 环境变量）。`--status` 需要带与启动时相同的 `--port` 才能查到那个实例；`--stop` 默认读取 pid 文件里记录的端口来确认归属（pid 文件路径可用 `OLLAMA_SERVE_PID_FILE` 覆盖）。`--dry-run` 只打印启动与拉取命令，不执行安装、启动和示例请求。

## 环境变量：macOS 上最容易踩的坑

`ollama serve` 从进程环境里读配置；而菜单栏 App 是图形界面程序，**在你的 shell 里 `export` 不会影响它**。三种启动方式各有一套设置方法：

### App 模式：用 launchctl

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
launchctl setenv OLLAMA_CONTEXT_LENGTH "8192"
launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"
# 然后重启 App：菜单栏 → Quit Ollama，再 open -a Ollama
```

注意 `launchctl setenv` 设置的是当前登录会话的环境，**重启/注销后失效**。想长期生效，可以写一个 LaunchAgent 在登录时执行 `launchctl setenv`，或者干脆用 shell 脚本/`brew services` 方式启动。

### 终端 / 脚本模式：直接前缀

```bash
OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

`run.sh` 会把调用方已经 `export` 的 `OLLAMA_*` 变量原样传给 serve，也支持 `--ctx`（= `OLLAMA_CONTEXT_LENGTH`）、`--host`、`--port` 参数。

### Homebrew service 模式

formula 已经默认设置了 `OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0`。要再加变量，可以用 `launchctl setenv` 后重启 service：

```bash
launchctl setenv OLLAMA_KEEP_ALIVE "30m"
brew services restart ollama
```

### 常用变量

| 变量 | 作用 |
| --- | --- |
| `OLLAMA_HOST` | 监听地址，默认 `127.0.0.1:11434`；`0.0.0.0:11434` 可让局域网访问，`127.0.0.1:11500` 可换端口 |
| `OLLAMA_MODELS` | 模型目录，默认 `~/.ollama/models`；可指到外置盘 |
| `OLLAMA_CONTEXT_LENGTH` | 默认上下文长度；不设时 Ollama 按显存选 **4k（<24 GiB）/ 32k（24–48 GiB）/ 256k（≥48 GiB）**，也是 `num_ctx` 的全局默认值 |
| `OLLAMA_KEEP_ALIVE` | 模型空闲多久卸载，默认 `5m`；`-1` 常驻 |
| `OLLAMA_NUM_PARALLEL` | 每个模型同时处理的请求数，默认 1；每路各占一份 KV cache |
| `OLLAMA_MAX_LOADED_MODELS` | 同时加载几个模型，默认 3（有 GPU 时 3×GPU 数） |
| `OLLAMA_MAX_QUEUE` | 忙时最多排队多少请求，默认 512，超出返回 503 |
| `OLLAMA_FLASH_ATTENTION` | 是否启用 FlashAttention；多数后端默认自动开启，`1` 强制、`0` 关闭 |
| `OLLAMA_KV_CACHE_TYPE` | KV cache 精度：`f16`（默认）、`q8_0`、`q4_0`（量化需要 FlashAttention） |
| `OLLAMA_ORIGINS` | 允许跨域访问的来源，默认只有 `127.0.0.1` / `0.0.0.0` |
| `OLLAMA_NO_CLOUD` | 设为 `1` 关闭云端模型与联网功能，纯本地运行 |
| `OLLAMA_DEBUG` | 设为 `1` 输出更详细的日志 |

完整列表可以直接看 CLI 自带帮助：`ollama serve --help`。

## 模型、目录与 API

### 模型从哪来

和 Docker 版一致：默认从 [Ollama 官方模型库](https://ollama.com/library)拉取，`name:tag` 的完整坐标是 `registry.ollama.ai/library/<name>:<tag>`；也支持 `hf.co/<user>/<repo>:<quant>` 这种 Hugging Face GGUF。Llama 家族各型号的体积、上下文和内存取舍见下一节。

```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b "用一句话解释 KV cache"
ollama list            # 或 ollama ls
ollama show llama3.2:3b
ollama rm llama3.2:1b
ollama stop llama3.2:3b   # 立刻卸载，释放内存
```

### 文件都在哪

| 路径 | 内容 |
| --- | --- |
| `~/.ollama/models` | 模型（`manifests/` 清单 + `blobs/` 权重分片） |
| `~/.ollama/logs/app.log`、`server.log` | App 与推理服务日志 |
| `~/.ollama/id_ed25519.pub` | 本机 Ollama 公钥（推模型/云端模型会用到） |
| `$(brew --prefix)/var/log/ollama.log` | Homebrew service 日志 |

换模型目录（比如外置盘）：

```bash
launchctl setenv OLLAMA_MODELS "/Volumes/SSD/ollama-models"   # App 模式后重启 App
OLLAMA_MODELS=/Volumes/SSD/ollama-models ollama serve          # 终端模式
```

### HTTP API

服务起来后地址是 `http://127.0.0.1:11434`：

```bash
MODEL="${MODEL:-llama3.2:3b}"

curl -s http://127.0.0.1:11434/                        # "Ollama is running"
curl -s http://127.0.0.1:11434/api/version
curl -s http://127.0.0.1:11434/api/tags

# 生成（stream=false 一次性返回）
curl -s http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"用一句话解释 KV cache\",\"stream\":false}"

# 对话
curl -s http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"stream\":false}"

# 当前加载的模型、占用与跑在 CPU 还是 GPU 上
curl -s http://127.0.0.1:11434/api/ps
```

OpenAI 兼容接口也在同一个端口：

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

Python 用 `openai` SDK 时把 `base_url` 设为 `http://127.0.0.1:11434/v1/`、`api_key` 随便填（如 `"ollama"`，本地服务会忽略）。

## 在 Mac 上跑 Llama：选型、内存与速度

Ollama 模型库里的 Llama 都是 Meta 官方模型，下面的 tag 都能直接 `ollama pull`。体积是 Ollama 默认 4-bit 量化（q4_K_M 级别）的下载大小；KV 列是 `num_ctx=128K` 时 fp16 KV cache 的**估算**（公式见下一节），不是模型权重。

| 模型 | tag 示例 | 下载体积 | 上下文 | 128K 上下文的 KV（fp16，估算） | 建议统一内存 |
| --- | --- | --- | --- | --- | --- |
| Llama 3.2 1B | `llama3.2:1b` | 1.3 GB | 128K | ~4 GiB | 8 GB 起 |
| Llama 3.2 3B | `llama3.2:3b`（默认） | 2.0 GB | 128K | ~14 GiB | 8–16 GB |
| Llama 3.1 8B | `llama3.1:8b` | 4.9 GB | 128K | ~16 GiB | 16 GB 起 |
| Llama 3.2 Vision 11B | `llama3.2-vision:11b` | 7.8 GB | 128K | —（另有视觉编码器） | 16–24 GB |
| Llama 3.2 Vision 90B | `llama3.2-vision:90b` | 55 GB | 128K | — | 64 GB 起 |
| Llama 3.1 / 3.3 70B | `llama3.1:70b` / `llama3.3:70b` | 43 GB | 128K | ~40 GiB | 64 GB 起，建议 96 GB |
| Llama 4 Scout（MoE，17B 激活） | `llama4:scout` | 67 GB | 10M | 视 `num_ctx` 而定 | 96 GB 起 |
| Llama 4 Maverick（MoE，17B 激活） | `llama4:maverick` | 245 GB | 1M | 视 `num_ctx` 而定 | 512 GB 级或服务器 |
| Llama 3.1 405B | `llama3.1:405b` | 243 GB | 128K | — | 512 GB 级，日常不现实 |

> KV cache 的长上下文开销不能忽略：128K 全开时 8B 光 KV 就约 16 GiB，16 GB Mac 还要给权重和系统留空间。日常先把 `num_ctx` 控制在 8K–16K；确实要长上下文时，用 `OLLAMA_KV_CACHE_TYPE=q8_0`（KV 减半）并看 `ollama ps` 的实际 `CONTEXT` 和 `SIZE`。
>
> Llama 3.x 的官方支持语言里**没有中文**；中文任务可以换仓库 Docker 文档里提到的 `qwen2.5:3b` / `qwen2.5:7b`。

### 按内存挑模型

```bash
# 8 GB：先跑 3B，验证链路
ollama run llama3.2:3b

# 16 GB：3B 日常够用，8B 也能跑（上下文别开太大）
ollama run llama3.1:8b

# 24–64 GB：8B + 更大上下文，或 11B 视觉模型（提示词里带图片路径）
ollama run llama3.2-vision:11b "这张图里有什么？/Users/you/Pictures/demo.png"

# 64 GB+：70B 的 4-bit 量化版本
ollama run llama3.3:70b
```

用本目录脚本跑同样的模型（会自动拉模型、调一次 API、打印占用）：

```bash
bash macos/ollama/run.sh --model llama3.2:3b
bash macos/ollama/run.sh --model llama3.1:8b --ctx 8192
```

### 量化怎么选

同一模型在模型库里通常有多个量化 tag，例如 8B 的 `llama3.1:8b-instruct-q4_K_M`、`llama3.1:8b-instruct-q8_0`、`llama3.1:8b-instruct-fp16`：

| 量化 | 大致体积（8B） | 说明 |
| --- | --- | --- |
| q4_K_M（默认 tag 的量化级别） | ~4.9 GB | 体积/质量平衡，日常用这个 |
| q8_0 | ~8.5 GB | 质量更好，用内存换质量 |
| fp16 | ~16 GB | 只在验证精度时用 |
| q2/q3 系列 | 更小 | 质量损失明显，内存极紧时才考虑 |

各模型有哪些 tag，看模型库的 Tags 页（如 <https://ollama.com/library/llama3.1/tags>），或直接 `ollama pull llama3.1:8b-instruct-q8_0` 试。

### 速度由什么决定

- **decode（逐 token 生成）大多卡在内存带宽**：同一模型，M 系列里 Pro/Max/Ultra 的内存带宽更高，tokens/s 也更高；GPU 核数更多主要改善 prefill（首 token）和长上下文。
- 确认没掉到 CPU：`ollama ps` 的 `PROCESSOR` 应为 `100% GPU`；出现 `xx%/yy% CPU/GPU` 说明模型/上下文装不下、部分层被 offload 到 CPU，速度会明显下降。
- 上下文越大 KV 越大，内存压力越大；`ollama ps` 的 `CONTEXT` 列可以确认实际分配的长度。
- 首次加载要读权重、编译 Metal shader，第一句会慢几秒到几十秒；模型空闲 5 分钟会被卸载（`OLLAMA_KEEP_ALIVE`），反复用同一个模型时保持默认即可。
- prompt 很长（贴代码/文档）时主要吃 prefill，首 token 延迟比短 prompt 明显。

### 和 llama.cpp / MLX 的关系

- Ollama 在 Apple Silicon 上同时带 Metal（llama.cpp 系）和 MLX 两个后端，模型格式是 GGUF，具体走哪条由 Ollama 自己选，日常不用关心。
- 想手动调 offload 层数、上下文长度、KV 量化等底层参数，可以改用 llama.cpp：安装与用法见 [`../llama-cpp/README.md`](../llama-cpp/README.md)。
- 想试 Apple 的 MLX 框架，用 `mlx-lm` 加载 Hugging Face 格式的权重；它和 Ollama 的 GGUF 不通用。

## Qwen3-Coder-30B：复杂代码开发的显存要求

前面按体积挑模型；这一节换成一个具体场景：**用 Qwen3-Coder-30B-A3B（Ollama tag `qwen3-coder:30b`）做仓库级 / Agent 式代码开发，显存要多少**，以及 10K token 的 prompt 在 prefill、decode 两个阶段分别吃多少内存。Mac 的统一内存和独显用同一本账。

### 结论：24 GB 独显起，统一内存按约 2/3 折算

| 目标 | 独显显存（Q4_K_M） | Mac 统一内存（Metal 预算约 2/3） | 说明 |
| --- | --- | --- | --- |
| 32K 上下文（复杂开发的最低档） | **24 GB** | 36 GB（32 GB 贴边，需实测） | 24 GB 上接近满载，再接显示器要留余量 |
| 64K 上下文（推荐） | 32 GB | 48 GB | Agent 长会话、多文件任务 |
| 128K / 轻量并发 | 48 GB | 64 GB | KV 用 q8_0；fp8 权重另算 |

显存账就一条公式：

```text
显存 ≈ 权重 + T × 每 token KV + 1–2 GiB（运行时 / 激活）
```

24 GB 独显上：17.3 GiB 权重 + 3 GiB（32K fp16 KV）+ 1–2 GiB ≈ 21–22 GiB，**能跑但只剩 2–3 GiB**。Mac 16 GB 连权重都放不下（macOS 给 Metal 的预算约 10.7 GiB，见「常见问题」里的 `vram-based default context` 日志），24 GB 内存的 Mac 同样放不下，别硬试 30B 级模型。

### 权重与 KV：每 token 96 KiB

[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) 的结构：30.5B 总参 / 3.3B 激活、48 层、4 个 KV 头、head_dim=128、128 专家（每 token 激活 8 个）、原生 262144 上下文。KV 按 GQA 公式：

```text
每 token KV（fp16） = 2 × 48 × 4 × 128 × 2 B = 96 KiB
```

| 上下文 | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | --- | --- | --- | --- | --- | --- |
| KV fp16 | 0.75 GiB | 1.5 GiB | 3 GiB | 6 GiB | 12 GiB | 24 GiB |
| KV q8_0 | 0.38 GiB | 0.75 GiB | 1.5 GiB | 3 GiB | 6 GiB | 12 GiB |

权重：Q4_K_M 约 17.3 GiB（Ollama 的 `qwen3-coder:30b` tag 显示 19 GB，是同一份 Q4 权重）、Q5_K_M 约 20 GiB、Q8_0 约 30 GiB、fp16 约 57 GiB。256K 是原生窗口，但光 fp16 KV 就要 24 GiB，单机实际只适合 32K–128K；更长要量化 KV，并接受 prefill 明显变慢。

### Prefill：10K prompt 的额外占用

Agent 一轮请求（system prompt + 工具定义 + 读几个文件）很容易到 10K token：

| 组成 | 大小 |
| --- | --- |
| KV 写入（10,000 × 96 KiB） | 0.92 GiB |
| 瞬态激活（hidden / QKV / MoE 中间张量） | 约 0.5 GiB（全量 10K） |
| 全量 logits（10,000 × 151936） | fp16 2.83 GiB / fp32 5.66 GiB（通常不需要） |
| eager attention 分数矩阵（不开 FlashAttention） | 约 6 GiB |

实际引擎不会四样全吃满：

- Ollama 底层 llama.cpp 默认**分块 prefill**（`-ub 512` / `-b 2048`），且只算最后一个 token 的 logits，所以 **10K prompt 的瞬态峰值约 0.8–1.5 GiB，不随 prompt 总长线性增长**；随 prompt 增长的是时间（prefill 是算力密集型）。
- Ollama 按 `num_ctx` **预分配** KV：设 `OLLAMA_CONTEXT_LENGTH=32768` 启动就占 3 GiB（fp16），和这次 prompt 长短无关；`ollama ps` 的 `SIZE` 是权重 + KV + 运行时。
- 用 HuggingFace Transformers 直接前向时要传 `logits_to_keep=1`，否则 10K prompt 多出 3–6 GiB logits；attention 要走 FlashAttention / SDPA，否则 24 GB 上容易 OOM。
- vLLM 同理：KV 按 `--gpu-memory-utilization` 预分配，24 GB 上要 `--max-model-len 32768` + `--kv-cache-dtype fp8` 才留得下 32K 的池子。

### Decode：每步只涨 96 KiB

| 每步组成 | 大小 |
| --- | --- |
| 读权重（3.3B 激活参数，Q4） | 约 1.7 GiB（读取量，不是新增占用） |
| 读 KV（10K prompt 后） | 0.92 GiB |
| KV 增量 | **+96 KiB/token**（q8_0 为 48 KiB） |
| 瞬态激活 + logits（1 token） | 0.2–0.5 GiB，基本固定 |

prompt 10K + 生成 2K = 12K token，KV ≈ 1.13 GiB。decode 没有 prefill 那样的峰值，显存只随总长度线性涨；卡的是内存带宽——每步要读约 1.7 GiB 激活权重 + 全部 KV，24 GB 档位的卡上 Q4 大约 80–150 tok/s（4090 级别），上下文越长越慢。这就是 [`example/KV_CACHE_READING.md`](../../example/KV_CACHE_READING.md) 说的「decode 越来越像读显存」。

### Ollama 上怎么配

```bash
# 24 GB：Q4 + 32K，KV 量化，单并发（接近满载）
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=1 ollama serve
ollama run qwen3-coder:30b

# 32 GB：64K + fp16 KV，Agent 长会话更稳
OLLAMA_FLASH_ATTENTION=1 OLLAMA_CONTEXT_LENGTH=65536 ollama serve
```

验证：`ollama ps` 里 `PROCESSOR` 应为 `100% GPU`，24 GB 档位的 `SIZE` 约 21–23 GB，`CONTEXT` 是实际分配长度。出现 `xx%/yy% CPU/GPU` 说明放不下：先降 `num_ctx`，再开 `OLLAMA_KV_CACHE_TYPE=q8_0`，最后才考虑换更小的量化。

## 对接 OpenCode（编程 Agent）

[OpenCode](https://opencode.ai) 是终端里的开源编程 Agent。它对 Ollama 有**内置 provider**：服务在 `http://127.0.0.1:11434` 时，OpenCode 会自动探测 `/api/tags`、`/api/show`，把可用的对话模型加进 `/models`，不需要账号，也不需要写配置。

```bash
# 1) 启动 Ollama 并拉一个模型（工具调用/上下文尽量好一点的）
ollama pull qwen2.5-coder:7b

# 2) 安装 OpenCode
curl -fsSL https://opencode.ai/install | bash

# 3) 在项目里打开，/models 选 Ollama 的模型
cd ~/code/your-project
opencode
```

也可以让 Ollama 直接配置并拉起 OpenCode：

```bash
ollama launch opencode
# 指定模型
ollama launch opencode --model qwen2.5-coder:7b
```

远程/代理的 Ollama（或者不想用自动探测）在 OpenCode 的 `opencode.jsonc` 里指定端点：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "ollama": {
      "settings": {
        "baseURL": "http://127.0.0.1:11434/v1"
      }
    }
  }
}
```

- `baseURL` 要带 `/v1`；OpenCode 会从它推导原生 `/api/tags`、`/api/show` 的发现地址。
- 远程服务开了认证时，在 `settings.apiKey` 里给 token。
- Ollama 的 OpenCode 接入文档建议模型上下文 **64K 以上**；但 Ollama 会把 `num_ctx` 限制在模型训练长度内（超过时日志会警告 `requested context size too large for model` 并压回）。`qwen2.5-coder:7b` 是 32K，所以设 `OLLAMA_CONTEXT_LENGTH=65536` 实际仍是 32K。
- 16 GB 上想用 Ollama 跑 64K，得换原生 64K+ 的模型（如 `llama3.1:8b` 的 128K，配合 `OLLAMA_KV_CACHE_TYPE=q8_0`），但代码能力不如 Qwen Coder；想 64K + Qwen Coder 只能用 llama.cpp 的 YaRN 外推，见 [`../llama-cpp/README.md`](../llama-cpp/README.md#64k-上下文)。现实一点的组合仍是本地 32K + OpenCode 自动压缩，长任务用云端模型。
- 本地模型写代码的选型和 OpenCode 的详细配置见 [`../llama-cpp/README.md`](../llama-cpp/README.md#对接-opencode本地代码-agent)。

## 对接 Cline（VS Code 编程 Agent）

[Cline](https://docs.cline.bot) 内置 Ollama 提供方，是本地代码 Agent 里配置成本最低的选项之一。

### 1. 启动 Ollama 并准备模型

```bash
ollama pull qwen2.5-coder:7b

# 关键：Ollama 默认上下文只有 4K，Cline 需要至少 32K
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

如果用的是菜单栏 App，在 App 的 Settings 里把 context length 调到 `32768`，然后重启 App。

模型选择（16 GB / 10.7 GiB Metal 预算）：

- `qwen3.5:4b`（4.0 GB）：快，32K/64K 都放得下，适合日常 Agent。
- `qwen3.5:9b`（6.6–7.6 GB，同家族 tools/thinking/256K）：质量更好，32K 仍能 100% GPU（权重 + 32K KV 约 1 GiB + 开销），64K 接近上限、需实测。
- `qwen2.5-coder:7b`：代码补全好，但 Agent 工具调用不稳定（见下），不推荐给 Cline。

### 2. Cline 里的设置

VS Code 安装 Cline 后，打开 Cline 设置 → **API Configuration**：

| 设置项 | 值 |
| --- | --- |
| API Provider | `Ollama` |
| Base URL | `http://localhost:11434` |
| Model | `qwen2.5-coder:7b`（下拉选择；列表为空先 `ollama pull`） |
| Context Window | `32768` |
| API Key | 留空（本地 Ollama 不需要） |

然后到 Cline 设置 → **Features**，打开 **Use Compact Prompt**（Cline 官方对本地推理的推荐设置，能明显省 token）。任务变长时新开一个 task，别让上下文无限增长。

### 3. 几个容易踩的坑

- **上下文两边都要设**：Ollama 侧 `OLLAMA_CONTEXT_LENGTH`（或 App 滑块）和 Cline 的 `Context Window` 都设 32768。只设 Cline 的话，Ollama 实际仍按 4K 截断，表现为「Cline 突然忘事/输出被截断」。
- **Qwen2.5-Coder 原生就是 32K**：在这条 Ollama 路线上不要写 64K，Ollama 会把 `num_ctx` 压回 32K（日志警告 `requested context size too large for model`）。要 64K 得换原生 64K+ 模型（如 `llama3.1:8b`）或改用 llama.cpp 的 YaRN。
- **工具调用**：Cline 的改文件/执行命令依赖 tool calling。`qwen2.5-coder:7b` 以代码见长；如果出现「只聊天不改文件 / 不调用工具」，换 `qwen3:8b`（Ollama Cline 文档的示例模型）或云端模型，也可以试下面的 OpenAI Compatible 入口。
- **内存**：16 GB 上 7B Q4 + 32K 上下文约 6–8 GB，再开浏览器/IDE 会紧；不要同时再跑一个 Ollama 模型。
- **远程 Ollama**：Base URL 换成 `http://<Mac-IP>:11434`，服务端用 `OLLAMA_HOST=0.0.0.0:11434` 启动；只在内网用，别暴露公网。

### 4. OpenAI Compatible 兜底

如果 Cline 的 Ollama provider 连不上或模型列表为空，可以走 OpenAI 兼容入口：

| 设置项 | 值 |
| --- | --- |
| API Provider | `OpenAI Compatible` |
| Base URL | `http://localhost:11434/v1` |
| API Key | `ollama`（Ollama 不校验） |
| Model ID | `qwen2.5-coder:7b` |
| Context Window | `32768` |

### 5. Cline Desktop

如果用的是 Cline Desktop，它的 **Local Models** 页面可以直接发现 Ollama、下载模型并自动填好上面的配置，比手动填更省事。

### 常见问题：OpenCode Desktop 2.x 与 CLI 1.x 共库冲突

如果你已经装了 **OpenCode Desktop 2.x**（`/Applications/OpenCode.app`），再用 `ollama launch opencode` 时，Ollama 可能会自动装一个 **v1 稳定版 CLI**（`~/.opencode/bin/opencode`，`opencode --version` 显示 `1.x`）。两者共用 `~/.local/share/opencode/opencode.db`，但数据库 schema 不同：

```text
Error: could not read OpenCode configuration: exit status 1
Run `opencode debug config` for details.
```

手动运行 `opencode debug config` 会看到：

```text
Database is not empty and has no session table
```

原因是这个 DB 由 Desktop 2.x 创建（表是 `session_v2` / `session_message`），而 v1 CLI 期待旧的 `session` 表，于是直接退出 1。

修法（推荐第一种）：

```bash
# 1) 用 Desktop 自带的 v2 CLI 替换 v1 CLI（先备份）
mv ~/.opencode/bin/opencode ~/.opencode/bin/opencode-v1.bak
ln -s /Applications/OpenCode.app/Contents/Resources/opencode-cli ~/.opencode/bin/opencode

# 2) 确认版本与配置读取
opencode --version          # 应显示 v2.x
opencode debug config       # 应正常输出，而不是 Database ... session table

# 3) 再启动
ollama launch opencode --model qwen3.5:4b
```

如果更想继续用 v1 CLI，也可以把 v2 的 DB 挪走让它重建（**会丢 Desktop 里的会话历史，先备份**）：

```bash
mkdir -p ~/opencode-db-backup
mv ~/.local/share/opencode/opencode.db* ~/opencode-db-backup/
```

另外，[`../llama-cpp/opencode.jsonc`](../llama-cpp/opencode.jsonc) 用的是 **v2 的 `providers` / `package` / `settings` 格式**；v1 CLI 不认这套，别混用。

## 上下文与 KV cache：和本仓库的关系

macOS 原生版和 Docker 版用的是同一套 Ollama，参数含义完全相同，只是内存来源从“容器能看到的宿主内存”变成了 Apple 统一内存。

- 默认上下文**不是**固定 4096：不设 `OLLAMA_CONTEXT_LENGTH` 时，Ollama 0.40 会按显存档位选 **4k（<24 GiB）/ 32k（24–48 GiB）/ 256k（≥48 GiB）**（官方 FAQ 仍写 4096，以 `ollama serve` 日志里的 `vram-based default context`、以及 `ollama ps` 的 `CONTEXT` 列为准）。可以用 `num_ctx` 在单次请求里覆盖，或用 Modelfile 的 `PARAMETER num_ctx` 固定下来；App 的 Settings 里也有 context length 滑块。
- KV cache 大小的算法和 [`docker/ollama/README.md`](../../docker/ollama/README.md) 里一致：

  ```text
  2（K 和 V）× n_layers × n_kv_heads × head_dim × 每个元素字节数
  ```

  Llama 3 8B（32 层、8 个 KV 头、head_dim=128、fp16）大约是 **128 KiB / token**，8192 token ≈ 1 GiB。上下文翻倍，KV 就翻倍；在 Mac 上这部分和模型权重一起占统一内存。

- `OLLAMA_NUM_PARALLEL=2` 会同时保留两份上下文，KV 占用翻倍；`OLLAMA_KV_CACHE_TYPE=q8_0` 把 KV 减半，`q4_0` 减到约四分之一（质量损失随 GQA 强度变化）。
- 用 API 显式设置上下文长度：

  ```bash
  MODEL="${MODEL:-llama3.2:3b}"
  curl -s http://127.0.0.1:11434/api/chat \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",
         \"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],
         \"stream\":false,
         \"options\":{\"num_ctx\":8192,\"num_predict\":256}}"
  ```

- `ollama ps` 的列含义：`SIZE` 是加载后的总占用（权重 + 上下文 + 运行时开销）；`PROCESSOR` 显示 `100% GPU` / `100% CPU` / `xx%/yy% CPU/GPU`；`CONTEXT` 是实际分配的上下文长度；`RUNNER` 是实际用的推理后端（Apple Silicon 上可能是 `mlx` 或 `llamacpp`，两者都走 Apple GPU，具体由 Ollama 按模型选择）。这是确认 Metal 有没有生效、默认档位是否如预期、以及 `num_ctx` 调大后内存涨了多少的最直接方式。

对比 [`../../example/kv_cache.py`](../../example/kv_cache.py)：那边是单请求的 K/V 追加与复用；Ollama 把同样的 K/V 放在 Metal GPU 的统一内存里，并处理多请求、采样和上下文管理。跨请求的前缀共享见 [`../../example/prefix_cache_demo.py`](../../example/prefix_cache_demo.py)。

## 和 Docker 版的差异

| | macOS 原生（本文） | Docker（[`docker/ollama/`](../../docker/ollama/README.md)） |
| --- | --- | --- |
| GPU | Apple Silicon 可用 Metal GPU | 用不到 Apple GPU，只能 CPU |
| 内存 | 统一内存，权重 + KV 一起占内存 | Docker Desktop 虚拟机的内存，同样只能 CPU |
| 模型位置 | `~/.ollama/models` | volume `ollama`（Docker Desktop 虚拟机内），删容器不丢模型 |
| 启动方式 | 菜单栏 App / `ollama serve` / `brew services` | `docker run` / 本仓库 `run.sh` |
| 日志 | `~/.ollama/logs/*.log` | `docker logs ollama` |
| 升级 | App 自动更新，或 `brew upgrade ollama` | `docker pull ollama/ollama` |
| 适合场景 | 日常在 Mac 上跑、想用 GPU | 练 Docker、复现 Linux 服务器环境 |

> 两种方式**不要同时跑**：都监听 `11434`，而且 Docker 版会把模型再下载一份进 volume，白占磁盘。想切换时先把一边停掉。

## 怎么自己测模型

不要只看榜单，用你自己机器 + 自己任务测。仓库里附了一个小脚本 [`eval_models.py`](eval_models.py)，做三件事：

1. 6 道带隐藏断言的 Python 题，模型只看到题目，生成代码后在子进程里跑断言；
2. 检查 `write_file` 工具调用是否返回 `tool_calls`；
3. 记录每题耗时。

```bash
# 一次测一个模型（避免模型切换/卸载的额外等待）
python3 macos/ollama/eval_models.py --models qwen3.5:4b --ctx 32768
python3 macos/ollama/eval_models.py --models qwen3.5:9b --ctx 32768

# 只跑前 3 题
python3 macos/ollama/eval_models.py --models qwen3.5:4b-q8_0 --tasks 3
```

输出示例：

```text
=== qwen3.5:4b（num_ctx=32768, 3 题） ===
  [PASS] is_palindrome      6.5s
  [PASS] flatten            2.8s
  [FAIL] top_k_frequent    12.6s AssertionError
  pass@1 = 2/3  avg=7.3s
  tool_calls = OK
```

注意：6 题只能看个大概，真正评估建议：

- **20 题以上**，来自你真实项目（函数题 + 带断言的 bug 修复）；
- 固定条件：同一 `num_ctx`、`temperature=0`、关闭 thinking、同一 prompt；
- 记录 **pass@1、平均耗时、工具调用成功率、峰值内存/是否 swap**；
- Agent 场景再加 5–10 个端到端任务：在临时 git 仓库里放一个失败测试，让 Cline/OpenCode 修，跑测试并统计“一次通过 / 需要人工改几次”；
- 一次只加载一个模型；如果 Ollama 中途重启或 `ollama ps` 卡在 `Stopping...`，先清内存/重启 App 再测。

## token 输出慢怎么定位

先把“慢”拆成三段，用 Ollama API 自带的计时字段量，而不是靠体感：

```bash
python3 -c '
import json, urllib.request, time
prompt = "用一句话解释 KV cache。" * 100
body = json.dumps({"model": "qwen3.5:4b", "prompt": prompt, "stream": False,
                   "options": {"num_ctx": 32768, "num_predict": 128, "temperature": 0}}).encode()
t0 = time.time()
with urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:11434/api/generate", data=body,
        headers={"Content-Type": "application/json"}), timeout=600) as r:
    d = json.load(r)
rate = lambda ns, c: c / (ns / 1e9) if ns else 0
print("wall=%.1fs load=%.1fs" % (time.time()-t0, d.get("load_duration",0)/1e9))
print("prefill=%.1f tok/s (%s tok)" % (rate(d.get("prompt_eval_duration",0), d.get("prompt_eval_count",0)), d.get("prompt_eval_count")))
print("decode =%.1f tok/s (%s tok)" % (rate(d.get("eval_duration",0), d.get("eval_count",0)), d.get("eval_count")))
'
```

读法：

- `load` 很大（几十秒到几分钟）→ 不是生成慢，是**模型加载/切换/卸载在等**；先看 `ollama ps` 里有没有模型卡在 `Stopping...`。
- `prefill` 慢 → 提示词/上下文太长，或 `num_ctx` 太大；Agent 每轮都会重发很长的 system prompt + 文件内容。
- `decode` 慢 → 先看 `ollama ps` 的 `PROCESSOR`：不是 `100% GPU` 就是被 offload 了（内存不够 / 上下文太大 / 模型太大）。

然后按顺序查：

1. **`ollama ps`**：`PROCESSOR`、`CONTEXT`、`SIZE`、`RUNNER`。出现 `xx%/yy% CPU/GPU` 或 `SIZE` 接近 10.7 GiB，就是装不下；`RUNNER=mlx` 和 `llamacpp` 不能直接横向比。
2. **内存 / swap**：

   ```bash
   sysctl vm.swapusage
   memory_pressure -Q | tail -2
   ```

   16 GB 的 Mac 上，如果 swap 已用好几 GB、空闲内存很低，模型切换/加载会非常慢，decode 也会被拖累。关掉不用的 Electron 应用、浏览器标签、其他 LLM 服务，只保留一个模型。
3. **模型切换**：`OLLAMA_MAX_LOADED_MODELS=1` 时切模型要先卸载旧的；MLX runner 偶发卡在 `Stopping...`（`ollama stop` 后 `ollama ps` 一直显示 Stopping…）。这时重启 Ollama App 清状态，并在一次 Agent 会话里只用一个模型。
4. **日志**：

   ```bash
   grep -E "llama-server started|offloaded|n_ctx|Flash Attention|loaded runners" ~/.ollama/logs/server.log | tail -30
   ```

   重点看加载耗时、是否 offload、实际 `n_ctx`、FlashAttention 是否启用。
5. **热 / 电源**：MacBook Air 无风扇，长时间满负载会降频：

   ```bash
   pmset -g therm            # 有 thermal warning 就是过热降频
   ```

   接电源、改善散热、缩短单次任务都能缓解。
6. **Agent 侧**：Cline / OpenCode 的 prompt 会随会话增长，prefill 越来越慢是正常的；开 `Use Compact Prompt`、及时新开 task、少塞大文件。测量时看 `prompt_eval_count`，不要拿第一轮和第十轮直接比。

本机实测基线（M2 / 16 GB / 32K 上下文）：

```text
qwen3.5:4b：prefill ≈ 252 tok/s，decode ≈ 21 tok/s（100% GPU，mlx）
qwen2.5-coder:7b 在 9b 仍驻留时切换过去：load ≈ 141s，prefill ≈ 110 tok/s，decode ≈ 12 tok/s
```

第二行说明“慢”很多时候是**模型切换 / 内存压力**，不是模型本身生成慢。

### 看懂 MLX runner 的日志

`RUNNER=mlx` 时 Ollama 用的是 `mlxrunner`，日志长这样：

```text
Prompt processing progress processed=10561 total=10562
POST /v1/chat/completions ... took=1m40s
speculative decode stats iterations=55 drafted=19 accepted=13 acceptance=0.68
prefix_cache: "cache hit" total=11012 matched=10630 cached=10630 left=382
pipeline: memory peak="9.35 GiB" held="7.62 GiB"
```

| 日志 | 含义 | 说明 |
| --- | --- | --- |
| `Prompt processing progress processed/total` | 正在 prefill（读 prompt） | 10.5K token 的 prompt 在 M2 上要几十秒到一两分钟，是首轮慢的主因 |
| 请求 `took=1m40s` | 整个请求耗时 | 要减去 prefill 才能得到 decode 时间 |
| `prefix_cache "cache hit" matched=10630/11012` | 前缀缓存命中 | 只重算 `left=382` 个 token，所以第二个请求从 1m40s 降到几秒；前缀（system prompt + tools）越稳定命中越高 |
| `speculative decode stats acceptance=0.68 avg_accepted=0.24` | 投机解码接受率 | 接受率低时收益有限；由模型自带的 draft/MTP 决定，普通请求无法用环境变量关掉 |
| `memory peak="9.35 GiB"` | 峰值内存 | 16 GB 机器的 Metal 预算约 10.7 GiB；9b 在 32K 已经接近上限，再涨上下文就会 swap/offload |

对应到你的日志：第一次 1m40s 主要是 **10562 token 的 prefill**；第二次因为 prefix cache 命中 10630 token，只处理 382 token，所以只要 5.9s。这不是“时快时慢”，而是 **prefix cache 生效与否**。

降低 prefill 成本的办法：用更小的模型（`qwen3.5:4b`）、压缩 system prompt/工具定义、少塞文件、保持前缀稳定让 cache 命中；如果 Agent 每轮都改前缀，cache 就失效，每轮都要重算 1 万 token。

### 一组 API 数字怎么算

```text
load_duration        = 5,621,212,041 ns  ≈ 5.62 s
prompt_eval_count    = 11
prompt_eval_duration =   600,988,583 ns  ≈ 0.60 s  → prefill ≈ 18 tok/s（token 太少，不代表真实 prefill 能力）
eval_count           = 124
eval_duration        = 8,815,842,875 ns  ≈ 8.82 s  → decode  ≈ 14.1 tok/s
```

- 总耗时 ≈ `load + prompt_eval + eval`，这组大约是 **15 秒**。
- prompt 只有 11 token，所以 prefill 可以忽略；慢在 **eval（生成 124 token）**。
- `load 5.6s` 是冷加载；如果每次请求都重新加载，会额外多 5 秒，记得让模型保持加载（`OLLAMA_KEEP_ALIVE`）。
- 9B 模型在 M2 上 decode 14 tok/s 属于正常范围（4B 约 21 tok/s）；如果 124 token 里有大量**思考 token**，体感会更慢。

### 先关掉 thinking 再比速度

同一个问题（“用一句话解释 KV cache”）在 `qwen3.5:9b` 上实测：

```text
think=true ：eval_count=128，9.39s，全部被思考占用，没有输出答案（撞到 num_predict 上限）
think=false：eval_count=37，2.80s，直接给出完整答案
```

所以 Agent 场景如果不需要长思考，先把 thinking 关掉：

- Ollama 原生 API：`"think": false`
- OpenAI 兼容接口：`"reasoning_effort": "none"`（实测能关掉 qwen3.5 的思考）
- OpenCode：在模型的 `settings.reasoningEffort` 或 reasoning variant 里关掉
- Cline：如果 UI 没有 thinking 开关，就只能换 `qwen3.5:4b` 或接受思考开销

### 让模型常驻（避免每次请求都等加载）

Ollama 默认空闲 **5 分钟**就卸载模型。常驻有三种设置方式，优先级：请求 `keep_alive` > 服务级 `OLLAMA_KEEP_ALIVE` > 默认 5m。

1. **服务级（影响所有模型，推荐给 Cline/OpenCode）**

   ```bash
   # CLI 前台启动
   OLLAMA_KEEP_ALIVE=-1 ollama serve    # -1 = 永久；也可以 30m / 24h
   ```

   App 模式（菜单栏 Ollama）：

   ```bash
   launchctl setenv OLLAMA_KEEP_ALIVE 24h
   osascript -e 'quit app "Ollama"'
   open -a Ollama
   ```

   `launchctl setenv` 只在当前登录会话有效；要每次开机生效，写一个 LaunchAgent 在登录时执行它，或者改用 CLI/本仓库脚本启动。

   用本仓库脚本时（先退出 App，让脚本自己起 serve）：

   ```bash
   OLLAMA_KEEP_ALIVE=30m ./macos/ollama/run.sh --no-pull --no-demo
   ```

2. **单次请求 `keep_alive`（覆盖服务级，适合 App 用户）**

   不需要改环境变量，直接给已加载的模型续期：

   ```bash
   curl -s http://127.0.0.1:11434/api/generate      -d '{"model":"qwen3.5:9b","keep_alive":-1}'
   ollama ps    # UNTIL 列应显示 Forever
   ```

   - `-1`：永久驻留
   - `"30m"` / `3600`：指定时长
   - `0`：请求结束后立刻卸载
   - 也可以直接预热：`ollama run qwen3.5:9b ""`（配合 `--keepalive 30m`）

3. **不要频繁切模型**：`OLLAMA_MAX_LOADED_MODELS=1` 时切模型会先卸载旧的，又回到 load 等待。一次 Agent 会话只用一个模型。

验证：

```bash
ollama ps
# UNTIL 列：Forever / 30 minutes from now ...
```

16 GB 机器上的取舍：

- `qwen3.5:9b` 常驻 `SIZE` 约 8.2 GB，加上 IDE/浏览器已经很紧；建议 `30m`~`1h`，或只在使用期间 `-1`。
- `qwen3.5:4b` 只占约 4 GB，常驻 `-1` 也比较安全。
- 退出 App、重启服务、重启机器都会清掉；App 重启后重新发一次 `keep_alive:-1` 的预加载请求即可。

### 怎么判断有没有触发 offload

- **llama.cpp runner**（`RUNNER=llamacpp`）：日志里 `load_tensors: offloaded 29/29 layers to GPU` 表示**全部层都在 GPU**，不是 CPU offload；只有 `ollama ps` 的 `PROCESSOR` 出现 `xx%/yy% CPU/GPU`，或日志里 GPU 层数小于总层数，才是真的把层放到了 CPU。
- **MLX runner**（`RUNNER=mlx`）：没有“按层 offload”这个概念，模型整体在统一内存里跑；`PROCESSOR` 通常一直显示 `100% GPU`，不能只看这一列判断是否吃紧。要看 `pipeline: memory peak/held` 和系统内存：

  ```bash
  ollama ps                 # SIZE / PROCESSOR / CONTEXT
  sysctl vm.swapusage       # swap 用了多少
  memory_pressure -Q | tail -2
  ```

  例如 `memory peak="9.85 GiB"`、`held="8.12 GiB"`，而 16 GB 机器的 Metal 预算约 10.7 GiB：**没有 offload，但已经到 92% 边缘**；如果 swap 同时用了 9 GB，系统在换页，生成会抖/变慢。

### prefill 到底占多少内存、为什么慢

先分清两件事：**KV cache 按 `num_ctx` 预留，不按这次 prompt 的长度预留**；prompt 再长，也只在预留的 KV 里占一段。

以 `qwen3.5:9b`、`num_ctx=32768` 为例：

- Qwen3.5 是混合注意力（32 层中 8 层 full attention），KV ≈ **32 KiB/token**。
- `num_ctx=32768` → 预留 KV ≈ **1 GiB**（启动/加载时就占）。
- 这次 prompt = 10,797 token → 实际用到约 **337 MiB** KV，其余仍是空闲容量。
- prefill 的临时激活按 chunk 算（MLX runner 的 chunk 是 2048 token），峰值大约 **1–1.5 GiB**（包含计算缓冲/前缀缓存快照）。
- 所以日志里 `memory peak 9.85 GiB` ≈ 权重 7.6 GiB + KV 1 GiB + prefill 临时缓冲 1 GiB 左右；`ollama ps SIZE 8.9 GB` 是常驻占用，不含瞬时峰值。

不同模型的 KV 估算：

| 模型 | KV/token | 32K 上下文 KV | 10.8K prompt 实际占用 |
| --- | --- | --- | --- |
| `qwen3.5:4b / 9b` | ~32 KiB | ~1 GiB | ~337 MiB |
| `qwen2.5-coder:7b` | 56 KiB | ~1.75 GiB | ~590 MiB |
| `llama3.2:3b` | 112 KiB | ~3.5 GiB | ~1.2 GiB |

把 `num_ctx` 从 32768 降到 16384，KV 预留直接减半；升到 65536 则翻倍，16 GB 机器上 9b 会超预算。

**为什么慢：** prefill 是计算密集型，不是内存带宽型。10.8K token、9B 参数的 prefill，FLOPs 有两条腿：

- **GEMM**（QKV/O 投影 + MLP + 24 层线性注意力）：粗算 `≈ 2 × 9B × 10.8K ≈ 194 TFLOPs`，对 token 数线性增长；
- **8 层 full attention**：`≈ 4 × 8 × d_qk × n²`；按 16 个 Q 头、head_dim 256（`d_qk = 4096`）估算，10.8K token 约 **15 TFLOPs**，只占 ~8%，要到约 14 万 token 才和 GEMM 持平，256K 时反超。

M2 的 GPU 算力大约只有数据中心卡的几十分之一：194 TFLOPs / 100 秒 ≈ **2 TFLOPS 有效算力**，和实测吻合。所以这个长度下**慢在 GEMM 把算力打满，而不是 attention 的 O(n²)**。按 100 秒算就是 **~108 tok/s 的 prefill**，对 M2 + 9B 是正常量级；和 decode 的 14 tok/s 是两回事。

再加上：
- 前缀缓存没命中时，每轮都要重算这 1 万 token；
- 9.85 GiB 峰值贴近 10.7 GiB 预算，swap 已用 9 GB，系统换页进一步拖慢；
- speculative decode 只影响生成阶段，救不了 prefill。

所以优化 prefill 的方向只有：**换小模型（4b）、缩短 prompt、保持前缀稳定让 prefix cache 命中、别开 64K**。

### 每层在做什么：一层的数据流

前面把 prefill 的 FLOPs 拆成「GEMM + 8 层 full attention」两条腿。这一节把 `qwen3.5:9b` 的一层打开，看这两条腿分别落在哪。参数取自模型自身的元数据（`ollama show` 和 GGUF 头）：32 层、hidden 4096、FFN 12288、16 个 Q 头 / 4 个 KV 头、head_dim 256；每 4 层插 1 层 full attention（第 4、8、…、32 层，共 8 层），其余 24 层是线性注意力（Gated DeltaNet 一类的递推结构）。

一层就是两个残差子层，prefill 和 decode 跑的是同一张图，只是张量形状不同：

```text
x ──► RMSNorm ──► attention 子层 ──► (+x) ──► RMSNorm ──► MLP 子层 ──► (+x)
```

| 子层 | 里面有什么 | prefill（n ≈ 10.8K） | decode（n = 1） |
| --- | --- | --- | --- |
| 线性注意力（24 层） | `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` → 短卷积（kernel 4）→ delta 规则递推 → 门控归一化 → `out_proj` | `[n, 4096]` 上的几组 GEMM（qkv / z / a / b 投影与 `out_proj`），递推部分逐 token 顺序算 | 只更新固定大小的递推状态（`state_size = 128`），不随上下文变长 |
| full attention（8 层） | q/k/v 投影 → q_norm / k_norm → 部分 RoPE（只转 64 维）→ 写 KV cache → GQA（16 Q 头 : 4 KV 头）→ `out_proj` | 投影是 GEMM；`QKᵀ` 是 `[n, n]`，只有这 8 层是 O(n²) | 每步读完整 KV（32 KiB × 上下文） |
| MLP（每层） | SwiGLU：gate / up 两个 `4096 → 12288`，相乘后再 down 回 `4096` | 每 token 约 `2 × 3 × 4096 × 12288 ≈ 0.30 GFLOPs` | GEMV，以读权重为主 |

几个能直接对上前面数字的结论：

- **KV 只来自 8 层 full attention。** 每 token 每层 `2 × 4 × 256 × 2 B = 4 KiB`，8 层正好 **32 KiB/token**，就是前面表格里的数；24 层线性注意力不占 KV cache（只维护固定大小的递推状态），所以 9B 的 KV 只有「32 层全是 full attention」同类模型的 1/4。`qwen3.5:4b` 只是宽度小（hidden 2560 / FFN 9216），层数和注意力配比相同，KV 同为 32 KiB/token。
- **prefill 的算力主要花在 MLP 和投影上。** MLP 一层每 token 约 0.30 GFLOPs，32 层 × 10.8K token ≈ **105 TFLOPs**，是那 194 TFLOPs 里最大的一块；attention 投影加上 8 层 full attention 的 15 TFLOPs 填掉剩下的部分。这就是「慢在 GEMM 把算力打满」的实际来源。
- **decode 走的是另一条路。** 每步输入 `[1, 4096]`，GEMM 退化成 GEMV，算力不再是瓶颈；瓶颈变成每步要读的权重和 KV（上下文 × 32 KiB）。所以 prefill 的优化方向是算力（换小模型、缩 prompt），decode 是带宽（KV 量化、减上下文）。

日志里 `speculative decode stats` 那行是投机解码的统计（草案 token 让主模型一次验证多个），只影响 decode，救不了 prefill。

换成 [Qwen3-Coder-30B](#qwen3-coder-30b复杂代码开发的显存要求) 是另一种组合：48 层**全部**是 full attention（4 个 KV 头、head_dim 128 → 96 KiB/token），MLP 位置换成 MoE——每层一个 router 在 128 个专家里选 8 个。decode 每步只读激活的 3.3B 参数，所以快；prefill 时 10.8K 个 token 会摊到大部分专家，GEMM 仍按 30.5B 总参算，所以贵。同样叫「一层」，稠密模型和 MoE 的账不一样。

## 日常开发要多大上下文

按使用方式分，不是“越大越好”：

| 场景 | 最小上下文 | 说明 |
| --- | --- | --- |
| 行内补全 / 单行代码 | 4K–8K | FIM 补全只需要光标附近 |
| 单文件问答 / 解释报错 | 8K–16K | 一个文件 + 问题 |
| Cline / OpenCode 日常 Agent | **16K 硬底线，32K 实用最小** | system prompt + 工具定义 + 读几个文件，很容易到 8K–12K；Cline 官方建议 ≥32K，Ollama 的 OpenCode 接入建议 64K |
| 多文件重构 / 仓库级任务 | 64K–128K | 需要读多个文件、多轮工具调用 |
| 长时间 Agent 会话 | 64K+ | 上下文会持续增长，压缩策略决定体验 |

参考你之前的真实日志：一个普通 Agent 请求的 prompt 已经 **10,797 token**。所以：

- `num_ctx=16K`：只剩约 5K 给输出和下一轮，很快就会触发压缩，**只能算硬底线**；
- `num_ctx=32K`：约 21K 余量，**日常开发的实用最小值**；
- `num_ctx=64K`：适合多文件任务，但这台 16 GB 上 9B 装不下，要用 4B 或云端。

不同 `num_ctx` 的 KV 内存（只算 KV，不含权重）：

| 模型 | KV/token | 8K | 16K | 32K | 64K |
| --- | --- | --- | --- | --- | --- |
| `qwen3.5:4b/9b` | 32 KiB | 0.25 GiB | 0.5 GiB | 1 GiB | 2 GiB |
| `qwen2.5-coder:7b` | 56 KiB | 0.44 GiB | 0.88 GiB | 1.75 GiB | 原生上限 32K |
| `llama3.2:3b` | 112 KiB | 0.88 GiB | 1.75 GiB | 3.5 GiB | 7 GiB |

加上权重和 prefill 临时缓冲后的结论（Metal 预算约 10.7 GiB）：

- `qwen3.5:4b`：4.0 GB 权重，32K/64K 都放得下，64K 还有余量；
- `qwen3.5:9b`：7.6 GB 权重，32K 已到 9.8 GiB 峰值，64K 会超预算；
- `qwen2.5-coder:7b`：32K 是原生上限，权重 + KV 约 7–8 GB，能跑但 Agent 工具调用不稳。

**建议**：这台机器上日常开发设 `num_ctx=32768`，模型用 `qwen3.5:4b`（速度/内存都宽松）或 `qwen3.5:9b`（质量更好、余量小）；`16K` 只在单文件/短会话时用；`8K` 不要给 Agent。

### 一次 7m45s 请求的拆解

```text
prefix_cache: total=26481 matched=11000 left=15483   # prompt 26.5K，命中 11K，还剩 15.5K 要 prefill
Prompt processing progress: 2048 → 15482，每 2048 块约 23–28s
speculative decode stats: iterations=2287 accepted=1108 acceptance=0.85
POST /v1/chat/completions took=7m45s
```

拆开看：

- **prefill**：从 10:46:53 到 10:49:53，约 **3 分钟** 处理 15,483 token → **约 86 tok/s**（每 2048 块 23–28s）。这部分是 9B 模型 + M2 的正常量级。
- **decode**：10:49:53 到 10:54:38，约 **4 分 45 秒**。按 2287 次迭代 + 1108 个被接受的 draft 粗略估算约 3.4K 输出 token → **约 12 tok/s**。
- 合计 ≈ 3 分钟 + 4 分 45 秒 = **7m45s**，和日志一致。

为什么这么慢：

- Agent 一次塞了 **26.5K prompt**，命中缓存后仍要重算 15.5K；
- 上下文已到 26.5K/32K，decode 每个 token 都要看更长的历史，速度比短上下文更低；
- 输出本身又很长（约 3.4K token），12 tok/s 下就是几分钟；
- 9B 在 16 GB 上已接近 10.7 GiB 预算，swap/换页也会拖慢。

优化方向：把 prompt 压到 10K–16K、保持前缀稳定提高缓存命中、让 Agent 输出小 patch、用 `qwen3.5:4b`；32K 已经接近上限，不要再加。

## 能不能看算子级耗时（类似 Linux perf）

macOS 没有 Linux `perf`，也没有现成的逐算子计时。可行的工具分三层：

### 1. CPU 侧：`sample`（无需 Xcode）

```bash
PID=$(lsof -nP -iTCP:11434 -sTCP:LISTEN -t | head -1)
sample "$PID" 10 -file /tmp/ollama.sample.txt
open /tmp/ollama.sample.txt
```

能看到 Ollama/MLX runner 的 CPU 栈热点（tokenizer、Go 调度、Metal 提交、内存拷贝），但**看不到 GPU kernel 的耗时**。

### 2. GPU 侧：`powermetrics`（需要 sudo）

```bash
sudo powermetrics --samplers gpu_power -i 500 -n 20
```

只能看 GPU 利用率/频率/功耗，不能定位到具体算子。

### 3. 真正到 kernel：Xcode / Metal GPU Trace

需要安装完整 **Xcode**（只装 Command Line Tools 没有 `xctrace`/Instruments）。两条路：

**A. Instruments 的 Metal System Trace（适合 MLX runner / qwen3.5）**

```bash
xcrun xctrace list templates | grep -i metal
PID=$(lsof -nP -iTCP:11434 -sTCP:LISTEN -t | head -1)
xcrun xctrace record --template 'Metal System Trace' \
  --attach "$PID" --time-limit 30s --output /tmp/ollama-metal.trace
open /tmp/ollama-metal.trace
```

抓取时在另一个终端发一个固定 prompt 的请求；然后在 Instruments 里看 command buffer / shader 的耗时分布，找最热的 kernel（matmul、attention、dequant、copy 等）。

**B. ggml Metal 的 GPU trace（适合 llama.cpp runner，如 llama3.2/qwen2.5-coder）**

ggml 自带 capture 支持：

```bash
# 先退出 App，手动启动并打开 Metal capture
METAL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=1 ollama serve
# 然后发请求，日志会出现：capturing graph in /tmp/perf-metal-<pid>.gputrace
```

生成的 `.gputrace` 用 Xcode 打开，可以看到每个 Metal kernel 的 GPU 时间。

注意：
- capture 会显著拖慢执行，只适合单次基准；
- `GGML_METAL_CAPTURE_COMPUTE=N` 表示跳过前 N 个 graph 再抓（先 warmup 再抓更有代表性）；
- MLX runner（qwen3.5）不走 ggml Metal，用方案 A；
- Ollama 不暴露逐算子计时，真要做 kernel 优化，直接用 llama.cpp 源码 + Xcode GPU trace 更顺。

**实际建议**：先确认 `ollama ps` 是 `100% GPU`、上下文/内存没超预算、prefix cache 生效；这些对体验的影响通常比换某个 kernel 大得多。算子级 profiling 更适合改 llama.cpp/MLX 的场景。

## 常见问题

**`Error: listen tcp 127.0.0.1:11434: bind: address already in use`**

说明已经有一个 Ollama 在跑。先查是谁：

```bash
lsof -nP -iTCP:11434 -sTCP:LISTEN
```

然后按来源停掉：菜单栏 App → Quit Ollama；`brew services stop ollama`；本脚本起的 → `bash macos/ollama/run.sh --stop`。

**`ollama list` 报连不上服务**

CLI 只是客户端，需要有服务在跑。`curl -s http://127.0.0.1:11434/api/version` 没响应就先启动服务（App 或 `ollama serve`）。

**App 图标在，但请求超时或报错**

看 `~/.ollama/logs/server.log`；App 层面的启动问题看 `app.log`。必要时 `launchctl setenv OLLAMA_DEBUG 1` 后重启 App 再复现。

**下载模型很慢 / 中途失败**

支持断点续传，重跑 `ollama pull` 即可。需要代理时只设 `HTTPS_PROXY`（官方建议**不要**设 `HTTP_PROXY`，可能干扰客户端连本地服务）；公司自签证书要装进系统钥匙串。

**内存吃紧 / 开始用 swap，机器变卡**

按代价从低到高：换更小的模型（`llama3.2:1b` / `3b`）；把 `num_ctx` 调小（`OLLAMA_CONTEXT_LENGTH` 或请求里的 `num_ctx`）；想更快卸载把 `OLLAMA_KEEP_ALIVE` 调小（如 `1m`，默认是 `5m`），或 `ollama stop <model>` 手动卸载；`OLLAMA_NUM_PARALLEL=1` 别开并发；把 `OLLAMA_KV_CACHE_TYPE` 从 `f16` 改成 `q8_0`。

**模型跑在 CPU 上 / 速度不对**

先看 `ollama ps` 的 `PROCESSOR` 列。Apple Silicon 正常应是 `100% GPU`；Intel Mac 只有 CPU。如果 Apple Silicon 上跑的是 Rosetta 转译的 x86_64 CLI，可能退回通用 CPU 库——用 `uname -m` 和 `file "$(command -v ollama)"` 确认，装 arm64 原生版本。

**`ollama ps` 的 CONTEXT 一直是 4096，怎么调大？**

`CONTEXT` 是**当前 Ollama 服务启动时**分配的默认上下文（16 GB 机器默认 4K），不是单次 `ollama run` 能改的；已经加载的模型也不会因为你在 shell 里 export 就变。改法：

- App 模式（菜单栏 Ollama）：Settings 里把 context length 滑块调到 `32768`，然后 Quit 再重开 App。
- CLI 模式：先退出 App，再带环境变量起服务：

  ```bash
  osascript -e 'quit app "Ollama"'
  OLLAMA_CONTEXT_LENGTH=32768 ollama serve
  ```

- 本仓库脚本：先停掉已有服务，再让脚本自己起 serve（脚本只在“自己启动 serve”时应用 `--ctx`）：

  ```bash
  ./macos/ollama/run.sh --ctx 32768 --no-pull --no-demo
  ```

重启后第一次加载模型，再用 `ollama ps` 确认 `CONTEXT` 变成 `32768`。16 GB 上建议 32K；64K 的前提是模型原生支持且 KV 装得下，别硬设。

**日志里的 `vram-based default context` 是什么意思？**

Ollama 启动时会探测 Metal 可用显存，按分档决定默认上下文：

```text
inference compute ... library=Metal name=MTL0 description="Apple M2" total="10.7 GiB"
vram-based default context total_vram="10.7 GiB" default_num_ctx=4096
```

- 16 GB 统一内存的 Mac，macOS 给 Metal 的预算约 **10.7 GiB**（其余留给系统），落在 `< 23 GiB` 档，所以默认 `4096`。
- 这个默认值只是“起点”，`OLLAMA_CONTEXT_LENGTH` 可以覆盖；**日志里即使还打印 `default_num_ctx=4096` 也不代表没生效**——那行只是 VRAM 分档的兜底值，实际加载优先级是：请求 `num_ctx` > Modelfile/模型参数 > `OLLAMA_CONTEXT_LENGTH` > VRAM 默认。用 `ollama ps` 的 `CONTEXT` 列验证真实值。
- 但**权重 + KV 要装进这 10.7 GiB 才能保持 `100% GPU`**，超了会 offload 到 CPU（`ollama ps` 的 `PROCESSOR` 会显示 CPU/GPU 混合）。
- 本机参考：`qwen2.5-coder:7b`（4.7 GB 权重 + 32K KV 约 1.75 GiB）≈ 7 GB，32K 可以稳稳 100% GPU；`llama3.2:3b` 的 64K KV 约 7 GiB + 2.6 GB 权重，已经接近 10.7 GiB 上限，容易掉 CPU。所以 16 GB 上 **32K 是安全档，64K 要先实测**。

**同时加载了两个模型，内存吃紧？**

`ollama ps` 里出现多行说明多个模型都在内存里（默认最多 3 个，空闲 5 分钟后卸载）。16 GB 上写代码只留一个：

```bash
ollama stop llama3.2:3b          # 停掉不用的
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m ollama serve   # 或启动时限制
```

**下载到一半 Mac 睡眠**

用 `caffeinate` 防止睡眠：`caffeinate -i ollama pull llama3.2:3b`，或 `caffeinate -i bash macos/ollama/run.sh`。

**改了环境变量没生效**

App 模式必须重启 App（菜单栏 → Quit，再 `open -a Ollama`）；`launchctl setenv` 只对当前登录会话有效，重启后要重新设置；`brew services` 模式用 `brew services restart ollama`。

## 清理 / 卸载

```bash
# 先停服务
brew services stop ollama          # 如果用的是 Homebrew 方式
osascript -e 'quit app "Ollama"'   # 如果用的是 App

# 官方文档列出的卸载路径
sudo rm -rf /Applications/Ollama.app
sudo rm -f /usr/local/bin/ollama
rm -rf "$HOME/Library/Application Support/Ollama"
rm -rf "$HOME/Library/Saved Application State/com.electron.ollama.savedState"
rm -rf "$HOME/Library/Caches/com.electron.ollama/"
rm -rf "$HOME/Library/Caches/ollama"
rm -rf "$HOME/Library/WebKit/com.electron.ollama"
rm -rf "$HOME/.ollama"             # 模型也在这里，删之前确认不再需要

# Homebrew 安装的话
brew uninstall ollama
```

> 如果系统设置的「登录项」里还残留 Ollama 条目，手动删掉即可。

> 只想腾模型空间、保留程序：`du -sh ~/.ollama/models` 看占用，`ollama rm <model>` 逐个删。

## 参考

- Ollama macOS 文档：<https://docs.ollama.com/macos>
- 本地 FAQ（环境变量、模型目录、并发、KV cache 量化）：<https://docs.ollama.com/faq>
- 上下文长度（按显存档位的默认值）：<https://docs.ollama.com/context-length>
- CLI 参考：<https://docs.ollama.com/cli>
- API 文档：<https://docs.ollama.com/api/introduction>
- 硬件支持（Metal）：<https://docs.ollama.com/gpu>
- 官方安装脚本：<https://ollama.com/install.sh>
- Docker 版示例：[docker/ollama/README.md](../../docker/ollama/README.md)
