# 在 macOS 上跑 Ollama / Llama（Apple GPU / Metal）

在 Mac 上直接跑 [Ollama](https://ollama.com/)：装官方 App 或 CLI，拉一个 Llama 模型，用 HTTP API 对话。文档既讲 Ollama 本身的安装/启动，也讲 Llama 家族在 Mac 上怎么选模型、内存怎么算。和 [`docker/ollama/`](../../docker/ollama/README.md) 是同一件事的两种跑法——那边是 Docker 容器（在 macOS 上只能用 CPU，Linux + NVIDIA 才用得到 GPU）；这边是 macOS 原生版，Apple Silicon 上会走 **Metal GPU**，模型权重和 KV cache 都放在统一内存里。

目录内容：

```text
macos/ollama/
├── run.sh     # 检查/安装 + 启动 serve + 拉模型 + 调一次 API 的脚本
└── README.md  # 本文件
```

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
