# 在 macOS 上用 llama.cpp 跑 Llama（Metal GPU）

用 [llama.cpp](https://github.com/ggml-org/llama.cpp) 在 Mac 上直接跑 Llama 的 GGUF 量化模型，起一个 OpenAI 兼容服务，并用 `-c`、`-ngl`、`-fa`、`-ctk/-ctv` 这些底层参数观察 KV cache 和 Metal 卸载行为。默认模型（Llama 3.2 3B）只用于演示；**写代码请换 Coder 模型，见下文「用 llama.cpp 写代码」**。llama.cpp 是 Ollama 底层的推理引擎之一，这份示例和 [`../ollama/README.md`](../ollama/README.md) 是互补关系：

| | 这份示例（llama.cpp） | [`macos/ollama/`](../ollama/README.md) |
| --- | --- | --- |
| 定位 | 底层引擎，参数全暴露 | 一体化运行时 + 模型库 + 菜单栏 App |
| 模型管理 | `-hf` 从 Hugging Face 拉 GGUF，或给本地 `.gguf` | `ollama pull`，官方模型库 |
| 参数控制 | `-c`、`-ngl`、`-fa`、`-ctk/-ctv`、`-np`、`-b/-ub` … | `OLLAMA_*` 环境变量 + `num_ctx` |
| 服务端口 | 默认 `9931`（v1.0） | 默认 `11434` |
| 适合 | 调参、对照 KV cache、试新特性、单一二进制 | 日常使用、省心、自动更新 |

目录内容：

```text
macos/llama-cpp/
├── run.sh         # 启动 serve + 等 /health + 调一次 API + 打印上下文/吞吐
├── opencode.jsonc # 对接 OpenCode 的示例配置（custom OpenAI-compatible provider）
└── README.md      # 本文件
```

## 前置条件

| 项 | 要求 |
| --- | --- |
| 系统 | macOS Sonoma（14）或更新（Apple Silicon 推荐） |
| 芯片 | Apple Silicon：Metal GPU + 统一内存；Intel Mac：只能 CPU |
| 内存 | 模型大小 + `-c` 对应的 KV cache + 运行时开销（见下文公式） |
| 磁盘 | 模型放 `~/Library/Caches/llama.cpp`（默认缓存），一个 3B Q4 模型约 2 GB |
| 网络 | 首次 `-hf` 需要访问 Hugging Face；国内可设 `HF_ENDPOINT=https://hf-mirror.com` |

## 安装（三种方式，选一种）

### 1. 官方脚本（Apple Silicon 推荐）

llama.cpp 从 v1.0 起提供了一个多命令的 `llama` 二进制，官方脚本按芯片下载预编译的 Metal 版本：

```bash
curl -LsSf https://llama.app/install.sh | sh
```

装到 `~/.local/bin/llama`（脚本会提示把 `~/.local/bin` 加进 `PATH`）。支持 M1–M5 的 Metal 预编译包；**Intel Mac 没有预编译包**，脚本会报错，改用方式 2 或 3。

```bash
llama --version
llama help            # 常用命令：serve / cli / download
llama help all        # 额外命令：bench / quantize / perplexity / fit-params
```

### 2. Homebrew

```bash
brew install llama.cpp
```

装的是同一个 v1.0 的 `llama` 二进制（通常还会带旧版 `llama-server`、`llama-cli` 等）。注意：**macOS 14（Sonoma）目前没有预编译 bottle**，会从源码编译（依赖 cmake、node、ggml、openssl@4，耗时长）；macOS 15+ 才有现成 bottle。在 Sonoma 上优先用方式 1。

### 3. 源码构建（想跟最新 master 时）

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build            # macOS 上 Metal 默认开启，不需要额外 -DGGML_METAL=ON
cmake --build build --config Release -j
./build/bin/llama --version
```

想关掉 Metal 编译用 `-DGGML_METAL=OFF`；运行时想强制只用 CPU，用 `-ngl 0`。

装完确认：

```bash
command -v llama        # ~/.local/bin/llama 或 /opt/homebrew/bin/llama
llama version
```

## 模型从哪来

llama.cpp 直接吃 GGUF 文件，三种来源：

```bash
# 1) 直接从 Hugging Face 仓库拉（quant 省略时按 Q4_K_M 找）
llama download -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
# 输出本地路径，例如 ~/Library/Caches/llama.cpp/...gguf

# 2) 直接启动时自动下载
llama serve -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

# 3) 本地已有 .gguf
llama serve -m ./Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

- macOS 的默认缓存目录是 `~/Library/Caches/llama.cpp`，可用 `LLAMA_CACHE` 改。
- 国内网络：`HF_ENDPOINT=https://hf-mirror.com llama serve -hf ...`。
- gated 仓库：`-hft <token>` 或环境变量 `HF_TOKEN`。
- Ollama 的模型其实也是 GGUF blob，但它按 manifest 存放，不建议直接把 `~/.ollama/models/blobs/*` 指给 llama.cpp；从 HF 拉一份更稳。

## 用 llama.cpp 写代码：模型怎么选

先说结论：**本文默认的 Llama 3.2 3B 是拿来演示 KV cache 的，不适合认真写代码**。它能回答简单语法问题，但代码能力、指令跟随和长代码上下文都明显弱于专门的 Coder 模型，容易编造 API 或依赖。真要用它写代码，把 `-hf` 换成下面的 Coder 模型（都是 GGUF，`llama serve -hf` 直接拉）：

| 模型 | Hugging Face 仓库（示例） | Q4 体积 | 16 GB M2 上 | 适合 |
| --- | --- | --- | --- | --- |
| Qwen2.5-Coder-7B-Instruct | `unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` | 4.36 GB | ✅ 推荐，`-c 8192`/`-c 16384` 都稳 | 日常补全、小项目、代码问答 |
| Qwen2.5-Coder-14B-Instruct | `unsloth/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M` | 8.37 GB | ⚠️ 能跑，建议 `-c 2048` 且关掉大内存应用 | 质量更好、复杂重构 |
| Qwen3-Coder-30B-A3B-Instruct（MoE） | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` | 17.3 GB | ❌ 16 GB 放不下，建议 24 GB+ | 代码 Agent、长任务 |
| gpt-oss-20b（MoE） | `ggml-org/gpt-oss-20b-GGUF:MXFP4` | 11.3 GB | ⚠️ 比较勉强，建议 `-c 2048`–`4096` 并关大应用 | 推理 + 代码，MXFP4 量化 |
| GLM-4.7-Flash | `ggml-org/GLM-4.7-Flash-GGUF:Q4_K` | 17.0 GB | ❌ 建议 24 GB+ | 代码/Agent |

> 体积、量化来自 Hugging Face 仓库的实际文件；16 GB 的判断按“权重 + KV + 运行时 + 系统”一起算，实际会受 macOS 其他进程影响。

KV cache 也要算进内存（比 Llama 3.2 3B 更要注意）：

- Qwen2.5-Coder-7B：28 层、4 个 KV 头、head_dim=128、fp16 → **56 KiB/token**；`-c 8192` ≈ 448 MiB，`-c 16384` ≈ 896 MiB。
- Qwen2.5-Coder-14B：48 层、8 个 KV 头 → **192 KiB/token**；`-c 2048` ≈ 384 MiB，`-c 4096` ≈ 768 MiB。
- 30B 级 MoE 的权重装不下时，Ollama 式的“自动缩上下文”在 llama.cpp 里没有，别硬开大 `-c`；宁可换小模型。

### 启动与接入编辑器

```bash
# 16 GB Mac 的日常选择：7B + 8K 上下文
bash macos/llama-cpp/run.sh \
  --hf unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M \
  --ctx 8192 --alias qwen2.5-coder-7b

# 质量优先：14B，但上下文压到 2K，先关掉浏览器/IDE 以外的大应用
bash macos/llama-cpp/run.sh \
  --hf unsloth/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M \
  --ctx 2048 --alias qwen2.5-coder-14b

# 直接 llama serve：代码任务用低温度更稳
llama serve -hf unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M \
  -c 8192 -ngl auto -fa auto -a qwen2.5-coder-7b --temp 0.2 --top-p 0.9
```

服务是 OpenAI 兼容的，编辑器/工具接 `http://127.0.0.1:9931/v1` 即可（`model` 填上面的 `--alias`，`api_key` 随便填；从别的机器访问再加 `--host 0.0.0.0` 和 `--api-key`）。常见接法：

- Continue / Cline / Roo Code 等 VS Code 插件：OpenAI Compatible provider，base URL `http://127.0.0.1:9931/v1`。
- Aider / Open WebUI / 自己写的脚本：同样走 `/v1/chat/completions`。
- 代码补全（FIM）：llama.cpp 有 `/infill` 端点，Qwen2.5-Coder 带 FIM 模板；需要客户端支持 llama.cpp 原生 infill，普通 OpenAI 补全协议不传 FIM 标记。

### 实际体验与建议

- 7B 在 M2 16 GB 上比较流畅，适合补全、写脚本、解释报错；复杂重构/长文件分析仍会力不从心。
- 14B 质量更好但明显慢，`-c` 超过 4K 容易和系统抢内存；只在需要时用。
- 想让模型“读整个仓库”，本地 16 GB 内存不现实，优先把相关文件/片段喂进上下文。
- 不想折腾 llama.cpp 参数的话，Ollama 里有现成的 `qwen2.5-coder:7b` / `qwen2.5-coder:14b`，见 [`../ollama/README.md`](../ollama/README.md)。

## 对接 OpenCode（本地代码 Agent）

[OpenCode](https://opencode.ai) 是一个终端里的开源编程 Agent：读项目、改文件、跑命令、调用工具。它支持 OpenAI 兼容的 provider，所以可以直接接 llama.cpp 的 `/v1` 服务。和 vLLM / LM Studio 不同，**llama.cpp 不在 OpenCode 的自动发现列表里**，需要手动写一个 custom provider 并显式列出模型。

### 1. 启动 llama.cpp 服务

Agent 场景需要相对大的上下文，16 GB M2 上比较现实的是 Qwen2.5-Coder-7B + 32K：

```bash
bash macos/llama-cpp/run.sh \
  --hf unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M \
  --ctx 32768 --alias qwen2.5-coder-7b
```

内存账：权重 4.36 GB + 32K KV（56 KiB/token ≈ 1.75 GiB）+ 运行时约 1–2 GB，16 GB 机器能放下，但别同时开太多大应用。

> 注意：Qwen2.5-Coder-7B 的代码能力好，但**不等价于 Agent 工具调用好**。实测在 Ollama 里它经常把工具调用输出成裸 JSON（不带 `<tool_call>` 标签），OpenCode/Cline 拿不到 `tool_calls`，表现为“只聊天不改文件”。如果遇到这种情况，Agent 场景换 `qwen3.5:4b`（Ollama，256K 原生、tools/thinking、MLX）或云端模型；Qwen2.5-Coder 更适合 FIM 补全和直接问答。

### 2. 安装并配置 OpenCode

```bash
curl -fsSL https://opencode.ai/install | bash
opencode --version
```

> 如果你的机器上已经装了 **OpenCode Desktop 2.x**（`/Applications/OpenCode.app`），不要再装它的 v1 CLI 去共用同一个数据库：v1/v2 的 `~/.local/share/opencode/opencode.db` schema 不同，会报 `Database is not empty and has no session table`，Ollama 侧表现为 `could not read OpenCode configuration: exit status 1`。用 Desktop 自带的 `Contents/Resources/opencode-cli`，或按 [`../ollama/README.md`](../ollama/README.md) 的「OpenCode Desktop 2.x 与 CLI 1.x 共库冲突」处理。

把本目录的 [`opencode.jsonc`](opencode.jsonc) 复制到项目根（只对当前项目生效）或全局配置目录：

```bash
# 项目级（推荐，跟着项目走）
cp macos/llama-cpp/opencode.jsonc ./opencode.jsonc

# 或全局
mkdir -p ~/.config/opencode
cp macos/llama-cpp/opencode.jsonc ~/.config/opencode/opencode.jsonc
```

核心内容就是：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "llamacpp": {
      "name": "llama.cpp (local)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://127.0.0.1:9931/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen2.5-coder-7b": {
          "name": "Qwen2.5 Coder 7B (llama.cpp)",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  },
  "model": "llamacpp/qwen2.5-coder-7b"
}
```

要点：

- `baseURL` 必须带 `/v1`，且端口和你 `llama serve --port` 一致。
- `models` 的 key 是 OpenCode 里的模型 ID，要和服务端 `-a/--alias` 对上；自定义 provider 不会自动发现模型，必须显式列。
- `limit.context` 要和服务端 `-c` 对齐（这里都是 32768），否则 OpenCode 可能发出超预算的请求，直接被 llama.cpp 截断/报错。
- llama.cpp 不带 `--api-key` 时不校验；如果带了，把 `settings.apiKey` 改成同一个值（或改用环境变量引用）。
- 旧版 OpenCode 的写法是 `provider` + `npm` + `options`，当前版本是 `providers` + `package` + `settings`；按你装的版本选，别混用。

### 64K 上下文

OpenCode 本身没有全局“默认 64K”的硬编码：不在 catalog 里的模型它会按 fallback 假设 200K context / 32K output，实际以 provider 返回或你在 `limit.context` 里写的值为准（见 [OpenCode models 文档](https://opencode.ai/docs/models/)）。**64K 来自 Ollama 的 OpenCode 接入建议**（“requires a context length of 64k or higher”，见 [Ollama OpenCode 文档](https://docs.ollama.com/integrations/opencode)）。在这台 16 GB 机器上要分两条路看：

- **Ollama 路线**：Ollama 会把 `num_ctx` 限制在模型训练长度内（源码 `llm/server.go` 里 `opts.NumCtx > trainCtx` 时直接压回，并打 `requested context size too large for model` 警告）。`qwen2.5-coder:7b` 是 32K，所以 `OLLAMA_CONTEXT_LENGTH=65536` 实际还是 32K。要用 Ollama 跑 64K，得换原生 64K+ 的模型（如 `llama3.1:8b` 的 128K，配合 `OLLAMA_KV_CACHE_TYPE=q8_0`），但它的代码能力不如 Qwen Coder。
- **llama.cpp 路线**：可以显式 `-c 65536`，并用 YaRN 把 32K 的 Qwen2.5-Coder-7B 外推到 64K：

  ```bash
  bash macos/llama-cpp/run.sh \
    --hf unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M \
    --ctx 65536 --alias qwen2.5-coder-7b \
    --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 32768 \
    --kv-type q8_0
  ```

  内存账：权重 4.36 GB + 64K KV（56 KiB/token × 65536 ≈ 3.5 GiB fp16，q8_0 约 1.75 GiB）+ 运行时；16 GB 能放下，但开着 IDE/浏览器就比较紧，建议 `--kv-type q8_0`。
  代价：Qwen2.5-Coder 原生是 32K，YaRN 外推到 64K 属于 extrapolation，长距离检索/精确引用的质量会下降；如果只是让 OpenCode 少触发压缩，32K + 自动压缩通常更稳。

  用了 64K 就把 OpenCode 配置里的 `limit.context` 改成 `65536`，并确认 `curl http://127.0.0.1:9931/props` 里的 `n_ctx` 也是 65536。

- **Qwen3.5 的 KV 比同尺寸稠密模型小**：它是混合注意力（32 层里只有 8 层 full attention，其余 linear attention），full attention 部分约 **32 KiB/token**，64K 只有约 2 GiB。所以 `qwen3.5:4b` 跑 64K 很轻松；`qwen3.5:9b`（权重 6.6–7.6 GB）跑 64K 会接近 10.7 GiB 的 Metal 预算，需要实测。
- **结论**：16 GB 上 64K 是“能跑但别指望和原生 128K 模型一样好用”；认真做仓库级 Agent，建议更大内存的机器或云端模型。

### 3. 在项目里使用

```bash
cd ~/code/your-project
opencode                 # 打开 TUI，首次用 /models 选 llamacpp/qwen2.5-coder-7b
# 或直接跑一次性任务（用配置里的默认模型）
opencode run "解释一下这个仓库的入口和主要模块"
```

TUI 里 `/models` 可以切换模型，`/help` 看命令；OpenCode 会在上下文快满时自动压缩历史。

### 4. 常见问题

- **`/models` 里看不到 llamacpp**：配置文件放错位置、JSONC 语法错误，或 provider ID 拼错；确认 `opencode.jsonc` 在项目根或 `~/.config/opencode/`。
- **请求 404/400**：`baseURL` 少了 `/v1`，或 `model` 和服务端 `-a` 别名不一致；先 `curl http://127.0.0.1:9931/v1/models` 确认。
- **Agent 不调用工具 / 只聊天**：模型工具调用能力不足或上下文不够；先保证 `/v1/chat/completions` 正常，再换更强的 Coder 模型。llama.cpp v1 默认启用 `--jinja`，模板没问题。
- **上下文超限**：把 `limit.context`、服务端 `-c`、模型原生窗口三者对齐；16 GB 上 7B 用 32K 是比较稳的上限。
- **想换成 Ollama**：OpenCode 对 Ollama 有内置 provider 和自动发现，见 [`../ollama/README.md`](../ollama/README.md)。

## 启动服务

### 一条命令

```bash
llama serve -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
  -c 4096 -ngl auto -fa auto -a llama3.2-3b \
  --host 127.0.0.1 --port 9931 --metrics
```

- `-c 4096`：上下文；**不要用默认的 `-c 0`**（0 表示用模型训练时的长度，Llama 3.2 是 128K，KV cache 会直接吃掉十几 GB）。
- `-ngl auto`：让 llama.cpp 自动决定卸载多少层到 Metal；`all` 全上 GPU，`0` 纯 CPU。
- `-a llama3.2-3b`：API 里的模型别名（`/v1/chat/completions` 的 `model` 字段）。
- 服务起来后浏览器打开 `http://127.0.0.1:9931` 就是内置的 **Llama Chat** Web UI；不需要就加 `--no-webui`。

默认监听 `127.0.0.1:9931`（v1.0 的默认端口；网上很多旧教程写 8080，是因为显式传了 `--port 8080`）。

### 本目录脚本

```bash
bash macos/llama-cpp/run.sh                       # 默认 Llama 3.2 3B Q4_K_M，端口 9931
bash macos/llama-cpp/run.sh --ctx 8192            # 8K 上下文（KV cache 翻倍）
bash macos/llama-cpp/run.sh --kv-type q8_0        # KV 量化，省一半（自动 -fa on）
bash macos/llama-cpp/run.sh --model ./model.gguf  # 本地 GGUF
bash macos/llama-cpp/run.sh --hf bartowski/Llama-3.1-8B-Instruct-GGUF:Q4_K_M
bash macos/llama-cpp/run.sh --install             # 没装时跑 llama.app 安装脚本
bash macos/llama-cpp/run.sh --dry-run             # 只打印命令
bash macos/llama-cpp/run.sh --status              # 看 /props、上下文、吞吐
bash macos/llama-cpp/run.sh --stop                # 停掉本脚本起的服务
```

脚本的后台进程日志在 `~/.llama-cpp/llama-serve.log`，pid 在 `~/.llama-cpp/llama-serve.pid`。首次启动会下载模型，`/health` 在加载完成前返回 503，脚本会一直等到 200。

### 常用参数速查

| 参数 | 作用 |
| --- | --- |
| `-m PATH` | 本地 GGUF 模型 |
| `-hf REPO[:QUANT]` | 从 Hugging Face 拉模型（quant 缺省按 Q4_K_M） |
| `-c N` / `--ctx-size N` | 上下文长度；0 = 模型训练长度（通常很大，慎用） |
| `-ngl N\|auto\|all` | 卸载到 GPU 的层数 |
| `-fa on\|off\|auto` | FlashAttention，默认 auto |
| `-ctk TYPE` / `-ctv TYPE` | K/V cache 的数据类型（`f16` / `q8_0` / `q4_0` …）；量化需要 `-fa on` |
| `-np N` | 并行槽（并发请求）数；v1 默认 auto |
| `-b N` / `-ub N` | 逻辑批 / 物理批大小，影响 prefill 速度与显存峰值 |
| `-t N` | CPU 线程数（默认自动） |
| `-fit on\|off` | 自动调整未显式指定的参数以适配设备内存（v1 默认 on） |
| `-lm MODE` | 加载方式：`mmap`（默认）/ `mlock` / `mmap+mlock` / `dio` |
| `--host` / `--port` | 监听地址与端口（默认 `127.0.0.1:9931`） |
| `--metrics` | 打开 Prometheus 指标端点 `/metrics` |
| `--no-webui` | 关掉内置 Web UI |
| `-a ALIAS` | API 模型别名 |
| `--api-key KEY` | 给 API 加认证（非本机访问时建议加） |

`llama serve --help` 输出全部参数；也可以看上游的 [server 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。

## API

服务地址 `http://127.0.0.1:9931`，同时有 llama.cpp 原生接口和 OpenAI 兼容接口：

```bash
MODEL="llama3.2-3b"

# 健康检查：加载中返回 503，就绪返回 200
curl -s http://127.0.0.1:9931/health

# 服务信息（上下文 n_ctx、槽位数、模型路径、chat template）
curl -s http://127.0.0.1:9931/props

# OpenAI 兼容对话
curl -s http://127.0.0.1:9931/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"stream\":false}"

# OpenAI 兼容模型列表
curl -s http://127.0.0.1:9931/v1/models

# llama.cpp 原生补全接口（不是 OpenAI 格式）
curl -s http://127.0.0.1:9931/completion \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"用一句话解释 KV cache\",\"n_predict\":128}"

# 监控：需要启动时加 --metrics
curl -s http://127.0.0.1:9931/metrics
```

Python 用 `openai` SDK 时把 `base_url` 设为 `http://127.0.0.1:9931/v1`、`api_key` 随便填（如 `"llama"`，本地未开 `--api-key` 时会忽略）。

多模态模型（如 Llama 3.2 Vision 的 GGUF 仓库）用 `-hf` 时如果仓库带 mmproj，会自动下载并在 `/v1/chat/completions` 里接受图片（base64 或 URL，具体看 `/props` 的 `modalities`）。

## 上下文与 KV cache：和本仓库的关系

llama.cpp 的 `-c` 直接决定 KV cache 的总容量；默认 `-c 0` 会用模型训练长度（Llama 3.2 是 128K），所以日常一定要显式给一个值。

- 每个 token 每层的 K/V 占用：

  ```text
  2（K 和 V）× n_layers × n_kv_heads × head_dim × 每个元素字节数
  ```

- 以 Llama 3.2 3B（28 层、8 个 KV 头、head_dim=128、fp16）为例：

  ```text
  2 × 28 × 8 × 128 × 2 B = 112 KiB / token
  -c 4096  ≈ 448 MiB
  -c 8192  ≈ 896 MiB
  -c 131072 ≈ 14 GiB
  ```

- 换成 Llama 3.1 8B（32 层、8 个 KV 头、head_dim=128）= 128 KiB/token：`-c 4096` ≈ 512 MiB，`-c 8192` ≈ 1 GiB。
- `-ctk q8_0 -ctv q8_0` 把 KV 压到约一半，`q4_0` 约四分之一；量化 V cache 需要 FlashAttention，所以要么 `-fa on`，要么像脚本一样在设置 `--kv-type` 时自动打开。
- `-c` 是 **KV pool 的总预算**。槽位数用默认值（auto）时，v1 默认 `kv_unified`：所有槽共享这 `-c` 个 token；显式写 `-np N` 时默认按槽拆分（每槽约 `n_ctx / N`），可以用 `-kvu` 强制统一。调并发前先算总内存。
- 观察实际分配：`/props` 的 `default_generation_settings.n_ctx`、服务日志里 `llama_context: n_ctx = ...` 和 `flash_attn = ...`、`ggml_metal`/`offloaded N/M layers` 行。

对照 [`../../example/kv_cache.py`](../../example/kv_cache.py)：那边是单请求的 K/V 追加与复用；llama.cpp 把同样的 K/V 放在 Metal 的统一内存里，并负责多请求、批处理和上下文管理。跨请求的前缀共享见 [`../../example/prefix_cache_demo.py`](../../example/prefix_cache_demo.py)。

## 性能与调优

- **decode 受内存带宽限制**：M 系列里 Pro/Max/Ultra 带宽更高，tokens/s 更高；`-ngl` 决定多少层在 GPU 上，掉到 CPU 会明显变慢。
- 先看日志确认 Metal 生效：启动日志里应有 `ggml_metal` / `offloaded N/M layers to GPU`；没有的话检查是不是用了 `-ngl 0` 或 Intel 机器。
- 长 prompt（贴代码/文档）主要吃 prefill，`-b/-ub` 会影响 prefill 速度和显存峰值；短回复的延迟更依赖首 token 时间。
- 用 `/metrics` 的 `prompt_tokens_seconds` / `predicted_tokens_seconds` 看吞吐；要跑基准用 `llama bench -m model.gguf -p 512 -n 128`。
- 模型加载用 mmap（默认）时，首次加载慢、内存占用看起来“省”，但推理时仍会驻留；想避免换页可以试 `-lm mmap+mlock`（内存要够）。
- v1 的 `-fit on`（默认）会自动调整**没有显式指定**的参数以适配设备内存；显式给了 `-c`/`-ngl` 时以你的值为准。

## 和 Ollama / Docker 版的关系

| | llama.cpp（本文） | Ollama（[`../ollama/`](../ollama/README.md)） | Docker Ollama（[`../../docker/ollama/`](../../docker/ollama/README.md)） |
| --- | --- | --- | --- |
| 模型管理 | `-hf` / 本地 GGUF | `ollama pull` 模型库 | 容器内 `ollama pull` |
| Metal GPU | 有 | 有 | 无（macOS 上 Docker 拿不到 GPU） |
| 参数粒度 | 最细（`-c`、`-ngl`、`-fa`、`-ctk/-ctv`、`-b/-ub`） | 中（`OLLAMA_*`、`num_ctx`） | 中 |
| 端口 | 9931 | 11434 | 宿主机映射 11434 |
| 适合 | 调参/实验/对照 KV cache | 日常使用、省心 | 练 Docker、Linux 服务器 |

llama.cpp 用 9931、Ollama 用 11434，端口不冲突，可以并存；但每套都会加载一份模型，同时跑要注意总内存。同一个端口不要起两个实例。

## 常见问题

**`/health` 一直是 503**

模型还在下载或加载。首次 `-hf` 要下载数 GB；用本脚本启动时看 `tail -f ~/.llama-cpp/llama-serve.log`，手动前台跑就在终端看进度；加载完成后返回 `{"status":"ok"}`。

**地址被占用（`bind: address already in use`）**

```bash
lsof -nP -iTCP:9931 -sTCP:LISTEN
```

可能是上一次没退出的 llama.cpp，或另一个脚本实例。本脚本起的用 `bash macos/llama-cpp/run.sh --stop`；手动起的在终端 `Ctrl+C`。

**模型下载慢 / 失败**

设镜像 `HF_ENDPOINT=https://hf-mirror.com`；gated 仓库要 `HF_TOKEN`/`-hft`；已经下载过的模型在 `~/Library/Caches/llama.cpp`，重复启动不会重下。

**内存不够 / 机器变卡**

优先降 `-c`（4096 → 2048），再用 `-ctk q8_0 -ctv q8_0`，再换更小的 quant（Q4_K_M → Q3_K_M）或更小的模型；`-ngl` 调小会把层放回 CPU，反而更慢，除非显存真的不够。

**确认 Metal 有没有用上**

看启动日志的 `ggml_metal` / `offloaded N/M layers`；`llama serve --list-devices` 可以列出可用设备；Intel Mac 只有 CPU。

**想用更长的上下文**

llama.cpp 支持 RoPE 缩放（`--rope-scaling`、`--rope-scale` 等），但 KV cache 会按实际 `-c` 增长，先按上面的公式算内存；Llama 3.2 训练上下文就是 128K，不需要外推。

**Web UI 打不开**

确认启动时没加 `--no-webui`，并且监听地址是 `127.0.0.1`（或从别的机器访问时用 `--host 0.0.0.0`，并考虑 `--api-key`）。

**Ollama 的模型能直接给 llama.cpp 用吗**

不建议直接指 `~/.ollama/models/blobs`：那是按 manifest 组织的内部结构。用 `-hf` 从 HF 拉同一模型的 GGUF；如果只有 Hugging Face 原始权重，可以用 llama.cpp 的 [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) 自己转。

## 清理 / 卸载

```bash
# 停掉本脚本起的服务
bash macos/llama-cpp/run.sh --stop

# 官方脚本安装（v1 自更新/重装都用这个）
curl -LsSf https://llama.app/install.sh | sh

# 彻底删除官方脚本装的二进制和缓存
rm -rf ~/.llama-app ~/.local/bin/llama
rm -rf ~/Library/Caches/llama.cpp      # 模型缓存，确认不再需要再删

# Homebrew 安装的
brew uninstall llama.cpp
```

## 参考

- llama.cpp 仓库：<https://github.com/ggml-org/llama.cpp>
- 安装文档：<https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md>
- 构建文档（Metal 默认开启）：<https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- server 参数与 API：<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- 命令行工具：<https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md>
- 官方安装脚本：<https://llama.app/install.sh>
- Ollama 版（自动管理模型、菜单栏 App）：[`../ollama/README.md`](../ollama/README.md)
- Docker 版（Linux 容器 / CPU）：[`../../docker/ollama/README.md`](../../docker/ollama/README.md)
