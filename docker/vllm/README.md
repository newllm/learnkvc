# 用 Docker 跑 vLLM（CPU / GPU，PagedAttention）

用 Docker 起 [vLLM](https://github.com/vllm-project/vllm) 的 OpenAI 兼容服务，验证 `/v1/chat/completions`，并观察 PagedAttention 的 KV cache 块信息。默认模型是 `Qwen/Qwen2.5-0.5B-Instruct`：体积小、无需申请权限，CPU 也能跑通。

vLLM 官方同时提供 GPU 和 CPU 镜像：

| 模式 | 镜像 | 说明 |
| --- | --- | --- |
| GPU | `vllm/vllm-openai:latest` | 需要 NVIDIA GPU + nvidia-container-toolkit |
| CPU | `vllm/vllm-openai-cpu:latest-x86_64` | 不需要 GPU；ARM 自动用 `latest-arm64` |

目录内容：

```text
docker/vllm/
├── run.sh     # 启动 + 等待就绪 + 调用接口 + 打印 KV cache 日志
└── README.md  # 本文件
```

## 前置条件

| 项 | 要求 |
| --- | --- |
| Docker | Docker Engine / Desktop，daemon 已启动 |
| CPU 模式 | Linux x86_64（AVX2 起，推荐 AVX512）或 ARM64；内存建议 ≥ 8 GB |
| GPU 模式 | NVIDIA GPU + 驱动，已安装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| 其他 | `curl`；gated 模型需要 Hugging Face token（`HF_TOKEN`） |

> macOS 上 Docker 跑的是 Linux 容器，Apple GPU 用不到；CPU 镜像可以用，但速度取决于机器。

## 镜像和模型从哪里来

| 东西 | 来源 |
| --- | --- |
| GPU 镜像 | Docker Hub 的 [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai) |
| CPU 镜像 | Docker Hub 的 [`vllm/vllm-openai-cpu`](https://hub.docker.com/r/vllm/vllm-openai-cpu)（`latest-x86_64` / `latest-arm64`） |
| 模型 | [Hugging Face](https://huggingface.co) 上的模型 id |

模型在容器启动时下载到 Hugging Face 缓存；脚本把宿主机的 `${HF_CACHE:-~/.cache/huggingface}` 挂到容器的 `/root/.cache/huggingface`，第二次启动不会重复下载。

gated 模型（如 Llama 系列）需要先在 Hugging Face 页面接受协议，然后：

```bash
HF_TOKEN=hf_xxx bash docker/vllm/run.sh --model meta-llama/Llama-3.2-3B-Instruct
```

## 快速开始

```bash
chmod +x docker/vllm/run.sh

bash docker/vllm/run.sh                # 有 NVIDIA GPU 用 GPU，否则自动用 CPU
bash docker/vllm/run.sh --cpu          # 强制 CPU 镜像
bash docker/vllm/run.sh --gpu          # 强制 GPU 镜像
```

脚本会依次：

1. 按 `nvidia-smi` 自动选择 CPU / GPU（可用 `--cpu` / `--gpu` 覆盖）；
2. 检查 docker、daemon、curl；GPU 模式额外检查 `nvidia-smi`；
3. 删掉同名旧容器（HF 缓存保留在宿主机目录）；
4. `docker run` 启动 vLLM，映射 `8000`，挂载 HF 缓存；
5. 轮询 `/health` 直到就绪（首次要下载模型、编译 kernel，可能几分钟）；
6. 从日志里过滤 `KV cache size` / `Maximum concurrency`，调一次 `/v1/chat/completions` 并打印回答。

其他常用参数：

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash docker/vllm/run.sh
MAX_MODEL_LEN=8192 bash docker/vllm/run.sh
bash docker/vllm/run.sh --dry-run      # 只打印 docker 命令
```

## CPU 模式

```bash
bash docker/vllm/run.sh --cpu

# 更贴近官方推荐的 CPU 配置
VLLM_CPU_KVCACHE_SPACE=8 \
VLLM_CPU_OMP_THREADS_BIND=0-15 \
DTYPE=bfloat16 \
bash docker/vllm/run.sh --cpu
```

CPU 相关的参数：

| 参数 | 作用 |
| --- | --- |
| `VLLM_CPU_KVCACHE_SPACE` | KV cache 空间，单位 GiB（如 `8`）；CPU 上不用 `--gpu-memory-utilization` |
| `VLLM_CPU_OMP_THREADS_BIND` | OpenMP 线程绑核，如 `0-15`、`auto`、`nobind` |
| `DTYPE=bfloat16` | CPU 上推荐显式指定 bfloat16，避免 fp16 在 CPU 上的数值/性能问题 |
| `TP` | CPU 也支持张量并行，建议按 NUMA 节点设置 |
| `BLOCK_SIZE` | CPU 默认 128；官方建议用 32 的倍数（8/16 会导致 kernel 效率低） |

脚本在 CPU 模式下会额外加 `--security-opt seccomp=unconfined --cap-add SYS_NICE --shm-size=4g`：vLLM CPU 的 NUMA 调用（`get_mempolicy` / `migrate_pages`）在默认 seccomp 下可能被拒绝，这两个参数是官方文档给出的最小放宽方式。如果你不能接受放宽 seccomp，可以自定义 profile 或忽略这些 warning（功能不受影响，只是 NUMA 优化可能不生效）。

**性能预期**：CPU 推理明显慢于 GPU，适合验证链路、学习 PagedAttention，不适合高吞吐服务。脚本会按 `uname -m` 自动选择 `latest-x86_64` 或 `latest-arm64`；要固定版本或换镜像时：

```bash
CPU_IMAGE=vllm/vllm-openai-cpu:<version>-x86_64 bash docker/vllm/run.sh --cpu
```

## GPU 模式

```bash
bash docker/vllm/run.sh --gpu
GPU_MEM_UTIL=0.8 TP=2 bash docker/vllm/run.sh --gpu
```

GPU 模式使用 `--gpus all --ipc=host`；`TP>1` 时 `--ipc=host` 是必须的（共享内存不足会导致 NCCL 报错）。`GPU_MEM_UTIL` 对应 `--gpu-memory-utilization`，决定启动时能划出多少 KV cache 块。

## 常用模型

| 模型 | 资源（约） | 说明 |
| --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | CPU 内存 2 GB+ / GPU 2 GB+ | 默认，链路验证 |
| `Qwen/Qwen2.5-1.5B-Instruct` | CPU 4 GB+ / GPU 4 GB+ | 质量更好一点 |
| `Qwen/Qwen2.5-7B-Instruct` | GPU 16 GB+ | CPU 也能跑但很慢 |
| `meta-llama/Llama-3.2-3B-Instruct` | GPU 8 GB+ | 需 HF 授权 + `HF_TOKEN` |

上表是粗略估算；实际还要加 KV cache。CPU 模式下 KV cache 由 `VLLM_CPU_KVCACHE_SPACE` 控制，GPU 模式下由 `--gpu-memory-utilization` 和 `--max-model-len` 推算。

## API 示例

服务起来后（`http://127.0.0.1:8000`，OpenAI 兼容）：

```bash
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

# 健康检查 / 模型列表
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models

# Chat
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":64}"

# 纯文本补全（不依赖 chat template）
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"KV cache 的作用是\",\"max_tokens\":64}"

# Prometheus 指标：块池使用率等
curl -s http://127.0.0.1:8000/metrics | grep -E 'vllm:(gpu_cache_usage_perc|num_requests_running)'
```

## PagedAttention 与参数

vLLM 的 KV cache 是固定大小块的池子，每个请求有一张 block table，把逻辑位置映射到物理块。对应关系：

| 参数 | 作用 |
| --- | --- |
| `--block-size` | 每块多少 token；GPU 默认 16，CPU 默认 128，CPU 建议 32 的倍数 |
| `--gpu-memory-utilization` | GPU 模式：允许使用的显存比例，决定块池大小 |
| `VLLM_CPU_KVCACHE_SPACE` | CPU 模式：KV cache 空间（GiB） |
| `--max-model-len` | 单请求最大上下文，决定一个请求最多要多少块，也影响并发数 |
| `--tensor-parallel-size` | 多卡/多 NUMA 并行 |

启动日志里能看到块池规模，例如：

```text
INFO ... GPU KV cache size: 100,000 tokens
INFO ... Maximum concurrency for 4,096 tokens per request: 24.41x
```

`/metrics` 里的 `vllm:gpu_cache_usage_perc` 是块池使用率；接近 1 时调度器会开始抢占请求。

这和 [`../../example/paged_attention.py`](../../example/paged_attention.py) 是同一件事：那边用 4 个 token 一块方便观察，`--block-size` 改变的就是这个粒度。

## 和本仓库的关系

- [`example/paged_attention.py`](../../example/paged_attention.py)：用 block table 把逻辑 token 映射到物理块，并验证 attention 与连续缓存一致；
- [`example/PAGED_ATTENTION_READING.md`](../../example/PAGED_ATTENTION_READING.md)：逐段讲解块池、`gather()`、causal mask 与 vLLM 的对应关系；
- 本目录：把同一套机制跑在真实的 vLLM 容器里，用参数和 `/metrics` 观察它。

## 常见问题

**CPU 启动报 illegal instruction**：CPU 需要 AVX2（推荐 AVX512），先用 `lscpu | grep avx` 确认；老 CPU 建议换机器或改用 Ollama 示例。

**`get_mempolicy: Operation not permitted`**：默认 seccomp 限制了 NUMA 系统调用；脚本已经加了 `seccomp=unconfined` 和 `SYS_NICE`，如仍出现可忽略（功能正常，NUMA 优化可能不生效）。

**启动时报显存不足（OOM）**：GPU 模式先降 `MAX_MODEL_LEN`（如 2048）、再降 `GPU_MEM_UTIL`（如 0.8）；CPU 模式降 `VLLM_CPU_KVCACHE_SPACE` 或换更小模型。

**模型下载失败（401/403）**：gated 模型需要 `HF_TOKEN`，并先在 Hugging Face 页面接受许可。

**第一次启动很慢**：要下载数 GB 权重并做编译/图捕获；后续启动会复用缓存。

**端口被占用**：`lsof -i :8000`，或 `PORT=8001`。

**多卡启动失败**：确认 `TP` 不超过卡数，并保留 `--ipc=host`。

## 清理

```bash
docker stop vllm
docker rm -f vllm
# 模型缓存默认在 ~/.cache/huggingface，需要时再手动清理
```
