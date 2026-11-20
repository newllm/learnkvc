# 用 Docker 跑 vLLM（PagedAttention）

用 Docker 起 [vLLM](https://github.com/vllm-project/vllm) 的 OpenAI 兼容服务，验证 `/v1/chat/completions`，并观察 PagedAttention 的 KV cache 块信息。默认模型是 `Qwen/Qwen2.5-0.5B-Instruct`：体积小、无需申请权限，适合先把链路跑通。

目录内容：

```text
docker/vllm/
├── run.sh     # 启动 + 等待就绪 + 调用一次接口 + 打印 KV cache 日志
└── README.md  # 本文件
```

## 前置条件

| 项 | 要求 |
| --- | --- |
| GPU | NVIDIA GPU + 驱动；`nvidia-smi` 正常 |
| Docker | Docker Engine / Desktop，已安装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| 显存 | 0.5B 模型约 2 GB 起；7B 模型建议 16 GB 以上（含 KV cache） |
| 其他 | `curl`；gated 模型需要 Hugging Face token（`HF_TOKEN`） |

> `vllm/vllm-openai` 官方镜像要求 NVIDIA GPU。没有 GPU 时可以用 [`../ollama/`](../ollama/)：同样是本地模型服务，CPU 也能跑。CPU 版 vLLM 需要从源码编译（`VLLM_TARGET_DEVICE=cpu`），不在本示例范围内。

## 镜像和模型从哪里来

| 东西 | 来源 |
| --- | --- |
| Docker 镜像 | Docker Hub 的 [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai)（`docker.io/vllm/vllm-openai:latest`） |
| 模型 | [Hugging Face](https://huggingface.co) 上的模型 id，如 `Qwen/Qwen2.5-0.5B-Instruct`、`meta-llama/Llama-3.2-3B-Instruct`（需授权） |

模型会在容器启动时下载到 Hugging Face 缓存目录；脚本把宿主机的 `${HF_CACHE:-~/.cache/huggingface}` 挂到容器的 `/root/.cache/huggingface`，所以第二次启动不会重复下载。

gated 模型（如 Llama 系列）需要先在 Hugging Face 页面接受协议，然后：

```bash
HF_TOKEN=hf_xxx bash docker/vllm/run.sh --model meta-llama/Llama-3.2-3B-Instruct
```

## 快速开始

```bash
chmod +x docker/vllm/run.sh
bash docker/vllm/run.sh
```

脚本会依次：

1. 检查 `docker`、daemon、`nvidia-smi`、`curl`；
2. 删掉同名旧容器（模型缓存在宿主机目录里，不会重复下载）；
3. `docker run` 启动 vLLM，映射 `8000`，挂载 HF 缓存，传 `--model` 等参数；
4. 轮询 `/health` 直到就绪（首次要下载模型、编译 kernel，可能几分钟）；
5. 从容器日志里过滤 `GPU KV cache size` / `Maximum concurrency`；
6. 调一次 `/v1/chat/completions` 并打印回答。

常用变体：

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash docker/vllm/run.sh
MAX_MODEL_LEN=8192 bash docker/vllm/run.sh          # 更长上下文，KV cache 占用更大
GPU_MEM_UTIL=0.8 bash docker/vllm/run.sh            # 给别的进程留显存
TP=2 bash docker/vllm/run.sh                        # 两张卡张量并行
BLOCK_SIZE=8 bash docker/vllm/run.sh                # 更细的 PagedAttention 块
bash docker/vllm/run.sh --dry-run                   # 只打印 docker 命令
```

## 常用模型

| 模型 | 显存（约） | 说明 |
| --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | 2 GB | 默认，链路验证 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 4 GB | 质量更好一点 |
| `Qwen/Qwen2.5-7B-Instruct` | 16 GB+ | 需要较大显存 |
| `meta-llama/Llama-3.2-3B-Instruct` | 8 GB+ | 需 HF 授权 + `HF_TOKEN` |

上表是 fp16 权重的粗略估算；实际占用还要加上 KV cache，vLLM 启动时会按 `--gpu-memory-utilization` 和 `--max-model-len` 计算能放多少块。

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

## PagedAttention 与启动参数

vLLM 的 KV cache 不是一整段连续显存，而是一个固定大小块的池子；每个请求有一张 block table，把逻辑位置映射到物理块。对应关系：

| vLLM 参数 | 作用 |
| --- | --- |
| `--block-size` | 每个块多少 token（默认 16；常用 8/16/32） |
| `--gpu-memory-utilization` | 允许 vLLM 使用的显存比例，启动时据此算出块池大小 |
| `--max-model-len` | 单请求最大上下文，决定一个请求最多要多少块，也影响并发数 |
| `--tensor-parallel-size` | 多卡张量并行；`>1` 时建议保留 `--ipc=host` |

启动日志里能看到块池规模，例如：

```text
INFO ... GPU KV cache size: 100,000 tokens
INFO ... Maximum concurrency for 4,096 tokens per request: 24.41x
```

`--block-size` 越小，最后一块的内部碎片越少，但 block table 更长、kernel 调度更碎；越大则相反。这和 [`../../example/paged_attention.py`](../../example/paged_attention.py) 里 `BLOCK_SIZE` 的取舍是同一件事，只是那边用 4 个 token 一块方便观察。

`/metrics` 里的 `vllm:gpu_cache_usage_perc` 是块池使用率；接近 1 时调度器会开始抢占（preempt）请求的 KV cache。

## 和本仓库的关系

- [`example/paged_attention.py`](../../example/paged_attention.py)：用 block table 把逻辑 token 映射到物理块，并验证 attention 结果与连续缓存一致；
- [`example/PAGED_ATTENTION_READING.md`](../../example/PAGED_ATTENTION_READING.md)：逐段讲解块池、`gather()`、causal mask 与 vLLM 的对应关系；
- 本目录：把同一套机制跑在真实的 vLLM 容器里，用 `--block-size` / `--gpu-memory-utilization` / `/metrics` 观察它。

## 常见问题

**`nvidia-smi` 失败 / 容器看不到 GPU**：确认驱动和 nvidia-container-toolkit 已安装并重启过 Docker：

```bash
nvidia-smi
docker info | grep -i runtime
docker run --rm --gpus all vllm/vllm-openai:latest --help | head
```

**启动时报显存不足（OOM）**：先降低 `MAX_MODEL_LEN`（如 2048），再降低 `GPU_MEM_UTIL`（如 0.8），或换更小的模型。

**模型下载失败（401/403）**：gated 模型需要 `HF_TOKEN`，并先在 Hugging Face 页面接受许可。

**第一次启动很慢**：要下载数 GB 权重，并做 CUDA graph / kernel 编译；后续启动会复用缓存。

**端口被占用**：`lsof -i :8000`，或 `PORT=8001`。

**多卡启动失败**：确认 `TP` 不超过卡数，并保留 `--ipc=host`（共享内存不足会导致 NCCL 报错）。

## 清理

```bash
docker stop vllm
docker rm -f vllm
# 模型缓存默认在 ~/.cache/huggingface，需要时再手动清理
```
