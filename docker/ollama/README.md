# 用 Docker 跑 Ollama（Llama 等模型）

一个可以直接照抄的本地大模型运行示例：用 Docker 起 [Ollama](https://ollama.com/)，拉一个 Llama 模型，然后用 HTTP API 对话。Ollama 内部是 llama.cpp，模型默认 4-bit 量化，CPU 也能跑，有 NVIDIA GPU 时会自动用上。

目录内容：

```text
docker/ollama/
├── run.sh     # 启动 + 拉模型 + 调一次 API 的脚本（带 CPU/GPU 判断）
└── README.md  # 本文件
```

## 前置条件

| 项 | 要求 |
| --- | --- |
| Docker | Docker Desktop 或 Docker Engine，daemon 已启动 |
| CPU 运行 | x86_64 / arm64 均可；内存 ≥ 模型大小 + 上下文开销 |
| GPU 运行 | NVIDIA 驱动 + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，`nvidia-smi` 能正常输出 |
| macOS | Docker 容器**用不到** Apple GPU（Metal），只能 CPU；想用 GPU 建议直接在 macOS 上装 Ollama |

## 快速开始

```bash
chmod +x docker/ollama/run.sh
bash docker/ollama/run.sh
```

脚本会依次：

1. 检查 `docker` 和 daemon；
2. 用 `nvidia-smi` 自动判断 CPU / GPU（也可用 `--cpu` / `--gpu` 强制）；
3. 删掉同名旧容器（模型数据在 volume 里，不会重复下载）；
4. `docker run` 启动 Ollama，映射 `11434`；
5. 轮询 `ollama list` 等待服务就绪；
6. `ollama pull` 拉取模型；
7. 调一次 `POST /api/generate`，并打印 `ollama ps` 的内存占用。

换模型、端口或强制 CPU：

```bash
MODEL=llama3.1:8b bash docker/ollama/run.sh
PORT=11500 bash docker/ollama/run.sh
bash docker/ollama/run.sh --cpu
bash docker/ollama/run.sh --dry-run    # 只打印将执行的 docker 命令
```

## CPU / GPU 的原始命令

脚本本质上就是这两条命令加上检查与等待：

```bash
# CPU
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# NVIDIA GPU（需要 nvidia-container-toolkit）
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

确认 GPU 是否真的进了容器：

```bash
docker exec ollama nvidia-smi          # 能看到显卡即成功
docker logs ollama | head              # 启动日志里会打印检测到的 GPU
```

## 常用模型

| 模型 | 下载体积（约） | 说明 |
| --- | --- | --- |
| `llama3.2:1b` | 1.3 GB | 最轻量，适合先验证链路 |
| `llama3.2:3b` | 2.0 GB | 默认示例，CPU 也能接受 |
| `llama3.1:8b` | 4.7 GB | 效果更好，建议 8 GB+ 显存 |
| `qwen2.5:3b` / `qwen2.5:7b` | 1.9 / 4.7 GB | 中文表现不错 |
| `deepseek-r1:7b` | 4.7 GB | 带推理链输出的蒸馏模型 |

体积以实际下载为准；Ollama 默认拉取 4-bit 量化版本，所以比原始 fp16 权重小很多。

## API 示例

服务起来后（`http://127.0.0.1:11434`）：

```bash
# 版本 / 已下载模型
curl -s http://127.0.0.1:11434/api/version
curl -s http://127.0.0.1:11434/api/tags

# 生成（stream=false 一次性返回）
curl -s http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","prompt":"用一句话解释 KV cache","stream":false}'

# 对话
curl -s http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"你好"}],"stream":false}'

# 当前加载的模型与内存占用
curl -s http://127.0.0.1:11434/api/ps
```

默认 `stream=true` 会返回 NDJSON 流；脚本里用 `stream:false` 方便直接解析。也可以进容器交互聊天：

```bash
docker exec -it ollama ollama run llama3.2:3b
```

## 数据、端口与常用环境变量

- 模型与配置存在 volume `ollama` 里，对应容器的 `/root/.ollama`；删容器不会删模型，删 volume 才会。
- 宿主机端口由 `-p` 映射，默认 `11434`；冲突时换 `PORT=11500`。
- 常用环境变量（较新版本的 Ollama 支持）：

| 变量 | 作用 |
| --- | --- |
| `OLLAMA_NUM_PARALLEL` | 同时处理的请求数；每路请求各自占一份 KV cache |
| `OLLAMA_MAX_LOADED_MODELS` | 同时加载几个模型 |
| `OLLAMA_KEEP_ALIVE` | 模型空闲多久后卸载（如 `5m`、`-1` 常驻） |
| `OLLAMA_KV_CACHE_TYPE` | KV cache 精度：`f16`（默认）、`q8_0`、`q4_0` |
| `OLLAMA_FLASH_ATTENTION` | 是否启用 FlashAttention（`1` 开启） |

传给容器的方式，例如把 KV cache 量化成 `q8_0` 并允许两路并发：

```bash
docker run -d --name ollama \
  -v ollama:/root/.ollama -p 11434:11434 \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -e OLLAMA_NUM_PARALLEL=2 \
  --restart unless-stopped \
  ollama/ollama
```

## 上下文与 KV cache：和本仓库的关系

这个仓库的前两个示例讲的就是 KV cache 怎么算、怎么复用；Ollama 是同一套东西在真实推理引擎里的样子（llama.cpp）：

- 请求的上下文长度 `num_ctx` 决定 KV cache 的大小。每个 token、每层的 KV 占用是：

  ```text
  2（K 和 V）× n_layers × n_kv_heads × head_dim × 每个元素字节数
  ```

  以 Llama 3 8B（32 层、8 个 KV 头、head_dim=128、fp16）为例：

  ```text
  2 × 32 × 8 × 128 × 2 B = 128 KiB / token
  8192 token ≈ 1 GiB KV cache
  ```

- `OLLAMA_NUM_PARALLEL=2` 会同时保留两份上下文，KV 显存翻倍；`OLLAMA_KV_CACHE_TYPE=q8_0` 能把 KV 占用减半。
- 用 API 显式设置上下文和最多生成多少 token：

  ```bash
  curl -s http://127.0.0.1:11434/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"llama3.2:3b",
         "messages":[{"role":"user","content":"你好"}],
         "stream":false,
         "options":{"num_ctx":8192,"num_predict":256}}'
  ```

- `docker exec ollama ollama ps` 里的 SIZE 是加载后的总占用（权重 + 上下文 + 运行时开销），可以用来验证 `num_ctx` 的效果。

对比 [`../../example/kv_cache.py`](../../example/kv_cache.py)：那边是单请求的 K/V 追加与复用；Ollama/llama.cpp 则把同样的 K/V 放在 GPU 显存里，并处理多请求、采样和上下文管理。更上层的跨请求前缀共享见 [`../../example/prefix_cache_demo.py`](../../example/prefix_cache_demo.py)。

## 常见问题

**GPU 没被用上**

```bash
nvidia-smi                       # 宿主机能识别显卡
docker info | grep -i runtime    # 应能看到 nvidia
docker exec ollama nvidia-smi    # 容器内也能识别
```

三者任一失败，通常是没装 `nvidia-container-toolkit`，或安装后没重启 Docker。

**端口被占用**：`lsof -i :11434` 找到进程，或用 `PORT=11500`。

**拉模型很慢或失败**：先试 `llama3.2:1b`；下载支持断点续传，重跑 `ollama pull` 即可；公司网络需要给 Docker 配代理。

**容器名冲突**：脚本会自动 `docker rm -f ollama`；手工执行时先删同名容器。

**macOS 上很慢**：Docker 里的 Ollama 只能用 CPU。想用 Apple GPU，请在 macOS 上直接安装 Ollama（`brew install ollama` 或官网 dmg），不要走容器。

## 清理

```bash
docker stop ollama                 # 停止
docker rm -f ollama                # 删除容器
docker volume rm ollama            # 删除已下载的模型（谨慎）
```
