#!/usr/bin/env bash
# 用 Docker 起一个 vLLM OpenAI 兼容服务，并用 PagedAttention 跑一次推理。
#
# vLLM 把 KV cache 放在固定大小的 block 池里，每个请求用 block table 索引；
# 这个脚本只负责把服务拉起来并验证接口：
#   镜像来源：Docker Hub 的 vllm/vllm-openai（GPU）与 vllm/vllm-openai-cpu（CPU）
#   模型来源：Hugging Face（默认 Qwen/Qwen2.5-0.5B-Instruct）
#
# 用法：
#   bash docker/vllm/run.sh                 # 有 NVIDIA GPU 用 GPU，否则用 CPU
#   bash docker/vllm/run.sh --cpu           # 强制 CPU（官方 CPU 镜像）
#   bash docker/vllm/run.sh --gpu           # 强制 GPU
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct bash docker/vllm/run.sh
#   VLLM_CPU_KVCACHE_SPACE=8 bash docker/vllm/run.sh --cpu
#   bash docker/vllm/run.sh --dry-run
#
# 环境变量：
#   ACCEL(auto|cpu|gpu) / MODEL / PORT / MAX_MODEL_LEN / GPU_MEM_UTIL / TP /
#   BLOCK_SIZE / DTYPE / HF_CACHE / HF_TOKEN / HEALTH_TIMEOUT /
#   VLLM_CPU_KVCACHE_SPACE / VLLM_CPU_OMP_THREADS_BIND / CPU_IMAGE / GPU_IMAGE

set -euo pipefail

CONTAINER="${CONTAINER:-vllm}"
GPU_IMAGE="${GPU_IMAGE:-vllm/vllm-openai:latest}"
case "$(uname -m)" in
    arm64|aarch64) DEFAULT_CPU_TAG="latest-arm64" ;;
    *)             DEFAULT_CPU_TAG="latest-x86_64" ;;
esac
CPU_IMAGE="${CPU_IMAGE:-vllm/vllm-openai-cpu:${DEFAULT_CPU_TAG}}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TP="${TP:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
DTYPE="${DTYPE:-}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
ACCEL="${ACCEL:-auto}"
DEMO=1
DRY_RUN=0

if [[ -t 1 ]]; then
    BLUE=$'\033[1;34m'
    RESET=$'\033[0m'
else
    BLUE=''
    RESET=''
fi

log() { printf '%s==>%s %s\n' "$BLUE" "$RESET" "$*"; }

run() {
    printf '  $'; printf ' %q' "$@"; printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    "$@"
}

usage() {
    cat <<'EOF'
用法: run.sh [选项]

  --model NAME           Hugging Face 模型 id（默认 Qwen/Qwen2.5-0.5B-Instruct）
  --port PORT            宿主机端口（默认 8000）
  --max-model-len N      最大上下文（默认 4096；越大 KV cache 占用越多）
  --gpu-mem-util F       GPU 显存上限比例（默认 0.90；CPU 模式不使用）
  --tp N                 CPU/GPU 并行度（GPU 为卡数，CPU 建议按 NUMA 节点）
  --block-size N         PagedAttention 块大小（不设则用 vLLM 默认：GPU 16 / CPU 128）
  --dtype TYPE           模型精度，如 bfloat16（CPU 上推荐显式指定）
  --timeout SEC          等待服务就绪的最长秒数（默认 600）
  --no-demo              不调用示例接口
  --cpu / --gpu          强制 CPU 或 GPU（默认自动检测 nvidia-smi）
  --dry-run              只打印将执行的 docker 命令
  -h, --help             显示本帮助

环境变量：ACCEL=auto|cpu|gpu / MODEL / PORT / MAX_MODEL_LEN / GPU_MEM_UTIL /
          TP / BLOCK_SIZE / DTYPE / HF_CACHE / HF_TOKEN / HEALTH_TIMEOUT /
          VLLM_CPU_KVCACHE_SPACE / VLLM_CPU_OMP_THREADS_BIND

CPU 模式说明：
  - 镜像 vllm/vllm-openai-cpu:latest-x86_64 需要 Linux x86_64 + AVX2（推荐 AVX512）；
  - KV cache 大小用 VLLM_CPU_KVCACHE_SPACE（GiB）控制，而不是 --gpu-mem-util；
  - 线程绑定可用 VLLM_CPU_OMP_THREADS_BIND=0-15 或 auto。

模型来源：Hugging Face（https://huggingface.co），gated 模型需要 HF_TOKEN。
镜像来源：Docker Hub 的 docker.io/vllm/vllm-openai（GPU）与
          docker.io/vllm/vllm-openai-cpu（CPU）。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "--model 需要模型 id" >&2; exit 1; }
            MODEL="$2"; shift 2 ;;
        --port)
            [[ $# -ge 2 ]] || { echo "--port 需要端口号" >&2; exit 1; }
            PORT="$2"; shift 2 ;;
        --max-model-len)
            [[ $# -ge 2 ]] || { echo "--max-model-len 需要数字" >&2; exit 1; }
            MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem-util)
            [[ $# -ge 2 ]] || { echo "--gpu-mem-util 需要比例" >&2; exit 1; }
            GPU_MEM_UTIL="$2"; shift 2 ;;
        --tp)
            [[ $# -ge 2 ]] || { echo "--tp 需要并行度" >&2; exit 1; }
            TP="$2"; shift 2 ;;
        --block-size)
            [[ $# -ge 2 ]] || { echo "--block-size 需要数字" >&2; exit 1; }
            BLOCK_SIZE="$2"; shift 2 ;;
        --dtype)
            [[ $# -ge 2 ]] || { echo "--dtype 需要类型" >&2; exit 1; }
            DTYPE="$2"; shift 2 ;;
        --timeout)
            [[ $# -ge 2 ]] || { echo "--timeout 需要秒数" >&2; exit 1; }
            HEALTH_TIMEOUT="$2"; shift 2 ;;
        --cpu) ACCEL=cpu; shift ;;
        --gpu) ACCEL=gpu; shift ;;
        --no-demo) DEMO=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "端口不合法: $PORT" >&2
    exit 1
fi
for value in "$MAX_MODEL_LEN" "$TP" "$HEALTH_TIMEOUT"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "需要正整数: $value" >&2
        exit 1
    fi
done
if [[ -n "$BLOCK_SIZE" ]] && ! { [[ "$BLOCK_SIZE" =~ ^[0-9]+$ ]] && (( BLOCK_SIZE > 0 )); }; then
    echo "--block-size 需要正整数: $BLOCK_SIZE" >&2
    exit 1
fi

# 自动选择 CPU / GPU。
case "$ACCEL" in
    auto)
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
            ACCEL=gpu
        else
            ACCEL=cpu
        fi
        ;;
    cpu|gpu) ;;
    *) echo "ACCEL 只能是 auto / cpu / gpu" >&2; exit 1 ;;
esac

if [[ "$ACCEL" == "gpu" ]]; then
    IMAGE="$GPU_IMAGE"
    log "使用 GPU（--gpus all，镜像 ${IMAGE}）"
else
    IMAGE="$CPU_IMAGE"
    log "使用 CPU（镜像 ${IMAGE}）"
fi

if [[ "$DRY_RUN" != 1 ]]; then
    command -v docker >/dev/null 2>&1 || {
        echo "未找到 docker，请先安装 Docker Desktop 或 Docker Engine。" >&2
        exit 1
    }
    docker info >/dev/null 2>&1 || {
        echo "Docker daemon 不可用，请先启动 Docker。" >&2
        exit 1
    }
    command -v curl >/dev/null 2>&1 || {
        echo "需要 curl 等待服务就绪并调用接口。" >&2
        exit 1
    }
    if [[ "$ACCEL" == "gpu" ]]; then
        command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 || {
            echo "GPU 模式需要 NVIDIA 驱动；没有 GPU 时去掉 --gpu，或加 --cpu 使用官方 CPU 镜像。" >&2
            exit 1
        }
    fi
    mkdir -p "$HF_CACHE"
fi

if [[ "$DRY_RUN" != 1 ]] && docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
    log "移除已有容器 ${CONTAINER}（HF 缓存保留在 ${HF_CACHE}）"
    docker rm -f "$CONTAINER" >/dev/null
fi

docker_args=(run -d --name "$CONTAINER"
    -p "$PORT":8000
    -v "$HF_CACHE":/root/.cache/huggingface)

if [[ "$ACCEL" == "gpu" ]]; then
    docker_args+=(--gpus all --ipc=host)
else
    # vLLM CPU 的 NUMA 调用在默认 seccomp 下可能被拒绝；官方文档建议这两个参数。
    docker_args+=(--security-opt seccomp=unconfined --cap-add SYS_NICE --shm-size=4g)
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args+=(-e HF_TOKEN)
fi
if [[ "$ACCEL" == "cpu" ]]; then
    for name in VLLM_CPU_KVCACHE_SPACE VLLM_CPU_OMP_THREADS_BIND; do
        if [[ -n "${!name:-}" ]]; then
            docker_args+=(-e "$name")
        fi
    done
fi

docker_args+=("$IMAGE")
# GPU 镜像的文档写法是 --model；CPU 镜像的入口是 `vllm serve`，文档用位置参数。
if [[ "$ACCEL" == "gpu" ]]; then
    docker_args+=(--model "$MODEL")
else
    docker_args+=("$MODEL")
fi
docker_args+=(--max-model-len "$MAX_MODEL_LEN" --tensor-parallel-size "$TP")

if [[ "$ACCEL" == "gpu" ]]; then
    docker_args+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
fi
if [[ -n "$BLOCK_SIZE" ]]; then
    docker_args+=(--block-size "$BLOCK_SIZE")
fi
if [[ -n "$DTYPE" ]]; then
    docker_args+=(--dtype "$DTYPE")
fi

log "启动 vLLM（模型：${MODEL}）"
if ! run docker "${docker_args[@]}"; then
    echo "容器启动失败，查看原因: docker logs $CONTAINER" >&2
    exit 1
fi

if [[ "$DRY_RUN" != 1 ]]; then
    log "等待 vLLM 就绪（最多 ${HEALTH_TIMEOUT}s，首次会下载模型并编译 kernel）"
    ready=0
    for ((i = 0; i < HEALTH_TIMEOUT; i += 5)); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 5
    done
    if [[ "$ready" != 1 ]]; then
        echo "启动超时。最近日志：" >&2
        docker logs --tail 40 "$CONTAINER" >&2 || true
        exit 1
    fi
    log "vLLM 已就绪"

    log "KV cache / 显存相关日志"
    docker logs "$CONTAINER" 2>&1 \
        | grep -iE "gpu kv cache|kv cache size|maximum concurrency|gpu blocks" \
        | tail -5 || true
fi

if [[ "$DEMO" == 1 && "$DRY_RUN" != 1 ]]; then
    log "调用 POST /v1/chat/completions"
    payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"用一句话解释 PagedAttention"}],"max_tokens":64,"temperature":0.7}' "$MODEL")
    response=$(curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$payload")
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$response" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'
    else
        printf '%s\n' "$response"
    fi
fi

cat <<EOF

完成。常用命令：
  OpenAI API   http://127.0.0.1:${PORT}/v1
  模型列表     curl http://127.0.0.1:${PORT}/v1/models
  健康检查     curl http://127.0.0.1:${PORT}/health
  指标         curl http://127.0.0.1:${PORT}/metrics | grep vllm:gpu_cache_usage_perc
  日志         docker logs -f ${CONTAINER}
  停止容器     docker stop ${CONTAINER}
  彻底清理     docker rm -f ${CONTAINER}

当前模式     ${ACCEL}（镜像 ${IMAGE}）
PagedAttention 的块表与 attention 读取方式见 example/PAGED_ATTENTION_READING.md；
本目录 README.md 里还有 CPU/GPU 参数与 KV cache 的对应关系。
EOF
