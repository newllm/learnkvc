#!/usr/bin/env bash
# 用 Docker 起一个 vLLM OpenAI 兼容服务，并用 PagedAttention 跑一次推理。
#
# vLLM 把 KV cache 放在固定大小的 block 池里（默认 16 token 一块），每个请求
# 用 block table 索引；这个脚本只负责把服务拉起来并验证接口：
#   镜像来源：Docker Hub 的 vllm/vllm-openai
#   模型来源：Hugging Face（默认 Qwen/Qwen2.5-0.5B-Instruct）
#
# 用法：
#   bash docker/vllm/run.sh
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct bash docker/vllm/run.sh
#   bash docker/vllm/run.sh --model meta-llama/Llama-3.2-3B-Instruct
#   MAX_MODEL_LEN=8192 bash docker/vllm/run.sh
#   bash docker/vllm/run.sh --dry-run
#
# 环境变量：
#   MODEL / PORT / MAX_MODEL_LEN / GPU_MEM_UTIL / TP / BLOCK_SIZE /
#   HF_CACHE / HF_TOKEN / HEALTH_TIMEOUT

set -euo pipefail

CONTAINER="${CONTAINER:-vllm}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TP="${TP:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
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
  --gpu-mem-util F       显存使用上限比例（默认 0.90）
  --tp N                 张量并行卡数（默认 1）
  --block-size N         PagedAttention 块大小（默认 16，常用 8/16/32）
  --timeout SEC          等待服务就绪的最长秒数（默认 600）
  --no-demo              不调用示例接口
  --dry-run              只打印将执行的 docker 命令
  -h, --help             显示本帮助

环境变量：MODEL / PORT / MAX_MODEL_LEN / GPU_MEM_UTIL / TP / BLOCK_SIZE /
          HF_CACHE / HF_TOKEN / HEALTH_TIMEOUT

模型来源：Hugging Face（https://huggingface.co），gated 模型需要 HF_TOKEN。
镜像来源：Docker Hub 的 docker.io/vllm/vllm-openai。
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
            [[ $# -ge 2 ]] || { echo "--tp 需要卡数" >&2; exit 1; }
            TP="$2"; shift 2 ;;
        --block-size)
            [[ $# -ge 2 ]] || { echo "--block-size 需要数字" >&2; exit 1; }
            BLOCK_SIZE="$2"; shift 2 ;;
        --timeout)
            [[ $# -ge 2 ]] || { echo "--timeout 需要秒数" >&2; exit 1; }
            HEALTH_TIMEOUT="$2"; shift 2 ;;
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
for value in "$MAX_MODEL_LEN" "$TP" "$BLOCK_SIZE" "$HEALTH_TIMEOUT"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "需要正整数: $value" >&2
        exit 1
    fi
done

if [[ "$DRY_RUN" != 1 ]]; then
    command -v docker >/dev/null 2>&1 || {
        echo "未找到 docker，请先安装 Docker Desktop 或 Docker Engine。" >&2
        exit 1
    }
    docker info >/dev/null 2>&1 || {
        echo "Docker daemon 不可用，请先启动 Docker。" >&2
        exit 1
    }
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 || {
        echo "vllm/vllm-openai 需要 NVIDIA GPU；没有 GPU 时请改用 docker/ollama 示例。" >&2
        exit 1
    }
    command -v curl >/dev/null 2>&1 || {
        echo "需要 curl 等待服务就绪并调用接口。" >&2
        exit 1
    }
    mkdir -p "$HF_CACHE"
fi

if [[ "$DRY_RUN" != 1 ]] && docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
    log "移除已有容器 ${CONTAINER}（HF 缓存保留在 ${HF_CACHE}）"
    docker rm -f "$CONTAINER" >/dev/null
fi

docker_args=(run -d --name "$CONTAINER"
    --gpus all
    --ipc=host
    -p "$PORT":8000
    -v "$HF_CACHE":/root/.cache/huggingface)
if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args+=(-e HF_TOKEN)
fi
docker_args+=("$IMAGE"
    --model "$MODEL"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --tensor-parallel-size "$TP"
    --block-size "$BLOCK_SIZE")

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

PagedAttention 的块表与 attention 读取方式见 example/PAGED_ATTENTION_READING.md；
本目录 README.md 里还有 GPU 前提、参数和 KV cache 的对应关系。
EOF
