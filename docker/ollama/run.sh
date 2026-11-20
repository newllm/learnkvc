#!/usr/bin/env bash
# 用 Docker 起一个 Ollama 服务，拉取 Llama 模型，并调用一次生成接口。
#
# 原始的启动命令只有两行：
#   docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
#   docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
#
# 这个脚本在此基础上补齐了日常真正需要的几步：CPU/GPU 自动判断、旧容器清理、
# 就绪等待、模型拉取、一次 /api/generate 调用，以及 ollama ps 内存占用展示。
# 每一行 docker 命令都会打印出来，方便对照学习。
#
# 用法：
#   bash docker/ollama/run.sh                 # 默认拉 llama3.2:3b，自动判断 CPU/GPU
#   MODEL=llama3.1:8b bash docker/ollama/run.sh
#   bash docker/ollama/run.sh --model qwen2.5:3b --port 11434
#   bash docker/ollama/run.sh --cpu           # 强制 CPU（没有 NVIDIA GPU 的机器）
#   bash docker/ollama/run.sh --dry-run       # 只打印命令，不真正执行
#   bash docker/ollama/run.sh --no-demo       # 不跑示例请求
#
# 环境变量： MODEL / PORT / GPU_MODE(auto|on|off) / PULL / DEMO / HEALTH_TIMEOUT

set -euo pipefail

CONTAINER="${CONTAINER:-ollama}"
IMAGE="${IMAGE:-ollama/ollama}"
VOLUME="${VOLUME:-ollama}"
MODEL="${MODEL:-llama3.2:3b}"
PORT="${PORT:-11434}"
GPU_MODE="${GPU_MODE:-auto}"
PULL="${PULL:-1}"
DEMO="${DEMO:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
用法: run.sh [选项]

  --model NAME      要拉取/使用的模型（默认 llama3.2:3b，也可以是 llama3.1:8b 等）
  --port PORT       映射到宿主机的端口（默认 11434）
  --gpu             强制使用 GPU（需要 NVIDIA 驱动 + nvidia-container-toolkit）
  --cpu             强制使用 CPU
  --no-pull         不执行 ollama pull
  --no-demo         不调用示例 /api/generate
  --timeout SEC     等待服务就绪的最长秒数（默认 180）
  --dry-run         只打印将要执行的 docker 命令
  -h, --help        显示本帮助

环境变量与选项等价：MODEL, PORT, GPU_MODE=auto|on|off, PULL=0|1, DEMO=0|1
EOF
}

if [[ -t 1 ]]; then
    BLUE=$'\033[1;34m'
    RESET=$'\033[0m'
else
    BLUE=''
    RESET=''
fi

log() { printf '%s==>%s %s\n' "$BLUE" "$RESET" "$*"; }

# 统一打印并执行；--dry-run 时只打印。
run() {
    printf '  $'; printf ' %q' "$@"; printf '\n'
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "--model 需要模型名" >&2; exit 1; }
            MODEL="$2"; shift 2 ;;
        --port)
            [[ $# -ge 2 ]] || { echo "--port 需要端口号" >&2; exit 1; }
            PORT="$2"; shift 2 ;;
        --gpu)      GPU_MODE=on; shift ;;
        --cpu)      GPU_MODE=off; shift ;;
        --no-pull)  PULL=0; shift ;;
        --no-demo)  DEMO=0; shift ;;
        --timeout)
            [[ $# -ge 2 ]] || { echo "--timeout 需要秒数" >&2; exit 1; }
            HEALTH_TIMEOUT="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "端口不合法: $PORT" >&2
    exit 1
fi
if ! [[ "$HEALTH_TIMEOUT" =~ ^[0-9]+$ ]] || (( HEALTH_TIMEOUT < 1 )); then
    echo "HEALTH_TIMEOUT 必须是正整数: $HEALTH_TIMEOUT" >&2
    exit 1
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
fi

# CPU / GPU：auto 时以宿主机能否运行 nvidia-smi 判断。
want_gpu=0
case "$GPU_MODE" in
    on)  want_gpu=1 ;;
    off) want_gpu=0 ;;
    auto)
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
            want_gpu=1
        fi
        ;;
    *) echo "GPU_MODE 只能是 auto / on / off" >&2; exit 1 ;;
esac

if [[ "$want_gpu" == 1 ]]; then
    log "使用 GPU（--gpus=all）；需要 NVIDIA 驱动和 nvidia-container-toolkit"
else
    log "使用 CPU（没有 NVIDIA GPU，或显式指定了 --cpu）"
fi

# 同名容器先删掉再重建；模型数据在 volume 里，不会重复下载。
if [[ "$DRY_RUN" != 1 ]] && docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
    log "移除已有容器 ${CONTAINER}（volume ${VOLUME} 保留）"
    docker rm -f "$CONTAINER" >/dev/null
fi

docker_args=(run -d --name "$CONTAINER"
    -v "$VOLUME":/root/.ollama
    -p "$PORT":11434)
if [[ "$want_gpu" == 1 ]]; then
    docker_args+=(--gpus=all)
fi
docker_args+=("$IMAGE")

log "启动容器"
if ! run docker "${docker_args[@]}"; then
    echo "容器启动失败。若使用 GPU，请确认已安装 nvidia-container-toolkit 并重启过 Docker。" >&2
    echo "查看原因: docker logs $CONTAINER" >&2
    exit 1
fi

if [[ "$DRY_RUN" != 1 ]]; then
    log "等待 Ollama 就绪（最多 ${HEALTH_TIMEOUT}s）"
    ready=0
    for ((i = 0; i < HEALTH_TIMEOUT; i += 2)); do
        if docker exec "$CONTAINER" ollama list >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    if [[ "$ready" != 1 ]]; then
        echo "启动超时，查看日志: docker logs $CONTAINER" >&2
        exit 1
    fi
    log "Ollama 已就绪"
fi

if [[ "$PULL" == 1 ]]; then
    log "拉取模型 ${MODEL}（首次会下载数 GB，可先用小模型试）"
    run docker exec "$CONTAINER" ollama pull "$MODEL"
fi

if [[ "$DEMO" == 1 && "$DRY_RUN" != 1 ]]; then
    command -v curl >/dev/null 2>&1 || {
        echo "缺少 curl，无法调用示例接口；可加 --no-demo 跳过。" >&2
        exit 1
    }
    log "调用 POST /api/generate"
    payload=$(printf '{"model":"%s","prompt":"用一句话解释 KV cache 为什么能加速推理","stream":false}' "$MODEL")
    response=$(curl -fsS "http://127.0.0.1:$PORT/api/generate" \
        -H 'Content-Type: application/json' -d "$payload")
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("response", "").strip())'
    else
        printf '%s\n' "$response"
    fi

    log "ollama ps：模型和上下文占用的实际内存"
    docker exec "$CONTAINER" ollama ps
fi

cat <<EOF

完成。常用命令：
  服务地址     http://127.0.0.1:$PORT
  交互聊天     docker exec -it $CONTAINER ollama run $MODEL
  模型列表     docker exec $CONTAINER ollama list
  查看占用     docker exec $CONTAINER ollama ps
  停止容器     docker stop $CONTAINER
  彻底清理     docker rm -f $CONTAINER && docker volume rm $VOLUME

更多说明（GPU 前提、API、上下文与 KV cache 的关系）见同目录 README.md。
EOF
