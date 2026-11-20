#!/usr/bin/env bash
# 在 macOS 上用 llama.cpp 跑 Llama（Metal GPU），起一个 OpenAI 兼容服务并调一次接口。
#
# 和 macos/ollama/run.sh 对应：Ollama 是“模型管理 + 默认参数”的一体化方案；
# llama.cpp 是底层引擎，参数直接暴露（-c、-ngl、-fa、-ctk/-ctv、-np ...），
# 适合想对照 KV cache 行为调参、或需要最新特性/单一二进制的场景。
#
# 脚本做的事：
#   1. 找到 llama 可执行文件（llama.cpp v1.0 的 `llama` 多命令二进制，或旧版 llama-server）
#   2. 用 Hugging Face 仓库（-hf）或本地 GGUF 起 `llama serve`
#   3. 轮询 /health 等模型加载完成
#   4. 调一次 /v1/chat/completions，打印 /props 的上下文信息和 /metrics 的 KV cache 使用率
#   5. 退出前打印常用命令；--stop 可停掉本脚本起的服务
#
# 用法：
#   bash macos/llama-cpp/run.sh                      # 默认 Llama 3.2 3B Q4_K_M，端口 9931
#   bash macos/llama-cpp/run.sh --ctx 8192           # 8K 上下文（KV cache 翻倍）
#   bash macos/llama-cpp/run.sh --kv-type q8_0       # KV cache 量化，省一半，要求 -fa on
#   bash macos/llama-cpp/run.sh --ctx 65536 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 32768  # 32K 模型扩到 64K
#   bash macos/llama-cpp/run.sh --model ./model.gguf # 用本地 GGUF
#   bash macos/llama-cpp/run.sh --hf bartowski/Llama-3.1-8B-Instruct-GGUF:Q4_K_M
#   bash macos/llama-cpp/run.sh --install            # 没装时跑 llama.app 安装脚本
#   bash macos/llama-cpp/run.sh --dry-run            # 只打印命令
#   bash macos/llama-cpp/run.sh --stop               # 停掉本脚本后台起的服务
#   bash macos/llama-cpp/run.sh --status             # 查看服务/上下文/KV 使用率
#
# 环境变量：MODEL（本地 GGUF 路径）/ HF_REPO / CTX / NGL / PORT / HOST /
#           KV_TYPE / FA / ALIAS / HEALTH_TIMEOUT / LLAMA_BIN /
#           LLAMA_SERVE_LOG / LLAMA_SERVE_PID_FILE / LLAMA_SERVE_LOCK
#
# 模型来源：默认 Hugging Face 的 GGUF 仓库，llama.cpp 会自动下载到
#           ~/Library/Caches/llama.cpp（可用 LLAMA_CACHE 或 HF_ENDPOINT 改）；
#           也可以指定本地 .gguf 文件。
# 安装来源：https://llama.app/install.sh（Apple Silicon 预编译 Metal 版，
#           装到 ~/.local/bin/llama）或 Homebrew（brew install llama.cpp）。
#
# 只支持 macOS；Linux 上同样的命令可以直接用，但本脚本不做适配。

set -euo pipefail

DEFAULT_BIN="${HOME}/.local/bin/llama"
LOG_FILE="${LLAMA_SERVE_LOG:-$HOME/.llama-cpp/llama-serve.log}"
PID_FILE="${LLAMA_SERVE_PID_FILE:-$HOME/.llama-cpp/llama-serve.pid}"
LOCK_DIR="${LLAMA_SERVE_LOCK:-$HOME/.llama-cpp/llama-serve.lock}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9931}"
CTX="${CTX:-4096}"
NGL="${NGL:-auto}"
KV_TYPE="${KV_TYPE:-}"
FA="${FA:-auto}"
ROPE_SCALING="${ROPE_SCALING:-}"
ROPE_SCALE="${ROPE_SCALE:-}"
YARN_ORIG_CTX="${YARN_ORIG_CTX:-}"
MODEL="${MODEL:-}"
HF_REPO="${HF_REPO:-bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M}"
ALIAS="${ALIAS:-llama3.2-3b}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
DEMO="${DEMO:-1}"
DRY_RUN="${DRY_RUN:-0}"
INSTALL=0
ACTION="start"

# 允许调用方 export 的 LLAMA_* 变量透传给 serve（例如 LLAMA_ARG_*、LLAMA_CACHE）。
passthrough_env=()
while IFS='=' read -r name _; do
    case "$name" in
        LLAMA_BIN|LLAMA_SERVE_LOG|LLAMA_SERVE_PID_FILE|LLAMA_SERVE_LOCK) continue ;;
        LLAMA_*) passthrough_env+=("$name=${!name}") ;;
    esac
done < <(env)

usage() {
    cat <<'EOF'
用法: run.sh [选项]

  --model PATH      本地 GGUF 模型路径（优先于 --hf）
  --hf REPO[:QUANT] Hugging Face 仓库，quant 可省略（默认按 Q4_K_M 找）
                    默认：bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
  --ctx N           上下文长度 -c（默认 4096；0 = 模型训练时的长度，通常很大）
  --ngl N|auto|all  卸载到 GPU 的层数 -ngl（默认 auto）
  --kv-type TYPE    KV cache 量化类型，同时设置 -ctk/-ctv（如 q8_0、q4_0；需配合 -fa on）
  --fa MODE         FlashAttention，-fa on|off|auto（默认 auto）
  --rope-scaling M  RoPE 缩放方式：none|linear|yarn（模型原生窗口不够时用）
  --rope-scale N    RoPE 缩放倍数，如 2 表示把 32K 扩到 64K
  --yarn-orig-ctx N YaRN 的原始上下文长度（如 32768）
  --alias NAME      API 里的模型别名（默认 llama3.2-3b）
  --port PORT       服务端口（默认 9931，llama.cpp v1 的默认值）
  --host ADDR       监听地址（默认 127.0.0.1）
  --install         没装 llama 时先跑 https://llama.app/install.sh
  --no-demo         不调用示例接口
  --timeout SEC     等待 /health 就绪的最长秒数（默认 900，首次含模型下载）
  --dry-run         只打印将要执行的命令
  --stop            停掉本脚本后台启动的 serve（不碰 Ollama/其他实例）
  --status          只查看服务、上下文与 KV cache 使用率
  -h, --help        显示本帮助

环境变量：
  MODEL / HF_REPO / CTX / NGL / KV_TYPE / FA / ALIAS / PORT / HOST
  ROPE_SCALING / ROPE_SCALE / YARN_ORIG_CTX
  HEALTH_TIMEOUT / DEMO=0|1 / LLAMA_BIN / LLAMA_SERVE_LOG / LLAMA_SERVE_PID_FILE
  其它 LLAMA_* 变量会原样传给 serve（如 LLAMA_CACHE、LLAMA_ARG_THREADS）

默认模型是 Llama 3.2 3B（Q4_K_M，约 2 GB），首次启动会从 Hugging Face 下载。
国内网络可设 HF_ENDPOINT=https://hf-mirror.com 加速。
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
            [[ $# -ge 2 ]] || { echo "--model 需要文件路径" >&2; exit 1; }
            MODEL="$2"; shift 2 ;;
        --hf)
            [[ $# -ge 2 ]] || { echo "--hf 需要仓库名" >&2; exit 1; }
            HF_REPO="$2"; shift 2 ;;
        --ctx)
            [[ $# -ge 2 ]] || { echo "--ctx 需要数值" >&2; exit 1; }
            CTX="$2"; shift 2 ;;
        --ngl)
            [[ $# -ge 2 ]] || { echo "--ngl 需要数值/auto/all" >&2; exit 1; }
            NGL="$2"; shift 2 ;;
        --kv-type)
            [[ $# -ge 2 ]] || { echo "--kv-type 需要类型" >&2; exit 1; }
            KV_TYPE="$2"; shift 2 ;;
        --fa)
            [[ $# -ge 2 ]] || { echo "--fa 需要 on/off/auto" >&2; exit 1; }
            FA="$2"; shift 2 ;;
        --rope-scaling)
            [[ $# -ge 2 ]] || { echo "--rope-scaling 需要 none/linear/yarn" >&2; exit 1; }
            ROPE_SCALING="$2"; shift 2 ;;
        --rope-scale)
            [[ $# -ge 2 ]] || { echo "--rope-scale 需要数值" >&2; exit 1; }
            ROPE_SCALE="$2"; shift 2 ;;
        --yarn-orig-ctx)
            [[ $# -ge 2 ]] || { echo "--yarn-orig-ctx 需要数值" >&2; exit 1; }
            YARN_ORIG_CTX="$2"; shift 2 ;;
        --alias)
            [[ $# -ge 2 ]] || { echo "--alias 需要名字" >&2; exit 1; }
            ALIAS="$2"; shift 2 ;;
        --port)
            [[ $# -ge 2 ]] || { echo "--port 需要端口号" >&2; exit 1; }
            PORT="$2"; shift 2 ;;
        --host)
            [[ $# -ge 2 ]] || { echo "--host 需要地址" >&2; exit 1; }
            HOST="$2"; shift 2 ;;
        --install)  INSTALL=1; shift ;;
        --no-demo)  DEMO=0; shift ;;
        --timeout)
            [[ $# -ge 2 ]] || { echo "--timeout 需要秒数" >&2; exit 1; }
            HEALTH_TIMEOUT="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --stop)     ACTION=stop; shift ;;
        --status)   ACTION=status; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "这个脚本按 macOS 使用体验编写；Linux 上可以手动执行同样的 llama serve 命令。" >&2
    exit 1
fi

# 数字校验：先确认是十进制串，再强制十进制（避免 08 被当成八进制）。
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then echo "端口不合法: $PORT" >&2; exit 1; fi
PORT=$((10#$PORT))
if (( PORT < 1 || PORT > 65535 )); then echo "端口不合法: $PORT" >&2; exit 1; fi
if ! [[ "$CTX" =~ ^[0-9]+$ ]]; then echo "--ctx 必须是非负整数: $CTX" >&2; exit 1; fi
CTX=$((10#$CTX))
if ! [[ "$HEALTH_TIMEOUT" =~ ^[0-9]+$ ]]; then echo "HEALTH_TIMEOUT 必须是正整数: $HEALTH_TIMEOUT" >&2; exit 1; fi
HEALTH_TIMEOUT=$((10#$HEALTH_TIMEOUT))
if (( HEALTH_TIMEOUT < 1 )); then echo "HEALTH_TIMEOUT 必须是正整数: $HEALTH_TIMEOUT" >&2; exit 1; fi
case "$FA" in on|off|auto) ;; *) echo "--fa 只能是 on/off/auto" >&2; exit 1 ;; esac
case "$ROPE_SCALING" in ""|none|linear|yarn) ;; *) echo "--rope-scaling 只能是 none/linear/yarn" >&2; exit 1 ;; esac
if [[ -n "$ROPE_SCALE" ]] && ! [[ "$ROPE_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--rope-scale 需要正数，如 2 或 2.0" >&2; exit 1
fi
if [[ -n "$YARN_ORIG_CTX" ]] && ! [[ "$YARN_ORIG_CTX" =~ ^[0-9]+$ ]]; then
    echo "--yarn-orig-ctx 需要正整数" >&2; exit 1
fi
case "$NGL" in auto|all) ;; *) ! [[ "$NGL" =~ ^[0-9]+$ ]] && { echo "--ngl 只能是数字/auto/all" >&2; exit 1; } ;; esac
if [[ -n "$KV_TYPE" ]]; then
    case "$KV_TYPE" in
        f16|q8_0|q4_0|q4_1|q5_0|q5_1|bf16|f32|iq4_nl|f8_e4m3) ;;
        *) echo "--kv-type 不支持: $KV_TYPE" >&2; exit 1 ;;
    esac
fi

# 监听 0.0.0.0 / :: 时，本机仍然走 127.0.0.1。
CLIENT_HOST="$HOST"
case "$HOST" in
    0.0.0.0|"[::]"|"::") CLIENT_HOST="127.0.0.1" ;;
esac
ENDPOINT="http://$CLIENT_HOST:$PORT"

# 找 llama 可执行文件：显式 LLAMA_BIN > PATH > ~/.local/bin > Homebrew > 旧版 llama-server。
find_llama() {
    if [[ -n "${LLAMA_BIN:-}" && -x "$LLAMA_BIN" ]]; then
        printf '%s' "$LLAMA_BIN"; return 0
    fi
    if command -v llama >/dev/null 2>&1; then
        command -v llama; return 0
    fi
    if [[ -x "$DEFAULT_BIN" ]]; then
        printf '%s' "$DEFAULT_BIN"; return 0
    fi
    if command -v brew >/dev/null 2>&1; then
        brew_prefix="$(brew --prefix 2>/dev/null || true)"
        if [[ -n "$brew_prefix" && -x "$brew_prefix/bin/llama" ]]; then
            printf '%s' "$brew_prefix/bin/llama"; return 0
        fi
    fi
    # 旧版 llama.cpp 没有统一的 llama 二进制，只有 llama-server。
    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server; return 0
    fi
    return 1
}

# llama 是 v1.0 的多命令二进制还是旧版 llama-server。
llama_is_multi() {
    local help_out
    help_out="$("$1" --help 2>&1 || true)"
    case "$help_out" in
        *"Available commands"*) return 0 ;;
        *) return 1 ;;
    esac
}

http_code() {
    curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$1" 2>/dev/null || true
}

server_ready() {
    [[ "$(http_code "$ENDPOINT/health")" == "200" ]]
}

# 服务在监听（无论是否加载完成：加载中 /health 返回 503）。
server_up() {
    local code
    code="$(http_code "$ENDPOINT/health")"
    [[ "$code" == "200" || "$code" == "503" ]]
}

wait_ready() {
    local waited=0 code
    while (( waited < HEALTH_TIMEOUT )); do
        code="$(http_code "$ENDPOINT/health")"
        if [[ "$code" == "200" ]]; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        if (( waited % 15 == 0 )); then
            log "仍在等待模型下载/加载…（${waited}s，/health=${code:-无响应}）"
        fi
    done
    return 1
}

show_log_tail() {
    [[ -f "$LOG_FILE" ]] || return 0
    echo "----- $LOG_FILE 最后 30 行 -----" >&2
    tail -n 30 "$LOG_FILE" >&2 || true
    echo "-------------------------------" >&2
}

# pid 文件格式：第一行 pid，第二行启动时使用的端口。
read_pid_file() {
    pid=""
    stored_port=""
    if [[ -f "$PID_FILE" ]]; then
        pid="$(sed -n '1p' "$PID_FILE" 2>/dev/null || true)"
        stored_port="$(sed -n '2p' "$PID_FILE" 2>/dev/null || true)"
    fi
}

pid_owns_port() {
    local p="$1" p_port="$2"
    [[ -n "$p" && -n "$p_port" ]] || return 1
    lsof -nP -iTCP:"$p_port" -sTCP:LISTEN -t 2>/dev/null | grep -Fxq "$p"
}

is_llama_pid() {
    local p="$1" p_port="$2" cmd=""
    cmd="$(ps -p "$p" -ww -o command= 2>/dev/null || true)"
    case "$cmd" in
        *llama*) return 0 ;;
    esac
    pid_owns_port "$p" "$p_port"
}

LOCK_HELD=0
release_lock() {
    if [[ "$LOCK_HELD" == 1 ]]; then
        rm -rf "$LOCK_DIR"
        LOCK_HELD=0
    fi
}
trap release_lock EXIT INT TERM

acquire_lock() {
    local i lock_pid
    for ((i = 0; i < 50; i += 1)); do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            printf '%s\n' "$$" > "$LOCK_DIR/pid" 2>/dev/null || true
            LOCK_HELD=1
            return 0
        fi
        lock_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
        if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
            rm -rf "$LOCK_DIR"
            continue
        fi
        sleep 0.2
    done
    return 1
}

stop_serve() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "没有找到本脚本后台启动的 serve（${PID_FILE}）。"
        echo "如果服务是手动在前台跑的，直接在对应终端 Ctrl+C。"
        return 0
    fi
    read_pid_file
    if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
        log "pid 文件内容无效，直接清理"
        rm -f "$PID_FILE"
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        log "pid=$pid 已不存在，清理 pid 文件"
        rm -f "$PID_FILE"
        if server_up; then
            echo "注意：$ENDPOINT 仍有响应，可能是别的 llama.cpp 实例；用 lsof -nP -iTCP:$PORT -sTCP:LISTEN 查。" >&2
        fi
        return 0
    fi
    if ! is_llama_pid "$pid" "${stored_port:-$PORT}"; then
        echo "拒绝停止 pid=${pid}：它不像是本脚本启动的 llama.cpp 进程。" >&2
        echo "请先确认：lsof -p ${pid}；确认无误后可手动 kill ${pid}，再删除 ${PID_FILE}。" >&2
        return 1
    fi
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "  （dry-run）将执行：kill $pid   # 本脚本启动的 llama serve"
        return 0
    fi
    log "停止 pid=${pid}（本脚本启动的 llama serve）"
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    if server_up; then
        echo "进程已结束，但 $ENDPOINT 仍有响应（可能还有别的实例在跑）。"
    else
        echo "已停止。"
    fi
}

# 打印 /props 里的上下文信息和 /metrics 里的吞吐指标。
show_runtime_info() {
    local props
    props="$(curl -fsS --max-time 5 "$ENDPOINT/props" 2>/dev/null || true)"
    if [[ -n "$props" ]] && command -v python3 >/dev/null 2>&1; then
        printf '%s' "$props" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit
g = d.get("default_generation_settings", {}) or {}
print("  n_ctx（实际上下文）: %s" % g.get("n_ctx", "?"))
print("  total_slots（并行槽）: %s" % d.get("total_slots", "?"))
print("  model_path: %s" % d.get("model_path", "?"))
print("  model_alias: %s" % d.get("model_alias", "?"))
' || true
    fi
    local metrics
    metrics="$(curl -fsS --max-time 5 "$ENDPOINT/metrics" 2>/dev/null || true)"
    if [[ -n "$metrics" ]]; then
        echo "  吞吐指标（最近窗口）："
        printf '%s\n' "$metrics" | grep -E "^llamacpp:(prompt_tokens_seconds|predicted_tokens_seconds|requests_processing)" | sed 's/^/    /' || true
    fi
}

status_serve() {
    if ! server_up; then
        echo "服务未运行（$ENDPOINT 无响应，或模型仍在加载）。"
        [[ -f "$PID_FILE" ]] && echo "提示：$PID_FILE 存在，可用 --stop 清理。"
        return 1
    fi
    echo "服务地址: $ENDPOINT"
    echo "健康检查: $(http_code "$ENDPOINT/health")（200 = 就绪，503 = 加载中）"
    show_runtime_info
    echo "  模型列表: $(curl -fsS --max-time 5 "$ENDPOINT/v1/models" 2>/dev/null | head -c 300 || true)"
}

if [[ "$ACTION" == "stop" ]]; then
    stop_serve
    exit 0
fi

if [[ "$ACTION" == "status" ]]; then
    LLAMA_BIN="$(find_llama || true)"
    status_serve
    exit $?
fi

# ---------------- 安装 ----------------
LLAMA_BIN="$(find_llama || true)"
if [[ -z "$LLAMA_BIN" ]]; then
    if [[ "$INSTALL" == 1 ]]; then
        ARCH="$(uname -m)"
        if [[ "$ARCH" == "x86_64" ]]; then
            echo "llama.app 只提供 Apple Silicon 的预编译包；Intel Mac 请用：" >&2
            echo "  brew install llama.cpp   # 可能从源码编译" >&2
            echo "或按 https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md 自行构建。" >&2
            exit 1
        fi
        log "未找到 llama，执行官方安装脚本（装到 ~/.local/bin/llama）"
        echo "  这一步会下载预编译的 Metal 版本；如果 ~/.local/bin 不在 PATH，脚本会提示。"
        if [[ "$DRY_RUN" == 1 ]]; then
            printf '  $ curl -LsSf https://llama.app/install.sh | sh\n'
        else
            run sh -c 'curl -LsSf https://llama.app/install.sh | sh'
        fi
        LLAMA_BIN="$(find_llama || true)"
        if [[ -z "$LLAMA_BIN" && "$DRY_RUN" != 1 ]]; then
            echo "安装后仍未找到 llama，请确认 ~/.local/bin 在 PATH 里，或手动指定 LLAMA_BIN。" >&2
            exit 1
        fi
        [[ -z "$LLAMA_BIN" ]] && LLAMA_BIN="llama"
    elif [[ "$DRY_RUN" == 1 ]]; then
        LLAMA_BIN="llama"
    else
        echo "未找到 llama。请先安装（任选一种）：" >&2
        echo "  1) 官方脚本（Apple Silicon，推荐）:" >&2
        echo "     curl -LsSf https://llama.app/install.sh | sh" >&2
        echo "  2) Homebrew: brew install llama.cpp" >&2
        echo "  3) 源码构建: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md" >&2
        echo "  也可以重跑本脚本并加 --install，让脚本执行方式 1。" >&2
        exit 1
    fi
fi

log "使用 llama: $LLAMA_BIN"
LLAMA_MULTI=1
if [[ -x "$LLAMA_BIN" ]] && ! llama_is_multi "$LLAMA_BIN"; then
    LLAMA_MULTI=0
    log "检测到旧版 llama-server；脚本会用顶层参数启动（交互命令为 llama-cli）"
fi
case "$(uname -m)" in
    arm64) log "检测到 Apple Silicon：Metal GPU + 统一内存，-ngl 默认 auto 会自动卸载能放下的层" ;;
    x86_64) log "检测到 Intel Mac：llama.cpp 只能用 CPU（-ngl 0）；建议换用 Apple Silicon 机器或降低模型规模" ;;
esac

# ---------------- 启动 ----------------
if server_up; then
    if [[ "$DRY_RUN" != 1 ]] && ! server_ready; then
        log "检测到服务正在加载模型（/health=503），等待就绪（最多 ${HEALTH_TIMEOUT}s）"
        if ! wait_ready; then
            echo "等待现有服务就绪超时；可查看它的日志，或先停掉再启动。" >&2
            exit 1
        fi
    fi
    log "检测到服务已在运行：${ENDPOINT}（复用，不重复启动）"
    if [[ "$HOST" != "127.0.0.1" || -n "$KV_TYPE" ]]; then
        log "注意：已有服务不是本脚本启动的，--host/--kv-type 等参数对它不生效"
    fi
elif [[ "$DRY_RUN" == 1 ]]; then
    LOCK_OK=1
else
    if ! acquire_lock; then
        echo "另一个 run.sh 正在启动 llama.cpp（锁：${LOCK_DIR}），请稍后重试。" >&2
        exit 1
    fi
    if server_up; then
        release_lock
        log "检测到服务已在运行：${ENDPOINT}（复用，不重复启动）"
        LOCK_OK=2
    else
        LOCK_OK=1
    fi
fi

if [[ "${LOCK_OK:-0}" == "1" ]]; then
    model_args=()
    if [[ -n "$MODEL" ]]; then
        if [[ "$DRY_RUN" != 1 && ! -f "$MODEL" ]]; then
            echo "本地模型不存在: $MODEL" >&2
            exit 1
        fi
        model_args=(-m "$MODEL")
        log "模型：本地文件 $MODEL"
    else
        model_args=(-hf "$HF_REPO")
        log "模型：${HF_REPO}（首次会从 Hugging Face 下载到 ~/Library/Caches/llama.cpp）"
    fi

    # 量化 KV cache 依赖 FlashAttention；auto 时显式打开，避免 -ctk/-ctv 不生效。
    if [[ -n "$KV_TYPE" && "$FA" == "auto" ]]; then
        FA=on
        log "设置 --kv-type 后需要 FlashAttention，已把 -fa auto 改为 on"
    fi
    serve_args=("${model_args[@]}" -c "$CTX" -ngl "$NGL" -fa "$FA" -a "$ALIAS"
        --host "$HOST" --port "$PORT" --metrics)
    [[ -n "$KV_TYPE" ]] && serve_args+=(-ctk "$KV_TYPE" -ctv "$KV_TYPE")
    [[ -n "$ROPE_SCALING" ]] && serve_args+=(--rope-scaling "$ROPE_SCALING")
    [[ -n "$ROPE_SCALE" ]] && serve_args+=(--rope-scale "$ROPE_SCALE")
    [[ -n "$YARN_ORIG_CTX" ]] && serve_args+=(--yarn-orig-ctx "$YARN_ORIG_CTX")

    # v1.0 的 llama 是多命令二进制（llama serve ...）；旧版只有 llama-server。
    if [[ "$LLAMA_MULTI" == 1 ]]; then
        serve_prefix=("$LLAMA_BIN" serve)
    else
        serve_prefix=("$LLAMA_BIN")
    fi

    log "后台启动 llama serve（日志: ${LOG_FILE}）"
    printf '  $ nohup'
    if (( ${#passthrough_env[@]} > 0 )); then
        printf ' env'
        printf ' %q' "${passthrough_env[@]}"
    fi
    printf ' %q' "${serve_prefix[@]}" "${serve_args[@]}"
    printf ' >> %q 2>&1 &\n' "$LOG_FILE"
    if [[ "$DRY_RUN" != 1 ]]; then
        mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
        if (( ${#passthrough_env[@]} > 0 )); then
            nohup env "${passthrough_env[@]}" "${serve_prefix[@]}" "${serve_args[@]}" >>"$LOG_FILE" 2>&1 &
        else
            nohup "${serve_prefix[@]}" "${serve_args[@]}" >>"$LOG_FILE" 2>&1 &
        fi
        serve_pid=$!
        printf '%s\n%s\n' "$serve_pid" "$PORT" > "$PID_FILE"
        log "serve pid=${serve_pid}，等待 /health = 200（最多 ${HEALTH_TIMEOUT}s；首次含模型下载）"
        if ! wait_ready; then
            echo "启动失败或超时。" >&2
            if kill -0 "$serve_pid" 2>/dev/null; then
                kill "$serve_pid" 2>/dev/null || true
            fi
            rm -f "$PID_FILE"
            release_lock
            show_log_tail
            echo "排查：端口占用（lsof -nP -iTCP:$PORT -sTCP:LISTEN）；模型下载失败时尝试 HF_ENDPOINT=https://hf-mirror.com。" >&2
            exit 1
        fi
        if ! kill -0 "$serve_pid" 2>/dev/null; then
            release_lock
            if server_up; then
                rm -f "$PID_FILE"
                log "端口已被别的 llama.cpp 实例占用，复用现有服务"
            else
                echo "serve 进程已退出但服务未就绪，查看日志：$LOG_FILE" >&2
                exit 1
            fi
        else
            log "llama.cpp 已就绪"
            release_lock
        fi
    fi
fi

# ---------------- 示例请求 ----------------
if [[ "$DEMO" == 1 && "$DRY_RUN" != 1 ]]; then
    command -v curl >/dev/null 2>&1 || {
        echo "缺少 curl，无法调用示例接口；可加 --no-demo 跳过。" >&2
        exit 1
    }
    log "调用 POST /v1/chat/completions（model=${ALIAS}）"
    payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"用一句话解释 KV cache 为什么能加速推理"}],"max_tokens":128,"stream":false}', "$ALIAS")
    response=$(curl -fsS --max-time 600 "$ENDPOINT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$payload")
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'
    else
        printf '%s\n' "$response"
    fi
    log "运行时信息（/props 与 /metrics）"
    show_runtime_info
fi

if [[ -n "$MODEL" ]]; then
    cli_model_arg="-m $MODEL"
else
    cli_model_arg="-hf $HF_REPO"
fi
if [[ "$LLAMA_MULTI" == 1 ]]; then
    cli_hint="llama cli $cli_model_arg"
else
    cli_hint="llama-cli $cli_model_arg"
fi

cat <<EOF

完成。常用命令：
  服务地址     $ENDPOINT            （浏览器打开是内置 Web UI）
  OpenAI 兼容  $ENDPOINT/v1
  交互聊天     $cli_hint
  健康检查     curl -s $ENDPOINT/health
  上下文信息   curl -s $ENDPOINT/props
  吞吐指标     curl -s $ENDPOINT/metrics | grep tokens_seconds
  查看日志     tail -f $LOG_FILE
  停止服务     bash macos/llama-cpp/run.sh --stop

参数提示：
  -c ${CTX}（上下文）直接决定 KV cache 大小；调大前先看内存。
  本机可用的 Metal 设备与卸载层数可以看日志开头的 "ggml_metal" / "offloaded N/M layers"。

更多说明（安装、模型来源、参数表、API、KV cache 与调优）见同目录 README.md。
EOF
