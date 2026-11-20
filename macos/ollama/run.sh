#!/usr/bin/env bash
# 在 macOS 上启动 Ollama（原生 Metal GPU），拉取模型并调用一次生成接口。
#
# 和 docker/ollama/run.sh 对应：那边是 Linux 容器里的 CPU 版（macOS 上 Docker
# 拿不到 Apple GPU）；这边直接跑 macOS 原生版，Apple Silicon 上会用 Metal GPU
# 和统一内存，Intel Mac 则是纯 CPU。
#
# 这个脚本在“安装 + 启动”之外补齐了日常真正需要的几步：找到 ollama 可执行文件、
# 后台启动 serve（或交给菜单栏 App）、等待就绪、模型拉取、一次 /api/generate
# 调用，以及 ollama ps 的显存/内存占用展示。每一行实际执行的命令都会打印出来。
#
# 用法：
#   bash macos/ollama/run.sh                      # 后台起 serve，默认拉 llama3.2:3b
#   MODELS="llama3.2:3b qwen2.5:3b" bash macos/ollama/run.sh    # 一次拉多个
#   MODEL=llama3.1:8b bash macos/ollama/run.sh                  # 单模型别名
#   bash macos/ollama/run.sh --model qwen2.5:3b --ctx 8192
#   bash macos/ollama/run.sh --app                # 用菜单栏 App 启动（登录项自启）
#   bash macos/ollama/run.sh --install            # 没装时先跑官方安装脚本（.app）
#   bash macos/ollama/run.sh --dry-run            # 只打印命令，不真正执行
#   bash macos/ollama/run.sh --stop               # 停掉本脚本后台起的 serve
#   bash macos/ollama/run.sh --status             # 只查看服务/模型/占用
#
# 环境变量： MODELS="a b"（空格/逗号分隔）/ MODEL（单模型别名）/
#           PORT / HOST / CTX / HEALTH_TIMEOUT / APP_MODE /
#           OLLAMA_SERVE_LOG（日志文件）/ OLLAMA_SERVE_PID_FILE（pid 文件）/
#           OLLAMA_BIN（指定 ollama 可执行文件）
#
# 模型来源：默认从 Ollama 官方模型库拉取（https://ollama.com/library），
#           name:tag 的完整坐标是 registry.ollama.ai/library/<name>:<tag>；
#           也支持 hf.co/<user>/<repo>:<quant> 这种 Hugging Face GGUF。
# 安装来源：官方 App（https://ollama.com/download/mac，dmg）或
#           install.sh（下载 Ollama-darwin.zip 解到 /Applications，并建 CLI 符号链接）。
#
# 只支持 macOS；Linux 请用 docker/ollama/run.sh，或直接跑官方 install.sh。

set -euo pipefail

APP_PATH="${OLLAMA_APP:-/Applications/Ollama.app}"
APP_BIN="$APP_PATH/Contents/Resources/ollama"
LOG_FILE="${OLLAMA_SERVE_LOG:-$HOME/.ollama/logs/ollama-serve.log}"
PID_FILE="${OLLAMA_SERVE_PID_FILE:-$HOME/.ollama/ollama-serve.pid}"
LOCK_DIR="${OLLAMA_SERVE_LOCK:-$HOME/.ollama/ollama-serve.lock}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-11434}"
CTX="${CTX:-}"
MODELS="${MODELS:-}"
MODEL="${MODEL:-}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
APP_MODE="${APP_MODE:-0}"
DRY_RUN="${DRY_RUN:-0}"
INSTALL=0
PULL="${PULL:-1}"
DEMO="${DEMO:-1}"
ACTION="start"

# 如果调用方 export 了 OLLAMA_HOST（macOS 上给 App 设环境变量时很常见），
# 例如 OLLAMA_HOST=0.0.0.0:11434，则没有 --host/--port 参数时把它作为默认值。
# 支持 http://host:port、host:port、:port；IPv6 暂不支持（明确报错，避免拼出非法 URL）。
if [[ -n "${OLLAMA_HOST:-}" ]]; then
    oh="${OLLAMA_HOST}"
    case "$oh" in
        *://*) oh="${oh#*://}" ;;
    esac
    oh="${oh%/}"
    case "$oh" in
        "") ;;
        \[*\]:*|\[*\]|*:*:*)
            echo "暂不支持 IPv6 形式的 OLLAMA_HOST=${OLLAMA_HOST}，请用 127.0.0.1 或 0.0.0.0。" >&2
            exit 1
            ;;
        :*)
            HOST="0.0.0.0"
            port_part="${oh#:}"
            [[ -n "$port_part" ]] && PORT="$port_part"
            ;;
        *:*)
            HOST="${oh%:*}"
            PORT="${oh##*:}"
            ;;
        *) HOST="$oh" ;;
    esac
fi

# 允许脚本接收任意 OLLAMA_* 环境变量并透传给 serve（例如调用方已经 export 了
# OLLAMA_KEEP_ALIVE、OLLAMA_NUM_PARALLEL、OLLAMA_KV_CACHE_TYPE 等）。
passthrough_env=()
while IFS='=' read -r name _; do
    case "$name" in
        OLLAMA_BIN|OLLAMA_APP|OLLAMA_SERVE_LOG|OLLAMA_SERVE_PID_FILE|OLLAMA_SERVE_LOCK|OLLAMA_HOST) continue ;;
        OLLAMA_*) passthrough_env+=("$name=${!name}") ;;
    esac
done < <(env)

usage() {
    cat <<'EOF'
用法: run.sh [选项]

  --model NAME      要拉取的模型，可重复；不传时用 MODELS / MODEL 环境变量
                    （默认 llama3.2:3b，也可以是 llama3.1:8b 等）
  --port PORT       服务端口（默认 11434）
  --host ADDR       监听地址（默认 127.0.0.1；0.0.0.0 表示局域网可见）
  --ctx N           默认上下文长度，对应 OLLAMA_CONTEXT_LENGTH
                    （不设时 Ollama 按显存档位选 4k/32k/256k）
  --app             用 /Applications/Ollama.app 启动（菜单栏 App，登录项自启）
  --install         没装 ollama 时先执行官方 install.sh（安装 Ollama.app）
  --no-pull         不执行 ollama pull
  --no-demo         不调用示例 /api/generate
  --timeout SEC     等待服务就绪的最长秒数（默认 120）
  --dry-run         只打印将要执行的命令
  --stop            停掉本脚本后台启动的 serve（不碰 App / brew services）
  --status          只查看服务、模型与占用，不启动
  -h, --help        显示本帮助

环境变量：
  MODELS="llama3.2:3b qwen2.5:3b"   要拉取的模型列表（空格或逗号分隔）
  MODEL=llama3.1:8b                 单个模型的别名，等价于 MODELS="llama3.1:8b"
  PORT / HOST / CTX / APP_MODE=0|1 / PULL=0|1 / DEMO=0|1 / HEALTH_TIMEOUT
  OLLAMA_SERVE_LOG / OLLAMA_SERVE_PID_FILE / OLLAMA_BIN
  其它 OLLAMA_* 变量会原样传给 `ollama serve`（见 README 的环境变量表）

优先级：--model（可重复） > MODELS > MODEL > 默认 llama3.2:3b

模型来源：默认 Ollama 官方模型库 https://ollama.com/library
          （完整坐标 registry.ollama.ai/library/<name>:<tag>）；
          也支持 hf.co/<user>/<repo>:<quant> 的 Hugging Face GGUF。
安装来源：官方 App（Ollama.dmg）或 https://ollama.com/install.sh。
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

cli_models=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "--model 需要模型名" >&2; exit 1; }
            cli_models+=("$2"); shift 2 ;;
        --port)
            [[ $# -ge 2 ]] || { echo "--port 需要端口号" >&2; exit 1; }
            PORT="$2"; shift 2 ;;
        --host)
            [[ $# -ge 2 ]] || { echo "--host 需要地址" >&2; exit 1; }
            HOST="$2"; shift 2 ;;
        --ctx)
            [[ $# -ge 2 ]] || { echo "--ctx 需要上下文长度" >&2; exit 1; }
            CTX="$2"; shift 2 ;;
        --app)      APP_MODE=1; shift ;;
        --install)  INSTALL=1; shift ;;
        --no-pull)  PULL=0; shift ;;
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

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "端口不合法: $PORT" >&2
    exit 1
fi
PORT=$((10#$PORT))
if (( PORT < 1 || PORT > 65535 )); then
    echo "端口不合法: $PORT" >&2
    exit 1
fi
if ! [[ "$HEALTH_TIMEOUT" =~ ^[0-9]+$ ]]; then
    echo "HEALTH_TIMEOUT 必须是正整数: $HEALTH_TIMEOUT" >&2
    exit 1
fi
HEALTH_TIMEOUT=$((10#$HEALTH_TIMEOUT))
if (( HEALTH_TIMEOUT < 1 )); then
    echo "HEALTH_TIMEOUT 必须是正整数: $HEALTH_TIMEOUT" >&2
    exit 1
fi
if [[ -n "$CTX" ]]; then
    if ! [[ "$CTX" =~ ^[0-9]+$ ]]; then
        echo "--ctx 必须是正整数: $CTX" >&2
        exit 1
    fi
    CTX=$((10#$CTX))
    if (( CTX < 1 )); then
        echo "--ctx 必须是正整数: $CTX" >&2
        exit 1
    fi
fi

# --app 启动的是固定监听 127.0.0.1:11434 的菜单栏 App，这几个参数传不进去。
if [[ "$APP_MODE" == 1 && ( "$PORT" != "11434" || "$HOST" != "127.0.0.1" || -n "$CTX" ) ]]; then
    echo "--app 模式不能搭配 --port/--host/--ctx：App 固定监听 127.0.0.1:11434。" >&2
    echo "若要改 App 的监听地址/上下文，用 launchctl setenv 后重启 App（见 README「环境变量」）。" >&2
    exit 1
fi

# 客户端访问地址：监听 0.0.0.0 / :: 时，本机仍然走 127.0.0.1。
CLIENT_HOST="$HOST"
case "$HOST" in
    0.0.0.0|"[::]"|"::") CLIENT_HOST="127.0.0.1" ;;
esac
ENDPOINT="http://$CLIENT_HOST:$PORT"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "这个脚本只支持 macOS。Linux 请用 docker/ollama/run.sh，或直接跑官方 install.sh。" >&2
    exit 1
fi

# 解析模型列表：--model 优先，其次 MODELS，最后 MODEL；支持空格/逗号分隔。
if (( ${#cli_models[@]} > 0 )); then
    raw_models="${cli_models[*]}"
else
    raw_models="${MODELS:-${MODEL:-llama3.2:3b}}"
fi
IFS=', ' read -r -a models <<< "$raw_models"
clean_models=()
for model_name in "${models[@]}"; do
    [[ -n "$model_name" ]] && clean_models+=("$model_name")
done
if (( ${#clean_models[@]} == 0 )); then
    echo "没有可用的模型名，请设置 MODELS 或使用 --model 指定。" >&2
    exit 1
fi
models=("${clean_models[@]}")

# 找到 ollama 可执行文件：显式 OLLAMA_BIN > PATH > /Applications/Ollama.app > Homebrew。
find_ollama() {
    if [[ -n "${OLLAMA_BIN:-}" && -x "$OLLAMA_BIN" ]]; then
        printf '%s' "$OLLAMA_BIN"; return 0
    fi
    if command -v ollama >/dev/null 2>&1; then
        command -v ollama; return 0
    fi
    if [[ -x "$APP_BIN" ]]; then
        printf '%s' "$APP_BIN"; return 0
    fi
    if command -v brew >/dev/null 2>&1; then
        brew_prefix="$(brew --prefix 2>/dev/null || true)"
        if [[ -n "$brew_prefix" && -x "$brew_prefix/bin/ollama" ]]; then
            printf '%s' "$brew_prefix/bin/ollama"; return 0
        fi
    fi
    return 1
}

# 服务是否已经在监听（任何方式启动的都算）。
server_up() {
    curl -fsS --max-time 2 "$ENDPOINT/api/version" >/dev/null 2>&1
}

# 等 /api/version 返回；超时返回非 0。
wait_ready() {
    local waited=0
    while (( waited < HEALTH_TIMEOUT )); do
        if server_up; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

show_log_tail() {
    [[ -f "$LOG_FILE" ]] || return 0
    echo "----- $LOG_FILE 最后 20 行 -----" >&2
    tail -n 20 "$LOG_FILE" >&2 || true
    echo "-------------------------------" >&2
}

# 调 CLI 时显式带上 OLLAMA_HOST，保证自定义端口下也能连到同一个服务。
run_ollama() {
    run env "OLLAMA_HOST=$CLIENT_HOST:$PORT" "$OLLAMA_BIN" "$@"
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

# pid 是否正在监听某个端口（ps 不可用或输出不明确时，用这个确认归属）。
pid_owns_port() {
    local p="$1" p_port="$2"
    [[ -n "$p" && -n "$p_port" ]] || return 1
    lsof -nP -iTCP:"$p_port" -sTCP:LISTEN -t 2>/dev/null | grep -Fxq "$p"
}

# 校验 pid 确实是 ollama 进程：命令行包含 ollama，或它正监听 pid 文件里记录的端口。
is_ollama_pid() {
    local p="$1" p_port="$2" cmd=""
    cmd="$(ps -p "$p" -ww -o command= 2>/dev/null || true)"
    case "$cmd" in
        *ollama*) return 0 ;;
    esac
    pid_owns_port "$p" "$p_port"
}

# ---- serve 启动锁：同一时间只允许一个 run.sh 启动 serve ----
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
            rm -rf "$LOCK_DIR"   # 上一个 run.sh 异常退出，清了重试
            continue
        fi
        sleep 0.2
    done
    return 1
}

stop_serve() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "没有找到本脚本后台启动的 serve（${PID_FILE}）。"
        echo "如果是菜单栏 App 或 brew services 起的，请用它们自己的方式停："
        echo "  菜单栏 Ollama 图标 -> Quit Ollama"
        echo "  brew services stop ollama"
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
            echo "注意：$ENDPOINT 仍有响应，可能是别的 Ollama 实例；用 lsof -nP -iTCP:$PORT -sTCP:LISTEN 查。" >&2
        fi
        return 0
    fi
    if ! is_ollama_pid "$pid" "${stored_port:-$PORT}"; then
        echo "拒绝停止 pid=${pid}：它不像是本脚本启动的 ollama serve。" >&2
        echo "请先确认：lsof -p ${pid}；确认无误后可手动 kill ${pid}，再删除 ${PID_FILE}。" >&2
        return 1
    fi
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "  （dry-run）将执行：kill $pid   # 本脚本启动的 serve"
        return 0
    fi
    log "停止 pid=${pid}（本脚本启动的 serve）"
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    if server_up; then
        echo "进程已结束，但 $ENDPOINT 仍有响应（可能还有别的 Ollama 在跑）。"
    else
        echo "已停止。"
    fi
}

status_serve() {
    if ! server_up; then
        echo "服务未运行（$ENDPOINT 无响应）。"
        [[ -f "$PID_FILE" ]] && echo "提示：$PID_FILE 存在，可用 --stop 清理。"
        return 1
    fi
    echo "服务地址: $ENDPOINT"
    curl -fsS --max-time 5 "$ENDPOINT/api/version" || true
    echo
    if [[ -n "${OLLAMA_BIN:-}" ]]; then
        env "OLLAMA_HOST=$CLIENT_HOST:$PORT" "$OLLAMA_BIN" list || true
        echo
        env "OLLAMA_HOST=$CLIENT_HOST:$PORT" "$OLLAMA_BIN" ps || true
    fi
}

if [[ "$ACTION" == "stop" ]]; then
    stop_serve
    exit 0
fi

if [[ "$ACTION" == "status" ]]; then
    OLLAMA_BIN="$(find_ollama || true)"
    status_serve
    exit $?
fi

# ---------------- 安装 ----------------
OLLAMA_BIN="$(find_ollama || true)"
if [[ -z "$OLLAMA_BIN" ]]; then
    if [[ "$INSTALL" == 1 ]]; then
        log "未找到 ollama，执行官方安装脚本（安装 /Applications/Ollama.app）"
        echo "  这一步会下载 Ollama.app，/usr/local/bin 符号链接可能需要输入密码。"
        if [[ "$DRY_RUN" == 1 ]]; then
            printf '  $ curl -fsSL https://ollama.com/install.sh | OLLAMA_NO_START=1 sh\n'
        else
            # 不让 install.sh 顺手启动 App，避免它和脚本接着起的 serve 抢 11434。
            run sh -c 'curl -fsSL https://ollama.com/install.sh | OLLAMA_NO_START=1 sh'
        fi
        OLLAMA_BIN="$(find_ollama || true)"
        if [[ -z "$OLLAMA_BIN" && "$DRY_RUN" != 1 ]]; then
            echo "安装后仍未找到 ollama 可执行文件，请检查 PATH 或 /Applications/Ollama.app。" >&2
            exit 1
        fi
        if [[ -z "$OLLAMA_BIN" ]]; then
            OLLAMA_BIN="ollama"
        fi
    elif [[ "$DRY_RUN" == 1 ]]; then
        OLLAMA_BIN="ollama"
    else
        echo "未找到 ollama。请先安装（任选一种）：" >&2
        echo "  1) 官网下载 App: https://ollama.com/download/mac" >&2
        echo "  2) 一键脚本:     curl -fsSL https://ollama.com/install.sh | sh" >&2
        echo "  3) Homebrew:     brew install ollama" >&2
        echo "  也可以重跑本脚本并加 --install，让脚本执行方式 2。" >&2
        exit 1
    fi
fi

log "使用 ollama: $OLLAMA_BIN"
case "$(uname -m)" in
    arm64) log "检测到 Apple Silicon：原生版会走 Metal GPU + 统一内存（Docker 版用不到 GPU）" ;;
    x86_64) log "检测到 Intel Mac：原生版是纯 CPU 推理" ;;
esac

# ---------------- 启动 ----------------
if server_up; then
    log "检测到服务已在运行：${ENDPOINT}（复用，不重复启动）"
    if [[ -n "$CTX" || "$HOST" != "127.0.0.1" ]]; then
        log "注意：已有服务不是本脚本启动的，--host/--ctx 等参数对它不生效；需要时先停掉再重启"
    fi
elif [[ "$APP_MODE" == 1 ]]; then
    log "用菜单栏 App 启动：$APP_PATH"
    if [[ ! -d "$APP_PATH" ]]; then
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "  （dry-run）没有找到 ${APP_PATH}，实际执行前需要先安装：https://ollama.com/download/mac"
        else
            echo "没有找到 ${APP_PATH}。请先安装官方 App，或去掉 --app 改用后台 serve 模式。" >&2
            exit 1
        fi
    fi
    # 官方 install.sh 用 --args hidden 启动，App 会同时起 HTTP 服务。
    if [[ "$APP_PATH" == "/Applications/Ollama.app" ]]; then
        run open -a Ollama --args hidden
    else
        run open -a "$APP_PATH" --args hidden
    fi
    if [[ "$DRY_RUN" != 1 ]]; then
        log "等待 Ollama 就绪（最多 ${HEALTH_TIMEOUT}s）"
        if ! wait_ready; then
            echo "启动超时。查看日志: $HOME/.ollama/logs/server.log" >&2
            exit 1
        fi
    fi
else
    # 先拿锁，避免两个 run.sh 同时起 serve：抢端口、互相覆盖 pid 文件。
    lock_ok=1
    if [[ "$DRY_RUN" != 1 ]]; then
        lock_ok=0
        if acquire_lock; then
            lock_ok=1
        else
            echo "另一个 run.sh 正在启动 Ollama（锁：${LOCK_DIR}），请稍后重试。" >&2
            exit 1
        fi
        if server_up; then
            # 拿锁期间别处已经起来了，直接复用。
            release_lock
            log "检测到服务已在运行：${ENDPOINT}（复用，不重复启动）"
            lock_ok=2
        fi
    fi

    if [[ "$lock_ok" == 1 ]]; then
        log "后台启动 ollama serve（日志: ${LOG_FILE}）"
        serve_env=("OLLAMA_HOST=$HOST:$PORT")
        [[ -n "$CTX" ]] && serve_env+=("OLLAMA_CONTEXT_LENGTH=$CTX")
        # 调用方显式 export 的 OLLAMA_* 变量优先级低于命令行参数：
        # OLLAMA_HOST / OLLAMA_CONTEXT_LENGTH 已由 --host/--port/--ctx 显式决定时不再透传。
        if (( ${#passthrough_env[@]} > 0 )); then
            for kv in "${passthrough_env[@]}"; do
                name="${kv%%=*}"
                [[ "$name" == "OLLAMA_HOST" ]] && continue
                [[ -n "$CTX" && "$name" == "OLLAMA_CONTEXT_LENGTH" ]] && continue
                serve_env+=("$kv")
            done
        fi

        printf '  $ nohup env'
        printf ' %q' "${serve_env[@]}" "$OLLAMA_BIN" serve
        printf ' >> %q 2>&1 &\n' "$LOG_FILE"
        if [[ "$DRY_RUN" != 1 ]]; then
            mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
            nohup env "${serve_env[@]}" "$OLLAMA_BIN" serve >>"$LOG_FILE" 2>&1 &
            serve_pid=$!
            printf '%s\n%s\n' "$serve_pid" "$PORT" > "$PID_FILE"
            log "serve pid=${serve_pid}，等待就绪（最多 ${HEALTH_TIMEOUT}s）"
            if ! wait_ready; then
                echo "启动失败或超时。" >&2
                if kill -0 "$serve_pid" 2>/dev/null; then
                    kill "$serve_pid" 2>/dev/null || true
                fi
                rm -f "$PID_FILE"
                release_lock
                show_log_tail
                echo "排查：端口是否被占用（lsof -nP -iTCP:$PORT -sTCP:LISTEN）" >&2
                exit 1
            fi
            if ! kill -0 "$serve_pid" 2>/dev/null; then
                release_lock
                if server_up; then
                    rm -f "$PID_FILE"
                    log "端口已被别的 Ollama 实例占用，复用现有服务"
                else
                    echo "serve 进程已退出但服务未就绪，查看日志：$LOG_FILE" >&2
                    exit 1
                fi
            else
                log "Ollama 已就绪"
                release_lock
            fi
        fi
    fi
fi

# ---------------- 拉模型 ----------------
if [[ "$PULL" == 1 ]]; then
    log "拉取模型：${models[*]}（首次会下载数 GB，可先用小模型试）"
    for model_name in "${models[@]}"; do
        run_ollama pull "$model_name"
    done
fi

# ---------------- 示例请求 ----------------
if [[ "$DEMO" == 1 && "$DRY_RUN" != 1 ]]; then
    command -v curl >/dev/null 2>&1 || {
        echo "缺少 curl，无法调用示例接口；可加 --no-demo 跳过。" >&2
        exit 1
    }
    log "调用 POST /api/generate（模型：${models[0]}）"
    payload=$(printf '{"model":"%s","prompt":"用一句话解释 KV cache 为什么能加速推理","stream":false}', "${models[0]}")
    response=$(curl -fsS --max-time 600 "$ENDPOINT/api/generate" \
        -H 'Content-Type: application/json' -d "$payload")
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("response", "").strip())'
    else
        printf '%s\n' "$response"
    fi

    log "ollama ps：模型和上下文占用的实际内存 / 是否跑在 GPU 上"
    run env "OLLAMA_HOST=$CLIENT_HOST:$PORT" "$OLLAMA_BIN" ps
fi

# 非默认端口时，CLI 需要 OLLAMA_HOST 指到同一个服务。
case "$(uname -m)" in
    arm64) processor_hint="（Apple Silicon 应为 100% GPU）" ;;
    *) processor_hint="（Intel Mac 为 100% CPU）" ;;
esac
cli_prefix=""
if [[ "$PORT" != "11434" || "$CLIENT_HOST" != "127.0.0.1" ]]; then
    cli_prefix="OLLAMA_HOST=$CLIENT_HOST:$PORT "
fi

cat <<EOF

完成。常用命令：
  服务地址     $ENDPOINT
  交互聊天     ${cli_prefix}ollama run ${models[0]}
  本次模型     ${models[*]}
  模型列表     ${cli_prefix}ollama list
  本次占用     ${cli_prefix}ollama ps        ${processor_hint}
  查看日志     tail -f ~/.ollama/logs/server.log   # App 起的服务
               tail -f $LOG_FILE                    # 本脚本后台起的 serve
  停止服务     bash macos/ollama/run.sh --stop
               菜单栏 Ollama 图标 -> Quit Ollama（App 模式）
               brew services stop ollama（Homebrew 模式）

更多说明（安装方式、App/brew/serve 三种启动、环境变量、KV cache 参数、
与 Docker 版的差异）见同目录 README.md。
EOF
