#!/usr/bin/env bash
# aitrade 开发服务管理脚本
# 用法:
#   ./dev.sh                    # 交互式菜单
#   ./dev.sh start backend      # 启动后端
#   ./dev.sh stop frontend      # 停止前端
#   ./dev.sh restart all        # 重启前后端
#   ./dev.sh restart-backend    # 重启后端
#   ./dev.sh restart-frontend   # 重启前端
#   ./dev.sh status             # 查看状态
#   ./dev.sh logs backend       # 查看后端日志

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"
RUN_DIR="${ROOT_DIR}/.run"
LOG_DIR="${RUN_DIR}/logs"

BACKEND_PID_FILE="${RUN_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUN_DIR}/frontend.pid"
BACKEND_LOG="${LOG_DIR}/backend.log"
FRONTEND_LOG="${LOG_DIR}/frontend.log"

BACKEND_PORT="${AITRADE_PORT:-8000}"
BACKEND_HOST="${AITRADE_HOST:-127.0.0.1}"
FRONTEND_PORT="${AITRADE_FRONTEND_PORT:-3000}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${BLUE}[信息]${NC} $*"; }
ok()    { echo -e "${GREEN}[成功]${NC} $*"; }
warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
error() { echo -e "${RED}[错误]${NC} $*" >&2; }

ensure_dirs() {
    mkdir -p "${RUN_DIR}" "${LOG_DIR}"
}

# 检查 PID 对应进程是否存活
is_pid_alive() {
    local pid="$1"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# 根据端口查找监听进程 PID
pid_on_port() {
    local port="$1"
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | head -n1
}

# 读取 PID 文件
read_pid() {
    local pid_file="$1"
    if [[ -f "${pid_file}" ]]; then
        cat "${pid_file}"
    fi
}

# 停止进程（含子进程）
kill_process() {
    local pid="$1"
    if ! is_pid_alive "${pid}"; then
        return 0
    fi
    # 先尝试优雅终止
    kill "${pid}" 2>/dev/null || true
    local i
    for i in {1..10}; do
        if ! is_pid_alive "${pid}"; then
            return 0
        fi
        sleep 0.3
    done
    # 强制终止进程组
    kill -9 "${pid}" 2>/dev/null || true
    # 清理可能残留的子进程
    pkill -P "${pid}" 2>/dev/null || true
}

# 按端口强制停止
kill_port() {
    local port="$1"
    local pids
    pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "${pids}" | xargs kill -9 2>/dev/null || true
    fi
}

# 检查 uv 是否可用
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# 检查 .venv 是否被其他 uv 进程占用
is_uv_venv_locked() {
    local lock_file="${BACKEND_DIR}/.venv/.lock"
    [[ -f "${lock_file}" ]] || return 1
    local holders
    holders="$(uv_venv_lock_pids)"
    [[ -n "${holders}" ]]
}

# 获取占用 .venv 锁的进程 PID（空格分隔）
uv_venv_lock_pids() {
    local lock_file="${BACKEND_DIR}/.venv/.lock"
    lsof -t "${lock_file}" 2>/dev/null | tr '\n' ' ' | sed 's/ $//' || true
}

# 等待 .venv 锁释放
wait_uv_lock_release() {
    local max_wait="${UV_LOCK_WAIT_TIMEOUT:-30}"
    local i
    if ! is_uv_venv_locked; then
        return 0
    fi
    warn "backend/.venv 正被占用 (PID: $(uv_venv_lock_pids))，等待释放 (最多 ${max_wait}s)..."
    for ((i=1; i<=max_wait; i++)); do
        if ! is_uv_venv_locked; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# 强制释放 .venv 锁（终止占用进程）
release_uv_venv_lock() {
    local pids
    pids="$(uv_venv_lock_pids)"
    if [[ -z "${pids}" ]]; then
        return 0
    fi
    warn "强制释放 backend/.venv 锁，终止进程: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 1
    pids="$(uv_venv_lock_pids)"
    if [[ -n "${pids}" ]]; then
        # shellcheck disable=SC2086
        kill -9 ${pids} 2>/dev/null || true
        sleep 0.5
    fi
    if is_uv_venv_locked; then
        error "无法释放 backend/.venv 锁，请手动执行: lsof ${BACKEND_DIR}/.venv/.lock"
        return 1
    fi
    ok "backend/.venv 锁已释放"
}

# 启动/同步前确保锁可用
ensure_uv_lock_available() {
    if wait_uv_lock_release; then
        return 0
    fi
    release_uv_venv_lock
}

# 停止与本项目后端相关的 uv 进程
stop_backend_uv_workers() {
    pkill -f "uv run uvicorn aitrade.main:app" 2>/dev/null || true
    local pids
    pids="$(uv_venv_lock_pids)"
    if [[ -n "${pids}" ]]; then
        # shellcheck disable=SC2086
        kill ${pids} 2>/dev/null || true
    fi
}

# 检查后端核心依赖是否已安装
backend_deps_ready() {
    (
        cd "${BACKEND_DIR}"
        uv run python -c "import uvicorn" >/dev/null 2>&1
    )
}

# 启动前同步后端依赖（前台执行，避免与 uv run 争抢锁）
ensure_backend_deps() {
    if backend_deps_ready; then
        return 0
    fi

    if is_uv_venv_locked; then
        error "backend/.venv 正被其他 uv 进程占用（如同步/安装依赖）"
        error "请等待其完成，或执行: lsof ${BACKEND_DIR}/.venv/.lock"
        return 1
    fi

    warn "后端依赖不完整，正在执行 uv sync（大包下载可能较慢，请耐心等待）..."
    info "提示: 网络慢时可设置 UV_HTTP_TIMEOUT=600"
    if ! (
        cd "${BACKEND_DIR}"
        UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}" uv sync
    ); then
        error "uv sync 失败，请手动执行: cd backend && UV_HTTP_TIMEOUT=600 uv sync -v"
        return 1
    fi

    if ! backend_deps_ready; then
        error "依赖同步后仍无法导入 uvicorn，请检查 pyproject.toml 与 uv.lock"
        return 1
    fi
    ok "后端依赖已就绪"
}

# 等待后端 HTTP 服务真正可用
wait_backend_ready() {
    local pid="$1"
    local max_wait="${BACKEND_START_TIMEOUT:-90}"
    local url="http://127.0.0.1:${BACKEND_PORT}/api/status"
    local i

    info "等待后端就绪 (最多 ${max_wait}s)..."
    for ((i=1; i<=max_wait; i++)); do
        if ! is_pid_alive "${pid}"; then
            return 1
        fi
        if http_check "${url}"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# 加载环境变量文件
load_env_file() {
    local env_file="$1"
    if [[ -f "${env_file}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi
}

# 获取服务运行状态: running | stopped | stale
service_state() {
    local name="$1"
    local pid_file port
    case "${name}" in
        backend)
            pid_file="${BACKEND_PID_FILE}"
            port="${BACKEND_PORT}"
            ;;
        frontend)
            pid_file="${FRONTEND_PID_FILE}"
            port="${FRONTEND_PORT}"
            ;;
        *)
            error "未知服务: ${name}"
            return 1
            ;;
    esac

    local pid
    pid="$(read_pid "${pid_file}")"
    if is_pid_alive "${pid}"; then
        echo "running"
        return 0
    fi

    local port_pid
    port_pid="$(pid_on_port "${port}")"
    if [[ -n "${port_pid}" ]]; then
        echo "stale"
        return 0
    fi

    echo "stopped"
}

start_backend() {
    ensure_dirs
    local state
    state="$(service_state backend)"
    if [[ "${state}" == "running" ]]; then
        warn "后端已在运行 (PID: $(read_pid "${BACKEND_PID_FILE}"))"
        return 0
    fi
    if [[ "${state}" == "stale" ]]; then
        warn "端口 ${BACKEND_PORT} 已被占用，尝试清理..."
        kill_port "${BACKEND_PORT}"
        sleep 0.5
    fi

    if ! ensure_uv; then
        error "未找到 uv，请先安装: https://docs.astral.sh/uv/getting-started/installation/"
        return 1
    fi

    if ! ensure_uv_lock_available; then
        return 1
    fi

    if ! ensure_backend_deps; then
        return 1
    fi

    load_env_file "${BACKEND_DIR}/.env"
    load_env_file "${ROOT_DIR}/.env"

    info "启动后端 (uv run) -> http://${BACKEND_HOST}:${BACKEND_PORT}"
    (
        cd "${BACKEND_DIR}"
        nohup uv run uvicorn aitrade.main:app \
            --host "${BACKEND_HOST}" \
            --port "${BACKEND_PORT}" \
            --reload \
            --log-level info \
            >> "${BACKEND_LOG}" 2>&1 &
        echo $! > "${BACKEND_PID_FILE}"
    )

    local pid
    pid="$(read_pid "${BACKEND_PID_FILE}")"
    if wait_backend_ready "${pid}"; then
        ok "后端已启动 (PID: ${pid})，日志: ${BACKEND_LOG}"
    else
        if is_pid_alive "${pid}"; then
            kill_process "${pid}"
        fi
        rm -f "${BACKEND_PID_FILE}"
        error "后端启动失败或未在超时内就绪，请查看日志: ${BACKEND_LOG}"
        error "常见原因: 依赖未装完、端口被占、网络下载超时"
        return 1
    fi
}

start_frontend() {
    ensure_dirs
    local state
    state="$(service_state frontend)"
    if [[ "${state}" == "running" ]]; then
        warn "前端已在运行 (PID: $(read_pid "${FRONTEND_PID_FILE}"))"
        return 0
    fi
    if [[ "${state}" == "stale" ]]; then
        warn "端口 ${FRONTEND_PORT} 已被占用，尝试清理..."
        kill_port "${FRONTEND_PORT}"
        sleep 0.5
    fi

    if ! command -v npm >/dev/null 2>&1; then
        error "未找到 npm，请先安装 Node.js"
        return 1
    fi
    if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
        warn "未检测到 node_modules，正在安装依赖..."
        (cd "${FRONTEND_DIR}" && npm install)
    fi

    load_env_file "${FRONTEND_DIR}/.env"

    info "启动前端 -> http://localhost:${FRONTEND_PORT}"
    (
        cd "${FRONTEND_DIR}"
        nohup npm run dev -- --host 127.0.0.1 --port "${FRONTEND_PORT}" \
            >> "${FRONTEND_LOG}" 2>&1 &
        echo $! > "${FRONTEND_PID_FILE}"
    )

    sleep 2
    if is_pid_alive "$(read_pid "${FRONTEND_PID_FILE}")"; then
        ok "前端已启动 (PID: $(read_pid "${FRONTEND_PID_FILE}"))，日志: ${FRONTEND_LOG}"
    else
        error "前端启动失败，请查看日志: ${FRONTEND_LOG}"
        return 1
    fi
}

stop_backend() {
    local pid state
    state="$(service_state backend)"
    pid="$(read_pid "${BACKEND_PID_FILE}")"

    if [[ "${state}" == "stopped" ]]; then
        info "后端未在运行"
        rm -f "${BACKEND_PID_FILE}"
        return 0
    fi

    info "停止后端..."
    if [[ -n "${pid}" ]]; then
        kill_process "${pid}"
    fi
    stop_backend_uv_workers
    kill_port "${BACKEND_PORT}"
    # 清理可能残留的 uv sync，避免 restart 时锁冲突
    if is_uv_venv_locked; then
        release_uv_venv_lock || true
    fi
    rm -f "${BACKEND_PID_FILE}"
    ok "后端已停止"
}

stop_frontend() {
    local pid state
    state="$(service_state frontend)"
    pid="$(read_pid "${FRONTEND_PID_FILE}")"

    if [[ "${state}" == "stopped" ]]; then
        info "前端未在运行"
        rm -f "${FRONTEND_PID_FILE}"
        return 0
    fi

    info "停止前端..."
    if [[ -n "${pid}" ]]; then
        kill_process "${pid}"
    fi
    kill_port "${FRONTEND_PORT}"
    rm -f "${FRONTEND_PID_FILE}"
    ok "前端已停止"
}

restart_service() {
    local target="$1"
    case "${target}" in
        backend)
            stop_backend || true
            start_backend
            ;;
        frontend)
            stop_frontend || true
            start_frontend
            ;;
        all)
            stop_backend || true
            stop_frontend || true
            start_backend
            start_frontend
            ;;
        *)
            error "未知目标: ${target}，可选: backend | frontend | all"
            return 1
            ;;
    esac
}

start_service() {
    local target="$1"
    case "${target}" in
        backend)  start_backend ;;
        frontend) start_frontend ;;
        all)
            start_backend
            start_frontend
            ;;
        *)
            error "未知目标: ${target}，可选: backend | frontend | all"
            return 1
            ;;
    esac
}

stop_service() {
    local target="$1"
    case "${target}" in
        backend)  stop_backend ;;
        frontend) stop_frontend ;;
        all)
            stop_backend
            stop_frontend
            ;;
        *)
            error "未知目标: ${target}，可选: backend | frontend | all"
            return 1
            ;;
    esac
}

# 探测 HTTP 服务是否可访问
http_check() {
    local url="$1"
    curl -sf --max-time 3 "${url}" >/dev/null 2>&1
}

print_service_status() {
    local name="$1"
    local label port url pid_file log_file
    case "${name}" in
        backend)
            label="后端 (FastAPI)"
            port="${BACKEND_PORT}"
            url="http://127.0.0.1:${port}/api/status"
            pid_file="${BACKEND_PID_FILE}"
            log_file="${BACKEND_LOG}"
            ;;
        frontend)
            label="前端 (Vite)"
            port="${FRONTEND_PORT}"
            url="http://127.0.0.1:${port}/"
            pid_file="${FRONTEND_PID_FILE}"
            log_file="${FRONTEND_LOG}"
            ;;
    esac

    local state pid port_pid health="不可达"
    state="$(service_state "${name}")"
    pid="$(read_pid "${pid_file}")"
    port_pid="$(pid_on_port "${port}")"

    echo -e "${CYAN}${label}${NC}"
    echo "  端口:   ${port}"
    echo "  地址:   ${url}"

    case "${state}" in
        running)
            echo -e "  状态:   ${GREEN}运行中${NC}"
            echo "  PID:    ${pid}"
            ;;
        stale)
            echo -e "  状态:   ${YELLOW}端口占用 (PID 文件失效)${NC}"
            echo "  端口PID: ${port_pid}"
            ;;
        stopped)
            echo -e "  状态:   ${RED}已停止${NC}"
            ;;
    esac

    if http_check "${url}"; then
        health="正常"
        echo -e "  健康:   ${GREEN}${health}${NC}"
    else
        echo -e "  健康:   ${YELLOW}${health}${NC}"
    fi
    echo "  日志:   ${log_file}"
    echo
}

show_status() {
    local target="${1:-all}"
    echo
    echo "========== aitrade 服务状态 =========="
    echo
    case "${target}" in
        backend)  print_service_status backend ;;
        frontend) print_service_status frontend ;;
        all)
            print_service_status backend
            print_service_status frontend
            ;;
        *)
            error "未知目标: ${target}"
            return 1
            ;;
    esac
}

show_logs() {
    local target="${1:-all}"
    case "${target}" in
        backend)
            ensure_dirs
            info "后端日志 (${BACKEND_LOG})，Ctrl+C 退出"
            touch "${BACKEND_LOG}"
            tail -f "${BACKEND_LOG}"
            ;;
        frontend)
            ensure_dirs
            info "前端日志 (${FRONTEND_LOG})，Ctrl+C 退出"
            touch "${FRONTEND_LOG}"
            tail -f "${FRONTEND_LOG}"
            ;;
        all)
            ensure_dirs
            info "合并日志，Ctrl+C 退出"
            touch "${BACKEND_LOG}" "${FRONTEND_LOG}"
            tail -f "${BACKEND_LOG}" "${FRONTEND_LOG}"
            ;;
        *)
            error "未知目标: ${target}"
            return 1
            ;;
    esac
}

usage() {
    cat <<EOF
aitrade 开发服务管理脚本

用法:
  $0                          交互式菜单
  $0 <命令> [目标]

命令:
  start             启动服务
  stop              停止服务
  restart           重启服务
  restart-backend   重启后端
  restart-frontend  重启前端
  rb                重启后端快捷命令
  rf                重启前端快捷命令
  status            查看状态
  logs              实时查看日志

目标:
  backend   仅后端 (端口 ${BACKEND_PORT})
  frontend  仅前端 (端口 ${FRONTEND_PORT})
  all       前后端

示例:
  $0 start all
  $0 stop backend
  $0 restart frontend
  $0 restart-backend
  $0 restart-frontend
  $0 rb
  $0 rf
  $0 status
  $0 logs backend

环境变量:
  AITRADE_PORT            后端端口 (默认 8000)
  AITRADE_HOST            后端监听地址 (默认 127.0.0.1)
  AITRADE_FRONTEND_PORT   前端端口 (默认 3000)
  UV_HTTP_TIMEOUT         uv sync 下载超时秒数 (默认 300)
  UV_LOCK_WAIT_TIMEOUT    等待 .venv 锁释放秒数 (默认 30)
  BACKEND_START_TIMEOUT   后端启动等待秒数 (默认 90)
EOF
}

interactive_menu() {
    while true; do
        echo
        echo "========== aitrade 开发服务管理 =========="
        echo "  1) 启动后端"
        echo "  2) 启动前端"
        echo "  3) 启动全部"
        echo "  4) 停止后端"
        echo "  5) 停止前端"
        echo "  6) 停止全部"
        echo "  7) 重启后端"
        echo "  8) 重启前端"
        echo "  9) 重启全部"
        echo " 10) 查看状态"
        echo " 11) 查看后端日志"
        echo " 12) 查看前端日志"
        echo "  0) 退出"
        echo "=========================================="
        read -rp "请选择 [0-12]: " choice
        case "${choice}" in
            1)  start_backend ;;
            2)  start_frontend ;;
            3)  start_backend; start_frontend ;;
            4)  stop_backend ;;
            5)  stop_frontend ;;
            6)  stop_backend; stop_frontend ;;
            7)  restart_service backend ;;
            8)  restart_service frontend ;;
            9)  restart_service all ;;
            10) show_status all ;;
            11) show_logs backend ;;
            12) show_logs frontend ;;
            0)  echo "再见"; exit 0 ;;
            *)  warn "无效选项，请重新输入" ;;
        esac
    done
}

main() {
    if [[ $# -eq 0 ]]; then
        interactive_menu
        return
    fi

    local cmd="${1:-}"
    local target="${2:-all}"

    case "${cmd}" in
        start)            start_service "${target}" ;;
        stop)             stop_service "${target}" ;;
        restart)          restart_service "${target}" ;;
        restart-backend|rb)
            restart_service backend
            ;;
        restart-frontend|rf)
            restart_service frontend
            ;;
        status)           show_status "${target}" ;;
        logs)             show_logs "${target}" ;;
        -h|--help|help) usage ;;
        *)
            error "未知命令: ${cmd}"
            echo
            usage
            exit 1
            ;;
    esac
}

main "$@"
