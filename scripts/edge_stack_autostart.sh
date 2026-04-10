#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${CHEK_EDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STACK_SCRIPT="${ROOT_DIR}/scripts/teleop_local_stack.sh"
ACTION="${1:-start}"
EDGE_RUNTIME_PROFILE="${EDGE_RUNTIME_PROFILE:-teleop_fullstack}"
STACK_BIND_HOST="${STACK_BIND_HOST:-0.0.0.0}"
STACK_PUBLIC_HOST="${STACK_PUBLIC_HOST:-${STACK_BIND_HOST}}"
EDGE_CONTROL_ENABLED="${EDGE_CONTROL_ENABLED:-1}"
EDGE_SIM_ENABLED="${EDGE_SIM_ENABLED:-0}"

if [[ ! -x "${STACK_SCRIPT}" ]]; then
  echo "缺少 teleop stack 启动脚本: ${STACK_SCRIPT}" >&2
  exit 1
fi

run_stack() {
  cd "${ROOT_DIR}"
  export EDGE_RUNTIME_PROFILE
  export EDGE_CONTROL_ENABLED
  export EDGE_SIM_ENABLED
  export STACK_CHECK_HOST=127.0.0.1
  case "${ACTION}" in
    start)
      exec bash "${STACK_SCRIPT}" start --release --no-sim --bind-host "${STACK_BIND_HOST}" --public-host "${STACK_PUBLIC_HOST}"
      ;;
    stop)
      exec bash "${STACK_SCRIPT}" stop
      ;;
    restart)
      exec bash "${STACK_SCRIPT}" restart --release --no-sim --bind-host "${STACK_BIND_HOST}" --public-host "${STACK_PUBLIC_HOST}"
      ;;
    status)
      exec bash "${STACK_SCRIPT}" status
      ;;
    *)
      echo "用法: $(basename "$0") {start|stop|restart|status}" >&2
      exit 2
      ;;
  esac
}

run_stack
