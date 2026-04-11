#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_ROOT_DIR="${CHEK_EDGE_ROOT:-${DEFAULT_ROOT_DIR}}"
ROOT_DIR="${SERVICE_ROOT_DIR}"
STACK_SCRIPT="${ROOT_DIR}/scripts/teleop_local_stack.sh"
ENV_FILE="${CHEK_EDGE_AUTOSTART_ENV_FILE:-/etc/default/chek-edge-autostart}"
ACTION="${1:-start}"
EDGE_RUNTIME_PROFILE="${EDGE_RUNTIME_PROFILE:-teleop_fullstack}"
STACK_BIND_HOST="${STACK_BIND_HOST:-0.0.0.0}"
STACK_PUBLIC_HOST="${STACK_PUBLIC_HOST:-${STACK_BIND_HOST}}"
EDGE_CONTROL_ENABLED="${EDGE_CONTROL_ENABLED:-1}"
EDGE_SIM_ENABLED="${EDGE_SIM_ENABLED:-0}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CHEK_EDGE_ROOT="${SERVICE_ROOT_DIR}"
EDGE_STACK_TOKEN="${CHEK_EDGE_TOKEN:-${EDGE_TOKEN:-chek-ego-miner-local-token}}"

if [[ ! -x "${STACK_SCRIPT}" ]]; then
  echo "缺少 teleop stack 启动脚本: ${STACK_SCRIPT}" >&2
  exit 1
fi

run_stack() {
  cd "${ROOT_DIR}"
  export EDGE_RUNTIME_PROFILE
  export EDGE_CONTROL_ENABLED
  export EDGE_SIM_ENABLED
  export EDGE_TOKEN="${EDGE_STACK_TOKEN}"
  export STACK_CHECK_HOST=127.0.0.1
  case "${ACTION}" in
    start)
      exec bash "${STACK_SCRIPT}" start --release --no-sim --bind-host "${STACK_BIND_HOST}" --public-host "${STACK_PUBLIC_HOST}" --edge-token "${EDGE_STACK_TOKEN}"
      ;;
    stop)
      exec bash "${STACK_SCRIPT}" stop
      ;;
    restart)
      exec bash "${STACK_SCRIPT}" restart --release --no-sim --bind-host "${STACK_BIND_HOST}" --public-host "${STACK_PUBLIC_HOST}" --edge-token "${EDGE_STACK_TOKEN}"
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
