#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${CHEK_EDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${ISAAC_ADAPTER_VENV_PATH:-${ROOT_DIR}/sim/teleop_adapter/.venv}"
PYTHON_BIN="${ISAAC_ADAPTER_PYTHON_BIN:-${VENV_PATH}/bin/python}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaacsim}"
ISAAC_PYTHON="${ISAAC_SIM_ROOT}/python.sh"
SCRIPT_PATH="${ROOT_DIR}/sim/teleop_adapter/teleop_adapter.py"

resolve_python_bin() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    echo "${candidate}"
    return 0
  fi
  command -v "${candidate}" 2>/dev/null || true
}

python_can_run_adapter() {
  local python_bin
  python_bin="$(resolve_python_bin "$1")"
  [[ -n "${python_bin}" && -x "${python_bin}" ]] || return 1
  "${python_bin}" -c 'import asyncio, json, urllib.request, websockets' >/dev/null 2>&1
}

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "缺少 Isaac adapter 脚本: ${SCRIPT_PATH}" >&2
  exit 1
fi

if ! python_can_run_adapter "${PYTHON_BIN}"; then
  if python_can_run_adapter "${ISAAC_PYTHON}"; then
    echo "[isaac-host] adapter venv 不可用，回退到 Isaac python: ${ISAAC_PYTHON}" >&2
    PYTHON_BIN="${ISAAC_PYTHON}"
  else
    echo "Isaac adapter Python 不存在或缺少依赖: ${PYTHON_BIN}" >&2
    echo "Isaac python fallback 也不可用: ${ISAAC_PYTHON}" >&2
    exit 1
  fi
fi

exec "${PYTHON_BIN}" "${SCRIPT_PATH}" "$@"
