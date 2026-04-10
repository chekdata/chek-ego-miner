#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PHONE_VENV="${ROOT_DIR}/runtime/phonevision-venv"
DEFAULT_ADAPTER_VENV="${ROOT_DIR}/sim/teleop_adapter/.venv"
VENV_PATH="${EDGE_PHONE_VISION_VENV_PATH:-${DEFAULT_PHONE_VENV}}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  VENV_PATH="${ISAAC_ADAPTER_VENV_PATH:-${DEFAULT_ADAPTER_VENV}}"
fi
PYTHON_BIN="${EDGE_PHONE_VISION_PYTHON_BIN:-${VENV_PATH}/bin/python}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaacsim}"
ISAAC_PYTHON="${ISAAC_SIM_ROOT}/python.sh"
SCRIPT_PATH="${ROOT_DIR}/scripts/edge_phone_vision_service.py"
HAND_MODEL_PATH="${EDGE_PHONE_VISION_HAND_MODEL:-${ROOT_DIR}/model-candidates/rtmlib/rtmpose-m-hand.onnx}"
POSE_MODEL_PATH="${EDGE_PHONE_VISION_POSE_MODEL:-${ROOT_DIR}/model-candidates/mediapipe/pose_landmarker_heavy.task}"
SERVICE_HOST="${EDGE_PHONE_VISION_HOST:-0.0.0.0}"
SERVICE_PORT="${EDGE_PHONE_VISION_PORT:-3031}"

resolve_python_bin() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    echo "${candidate}"
    return 0
  fi
  command -v "${candidate}" 2>/dev/null || true
}

python_can_run_service() {
  local python_bin
  python_bin="$(resolve_python_bin "$1")"
  [[ -n "${python_bin}" && -x "${python_bin}" ]] || return 1
  "${python_bin}" -c 'import cv2, mediapipe, numpy' >/dev/null 2>&1
}

select_python_bin() {
  local candidate resolved
  for candidate in \
    "${PYTHON_BIN}" \
    "${ISAAC_PYTHON}" \
    python3.10 \
    python3.11 \
    python3.12 \
    python3.13 \
    python3
  do
    [[ -n "${candidate}" ]] || continue
    if python_can_run_service "${candidate}"; then
      resolved="$(resolve_python_bin "${candidate}")"
      [[ -n "${resolved}" ]] || continue
      echo "${resolved}"
      return 0
    fi
  done
  return 1
}

SELECTED_PYTHON_BIN="$(select_python_bin || true)"
if [[ -z "${SELECTED_PYTHON_BIN}" ]]; then
  echo "Edge phone vision Python 不存在或缺少依赖: ${PYTHON_BIN}" >&2
  echo "Isaac python fallback 也不可用: ${ISAAC_PYTHON}" >&2
  echo "额外尝试的解释器: python3.10 python3.11 python3.12 python3.13 python3" >&2
  exit 1
fi

if [[ "${SELECTED_PYTHON_BIN}" != "$(resolve_python_bin "${PYTHON_BIN}")" ]]; then
  echo "[edge-phone-vision] 自动切换到兼容解释器: ${SELECTED_PYTHON_BIN}" >&2
fi

PYTHON_BIN="${SELECTED_PYTHON_BIN}"

exec "${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --host "${SERVICE_HOST}" \
  --port "${SERVICE_PORT}" \
  --rtmpose-hand-model "${HAND_MODEL_PATH}" \
  --pose-model "${POSE_MODEL_PATH}" \
  "$@"
