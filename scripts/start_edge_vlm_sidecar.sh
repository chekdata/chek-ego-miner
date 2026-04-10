#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODELS_ROOT="${EDGE_VLM_MODELS_ROOT:-${ROOT_DIR}/model-candidates/huggingface}"
DEFAULT_PRIMARY_MODEL_PATH="${DEFAULT_MODELS_ROOT}/SmolVLM2-500M-Video-Instruct"
DEFAULT_FALLBACK_MODEL_PATH="${DEFAULT_MODELS_ROOT}/SmolVLM2-256M-Video-Instruct"
PYTHON_BIN="${EDGE_VLM_SIDECAR_PYTHON_BIN:-python3}"
SCRIPT_PATH="${ROOT_DIR}/scripts/edge_vlm_sidecar.py"
PRIMARY_MODEL_ID="${EDGE_VLM_MODEL_ID:-SmolVLM2-500M}"
FALLBACK_MODEL_ID="${EDGE_VLM_FALLBACK_MODEL_ID:-SmolVLM2-256M}"
PRIMARY_MODEL_PATH="${EDGE_VLM_PRIMARY_MODEL_PATH:-${DEFAULT_PRIMARY_MODEL_PATH}}"
FALLBACK_MODEL_PATH="${EDGE_VLM_FALLBACK_MODEL_PATH:-${DEFAULT_FALLBACK_MODEL_PATH}}"
SERVICE_BASE="${EDGE_VLM_SIDECAR_BASE:-http://127.0.0.1:3032}"
RUNTIME_DEVICE="${EDGE_VLM_RUNTIME_DEVICE:-auto}"
REPO_VENV_PYTHON="${ROOT_DIR}/runtime/vlm-venv/bin/python"

resolve_python_bin() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    echo "${candidate}"
    return 0
  fi
  command -v "${candidate}" 2>/dev/null || true
}

python_can_run_sidecar() {
  local python_bin
  python_bin="$(resolve_python_bin "$1")"
  [[ -n "${python_bin}" && -x "${python_bin}" ]] || return 1
  "${python_bin}" -c 'import PIL, torch, transformers, huggingface_hub, num2words' >/dev/null 2>&1
}

select_python_bin() {
  local candidate resolved
  for candidate in \
    "${PYTHON_BIN}" \
    "${REPO_VENV_PYTHON}" \
    python3.10 \
    python3.11 \
    python3.12 \
    python3.13 \
    python3
  do
    [[ -n "${candidate}" ]] || continue
    if python_can_run_sidecar "${candidate}"; then
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
  echo "Edge VLM Python 不存在或缺少依赖: ${PYTHON_BIN}" >&2
  echo "先执行: python3 -m pip install --user -r ${ROOT_DIR}/scripts/edge_vlm_requirements.txt" >&2
  exit 1
fi

if [[ "${SELECTED_PYTHON_BIN}" != "$(resolve_python_bin "${PYTHON_BIN}")" ]]; then
  echo "[edge-vlm] 自动切换到兼容解释器: ${SELECTED_PYTHON_BIN}" >&2
fi

if [[ ! -f "${PRIMARY_MODEL_PATH}/config.json" ]]; then
  echo "缺少主 VLM 模型目录: ${PRIMARY_MODEL_PATH}" >&2
  echo "先执行: ./cli/chek-ego-miner fetch-vlm-models --primary-model-id ${PRIMARY_MODEL_ID}" >&2
  exit 1
fi

if [[ -n "${FALLBACK_MODEL_PATH}" && ! -f "${FALLBACK_MODEL_PATH}/config.json" ]]; then
  echo "[edge-vlm] fallback 模型不存在，将忽略 fallback: ${FALLBACK_MODEL_PATH}" >&2
  FALLBACK_MODEL_PATH=""
fi

exec "${SELECTED_PYTHON_BIN}" "${SCRIPT_PATH}" \
  --base "${SERVICE_BASE}" \
  --primary-model-id "${PRIMARY_MODEL_ID}" \
  --fallback-model-id "${FALLBACK_MODEL_ID}" \
  --primary-model-path "${PRIMARY_MODEL_PATH}" \
  --fallback-model-path "${FALLBACK_MODEL_PATH}" \
  --runtime-device "${RUNTIME_DEVICE}" \
  "$@"
