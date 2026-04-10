#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_VENV="${CHEK_JETSON_VLM_SOURCE_VENV:-${HOME}/CHEK-humanoid/runtime/vlm-gpu-wheel-venv}"
SOURCE_PRIMARY_MODEL="${CHEK_JETSON_VLM_SOURCE_PRIMARY_MODEL:-${HOME}/CHEK-humanoid/runtime/ms-smol-500m-skel}"
SOURCE_FALLBACK_MODEL="${CHEK_JETSON_VLM_SOURCE_FALLBACK_MODEL:-${HOME}/CHEK-humanoid/runtime/ms-smol-256m-skel}"
TARGET_VENV="${ROOT_DIR}/runtime/vlm-venv"
TARGET_MODELS_ROOT="${ROOT_DIR}/model-candidates/huggingface"
TARGET_PRIMARY_MODEL="${TARGET_MODELS_ROOT}/SmolVLM2-500M-Video-Instruct"
TARGET_FALLBACK_MODEL="${TARGET_MODELS_ROOT}/SmolVLM2-256M-Video-Instruct"
LINK_MODE="symlink"
FORCE=0

usage() {
  cat <<'EOF'
Usage: bootstrap_jetson_vlm_runtime.sh [options]

Wire an existing Jetson VLM runtime into the public repo layout.

Options:
  --source-venv PATH
  --source-primary-model PATH
  --source-fallback-model PATH
  --copy                      copy assets instead of symlinking them
  --force                     replace existing targets
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-venv)
      SOURCE_VENV="$2"
      shift 2
      ;;
    --source-primary-model)
      SOURCE_PRIMARY_MODEL="$2"
      shift 2
      ;;
    --source-fallback-model)
      SOURCE_FALLBACK_MODEL="$2"
      shift 2
      ;;
    --copy)
      LINK_MODE="copy"
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ensure_source() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

replace_target() {
  local source="$1"
  local target="$2"
  ensure_source "${source}" "${target##*/} source"
  mkdir -p "$(dirname "${target}")"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "${FORCE}" != "1" ]]; then
      echo "Target already exists: ${target} (use --force to replace)" >&2
      exit 1
    fi
    rm -rf "${target}"
  fi
  if [[ "${LINK_MODE}" == "copy" ]]; then
    cp -R "${source}" "${target}"
  else
    ln -s "${source}" "${target}"
  fi
}

ensure_source "${SOURCE_VENV}" "VLM venv"
ensure_source "${SOURCE_PRIMARY_MODEL}" "primary VLM model"
ensure_source "${SOURCE_FALLBACK_MODEL}" "fallback VLM model"

replace_target "${SOURCE_VENV}" "${TARGET_VENV}"
replace_target "${SOURCE_PRIMARY_MODEL}" "${TARGET_PRIMARY_MODEL}"
replace_target "${SOURCE_FALLBACK_MODEL}" "${TARGET_FALLBACK_MODEL}"

echo "jetson_vlm_bootstrap_ok"
echo "root=${ROOT_DIR}"
echo "venv=${TARGET_VENV}"
echo "primary_model=${TARGET_PRIMARY_MODEL}"
echo "fallback_model=${TARGET_FALLBACK_MODEL}"
