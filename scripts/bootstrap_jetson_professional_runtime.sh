#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_ROOT="${CHEK_JETSON_PRO_SOURCE_ROOT:-${HOME}/CHEK-humanoid}"
SOURCE_CALIBRATION="${CHEK_JETSON_PRO_SOURCE_CALIBRATION:-${SOURCE_ROOT}/data/ruview/runtime/stereo_pair_calibration.json}"
SOURCE_WIFI_MODEL="${CHEK_JETSON_PRO_SOURCE_WIFI_MODEL:-${SOURCE_ROOT}/RuView/rust-port/wifi-densepose-rs/data/models/trained-supervised-20260405_145457.rvf}"
SOURCE_WIFI_BINARY="${CHEK_JETSON_PRO_SOURCE_WIFI_BINARY:-${SOURCE_ROOT}/RuView/rust-port/wifi-densepose-rs/target/debug/sensing-server}"
SOURCE_WIFI_UI="${CHEK_JETSON_PRO_SOURCE_WIFI_UI:-${SOURCE_ROOT}/RuView/ui}"
SOURCE_EDGE_BINARY="${CHEK_JETSON_PRO_SOURCE_EDGE_BINARY:-${SOURCE_ROOT}/edge-orchestrator/target/debug/edge-orchestrator}"
SOURCE_LEAP_BINARY="${CHEK_JETSON_PRO_SOURCE_LEAP_BINARY:-${SOURCE_ROOT}/ruview-leap-bridge/target/debug/ruview-leap-bridge}"
SOURCE_UNITREE_BINARY="${CHEK_JETSON_PRO_SOURCE_UNITREE_BINARY:-${SOURCE_ROOT}/ruview-unitree-bridge/target/debug/ruview-unitree-bridge}"
SOURCE_WORKSTATION_DIST="${CHEK_JETSON_PRO_SOURCE_WORKSTATION_DIST:-${SOURCE_ROOT}/RuView/ui-react/dist}"

TARGET_CALIBRATION="${ROOT_DIR}/data/ruview/runtime/stereo_pair_calibration.json"
TARGET_WIFI_MODEL="${ROOT_DIR}/RuView/rust-port/wifi-densepose-rs/data/models/trained-supervised-live.rvf"
TARGET_WIFI_BINARY="${ROOT_DIR}/RuView/rust-port/wifi-densepose-rs/target/debug/sensing-server"
TARGET_WIFI_UI="${ROOT_DIR}/RuView/ui"
TARGET_EDGE_BINARY="${ROOT_DIR}/edge-orchestrator/target/debug/edge-orchestrator"
TARGET_EDGE_BINARY_RELEASE="${ROOT_DIR}/edge-orchestrator/target/release/edge-orchestrator"
TARGET_LEAP_BINARY="${ROOT_DIR}/ruview-leap-bridge/target/debug/ruview-leap-bridge"
TARGET_LEAP_BINARY_RELEASE="${ROOT_DIR}/ruview-leap-bridge/target/release/ruview-leap-bridge"
TARGET_UNITREE_BINARY="${ROOT_DIR}/ruview-unitree-bridge/target/debug/ruview-unitree-bridge"
TARGET_UNITREE_BINARY_RELEASE="${ROOT_DIR}/ruview-unitree-bridge/target/release/ruview-unitree-bridge"
TARGET_WORKSTATION_DIST="${ROOT_DIR}/RuView/ui-react/dist"

LINK_MODE="symlink"
FORCE=0
WITH_VLM=1
VLM_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bootstrap_jetson_professional_runtime.sh [options]

Wire host-local Jetson professional runtime assets into the public repo layout.

Options:
  --source-root PATH
  --source-calibration PATH
  --source-wifi-model PATH
  --source-wifi-binary PATH
  --source-wifi-ui PATH
  --source-edge-binary PATH
  --source-leap-binary PATH
  --source-unitree-binary PATH
  --source-workstation-dist PATH
  --copy                      copy assets instead of symlinking them
  --force                     replace existing targets
  --without-vlm              skip the VLM runtime bootstrap
  --with-vlm-arg ARG         extra arg forwarded to bootstrap_jetson_vlm_runtime.sh
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --source-calibration)
      SOURCE_CALIBRATION="$2"
      shift 2
      ;;
    --source-wifi-model)
      SOURCE_WIFI_MODEL="$2"
      shift 2
      ;;
    --source-wifi-binary)
      SOURCE_WIFI_BINARY="$2"
      shift 2
      ;;
    --source-wifi-ui)
      SOURCE_WIFI_UI="$2"
      shift 2
      ;;
    --source-edge-binary)
      SOURCE_EDGE_BINARY="$2"
      shift 2
      ;;
    --source-leap-binary)
      SOURCE_LEAP_BINARY="$2"
      shift 2
      ;;
    --source-unitree-binary)
      SOURCE_UNITREE_BINARY="$2"
      shift 2
      ;;
    --source-workstation-dist)
      SOURCE_WORKSTATION_DIST="$2"
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
    --without-vlm)
      WITH_VLM=0
      shift
      ;;
    --with-vlm-arg)
      VLM_EXTRA_ARGS+=("$2")
      shift 2
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

replace_target "${SOURCE_CALIBRATION}" "${TARGET_CALIBRATION}"
replace_target "${SOURCE_WIFI_MODEL}" "${TARGET_WIFI_MODEL}"
replace_target "${SOURCE_WIFI_BINARY}" "${TARGET_WIFI_BINARY}"
replace_target "${SOURCE_WIFI_UI}" "${TARGET_WIFI_UI}"
replace_target "${SOURCE_EDGE_BINARY}" "${TARGET_EDGE_BINARY}"
replace_target "${SOURCE_EDGE_BINARY}" "${TARGET_EDGE_BINARY_RELEASE}"
replace_target "${SOURCE_LEAP_BINARY}" "${TARGET_LEAP_BINARY}"
replace_target "${SOURCE_LEAP_BINARY}" "${TARGET_LEAP_BINARY_RELEASE}"
replace_target "${SOURCE_UNITREE_BINARY}" "${TARGET_UNITREE_BINARY}"
replace_target "${SOURCE_UNITREE_BINARY}" "${TARGET_UNITREE_BINARY_RELEASE}"
replace_target "${SOURCE_WORKSTATION_DIST}" "${TARGET_WORKSTATION_DIST}"

if [[ "${WITH_VLM}" == "1" ]]; then
  "${SCRIPT_DIR}/bootstrap_jetson_vlm_runtime.sh" --force "${VLM_EXTRA_ARGS[@]}"
fi

echo "jetson_professional_bootstrap_ok"
echo "root=${ROOT_DIR}"
echo "stereo_calibration=${TARGET_CALIBRATION}"
echo "wifi_model=${TARGET_WIFI_MODEL}"
echo "wifi_binary=${TARGET_WIFI_BINARY}"
echo "wifi_ui=${TARGET_WIFI_UI}"
echo "edge_binary=${TARGET_EDGE_BINARY}"
echo "edge_binary_release=${TARGET_EDGE_BINARY_RELEASE}"
echo "leap_binary=${TARGET_LEAP_BINARY}"
echo "leap_binary_release=${TARGET_LEAP_BINARY_RELEASE}"
echo "unitree_binary=${TARGET_UNITREE_BINARY}"
echo "unitree_binary_release=${TARGET_UNITREE_BINARY_RELEASE}"
echo "workstation_dist=${TARGET_WORKSTATION_DIST}"
