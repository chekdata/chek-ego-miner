#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${CHEK_EDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BRIDGE_PATH="${ROOT_DIR}/scripts/wifi_pose_bridge.py"

if [[ ! -f "${BRIDGE_PATH}" ]]; then
  echo "缺少 Wi-Fi bridge 脚本: ${BRIDGE_PATH}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec python3 "${BRIDGE_PATH}" "$@"
