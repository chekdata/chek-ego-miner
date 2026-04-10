#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${CHEK_EDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PRODUCER_PATH="${ROOT_DIR}/scripts/stereo_pose_producer.py"

if [[ ! -f "${PRODUCER_PATH}" ]]; then
  echo "缺少 stereo producer: ${PRODUCER_PATH}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
exec python3 "${PRODUCER_PATH}" "$@"
