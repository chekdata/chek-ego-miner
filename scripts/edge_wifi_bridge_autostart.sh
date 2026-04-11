#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_ROOT_DIR="${CHEK_EDGE_ROOT:-${DEFAULT_ROOT_DIR}}"
ROOT_DIR="${SERVICE_ROOT_DIR}"
ENV_FILE="${CHEK_EDGE_AUTOSTART_ENV_FILE:-/etc/default/chek-edge-autostart}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ "${CHEK_ENABLE_WIFI_POSE:-1}" != "1" ]]; then
  exit 0
fi

if [[ "${WIFI_TRACKED_POSE_DIRECT_ENABLED:-0}" == "1" ]]; then
  echo "wifi bridge skipped because direct tracked pose polling is enabled"
  exec sleep infinity
fi

CHEK_EDGE_ROOT="${SERVICE_ROOT_DIR}"
CHEK_EDGE_USER="${CHEK_EDGE_USER:-$(stat -c %U "${ROOT_DIR}" 2>/dev/null || echo "${USER}")}"
CHEK_EDGE_HTTP_PORT="${CHEK_EDGE_HTTP_PORT:-8080}"
CHEK_EDGE_TOKEN="${CHEK_EDGE_TOKEN:-chek-ego-miner-local-token}"
CHEK_EDGE_FOLLOW_ACTIVE_SESSION="${CHEK_EDGE_FOLLOW_ACTIVE_SESSION:-1}"
CHEK_EDGE_TRIP_ID="${CHEK_EDGE_TRIP_ID:-}"
CHEK_EDGE_SESSION_ID="${CHEK_EDGE_SESSION_ID:-}"
CHEK_WIFI_HTTP_PORT="${CHEK_WIFI_HTTP_PORT:-18080}"
CHEK_WIFI_DEVICE_ID="${CHEK_WIFI_DEVICE_ID:-wifi-densepose-001}"
CHEK_WIFI_TIME_SYNC_INTERVAL_S="${CHEK_WIFI_TIME_SYNC_INTERVAL_S:-2.0}"
CHEK_WIFI_TIME_SYNC_SAMPLE_COUNT="${CHEK_WIFI_TIME_SYNC_SAMPLE_COUNT:-5}"

run_edge_user_bash() {
  local command="$1"
  if [[ "$(id -u)" == "0" ]]; then
    exec runuser -u "${CHEK_EDGE_USER}" -- bash -lc "${command}"
  fi
  exec bash -lc "${command}"
}

pkill -f 'wifi_pose_bridge.py' 2>/dev/null || true

session_args=(
  --trip-id "${CHEK_EDGE_TRIP_ID}"
  --session-id "${CHEK_EDGE_SESSION_ID}"
)
if [[ "${CHEK_EDGE_FOLLOW_ACTIVE_SESSION}" == "1" ]]; then
  session_args=(
    --trip-id ""
    --session-id ""
  )
fi
printf -v session_args_escaped ' %q' "${session_args[@]}"

command="export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH\"; cd \"${CHEK_EDGE_ROOT}\"; exec python3 ./scripts/wifi_pose_bridge.py --wifi-base-url \"http://127.0.0.1:${CHEK_WIFI_HTTP_PORT}\" --edge-base-url \"http://127.0.0.1:${CHEK_EDGE_HTTP_PORT}\" --edge-token \"${CHEK_EDGE_TOKEN}\"${session_args_escaped} --device-id \"${CHEK_WIFI_DEVICE_ID}\" --time-sync-interval-s \"${CHEK_WIFI_TIME_SYNC_INTERVAL_S}\" --time-sync-sample-count \"${CHEK_WIFI_TIME_SYNC_SAMPLE_COUNT}\""

run_edge_user_bash "${command}"
