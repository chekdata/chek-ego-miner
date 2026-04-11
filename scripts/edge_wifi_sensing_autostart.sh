#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_ROOT_DIR="${CHEK_EDGE_ROOT:-${DEFAULT_ROOT_DIR}}"
ROOT_DIR="${SERVICE_ROOT_DIR}"
ENV_FILE="${CHEK_EDGE_AUTOSTART_ENV_FILE:-/etc/default/chek-edge-autostart}"
WIFI_DIR="${ROOT_DIR}/RuView/rust-port/wifi-densepose-rs"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ "${CHEK_ENABLE_WIFI_POSE:-1}" != "1" ]]; then
  exit 0
fi

CHEK_EDGE_ROOT="${SERVICE_ROOT_DIR}"
CHEK_EDGE_USER="${CHEK_EDGE_USER:-$(stat -c %U "${ROOT_DIR}" 2>/dev/null || echo "${USER}")}"
CHEK_WIFI_HTTP_PORT="${CHEK_WIFI_HTTP_PORT:-18080}"
CHEK_WIFI_WS_PORT="${CHEK_WIFI_WS_PORT:-18765}"
CHEK_WIFI_UDP_PORT="${CHEK_WIFI_UDP_PORT:-5006}"
CHEK_WIFI_SOURCE="${CHEK_WIFI_SOURCE:-esp32}"
CHEK_WIFI_DISABLE_LEGACY_ESP32_PORT_FALLBACK="${CHEK_WIFI_DISABLE_LEGACY_ESP32_PORT_FALLBACK:-1}"
CHEK_WIFI_MODEL_PATH="${CHEK_WIFI_MODEL_PATH:-${WIFI_DIR}/data/models/trained-supervised-live.rvf}"
CHEK_WIFI_UI_PATH="${CHEK_WIFI_UI_PATH:-${ROOT_DIR}/RuView/ui}"
REPO_WIFI_MODEL_PATH="${WIFI_DIR}/data/models/trained-supervised-live.rvf"
REPO_WIFI_UI_PATH="${ROOT_DIR}/RuView/ui"

if [[ ! -f "${CHEK_WIFI_MODEL_PATH}" && -f "${REPO_WIFI_MODEL_PATH}" ]]; then
  CHEK_WIFI_MODEL_PATH="${REPO_WIFI_MODEL_PATH}"
fi

if [[ ! -d "${CHEK_WIFI_UI_PATH}" && -d "${REPO_WIFI_UI_PATH}" ]]; then
  CHEK_WIFI_UI_PATH="${REPO_WIFI_UI_PATH}"
fi

run_edge_user_bash() {
  local command="$1"
  if [[ "$(id -u)" == "0" ]]; then
    exec runuser -u "${CHEK_EDGE_USER}" -- bash -lc "${command}"
  fi
  exec bash -lc "${command}"
}

terminate_matching_sensing() {
  local http_port="${1:?http_port is required}"
  local -a pids=()
  while read -r pid cmdline; do
    [[ -n "${pid:-}" && -n "${cmdline:-}" ]] || continue
    case "${cmdline}" in
      *wifi-densepose-sensing-server*|*sensing-server*) ;;
      *) continue ;;
    esac
    case "${cmdline}" in
      *"--http-port ${http_port}"*|*"--http-port \"${http_port}\""*) pids+=("${pid}") ;;
    esac
  done < <(ps -eo pid=,args=)
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}

if [[ ! -d "${WIFI_DIR}" ]]; then
  echo "缺少 Wi-Fi sensing 运行目录: ${WIFI_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CHEK_WIFI_MODEL_PATH}" ]]; then
  echo "缺少 Wi-Fi sensing 模型: ${CHEK_WIFI_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CHEK_WIFI_UI_PATH}" ]]; then
  echo "缺少 Wi-Fi sensing UI 目录: ${CHEK_WIFI_UI_PATH}" >&2
  exit 1
fi

terminate_matching_sensing "${CHEK_WIFI_HTTP_PORT}"

command="export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH\"; cd \"${WIFI_DIR}\"; extra_args=(); if [[ \"${CHEK_WIFI_DISABLE_LEGACY_ESP32_PORT_FALLBACK}\" == \"1\" ]]; then extra_args+=(--disable-legacy-esp32-port-fallback); fi; if [[ -x ./target/debug/sensing-server ]]; then exec ./target/debug/sensing-server --http-port \"${CHEK_WIFI_HTTP_PORT}\" --ws-port \"${CHEK_WIFI_WS_PORT}\" --udp-port \"${CHEK_WIFI_UDP_PORT}\" --source \"${CHEK_WIFI_SOURCE}\" --model \"${CHEK_WIFI_MODEL_PATH}\" --load-rvf \"${CHEK_WIFI_MODEL_PATH}\" --ui-path \"${CHEK_WIFI_UI_PATH}\" \"\${extra_args[@]}\"; else exec cargo run -p wifi-densepose-sensing-server -- --http-port \"${CHEK_WIFI_HTTP_PORT}\" --ws-port \"${CHEK_WIFI_WS_PORT}\" --udp-port \"${CHEK_WIFI_UDP_PORT}\" --source \"${CHEK_WIFI_SOURCE}\" --model \"${CHEK_WIFI_MODEL_PATH}\" --load-rvf \"${CHEK_WIFI_MODEL_PATH}\" --ui-path \"${CHEK_WIFI_UI_PATH}\" \"\${extra_args[@]}\"; fi"

run_edge_user_bash "${command}"
