#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${CHEK_EDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WIFI_DIR="${ROOT_DIR}/RuView/rust-port/wifi-densepose-rs"

if [[ ! -d "${WIFI_DIR}" ]]; then
  echo "缺少 Wi-Fi sensing 运行目录: ${WIFI_DIR}" >&2
  exit 1
fi

cd "${WIFI_DIR}"
if [[ -x "./target/debug/sensing-server" ]]; then
  exec ./target/debug/sensing-server "$@"
fi
exec cargo run -p wifi-densepose-sensing-server -- "$@"
