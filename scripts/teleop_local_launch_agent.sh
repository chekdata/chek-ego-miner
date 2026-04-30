#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${CHEK_EDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STACK_SCRIPT="${ROOT_DIR}/scripts/teleop_local_stack.sh"
RUNTIME_PROFILE="${EDGE_RUNTIME_PROFILE:-capture_plus_facts}"
CONTROL_ENABLED="${EDGE_CONTROL_ENABLED:-0}"
SIM_ENABLED="${EDGE_SIM_ENABLED:-0}"
STACK_BIND_HOST="${STACK_BIND_HOST:-127.0.0.1}"
STACK_CHECK_HOST="${STACK_CHECK_HOST:-${STACK_BIND_HOST}}"
STACK_PUBLIC_HOST="${STACK_PUBLIC_HOST:-${STACK_BIND_HOST}}"
EDGE_HTTP_PORT="${EDGE_HTTP_PORT:-8080}"
EDGE_WS_PORT="${EDGE_WS_PORT:-28765}"
VIEWER_PORT="${VIEWER_PORT:-3010}"
CHECK_INTERVAL="${CHEK_EDGE_AGENT_CHECK_INTERVAL:-30}"

log() {
    printf '[chek-edge-launch-agent] %s\n' "$*"
}

run_stack() {
    EDGE_RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
    EDGE_CONTROL_ENABLED="${CONTROL_ENABLED}" \
    EDGE_SIM_ENABLED="${SIM_ENABLED}" \
    CHEK_EDGE_ROOT="${ROOT_DIR}" \
    "${STACK_SCRIPT}" "$@"
}

stack_start() {
    local args=(start --edge-http-port "${EDGE_HTTP_PORT}" --edge-ws-port "${EDGE_WS_PORT}" --viewer-port "${VIEWER_PORT}" --bind-host "${STACK_BIND_HOST}" --public-host "${STACK_PUBLIC_HOST}")
    if [[ "${SIM_ENABLED}" == "0" ]]; then
        args+=(--no-sim)
    fi
    log "starting local teleop stack"
    run_stack "${args[@]}"
}

stack_restart() {
    local args=(restart --edge-http-port "${EDGE_HTTP_PORT}" --edge-ws-port "${EDGE_WS_PORT}" --viewer-port "${VIEWER_PORT}" --bind-host "${STACK_BIND_HOST}" --public-host "${STACK_PUBLIC_HOST}")
    if [[ "${SIM_ENABLED}" == "0" ]]; then
        args+=(--no-sim)
    fi
    log "restarting local teleop stack"
    run_stack "${args[@]}"
}

stack_stop() {
    log "stopping local teleop stack"
    run_stack stop || true
}

stack_status() {
    curl -fsS "http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}/health" >/dev/null 2>&1 \
        && curl -fsS "http://${STACK_CHECK_HOST}:${VIEWER_PORT}/healthz" >/dev/null 2>&1
}

cleanup() {
    stack_stop
}

trap cleanup EXIT INT TERM

if stack_status; then
    log "stack already healthy, skip initial start"
else
    stack_start
fi

while true; do
    sleep "${CHECK_INTERVAL}"
    if ! stack_status; then
        log "health probe failed, recovering stack"
        stack_restart
    fi
done
