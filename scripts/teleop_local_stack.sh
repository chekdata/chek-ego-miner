#!/usr/bin/env bash

set -euo pipefail

USER_HOME="${HOME:-}"
if [[ -z "${USER_HOME}" ]]; then
    USER_HOME="$(python3 - <<'PY'
from pathlib import Path
print(Path.home())
PY
)"
fi

export PATH="${USER_HOME}/.cargo/bin:${USER_HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_DIR="${ROOT_DIR}/edge-orchestrator"
LEAP_DIR="${ROOT_DIR}/ruview-leap-bridge"
UNITREE_DIR="${ROOT_DIR}/ruview-unitree-bridge"
SIM_SCRIPT="${ROOT_DIR}/scripts/operator_debug_sim.py"
LEGACY_UI_DIR="${ROOT_DIR}/RuView/ui"
WORKSTATION_DIR="${ROOT_DIR}/RuView/ui-react"
WORKSTATION_DIST_DIR="${WORKSTATION_DIR}/dist"
WORKSTATION_SERVER_SCRIPT="${WORKSTATION_DIR}/scripts/workstation_server.py"
RUNTIME_DIR="${EDGE_DIR}/target/codex-local/teleop-stack"
META_FILE="${RUNTIME_DIR}/stack.env"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

EDGE_HTTP_PORT=8080
EDGE_WS_PORT=8765
EDGE_CSI_PORT=5505
SENSING_HTTP_PORT="${SENSING_HTTP_PORT:-18080}"
REPLAY_HTTP_PORT="${REPLAY_HTTP_PORT:-3020}"
EDGE_CSI_MIRROR_ADDR="${EDGE_CSI_MIRROR_ADDR:-}"
LEAP_HTTP_PORT=8090
UNITREE_HTTP_PORT=8091
VIEWER_PORT=3010
SIM_CONTROL_PORT=3011
EDGE_TOKEN="${EDGE_TOKEN:-chek-ego-miner-local-token}"
TRIP_ID="${TRIP_ID:-trip-local-debug-001}"
SESSION_ID="${SESSION_ID:-sess-local-debug-001}"
DEVICE_ID="${DEVICE_ID:-capture-sim-local-001}"
OPERATOR_ID="${OPERATOR_ID:-operator-local-001}"
SIM_FPS=20
ENABLE_SIM=1
SIM_FLAG_EXPLICIT=0
STACK_BIND_HOST="${STACK_BIND_HOST:-127.0.0.1}"
STACK_CHECK_HOST="${STACK_CHECK_HOST:-127.0.0.1}"
STACK_PUBLIC_HOST="${STACK_PUBLIC_HOST:-${STACK_BIND_HOST}}"
USE_RELEASE=0
FOLLOW_LOGS=0
LOG_SERVICE="all"
EDGE_RUNTIME_PROFILE="${EDGE_RUNTIME_PROFILE:-teleop_fullstack}"
CONTROL_STACK_ENABLED=1

runtime_profile_default_flag() {
    local profile="${1:-teleop_fullstack}"
    local feature="${2:?feature is required}"
    case "${profile}" in
        raw_capture_only)
            case "${feature}" in
                phone_ingest|fusion) echo 1 ;;
                control|sim|stereo|wifi|vlm|preview) echo 0 ;;
                *) echo 0 ;;
            esac
            ;;
        capture_plus_facts)
            case "${feature}" in
                phone_ingest|fusion) echo 1 ;;
                control|sim|stereo|wifi|vlm|preview) echo 0 ;;
                *) echo 0 ;;
            esac
            ;;
        capture_plus_vlm)
            case "${feature}" in
                phone_ingest|fusion|vlm|preview) echo 1 ;;
                control|sim|stereo|wifi) echo 0 ;;
                *) echo 0 ;;
            esac
            ;;
        teleop_fullstack|*)
            case "${feature}" in
                phone_ingest|fusion|control|stereo) echo 1 ;;
                sim|wifi|vlm|preview) echo 0 ;;
                *) echo 0 ;;
            esac
            ;;
    esac
}

apply_runtime_profile_defaults() {
    local control_default sim_default
    control_default="$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" control)"
    sim_default="$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" sim)"
    CONTROL_STACK_ENABLED="${EDGE_CONTROL_ENABLED:-${control_default}}"
    if [[ "${SIM_FLAG_EXPLICIT}" -eq 0 ]]; then
        ENABLE_SIM="${EDGE_SIM_ENABLED:-${sim_default}}"
    fi
}

service_pid_file() {
    echo "${RUNTIME_DIR}/$1.pid"
}

service_log_file() {
    echo "${RUNTIME_DIR}/$1.log"
}

usage() {
    cat <<'EOF'
用法：
  ./scripts/teleop_local_stack.sh start [选项]
  ./scripts/teleop_local_stack.sh stop
  ./scripts/teleop_local_stack.sh restart [选项]
  ./scripts/teleop_local_stack.sh status
  ./scripts/teleop_local_stack.sh logs [edge|leap|unitree|sim|viewer] [--follow]

说明：
  start     启动 edge-orchestrator + LEAP bridge + Unitree bridge + operator 模拟器 + React 工作站
  stop      停止整套本机遥操调试栈
  restart   先 stop 再 start
  status    查看各服务进程、健康检查与 edge control/state
  logs      查看日志；默认输出全部服务的最近日志

start/restart 选项：
  --release                 使用 release 构建产物
  --edge-http-port PORT     edge HTTP 端口，默认 8080
  --edge-ws-port PORT       edge WS 端口，默认 8765
  --edge-csi-port PORT      edge CSI UDP 端口，默认 5505
  --leap-http-port PORT     leap bridge HTTP 端口，默认 8090
  --unitree-http-port PORT  unitree bridge HTTP 端口，默认 8091
  --viewer-port PORT        调试页面端口，默认 3010
  --sim-control-port PORT   模拟器控制接口端口，默认 3011
  --bind-host HOST          服务监听地址，默认 127.0.0.1
  --public-host HOST        对外展示地址，默认与 bind-host 相同
  --edge-token TOKEN        edge 鉴权 token
  --trip-id ID              模拟 trip_id
  --session-id ID           模拟 session_id
  --device-id ID            模拟 device_id
  --operator-id ID          模拟 operator_id
  --sim-fps FPS             模拟器发送频率，默认 20
  --no-sim                  不启动 operator 模拟器，适合真机/真实传感器联调
EOF
}

parse_start_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --release)
                USE_RELEASE=1
                shift
                ;;
            --edge-http-port)
                [[ $# -ge 2 ]] || { echo "缺少 --edge-http-port 参数值" >&2; exit 1; }
                EDGE_HTTP_PORT="$2"
                shift 2
                ;;
            --edge-ws-port)
                [[ $# -ge 2 ]] || { echo "缺少 --edge-ws-port 参数值" >&2; exit 1; }
                EDGE_WS_PORT="$2"
                shift 2
                ;;
            --edge-csi-port)
                [[ $# -ge 2 ]] || { echo "缺少 --edge-csi-port 参数值" >&2; exit 1; }
                EDGE_CSI_PORT="$2"
                shift 2
                ;;
            --leap-http-port)
                [[ $# -ge 2 ]] || { echo "缺少 --leap-http-port 参数值" >&2; exit 1; }
                LEAP_HTTP_PORT="$2"
                shift 2
                ;;
            --unitree-http-port)
                [[ $# -ge 2 ]] || { echo "缺少 --unitree-http-port 参数值" >&2; exit 1; }
                UNITREE_HTTP_PORT="$2"
                shift 2
                ;;
            --viewer-port)
                [[ $# -ge 2 ]] || { echo "缺少 --viewer-port 参数值" >&2; exit 1; }
                VIEWER_PORT="$2"
                shift 2
                ;;
            --sim-control-port)
                [[ $# -ge 2 ]] || { echo "缺少 --sim-control-port 参数值" >&2; exit 1; }
                SIM_CONTROL_PORT="$2"
                shift 2
                ;;
            --bind-host)
                [[ $# -ge 2 ]] || { echo "缺少 --bind-host 参数值" >&2; exit 1; }
                STACK_BIND_HOST="$2"
                shift 2
                ;;
            --public-host)
                [[ $# -ge 2 ]] || { echo "缺少 --public-host 参数值" >&2; exit 1; }
                STACK_PUBLIC_HOST="$2"
                shift 2
                ;;
            --edge-token)
                [[ $# -ge 2 ]] || { echo "缺少 --edge-token 参数值" >&2; exit 1; }
                EDGE_TOKEN="$2"
                shift 2
                ;;
            --trip-id)
                [[ $# -ge 2 ]] || { echo "缺少 --trip-id 参数值" >&2; exit 1; }
                TRIP_ID="$2"
                shift 2
                ;;
            --session-id)
                [[ $# -ge 2 ]] || { echo "缺少 --session-id 参数值" >&2; exit 1; }
                SESSION_ID="$2"
                shift 2
                ;;
            --device-id)
                [[ $# -ge 2 ]] || { echo "缺少 --device-id 参数值" >&2; exit 1; }
                DEVICE_ID="$2"
                shift 2
                ;;
            --operator-id)
                [[ $# -ge 2 ]] || { echo "缺少 --operator-id 参数值" >&2; exit 1; }
                OPERATOR_ID="$2"
                shift 2
                ;;
            --sim-fps)
                [[ $# -ge 2 ]] || { echo "缺少 --sim-fps 参数值" >&2; exit 1; }
                SIM_FPS="$2"
                shift 2
                ;;
            --no-sim)
                ENABLE_SIM=0
                SIM_FLAG_EXPLICIT=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "不支持的参数：$1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done
}

parse_logs_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            edge|leap|unitree|sim|viewer|all)
                LOG_SERVICE="$1"
                shift
                ;;
            --follow|-f)
                FOLLOW_LOGS=1
                shift
                ;;
            *)
                echo "不支持的参数：$1" >&2
                exit 1
                ;;
        esac
    done
}

ensure_runtime_dir() {
    mkdir -p "${RUNTIME_DIR}"
}

load_meta() {
    if [[ -f "${META_FILE}" ]]; then
        # shellcheck disable=SC1090
        source "${META_FILE}"
    fi
}

save_meta() {
    cat > "${META_FILE}" <<EOF
EDGE_HTTP_PORT=${EDGE_HTTP_PORT}
EDGE_WS_PORT=${EDGE_WS_PORT}
EDGE_CSI_PORT=${EDGE_CSI_PORT}
EDGE_CSI_MIRROR_ADDR=${EDGE_CSI_MIRROR_ADDR}
LEAP_HTTP_PORT=${LEAP_HTTP_PORT}
UNITREE_HTTP_PORT=${UNITREE_HTTP_PORT}
VIEWER_PORT=${VIEWER_PORT}
SIM_CONTROL_PORT=${SIM_CONTROL_PORT}
STACK_BIND_HOST=${STACK_BIND_HOST}
STACK_CHECK_HOST=${STACK_CHECK_HOST}
STACK_PUBLIC_HOST=${STACK_PUBLIC_HOST}
EDGE_TOKEN=${EDGE_TOKEN}
TRIP_ID=${TRIP_ID}
SESSION_ID=${SESSION_ID}
DEVICE_ID=${DEVICE_ID}
OPERATOR_ID=${OPERATOR_ID}
SIM_FPS=${SIM_FPS}
ENABLE_SIM=${ENABLE_SIM}
EDGE_RUNTIME_PROFILE=${EDGE_RUNTIME_PROFILE}
CONTROL_STACK_ENABLED=${CONTROL_STACK_ENABLED}
USE_RELEASE=${USE_RELEASE}
EOF
}

is_pid_running() {
    local pid_file="$1"
    [[ -f "${pid_file}" ]] || return 1
    local pid
    pid="$(cat "${pid_file}")"
    [[ -n "${pid}" ]] || return 1
    kill -0 "${pid}" >/dev/null 2>&1 || return 1

    # In some Linux/container environments orphaned children can remain as
    # zombies until PID 1 reaps them. `kill -0` still succeeds for zombies, so
    # treat `Z*` states as no longer running to avoid false restart failures.
    local stat
    stat="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')" || return 1
    [[ -n "${stat}" ]] || return 1
    [[ "${stat}" == Z* ]] && return 1
    return 0
}

stop_service() {
    local service="$1"
    local pid_file
    pid_file="$(service_pid_file "${service}")"

    if ! is_pid_running "${pid_file}"; then
        rm -f "${pid_file}"
        return 0
    fi

    local pid
    pid="$(cat "${pid_file}")"
    echo "停止 ${service}，PID=${pid}"
    kill "${pid}" >/dev/null 2>&1 || true

    local attempt=0
    while is_pid_running "${pid_file}"; do
        sleep 1
        attempt=$((attempt + 1))
        if [[ "${attempt}" -ge 10 ]]; then
            echo "${service} 仍未退出，发送 SIGKILL"
            kill -9 "${pid}" >/dev/null 2>&1 || true
            break
        fi
    done

    rm -f "${pid_file}"
}

check_port_free() {
    local port="$1"
    if lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null 2>&1 \
        || lsof -nP -iUDP:"${port}" >/dev/null 2>&1
    then
        echo "端口 ${port} 已被占用，请先释放或改用其他端口。" >&2
        return 1
    fi
}

resolve_binary_path() {
    local service="$1"
    local repo_dir binary_name profile debug_path release_path
    case "${service}" in
        edge)
            repo_dir="${EDGE_DIR}"
            binary_name="edge-orchestrator"
            ;;
        leap)
            repo_dir="${LEAP_DIR}"
            binary_name="ruview-leap-bridge"
            ;;
        unitree)
            repo_dir="${UNITREE_DIR}"
            binary_name="ruview-unitree-bridge"
            ;;
        *)
            echo "未知服务：${service}" >&2
            exit 1
            ;;
    esac

    debug_path="${repo_dir}/target/debug/${binary_name}"
    release_path="${repo_dir}/target/release/${binary_name}"
    if [[ "${USE_RELEASE}" -eq 1 ]]; then
        echo "${release_path}"
        return 0
    fi
    if [[ -x "${debug_path}" ]]; then
        echo "${debug_path}"
        return 0
    fi
    if [[ -x "${release_path}" ]]; then
        echo "${release_path}"
        return 0
    fi
    echo "${debug_path}"
}

build_binary_if_needed() {
    local service="$1"
    local repo_dir binary_path
    case "${service}" in
        edge) repo_dir="${EDGE_DIR}" ;;
        leap) repo_dir="${LEAP_DIR}" ;;
        unitree) repo_dir="${UNITREE_DIR}" ;;
        *)
            echo "未知服务：${service}" >&2
            exit 1
            ;;
    esac
    binary_path="$(resolve_binary_path "${service}")"

    if [[ -x "${binary_path}" ]]; then
        return 0
    fi

    echo "未检测到 ${service} 可执行文件，开始构建..."
    if [[ "${USE_RELEASE}" -eq 1 ]]; then
        (cd "${repo_dir}" && cargo build --release)
    else
        (cd "${repo_dir}" && cargo build)
    fi
}

launch_background() {
    local workdir="$1"
    local pid_file="$2"
    local log_file="$3"
    shift 3

    python3 - "${workdir}" "${pid_file}" "${log_file}" "$@" <<'PY'
import os
import subprocess
import sys

workdir, pid_file, log_file, *command = sys.argv[1:]
os.makedirs(os.path.dirname(pid_file), exist_ok=True)

with open(log_file, "ab", buffering=0) as log_file_handle:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=log_file_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )

with open(pid_file, "w", encoding="utf-8") as pid_handle:
    pid_handle.write(str(process.pid))
PY
}

wait_for_http_ok() {
    local url="$1"
    local attempt=0
    while [[ "${attempt}" -lt 60 ]]; do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "等待接口就绪超时：${url}" >&2
    return 1
}

ensure_workstation_dist() {
    local log_file="$1"
    if [[ -f "${WORKSTATION_DIST_DIR}/index.html" ]]; then
        return 0
    fi

    if ! command -v npm >/dev/null 2>&1; then
        echo "未检测到 React workstation dist，且 npm 不可用：${WORKSTATION_DIST_DIR}/index.html" >&2
        return 1
    fi

    echo "未检测到 React workstation dist，开始构建..." >> "${log_file}"
    (
        cd "${WORKSTATION_DIR}"
        npm run build >> "${log_file}" 2>&1
    )
}

wait_for_edge_armed() {
    local url="http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}/control/state"
    local attempt=0
    while [[ "${attempt}" -lt 60 ]]; do
        if python3 - "${url}" "${EDGE_TOKEN}" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
token = sys.argv[2]
request = urllib.request.Request(
    url=url,
    method="GET",
    headers={"Authorization": f"Bearer {token}"},
)
try:
    with urllib.request.urlopen(request, timeout=2.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)

preflight = payload.get("preflight") or {}
deadman = payload.get("deadman") or {}
armed = (
    payload.get("state") == "armed"
    and preflight.get("unitree_bridge_ready")
    and preflight.get("leap_bridge_ready")
    and preflight.get("time_sync_ok")
    and preflight.get("extrinsic_ok")
    and preflight.get("lan_control_ok")
    and deadman.get("link_ok")
    and deadman.get("pressed")
)
raise SystemExit(0 if armed else 1)
PY
        then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "等待 edge 进入 armed 超时" >&2
    return 1
}

start_edge() {
    build_binary_if_needed edge
    local pid_file log_file binary_path
    pid_file="$(service_pid_file edge)"
    log_file="$(service_log_file edge)"
    binary_path="$(resolve_binary_path edge)"
    : > "${log_file}"
    ensure_workstation_dist "${log_file}"

    launch_background "${EDGE_DIR}" "${pid_file}" "${log_file}" \
        env \
        EDGE_HTTP_ADDR="${STACK_BIND_HOST}:${EDGE_HTTP_PORT}" \
        EDGE_WS_ADDR="${STACK_BIND_HOST}:${EDGE_WS_PORT}" \
        CSI_UDP_BIND="${STACK_BIND_HOST}:${EDGE_CSI_PORT}" \
        CSI_UDP_MIRROR_ADDR="${EDGE_CSI_MIRROR_ADDR}" \
        EDGE_DATA_DIR="${RUNTIME_DIR}/edge-data" \
        EDGE_TOKEN="${EDGE_TOKEN}" \
        EDGE_ALLOW_SIMULATED_CAPTURE="${EDGE_ALLOW_SIMULATED_CAPTURE:-0}" \
        EDGE_UI_DIST_DIR="${WORKSTATION_DIST_DIR}" \
        EDGE_SENSING_PROXY_BASE="http://${STACK_CHECK_HOST}:${SENSING_HTTP_PORT}" \
        EDGE_SIM_CONTROL_PROXY_BASE="http://${STACK_CHECK_HOST}:${SIM_CONTROL_PORT}" \
        EDGE_REPLAY_PROXY_BASE="http://${STACK_CHECK_HOST}:${REPLAY_HTTP_PORT}" \
        EDGE_PHONE_VISION_SERVICE_BASE="${EDGE_PHONE_VISION_SERVICE_BASE:-http://${STACK_CHECK_HOST}:3031}" \
        EDGE_PHONE_VISION_SERVICE_PATH="${EDGE_PHONE_VISION_SERVICE_PATH:-${ROOT_DIR}/scripts/edge_phone_vision_service.py}" \
        EDGE_PHONE_VISION_PYTHON_BIN="${EDGE_PHONE_VISION_PYTHON_BIN:-python3}" \
        EDGE_PHONE_VISION_SERVICE_AUTOSTART="${EDGE_PHONE_VISION_SERVICE_AUTOSTART:-0}" \
        EDGE_PHONE_VISION_PROCESSING_ENABLED="${EDGE_PHONE_VISION_PROCESSING_ENABLED:-1}" \
        EDGE_RUNTIME_PROFILE="${EDGE_RUNTIME_PROFILE}" \
        EDGE_PHONE_INGEST_ENABLED="${EDGE_PHONE_INGEST_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" phone_ingest)}" \
        EDGE_STEREO_ENABLED="${EDGE_STEREO_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" stereo)}" \
        EDGE_WIFI_ENABLED="${EDGE_WIFI_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" wifi)}" \
        EDGE_FUSION_ENABLED="${EDGE_FUSION_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" fusion)}" \
        EDGE_CONTROL_ENABLED="${EDGE_CONTROL_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" control)}" \
        EDGE_SIM_ENABLED="${EDGE_SIM_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" sim)}" \
        EDGE_VLM_INDEXING_ENABLED="${EDGE_VLM_INDEXING_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" vlm)}" \
        EDGE_PREVIEW_GENERATION_ENABLED="${EDGE_PREVIEW_GENERATION_ENABLED:-$(runtime_profile_default_flag "${EDGE_RUNTIME_PROFILE}" preview)}" \
        EDGE_VLM_MODEL_ID="${EDGE_VLM_MODEL_ID:-SmolVLM2-500M}" \
        EDGE_VLM_FALLBACK_MODEL_ID="${EDGE_VLM_FALLBACK_MODEL_ID:-SmolVLM2-256M}" \
        EDGE_VLM_KEYFRAME_INTERVAL_MS="${EDGE_VLM_KEYFRAME_INTERVAL_MS:-3000}" \
        EDGE_VLM_EVENT_TRIGGER_ENABLED="${EDGE_VLM_EVENT_TRIGGER_ENABLED:-1}" \
        EDGE_VLM_EVENT_TRIGGER_CAMERA_MODE_CHANGE_ENABLED="${EDGE_VLM_EVENT_TRIGGER_CAMERA_MODE_CHANGE_ENABLED:-1}" \
        EDGE_VLM_SIDECAR_BASE="${EDGE_VLM_SIDECAR_BASE:-}" \
        EDGE_VLM_SIDECAR_PATH="${EDGE_VLM_SIDECAR_PATH:-${ROOT_DIR}/scripts/edge_vlm_sidecar.py}" \
        EDGE_VLM_SIDECAR_PYTHON_BIN="${EDGE_VLM_SIDECAR_PYTHON_BIN:-python3}" \
        EDGE_VLM_SIDECAR_AUTOSTART="${EDGE_VLM_SIDECAR_AUTOSTART:-0}" \
        EDGE_VLM_PRIMARY_MODEL_PATH="${EDGE_VLM_PRIMARY_MODEL_PATH:-}" \
        EDGE_VLM_FALLBACK_MODEL_PATH="${EDGE_VLM_FALLBACK_MODEL_PATH:-}" \
        EDGE_VLM_RUNTIME_DEVICE="${EDGE_VLM_RUNTIME_DEVICE:-auto}" \
        EDGE_VLM_EDGE_LONGEST_SIDE_PX="${EDGE_VLM_EDGE_LONGEST_SIDE_PX:-256}" \
        EDGE_VLM_EDGE_IMAGE_SEQ_LEN="${EDGE_VLM_EDGE_IMAGE_SEQ_LEN:-16}" \
        EDGE_VLM_DISABLE_IMAGE_SPLITTING="${EDGE_VLM_DISABLE_IMAGE_SPLITTING:-1}" \
        EDGE_VLM_INFERENCE_TIMEOUT_MS="${EDGE_VLM_INFERENCE_TIMEOUT_MS:-3500}" \
        EDGE_VLM_AUTO_FALLBACK_LATENCY_MS="${EDGE_VLM_AUTO_FALLBACK_LATENCY_MS:-2200}" \
        EDGE_VLM_AUTO_FALLBACK_COOLDOWN_MS="${EDGE_VLM_AUTO_FALLBACK_COOLDOWN_MS:-60000}" \
        EDGE_VLM_MAX_CONSECUTIVE_FAILURES="${EDGE_VLM_MAX_CONSECUTIVE_FAILURES:-2}" \
        EDGE_CROWD_UPLOAD_POLICY_MODE="${EDGE_CROWD_UPLOAD_POLICY_MODE:-full_raw_mirror}" \
        EXTRINSIC_VERSION="local-debug-0.1.0" \
        RUST_LOG="${RUST_LOG:-info}" \
        "${binary_path}"

    wait_for_http_ok "http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}/health"
}

start_leap() {
    build_binary_if_needed leap
    local pid_file log_file binary_path
    pid_file="$(service_pid_file leap)"
    log_file="$(service_log_file leap)"
    binary_path="$(resolve_binary_path leap)"
    : > "${log_file}"

    launch_background "${LEAP_DIR}" "${pid_file}" "${log_file}" \
        env \
        EDGE_TELEOP_WS_URL="ws://${STACK_CHECK_HOST}:${EDGE_WS_PORT}/stream/teleop" \
        EDGE_TOKEN="${EDGE_TOKEN}" \
        HTTP_ADDR="${STACK_BIND_HOST}:${LEAP_HTTP_PORT}" \
        BRIDGE_ID="leap-bridge-01" \
        RUST_LOG="${RUST_LOG:-info}" \
        "${binary_path}"

    wait_for_http_ok "http://${STACK_CHECK_HOST}:${LEAP_HTTP_PORT}/health"
}

start_unitree() {
    build_binary_if_needed unitree
    local pid_file log_file binary_path
    pid_file="$(service_pid_file unitree)"
    log_file="$(service_log_file unitree)"
    binary_path="$(resolve_binary_path unitree)"
    : > "${log_file}"

    launch_background "${UNITREE_DIR}" "${pid_file}" "${log_file}" \
        env \
        EDGE_TELEOP_WS_URL="ws://${STACK_CHECK_HOST}:${EDGE_WS_PORT}/stream/teleop" \
        EDGE_TOKEN="${EDGE_TOKEN}" \
        HTTP_ADDR="${STACK_BIND_HOST}:${UNITREE_HTTP_PORT}" \
        BRIDGE_ID="unitree-bridge-01" \
        RUST_LOG="${RUST_LOG:-info}" \
        "${binary_path}"

    wait_for_http_ok "http://${STACK_CHECK_HOST}:${UNITREE_HTTP_PORT}/health"
}

start_sim() {
    local pid_file log_file
    pid_file="$(service_pid_file sim)"
    log_file="$(service_log_file sim)"
    : > "${log_file}"

    if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
raise SystemExit(0 if importlib.util.find_spec("websockets") else 1)
PY
    then
        echo "缺少 Python 依赖 websockets，请先执行：python3 -m pip install websockets" >&2
        return 1
    fi

    launch_background "${ROOT_DIR}" "${pid_file}" "${log_file}" \
        python3 "${SIM_SCRIPT}" \
        --http-base "http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}" \
        --ws-base "ws://${STACK_CHECK_HOST}:${EDGE_WS_PORT}" \
        --edge-token "${EDGE_TOKEN}" \
        --trip-id "${TRIP_ID}" \
        --session-id "${SESSION_ID}" \
        --device-id "${DEVICE_ID}" \
        --operator-id "${OPERATOR_ID}" \
        --control-port "${SIM_CONTROL_PORT}" \
        --fps "${SIM_FPS}"

    wait_for_edge_armed
    wait_for_http_ok "http://${STACK_CHECK_HOST}:${SIM_CONTROL_PORT}/state"
}

start_viewer() {
    local pid_file log_file
    pid_file="$(service_pid_file viewer)"
    log_file="$(service_log_file viewer)"
    : > "${log_file}"

    ensure_workstation_dist "${log_file}"

    launch_background "${ROOT_DIR}" "${pid_file}" "${log_file}" \
        python3 "${WORKSTATION_SERVER_SCRIPT}" \
        --bind "${STACK_BIND_HOST}" \
        --port "${VIEWER_PORT}" \
        --dist-dir "${WORKSTATION_DIST_DIR}" \
        --edge-http-base "http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}" \
        --sensing-http-base "http://${STACK_CHECK_HOST}:${SENSING_HTTP_PORT}" \
        --sim-control-http-base "http://${STACK_CHECK_HOST}:${SIM_CONTROL_PORT}" \
        --replay-http-base "http://${STACK_CHECK_HOST}:${REPLAY_HTTP_PORT}"

    wait_for_http_ok "http://${STACK_CHECK_HOST}:${VIEWER_PORT}/healthz"
}

print_summary() {
    local workstation_url viewer_url
    if [[ "${ENABLE_SIM}" -eq 1 ]]; then
        workstation_url="http://${STACK_PUBLIC_HOST}:${EDGE_HTTP_PORT}/?token=${EDGE_TOKEN}#/overview"
        viewer_url="http://${STACK_PUBLIC_HOST}:${VIEWER_PORT}/?token=${EDGE_TOKEN}#/overview"
    else
        workstation_url="http://${STACK_PUBLIC_HOST}:${EDGE_HTTP_PORT}/?token=${EDGE_TOKEN}#/overview"
        viewer_url="http://${STACK_PUBLIC_HOST}:${VIEWER_PORT}/?token=${EDGE_TOKEN}#/overview"
    fi
    cat <<EOF
本机遥操调试栈已启动
  edge HTTP:      http://${STACK_PUBLIC_HOST}:${EDGE_HTTP_PORT}/health
  edge control:   http://${STACK_PUBLIC_HOST}:${EDGE_HTTP_PORT}/control/state
  edge WS fusion: ws://${STACK_PUBLIC_HOST}:${EDGE_WS_PORT}/stream/fusion
  leap health:    $( [[ "${CONTROL_STACK_ENABLED}" == "1" ]] && echo "http://${STACK_PUBLIC_HOST}:${LEAP_HTTP_PORT}/health" || echo "未启动（control disabled）" )
  unitree health: $( [[ "${CONTROL_STACK_ENABLED}" == "1" ]] && echo "http://${STACK_PUBLIC_HOST}:${UNITREE_HTTP_PORT}/health" || echo "未启动（control disabled）" )
  工作站入口:     ${workstation_url}
  独立预览:       ${viewer_url}
  模拟控制:       $( [[ "${ENABLE_SIM}" -eq 1 ]] && echo "http://${STACK_PUBLIC_HOST}:${SIM_CONTROL_PORT}/state" || echo "未启动（真机/真实传感器模式）" )
  trip/session:   ${TRIP_ID} / ${SESSION_ID}
  runtime profile:${EDGE_RUNTIME_PROFILE}
  运行目录:       ${RUNTIME_DIR}
EOF
}

start_stack() {
    parse_start_args "$@"
    apply_runtime_profile_defaults
    ensure_runtime_dir

    for service in edge leap unitree sim viewer; do
        if is_pid_running "$(service_pid_file "${service}")"; then
            echo "${service} 已在运行，请先 stop 或执行 restart。" >&2
            return 1
        fi
    done

    check_port_free "${EDGE_HTTP_PORT}"
    check_port_free "${EDGE_WS_PORT}"
    if [[ "${CONTROL_STACK_ENABLED}" == "1" ]]; then
        check_port_free "${LEAP_HTTP_PORT}"
        check_port_free "${UNITREE_HTTP_PORT}"
    fi
    check_port_free "${EDGE_CSI_PORT}"
    check_port_free "${VIEWER_PORT}"
    if [[ "${ENABLE_SIM}" -eq 1 ]]; then
        check_port_free "${SIM_CONTROL_PORT}"
    fi

    save_meta

    local started=()
    cleanup_on_error() {
        for service in viewer sim unitree leap edge; do
            stop_service "${service}" || true
        done
    }
    trap cleanup_on_error ERR

    start_edge
    started+=("edge")
    if [[ "${CONTROL_STACK_ENABLED}" == "1" ]]; then
        start_leap
        started+=("leap")
        start_unitree
        started+=("unitree")
    fi
    if [[ "${ENABLE_SIM}" -eq 1 ]]; then
        start_sim
        started+=("sim")
    fi
    start_viewer
    started+=("viewer")

    trap - ERR
    print_summary
}

stop_stack() {
    load_meta
    for service in viewer sim unitree leap edge; do
        stop_service "${service}"
    done
}

status_stack() {
    load_meta
    local edge_running=0
    local leap_running=0
    local unitree_running=0
    local viewer_running=0
    local sim_running=0

    for service in edge leap unitree sim viewer; do
        local pid_file pid log_file
        pid_file="$(service_pid_file "${service}")"
        log_file="$(service_log_file "${service}")"
        if is_pid_running "${pid_file}"; then
            pid="$(cat "${pid_file}")"
            echo "${service}: 运行中 (PID=${pid})"
            case "${service}" in
                edge) edge_running=1 ;;
                leap) leap_running=1 ;;
                unitree) unitree_running=1 ;;
                sim) sim_running=1 ;;
                viewer) viewer_running=1 ;;
            esac
        else
            echo "${service}: 未运行"
        fi
        if [[ -f "${log_file}" ]]; then
            echo "  日志: ${log_file}"
        fi
    done

    echo ""
    echo "[edge /health]"
    if [[ "${edge_running}" -eq 1 ]]; then
        curl -fsS "http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}/health" || echo "请求失败"
    else
        echo "edge 未运行，跳过探活"
    fi
    echo ""
    echo "[edge /control/state]"
    if [[ "${edge_running}" -eq 1 ]]; then
        curl -fsS "http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}/control/state" \
            -H "Authorization: Bearer ${EDGE_TOKEN}" || echo "请求失败"
    else
        echo "edge 未运行，跳过探活"
    fi
    echo ""
    echo "[edge /association/hint]"
    if [[ "${edge_running}" -eq 1 ]]; then
        curl -fsS "http://${STACK_CHECK_HOST}:${EDGE_HTTP_PORT}/association/hint" \
            -H "Authorization: Bearer ${EDGE_TOKEN}" || echo "请求失败"
    else
        echo "edge 未运行，跳过关联提示"
    fi
    echo ""
    echo "[leap /health]"
    if [[ "${leap_running}" -eq 1 ]]; then
        curl -fsS "http://${STACK_CHECK_HOST}:${LEAP_HTTP_PORT}/health" || echo "请求失败"
    else
        echo "leap 未运行，跳过探活"
    fi
    echo ""
    echo "[unitree /health]"
    if [[ "${unitree_running}" -eq 1 ]]; then
        curl -fsS "http://${STACK_CHECK_HOST}:${UNITREE_HTTP_PORT}/health" || echo "请求失败"
    else
        echo "unitree 未运行，跳过探活"
    fi
    echo ""
    echo "[sensing /api/v1/stream/status]"
    if curl -fsS "http://${STACK_CHECK_HOST}:${SENSING_HTTP_PORT}/api/v1/stream/status" >/dev/null 2>&1; then
        curl -fsS "http://${STACK_CHECK_HOST}:${SENSING_HTTP_PORT}/api/v1/stream/status" || echo "请求失败"
    else
        echo "Wi‑Fi sensing server 未运行：http://${STACK_CHECK_HOST}:${SENSING_HTTP_PORT}"
    fi
    echo ""
    echo "[real-source preflight]"
    python3 - "${SENSING_HTTP_PORT}" <<'PY'
import json
import glob
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

sensing_port = sys.argv[1]

def detect_video_devices():
    if platform.system() == "Darwin":
        devices = []
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                completed = subprocess.run(
                    [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=8,
                )
                in_video = False
                for raw_line in f"{completed.stdout}\n{completed.stderr}".splitlines():
                    line = raw_line.strip()
                    if "AVFoundation video devices:" in line:
                        in_video = True
                        continue
                    if "AVFoundation audio devices:" in line:
                        in_video = False
                        continue
                    if not in_video:
                        continue
                    match = re.search(r"\]\s+\[(\d+)\]\s+(.+)$", line)
                    if match and not match.group(2).strip().lower().startswith("capture screen"):
                        devices.append(f"avfoundation:{match.group(1)}:{match.group(2).strip()}")
            except Exception:
                devices = []
        return devices
    return sorted(glob.glob("/dev/video*"))

video_devices = detect_video_devices()
stereo_hint = {
    "host_video_devices": video_devices,
    "stereo_local_producer_ready": len(video_devices) >= 2,
    "stereo_local_reason": (
        "本机已发现至少 2 个视频设备，可尝试启动 stereo_pose_producer.py"
        if len(video_devices) >= 2
        else "本机未发现至少 2 个视频设备，stereo_pose_producer.py 无法在当前主机直接起双目采集"
    ),
}
wifi_url = f"http://127.0.0.1:{sensing_port}/api/v1/stream/status"
try:
    with urllib.request.urlopen(wifi_url, timeout=1.5) as response:
        wifi_payload = json.loads(response.read().decode("utf-8"))
    wifi_hint = {
        "wifi_sensing_ready": True,
        "wifi_reason": "Wi‑Fi sensing server 在线",
        "stream_status": wifi_payload,
    }
except Exception:
    wifi_hint = {
        "wifi_sensing_ready": False,
        "wifi_reason": f"Wi‑Fi sensing server 不在线：{wifi_url}",
    }
print(json.dumps({"stereo": stereo_hint, "wifi": wifi_hint}, ensure_ascii=False, indent=2))
PY
    echo ""
    echo "[viewer]"
    if [[ "${viewer_running}" -eq 1 ]]; then
        if [[ "${sim_running}" -eq 1 ]]; then
            echo "http://${STACK_PUBLIC_HOST}:${VIEWER_PORT}/?token=${EDGE_TOKEN}&control=http://${STACK_PUBLIC_HOST}:${SIM_CONTROL_PORT}#/overview"
        else
            echo "http://${STACK_PUBLIC_HOST}:${VIEWER_PORT}/?token=${EDGE_TOKEN}#/overview"
        fi
    else
        echo "viewer 未运行，跳过探活"
    fi
    echo ""
    echo "[sim control]"
    if [[ "${sim_running}" -eq 1 ]]; then
        curl -fsS "http://${STACK_CHECK_HOST}:${SIM_CONTROL_PORT}/state" || echo "请求失败"
    else
        echo "sim 未运行，当前为真机/真实传感器模式"
    fi
    echo ""
}

logs_stack() {
    parse_logs_args "$@"
    load_meta

    if [[ "${FOLLOW_LOGS}" -eq 1 && "${LOG_SERVICE}" == "all" ]]; then
        echo "--follow 需要指定单个服务：edge | leap | unitree | sim | viewer" >&2
        exit 1
    fi

    if [[ "${LOG_SERVICE}" == "all" ]]; then
        for service in edge leap unitree sim viewer; do
            local log_file
            log_file="$(service_log_file "${service}")"
            echo "===== ${service} ====="
            if [[ -f "${log_file}" ]]; then
                tail -n 40 "${log_file}"
            else
                echo "暂无日志：${log_file}"
            fi
            echo ""
        done
        return 0
    fi

    local log_file
    log_file="$(service_log_file "${LOG_SERVICE}")"
    if [[ ! -f "${log_file}" ]]; then
        echo "暂无日志：${log_file}"
        return 0
    fi

    if [[ "${FOLLOW_LOGS}" -eq 1 ]]; then
        tail -f "${log_file}"
    else
        tail -n 200 "${log_file}"
    fi
}

restart_stack() {
    stop_stack
    start_stack "$@"
}

case "${COMMAND}" in
    start)
        start_stack "$@"
        ;;
    stop)
        stop_stack
        ;;
    restart)
        restart_stack "$@"
        ;;
    status)
        status_stack
        ;;
    logs)
        logs_stack "$@"
        ;;
    help|-h|--help|"")
        usage
        ;;
    *)
        echo "不支持的命令：${COMMAND}" >&2
        usage >&2
        exit 1
        ;;
esac
