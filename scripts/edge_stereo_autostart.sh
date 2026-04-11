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

if [[ "${CHEK_ENABLE_STEREO:-1}" != "1" ]]; then
  exit 0
fi

CHEK_EDGE_ROOT="${SERVICE_ROOT_DIR}"
CHEK_EDGE_USER="${CHEK_EDGE_USER:-$(stat -c %U "${ROOT_DIR}" 2>/dev/null || echo "${USER}")}"
CHEK_EDGE_HTTP_PORT="${CHEK_EDGE_HTTP_PORT:-8080}"
CHEK_EDGE_TOKEN="${CHEK_EDGE_TOKEN:-chek-ego-miner-local-token}"
CHEK_EDGE_FOLLOW_ACTIVE_SESSION="${CHEK_EDGE_FOLLOW_ACTIVE_SESSION:-1}"
CHEK_EDGE_TRIP_ID="${CHEK_EDGE_TRIP_ID:-}"
CHEK_EDGE_SESSION_ID="${CHEK_EDGE_SESSION_ID:-}"

CHEK_STEREO_DEVICE_ID="${CHEK_STEREO_DEVICE_ID:-stereo-uvc-001}"
CHEK_STEREO_INPUT_MODE="${CHEK_STEREO_INPUT_MODE:-uvc-sbs}"
CHEK_STEREO_VIDEO_DEVICE="${CHEK_STEREO_VIDEO_DEVICE:-/dev/video0}"
CHEK_STEREO_WIDTH="${CHEK_STEREO_WIDTH:-1920}"
CHEK_STEREO_HEIGHT="${CHEK_STEREO_HEIGHT:-1520}"
CHEK_STEREO_FPS="${CHEK_STEREO_FPS:-6}"
CHEK_STEREO_CALIBRATION_PATH="${CHEK_STEREO_CALIBRATION_PATH:-${CHEK_EDGE_ROOT}/data/ruview/runtime/stereo_pair_calibration.json}"
CHEK_STEREO_MODE_PROBE_TIMEOUT_S="${CHEK_STEREO_MODE_PROBE_TIMEOUT_S:-8}"
CHEK_STEREO_UVC_SBS_MODE_CANDIDATES="${CHEK_STEREO_UVC_SBS_MODE_CANDIDATES:-1920x1080@30 1280x720@30}"
CHEK_STEREO_INFERENCE_WIDTH="${CHEK_STEREO_INFERENCE_WIDTH:-480}"
CHEK_STEREO_INFERENCE_HEIGHT="${CHEK_STEREO_INFERENCE_HEIGHT:-380}"
CHEK_STEREO_MODEL_MODE="${CHEK_STEREO_MODEL_MODE:-lightweight}"
CHEK_STEREO_BODY_SCORE_THRESHOLD="${CHEK_STEREO_BODY_SCORE_THRESHOLD:-0.35}"
CHEK_STEREO_JOINT_SCORE_THRESHOLD="${CHEK_STEREO_JOINT_SCORE_THRESHOLD:-0.25}"
CHEK_STEREO_MAX_PERSONS="${CHEK_STEREO_MAX_PERSONS:-4}"
CHEK_STEREO_DET_FREQUENCY="${CHEK_STEREO_DET_FREQUENCY:-3}"
CHEK_STEREO_TRACKING_THR="${CHEK_STEREO_TRACKING_THR:-0.3}"
CHEK_STEREO_PARALLEL_INFERENCE_WORKERS="${CHEK_STEREO_PARALLEL_INFERENCE_WORKERS:-2}"
CHEK_STEREO_TIME_SYNC_INTERVAL_S="${CHEK_STEREO_TIME_SYNC_INTERVAL_S:-2.0}"
CHEK_STEREO_TIME_SYNC_SAMPLE_COUNT="${CHEK_STEREO_TIME_SYNC_SAMPLE_COUNT:-5}"
CHEK_STEREO_TIME_SYNC_TIMEOUT_S="${CHEK_STEREO_TIME_SYNC_TIMEOUT_S:-8.0}"
CHEK_STEREO_LOW_ROI_SECOND_PASS="${CHEK_STEREO_LOW_ROI_SECOND_PASS:-1}"
CHEK_STEREO_LOW_ROI_TOP_FRACTION="${CHEK_STEREO_LOW_ROI_TOP_FRACTION:-0.38}"
CHEK_STEREO_LOW_ROI_HEIGHT_FRACTION="${CHEK_STEREO_LOW_ROI_HEIGHT_FRACTION:-0.62}"
CHEK_STEREO_LOW_ROI_LEFT_FRACTION="${CHEK_STEREO_LOW_ROI_LEFT_FRACTION:-0.0}"
CHEK_STEREO_LOW_ROI_WIDTH_FRACTION="${CHEK_STEREO_LOW_ROI_WIDTH_FRACTION:-1.0}"
CHEK_STEREO_LOW_ROI_INFERENCE_WIDTH="${CHEK_STEREO_LOW_ROI_INFERENCE_WIDTH:-0}"
CHEK_STEREO_LOW_ROI_INFERENCE_HEIGHT="${CHEK_STEREO_LOW_ROI_INFERENCE_HEIGHT:-0}"
CHEK_STEREO_LOW_ROI_BODY_SCORE_THRESHOLD="${CHEK_STEREO_LOW_ROI_BODY_SCORE_THRESHOLD:-0.24}"
CHEK_STEREO_LOW_ROI_JOINT_SCORE_THRESHOLD="${CHEK_STEREO_LOW_ROI_JOINT_SCORE_THRESHOLD:-0.18}"
CHEK_STEREO_LOW_ROI_SECOND_PASS_INTERVAL="${CHEK_STEREO_LOW_ROI_SECOND_PASS_INTERVAL:-3}"
CHEK_STEREO_LEFT_FRAME="${CHEK_STEREO_LEFT_FRAME:-/tmp/stereo-uvc-left.jpg}"
CHEK_STEREO_RIGHT_FRAME="${CHEK_STEREO_RIGHT_FRAME:-/tmp/stereo-uvc-right.jpg}"
CHEK_STEREO_PREVIEW="${CHEK_STEREO_PREVIEW:-/tmp/stereo-uvc-preview.jpg}"
CHEK_STEREO_RAW_UPLOAD_QUEUE_SIZE="${CHEK_STEREO_RAW_UPLOAD_QUEUE_SIZE:-4}"
CHEK_STEREO_ASSOCIATION_HINT_URL="${CHEK_STEREO_ASSOCIATION_HINT_URL:-http://127.0.0.1:${CHEK_EDGE_HTTP_PORT}/association/hint}"
CHEK_STEREO_FX_PX="${CHEK_STEREO_FX_PX:-1200}"
CHEK_STEREO_FY_PX="${CHEK_STEREO_FY_PX:-1200}"
CHEK_STEREO_CX_PX="${CHEK_STEREO_CX_PX:-960}"
CHEK_STEREO_CY_PX="${CHEK_STEREO_CY_PX:-540}"
CHEK_STEREO_BASELINE_M="${CHEK_STEREO_BASELINE_M:-0.06}"
CHEK_STEREO_UPLOAD_RAW_MEDIA="${CHEK_STEREO_UPLOAD_RAW_MEDIA:-1}"
CHEK_STEREO_RAW_CHUNK_DURATION_S="${CHEK_STEREO_RAW_CHUNK_DURATION_S:-2}"
CHEK_STEREO_RAW_UPLOAD_TIMEOUT_S="${CHEK_STEREO_RAW_UPLOAD_TIMEOUT_S:-8}"

REPO_STEREO_CALIBRATION_PATH="${CHEK_EDGE_ROOT}/data/ruview/runtime/stereo_pair_calibration.json"
if [[ ! -f "${CHEK_STEREO_CALIBRATION_PATH}" && -f "${REPO_STEREO_CALIBRATION_PATH}" ]]; then
  CHEK_STEREO_CALIBRATION_PATH="${REPO_STEREO_CALIBRATION_PATH}"
fi

run_edge_user_bash() {
  local command="$1"
  if [[ "$(id -u)" == "0" ]]; then
    exec runuser -u "${CHEK_EDGE_USER}" -- bash -lc "${command}"
  fi
  exec bash -lc "${command}"
}

resolve_uvc_sbs_device() {
  local configured="${CHEK_STEREO_VIDEO_DEVICE}"
  if [[ -c "${configured}" ]]; then
    printf '%s\n' "${configured}"
    return 0
  fi
  local dev=""
  for dev in /dev/video*; do
    [[ -c "${dev}" ]] || continue
    if ! command -v v4l2-ctl >/dev/null 2>&1; then
      continue
    fi
    local info
    info="$(v4l2-ctl -d "${dev}" --all 2>/dev/null || true)"
    [[ "${info}" == *"Driver name      : uvcvideo"* ]] || continue
    [[ "${info}" == *"Format Video Capture:"* ]] || continue
    printf '%s\n' "${dev}"
    return 0
  done
  return 1
}

parse_uvc_sbs_mode_spec() {
  local spec="$1"
  if [[ ! "${spec}" =~ ^([0-9]+)x([0-9]+)@([0-9]+([.][0-9]+)?)$ ]]; then
    return 1
  fi
  printf '%s %s %s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
}

probe_uvc_sbs_mode() {
  local device="$1"
  if ! command -v v4l2-ctl >/dev/null 2>&1; then
    return 0
  fi

  local configured_mode="${CHEK_STEREO_WIDTH}x${CHEK_STEREO_HEIGHT}@${CHEK_STEREO_FPS}"
  local mode_specs=("${configured_mode}")
  local spec
  for spec in ${CHEK_STEREO_UVC_SBS_MODE_CANDIDATES}; do
    [[ -n "${spec}" ]] || continue
    if [[ " ${mode_specs[*]} " != *" ${spec} "* ]]; then
      mode_specs+=("${spec}")
    fi
  done

  local probe_file probe_log parsed eye_width eye_height fps full_width status size
  probe_file="$(mktemp /tmp/chek-stereo-mode-probe-XXXXXX.bin)"
  probe_log="${probe_file}.log"
  trap 'rm -f "${probe_file}" "${probe_log}"' RETURN

  for spec in "${mode_specs[@]}"; do
    if ! parsed="$(parse_uvc_sbs_mode_spec "${spec}")"; then
      continue
    fi
    read -r eye_width eye_height fps <<<"${parsed}"
    full_width=$(( eye_width * 2 ))
    rm -f "${probe_file}" "${probe_log}"
    timeout "${CHEK_STEREO_MODE_PROBE_TIMEOUT_S}s" \
      v4l2-ctl -d "${device}" \
        --set-fmt-video="width=${full_width},height=${eye_height},pixelformat=MJPG" \
        --set-parm="${fps}" \
        --stream-mmap=3 \
        --stream-count=1 \
        --stream-to="${probe_file}" >"${probe_log}" 2>&1
    status=$?
    size="$(stat -c %s "${probe_file}" 2>/dev/null || echo 0)"
    if [[ "${size}" -gt 0 ]]; then
      CHEK_STEREO_WIDTH="${eye_width}"
      CHEK_STEREO_HEIGHT="${eye_height}"
      CHEK_STEREO_FPS="${fps}"
      return 0
    fi
    echo "[stereo] probe failed for ${spec}: status=${status} size=${size}" >&2
  done
  return 1
}

pkill -f 'stereo_pose_producer.py' 2>/dev/null || true

resolved_stereo_device="${CHEK_STEREO_VIDEO_DEVICE}"
if [[ "${CHEK_STEREO_INPUT_MODE}" == "uvc-sbs" ]]; then
  if resolved="$(resolve_uvc_sbs_device)"; then
    resolved_stereo_device="${resolved}"
  else
    echo "[stereo] no usable UVC capture device found" >&2
    exit 1
  fi
  if ! probe_uvc_sbs_mode "${resolved_stereo_device}"; then
    echo "[stereo] no working MJPG UVC SBS mode found on ${resolved_stereo_device}" >&2
    exit 1
  fi
fi

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

low_roi_args=""
if [[ "${CHEK_STEREO_LOW_ROI_SECOND_PASS}" == "1" ]]; then
  low_roi_args=" --low-roi-second-pass --low-roi-second-pass-interval \"${CHEK_STEREO_LOW_ROI_SECOND_PASS_INTERVAL}\" --low-roi-top-fraction \"${CHEK_STEREO_LOW_ROI_TOP_FRACTION}\" --low-roi-height-fraction \"${CHEK_STEREO_LOW_ROI_HEIGHT_FRACTION}\" --low-roi-left-fraction \"${CHEK_STEREO_LOW_ROI_LEFT_FRACTION}\" --low-roi-width-fraction \"${CHEK_STEREO_LOW_ROI_WIDTH_FRACTION}\" --low-roi-inference-width \"${CHEK_STEREO_LOW_ROI_INFERENCE_WIDTH}\" --low-roi-inference-height \"${CHEK_STEREO_LOW_ROI_INFERENCE_HEIGHT}\" --low-roi-body-score-threshold \"${CHEK_STEREO_LOW_ROI_BODY_SCORE_THRESHOLD}\" --low-roi-joint-score-threshold \"${CHEK_STEREO_LOW_ROI_JOINT_SCORE_THRESHOLD}\""
fi

command="export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH\"; cd \"${CHEK_EDGE_ROOT}\"; exec python3 ./scripts/stereo_pose_producer.py --http-base \"http://127.0.0.1:${CHEK_EDGE_HTTP_PORT}\" --edge-token \"${CHEK_EDGE_TOKEN}\"${session_args_escaped} --device-id \"${CHEK_STEREO_DEVICE_ID}\" --input-mode \"${CHEK_STEREO_INPUT_MODE}\" --stereo-device \"${resolved_stereo_device}\" --width \"${CHEK_STEREO_WIDTH}\" --height \"${CHEK_STEREO_HEIGHT}\" --inference-width \"${CHEK_STEREO_INFERENCE_WIDTH}\" --inference-height \"${CHEK_STEREO_INFERENCE_HEIGHT}\" --model-mode \"${CHEK_STEREO_MODEL_MODE}\" --body-score-threshold \"${CHEK_STEREO_BODY_SCORE_THRESHOLD}\" --joint-score-threshold \"${CHEK_STEREO_JOINT_SCORE_THRESHOLD}\" --max-persons \"${CHEK_STEREO_MAX_PERSONS}\" --det-frequency \"${CHEK_STEREO_DET_FREQUENCY}\" --tracking-thr \"${CHEK_STEREO_TRACKING_THR}\" --parallel-inference-workers \"${CHEK_STEREO_PARALLEL_INFERENCE_WORKERS}\" --time-sync-interval-s \"${CHEK_STEREO_TIME_SYNC_INTERVAL_S}\" --time-sync-sample-count \"${CHEK_STEREO_TIME_SYNC_SAMPLE_COUNT}\" --time-sync-timeout-s \"${CHEK_STEREO_TIME_SYNC_TIMEOUT_S}\"${low_roi_args} --calibration-path \"${CHEK_STEREO_CALIBRATION_PATH}\" --fx-px \"${CHEK_STEREO_FX_PX}\" --fy-px \"${CHEK_STEREO_FY_PX}\" --cx-px \"${CHEK_STEREO_CX_PX}\" --cy-px \"${CHEK_STEREO_CY_PX}\" --baseline-m \"${CHEK_STEREO_BASELINE_M}\" --fps \"${CHEK_STEREO_FPS}\" --association-hint-url \"${CHEK_STEREO_ASSOCIATION_HINT_URL}\" --debug-preview \"${CHEK_STEREO_PREVIEW}\" --debug-left-frame \"${CHEK_STEREO_LEFT_FRAME}\" --debug-right-frame \"${CHEK_STEREO_RIGHT_FRAME}\""

if [[ "${CHEK_STEREO_UPLOAD_RAW_MEDIA}" == "1" ]]; then
  command="${command} --upload-raw-media --raw-chunk-duration-s \"${CHEK_STEREO_RAW_CHUNK_DURATION_S}\" --raw-upload-timeout-s \"${CHEK_STEREO_RAW_UPLOAD_TIMEOUT_S}\" --raw-upload-queue-size \"${CHEK_STEREO_RAW_UPLOAD_QUEUE_SIZE}\""
fi

run_edge_user_bash "${command}"
