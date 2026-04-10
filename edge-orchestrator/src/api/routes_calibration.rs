use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::process::Command;
use tokio::time::{sleep, Duration};

use crate::calibration::{solve_iphone_stereo_transform, solve_similarity_transform, PointPair};
use crate::operator::{canonicalize_body_points_3d, canonicalize_hand_points_3d};
use crate::AppState;

const IPHONE_STEREO_MAX_ACCEPTABLE_RMS_M: f32 = 0.25;
const IPHONE_STEREO_MAX_REGRESSION_FACTOR: f32 = 2.5;
const STEREO_BASELINE_MIN_FACTOR: f32 = 0.25;
const STEREO_BASELINE_MAX_FACTOR: f32 = 4.0;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/calibration/stereo/current", get(get_stereo_current))
        .route(
            "/calibration/stereo/solver/current",
            get(get_stereo_solver_current),
        )
        .route("/calibration/stereo/solve", post(post_stereo_solve))
        .route("/calibration/stereo/apply", post(post_apply_stereo))
        .route("/calibration/iphone-stereo/current", get(get_current))
        .route("/calibration/iphone-stereo/solve", post(post_solve))
        .route("/calibration/wifi-stereo/current", get(get_wifi_current))
        .route("/calibration/wifi-stereo/solve", post(post_wifi_solve))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct SolveRequest {
    sample_frames: Option<usize>,
    sample_interval_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct StereoSolveRequest {
    sample_frames: Option<usize>,
    sample_interval_ms: Option<u64>,
    board_squares_x: Option<usize>,
    board_squares_y: Option<usize>,
    square_size_mm: Option<f32>,
    marker_size_mm: Option<f32>,
    min_valid_pairs: Option<usize>,
    min_common_corners: Option<usize>,
    dictionary_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApplyStereoCalibrationRequest {
    calibration: StereoRuntimeCalibrationPayload,
    restart_service: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StereoCameraIntrinsicsPayload {
    fx_px: f32,
    fy_px: f32,
    cx_px: f32,
    cy_px: f32,
    reference_image_w: u32,
    reference_image_h: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StereoRuntimeCalibrationPayload {
    sensor_frame: String,
    operator_frame: String,
    extrinsic_version: String,
    baseline_m: f32,
    left_intrinsics: StereoCameraIntrinsicsPayload,
    right_intrinsics: StereoCameraIntrinsicsPayload,
    #[serde(default)]
    capture_image_w: Option<u32>,
    #[serde(default)]
    capture_image_h: Option<u32>,
    #[serde(default)]
    inference_image_w: Option<u32>,
    #[serde(default)]
    inference_image_h: Option<u32>,
    #[serde(default)]
    model_mode: Option<String>,
    #[serde(default)]
    solver_summary: Option<serde_json::Value>,
}

async fn get_current(State(state): State<AppState>) -> Json<serde_json::Value> {
    let calibration = state.iphone_stereo_calibration.snapshot();
    Json(json!({
        "available": calibration.is_some(),
        "path": state.iphone_stereo_calibration.path(),
        "calibration": calibration,
    }))
}

async fn get_stereo_current(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snapshot = state.stereo.snapshot(state.config.stereo_stale_ms);
    let runtime_path = stereo_runtime_calibration_path(&state);
    let runtime_calibration = load_runtime_stereo_calibration(&runtime_path);
    Json(json!({
        "available": snapshot.calibration.is_some(),
        "path": "live_stereo_packet.calibration",
        "runtime_path": runtime_path.display().to_string(),
        "runtime_available": runtime_calibration.is_some(),
        "fresh": snapshot.fresh,
        "device_id": snapshot.device_id,
        "body_space": snapshot.body_space,
        "last_edge_time_ns": snapshot.last_edge_time_ns,
        "stereo_confidence": snapshot.stereo_confidence,
        "calibration": snapshot.calibration,
        "runtime_calibration": runtime_calibration,
    }))
}

async fn get_stereo_solver_current(State(state): State<AppState>) -> Json<serde_json::Value> {
    let status_path = stereo_solver_status_path(&state);
    let status = fs::read_to_string(&status_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok());
    let now_ms = unix_time_ms();
    let status_updated_at_ms = status
        .as_ref()
        .and_then(|value| value.get("updated_at_ms"))
        .and_then(|value| value.as_u64());
    let status_age_ms = status_updated_at_ms.and_then(|updated| now_ms.checked_sub(updated));
    let status_fresh = status_age_ms.map(|age| age <= 5_000).unwrap_or(false);

    Json(json!({
        "available": status.is_some(),
        "status_json_path": status_path.display().to_string(),
        "status_updated_at_ms": status_updated_at_ms,
        "status_age_ms": status_age_ms,
        "status_fresh": status_fresh,
        "status": status,
    }))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

async fn get_wifi_current(State(state): State<AppState>) -> Json<serde_json::Value> {
    let calibration = state.wifi_stereo_calibration.snapshot();
    Json(json!({
        "available": calibration.is_some(),
        "path": state.wifi_stereo_calibration.path(),
        "calibration": calibration,
    }))
}

async fn post_apply_stereo(
    State(state): State<AppState>,
    Json(req): Json<ApplyStereoCalibrationRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    validate_stereo_runtime_calibration(&req.calibration)
        .map_err(|error| (StatusCode::BAD_REQUEST, error))?;

    let path = stereo_runtime_calibration_path(&state);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建双目标定目录失败: {error}"),
            )
        })?;
    }

    let raw = serde_json::to_vec_pretty(&req.calibration).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("序列化双目标定失败: {error}"),
        )
    })?;
    tokio::fs::write(&path, raw).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("保存双目标定失败: {error}"),
        )
    })?;

    let restarted = req.restart_service.unwrap_or(true);
    let requested_extrinsic_version = req.calibration.extrinsic_version.clone();
    if restarted {
        restart_stereo_service().await?;
        sleep(Duration::from_secs(2)).await;

        let mut live_calibration: Option<serde_json::Value> = None;
        let mut live_loaded = false;
        for _attempt in 0..12 {
            let snapshot = state
                .stereo
                .snapshot(state.config.stereo_stale_ms.max(1_000));
            live_calibration = snapshot.calibration.clone();
            if let Some(current) = live_calibration.as_ref() {
                if stereo_runtime_calibration_matches(current, &req.calibration) {
                    live_loaded = true;
                    break;
                }
            }
            sleep(Duration::from_millis(500)).await;
        }

        if !live_loaded {
            let live_extrinsic_version =
                read_string_field(live_calibration.as_ref(), "extrinsic_version")
                    .unwrap_or("<missing>");
            return Err((
                StatusCode::CONFLICT,
                format!(
                    "双目标定已写入 {} 并重启双目服务，但 live calibration 仍未切到请求的 extrinsic_version={}（当前 live 为 {}）。请检查 stereo producer 实际加载路径是否与 runtime_path 一致。",
                    path.display(),
                    requested_extrinsic_version,
                    live_extrinsic_version,
                ),
            ));
        }
    }

    Ok(Json(json!({
        "ok": true,
        "path": path.display().to_string(),
        "restarted": restarted,
        "service_name": "chek-edge-stereo.service",
        "calibration": req.calibration,
    })))
}

async fn post_stereo_solve(
    State(state): State<AppState>,
    Json(req): Json<StereoSolveRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sample_frames = req.sample_frames.unwrap_or(24).clamp(8, 60);
    let sample_interval_ms = req.sample_interval_ms.unwrap_or(350).clamp(80, 1_500);
    let board_squares_x = req.board_squares_x.unwrap_or(10).clamp(4, 32);
    let board_squares_y = req.board_squares_y.unwrap_or(7).clamp(4, 32);
    let square_size_mm = req.square_size_mm.unwrap_or(24.0).clamp(5.0, 500.0);
    let marker_size_mm = req.marker_size_mm.unwrap_or(18.0).clamp(3.0, 490.0);
    let min_valid_pairs = req.min_valid_pairs.unwrap_or(18).clamp(6, 80);
    let min_common_corners = req.min_common_corners.unwrap_or(12).clamp(4, 200);
    let dictionary_name = req
        .dictionary_name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("DICT_4X4_50");

    if marker_size_mm >= square_size_mm {
        return Err((
            StatusCode::BAD_REQUEST,
            "marker_size_mm 必须小于 square_size_mm".to_string(),
        ));
    }

    let runtime_path = stereo_runtime_calibration_path(&state);
    let runtime_calibration = load_runtime_stereo_calibration(&runtime_path);
    let snapshot = state
        .stereo
        .snapshot(state.config.stereo_stale_ms.max(1_000));
    let preferred_calibration = snapshot
        .calibration
        .as_ref()
        .or(runtime_calibration.as_ref());

    let preview_path = PathBuf::from(state.config.stereo_preview_path.trim());
    let left_image_path = PathBuf::from(state.config.stereo_left_frame_path.trim());
    let right_image_path = PathBuf::from(state.config.stereo_right_frame_path.trim());
    let high_res_frames_available = left_image_path.exists() && right_image_path.exists();
    if !high_res_frames_available && !preview_path.exists() {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "双目标定图像不存在：{}，且高分辨率左右目快照也不存在（{}, {}）。先确认 chek-edge-stereo.service 正在产出图像。",
                preview_path.display(),
                left_image_path.display(),
                right_image_path.display(),
            ),
        ));
    }

    let solver_path = PathBuf::from(state.config.stereo_calibration_solver_path.trim());
    let debug_dir = allocate_stereo_solver_debug_dir(&state).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("创建双目标定调试目录失败: {error}"),
        )
    })?;
    let status_path = stereo_solver_status_path(&state);
    if let Some(parent) = status_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建双目标定状态目录失败: {error}"),
            )
        })?;
    }
    let python_bin = state.config.stereo_calibration_python_bin.trim();
    let mut command = Command::new(if python_bin.is_empty() {
        "python3"
    } else {
        python_bin
    });
    command
        .arg(&solver_path)
        .arg("--preview-path")
        .arg(&preview_path)
        .arg("--left-image-path")
        .arg(&left_image_path)
        .arg("--right-image-path")
        .arg(&right_image_path)
        .arg("--debug-dir")
        .arg(&debug_dir)
        .arg("--status-json-path")
        .arg(&status_path)
        .arg("--sample-frames")
        .arg(sample_frames.to_string())
        .arg("--sample-interval-ms")
        .arg(sample_interval_ms.to_string())
        .arg("--board-squares-x")
        .arg(board_squares_x.to_string())
        .arg("--board-squares-y")
        .arg(board_squares_y.to_string())
        .arg("--square-size-mm")
        .arg(square_size_mm.to_string())
        .arg("--marker-size-mm")
        .arg(marker_size_mm.to_string())
        .arg("--min-valid-pairs")
        .arg(min_valid_pairs.to_string())
        .arg("--min-common-corners")
        .arg(min_common_corners.to_string())
        .arg("--dictionary-name")
        .arg(dictionary_name)
        .arg("--sensor-frame")
        .arg(
            read_string_field(preferred_calibration, "sensor_frame").unwrap_or("stereo_pair_frame"),
        )
        .arg("--operator-frame")
        .arg(read_string_field(preferred_calibration, "operator_frame").unwrap_or("operator_frame"))
        .arg("--extrinsic-version")
        .arg(read_string_field(preferred_calibration, "extrinsic_version").unwrap_or_default())
        .arg("--model-mode")
        .arg(read_string_field(preferred_calibration, "model_mode").unwrap_or_default())
        .arg("--capture-image-width")
        .arg(
            read_u32_field(preferred_calibration, "capture_image_w")
                .or_else(|| {
                    read_nested_u32_field(
                        preferred_calibration,
                        "left_intrinsics",
                        "reference_image_w",
                    )
                })
                .unwrap_or_default()
                .to_string(),
        )
        .arg("--capture-image-height")
        .arg(
            read_u32_field(preferred_calibration, "capture_image_h")
                .or_else(|| {
                    read_nested_u32_field(
                        preferred_calibration,
                        "left_intrinsics",
                        "reference_image_h",
                    )
                })
                .unwrap_or_default()
                .to_string(),
        )
        .arg("--inference-image-width")
        .arg(
            read_u32_field(preferred_calibration, "inference_image_w")
                .unwrap_or_default()
                .to_string(),
        )
        .arg("--inference-image-height")
        .arg(
            read_u32_field(preferred_calibration, "inference_image_h")
                .unwrap_or_default()
                .to_string(),
        );

    let output = command.output().await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "启动双目标定求解器失败（{} via {}）: {error}",
                solver_path.display(),
                if python_bin.is_empty() {
                    "python3"
                } else {
                    python_bin
                }
            ),
        )
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err((
            StatusCode::BAD_REQUEST,
            if detail.is_empty() {
                "双目标定求解失败。请确认标定板已进入预览且角点清晰。".to_string()
            } else {
                format!("双目标定求解失败：{detail}")
            },
        ));
    }

    let mut response = parse_solver_json_output(&output.stdout).map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!("双目标定求解器输出不是合法 JSON: {error}"),
        )
    })?;
    validate_solved_stereo_baseline(preferred_calibration, &response)?;
    if let Some(object) = response.as_object_mut() {
        object.insert(
            "runtime_path".to_string(),
            json!(runtime_path.display().to_string()),
        );
        object.insert(
            "preview_path".to_string(),
            json!(preview_path.display().to_string()),
        );
        object.insert(
            "left_image_path".to_string(),
            json!(left_image_path.display().to_string()),
        );
        object.insert(
            "right_image_path".to_string(),
            json!(right_image_path.display().to_string()),
        );
        object.insert(
            "high_res_frames_available".to_string(),
            json!(high_res_frames_available),
        );
        object.insert(
            "debug_dir".to_string(),
            json!(debug_dir.display().to_string()),
        );
        object.insert(
            "status_json_path".to_string(),
            json!(status_path.display().to_string()),
        );
        object.insert(
            "live_calibration_available".to_string(),
            json!(snapshot.calibration.is_some()),
        );
    }

    Ok(Json(response))
}

fn stereo_solver_status_path(state: &AppState) -> PathBuf {
    PathBuf::from(&state.config.data_dir)
        .join("runtime")
        .join("stereo_calibration_solver")
        .join("current_status.json")
}

fn parse_solver_json_output(raw_stdout: &[u8]) -> Result<serde_json::Value, String> {
    let stdout = String::from_utf8_lossy(raw_stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err("solver stdout 为空".to_string());
    }
    let json_start = trimmed
        .find('{')
        .ok_or_else(|| "solver stdout 中未找到 JSON 起始位置".to_string())?;
    let json_end = trimmed
        .rfind('}')
        .ok_or_else(|| "solver stdout 中未找到 JSON 结束位置".to_string())?;
    if json_end < json_start {
        return Err("solver stdout 中 JSON 起止位置异常".to_string());
    }
    serde_json::from_str::<serde_json::Value>(&trimmed[json_start..=json_end])
        .map_err(|error| format!("{error}; stdout={trimmed}"))
}

fn validate_solved_stereo_baseline(
    preferred_calibration: Option<&serde_json::Value>,
    response: &serde_json::Value,
) -> Result<(), (StatusCode, String)> {
    let expected_baseline = read_f32_field(preferred_calibration, "baseline_m");
    let solved_baseline = response
        .get("calibration")
        .and_then(|value| value.get("baseline_m"))
        .and_then(|value| value.as_f64())
        .map(|value| value as f32);
    let (Some(expected_baseline), Some(solved_baseline)) = (expected_baseline, solved_baseline)
    else {
        return Ok(());
    };
    if expected_baseline <= 0.0 || !expected_baseline.is_finite() || !solved_baseline.is_finite() {
        return Ok(());
    }
    let min_allowed = expected_baseline * STEREO_BASELINE_MIN_FACTOR;
    let max_allowed = expected_baseline * STEREO_BASELINE_MAX_FACTOR;
    if solved_baseline < min_allowed || solved_baseline > max_allowed {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "双目标定求解结果未通过基线校验：本次 baseline={:.3}m，当前参考 baseline={:.3}m，允许范围 [{:.3}, {:.3}]m。请重新采集更丰富的板姿态，暂勿 apply 本次结果。",
                solved_baseline,
                expected_baseline,
                min_allowed,
                max_allowed,
            ),
        ));
    }
    Ok(())
}

fn allocate_stereo_solver_debug_dir(state: &AppState) -> std::io::Result<PathBuf> {
    let root = PathBuf::from(&state.config.data_dir)
        .join("runtime")
        .join("stereo_calibration_solver");
    fs::create_dir_all(&root)?;
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let dir = root.join(format!("run-{now_ms}"));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

async fn post_solve(
    State(state): State<AppState>,
    Json(req): Json<SolveRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sample_frames = req.sample_frames.unwrap_or(12).clamp(4, 80);
    let sample_interval_ms = req.sample_interval_ms.unwrap_or(50).clamp(20, 250);
    let calibration_vision_stale_ms = state.config.vision_stale_ms.max(600);
    let calibration_stereo_stale_ms = state.config.stereo_stale_ms.max(1_000);
    let max_attempts = sample_frames.saturating_mul(80);

    let previous_calibration = state.iphone_stereo_calibration.snapshot();
    let mut pairs: Vec<PointPair> = Vec::new();
    let mut captured_frames = 0usize;
    let mut sampled_stereo_frames = 0usize;
    let mut last_sampled_stereo_edge_time_ns: Option<u64> = None;
    for index in 0..max_attempts {
        let vision = state.vision.snapshot(calibration_vision_stale_ms);
        let stereo = state.stereo.snapshot(calibration_stereo_stale_ms);
        let stereo_edge_time_ns = stereo.last_edge_time_ns;
        if stereo_edge_time_ns > 0 && last_sampled_stereo_edge_time_ns != Some(stereo_edge_time_ns)
        {
            last_sampled_stereo_edge_time_ns = Some(stereo_edge_time_ns);
            sampled_stereo_frames += 1;
            let frame_pairs = build_live_pairs(&state, &vision, &stereo);
            if !frame_pairs.is_empty() {
                captured_frames += 1;
                pairs.extend(frame_pairs);
                if captured_frames >= sample_frames {
                    break;
                }
            }
        }
        if index + 1 < max_attempts {
            sleep(Duration::from_millis(sample_interval_ms)).await;
        }
    }

    let solved = solve_iphone_stereo_transform(&pairs, state.gate.edge_time_ns())
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    if solved.rms_error_m > IPHONE_STEREO_MAX_ACCEPTABLE_RMS_M {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "iphone-stereo 标定未保存：RMS {:.3}m 超过上限 {:.3}m",
                solved.rms_error_m, IPHONE_STEREO_MAX_ACCEPTABLE_RMS_M
            ),
        ));
    }
    if let Some(previous) = previous_calibration.as_ref() {
        if previous.rms_error_m.is_finite()
            && solved.rms_error_m > previous.rms_error_m * IPHONE_STEREO_MAX_REGRESSION_FACTOR
        {
            return Err((
                StatusCode::CONFLICT,
                format!(
                    "iphone-stereo 标定未保存：RMS {:.3}m 相比现有 {:.3}m 退化过大",
                    solved.rms_error_m, previous.rms_error_m
                ),
            ));
        }
    }
    state
        .iphone_stereo_calibration
        .save(solved.clone())
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    Ok(Json(json!({
        "ok": true,
        "captured_frames": captured_frames,
        "sampled_stereo_frames": sampled_stereo_frames,
        "pair_count": pairs.len(),
        "calibration": solved,
    })))
}

async fn post_wifi_solve(
    State(state): State<AppState>,
    Json(req): Json<SolveRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sample_frames = req.sample_frames.unwrap_or(80).clamp(8, 320);
    let sample_interval_ms = req.sample_interval_ms.unwrap_or(50).clamp(20, 250);
    let calibration_wifi_stale_ms = state.config.wifi_pose_stale_ms.max(1_000);
    let calibration_stereo_stale_ms = state.config.stereo_stale_ms.max(1_000);

    let mut pairs: Vec<PointPair> = Vec::new();
    let mut captured_frames = 0usize;
    for index in 0..sample_frames {
        let wifi = state.wifi_pose.snapshot(calibration_wifi_stale_ms);
        let stereo = state.stereo.snapshot(calibration_stereo_stale_ms);
        let frame_pairs = build_live_pairs_wifi_stereo(&state, &wifi, &stereo);
        if !frame_pairs.is_empty() {
            captured_frames += 1;
            pairs.extend(frame_pairs);
        }
        if index + 1 < sample_frames {
            sleep(Duration::from_millis(sample_interval_ms)).await;
        }
    }

    let solved = solve_similarity_transform(&pairs, state.gate.edge_time_ns())
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    state
        .wifi_stereo_calibration
        .save(solved.clone())
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;

    Ok(Json(json!({
        "ok": true,
        "captured_frames": captured_frames,
        "pair_count": pairs.len(),
        "calibration": solved,
    })))
}

fn build_live_pairs(
    state: &AppState,
    vision: &crate::sensing::VisionSnapshot,
    stereo: &crate::sensing::StereoSnapshot,
) -> Vec<PointPair> {
    if !vision.fresh || !stereo.fresh {
        return Vec::new();
    }
    let _ = state;
    let vision_hand = canonicalize_hand_points_3d(&vision.hand_kpts_3d, vision.hand_layout);
    let stereo_body = canonicalize_body_points_3d(&stereo.body_kpts_3d, stereo.body_layout);

    build_iphone_stereo_pairs(&vision_hand, &stereo_body)
}

fn build_live_pairs_wifi_stereo(
    state: &AppState,
    wifi: &crate::sensing::WifiPoseSnapshot,
    stereo: &crate::sensing::StereoSnapshot,
) -> Vec<PointPair> {
    if !wifi.fresh || !stereo.fresh {
        return Vec::new();
    }
    let _ = state;
    let wifi_body = canonicalize_body_points_3d(&wifi.body_kpts_3d, wifi.body_layout);
    let stereo_body = canonicalize_body_points_3d(&stereo.body_kpts_3d, stereo.body_layout);

    build_wifi_stereo_pairs(&wifi_body, &stereo_body)
}

fn build_iphone_stereo_pairs(vision_hand: &[[f32; 3]], stereo_body: &[[f32; 3]]) -> Vec<PointPair> {
    let mut pairs = Vec::new();
    if let (Some(iphone_left), Some(stereo_left)) =
        (hand_wrist(vision_hand, true), body_joint(stereo_body, 9))
    {
        pairs.push(PointPair {
            source: iphone_left,
            target: stereo_left,
        });
    }
    if let (Some(iphone_right), Some(stereo_right)) =
        (hand_wrist(vision_hand, false), body_joint(stereo_body, 10))
    {
        pairs.push(PointPair {
            source: iphone_right,
            target: stereo_right,
        });
    }
    pairs
}

fn build_wifi_stereo_pairs(wifi_body: &[[f32; 3]], stereo_body: &[[f32; 3]]) -> Vec<PointPair> {
    let mut pairs = Vec::new();
    for index in [5usize, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] {
        if let (Some(wifi_point), Some(stereo_point)) =
            (body_joint(wifi_body, index), body_joint(stereo_body, index))
        {
            pairs.push(PointPair {
                source: wifi_point,
                target: stereo_point,
            });
        }
    }
    pairs
}

fn body_joint(points: &[[f32; 3]], index: usize) -> Option<[f32; 3]> {
    points.get(index).copied().filter(valid_point)
}

fn hand_wrist(points: &[[f32; 3]], is_left: bool) -> Option<[f32; 3]> {
    let index = if is_left { 0 } else { 21 };
    points.get(index).copied().filter(valid_point)
}

fn valid_point(point: &[f32; 3]) -> bool {
    point.iter().all(|value| value.is_finite()) && point.iter().any(|value| value.abs() > 1e-6)
}

fn stereo_runtime_calibration_path(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.stereo_runtime_calibration_path.trim())
}

fn load_runtime_stereo_calibration(path: &Path) -> Option<serde_json::Value> {
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
}

fn read_string_field<'a>(value: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    value
        .and_then(|entry| entry.get(key))
        .and_then(|entry| entry.as_str())
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
}

fn read_u32_field(value: Option<&serde_json::Value>, key: &str) -> Option<u32> {
    value
        .and_then(|entry| entry.get(key))
        .and_then(|entry| entry.as_u64())
        .and_then(|entry| u32::try_from(entry).ok())
}

fn read_f32_field(value: Option<&serde_json::Value>, key: &str) -> Option<f32> {
    value
        .and_then(|entry| entry.get(key))
        .and_then(|entry| entry.as_f64())
        .map(|entry| entry as f32)
        .filter(|entry| entry.is_finite())
}

fn read_nested_u32_field(
    value: Option<&serde_json::Value>,
    outer_key: &str,
    inner_key: &str,
) -> Option<u32> {
    value
        .and_then(|entry| entry.get(outer_key))
        .and_then(|entry| entry.get(inner_key))
        .and_then(|entry| entry.as_u64())
        .and_then(|entry| u32::try_from(entry).ok())
}

fn validate_stereo_runtime_calibration(
    calibration: &StereoRuntimeCalibrationPayload,
) -> Result<(), String> {
    if calibration.sensor_frame.trim().is_empty() {
        return Err("sensor_frame 不能为空".to_string());
    }
    if calibration.operator_frame.trim().is_empty() {
        return Err("operator_frame 不能为空".to_string());
    }
    if calibration.extrinsic_version.trim().is_empty() {
        return Err("extrinsic_version 不能为空".to_string());
    }
    if !calibration.baseline_m.is_finite() || calibration.baseline_m <= 0.0 {
        return Err("baseline_m 必须是正数".to_string());
    }

    validate_intrinsics("left_intrinsics", &calibration.left_intrinsics)?;
    validate_intrinsics("right_intrinsics", &calibration.right_intrinsics)?;

    for (label, value) in [
        ("capture_image_w", calibration.capture_image_w),
        ("capture_image_h", calibration.capture_image_h),
        ("inference_image_w", calibration.inference_image_w),
        ("inference_image_h", calibration.inference_image_h),
    ] {
        if let Some(number) = value {
            if number == 0 {
                return Err(format!("{label} 必须大于 0"));
            }
        }
    }

    Ok(())
}

fn validate_intrinsics(
    label: &str,
    intrinsics: &StereoCameraIntrinsicsPayload,
) -> Result<(), String> {
    if !intrinsics.fx_px.is_finite() || intrinsics.fx_px <= 0.0 {
        return Err(format!("{label}.fx_px 必须是正数"));
    }
    if !intrinsics.fy_px.is_finite() || intrinsics.fy_px <= 0.0 {
        return Err(format!("{label}.fy_px 必须是正数"));
    }
    if !intrinsics.cx_px.is_finite() {
        return Err(format!("{label}.cx_px 必须是有限值"));
    }
    if !intrinsics.cy_px.is_finite() {
        return Err(format!("{label}.cy_px 必须是有限值"));
    }
    if intrinsics.reference_image_w == 0 || intrinsics.reference_image_h == 0 {
        return Err(format!(
            "{label}.reference_image_w/reference_image_h 必须大于 0"
        ));
    }
    Ok(())
}

fn stereo_runtime_calibration_matches(
    live: &serde_json::Value,
    expected: &StereoRuntimeCalibrationPayload,
) -> bool {
    read_string_field(Some(live), "sensor_frame") == Some(expected.sensor_frame.as_str())
        && read_string_field(Some(live), "operator_frame") == Some(expected.operator_frame.as_str())
        && read_string_field(Some(live), "extrinsic_version")
            == Some(expected.extrinsic_version.as_str())
        && json_f32_matches(Some(live), "baseline_m", expected.baseline_m)
        && json_nested_f32_matches(
            Some(live),
            "left_intrinsics",
            "fx_px",
            expected.left_intrinsics.fx_px,
        )
        && json_nested_f32_matches(
            Some(live),
            "left_intrinsics",
            "fy_px",
            expected.left_intrinsics.fy_px,
        )
        && json_nested_f32_matches(
            Some(live),
            "right_intrinsics",
            "fx_px",
            expected.right_intrinsics.fx_px,
        )
        && json_nested_f32_matches(
            Some(live),
            "right_intrinsics",
            "fy_px",
            expected.right_intrinsics.fy_px,
        )
}

fn json_f32_matches(value: Option<&serde_json::Value>, key: &str, expected: f32) -> bool {
    value
        .and_then(|entry| entry.get(key))
        .and_then(|entry| entry.as_f64())
        .map(|current| (current - f64::from(expected)).abs() <= 1e-4)
        .unwrap_or(false)
}

fn json_nested_f32_matches(
    value: Option<&serde_json::Value>,
    outer_key: &str,
    inner_key: &str,
    expected: f32,
) -> bool {
    value
        .and_then(|entry| entry.get(outer_key))
        .and_then(|entry| entry.get(inner_key))
        .and_then(|entry| entry.as_f64())
        .map(|current| (current - f64::from(expected)).abs() <= 1e-4)
        .unwrap_or(false)
}

async fn restart_stereo_service() -> Result<(), (StatusCode, String)> {
    let output = Command::new("bash")
        .arg("-lc")
        .arg("set -euo pipefail; sudo -n systemctl restart chek-edge-stereo.service")
        .output()
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("重启双目服务失败: {error}"),
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err((
            StatusCode::BAD_GATEWAY,
            if stderr.is_empty() {
                "重启双目服务失败。".to_string()
            } else {
                format!("重启双目服务失败：{stderr}")
            },
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        build_iphone_stereo_pairs, build_wifi_stereo_pairs, validate_stereo_runtime_calibration,
        StereoCameraIntrinsicsPayload, StereoRuntimeCalibrationPayload,
    };

    #[test]
    fn iphone_stereo_pairs_should_use_raw_hand_wrist_points() {
        let mut vision_hand = vec![[0.0_f32; 3]; 42];
        vision_hand[0] = [0.3, 0.2, 0.8];
        vision_hand[21] = [-0.4, 0.1, 0.9];
        let mut stereo_body = vec![[0.0_f32; 3]; 17];
        stereo_body[9] = [0.35, 0.25, 0.82];
        stereo_body[10] = [-0.45, 0.12, 0.88];

        let pairs = build_iphone_stereo_pairs(&vision_hand, &stereo_body);

        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].source, [0.3, 0.2, 0.8]);
        assert_eq!(pairs[0].target, [0.35, 0.25, 0.82]);
        assert_eq!(pairs[1].source, [-0.4, 0.1, 0.9]);
        assert_eq!(pairs[1].target, [-0.45, 0.12, 0.88]);
    }

    #[test]
    fn wifi_stereo_pairs_should_keep_matching_body_indices() {
        let mut wifi_body = vec![[0.0_f32; 3]; 17];
        let mut stereo_body = vec![[0.0_f32; 3]; 17];
        for index in [5usize, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] {
            wifi_body[index] = [index as f32, 1.0, 2.0];
            stereo_body[index] = [index as f32 + 0.1, 1.1, 2.1];
        }

        let pairs = build_wifi_stereo_pairs(&wifi_body, &stereo_body);

        assert_eq!(pairs.len(), 12);
        assert_eq!(pairs.first().map(|pair| pair.source), Some([5.0, 1.0, 2.0]));
        assert_eq!(pairs.last().map(|pair| pair.target), Some([16.1, 1.1, 2.1]));
    }

    fn sample_runtime_calibration() -> StereoRuntimeCalibrationPayload {
        StereoRuntimeCalibrationPayload {
            sensor_frame: "stereo_pair_frame".to_string(),
            operator_frame: "operator_frame".to_string(),
            extrinsic_version: "stereo-live-v1".to_string(),
            baseline_m: 0.06,
            left_intrinsics: StereoCameraIntrinsicsPayload {
                fx_px: 1200.0,
                fy_px: 1200.0,
                cx_px: 960.0,
                cy_px: 760.0,
                reference_image_w: 1920,
                reference_image_h: 1520,
            },
            right_intrinsics: StereoCameraIntrinsicsPayload {
                fx_px: 1200.0,
                fy_px: 1200.0,
                cx_px: 960.0,
                cy_px: 760.0,
                reference_image_w: 1920,
                reference_image_h: 1520,
            },
            capture_image_w: Some(1920),
            capture_image_h: Some(1520),
            inference_image_w: Some(640),
            inference_image_h: Some(506),
            model_mode: Some("lightweight".to_string()),
            solver_summary: Some(json!({
                "stereo_rms_px": 0.42
            })),
        }
    }

    #[test]
    fn stereo_runtime_calibration_validation_accepts_well_formed_payload() {
        let calibration = sample_runtime_calibration();
        assert!(validate_stereo_runtime_calibration(&calibration).is_ok());
    }

    #[test]
    fn stereo_runtime_calibration_validation_rejects_non_positive_baseline() {
        let mut calibration = sample_runtime_calibration();
        calibration.baseline_m = 0.0;

        let error =
            validate_stereo_runtime_calibration(&calibration).expect_err("baseline should fail");

        assert!(error.contains("baseline_m"));
    }
}
