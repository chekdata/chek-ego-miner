use axum::extract::{DefaultBodyLimit, State};
use axum::{routing::post, Json, Router};
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::operator::{CANONICAL_BODY_FRAME, STEREO_PAIR_FRAME};
use crate::sensing::{
    BodyKeypointLayout, HandKeypointLayout, VisionDevicePose, VisionImuSample, WifiPoseDiagnostics,
};
use crate::AppState;

const PHONE_VISION_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Edge 本地 ingest：双目 3D pose（用于无 ROS2/gRPC/SDK 时的联调入口）。
///
/// 注意：这是“工程联调口”，不是公网接口；默认仍受 `EDGE_TOKEN` 鉴权保护。
#[derive(Deserialize)]
pub struct StereoPoseIngestRequest {
    pub schema_version: String,
    pub trip_id: Option<String>,
    pub session_id: Option<String>,
    pub device_id: Option<String>,
    pub operator_track_id: Option<String>,
    pub source_time_ns: u64,
    pub left_frame_id: u64,
    pub right_frame_id: u64,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    pub left_body_kpts_2d: Option<Vec<[f32; 2]>>,
    pub right_body_kpts_2d: Option<Vec<[f32; 2]>>,
    pub body_layout: Option<String>,
    pub hand_layout: Option<String>,
    pub body_space: Option<String>,
    pub hand_space: Option<String>,
    pub calibration: Option<serde_json::Value>,
    pub stereo_confidence: f32,
    pub persons: Option<Vec<StereoPosePersonIngest>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StereoPosePersonIngest {
    pub operator_track_id: Option<String>,
    pub body_kpts_3d: Vec<[f32; 3]>,
    #[serde(default)]
    pub hand_kpts_3d: Vec<[f32; 3]>,
    pub left_body_kpts_2d: Option<Vec<[f32; 2]>>,
    pub right_body_kpts_2d: Option<Vec<[f32; 2]>>,
    pub stereo_confidence: Option<f32>,
    #[serde(default)]
    pub selection: Option<StereoPoseSelectionIngest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StereoPoseSelectionIngest {
    pub selection_reason: Option<String>,
    pub source_tag_left: Option<String>,
    pub source_tag_right: Option<String>,
    pub hand_hint_gap_m: Option<f32>,
    pub continuity_gap_m: Option<f32>,
}

#[derive(Deserialize)]
pub struct WifiPoseIngestRequest {
    pub schema_version: String,
    pub trip_id: Option<String>,
    pub session_id: Option<String>,
    pub device_id: Option<String>,
    pub operator_track_id: Option<String>,
    pub source_time_ns: u64,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub body_layout: Option<String>,
    pub body_confidence: f32,
    pub source_label: Option<String>,
    pub person_id: Option<u32>,
    pub total_persons: Option<u32>,
    pub body_space: Option<String>,
    pub raw_body_kpts_3d: Option<Vec<[f32; 3]>>,
    pub raw_body_layout: Option<String>,
    pub raw_body_space: Option<String>,
    pub calibration: Option<serde_json::Value>,
    pub diagnostics: Option<WifiPoseDiagnosticsIngest>,
}

#[derive(Deserialize, Serialize)]
pub struct WifiPoseDiagnosticsIngest {
    pub layout_node_count: Option<usize>,
    pub layout_score: Option<f32>,
    pub zone_score: Option<f32>,
    pub zone_summary_reliable: Option<bool>,
    pub motion_energy: Option<f32>,
    pub doppler_hz: Option<f32>,
    pub signal_quality: Option<f32>,
    pub vital_signal_quality: Option<f32>,
    pub stream_fps: Option<f32>,
    pub lifecycle_state: Option<String>,
    pub coherence_gate_decision: Option<String>,
    pub target_space: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PhoneVisionCameraCalibration {
    #[serde(alias = "fxPx")]
    pub fx_px: f32,
    #[serde(alias = "fyPx")]
    pub fy_px: f32,
    #[serde(alias = "cxPx")]
    pub cx_px: f32,
    #[serde(alias = "cyPx")]
    pub cy_px: f32,
    #[serde(alias = "referenceImageW")]
    pub reference_image_w: u32,
    #[serde(alias = "referenceImageH")]
    pub reference_image_h: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PhoneVisionFrameIngestRequest {
    pub schema_version: String,
    #[serde(alias = "tripId")]
    pub trip_id: Option<String>,
    #[serde(alias = "sessionId")]
    pub session_id: Option<String>,
    #[serde(alias = "deviceId")]
    pub device_id: Option<String>,
    #[serde(alias = "operatorTrackId")]
    pub operator_track_id: Option<String>,
    #[serde(alias = "sourceTimeNs")]
    pub source_time_ns: u64,
    #[serde(alias = "frameId")]
    pub frame_id: u64,
    #[serde(alias = "cameraMode")]
    pub camera_mode: String,
    #[serde(alias = "imageW", alias = "imageWidth")]
    pub image_w: u32,
    #[serde(alias = "imageH", alias = "imageHeight")]
    pub image_h: u32,
    #[serde(alias = "sensorImageW", alias = "sensorWidth")]
    pub sensor_image_w: u32,
    #[serde(alias = "sensorImageH", alias = "sensorHeight")]
    pub sensor_image_h: u32,
    #[serde(alias = "normalizedWasRotatedRight")]
    pub normalized_was_rotated_right: bool,
    #[serde(alias = "cameraHasDepth")]
    pub camera_has_depth: bool,
    #[serde(alias = "cameraCalibration")]
    pub camera_calibration: Option<PhoneVisionCameraCalibration>,
    #[serde(
        alias = "devicePose",
        alias = "phonePose",
        alias = "cameraPose",
        alias = "worldPose"
    )]
    pub device_pose: Option<VisionDevicePose>,
    #[serde(alias = "deviceMotion", alias = "motion")]
    pub imu: Option<VisionImuSample>,
    #[serde(alias = "primaryImageJpegB64")]
    pub primary_image_jpeg_b64: String,
    #[serde(alias = "auxImageJpegB64")]
    pub aux_image_jpeg_b64: Option<String>,
    #[serde(alias = "depthF32B64")]
    pub depth_f32_b64: Option<String>,
    #[serde(alias = "depthW", alias = "depthWidth")]
    pub depth_w: Option<u32>,
    #[serde(alias = "depthH", alias = "depthHeight")]
    pub depth_h: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PhoneVisionServiceError {
    pub code: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct PhoneVisionServiceResponse {
    pub ok: bool,
    pub capture_pose_packet: Option<serde_json::Value>,
    pub diagnostics: Option<serde_json::Value>,
    pub error: Option<PhoneVisionServiceError>,
}

fn normalized_optional_string(value: Option<&str>) -> String {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("")
        .to_string()
}

fn build_phone_ingress_status_update(
    req: &PhoneVisionFrameIngestRequest,
    edge_time_ns: u64,
    effective_trip_id: Option<&str>,
    effective_session_id: Option<&str>,
    device_id: Option<&str>,
    operator_track_id: Option<&str>,
) -> crate::control::gate::PhoneIngressStatusUpdate {
    crate::control::gate::PhoneIngressStatusUpdate {
        edge_time_ns,
        frame_id: Some(req.frame_id),
        request_trip_id: normalized_optional_string(req.trip_id.as_deref()),
        request_session_id: normalized_optional_string(req.session_id.as_deref()),
        effective_trip_id: normalized_optional_string(effective_trip_id),
        effective_session_id: normalized_optional_string(effective_session_id),
        device_id: normalized_optional_string(device_id.or(req.device_id.as_deref())),
        operator_track_id: normalized_optional_string(
            operator_track_id.or(req.operator_track_id.as_deref()),
        ),
        camera_mode: req.camera_mode.trim().to_string(),
        camera_has_depth: Some(req.camera_has_depth),
    }
}

fn build_phone_passthrough_packet(
    req: &PhoneVisionFrameIngestRequest,
    trip_id: &str,
    session_id: &str,
) -> serde_json::Value {
    let operator_track_id = req
        .operator_track_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("primary_operator");
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": req.device_id.clone().unwrap_or_default(),
        "operator_track_id": operator_track_id,
        "source_time_ns": req.source_time_ns,
        "frame_id": req.frame_id,
        "camera": {
            "mode": req.camera_mode,
            "image_w": req.image_w,
            "image_h": req.image_h,
            "sensor_image_w": req.sensor_image_w,
            "sensor_image_h": req.sensor_image_h,
            "normalized_was_rotated_right": req.normalized_was_rotated_right,
            "has_depth": req.camera_has_depth,
            "calibration": req.camera_calibration,
        },
        "capture_profile": {
            "body_3d_source": "none",
            "hand_3d_source": "none",
            "execution_mode": "device_pose_passthrough",
            "vision_processing_enabled": false,
        },
        "confidence": {
            "body": 0.0,
            "hand": 0.0,
        },
        "device_pose": req.device_pose,
        "imu": req.imu,
        "body_kpts_2d": [],
        "hand_kpts_2d": [],
        "body_kpts_3d": [],
        "hand_kpts_3d": [],
    })
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/ingest/stereo_pose", post(post_stereo_pose))
        .route("/ingest/wifi_pose", post(post_wifi_pose))
        .route("/ingest/phone_vision_frame", post(post_phone_vision_frame))
        .layer(DefaultBodyLimit::max(PHONE_VISION_MAX_BODY_BYTES))
        .with_state(state)
}

async fn post_stereo_pose(
    State(state): State<AppState>,
    Json(req): Json<StereoPoseIngestRequest>,
) -> Json<serde_json::Value> {
    if req.schema_version != "1.0.0" {
        return Json(serde_json::json!({
            "ok": false,
            "error": { "code": "invalid_schema_version", "message": "schema_version 必须为 1.0.0" }
        }));
    }

    let recv_time_ns = state.gate.edge_time_ns();
    let device_id = req.device_id.clone().unwrap_or_default();
    let edge_time_ns = if device_id.trim().is_empty() {
        recv_time_ns
    } else {
        state
            .gate
            .map_source_time_to_edge(device_id.as_str(), req.source_time_ns, recv_time_ns)
            .0
    };

    let stereo_body_space = req
        .body_space
        .clone()
        .or_else(|| {
            req.calibration
                .as_ref()
                .and_then(|value| value.get("sensor_frame"))
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| STEREO_PAIR_FRAME.to_string());
    let stereo_hand_space = req
        .hand_space
        .clone()
        .unwrap_or_else(|| stereo_body_space.clone());
    let stereo_persons = req
        .persons
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|person| crate::sensing::StereoTrackedPersonSnapshot {
            operator_track_id: person.operator_track_id,
            stereo_confidence: person
                .stereo_confidence
                .unwrap_or(req.stereo_confidence)
                .clamp(0.0, 1.0),
            body_kpts_3d: person.body_kpts_3d,
            hand_kpts_3d: person.hand_kpts_3d,
            left_body_kpts_2d: person.left_body_kpts_2d.unwrap_or_default(),
            right_body_kpts_2d: person.right_body_kpts_2d.unwrap_or_default(),
            selection_reason: person
                .selection
                .as_ref()
                .and_then(|selection| selection.selection_reason.clone())
                .unwrap_or_default(),
            source_tag_left: person
                .selection
                .as_ref()
                .and_then(|selection| selection.source_tag_left.clone())
                .unwrap_or_default(),
            source_tag_right: person
                .selection
                .as_ref()
                .and_then(|selection| selection.source_tag_right.clone())
                .unwrap_or_default(),
            hand_hint_gap_m: person
                .selection
                .as_ref()
                .and_then(|selection| selection.hand_hint_gap_m),
            continuity_gap_m: person
                .selection
                .as_ref()
                .and_then(|selection| selection.continuity_gap_m),
        })
        .collect::<Vec<_>>();

    // 1) 更新 stereo store（供融合回显/控制输出使用）
    let accepted_stereo_packet = state.stereo.ingest_pose3d(
        Some(device_id.clone()),
        req.calibration.clone(),
        req.body_kpts_3d.clone(),
        req.hand_kpts_3d.clone(),
        req.left_body_kpts_2d.clone().unwrap_or_default(),
        req.right_body_kpts_2d.clone().unwrap_or_default(),
        BodyKeypointLayout::resolve(req.body_layout.as_deref(), &[req.body_kpts_3d.len()]),
        HandKeypointLayout::resolve(req.hand_layout.as_deref(), &[req.hand_kpts_3d.len()]),
        stereo_body_space.clone(),
        stereo_hand_space.clone(),
        stereo_persons,
        req.operator_track_id.clone(),
        req.stereo_confidence,
        edge_time_ns,
        recv_time_ns,
    );
    if !accepted_stereo_packet {
        return Json(serde_json::json!({
            "ok": true,
            "ignored": true,
            "reason": "shadowed_by_real_stereo"
        }));
    }

    // 2) 会话内落盘（raw/stereo/pose3d.jsonl）
    let mut trip_id = req.trip_id.unwrap_or_default();
    let mut session_id = req.session_id.unwrap_or_default();
    if trip_id.trim().is_empty() || session_id.trim().is_empty() {
        let snap = state.session.snapshot();
        trip_id = snap.trip_id;
        session_id = snap.session_id;
    }

    if !trip_id.trim().is_empty() && !session_id.trim().is_empty() {
        let v = serde_json::json!({
            "type": "stereo_pose_packet",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "operator_track_id": req.operator_track_id,
            "source_time_ns": req.source_time_ns,
            "left_frame_id": req.left_frame_id,
            "right_frame_id": req.right_frame_id,
            "recv_time_ns": recv_time_ns,
            "edge_time_ns": edge_time_ns,
            "body_kpts_3d": req.body_kpts_3d,
            "hand_kpts_3d": req.hand_kpts_3d,
            "left_body_kpts_2d": req.left_body_kpts_2d,
            "right_body_kpts_2d": req.right_body_kpts_2d,
            "body_layout": req.body_layout,
            "hand_layout": req.hand_layout,
            "body_space": stereo_body_space,
            "hand_space": stereo_hand_space,
            "calibration": req.calibration,
            "stereo_confidence": req.stereo_confidence,
            "persons": req.persons
        });
        let recorder = state.recorder.clone();
        let protocol = state.protocol.clone();
        let config = state.config.clone();
        let trip_id_for_record = trip_id.clone();
        let session_id_for_record = session_id.clone();
        tokio::spawn(async move {
            recorder
                .record_stereo_pose3d(
                    &protocol,
                    &config,
                    &trip_id_for_record,
                    &session_id_for_record,
                    &v,
                )
                .await;
        });
    }

    Json(serde_json::json!({ "ok": true }))
}

async fn post_wifi_pose(
    State(state): State<AppState>,
    Json(req): Json<WifiPoseIngestRequest>,
) -> Json<serde_json::Value> {
    if req.schema_version != "1.0.0" {
        return Json(serde_json::json!({
            "ok": false,
            "error": { "code": "invalid_schema_version", "message": "schema_version 必须为 1.0.0" }
        }));
    }

    let recv_time_ns = state.gate.edge_time_ns();
    let device_id = req.device_id.clone().unwrap_or_default();
    let edge_time_ns = if device_id.trim().is_empty() {
        recv_time_ns
    } else {
        state
            .gate
            .map_source_time_to_edge(device_id.as_str(), req.source_time_ns, recv_time_ns)
            .0
    };
    let body_layout =
        BodyKeypointLayout::resolve(req.body_layout.as_deref(), &[req.body_kpts_3d.len()]);

    state.wifi_pose.ingest_pose3d(
        req.body_kpts_3d.clone(),
        body_layout,
        req.body_space
            .clone()
            .unwrap_or_else(|| CANONICAL_BODY_FRAME.to_string()),
        req.operator_track_id.clone(),
        req.body_confidence,
        req.diagnostics
            .as_ref()
            .map(|diag| WifiPoseDiagnostics {
                layout_node_count: diag.layout_node_count.unwrap_or(0),
                layout_score: diag.layout_score.unwrap_or(0.0).clamp(0.0, 1.0),
                zone_score: diag.zone_score.unwrap_or(0.0).clamp(0.0, 1.0),
                zone_summary_reliable: diag.zone_summary_reliable.unwrap_or(false),
                motion_energy: diag.motion_energy.unwrap_or(0.0).max(0.0),
                doppler_hz: diag.doppler_hz.unwrap_or(0.0).abs(),
                signal_quality: diag.signal_quality.unwrap_or(0.0).clamp(0.0, 1.0),
                vital_signal_quality: diag.vital_signal_quality.map(|v| v.clamp(0.0, 1.0)),
                stream_fps: diag.stream_fps.unwrap_or(0.0).max(0.0),
                lifecycle_state: diag.lifecycle_state.clone().unwrap_or_default(),
                coherence_gate_decision: diag.coherence_gate_decision.clone().unwrap_or_default(),
                target_space: diag.target_space.clone().unwrap_or_else(|| {
                    req.body_space
                        .clone()
                        .unwrap_or_else(|| CANONICAL_BODY_FRAME.to_string())
                }),
            })
            .unwrap_or_default(),
        edge_time_ns,
        recv_time_ns,
    );

    let mut trip_id = req.trip_id.unwrap_or_default();
    let mut session_id = req.session_id.unwrap_or_default();
    if trip_id.trim().is_empty() || session_id.trim().is_empty() {
        let snap = state.session.snapshot();
        trip_id = snap.trip_id;
        session_id = snap.session_id;
    }

    if !trip_id.trim().is_empty() && !session_id.trim().is_empty() {
        let v = serde_json::json!({
            "type": "wifi_pose_packet",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "operator_track_id": req.operator_track_id,
            "source_time_ns": req.source_time_ns,
            "recv_time_ns": recv_time_ns,
            "edge_time_ns": edge_time_ns,
            "body_layout": body_layout.as_str(),
            "body_space": req.body_space,
            "body_kpts_3d": req.body_kpts_3d,
            "body_confidence": req.body_confidence,
            "source_label": req.source_label,
            "person_id": req.person_id,
            "total_persons": req.total_persons,
            "raw_body_layout": req.raw_body_layout,
            "raw_body_space": req.raw_body_space,
            "raw_body_kpts_3d": req.raw_body_kpts_3d,
            "calibration": req.calibration,
            "diagnostics": req.diagnostics,
        });
        let recorder = state.recorder.clone();
        let protocol = state.protocol.clone();
        let config = state.config.clone();
        let trip_id_for_record = trip_id.clone();
        let session_id_for_record = session_id.clone();
        tokio::spawn(async move {
            recorder
                .record_wifi_pose3d(
                    &protocol,
                    &config,
                    &trip_id_for_record,
                    &session_id_for_record,
                    &v,
                )
                .await;
        });
    }

    Json(serde_json::json!({ "ok": true }))
}

async fn post_phone_vision_frame(
    State(state): State<AppState>,
    Json(mut req): Json<PhoneVisionFrameIngestRequest>,
) -> Json<serde_json::Value> {
    let recv_time_ns = state.gate.edge_time_ns();
    tracing::info!(
        trip_id = req.trip_id.as_deref().unwrap_or(""),
        session_id = req.session_id.as_deref().unwrap_or(""),
        device_id = req.device_id.as_deref().unwrap_or(""),
        operator_track_id = req.operator_track_id.as_deref().unwrap_or(""),
        frame_id = req.frame_id,
        source_time_ns = req.source_time_ns,
        camera_mode = req.camera_mode,
        camera_has_depth = req.camera_has_depth,
        has_aux = req
            .aux_image_jpeg_b64
            .as_deref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false),
        has_depth_payload = req
            .depth_f32_b64
            .as_deref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false),
        "phone_vision_frame ingest received"
    );

    if req.schema_version != "1.0.0" {
        state.phone_ingress_status.record_error(
            build_phone_ingress_status_update(
                &req,
                recv_time_ns,
                None,
                None,
                req.device_id.as_deref(),
                req.operator_track_id.as_deref(),
            ),
            "invalid_schema_version",
            "schema_version 必须为 1.0.0".to_string(),
        );
        return Json(serde_json::json!({
            "ok": false,
            "error": { "code": "invalid_schema_version", "message": "schema_version 必须为 1.0.0" }
        }));
    }

    if req.primary_image_jpeg_b64.trim().is_empty() {
        state.phone_ingress_status.record_error(
            build_phone_ingress_status_update(
                &req,
                recv_time_ns,
                None,
                None,
                req.device_id.as_deref(),
                req.operator_track_id.as_deref(),
            ),
            "missing_primary_image",
            "primary_image_jpeg_b64 不能为空".to_string(),
        );
        return Json(serde_json::json!({
            "ok": false,
            "error": { "code": "missing_primary_image", "message": "primary_image_jpeg_b64 不能为空" }
        }));
    }

    if !state.config.phone_ingest_enabled {
        state.phone_ingress_status.record_rejected(
            build_phone_ingress_status_update(
                &req,
                recv_time_ns,
                None,
                None,
                req.device_id.as_deref(),
                req.operator_track_id.as_deref(),
            ),
            "phone_ingest_disabled",
            "phone vision ingest disabled by runtime profile",
        );
        metrics::counter!("phone_vision_frame_ingest_ignored_count").increment(1);
        return Json(serde_json::json!({
            "ok": false,
            "error": {
                "code": "phone_ingest_disabled",
                "message": "phone vision ingest disabled by runtime profile"
            }
        }));
    }

    if req.trip_id.as_deref().unwrap_or("").trim().is_empty()
        || req.session_id.as_deref().unwrap_or("").trim().is_empty()
    {
        let snap = state.session.snapshot();
        if req.trip_id.as_deref().unwrap_or("").trim().is_empty() {
            req.trip_id = Some(snap.trip_id);
        }
        if req.session_id.as_deref().unwrap_or("").trim().is_empty() {
            req.session_id = Some(snap.session_id);
        }
    }

    let Some(trip_id) = req
        .trip_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
    else {
        state.phone_ingress_status.record_error(
            build_phone_ingress_status_update(
                &req,
                recv_time_ns,
                None,
                None,
                req.device_id.as_deref(),
                req.operator_track_id.as_deref(),
            ),
            "missing_trip_id",
            "trip_id 不能为空".to_string(),
        );
        return Json(serde_json::json!({
            "ok": false,
            "error": { "code": "missing_trip_id", "message": "trip_id 不能为空" }
        }));
    };
    let Some(session_id) = req
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
    else {
        state.phone_ingress_status.record_error(
            build_phone_ingress_status_update(
                &req,
                recv_time_ns,
                Some(trip_id.as_str()),
                None,
                req.device_id.as_deref(),
                req.operator_track_id.as_deref(),
            ),
            "missing_session_id",
            "session_id 不能为空".to_string(),
        );
        return Json(serde_json::json!({
            "ok": false,
            "error": { "code": "missing_session_id", "message": "session_id 不能为空" }
        }));
    };

    state
        .session
        .set_active(trip_id.clone(), session_id.clone());

    let request_status = build_phone_ingress_status_update(
        &req,
        recv_time_ns,
        Some(trip_id.as_str()),
        Some(session_id.as_str()),
        req.device_id.as_deref(),
        req.operator_track_id.as_deref(),
    );
    state
        .phone_ingress_status
        .record_attempt(request_status.clone());
    let (mut packet, diagnostics) = if !state.config.phone_vision_processing_enabled {
        (
            build_phone_passthrough_packet(&req, &trip_id, &session_id),
            Some(serde_json::json!({
                "mode": "device_pose_passthrough",
                "vision_processing_enabled": false,
                "camera_mode": req.camera_mode,
            })),
        )
    } else {
        let service_url = format!(
            "{}/infer",
            state.config.phone_vision_service_base.trim_end_matches('/')
        );

        let service_response = match state.http_client.post(&service_url).json(&req).send().await {
            Ok(response) => {
                let status = response.status();
                let body = match response.text().await {
                    Ok(text) => text,
                    Err(error) => {
                        let message = error.to_string();
                        state.phone_ingress_status.record_error(
                            request_status.clone(),
                            "phone_vision_service_body_read_fail",
                            message.clone(),
                        );
                        return Json(serde_json::json!({
                            "ok": false,
                            "error": {
                                "code": "phone_vision_service_body_read_fail",
                                "message": message,
                                "service_url": service_url,
                            }
                        }));
                    }
                };
                if !status.is_success() {
                    let message = format!(
                        "status={} body_prefix={}",
                        status.as_u16(),
                        body.chars().take(300).collect::<String>()
                    );
                    state.phone_ingress_status.record_error(
                        request_status.clone(),
                        "phone_vision_service_http_fail",
                        message.clone(),
                    );
                    return Json(serde_json::json!({
                        "ok": false,
                        "error": {
                            "code": "phone_vision_service_http_fail",
                            "message": message,
                            "service_url": service_url,
                        }
                    }));
                }
                match serde_json::from_str::<PhoneVisionServiceResponse>(&body) {
                    Ok(payload) => payload,
                    Err(error) => {
                        let message = error.to_string();
                        let body_prefix = body.chars().take(300).collect::<String>();
                        state.phone_ingress_status.record_error(
                            request_status.clone(),
                            "phone_vision_service_bad_json",
                            format!("{message} body_prefix={body_prefix}"),
                        );
                        return Json(serde_json::json!({
                            "ok": false,
                            "error": {
                                "code": "phone_vision_service_bad_json",
                                "message": message,
                                "body_prefix": body_prefix,
                                "service_url": service_url,
                            }
                        }));
                    }
                }
            }
            Err(error) => {
                let message = error.to_string();
                state.phone_ingress_status.record_error(
                    request_status.clone(),
                    "phone_vision_service_unreachable",
                    message.clone(),
                );
                return Json(serde_json::json!({
                    "ok": false,
                    "error": {
                        "code": "phone_vision_service_unreachable",
                        "message": message,
                        "service_url": service_url,
                    }
                }));
            }
        };

        if !service_response.ok {
            let service_error = service_response.error.unwrap_or(PhoneVisionServiceError {
                code: Some("phone_vision_service_failed".to_string()),
                message: Some("phone vision service returned ok=false".to_string()),
            });
            let error_code = service_error
                .code
                .clone()
                .unwrap_or_else(|| "phone_vision_service_failed".to_string());
            let error_message = service_error
                .message
                .clone()
                .unwrap_or_else(|| "phone vision service returned ok=false".to_string());
            state.phone_ingress_status.record_error(
                request_status.clone(),
                error_code.as_str(),
                error_message,
            );
            return Json(serde_json::json!({
                "ok": false,
                "error": service_error,
                "diagnostics": service_response.diagnostics,
            }));
        }

        let Some(packet) = service_response.capture_pose_packet else {
            state.phone_ingress_status.record_error(
                request_status.clone(),
                "missing_capture_pose_packet",
                "phone vision service 没有返回 capture_pose_packet".to_string(),
            );
            return Json(serde_json::json!({
                "ok": false,
                "error": { "code": "missing_capture_pose_packet", "message": "phone vision service 没有返回 capture_pose_packet" }
            }));
        };

        (packet, service_response.diagnostics)
    };

    let device_id = packet
        .get("device_id")
        .and_then(|value| value.as_str())
        .or(req.device_id.as_deref())
        .unwrap_or("")
        .to_string();
    let operator_track_id = packet
        .get("operator_track_id")
        .and_then(|value| value.as_str())
        .or(req.operator_track_id.as_deref())
        .unwrap_or("")
        .to_string();
    let source_time_ns = packet
        .get("source_time_ns")
        .and_then(|value| value.as_u64())
        .unwrap_or(req.source_time_ns);
    let edge_time_ns = if device_id.trim().is_empty() {
        recv_time_ns
    } else {
        state
            .gate
            .map_source_time_to_edge(device_id.as_str(), source_time_ns, recv_time_ns)
            .0
    };
    let final_status = build_phone_ingress_status_update(
        &req,
        edge_time_ns,
        Some(trip_id.as_str()),
        Some(session_id.as_str()),
        Some(device_id.as_str()),
        Some(operator_track_id.as_str()),
    );

    let should_decode_preview =
        state.config.vlm_indexing_enabled || state.config.preview_generation_enabled;
    let (primary_image_bytes, primary_image_decode_error) = if should_decode_preview {
        match base64::engine::general_purpose::STANDARD.decode(req.primary_image_jpeg_b64.trim()) {
            Ok(bytes) => (Some(bytes), None),
            Err(error) => (None, Some(error.to_string())),
        }
    } else {
        (None, None)
    };
    let (depth_bytes, depth_decode_error) = match req
        .depth_f32_b64
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(payload) => match base64::engine::general_purpose::STANDARD.decode(payload) {
            Ok(bytes) => (Some(bytes), None),
            Err(error) => (None, Some(error.to_string())),
        },
        None => (None, None),
    };
    let depth_relpath = if depth_bytes.is_some() {
        Some(format!(
            "raw/iphone/wide/depth/frame_{:010}__{:020}.f32le",
            req.frame_id, req.source_time_ns
        ))
    } else {
        None
    };
    let raw_phone_input = serde_json::json!({
        "type": "phone_vision_input",
        "schema_version": "1.0.0",
        "trip_id": trip_id.clone(),
        "session_id": session_id.clone(),
        "device_id": req.device_id.clone().unwrap_or_else(|| device_id.clone()),
        "operator_track_id": req.operator_track_id.clone(),
        "source_time_ns": req.source_time_ns,
        "recv_time_ns": recv_time_ns,
        "edge_time_ns": edge_time_ns,
        "frame_id": req.frame_id,
        "camera_mode": req.camera_mode.clone(),
        "image_w": req.image_w,
        "image_h": req.image_h,
        "sensor_image_w": req.sensor_image_w,
        "sensor_image_h": req.sensor_image_h,
        "normalized_was_rotated_right": req.normalized_was_rotated_right,
        "camera_has_depth": req.camera_has_depth,
        "has_aux_image": req
            .aux_image_jpeg_b64
            .as_deref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false),
        "camera_calibration": req.camera_calibration.clone(),
        "device_pose": req.device_pose.clone(),
        "imu": req.imu.clone(),
        "primary_image_decode_error": primary_image_decode_error,
        "primary_image_byte_count": primary_image_bytes.as_ref().map(|value| value.len()),
        "depth_w": req.depth_w,
        "depth_h": req.depth_h,
        "depth_relpath": depth_relpath,
        "depth_payload_decode_error": depth_decode_error,
    });

    {
        let recorder = state.recorder.clone();
        let protocol = state.protocol.clone();
        let config = state.config.clone();
        let trip_id_for_record = trip_id.clone();
        let session_id_for_record = session_id.clone();
        let raw_phone_input_for_record = raw_phone_input.clone();
        let primary_image_bytes_for_record = primary_image_bytes.clone();
        let depth_bytes_for_record = depth_bytes.clone();
        tokio::spawn(async move {
            recorder
                .record_phone_vision_input(
                    &protocol,
                    &config,
                    &trip_id_for_record,
                    &session_id_for_record,
                    &raw_phone_input_for_record,
                    primary_image_bytes_for_record,
                    depth_bytes_for_record.as_deref(),
                )
                .await;
        });
    }

    if let Some(obj) = packet.as_object_mut() {
        obj.insert("trip_id".to_string(), serde_json::json!(trip_id));
        obj.insert("session_id".to_string(), serde_json::json!(session_id));
        obj.insert("recv_time_ns".to_string(), serde_json::json!(recv_time_ns));
        obj.insert("edge_time_ns".to_string(), serde_json::json!(edge_time_ns));
        if !obj.contains_key("diagnostics") {
            if let Some(service_diag) = diagnostics.clone() {
                obj.insert("diagnostics".to_string(), service_diag);
            }
        }
        if !obj.contains_key("device_pose") {
            if let Some(device_pose) = req.device_pose.as_ref() {
                if let Ok(value) = serde_json::to_value(device_pose) {
                    obj.insert("device_pose".to_string(), value);
                }
            }
        }
        if !obj.contains_key("imu") {
            if let Some(imu) = req.imu.as_ref() {
                if let Ok(value) = serde_json::to_value(imu) {
                    obj.insert("imu".to_string(), value);
                }
            }
        }
    }

    metrics::counter!("phone_vision_frame_ingest_count").increment(1);
    let accepted_capture_pose_packet =
        state
            .vision
            .ingest_capture_pose_json(&packet, edge_time_ns, recv_time_ns);
    if accepted_capture_pose_packet {
        state.phone_ingress_status.record_accepted(
            final_status.clone(),
            "capture_pose_packet accepted for recorder",
        );
        tracing::info!(
            trip_id,
            session_id,
            device_id = device_id,
            frame_id = req.frame_id,
            source_time_ns,
            edge_time_ns,
            "phone_vision_frame accepted for recorder"
        );
        let recorder = state.recorder.clone();
        let protocol = state.protocol.clone();
        let config = state.config.clone();
        let packet_for_record = packet.clone();
        let trip_id_for_log = trip_id.clone();
        let session_id_for_log = session_id.clone();
        let device_id_for_log = device_id.clone();
        let frame_id_for_log = req.frame_id;
        tokio::spawn(async move {
            recorder
                .record_capture_pose(&protocol, &config, &packet_for_record)
                .await;
            tracing::info!(
                trip_id = trip_id_for_log,
                session_id = session_id_for_log,
                device_id = device_id_for_log,
                frame_id = frame_id_for_log,
                source_time_ns,
                edge_time_ns,
                "phone_vision_frame recorder_write_completed"
            );
        });
    } else {
        state.phone_ingress_status.record_rejected(
            final_status.clone(),
            "rejected_by_vision_store",
            "vision store rejected capture_pose_packet",
        );
        metrics::counter!("phone_vision_frame_ingest_ignored_count").increment(1);
        tracing::warn!(
            trip_id,
            session_id,
            device_id = device_id,
            frame_id = req.frame_id,
            source_time_ns,
            edge_time_ns,
            "phone_vision_frame rejected by vision store"
        );
    }

    tracing::info!(
        trip_id,
        session_id,
        device_id = device_id,
        frame_id = req.frame_id,
        source_time_ns,
        edge_time_ns,
        accepted = accepted_capture_pose_packet,
        service_diag = ?diagnostics,
        "phone_vision_frame ingest completed"
    );

    Json(serde_json::json!({
        "ok": true,
        "accepted": accepted_capture_pose_packet,
        "capture_pose_packet": packet,
        "diagnostics": diagnostics,
    }))
}
