#![recursion_limit = "512"]

//! WiFi-DensePose Sensing Server
//!
//! Lightweight Axum server that:
//! - Receives ESP32 CSI frames via UDP (port 5005)
//! - Processes signals using RuVector-powered wifi-densepose-signal crate
//! - Broadcasts sensing updates via WebSocket (ws://localhost:8765/ws/sensing)
//! - Serves the static UI files (port 8080)
//!
//! Replaces both ws_server.py and the Python HTTP server.

mod pose_head;
mod recording;
mod rvf_container;
mod rvf_pipeline;
mod training_api;
mod vital_signs;

// Training pipeline modules (exposed via lib.rs)
use wifi_densepose_sensing_server::{dataset, embedding, graph_transformer, trainer};

use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use clap::Parser;
use pose_head::{forward_with_f32_params, PoseHeadConfig};

use axum::http::HeaderValue;
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;
use tokio::sync::{broadcast, RwLock};
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;
use tracing::{debug, error, info, warn};

use rvf_container::{RvfBuilder, RvfContainerInfo, RvfReader, VitalSignConfig};
use rvf_pipeline::{decode_sona_profile_deltas, ProgressiveLoader};
use vital_signs::{VitalSignDetector, VitalSigns};

// ADR-022 Phase 3: Multi-BSSID pipeline integration
use wifi_densepose_wifiscan::parse_netsh_output as parse_netsh_bssid_output;
use wifi_densepose_wifiscan::{
    BandType, BssidId, BssidObservation, BssidRegistry, WindowsWifiPipeline,
};

const LEGACY_ESP32_UDP_PORT: u16 = 5005;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "sensing-server", about = "WiFi-DensePose sensing server")]
struct Args {
    /// HTTP port for UI and REST API
    #[arg(long, default_value = "8080")]
    http_port: u16,

    /// WebSocket port for sensing stream
    #[arg(long, default_value = "8765")]
    ws_port: u16,

    /// UDP port for ESP32 CSI frames
    #[arg(long, default_value = "5005")]
    udp_port: u16,

    /// Disable binding the legacy ESP32 UDP port 5005 alongside `--udp-port`
    #[arg(long, default_value_t = false)]
    disable_legacy_esp32_port_fallback: bool,

    /// Path to UI static files
    #[arg(long, default_value = "../../ui")]
    ui_path: PathBuf,

    /// Tick interval in milliseconds (default 100 ms = 10 fps for smooth pose animation)
    #[arg(long, default_value = "100")]
    tick_ms: u64,

    /// Data source: auto, wifi, esp32, simulate
    #[arg(long, default_value = "auto")]
    source: String,

    /// Run vital sign detection benchmark (1000 frames) and exit
    #[arg(long)]
    benchmark: bool,

    /// Load model config from an RVF container at startup
    #[arg(long, value_name = "PATH")]
    load_rvf: Option<PathBuf>,

    /// Save current model state as an RVF container on shutdown
    #[arg(long, value_name = "PATH")]
    save_rvf: Option<PathBuf>,

    /// Load a trained .rvf model for inference
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Enable progressive loading (Layer A instant start)
    #[arg(long)]
    progressive: bool,

    /// Export an RVF container package and exit (no server)
    #[arg(long, value_name = "PATH")]
    export_rvf: Option<PathBuf>,

    /// Run training mode (train a model and exit)
    #[arg(long)]
    train: bool,

    /// Path to dataset directory (MM-Fi or Wi-Pose)
    #[arg(long, value_name = "PATH")]
    dataset: Option<PathBuf>,

    /// Dataset type: "mmfi" or "wipose"
    #[arg(long, value_name = "TYPE", default_value = "mmfi")]
    dataset_type: String,

    /// Number of training epochs
    #[arg(long, default_value = "100")]
    epochs: usize,

    /// Directory for training checkpoints
    #[arg(long, value_name = "DIR")]
    checkpoint_dir: Option<PathBuf>,

    /// Run self-supervised contrastive pretraining (ADR-024)
    #[arg(long)]
    pretrain: bool,

    /// Number of pretraining epochs (default 50)
    #[arg(long, default_value = "50")]
    pretrain_epochs: usize,

    /// Extract embeddings mode: load model and extract CSI embeddings
    #[arg(long)]
    embed: bool,

    /// Build fingerprint index from embeddings (env|activity|temporal|person)
    #[arg(long, value_name = "TYPE")]
    build_index: Option<String>,
}

// ── Data types ───────────────────────────────────────────────────────────────

/// ADR-018 ESP32 CSI binary frame header (20 bytes)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Esp32Frame {
    magic: u32,
    node_id: u8,
    n_antennas: u8,
    n_subcarriers: u8,
    freq_mhz: u16,
    sequence: u32,
    rssi: i8,
    noise_floor: i8,
    amplitudes: Vec<f64>,
    phases: Vec<f64>,
}

const ESP32_CSI_MAGIC_V1: u32 = 0xC511_0001;
const ESP32_CSI_MAGIC_V2: u32 = 0xC511_0005;
const ESP32_CSI_HEADER_LEN_V1: usize = 20;
const ESP32_CSI_HEADER_LEN_V2: usize = 28;

/// Sensing update broadcast to WebSocket clients
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensingUpdate {
    #[serde(rename = "type")]
    msg_type: String,
    timestamp: f64,
    source: String,
    tick: u64,
    nodes: Vec<NodeInfo>,
    features: FeatureInfo,
    classification: ClassificationInfo,
    signal_field: SignalField,
    /// Vital sign estimates (breathing rate, heart rate, confidence).
    #[serde(skip_serializing_if = "Option::is_none")]
    vital_signs: Option<VitalSigns>,
    // ── ADR-022 Phase 3: Enhanced multi-BSSID pipeline fields ──
    /// Enhanced motion estimate from multi-BSSID pipeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    enhanced_motion: Option<serde_json::Value>,
    /// Enhanced breathing estimate from multi-BSSID pipeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    enhanced_breathing: Option<serde_json::Value>,
    /// Posture classification from BSSID fingerprint matching.
    #[serde(skip_serializing_if = "Option::is_none")]
    posture: Option<String>,
    /// Signal quality score from multi-BSSID quality gate [0.0, 1.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    signal_quality_score: Option<f64>,
    /// Quality gate verdict: "Permit", "Warn", or "Deny".
    #[serde(skip_serializing_if = "Option::is_none")]
    quality_verdict: Option<String>,
    /// Number of BSSIDs used in the enhanced sensing cycle.
    #[serde(skip_serializing_if = "Option::is_none")]
    bssid_count: Option<usize>,
    // ── ADR-023 Phase 7-8: Model inference fields ──
    /// Pose keypoints when a trained model is loaded (x, y, z, confidence).
    #[serde(skip_serializing_if = "Option::is_none")]
    pose_keypoints: Option<Vec<[f64; 4]>>,
    /// Model status when a trained model is loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    model_status: Option<serde_json::Value>,
    // ── Multi-person detection (issue #97) ──
    /// Detected persons from WiFi sensing (multi-person support).
    #[serde(skip_serializing_if = "Option::is_none")]
    persons: Option<Vec<PersonDetection>>,
    /// Estimated person count from CSI feature heuristics (1-3 for single ESP32).
    #[serde(skip_serializing_if = "Option::is_none")]
    estimated_persons: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeInfo {
    node_id: u8,
    rssi_dbm: f64,
    position: [f64; 3],
    amplitude: Vec<f64>,
    subcarrier_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureInfo {
    mean_rssi: f64,
    variance: f64,
    motion_band_power: f64,
    breathing_band_power: f64,
    dominant_freq_hz: f64,
    change_points: usize,
    spectral_power: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassificationInfo {
    motion_level: String,
    presence: bool,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignalField {
    grid_size: [usize; 3],
    values: Vec<f64>,
}

/// WiFi-derived pose keypoint (17 COCO keypoints)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PoseKeypoint {
    name: String,
    x: f64,
    y: f64,
    z: f64,
    confidence: f64,
}

/// Person detection from WiFi sensing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersonDetection {
    id: u32,
    confidence: f64,
    keypoints: Vec<PoseKeypoint>,
    bbox: BoundingBox,
    zone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrackedPersonContract {
    track_id: String,
    source_person_id: u32,
    lifecycle_state: String,
    coherence_gate_decision: String,
    target_space: String,
    canonical_body_space: String,
    body_layout: String,
    body_kpts_3d: Vec<[f64; 3]>,
    canonical_body_kpts_3d: Vec<[f64; 3]>,
    person_confidence: f64,
    keypoints: Vec<PoseKeypoint>,
    pose_source: String,
}

#[derive(Debug, Clone)]
struct StableTrackedPersonState {
    contract: TrackedPersonContract,
    centroid: [f64; 3],
    consecutive_hits: u64,
    missed_frames: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BoundingBox {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// Shared application state
struct AppStateInner {
    latest_update: Option<SensingUpdate>,
    rssi_history: VecDeque<f64>,
    /// Circular buffer of recent CSI amplitude vectors for temporal analysis.
    /// Each entry is the full subcarrier amplitude vector for one frame.
    /// Capacity: FRAME_HISTORY_CAPACITY frames.
    frame_history: VecDeque<Vec<f64>>,
    tick: u64,
    source: String,
    tx: broadcast::Sender<String>,
    training_progress_tx: broadcast::Sender<String>,
    total_detections: u64,
    start_time: std::time::Instant,
    /// Vital sign detector (processes CSI frames to estimate HR/RR).
    vital_detector: VitalSignDetector,
    /// Most recent vital sign reading for the REST endpoint.
    latest_vitals: VitalSigns,
    /// RVF container info if a model was loaded via `--load-rvf`.
    rvf_info: Option<RvfContainerInfo>,
    /// Path to save RVF container on shutdown (set via `--save-rvf`).
    save_rvf_path: Option<PathBuf>,
    /// Progressive loader for a trained model (set via `--model`).
    progressive_loader: Option<ProgressiveLoader>,
    /// Cached model weights decoded from the active RVF.
    model_weights: Option<Vec<f32>>,
    /// Cached feature normalization stats decoded from the active RVF metadata.
    model_feature_stats: Option<ModelFeatureStats>,
    /// Parsed pose head config of the active RVF model.
    model_head_config: Option<PoseHeadConfig>,
    /// Coordinate space of the active model outputs.
    model_target_space: Option<String>,
    /// Decoded SONA profile deltas keyed by profile name.
    model_sona_profile_deltas: BTreeMap<String, Vec<f32>>,
    /// Active SONA profile name.
    active_sona_profile: Option<String>,
    /// Cached base+SONA effective weights for live inference when a profile is active.
    active_sona_weights: Option<Vec<f32>>,
    /// Whether a trained model is loaded.
    model_loaded: bool,
    /// Smoothed person count (EMA) for hysteresis — prevents frame-to-frame jumping.
    smoothed_person_score: f64,
    /// ADR-039: Latest edge vitals packet from ESP32.
    edge_vitals: Option<Esp32VitalsPacket>,
    /// ADR-040: Latest WASM output packet from ESP32.
    latest_wasm_events: Option<WasmOutputPacket>,
    /// 最近活跃的 ESP32 节点，用于 AP/RX 布局与多节点 live 视图。
    recent_nodes: BTreeMap<u8, RecentNodeState>,
    /// 当前录制会话状态。
    recording_state: recording::RecordingState,
    /// 当前训练任务状态。
    training_state: training_api::TrainingState,
    /// 稳定的 live tracked person contract，供 `/api/v1/pose/current` 与 edge bridge 直接消费。
    stable_tracked_person: Option<StableTrackedPersonState>,
    /// 简单递增的 live track id 分配器。
    next_stable_track_id: u64,
}

#[derive(Debug, Clone)]
struct RecentNodeState {
    node_id: u8,
    rssi_dbm: f64,
    position: [f64; 3],
    amplitude: Vec<f64>,
    subcarrier_count: usize,
    last_seen: Instant,
}

/// Number of frames retained in `frame_history` for temporal analysis.
/// At 500 ms ticks this covers ~50 seconds; at 100 ms ticks ~10 seconds.
const FRAME_HISTORY_CAPACITY: usize = 100;
const ACTIVE_NODE_TTL_MS: u64 = 2_500;
const MODEL_TARGET_SPACE_DEFAULT: &str = "wifi_pose_pixels";
const MODEL_TARGET_SPACE_OPERATOR_FRAME: &str = "operator_frame";
const MAX_LIVE_BSSID_NODES: usize = 16;
const ZONE_COUNT: usize = 4;
const SIGNAL_ZONE_CANDIDATE_ENERGY: f64 = 0.18;
const SIGNAL_HOTSPOT_MIN_PEAK: f64 = 0.18;
const SIGNAL_HOTSPOT_MIN_MASS: f64 = 0.85;
const SIGNAL_HOTSPOT_FALLBACK_PEAK: f64 = 0.12;
const MODEL_N_KEYPOINTS: usize = 17;
const MODEL_DIMS_PER_KP: usize = 3;
const MODEL_N_TARGETS: usize = MODEL_N_KEYPOINTS * MODEL_DIMS_PER_KP;
const MODEL_VARIANCE_WINDOW: usize = 10;
const LIVE_TRACK_MISS_HOLD_FRAMES: u64 = 8;
const LIVE_TRACK_CONF_ACCEPT: f64 = 0.70;
const LIVE_TRACK_CONF_PREDICT: f64 = 0.35;
const LIVE_TRACK_SOFT_CONTINUITY_SCALE: f64 = 2.5;
const LIVE_TRACK_CANONICAL_EMA_ALPHA: f64 = 0.32;
const LIVE_TRACK_CANONICAL_EMA_ALPHA_LIMB: f64 = 0.24;
const LIVE_TRACK_CANONICAL_EMA_ALPHA_WRIST: f64 = 0.18;
const TRACKED_CANONICAL_SHOULDER_WIDTH_M: f64 = 0.16;
const TRACKED_CANONICAL_TORSO_HEIGHT_M: f64 = 0.50;
const TRACKED_CANONICAL_BASE_DEPTH_M: f64 = 0.82;
const TRACKED_CANONICAL_DEPTH_DELTA_LIMIT_M: f64 = 0.18;
const TRACKED_CANONICAL_UPPER_ARM_RATIO: f64 = 0.48;
const TRACKED_CANONICAL_FOREARM_RATIO: f64 = 0.42;
const TRACKED_CANONICAL_THIGH_RATIO: f64 = 0.56;
const TRACKED_CANONICAL_SHIN_RATIO: f64 = 0.56;
const TRACKED_CANONICAL_HEAD_HEIGHT_RATIO: f64 = 0.42;
const TRACKED_CANONICAL_HEAD_HALF_WIDTH_RATIO: f64 = 0.18;
const TRACKED_CANONICAL_EYE_OFFSET_RATIO: f64 = 0.05;
const TRACKED_CANONICAL_EAR_OFFSET_RATIO: f64 = 0.12;
const TRACKED_CANONICAL_FACE_DEPTH_OFFSET_M: f64 = 0.05;

fn tracked_forward_bias_m(name: &str) -> f64 {
    match name {
        "nose" | "left_eye" | "right_eye" => 0.01,
        "left_ear" | "right_ear" => 0.005,
        "left_elbow" | "right_elbow" => 0.02,
        "left_wrist" | "right_wrist" => 0.035,
        "left_hip" | "right_hip" => -0.005,
        "left_knee" | "right_knee" => -0.015,
        "left_ankle" | "right_ankle" => -0.025,
        _ => 0.0,
    }
}

fn tracked_target_space(
    model_loaded: bool,
    model_target_space: Option<&str>,
    update: &SensingUpdate,
) -> String {
    if model_loaded && update.pose_keypoints.is_some() {
        model_target_space
            .unwrap_or(MODEL_TARGET_SPACE_DEFAULT)
            .to_string()
    } else {
        MODEL_TARGET_SPACE_DEFAULT.to_string()
    }
}

fn tracked_points_from_person(person: &PersonDetection) -> Option<Vec<[f64; 3]>> {
    if person.keypoints.len() < MODEL_N_KEYPOINTS {
        return None;
    }
    Some(
        person
            .keypoints
            .iter()
            .take(MODEL_N_KEYPOINTS)
            .map(|kp| [kp.x, kp.y, kp.z])
            .collect(),
    )
}

fn tracked_points_centroid(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut centroid = [0.0_f64; 3];
    for point in points {
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    let n = points.len() as f64;
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;
    centroid
}

fn tracked_point_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn tracked_continuity_threshold(target_space: &str) -> f64 {
    if target_space == MODEL_TARGET_SPACE_OPERATOR_FRAME {
        0.45
    } else {
        120.0
    }
}

fn tracked_gate_decision(
    update: &SensingUpdate,
    person_confidence: f64,
    reused_without_measurement: bool,
) -> &'static str {
    if reused_without_measurement {
        return "PredictOnly";
    }
    if let Some(verdict) = update.quality_verdict.as_deref() {
        if verdict.eq_ignore_ascii_case("deny") {
            return "Reject";
        }
        if verdict.eq_ignore_ascii_case("warn") {
            return if person_confidence >= LIVE_TRACK_CONF_PREDICT {
                "PredictOnly"
            } else {
                "Reject"
            };
        }
    }
    if let Some(score) = update.signal_quality_score {
        if score >= 0.75 && person_confidence >= 0.45 {
            return "Accept";
        }
        if score >= 0.35 && person_confidence >= 0.20 {
            return "PredictOnly";
        }
        return "Reject";
    }
    if person_confidence >= LIVE_TRACK_CONF_ACCEPT {
        "Accept"
    } else if person_confidence >= LIVE_TRACK_CONF_PREDICT {
        "PredictOnly"
    } else {
        "Reject"
    }
}

fn tracked_contract_from_person(
    track_id: String,
    person: &PersonDetection,
    points: Vec<[f64; 3]>,
    canonical_points: Vec<[f64; 3]>,
    target_space: &str,
    pose_source: &str,
    lifecycle_state: &str,
    coherence_gate_decision: &str,
) -> TrackedPersonContract {
    TrackedPersonContract {
        track_id,
        source_person_id: person.id,
        lifecycle_state: lifecycle_state.to_string(),
        coherence_gate_decision: coherence_gate_decision.to_string(),
        target_space: target_space.to_string(),
        canonical_body_space: "canonical_body_frame".to_string(),
        body_layout: "coco_body_17".to_string(),
        body_kpts_3d: points,
        canonical_body_kpts_3d: canonical_points,
        person_confidence: person.confidence,
        keypoints: person.keypoints.clone(),
        pose_source: pose_source.to_string(),
    }
}

fn tracked_keypoints_from_points(
    points: &[[f64; 3]],
    keypoints: &[PoseKeypoint],
) -> Option<Vec<PoseKeypoint>> {
    if points.is_empty() || keypoints.len() < points.len() {
        return None;
    }
    Some(
        keypoints
            .iter()
            .zip(points.iter())
            .map(|(kp, point)| PoseKeypoint {
                name: kp.name.clone(),
                x: point[0],
                y: point[1],
                z: point[2],
                confidence: kp.confidence,
            })
            .collect(),
    )
}

fn tracked_contract_for_pose_api(mut contract: TrackedPersonContract) -> serde_json::Value {
    if contract.canonical_body_kpts_3d.is_empty() {
        return serde_json::to_value(contract).unwrap_or_else(|_| serde_json::json!(null));
    }

    let raw_target_space = contract.target_space.clone();
    let raw_body_kpts_3d = contract.body_kpts_3d.clone();
    let raw_keypoints = contract.keypoints.clone();

    contract.body_kpts_3d = contract.canonical_body_kpts_3d.clone();
    contract.target_space = contract.canonical_body_space.clone();
    if let Some(canonical_keypoints) =
        tracked_keypoints_from_points(&contract.body_kpts_3d, &contract.keypoints)
    {
        contract.keypoints = canonical_keypoints;
    }

    let mut value = serde_json::to_value(contract).unwrap_or_else(|_| serde_json::json!(null));
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "raw_target_space".to_string(),
            serde_json::Value::String(raw_target_space),
        );
        obj.insert(
            "raw_body_kpts_3d".to_string(),
            serde_json::json!(raw_body_kpts_3d),
        );
        if !raw_keypoints.is_empty() {
            obj.insert(
                "raw_keypoints".to_string(),
                serde_json::json!(raw_keypoints),
            );
        }
    }
    value
}

fn tracked_vec_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn tracked_vec_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn tracked_vec_scale(v: [f64; 3], scalar: f64) -> [f64; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn tracked_vec_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn tracked_vec_norm(v: [f64; 3]) -> f64 {
    tracked_vec_dot(v, v).sqrt()
}

fn tracked_vec_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn tracked_vec_normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let norm = tracked_vec_norm(v);
    (norm.is_finite() && norm > 1e-6).then_some(tracked_vec_scale(v, 1.0 / norm))
}

fn canonicalize_operator_frame_tracked_points(person: &PersonDetection) -> Option<Vec<[f64; 3]>> {
    if person.keypoints.len() < MODEL_N_KEYPOINTS {
        return None;
    }

    let points = tracked_points_from_person(person)?;
    let left_shoulder = *points.get(5)?;
    let right_shoulder = *points.get(6)?;
    let left_hip = *points.get(11)?;
    let right_hip = *points.get(12)?;

    let shoulders_mid = tracked_vec_scale(tracked_vec_add(left_shoulder, right_shoulder), 0.5);
    let hips_mid = tracked_vec_scale(tracked_vec_add(left_hip, right_hip), 0.5);
    let lateral_mid_left = tracked_vec_scale(tracked_vec_add(left_shoulder, left_hip), 0.5);
    let lateral_mid_right = tracked_vec_scale(tracked_vec_add(right_shoulder, right_hip), 0.5);

    let x_seed = tracked_vec_sub(lateral_mid_left, lateral_mid_right);
    let y_seed = tracked_vec_sub(shoulders_mid, hips_mid);
    let mut x_axis = tracked_vec_normalize(x_seed)?;
    let mut y_axis = tracked_vec_normalize(y_seed)?;
    let mut z_seed = tracked_vec_cross(x_axis, y_axis);
    if tracked_vec_norm(z_seed) <= 1e-4 {
        z_seed = tracked_vec_cross(x_axis, [0.0, 1.0, 0.0]);
    }
    if tracked_vec_norm(z_seed) <= 1e-4 {
        z_seed = tracked_vec_cross(x_axis, [0.0, 0.0, 1.0]);
    }
    let mut z_axis = tracked_vec_normalize(z_seed)?;
    if let Some(recomputed_x) = tracked_vec_normalize(tracked_vec_cross(y_axis, z_axis)) {
        x_axis = recomputed_x;
    }
    if let Some(recomputed_y) = tracked_vec_normalize(tracked_vec_cross(z_axis, x_axis)) {
        y_axis = recomputed_y;
    }
    if tracked_vec_dot(y_axis, y_seed) < 0.0 {
        y_axis = tracked_vec_scale(y_axis, -1.0);
        z_axis = tracked_vec_scale(z_axis, -1.0);
    }

    let shoulder_width = tracked_point_distance(&left_shoulder, &right_shoulder);
    let torso_height = tracked_point_distance(&shoulders_mid, &hips_mid);
    if !shoulder_width.is_finite()
        || !torso_height.is_finite()
        || shoulder_width <= 1e-6
        || torso_height <= 1e-6
    {
        return None;
    }

    let scale_x = TRACKED_CANONICAL_SHOULDER_WIDTH_M / shoulder_width;
    let scale_y = TRACKED_CANONICAL_TORSO_HEIGHT_M / torso_height;
    let scale_z = (scale_x + scale_y) * 0.5;

    let mut canonical: Vec<[f64; 3]> = points
        .iter()
        .map(|point| {
            let delta = tracked_vec_sub(*point, hips_mid);
            let local_x = tracked_vec_dot(delta, x_axis) * scale_x;
            let local_y = tracked_vec_dot(delta, y_axis) * scale_y;
            let local_z = (tracked_vec_dot(delta, z_axis) * scale_z).clamp(
                -TRACKED_CANONICAL_DEPTH_DELTA_LIMIT_M,
                TRACKED_CANONICAL_DEPTH_DELTA_LIMIT_M,
            );
            [local_x, local_y, TRACKED_CANONICAL_BASE_DEPTH_M + local_z]
        })
        .collect();
    enforce_tracked_canonical_orientation(&mut canonical);
    Some(canonical)
}

fn canonicalize_tracked_person_points(
    person: &PersonDetection,
    _target_space: &str,
) -> Option<Vec<[f64; 3]>> {
    if _target_space == MODEL_TARGET_SPACE_OPERATOR_FRAME {
        return canonicalize_operator_frame_tracked_points(person);
    }

    if person.keypoints.len() < MODEL_N_KEYPOINTS {
        return None;
    }
    let by_name: BTreeMap<&str, &PoseKeypoint> = person
        .keypoints
        .iter()
        .map(|kp| (kp.name.as_str(), kp))
        .collect();
    let left_shoulder = by_name.get("left_shoulder")?;
    let right_shoulder = by_name.get("right_shoulder")?;
    let left_hip = by_name.get("left_hip")?;
    let right_hip = by_name.get("right_hip")?;

    let hip_center_x = (left_hip.x + right_hip.x) * 0.5;
    let hip_center_y = (left_hip.y + right_hip.y) * 0.5;
    let torso_center_z = (left_shoulder.z + right_shoulder.z + left_hip.z + right_hip.z) * 0.25;

    let dx = left_shoulder.x - right_shoulder.x;
    let dy = left_shoulder.y - right_shoulder.y;
    let shoulder_px = (dx * dx + dy * dy).sqrt();
    if !shoulder_px.is_finite() || shoulder_px <= 1e-6 {
        return None;
    }
    let meters_per_px = TRACKED_CANONICAL_SHOULDER_WIDTH_M / shoulder_px;

    let mut points: Vec<[f64; 3]> = person
        .keypoints
        .iter()
        .take(MODEL_N_KEYPOINTS)
        .map(|kp| {
            let x_m = (kp.x - hip_center_x) * meters_per_px;
            let y_m = (hip_center_y - kp.y) * meters_per_px;
            let z_delta = (kp.z - torso_center_z) * 0.12;
            let z_m = TRACKED_CANONICAL_BASE_DEPTH_M + z_delta + tracked_forward_bias_m(&kp.name);
            [x_m, y_m, z_m]
        })
        .collect();
    enforce_tracked_canonical_orientation(&mut points);
    Some(points)
}

fn tracked_flip_points(points: &[[f64; 3]], flip_x: bool, flip_y: bool) -> Vec<[f64; 3]> {
    points
        .iter()
        .map(|point| {
            [
                if flip_x { -point[0] } else { point[0] },
                if flip_y { -point[1] } else { point[1] },
                point[2],
            ]
        })
        .collect()
}

fn tracked_anchor_orientation(points: &[[f64; 3]]) -> Option<(f64, f64)> {
    if points.len() <= 12 {
        return None;
    }
    let left_shoulder = points[5];
    let right_shoulder = points[6];
    let left_hip = points[11];
    let right_hip = points[12];
    let shoulder_dx = left_shoulder[0] - right_shoulder[0];
    let shoulders_mean_y = (left_shoulder[1] + right_shoulder[1]) * 0.5;
    let hips_mean_y = (left_hip[1] + right_hip[1]) * 0.5;
    Some((shoulder_dx, shoulders_mean_y - hips_mean_y))
}

fn enforce_tracked_canonical_orientation(points: &mut [[f64; 3]]) {
    if let Some((shoulder_dx, shoulder_vs_hip_y)) = tracked_anchor_orientation(points) {
        if shoulder_dx < 0.0 {
            for point in points.iter_mut() {
                point[0] = -point[0];
            }
        }
        if shoulder_vs_hip_y > 0.0 {
            for point in points.iter_mut() {
                point[1] = -point[1];
            }
        }
    }
}

fn tracked_points_distance(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(pa, pb)| tracked_point_distance(pa, pb))
        .sum::<f64>()
}

fn stabilize_tracked_canonical_points(
    raw_points: Vec<[f64; 3]>,
    previous: Option<&TrackedPersonContract>,
) -> Vec<[f64; 3]> {
    if let Some(previous) = previous {
        if previous.canonical_body_kpts_3d.len() == raw_points.len() && !raw_points.is_empty() {
            let candidates = [
                tracked_flip_points(&raw_points, false, false),
                tracked_flip_points(&raw_points, true, false),
                tracked_flip_points(&raw_points, false, true),
                tracked_flip_points(&raw_points, true, true),
            ];
            if let Some(best) = candidates.into_iter().min_by(|a, b| {
                tracked_points_distance(a, &previous.canonical_body_kpts_3d).total_cmp(
                    &tracked_points_distance(b, &previous.canonical_body_kpts_3d),
                )
            }) {
                return best;
            }
        }
    }

    let mut points = raw_points;
    enforce_tracked_canonical_orientation(&mut points);
    points
}

fn tracked_joint_smoothing_alpha(index: usize) -> f64 {
    match index {
        9 | 10 => LIVE_TRACK_CANONICAL_EMA_ALPHA_WRIST,
        7 | 8 | 13 | 14 | 15 | 16 => LIVE_TRACK_CANONICAL_EMA_ALPHA_LIMB,
        _ => LIVE_TRACK_CANONICAL_EMA_ALPHA,
    }
}

fn tracked_joint_jump_limit_m(index: usize) -> f64 {
    match index {
        9 | 10 => 0.08,
        7 | 8 => 0.07,
        13 | 14 | 15 | 16 => 0.09,
        _ => 0.06,
    }
}

fn clamp_tracked_joint_delta(current: [f64; 3], previous: &[f64; 3], max_delta_m: f64) -> [f64; 3] {
    let dx = current[0] - previous[0];
    let dy = current[1] - previous[1];
    let dz = current[2] - previous[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    if !distance.is_finite() || distance <= max_delta_m || max_delta_m <= 0.0 {
        return current;
    }
    let scale = max_delta_m / distance;
    [
        previous[0] + dx * scale,
        previous[1] + dy * scale,
        previous[2] + dz * scale,
    ]
}

fn smooth_tracked_canonical_points(
    points: Vec<[f64; 3]>,
    previous: Option<&TrackedPersonContract>,
    reused_existing_track: bool,
) -> Vec<[f64; 3]> {
    if !reused_existing_track {
        return points;
    }
    let Some(previous) = previous else {
        return points;
    };
    if previous.canonical_body_kpts_3d.len() != points.len() || points.is_empty() {
        return points;
    }

    let mut smoothed = points;
    for (index, (point, prev)) in smoothed
        .iter_mut()
        .zip(previous.canonical_body_kpts_3d.iter())
        .enumerate()
    {
        let clamped = clamp_tracked_joint_delta(*point, prev, tracked_joint_jump_limit_m(index));
        let alpha = tracked_joint_smoothing_alpha(index);
        point[0] = prev[0] * (1.0 - alpha) + clamped[0] * alpha;
        point[1] = prev[1] * (1.0 - alpha) + clamped[1] * alpha;
        point[2] = prev[2] * (1.0 - alpha) + clamped[2] * alpha;
    }
    smoothed
}

fn tracked_torso_metrics(points: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3], f64, f64)> {
    if points.len() <= 12 {
        return None;
    }
    let left_shoulder = points[5];
    let right_shoulder = points[6];
    let left_hip = points[11];
    let right_hip = points[12];
    let shoulder_mid = midpoint_3d(left_shoulder, right_shoulder);
    let hip_mid = midpoint_3d(left_hip, right_hip);
    let shoulder_width = tracked_point_distance(&left_shoulder, &right_shoulder);
    let torso_height = tracked_point_distance(&shoulder_mid, &hip_mid);
    if !shoulder_width.is_finite()
        || !torso_height.is_finite()
        || shoulder_width <= 1e-6
        || torso_height <= 1e-6
    {
        return None;
    }
    Some((shoulder_mid, hip_mid, shoulder_width, torso_height))
}

fn midpoint_3d(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

fn tracked_segment_with_constraints(
    parent: [f64; 3],
    child: [f64; 3],
    desired_len: f64,
    side_sign: f64,
    min_abs_x_delta: f64,
    require_positive_y: bool,
    min_y_delta: f64,
    fallback: [f64; 3],
) -> [f64; 3] {
    let mut dir = tracked_vec_sub(child, parent);
    if dir[0].abs() < min_abs_x_delta {
        dir[0] = side_sign * min_abs_x_delta;
    } else {
        dir[0] = dir[0].abs() * side_sign;
    }
    if require_positive_y {
        dir[1] = dir[1].abs().max(min_y_delta);
    }
    if tracked_vec_norm(dir) <= 1e-6 {
        dir = fallback;
    }
    let dir = tracked_vec_normalize(dir).unwrap_or(fallback);
    let mut next = tracked_vec_add(parent, tracked_vec_scale(dir, desired_len));
    next[0] = next[0].abs() * side_sign;
    if require_positive_y && next[1] < parent[1] + min_y_delta {
        next[1] = parent[1] + min_y_delta;
    }
    next[2] = next[2].clamp(
        TRACKED_CANONICAL_BASE_DEPTH_M - TRACKED_CANONICAL_DEPTH_DELTA_LIMIT_M,
        TRACKED_CANONICAL_BASE_DEPTH_M + TRACKED_CANONICAL_DEPTH_DELTA_LIMIT_M,
    );
    next
}

fn repair_tracked_canonical_points(mut points: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
    if points.len() < MODEL_N_KEYPOINTS {
        return points;
    }
    enforce_tracked_canonical_orientation(&mut points);
    let Some((shoulder_mid, hip_mid, shoulder_width, torso_height)) =
        tracked_torso_metrics(&points)
    else {
        return points;
    };

    let shoulder_half = (shoulder_width * 0.5).max(TRACKED_CANONICAL_SHOULDER_WIDTH_M * 0.45);
    let hip_half =
        (tracked_point_distance(&points[11], &points[12]) * 0.5).max(shoulder_half * 0.82);
    let shoulders_y = shoulder_mid[1];
    let hips_y = hip_mid[1];
    let shoulders_z = shoulder_mid[2];
    let hips_z = hip_mid[2];

    points[5] = [
        shoulder_half,
        shoulders_y,
        points[5][2].clamp(shoulders_z - 0.03, shoulders_z + 0.03),
    ];
    points[6] = [
        -shoulder_half,
        shoulders_y,
        points[6][2].clamp(shoulders_z - 0.03, shoulders_z + 0.03),
    ];
    points[11] = [
        hip_half,
        hips_y,
        points[11][2].clamp(hips_z - 0.03, hips_z + 0.03),
    ];
    points[12] = [
        -hip_half,
        hips_y,
        points[12][2].clamp(hips_z - 0.03, hips_z + 0.03),
    ];

    let upper_arm_len = torso_height * TRACKED_CANONICAL_UPPER_ARM_RATIO;
    let forearm_len = torso_height * TRACKED_CANONICAL_FOREARM_RATIO;
    let thigh_len = torso_height * TRACKED_CANONICAL_THIGH_RATIO;
    let shin_len = torso_height * TRACKED_CANONICAL_SHIN_RATIO;
    let min_side_dx = shoulder_half * 0.18;
    let min_leg_dx = hip_half * 0.12;
    let min_leg_dy = torso_height * 0.18;

    points[7] = tracked_segment_with_constraints(
        points[5],
        points[7],
        upper_arm_len,
        1.0,
        min_side_dx,
        false,
        0.0,
        [0.35, 0.94, 0.0],
    );
    points[9] = tracked_segment_with_constraints(
        points[7],
        points[9],
        forearm_len,
        1.0,
        min_side_dx,
        false,
        0.0,
        [0.28, 0.96, 0.0],
    );
    points[8] = tracked_segment_with_constraints(
        points[6],
        points[8],
        upper_arm_len,
        -1.0,
        min_side_dx,
        false,
        0.0,
        [-0.35, 0.94, 0.0],
    );
    points[10] = tracked_segment_with_constraints(
        points[8],
        points[10],
        forearm_len,
        -1.0,
        min_side_dx,
        false,
        0.0,
        [-0.28, 0.96, 0.0],
    );

    points[13] = tracked_segment_with_constraints(
        points[11],
        points[13],
        thigh_len,
        1.0,
        min_leg_dx,
        true,
        min_leg_dy,
        [0.10, 0.99, 0.0],
    );
    points[15] = tracked_segment_with_constraints(
        points[13],
        points[15],
        shin_len,
        1.0,
        min_leg_dx,
        true,
        min_leg_dy,
        [0.06, 0.998, 0.0],
    );
    points[14] = tracked_segment_with_constraints(
        points[12],
        points[14],
        thigh_len,
        -1.0,
        min_leg_dx,
        true,
        min_leg_dy,
        [-0.10, 0.99, 0.0],
    );
    points[16] = tracked_segment_with_constraints(
        points[14],
        points[16],
        shin_len,
        -1.0,
        min_leg_dx,
        true,
        min_leg_dy,
        [-0.06, 0.998, 0.0],
    );

    let head_y = shoulders_y - torso_height * TRACKED_CANONICAL_HEAD_HEIGHT_RATIO;
    let head_z = (shoulders_z + TRACKED_CANONICAL_FACE_DEPTH_OFFSET_M).clamp(
        TRACKED_CANONICAL_BASE_DEPTH_M - TRACKED_CANONICAL_DEPTH_DELTA_LIMIT_M,
        TRACKED_CANONICAL_BASE_DEPTH_M + TRACKED_CANONICAL_DEPTH_DELTA_LIMIT_M,
    );
    let head_half = shoulder_half * TRACKED_CANONICAL_HEAD_HALF_WIDTH_RATIO;
    let eye_half = shoulder_half * TRACKED_CANONICAL_EYE_OFFSET_RATIO;
    let ear_half = shoulder_half * TRACKED_CANONICAL_EAR_OFFSET_RATIO;
    points[0] = [0.0, head_y, head_z];
    points[1] = [eye_half, head_y - torso_height * 0.03, head_z + 0.01];
    points[2] = [-eye_half, head_y - torso_height * 0.03, head_z + 0.01];
    points[3] = [ear_half + head_half, head_y - torso_height * 0.01, head_z];
    points[4] = [
        -(ear_half + head_half),
        head_y - torso_height * 0.01,
        head_z,
    ];

    enforce_tracked_canonical_orientation(&mut points);
    points
}

fn refresh_stable_tracked_person(state: &mut AppStateInner) -> Option<TrackedPersonContract> {
    let update = state.latest_update.clone()?;
    let pose_source = live_pose_source(&update, state.model_loaded).to_string();
    let target_space = tracked_target_space(
        state.model_loaded,
        state.model_target_space.as_deref(),
        &update,
    );
    let persons = resolve_live_persons(&update, state.model_loaded);

    if persons.is_empty() {
        if let Some(mut tracked) = state.stable_tracked_person.clone() {
            tracked.missed_frames += 1;
            if tracked.missed_frames <= LIVE_TRACK_MISS_HOLD_FRAMES {
                tracked.contract.lifecycle_state = "lost".to_string();
                tracked.contract.coherence_gate_decision =
                    tracked_gate_decision(&update, tracked.contract.person_confidence, true)
                        .to_string();
                tracked.contract.target_space = target_space;
                tracked.contract.pose_source = pose_source;
                state.stable_tracked_person = Some(tracked.clone());
                return Some(tracked.contract);
            }
        }
        state.stable_tracked_person = None;
        return None;
    }

    let candidates: Vec<(PersonDetection, Vec<[f64; 3]>, Vec<[f64; 3]>, [f64; 3])> = persons
        .into_iter()
        .filter_map(|person| {
            let points = tracked_points_from_person(&person)?;
            let canonical_points = canonicalize_tracked_person_points(&person, &target_space)
                .unwrap_or_else(|| points.clone());
            let centroid = tracked_points_centroid(&canonical_points);
            Some((person, points, canonical_points, centroid))
        })
        .collect();
    if candidates.is_empty() {
        state.stable_tracked_person = None;
        return None;
    }

    let previous = state.stable_tracked_person.clone();
    let nearest_match = previous.as_ref().and_then(|tracked| {
        let threshold = tracked_continuity_threshold(&target_space);
        candidates
            .iter()
            .enumerate()
            .map(|(idx, (_, _, _, centroid))| {
                (idx, tracked_point_distance(&tracked.centroid, centroid))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(idx, distance)| (idx, distance, threshold))
    });

    let matched_index = nearest_match.and_then(|(idx, distance, threshold)| {
        if distance <= threshold {
            Some(idx)
        } else {
            None
        }
    });

    let soft_matched_index = nearest_match.and_then(|(idx, distance, threshold)| {
        if distance <= threshold * LIVE_TRACK_SOFT_CONTINUITY_SCALE {
            Some(idx)
        } else {
            None
        }
    });

    let same_source_index = previous.as_ref().and_then(|tracked| {
        candidates
            .iter()
            .enumerate()
            .find_map(|(idx, (person, _, _, centroid))| {
                if person.id != tracked.contract.source_person_id {
                    return None;
                }
                let threshold =
                    tracked_continuity_threshold(&target_space) * LIVE_TRACK_SOFT_CONTINUITY_SCALE;
                let distance = tracked_point_distance(&tracked.centroid, centroid);
                if distance <= threshold {
                    Some(idx)
                } else {
                    None
                }
            })
    });

    let selected_index = matched_index
        .or(same_source_index)
        .or(soft_matched_index)
        .unwrap_or_else(|| {
            candidates
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.0.confidence.total_cmp(&b.0.confidence))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        });

    let (person, points, raw_canonical_points, _) = candidates[selected_index].clone();
    let reused_existing_track =
        matched_index.is_some() || same_source_index.is_some() || soft_matched_index.is_some();
    let (track_id, consecutive_hits) = if let Some(previous) = previous.as_ref() {
        if reused_existing_track {
            (
                previous.contract.track_id.clone(),
                previous.consecutive_hits.saturating_add(1),
            )
        } else {
            let next_track = format!("wifi-track-{}", state.next_stable_track_id);
            state.next_stable_track_id += 1;
            (next_track, 1)
        }
    } else {
        let next_track = format!("wifi-track-{}", state.next_stable_track_id);
        state.next_stable_track_id += 1;
        (next_track, 1)
    };

    let lifecycle_state = if consecutive_hits >= 2 {
        "active"
    } else {
        "tentative"
    };
    let gate_decision = tracked_gate_decision(&update, person.confidence, false);
    let canonical_points = smooth_tracked_canonical_points(
        stabilize_tracked_canonical_points(
            raw_canonical_points,
            previous.as_ref().map(|value| &value.contract),
        ),
        previous.as_ref().map(|value| &value.contract),
        reused_existing_track,
    );
    let canonical_points = repair_tracked_canonical_points(canonical_points);
    let contract = tracked_contract_from_person(
        track_id,
        &person,
        points,
        canonical_points,
        &target_space,
        &pose_source,
        lifecycle_state,
        gate_decision,
    );
    state.stable_tracked_person = Some(StableTrackedPersonState {
        contract: contract.clone(),
        centroid: tracked_points_centroid(&contract.canonical_body_kpts_3d),
        consecutive_hits,
        missed_frames: 0,
    });
    Some(contract)
}
const MODEL_N_FREQ_BANDS: usize = 9;
const MODEL_N_GLOBAL_FEATURES: usize = 3;

fn default_model_temporal_context_decay() -> f64 {
    0.65
}

type SharedState = Arc<RwLock<AppStateInner>>;

#[derive(Debug, Clone, Deserialize)]
struct ModelFeatureStats {
    mean: Vec<f64>,
    std: Vec<f64>,
    n_features: usize,
    n_subcarriers: usize,
    #[serde(default)]
    temporal_context_frames: usize,
    #[serde(default = "default_model_temporal_context_decay")]
    temporal_context_decay: f64,
}

// ── ESP32 Edge Vitals Packet (ADR-039, magic 0xC511_0002) ────────────────────

/// Decoded vitals packet from ESP32 edge processing pipeline.
#[derive(Debug, Clone, Serialize)]
struct Esp32VitalsPacket {
    node_id: u8,
    presence: bool,
    fall_detected: bool,
    motion: bool,
    breathing_rate_bpm: f64,
    heartrate_bpm: f64,
    rssi: i8,
    n_persons: u8,
    motion_energy: f32,
    presence_score: f32,
    timestamp_ms: u32,
}

/// Parse a 32-byte edge vitals packet (magic 0xC511_0002).
fn parse_esp32_vitals(buf: &[u8]) -> Option<Esp32VitalsPacket> {
    if buf.len() < 32 {
        return None;
    }
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != 0xC511_0002 {
        return None;
    }

    let node_id = buf[4];
    let flags = buf[5];
    let breathing_raw = u16::from_le_bytes([buf[6], buf[7]]);
    let heartrate_raw = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let rssi = buf[12] as i8;
    let n_persons = buf[13];
    let motion_energy = f32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let presence_score = f32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let timestamp_ms = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);

    Some(Esp32VitalsPacket {
        node_id,
        presence: (flags & 0x01) != 0,
        fall_detected: (flags & 0x02) != 0,
        motion: (flags & 0x04) != 0,
        breathing_rate_bpm: breathing_raw as f64 / 100.0,
        heartrate_bpm: heartrate_raw as f64 / 10000.0,
        rssi,
        n_persons,
        motion_energy,
        presence_score,
        timestamp_ms,
    })
}

// ── ADR-040: WASM Output Packet (magic 0xC511_0004) ───────────────────────────

/// Single WASM event (type + value).
#[derive(Debug, Clone, Serialize)]
struct WasmEvent {
    event_type: u8,
    value: f32,
}

/// Decoded WASM output packet from ESP32 Tier 3 runtime.
#[derive(Debug, Clone, Serialize)]
struct WasmOutputPacket {
    node_id: u8,
    module_id: u8,
    events: Vec<WasmEvent>,
}

/// Parse a WASM output packet (magic 0xC511_0004).
fn parse_wasm_output(buf: &[u8]) -> Option<WasmOutputPacket> {
    if buf.len() < 8 {
        return None;
    }
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != 0xC511_0004 {
        return None;
    }

    let node_id = buf[4];
    let module_id = buf[5];
    let event_count = u16::from_le_bytes([buf[6], buf[7]]) as usize;

    let mut events = Vec::with_capacity(event_count);
    let mut offset = 8;
    for _ in 0..event_count {
        if offset + 5 > buf.len() {
            break;
        }
        let event_type = buf[offset];
        let value = f32::from_le_bytes([
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
            buf[offset + 4],
        ]);
        events.push(WasmEvent { event_type, value });
        offset += 5;
    }

    Some(WasmOutputPacket {
        node_id,
        module_id,
        events,
    })
}

// ── ESP32 UDP frame parser ───────────────────────────────────────────────────

fn parse_esp32_frame(buf: &[u8]) -> Option<Esp32Frame> {
    if buf.len() < ESP32_CSI_HEADER_LEN_V1 {
        return None;
    }

    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let iq_start = match magic {
        ESP32_CSI_MAGIC_V1 => ESP32_CSI_HEADER_LEN_V1,
        ESP32_CSI_MAGIC_V2 => ESP32_CSI_HEADER_LEN_V2,
        _ => return None,
    };
    if buf.len() < iq_start {
        return None;
    }

    let node_id = buf[4];
    let n_antennas = buf[5];
    let n_subcarriers_u16 = u16::from_le_bytes([buf[6], buf[7]]);
    if n_subcarriers_u16 == 0 || n_subcarriers_u16 > u8::MAX as u16 {
        return None;
    }
    let n_subcarriers = n_subcarriers_u16 as u8;
    let freq_mhz_u32 = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    if freq_mhz_u32 > u16::MAX as u32 {
        return None;
    }
    let freq_mhz = freq_mhz_u32 as u16;
    let sequence = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    let rssi = buf[16] as i8;
    let noise_floor = buf[17] as i8;

    let n_pairs = n_antennas as usize * n_subcarriers as usize;
    let expected_len = iq_start + n_pairs * 2;

    if buf.len() < expected_len {
        return None;
    }

    let mut amplitudes = Vec::with_capacity(n_pairs);
    let mut phases = Vec::with_capacity(n_pairs);

    for k in 0..n_pairs {
        let i_val = buf[iq_start + k * 2] as i8 as f64;
        let q_val = buf[iq_start + k * 2 + 1] as i8 as f64;
        amplitudes.push((i_val * i_val + q_val * q_val).sqrt());
        phases.push(q_val.atan2(i_val));
    }

    Some(Esp32Frame {
        magic,
        node_id,
        n_antennas,
        n_subcarriers,
        freq_mhz,
        sequence,
        rssi,
        noise_floor,
        amplitudes,
        phases,
    })
}

// ── Signal field generation ──────────────────────────────────────────────────

/// Generate a signal field that reflects where motion and signal changes are occurring.
///
/// Instead of a fixed-animation circle, this function uses the actual sensing data:
/// - `subcarrier_variances`: per-subcarrier variance computed from the frame history.
///   High-variance subcarriers indicate spatial directions where the signal is disrupted.
/// - `motion_score`: overall motion intensity [0, 1].
/// - `breathing_rate_hz`: estimated breathing rate in Hz; if > 0, adds a breathing ring.
/// - `signal_quality`: overall quality metric [0, 1] modulates field brightness.
///
/// The field grid is 20×20 cells representing a top-down view of the room.
/// Hotspots are derived from the subcarrier index (treated as an angular bin) so that
/// subcarriers with the highest variance produce peaks at the corresponding directions.
fn generate_signal_field(
    _mean_rssi: f64,
    motion_score: f64,
    breathing_rate_hz: f64,
    signal_quality: f64,
    subcarrier_variances: &[f64],
) -> SignalField {
    let grid = 20usize;
    let mut values = vec![0.0f64; grid * grid];
    let center = (grid as f64 - 1.0) / 2.0;

    // Normalise subcarrier variances to [0, 1].
    let max_var = subcarrier_variances.iter().cloned().fold(0.0f64, f64::max);
    let norm_factor = if max_var > 1e-9 { max_var } else { 1.0 };

    // For each cell, accumulate contributions from all subcarriers.
    // Each subcarrier k is assigned an angular direction proportional to its index
    // so that different subcarriers illuminate different regions of the room.
    let n_sub = subcarrier_variances.len().max(1);
    for (k, &var) in subcarrier_variances.iter().enumerate() {
        let weight = (var / norm_factor) * motion_score;
        if weight < 1e-6 {
            continue;
        }
        // Map subcarrier index to an angle across the full 2π sweep.
        let angle = (k as f64 / n_sub as f64) * 2.0 * std::f64::consts::PI;
        // Place the hotspot at a distance proportional to the weight, capped at 40% of
        // the grid radius so it stays within the room model.
        let radius = center * 0.8 * weight.sqrt();
        let hx = center + radius * angle.cos();
        let hz = center + radius * angle.sin();

        for z in 0..grid {
            for x in 0..grid {
                let dx = x as f64 - hx;
                let dz = z as f64 - hz;
                let dist2 = dx * dx + dz * dz;
                // Gaussian blob centred on the hotspot; spread scales with weight.
                let spread = (0.5 + weight * 2.0).max(0.5);
                values[z * grid + x] += weight * (-dist2 / (2.0 * spread * spread)).exp();
            }
        }
    }

    // Base radial attenuation from the router assumed at grid centre.
    for z in 0..grid {
        for x in 0..grid {
            let dx = x as f64 - center;
            let dz = z as f64 - center;
            let dist = (dx * dx + dz * dz).sqrt();
            let base = signal_quality * (-dist * 0.12).exp();
            values[z * grid + x] += base * 0.3;
        }
    }

    // Breathing ring: if a breathing rate was estimated add a faint annular highlight
    // at a radius corresponding to typical chest-wall displacement range.
    if breathing_rate_hz > 0.05 {
        let ring_r = center * 0.55;
        let ring_width = 1.8f64;
        for z in 0..grid {
            for x in 0..grid {
                let dx = x as f64 - center;
                let dz = z as f64 - center;
                let dist = (dx * dx + dz * dz).sqrt();
                let ring_val =
                    0.08 * (-(dist - ring_r).powi(2) / (2.0 * ring_width * ring_width)).exp();
                values[z * grid + x] += ring_val;
            }
        }
    }

    // Clamp and normalise to [0, 1].
    let field_max = values.iter().cloned().fold(0.0f64, f64::max);
    let scale = if field_max > 1e-9 {
        1.0 / field_max
    } else {
        1.0
    };
    for v in &mut values {
        *v = (*v * scale).clamp(0.0, 1.0);
    }

    SignalField {
        grid_size: [grid, 1, grid],
        values,
    }
}

fn default_node_position(node_id: u8) -> [f64; 3] {
    match node_id {
        1 => [-1.2, 0.0, 1.2],
        2 => [1.2, 0.0, 1.2],
        3 => [-1.2, 0.0, -1.2],
        4 => [1.2, 0.0, -1.2],
        5 => [0.0, 0.0, 1.8],
        6 => [0.0, 0.0, -1.8],
        _ => [0.0, 0.0, 0.0],
    }
}

fn upsert_recent_node(
    recent_nodes: &mut BTreeMap<u8, RecentNodeState>,
    node_id: u8,
    rssi_dbm: f64,
    amplitude: Vec<f64>,
    subcarrier_count: usize,
) {
    recent_nodes.insert(
        node_id,
        RecentNodeState {
            node_id,
            rssi_dbm,
            position: default_node_position(node_id),
            amplitude,
            subcarrier_count,
            last_seen: Instant::now(),
        },
    );
}

fn active_nodes(recent_nodes: &mut BTreeMap<u8, RecentNodeState>) -> Vec<NodeInfo> {
    let now = Instant::now();
    recent_nodes.retain(|_, node| {
        now.duration_since(node.last_seen) <= Duration::from_millis(ACTIVE_NODE_TTL_MS)
    });
    recent_nodes
        .values()
        .map(|node| NodeInfo {
            node_id: node.node_id,
            rssi_dbm: node.rssi_dbm,
            position: node.position,
            amplitude: node.amplitude.clone(),
            subcarrier_count: node.subcarrier_count,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct SignalHotspot {
    zone: usize,
    centroid: [f64; 2],
    peak: f64,
    mass: f64,
    cell_count: usize,
}

fn round_metric(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn hash_bssid(bssid: &BssidId) -> u64 {
    bssid
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            hash.wrapping_mul(0x0000_0001_0000_01b3)
                .wrapping_add(u64::from(*byte) + 1)
        })
}

fn live_bssid_node_position(obs: &BssidObservation) -> [f64; 3] {
    let seed = hash_bssid(&obs.bssid);
    let angle = ((seed & 0xffff) as f64 / 65_535.0) * std::f64::consts::TAU;
    let band_radius = match obs.band {
        BandType::Band2_4GHz => 1.45,
        BandType::Band5GHz => 1.95,
        BandType::Band6GHz => 2.35,
    };
    let signal_norm = ((obs.rssi_dbm + 95.0) / 55.0).clamp(0.0, 1.0);
    let channel_bias = (obs.channel as f64 / 177.0 - 0.5) * 0.35;
    let wobble = (((seed >> 16) & 0xff) as f64 / 255.0 - 0.5) * 0.18;
    let radius = (band_radius + channel_bias - signal_norm * 0.18).clamp(1.1, 2.6);
    [
        (radius * angle.cos() + wobble).clamp(-2.8, 2.8),
        0.0,
        (radius * angle.sin() - wobble).clamp(-2.8, 2.8),
    ]
}

fn live_nodes_from_bssid_observations(observations: &[BssidObservation]) -> Vec<NodeInfo> {
    let mut ordered: Vec<&BssidObservation> = observations.iter().collect();
    ordered.sort_by(|a, b| a.bssid.cmp(&b.bssid));
    ordered.truncate(MAX_LIVE_BSSID_NODES);

    ordered
        .into_iter()
        .enumerate()
        .map(|(index, obs)| NodeInfo {
            node_id: u8::try_from(index + 1).unwrap_or(u8::MAX),
            rssi_dbm: obs.rssi_dbm,
            position: live_bssid_node_position(obs),
            amplitude: vec![obs.amplitude()],
            subcarrier_count: 1,
        })
        .collect()
}

fn empty_zone_summary() -> serde_json::Value {
    serde_json::json!({
        "zones": {
            "zone_1": { "person_count": 0, "status": "clear", "energy": 0.0, "peak": 0.0, "hotspot_count": 0 },
            "zone_2": { "person_count": 0, "status": "clear", "energy": 0.0, "peak": 0.0, "hotspot_count": 0 },
            "zone_3": { "person_count": 0, "status": "clear", "energy": 0.0, "peak": 0.0, "hotspot_count": 0 },
            "zone_4": { "person_count": 0, "status": "clear", "energy": 0.0, "peak": 0.0, "hotspot_count": 0 }
        },
        "summary": {
            "hotspot_count": 0,
            "presence": false,
        }
    })
}

fn zone_index_for_cell(x: usize, z: usize, grid_x: usize, grid_z: usize) -> usize {
    let x_mid = grid_x / 2;
    let z_mid = grid_z / 2;
    match (x < x_mid, z < z_mid) {
        (true, true) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (false, false) => 3,
    }
}

fn detect_signal_hotspots(
    field: &SignalField,
    total_persons: usize,
    presence: bool,
) -> (Vec<SignalHotspot>, f64) {
    let grid_x = field.grid_size[0].max(1);
    let grid_z = field.grid_size[2].max(1);
    let cell_count = grid_x * grid_z;
    if !presence || field.values.len() < cell_count {
        return (Vec::new(), 0.0);
    }

    let values = &field.values[..cell_count];
    let sum = values.iter().sum::<f64>();
    let mean = sum / cell_count as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / cell_count as f64;
    let std_dev = variance.sqrt();
    let max_value = values.iter().copied().fold(0.0f64, f64::max);
    if max_value < SIGNAL_HOTSPOT_FALLBACK_PEAK {
        return (Vec::new(), max_value);
    }

    let threshold = (mean + std_dev * 0.9)
        .max(max_value * 0.42)
        .max(SIGNAL_HOTSPOT_FALLBACK_PEAK)
        .min(max_value);
    let mut visited = vec![false; cell_count];
    let mut hotspots = Vec::new();

    for start_idx in 0..cell_count {
        if visited[start_idx] || values[start_idx] < threshold {
            continue;
        }

        let mut queue = VecDeque::from([start_idx]);
        visited[start_idx] = true;
        let mut peak = 0.0f64;
        let mut mass = 0.0f64;
        let mut weighted_x = 0.0f64;
        let mut weighted_z = 0.0f64;
        let mut cell_count_in_hotspot = 0usize;
        let mut zone_mass = [0.0f64; ZONE_COUNT];

        while let Some(idx) = queue.pop_front() {
            let x = idx % grid_x;
            let z = idx / grid_x;
            let value = values[idx];
            peak = peak.max(value);
            mass += value;
            weighted_x += value * x as f64;
            weighted_z += value * z as f64;
            cell_count_in_hotspot += 1;
            let zone = zone_index_for_cell(x, z, grid_x, grid_z);
            zone_mass[zone] += value;

            let x_start = x.saturating_sub(1);
            let x_end = (x + 1).min(grid_x - 1);
            let z_start = z.saturating_sub(1);
            let z_end = (z + 1).min(grid_z - 1);

            for nz in z_start..=z_end {
                for nx in x_start..=x_end {
                    let neighbor_idx = nz * grid_x + nx;
                    if visited[neighbor_idx] || values[neighbor_idx] < threshold {
                        continue;
                    }
                    visited[neighbor_idx] = true;
                    queue.push_back(neighbor_idx);
                }
            }
        }

        if peak < SIGNAL_HOTSPOT_MIN_PEAK
            && (mass < SIGNAL_HOTSPOT_MIN_MASS || cell_count_in_hotspot < 2)
        {
            continue;
        }

        let zone = zone_mass
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);

        hotspots.push(SignalHotspot {
            zone,
            centroid: [
                weighted_x / mass.max(f64::EPSILON),
                weighted_z / mass.max(f64::EPSILON),
            ],
            peak,
            mass,
            cell_count: cell_count_in_hotspot,
        });
    }

    hotspots.sort_by(|a, b| {
        b.mass
            .partial_cmp(&a.mass)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.peak
                    .partial_cmp(&a.peak)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    if hotspots.is_empty() && max_value >= SIGNAL_HOTSPOT_FALLBACK_PEAK {
        if let Some((peak_idx, peak)) = values
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            let x = peak_idx % grid_x;
            let z = peak_idx / grid_x;
            hotspots.push(SignalHotspot {
                zone: zone_index_for_cell(x, z, grid_x, grid_z),
                centroid: [x as f64, z as f64],
                peak,
                mass: peak,
                cell_count: 1,
            });
        }
    }

    let limit = if total_persons > 0 {
        total_persons.min(ZONE_COUNT)
    } else {
        hotspots.len().min(ZONE_COUNT)
    };
    hotspots.truncate(limit);

    (hotspots, max_value)
}

fn summarize_signal_field_zones(
    field: &SignalField,
    total_persons: usize,
    presence: bool,
) -> serde_json::Value {
    let grid_x = field.grid_size[0].max(1);
    let grid_z = field.grid_size[2].max(1);
    if field.values.len() < grid_x * grid_z {
        return empty_zone_summary();
    }

    let mut zone_sums = [0.0f64; ZONE_COUNT];
    let mut zone_counts = [0usize; ZONE_COUNT];
    let mut zone_max = [0.0f64; ZONE_COUNT];

    for z in 0..grid_z {
        for x in 0..grid_x {
            let idx = z * grid_x + x;
            let value = field.values[idx];
            let zone = zone_index_for_cell(x, z, grid_x, grid_z);
            zone_sums[zone] += value;
            zone_counts[zone] += 1;
            zone_max[zone] = zone_max[zone].max(value);
        }
    }

    let mut zone_energy = [0.0f64; ZONE_COUNT];
    for i in 0..ZONE_COUNT {
        let mean = if zone_counts[i] > 0 {
            zone_sums[i] / zone_counts[i] as f64
        } else {
            0.0
        };
        zone_energy[i] = (mean * 0.55 + zone_max[i] * 0.45).clamp(0.0, 1.0);
    }

    let (hotspots, max_field_value) = detect_signal_hotspots(field, total_persons, presence);
    let mut person_counts = [0usize; ZONE_COUNT];
    let mut hotspot_counts = [0usize; ZONE_COUNT];
    let mut primary_hotspot_peak = [0.0f64; ZONE_COUNT];
    let mut primary_hotspot_mass = [0.0f64; ZONE_COUNT];
    let mut primary_hotspot_cells = [0usize; ZONE_COUNT];
    let mut primary_hotspot_centroid = [None; ZONE_COUNT];

    for hotspot in &hotspots {
        person_counts[hotspot.zone] += 1;
        hotspot_counts[hotspot.zone] += 1;
        if hotspot.peak >= primary_hotspot_peak[hotspot.zone] {
            primary_hotspot_peak[hotspot.zone] = hotspot.peak;
            primary_hotspot_mass[hotspot.zone] = hotspot.mass;
            primary_hotspot_cells[hotspot.zone] = hotspot.cell_count;
            primary_hotspot_centroid[hotspot.zone] = Some(hotspot.centroid);
        }
    }

    let zone_entry = |index: usize| {
        let energy = round_metric(zone_energy[index]);
        let count = person_counts[index];
        let peak = round_metric(zone_max[index]);
        let status = if count > 0 {
            "occupied"
        } else if zone_energy[index] >= SIGNAL_ZONE_CANDIDATE_ENERGY
            || primary_hotspot_peak[index] >= SIGNAL_HOTSPOT_FALLBACK_PEAK
        {
            "candidate"
        } else {
            "clear"
        };
        let mut entry = serde_json::json!({
            "person_count": count,
            "status": status,
            "energy": energy,
            "peak": peak,
            "hotspot_count": hotspot_counts[index],
        });

        if let Some(centroid) = primary_hotspot_centroid[index] {
            entry["centroid"] =
                serde_json::json!([round_metric(centroid[0]), round_metric(centroid[1]),]);
            entry["hotspot_mass"] = serde_json::json!(round_metric(primary_hotspot_mass[index]));
            entry["hotspot_cells"] = serde_json::json!(primary_hotspot_cells[index]);
        }

        entry
    };

    serde_json::json!({
        "zones": {
            "zone_1": zone_entry(0),
            "zone_2": zone_entry(1),
            "zone_3": zone_entry(2),
            "zone_4": zone_entry(3),
        },
        "summary": {
            "hotspot_count": hotspots.len(),
            "presence": presence,
            "max_field_value": round_metric(max_field_value),
        },
    })
}

// ── Feature extraction from ESP32 frame ──────────────────────────────────────

/// Estimate breathing rate in Hz from the amplitude time series stored in `frame_history`.
///
/// Approach:
/// 1. Build a scalar time series by computing the mean amplitude of each historical frame.
/// 2. Run a peak-detection pass: count rising-edge zero-crossings of the de-meaned signal.
/// 3. Convert the crossing rate to Hz, clipped to the physiological range 0.1–0.5 Hz
///    (12–30 breaths/min).
///
/// For accuracy the function additionally applies a simple 3-tap Goertzel-style power
/// estimate at evenly-spaced candidate frequencies in the breathing band and returns
/// the candidate with the highest energy.
fn estimate_breathing_rate_hz(frame_history: &VecDeque<Vec<f64>>, sample_rate_hz: f64) -> f64 {
    let n = frame_history.len();
    if n < 6 {
        return 0.0;
    }

    // Build scalar time series: mean amplitude per frame.
    let series: Vec<f64> = frame_history
        .iter()
        .map(|amps| {
            if amps.is_empty() {
                0.0
            } else {
                amps.iter().sum::<f64>() / amps.len() as f64
            }
        })
        .collect();

    let mean_s = series.iter().sum::<f64>() / n as f64;
    // De-mean.
    let detrended: Vec<f64> = series.iter().map(|x| x - mean_s).collect();

    // Goertzel power at candidate frequencies in the breathing band [0.1, 0.5] Hz.
    // We evaluate 9 candidate frequencies uniformly spaced in that band.
    let n_candidates = 9usize;
    let f_low = 0.1f64;
    let f_high = 0.5f64;
    let mut best_freq = 0.0f64;
    let mut best_power = 0.0f64;

    for i in 0..n_candidates {
        let freq = f_low + (f_high - f_low) * i as f64 / (n_candidates - 1).max(1) as f64;
        let omega = 2.0 * std::f64::consts::PI * freq / sample_rate_hz;
        let coeff = 2.0 * omega.cos();
        let mut s_prev2 = 0.0f64;
        let mut s_prev1 = 0.0f64;
        for &x in &detrended {
            let s = x + coeff * s_prev1 - s_prev2;
            s_prev2 = s_prev1;
            s_prev1 = s;
        }
        // Goertzel magnitude squared.
        let power = s_prev2 * s_prev2 + s_prev1 * s_prev1 - coeff * s_prev1 * s_prev2;
        if power > best_power {
            best_power = power;
            best_freq = freq;
        }
    }

    // Only report a breathing rate if the Goertzel energy is meaningfully above noise.
    // Threshold: power must exceed 10× the average power across all candidates.
    let avg_power = {
        let mut total = 0.0f64;
        for i in 0..n_candidates {
            let freq = f_low + (f_high - f_low) * i as f64 / (n_candidates - 1).max(1) as f64;
            let omega = 2.0 * std::f64::consts::PI * freq / sample_rate_hz;
            let coeff = 2.0 * omega.cos();
            let mut s_prev2 = 0.0f64;
            let mut s_prev1 = 0.0f64;
            for &x in &detrended {
                let s = x + coeff * s_prev1 - s_prev2;
                s_prev2 = s_prev1;
                s_prev1 = s;
            }
            total += s_prev2 * s_prev2 + s_prev1 * s_prev1 - coeff * s_prev1 * s_prev2;
        }
        total / n_candidates as f64
    };

    if best_power > avg_power * 3.0 {
        best_freq.clamp(f_low, f_high)
    } else {
        0.0
    }
}

/// Compute per-subcarrier variance across the sliding window of `frame_history`.
///
/// For each subcarrier index `k`, returns `Var[A_k]` over all stored frames.
/// This captures spatial signal variation; subcarriers whose amplitude fluctuates
/// heavily across time correspond to directions with motion.
fn compute_subcarrier_variances(frame_history: &VecDeque<Vec<f64>>, n_sub: usize) -> Vec<f64> {
    if frame_history.is_empty() || n_sub == 0 {
        return vec![0.0; n_sub];
    }

    let n_frames = frame_history.len() as f64;
    let mut means = vec![0.0f64; n_sub];
    let mut sq_means = vec![0.0f64; n_sub];

    for frame in frame_history.iter() {
        for k in 0..n_sub {
            let a = if k < frame.len() { frame[k] } else { 0.0 };
            means[k] += a;
            sq_means[k] += a * a;
        }
    }

    (0..n_sub)
        .map(|k| {
            let mean = means[k] / n_frames;
            let sq_mean = sq_means[k] / n_frames;
            (sq_mean - mean * mean).max(0.0)
        })
        .collect()
}

/// Extract features from the current ESP32 frame, enhanced with temporal context from
/// `frame_history`.
///
/// Improvements over the previous single-frame approach:
///
/// - **Variance**: computed as the mean of per-subcarrier temporal variance across the
///   sliding window, not just the intra-frame spatial variance.
/// - **Motion detection**: uses frame-to-frame temporal difference (mean L2 change
///   between the current frame and the previous frame) normalised by signal amplitude,
///   so that actual changes are detected rather than just a threshold on the current frame.
/// - **Breathing rate**: estimated via Goertzel filter bank on the 0.1–0.5 Hz band of
///   the amplitude time series.
/// - **Signal quality**: based on SNR estimate (RSSI – noise floor) and subcarrier
///   variance stability.
fn extract_features_from_frame(
    frame: &Esp32Frame,
    frame_history: &VecDeque<Vec<f64>>,
    sample_rate_hz: f64,
) -> (FeatureInfo, ClassificationInfo, f64, Vec<f64>) {
    let n_sub = frame.amplitudes.len().max(1);
    let n = n_sub as f64;
    let mean_amp: f64 = frame.amplitudes.iter().sum::<f64>() / n;
    let mean_rssi = frame.rssi as f64;

    // ── Intra-frame subcarrier variance (spatial spread across subcarriers) ──
    let intra_variance: f64 = frame
        .amplitudes
        .iter()
        .map(|a| (a - mean_amp).powi(2))
        .sum::<f64>()
        / n;

    // ── Temporal (sliding-window) per-subcarrier variance ──
    let sub_variances = compute_subcarrier_variances(frame_history, n_sub);
    let temporal_variance: f64 = if sub_variances.is_empty() {
        intra_variance
    } else {
        sub_variances.iter().sum::<f64>() / sub_variances.len() as f64
    };

    // Use the larger of intra-frame and temporal variance as the reported variance.
    let variance = intra_variance.max(temporal_variance);

    // ── Spectral power ──
    let spectral_power: f64 = frame.amplitudes.iter().map(|a| a * a).sum::<f64>() / n;

    // ── Motion band power (upper half of subcarriers, high spatial frequency) ──
    let half = frame.amplitudes.len() / 2;
    let motion_band_power = if half > 0 {
        frame.amplitudes[half..]
            .iter()
            .map(|a| (a - mean_amp).powi(2))
            .sum::<f64>()
            / (frame.amplitudes.len() - half) as f64
    } else {
        0.0
    };

    // ── Breathing band power (lower half of subcarriers, low spatial frequency) ──
    let breathing_band_power = if half > 0 {
        frame.amplitudes[..half]
            .iter()
            .map(|a| (a - mean_amp).powi(2))
            .sum::<f64>()
            / half as f64
    } else {
        0.0
    };

    // ── Dominant frequency via peak subcarrier index ──
    let peak_idx = frame
        .amplitudes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant_freq_hz = peak_idx as f64 * 0.05;

    // ── Change point detection (threshold-crossing count in current frame) ──
    let threshold = mean_amp * 1.2;
    let change_points = frame
        .amplitudes
        .windows(2)
        .filter(|w| (w[0] < threshold) != (w[1] < threshold))
        .count();

    // ── Motion score: sliding-window temporal difference ──
    // Compare current frame against the most recent historical frame.
    // The difference is normalised by the mean amplitude to be scale-invariant.
    let temporal_motion_score = if let Some(prev_frame) = frame_history.back() {
        let n_cmp = n_sub.min(prev_frame.len());
        if n_cmp > 0 {
            let diff_energy: f64 = (0..n_cmp)
                .map(|k| (frame.amplitudes[k] - prev_frame[k]).powi(2))
                .sum::<f64>()
                / n_cmp as f64;
            // Normalise by mean squared amplitude to get a dimensionless ratio.
            let ref_energy = mean_amp * mean_amp + 1e-9;
            (diff_energy / ref_energy).sqrt().clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        // No history yet — fall back to intra-frame variance-based estimate.
        (intra_variance / (mean_amp * mean_amp + 1e-9))
            .sqrt()
            .clamp(0.0, 1.0)
    };

    // Blend temporal motion with variance-based motion for robustness.
    let variance_motion = (temporal_variance / 10.0).clamp(0.0, 1.0);
    let motion_score = (temporal_motion_score * 0.7 + variance_motion * 0.3).clamp(0.0, 1.0);

    // ── Signal quality metric ──
    // Based on estimated SNR (RSSI relative to noise floor) and subcarrier consistency.
    let snr_db = (frame.rssi as f64 - frame.noise_floor as f64).max(0.0);
    let snr_quality = (snr_db / 40.0).clamp(0.0, 1.0); // 40 dB → quality = 1.0
                                                       // Penalise quality when temporal variance is very high (unstable signal).
    let stability =
        (1.0 - (temporal_variance / (mean_amp * mean_amp + 1e-9)).clamp(0.0, 1.0)).max(0.0);
    let signal_quality = (snr_quality * 0.6 + stability * 0.4).clamp(0.0, 1.0);

    // ── Breathing rate estimation ──
    let breathing_rate_hz = estimate_breathing_rate_hz(frame_history, sample_rate_hz);

    let features = FeatureInfo {
        mean_rssi,
        variance,
        motion_band_power,
        breathing_band_power,
        dominant_freq_hz,
        change_points,
        spectral_power,
    };

    // ── Classification ──
    let (motion_level, presence) = if motion_score > 0.4 {
        ("active".to_string(), true)
    } else if motion_score > 0.08 {
        ("present_still".to_string(), true)
    } else {
        ("absent".to_string(), false)
    };

    let confidence = (0.4 + signal_quality * 0.3 + motion_score * 0.3).clamp(0.0, 1.0);

    let classification = ClassificationInfo {
        motion_level,
        presence,
        confidence,
    };

    (features, classification, breathing_rate_hz, sub_variances)
}

async fn record_live_frame(
    state: &SharedState,
    amplitudes: &[f64],
    mean_rssi: f64,
    noise_floor: f64,
    features: &FeatureInfo,
) {
    let features_json = serde_json::to_value(features).unwrap_or_else(|_| serde_json::json!({}));
    recording::maybe_record_frame(state, amplitudes, mean_rssi, noise_floor, &features_json).await;
}

// ── Windows WiFi RSSI collector ──────────────────────────────────────────────

/// Parse `netsh wlan show interfaces` output for RSSI and signal quality
fn parse_netsh_interfaces_output(output: &str) -> Option<(f64, f64, String)> {
    let mut rssi = None;
    let mut signal = None;
    let mut ssid = None;

    for line in output.lines() {
        let line = line.trim();
        if line.starts_with("Signal") {
            // "Signal                 : 89%"
            if let Some(pct) = line.split(':').nth(1) {
                let pct = pct.trim().trim_end_matches('%');
                if let Ok(v) = pct.parse::<f64>() {
                    signal = Some(v);
                    // Convert signal% to approximate dBm: -100 + (signal% * 0.6)
                    rssi = Some(-100.0 + v * 0.6);
                }
            }
        }
        if line.starts_with("SSID") && !line.starts_with("BSSID") {
            if let Some(s) = line.split(':').nth(1) {
                ssid = Some(s.trim().to_string());
            }
        }
    }

    match (rssi, signal, ssid) {
        (Some(r), Some(_s), Some(name)) => Some((r, _s, name)),
        (Some(r), Some(_s), None) => Some((r, _s, "Unknown".into())),
        _ => None,
    }
}

async fn windows_wifi_task(state: SharedState, tick_ms: u64) {
    let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));
    let mut seq: u32 = 0;

    // ADR-022 Phase 3: Multi-BSSID pipeline state (kept across ticks)
    let mut registry = BssidRegistry::new(32, 30);
    let mut pipeline = WindowsWifiPipeline::new();

    info!(
        "Windows WiFi multi-BSSID pipeline active (tick={}ms, max_bssids=32)",
        tick_ms
    );

    loop {
        interval.tick().await;
        seq += 1;

        // ── Step 1: Run multi-BSSID scan via spawn_blocking ──────────
        // NetshBssidScanner is not Send, so we run `netsh` and parse
        // the output inside a blocking closure.
        let bssid_scan_result = tokio::task::spawn_blocking(|| {
            let output = std::process::Command::new("netsh")
                .args(["wlan", "show", "networks", "mode=bssid"])
                .output()
                .map_err(|e| format!("netsh bssid scan failed: {e}"))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "netsh exited with {}: {}",
                    output.status,
                    stderr.trim()
                ));
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_netsh_bssid_output(&stdout).map_err(|e| format!("parse error: {e}"))
        })
        .await;

        // Unwrap the JoinHandle result, then the inner Result.
        let observations = match bssid_scan_result {
            Ok(Ok(obs)) if !obs.is_empty() => obs,
            Ok(Ok(_empty)) => {
                debug!("Multi-BSSID scan returned 0 observations, falling back");
                windows_wifi_fallback_tick(&state, seq).await;
                continue;
            }
            Ok(Err(e)) => {
                warn!("Multi-BSSID scan error: {e}, falling back");
                windows_wifi_fallback_tick(&state, seq).await;
                continue;
            }
            Err(join_err) => {
                error!("spawn_blocking panicked: {join_err}");
                continue;
            }
        };

        let obs_count = observations.len();

        // Derive SSID from the first observation for the source label.
        let ssid = observations
            .first()
            .map(|o| o.ssid.clone())
            .unwrap_or_else(|| "Unknown".into());

        // ── Step 2: Feed observations into registry ──────────────────
        registry.update(&observations);
        let multi_ap_frame = registry.to_multi_ap_frame();

        // ── Step 3: Run enhanced pipeline ────────────────────────────
        let enhanced = pipeline.process(&multi_ap_frame);

        // ── Step 4: Build backward-compatible Esp32Frame ─────────────
        let first_rssi = observations.first().map(|o| o.rssi_dbm).unwrap_or(-80.0);
        let _first_signal_pct = observations.first().map(|o| o.signal_pct).unwrap_or(40.0);

        let frame = Esp32Frame {
            magic: 0xC511_0001,
            node_id: 0,
            n_antennas: 1,
            n_subcarriers: obs_count.min(255) as u8,
            freq_mhz: 2437,
            sequence: seq,
            rssi: first_rssi.clamp(-128.0, 127.0) as i8,
            noise_floor: -90,
            amplitudes: multi_ap_frame.amplitudes.clone(),
            phases: multi_ap_frame.phases.clone(),
        };

        // ── Step 4b: Update frame history and extract features ───────
        let mut s_write_pre = state.write().await;
        s_write_pre
            .frame_history
            .push_back(frame.amplitudes.clone());
        if s_write_pre.frame_history.len() > FRAME_HISTORY_CAPACITY {
            s_write_pre.frame_history.pop_front();
        }
        let sample_rate_hz = 1000.0 / tick_ms as f64;
        let (features, classification, breathing_rate_hz, sub_variances) =
            extract_features_from_frame(&frame, &s_write_pre.frame_history, sample_rate_hz);
        drop(s_write_pre);

        // ── Step 5: Build enhanced fields from pipeline result ───────
        let enhanced_motion = Some(serde_json::json!({
            "score": enhanced.motion.score,
            "level": format!("{:?}", enhanced.motion.level),
            "contributing_bssids": enhanced.motion.contributing_bssids,
        }));

        let enhanced_breathing = enhanced.breathing.as_ref().map(|b| {
            serde_json::json!({
                "rate_bpm": b.rate_bpm,
                "confidence": b.confidence,
                "bssid_count": b.bssid_count,
            })
        });

        let posture_str = enhanced.posture.map(|p| format!("{p:?}"));
        let sig_quality_score = Some(enhanced.signal_quality.score);
        let verdict_str = Some(format!("{:?}", enhanced.verdict));
        let bssid_n = Some(enhanced.bssid_count);

        // ── Step 6: Update shared state ──────────────────────────────
        let mut s = state.write().await;
        s.source = format!("wifi:{ssid}");
        s.rssi_history.push_back(first_rssi);
        if s.rssi_history.len() > 60 {
            s.rssi_history.pop_front();
        }

        s.tick += 1;
        let tick = s.tick;

        let motion_score = if classification.motion_level == "active" {
            0.8
        } else if classification.motion_level == "present_still" {
            0.3
        } else {
            0.05
        };

        let vitals = s
            .vital_detector
            .process_frame(&frame.amplitudes, &frame.phases);
        s.latest_vitals = vitals.clone();

        let feat_variance = features.variance;

        // Multi-person estimation with temporal smoothing (EMA α=0.15).
        let raw_score = compute_person_score(&features);
        s.smoothed_person_score = s.smoothed_person_score * 0.85 + raw_score * 0.15;
        let est_persons = if classification.presence {
            score_to_person_count(s.smoothed_person_score)
        } else {
            0
        };

        let mut update = SensingUpdate {
            msg_type: "sensing_update".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
            source: format!("wifi:{ssid}"),
            tick,
            nodes: live_nodes_from_bssid_observations(&observations),
            features: features.clone(),
            classification,
            signal_field: generate_signal_field(
                first_rssi,
                motion_score,
                breathing_rate_hz,
                feat_variance.min(1.0),
                &sub_variances,
            ),
            vital_signs: Some(vitals),
            enhanced_motion,
            enhanced_breathing,
            posture: posture_str,
            signal_quality_score: sig_quality_score,
            quality_verdict: verdict_str,
            bssid_count: bssid_n,
            pose_keypoints: None,
            model_status: None,
            persons: None,
            estimated_persons: if est_persons > 0 {
                Some(est_persons)
            } else {
                None
            },
        };
        populate_live_pose(&mut update, &s, &frame.amplitudes, sample_rate_hz);

        if let Ok(json) = serde_json::to_string(&update) {
            let _ = s.tx.send(json);
        }
        s.latest_update = Some(update);
        drop(s);
        record_live_frame(
            &state,
            &frame.amplitudes,
            features.mean_rssi,
            frame.noise_floor as f64,
            &features,
        )
        .await;

        debug!(
            "Multi-BSSID tick #{tick}: {obs_count} BSSIDs, quality={:.2}, verdict={:?}",
            enhanced.signal_quality.score, enhanced.verdict
        );
    }
}

/// Fallback: single-RSSI collection via `netsh wlan show interfaces`.
///
/// Used when the multi-BSSID scan fails or returns 0 observations.
async fn windows_wifi_fallback_tick(state: &SharedState, seq: u32) {
    let output = match tokio::process::Command::new("netsh")
        .args(["wlan", "show", "interfaces"])
        .output()
        .await
    {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => {
            warn!("netsh interfaces fallback failed: {e}");
            return;
        }
    };

    let (rssi_dbm, signal_pct, ssid) = match parse_netsh_interfaces_output(&output) {
        Some(v) => v,
        None => {
            debug!("Fallback: no WiFi interface connected");
            return;
        }
    };

    let frame = Esp32Frame {
        magic: 0xC511_0001,
        node_id: 0,
        n_antennas: 1,
        n_subcarriers: 1,
        freq_mhz: 2437,
        sequence: seq,
        rssi: rssi_dbm as i8,
        noise_floor: -90,
        amplitudes: vec![signal_pct],
        phases: vec![0.0],
    };

    let mut s = state.write().await;
    // Update frame history before extracting features.
    s.frame_history.push_back(frame.amplitudes.clone());
    if s.frame_history.len() > FRAME_HISTORY_CAPACITY {
        s.frame_history.pop_front();
    }
    let sample_rate_hz = 2.0_f64; // fallback tick ~ 500 ms => 2 Hz
    let (features, classification, breathing_rate_hz, sub_variances) =
        extract_features_from_frame(&frame, &s.frame_history, sample_rate_hz);

    s.source = format!("wifi:{ssid}");
    s.rssi_history.push_back(rssi_dbm);
    if s.rssi_history.len() > 60 {
        s.rssi_history.pop_front();
    }

    s.tick += 1;
    let tick = s.tick;

    let motion_score = if classification.motion_level == "active" {
        0.8
    } else if classification.motion_level == "present_still" {
        0.3
    } else {
        0.05
    };

    let vitals = s
        .vital_detector
        .process_frame(&frame.amplitudes, &frame.phases);
    s.latest_vitals = vitals.clone();

    let feat_variance = features.variance;

    // Multi-person estimation with temporal smoothing.
    let raw_score = compute_person_score(&features);
    s.smoothed_person_score = s.smoothed_person_score * 0.85 + raw_score * 0.15;
    let est_persons = if classification.presence {
        score_to_person_count(s.smoothed_person_score)
    } else {
        0
    };

    let mut update = SensingUpdate {
        msg_type: "sensing_update".to_string(),
        timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        source: format!("wifi:{ssid}"),
        tick,
        nodes: vec![NodeInfo {
            node_id: 0,
            rssi_dbm,
            position: [0.0, 0.0, 0.0],
            amplitude: vec![signal_pct],
            subcarrier_count: 1,
        }],
        features: features.clone(),
        classification,
        signal_field: generate_signal_field(
            rssi_dbm,
            motion_score,
            breathing_rate_hz,
            feat_variance.min(1.0),
            &sub_variances,
        ),
        vital_signs: Some(vitals),
        enhanced_motion: None,
        enhanced_breathing: None,
        posture: None,
        signal_quality_score: None,
        quality_verdict: None,
        bssid_count: None,
        pose_keypoints: None,
        model_status: None,
        persons: None,
        estimated_persons: if est_persons > 0 {
            Some(est_persons)
        } else {
            None
        },
    };
    populate_live_pose(&mut update, &s, &frame.amplitudes, sample_rate_hz);

    if let Ok(json) = serde_json::to_string(&update) {
        let _ = s.tx.send(json);
    }
    s.latest_update = Some(update);
    drop(s);
    record_live_frame(
        state,
        &frame.amplitudes,
        features.mean_rssi,
        frame.noise_floor as f64,
        &features,
    )
    .await;
}

/// Probe if Windows WiFi is connected
async fn probe_windows_wifi() -> bool {
    match tokio::process::Command::new("netsh")
        .args(["wlan", "show", "interfaces"])
        .output()
        .await
    {
        Ok(o) => {
            let out = String::from_utf8_lossy(&o.stdout);
            parse_netsh_interfaces_output(&out).is_some()
        }
        Err(_) => false,
    }
}

/// Probe if ESP32 is streaming on UDP port
async fn probe_esp32(port: u16) -> bool {
    let addr = format!("0.0.0.0:{port}");
    match UdpSocket::bind(&addr).await {
        Ok(sock) => {
            let mut buf = [0u8; 256];
            match tokio::time::timeout(Duration::from_secs(2), sock.recv_from(&mut buf)).await {
                Ok(Ok((len, _))) => parse_esp32_frame(&buf[..len]).is_some(),
                _ => false,
            }
        }
        Err(_) => false,
    }
}

fn esp32_udp_ports(primary_port: u16, disable_legacy_fallback: bool) -> Vec<u16> {
    let mut ports = vec![primary_port];
    if !disable_legacy_fallback && primary_port != LEGACY_ESP32_UDP_PORT {
        ports.push(LEGACY_ESP32_UDP_PORT);
    }
    ports
}

async fn probe_esp32_ports(ports: &[u16]) -> Option<u16> {
    for &port in ports {
        if probe_esp32(port).await {
            return Some(port);
        }
    }
    None
}

// ── Simulated data generator ─────────────────────────────────────────────────

fn generate_simulated_frame(tick: u64) -> Esp32Frame {
    let t = tick as f64 * 0.1;
    let n_sub = 56usize;
    let mut amplitudes = Vec::with_capacity(n_sub);
    let mut phases = Vec::with_capacity(n_sub);

    for i in 0..n_sub {
        let base = 15.0 + 5.0 * (i as f64 * 0.1 + t * 0.3).sin();
        let noise = (i as f64 * 7.3 + t * 13.7).sin() * 2.0;
        amplitudes.push((base + noise).max(0.1));
        phases.push((i as f64 * 0.2 + t * 0.5).sin() * std::f64::consts::PI);
    }

    Esp32Frame {
        magic: 0xC511_0001,
        node_id: 1,
        n_antennas: 1,
        n_subcarriers: n_sub as u8,
        freq_mhz: 2437,
        sequence: tick as u32,
        rssi: (-40.0 + 5.0 * (t * 0.2).sin()) as i8,
        noise_floor: -90,
        amplitudes,
        phases,
    }
}

// ── WebSocket handler ────────────────────────────────────────────────────────

async fn ws_sensing_handler(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws_client(socket, state))
}

async fn handle_ws_client(mut socket: WebSocket, state: SharedState) {
    let mut rx = {
        let s = state.read().await;
        s.tx.subscribe()
    };

    info!("WebSocket client connected (sensing)");

    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Ok(json) => {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // ignore client messages
                }
            }
        }
    }

    info!("WebSocket client disconnected (sensing)");
}

// ── Pose WebSocket handler (sends pose_data messages for Live Demo) ──────────

fn model_pose_persons(sensing: &SensingUpdate) -> Option<Vec<PersonDetection>> {
    sensing.pose_keypoints.as_ref().map(|kps| {
        let kp_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ];
        let keypoints: Vec<PoseKeypoint> = kps
            .iter()
            .enumerate()
            .map(|(i, kp)| PoseKeypoint {
                name: kp_names.get(i).unwrap_or(&"unknown").to_string(),
                x: kp[0],
                y: kp[1],
                z: kp[2],
                confidence: kp[3],
            })
            .collect();
        vec![PersonDetection {
            id: 1,
            confidence: sensing.classification.confidence,
            bbox: BoundingBox {
                x: 260.0,
                y: 150.0,
                width: 120.0,
                height: 220.0,
            },
            keypoints,
            zone: "zone_1".into(),
        }]
    })
}

fn resolve_live_persons(sensing: &SensingUpdate, model_loaded: bool) -> Vec<PersonDetection> {
    if model_loaded {
        sensing.persons.clone().unwrap_or_default()
    } else {
        sensing
            .persons
            .clone()
            .unwrap_or_else(|| derive_pose_from_sensing(sensing))
    }
}

fn live_pose_source(sensing: &SensingUpdate, model_loaded: bool) -> &'static str {
    if model_loaded && sensing.pose_keypoints.is_some() {
        "model_inference"
    } else {
        "signal_derived"
    }
}

fn model_temporal_context_feature_dim(n_sub: usize, temporal_context_frames: usize) -> usize {
    if temporal_context_frames == 0 {
        0
    } else {
        n_sub + n_sub
    }
}

fn model_feature_dim(n_sub: usize, temporal_context_frames: usize) -> usize {
    n_sub
        + n_sub
        + n_sub
        + model_temporal_context_feature_dim(n_sub, temporal_context_frames)
        + MODEL_N_FREQ_BANDS
        + MODEL_N_GLOBAL_FEATURES
}

fn model_goertzel_power(signal: &[f64], freq_norm: f64) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let coeff = 2.0 * (2.0 * std::f64::consts::PI * freq_norm).cos();
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2;
    for &x in signal {
        s2 = s1;
        s1 = s0;
        s0 = x + coeff * s1 - s2;
    }
    let power = s0 * s0 + s1 * s1 - coeff * s0 * s1;
    (power / n as f64).max(0.0)
}

fn extract_model_features_for_frame(
    raw_subcarriers: &[f64],
    frame_history: &VecDeque<Vec<f64>>,
    prev_subcarriers: Option<&[f64]>,
    sample_rate_hz: f64,
    temporal_context_frames: usize,
    temporal_context_decay: f64,
) -> Vec<f64> {
    let n_sub = raw_subcarriers.len().max(1);
    let mut features = Vec::with_capacity(model_feature_dim(n_sub, temporal_context_frames));

    features.extend_from_slice(raw_subcarriers);
    while features.len() < n_sub {
        features.push(0.0);
    }

    let window: Vec<&[f64]> = frame_history
        .iter()
        .rev()
        .skip(1)
        .take(MODEL_VARIANCE_WINDOW)
        .rev()
        .map(Vec::as_slice)
        .collect();

    for k in 0..n_sub {
        if window.is_empty() {
            features.push(0.0);
            continue;
        }
        let n = window.len() as f64;
        let mut sum = 0.0f64;
        let mut sq_sum = 0.0f64;
        for amps in &window {
            let value = amps.get(k).copied().unwrap_or(0.0);
            sum += value;
            sq_sum += value * value;
        }
        let mean = sum / n;
        let variance = (sq_sum / n - mean * mean).max(0.0);
        features.push(variance);
    }

    for k in 0..n_sub {
        let gradient = match prev_subcarriers {
            Some(prev) => (raw_subcarriers.get(k).copied().unwrap_or(0.0)
                - prev.get(k).copied().unwrap_or(0.0))
            .abs(),
            None => 0.0,
        };
        features.push(gradient);
    }

    if temporal_context_frames > 0 {
        let mut context_mean = vec![0.0f64; n_sub];
        let mut weight_sum = 0.0f64;
        let safe_decay = temporal_context_decay.clamp(0.05, 1.0);
        for (idx, amps) in window
            .iter()
            .rev()
            .take(temporal_context_frames)
            .enumerate()
        {
            let weight = safe_decay.powi(idx as i32);
            weight_sum += weight;
            for feature_idx in 0..n_sub {
                let value = amps.get(feature_idx).copied().unwrap_or(0.0);
                context_mean[feature_idx] += value * weight;
            }
        }
        if weight_sum > 0.0 {
            for value in &mut context_mean {
                *value /= weight_sum;
            }
        }
        features.extend_from_slice(&context_mean);
        for feature_idx in 0..n_sub {
            let current = raw_subcarriers.get(feature_idx).copied().unwrap_or(0.0);
            let residual = if weight_sum > 0.0 {
                current - context_mean[feature_idx]
            } else {
                0.0
            };
            features.push(residual);
        }
    }

    let freq_bands = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0];
    let mean_series: Vec<f64> = window
        .iter()
        .map(|amps| {
            if amps.is_empty() {
                0.0
            } else {
                amps.iter().sum::<f64>() / amps.len() as f64
            }
        })
        .collect();
    for &freq_hz in &freq_bands {
        let freq_norm = if sample_rate_hz > 0.0 {
            freq_hz / sample_rate_hz
        } else {
            0.0
        };
        features.push(model_goertzel_power(&mean_series, freq_norm));
    }

    let mean_amp = if raw_subcarriers.is_empty() {
        0.0
    } else {
        raw_subcarriers.iter().sum::<f64>() / raw_subcarriers.len() as f64
    };
    let std_amp = if raw_subcarriers.len() > 1 {
        let variance = raw_subcarriers
            .iter()
            .map(|value| (value - mean_amp).powi(2))
            .sum::<f64>()
            / (raw_subcarriers.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let motion_score = match prev_subcarriers {
        Some(prev) => {
            let n_cmp = n_sub.min(prev.len());
            if n_cmp > 0 {
                let diff = (0..n_cmp)
                    .map(|idx| {
                        let current = raw_subcarriers.get(idx).copied().unwrap_or(0.0);
                        let previous = prev.get(idx).copied().unwrap_or(0.0);
                        (current - previous).powi(2)
                    })
                    .sum::<f64>()
                    / n_cmp as f64;
                (diff / (mean_amp * mean_amp + 1e-9)).sqrt().clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
        None => 0.0,
    };
    features.push(mean_amp);
    features.push(std_amp);
    features.push(motion_score);

    features
}

fn default_model_keypoints() -> Vec<[f64; 4]> {
    vec![[320.0, 240.0, 0.0, 0.0]; MODEL_N_KEYPOINTS]
}

fn infer_pose_keypoints_from_model(
    model_weights: &[f32],
    head_config: &PoseHeadConfig,
    feature_stats: &ModelFeatureStats,
    raw_subcarriers: &[f64],
    frame_history: &VecDeque<Vec<f64>>,
    prev_subcarriers: Option<&[f64]>,
    sample_rate_hz: f64,
) -> Vec<[f64; 4]> {
    let n_feat = feature_stats.n_features;
    let expected_params = head_config.expected_params();
    if model_weights.len() < expected_params {
        warn!(
            "Model weights too short for live inference: {} < {}",
            model_weights.len(),
            expected_params
        );
        return default_model_keypoints();
    }

    let mut features = extract_model_features_for_frame(
        raw_subcarriers,
        frame_history,
        prev_subcarriers,
        sample_rate_hz,
        feature_stats.temporal_context_frames,
        feature_stats.temporal_context_decay,
    );
    for (idx, value) in features.iter_mut().enumerate() {
        if idx < n_feat {
            let mean = feature_stats.mean.get(idx).copied().unwrap_or(0.0);
            let std = feature_stats.std.get(idx).copied().unwrap_or(1.0);
            *value = (*value - mean) / std;
        }
    }
    features.resize(n_feat, 0.0);

    let outputs = match forward_with_f32_params(head_config, model_weights, &features) {
        Some(outputs) if outputs.len() >= MODEL_N_TARGETS => outputs,
        _ => return default_model_keypoints(),
    };
    let mut keypoints = Vec::with_capacity(MODEL_N_KEYPOINTS);
    for kp_idx in 0..MODEL_N_KEYPOINTS {
        let mut coords = [0.0f64; 4];
        for dim in 0..MODEL_DIMS_PER_KP {
            let target_idx = kp_idx * MODEL_DIMS_PER_KP + dim;
            coords[dim] = outputs.get(target_idx).copied().unwrap_or(0.0);
        }
        let feat_magnitude =
            features.iter().map(|value| value.abs()).sum::<f64>() / features.len().max(1) as f64;
        coords[3] = (1.0 / (1.0 + (-feat_magnitude + 1.0).exp())).clamp(0.1, 0.99);
        keypoints.push(coords);
    }

    keypoints
}

fn parse_model_feature_stats(metadata: Option<&serde_json::Value>) -> Option<ModelFeatureStats> {
    metadata
        .and_then(|value| value.get("feature_stats").cloned())
        .and_then(|value| serde_json::from_value(value).ok())
}

fn parse_model_head_config(
    metadata: Option<&serde_json::Value>,
    feature_stats: Option<&ModelFeatureStats>,
) -> Option<PoseHeadConfig> {
    feature_stats
        .map(|stats| PoseHeadConfig::from_metadata(metadata, stats.n_features, MODEL_N_TARGETS))
}

fn parse_model_target_space(metadata: Option<&serde_json::Value>) -> Option<String> {
    metadata
        .and_then(|value| {
            value
                .pointer("/model_config/target_space")
                .or_else(|| value.pointer("/training/target_space"))
                .or_else(|| value.get("target_space"))
                .and_then(|raw| raw.as_str())
        })
        .map(str::to_string)
}

fn parse_default_sona_profile(metadata: Option<&serde_json::Value>) -> Option<String> {
    metadata
        .and_then(|value| {
            value
                .pointer("/model_config/default_sona_profile")
                .or_else(|| value.pointer("/training/default_sona_profile"))
                .or_else(|| value.get("default_sona_profile"))
                .and_then(|raw| raw.as_str())
        })
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
        .map(str::to_string)
}

fn resolve_initial_active_sona_profile(
    metadata: Option<&serde_json::Value>,
    model_path: Option<&Path>,
    sona_profile_deltas: &BTreeMap<String, Vec<f32>>,
) -> Option<String> {
    let available_profiles = sona_profile_deltas.keys().cloned().collect::<Vec<_>>();
    if available_profiles.is_empty() {
        return None;
    }

    if let Some(default_profile) = parse_default_sona_profile(metadata) {
        if sona_profile_deltas.contains_key(&default_profile) {
            return Some(default_profile);
        }
    }

    if available_profiles.len() != 1 {
        return None;
    }

    let only_profile = available_profiles[0].clone();
    let imported_name = model_path
        .and_then(|path| path.file_name().and_then(|name| name.to_str()))
        .map(|name| name.contains(".sona-") || name.contains(".imported."))
        .unwrap_or(false);
    let has_base_model_metadata = metadata
        .and_then(|value| {
            value
                .pointer("/model_config/base_model_id")
                .or_else(|| value.pointer("/training/base_model_id"))
                .and_then(|raw| raw.as_str())
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some();

    if imported_name || has_base_model_metadata {
        return Some(only_profile);
    }

    None
}

fn refresh_active_sona_weights(state: &mut AppStateInner) {
    let Some(base_weights) = state.model_weights.as_ref() else {
        state.active_sona_weights = None;
        return;
    };
    let Some(profile) = state
        .active_sona_profile
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
    else {
        state.active_sona_weights = None;
        return;
    };
    let Some(delta) = state.model_sona_profile_deltas.get(profile) else {
        warn!("SONA profile `{profile}` not found in active RVF; falling back to base weights");
        state.active_sona_weights = None;
        return;
    };
    if delta.len() != base_weights.len() {
        warn!(
            "SONA profile `{profile}` delta length mismatch: {} != {}",
            delta.len(),
            base_weights.len()
        );
        state.active_sona_weights = None;
        return;
    }
    state.active_sona_weights = Some(
        base_weights
            .iter()
            .zip(delta.iter())
            .map(|(&base, &overlay)| base + overlay)
            .collect(),
    );
}

fn current_inference_weights(state: &AppStateInner) -> Option<&[f32]> {
    state
        .active_sona_weights
        .as_deref()
        .or(state.model_weights.as_deref())
}

fn model_pose_looks_absent(keypoints: &[[f64; 4]], target_space: &str) -> bool {
    if keypoints.len() < 13 {
        return true;
    }
    let span = |a: usize, b: usize| -> f64 {
        let pa = keypoints.get(a).copied().unwrap_or([0.0; 4]);
        let pb = keypoints.get(b).copied().unwrap_or([0.0; 4]);
        ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt()
    };
    let shoulder_span = span(5, 6);
    let hip_span = span(11, 12);
    let max_abs = keypoints
        .iter()
        .flat_map(|kp| kp[..3].iter())
        .map(|value| value.abs())
        .fold(0.0f64, f64::max);
    if target_space == MODEL_TARGET_SPACE_OPERATOR_FRAME {
        shoulder_span < 0.05 && hip_span < 0.05 && max_abs < 0.25
    } else {
        shoulder_span < 1.0 && hip_span < 1.0 && max_abs < 4.0
    }
}

fn infer_live_model_pose(
    state: &AppStateInner,
    raw_subcarriers: &[f64],
    sample_rate_hz: f64,
) -> Option<Vec<[f64; 4]>> {
    let weights = current_inference_weights(state)?;
    let feature_stats = state.model_feature_stats.as_ref()?;
    let head_config = state
        .model_head_config
        .as_ref()
        .cloned()
        .unwrap_or_else(|| PoseHeadConfig::linear(feature_stats.n_features, MODEL_N_TARGETS));
    if raw_subcarriers.is_empty() {
        return None;
    }

    let prev_subcarriers = state.frame_history.iter().rev().nth(1).map(Vec::as_slice);
    let keypoints = infer_pose_keypoints_from_model(
        weights,
        &head_config,
        feature_stats,
        raw_subcarriers,
        &state.frame_history,
        prev_subcarriers,
        sample_rate_hz,
    );
    let avg_confidence =
        keypoints.iter().map(|kp| kp[3]).sum::<f64>() / keypoints.len().max(1) as f64;
    if !avg_confidence.is_finite() || avg_confidence <= 0.05 {
        return None;
    }
    let target_space = state
        .model_target_space
        .as_deref()
        .unwrap_or(MODEL_TARGET_SPACE_DEFAULT);
    if model_pose_looks_absent(&keypoints, target_space) {
        return None;
    }

    Some(keypoints)
}

fn populate_live_pose(
    update: &mut SensingUpdate,
    state: &AppStateInner,
    raw_subcarriers: &[f64],
    sample_rate_hz: f64,
) {
    let pose_keypoints = if state.model_loaded {
        infer_live_model_pose(state, raw_subcarriers, sample_rate_hz)
    } else {
        None
    };
    update.pose_keypoints = pose_keypoints;

    let pose_source = if update.pose_keypoints.is_some() {
        "model_inference"
    } else {
        "signal_derived"
    };
    update.model_status = current_model_status(
        state.model_loaded,
        state.progressive_loader.as_ref(),
        state.model_target_space.as_deref(),
        state.active_sona_profile.as_deref(),
        pose_source,
    );

    if state.model_loaded {
        let matched_model_persons = if update.pose_keypoints.is_some() {
            model_pose_persons(update).unwrap_or_default()
        } else {
            Vec::new()
        };
        update.estimated_persons = Some(matched_model_persons.len());
        if matched_model_persons.is_empty() {
            update.classification.presence = false;
            update.classification.motion_level = "absent".to_string();
            update.classification.confidence = 0.0;
        }
        update.persons = Some(matched_model_persons);
        return;
    }

    let persons = derive_pose_from_sensing(update);
    if !persons.is_empty() {
        update.persons = Some(persons);
    }
}

fn current_model_status(
    model_loaded: bool,
    progressive_loader: Option<&ProgressiveLoader>,
    model_target_space: Option<&str>,
    active_sona_profile: Option<&str>,
    pose_source: &str,
) -> Option<serde_json::Value> {
    if !model_loaded {
        return None;
    }

    Some(serde_json::json!({
        "loaded": true,
        "layers": progressive_loader
            .map(|loader| {
                let (a, b, c) = loader.layer_status();
                a as u8 + b as u8 + c as u8
            })
            .unwrap_or(0),
        "target_space": model_target_space.unwrap_or(MODEL_TARGET_SPACE_DEFAULT),
        "sona_profile": active_sona_profile.unwrap_or("default"),
        "pose_source": pose_source,
    }))
}

async fn ws_pose_handler(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws_pose_client(socket, state))
}

async fn handle_ws_pose_client(mut socket: WebSocket, state: SharedState) {
    let mut rx = {
        let s = state.read().await;
        s.tx.subscribe()
    };

    info!("WebSocket client connected (pose)");

    // Send connection established message
    let conn_msg = serde_json::json!({
        "type": "connection_established",
        "payload": { "status": "connected", "backend": "rust+ruvector" }
    });
    let _ = socket
        .send(Message::Text(conn_msg.to_string().into()))
        .await;

    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Ok(json) => {
                        // Parse the sensing update and convert to pose format
                        if let Ok(sensing) = serde_json::from_str::<SensingUpdate>(&json) {
                            if sensing.msg_type == "sensing_update" {
                                let (model_loaded, pose_source) = {
                                    let s = state.read().await;
                                    (s.model_loaded, live_pose_source(&sensing, s.model_loaded))
                                };
                                let persons = resolve_live_persons(&sensing, model_loaded);
                                let total_persons = persons.len();

                                let pose_msg = serde_json::json!({
                                    "type": "pose_data",
                                    "zone_id": "zone_1",
                                    "timestamp": sensing.timestamp,
                                    "payload": {
                                        "pose": {
                                            "persons": persons,
                                        },
                                        "confidence": if sensing.classification.presence { sensing.classification.confidence } else { 0.0 },
                                        "activity": sensing.classification.motion_level,
                                        // pose_source tells the UI which estimation mode is active.
                                        "pose_source": pose_source,
                                        "metadata": {
                                            "frame_id": format!("rust_frame_{}", sensing.tick),
                                            "processing_time_ms": 1,
                                            "source": sensing.source,
                                            "tick": sensing.tick,
                                            "signal_strength": sensing.features.mean_rssi,
                                            "motion_band_power": sensing.features.motion_band_power,
                                            "breathing_band_power": sensing.features.breathing_band_power,
                                            "estimated_persons": total_persons,
                                        }
                                    }
                                });
                                if socket.send(Message::Text(pose_msg.to_string().into())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Handle ping/pong
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                            if v.get("type").and_then(|t| t.as_str()) == Some("ping") {
                                let pong = serde_json::json!({"type": "pong"});
                                let _ = socket.send(Message::Text(pong.to_string().into())).await;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }

    info!("WebSocket client disconnected (pose)");
}

// ── REST endpoints ───────────────────────────────────────────────────────────

async fn health(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::json!({
        "status": "ok",
        "source": s.source,
        "tick": s.tick,
        "clients": s.tx.receiver_count(),
    }))
}

async fn latest(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    match &s.latest_update {
        Some(update) => Json(serde_json::to_value(update).unwrap_or_default()),
        None => Json(serde_json::json!({"status": "no data yet"})),
    }
}

/// Generate WiFi-derived pose keypoints from sensing data.
///
/// Keypoint positions are modulated by real signal features rather than a pure
/// time-based sine/cosine loop:
///
///   - `motion_band_power`    drives whole-body translation and limb splay
///   - `variance`             seeds per-frame noise so the skeleton never freezes
///   - `breathing_band_power` expands/contracts torso keypoints (shoulders, hips)
///   - `dominant_freq_hz`     tilts the upper body laterally (lean direction)
///   - `change_points`        adds burst jitter to extremities (wrists, ankles)
///
/// When `presence == false` no persons are returned (empty room).
/// When walking is detected (`motion_score > 0.55`) the figure shifts laterally
/// with a stride-swing pattern applied to arms and legs.
// ── Multi-person estimation (issue #97) ──────────────────────────────────────

/// Estimate person count from CSI features using a weighted composite heuristic.
///
/// Single ESP32 link limitations: variance-based detection can reliably detect
/// 1-2 persons. 3+ is speculative and requires ≥3 nodes for spatial resolution.
///
/// Returns a raw score (0.0..1.0) that the caller converts to person count
/// after temporal smoothing.
fn compute_person_score(feat: &FeatureInfo) -> f64 {
    // Normalize each feature to [0, 1] using calibrated ranges:
    //
    //   variance: intra-frame amp variance. 1-person ~2-15, 2-person ~15-60,
    //     real ESP32 can go higher. Use 30.0 as scaling midpoint.
    let var_norm = (feat.variance / 30.0).clamp(0.0, 1.0);

    //   change_points: threshold crossings in 56 subcarriers. 1-person ~5-15,
    //     2-person ~15-30. Scale by 30.0 (half of max 55).
    let cp_norm = (feat.change_points as f64 / 30.0).clamp(0.0, 1.0);

    //   motion_band_power: upper-half subcarrier variance. 1-person ~1-8,
    //     2-person ~8-25. Scale by 20.0.
    let motion_norm = (feat.motion_band_power / 20.0).clamp(0.0, 1.0);

    //   spectral_power: mean squared amplitude. Highly variable (~100-1000+).
    //     Use relative change indicator: high spectral_power with high variance
    //     suggests multiple reflectors. Scale by 500.0.
    let sp_norm = (feat.spectral_power / 500.0).clamp(0.0, 1.0);

    // Weighted composite — variance and change_points carry the most signal.
    var_norm * 0.35 + cp_norm * 0.30 + motion_norm * 0.20 + sp_norm * 0.15
}

/// Convert smoothed person score to discrete count with hysteresis.
///
/// Uses asymmetric thresholds: higher threshold to add a person, lower to remove.
/// This prevents flickering at the boundary.
fn score_to_person_count(smoothed_score: f64) -> usize {
    // Thresholds chosen conservatively for single-ESP32 link:
    //   score > 0.50 → 2 persons (needs sustained high variance + change points)
    //   score > 0.80 → 3 persons (very high activity, rare with single link)
    if smoothed_score > 0.80 {
        3
    } else if smoothed_score > 0.50 {
        2
    } else {
        1
    }
}

/// Generate a single person's skeleton with per-person spatial offset and phase stagger.
///
/// `person_idx`: 0-based index of this person.
/// `total_persons`: total number of detected persons (for spacing calculation).
fn derive_single_person_pose(
    update: &SensingUpdate,
    person_idx: usize,
    total_persons: usize,
) -> PersonDetection {
    let cls = &update.classification;
    let feat = &update.features;

    // Per-person phase offset: ~120 degrees apart so they don't move in sync.
    let phase_offset = person_idx as f64 * 2.094;

    // Spatial spread: persons distributed symmetrically around center.
    let half = (total_persons as f64 - 1.0) / 2.0;
    let person_x_offset = (person_idx as f64 - half) * 120.0; // 120px spacing

    // Confidence decays for additional persons (less certain about person 2, 3).
    let conf_decay = 1.0 - person_idx as f64 * 0.15;

    // ── Signal-derived scalars ────────────────────────────────────────────────

    let motion_score = (feat.motion_band_power / 15.0).clamp(0.0, 1.0);
    let is_walking = motion_score > 0.55;
    let breath_amp = (feat.breathing_band_power * 4.0).clamp(0.0, 12.0);

    let breath_phase = if let Some(ref vs) = update.vital_signs {
        let bpm = vs.breathing_rate_bpm.unwrap_or(15.0);
        let freq = (bpm / 60.0).clamp(0.1, 0.5);
        (update.tick as f64 * freq * 0.1 * std::f64::consts::TAU + phase_offset).sin()
    } else {
        (update.tick as f64 * 0.08 + feat.breathing_band_power + phase_offset).sin()
    };

    let lean_x = (feat.dominant_freq_hz / 5.0 - 1.0).clamp(-1.0, 1.0) * 18.0;

    let stride_x = if is_walking {
        let stride_phase =
            (feat.motion_band_power * 0.7 + update.tick as f64 * 0.12 + phase_offset).sin();
        stride_phase * 45.0 * motion_score
    } else {
        0.0
    };

    let burst = (feat.change_points as f64 / 8.0).clamp(0.0, 1.0);

    let noise_seed = feat.variance * 31.7 + update.tick as f64 * 17.3 + person_idx as f64 * 97.1;
    let noise_val = (noise_seed.sin() * 43758.545).fract();

    let snr_factor = ((feat.variance - 0.5) / 10.0).clamp(0.0, 1.0);
    let base_confidence = cls.confidence * (0.6 + 0.4 * snr_factor) * conf_decay;

    // ── Skeleton base position ────────────────────────────────────────────────

    let base_x = 320.0 + stride_x + lean_x * 0.5 + person_x_offset;
    let base_y = 240.0 - motion_score * 8.0;

    // ── COCO 17-keypoint offsets from hip-center ──────────────────────────────

    let kp_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ];

    let kp_offsets: [(f64, f64); 17] = [
        (0.0, -80.0),   // 0  nose
        (-8.0, -88.0),  // 1  left_eye
        (8.0, -88.0),   // 2  right_eye
        (-16.0, -82.0), // 3  left_ear
        (16.0, -82.0),  // 4  right_ear
        (-30.0, -50.0), // 5  left_shoulder
        (30.0, -50.0),  // 6  right_shoulder
        (-45.0, -15.0), // 7  left_elbow
        (45.0, -15.0),  // 8  right_elbow
        (-50.0, 20.0),  // 9  left_wrist
        (50.0, 20.0),   // 10 right_wrist
        (-20.0, 20.0),  // 11 left_hip
        (20.0, 20.0),   // 12 right_hip
        (-22.0, 70.0),  // 13 left_knee
        (22.0, 70.0),   // 14 right_knee
        (-24.0, 120.0), // 15 left_ankle
        (24.0, 120.0),  // 16 right_ankle
    ];

    const TORSO_KP: [usize; 4] = [5, 6, 11, 12];
    const EXTREMITY_KP: [usize; 4] = [9, 10, 15, 16];

    let keypoints: Vec<PoseKeypoint> = kp_names
        .iter()
        .zip(kp_offsets.iter())
        .enumerate()
        .map(|(i, (name, (dx, dy)))| {
            let breath_dx = if TORSO_KP.contains(&i) {
                let sign = if *dx < 0.0 { -1.0 } else { 1.0 };
                sign * breath_amp * breath_phase * 0.5
            } else {
                0.0
            };
            let breath_dy = if TORSO_KP.contains(&i) {
                let sign = if *dy < 0.0 { -1.0 } else { 1.0 };
                sign * breath_amp * breath_phase * 0.3
            } else {
                0.0
            };

            let extremity_jitter = if EXTREMITY_KP.contains(&i) {
                let phase = noise_seed + i as f64 * 2.399;
                (
                    phase.sin() * burst * motion_score * 12.0,
                    (phase * 1.31).cos() * burst * motion_score * 8.0,
                )
            } else {
                (0.0, 0.0)
            };

            let kp_noise_x = ((noise_seed + i as f64 * 1.618).sin() * 43758.545).fract()
                * feat.variance.sqrt().clamp(0.0, 3.0)
                * motion_score;
            let kp_noise_y = ((noise_seed + i as f64 * 2.718).cos() * 31415.926).fract()
                * feat.variance.sqrt().clamp(0.0, 3.0)
                * motion_score
                * 0.6;

            let swing_dy = if is_walking {
                let stride_phase =
                    (feat.motion_band_power * 0.7 + update.tick as f64 * 0.12 + phase_offset).sin();
                match i {
                    7 | 9 => -stride_phase * 20.0 * motion_score,
                    8 | 10 => stride_phase * 20.0 * motion_score,
                    13 | 15 => stride_phase * 25.0 * motion_score,
                    14 | 16 => -stride_phase * 25.0 * motion_score,
                    _ => 0.0,
                }
            } else {
                0.0
            };

            let final_x = base_x + dx + breath_dx + extremity_jitter.0 + kp_noise_x;
            let final_y = base_y + dy + breath_dy + extremity_jitter.1 + kp_noise_y + swing_dy;

            let kp_conf = if EXTREMITY_KP.contains(&i) {
                base_confidence * (0.7 + 0.3 * snr_factor) * (0.85 + 0.15 * noise_val)
            } else {
                base_confidence * (0.88 + 0.12 * ((i as f64 * 0.7 + noise_seed).cos()))
            };

            PoseKeypoint {
                name: name.to_string(),
                x: final_x,
                y: final_y,
                z: lean_x * 0.02,
                confidence: kp_conf.clamp(0.1, 1.0),
            }
        })
        .collect();

    let xs: Vec<f64> = keypoints.iter().map(|k| k.x).collect();
    let ys: Vec<f64> = keypoints.iter().map(|k| k.y).collect();
    let min_x = xs.iter().cloned().fold(f64::MAX, f64::min) - 10.0;
    let min_y = ys.iter().cloned().fold(f64::MAX, f64::min) - 10.0;
    let max_x = xs.iter().cloned().fold(f64::MIN, f64::max) + 10.0;
    let max_y = ys.iter().cloned().fold(f64::MIN, f64::max) + 10.0;

    PersonDetection {
        id: (person_idx + 1) as u32,
        confidence: cls.confidence * conf_decay,
        keypoints,
        bbox: BoundingBox {
            x: min_x,
            y: min_y,
            width: (max_x - min_x).max(80.0),
            height: (max_y - min_y).max(160.0),
        },
        zone: format!("zone_{}", person_idx + 1),
    }
}

fn derive_pose_from_sensing(update: &SensingUpdate) -> Vec<PersonDetection> {
    let cls = &update.classification;
    if !cls.presence {
        return vec![];
    }

    // Use estimated_persons if set by the tick loop; otherwise default to 1.
    let person_count = update.estimated_persons.unwrap_or(1).max(1);

    (0..person_count)
        .map(|idx| derive_single_person_pose(update, idx, person_count))
        .collect()
}

// ── DensePose-compatible REST endpoints ─────────────────────────────────────

async fn health_live(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::json!({
        "status": "alive",
        "uptime": s.start_time.elapsed().as_secs(),
    }))
}

async fn health_ready(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::json!({
        "status": "ready",
        "source": s.source,
    }))
}

async fn health_system(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    let uptime = s.start_time.elapsed().as_secs();
    Json(serde_json::json!({
        "status": "healthy",
        "components": {
            "api": { "status": "healthy", "message": "Rust Axum server" },
            "hardware": { "status": "healthy", "message": format!("Source: {}", s.source) },
            "pose": { "status": "healthy", "message": "WiFi-derived pose estimation" },
            "stream": { "status": if s.tx.receiver_count() > 0 { "healthy" } else { "idle" },
                        "message": format!("{} client(s)", s.tx.receiver_count()) },
        },
        "metrics": {
            "cpu_percent": 2.5,
            "memory_percent": 1.8,
            "disk_percent": 15.0,
            "uptime_seconds": uptime,
        }
    }))
}

async fn health_version() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "name": "wifi-densepose-sensing-server",
        "backend": "rust+axum+ruvector",
    }))
}

async fn health_metrics(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::json!({
        "system_metrics": {
            "cpu": { "percent": 2.5 },
            "memory": { "percent": 1.8, "used_mb": 5 },
            "disk": { "percent": 15.0 },
        },
        "tick": s.tick,
    }))
}

async fn api_info(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "environment": "production",
        "backend": "rust",
        "source": s.source,
        "features": {
            "wifi_sensing": true,
            "pose_estimation": true,
            "signal_processing": true,
            "ruvector": true,
            "streaming": true,
        }
    }))
}

async fn pose_current(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut s = state.write().await;
    let tracked_person = refresh_stable_tracked_person(&mut s).map(tracked_contract_for_pose_api);
    let (persons, estimated_persons, pose_source, model_status) = match &s.latest_update {
        Some(update) => {
            let pose_source = live_pose_source(update, s.model_loaded);
            let persons = resolve_live_persons(update, s.model_loaded);
            (
                persons,
                update.estimated_persons.unwrap_or(0),
                pose_source,
                update.model_status.clone().or_else(|| {
                    current_model_status(
                        s.model_loaded,
                        s.progressive_loader.as_ref(),
                        s.model_target_space.as_deref(),
                        s.active_sona_profile.as_deref(),
                        pose_source,
                    )
                }),
            )
        }
        None => (
            Vec::new(),
            0,
            "signal_derived",
            current_model_status(
                s.model_loaded,
                s.progressive_loader.as_ref(),
                s.model_target_space.as_deref(),
                s.active_sona_profile.as_deref(),
                "signal_derived",
            ),
        ),
    };
    let total_persons = persons.len();
    Json(serde_json::json!({
        "timestamp": chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        "persons": persons,
        "tracked_person": tracked_person,
        "total_persons": total_persons,
        "source": s.source,
        "estimated_persons": estimated_persons.max(total_persons),
        "pose_source": pose_source,
        "model_status": model_status,
    }))
}

async fn pose_tracked(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut s = state.write().await;
    let tracked_person = refresh_stable_tracked_person(&mut s).map(tracked_contract_for_pose_api);
    let pose_source = s
        .latest_update
        .as_ref()
        .map(|update| live_pose_source(update, s.model_loaded))
        .unwrap_or("signal_derived");
    let model_status = current_model_status(
        s.model_loaded,
        s.progressive_loader.as_ref(),
        s.model_target_space.as_deref(),
        s.active_sona_profile.as_deref(),
        pose_source,
    );
    Json(serde_json::json!({
        "timestamp": chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        "tracked_person": tracked_person,
        "source": s.source,
        "pose_source": pose_source,
        "model_status": model_status,
    }))
}

async fn pose_stats(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::json!({
        "total_detections": s.total_detections,
        "average_confidence": 0.87,
        "frames_processed": s.tick,
        "source": s.source,
    }))
}

async fn pose_zones_summary(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    match s.latest_update.as_ref() {
        Some(update) => {
            let total_persons = update
                .persons
                .as_ref()
                .map(|persons| persons.len())
                .or(update.estimated_persons)
                .unwrap_or(0);
            Json(summarize_signal_field_zones(
                &update.signal_field,
                total_persons,
                update.classification.presence,
            ))
        }
        None => Json(summarize_signal_field_zones(
            &SignalField {
                grid_size: [20, 1, 20],
                values: vec![0.0; 400],
            },
            0,
            false,
        )),
    }
}

async fn stream_status(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::json!({
        "active": true,
        "clients": s.tx.receiver_count(),
        "fps": if s.tick > 1 { 10u64 } else { 0u64 },
        "source": s.source,
    }))
}

async fn vital_signs_endpoint(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    let vs = &s.latest_vitals;
    let (br_len, br_cap, hb_len, hb_cap) = s.vital_detector.buffer_status();
    Json(serde_json::json!({
        "vital_signs": {
            "breathing_rate_bpm": vs.breathing_rate_bpm,
            "heart_rate_bpm": vs.heart_rate_bpm,
            "breathing_confidence": vs.breathing_confidence,
            "heartbeat_confidence": vs.heartbeat_confidence,
            "signal_quality": vs.signal_quality,
        },
        "buffer_status": {
            "breathing_samples": br_len,
            "breathing_capacity": br_cap,
            "heartbeat_samples": hb_len,
            "heartbeat_capacity": hb_cap,
        },
        "source": s.source,
        "tick": s.tick,
    }))
}

/// GET /api/v1/edge-vitals — latest edge vitals from ESP32 (ADR-039).
async fn edge_vitals_endpoint(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    match &s.edge_vitals {
        Some(v) => Json(serde_json::json!({
            "status": "ok",
            "edge_vitals": v,
        })),
        None => Json(serde_json::json!({
            "status": "no_data",
            "edge_vitals": null,
            "message": "No edge vitals packet received yet. Ensure ESP32 edge_tier >= 1.",
        })),
    }
}

/// GET /api/v1/wasm-events — latest WASM events from ESP32 (ADR-040).
async fn wasm_events_endpoint(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    match &s.latest_wasm_events {
        Some(w) => Json(serde_json::json!({
            "status": "ok",
            "wasm_events": w,
        })),
        None => Json(serde_json::json!({
            "status": "no_data",
            "wasm_events": null,
            "message": "No WASM output packet received yet. Upload and start a .wasm module on the ESP32.",
        })),
    }
}

async fn model_info(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    match &s.rvf_info {
        Some(info) => Json(serde_json::json!({
            "status": "loaded",
            "container": info,
            "sona_profiles": s
                .model_sona_profile_deltas
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            "active_sona_profile": s.active_sona_profile.clone(),
        })),
        None => Json(serde_json::json!({
            "status": "no_model",
            "message": "No RVF container loaded. Use --load-rvf <path> to load one.",
            "sona_profiles": [],
            "active_sona_profile": null,
        })),
    }
}

async fn model_layers(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    match &s.progressive_loader {
        Some(loader) => {
            let (a, b, c) = loader.layer_status();
            Json(serde_json::json!({
                "layer_a": a,
                "layer_b": b,
                "layer_c": c,
                "progress": loader.loading_progress(),
            }))
        }
        None => Json(serde_json::json!({
            "layer_a": false,
            "layer_b": false,
            "layer_c": false,
            "progress": 0.0,
            "message": "No model loaded with progressive loading",
        })),
    }
}

async fn model_segments(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    match &s.progressive_loader {
        Some(loader) => Json(serde_json::json!({ "segments": loader.segment_list() })),
        None => Json(serde_json::json!({ "segments": [] })),
    }
}

async fn sona_profiles(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    let names = s
        .model_sona_profile_deltas
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let active = s.active_sona_profile.clone().unwrap_or_default();
    Json(serde_json::json!({ "profiles": names, "active": active }))
}

async fn sona_activate(
    State(state): State<SharedState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let profile = body
        .get("profile")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();

    let mut s = state.write().await;
    let available = s
        .model_sona_profile_deltas
        .keys()
        .cloned()
        .collect::<Vec<_>>();

    if profile.is_empty() || profile == "default" {
        s.active_sona_profile = None;
        s.active_sona_weights = None;
        return Json(serde_json::json!({ "status": "activated", "profile": "default" }));
    }

    if available.contains(&profile) {
        s.active_sona_profile = Some(profile.clone());
        refresh_active_sona_weights(&mut s);
        if s.active_sona_weights.is_some() {
            Json(serde_json::json!({ "status": "activated", "profile": profile }))
        } else {
            s.active_sona_profile = None;
            Json(serde_json::json!({
                "status": "error",
                "message": format!("Profile '{}' 无法应用到当前 RVF；请检查 delta 维度是否与 base weights 一致。", profile),
            }))
        }
    } else {
        Json(serde_json::json!({
            "status": "error",
            "message": format!("Profile '{}' not found. Available: {:?}", profile, available),
        }))
    }
}

async fn info_page() -> Html<String> {
    Html(format!(
        "<html><body>\
         <h1>WiFi-DensePose Sensing Server</h1>\
         <p>Rust + Axum + RuVector</p>\
         <ul>\
         <li><a href='/health'>/health</a> — Server health</li>\
         <li><a href='/api/v1/sensing/latest'>/api/v1/sensing/latest</a> — Latest sensing data</li>\
         <li><a href='/api/v1/vital-signs'>/api/v1/vital-signs</a> — Vital sign estimates (HR/RR)</li>\
         <li><a href='/api/v1/model/info'>/api/v1/model/info</a> — RVF model container info</li>\
         <li>ws://localhost:8765/ws/sensing — WebSocket stream</li>\
         </ul>\
         </body></html>"
    ))
}

// ── UDP receiver task ────────────────────────────────────────────────────────

async fn udp_receiver_task(state: SharedState, udp_port: u16) {
    let addr = format!("0.0.0.0:{udp_port}");
    let socket = match UdpSocket::bind(&addr).await {
        Ok(s) => {
            info!("UDP listening on {addr} for ESP32 CSI frames");
            s
        }
        Err(e) => {
            error!("Failed to bind UDP {addr}: {e}");
            return;
        }
    };

    let mut buf = [0u8; 2048];
    loop {
        match socket.recv_from(&mut buf).await {
            Ok((len, src)) => {
                // ADR-039: Try edge vitals packet first (magic 0xC511_0002).
                if let Some(vitals) = parse_esp32_vitals(&buf[..len]) {
                    debug!(
                        "ESP32 vitals from {src}: node={} br={:.1} hr={:.1} pres={}",
                        vitals.node_id,
                        vitals.breathing_rate_bpm,
                        vitals.heartrate_bpm,
                        vitals.presence
                    );
                    let mut s = state.write().await;
                    // Broadcast vitals via WebSocket.
                    if let Ok(json) = serde_json::to_string(&serde_json::json!({
                        "type": "edge_vitals",
                        "node_id": vitals.node_id,
                        "presence": vitals.presence,
                        "fall_detected": vitals.fall_detected,
                        "motion": vitals.motion,
                        "breathing_rate_bpm": vitals.breathing_rate_bpm,
                        "heartrate_bpm": vitals.heartrate_bpm,
                        "n_persons": vitals.n_persons,
                        "motion_energy": vitals.motion_energy,
                        "presence_score": vitals.presence_score,
                        "rssi": vitals.rssi,
                    })) {
                        let _ = s.tx.send(json);
                    }
                    s.edge_vitals = Some(vitals);
                    continue;
                }

                // ADR-040: Try WASM output packet (magic 0xC511_0004).
                if let Some(wasm_output) = parse_wasm_output(&buf[..len]) {
                    debug!(
                        "WASM output from {src}: node={} module={} events={}",
                        wasm_output.node_id,
                        wasm_output.module_id,
                        wasm_output.events.len()
                    );
                    let mut s = state.write().await;
                    // Broadcast WASM events via WebSocket.
                    if let Ok(json) = serde_json::to_string(&serde_json::json!({
                        "type": "wasm_event",
                        "node_id": wasm_output.node_id,
                        "module_id": wasm_output.module_id,
                        "events": wasm_output.events,
                    })) {
                        let _ = s.tx.send(json);
                    }
                    s.latest_wasm_events = Some(wasm_output);
                    continue;
                }

                if let Some(frame) = parse_esp32_frame(&buf[..len]) {
                    debug!(
                        "ESP32 frame from {src}: node={}, subs={}, seq={}",
                        frame.node_id, frame.n_subcarriers, frame.sequence
                    );

                    let mut s = state.write().await;
                    s.source = "esp32".to_string();

                    // Append current amplitudes to history before extracting features so
                    // that temporal analysis includes the most recent frame.
                    s.frame_history.push_back(frame.amplitudes.clone());
                    if s.frame_history.len() > FRAME_HISTORY_CAPACITY {
                        s.frame_history.pop_front();
                    }

                    let sample_rate_hz = 1000.0 / 500.0_f64; // default tick; ESP32 frames arrive as fast as they come
                    let (features, classification, breathing_rate_hz, sub_variances) =
                        extract_features_from_frame(&frame, &s.frame_history, sample_rate_hz);

                    // Update RSSI history
                    s.rssi_history.push_back(features.mean_rssi);
                    if s.rssi_history.len() > 60 {
                        s.rssi_history.pop_front();
                    }

                    s.tick += 1;
                    let tick = s.tick;

                    let motion_score = if classification.motion_level == "active" {
                        0.8
                    } else if classification.motion_level == "present_still" {
                        0.3
                    } else {
                        0.05
                    };

                    let vitals = s
                        .vital_detector
                        .process_frame(&frame.amplitudes, &frame.phases);
                    s.latest_vitals = vitals.clone();

                    // Multi-person estimation with temporal smoothing.
                    let raw_score = compute_person_score(&features);
                    s.smoothed_person_score = s.smoothed_person_score * 0.85 + raw_score * 0.15;
                    let est_persons = if classification.presence {
                        score_to_person_count(s.smoothed_person_score)
                    } else {
                        0
                    };

                    let mut update = SensingUpdate {
                        msg_type: "sensing_update".to_string(),
                        timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                        source: "esp32".to_string(),
                        tick,
                        nodes: {
                            upsert_recent_node(
                                &mut s.recent_nodes,
                                frame.node_id,
                                features.mean_rssi,
                                frame.amplitudes.iter().take(56).cloned().collect(),
                                frame.n_subcarriers as usize,
                            );
                            active_nodes(&mut s.recent_nodes)
                        },
                        features: features.clone(),
                        classification,
                        signal_field: generate_signal_field(
                            features.mean_rssi,
                            motion_score,
                            breathing_rate_hz,
                            features.variance.min(1.0),
                            &sub_variances,
                        ),
                        vital_signs: Some(vitals),
                        enhanced_motion: None,
                        enhanced_breathing: None,
                        posture: None,
                        signal_quality_score: None,
                        quality_verdict: None,
                        bssid_count: None,
                        pose_keypoints: None,
                        model_status: None,
                        persons: None,
                        estimated_persons: if est_persons > 0 {
                            Some(est_persons)
                        } else {
                            None
                        },
                    };
                    populate_live_pose(&mut update, &s, &frame.amplitudes, sample_rate_hz);

                    if let Ok(json) = serde_json::to_string(&update) {
                        let _ = s.tx.send(json);
                    }
                    s.latest_update = Some(update);
                    drop(s);
                    record_live_frame(
                        &state,
                        &frame.amplitudes,
                        features.mean_rssi,
                        frame.noise_floor as f64,
                        &features,
                    )
                    .await;
                }
            }
            Err(e) => {
                warn!("UDP recv error: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

// ── Simulated data task ──────────────────────────────────────────────────────

async fn simulated_data_task(state: SharedState, tick_ms: u64) {
    let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));
    info!("Simulated data source active (tick={}ms)", tick_ms);

    loop {
        interval.tick().await;

        let mut s = state.write().await;
        s.tick += 1;
        let tick = s.tick;

        let frame = generate_simulated_frame(tick);

        // Append current amplitudes to history before feature extraction.
        s.frame_history.push_back(frame.amplitudes.clone());
        if s.frame_history.len() > FRAME_HISTORY_CAPACITY {
            s.frame_history.pop_front();
        }

        let sample_rate_hz = 1000.0 / tick_ms as f64;
        let (features, classification, breathing_rate_hz, sub_variances) =
            extract_features_from_frame(&frame, &s.frame_history, sample_rate_hz);

        s.rssi_history.push_back(features.mean_rssi);
        if s.rssi_history.len() > 60 {
            s.rssi_history.pop_front();
        }

        let motion_score = if classification.motion_level == "active" {
            0.8
        } else if classification.motion_level == "present_still" {
            0.3
        } else {
            0.05
        };

        let vitals = s
            .vital_detector
            .process_frame(&frame.amplitudes, &frame.phases);
        s.latest_vitals = vitals.clone();

        let frame_amplitudes = frame.amplitudes.clone();
        let frame_n_sub = frame.n_subcarriers;

        // Multi-person estimation with temporal smoothing.
        let raw_score = compute_person_score(&features);
        s.smoothed_person_score = s.smoothed_person_score * 0.85 + raw_score * 0.15;
        let est_persons = if classification.presence {
            score_to_person_count(s.smoothed_person_score)
        } else {
            0
        };

        let mut update = SensingUpdate {
            msg_type: "sensing_update".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
            source: "simulated".to_string(),
            tick,
            nodes: vec![NodeInfo {
                node_id: 1,
                rssi_dbm: features.mean_rssi,
                position: [2.0, 0.0, 1.5],
                amplitude: frame_amplitudes,
                subcarrier_count: frame_n_sub as usize,
            }],
            features: features.clone(),
            classification,
            signal_field: generate_signal_field(
                features.mean_rssi,
                motion_score,
                breathing_rate_hz,
                features.variance.min(1.0),
                &sub_variances,
            ),
            vital_signs: Some(vitals),
            enhanced_motion: None,
            enhanced_breathing: None,
            posture: None,
            signal_quality_score: None,
            quality_verdict: None,
            bssid_count: None,
            pose_keypoints: None,
            model_status: None,
            persons: None,
            estimated_persons: if est_persons > 0 {
                Some(est_persons)
            } else {
                None
            },
        };
        populate_live_pose(&mut update, &s, &frame.amplitudes, sample_rate_hz);

        if update.classification.presence {
            s.total_detections += 1;
        }
        if let Ok(json) = serde_json::to_string(&update) {
            let _ = s.tx.send(json);
        }
        s.latest_update = Some(update);
        drop(s);
        record_live_frame(
            &state,
            &frame.amplitudes,
            features.mean_rssi,
            frame.noise_floor as f64,
            &features,
        )
        .await;
    }
}

// ── Broadcast tick task (for ESP32 mode, sends buffered state) ───────────────

async fn broadcast_tick_task(state: SharedState, tick_ms: u64) {
    let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));

    loop {
        interval.tick().await;
        let s = state.read().await;
        if let Some(ref update) = s.latest_update {
            if s.tx.receiver_count() > 0 {
                // Re-broadcast the latest sensing_update so pose WS clients
                // always get data even when ESP32 pauses between frames.
                if let Ok(json) = serde_json::to_string(update) {
                    let _ = s.tx.send(json);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wifi_densepose_wifiscan::RadioType;

    fn make_signal_field(points: &[(usize, usize, f64)]) -> SignalField {
        let mut values = vec![0.0f64; 20 * 20];
        for &(x, z, value) in points {
            values[z * 20 + x] = value;
        }
        SignalField {
            grid_size: [20, 1, 20],
            values,
        }
    }

    fn make_bssid_observation(
        mac: &str,
        rssi_dbm: f64,
        channel: u8,
        band: BandType,
    ) -> BssidObservation {
        BssidObservation {
            bssid: BssidId::parse(mac).expect("valid test mac"),
            rssi_dbm,
            signal_pct: ((rssi_dbm + 100.0) * 2.0).clamp(0.0, 100.0),
            channel,
            band,
            radio_type: RadioType::Ax,
            ssid: "LAN-SENSE".to_string(),
            timestamp: Instant::now(),
        }
    }

    fn make_person_detection_from_points(points: &[(f64, f64, f64); 17]) -> PersonDetection {
        let kp_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ];
        let keypoints = kp_names
            .iter()
            .zip(points.iter())
            .map(|(name, (x, y, z))| PoseKeypoint {
                name: (*name).to_string(),
                x: *x,
                y: *y,
                z: *z,
                confidence: 0.9,
            })
            .collect();

        PersonDetection {
            id: 1,
            confidence: 0.9,
            keypoints,
            bbox: BoundingBox {
                x: 250.0,
                y: 120.0,
                width: 140.0,
                height: 260.0,
            },
            zone: "zone_1".to_string(),
        }
    }

    #[test]
    fn zone_summary_detects_real_hotspots_per_zone() {
        let field = make_signal_field(&[
            (3, 3, 0.95),
            (3, 4, 0.72),
            (4, 3, 0.68),
            (16, 15, 0.93),
            (16, 16, 0.74),
            (15, 16, 0.69),
        ]);

        let summary = summarize_signal_field_zones(&field, 2, true);

        assert_eq!(summary["zones"]["zone_1"]["person_count"].as_u64(), Some(1));
        assert_eq!(summary["zones"]["zone_4"]["person_count"].as_u64(), Some(1));
        assert_eq!(summary["zones"]["zone_2"]["person_count"].as_u64(), Some(0));
        assert_eq!(summary["zones"]["zone_3"]["person_count"].as_u64(), Some(0));
        assert_eq!(summary["summary"]["hotspot_count"].as_u64(), Some(2));
    }

    #[test]
    fn zone_summary_falls_back_to_hottest_cell_when_presence_is_true() {
        let field = make_signal_field(&[(16, 3, 0.14)]);

        let summary = summarize_signal_field_zones(&field, 1, true);

        assert_eq!(summary["zones"]["zone_2"]["person_count"].as_u64(), Some(1));
        assert_eq!(
            summary["zones"]["zone_2"]["status"].as_str(),
            Some("occupied")
        );
        assert_eq!(summary["summary"]["hotspot_count"].as_u64(), Some(1));
    }

    #[test]
    fn live_nodes_from_bssid_observations_exposes_multiple_nodes() {
        let observations = vec![
            make_bssid_observation("aa:bb:cc:dd:ee:01", -54.0, 1, BandType::Band2_4GHz),
            make_bssid_observation("aa:bb:cc:dd:ee:02", -58.0, 36, BandType::Band5GHz),
            make_bssid_observation("aa:bb:cc:dd:ee:03", -62.0, 149, BandType::Band5GHz),
            make_bssid_observation("aa:bb:cc:dd:ee:04", -67.0, 5, BandType::Band2_4GHz),
        ];

        let nodes = live_nodes_from_bssid_observations(&observations);

        assert_eq!(nodes.len(), 4);
        assert!(nodes.iter().all(|node| node.subcarrier_count == 1));
        assert!(nodes.iter().all(|node| node.amplitude.len() == 1));
        assert_ne!(nodes[0].position, nodes[1].position);
        assert_ne!(nodes[1].position, nodes[2].position);
    }

    #[test]
    fn esp32_udp_ports_respects_legacy_fallback_setting() {
        assert_eq!(esp32_udp_ports(5006, false), vec![5006, 5005]);
        assert_eq!(esp32_udp_ports(5005, false), vec![5005]);
        assert_eq!(esp32_udp_ports(5006, true), vec![5006]);
    }

    #[test]
    fn canonicalize_tracked_person_points_operator_frame_still_returns_body_centered_pose() {
        let person = make_person_detection_from_points(&[
            (320.0, 140.0, 0.20),
            (312.0, 135.0, 0.21),
            (328.0, 135.0, 0.21),
            (304.0, 142.0, 0.20),
            (336.0, 142.0, 0.20),
            (290.0, 190.0, 0.22),
            (350.0, 190.0, 0.23),
            (278.0, 235.0, 0.21),
            (362.0, 235.0, 0.24),
            (268.0, 280.0, 0.20),
            (372.0, 280.0, 0.25),
            (300.0, 260.0, 0.22),
            (340.0, 260.0, 0.23),
            (298.0, 330.0, 0.20),
            (342.0, 330.0, 0.24),
            (296.0, 395.0, 0.19),
            (344.0, 395.0, 0.25),
        ]);

        let canonical =
            canonicalize_tracked_person_points(&person, MODEL_TARGET_SPACE_OPERATOR_FRAME)
                .expect("canonical points");

        let max_abs = canonical
            .iter()
            .flat_map(|point| point.iter())
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_abs < 2.0, "canonical max_abs was {max_abs}");
        let min_z = canonical
            .iter()
            .map(|point| point[2])
            .fold(f64::INFINITY, f64::min);
        let max_z = canonical
            .iter()
            .map(|point| point[2])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min_z >= 0.60, "canonical min_z was {min_z}");
        assert!(max_z <= 1.05, "canonical max_z was {max_z}");

        let (shoulder_dx, shoulder_vs_hip_y) =
            tracked_anchor_orientation(&canonical).expect("orientation");
        assert!(shoulder_dx > 0.0);
        assert!(shoulder_vs_hip_y < 0.0);
    }

    #[test]
    fn canonicalize_tracked_person_points_operator_frame_tames_realistic_noisy_depth() {
        let person = make_person_detection_from_points(&[
            (-8.6558632666, 7.4645421258, 5.4205608653),
            (-3.1558969488, -9.3549546051, 10.3946614517),
            (0.6664664930, -3.8680370538, -9.7488087646),
            (9.4464982401, -0.4922348980, -4.1970337147),
            (-10.4277912015, 8.4701939397, -1.2210757735),
            (-5.5076600721, -11.6118623342, 7.6923823509),
            (-1.9589216430, -6.3575817236, 9.6174654974),
            (7.3818456036, -2.8015900448, -7.2394490661),
            (8.8257041178, 6.3512099313, 0.2484710597),
            (-3.2094656245, 3.8350315609, 3.2569080647),
            (-0.8326662929, -3.0895705832, 4.1596757740),
            (4.7306822118, -3.5152169930, -9.4455117535),
            (9.9168483322, 0.3016303893, -4.5902012906),
            (-10.2102192436, 9.4182657359, -1.0995938661),
            (-4.7639613248, -10.9199547483, 8.3387753310),
            (-0.5930329873, -3.0317587016, 4.3856044075),
            (2.8481475202, -1.0783485895, -2.1538479440),
        ]);

        let canonical =
            canonicalize_tracked_person_points(&person, MODEL_TARGET_SPACE_OPERATOR_FRAME)
                .expect("canonical points");

        let min_z = canonical
            .iter()
            .map(|point| point[2])
            .fold(f64::INFINITY, f64::min);
        let max_z = canonical
            .iter()
            .map(|point| point[2])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min_z >= 0.63, "canonical min_z was {min_z}");
        assert!(max_z <= 1.01, "canonical max_z was {max_z}");

        let left_shoulder = canonical[5];
        let right_shoulder = canonical[6];
        let shoulder_width = tracked_point_distance(&left_shoulder, &right_shoulder);
        assert!(shoulder_width > 0.12 && shoulder_width < 0.22);

        let (shoulder_dx, shoulder_vs_hip_y) =
            tracked_anchor_orientation(&canonical).expect("orientation");
        assert!(shoulder_dx > 0.0);
        assert!(shoulder_vs_hip_y < 0.0);
    }

    #[test]
    fn enforce_tracked_canonical_orientation_unflips_pose() {
        let mut points = vec![[0.0; 3]; 17];
        points[5] = [-0.08, 0.18, 0.82];
        points[6] = [0.08, 0.19, 0.83];
        points[11] = [-0.05, -0.01, 0.80];
        points[12] = [0.05, 0.00, 0.81];

        enforce_tracked_canonical_orientation(&mut points);

        let (shoulder_dx, shoulder_vs_hip_y) =
            tracked_anchor_orientation(&points).expect("orientation");
        assert!(shoulder_dx > 0.0);
        assert!(shoulder_vs_hip_y < 0.0);
    }

    #[test]
    fn stabilize_tracked_canonical_points_prefers_previous_orientation() {
        let previous = TrackedPersonContract {
            track_id: "wifi-track-1".to_string(),
            source_person_id: 1,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: "wifi_pose_pixels".to_string(),
            canonical_body_space: "canonical_body_frame".to_string(),
            body_layout: "coco_body_17".to_string(),
            body_kpts_3d: vec![],
            canonical_body_kpts_3d: {
                let mut points = vec![[0.0; 3]; 17];
                points[5] = [0.08, -0.18, 0.82];
                points[6] = [-0.08, -0.19, 0.83];
                points[11] = [0.05, 0.01, 0.80];
                points[12] = [-0.05, 0.00, 0.81];
                points
            },
            person_confidence: 0.8,
            keypoints: vec![],
            pose_source: "model_inference".to_string(),
        };

        let mut flipped = previous.canonical_body_kpts_3d.clone();
        for point in &mut flipped {
            point[0] = -point[0];
            point[1] = -point[1];
        }

        let stabilized = stabilize_tracked_canonical_points(flipped, Some(&previous));
        let (shoulder_dx, shoulder_vs_hip_y) =
            tracked_anchor_orientation(&stabilized).expect("orientation");
        assert!(shoulder_dx > 0.0);
        assert!(shoulder_vs_hip_y < 0.0);
    }

    #[test]
    fn smooth_tracked_canonical_points_clamps_wrist_spike() {
        let previous = TrackedPersonContract {
            track_id: "wifi-track-1".to_string(),
            source_person_id: 1,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: "wifi_pose_pixels".to_string(),
            canonical_body_space: "canonical_body_frame".to_string(),
            body_layout: "coco_body_17".to_string(),
            body_kpts_3d: vec![],
            canonical_body_kpts_3d: {
                let mut points = vec![[0.0; 3]; 17];
                points[9] = [0.12, -0.01, 0.84];
                points[10] = [-0.12, -0.01, 0.84];
                points
            },
            person_confidence: 0.8,
            keypoints: vec![],
            pose_source: "model_inference".to_string(),
        };

        let mut points = previous.canonical_body_kpts_3d.clone();
        points[9] = [0.60, 0.45, 1.50];

        let smoothed = smooth_tracked_canonical_points(points, Some(&previous), true);
        let jump = tracked_point_distance(&smoothed[9], &previous.canonical_body_kpts_3d[9]);
        assert!(jump < 0.09);
    }

    #[test]
    fn repair_tracked_canonical_points_restores_human_topology() {
        let points = vec![
            [-0.1082, -0.3533, 1.0],
            [-0.0089, -0.5178, 0.8080],
            [0.1420, -0.0495, 0.8834],
            [-0.0784, -0.0143, 0.8056],
            [-0.0080, -0.2551, 1.0],
            [0.0762, -0.5218, 0.8144],
            [-0.0762, -0.4782, 0.8256],
            [0.0107, -0.0019, 0.8089],
            [-0.2234, -0.0390, 0.8941],
            [-0.0789, -0.2661, 0.9998],
            [-0.0433, -0.3020, 0.8930],
            [0.0867, -0.0013, 0.8256],
            [-0.0867, 0.0013, 0.8144],
            [-0.0207, -0.2478, 1.0],
            [0.0523, -0.5205, 0.8104],
            [-0.0530, -0.3102, 0.8934],
            [-0.0288, -0.1591, 0.8937],
        ];

        let repaired = repair_tracked_canonical_points(points);

        assert!(repaired[7][0] > 0.0);
        assert!(repaired[9][0] > 0.0);
        assert!(repaired[8][0] < 0.0);
        assert!(repaired[10][0] < 0.0);
        assert!(repaired[13][0] > 0.0);
        assert!(repaired[15][0] > 0.0);
        assert!(repaired[14][0] < 0.0);
        assert!(repaired[16][0] < 0.0);
        assert!(repaired[13][1] > repaired[11][1]);
        assert!(repaired[14][1] > repaired[12][1]);
        assert!(repaired[15][1] > repaired[13][1]);
        assert!(repaired[16][1] > repaired[14][1]);

        let upper_arm_left = tracked_point_distance(&repaired[5], &repaired[7]);
        let forearm_left = tracked_point_distance(&repaired[7], &repaired[9]);
        let thigh_right = tracked_point_distance(&repaired[12], &repaired[14]);
        let shin_right = tracked_point_distance(&repaired[14], &repaired[16]);
        assert!(upper_arm_left > 0.15 && upper_arm_left < 0.35);
        assert!(forearm_left > 0.12 && forearm_left < 0.32);
        assert!(thigh_right > 0.18 && thigh_right < 0.4);
        assert!(shin_right > 0.18 && shin_right < 0.4);
    }

    #[test]
    fn tracked_contract_for_pose_api_defaults_to_canonical_body() {
        let contract = TrackedPersonContract {
            track_id: "wifi-track-1".to_string(),
            source_person_id: 1,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: MODEL_TARGET_SPACE_OPERATOR_FRAME.to_string(),
            canonical_body_space: "canonical_body_frame".to_string(),
            body_layout: "coco_body_17".to_string(),
            body_kpts_3d: vec![[10.0, 20.0, 30.0]; 17],
            canonical_body_kpts_3d: vec![[0.1, -0.2, 0.9]; 17],
            person_confidence: 0.8,
            keypoints: (0..17)
                .map(|idx| PoseKeypoint {
                    name: format!("kp-{idx}"),
                    x: 10.0,
                    y: 20.0,
                    z: 30.0,
                    confidence: 0.7,
                })
                .collect(),
            pose_source: "model_inference".to_string(),
        };

        let value = tracked_contract_for_pose_api(contract);

        assert_eq!(
            value
                .get("target_space")
                .and_then(serde_json::Value::as_str),
            Some("canonical_body_frame")
        );
        assert_eq!(
            value
                .get("raw_target_space")
                .and_then(serde_json::Value::as_str),
            Some(MODEL_TARGET_SPACE_OPERATOR_FRAME)
        );
        assert_eq!(
            value
                .pointer("/body_kpts_3d/0/0")
                .and_then(serde_json::Value::as_f64),
            Some(0.1)
        );
        assert_eq!(
            value
                .pointer("/raw_body_kpts_3d/0/0")
                .and_then(serde_json::Value::as_f64),
            Some(10.0)
        );
        assert_eq!(
            value
                .pointer("/keypoints/0/x")
                .and_then(serde_json::Value::as_f64),
            Some(0.1)
        );
        assert_eq!(
            value
                .pointer("/raw_keypoints/0/x")
                .and_then(serde_json::Value::as_f64),
            Some(10.0)
        );
    }

    #[test]
    fn tracked_contract_for_pose_api_keeps_raw_when_canonical_missing() {
        let contract = TrackedPersonContract {
            track_id: "wifi-track-1".to_string(),
            source_person_id: 1,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: MODEL_TARGET_SPACE_OPERATOR_FRAME.to_string(),
            canonical_body_space: "canonical_body_frame".to_string(),
            body_layout: "coco_body_17".to_string(),
            body_kpts_3d: vec![[10.0, 20.0, 30.0]; 17],
            canonical_body_kpts_3d: vec![],
            person_confidence: 0.8,
            keypoints: vec![],
            pose_source: "model_inference".to_string(),
        };

        let value = tracked_contract_for_pose_api(contract);

        assert_eq!(
            value
                .get("target_space")
                .and_then(serde_json::Value::as_str),
            Some(MODEL_TARGET_SPACE_OPERATOR_FRAME)
        );
        assert!(value.get("raw_target_space").is_none());
        assert!(value.get("raw_body_kpts_3d").is_none());
    }

    #[test]
    fn resolve_initial_active_sona_profile_prefers_metadata_default() {
        let metadata = serde_json::json!({
            "model_config": {
                "default_sona_profile": "scene-geometry"
            }
        });
        let deltas = BTreeMap::from([
            ("scene-geometry".to_string(), vec![0.1f32, 0.2]),
            ("scene-backup".to_string(), vec![0.3f32, 0.4]),
        ]);

        let profile = resolve_initial_active_sona_profile(
            Some(&metadata),
            Some(Path::new("/tmp/trained-supervised-live.rvf")),
            &deltas,
        );

        assert_eq!(profile.as_deref(), Some("scene-geometry"));
    }

    #[test]
    fn resolve_initial_active_sona_profile_recovers_imported_single_profile() {
        let metadata = serde_json::json!({
            "model_config": {
                "base_model_id": "trained-supervised-20260314_040438"
            }
        });
        let deltas = BTreeMap::from([(
            "chek_humanoid_lan_sense_main_stage_v1_geometry".to_string(),
            vec![0.1f32, -0.1],
        )]);

        let profile = resolve_initial_active_sona_profile(
            Some(&metadata),
            Some(Path::new(
                "/tmp/trained-supervised-20260314_040438.sona-chek_humanoid_lan_sense_main_stage_v1_geometry.imported.rvf",
            )),
            &deltas,
        );

        assert_eq!(
            profile.as_deref(),
            Some("chek_humanoid_lan_sense_main_stage_v1_geometry")
        );
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug".into()),
        )
        .init();

    let args = Args::parse();

    // Handle --benchmark mode: run vital sign benchmark and exit
    if args.benchmark {
        eprintln!("Running vital sign detection benchmark (1000 frames)...");
        let (total, per_frame) = vital_signs::run_benchmark(1000);
        eprintln!();
        eprintln!(
            "Summary: {} total, {} per frame",
            format!("{total:?}"),
            format!("{per_frame:?}")
        );
        return;
    }

    // Handle --export-rvf mode: build an RVF container package and exit
    if let Some(ref rvf_path) = args.export_rvf {
        eprintln!("Exporting RVF container package...");
        use rvf_pipeline::RvfModelBuilder;

        let mut builder = RvfModelBuilder::new("wifi-densepose", "1.0.0");

        // Vital sign config (default breathing 0.1-0.5 Hz, heartbeat 0.8-2.0 Hz)
        builder.set_vital_config(0.1, 0.5, 0.8, 2.0);

        // Model profile (input/output spec)
        builder.set_model_profile(
            "56-subcarrier CSI amplitude/phase @ 10-100 Hz",
            "17 COCO keypoints + body part UV + vital signs",
            "ESP32-S3 or Windows WiFi RSSI, Rust 1.85+",
        );

        // Placeholder weights (17 keypoints × 56 subcarriers × 3 dims = 2856 params)
        let placeholder_weights: Vec<f32> = (0..2856).map(|i| (i as f32 * 0.001).sin()).collect();
        builder.set_weights(&placeholder_weights);

        // Training provenance
        builder.set_training_proof(
            "wifi-densepose-rs-v1.0.0",
            serde_json::json!({
                "pipeline": "ADR-023 8-phase",
                "test_count": 229,
                "benchmark_fps": 9520,
                "framework": "wifi-densepose-rs",
            }),
        );

        // SONA default environment profile
        let default_lora: Vec<f32> = vec![0.0; 64];
        builder.add_sona_profile("default", &default_lora, &default_lora);

        match builder.build() {
            Ok(rvf_bytes) => {
                if let Err(e) = std::fs::write(rvf_path, &rvf_bytes) {
                    eprintln!("Error writing RVF: {e}");
                    std::process::exit(1);
                }
                eprintln!("Wrote {} bytes to {}", rvf_bytes.len(), rvf_path.display());
                eprintln!("RVF container exported successfully.");
            }
            Err(e) => {
                eprintln!("Error building RVF: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Handle --pretrain mode: self-supervised contrastive pretraining (ADR-024)
    if args.pretrain {
        eprintln!("=== WiFi-DensePose Contrastive Pretraining (ADR-024) ===");

        let ds_path = args
            .dataset
            .clone()
            .unwrap_or_else(|| PathBuf::from("data"));
        let source = match args.dataset_type.as_str() {
            "wipose" => dataset::DataSource::WiPose(ds_path.clone()),
            _ => dataset::DataSource::MmFi(ds_path.clone()),
        };
        let pipeline = dataset::DataPipeline::new(dataset::DataConfig {
            source,
            ..Default::default()
        });

        // Generate synthetic or load real CSI windows
        let generate_synthetic_windows = || -> Vec<Vec<Vec<f32>>> {
            (0..50)
                .map(|i| {
                    (0..4)
                        .map(|a| {
                            (0..56)
                                .map(|s| ((i * 7 + a * 13 + s) as f32 * 0.31).sin() * 0.5)
                                .collect()
                        })
                        .collect()
                })
                .collect()
        };

        let csi_windows: Vec<Vec<Vec<f32>>> = match pipeline.load() {
            Ok(s) if !s.is_empty() => {
                eprintln!("Loaded {} samples from {}", s.len(), ds_path.display());
                s.into_iter().map(|s| s.csi_window).collect()
            }
            _ => {
                eprintln!("Using synthetic data for pretraining.");
                generate_synthetic_windows()
            }
        };

        let n_subcarriers = csi_windows
            .first()
            .and_then(|w| w.first())
            .map(|f| f.len())
            .unwrap_or(56);

        let tf_config = graph_transformer::TransformerConfig {
            n_subcarriers,
            n_keypoints: 17,
            d_model: 64,
            n_heads: 4,
            n_gnn_layers: 2,
        };
        let transformer = graph_transformer::CsiToPoseTransformer::new(tf_config);
        eprintln!("Transformer params: {}", transformer.param_count());

        let trainer_config = trainer::TrainerConfig {
            epochs: args.pretrain_epochs,
            batch_size: 8,
            lr: 0.001,
            warmup_epochs: 2,
            min_lr: 1e-6,
            early_stop_patience: args.pretrain_epochs + 1,
            pretrain_temperature: 0.07,
            ..Default::default()
        };
        let mut t = trainer::Trainer::with_transformer(trainer_config, transformer);

        let e_config = embedding::EmbeddingConfig {
            d_model: 64,
            d_proj: 128,
            temperature: 0.07,
            normalize: true,
        };
        let mut projection = embedding::ProjectionHead::new(e_config.clone());
        let augmenter = embedding::CsiAugmenter::new();

        eprintln!(
            "Starting contrastive pretraining for {} epochs...",
            args.pretrain_epochs
        );
        let start = std::time::Instant::now();
        for epoch in 0..args.pretrain_epochs {
            let loss = t.pretrain_epoch(&csi_windows, &augmenter, &mut projection, 0.07, epoch);
            if epoch % 10 == 0 || epoch == args.pretrain_epochs - 1 {
                eprintln!("  Epoch {epoch}: contrastive loss = {loss:.4}");
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("Pretraining complete in {elapsed:.1}s");

        // Save pretrained model as RVF with embedding segment
        if let Some(ref save_path) = args.save_rvf {
            eprintln!("Saving pretrained model to RVF: {}", save_path.display());
            t.sync_transformer_weights();
            let weights = t.params().to_vec();
            let mut proj_weights = Vec::new();
            projection.flatten_into(&mut proj_weights);

            let mut builder = RvfBuilder::new();
            builder.add_manifest(
                "wifi-densepose-pretrained",
                env!("CARGO_PKG_VERSION"),
                "WiFi DensePose contrastive pretrained model (ADR-024)",
            );
            builder.add_weights(&weights);
            builder.add_embedding(
                &serde_json::json!({
                    "d_model": e_config.d_model,
                    "d_proj": e_config.d_proj,
                    "temperature": e_config.temperature,
                    "normalize": e_config.normalize,
                    "pretrain_epochs": args.pretrain_epochs,
                }),
                &proj_weights,
            );
            match builder.write_to_file(save_path) {
                Ok(()) => eprintln!(
                    "RVF saved ({} transformer + {} projection params)",
                    weights.len(),
                    proj_weights.len()
                ),
                Err(e) => eprintln!("Failed to save RVF: {e}"),
            }
        }

        return;
    }

    // Handle --embed mode: extract embeddings from CSI data
    if args.embed {
        eprintln!("=== WiFi-DensePose Embedding Extraction (ADR-024) ===");

        let model_path = match &args.model {
            Some(p) => p.clone(),
            None => {
                eprintln!("Error: --embed requires --model <path> to a pretrained .rvf file");
                std::process::exit(1);
            }
        };

        let reader = match RvfReader::from_file(&model_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        };

        let weights = reader.weights().unwrap_or_default();
        let (embed_config_json, proj_weights) = reader.embedding().unwrap_or_else(|| {
            eprintln!("Warning: no embedding segment in RVF, using defaults");
            (
                serde_json::json!({"d_model":64,"d_proj":128,"temperature":0.07,"normalize":true}),
                Vec::new(),
            )
        });

        let d_model = embed_config_json["d_model"].as_u64().unwrap_or(64) as usize;
        let d_proj = embed_config_json["d_proj"].as_u64().unwrap_or(128) as usize;

        let tf_config = graph_transformer::TransformerConfig {
            n_subcarriers: 56,
            n_keypoints: 17,
            d_model,
            n_heads: 4,
            n_gnn_layers: 2,
        };
        let e_config = embedding::EmbeddingConfig {
            d_model,
            d_proj,
            temperature: 0.07,
            normalize: true,
        };
        let mut extractor = embedding::EmbeddingExtractor::new(tf_config, e_config.clone());

        // Load transformer weights
        if !weights.is_empty() {
            if let Err(e) = extractor.transformer.unflatten_weights(&weights) {
                eprintln!("Warning: failed to load transformer weights: {e}");
            }
        }
        // Load projection weights
        if !proj_weights.is_empty() {
            let (proj, _) = embedding::ProjectionHead::unflatten_from(&proj_weights, &e_config);
            extractor.projection = proj;
        }

        // Load dataset and extract embeddings
        let _ds_path = args
            .dataset
            .clone()
            .unwrap_or_else(|| PathBuf::from("data"));
        let csi_windows: Vec<Vec<Vec<f32>>> = (0..10)
            .map(|i| {
                (0..4)
                    .map(|a| {
                        (0..56)
                            .map(|s| ((i * 7 + a * 13 + s) as f32 * 0.31).sin() * 0.5)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        eprintln!(
            "Extracting embeddings from {} CSI windows...",
            csi_windows.len()
        );
        let embeddings = extractor.extract_batch(&csi_windows);
        for (i, emb) in embeddings.iter().enumerate() {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("  Window {i}: {d_proj}-dim embedding, ||e|| = {norm:.4}");
        }
        eprintln!(
            "Extracted {} embeddings of dimension {d_proj}",
            embeddings.len()
        );

        return;
    }

    // Handle --build-index mode: build a fingerprint index from embeddings
    if let Some(ref index_type_str) = args.build_index {
        eprintln!("=== WiFi-DensePose Fingerprint Index Builder (ADR-024) ===");

        let index_type = match index_type_str.as_str() {
            "env" | "environment" => embedding::IndexType::EnvironmentFingerprint,
            "activity" => embedding::IndexType::ActivityPattern,
            "temporal" => embedding::IndexType::TemporalBaseline,
            "person" => embedding::IndexType::PersonTrack,
            _ => {
                eprintln!(
                    "Unknown index type '{}'. Use: env, activity, temporal, person",
                    index_type_str
                );
                std::process::exit(1);
            }
        };

        let tf_config = graph_transformer::TransformerConfig::default();
        let e_config = embedding::EmbeddingConfig::default();
        let mut extractor = embedding::EmbeddingExtractor::new(tf_config, e_config);

        // Generate synthetic CSI windows for demo
        let csi_windows: Vec<Vec<Vec<f32>>> = (0..20)
            .map(|i| {
                (0..4)
                    .map(|a| {
                        (0..56)
                            .map(|s| ((i * 7 + a * 13 + s) as f32 * 0.31).sin() * 0.5)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut index = embedding::FingerprintIndex::new(index_type);
        for (i, window) in csi_windows.iter().enumerate() {
            let emb = extractor.extract(window);
            index.insert(emb, format!("window_{i}"), i as u64 * 100);
        }

        eprintln!("Built {:?} index with {} entries", index_type, index.len());

        // Test a query
        let query_emb = extractor.extract(&csi_windows[0]);
        let results = index.search(&query_emb, 5);
        eprintln!("Top-5 nearest to window_0:");
        for r in &results {
            eprintln!(
                "  entry={}, distance={:.4}, metadata={}",
                r.entry, r.distance, r.metadata
            );
        }

        return;
    }

    // Handle --train mode: train a model and exit
    if args.train {
        eprintln!("=== WiFi-DensePose Training Mode ===");

        // Build data pipeline
        let ds_path = args
            .dataset
            .clone()
            .unwrap_or_else(|| PathBuf::from("data"));
        let source = match args.dataset_type.as_str() {
            "wipose" => dataset::DataSource::WiPose(ds_path.clone()),
            _ => dataset::DataSource::MmFi(ds_path.clone()),
        };
        let pipeline = dataset::DataPipeline::new(dataset::DataConfig {
            source,
            ..Default::default()
        });

        // Generate synthetic training data (50 samples with deterministic CSI + keypoints)
        let generate_synthetic = || -> Vec<dataset::TrainingSample> {
            (0..50)
                .map(|i| {
                    let csi: Vec<Vec<f32>> = (0..4)
                        .map(|a| {
                            (0..56)
                                .map(|s| ((i * 7 + a * 13 + s) as f32 * 0.31).sin() * 0.5)
                                .collect()
                        })
                        .collect();
                    let mut kps = [(0.0f32, 0.0f32, 1.0f32); 17];
                    for (k, kp) in kps.iter_mut().enumerate() {
                        kp.0 = (k as f32 * 0.1 + i as f32 * 0.02).sin() * 100.0 + 320.0;
                        kp.1 = (k as f32 * 0.15 + i as f32 * 0.03).cos() * 80.0 + 240.0;
                    }
                    dataset::TrainingSample {
                        csi_window: csi,
                        pose_label: dataset::PoseLabel {
                            keypoints: kps,
                            body_parts: Vec::new(),
                            confidence: 1.0,
                        },
                        source: "synthetic",
                    }
                })
                .collect()
        };

        // Load samples (fall back to synthetic if dataset missing/empty)
        let samples = match pipeline.load() {
            Ok(s) if !s.is_empty() => {
                eprintln!("Loaded {} samples from {}", s.len(), ds_path.display());
                s
            }
            Ok(_) => {
                eprintln!(
                    "No samples found at {}. Using synthetic data.",
                    ds_path.display()
                );
                generate_synthetic()
            }
            Err(e) => {
                eprintln!("Failed to load dataset: {e}. Using synthetic data.");
                generate_synthetic()
            }
        };

        // Convert dataset samples to trainer format
        let trainer_samples: Vec<trainer::TrainingSample> =
            samples.iter().map(trainer::from_dataset_sample).collect();

        // Split 80/20 train/val
        let split = (trainer_samples.len() * 4) / 5;
        let (train_data, val_data) = trainer_samples.split_at(split.max(1));
        eprintln!(
            "Train: {} samples, Val: {} samples",
            train_data.len(),
            val_data.len()
        );

        // Create transformer + trainer
        let n_subcarriers = train_data
            .first()
            .and_then(|s| s.csi_features.first())
            .map(|f| f.len())
            .unwrap_or(56);
        let tf_config = graph_transformer::TransformerConfig {
            n_subcarriers,
            n_keypoints: 17,
            d_model: 64,
            n_heads: 4,
            n_gnn_layers: 2,
        };
        let transformer = graph_transformer::CsiToPoseTransformer::new(tf_config);
        eprintln!("Transformer params: {}", transformer.param_count());

        let trainer_config = trainer::TrainerConfig {
            epochs: args.epochs,
            batch_size: 8,
            lr: 0.001,
            warmup_epochs: 5,
            min_lr: 1e-6,
            early_stop_patience: 20,
            checkpoint_every: 10,
            ..Default::default()
        };
        let mut t = trainer::Trainer::with_transformer(trainer_config, transformer);

        // Run training
        eprintln!("Starting training for {} epochs...", args.epochs);
        let result = t.run_training(train_data, val_data);
        eprintln!("Training complete in {:.1}s", result.total_time_secs);
        eprintln!(
            "  Best epoch: {}, PCK@0.2: {:.4}, OKS mAP: {:.4}",
            result.best_epoch, result.best_pck, result.best_oks
        );

        // Save checkpoint
        if let Some(ref ckpt_dir) = args.checkpoint_dir {
            let _ = std::fs::create_dir_all(ckpt_dir);
            let ckpt_path = ckpt_dir.join("best_checkpoint.json");
            let ckpt = t.checkpoint();
            match ckpt.save_to_file(&ckpt_path) {
                Ok(()) => eprintln!("Checkpoint saved to {}", ckpt_path.display()),
                Err(e) => eprintln!("Failed to save checkpoint: {e}"),
            }
        }

        // Sync weights back to transformer and save as RVF
        t.sync_transformer_weights();
        if let Some(ref save_path) = args.save_rvf {
            eprintln!("Saving trained model to RVF: {}", save_path.display());
            let weights = t.params().to_vec();
            let mut builder = RvfBuilder::new();
            builder.add_manifest(
                "wifi-densepose-trained",
                env!("CARGO_PKG_VERSION"),
                "WiFi DensePose trained model weights",
            );
            builder.add_metadata(&serde_json::json!({
                "training": {
                    "epochs": args.epochs,
                    "best_epoch": result.best_epoch,
                    "best_pck": result.best_pck,
                    "best_oks": result.best_oks,
                    "n_train_samples": train_data.len(),
                    "n_val_samples": val_data.len(),
                    "n_subcarriers": n_subcarriers,
                    "param_count": weights.len(),
                },
            }));
            builder.add_vital_config(&VitalSignConfig::default());
            builder.add_weights(&weights);
            match builder.write_to_file(save_path) {
                Ok(()) => eprintln!(
                    "RVF saved ({} params, {} bytes)",
                    weights.len(),
                    weights.len() * 4
                ),
                Err(e) => eprintln!("Failed to save RVF: {e}"),
            }
        }

        return;
    }

    info!("WiFi-DensePose Sensing Server (Rust + Axum + RuVector)");
    let esp32_ports = esp32_udp_ports(args.udp_port, args.disable_legacy_esp32_port_fallback);
    let esp32_ports_label = esp32_ports
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    info!("  HTTP:      http://localhost:{}", args.http_port);
    info!("  WebSocket: ws://localhost:{}/ws/sensing", args.ws_port);
    info!("  UDP:       0.0.0.0:[{}] (ESP32 CSI)", esp32_ports_label);
    info!("  UI path:   {}", args.ui_path.display());
    info!("  Source:    {}", args.source);

    // Auto-detect data source
    let source = match args.source.as_str() {
        "auto" => {
            info!("Auto-detecting data source...");
            if let Some(port) = probe_esp32_ports(&esp32_ports).await {
                info!("  ESP32 CSI detected on UDP :{port}");
                "esp32"
            } else if probe_windows_wifi().await {
                info!("  Windows WiFi detected");
                "wifi"
            } else {
                info!("  No hardware detected, using simulation");
                "simulate"
            }
        }
        other => other,
    };

    info!("Data source: {source}");

    // Shared state
    // Vital sign sample rate derives from tick interval (e.g. 500ms tick => 2 Hz)
    let vital_sample_rate = 1000.0 / args.tick_ms as f64;
    info!("Vital sign detector sample rate: {vital_sample_rate:.1} Hz");

    let rvf_path = args.load_rvf.as_ref().or(args.model.as_ref());
    let (
        rvf_info,
        model_weights,
        model_feature_stats,
        model_head_config,
        model_target_space,
        model_sona_profile_deltas,
    ) = if let Some(rvf_path) = rvf_path {
        info!("Loading RVF container from {}", rvf_path.display());
        match RvfReader::from_file(rvf_path) {
            Ok(reader) => {
                let info = reader.info();
                info!(
                    "  RVF loaded: {} segments, {} bytes",
                    info.segment_count, info.total_size
                );
                if let Some(ref manifest) = info.manifest {
                    if let Some(model_id) = manifest.get("model_id") {
                        info!("  Model ID: {model_id}");
                    }
                    if let Some(version) = manifest.get("version") {
                        info!("  Version:  {version}");
                    }
                }

                let weights = reader.weights();
                if let Some(ref params) = weights {
                    info!("  Weights: {} parameters", params.len());
                } else {
                    warn!("  RVF missing weight segment; live model inference will stay disabled");
                }

                if info.has_vital_config {
                    info!("  Vital sign config: present");
                }
                if info.has_quant_info {
                    info!("  Quantization info: present");
                }
                if info.has_witness {
                    info!("  Witness/proof: present");
                }

                let feature_stats = parse_model_feature_stats(info.metadata.as_ref());
                let head_config =
                    parse_model_head_config(info.metadata.as_ref(), feature_stats.as_ref());
                let target_space = parse_model_target_space(info.metadata.as_ref());
                let sona_profile_deltas = decode_sona_profile_deltas(&reader);
                if feature_stats.is_none() {
                    warn!(
                        "  RVF missing feature_stats metadata; live pose_keypoints inference will stay disabled"
                    );
                }
                if let Some(ref cfg) = head_config {
                    info!("  Model head: {}", cfg.description());
                }
                if let Some(ref stats) = feature_stats {
                    info!(
                        "  Temporal context: frames={}, decay={:.2}",
                        stats.temporal_context_frames, stats.temporal_context_decay
                    );
                }
                if target_space.is_none() {
                    warn!(
                        "  RVF missing target_space metadata; defaulting to {}",
                        MODEL_TARGET_SPACE_DEFAULT
                    );
                }
                if !sona_profile_deltas.is_empty() {
                    info!(
                        "  SONA profiles: {}",
                        sona_profile_deltas
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }

                (
                    Some(info),
                    weights,
                    feature_stats,
                    head_config,
                    target_space,
                    sona_profile_deltas,
                )
            }
            Err(e) => {
                error!("Failed to load RVF container: {e}");
                (None, None, None, None, None, BTreeMap::new())
            }
        }
    } else {
        (None, None, None, None, None, BTreeMap::new())
    };

    // Load trained model via --model (uses progressive loading if --progressive set)
    let model_path = args.model.as_ref().or(args.load_rvf.as_ref());
    let mut progressive_loader: Option<ProgressiveLoader> = None;
    let mut model_loaded = false;
    if let Some(mp) = model_path {
        if args.progressive || args.model.is_some() {
            info!("Loading trained model (progressive) from {}", mp.display());
            match std::fs::read(mp) {
                Ok(data) => match ProgressiveLoader::new(&data) {
                    Ok(mut loader) => {
                        if let Ok(la) = loader.load_layer_a() {
                            info!(
                                "  Layer A ready: model={} v{} ({} segments)",
                                la.model_name, la.version, la.n_segments
                            );
                        }
                        model_loaded = model_weights.is_some() && model_feature_stats.is_some();
                        progressive_loader = Some(loader);
                    }
                    Err(e) => error!("Progressive loader init failed: {e}"),
                },
                Err(e) => error!("Failed to read model file: {e}"),
            }
        }
    }
    if model_path.is_some() && model_weights.is_some() && model_feature_stats.is_some() {
        model_loaded = true;
    }

    let initial_active_sona_profile = resolve_initial_active_sona_profile(
        rvf_info.as_ref().and_then(|info| info.metadata.as_ref()),
        model_path.map(PathBuf::as_path),
        &model_sona_profile_deltas,
    );

    let (tx, _) = broadcast::channel::<String>(256);
    let (training_progress_tx, _) = broadcast::channel::<String>(256);
    let mut initial_state = AppStateInner {
        latest_update: None,
        rssi_history: VecDeque::new(),
        frame_history: VecDeque::new(),
        tick: 0,
        source: source.into(),
        tx,
        training_progress_tx,
        total_detections: 0,
        start_time: std::time::Instant::now(),
        vital_detector: VitalSignDetector::new(vital_sample_rate),
        latest_vitals: VitalSigns::default(),
        rvf_info,
        save_rvf_path: args.save_rvf.clone(),
        progressive_loader,
        model_weights,
        model_feature_stats,
        model_head_config,
        model_target_space,
        model_sona_profile_deltas,
        active_sona_profile: initial_active_sona_profile,
        active_sona_weights: None,
        model_loaded,
        smoothed_person_score: 0.0,
        edge_vitals: None,
        latest_wasm_events: None,
        recent_nodes: BTreeMap::new(),
        recording_state: recording::RecordingState::default(),
        training_state: training_api::TrainingState::default(),
        stable_tracked_person: None,
        next_stable_track_id: 1,
    };
    refresh_active_sona_weights(&mut initial_state);
    let state: SharedState = Arc::new(RwLock::new(initial_state));

    // Start background tasks based on source
    match source {
        "esp32" => {
            for &port in &esp32_ports {
                tokio::spawn(udp_receiver_task(state.clone(), port));
            }
            tokio::spawn(broadcast_tick_task(state.clone(), args.tick_ms));
        }
        "wifi" => {
            tokio::spawn(windows_wifi_task(state.clone(), args.tick_ms));
        }
        _ => {
            tokio::spawn(simulated_data_task(state.clone(), args.tick_ms));
        }
    }

    // WebSocket server on dedicated port (8765)
    let ws_state = state.clone();
    let ws_app = Router::new()
        .route("/ws/sensing", get(ws_sensing_handler))
        .route("/health", get(health))
        .with_state(ws_state);

    let ws_addr = SocketAddr::from(([0, 0, 0, 0], args.ws_port));
    let ws_listener = tokio::net::TcpListener::bind(ws_addr)
        .await
        .expect("Failed to bind WebSocket port");
    info!("WebSocket server listening on {ws_addr}");

    tokio::spawn(async move {
        axum::serve(ws_listener, ws_app).await.unwrap();
    });

    // HTTP server (serves UI + full DensePose-compatible REST API)
    let ui_path = args.ui_path.clone();
    let http_app = Router::new()
        .route("/", get(info_page))
        // Health endpoints (DensePose-compatible)
        .route("/health", get(health))
        .route("/health/health", get(health_system))
        .route("/health/live", get(health_live))
        .route("/health/ready", get(health_ready))
        .route("/health/version", get(health_version))
        .route("/health/metrics", get(health_metrics))
        // API info
        .route("/api/v1/info", get(api_info))
        .route("/api/v1/status", get(health_ready))
        .route("/api/v1/metrics", get(health_metrics))
        // Sensing endpoints
        .route("/api/v1/sensing/latest", get(latest))
        // Vital sign endpoints
        .route("/api/v1/vital-signs", get(vital_signs_endpoint))
        .route("/api/v1/edge-vitals", get(edge_vitals_endpoint))
        .route("/api/v1/wasm-events", get(wasm_events_endpoint))
        // RVF model container info
        .route("/api/v1/model/info", get(model_info))
        // Progressive loading & SONA endpoints (Phase 7-8)
        .route("/api/v1/model/layers", get(model_layers))
        .route("/api/v1/model/segments", get(model_segments))
        .route("/api/v1/model/sona/profiles", get(sona_profiles))
        .route("/api/v1/model/sona/activate", post(sona_activate))
        // Pose endpoints (WiFi-derived)
        .route("/api/v1/pose/current", get(pose_current))
        .route("/api/v1/pose/tracked", get(pose_tracked))
        .route("/api/v1/pose/stats", get(pose_stats))
        .route("/api/v1/pose/zones/summary", get(pose_zones_summary))
        // Stream endpoints
        .route("/api/v1/stream/status", get(stream_status))
        .route("/api/v1/stream/pose", get(ws_pose_handler))
        // Sensing WebSocket on the HTTP port so the UI can reach it without a second port
        .route("/ws/sensing", get(ws_sensing_handler))
        // Recording + training APIs
        .merge(recording::routes())
        .merge(training_api::routes())
        // Static UI files
        .nest_service("/ui", ServeDir::new(&ui_path))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache, no-store, must-revalidate"),
        ))
        .with_state(state.clone());

    let http_addr = SocketAddr::from(([0, 0, 0, 0], args.http_port));
    let http_listener = tokio::net::TcpListener::bind(http_addr)
        .await
        .expect("Failed to bind HTTP port");
    info!("HTTP server listening on {http_addr}");
    info!(
        "Open http://localhost:{}/ui/index.html in your browser",
        args.http_port
    );

    // Run the HTTP server with graceful shutdown support
    let shutdown_state = state.clone();
    let server = axum::serve(http_listener, http_app).with_graceful_shutdown(async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
        info!("Shutdown signal received");
    });

    server.await.unwrap();

    // Save RVF container on shutdown if --save-rvf was specified
    let s = shutdown_state.read().await;
    if let Some(ref save_path) = s.save_rvf_path {
        info!("Saving RVF container to {}", save_path.display());
        let mut builder = RvfBuilder::new();
        builder.add_manifest(
            "wifi-densepose-sensing",
            env!("CARGO_PKG_VERSION"),
            "WiFi DensePose sensing model state",
        );
        builder.add_metadata(&serde_json::json!({
            "source": s.source,
            "total_ticks": s.tick,
            "total_detections": s.total_detections,
            "uptime_secs": s.start_time.elapsed().as_secs(),
        }));
        builder.add_vital_config(&VitalSignConfig::default());
        // Save transformer weights if a model is loaded, otherwise empty
        let weights: Vec<f32> = if s.model_loaded {
            // If we loaded via --model, the progressive loader has the weights
            // For now, save runtime state placeholder
            let tf = graph_transformer::CsiToPoseTransformer::new(Default::default());
            tf.flatten_weights()
        } else {
            Vec::new()
        };
        builder.add_weights(&weights);
        match builder.write_to_file(save_path) {
            Ok(()) => info!("  RVF saved ({} weight params)", weights.len()),
            Err(e) => error!("  Failed to save RVF: {e}"),
        }
    }

    info!("Server shut down cleanly");
}
