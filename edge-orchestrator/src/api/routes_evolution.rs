use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::{
    routing::{get, post},
    Json, Router,
};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::process::Command;
use tokio::time::sleep;

use crate::calibration::IphoneStereoExtrinsic;
use crate::operator::{canonical_body_points_3d, OperatorSnapshot, OperatorSource};
use crate::sensing::{CsiSnapshot, StereoSnapshot, VisionDevicePose, VisionSnapshot, WifiPoseSnapshot};
use crate::AppState;

const DEFAULT_SCENE_ID: &str = "chek_humanoid_lan_sense_main_stage_v1";
const DEFAULT_SCENE_NAME: &str = "CHEK Humanoid / LAN-SENSE / 主采集区 / V1";
const DEFAULT_SCENE_NAME_RULE: &str =
    "建议用“品牌 / 网络 / 空间 / 版本”，例如 CHEK Humanoid / LAN-SENSE / 主采集区 / V1。";
const DEFAULT_SCENE_ID_RULE: &str =
    "系统标识统一用小写英文和下划线，例如 chek_humanoid_lan_sense_main_stage_v1。";
const MIN_TEACHER_BODY_JOINTS: usize = 8;
const DEFAULT_SCENE_HISTORY_LIMIT: usize = 24;
const DEFAULT_MIN_RECORDING_QUALITY: f64 = 0.70;
const DEFAULT_ZERO_SHOT_WARMUP_SECS: u64 = 10;
const DEFAULT_ZERO_SHOT_VALIDATION_SECS: u64 = 15;
const DEFAULT_FEW_SHOT_LORA_RANK: u8 = 8;
const DEFAULT_FEW_SHOT_LORA_EPOCHS: u32 = 30;
const FEW_SHOT_GATE_MAX_DOMAIN_GAP_RATIO: f64 = 1.5;
const FEW_SHOT_GATE_MIN_ADAPTATION_SPEEDUP: f64 = 5.0;
const GUIDE_PHASE_ADVANCE_COOLDOWN_MS: u64 = 900;
const GUIDE_PHASE_HOLD_DEFAULT_MS: u64 = 700;
const GUIDE_CENTER_TARGET_TOLERANCE_M: f32 = 0.18;
const GUIDE_CENTER_START_TOLERANCE_M: f32 = 0.90;
const COCO_NOSE: usize = 0;
const COCO_LEFT_SHOULDER: usize = 5;
const COCO_RIGHT_SHOULDER: usize = 6;
const COCO_LEFT_WRIST: usize = 9;
const COCO_RIGHT_WRIST: usize = 10;
const COCO_LEFT_HIP: usize = 11;
const COCO_RIGHT_HIP: usize = 12;
const COCO_LEFT_KNEE: usize = 13;
const COCO_RIGHT_KNEE: usize = 14;
const COCO_LEFT_ANKLE: usize = 15;
const COCO_RIGHT_ANKLE: usize = 16;

#[derive(Debug, Clone, Copy)]
struct LocalCrossDomainMetrics {
    in_domain_mpjpe: f64,
    cross_domain_mpjpe: f64,
    few_shot_mpjpe: f64,
    cross_hardware_mpjpe: f64,
    domain_gap_ratio: f64,
    adaptation_speedup: f64,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/evolution/summary", get(get_current))
        .route("/evolution/session/current", get(get_current))
        .route(
            "/evolution/scene/geometry/auto-draft",
            post(post_scene_geometry_auto_draft),
        )
        .route(
            "/evolution/scene/geometry",
            get(get_scene_geometry).post(post_scene_geometry),
        )
        .route(
            "/evolution/model/validate-zero-shot",
            post(post_validate_zero_shot),
        )
        .route(
            "/evolution/model/apply-zero-shot",
            post(post_apply_zero_shot),
        )
        .route(
            "/evolution/model/rollback-zero-shot",
            post(post_rollback_zero_shot),
        )
        .route(
            "/evolution/model/calibrate-few-shot",
            post(post_calibrate_few_shot),
        )
        .route(
            "/evolution/model/preview-few-shot",
            post(post_preview_few_shot),
        )
        .route(
            "/evolution/model/rollback-few-shot",
            post(post_rollback_few_shot),
        )
        .route(
            "/evolution/model/report-few-shot-evaluation",
            post(post_report_few_shot_evaluation),
        )
        .route(
            "/evolution/model/import-few-shot-evaluation",
            post(post_import_few_shot_evaluation),
        )
        .route(
            "/evolution/model/import-latest-few-shot-evaluation",
            post(post_import_latest_few_shot_evaluation),
        )
        .route(
            "/evolution/model/evaluate-few-shot",
            post(post_evaluate_few_shot),
        )
        .route(
            "/evolution/model/evaluate-few-shot-recordings",
            post(post_evaluate_few_shot_recordings),
        )
        .route(
            "/evolution/model/evaluate-few-shot-benchmark-summaries",
            post(post_evaluate_few_shot_benchmark_summaries),
        )
        .route("/evolution/model/apply-few-shot", post(post_apply_few_shot))
        .route("/evolution/session/start", post(post_session_start))
        .route("/evolution/session/step/start", post(post_step_start))
        .route("/evolution/session/step/stop", post(post_step_stop))
        .route("/evolution/session/train", post(post_train))
        .route("/evolution/session/apply", post(post_apply))
        .with_state(state)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionStepState {
    code: String,
    title: String,
    instruction: String,
    label: String,
    duration_secs: u64,
    status: String,
    recording_id: Option<String>,
    frame_count: Option<u64>,
    quality_score: Option<f64>,
    started_at_ms: Option<u64>,
    completed_at_ms: Option<u64>,
    #[serde(default)]
    assessed_score: Option<f64>,
    #[serde(default)]
    assessed_tone: String,
    #[serde(default)]
    assessed_summary: Option<String>,
    #[serde(default)]
    rerecord_recommended: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionSessionState {
    capture_session_id: Option<String>,
    scene_id: String,
    scene_name: String,
    status: String,
    created_at_ms: Option<u64>,
    updated_at_ms: u64,
    current_step_code: Option<String>,
    training_requested_at_ms: Option<u64>,
    applied_at_ms: Option<u64>,
    #[serde(default)]
    latest_training_report_id: Option<String>,
    #[serde(default)]
    latest_applied_report_id: Option<String>,
    #[serde(default)]
    latest_zero_shot_validation_id: Option<String>,
    #[serde(default)]
    latest_few_shot_candidate_id: Option<String>,
    #[serde(default)]
    latest_applied_few_shot_candidate_id: Option<String>,
    #[serde(default)]
    active_training_kind: Option<String>,
    steps: Vec<EvolutionStepState>,
    #[serde(default)]
    step_runtimes: Vec<EvolutionStepRuntimeState>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct EvolutionStepRuntimeState {
    step_code: String,
    phase_index: usize,
    max_phase_index: usize,
    phase_total: usize,
    phase_completed: bool,
    last_advanced_at_ms: Option<u64>,
    phase_hold_started_at_ms: Option<u64>,
    #[serde(default)]
    lead_side: Option<String>,
    #[serde(default)]
    baseline_left_wrist_side_offset_m: Option<f32>,
    #[serde(default)]
    baseline_right_wrist_side_offset_m: Option<f32>,
    #[serde(default)]
    baseline_left_wrist_forward_offset_m: Option<f32>,
    #[serde(default)]
    baseline_right_wrist_forward_offset_m: Option<f32>,
    #[serde(default)]
    assessed_recording_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct EvolutionSceneInfo {
    scene_id: String,
    scene_name: String,
    display_name_rule: &'static str,
    system_id_rule: &'static str,
}

#[derive(Debug, Serialize)]
struct EvolutionReadiness {
    csi_ready: bool,
    stereo_ready: bool,
    wifi_ready: bool,
    phone_ready: bool,
    robot_ready: bool,
    teacher_ready: bool,
    room_empty_ready: bool,
    full_multimodal_ready: bool,
    pose_capture_ready: bool,
    anchor_source: String,
    selected_operator_track_id: Option<String>,
    iphone_visible_hand_count: usize,
    hand_match_count: usize,
    hand_match_score: f32,
    left_wrist_gap_m: Option<f32>,
    right_wrist_gap_m: Option<f32>,
    suggested_quality_score: f64,
    issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionModelSummary {
    loaded: bool,
    status: String,
    model_id: Option<String>,
    target_space: Option<String>,
    scene_id: Option<String>,
    base_model_id: Option<String>,
    global_parent_model_id: Option<String>,
    best_pck: Option<f64>,
    scene_history_used: bool,
    scene_history_sample_count: usize,
    adaptation_mode: String,
    sona_profiles: Vec<String>,
    active_sona_profile: Option<String>,
    lora_profiles: Vec<String>,
    active_lora_profile: Option<String>,
    raw: Value,
}

#[derive(Debug, Serialize)]
struct EvolutionRecordingSummary {
    active: bool,
    session_id: Option<String>,
    session_name: Option<String>,
    capture_session_id: Option<String>,
    step_code: Option<String>,
    frame_count: u64,
    duration_secs: Option<u64>,
    raw: Value,
}

#[derive(Debug, Serialize)]
struct EvolutionTrainingSummary {
    active: bool,
    phase: String,
    epoch: u32,
    total_epochs: u32,
    best_pck: f64,
    best_epoch: u32,
    val_pck: f64,
    eta_secs: Option<u64>,
    raw: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct EvolutionTrainingQualityGate {
    eligible_dataset_count: usize,
    eligible_empty_count: usize,
    eligible_pose_count: usize,
    skipped_rerecord_dataset_ids: Vec<String>,
    skipped_rerecord_step_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionTrainingSelectedRecording {
    dataset_id: String,
    step_code: String,
    title: String,
    label: String,
    quality_score: Option<f64>,
    rerecord_recommended: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionTrainingArtifact {
    train_job_id: String,
    status: String,
    train_type: String,
    scene_id: String,
    scene_name: String,
    capture_session_id: Option<String>,
    requested_at_ms: u64,
    applied_at_ms: Option<u64>,
    base_model_id: Option<String>,
    applied_model_id: Option<String>,
    #[serde(default)]
    candidate_model_id: Option<String>,
    #[serde(default)]
    candidate_model_created_at: Option<String>,
    adaptation_mode: String,
    explicit_dataset_ids: Vec<String>,
    resolved_dataset_ids: Vec<String>,
    added_history_dataset_ids: Vec<String>,
    selected_recordings: Vec<EvolutionTrainingSelectedRecording>,
    scene_history_used: bool,
    scene_history_sample_count: usize,
    min_recording_quality: f64,
    quality_gate: EvolutionTrainingQualityGate,
    lora_profile_id: Option<String>,
    geometry_profile_id: Option<String>,
    best_pck: Option<f64>,
    best_epoch: Option<u32>,
    evaluator_summary: Option<Value>,
    cross_domain_summary: Option<Value>,
    promotion_gate_status: String,
}

#[derive(Debug, Serialize)]
struct EvolutionDiagnostics {
    upstream_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionGuideCheck {
    label: String,
    ok: bool,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionStepGuide {
    step_code: String,
    ready: bool,
    tone: String,
    headline: String,
    detail: String,
    progress: f64,
    phase_index: usize,
    phase_total: usize,
    phase_label: Option<String>,
    phase_detail: Option<String>,
    phase_completed: bool,
    phase_hold_ms: Option<u64>,
    phase_hold_target_ms: Option<u64>,
    phase_hold_progress: Option<f64>,
    checks: Vec<EvolutionGuideCheck>,
}

#[derive(Debug, Serialize)]
struct EvolutionCurrentResponse {
    ok: bool,
    scene: EvolutionSceneInfo,
    session: EvolutionSessionState,
    readiness: EvolutionReadiness,
    current_step_guide: Option<EvolutionStepGuide>,
    step_guides: Vec<EvolutionStepGuide>,
    model: EvolutionModelSummary,
    recording: EvolutionRecordingSummary,
    training: EvolutionTrainingSummary,
    training_report: Option<EvolutionTrainingArtifact>,
    zero_shot_validation: Option<EvolutionZeroShotValidationArtifact>,
    few_shot_candidate: Option<EvolutionFewShotCandidateArtifact>,
    few_shot_evaluator_inbox_dir: String,
    few_shot_benchmark_discovery: EvolutionFewShotBenchmarkDiscovery,
    diagnostics: EvolutionDiagnostics,
}

#[derive(Debug, Clone, Default, Serialize)]
struct EvolutionFewShotBenchmarkDiscovery {
    capture_root_dir: String,
    latest_preapply_summary_path: Option<String>,
    latest_postapply_summary_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionSceneGeometryAnchor {
    id: String,
    #[serde(default)]
    label: Option<String>,
    position_m: [f64; 3],
    #[serde(default = "default_rotation_deg")]
    rotation_deg: [f64; 3],
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionSceneGeometry {
    scene_id: String,
    scene_name: String,
    coordinate_frame_version: String,
    updated_at_ms: u64,
    #[serde(default = "default_geometry_source")]
    source: String,
    ap_nodes: Vec<EvolutionSceneGeometryAnchor>,
    stereo_rig: EvolutionSceneGeometryAnchor,
    #[serde(default)]
    phone_pose: Option<EvolutionSceneGeometryAnchor>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionSceneGeometrySummary {
    coordinate_frame_version: Option<String>,
    ap_count: usize,
    stereo_defined: bool,
    phone_defined: bool,
    updated_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionSceneGeometryResponse {
    scene_id: String,
    scene_name: String,
    exists: bool,
    summary: EvolutionSceneGeometrySummary,
    geometry: Option<EvolutionSceneGeometry>,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionSceneGeometryAutoDraftSourceBreakdown {
    ap_nodes: String,
    stereo_rig: String,
    phone_pose: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionSceneGeometryAutoDraftConfidenceBreakdown {
    ap_nodes: f64,
    stereo_rig: f64,
    phone_pose: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionSceneGeometryAutoDraftResponse {
    scene_id: String,
    scene_name: String,
    generated_at_ms: u64,
    ready_to_save: bool,
    confidence_score: f64,
    message: String,
    warnings: Vec<String>,
    missing_fields: Vec<String>,
    source_breakdown: EvolutionSceneGeometryAutoDraftSourceBreakdown,
    confidence_breakdown: EvolutionSceneGeometryAutoDraftConfidenceBreakdown,
    geometry: EvolutionSceneGeometry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionZeroShotValidationArtifact {
    validation_id: String,
    scene_id: String,
    scene_name: String,
    validated_at_ms: u64,
    status: String,
    message: String,
    adaptation_mode: String,
    validation_mode: String,
    warmup_secs: u64,
    validation_secs: u64,
    geometry_ready: bool,
    model_ready: bool,
    readiness_ready: bool,
    live_model_id: Option<String>,
    base_model_id: Option<String>,
    geometry_summary: EvolutionSceneGeometrySummary,
    blockers: Vec<String>,
    compare_status: String,
    compare_summary: Option<Value>,
    promotion_gate_status: String,
    #[serde(default)]
    promotion_gate_reason: Option<String>,
    #[serde(default)]
    applied_at_ms: Option<u64>,
    #[serde(default)]
    rolled_back_at_ms: Option<u64>,
    #[serde(default)]
    auto_policy_status: String,
    #[serde(default)]
    auto_policy_reason: Option<String>,
    #[serde(default)]
    auto_apply_min_improvement_mm: Option<f64>,
    #[serde(default)]
    auto_applied_at_ms: Option<u64>,
    #[serde(default)]
    auto_rolled_back_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionFewShotCandidateArtifact {
    calibration_id: String,
    scene_id: String,
    scene_name: String,
    source_capture_session_id: Option<String>,
    requested_at_ms: u64,
    completed_at_ms: Option<u64>,
    previewed_at_ms: Option<u64>,
    rolled_back_at_ms: Option<u64>,
    #[serde(default)]
    applied_at_ms: Option<u64>,
    status: String,
    runtime_path: String,
    profile_name: String,
    base_model_id: String,
    candidate_model_id: Option<String>,
    candidate_model_created_at: Option<String>,
    source_dataset_ids: Vec<String>,
    selected_recordings: Vec<EvolutionTrainingSelectedRecording>,
    min_recording_quality: f64,
    quality_gate: EvolutionTrainingQualityGate,
    rank: u8,
    epochs: u32,
    best_pck: Option<f64>,
    best_epoch: Option<u32>,
    before_metrics: Option<Value>,
    after_metrics: Option<Value>,
    #[serde(default)]
    evaluator_summary: Option<Value>,
    #[serde(default)]
    cross_domain_summary: Option<Value>,
    promotion_gate_status: String,
    #[serde(default)]
    promotion_gate_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EvolutionSessionStartRequest {
    scene_name: Option<String>,
    scene_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EvolutionStepStartRequest {
    step_code: String,
}

#[derive(Debug, Deserialize)]
struct EvolutionSceneGeometryUpsertRequest {
    coordinate_frame_version: String,
    #[serde(default)]
    ap_nodes: Vec<EvolutionSceneGeometryAnchor>,
    #[serde(default)]
    stereo_rig: Option<EvolutionSceneGeometryAnchor>,
    #[serde(default)]
    phone_pose: Option<EvolutionSceneGeometryAnchor>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionZeroShotValidationRequest {
    #[serde(default)]
    warmup_secs: Option<u64>,
    #[serde(default)]
    validation_secs: Option<u64>,
    #[serde(default)]
    dataset_ids: Vec<String>,
    #[serde(default)]
    benchmark_summary_path: Option<String>,
    #[serde(default)]
    candidate_model_id: Option<String>,
    #[serde(default)]
    candidate_sona_profile: Option<String>,
    #[serde(default)]
    auto_apply: Option<bool>,
    #[serde(default)]
    auto_rollback_if_regressed: Option<bool>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionFewShotCalibrationRequest {
    #[serde(default)]
    rank: Option<u8>,
    #[serde(default)]
    epochs: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionFewShotEvaluationReportRequest {
    #[serde(default)]
    calibration_id: Option<String>,
    #[serde(default)]
    metric_name: Option<String>,
    #[serde(default)]
    in_domain_metric: Option<f64>,
    #[serde(default)]
    unseen_room_zero_shot_metric: Option<f64>,
    #[serde(default)]
    unseen_room_few_shot_metric: Option<f64>,
    #[serde(default)]
    cross_hardware_metric: Option<f64>,
    #[serde(default)]
    domain_gap_ratio: Option<f64>,
    #[serde(default)]
    few_shot_improvement_delta: Option<f64>,
    #[serde(default)]
    adaptation_speedup: Option<f64>,
    #[serde(default)]
    hardware_type: Option<String>,
    passed: bool,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct EvolutionFewShotEvaluationSample {
    bucket: String,
    predicted: Vec<f32>,
    ground_truth: Vec<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionFewShotEvaluateRequest {
    #[serde(default)]
    calibration_id: Option<String>,
    #[serde(default)]
    hardware_type: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    samples: Vec<EvolutionFewShotEvaluationSample>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct EvolutionFewShotRecordingEvalBucketRequest {
    bucket: String,
    #[serde(default)]
    dataset_ids: Vec<String>,
    #[serde(default)]
    model_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionFewShotEvaluateRecordingsRequest {
    #[serde(default)]
    calibration_id: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    producer: Option<String>,
    #[serde(default)]
    include_samples: bool,
    #[serde(default)]
    buckets: Vec<EvolutionFewShotRecordingEvalBucketRequest>,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionFewShotEvaluateBenchmarkSummariesRequest {
    #[serde(default)]
    calibration_id: Option<String>,
    #[serde(default)]
    in_domain_summary_path: Option<String>,
    zero_shot_summary_path: String,
    few_shot_summary_path: String,
    #[serde(default)]
    cross_hardware_summary_path: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    producer: Option<String>,
    #[serde(default)]
    include_samples: bool,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionFewShotEvaluationImportRequest {
    #[serde(default)]
    calibration_id: Option<String>,
    #[serde(default)]
    artifact_path: Option<String>,
    #[serde(default)]
    artifact: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
struct EvolutionFewShotLatestEvaluationImportRequest {
    #[serde(default)]
    calibration_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ExternalRecordingStatus {
    active: bool,
    session_id: Option<String>,
    session_name: Option<String>,
    capture_session_id: Option<String>,
    step_code: Option<String>,
    frame_count: u64,
    duration_secs: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExternalRecordingSession {
    id: String,
    capture_session_id: Option<String>,
    step_code: Option<String>,
    frame_count: u64,
    quality_score: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct ExternalRecordingList {
    recordings: Vec<ExternalRecordingSession>,
}

#[derive(Debug, Default, Deserialize)]
struct BenchmarkSummaryStep {
    recording_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct BenchmarkSummary {
    #[serde(default)]
    steps: Vec<BenchmarkSummaryStep>,
}

#[derive(Debug, Default, Deserialize)]
struct ExternalTrainingStatus {
    active: bool,
    epoch: u32,
    total_epochs: u32,
    val_pck: f64,
    best_pck: f64,
    best_epoch: u32,
    eta_secs: Option<u64>,
    phase: String,
}

#[derive(Debug, Default, Deserialize)]
struct ExternalLoraProfilesState {
    #[serde(default)]
    profiles: Vec<String>,
    #[serde(default)]
    active: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ExternalSonaProfilesState {
    #[serde(default)]
    profiles: Vec<String>,
    #[serde(default)]
    active: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ExternalModelCatalogItem {
    id: String,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    pck_score: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct ExternalModelCatalog {
    #[serde(default)]
    models: Vec<ExternalModelCatalogItem>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn default_rotation_deg() -> [f64; 3] {
    [0.0, 0.0, 0.0]
}

fn radians_to_degrees(values: [f64; 3]) -> [f64; 3] {
    [
        values[0].to_degrees(),
        values[1].to_degrees(),
        values[2].to_degrees(),
    ]
}

fn degrees_to_radians(values: [f64; 3]) -> [f64; 3] {
    [
        values[0].to_radians(),
        values[1].to_radians(),
        values[2].to_radians(),
    ]
}

fn build_scene_geometry_template(scene_id: &str, scene_name: &str) -> EvolutionSceneGeometry {
    EvolutionSceneGeometry {
        scene_id: scene_id.to_string(),
        scene_name: scene_name.to_string(),
        coordinate_frame_version: "chek-room-v1".to_string(),
        updated_at_ms: now_ms(),
        source: "auto_draft".to_string(),
        ap_nodes: vec![EvolutionSceneGeometryAnchor {
            id: format!("{scene_id}-ap-01"),
            label: Some("主 AP".to_string()),
            position_m: [0.0, 2.4, 0.0],
            rotation_deg: default_rotation_deg(),
            notes: Some("自动草稿默认 AP 占位，请按现场安装位确认。".to_string()),
        }],
        stereo_rig: EvolutionSceneGeometryAnchor {
            id: format!("{scene_id}-stereo-01"),
            label: Some(format!("{scene_name} 双目")),
            position_m: [1.8, 1.45, 0.0],
            rotation_deg: [0.0, -90.0, 0.0],
            notes: Some("自动草稿默认双目位姿，请按现场安装位确认。".to_string()),
        },
        phone_pose: None,
        notes: Some("自动草稿：请确认 AP / 双目 / 手机位姿后再保存。".to_string()),
    }
}

fn scene_uses_default_stage_layout(session: &EvolutionSessionState) -> bool {
    session.scene_id.trim() == DEFAULT_SCENE_ID
}

fn compose_anchor_pose(
    base_anchor: &EvolutionSceneGeometryAnchor,
    translation_m: [f32; 3],
    rotation_quat_xyzw: [f32; 4],
    id: String,
    label: String,
    notes: Option<String>,
) -> EvolutionSceneGeometryAnchor {
    let base_rpy = degrees_to_radians(base_anchor.rotation_deg);
    let base_rot = UnitQuaternion::from_euler_angles(base_rpy[0], base_rpy[1], base_rpy[2]);
    let local_translation = Vector3::new(
        translation_m[0] as f64,
        translation_m[1] as f64,
        translation_m[2] as f64,
    );
    let world_translation = base_rot.transform_vector(&local_translation)
        + Vector3::new(
            base_anchor.position_m[0],
            base_anchor.position_m[1],
            base_anchor.position_m[2],
        );
    let local_rot = UnitQuaternion::from_quaternion(Quaternion::new(
        rotation_quat_xyzw[3] as f64,
        rotation_quat_xyzw[0] as f64,
        rotation_quat_xyzw[1] as f64,
        rotation_quat_xyzw[2] as f64,
    ));
    let world_rot = base_rot * local_rot;
    let (rx, ry, rz) = world_rot.euler_angles();
    EvolutionSceneGeometryAnchor {
        id,
        label: Some(label),
        position_m: [
            world_translation.x,
            world_translation.y,
            world_translation.z,
        ],
        rotation_deg: radians_to_degrees([rx, ry, rz]),
        notes,
    }
}

fn derive_phone_pose_from_iphone_stereo(
    scene_id: &str,
    stereo_rig: &EvolutionSceneGeometryAnchor,
    calibration: &IphoneStereoExtrinsic,
) -> Option<EvolutionSceneGeometryAnchor> {
    if calibration.target_frame.trim() != "stereo_pair_frame"
        && calibration.target_frame.trim() != "operator_frame"
    {
        return None;
    }
    Some(compose_anchor_pose(
        stereo_rig,
        calibration.extrinsic_translation_m,
        calibration.extrinsic_rotation_quat,
        format!("{scene_id}-phone-01"),
        "胸前 iPhone".to_string(),
        Some(format!(
            "自动草稿：由 iphone_stereo 标定回填，RMS {:.3}m。",
            calibration.rms_error_m
        )),
    ))
}

fn merge_geometry_notes(existing: Option<&str>, extra: &str) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(value) = existing {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }
    if !extra.trim().is_empty() {
        parts.push(extra.trim().to_string());
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn build_scene_geometry_auto_draft(
    session: &EvolutionSessionState,
    existing: Option<EvolutionSceneGeometry>,
    iphone_calibration: Option<IphoneStereoExtrinsic>,
) -> EvolutionSceneGeometryAutoDraftResponse {
    let generated_at_ms = now_ms();
    let default_geometry = build_scene_geometry_template(&session.scene_id, &session.scene_name);
    let mut geometry = existing.clone().unwrap_or_else(|| default_geometry.clone());
    geometry.scene_id = session.scene_id.clone();
    geometry.scene_name = session.scene_name.clone();
    geometry.updated_at_ms = generated_at_ms;
    geometry.source = "auto_draft".to_string();

    let mut warnings = Vec::new();
    let mut missing_fields = Vec::new();
    let ap_confidence;
    let stereo_confidence;
    let mut phone_confidence = 0.0_f64;

    let ap_nodes_source = if existing
        .as_ref()
        .map(|value| !value.ap_nodes.is_empty())
        .unwrap_or(false)
    {
        ap_confidence = 0.92;
        "existing_geometry".to_string()
    } else if scene_uses_default_stage_layout(session) {
        geometry.ap_nodes = default_geometry.ap_nodes.clone();
        if let Some(anchor) = geometry.ap_nodes.first_mut() {
            anchor.notes = merge_geometry_notes(
                anchor.notes.as_deref(),
                "自动草稿：按 CHEK 主采集区默认 AP 布局回填，请按现场安装位确认。",
            );
        }
        warnings.push("AP 位置已按 CHEK 主采集区默认布局回填，请按现场安装位确认。".to_string());
        ap_confidence = 0.68;
        "stage_default_layout".to_string()
    } else {
        geometry.ap_nodes = default_geometry.ap_nodes.clone();
        warnings.push("AP 位置当前仍是模板占位，请按现场安装位确认。".to_string());
        ap_confidence = 0.28;
        "template_placeholder".to_string()
    };

    let stereo_rig_source = if existing.is_some() {
        stereo_confidence = 0.88;
        "existing_geometry".to_string()
    } else if scene_uses_default_stage_layout(session) {
        geometry.stereo_rig = default_geometry.stereo_rig.clone();
        geometry.stereo_rig.notes = merge_geometry_notes(
            geometry.stereo_rig.notes.as_deref(),
            "自动草稿：按 CHEK 主采集区默认双目机位回填，请按现场机位确认。",
        );
        warnings.push("双目位姿已按 CHEK 主采集区默认布局回填，请按现场机位确认。".to_string());
        stereo_confidence = 0.64;
        "stage_default_layout".to_string()
    } else {
        geometry.stereo_rig = default_geometry.stereo_rig.clone();
        warnings.push("双目位姿当前仍是模板初值，请按现场机位确认。".to_string());
        stereo_confidence = 0.35;
        "template_default".to_string()
    };

    let mut phone_pose_source = if geometry.phone_pose.is_some() {
        phone_confidence = 0.55;
        "existing_geometry".to_string()
    } else {
        "not_available".to_string()
    };

    if let Some(calibration) = iphone_calibration.as_ref() {
        if let Some(phone_pose) = derive_phone_pose_from_iphone_stereo(
            &session.scene_id,
            &geometry.stereo_rig,
            calibration,
        ) {
            geometry.phone_pose = Some(phone_pose);
            phone_pose_source = "iphone_stereo_calibration".to_string();
            phone_confidence = if calibration.rms_error_m <= 0.08 {
                0.92
            } else if calibration.rms_error_m <= 0.15 {
                0.78
            } else {
                warnings.push(format!(
                    "iphone_stereo 标定 RMS 为 {:.3}m，手机位姿仅作粗略草稿。",
                    calibration.rms_error_m
                ));
                0.42
            };
        } else if geometry.phone_pose.is_none() {
            warnings.push("当前 iphone_stereo 标定 target_frame 不是 stereo_pair_frame，无法自动回填手机位姿。".to_string());
            missing_fields.push("phone_pose".to_string());
        }
    } else if geometry.phone_pose.is_none() {
        warnings.push("当前没有 iphone_stereo 标定，手机位姿仍需人工确认。".to_string());
        missing_fields.push("phone_pose".to_string());
    }

    geometry.notes = merge_geometry_notes(
        geometry.notes.as_deref(),
        "自动草稿：已尽量回填现场可推断信息，请确认后保存。",
    );

    let ready_to_save = !geometry.ap_nodes.is_empty() && geometry.stereo_rig.id.trim().len() > 0;
    let confidence_score =
        (ap_confidence * 0.35 + stereo_confidence * 0.35 + phone_confidence * 0.30).clamp(0.0, 1.0);
    let message = if ready_to_save {
        if scene_uses_default_stage_layout(session) {
            "已生成 scene geometry 自动草稿：当前 scene 已优先按 CHEK 主采集区默认布局回填 AP / 双目，手机位姿优先按 iphone_stereo 标定回填。".to_string()
        } else {
            "已生成 scene geometry 自动草稿：AP / 双目若无现成 geometry 会先用模板占位，手机位姿优先按 iphone_stereo 标定回填。".to_string()
        }
    } else {
        "当前只能生成部分 geometry 草稿，仍缺少必要字段。".to_string()
    };

    EvolutionSceneGeometryAutoDraftResponse {
        scene_id: session.scene_id.clone(),
        scene_name: session.scene_name.clone(),
        generated_at_ms,
        ready_to_save,
        confidence_score,
        message,
        warnings,
        missing_fields,
        source_breakdown: EvolutionSceneGeometryAutoDraftSourceBreakdown {
            ap_nodes: ap_nodes_source,
            stereo_rig: stereo_rig_source,
            phone_pose: phone_pose_source,
        },
        confidence_breakdown: EvolutionSceneGeometryAutoDraftConfidenceBreakdown {
            ap_nodes: ap_confidence,
            stereo_rig: stereo_confidence,
            phone_pose: phone_confidence,
        },
        geometry,
    }
}

fn training_phase_failed(phase: &str) -> bool {
    let normalized = phase.trim().to_ascii_lowercase();
    normalized == "failed" || normalized.starts_with("failed_")
}

fn default_geometry_source() -> String {
    "manual".to_string()
}

fn evolution_session_path(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.data_dir.trim()).join("runtime/environment_evolution_session.json")
}

fn evolution_training_reports_dir(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.data_dir.trim()).join("runtime/environment_evolution_reports")
}

fn evolution_training_report_path(state: &AppState, train_job_id: &str) -> PathBuf {
    evolution_training_reports_dir(state).join(format!("{train_job_id}.json"))
}

fn evolution_zero_shot_reports_dir(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.data_dir.trim())
        .join("runtime/environment_evolution_zero_shot_reports")
}

fn evolution_zero_shot_report_path(state: &AppState, validation_id: &str) -> PathBuf {
    evolution_zero_shot_reports_dir(state).join(format!("{validation_id}.json"))
}

fn evolution_few_shot_candidates_dir(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.data_dir.trim())
        .join("runtime/environment_evolution_few_shot_candidates")
}

fn evolution_few_shot_candidate_path(state: &AppState, calibration_id: &str) -> PathBuf {
    evolution_few_shot_candidates_dir(state).join(format!("{calibration_id}.json"))
}

fn evolution_few_shot_imports_dir(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.data_dir.trim())
        .join("runtime/environment_evolution_few_shot_imports")
}

fn evolution_few_shot_import_path(
    state: &AppState,
    calibration_id: &str,
    imported_at_ms: u64,
) -> PathBuf {
    evolution_few_shot_imports_dir(state).join(format!("{calibration_id}-{imported_at_ms}.json"))
}

fn evolution_few_shot_inbox_dir(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.few_shot_evaluator_inbox_dir.trim())
}

fn evolution_benchmark_capture_dir(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.few_shot_benchmark_capture_dir.trim())
}

fn evolution_scene_geometry_dir(state: &AppState) -> PathBuf {
    PathBuf::from(state.config.data_dir.trim()).join("runtime/environment_evolution_geometry")
}

fn evolution_scene_geometry_path(state: &AppState, scene_id: &str) -> PathBuf {
    evolution_scene_geometry_dir(state).join(format!("{scene_id}.json"))
}

fn default_steps() -> Vec<EvolutionStepState> {
    [
        (
            "empty_room_01",
            "空房间基线 1",
            "请人工确认现场已清空，再保持 20 秒。",
            "empty",
            20,
        ),
        (
            "empty_room_02",
            "空房间基线 2",
            "继续保持空房间，再录 20 秒。",
            "empty",
            20,
        ),
        (
            "pose_idle_front_02",
            "正面站立",
            "正对双目，双臂自然下垂，保持轻微呼吸。",
            "pose",
            18,
        ),
        (
            "pose_turn_lr_01",
            "左右转体",
            "缓慢左转 90°，回中，再右转 90°，回中。",
            "pose",
            48,
        ),
        (
            "pose_reach_lr_01",
            "左右前伸",
            "左右手交替平举或前伸，动作尽量完整。",
            "pose",
            40,
        ),
        (
            "pose_arms_up_down_01",
            "双臂上举",
            "双臂从身体两侧抬到头顶，再自然放下。",
            "pose",
            32,
        ),
        (
            "pose_step_in_place_01",
            "原地踏步",
            "小步幅原地踏步，不要离开双目中心区域。",
            "pose",
            32,
        ),
        (
            "pose_bend_squat_01",
            "弯腰下蹲",
            "轻微下蹲或前屈，再恢复站立。",
            "pose",
            32,
        ),
    ]
    .into_iter()
    .map(
        |(code, title, instruction, label, duration_secs)| EvolutionStepState {
            code: code.to_string(),
            title: title.to_string(),
            instruction: instruction.to_string(),
            label: label.to_string(),
            duration_secs,
            status: "pending".to_string(),
            recording_id: None,
            frame_count: None,
            quality_score: None,
            started_at_ms: None,
            completed_at_ms: None,
            assessed_score: None,
            assessed_tone: String::new(),
            assessed_summary: None,
            rerecord_recommended: false,
        },
    )
    .collect()
}

fn canonical_step_duration_secs(step_code: &str) -> Option<u64> {
    match step_code {
        "empty_room_01" | "empty_room_02" => Some(20),
        "pose_idle_front_02" => Some(18),
        "pose_turn_lr_01" => Some(48),
        "pose_reach_lr_01" => Some(40),
        "pose_arms_up_down_01" | "pose_step_in_place_01" | "pose_bend_squat_01" => Some(32),
        _ => None,
    }
}

fn normalize_session_step_durations(session: &mut EvolutionSessionState) {
    for step in &mut session.steps {
        if let Some(duration_secs) = canonical_step_duration_secs(&step.code) {
            step.duration_secs = duration_secs;
        }
    }
}

fn default_session(scene_id: String, scene_name: String) -> EvolutionSessionState {
    EvolutionSessionState {
        capture_session_id: None,
        scene_id,
        scene_name,
        status: "idle".to_string(),
        created_at_ms: None,
        updated_at_ms: now_ms(),
        current_step_code: default_steps().first().map(|step| step.code.clone()),
        training_requested_at_ms: None,
        applied_at_ms: None,
        latest_training_report_id: None,
        latest_applied_report_id: None,
        latest_zero_shot_validation_id: None,
        latest_few_shot_candidate_id: None,
        latest_applied_few_shot_candidate_id: None,
        active_training_kind: None,
        steps: default_steps(),
        step_runtimes: Vec::new(),
    }
}

fn step_phase_labels(step_code: &str) -> &'static [&'static str] {
    match step_code {
        "pose_turn_lr_01" => &["左转到位", "回到正面", "右转到位", "回到正面"],
        "pose_reach_lr_01" => &["左手伸展", "回到起始位", "右手伸展", "回到起始位"],
        "pose_arms_up_down_01" => &["双臂举到位", "双臂放下"],
        "pose_step_in_place_01" => &["抬左脚一步", "换右脚一步"],
        "pose_bend_squat_01" => &["下蹲或前屈到位", "回正站立"],
        _ => &[],
    }
}

fn runtime_phase_label(
    step_code: &str,
    phase_index: usize,
    lead_side: Option<&str>,
) -> Option<&'static str> {
    match step_code {
        "pose_turn_lr_01" if lead_side == Some("right") => match phase_index {
            0 => Some("右转到位"),
            1 => Some("回到正面"),
            2 => Some("左转到位"),
            3 => Some("回到正面"),
            _ => None,
        },
        "pose_reach_lr_01" if lead_side == Some("right") => match phase_index {
            0 => Some("右手伸展"),
            1 => Some("回到起始位"),
            2 => Some("左手伸展"),
            3 => Some("回到起始位"),
            _ => None,
        },
        "pose_step_in_place_01" if lead_side == Some("right") => match phase_index {
            0 => Some("抬右脚一步"),
            1 => Some("换左脚一步"),
            _ => None,
        },
        _ => step_phase_labels(step_code).get(phase_index).copied(),
    }
}

fn phase_hold_target_ms(step_code: &str, phase_index: usize) -> u64 {
    match step_code {
        "pose_turn_lr_01" => 800,
        "pose_reach_lr_01" => {
            if phase_index % 2 == 0 {
                650
            } else {
                500
            }
        }
        "pose_arms_up_down_01" => 350,
        "pose_step_in_place_01" => 450,
        "pose_bend_squat_01" => 650,
        _ => GUIDE_PHASE_HOLD_DEFAULT_MS,
    }
}

fn sync_step_runtimes(session: &mut EvolutionSessionState) {
    for step in &session.steps {
        if session
            .step_runtimes
            .iter()
            .any(|runtime| runtime.step_code == step.code)
        {
            continue;
        }
        session.step_runtimes.push(EvolutionStepRuntimeState {
            step_code: step.code.clone(),
            phase_total: step_phase_labels(&step.code).len(),
            ..EvolutionStepRuntimeState::default()
        });
    }

    session.step_runtimes.retain(|runtime| {
        session
            .steps
            .iter()
            .any(|step| step.code == runtime.step_code)
    });
}

fn step_runtime<'a>(
    session: &'a EvolutionSessionState,
    step_code: &str,
) -> Option<&'a EvolutionStepRuntimeState> {
    session
        .step_runtimes
        .iter()
        .find(|runtime| runtime.step_code == step_code)
}

fn step_runtime_mut<'a>(
    session: &'a mut EvolutionSessionState,
    step_code: &str,
) -> Option<&'a mut EvolutionStepRuntimeState> {
    session
        .step_runtimes
        .iter_mut()
        .find(|runtime| runtime.step_code == step_code)
}

fn normalize_scene_id(value: Option<&str>) -> String {
    let raw = value.unwrap_or_default().trim();
    if raw.is_empty() {
        return DEFAULT_SCENE_ID.to_string();
    }

    let mut normalized = String::with_capacity(raw.len());
    let mut prev_sep = false;
    for ch in raw.chars() {
        let mapped = if ch.is_ascii_alphanumeric() {
            prev_sep = false;
            ch.to_ascii_lowercase()
        } else if ch == '-' || ch == '_' || ch.is_whitespace() {
            if prev_sep {
                continue;
            }
            prev_sep = true;
            '_'
        } else {
            continue;
        };
        normalized.push(mapped);
    }

    let normalized = normalized.trim_matches('_');
    if normalized.is_empty() {
        DEFAULT_SCENE_ID.to_string()
    } else {
        normalized.to_string()
    }
}

async fn load_session(state: &AppState) -> EvolutionSessionState {
    let path = evolution_session_path(state);
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => serde_json::from_str(&content)
            .map(|mut session: EvolutionSessionState| {
                normalize_session_step_durations(&mut session);
                session
            })
            .unwrap_or_else(|_| {
                default_session(DEFAULT_SCENE_ID.to_string(), DEFAULT_SCENE_NAME.to_string())
            }),
        Err(_) => default_session(DEFAULT_SCENE_ID.to_string(), DEFAULT_SCENE_NAME.to_string()),
    }
}

async fn save_session(
    state: &AppState,
    session: &EvolutionSessionState,
) -> Result<(), (StatusCode, String)> {
    let path = evolution_session_path(state);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建环境自进化目录失败: {error}"),
            )
        })?;
    }
    let payload = serde_json::to_vec_pretty(session).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("环境自进化状态序列化失败: {error}"),
        )
    })?;
    tokio::fs::write(&path, payload).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("环境自进化状态写入失败: {error}"),
        )
    })?;
    Ok(())
}

async fn load_training_report(
    state: &AppState,
    train_job_id: &str,
) -> Option<EvolutionTrainingArtifact> {
    let path = evolution_training_report_path(state, train_job_id);
    tokio::fs::read_to_string(&path)
        .await
        .ok()
        .and_then(|content| serde_json::from_str::<EvolutionTrainingArtifact>(&content).ok())
}

async fn save_training_report(
    state: &AppState,
    report: &EvolutionTrainingArtifact,
) -> Result<(), (StatusCode, String)> {
    let path = evolution_training_report_path(state, &report.train_job_id);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建训练报告目录失败: {error}"),
            )
        })?;
    }
    let payload = serde_json::to_vec_pretty(report).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("训练报告序列化失败: {error}"),
        )
    })?;
    tokio::fs::write(&path, payload).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("训练报告写入失败: {error}"),
        )
    })?;
    Ok(())
}

async fn load_zero_shot_validation_report(
    state: &AppState,
    validation_id: &str,
) -> Option<EvolutionZeroShotValidationArtifact> {
    let path = evolution_zero_shot_report_path(state, validation_id);
    tokio::fs::read_to_string(&path)
        .await
        .ok()
        .and_then(|content| {
            serde_json::from_str::<EvolutionZeroShotValidationArtifact>(&content).ok()
        })
}

async fn save_zero_shot_validation_report(
    state: &AppState,
    report: &EvolutionZeroShotValidationArtifact,
) -> Result<(), (StatusCode, String)> {
    let path = evolution_zero_shot_report_path(state, &report.validation_id);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建 zero-shot 报告目录失败: {error}"),
            )
        })?;
    }
    let payload = serde_json::to_vec_pretty(report).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("zero-shot 报告序列化失败: {error}"),
        )
    })?;
    tokio::fs::write(&path, payload).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("zero-shot 报告写入失败: {error}"),
        )
    })?;
    Ok(())
}

async fn load_runtime_model_summary(
    state: &AppState,
) -> Result<EvolutionModelSummary, (StatusCode, String)> {
    let model_raw = sensing_get_json::<Value>(state, "/api/v1/model/info")
        .await
        .map_err(|error| {
            (
                StatusCode::BAD_GATEWAY,
                format!("获取 live model 状态失败: {error}"),
            )
        })?;
    let lora_profiles_raw = sensing_get_json::<Value>(state, "/api/v1/models/lora/profiles")
        .await
        .unwrap_or_else(|_| json!({}));
    let sona_profiles_raw = sensing_get_json::<Value>(state, "/api/v1/model/sona/profiles")
        .await
        .unwrap_or_else(|_| json!({}));
    let lora_state =
        serde_json::from_value::<ExternalLoraProfilesState>(lora_profiles_raw).unwrap_or_default();
    let sona_state =
        serde_json::from_value::<ExternalSonaProfilesState>(sona_profiles_raw).unwrap_or_default();
    Ok(summarize_model_info(model_raw, &lora_state, &sona_state))
}

async fn load_few_shot_candidate_report(
    state: &AppState,
    calibration_id: &str,
) -> Option<EvolutionFewShotCandidateArtifact> {
    let path = evolution_few_shot_candidate_path(state, calibration_id);
    tokio::fs::read_to_string(&path)
        .await
        .ok()
        .and_then(|content| {
            serde_json::from_str::<EvolutionFewShotCandidateArtifact>(&content).ok()
        })
}

async fn save_few_shot_candidate_report(
    state: &AppState,
    report: &EvolutionFewShotCandidateArtifact,
) -> Result<(), (StatusCode, String)> {
    let path = evolution_few_shot_candidate_path(state, &report.calibration_id);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建 few-shot 候选目录失败: {error}"),
            )
        })?;
    }
    let payload = serde_json::to_vec_pretty(report).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("few-shot 候选序列化失败: {error}"),
        )
    })?;
    tokio::fs::write(&path, payload).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("few-shot 候选写入失败: {error}"),
        )
    })?;
    Ok(())
}

async fn save_few_shot_import_snapshot(
    state: &AppState,
    calibration_id: &str,
    imported_at_ms: u64,
    artifact: &Value,
) -> Result<String, (StatusCode, String)> {
    let path = evolution_few_shot_import_path(state, calibration_id, imported_at_ms);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建 few-shot evaluator 导入目录失败: {error}"),
            )
        })?;
    }
    let payload = serde_json::to_vec_pretty(artifact).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("few-shot evaluator artifact 序列化失败: {error}"),
        )
    })?;
    tokio::fs::write(&path, payload).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("few-shot evaluator artifact 写入失败: {error}"),
        )
    })?;
    Ok(path.to_string_lossy().to_string())
}

async fn load_scene_geometry(state: &AppState, scene_id: &str) -> Option<EvolutionSceneGeometry> {
    let path = evolution_scene_geometry_path(state, scene_id);
    tokio::fs::read_to_string(&path)
        .await
        .ok()
        .and_then(|content| serde_json::from_str::<EvolutionSceneGeometry>(&content).ok())
}

async fn save_scene_geometry(
    state: &AppState,
    geometry: &EvolutionSceneGeometry,
) -> Result<(), (StatusCode, String)> {
    let path = evolution_scene_geometry_path(state, &geometry.scene_id);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("创建 geometry 目录失败: {error}"),
            )
        })?;
    }
    let payload = serde_json::to_vec_pretty(geometry).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("scene geometry 序列化失败: {error}"),
        )
    })?;
    tokio::fs::write(&path, payload).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("scene geometry 写入失败: {error}"),
        )
    })?;
    Ok(())
}

fn json_string_array(raw: &Value, pointer: &str) -> Vec<String> {
    raw.pointer(pointer)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn infer_adaptation_mode(
    scene_history_used: bool,
    geometry_profile_id: Option<&str>,
    lora_profile_id: Option<&str>,
) -> String {
    if lora_profile_id.is_some() {
        "lora-adapted".to_string()
    } else if geometry_profile_id.is_some() {
        "geometry-conditioned".to_string()
    } else if scene_history_used {
        "scene-history".to_string()
    } else {
        "none".to_string()
    }
}

fn build_scene_geometry_response(
    scene_id: String,
    scene_name: String,
    geometry: Option<EvolutionSceneGeometry>,
) -> EvolutionSceneGeometryResponse {
    let summary = geometry.as_ref().map_or(
        EvolutionSceneGeometrySummary {
            coordinate_frame_version: None,
            ap_count: 0,
            stereo_defined: false,
            phone_defined: false,
            updated_at_ms: None,
        },
        |geometry| EvolutionSceneGeometrySummary {
            coordinate_frame_version: Some(geometry.coordinate_frame_version.clone()),
            ap_count: geometry.ap_nodes.len(),
            stereo_defined: true,
            phone_defined: geometry.phone_pose.is_some(),
            updated_at_ms: Some(geometry.updated_at_ms),
        },
    );

    EvolutionSceneGeometryResponse {
        scene_id,
        scene_name,
        exists: geometry.is_some(),
        summary,
        geometry,
    }
}

fn build_zero_shot_validation_artifact(
    session: &EvolutionSessionState,
    geometry_response: &EvolutionSceneGeometryResponse,
    readiness: &EvolutionReadiness,
    model: &EvolutionModelSummary,
    validation_id: String,
    validated_at_ms: u64,
    warmup_secs: u64,
    validation_secs: u64,
) -> EvolutionZeroShotValidationArtifact {
    let mut blockers = Vec::new();
    if !geometry_response.exists {
        blockers.push("当前 scene 还没有保存 geometry。".to_string());
    }
    if !model.loaded {
        blockers.push("当前还没有加载 live model。".to_string());
    }
    if !readiness.csi_ready {
        blockers.push("CSI 还没稳定，zero-shot warmup 还不能开始。".to_string());
    }
    if !readiness.stereo_ready {
        blockers.push("双目还没锁定操作者，暂时无法做短验证片段。".to_string());
    }
    if !readiness.phone_ready {
        blockers.push("手机 shared/world-frame device_pose 还没稳定在线，完整数据集预检暂时不能开始。".to_string());
    }
    if !readiness.robot_ready {
        blockers.push("机器人状态链路还没就绪，完整数据集预检暂时不能开始。".to_string());
    }

    let status = if blockers.is_empty() {
        "preflight_ready"
    } else {
        "blocked"
    };
    let message = if blockers.is_empty() {
        "scene geometry、live model 和采集前条件已齐。当前会先落盘 zero-shot 预检结果；如果再提供 validation recordings，就能比较 plain base 与 geometry-conditioned profile 路径。"
            .to_string()
    } else {
        blockers
            .first()
            .cloned()
            .unwrap_or_else(|| "当前无法启动 zero-shot 预检。".to_string())
    };

    EvolutionZeroShotValidationArtifact {
        validation_id,
        scene_id: session.scene_id.clone(),
        scene_name: session.scene_name.clone(),
        validated_at_ms,
        status: status.to_string(),
        message,
        adaptation_mode: "geometry-conditioned".to_string(),
        validation_mode: "preflight_only".to_string(),
        warmup_secs,
        validation_secs,
        geometry_ready: geometry_response.exists,
        model_ready: model.loaded,
        readiness_ready: readiness.full_multimodal_ready,
        live_model_id: model.model_id.clone(),
        base_model_id: model
            .base_model_id
            .clone()
            .or(model.global_parent_model_id.clone()),
        geometry_summary: geometry_response.summary.clone(),
        blockers,
        compare_status: if status == "blocked" {
            "blocked".to_string()
        } else {
            "not_wired".to_string()
        },
        compare_summary: None,
        promotion_gate_status: "preview_only".to_string(),
        promotion_gate_reason: Some(
            "当前 zero-shot 先做 shadow validation；只有 compare 通过后才允许人工 apply。"
                .to_string(),
        ),
        applied_at_ms: None,
        rolled_back_at_ms: None,
        auto_policy_status: "manual_only".to_string(),
        auto_policy_reason: Some("当前 zero-shot 默认保持人工 gate。".to_string()),
        auto_apply_min_improvement_mm: None,
        auto_applied_at_ms: None,
        auto_rolled_back_at_ms: None,
    }
}

fn zero_shot_compare_passed(summary: Option<&Value>) -> Option<bool> {
    summary
        .and_then(|value| value.get("passed"))
        .and_then(Value::as_bool)
}

fn zero_shot_compare_improvement_delta(summary: Option<&Value>) -> Option<f64> {
    summary
        .and_then(|value| value.get("improvement_delta"))
        .and_then(Value::as_f64)
}

fn zero_shot_candidate_profile_from_summary(summary: Option<&Value>) -> Option<String> {
    summary
        .and_then(|value| value.get("candidate_sona_profile"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
        .map(ToOwned::to_owned)
}

fn zero_shot_compare_has_complete_metrics(summary: Option<&Value>) -> bool {
    summary
        .and_then(|value| value.get("base_metric"))
        .and_then(Value::as_f64)
        .is_some()
        && summary
            .and_then(|value| value.get("geometry_conditioned_metric"))
            .and_then(Value::as_f64)
            .is_some()
        && summary
            .and_then(|value| value.get("improvement_delta"))
            .and_then(Value::as_f64)
            .is_some()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ZeroShotAutoPolicyAction {
    Apply(String),
    Rollback(String),
}

#[derive(Debug, Clone)]
struct ZeroShotAutoPolicyDecision {
    status: &'static str,
    reason: String,
    action: Option<ZeroShotAutoPolicyAction>,
}

fn decide_zero_shot_auto_policy(
    report: &EvolutionZeroShotValidationArtifact,
    model: &EvolutionModelSummary,
    auto_apply_enabled: bool,
    auto_rollback_enabled: bool,
    min_improvement_mm: f64,
) -> ZeroShotAutoPolicyDecision {
    let candidate_profile =
        zero_shot_candidate_profile_from_summary(report.compare_summary.as_ref());
    let compare_passed = zero_shot_compare_passed(report.compare_summary.as_ref());
    let improvement_delta = zero_shot_compare_improvement_delta(report.compare_summary.as_ref());
    let active_matches = model.active_sona_profile.as_deref() == candidate_profile.as_deref();

    if !auto_apply_enabled && !auto_rollback_enabled {
        return ZeroShotAutoPolicyDecision {
            status: "disabled",
            reason: "当前 zero-shot 仍采用手动 gate；自动升配和自动回滚都未启用。".to_string(),
            action: None,
        };
    }
    let Some(candidate_profile) = candidate_profile else {
        return ZeroShotAutoPolicyDecision {
            status: "skipped_profile_missing",
            reason: "当前 compare 没有关联到可操作的 geometry-conditioned SONA profile。"
                .to_string(),
            action: None,
        };
    };
    if report.compare_status != "measured"
        || !zero_shot_compare_has_complete_metrics(report.compare_summary.as_ref())
    {
        return ZeroShotAutoPolicyDecision {
            status: "skipped_compare_incomplete",
            reason: "zero-shot compare 还没形成完整 measured 指标，暂时不触发自动策略。"
                .to_string(),
            action: None,
        };
    }
    if active_matches && auto_rollback_enabled && compare_passed == Some(false) {
        return ZeroShotAutoPolicyDecision {
            status: "rollback_pending",
            reason: format!(
                "当前 live 已激活 `{candidate_profile}`，但最新 zero-shot compare 回归了 {} mm，准备自动回滚。",
                improvement_delta
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string())
            ),
            action: Some(ZeroShotAutoPolicyAction::Rollback(candidate_profile)),
        };
    }
    if active_matches && compare_passed == Some(true) {
        return ZeroShotAutoPolicyDecision {
            status: "already_promoted",
            reason: format!(
                "当前 live 已经激活 geometry-conditioned profile `{candidate_profile}`。"
            ),
            action: None,
        };
    }
    if report.promotion_gate_status != "eligible_for_apply"
        && report.promotion_gate_status != "promoted"
    {
        return ZeroShotAutoPolicyDecision {
            status: "skipped_gate_blocked",
            reason: report
                .promotion_gate_reason
                .clone()
                .unwrap_or_else(|| "zero-shot candidate 还没有达到自动升配门槛。".to_string()),
            action: None,
        };
    }
    if !auto_apply_enabled {
        return ZeroShotAutoPolicyDecision {
            status: "manual_apply_only",
            reason: "自动回滚已启用，但自动升配仍关闭；继续保持人工 apply。".to_string(),
            action: None,
        };
    }
    let Some(improvement_delta) = improvement_delta else {
        return ZeroShotAutoPolicyDecision {
            status: "skipped_compare_incomplete",
            reason: "zero-shot compare 缺少 improvement delta，暂时不触发自动升配。".to_string(),
            action: None,
        };
    };
    if improvement_delta < min_improvement_mm {
        return ZeroShotAutoPolicyDecision {
            status: "skipped_below_threshold",
            reason: format!(
                "当前 delta={improvement_delta:.3} mm，低于自动升配阈值 {min_improvement_mm:.3} mm。"
            ),
            action: None,
        };
    }
    ZeroShotAutoPolicyDecision {
        status: "apply_pending",
        reason: format!(
            "当前 delta={improvement_delta:.3} mm，已达到自动升配阈值 {min_improvement_mm:.3} mm，准备自动应用 `{candidate_profile}`。"
        ),
        action: Some(ZeroShotAutoPolicyAction::Apply(candidate_profile)),
    }
}

fn zero_shot_live_lineage_matches_validation(
    report: &EvolutionZeroShotValidationArtifact,
    model: &EvolutionModelSummary,
) -> bool {
    let lineage = [
        model.model_id.as_deref(),
        model.base_model_id.as_deref(),
        model.global_parent_model_id.as_deref(),
    ];
    let report_ids = [
        report
            .compare_summary
            .as_ref()
            .and_then(|value| value.get("candidate_model_id"))
            .and_then(Value::as_str),
        report
            .compare_summary
            .as_ref()
            .and_then(|value| value.get("base_model_id"))
            .and_then(Value::as_str),
        report.base_model_id.as_deref(),
        report.live_model_id.as_deref(),
    ];
    report_ids
        .iter()
        .flatten()
        .any(|candidate| lineage.iter().flatten().any(|live| live == candidate))
}

fn sync_zero_shot_promotion_gate(
    report: &mut EvolutionZeroShotValidationArtifact,
    model: &EvolutionModelSummary,
) -> bool {
    let candidate_profile =
        zero_shot_candidate_profile_from_summary(report.compare_summary.as_ref());
    let compare_passed = zero_shot_compare_passed(report.compare_summary.as_ref());
    let metrics_complete = zero_shot_compare_has_complete_metrics(report.compare_summary.as_ref());

    let (next_status, next_reason) = if report.compare_status != "measured" {
        (
            "preview_only",
            "当前还没有 measured zero-shot compare，先停留在 shadow validation。".to_string(),
        )
    } else if !metrics_complete {
        (
            "blocked_incomplete_compare",
            "zero-shot compare 已执行，但指标还不完整，暂时不能升配。".to_string(),
        )
    } else if candidate_profile.is_none() {
        (
            "blocked_profile_missing",
            "当前 compare 没有关联到可应用的 geometry-conditioned SONA profile。".to_string(),
        )
    } else if !model.loaded {
        (
            "blocked_model_unloaded",
            "当前还没有加载 live model，暂时不能应用 zero-shot profile。".to_string(),
        )
    } else if !zero_shot_live_lineage_matches_validation(report, model) {
        (
            "blocked_model_changed",
            "当前 live model 已经切到另一条谱系，不再是这次 zero-shot compare 的基线。".to_string(),
        )
    } else if !model
        .sona_profiles
        .iter()
        .any(|profile| Some(profile.as_str()) == candidate_profile.as_deref())
    {
        (
            "blocked_profile_missing",
            format!(
                "当前 live RVF 中找不到 profile `{}`，请先导入或重新加载对应 profiled RVF。",
                candidate_profile.as_deref().unwrap_or("-")
            ),
        )
    } else if compare_passed != Some(true) {
        (
            "blocked_regressed",
            "geometry-conditioned path 还没有在 shadow compare 中稳定优于 base。".to_string(),
        )
    } else if model.active_sona_profile.as_deref() == candidate_profile.as_deref() {
        (
            "promoted",
            format!(
                "当前 live 已激活 geometry-conditioned SONA profile `{}`。",
                candidate_profile.as_deref().unwrap_or("-")
            ),
        )
    } else {
        (
            "eligible_for_apply",
            format!(
                "geometry-conditioned SONA profile `{}` 已在同批 recordings 上优于 base，可人工 apply。",
                candidate_profile.as_deref().unwrap_or("-")
            ),
        )
    };

    let mut changed = false;
    if report.promotion_gate_status != next_status {
        report.promotion_gate_status = next_status.to_string();
        changed = true;
    }
    if report.promotion_gate_reason.as_deref() != Some(next_reason.as_str()) {
        report.promotion_gate_reason = Some(next_reason);
        changed = true;
    }

    let next_report_status = if next_status == "promoted" {
        Some("promoted")
    } else if report.compare_status == "measured" && compare_passed == Some(true) {
        Some("shadow_validated")
    } else if report.compare_status == "measured" && metrics_complete {
        Some("shadow_regressed")
    } else {
        None
    };
    if let Some(next_report_status) = next_report_status {
        if report.status != next_report_status {
            report.status = next_report_status.to_string();
            changed = true;
        }
    }

    changed
}

fn finalize_zero_shot_validation_with_compare(
    validation: &mut EvolutionZeroShotValidationArtifact,
    compare_summary: Value,
) {
    let passed = compare_summary
        .get("passed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let base_metric = compare_summary.get("base_metric").and_then(Value::as_f64);
    let candidate_metric = compare_summary
        .get("geometry_conditioned_metric")
        .and_then(Value::as_f64);
    let improvement_delta = compare_summary
        .get("improvement_delta")
        .and_then(Value::as_f64);

    validation.validation_mode = "recording_compare".to_string();
    validation.compare_status = "measured".to_string();
    validation.compare_summary = Some(compare_summary);
    validation.status = if passed {
        "shadow_validated".to_string()
    } else {
        "shadow_regressed".to_string()
    };
    validation.message = if let (Some(base), Some(candidate), Some(delta)) =
        (base_metric, candidate_metric, improvement_delta)
    {
        if passed {
            format!(
                "geometry-conditioned path 在同一批 validation recordings 上优于 base：{base:.2} -> {candidate:.2} mm，提升 {delta:.2} mm。"
            )
        } else {
            format!(
                "geometry-conditioned path 在同一批 validation recordings 上没有优于 base：{base:.2} -> {candidate:.2} mm，变化 {delta:.2} mm。"
            )
        }
    } else {
        "zero-shot compare 已执行，但返回的指标不完整。".to_string()
    };
}

async fn activate_sona_profile(
    state: &AppState,
    profile: &str,
    fallback_message: &str,
) -> Result<(), (StatusCode, String)> {
    let response = sensing_post_json(
        state,
        "/api/v1/model/sona/activate",
        json!({ "profile": profile }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or(fallback_message)
                .to_string(),
        ));
    }

    Ok(())
}

async fn maybe_run_zero_shot_auto_policy(
    state: &AppState,
    session: &mut EvolutionSessionState,
    report: &mut EvolutionZeroShotValidationArtifact,
    auto_apply_enabled: bool,
    auto_rollback_enabled: bool,
    min_improvement_mm: f64,
) -> Result<bool, (StatusCode, String)> {
    report.auto_apply_min_improvement_mm = Some(min_improvement_mm);

    let training = sensing_get_json::<ExternalTrainingStatus>(state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        report.auto_policy_status = "skipped_training_active".to_string();
        report.auto_policy_reason =
            Some("当前训练还在进行中，自动 zero-shot 策略先不触发。".to_string());
        return Ok(false);
    }

    let model = load_runtime_model_summary(state).await?;
    sync_zero_shot_promotion_gate(report, &model);
    let decision = decide_zero_shot_auto_policy(
        report,
        &model,
        auto_apply_enabled,
        auto_rollback_enabled,
        min_improvement_mm,
    );
    report.auto_policy_status = decision.status.to_string();
    report.auto_policy_reason = Some(decision.reason);

    match decision.action {
        Some(ZeroShotAutoPolicyAction::Apply(candidate_profile)) => {
            activate_sona_profile(
                state,
                &candidate_profile,
                "自动应用 zero-shot geometry profile 失败。",
            )
            .await?;
            let applied_at_ms = now_ms();
            report.applied_at_ms = Some(applied_at_ms);
            report.rolled_back_at_ms = None;
            report.auto_applied_at_ms = Some(applied_at_ms);
            report.status = "promoted".to_string();
            session.applied_at_ms = Some(applied_at_ms);
            session.status = "applied".to_string();
            session.updated_at_ms = applied_at_ms;
            let refreshed_model = load_runtime_model_summary(state).await?;
            sync_zero_shot_promotion_gate(report, &refreshed_model);
            report.auto_policy_status = "auto_applied".to_string();
            report.auto_policy_reason = Some(format!(
                "自动应用 geometry-conditioned profile `{candidate_profile}` 成功。"
            ));
            Ok(true)
        }
        Some(ZeroShotAutoPolicyAction::Rollback(candidate_profile)) => {
            activate_sona_profile(
                state,
                "default",
                "自动回滚 zero-shot geometry profile 失败。",
            )
            .await?;
            let rolled_back_at_ms = now_ms();
            report.rolled_back_at_ms = Some(rolled_back_at_ms);
            report.auto_rolled_back_at_ms = Some(rolled_back_at_ms);
            session.updated_at_ms = rolled_back_at_ms;
            let refreshed_model = load_runtime_model_summary(state).await?;
            sync_zero_shot_promotion_gate(report, &refreshed_model);
            report.auto_policy_status = "auto_rolled_back".to_string();
            report.auto_policy_reason = Some(format!(
                "自动回滚 geometry-conditioned profile `{candidate_profile}` 成功。"
            ));
            Ok(true)
        }
        None => Ok(false),
    }
}

fn validate_geometry_anchor(
    anchor: &EvolutionSceneGeometryAnchor,
    kind_label: &str,
) -> Result<(), String> {
    if anchor.id.trim().is_empty() {
        return Err(format!("{kind_label} 的 id 不能为空。"));
    }
    if anchor
        .position_m
        .iter()
        .chain(anchor.rotation_deg.iter())
        .any(|value| !value.is_finite())
    {
        return Err(format!("{kind_label} 的坐标或角度必须是有限数字。"));
    }
    Ok(())
}

fn validate_scene_geometry_request(
    req: &EvolutionSceneGeometryUpsertRequest,
) -> Result<(), String> {
    if req.coordinate_frame_version.trim().is_empty() {
        return Err("coordinate_frame_version 不能为空。".to_string());
    }
    if req.ap_nodes.is_empty() {
        return Err("至少需要提供 1 个 AP 节点。".to_string());
    }
    let mut seen_ids = std::collections::HashSet::new();
    for anchor in &req.ap_nodes {
        validate_geometry_anchor(anchor, "AP 节点")?;
        if !seen_ids.insert(anchor.id.trim().to_string()) {
            return Err(format!("AP 节点 id 重复：{}。", anchor.id));
        }
    }
    let stereo = req
        .stereo_rig
        .as_ref()
        .ok_or_else(|| "stereo_rig 不能为空。".to_string())?;
    validate_geometry_anchor(stereo, "双目位姿")?;
    if let Some(phone_pose) = &req.phone_pose {
        validate_geometry_anchor(phone_pose, "手机位姿")?;
    }
    Ok(())
}

async fn sensing_get_json<T: DeserializeOwned>(state: &AppState, path: &str) -> Result<T, String> {
    let url = format!(
        "{}{}",
        state.config.sensing_proxy_base.trim_end_matches('/'),
        path
    );
    let response = state
        .http_client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("读取 sensing 接口失败: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("sensing 返回 {}", response.status()));
    }
    response
        .json::<T>()
        .await
        .map_err(|error| format!("解析 sensing 响应失败: {error}"))
}

async fn sensing_post_json(state: &AppState, path: &str, body: Value) -> Result<Value, String> {
    let url = format!(
        "{}{}",
        state.config.sensing_proxy_base.trim_end_matches('/'),
        path
    );
    let response = state
        .http_client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|error| format!("调用 sensing 接口失败: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("sensing 返回 {}", response.status()));
    }
    response
        .json::<Value>()
        .await
        .map_err(|error| format!("解析 sensing 响应失败: {error}"))
}

fn count_valid_points(points: &[[f32; 3]]) -> usize {
    points
        .iter()
        .filter(|point| {
            point.iter().all(|value| value.is_finite())
                && point.iter().any(|value| value.abs() > 1e-6)
        })
        .count()
}

fn valid_point3(point: &[f32; 3]) -> bool {
    point.iter().all(|value| value.is_finite()) && point.iter().any(|value| value.abs() > 1e-6)
}

fn point3(points: &[[f32; 3]], index: usize) -> Option<[f32; 3]> {
    points.get(index).copied().filter(valid_point3)
}

fn midpoint3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

#[derive(Debug, Clone, Copy)]
struct GuidePoseMetrics {
    torso_height_m: f32,
    center_x_m: f32,
    center_uses_stereo: bool,
    shoulder_level_gap_m: f32,
    facing_delta_z_m: f32,
    signed_shoulder_delta_z_m: f32,
    left_wrist_above_shoulder_m: f32,
    right_wrist_above_shoulder_m: f32,
    left_wrist_side_offset_m: f32,
    right_wrist_side_offset_m: f32,
    left_wrist_forward_offset_m: f32,
    right_wrist_forward_offset_m: f32,
    forward_lean_m: f32,
    knee_drop_m: f32,
    left_ankle_raise_m: f32,
    right_ankle_raise_m: f32,
}

fn body_metrics_from_body(body: &[[f32; 3]]) -> Option<GuidePoseMetrics> {
    let nose = point3(body, COCO_NOSE)?;
    let left_shoulder = point3(body, COCO_LEFT_SHOULDER)?;
    let right_shoulder = point3(body, COCO_RIGHT_SHOULDER)?;
    let left_wrist = point3(body, COCO_LEFT_WRIST)?;
    let right_wrist = point3(body, COCO_RIGHT_WRIST)?;
    let left_hip = point3(body, COCO_LEFT_HIP)?;
    let right_hip = point3(body, COCO_RIGHT_HIP)?;
    let left_knee = point3(body, COCO_LEFT_KNEE)?;
    let right_knee = point3(body, COCO_RIGHT_KNEE)?;
    let left_ankle = point3(body, COCO_LEFT_ANKLE)?;
    let right_ankle = point3(body, COCO_RIGHT_ANKLE)?;

    let shoulder_center = midpoint3(left_shoulder, right_shoulder);
    let hip_center = midpoint3(left_hip, right_hip);
    let knee_center = midpoint3(left_knee, right_knee);
    let torso_height_m = shoulder_center[1] - hip_center[1];

    Some(GuidePoseMetrics {
        torso_height_m,
        center_x_m: (shoulder_center[0] + hip_center[0]) * 0.5,
        center_uses_stereo: false,
        shoulder_level_gap_m: (left_shoulder[1] - right_shoulder[1]).abs(),
        facing_delta_z_m: (left_shoulder[2] - right_shoulder[2]).abs(),
        signed_shoulder_delta_z_m: left_shoulder[2] - right_shoulder[2],
        left_wrist_above_shoulder_m: left_wrist[1] - left_shoulder[1],
        right_wrist_above_shoulder_m: right_wrist[1] - right_shoulder[1],
        left_wrist_side_offset_m: (left_wrist[0] - left_shoulder[0]).abs(),
        right_wrist_side_offset_m: (right_wrist[0] - right_shoulder[0]).abs(),
        left_wrist_forward_offset_m: left_wrist[2] - left_shoulder[2],
        right_wrist_forward_offset_m: right_wrist[2] - right_shoulder[2],
        forward_lean_m: nose[2] - hip_center[2],
        knee_drop_m: hip_center[1] - knee_center[1],
        left_ankle_raise_m: left_ankle[1] - right_ankle[1],
        right_ankle_raise_m: right_ankle[1] - left_ankle[1],
    })
}

fn body_metrics(operator: &OperatorSnapshot, stereo: &StereoSnapshot) -> Option<GuidePoseMetrics> {
    let stereo_body = canonical_body_points_3d(&stereo.body_kpts_3d, stereo.body_layout);
    if let Some(mut stereo_metrics) = body_metrics_from_body(&stereo_body) {
        stereo_metrics.center_uses_stereo = true;
        return Some(stereo_metrics);
    }
    body_metrics_from_body(&operator.estimate.operator_state.body_kpts_3d)
}

fn seed_runtime_baseline(runtime: &mut EvolutionStepRuntimeState, metrics: &GuidePoseMetrics) {
    runtime.baseline_left_wrist_side_offset_m = Some(metrics.left_wrist_side_offset_m);
    runtime.baseline_right_wrist_side_offset_m = Some(metrics.right_wrist_side_offset_m);
    runtime.baseline_left_wrist_forward_offset_m = Some(metrics.left_wrist_forward_offset_m);
    runtime.baseline_right_wrist_forward_offset_m = Some(metrics.right_wrist_forward_offset_m);
}

fn clear_runtime_tracking(runtime: &mut EvolutionStepRuntimeState) {
    runtime.phase_index = 0;
    runtime.max_phase_index = 0;
    runtime.phase_completed = false;
    runtime.last_advanced_at_ms = None;
    runtime.phase_hold_started_at_ms = None;
    runtime.lead_side = None;
    runtime.baseline_left_wrist_side_offset_m = None;
    runtime.baseline_right_wrist_side_offset_m = None;
    runtime.baseline_left_wrist_forward_offset_m = None;
    runtime.baseline_right_wrist_forward_offset_m = None;
    runtime.assessed_recording_id = None;
}

fn reach_extension_delta(
    side_offset_m: f32,
    forward_offset_m: f32,
    baseline_side_offset_m: Option<f32>,
    baseline_forward_offset_m: Option<f32>,
) -> f32 {
    let baseline_side = baseline_side_offset_m.unwrap_or(0.0);
    let baseline_forward = baseline_forward_offset_m.unwrap_or(0.0);
    let side_delta = (side_offset_m - baseline_side).max(0.0);
    let forward_delta = (forward_offset_m - baseline_forward).max(0.0);
    side_delta.max(forward_delta)
}

fn hands_reset_from_baseline(
    metrics: &GuidePoseMetrics,
    runtime: &EvolutionStepRuntimeState,
) -> bool {
    let left_delta = reach_extension_delta(
        metrics.left_wrist_side_offset_m,
        metrics.left_wrist_forward_offset_m,
        runtime.baseline_left_wrist_side_offset_m,
        runtime.baseline_left_wrist_forward_offset_m,
    );
    let right_delta = reach_extension_delta(
        metrics.right_wrist_side_offset_m,
        metrics.right_wrist_forward_offset_m,
        runtime.baseline_right_wrist_side_offset_m,
        runtime.baseline_right_wrist_forward_offset_m,
    );
    left_delta <= 0.04 && right_delta <= 0.04
}

fn left_reach_from_baseline(
    metrics: &GuidePoseMetrics,
    runtime: &EvolutionStepRuntimeState,
) -> bool {
    reach_extension_delta(
        metrics.left_wrist_side_offset_m,
        metrics.left_wrist_forward_offset_m,
        runtime.baseline_left_wrist_side_offset_m,
        runtime.baseline_left_wrist_forward_offset_m,
    ) >= 0.10
}

fn right_reach_from_baseline(
    metrics: &GuidePoseMetrics,
    runtime: &EvolutionStepRuntimeState,
) -> bool {
    reach_extension_delta(
        metrics.right_wrist_side_offset_m,
        metrics.right_wrist_forward_offset_m,
        runtime.baseline_right_wrist_side_offset_m,
        runtime.baseline_right_wrist_forward_offset_m,
    ) >= 0.10
}

fn center_basis_detail(center_uses_stereo: bool) -> &'static str {
    if center_uses_stereo {
        "这里看的是双目骨架中心，不是双目拼接中缝。"
    } else {
        "这里看的是融合骨架中心，不是双目拼接中缝。"
    }
}

fn center_direction_detail(
    center_x_m: f32,
    centered: bool,
    ideally_centered: bool,
    center_uses_stereo: bool,
) -> String {
    let basis = center_basis_detail(center_uses_stereo);
    if ideally_centered {
        return format!("位置已经对了。{}", basis);
    }

    let move_hint = if center_x_m > 0.0 {
        "再往左一点"
    } else {
        "再往右一点"
    };

    if centered {
        format!(
            "{}会更理想。左右画面都完整看到你时就能开始。{}",
            move_hint, basis
        )
    } else {
        format!("{}，先回到双目覆盖区。{}", move_hint, basis)
    }
}

fn build_step_guide(
    step: &EvolutionStepState,
    readiness: &EvolutionReadiness,
    operator: &OperatorSnapshot,
    stereo: &StereoSnapshot,
    runtime: Option<&EvolutionStepRuntimeState>,
) -> EvolutionStepGuide {
    let now = now_ms();
    let metrics = body_metrics(operator, stereo);
    let phase_labels = step_phase_labels(&step.code);
    let phase_total = phase_labels.len();
    let phase_index = runtime
        .map(|value| value.phase_index.min(phase_total))
        .unwrap_or(0);
    let phase_completed = runtime
        .map(|value| value.phase_completed || (phase_total > 0 && value.phase_index >= phase_total))
        .unwrap_or(false);
    let lead_side = runtime.and_then(|value| value.lead_side.as_deref());
    let phase_label = if phase_total > 0 {
        if phase_index < phase_total {
            runtime_phase_label(&step.code, phase_index, lead_side).map(|label| label.to_string())
        } else {
            runtime_phase_label(&step.code, phase_total.saturating_sub(1), lead_side)
                .or_else(|| phase_labels.last().copied())
                .map(|label| label.to_string())
        }
    } else {
        None
    };
    let phase_hold_target_ms = if phase_total > 0 && !phase_completed && step.status == "recording"
    {
        Some(phase_hold_target_ms(
            &step.code,
            phase_index.min(phase_total.saturating_sub(1)),
        ))
    } else {
        None
    };
    let phase_hold_ms = runtime
        .and_then(|value| value.phase_hold_started_at_ms)
        .and_then(|started_at_ms| {
            let elapsed = now.saturating_sub(started_at_ms);
            if elapsed > 0 {
                Some(elapsed)
            } else {
                None
            }
        });
    let phase_hold_progress = phase_hold_target_ms
        .map(|target_ms| (phase_hold_ms.unwrap_or(0) as f64 / target_ms as f64).clamp(0.0, 1.0));

    let build_check = |label: &str, ok: bool, detail: String| EvolutionGuideCheck {
        label: label.to_string(),
        ok,
        detail,
    };

    if step.status == "recorded" {
        return EvolutionStepGuide {
            step_code: step.code.clone(),
            ready: true,
            tone: "ok".to_string(),
            headline: "这一段已经录好，如需替换可以重新录一次。".to_string(),
            detail: "如果现场状态明显更稳，可以点击“重录这一段”覆盖旧样本。".to_string(),
            progress: 1.0,
            phase_index: phase_total,
            phase_total,
            phase_label,
            phase_detail: Some("这一段已经采完，不需要再跟阶段提示。".to_string()),
            phase_completed: true,
            phase_hold_ms: None,
            phase_hold_target_ms: None,
            phase_hold_progress: None,
            checks: vec![build_check(
                "录制结果",
                true,
                "当前步骤已经有可用录制。".to_string(),
            )],
        };
    }

    let mut checks = Vec::new();
    let headline;
    let detail = match step.code.as_str() {
        "empty_room_01" => {
            "请人工确认现场已清空；系统不把 CSI 人数结果当门槛。开始后原地等待 20 秒。".to_string()
        }
        "empty_room_02" => {
            "继续保持空房间；是否清空请以现场人工确认为准，完成第二段 20 秒基线。".to_string()
        }
        "pose_idle_front_02" => {
            "开始后保持正面站立和自然呼吸，不要突然摆手或离开双目中心。".to_string()
        }
        "pose_turn_lr_01" => {
            "开始后按“左转 90° → 回中 → 右转 90° → 回中”的顺序缓慢完成。".to_string()
        }
        "pose_reach_lr_01" => "开始后左右手交替平举或前伸，动作尽量完整，不要急促。".to_string(),
        "pose_arms_up_down_01" => "开始后双臂从身体两侧举到头顶，再自然放下。".to_string(),
        "pose_step_in_place_01" => "开始后保持身体在中心区域，小步幅原地踏步即可。".to_string(),
        "pose_bend_squat_01" => "开始后轻微下蹲或前屈，再平稳回到站立。".to_string(),
        _ => "按屏幕提示完成这一段动作。".to_string(),
    };
    let mut phase_detail = None;

    match step.code.as_str() {
        "empty_room_01" | "empty_room_02" => {
            checks.push(build_check(
                "CSI 在线",
                readiness.csi_ready,
                if readiness.csi_ready {
                    "节点收包稳定，可以采空房间基线。".to_string()
                } else {
                    "CSI 还没稳定，请先确认节点持续在线。".to_string()
                },
            ));
            checks.push(build_check(
                "人工确认",
                true,
                "是否清空请由现场确认；CSI 可能穿墙，不拿自动人数当门槛。".to_string(),
            ));

            headline = if !readiness.csi_ready {
                "先把 CSI 节点拉稳，再录空房间。".to_string()
            } else {
                "CSI 已稳定，可以开始录空房间基线。".to_string()
            };
        }
        _ => {
            let base_ready = readiness.pose_capture_ready;
            checks.push(build_check(
                "采集链路过线",
                base_ready,
                if base_ready {
                    format!(
                        "双目 / teacher / 手机 / 机器人主链都已过线，Wi‑Fi 先验也在线，anchor={}。",
                        readiness.anchor_source
                    )
                } else {
                    "请先让双目 / teacher / 手机 / 机器人主链稳定，并确认 Wi‑Fi 先验在线。".to_string()
                },
            ));
            checks.push(build_check(
                "手机姿态在线",
                readiness.phone_ready,
                if readiness.phone_ready {
                    "手机 shared/world-frame device_pose 已持续上报，可并入完整多模态会话。".to_string()
                } else {
                    "手机还没提供稳定的 shared/world-frame device_pose，暂时不满足完整数据集门槛。".to_string()
                },
            ));
            checks.push(build_check(
                "机器人状态在线",
                readiness.robot_ready,
                if readiness.robot_ready {
                    "机器人控制 / 状态链路在线，可以并入完整多模态会话。".to_string()
                } else {
                    "机器人桥接、时间同步、外参或 LAN 控制还没就绪。".to_string()
                },
            ));

            if let Some(metrics) = metrics {
                let centered = metrics.center_x_m.abs() <= GUIDE_CENTER_START_TOLERANCE_M;
                let ideally_centered = metrics.center_x_m.abs() <= GUIDE_CENTER_TARGET_TOLERANCE_M;
                let shoulders_level = metrics.shoulder_level_gap_m <= 0.05;
                let facing_front = metrics.facing_delta_z_m <= 0.08;
                let hands_lowered = metrics.left_wrist_above_shoulder_m <= 0.02
                    && metrics.right_wrist_above_shoulder_m <= 0.02;
                let hands_raised = metrics.left_wrist_above_shoulder_m >= 0.05
                    && metrics.right_wrist_above_shoulder_m >= 0.05;
                let left_reach = runtime
                    .map(|value| left_reach_from_baseline(&metrics, value))
                    .unwrap_or(false);
                let right_reach = runtime
                    .map(|value| right_reach_from_baseline(&metrics, value))
                    .unwrap_or(false);
                let hands_reset = runtime
                    .map(|value| hands_reset_from_baseline(&metrics, value))
                    .unwrap_or(hands_lowered);
                let lead_side = runtime.and_then(|value| value.lead_side.as_deref());
                let upright = metrics.torso_height_m >= 0.10;
                let feet_visible = metrics.knee_drop_m >= 0.14;
                let bend_depth =
                    metrics.forward_lean_m.abs() >= 0.05 || metrics.knee_drop_m <= 0.18;
                let center_detail = center_direction_detail(
                    metrics.center_x_m,
                    centered,
                    ideally_centered,
                    metrics.center_uses_stereo,
                );

                match step.code.as_str() {
                    "pose_idle_front_02" => {
                        checks.push(build_check(
                            "留在双目覆盖区",
                            centered,
                            center_detail.clone(),
                        ));
                        checks.push(build_check(
                            "身体正对双目",
                            facing_front && shoulders_level,
                            format!(
                                "左右肩高差 {:.3} m，前后差 {:.3} m。",
                                metrics.shoulder_level_gap_m, metrics.facing_delta_z_m
                            ),
                        ));
                        checks.push(build_check(
                            "双臂自然下垂",
                            hands_lowered,
                            format!(
                                "左/右手相对肩部高度 {:.3} / {:.3} m。",
                                metrics.left_wrist_above_shoulder_m,
                                metrics.right_wrist_above_shoulder_m
                            ),
                        ));

                        headline = if base_ready
                            && centered
                            && facing_front
                            && shoulders_level
                            && hands_lowered
                        {
                            "起始姿态已经到位，可以开始录这一段。".to_string()
                        } else if !base_ready {
                            "先让 Wi‑Fi、双目和 teacher 稳定下来。".to_string()
                        } else if !centered {
                            "先让左右两个画面都完整看到你，再开始。".to_string()
                        } else if !(facing_front && shoulders_level) {
                            "身体再正对一点双目，肩膀放平。".to_string()
                        } else {
                            "双臂先自然放到身体两侧，再开始。".to_string()
                        };
                        phase_detail = Some("这一段只需要稳定保持正面站姿。".to_string());
                    }
                    "pose_turn_lr_01" => {
                        phase_detail = Some(match phase_index {
                            0 => {
                                if lead_side == Some("right") {
                                    "录制开始后先缓慢向右转，直到肩膀前后差明显。".to_string()
                                } else {
                                    "录制开始后先缓慢向左转，直到肩膀前后差明显。".to_string()
                                }
                            }
                            1 => {
                                if lead_side == Some("right") {
                                    "右转到位后，请回到正面站姿。".to_string()
                                } else {
                                    "左转到位后，请回到正面站姿。".to_string()
                                }
                            }
                            2 => {
                                if lead_side == Some("right") {
                                    "回到正面后，再向左转到位。".to_string()
                                } else {
                                    "回到正面后，再向右转到位。".to_string()
                                }
                            }
                            _ => {
                                if lead_side == Some("right") {
                                    "左转到位后，再回到正面结束这一段。".to_string()
                                } else {
                                    "右转到位后，再回到正面结束这一段。".to_string()
                                }
                            }
                        });
                        checks.push(build_check(
                            "留在双目覆盖区",
                            centered,
                            center_detail.clone(),
                        ));
                        checks.push(build_check(
                            "回到正面起始位",
                            facing_front,
                            format!(
                                "左右肩前后差 {:.3} m，建议先回到正面。",
                                metrics.facing_delta_z_m
                            ),
                        ));
                        checks.push(build_check(
                            "双臂保持自然",
                            hands_lowered,
                            format!(
                                "左/右手相对肩部高度 {:.3} / {:.3} m。",
                                metrics.left_wrist_above_shoulder_m,
                                metrics.right_wrist_above_shoulder_m
                            ),
                        ));

                        headline = if step.status == "recording" {
                            "正在录制，请按“左转 → 回中 → 右转 → 回中”完整做完。".to_string()
                        } else if base_ready && centered && facing_front && hands_lowered {
                            "起始姿态已经到位，可以开始录左右转体。".to_string()
                        } else if !base_ready {
                            "先把主链路稳定下来，再开始转体。".to_string()
                        } else if !centered {
                            "先让左右两个画面都完整看到你，再开始转体。".to_string()
                        } else if !facing_front {
                            "先回到正面起始位，再开始转体。".to_string()
                        } else {
                            "双臂先自然放下，再开始转体。".to_string()
                        };
                    }
                    "pose_reach_lr_01" => {
                        phase_detail = Some(match phase_index {
                            0 => {
                                if lead_side == Some("right") {
                                    "录制开始后先做右手前伸或平举。".to_string()
                                } else {
                                    "录制开始后先做左手前伸或平举。".to_string()
                                }
                            }
                            1 => "这一侧完成后，双手先回到身体两侧。".to_string(),
                            2 => {
                                if lead_side == Some("right") {
                                    "接着做左手前伸或平举。".to_string()
                                } else {
                                    "接着做右手前伸或平举。".to_string()
                                }
                            }
                            _ => "另一侧动作完成后，再回到起始位结束这一段。".to_string(),
                        });
                        checks.push(build_check(
                            "留在双目覆盖区",
                            centered,
                            center_detail.clone(),
                        ));
                        checks.push(build_check(
                            "起始站姿稳定",
                            upright && facing_front,
                            format!(
                                "躯干高度 {:.3} m，左右肩前后差 {:.3} m。",
                                metrics.torso_height_m, metrics.facing_delta_z_m
                            ),
                        ));
                        checks.push(build_check(
                            "双手已回到起始位",
                            hands_reset,
                            if step.status == "recording" {
                                "双手回到起始位后，系统才会切到下一侧。".to_string()
                            } else {
                                format!(
                                    "左/右手相对肩部高度 {:.3} / {:.3} m。",
                                    metrics.left_wrist_above_shoulder_m,
                                    metrics.right_wrist_above_shoulder_m
                                )
                            },
                        ));
                        checks.push(build_check(
                            "当前伸展幅度可见",
                            step.status != "recording" || left_reach || right_reach,
                            if step.status == "recording" {
                                format!(
                                    "左/右手相对起始位的伸展增量 {:.3} / {:.3} m，录制时请交替伸展。",
                                    reach_extension_delta(
                                        metrics.left_wrist_side_offset_m,
                                        metrics.left_wrist_forward_offset_m,
                                        runtime.and_then(|value| value.baseline_left_wrist_side_offset_m),
                                        runtime.and_then(|value| value.baseline_left_wrist_forward_offset_m),
                                    ),
                                    reach_extension_delta(
                                        metrics.right_wrist_side_offset_m,
                                        metrics.right_wrist_forward_offset_m,
                                        runtime.and_then(|value| value.baseline_right_wrist_side_offset_m),
                                        runtime.and_then(|value| value.baseline_right_wrist_forward_offset_m),
                                    ),
                                )
                            } else {
                                "正式录制后系统会检查左右手是否有明显前伸。".to_string()
                            },
                        ));

                        headline = if step.status == "recording" {
                            "正在录制，请左右手交替完整伸展。".to_string()
                        } else if base_ready && centered && upright && facing_front && hands_lowered
                        {
                            "起始姿态已经到位，可以开始录左右前伸。".to_string()
                        } else if !base_ready {
                            "先让 Wi‑Fi、双目和 teacher 稳下来。".to_string()
                        } else if !centered {
                            "先让左右两个画面都完整看到你，再开始前伸。".to_string()
                        } else if !hands_lowered {
                            "双手先回到身体两侧，再开始前伸。".to_string()
                        } else {
                            "身体先站直并正对双目，再开始前伸。".to_string()
                        };
                    }
                    "pose_arms_up_down_01" => {
                        phase_detail = Some(match phase_index {
                            0 => "录制开始后把双臂从身体两侧举到头顶上方。".to_string(),
                            _ => "双臂举到位后，再自然放回身体两侧。".to_string(),
                        });
                        checks.push(build_check(
                            "留在双目覆盖区",
                            centered,
                            center_detail.clone(),
                        ));
                        checks.push(build_check(
                            "起始站姿稳定",
                            upright && facing_front,
                            format!(
                                "躯干高度 {:.3} m，左右肩前后差 {:.3} m。",
                                metrics.torso_height_m, metrics.facing_delta_z_m
                            ),
                        ));
                        checks.push(build_check(
                            "双臂已回到起始位",
                            hands_lowered,
                            format!(
                                "左/右手相对肩部高度 {:.3} / {:.3} m。",
                                metrics.left_wrist_above_shoulder_m,
                                metrics.right_wrist_above_shoulder_m
                            ),
                        ));
                        checks.push(build_check(
                            "抬手幅度可见",
                            step.status != "recording" || hands_raised,
                            if step.status == "recording" {
                                format!(
                                    "左/右手当前抬高 {:.3} / {:.3} m，高于肩膀时会更稳。",
                                    metrics.left_wrist_above_shoulder_m,
                                    metrics.right_wrist_above_shoulder_m
                                )
                            } else {
                                "正式录制后系统会检查双臂是否明显举过肩膀。".to_string()
                            },
                        ));

                        headline = if step.status == "recording" {
                            "正在录制，请完整做“上举 → 放下”这一组动作。".to_string()
                        } else if base_ready && centered && upright && facing_front && hands_lowered
                        {
                            "起始姿态已经到位，可以开始录双臂上举。".to_string()
                        } else if !base_ready {
                            "先把采集链路稳定下来，再开始抬手。".to_string()
                        } else if !centered {
                            "先让左右两个画面都完整看到你，再开始抬手。".to_string()
                        } else if !hands_lowered {
                            "双臂先自然放下，再开始这一段。".to_string()
                        } else {
                            "身体先站直并正对双目，再开始抬手。".to_string()
                        };
                    }
                    "pose_step_in_place_01" => {
                        phase_detail = Some(match phase_index {
                            0 => {
                                if lead_side == Some("right") {
                                    "录制开始后先抬右脚做一步小踏步。".to_string()
                                } else {
                                    "录制开始后先抬左脚做一步小踏步。".to_string()
                                }
                            }
                            _ => {
                                if lead_side == Some("right") {
                                    "右脚完成后，换左脚做一步小踏步。".to_string()
                                } else {
                                    "左脚完成后，换右脚做一步小踏步。".to_string()
                                }
                            }
                        });
                        checks.push(build_check(
                            "留在双目覆盖区",
                            centered,
                            center_detail.clone(),
                        ));
                        checks.push(build_check(
                            "身体站稳",
                            upright && facing_front,
                            format!(
                                "躯干高度 {:.3} m，左右肩前后差 {:.3} m。",
                                metrics.torso_height_m, metrics.facing_delta_z_m
                            ),
                        ));
                        checks.push(build_check(
                            "双脚都在可见范围",
                            feet_visible,
                            format!(
                                "髋到膝的可见高度 {:.3} m，建议双脚都留在画面里。",
                                metrics.knee_drop_m
                            ),
                        ));

                        headline = if step.status == "recording" {
                            "正在录制，请在原地小步踏步，不要偏离中心。".to_string()
                        } else if base_ready && centered && upright && facing_front && feet_visible
                        {
                            "起始姿态已经到位，可以开始录原地踏步。".to_string()
                        } else if !base_ready {
                            "先让主链路稳定，再录原地踏步。".to_string()
                        } else if !centered {
                            "先让左右两个画面都完整看到你，再开始踏步。".to_string()
                        } else if !feet_visible {
                            "请让双脚都留在双目画面里，再开始。".to_string()
                        } else {
                            "先站直并正对双目，再开始踏步。".to_string()
                        };
                    }
                    "pose_bend_squat_01" => {
                        phase_detail = Some(match phase_index {
                            0 => "录制开始后先做一次明显的轻微下蹲或前屈。".to_string(),
                            _ => "下蹲或前屈到位后，再回到正直站姿。".to_string(),
                        });
                        checks.push(build_check(
                            "留在双目覆盖区",
                            centered,
                            center_detail.clone(),
                        ));
                        checks.push(build_check(
                            "起始站姿稳定",
                            upright && facing_front,
                            format!(
                                "躯干高度 {:.3} m，左右肩前后差 {:.3} m。",
                                metrics.torso_height_m, metrics.facing_delta_z_m
                            ),
                        ));
                        checks.push(build_check(
                            "双臂自然下垂",
                            hands_lowered,
                            format!(
                                "左/右手相对肩部高度 {:.3} / {:.3} m。",
                                metrics.left_wrist_above_shoulder_m,
                                metrics.right_wrist_above_shoulder_m
                            ),
                        ));
                        checks.push(build_check(
                            "下蹲/前屈幅度可见",
                            step.status != "recording" || bend_depth,
                            if step.status == "recording" {
                                format!(
                                    "当前前屈/下蹲特征 {:.3} / {:.3}，录制时请明显弯一点再回正。",
                                    metrics.forward_lean_m, metrics.knee_drop_m
                                )
                            } else {
                                "正式录制后系统会检查下蹲或前屈幅度。".to_string()
                            },
                        ));

                        headline = if step.status == "recording" {
                            "正在录制，请做一次完整的轻微下蹲或前屈，再平稳回正。".to_string()
                        } else if base_ready && centered && upright && facing_front && hands_lowered
                        {
                            "起始姿态已经到位，可以开始录弯腰下蹲。".to_string()
                        } else if !base_ready {
                            "先让采集链路过线，再开始这一段。".to_string()
                        } else if !centered {
                            "先让左右两个画面都完整看到你，再开始。".to_string()
                        } else if !hands_lowered {
                            "双臂先自然放下，再开始这一段。".to_string()
                        } else {
                            "身体先站直并正对双目，再开始弯腰下蹲。".to_string()
                        };
                    }
                    _ => {
                        checks.push(build_check(
                            "采集链路过线",
                            base_ready,
                            if base_ready {
                                "当前已经可以开始采集。".to_string()
                            } else {
                                "请先让 Wi‑Fi、双目和 teacher 稳定。".to_string()
                            },
                        ));
                        headline = if base_ready {
                            "当前动作可以开始录制。".to_string()
                        } else {
                            "先把采集链路稳定下来。".to_string()
                        };
                    }
                }
            } else {
                checks.push(build_check(
                    "当前看到完整身体",
                    false,
                    "还没有拿到稳定的 17 点身体骨架，请先站到左右两个画面都能稳定看到你的区域。"
                        .to_string(),
                ));
                headline = "先让双目和融合骨架稳定显示整个人体，再开始。".to_string();
            }
        }
    }

    let passed = checks.iter().filter(|check| check.ok).count();
    let total = checks.len().max(1);
    let check_progress = passed as f64 / total as f64;
    let progress = if step.status == "recording" && phase_total > 0 {
        (phase_index.min(phase_total) as f64 / phase_total as f64).max(check_progress)
    } else {
        check_progress
    };
    let ready = checks.iter().all(|check| check.ok);
    let tone = if ready {
        "ok"
    } else if progress >= 0.5 {
        "warn"
    } else {
        "danger"
    };
    let headline = if step.status == "recording" && phase_total > 0 && !phase_completed {
        format!(
            "正在录制：当前阶段“{}”。",
            phase_label.as_deref().unwrap_or("继续按提示动作")
        )
    } else {
        headline
    };

    EvolutionStepGuide {
        step_code: step.code.clone(),
        ready,
        tone: tone.to_string(),
        headline,
        detail,
        progress,
        phase_index,
        phase_total,
        phase_label,
        phase_detail,
        phase_completed,
        phase_hold_ms,
        phase_hold_target_ms,
        phase_hold_progress,
        checks,
    }
}

fn build_step_guides(
    session: &EvolutionSessionState,
    readiness: &EvolutionReadiness,
    operator: &OperatorSnapshot,
    stereo: &StereoSnapshot,
) -> Vec<EvolutionStepGuide> {
    session
        .steps
        .iter()
        .map(|step| {
            build_step_guide(
                step,
                readiness,
                operator,
                stereo,
                step_runtime(session, &step.code),
            )
        })
        .collect()
}

fn phase_reached(
    step_code: &str,
    phase_index: usize,
    metrics: &GuidePoseMetrics,
    runtime: &mut EvolutionStepRuntimeState,
) -> bool {
    let hands_lowered =
        metrics.left_wrist_above_shoulder_m <= 0.02 && metrics.right_wrist_above_shoulder_m <= 0.02;

    match step_code {
        "pose_turn_lr_01" => {
            if runtime.lead_side.is_none() {
                if metrics.signed_shoulder_delta_z_m >= 0.035 {
                    runtime.lead_side = Some("left".to_string());
                } else if metrics.signed_shoulder_delta_z_m <= -0.035 {
                    runtime.lead_side = Some("right".to_string());
                }
            }
            let right_first = runtime.lead_side.as_deref() == Some("right");
            match phase_index {
                0 => {
                    if right_first {
                        metrics.signed_shoulder_delta_z_m <= -0.035
                    } else {
                        metrics.signed_shoulder_delta_z_m >= 0.035
                    }
                }
                1 => metrics.facing_delta_z_m <= 0.05,
                2 => {
                    if right_first {
                        metrics.signed_shoulder_delta_z_m >= 0.035
                    } else {
                        metrics.signed_shoulder_delta_z_m <= -0.035
                    }
                }
                3 => metrics.facing_delta_z_m <= 0.05,
                _ => true,
            }
        }
        "pose_reach_lr_01" => {
            if runtime.lead_side.is_none() {
                let left_reach = left_reach_from_baseline(metrics, runtime);
                let right_reach = right_reach_from_baseline(metrics, runtime);
                if left_reach && !right_reach {
                    runtime.lead_side = Some("left".to_string());
                } else if right_reach && !left_reach {
                    runtime.lead_side = Some("right".to_string());
                }
            }
            let right_first = runtime.lead_side.as_deref() == Some("right");
            match phase_index {
                0 => {
                    if right_first {
                        right_reach_from_baseline(metrics, runtime)
                    } else {
                        left_reach_from_baseline(metrics, runtime)
                    }
                }
                1 => hands_reset_from_baseline(metrics, runtime),
                2 => {
                    if right_first {
                        left_reach_from_baseline(metrics, runtime)
                    } else {
                        right_reach_from_baseline(metrics, runtime)
                    }
                }
                3 => hands_reset_from_baseline(metrics, runtime),
                _ => true,
            }
        }
        "pose_arms_up_down_01" => match phase_index {
            0 => {
                metrics.left_wrist_above_shoulder_m >= 0.05
                    && metrics.right_wrist_above_shoulder_m >= 0.05
            }
            1 => hands_lowered,
            _ => true,
        },
        "pose_step_in_place_01" => {
            if runtime.lead_side.is_none() {
                if metrics.left_ankle_raise_m >= 0.045 {
                    runtime.lead_side = Some("left".to_string());
                } else if metrics.right_ankle_raise_m >= 0.045 {
                    runtime.lead_side = Some("right".to_string());
                }
            }
            let right_first = runtime.lead_side.as_deref() == Some("right");
            match phase_index {
                0 => {
                    if right_first {
                        metrics.right_ankle_raise_m >= 0.045
                    } else {
                        metrics.left_ankle_raise_m >= 0.045
                    }
                }
                1 => {
                    if right_first {
                        metrics.left_ankle_raise_m >= 0.045
                    } else {
                        metrics.right_ankle_raise_m >= 0.045
                    }
                }
                _ => true,
            }
        }
        "pose_bend_squat_01" => match phase_index {
            0 => metrics.forward_lean_m.abs() >= 0.05 || metrics.knee_drop_m <= 0.17,
            1 => metrics.forward_lean_m.abs() <= 0.03 && metrics.torso_height_m >= 0.10,
            _ => true,
        },
        _ => false,
    }
}

#[derive(Debug, Deserialize)]
struct TeacherRecordingFrame {
    timestamp: Option<f64>,
    #[serde(default)]
    stereo_body_kpts_3d: Vec<[f32; 3]>,
    #[serde(default)]
    body_kpts_3d: Vec<[f32; 3]>,
    #[serde(default)]
    fused_body_kpts_3d: Vec<[f32; 3]>,
}

fn teacher_recordings_root() -> PathBuf {
    if let Ok(value) = std::env::var("EDGE_EVOLUTION_RECORDINGS_DIR") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../RuView/rust-port/wifi-densepose-rs/data/recordings")
}

fn teacher_recording_path(recording_id: &str) -> PathBuf {
    teacher_recordings_root().join(format!("{recording_id}.teacher.jsonl"))
}

fn load_teacher_metrics_frames(path: &Path) -> Option<Vec<(u64, GuidePoseMetrics)>> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut frames = Vec::new();

    for line in reader.lines() {
        let Ok(line) = line else {
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(frame) = serde_json::from_str::<TeacherRecordingFrame>(&line) else {
            continue;
        };
        let Some(timestamp) = frame.timestamp else {
            continue;
        };
        let body = if !frame.stereo_body_kpts_3d.is_empty() {
            &frame.stereo_body_kpts_3d
        } else if !frame.body_kpts_3d.is_empty() {
            &frame.body_kpts_3d
        } else {
            &frame.fused_body_kpts_3d
        };
        let Some(metrics) = body_metrics_from_body(body) else {
            continue;
        };
        frames.push((((timestamp.max(0.0)) * 1000.0).round() as u64, metrics));
    }

    if frames.is_empty() {
        None
    } else {
        Some(frames)
    }
}

fn analyze_recorded_phase_progress(
    step_code: &str,
    recording_id: &str,
) -> Option<EvolutionStepRuntimeState> {
    let frames = load_teacher_metrics_frames(&teacher_recording_path(recording_id))?;
    let phase_total = step_phase_labels(step_code).len();
    let first_metrics = frames.first().map(|(_, metrics)| *metrics)?;
    let mut runtime = EvolutionStepRuntimeState {
        step_code: step_code.to_string(),
        phase_total,
        ..EvolutionStepRuntimeState::default()
    };
    seed_runtime_baseline(&mut runtime, &first_metrics);

    for (timestamp_ms, metrics) in frames {
        if runtime.phase_total == 0 || runtime.phase_completed {
            break;
        }
        if runtime.phase_index >= runtime.phase_total {
            runtime.phase_completed = true;
            break;
        }
        let can_advance = runtime
            .last_advanced_at_ms
            .map(|value| timestamp_ms.saturating_sub(value) >= GUIDE_PHASE_ADVANCE_COOLDOWN_MS)
            .unwrap_or(true);
        if !can_advance {
            continue;
        }
        if phase_reached(step_code, runtime.phase_index, &metrics, &mut runtime) {
            if runtime.phase_hold_started_at_ms.is_none() {
                runtime.phase_hold_started_at_ms = Some(timestamp_ms);
                continue;
            }
            let hold_target_ms = phase_hold_target_ms(
                step_code,
                runtime
                    .phase_index
                    .min(runtime.phase_total.saturating_sub(1)),
            );
            let held_long_enough = runtime
                .phase_hold_started_at_ms
                .map(|started_at_ms| timestamp_ms.saturating_sub(started_at_ms) >= hold_target_ms)
                .unwrap_or(false);
            if held_long_enough {
                runtime.phase_index += 1;
                runtime.max_phase_index = runtime.max_phase_index.max(runtime.phase_index);
                runtime.last_advanced_at_ms = Some(timestamp_ms);
                runtime.phase_hold_started_at_ms = None;
                if runtime.phase_index >= runtime.phase_total {
                    runtime.phase_completed = true;
                }
            }
        } else {
            runtime.phase_hold_started_at_ms = None;
        }
    }

    runtime.phase_index = runtime.max_phase_index.min(runtime.phase_total);
    runtime.phase_completed = runtime.max_phase_index >= runtime.phase_total;
    runtime.phase_hold_started_at_ms = None;
    runtime.assessed_recording_id = Some(recording_id.to_string());
    Some(runtime)
}

fn sync_recorded_step_runtimes_from_teacher_files(session: &mut EvolutionSessionState) {
    sync_step_runtimes(session);

    let recorded_steps: Vec<(String, String)> = session
        .steps
        .iter()
        .filter(|step| step.status == "recorded")
        .filter_map(|step| {
            step.recording_id
                .as_ref()
                .map(|recording_id| (step.code.clone(), recording_id.clone()))
        })
        .collect();

    for (step_code, recording_id) in recorded_steps {
        let Some(runtime) = step_runtime_mut(session, &step_code) else {
            continue;
        };
        if runtime.assessed_recording_id.as_deref() == Some(recording_id.as_str()) {
            continue;
        }
        if let Some(analyzed) = analyze_recorded_phase_progress(&step_code, &recording_id) {
            runtime.phase_total = analyzed.phase_total;
            runtime.phase_index = analyzed.phase_index;
            runtime.max_phase_index = analyzed.max_phase_index;
            runtime.phase_completed = analyzed.phase_completed;
            runtime.lead_side = analyzed.lead_side;
            runtime.assessed_recording_id = analyzed.assessed_recording_id;
            runtime.phase_hold_started_at_ms = None;
            runtime.last_advanced_at_ms = None;
        }
    }
}

fn assess_recorded_step(
    step: &EvolutionStepState,
    runtime: Option<&EvolutionStepRuntimeState>,
) -> (Option<f64>, String, Option<String>, bool) {
    let Some(frame_count) = step.frame_count else {
        return (
            None,
            "danger".to_string(),
            Some("这一段还没有有效帧，建议重录。".to_string()),
            true,
        );
    };

    let target_frames = (step.duration_secs as f64 * 8.0).max(1.0);
    let frame_ratio = (frame_count as f64 / target_frames).clamp(0.0, 1.0);
    let source_quality = step.quality_score.unwrap_or(0.0).clamp(0.0, 1.0);
    let (phase_ratio, phase_detail) = if step.label == "pose" {
        let phase_total = runtime.map(|value| value.phase_total).unwrap_or(0);
        let phase_done = runtime
            .map(|value| value.max_phase_index.max(value.phase_index))
            .unwrap_or(0);
        if phase_total == 0 {
            (1.0, None)
        } else {
            let ratio = (phase_done as f64 / phase_total as f64).clamp(0.0, 1.0);
            (
                ratio,
                Some(format!(
                    "动作阶段完成 {}/{}。",
                    phase_done.min(phase_total),
                    phase_total
                )),
            )
        }
    } else {
        (1.0, None)
    };

    let score = if step.label == "empty" {
        (source_quality * 0.58 + frame_ratio * 0.42).clamp(0.0, 1.0)
    } else {
        (source_quality * 0.42 + frame_ratio * 0.28 + phase_ratio * 0.30).clamp(0.0, 1.0)
    };

    let mut reasons = Vec::new();
    if frame_ratio < 0.75 {
        reasons.push(format!("有效帧偏少（{frame_count} 帧）。"));
    }
    if source_quality < 0.72 {
        reasons.push(format!(
            "采集起始质量偏低（{:.0}%）。",
            source_quality * 100.0
        ));
    }
    if let Some(phase_detail) = phase_detail {
        if phase_ratio < 0.95 {
            reasons.push(phase_detail);
        }
    }

    let (tone, summary, rerecord_recommended) =
        if step.label == "pose" && (phase_ratio < 0.95 || frame_ratio < 0.72 || score < 0.68) {
            (
                "danger".to_string(),
                Some(if reasons.is_empty() {
                    "这一段动作完整度不足，建议重录。".to_string()
                } else {
                    format!("建议重录：{}", reasons.join(" "))
                }),
                true,
            )
        } else if score < 0.82 {
            (
                "warn".to_string(),
                Some(if reasons.is_empty() {
                    "这一段可以先用于训练，但更建议现场更稳时补录一段。".to_string()
                } else {
                    format!("这一段可训练，但建议补录：{}", reasons.join(" "))
                }),
                false,
            )
        } else {
            (
                "ok".to_string(),
                Some("这一段质量稳定，可以直接进入训练。".to_string()),
                false,
            )
        };

    (Some(score), tone, summary, rerecord_recommended)
}

fn phase_tracking_ready(readiness: &EvolutionReadiness) -> bool {
    readiness.stereo_ready && readiness.teacher_ready
}

fn update_step_runtimes(
    session: &mut EvolutionSessionState,
    recording: &EvolutionRecordingSummary,
    readiness: &EvolutionReadiness,
    operator: &OperatorSnapshot,
    stereo: &StereoSnapshot,
) {
    sync_step_runtimes(session);
    let now = now_ms();
    let step_states: Vec<(String, String)> = session
        .steps
        .iter()
        .map(|step| (step.code.clone(), step.status.clone()))
        .collect();

    for (step_code, step_status) in step_states {
        let Some(runtime) = step_runtime_mut(session, &step_code) else {
            continue;
        };

        runtime.phase_total = step_phase_labels(&step_code).len();

        if step_status == "recorded" {
            runtime.phase_index = runtime.max_phase_index.min(runtime.phase_total);
            runtime.phase_completed = runtime.max_phase_index >= runtime.phase_total;
            runtime.phase_hold_started_at_ms = None;
            continue;
        }

        if !recording.active || recording.step_code.as_deref() != Some(step_code.as_str()) {
            if step_status != "recording" {
                clear_runtime_tracking(runtime);
            }
            continue;
        }

        if runtime.phase_total == 0 || runtime.phase_completed || !phase_tracking_ready(readiness) {
            continue;
        }

        let Some(metrics) = body_metrics(operator, stereo) else {
            continue;
        };

        if runtime.baseline_left_wrist_side_offset_m.is_none()
            || runtime.baseline_right_wrist_side_offset_m.is_none()
            || runtime.baseline_left_wrist_forward_offset_m.is_none()
            || runtime.baseline_right_wrist_forward_offset_m.is_none()
        {
            seed_runtime_baseline(runtime, &metrics);
        }

        if runtime.phase_index >= runtime.phase_total {
            runtime.phase_completed = true;
            continue;
        }

        let can_advance = runtime
            .last_advanced_at_ms
            .map(|value| now.saturating_sub(value) >= GUIDE_PHASE_ADVANCE_COOLDOWN_MS)
            .unwrap_or(true);
        if !can_advance {
            continue;
        }

        if phase_reached(&step_code, runtime.phase_index, &metrics, runtime) {
            if runtime.phase_hold_started_at_ms.is_none() {
                runtime.phase_hold_started_at_ms = Some(now);
                continue;
            }

            let hold_target_ms = phase_hold_target_ms(
                &step_code,
                runtime
                    .phase_index
                    .min(runtime.phase_total.saturating_sub(1)),
            );
            let held_long_enough = runtime
                .phase_hold_started_at_ms
                .map(|started_at_ms| now.saturating_sub(started_at_ms) >= hold_target_ms)
                .unwrap_or(false);
            if held_long_enough {
                runtime.phase_index += 1;
                runtime.max_phase_index = runtime.max_phase_index.max(runtime.phase_index);
                runtime.last_advanced_at_ms = Some(now);
                runtime.phase_hold_started_at_ms = None;
                if runtime.phase_index >= runtime.phase_total {
                    runtime.phase_completed = true;
                }
            }
        } else {
            runtime.phase_hold_started_at_ms = None;
        }
    }

    let runtime_snapshots: Vec<(String, EvolutionStepRuntimeState)> = session
        .step_runtimes
        .iter()
        .cloned()
        .map(|runtime| (runtime.step_code.clone(), runtime))
        .collect();

    for step in &mut session.steps {
        let runtime = runtime_snapshots
            .iter()
            .find(|(step_code, _)| step_code == &step.code)
            .map(|(_, runtime)| runtime);
        let (assessed_score, assessed_tone, assessed_summary, rerecord_recommended) =
            if step.status == "recorded" {
                assess_recorded_step(step, runtime)
            } else {
                (None, String::new(), None, false)
            };
        step.assessed_score = assessed_score;
        step.assessed_tone = assessed_tone;
        step.assessed_summary = assessed_summary;
        step.rerecord_recommended = rerecord_recommended;
    }
}

fn teacher_ready(operator: &OperatorSnapshot, stereo: &StereoSnapshot) -> bool {
    let fused_valid = count_valid_points(&operator.estimate.operator_state.body_kpts_3d);
    let stereo_valid = count_valid_points(&stereo.body_kpts_3d);
    let fused_uses_stereo = matches!(
        operator.estimate.source,
        OperatorSource::FusedMultiSource3d
            | OperatorSource::FusedStereoVision3d
            | OperatorSource::FusedStereoVision2dProjected
    ) || operator
        .estimate
        .association
        .anchor_source
        .contains("stereo");

    (fused_valid >= MIN_TEACHER_BODY_JOINTS && fused_uses_stereo)
        || stereo_valid >= MIN_TEACHER_BODY_JOINTS
}

fn robot_capture_ready(state: &AppState) -> bool {
    let bridge = state.bridge_store.snapshot(state.config.bridge_stale_ms);
    bridge.unitree_ready
        && bridge.lan_control_ok
        && state.gate.time_sync_ok(
            state.config.time_sync_ok_window_ms,
            state.config.time_sync_rtt_ok_ms,
        )
        && state.gate.extrinsic_ok()
}

fn phone_device_pose_ready(device_pose: Option<&VisionDevicePose>) -> bool {
    let Some(device_pose) = device_pose else {
        return false;
    };
    let target_space = device_pose.target_space.trim();
    let shared_frame_ready =
        target_space == "ios_arkit_world_frame" || target_space == "stereo_pair_frame";
    if !shared_frame_ready {
        return false;
    }
    let source = device_pose.source.trim();
    if source.is_empty() || source == "ios_device_attitude" {
        return false;
    }
    let has_finite_basis = device_pose
        .right_vector
        .iter()
        .chain(device_pose.up_vector.iter())
        .chain(device_pose.forward_vector.iter())
        .flatten()
        .all(|value| value.is_finite());
    let has_rotation = device_pose
        .rotation_deg
        .iter()
        .flatten()
        .all(|value| value.is_finite());
    has_finite_basis || has_rotation
}

fn build_readiness(
    vision: &VisionSnapshot,
    stereo: &StereoSnapshot,
    wifi: &WifiPoseSnapshot,
    csi: &CsiSnapshot,
    operator: &OperatorSnapshot,
    robot_ready: bool,
) -> EvolutionReadiness {
    let phone_ready = vision.fresh && phone_device_pose_ready(vision.device_pose.as_ref());
    let stereo_ready =
        stereo.fresh && count_valid_points(&stereo.body_kpts_3d) >= MIN_TEACHER_BODY_JOINTS;
    let wifi_ready =
        wifi.fresh && count_valid_points(&wifi.body_kpts_3d) >= MIN_TEACHER_BODY_JOINTS;
    let teacher_ready = teacher_ready(operator, stereo);
    let anchor_source = operator.estimate.association.anchor_source.to_string();
    let selected_operator_track_id = operator
        .estimate
        .association
        .selected_operator_track_id
        .clone();
    let iphone_visible_hand_count = operator.estimate.association.iphone_visible_hand_count;
    let hand_match_count = operator.estimate.association.hand_match_count;
    let hand_match_score = operator.estimate.association.hand_match_score;
    let left_wrist_gap_m = operator.estimate.association.left_wrist_gap_m;
    let right_wrist_gap_m = operator.estimate.association.right_wrist_gap_m;

    let full_multimodal_ready =
        csi.fresh && stereo_ready && wifi_ready && teacher_ready && phone_ready && robot_ready;
    let pose_capture_ready = full_multimodal_ready;
    let room_empty_ready = csi.fresh;

    let mut suggested_quality_score = 0.0_f64;
    if csi.fresh {
        suggested_quality_score += 0.28;
    }
    if wifi_ready {
        suggested_quality_score += 0.24;
    }
    if stereo_ready {
        suggested_quality_score += 0.24;
    }
    if teacher_ready {
        suggested_quality_score += 0.12;
    }
    if phone_ready && hand_match_score >= 0.45 {
        suggested_quality_score += 0.12;
    }
    if robot_ready {
        suggested_quality_score += 0.08;
    }
    if let Some(gap) = left_wrist_gap_m {
        if gap > 0.22 {
            suggested_quality_score -= 0.08;
        }
    }
    if let Some(gap) = right_wrist_gap_m {
        if gap > 0.22 {
            suggested_quality_score -= 0.08;
        }
    }
    let suggested_quality_score = suggested_quality_score.clamp(0.0, 1.0);

    let mut issues = Vec::new();
    if !csi.fresh {
        issues.push("CSI 还没稳定收包，请先确认 6 个节点持续在线。".to_string());
    }
    if !stereo_ready {
        issues.push("双目还没稳定看到当前动作，请先站进双目视野中心。".to_string());
    }
    if !wifi_ready {
        issues.push("Wi‑Fi 人体还没稳定，请先确认 CSI 正在持续发包。".to_string());
    }
    if !teacher_ready {
        issues.push("当前 teacher 还没过线，先让双目和融合骨架稳定几秒。".to_string());
    }
    if !phone_ready {
        issues.push("手机 shared/world-frame device_pose 还没稳定在线，请确认 ARKit 主路正在持续上报。".to_string());
    }
    if !robot_ready {
        issues.push("机器人状态链路还没就绪，请先确认 Unitree bridge、时间同步、外参和 LAN 控制在线。".to_string());
    }
    if phone_ready && iphone_visible_hand_count > 0 && hand_match_score < 0.45 {
        issues.push(
            "手机手部还没和身体对齐，至少保持一只手在手机和双目共同视野里；双手都在会更稳。"
                .to_string(),
        );
    }
    if csi.fresh && (stereo.fresh || vision.fresh || selected_operator_track_id.is_some()) {
        issues.push(
            "空房间是否清空请以现场人工确认为准；CSI 可能穿墙，不把自动人数结果当硬门槛。"
                .to_string(),
        );
    }

    EvolutionReadiness {
        csi_ready: csi.fresh,
        stereo_ready,
        wifi_ready,
        phone_ready,
        robot_ready,
        teacher_ready,
        room_empty_ready,
        full_multimodal_ready,
        pose_capture_ready,
        anchor_source,
        selected_operator_track_id,
        iphone_visible_hand_count,
        hand_match_count,
        hand_match_score,
        left_wrist_gap_m,
        right_wrist_gap_m,
        suggested_quality_score,
        issues,
    }
}

fn summarize_model_info(
    raw: Value,
    lora_state: &ExternalLoraProfilesState,
    sona_state: &ExternalSonaProfilesState,
) -> EvolutionModelSummary {
    let status = raw
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let model_id = raw
        .pointer("/container/manifest/model_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let target_space = raw
        .pointer("/container/metadata/target_space")
        .or_else(|| raw.pointer("/container/metadata/training/target_space"))
        .or_else(|| raw.pointer("/container/metadata/model_config/target_space"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let scene_id = raw
        .pointer("/container/metadata/scene_id")
        .or_else(|| raw.pointer("/container/metadata/training/scene_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let base_model_id = raw
        .pointer("/container/metadata/training/base_model_id")
        .or_else(|| raw.pointer("/container/metadata/model_config/base_model_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let global_parent_model_id = raw
        .pointer("/container/metadata/global_parent_model_id")
        .or_else(|| raw.pointer("/container/metadata/training/base_model_hint"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let best_pck = raw
        .pointer("/container/metadata/training/best_pck")
        .and_then(Value::as_f64);
    let added_history_dataset_ids = json_string_array(
        &raw,
        "/container/metadata/training/added_history_dataset_ids",
    );
    let active_lora_profile = lora_state.active.clone();
    let active_sona_profile = sona_state
        .active
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
        .map(ToOwned::to_owned);
    let training_type = raw
        .pointer("/container/metadata/training/type")
        .and_then(Value::as_str);
    let inferred_lora_runtime = active_lora_profile.is_some()
        || training_type == Some("lora")
        || model_id
            .as_deref()
            .map(|value| value.starts_with("trained-lora-"))
            .unwrap_or(false);
    let adaptation_mode = infer_adaptation_mode(
        !added_history_dataset_ids.is_empty(),
        active_sona_profile.as_deref(),
        if inferred_lora_runtime {
            Some("few-shot-candidate")
        } else {
            None
        },
    );

    EvolutionModelSummary {
        loaded: status == "loaded",
        status,
        model_id,
        target_space,
        scene_id,
        base_model_id,
        global_parent_model_id,
        best_pck,
        scene_history_used: !added_history_dataset_ids.is_empty(),
        scene_history_sample_count: added_history_dataset_ids.len(),
        adaptation_mode,
        sona_profiles: sona_state.profiles.clone(),
        active_sona_profile,
        lora_profiles: lora_state.profiles.clone(),
        active_lora_profile,
        raw,
    }
}

fn summarize_training_status(raw: Value) -> EvolutionTrainingSummary {
    let parsed = serde_json::from_value::<ExternalTrainingStatus>(raw.clone()).unwrap_or_default();
    EvolutionTrainingSummary {
        active: parsed.active,
        phase: parsed.phase,
        epoch: parsed.epoch,
        total_epochs: parsed.total_epochs,
        best_pck: parsed.best_pck,
        best_epoch: parsed.best_epoch,
        val_pck: parsed.val_pck,
        eta_secs: parsed.eta_secs,
        raw,
    }
}

fn summarize_recording_status(raw: Value) -> EvolutionRecordingSummary {
    let parsed = serde_json::from_value::<ExternalRecordingStatus>(raw.clone()).unwrap_or_default();
    EvolutionRecordingSummary {
        active: parsed.active,
        session_id: parsed.session_id,
        session_name: parsed.session_name,
        capture_session_id: parsed.capture_session_id,
        step_code: parsed.step_code,
        frame_count: parsed.frame_count,
        duration_secs: parsed.duration_secs,
        raw,
    }
}

fn build_quality_gate(
    session: &EvolutionSessionState,
    eligible_steps: &[&EvolutionStepState],
    eligible_dataset_ids_len: usize,
) -> EvolutionTrainingQualityGate {
    EvolutionTrainingQualityGate {
        eligible_dataset_count: eligible_dataset_ids_len,
        eligible_empty_count: eligible_steps
            .iter()
            .filter(|step| step.label == "empty")
            .count(),
        eligible_pose_count: eligible_steps
            .iter()
            .filter(|step| step.label == "pose")
            .count(),
        skipped_rerecord_dataset_ids: session
            .steps
            .iter()
            .filter(|step| step.rerecord_recommended)
            .filter_map(|step| step.recording_id.clone())
            .collect(),
        skipped_rerecord_step_codes: session
            .steps
            .iter()
            .filter(|step| step.rerecord_recommended)
            .map(|step| step.code.clone())
            .collect(),
    }
}

fn build_selected_recordings(
    eligible_steps: &[&EvolutionStepState],
) -> Vec<EvolutionTrainingSelectedRecording> {
    eligible_steps
        .iter()
        .filter_map(|step| {
            step.recording_id
                .as_ref()
                .map(|dataset_id| EvolutionTrainingSelectedRecording {
                    dataset_id: dataset_id.clone(),
                    step_code: step.code.clone(),
                    title: step.title.clone(),
                    label: step.label.clone(),
                    quality_score: step.quality_score,
                    rerecord_recommended: step.rerecord_recommended,
                })
        })
        .collect()
}

fn build_training_report_artifact(
    train_job_id: String,
    requested_at_ms: u64,
    session: &EvolutionSessionState,
    eligible_steps: &[&EvolutionStepState],
    explicit_dataset_ids: Vec<String>,
    train_response: &Value,
    current_model: &EvolutionModelSummary,
    min_recording_quality: f64,
) -> EvolutionTrainingArtifact {
    let added_history_dataset_ids = json_string_array(train_response, "/added_history_dataset_ids");
    let resolved_dataset_ids = json_string_array(train_response, "/resolved_dataset_ids");
    let scene_history_sample_count = added_history_dataset_ids.len();
    let scene_history_used = !added_history_dataset_ids.is_empty();
    let quality_gate = build_quality_gate(session, eligible_steps, explicit_dataset_ids.len());
    let selected_recordings = build_selected_recordings(eligible_steps);
    let base_model_id = train_response
        .get("base_model_hint")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| current_model.base_model_id.clone())
        .or_else(|| current_model.model_id.clone());

    EvolutionTrainingArtifact {
        train_job_id,
        status: "running".to_string(),
        train_type: train_response
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("supervised")
            .to_string(),
        scene_id: session.scene_id.clone(),
        scene_name: session.scene_name.clone(),
        capture_session_id: session.capture_session_id.clone(),
        requested_at_ms,
        applied_at_ms: None,
        base_model_id,
        applied_model_id: None,
        candidate_model_id: None,
        candidate_model_created_at: None,
        adaptation_mode: infer_adaptation_mode(scene_history_used, None, None),
        explicit_dataset_ids,
        resolved_dataset_ids,
        added_history_dataset_ids,
        selected_recordings,
        scene_history_used,
        scene_history_sample_count,
        min_recording_quality,
        quality_gate,
        lora_profile_id: None,
        geometry_profile_id: None,
        best_pck: None,
        best_epoch: None,
        evaluator_summary: None,
        cross_domain_summary: None,
        promotion_gate_status: "not_available".to_string(),
    }
}

async fn find_latest_supervised_candidate_model(
    state: &AppState,
    report: &EvolutionTrainingArtifact,
) -> Option<ExternalModelCatalogItem> {
    if let Ok(raw) = sensing_get_json::<Value>(state, "/api/v1/models").await {
        if let Ok(catalog) = serde_json::from_value::<ExternalModelCatalog>(raw) {
            let best_pck = report.best_pck;

            let mut matching: Vec<ExternalModelCatalogItem> = catalog
                .models
                .iter()
                .filter(|model| model.id.starts_with("trained-supervised-"))
                .filter(|model| model.id != "trained-supervised-live")
                .filter(|model| report.base_model_id.as_deref() != Some(model.id.as_str()))
                .filter(|model| {
                    if let Some(expected) = best_pck {
                        model
                            .pck_score
                            .map(|value| (value - expected).abs() <= 1e-6)
                            .unwrap_or(false)
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();
            matching.sort_by(|left, right| left.id.cmp(&right.id));
            if let Some(model) = matching.pop() {
                return Some(model);
            }

            let mut fallback: Vec<ExternalModelCatalogItem> = catalog
                .models
                .into_iter()
                .filter(|model| model.id.starts_with("trained-supervised-"))
                .filter(|model| model.id != "trained-supervised-live")
                .filter(|model| report.base_model_id.as_deref() != Some(model.id.as_str()))
                .collect();
            fallback.sort_by(|left, right| left.id.cmp(&right.id));
            if let Some(model) = fallback.pop() {
                return Some(model);
            }
        }
    }

    let data_dir = PathBuf::from(state.config.data_dir.trim());
    for ancestor in data_dir.ancestors() {
        let models_dir = ancestor.join("RuView/rust-port/wifi-densepose-rs/data/models");
        let mut entries = match tokio::fs::read_dir(&models_dir).await {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        let mut fallback = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("rvf") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            if !stem.starts_with("trained-supervised-") || stem == "trained-supervised-live" {
                continue;
            }
            if report.base_model_id.as_deref() == Some(stem) {
                continue;
            }
            fallback.push(ExternalModelCatalogItem {
                id: stem.to_string(),
                created_at: String::new(),
                pck_score: None,
            });
        }
        fallback.sort_by(|left, right| left.id.cmp(&right.id));
        if let Some(model) = fallback.pop() {
            return Some(model);
        }
    }

    None
}

async fn sync_training_report_artifact(
    state: &AppState,
    session: &mut EvolutionSessionState,
    training: &EvolutionTrainingSummary,
    model: &EvolutionModelSummary,
    supervised_training_active: bool,
) -> Option<EvolutionTrainingArtifact> {
    let train_job_id = session.latest_training_report_id.clone()?;
    let mut report = load_training_report(state, &train_job_id).await?;
    let mut changed = false;

    if training.best_pck > 0.0 && report.best_pck != Some(training.best_pck) {
        report.best_pck = Some(training.best_pck);
        changed = true;
    }
    if training.best_epoch > 0 && report.best_epoch != Some(training.best_epoch) {
        report.best_epoch = Some(training.best_epoch);
        changed = true;
    }

    if !supervised_training_active && report.candidate_model_id.is_none() {
        if let Some(candidate_model) = find_latest_supervised_candidate_model(state, &report).await
        {
            report.candidate_model_id = Some(candidate_model.id.clone());
            report.candidate_model_created_at = if candidate_model.created_at.trim().is_empty() {
                None
            } else {
                Some(candidate_model.created_at)
            };
            changed = true;
        }
    }

    let training_failed = training_phase_failed(&training.phase);
    let next_status = if supervised_training_active {
        "running"
    } else if training_failed {
        "failed"
    } else if session.latest_applied_report_id.as_deref() == Some(train_job_id.as_str()) {
        "applied"
    } else if report.candidate_model_id.is_some() {
        "ready_to_apply"
    } else {
        "candidate_pending_export"
    };
    if report.status != next_status {
        report.status = next_status.to_string();
        changed = true;
    }

    if session.latest_applied_report_id.as_deref() == Some(train_job_id.as_str()) {
        if report.applied_at_ms != session.applied_at_ms {
            report.applied_at_ms = session.applied_at_ms;
            changed = true;
        }
        if model.model_id.is_some() && report.applied_model_id != model.model_id {
            report.applied_model_id = model.model_id.clone();
            changed = true;
        }
    }

    let current_mode = infer_adaptation_mode(
        report.scene_history_used,
        report.geometry_profile_id.as_deref(),
        report.lora_profile_id.as_deref(),
    );
    if report.adaptation_mode != current_mode {
        report.adaptation_mode = current_mode;
        changed = true;
    }

    let next_gate = if supervised_training_active {
        "running"
    } else if training_failed {
        "failed"
    } else if session.latest_applied_report_id.as_deref() == Some(train_job_id.as_str()) {
        "applied"
    } else if report.candidate_model_id.is_some() {
        "ready_to_apply"
    } else {
        "candidate_pending_export"
    };
    if report.promotion_gate_status != next_gate {
        report.promotion_gate_status = next_gate.to_string();
        changed = true;
    }

    if changed {
        let _ = save_training_report(state, &report).await;
    }

    Some(report)
}

fn build_few_shot_candidate_artifact(
    calibration_id: String,
    requested_at_ms: u64,
    session: &EvolutionSessionState,
    eligible_steps: &[&EvolutionStepState],
    source_dataset_ids: Vec<String>,
    current_model: &EvolutionModelSummary,
    profile_name: String,
    rank: u8,
    epochs: u32,
    min_recording_quality: f64,
) -> Result<EvolutionFewShotCandidateArtifact, (StatusCode, String)> {
    let base_model_id = current_model.model_id.clone().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "当前还没有加载 live model，few-shot 校准无法绑定 base model。".to_string(),
        )
    })?;
    let quality_gate = build_quality_gate(session, eligible_steps, source_dataset_ids.len());
    let selected_recordings = build_selected_recordings(eligible_steps);

    Ok(EvolutionFewShotCandidateArtifact {
        calibration_id,
        scene_id: session.scene_id.clone(),
        scene_name: session.scene_name.clone(),
        source_capture_session_id: session.capture_session_id.clone(),
        requested_at_ms,
        completed_at_ms: None,
        previewed_at_ms: None,
        rolled_back_at_ms: None,
        applied_at_ms: None,
        status: "running".to_string(),
        runtime_path: "train_lora_v1_candidate_model".to_string(),
        profile_name,
        base_model_id: base_model_id.clone(),
        candidate_model_id: None,
        candidate_model_created_at: None,
        source_dataset_ids,
        selected_recordings,
        min_recording_quality,
        quality_gate,
        rank,
        epochs,
        best_pck: None,
        best_epoch: None,
        before_metrics: Some(json!({
            "model_id": base_model_id,
            "best_pck": current_model.best_pck,
            "adaptation_mode": current_model.adaptation_mode,
        })),
        after_metrics: None,
        evaluator_summary: None,
        cross_domain_summary: None,
        promotion_gate_status: "candidate_only".to_string(),
        promotion_gate_reason: Some("候选模型还没导出完成，目前只能等待训练结束。".to_string()),
    })
}

fn few_shot_metric_f64(value: Option<&Value>, key: &str) -> Option<f64> {
    value
        .and_then(|inner| inner.get(key))
        .and_then(Value::as_f64)
}

fn build_missing_few_shot_cross_domain_summary() -> Value {
    json!({
        "status": "missing",
        "source": "not_wired_yet",
        "required_buckets": [
            "in_domain",
            "unseen_room_zero_shot",
            "unseen_room_few_shot",
            "cross_hardware"
        ],
        "reason": "cross-domain evaluator summary not attached yet",
    })
}

fn build_few_shot_evaluator_summary(report: &EvolutionFewShotCandidateArtifact) -> Option<Value> {
    let base_best_pck = few_shot_metric_f64(report.before_metrics.as_ref(), "best_pck");
    let candidate_best_pck =
        few_shot_metric_f64(report.after_metrics.as_ref(), "best_pck").or(report.best_pck);
    let best_pck_delta = match (candidate_best_pck, base_best_pck) {
        (Some(candidate), Some(base)) => Some(candidate - base),
        _ => None,
    };

    if base_best_pck.is_none()
        && candidate_best_pck.is_none()
        && report.candidate_model_id.is_none()
    {
        return None;
    }

    Some(json!({
        "source": "train_lora_v1_candidate_metrics",
        "scope": "in_domain_training_metrics_only",
        "cross_domain_ready": false,
        "metric_name": "pck",
        "base_model_id": report.base_model_id,
        "candidate_model_id": report.candidate_model_id,
        "base_best_pck": base_best_pck,
        "candidate_best_pck": candidate_best_pck,
        "best_pck_delta": best_pck_delta,
    }))
}

fn normalize_few_shot_bucket(value: &str) -> Option<&'static str> {
    match value.trim() {
        "in_domain" => Some("in_domain"),
        "unseen_room_zero_shot" => Some("unseen_room_zero_shot"),
        "unseen_room_few_shot" => Some("unseen_room_few_shot"),
        "cross_hardware" => Some("cross_hardware"),
        _ => None,
    }
}

fn dedupe_non_empty_strings(values: &[String]) -> Vec<String> {
    let mut deduped = Vec::new();
    for value in values {
        let normalized = value.trim();
        if normalized.is_empty() {
            continue;
        }
        if !deduped.iter().any(|existing| existing == normalized) {
            deduped.push(normalized.to_string());
        }
    }
    deduped
}

fn benchmark_summary_recording_ids(summary: &BenchmarkSummary) -> Vec<String> {
    let mut dataset_ids = Vec::new();
    for step in &summary.steps {
        if let Some(recording_id) = step.recording_id.as_deref() {
            let normalized = recording_id.trim();
            if normalized.is_empty() {
                continue;
            }
            if !dataset_ids.iter().any(|existing| existing == normalized) {
                dataset_ids.push(normalized.to_string());
            }
        }
    }
    dataset_ids
}

async fn load_benchmark_summary_dataset_ids(
    summary_path: &str,
) -> Result<(Vec<String>, BenchmarkSummary), (StatusCode, String)> {
    let trimmed = summary_path.trim();
    if trimmed.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "benchmark summary 路径不能为空。".to_string(),
        ));
    }
    let path = PathBuf::from(trimmed);
    let content = tokio::fs::read_to_string(&path).await.map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            format!("读取 benchmark summary 失败: {} ({error})", path.display()),
        )
    })?;
    let summary = serde_json::from_str::<BenchmarkSummary>(&content).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            format!(
                "benchmark summary 不是合法 JSON: {} ({error})",
                path.display()
            ),
        )
    })?;
    let dataset_ids = benchmark_summary_recording_ids(&summary);
    if dataset_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "benchmark summary 里没有可用 recording_id: {}",
                path.display()
            ),
        ));
    }
    Ok((dataset_ids, summary))
}

async fn resolve_zero_shot_compare_dataset_ids(
    req: &EvolutionZeroShotValidationRequest,
) -> Result<(Vec<String>, Option<String>), (StatusCode, String)> {
    let mut dataset_ids = Vec::new();
    let benchmark_summary_path = req
        .benchmark_summary_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string());

    if let Some(path) = benchmark_summary_path.as_deref() {
        let (summary_dataset_ids, _summary) = load_benchmark_summary_dataset_ids(path).await?;
        dataset_ids.extend(summary_dataset_ids);
    }
    dataset_ids.extend(req.dataset_ids.clone());

    Ok((
        dedupe_non_empty_strings(&dataset_ids),
        benchmark_summary_path,
    ))
}

fn zero_shot_eval_metric(response: &Value) -> Option<f64> {
    response
        .pointer("/summary/metrics/buckets/unseen_room_zero_shot")
        .and_then(Value::as_f64)
        .or_else(|| {
            response
                .pointer("/summary/bucket_metrics/unseen_room_zero_shot")
                .and_then(Value::as_f64)
        })
        .or_else(|| {
            response
                .pointer("/artifact/cross_domain_summary/metrics/buckets/unseen_room_zero_shot")
                .and_then(Value::as_f64)
        })
        .or_else(|| {
            response
                .pointer("/artifact/cross_domain_summary/bucket_metrics/unseen_room_zero_shot")
                .and_then(Value::as_f64)
        })
}

fn zero_shot_eval_sample_count(response: &Value) -> usize {
    response
        .get("sample_count")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .or_else(|| {
            response
                .pointer("/summary/sample_counts/unseen_room_zero_shot")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
        })
        .or_else(|| {
            response
                .pointer("/artifact/cross_domain_summary/sample_counts/unseen_room_zero_shot")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
        })
        .unwrap_or(0)
}

fn resolve_zero_shot_candidate_sona_profile(
    req: &EvolutionZeroShotValidationRequest,
    session: &EvolutionSessionState,
    geometry: &EvolutionSceneGeometryResponse,
    available_profiles: &[String],
) -> Option<String> {
    let explicit = req
        .candidate_sona_profile
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default");
    if let Some(explicit) = explicit {
        return available_profiles
            .iter()
            .find(|profile| profile.as_str() == explicit)
            .cloned();
    }

    let mut candidates = Vec::new();
    let mut push = |value: Option<&str>| {
        let Some(raw) = value.map(str::trim).filter(|item| !item.is_empty()) else {
            return;
        };
        for variant in [
            raw.to_string(),
            raw.replace('-', "_"),
            raw.replace('_', "-"),
            format!("{raw}-geometry"),
            format!("{raw}_geometry"),
            format!("{raw}-zero-shot"),
            format!("{raw}_zero_shot"),
        ] {
            if !variant.is_empty() && !candidates.iter().any(|item| item == &variant) {
                candidates.push(variant);
            }
        }
        let normalized = normalize_scene_id(Some(raw));
        for variant in [
            normalized.clone(),
            normalized.replace('-', "_"),
            normalized.replace('_', "-"),
            format!("{normalized}-geometry"),
            format!("{normalized}_geometry"),
        ] {
            if !candidates.iter().any(|item| item == &variant) {
                candidates.push(variant);
            }
        }
    };
    push(geometry.summary.coordinate_frame_version.as_deref());
    push(Some(session.scene_id.as_str()));
    push(Some(session.scene_name.as_str()));

    for candidate in candidates {
        if let Some(profile) = available_profiles
            .iter()
            .find(|profile| profile.as_str() == candidate)
        {
            return Some(profile.clone());
        }
    }

    let non_default_profiles = available_profiles
        .iter()
        .filter(|profile| !profile.trim().is_empty() && profile.as_str() != "default")
        .cloned()
        .collect::<Vec<_>>();
    if non_default_profiles.len() == 1 {
        non_default_profiles.into_iter().next()
    } else {
        None
    }
}

fn zero_shot_compare_can_bypass_geometry_blocker(
    validation: &EvolutionZeroShotValidationArtifact,
    compare_dataset_ids: &[String],
    explicit_candidate_sona_profile: Option<&str>,
) -> bool {
    !compare_dataset_ids.is_empty()
        && explicit_candidate_sona_profile
            .map(str::trim)
            .is_some_and(|value| !value.is_empty() && value != "default")
        && validation.model_ready
        && validation.readiness_ready
        && !validation.geometry_ready
        && !validation.blockers.is_empty()
        && validation
            .blockers
            .iter()
            .all(|blocker| blocker == "当前 scene 还没有保存 geometry。")
}

fn resolve_zero_shot_base_model_id(
    validation: &EvolutionZeroShotValidationArtifact,
    model: &EvolutionModelSummary,
    candidate_model_id: Option<&String>,
) -> Option<String> {
    validation
        .base_model_id
        .clone()
        .or(model.base_model_id.clone())
        .or(model.global_parent_model_id.clone())
        .or(model.model_id.clone())
        .or(validation.live_model_id.clone())
        .or_else(|| candidate_model_id.cloned())
}

fn zero_shot_live_model_alias_for(model_id: &str) -> &'static str {
    if model_id.starts_with("trained-pretrain-") {
        "trained-pretrain-live.rvf"
    } else if model_id.starts_with("trained-global-base-") {
        "trained-global-base-live.rvf"
    } else {
        "trained-supervised-live.rvf"
    }
}

fn resolve_zero_shot_eval_model_hint(
    requested_model_id: &str,
    requested_sona_profile: Option<&str>,
    live_model: &EvolutionModelSummary,
) -> String {
    let Some(profile) = requested_sona_profile
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
    else {
        return requested_model_id.to_string();
    };
    let live_matches_requested = live_model.model_id.as_deref() == Some(requested_model_id)
        || live_model.base_model_id.as_deref() == Some(requested_model_id)
        || live_model.global_parent_model_id.as_deref() == Some(requested_model_id);
    if live_matches_requested
        && live_model
            .sona_profiles
            .iter()
            .any(|candidate| candidate == profile)
    {
        zero_shot_live_model_alias_for(requested_model_id).to_string()
    } else {
        requested_model_id.to_string()
    }
}

async fn run_zero_shot_recording_eval(
    state: &AppState,
    validation_id: &str,
    dataset_ids: &[String],
    model_id: &str,
    sona_profile: Option<&str>,
    notes: Option<&str>,
    producer: &str,
) -> Result<Value, (StatusCode, String)> {
    let response = sensing_post_json(
        state,
        "/api/v1/eval/cross-domain-recordings",
        json!({
            "calibration_id": validation_id,
            "notes": notes,
            "producer": producer,
            "include_samples": false,
            "buckets": [{
                "bucket": "unseen_room_zero_shot",
                "dataset_ids": dataset_ids,
                "model_id": model_id,
                "sona_profile": sona_profile,
            }],
        }),
    )
    .await
    .map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!("调用 zero-shot recording evaluator 失败: {error}"),
        )
    })?;
    if response
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default()
        != "ok"
    {
        let message = response
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("zero-shot recording evaluator 没有返回可用结果。")
            .to_string();
        return Err((StatusCode::BAD_GATEWAY, message));
    }
    Ok(response)
}

fn build_zero_shot_compare_summary(
    dataset_ids: &[String],
    benchmark_summary_path: Option<String>,
    base_model_id: &str,
    candidate_model_id: &str,
    base_sona_profile: Option<&str>,
    candidate_sona_profile: Option<&str>,
    base_eval: &Value,
    candidate_eval: &Value,
    notes: Option<String>,
) -> Value {
    let base_metric = zero_shot_eval_metric(base_eval);
    let candidate_metric = zero_shot_eval_metric(candidate_eval);
    let improvement_delta = match (base_metric, candidate_metric) {
        (Some(base), Some(candidate)) => Some(base - candidate),
        _ => None,
    };
    let passed = improvement_delta.map(|delta| delta > 0.0).unwrap_or(false);

    json!({
        "source": "recording_compare_v1",
        "metric_name": "mpjpe_mm",
        "dataset_ids": dataset_ids,
        "dataset_count": dataset_ids.len(),
        "benchmark_summary_path": benchmark_summary_path,
        "base_model_id": base_model_id,
        "candidate_model_id": candidate_model_id,
        "base_sona_profile": base_sona_profile,
        "candidate_sona_profile": candidate_sona_profile,
        "base_metric": base_metric,
        "geometry_conditioned_metric": candidate_metric,
        "improvement_delta": improvement_delta,
        "passed": passed,
        "sample_counts": {
            "base": zero_shot_eval_sample_count(base_eval),
            "candidate": zero_shot_eval_sample_count(candidate_eval),
        },
        "evaluations": {
            "base": base_eval.get("summary").cloned(),
            "candidate": candidate_eval.get("summary").cloned(),
        },
        "notes": notes,
    })
}

async fn discover_latest_benchmark_summary_paths(
    capture_root: &Path,
) -> (Option<String>, Option<String>) {
    let mut latest_preapply: Option<(SystemTime, String)> = None;
    let mut latest_postapply: Option<(SystemTime, String)> = None;
    let mut entries = match tokio::fs::read_dir(capture_root).await {
        Ok(entries) => entries,
        Err(_) => return (None, None),
    };

    loop {
        let Some(entry) = (match entries.next_entry().await {
            Ok(entry) => entry,
            Err(_) => {
                return (
                    latest_preapply.map(|(_, path)| path),
                    latest_postapply.map(|(_, path)| path),
                )
            }
        }) else {
            break;
        };

        let file_type = match entry.file_type().await {
            Ok(file_type) if file_type.is_dir() => file_type,
            _ => continue,
        };
        if !file_type.is_dir() {
            continue;
        }

        let directory_name = entry.file_name().to_string_lossy().to_string();
        let phase = if directory_name.starts_with("wifi-pose-benchmark-preapply-") {
            Some("preapply")
        } else if directory_name.starts_with("wifi-pose-benchmark-postapply-") {
            Some("postapply")
        } else {
            None
        };
        let Some(phase) = phase else {
            continue;
        };

        let summary_path = entry.path().join("summary.json");
        let metadata = match tokio::fs::metadata(&summary_path).await {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => continue,
        };
        let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
        let path_string = summary_path.to_string_lossy().to_string();

        match phase {
            "preapply" => {
                if latest_preapply
                    .as_ref()
                    .map(|(current_modified, _)| modified > *current_modified)
                    .unwrap_or(true)
                {
                    latest_preapply = Some((modified, path_string));
                }
            }
            "postapply" => {
                if latest_postapply
                    .as_ref()
                    .map(|(current_modified, _)| modified > *current_modified)
                    .unwrap_or(true)
                {
                    latest_postapply = Some((modified, path_string));
                }
            }
            _ => {}
        }
    }

    (
        latest_preapply.map(|(_, path)| path),
        latest_postapply.map(|(_, path)| path),
    )
}

fn few_shot_bucket_domain_label(value: &str) -> Option<u32> {
    match normalize_few_shot_bucket(value)? {
        "in_domain" => Some(0),
        "unseen_room_zero_shot" => Some(1),
        "unseen_room_few_shot" => Some(2),
        "cross_hardware" => Some(3),
        _ => None,
    }
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn local_mpjpe(predicted: &[f32], ground_truth: &[f32], joint_count: usize) -> f64 {
    if joint_count == 0 {
        return 0.0;
    }

    let total = (0..joint_count)
        .map(|joint_index| {
            let base = joint_index * 3;
            let dx = predicted.get(base).copied().unwrap_or(0.0) as f64
                - ground_truth.get(base).copied().unwrap_or(0.0) as f64;
            let dy = predicted.get(base + 1).copied().unwrap_or(0.0) as f64
                - ground_truth.get(base + 1).copied().unwrap_or(0.0) as f64;
            let dz = predicted.get(base + 2).copied().unwrap_or(0.0) as f64
                - ground_truth.get(base + 2).copied().unwrap_or(0.0) as f64;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .sum::<f64>();
    total / joint_count as f64
}

fn evaluate_cross_domain_metrics(
    predictions: &[(Vec<f32>, Vec<f32>)],
    domain_labels: &[u32],
    joint_count: usize,
) -> LocalCrossDomainMetrics {
    let mut in_domain_errors = Vec::new();
    let mut cross_domain_errors = Vec::new();
    let mut few_shot_errors = Vec::new();
    let mut cross_hardware_errors = Vec::new();

    for (index, (predicted, ground_truth)) in predictions.iter().enumerate() {
        let domain_label = domain_labels.get(index).copied().unwrap_or_default();
        let error = local_mpjpe(predicted, ground_truth, joint_count);
        match domain_label {
            0 => in_domain_errors.push(error),
            2 => {
                few_shot_errors.push(error);
                cross_domain_errors.push(error);
            }
            3 => {
                cross_hardware_errors.push(error);
                cross_domain_errors.push(error);
            }
            _ => cross_domain_errors.push(error),
        }
    }

    let in_domain_mpjpe = mean_f64(&in_domain_errors).unwrap_or(0.0);
    let cross_domain_mpjpe = mean_f64(&cross_domain_errors).unwrap_or(0.0);
    let few_shot_mpjpe =
        mean_f64(&few_shot_errors).unwrap_or_else(|| (in_domain_mpjpe + cross_domain_mpjpe) / 2.0);
    let cross_hardware_mpjpe = mean_f64(&cross_hardware_errors).unwrap_or(cross_domain_mpjpe);
    let domain_gap_ratio = if in_domain_mpjpe > 1e-10 {
        cross_domain_mpjpe / in_domain_mpjpe
    } else if cross_domain_mpjpe > 1e-10 {
        f64::INFINITY
    } else {
        1.0
    };
    let adaptation_speedup = if few_shot_mpjpe > 1e-10 {
        cross_domain_mpjpe / few_shot_mpjpe
    } else {
        1.0
    };

    LocalCrossDomainMetrics {
        in_domain_mpjpe,
        cross_domain_mpjpe,
        few_shot_mpjpe,
        cross_hardware_mpjpe,
        domain_gap_ratio,
        adaptation_speedup,
    }
}

fn build_few_shot_recording_eval_body(
    req: &EvolutionFewShotEvaluateRecordingsRequest,
    report: &EvolutionFewShotCandidateArtifact,
) -> Result<Value, (StatusCode, String)> {
    if req.buckets.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "请至少提供一个 recording eval bucket。".to_string(),
        ));
    }

    let producer = req
        .producer
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("edge_orchestrator_recording_evaluator")
        .to_string();
    let mut buckets = Vec::new();

    for bucket in &req.buckets {
        let bucket_name = normalize_few_shot_bucket(&bucket.bucket).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                format!("不支持的 recording eval bucket: {}", bucket.bucket),
            )
        })?;
        let dataset_ids = dedupe_non_empty_strings(&bucket.dataset_ids);
        if dataset_ids.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("bucket `{bucket_name}` 缺少 dataset_ids。"),
            ));
        }

        let model_id = bucket
            .model_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string())
            .or_else(|| match bucket_name {
                "unseen_room_few_shot" => report.candidate_model_id.clone(),
                "in_domain" | "unseen_room_zero_shot" | "cross_hardware" => {
                    Some(report.base_model_id.clone())
                }
                _ => None,
            })
            .ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("bucket `{bucket_name}` 还没有可用 model_id。"),
                )
            })?;

        buckets.push(json!({
            "bucket": bucket_name,
            "dataset_ids": dataset_ids,
            "model_id": model_id,
        }));
    }

    Ok(json!({
        "calibration_id": req
            .calibration_id
            .clone()
            .unwrap_or_else(|| report.calibration_id.clone()),
        "notes": req.notes.clone(),
        "producer": producer,
        "include_samples": req.include_samples,
        "buckets": buckets,
    }))
}

async fn build_few_shot_recording_eval_body_from_benchmark_summaries(
    req: &EvolutionFewShotEvaluateBenchmarkSummariesRequest,
    report: &EvolutionFewShotCandidateArtifact,
) -> Result<Value, (StatusCode, String)> {
    let (zero_shot_dataset_ids, _zero_shot_summary) =
        load_benchmark_summary_dataset_ids(&req.zero_shot_summary_path).await?;
    let (few_shot_dataset_ids, _few_shot_summary) =
        load_benchmark_summary_dataset_ids(&req.few_shot_summary_path).await?;

    let in_domain_bucket = if let Some(path) = req.in_domain_summary_path.as_deref() {
        let (dataset_ids, _summary) = load_benchmark_summary_dataset_ids(path).await?;
        Some(EvolutionFewShotRecordingEvalBucketRequest {
            bucket: "in_domain".to_string(),
            dataset_ids,
            model_id: None,
        })
    } else {
        None
    };

    let cross_hardware_bucket = if let Some(path) = req.cross_hardware_summary_path.as_deref() {
        let (dataset_ids, _summary) = load_benchmark_summary_dataset_ids(path).await?;
        Some(EvolutionFewShotRecordingEvalBucketRequest {
            bucket: "cross_hardware".to_string(),
            dataset_ids,
            model_id: None,
        })
    } else {
        None
    };

    build_few_shot_recording_eval_body(
        &EvolutionFewShotEvaluateRecordingsRequest {
            calibration_id: req.calibration_id.clone(),
            notes: req.notes.clone(),
            producer: req.producer.clone(),
            include_samples: req.include_samples,
            buckets: [
                in_domain_bucket,
                Some(EvolutionFewShotRecordingEvalBucketRequest {
                    bucket: "unseen_room_zero_shot".to_string(),
                    dataset_ids: zero_shot_dataset_ids,
                    model_id: None,
                }),
                Some(EvolutionFewShotRecordingEvalBucketRequest {
                    bucket: "unseen_room_few_shot".to_string(),
                    dataset_ids: few_shot_dataset_ids,
                    model_id: None,
                }),
                cross_hardware_bucket,
            ]
            .into_iter()
            .flatten()
            .collect(),
        },
        report,
    )
}

fn build_few_shot_cross_domain_summary_from_samples(
    req: &EvolutionFewShotEvaluateRequest,
) -> Result<Value, (StatusCode, String)> {
    if req.samples.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "请至少提供一条 evaluator sample。".to_string(),
        ));
    }

    let mut joint_count: Option<usize> = None;
    let mut pairs = Vec::with_capacity(req.samples.len());
    let mut domain_labels = Vec::with_capacity(req.samples.len());
    let mut in_domain_errors = Vec::new();
    let mut zero_shot_errors = Vec::new();
    let mut few_shot_errors = Vec::new();
    let mut cross_hardware_errors = Vec::new();

    for (index, sample) in req.samples.iter().enumerate() {
        let bucket = normalize_few_shot_bucket(&sample.bucket).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                format!(
                    "第 {} 条 sample 的 bucket 无效：{}",
                    index + 1,
                    sample.bucket
                ),
            )
        })?;
        if sample.predicted.len() != sample.ground_truth.len() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "第 {} 条 sample 的 predicted / ground_truth 长度不一致。",
                    index + 1
                ),
            ));
        }
        if sample.predicted.is_empty() || sample.predicted.len() % 3 != 0 {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "第 {} 条 sample 的关键点向量长度必须是 3 的倍数且不能为空。",
                    index + 1
                ),
            ));
        }

        let sample_joint_count = sample.predicted.len() / 3;
        if let Some(expected_joint_count) = joint_count {
            if expected_joint_count != sample_joint_count {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("第 {} 条 sample 的 joint 数量与前面不一致。", index + 1),
                ));
            }
        } else {
            joint_count = Some(sample_joint_count);
        }

        let sample_error = local_mpjpe(&sample.predicted, &sample.ground_truth, sample_joint_count);
        match bucket {
            "in_domain" => in_domain_errors.push(sample_error),
            "unseen_room_zero_shot" => zero_shot_errors.push(sample_error),
            "unseen_room_few_shot" => few_shot_errors.push(sample_error),
            "cross_hardware" => cross_hardware_errors.push(sample_error),
            _ => {}
        }

        pairs.push((sample.predicted.clone(), sample.ground_truth.clone()));
        domain_labels.push(few_shot_bucket_domain_label(bucket).unwrap_or_default());
    }

    let joint_count = joint_count.unwrap_or_default();
    let metrics = evaluate_cross_domain_metrics(&pairs, &domain_labels, joint_count);

    let in_domain_metric = mean_f64(&in_domain_errors);
    let unseen_room_zero_shot_metric = mean_f64(&zero_shot_errors);
    let unseen_room_few_shot_metric = mean_f64(&few_shot_errors).or_else(|| {
        if few_shot_errors.is_empty() {
            None
        } else {
            Some(metrics.few_shot_mpjpe)
        }
    });
    let cross_hardware_metric = mean_f64(&cross_hardware_errors);
    let domain_gap_ratio = metrics.domain_gap_ratio;
    let adaptation_speedup = metrics.adaptation_speedup;
    let few_shot_improvement_delta =
        match (unseen_room_zero_shot_metric, unseen_room_few_shot_metric) {
            (Some(zero_shot), Some(few_shot)) => Some(zero_shot - few_shot),
            _ => None,
        };

    let has_in_domain = in_domain_metric.is_some();
    let has_zero_shot = unseen_room_zero_shot_metric.is_some();
    let has_few_shot = unseen_room_few_shot_metric.is_some();
    let improvement_ok = few_shot_improvement_delta.is_some_and(|delta| delta > 0.0);
    let domain_gap_ok =
        domain_gap_ratio.is_finite() && domain_gap_ratio < FEW_SHOT_GATE_MAX_DOMAIN_GAP_RATIO;
    let adaptation_speedup_ok =
        adaptation_speedup.is_finite() && adaptation_speedup > FEW_SHOT_GATE_MIN_ADAPTATION_SPEEDUP;
    let passed = has_in_domain
        && has_zero_shot
        && has_few_shot
        && improvement_ok
        && domain_gap_ok
        && adaptation_speedup_ok;

    Ok(json!({
        "source": "cross_domain_evaluator_v1",
        "metric_name": "mpjpe_mm",
        "reported_at_ms": now_ms(),
        "sample_count": req.samples.len(),
        "joint_count": joint_count,
        "bucket_counts": {
            "in_domain": in_domain_errors.len(),
            "unseen_room_zero_shot": zero_shot_errors.len(),
            "unseen_room_few_shot": few_shot_errors.len(),
            "cross_hardware": cross_hardware_errors.len(),
        },
        "buckets": {
            "in_domain": in_domain_metric,
            "unseen_room_zero_shot": unseen_room_zero_shot_metric,
            "unseen_room_few_shot": unseen_room_few_shot_metric,
            "cross_hardware": cross_hardware_metric,
        },
        "cross_domain_metric": metrics.cross_domain_mpjpe,
        "cross_hardware_metric": metrics.cross_hardware_mpjpe,
        "in_domain_metric": metrics.in_domain_mpjpe,
        "domain_gap_ratio": domain_gap_ratio,
        "few_shot_improvement_delta": few_shot_improvement_delta,
        "adaptation_speedup": adaptation_speedup,
        "hardware_type": req.hardware_type,
        "notes": req.notes,
        "passed": passed,
        "gate": {
            "passed": passed,
            "thresholds": {
                "max_domain_gap_ratio": FEW_SHOT_GATE_MAX_DOMAIN_GAP_RATIO,
                "min_adaptation_speedup": FEW_SHOT_GATE_MIN_ADAPTATION_SPEEDUP,
                "require_positive_few_shot_delta": true,
            },
            "checks": [
                {
                    "key": "in_domain_bucket_present",
                    "ok": has_in_domain,
                    "detail": if has_in_domain { "已提供 in-domain 样本。" } else { "缺少 in-domain 样本。" },
                },
                {
                    "key": "zero_shot_bucket_present",
                    "ok": has_zero_shot,
                    "detail": if has_zero_shot { "已提供 unseen_room_zero_shot 样本。" } else { "缺少 unseen_room_zero_shot 样本。" },
                },
                {
                    "key": "few_shot_bucket_present",
                    "ok": has_few_shot,
                    "detail": if has_few_shot { "已提供 unseen_room_few_shot 样本。" } else { "缺少 unseen_room_few_shot 样本。" },
                },
                {
                    "key": "few_shot_improves_over_zero_shot",
                    "ok": improvement_ok,
                    "detail": if let Some(delta) = few_shot_improvement_delta {
                        format!("few-shot improvement delta = {:.2} mm。", delta)
                    } else {
                        "还无法比较 zero-shot 与 few-shot 的差值。".to_string()
                    },
                },
                {
                    "key": "domain_gap_ratio",
                    "ok": domain_gap_ok,
                    "detail": format!(
                        "domain_gap_ratio = {:.3}，阈值 < {:.1}。",
                        domain_gap_ratio,
                        FEW_SHOT_GATE_MAX_DOMAIN_GAP_RATIO
                    ),
                },
                {
                    "key": "adaptation_speedup",
                    "ok": adaptation_speedup_ok,
                    "detail": format!(
                        "adaptation_speedup = {:.3}x，阈值 > {:.1}x。",
                        adaptation_speedup,
                        FEW_SHOT_GATE_MIN_ADAPTATION_SPEEDUP
                    ),
                }
            ]
        }
    }))
}

fn build_few_shot_cross_domain_summary_from_report(
    req: &EvolutionFewShotEvaluationReportRequest,
    source: &str,
) -> Value {
    let few_shot_improvement_delta = req.few_shot_improvement_delta.or_else(|| {
        match (
            req.unseen_room_zero_shot_metric,
            req.unseen_room_few_shot_metric,
        ) {
            (Some(zero_shot), Some(few_shot)) => Some(zero_shot - few_shot),
            _ => None,
        }
    });

    json!({
        "source": source,
        "metric_name": req.metric_name.clone().filter(|value| !value.trim().is_empty()).unwrap_or_else(|| "mpjpe_mm".to_string()),
        "buckets": {
            "in_domain": req.in_domain_metric,
            "unseen_room_zero_shot": req.unseen_room_zero_shot_metric,
            "unseen_room_few_shot": req.unseen_room_few_shot_metric,
            "cross_hardware": req.cross_hardware_metric,
        },
        "domain_gap_ratio": req.domain_gap_ratio,
        "few_shot_improvement_delta": few_shot_improvement_delta,
        "adaptation_speedup": req.adaptation_speedup,
        "hardware_type": req.hardware_type,
        "passed": req.passed,
        "notes": req.notes,
        "reported_at_ms": now_ms(),
        "gate": {
            "passed": req.passed,
        }
    })
}

fn json_string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn json_f64_field(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn json_bool_field(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn build_few_shot_report_request_from_value(
    raw: &Value,
) -> Result<EvolutionFewShotEvaluationReportRequest, (StatusCode, String)> {
    let source = raw
        .get("cross_domain_summary")
        .filter(|value| value.is_object())
        .unwrap_or(raw);
    let buckets = source
        .get("buckets")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));

    let passed = json_bool_field(source, "passed")
        .or_else(|| source.pointer("/gate/passed").and_then(Value::as_bool))
        .or_else(|| {
            json_string_field(source, "status")
                .map(|value| matches!(value.as_str(), "pass" | "passed" | "ok"))
        })
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "导入的 evaluator artifact 缺少 passed / gate.passed 字段。".to_string(),
            )
        })?;

    Ok(EvolutionFewShotEvaluationReportRequest {
        calibration_id: json_string_field(raw, "calibration_id"),
        metric_name: json_string_field(source, "metric_name"),
        in_domain_metric: json_f64_field(&buckets, "in_domain")
            .or_else(|| json_f64_field(source, "in_domain_metric")),
        unseen_room_zero_shot_metric: json_f64_field(&buckets, "unseen_room_zero_shot")
            .or_else(|| json_f64_field(source, "unseen_room_zero_shot_metric")),
        unseen_room_few_shot_metric: json_f64_field(&buckets, "unseen_room_few_shot")
            .or_else(|| json_f64_field(source, "unseen_room_few_shot_metric")),
        cross_hardware_metric: json_f64_field(&buckets, "cross_hardware")
            .or_else(|| json_f64_field(source, "cross_hardware_metric")),
        domain_gap_ratio: json_f64_field(source, "domain_gap_ratio"),
        few_shot_improvement_delta: json_f64_field(source, "few_shot_improvement_delta"),
        adaptation_speedup: json_f64_field(source, "adaptation_speedup"),
        hardware_type: json_string_field(source, "hardware_type"),
        passed,
        notes: json_string_field(source, "notes"),
    })
}

fn annotate_imported_few_shot_cross_domain_summary(
    mut summary: Value,
    source_artifact_path: Option<String>,
    snapshot_path: String,
    imported_at_ms: u64,
    schema_version: Option<String>,
) -> Value {
    if let Some(object) = summary.as_object_mut() {
        object.insert(
            "imported_artifact".to_string(),
            json!({
                "source_path": source_artifact_path,
                "snapshot_path": snapshot_path,
                "imported_at_ms": imported_at_ms,
                "schema_version": schema_version,
            }),
        );
    }
    summary
}

fn build_few_shot_cross_domain_summary_from_imported_report(
    artifact: &Value,
    req: &EvolutionFewShotEvaluationReportRequest,
) -> Value {
    let fallback = build_few_shot_cross_domain_summary_from_report(req, "imported_report_v1");
    let nested_summary = artifact
        .get("cross_domain_summary")
        .filter(|value| value.is_object())
        .cloned();
    let has_nested_summary = nested_summary.is_some();
    let mut summary = nested_summary.unwrap_or_else(|| artifact.clone());

    if !has_nested_summary {
        if let Some(object) = summary.as_object_mut() {
            for key in [
                "calibration_id",
                "schema_version",
                "producer",
                "generated_at",
            ] {
                object.remove(key);
            }
        }
    }

    let Some(summary_object) = summary.as_object_mut() else {
        return fallback;
    };
    let Some(fallback_object) = fallback.as_object() else {
        return summary;
    };

    for (key, value) in fallback_object {
        if key == "gate" {
            let gate_entry = summary_object
                .entry("gate".to_string())
                .or_insert_with(|| json!({}));
            if let Some(gate_object) = gate_entry.as_object_mut() {
                if let Some(fallback_gate_object) = value.as_object() {
                    for (gate_key, gate_value) in fallback_gate_object {
                        gate_object
                            .entry(gate_key.clone())
                            .or_insert_with(|| gate_value.clone());
                    }
                }
            } else {
                *gate_entry = value.clone();
            }
            continue;
        }
        summary_object
            .entry(key.clone())
            .or_insert_with(|| value.clone());
    }

    summary
}

fn build_few_shot_cross_domain_summary_from_imported_artifact(
    artifact: &Value,
    source_artifact_path: Option<String>,
    snapshot_path: String,
    imported_at_ms: u64,
) -> Result<(Option<String>, Value), (StatusCode, String)> {
    let calibration_id = json_string_field(artifact, "calibration_id");
    let schema_version = json_string_field(artifact, "schema_version");

    let summary = if artifact.get("samples").and_then(Value::as_array).is_some() {
        let req = serde_json::from_value::<EvolutionFewShotEvaluateRequest>(artifact.clone())
            .map_err(|error| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("few-shot evaluator samples artifact 解析失败: {error}"),
                )
            })?;
        build_few_shot_cross_domain_summary_from_samples(&req)?
    } else {
        let req = build_few_shot_report_request_from_value(artifact)?;
        build_few_shot_cross_domain_summary_from_imported_report(artifact, &req)
    };

    Ok((
        calibration_id,
        annotate_imported_few_shot_cross_domain_summary(
            summary,
            source_artifact_path,
            snapshot_path,
            imported_at_ms,
            schema_version,
        ),
    ))
}

fn few_shot_artifact_matches_calibration_id(
    artifact: &Value,
    artifact_path: &Path,
    calibration_id: &str,
) -> bool {
    if json_string_field(artifact, "calibration_id").as_deref() == Some(calibration_id) {
        return true;
    }

    artifact_path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.contains(calibration_id))
        .unwrap_or(false)
}

async fn import_few_shot_artifact(
    state: &AppState,
    calibration_id_hint: Option<String>,
    source_artifact_path: Option<String>,
    artifact: Value,
) -> Result<EvolutionCurrentResponse, (StatusCode, String)> {
    let session = load_session(state).await;
    let imported_at_ms = now_ms();
    let calibration_id_for_snapshot = calibration_id_hint
        .clone()
        .or_else(|| json_string_field(&artifact, "calibration_id"))
        .or_else(|| session.latest_few_shot_candidate_id.clone())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 few-shot candidate，无法导入 evaluator artifact。".to_string(),
            )
        })?;
    let snapshot_path = save_few_shot_import_snapshot(
        state,
        &calibration_id_for_snapshot,
        imported_at_ms,
        &artifact,
    )
    .await?;
    let (artifact_calibration_id, summary) =
        build_few_shot_cross_domain_summary_from_imported_artifact(
            &artifact,
            source_artifact_path,
            snapshot_path,
            imported_at_ms,
        )?;
    let calibration_id = calibration_id_hint
        .or(artifact_calibration_id)
        .or_else(|| session.latest_few_shot_candidate_id.clone())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "导入 artifact 后仍无法确定 calibration_id。".to_string(),
            )
        })?;
    let mut report = load_few_shot_candidate_report(state, &calibration_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "few-shot candidate 报告不存在，请重新发起校准。".to_string(),
            )
        })?;

    report.evaluator_summary = build_few_shot_evaluator_summary(&report);
    report.cross_domain_summary = Some(summary);
    sync_few_shot_promotion_gate(&mut report, &session);
    save_few_shot_candidate_report(state, &report).await?;

    Ok(build_current_response(state).await)
}

async fn load_latest_few_shot_inbox_artifact(
    state: &AppState,
    calibration_id: &str,
) -> Result<(String, Value), (StatusCode, String)> {
    let inbox_dir = evolution_few_shot_inbox_dir(state);
    let mut entries = tokio::fs::read_dir(&inbox_dir).await.map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            format!(
                "few-shot evaluator inbox 不可用: {} ({error})",
                inbox_dir.display()
            ),
        )
    })?;

    let mut latest: Option<(u128, String, Value)> = None;
    while let Some(entry) = entries.next_entry().await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("遍历 few-shot evaluator inbox 失败: {error}"),
        )
    })? {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let metadata = match entry.metadata().await {
            Ok(value) => value,
            Err(_) => continue,
        };
        let modified_at_ms = metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
            .map(|value| value.as_millis())
            .unwrap_or(0);
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(value) => value,
            Err(_) => continue,
        };
        let artifact = match serde_json::from_str::<Value>(&content) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if !few_shot_artifact_matches_calibration_id(&artifact, &path, calibration_id) {
            continue;
        }
        let artifact_path = path.to_string_lossy().to_string();
        let should_replace = latest
            .as_ref()
            .map(|(current_ms, _, _)| modified_at_ms >= *current_ms)
            .unwrap_or(true);
        if should_replace {
            latest = Some((modified_at_ms, artifact_path, artifact));
        }
    }

    latest
        .map(|(_, path, artifact)| (path, artifact))
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                format!(
                    "在 {} 里没有找到 calibration_id={} 的 evaluator artifact。",
                    inbox_dir.display(),
                    calibration_id
                ),
            )
        })
}

fn few_shot_cross_domain_report_passed(summary: Option<&Value>) -> Option<bool> {
    summary
        .and_then(|value| value.get("passed"))
        .and_then(Value::as_bool)
        .or_else(|| {
            summary
                .and_then(|value| value.pointer("/gate/passed"))
                .and_then(Value::as_bool)
        })
}

fn sync_few_shot_promotion_gate(
    report: &mut EvolutionFewShotCandidateArtifact,
    session: &EvolutionSessionState,
) -> bool {
    let (next_status, next_reason) = if session.latest_applied_few_shot_candidate_id.as_deref()
        == Some(report.calibration_id.as_str())
        || report.status == "active"
    {
        ("promoted".to_string(), None)
    } else if report.candidate_model_id.is_none() {
        (
            "candidate_only".to_string(),
            Some("候选模型还没导出完成，目前只能等待训练结束。".to_string()),
        )
    } else if report
        .evaluator_summary
        .as_ref()
        .and_then(|value| value.get("best_pck_delta"))
        .and_then(Value::as_f64)
        .is_some_and(|delta| delta < 0.0)
    {
        let delta = report
            .evaluator_summary
            .as_ref()
            .and_then(|value| value.get("best_pck_delta"))
            .and_then(Value::as_f64)
            .unwrap_or_default();
        (
            "blocked_regressed_in_domain".to_string(),
            Some(format!(
                "候选 best PCK 比 base 下降了 {:.1}%，先不要升配。",
                delta.abs() * 100.0
            )),
        )
    } else {
        match few_shot_cross_domain_report_passed(report.cross_domain_summary.as_ref()) {
            Some(true) => (
                "eligible_for_apply".to_string(),
                Some("cross-domain evaluator 已通过，可以人工升配为默认 live 路径。".to_string()),
            ),
            Some(false) => (
                "blocked_cross_domain_eval".to_string(),
                Some("cross-domain evaluator 未通过，默认 live 路径仍被 gate 拦住。".to_string()),
            ),
            None => (
                "blocked_missing_cross_domain_eval".to_string(),
                Some(
                    "还没有 cross-domain evaluator summary，当前只能预览，不能升配为默认 live。"
                        .to_string(),
                ),
            ),
        }
    };

    let mut changed = false;
    if report.promotion_gate_status != next_status {
        report.promotion_gate_status = next_status;
        changed = true;
    }
    if report.promotion_gate_reason != next_reason {
        report.promotion_gate_reason = next_reason;
        changed = true;
    }
    changed
}

async fn find_latest_few_shot_candidate_model(
    state: &AppState,
    report: &EvolutionFewShotCandidateArtifact,
) -> Option<ExternalModelCatalogItem> {
    let raw = sensing_get_json::<Value>(state, "/api/v1/models")
        .await
        .ok()?;
    let catalog = serde_json::from_value::<ExternalModelCatalog>(raw).ok()?;
    catalog
        .models
        .into_iter()
        .filter(|model| model.id.starts_with("trained-lora-"))
        .filter(|model| model.id != report.base_model_id)
        .max_by(|left, right| left.id.cmp(&right.id))
}

async fn sync_few_shot_candidate_artifact(
    state: &AppState,
    session: &mut EvolutionSessionState,
    training: &EvolutionTrainingSummary,
    model: &EvolutionModelSummary,
    few_shot_training_active: bool,
) -> Option<EvolutionFewShotCandidateArtifact> {
    let calibration_id = session.latest_few_shot_candidate_id.clone()?;
    let mut report = load_few_shot_candidate_report(state, &calibration_id).await?;
    let mut changed = false;

    if session.latest_applied_few_shot_candidate_id.as_deref()
        == Some(report.calibration_id.as_str())
        && model.model_id.as_deref() != report.candidate_model_id.as_deref()
    {
        session.latest_applied_few_shot_candidate_id = None;
    }

    if training.best_pck > 0.0 && report.best_pck != Some(training.best_pck) {
        report.best_pck = Some(training.best_pck);
        changed = true;
    }
    if training.best_epoch > 0 && report.best_epoch != Some(training.best_epoch) {
        report.best_epoch = Some(training.best_epoch);
        changed = true;
    }

    if few_shot_training_active {
        if report.status != "running" {
            report.status = "running".to_string();
            changed = true;
        }
    } else if report.candidate_model_id.is_none() {
        if let Some(candidate_model) = find_latest_few_shot_candidate_model(state, &report).await {
            report.candidate_model_id = Some(candidate_model.id.clone());
            report.candidate_model_created_at = if candidate_model.created_at.trim().is_empty() {
                None
            } else {
                Some(candidate_model.created_at.clone())
            };
            report.completed_at_ms = Some(now_ms());
            report.after_metrics = Some(json!({
                "candidate_model_id": candidate_model.id,
                "best_pck": report.best_pck.or(candidate_model.pck_score),
                "best_epoch": report.best_epoch,
            }));
            report.status = "candidate".to_string();
            changed = true;
        }
    } else if session.latest_applied_few_shot_candidate_id.as_deref()
        == Some(report.calibration_id.as_str())
        && model.model_id.as_deref() == report.candidate_model_id.as_deref()
    {
        if report.status != "active" {
            report.status = "active".to_string();
            changed = true;
        }
    } else if model.model_id.as_deref() == report.candidate_model_id.as_deref() {
        if report.status != "preview_active" {
            report.status = "preview_active".to_string();
            changed = true;
        }
    } else if model.model_id.as_deref() == Some(report.base_model_id.as_str()) {
        if matches!(report.status.as_str(), "preview_active" | "active") {
            report.status = "rolled_back".to_string();
            changed = true;
        } else if report.status == "running" {
            report.status = "candidate_pending_export".to_string();
            changed = true;
        }
    } else if report.candidate_model_id.is_some() && report.status == "running" {
        report.status = "candidate".to_string();
        changed = true;
    }

    let next_evaluator_summary = build_few_shot_evaluator_summary(&report);
    if report.evaluator_summary != next_evaluator_summary {
        report.evaluator_summary = next_evaluator_summary;
        changed = true;
    }
    if report.cross_domain_summary.is_none() && report.candidate_model_id.is_some() {
        report.cross_domain_summary = Some(build_missing_few_shot_cross_domain_summary());
        changed = true;
    }
    if sync_few_shot_promotion_gate(&mut report, session) {
        changed = true;
    }

    if changed {
        let _ = save_few_shot_candidate_report(state, &report).await;
    }

    Some(report)
}

fn refresh_session_from_recordings(
    session: &mut EvolutionSessionState,
    recording_status: &EvolutionRecordingSummary,
    recordings: &[ExternalRecordingSession],
    training: &EvolutionTrainingSummary,
) {
    let now = now_ms();
    let active_capture_session_id = recording_status.capture_session_id.as_deref();
    let active_step_code = recording_status.step_code.as_deref();

    for step in &mut session.steps {
        let latest_recording =
            session
                .capture_session_id
                .as_deref()
                .and_then(|capture_session_id| {
                    recordings
                        .iter()
                        .filter(|recording| {
                            recording.capture_session_id.as_deref() == Some(capture_session_id)
                                && recording.step_code.as_deref() == Some(step.code.as_str())
                        })
                        .max_by(|left, right| left.id.cmp(&right.id))
                });

        if recording_status.active
            && active_capture_session_id == session.capture_session_id.as_deref()
            && active_step_code == Some(step.code.as_str())
        {
            step.status = "recording".to_string();
            step.recording_id = recording_status.session_id.clone();
            step.frame_count = Some(recording_status.frame_count);
            if step.started_at_ms.is_none() {
                step.started_at_ms = Some(now);
            }
            continue;
        }

        if let Some(recording) = latest_recording {
            step.status = "recorded".to_string();
            step.recording_id = Some(recording.id.clone());
            step.frame_count = Some(recording.frame_count);
            step.quality_score = recording.quality_score;
            if step.started_at_ms.is_none() {
                step.started_at_ms = Some(now);
            }
            if step.completed_at_ms.is_none() {
                step.completed_at_ms = Some(now);
            }
            continue;
        }

        step.status = "pending".to_string();
        step.recording_id = None;
        step.frame_count = None;
        step.quality_score = None;
        step.started_at_ms = None;
        step.completed_at_ms = None;
    }

    session.current_step_code = session
        .steps
        .iter()
        .find(|step| step.status != "recorded")
        .map(|step| step.code.clone());
    session.updated_at_ms = now;

    let all_recorded = session.steps.iter().all(|step| step.status == "recorded");
    let has_active_recording = session.steps.iter().any(|step| step.status == "recording");
    session.status = if training.active {
        "training".to_string()
    } else if has_active_recording {
        "recording".to_string()
    } else if session.applied_at_ms.is_some() {
        "applied".to_string()
    } else if session.training_requested_at_ms.is_some() {
        "ready_to_apply".to_string()
    } else if all_recorded && session.capture_session_id.is_some() {
        "ready_to_train".to_string()
    } else if session.capture_session_id.is_some() {
        "collecting".to_string()
    } else {
        "idle".to_string()
    };
}

async fn build_current_response(state: &AppState) -> EvolutionCurrentResponse {
    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let wifi = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let csi = state.csi.snapshot(state.config.csi_stale_ms);
    let operator = state.operator.snapshot(state.config.operator_hold_ms);
    let readiness = build_readiness(
        &vision,
        &stereo,
        &wifi,
        &csi,
        &operator,
        robot_capture_ready(state),
    );

    let mut upstream_errors = Vec::new();

    let recording_status_raw =
        match sensing_get_json::<Value>(state, "/api/v1/recording/status").await {
            Ok(value) => value,
            Err(error) => {
                upstream_errors.push(error);
                json!({})
            }
        };
    let recording_list_raw = match sensing_get_json::<Value>(state, "/api/v1/recording/list").await
    {
        Ok(value) => value,
        Err(error) => {
            upstream_errors.push(error);
            json!({"recordings": []})
        }
    };
    let training_raw = match sensing_get_json::<Value>(state, "/api/v1/train/status").await {
        Ok(value) => value,
        Err(error) => {
            upstream_errors.push(error);
            json!({})
        }
    };
    let model_raw = match sensing_get_json::<Value>(state, "/api/v1/model/info").await {
        Ok(value) => value,
        Err(error) => {
            upstream_errors.push(error);
            json!({"status": "unavailable"})
        }
    };
    let lora_profiles_raw =
        match sensing_get_json::<Value>(state, "/api/v1/models/lora/profiles").await {
            Ok(value) => value,
            Err(error) => {
                upstream_errors.push(error);
                json!({})
            }
        };
    let sona_profiles_raw =
        match sensing_get_json::<Value>(state, "/api/v1/model/sona/profiles").await {
            Ok(value) => value,
            Err(error) => {
                upstream_errors.push(error);
                json!({})
            }
        };

    let recording = summarize_recording_status(recording_status_raw);
    let training = summarize_training_status(training_raw);
    let lora_state =
        serde_json::from_value::<ExternalLoraProfilesState>(lora_profiles_raw).unwrap_or_default();
    let sona_state =
        serde_json::from_value::<ExternalSonaProfilesState>(sona_profiles_raw).unwrap_or_default();
    let model = summarize_model_info(model_raw, &lora_state, &sona_state);
    let recordings = serde_json::from_value::<ExternalRecordingList>(recording_list_raw)
        .unwrap_or_default()
        .recordings;
    let benchmark_capture_root = evolution_benchmark_capture_dir(state);
    let (latest_preapply_summary_path, latest_postapply_summary_path) =
        discover_latest_benchmark_summary_paths(&benchmark_capture_root).await;

    let mut session = load_session(state).await;
    sync_step_runtimes(&mut session);
    refresh_session_from_recordings(&mut session, &recording, &recordings, &training);
    sync_recorded_step_runtimes_from_teacher_files(&mut session);
    update_step_runtimes(&mut session, &recording, &readiness, &operator, &stereo);
    let supervised_training_active =
        training.active && session.active_training_kind.as_deref() == Some("supervised");
    let few_shot_training_active =
        training.active && session.active_training_kind.as_deref() == Some("few_shot_lora");
    let training_report = sync_training_report_artifact(
        state,
        &mut session,
        &training,
        &model,
        supervised_training_active,
    )
    .await;
    let zero_shot_validation = match session.latest_zero_shot_validation_id.as_deref() {
        Some(validation_id) => {
            let mut report = load_zero_shot_validation_report(state, validation_id).await;
            if let Some(report_mut) = report.as_mut() {
                if sync_zero_shot_promotion_gate(report_mut, &model) {
                    let _ = save_zero_shot_validation_report(state, report_mut).await;
                }
            }
            report
        }
        None => None,
    };
    let few_shot_candidate = sync_few_shot_candidate_artifact(
        state,
        &mut session,
        &training,
        &model,
        few_shot_training_active,
    )
    .await;
    if !training.active && session.active_training_kind.is_some() {
        session.active_training_kind = None;
    }
    let step_guides = build_step_guides(&session, &readiness, &operator, &stereo);
    let current_step_guide = session.current_step_code.as_ref().and_then(|step_code| {
        step_guides
            .iter()
            .find(|guide| guide.step_code == *step_code)
            .cloned()
    });
    let _ = save_session(state, &session).await;

    EvolutionCurrentResponse {
        ok: true,
        scene: EvolutionSceneInfo {
            scene_id: session.scene_id.clone(),
            scene_name: session.scene_name.clone(),
            display_name_rule: DEFAULT_SCENE_NAME_RULE,
            system_id_rule: DEFAULT_SCENE_ID_RULE,
        },
        session,
        readiness,
        current_step_guide,
        step_guides,
        model,
        recording,
        training,
        training_report,
        zero_shot_validation,
        few_shot_candidate,
        few_shot_evaluator_inbox_dir: evolution_few_shot_inbox_dir(state)
            .to_string_lossy()
            .to_string(),
        few_shot_benchmark_discovery: EvolutionFewShotBenchmarkDiscovery {
            capture_root_dir: benchmark_capture_root.to_string_lossy().to_string(),
            latest_preapply_summary_path,
            latest_postapply_summary_path,
        },
        diagnostics: EvolutionDiagnostics { upstream_errors },
    }
}

async fn get_current(State(state): State<AppState>) -> Json<EvolutionCurrentResponse> {
    Json(build_current_response(&state).await)
}

async fn get_scene_geometry(State(state): State<AppState>) -> Json<EvolutionSceneGeometryResponse> {
    let session = load_session(&state).await;
    let geometry = load_scene_geometry(&state, &session.scene_id).await;
    Json(build_scene_geometry_response(
        session.scene_id,
        session.scene_name,
        geometry,
    ))
}

async fn post_scene_geometry_auto_draft(
    State(state): State<AppState>,
) -> Json<EvolutionSceneGeometryAutoDraftResponse> {
    let session = load_session(&state).await;
    let geometry = load_scene_geometry(&state, &session.scene_id).await;
    let iphone_calibration = state.iphone_stereo_calibration.snapshot();
    Json(build_scene_geometry_auto_draft(
        &session,
        geometry,
        iphone_calibration,
    ))
}

async fn post_scene_geometry(
    State(state): State<AppState>,
    Json(req): Json<EvolutionSceneGeometryUpsertRequest>,
) -> Result<Json<EvolutionSceneGeometryResponse>, (StatusCode, String)> {
    validate_scene_geometry_request(&req).map_err(|error| (StatusCode::BAD_REQUEST, error))?;

    let session = load_session(&state).await;
    let geometry = EvolutionSceneGeometry {
        scene_id: session.scene_id.clone(),
        scene_name: session.scene_name.clone(),
        coordinate_frame_version: req.coordinate_frame_version.trim().to_string(),
        updated_at_ms: now_ms(),
        source: default_geometry_source(),
        ap_nodes: req.ap_nodes,
        stereo_rig: req
            .stereo_rig
            .ok_or_else(|| (StatusCode::BAD_REQUEST, "stereo_rig 不能为空。".to_string()))?,
        phone_pose: req.phone_pose,
        notes: req
            .notes
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    };
    save_scene_geometry(&state, &geometry).await?;

    Ok(Json(build_scene_geometry_response(
        geometry.scene_id.clone(),
        geometry.scene_name.clone(),
        Some(geometry),
    )))
}

async fn post_validate_zero_shot(
    State(state): State<AppState>,
    Json(req): Json<EvolutionZeroShotValidationRequest>,
) -> Result<Json<EvolutionZeroShotValidationArtifact>, (StatusCode, String)> {
    let mut session = load_session(&state).await;
    let geometry = load_scene_geometry(&state, &session.scene_id).await;
    let geometry_response = build_scene_geometry_response(
        session.scene_id.clone(),
        session.scene_name.clone(),
        geometry,
    );

    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let wifi = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let csi = state.csi.snapshot(state.config.csi_stale_ms);
    let operator = state.operator.snapshot(state.config.operator_hold_ms);
    let readiness = build_readiness(
        &vision,
        &stereo,
        &wifi,
        &csi,
        &operator,
        robot_capture_ready(&state),
    );

    let model_raw = sensing_get_json::<Value>(&state, "/api/v1/model/info")
        .await
        .unwrap_or_else(|_| json!({"status": "unavailable"}));
    let lora_profiles_raw = sensing_get_json::<Value>(&state, "/api/v1/models/lora/profiles")
        .await
        .unwrap_or_else(|_| json!({}));
    let sona_profiles_raw = sensing_get_json::<Value>(&state, "/api/v1/model/sona/profiles")
        .await
        .unwrap_or_else(|_| json!({}));
    let lora_state =
        serde_json::from_value::<ExternalLoraProfilesState>(lora_profiles_raw).unwrap_or_default();
    let sona_state =
        serde_json::from_value::<ExternalSonaProfilesState>(sona_profiles_raw).unwrap_or_default();
    let model = summarize_model_info(model_raw, &lora_state, &sona_state);

    let validated_at_ms = now_ms();
    let mut validation = build_zero_shot_validation_artifact(
        &session,
        &geometry_response,
        &readiness,
        &model,
        format!("zero-shot-{}-{validated_at_ms}", session.scene_id),
        validated_at_ms,
        req.warmup_secs.unwrap_or(DEFAULT_ZERO_SHOT_WARMUP_SECS),
        req.validation_secs
            .unwrap_or(DEFAULT_ZERO_SHOT_VALIDATION_SECS),
    );

    let (compare_dataset_ids, benchmark_summary_path) =
        resolve_zero_shot_compare_dataset_ids(&req).await?;
    let explicit_candidate_sona_profile = req
        .candidate_sona_profile
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
        .map(ToOwned::to_owned);
    if zero_shot_compare_can_bypass_geometry_blocker(
        &validation,
        &compare_dataset_ids,
        explicit_candidate_sona_profile.as_deref(),
    ) {
        validation.status = "preflight_ready".to_string();
        validation.compare_status = "not_wired".to_string();
        validation.message =
            "当前 scene 尚未保存 geometry，但已显式指定 geometry-conditioned SONA profile 和 validation recordings；先按 profile compare 执行。"
                .to_string();
        validation
            .blockers
            .retain(|blocker| blocker != "当前 scene 还没有保存 geometry。");
    }
    if !compare_dataset_ids.is_empty() && validation.status != "blocked" {
        let base_sona_profile: Option<String> = None;
        let candidate_model_id = req
            .candidate_model_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_string())
            .or_else(|| validation.live_model_id.clone());
        validation.base_model_id =
            resolve_zero_shot_base_model_id(&validation, &model, candidate_model_id.as_ref());
        let base_model_id = validation.base_model_id.clone();
        let candidate_sona_profile = resolve_zero_shot_candidate_sona_profile(
            &req,
            &session,
            &geometry_response,
            &sona_state.profiles,
        );

        validation.validation_mode = "recording_compare_requested".to_string();

        if explicit_candidate_sona_profile.is_some() && candidate_sona_profile.is_none() {
            validation.status = "compare_blocked".to_string();
            validation.compare_status = "profile_missing".to_string();
            validation.message =
                "指定的 geometry-conditioned SONA profile 当前不存在于 active RVF 中。".to_string();
            validation.compare_summary = Some(json!({
                "source": "recording_compare_v1",
                "dataset_ids": compare_dataset_ids,
                "dataset_count": compare_dataset_ids.len(),
                "benchmark_summary_path": benchmark_summary_path,
                "base_model_id": validation.base_model_id,
                "candidate_model_id": req.candidate_model_id.clone().or(validation.live_model_id.clone()),
                "base_sona_profile": base_sona_profile,
                "candidate_sona_profile": explicit_candidate_sona_profile,
                "available_sona_profiles": sona_state.profiles,
                "passed": false,
                "blocked_reason": "profile_missing",
                "notes": req.notes,
            }));
        } else {
            match (base_model_id, candidate_model_id) {
                (Some(base_model_id), Some(candidate_model_id))
                    if base_model_id != candidate_model_id
                        || base_sona_profile != candidate_sona_profile =>
                {
                    let base_eval_model_hint = resolve_zero_shot_eval_model_hint(
                        &base_model_id,
                        base_sona_profile.as_deref(),
                        &model,
                    );
                    let candidate_eval_model_hint = resolve_zero_shot_eval_model_hint(
                        &candidate_model_id,
                        candidate_sona_profile.as_deref(),
                        &model,
                    );
                    let base_eval = run_zero_shot_recording_eval(
                        &state,
                        &validation.validation_id,
                        &compare_dataset_ids,
                        &base_eval_model_hint,
                        base_sona_profile.as_deref(),
                        req.notes.as_deref(),
                        "edge_orchestrator_zero_shot_base_compare",
                    )
                    .await?;
                    let candidate_eval = run_zero_shot_recording_eval(
                        &state,
                        &validation.validation_id,
                        &compare_dataset_ids,
                        &candidate_eval_model_hint,
                        candidate_sona_profile.as_deref(),
                        req.notes.as_deref(),
                        "edge_orchestrator_zero_shot_candidate_compare",
                    )
                    .await?;
                    let compare_summary = build_zero_shot_compare_summary(
                        &compare_dataset_ids,
                        benchmark_summary_path.clone(),
                        &base_model_id,
                        &candidate_model_id,
                        base_sona_profile.as_deref(),
                        candidate_sona_profile.as_deref(),
                        &base_eval,
                        &candidate_eval,
                        req.notes.clone(),
                    );
                    finalize_zero_shot_validation_with_compare(&mut validation, compare_summary);
                }
                (Some(base_model_id), Some(candidate_model_id)) => {
                    validation.status = "compare_blocked".to_string();
                    validation.compare_status = "same_model".to_string();
                    validation.message = "当前 candidate 与 base 的 model/profile 组合相同；无法生成 zero-shot compare。请显式指定不同的 candidate_model_id，或提供 geometry-conditioned SONA profile。".to_string();
                    validation.compare_summary = Some(json!({
                        "source": "recording_compare_v1",
                        "dataset_ids": compare_dataset_ids,
                        "dataset_count": compare_dataset_ids.len(),
                        "benchmark_summary_path": benchmark_summary_path,
                        "base_model_id": base_model_id,
                        "candidate_model_id": candidate_model_id,
                        "base_sona_profile": base_sona_profile,
                        "candidate_sona_profile": candidate_sona_profile,
                        "passed": false,
                        "blocked_reason": "same_model",
                        "notes": req.notes,
                    }));
                }
                _ => {
                    validation.status = "compare_blocked".to_string();
                    validation.compare_status = "candidate_missing".to_string();
                    validation.message = "zero-shot compare 需要 base model 和 geometry-conditioned candidate model；当前至少缺少一个。".to_string();
                    validation.compare_summary = Some(json!({
                        "source": "recording_compare_v1",
                        "dataset_ids": compare_dataset_ids,
                        "dataset_count": compare_dataset_ids.len(),
                        "benchmark_summary_path": benchmark_summary_path,
                        "base_model_id": validation.base_model_id,
                        "candidate_model_id": req.candidate_model_id.clone().or(validation.live_model_id.clone()),
                        "base_sona_profile": base_sona_profile,
                        "candidate_sona_profile": candidate_sona_profile,
                        "passed": false,
                        "blocked_reason": "candidate_missing",
                        "notes": req.notes,
                    }));
                }
            }
        }
    }
    sync_zero_shot_promotion_gate(&mut validation, &model);
    let auto_apply_enabled = req
        .auto_apply
        .unwrap_or(state.config.zero_shot_auto_apply_enabled);
    let auto_rollback_enabled = req
        .auto_rollback_if_regressed
        .unwrap_or(state.config.zero_shot_auto_rollback_enabled);
    let auto_apply_min_improvement_mm =
        f64::from(state.config.zero_shot_auto_apply_min_improvement_mm);
    let _ = maybe_run_zero_shot_auto_policy(
        &state,
        &mut session,
        &mut validation,
        auto_apply_enabled,
        auto_rollback_enabled,
        auto_apply_min_improvement_mm,
    )
    .await?;
    save_zero_shot_validation_report(&state, &validation).await?;
    session.latest_zero_shot_validation_id = Some(validation.validation_id.clone());
    session.updated_at_ms = session.updated_at_ms.max(validated_at_ms);
    save_session(&state, &session).await?;

    Ok(Json(validation))
}

async fn post_apply_zero_shot(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "训练还在进行中，请等当前训练稳定后再应用 zero-shot profile。".to_string(),
        ));
    }

    let mut session = load_session(&state).await;
    let validation_id = session
        .latest_zero_shot_validation_id
        .clone()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 zero-shot validation artifact，请先运行 compare。".to_string(),
            )
        })?;
    let mut report = load_zero_shot_validation_report(&state, &validation_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "zero-shot validation 报告不存在，请重新运行 compare。".to_string(),
            )
        })?;
    let model = load_runtime_model_summary(&state).await?;
    sync_zero_shot_promotion_gate(&mut report, &model);

    if report.promotion_gate_status != "eligible_for_apply"
        && report.promotion_gate_status != "promoted"
    {
        let reason = report
            .promotion_gate_reason
            .clone()
            .unwrap_or_else(|| "zero-shot candidate 还没有达到应用门槛。".to_string());
        save_zero_shot_validation_report(&state, &report).await?;
        return Err((StatusCode::BAD_REQUEST, reason));
    }

    let candidate_profile = zero_shot_candidate_profile_from_summary(
        report.compare_summary.as_ref(),
    )
    .ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "当前 zero-shot compare 没有关联到可应用的 geometry-conditioned profile。".to_string(),
        )
    })?;
    let response = sensing_post_json(
        &state,
        "/api/v1/model/sona/activate",
        json!({ "profile": candidate_profile }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("应用 zero-shot geometry profile 失败。")
                .to_string(),
        ));
    }

    let applied_at_ms = now_ms();
    report.applied_at_ms = Some(applied_at_ms);
    report.rolled_back_at_ms = None;
    report.status = "promoted".to_string();
    session.applied_at_ms = Some(applied_at_ms);
    session.status = "applied".to_string();
    session.updated_at_ms = applied_at_ms;
    save_zero_shot_validation_report(&state, &report).await?;
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_rollback_zero_shot(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let mut session = load_session(&state).await;
    let validation_id = session
        .latest_zero_shot_validation_id
        .clone()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 zero-shot validation artifact。".to_string(),
            )
        })?;
    let mut report = load_zero_shot_validation_report(&state, &validation_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "zero-shot validation 报告不存在，请重新运行 compare。".to_string(),
            )
        })?;
    let model = load_runtime_model_summary(&state).await?;
    sync_zero_shot_promotion_gate(&mut report, &model);

    let candidate_profile = zero_shot_candidate_profile_from_summary(
        report.compare_summary.as_ref(),
    )
    .ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "当前 zero-shot compare 没有关联到可回滚的 geometry-conditioned profile。".to_string(),
        )
    })?;

    if let Some(active_profile) = model.active_sona_profile.as_deref() {
        if active_profile != candidate_profile {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "当前 live 激活的是 `{active_profile}`，不是最近这次 zero-shot candidate `{candidate_profile}`。"
                ),
            ));
        }
    } else {
        save_zero_shot_validation_report(&state, &report).await?;
        return Ok(Json(build_current_response(&state).await));
    }

    let response = sensing_post_json(
        &state,
        "/api/v1/model/sona/activate",
        json!({ "profile": "default" }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("回滚 zero-shot geometry profile 失败。")
                .to_string(),
        ));
    }

    let rolled_back_at_ms = now_ms();
    report.rolled_back_at_ms = Some(rolled_back_at_ms);
    session.updated_at_ms = rolled_back_at_ms;
    save_zero_shot_validation_report(&state, &report).await?;
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_calibrate_few_shot(
    State(state): State<AppState>,
    Json(req): Json<EvolutionFewShotCalibrationRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "当前已有训练任务在运行，请等当前任务结束后再发起 few-shot 校准。".to_string(),
        ));
    }

    let mut session = load_session(&state).await;
    if session.capture_session_id.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "请先完成一轮环境采集，再开始 few-shot 校准。".to_string(),
        ));
    }

    let eligible_steps: Vec<&EvolutionStepState> = session
        .steps
        .iter()
        .filter(|step| step.status == "recorded" && !step.rerecord_recommended)
        .collect();
    let dataset_ids: Vec<String> = eligible_steps
        .iter()
        .filter_map(|step| step.recording_id.clone())
        .collect();
    if dataset_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "当前没有可用于 few-shot 校准的高质量样本，请先补录或重录。".to_string(),
        ));
    }

    let eligible_empty_count = eligible_steps
        .iter()
        .filter(|step| step.label == "empty")
        .count();
    let eligible_pose_count = eligible_steps
        .iter()
        .filter(|step| step.label == "pose")
        .count();
    if eligible_empty_count == 0 || eligible_pose_count == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot 校准当前仍要求同时保留空房间和动作样本，以便生成可回看的 candidate。"
                .to_string(),
        ));
    }

    let current_model_raw = sensing_get_json::<Value>(&state, "/api/v1/model/info")
        .await
        .unwrap_or_else(|_| json!({"status": "unavailable"}));
    let lora_profiles_raw = sensing_get_json::<Value>(&state, "/api/v1/models/lora/profiles")
        .await
        .unwrap_or_else(|_| json!({}));
    let sona_profiles_raw = sensing_get_json::<Value>(&state, "/api/v1/model/sona/profiles")
        .await
        .unwrap_or_else(|_| json!({}));
    let lora_state =
        serde_json::from_value::<ExternalLoraProfilesState>(lora_profiles_raw).unwrap_or_default();
    let sona_state =
        serde_json::from_value::<ExternalSonaProfilesState>(sona_profiles_raw).unwrap_or_default();
    let current_model = summarize_model_info(current_model_raw, &lora_state, &sona_state);
    let Some(base_model_id) = current_model.model_id.clone() else {
        return Err((
            StatusCode::BAD_REQUEST,
            "当前还没有加载 live model，few-shot 校准无法绑定 base model。".to_string(),
        ));
    };

    let requested_at_ms = now_ms();
    let calibration_id = format!("few-shot-{requested_at_ms}");
    let rank = req.rank.unwrap_or(DEFAULT_FEW_SHOT_LORA_RANK).max(1);
    let epochs = req.epochs.unwrap_or(DEFAULT_FEW_SHOT_LORA_EPOCHS).max(1);
    let profile_name = format!("{}-few-shot-{requested_at_ms}", session.scene_id);

    let response = sensing_post_json(
        &state,
        "/api/v1/train/lora",
        json!({
            "base_model_id": base_model_id,
            "dataset_ids": dataset_ids,
            "profile_name": profile_name,
            "rank": rank,
            "epochs": epochs,
        }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("few-shot 校准启动失败。")
                .to_string(),
        ));
    }

    let mut report = build_few_shot_candidate_artifact(
        calibration_id.clone(),
        requested_at_ms,
        &session,
        &eligible_steps,
        response
            .get("dataset_ids")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_else(|| {
                eligible_steps
                    .iter()
                    .filter_map(|step| step.recording_id.clone())
                    .collect()
            }),
        &current_model,
        profile_name,
        rank,
        epochs,
        DEFAULT_MIN_RECORDING_QUALITY,
    )?;
    sync_few_shot_promotion_gate(&mut report, &session);
    save_few_shot_candidate_report(&state, &report).await?;

    session.latest_few_shot_candidate_id = Some(calibration_id);
    session.latest_applied_few_shot_candidate_id = None;
    session.active_training_kind = Some("few_shot_lora".to_string());
    session.status = "training".to_string();
    session.updated_at_ms = requested_at_ms;
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_preview_few_shot(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot 校准还在进行中，请等 candidate 生成后再预览。".to_string(),
        ));
    }

    let mut session = load_session(&state).await;
    let calibration_id = session
        .latest_few_shot_candidate_id
        .clone()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 few-shot candidate，请先运行校准。".to_string(),
            )
        })?;
    let mut report = load_few_shot_candidate_report(&state, &calibration_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "few-shot candidate 报告不存在，请重新发起校准。".to_string(),
            )
        })?;

    if report.candidate_model_id.is_none() {
        if let Some(candidate_model) = find_latest_few_shot_candidate_model(&state, &report).await {
            report.candidate_model_id = Some(candidate_model.id.clone());
            report.candidate_model_created_at = if candidate_model.created_at.trim().is_empty() {
                None
            } else {
                Some(candidate_model.created_at)
            };
            report.completed_at_ms.get_or_insert(now_ms());
            report.after_metrics = Some(json!({
                "candidate_model_id": candidate_model.id,
                "best_pck": report.best_pck.or(candidate_model.pck_score),
                "best_epoch": report.best_epoch,
            }));
            report.status = "candidate".to_string();
            let _ = save_few_shot_candidate_report(&state, &report).await;
        }
    }

    let candidate_model_id = report.candidate_model_id.clone().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "few-shot candidate 还没导出成可加载模型，请稍后刷新再试。".to_string(),
        )
    })?;
    let response = sensing_post_json(
        &state,
        "/api/v1/models/load",
        json!({ "model_id": candidate_model_id }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("预览 few-shot candidate 失败。")
                .to_string(),
        ));
    }

    let previewed_at_ms = now_ms();
    if session.latest_applied_few_shot_candidate_id.as_deref() == Some(calibration_id.as_str()) {
        report.status = "active".to_string();
    } else {
        report.status = "preview_active".to_string();
    }
    report.previewed_at_ms = Some(previewed_at_ms);
    report.rolled_back_at_ms = None;
    sync_few_shot_promotion_gate(&mut report, &session);
    save_few_shot_candidate_report(&state, &report).await?;

    session.updated_at_ms = previewed_at_ms;
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_rollback_few_shot(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let mut session = load_session(&state).await;
    let calibration_id = session
        .latest_few_shot_candidate_id
        .clone()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 few-shot candidate，无需回滚。".to_string(),
            )
        })?;
    let mut report = load_few_shot_candidate_report(&state, &calibration_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "few-shot candidate 报告不存在，请重新发起校准。".to_string(),
            )
        })?;

    let response = sensing_post_json(
        &state,
        "/api/v1/models/load",
        json!({ "model_id": report.base_model_id }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("回滚到基础模型失败。")
                .to_string(),
        ));
    }

    let rolled_back_at_ms = now_ms();
    report.status = "rolled_back".to_string();
    report.rolled_back_at_ms = Some(rolled_back_at_ms);
    session.latest_applied_few_shot_candidate_id = None;
    sync_few_shot_promotion_gate(&mut report, &session);
    save_few_shot_candidate_report(&state, &report).await?;

    session.updated_at_ms = rolled_back_at_ms;
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_report_few_shot_evaluation(
    State(state): State<AppState>,
    Json(req): Json<EvolutionFewShotEvaluationReportRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot 校准还在进行中，请等 candidate 生成后再回填 evaluator 报告。".to_string(),
        ));
    }

    let session = load_session(&state).await;
    let calibration_id = req
        .calibration_id
        .clone()
        .or_else(|| session.latest_few_shot_candidate_id.clone())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 few-shot candidate，无法回填 evaluator 报告。".to_string(),
            )
        })?;
    let mut report = load_few_shot_candidate_report(&state, &calibration_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "few-shot candidate 报告不存在，请重新发起校准。".to_string(),
            )
        })?;

    report.evaluator_summary = build_few_shot_evaluator_summary(&report);
    report.cross_domain_summary = Some(build_few_shot_cross_domain_summary_from_report(
        &req,
        "manual_report_v1",
    ));
    sync_few_shot_promotion_gate(&mut report, &session);
    save_few_shot_candidate_report(&state, &report).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_import_few_shot_evaluation(
    State(state): State<AppState>,
    Json(req): Json<EvolutionFewShotEvaluationImportRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot 校准还在进行中，请等 candidate 生成后再导入 evaluator artifact。".to_string(),
        ));
    }

    let source_artifact_path = req
        .artifact_path
        .clone()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let artifact = if let Some(artifact) = req.artifact {
        artifact
    } else if let Some(path) = source_artifact_path.as_deref() {
        let content = tokio::fs::read_to_string(path).await.map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                format!("读取 evaluator artifact 失败: {error}"),
            )
        })?;
        serde_json::from_str::<Value>(&content).map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                format!("evaluator artifact 不是合法 JSON: {error}"),
            )
        })?
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            "请提供 artifact_path 或 artifact。".to_string(),
        ));
    };

    Ok(Json(
        import_few_shot_artifact(
            &state,
            req.calibration_id.clone(),
            source_artifact_path,
            artifact,
        )
        .await?,
    ))
}

async fn post_import_latest_few_shot_evaluation(
    State(state): State<AppState>,
    Json(req): Json<EvolutionFewShotLatestEvaluationImportRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot 校准还在进行中，请等 candidate 生成后再导入最新 evaluator artifact。"
                .to_string(),
        ));
    }

    let session = load_session(&state).await;
    let calibration_id = req
        .calibration_id
        .clone()
        .or_else(|| session.latest_few_shot_candidate_id.clone())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 few-shot candidate，无法导入最新 evaluator artifact。".to_string(),
            )
        })?;
    let (artifact_path, artifact) =
        load_latest_few_shot_inbox_artifact(&state, &calibration_id).await?;

    Ok(Json(
        import_few_shot_artifact(&state, Some(calibration_id), Some(artifact_path), artifact)
            .await?,
    ))
}

async fn post_evaluate_few_shot(
    State(state): State<AppState>,
    Json(req): Json<EvolutionFewShotEvaluateRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot 校准还在进行中，请等 candidate 生成后再运行 evaluator。".to_string(),
        ));
    }

    let session = load_session(&state).await;
    let calibration_id = req
        .calibration_id
        .clone()
        .or_else(|| session.latest_few_shot_candidate_id.clone())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 few-shot candidate，无法运行 evaluator。".to_string(),
            )
        })?;
    let mut report = load_few_shot_candidate_report(&state, &calibration_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "few-shot candidate 报告不存在，请重新发起校准。".to_string(),
            )
        })?;

    if report.candidate_model_id.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot candidate 还没导出成可加载模型，请稍后刷新再试。".to_string(),
        ));
    }

    report.evaluator_summary = build_few_shot_evaluator_summary(&report);
    report.cross_domain_summary = Some(build_few_shot_cross_domain_summary_from_samples(&req)?);
    sync_few_shot_promotion_gate(&mut report, &session);
    save_few_shot_candidate_report(&state, &report).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn prepare_few_shot_recording_evaluation(
    state: &AppState,
    calibration_id_hint: Option<String>,
    busy_message: &str,
    unavailable_message: &str,
) -> Result<(String, EvolutionFewShotCandidateArtifact), (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((StatusCode::BAD_REQUEST, busy_message.to_string()));
    }

    let session = load_session(state).await;
    let calibration_id = calibration_id_hint
        .or_else(|| session.latest_few_shot_candidate_id.clone())
        .ok_or_else(|| (StatusCode::BAD_REQUEST, unavailable_message.to_string()))?;
    let mut report = load_few_shot_candidate_report(state, &calibration_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "few-shot candidate 报告不存在，请重新发起校准。".to_string(),
            )
        })?;

    if report.candidate_model_id.is_none() {
        if let Some(candidate_model) = find_latest_few_shot_candidate_model(state, &report).await {
            report.candidate_model_id = Some(candidate_model.id.clone());
            report.candidate_model_created_at = if candidate_model.created_at.trim().is_empty() {
                None
            } else {
                Some(candidate_model.created_at)
            };
            report.completed_at_ms.get_or_insert(now_ms());
            report.after_metrics = Some(json!({
                "candidate_model_id": candidate_model.id,
                "best_pck": report.best_pck.or(candidate_model.pck_score),
                "best_epoch": report.best_epoch,
            }));
            save_few_shot_candidate_report(state, &report).await?;
        }
    }

    if report.candidate_model_id.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "few-shot candidate 还没导出成可加载模型，请稍后刷新再试。".to_string(),
        ));
    }

    Ok((calibration_id, report))
}

async fn execute_few_shot_recording_eval(
    state: &AppState,
    calibration_id: String,
    eval_body: Value,
) -> Result<EvolutionCurrentResponse, (StatusCode, String)> {
    let response = sensing_post_json(state, "/api/v1/eval/cross-domain-recordings", eval_body)
        .await
        .map_err(|error| {
            (
                StatusCode::BAD_GATEWAY,
                format!("调用 recording evaluator 失败: {error}"),
            )
        })?;
    if response
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        != "ok"
    {
        let message = response
            .get("message")
            .and_then(|value| value.as_str())
            .unwrap_or("recording evaluator 没有返回可用 artifact。")
            .to_string();
        return Err((StatusCode::BAD_GATEWAY, message));
    }
    let artifact = response.get("artifact").cloned().ok_or_else(|| {
        (
            StatusCode::BAD_GATEWAY,
            "recording evaluator 没有返回 artifact。".to_string(),
        )
    })?;
    if !artifact.is_object() {
        return Err((
            StatusCode::BAD_GATEWAY,
            "recording evaluator 返回的 artifact 不是合法对象。".to_string(),
        ));
    }

    import_few_shot_artifact(
        state,
        Some(calibration_id),
        Some("sensing:/api/v1/eval/cross-domain-recordings".to_string()),
        artifact,
    )
    .await
}

async fn post_evaluate_few_shot_recordings(
    State(state): State<AppState>,
    Json(req): Json<EvolutionFewShotEvaluateRecordingsRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let (calibration_id, report) = prepare_few_shot_recording_evaluation(
        &state,
        req.calibration_id.clone(),
        "few-shot 校准还在进行中，请等 candidate 生成后再运行 recording evaluator。",
        "当前还没有 few-shot candidate，无法运行 recording evaluator。",
    )
    .await?;
    let eval_body = build_few_shot_recording_eval_body(&req, &report)?;
    Ok(Json(
        execute_few_shot_recording_eval(&state, calibration_id, eval_body).await?,
    ))
}

async fn post_evaluate_few_shot_benchmark_summaries(
    State(state): State<AppState>,
    Json(req): Json<EvolutionFewShotEvaluateBenchmarkSummariesRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let (calibration_id, report) = prepare_few_shot_recording_evaluation(
        &state,
        req.calibration_id.clone(),
        "few-shot 校准还在进行中，请等 candidate 生成后再运行 benchmark summary evaluator。",
        "当前还没有 few-shot candidate，无法运行 benchmark summary evaluator。",
    )
    .await?;
    let eval_body =
        build_few_shot_recording_eval_body_from_benchmark_summaries(&req, &report).await?;
    Ok(Json(
        execute_few_shot_recording_eval(&state, calibration_id, eval_body).await?,
    ))
}

async fn post_apply_few_shot(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "训练还在进行中，请等 few-shot candidate 稳定后再应用。".to_string(),
        ));
    }

    let mut session = load_session(&state).await;
    let calibration_id = session
        .latest_few_shot_candidate_id
        .clone()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "当前还没有 few-shot candidate，请先运行校准。".to_string(),
            )
        })?;
    let mut report = load_few_shot_candidate_report(&state, &calibration_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "few-shot candidate 报告不存在，请重新发起校准。".to_string(),
            )
        })?;

    if report.candidate_model_id.is_none() {
        if let Some(candidate_model) = find_latest_few_shot_candidate_model(&state, &report).await {
            report.candidate_model_id = Some(candidate_model.id.clone());
            report.candidate_model_created_at = if candidate_model.created_at.trim().is_empty() {
                None
            } else {
                Some(candidate_model.created_at)
            };
            report.completed_at_ms.get_or_insert(now_ms());
            report.after_metrics = Some(json!({
                "candidate_model_id": candidate_model.id,
                "best_pck": report.best_pck.or(candidate_model.pck_score),
                "best_epoch": report.best_epoch,
            }));
        }
    }

    report.evaluator_summary = build_few_shot_evaluator_summary(&report);
    if report.cross_domain_summary.is_none() && report.candidate_model_id.is_some() {
        report.cross_domain_summary = Some(build_missing_few_shot_cross_domain_summary());
    }
    sync_few_shot_promotion_gate(&mut report, &session);

    if report.promotion_gate_status != "eligible_for_apply"
        && report.promotion_gate_status != "promoted"
    {
        let reason = report
            .promotion_gate_reason
            .clone()
            .unwrap_or_else(|| "few-shot candidate 还没有达到应用门槛。".to_string());
        save_few_shot_candidate_report(&state, &report).await?;
        return Err((StatusCode::BAD_REQUEST, reason));
    }

    let candidate_model_id = report.candidate_model_id.clone().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "few-shot candidate 还没导出成可加载模型，请稍后刷新再试。".to_string(),
        )
    })?;
    let response = sensing_post_json(
        &state,
        "/api/v1/models/load",
        json!({ "model_id": candidate_model_id }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("应用 few-shot candidate 失败。")
                .to_string(),
        ));
    }

    let applied_at_ms = now_ms();
    report.status = "active".to_string();
    report.applied_at_ms = Some(applied_at_ms);
    report.rolled_back_at_ms = None;
    session.applied_at_ms = Some(applied_at_ms);
    session.latest_applied_few_shot_candidate_id = Some(calibration_id);
    session.status = "applied".to_string();
    session.updated_at_ms = applied_at_ms;
    sync_few_shot_promotion_gate(&mut report, &session);
    save_few_shot_candidate_report(&state, &report).await?;
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_session_start(
    State(state): State<AppState>,
    Json(req): Json<EvolutionSessionStartRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let now = now_ms();
    let scene_id = normalize_scene_id(req.scene_id.as_deref());
    let scene_name = req
        .scene_name
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_SCENE_NAME.to_string());

    let session = EvolutionSessionState {
        capture_session_id: Some(format!("evolution-{now}")),
        scene_id,
        scene_name,
        status: "collecting".to_string(),
        created_at_ms: Some(now),
        updated_at_ms: now,
        current_step_code: default_steps().first().map(|step| step.code.clone()),
        training_requested_at_ms: None,
        applied_at_ms: None,
        latest_training_report_id: None,
        latest_applied_report_id: None,
        latest_zero_shot_validation_id: None,
        latest_few_shot_candidate_id: None,
        latest_applied_few_shot_candidate_id: None,
        active_training_kind: None,
        steps: default_steps(),
        step_runtimes: Vec::new(),
    };
    let mut session = session;
    sync_step_runtimes(&mut session);
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_step_start(
    State(state): State<AppState>,
    Json(req): Json<EvolutionStepStartRequest>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let mut session = load_session(&state).await;
    sync_step_runtimes(&mut session);
    let Some(capture_session_id) = session.capture_session_id.clone() else {
        return Err((
            StatusCode::BAD_REQUEST,
            "请先开始一次环境自进化流程。".to_string(),
        ));
    };
    let Some(step_index) = session
        .steps
        .iter()
        .position(|step| step.code == req.step_code)
    else {
        return Err((StatusCode::NOT_FOUND, "未找到这个采集动作。".to_string()));
    };
    let step_snapshot = session.steps[step_index].clone();

    let readiness = {
        let vision = state.vision.snapshot(state.config.vision_stale_ms);
        let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
        let wifi = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
        let csi = state.csi.snapshot(state.config.csi_stale_ms);
        let operator = state.operator.snapshot(state.config.operator_hold_ms);
        (
            build_readiness(
                &vision,
                &stereo,
                &wifi,
                &csi,
                &operator,
                robot_capture_ready(&state),
            ),
            operator,
            stereo,
        )
    };
    let (readiness, operator, stereo) = readiness;

    if step_snapshot.status != "recorded"
        && session.current_step_code.as_deref() != Some(step_snapshot.code.as_str())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "请先完成当前步骤：{}。",
                session.current_step_code.as_deref().unwrap_or("当前动作")
            ),
        ));
    }

    let guide = build_step_guide(
        &step_snapshot,
        &readiness,
        &operator,
        &stereo,
        step_runtime(&session, &step_snapshot.code),
    );
    if !guide.ready {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("{} {}", guide.headline, guide.detail),
        ));
    }

    let quality_score = if step_snapshot.label == "empty" {
        if readiness.room_empty_ready {
            1.0
        } else {
            0.55
        }
    } else {
        readiness.suggested_quality_score
    };

    let response = sensing_post_json(
        &state,
        "/api/v1/recording/start",
        json!({
            "session_name": format!("{}__{}", session.scene_id, step_snapshot.code),
            "label": step_snapshot.label,
            "duration_secs": step_snapshot.duration_secs,
            "scene_id": session.scene_id,
            "scene_name": session.scene_name,
            "capture_session_id": capture_session_id,
            "step_code": step_snapshot.code,
            "quality_score": quality_score,
        }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("启动录制失败。")
                .to_string(),
        ));
    }

    let now = now_ms();
    let step = &mut session.steps[step_index];
    step.status = "recording".to_string();
    step.recording_id = response
        .get("session_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    step.frame_count = Some(0);
    step.quality_score = Some(quality_score);
    step.started_at_ms = Some(now);
    step.completed_at_ms = None;
    session.status = "recording".to_string();
    session.updated_at_ms = now;
    session.current_step_code = Some(step_snapshot.code.clone());
    if let Some(runtime) = step_runtime_mut(&mut session, &step_snapshot.code) {
        clear_runtime_tracking(runtime);
        runtime.phase_total = step_phase_labels(&step_snapshot.code).len();
        if let Some(metrics) = body_metrics(&operator, &stereo) {
            seed_runtime_baseline(runtime, &metrics);
        }
    }
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_step_stop(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let response = sensing_post_json(&state, "/api/v1/recording/stop", json!({}))
        .await
        .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("停止录制失败。")
                .to_string(),
        ));
    }

    Ok(Json(build_current_response(&state).await))
}

async fn post_train(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let mut session = load_session(&state).await;
    if session.capture_session_id.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "请先完成一轮环境采集，再开始训练。".to_string(),
        ));
    }
    if session.steps.iter().any(|step| step.status != "recorded") {
        return Err((
            StatusCode::BAD_REQUEST,
            "还有动作没有录完，请先把当前引导步骤录齐。".to_string(),
        ));
    }

    let eligible_steps: Vec<&EvolutionStepState> = session
        .steps
        .iter()
        .filter(|step| step.status == "recorded" && !step.rerecord_recommended)
        .collect();
    let dataset_ids: Vec<String> = eligible_steps
        .iter()
        .filter_map(|step| step.recording_id.clone())
        .collect();
    if dataset_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "当前没有达到训练门槛的样本，请先按建议重录质量不足的动作。".to_string(),
        ));
    }

    let eligible_empty_count = eligible_steps
        .iter()
        .filter(|step| step.label == "empty")
        .count();
    let eligible_pose_count = eligible_steps
        .iter()
        .filter(|step| step.label == "pose")
        .count();
    if eligible_empty_count == 0 || eligible_pose_count == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "当前高质量样本还不够，至少需要空房间和动作样本各保留一部分。".to_string(),
        ));
    }

    let current_model_raw = sensing_get_json::<Value>(&state, "/api/v1/model/info")
        .await
        .unwrap_or_else(|_| json!({"status": "unavailable"}));
    let current_model = summarize_model_info(
        current_model_raw,
        &ExternalLoraProfilesState::default(),
        &ExternalSonaProfilesState::default(),
    );
    let requested_at_ms = now_ms();
    let train_job_id = format!("train-{requested_at_ms}");
    let supervised_base_model_hint = current_model
        .model_id
        .clone()
        .or_else(|| current_model.base_model_id.clone());

    let response = sensing_post_json(
        &state,
        "/api/v1/train/start",
        json!({
            "dataset_ids": dataset_ids,
            "scene_id": session.scene_id,
            "include_scene_history": true,
            "scene_history_limit": DEFAULT_SCENE_HISTORY_LIMIT,
            "min_recording_quality": DEFAULT_MIN_RECORDING_QUALITY,
            "config": {
                "epochs": 40,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "early_stopping_patience": 20,
                "warmup_epochs": 5,
                "pretrained_rvf": supervised_base_model_hint
            }
        }),
    )
    .await
    .map_err(|error| (StatusCode::BAD_GATEWAY, error))?;

    if response.get("status").and_then(Value::as_str) == Some("error") {
        return Err((
            StatusCode::BAD_REQUEST,
            response
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("训练启动失败。")
                .to_string(),
        ));
    }

    let report = build_training_report_artifact(
        train_job_id.clone(),
        requested_at_ms,
        &session,
        &eligible_steps,
        dataset_ids,
        &response,
        &current_model,
        DEFAULT_MIN_RECORDING_QUALITY,
    );
    save_training_report(&state, &report).await?;

    session.training_requested_at_ms = Some(requested_at_ms);
    session.applied_at_ms = None;
    session.latest_training_report_id = Some(train_job_id);
    session.latest_applied_report_id = None;
    session.latest_applied_few_shot_candidate_id = None;
    session.active_training_kind = Some("supervised".to_string());
    session.status = "training".to_string();
    session.updated_at_ms = requested_at_ms;
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

async fn post_apply(
    State(state): State<AppState>,
) -> Result<Json<EvolutionCurrentResponse>, (StatusCode, String)> {
    let training = sensing_get_json::<ExternalTrainingStatus>(&state, "/api/v1/train/status")
        .await
        .unwrap_or_default();
    if training.active {
        return Err((
            StatusCode::BAD_REQUEST,
            "训练还在进行中，请等训练结束后再应用到当前环境。".to_string(),
        ));
    }

    let mut session = load_session(&state).await;
    let train_job_id = session.latest_training_report_id.clone().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "当前还没有可应用的 supervised candidate。".to_string(),
        )
    })?;
    let report = load_training_report(&state, &train_job_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                "训练报告不存在，请重新发起训练。".to_string(),
            )
        })?;
    let candidate_model_id = report.candidate_model_id.clone().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "supervised candidate 还没导出成可加载模型，请稍后刷新再试。".to_string(),
        )
    })?;

    let output = Command::new("bash")
        .arg("-lc")
        .arg(
            "set -euo pipefail; \
             source /etc/default/chek-edge-autostart; \
             : \"${CHEK_EDGE_ROOT:?missing CHEK_EDGE_ROOT}\"; \
             : \"${CHEK_WIFI_MODEL_PATH:?missing CHEK_WIFI_MODEL_PATH}\"; \
             candidate_path=\"${CHEK_EDGE_ROOT}/RuView/rust-port/wifi-densepose-rs/data/models/${CHEK_EVOLUTION_CANDIDATE_MODEL_ID}.rvf\"; \
             if [[ ! -f \"${candidate_path}\" ]]; then \
               echo \"supervised candidate not found: ${candidate_path}\" >&2; \
               exit 12; \
             fi; \
             cp -f \"${candidate_path}\" \"${CHEK_WIFI_MODEL_PATH}\"; \
             sudo -n systemctl restart chek-edge-wifi-sensing.service",
        )
        .env("CHEK_EVOLUTION_CANDIDATE_MODEL_ID", &candidate_model_id)
        .output()
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("重启 Wi‑Fi sensing 服务失败: {error}"),
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err((
            StatusCode::BAD_GATEWAY,
            if stderr.is_empty() {
                "应用新模型失败：无法重启 Wi‑Fi sensing 服务。".to_string()
            } else {
                format!("应用新模型失败：{stderr}")
            },
        ));
    }

    sleep(Duration::from_secs(2)).await;

    session.applied_at_ms = Some(now_ms());
    session.latest_applied_report_id = session.latest_training_report_id.clone();
    session.latest_applied_few_shot_candidate_id = None;
    session.status = "applied".to_string();
    session.updated_at_ms = now_ms();
    save_session(&state, &session).await?;

    Ok(Json(build_current_response(&state).await))
}

#[cfg(test)]
mod tests {
    use super::{
        benchmark_summary_recording_ids, body_metrics_from_body, build_few_shot_candidate_artifact,
        build_few_shot_cross_domain_summary_from_imported_artifact,
        build_few_shot_cross_domain_summary_from_samples, build_few_shot_evaluator_summary,
        build_few_shot_recording_eval_body,
        build_few_shot_recording_eval_body_from_benchmark_summaries,
        build_few_shot_report_request_from_value, build_missing_few_shot_cross_domain_summary,
        build_readiness, build_scene_geometry_auto_draft, build_zero_shot_compare_summary,
        build_zero_shot_validation_artifact, default_session,
        discover_latest_benchmark_summary_paths, few_shot_artifact_matches_calibration_id,
        finalize_zero_shot_validation_with_compare, normalize_scene_id,
        normalize_session_step_durations, phase_reached, phase_tracking_ready,
        resolve_zero_shot_candidate_sona_profile, resolve_zero_shot_compare_dataset_ids,
        seed_runtime_baseline, step_runtime_mut, summarize_model_info,
        sync_few_shot_promotion_gate, sync_step_runtimes, training_phase_failed,
        update_step_runtimes, validate_scene_geometry_request, BenchmarkSummary,
        EvolutionFewShotCandidateArtifact, EvolutionFewShotEvaluateBenchmarkSummariesRequest,
        EvolutionFewShotEvaluateRecordingsRequest, EvolutionFewShotEvaluateRequest,
        EvolutionFewShotEvaluationSample, EvolutionFewShotRecordingEvalBucketRequest,
        EvolutionModelSummary, EvolutionReadiness, EvolutionRecordingSummary,
        EvolutionSceneGeometryAnchor, EvolutionSceneGeometryResponse,
        EvolutionSceneGeometrySummary, EvolutionSceneGeometryUpsertRequest,
        EvolutionStepRuntimeState, EvolutionZeroShotValidationRequest, ExternalLoraProfilesState,
        ExternalSonaProfilesState, OperatorSnapshot, DEFAULT_SCENE_ID, DEFAULT_SCENE_NAME,
    };
    use crate::operator::OperatorEstimate;
    use crate::sensing::{
        BodyKeypointLayout, CsiSnapshot, StereoSnapshot, VisionDevicePose, VisionSnapshot,
        WifiPoseSnapshot,
    };
    use axum::http::StatusCode;
    use serde_json::json;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn default_operator_snapshot() -> OperatorSnapshot {
        OperatorSnapshot {
            estimate: OperatorEstimate::default(),
            fresh: false,
        }
    }

    fn sample_turn_left_body() -> Vec<[f32; 3]> {
        vec![
            [0.0, 1.72, 0.02],
            [-0.03, 1.75, 0.04],
            [0.03, 1.75, 0.04],
            [-0.06, 1.73, 0.02],
            [0.06, 1.73, 0.02],
            [-0.18, 1.52, 0.12],
            [0.18, 1.52, -0.08],
            [-0.22, 1.18, 0.03],
            [0.22, 1.18, 0.03],
            [-0.20, 0.92, 0.12],
            [0.20, 0.92, -0.08],
            [-0.14, 0.98, 0.08],
            [0.14, 0.98, -0.04],
            [-0.15, 0.54, 0.08],
            [0.15, 0.54, -0.04],
            [-0.16, 0.08, 0.08],
            [0.16, 0.08, -0.04],
        ]
    }

    fn sample_front_body() -> Vec<[f32; 3]> {
        vec![
            [0.0, 1.72, 0.02],
            [-0.03, 1.75, 0.04],
            [0.03, 1.75, 0.04],
            [-0.06, 1.73, 0.02],
            [0.06, 1.73, 0.02],
            [-0.18, 1.52, 0.02],
            [0.18, 1.52, 0.02],
            [-0.12, 1.18, 0.02],
            [0.12, 1.18, 0.02],
            [-0.1, 0.92, 0.02],
            [0.1, 0.92, 0.02],
            [-0.14, 0.98, 0.0],
            [0.14, 0.98, 0.0],
            [-0.15, 0.54, 0.01],
            [0.15, 0.54, 0.01],
            [-0.16, 0.08, 0.01],
            [0.16, 0.08, 0.01],
        ]
    }

    fn sample_turn_right_body() -> Vec<[f32; 3]> {
        vec![
            [0.0, 1.72, 0.02],
            [-0.03, 1.75, 0.04],
            [0.03, 1.75, 0.04],
            [-0.06, 1.73, 0.02],
            [0.06, 1.73, 0.02],
            [-0.18, 1.52, -0.08],
            [0.18, 1.52, 0.12],
            [-0.22, 1.18, 0.03],
            [0.22, 1.18, 0.03],
            [-0.20, 0.92, -0.08],
            [0.20, 0.92, 0.12],
            [-0.14, 0.98, -0.04],
            [0.14, 0.98, 0.08],
            [-0.15, 0.54, -0.04],
            [0.15, 0.54, 0.08],
            [-0.16, 0.08, -0.04],
            [0.16, 0.08, 0.08],
        ]
    }

    fn sample_reach_right_body() -> Vec<[f32; 3]> {
        let mut body = sample_front_body();
        body[10] = [0.36, 1.3, 0.18];
        body
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "edge-orchestrator-{prefix}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn write_benchmark_summary(
        root: &Path,
        directory_name: &str,
        recording_ids: &[&str],
    ) -> PathBuf {
        let directory = root.join(directory_name);
        fs::create_dir_all(&directory).expect("create benchmark dir");
        let steps = recording_ids
            .iter()
            .map(|recording_id| json!({ "recording_id": recording_id }))
            .collect::<Vec<_>>();
        let summary_path = directory.join("summary.json");
        fs::write(
            &summary_path,
            serde_json::to_vec(&json!({ "steps": steps })).expect("summary json"),
        )
        .expect("write benchmark summary");
        summary_path
    }

    #[test]
    fn normalize_scene_id_should_keep_formal_slug() {
        assert_eq!(
            normalize_scene_id(Some("chek_humanoid_lan_sense_main_stage_v1")),
            "chek_humanoid_lan_sense_main_stage_v1"
        );
    }

    #[test]
    fn normalize_scene_id_should_fallback_to_default_when_empty() {
        assert_eq!(normalize_scene_id(Some("   ")), DEFAULT_SCENE_ID);
    }

    #[test]
    fn default_session_should_start_idle() {
        let session = default_session("scene_a".to_string(), "Scene A".to_string());
        assert_eq!(session.status, "idle");
        assert_eq!(session.steps.len(), 8);
    }

    #[test]
    fn normalize_session_step_durations_should_upgrade_old_pose_windows() {
        let mut session = default_session("scene_a".to_string(), "Scene A".to_string());
        let turn = session
            .steps
            .iter_mut()
            .find(|step| step.code == "pose_turn_lr_01")
            .expect("turn step");
        turn.duration_secs = 20;
        let bend = session
            .steps
            .iter_mut()
            .find(|step| step.code == "pose_bend_squat_01")
            .expect("bend step");
        bend.duration_secs = 18;

        normalize_session_step_durations(&mut session);

        let turn = session
            .steps
            .iter()
            .find(|step| step.code == "pose_turn_lr_01")
            .expect("turn step normalized");
        let bend = session
            .steps
            .iter()
            .find(|step| step.code == "pose_bend_squat_01")
            .expect("bend step normalized");
        assert_eq!(turn.duration_secs, 48);
        assert_eq!(bend.duration_secs, 32);
    }

    #[test]
    fn scene_geometry_validation_requires_ap_and_stereo() {
        let invalid = EvolutionSceneGeometryUpsertRequest {
            coordinate_frame_version: "chek-room-v1".to_string(),
            ap_nodes: vec![],
            stereo_rig: None,
            phone_pose: None,
            notes: None,
        };
        assert!(validate_scene_geometry_request(&invalid).is_err());

        let valid = EvolutionSceneGeometryUpsertRequest {
            coordinate_frame_version: "chek-room-v1".to_string(),
            ap_nodes: vec![EvolutionSceneGeometryAnchor {
                id: "ap-01".to_string(),
                label: Some("主 AP".to_string()),
                position_m: [0.0, 2.4, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                notes: None,
            }],
            stereo_rig: Some(EvolutionSceneGeometryAnchor {
                id: "stereo-01".to_string(),
                label: Some("双目主机位".to_string()),
                position_m: [1.0, 1.4, 0.0],
                rotation_deg: [0.0, -90.0, 0.0],
                notes: None,
            }),
            phone_pose: None,
            notes: None,
        };
        assert!(validate_scene_geometry_request(&valid).is_ok());
    }

    #[test]
    fn auto_draft_should_use_stage_default_layout_for_default_scene() {
        let session = default_session(DEFAULT_SCENE_ID.to_string(), DEFAULT_SCENE_NAME.to_string());
        let draft = build_scene_geometry_auto_draft(&session, None, None);
        assert!(draft.ready_to_save);
        assert_eq!(draft.source_breakdown.ap_nodes, "stage_default_layout");
        assert_eq!(draft.source_breakdown.stereo_rig, "stage_default_layout");
        assert!(draft.confidence_breakdown.ap_nodes > 0.6);
        assert!(draft.confidence_breakdown.stereo_rig > 0.6);
        assert!(draft
            .warnings
            .iter()
            .any(|warning| warning.contains("CHEK 主采集区默认布局")));
    }

    #[test]
    fn auto_draft_should_keep_template_sources_for_non_default_scene() {
        let session = default_session("scene_a".to_string(), "Scene A".to_string());
        let draft = build_scene_geometry_auto_draft(&session, None, None);
        assert_eq!(draft.source_breakdown.ap_nodes, "template_placeholder");
        assert_eq!(draft.source_breakdown.stereo_rig, "template_default");
        assert!(draft.confidence_breakdown.ap_nodes < 0.4);
        assert!(draft.confidence_breakdown.stereo_rig < 0.4);
    }

    #[test]
    fn zero_shot_validation_artifact_should_report_blockers_honestly() {
        let session = default_session("scene_a".to_string(), "Scene A".to_string());
        let geometry = EvolutionSceneGeometryResponse {
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            exists: false,
            summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: None,
                ap_count: 0,
                stereo_defined: false,
                phone_defined: false,
                updated_at_ms: None,
            },
            geometry: None,
        };
        let readiness = EvolutionReadiness {
            csi_ready: false,
            stereo_ready: false,
            wifi_ready: true,
            phone_ready: false,
            robot_ready: false,
            teacher_ready: true,
            room_empty_ready: true,
            full_multimodal_ready: false,
            pose_capture_ready: false,
            anchor_source: "stereo+wifi_prior".to_string(),
            selected_operator_track_id: None,
            iphone_visible_hand_count: 0,
            hand_match_count: 0,
            hand_match_score: 0.0,
            left_wrist_gap_m: None,
            right_wrist_gap_m: None,
            suggested_quality_score: 0.64,
            issues: Vec::new(),
        };
        let model = EvolutionModelSummary {
            loaded: false,
            status: "unavailable".to_string(),
            model_id: None,
            target_space: None,
            scene_id: Some("scene_a".to_string()),
            base_model_id: None,
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "none".to_string(),
            sona_profiles: Vec::new(),
            active_sona_profile: None,
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let artifact = build_zero_shot_validation_artifact(
            &session,
            &geometry,
            &readiness,
            &model,
            "zero-shot-scene_a-1".to_string(),
            1,
            10,
            15,
        );

        assert_eq!(artifact.status, "blocked");
        assert_eq!(artifact.validation_mode, "preflight_only");
        assert_eq!(artifact.compare_status, "blocked");
        assert_eq!(artifact.promotion_gate_status, "preview_only");
        assert_eq!(artifact.blockers.len(), 6);
    }

    #[test]
    fn zero_shot_validation_artifact_should_mark_ready_preflight() {
        let session = default_session("scene_a".to_string(), "Scene A".to_string());
        let geometry = EvolutionSceneGeometryResponse {
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            exists: true,
            summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("chek-room-v1".to_string()),
                ap_count: 1,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(123),
            },
            geometry: None,
        };
        let readiness = EvolutionReadiness {
            csi_ready: true,
            stereo_ready: true,
            wifi_ready: true,
            phone_ready: true,
            robot_ready: true,
            teacher_ready: true,
            room_empty_ready: true,
            full_multimodal_ready: true,
            pose_capture_ready: true,
            anchor_source: "stereo+wifi_prior".to_string(),
            selected_operator_track_id: Some("stereo-person-1".to_string()),
            iphone_visible_hand_count: 0,
            hand_match_count: 0,
            hand_match_score: 0.0,
            left_wrist_gap_m: None,
            right_wrist_gap_m: None,
            suggested_quality_score: 0.88,
            issues: Vec::new(),
        };
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("scene_a_candidate".to_string()),
            target_space: None,
            scene_id: Some("scene_a".to_string()),
            base_model_id: Some("global_base_v1".to_string()),
            global_parent_model_id: None,
            best_pck: Some(0.83),
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: Some("scene_a_geometry".to_string()),
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let artifact = build_zero_shot_validation_artifact(
            &session,
            &geometry,
            &readiness,
            &model,
            "zero-shot-scene_a-2".to_string(),
            2,
            10,
            15,
        );

        assert_eq!(artifact.status, "preflight_ready");
        assert!(artifact.blockers.is_empty());
        assert_eq!(artifact.compare_status, "not_wired");
        assert_eq!(artifact.base_model_id.as_deref(), Some("global_base_v1"));
    }

    #[tokio::test]
    async fn zero_shot_compare_dataset_ids_should_merge_summary_and_inline_ids() {
        let root = unique_temp_dir("zero-shot-datasets");
        fs::create_dir_all(&root).expect("create root");
        let summary_path = write_benchmark_summary(
            &root,
            "wifi-pose-benchmark-validation-20260315-011500",
            &[" room-b-1 ", "room-b-2"],
        );

        let (dataset_ids, benchmark_summary_path) =
            resolve_zero_shot_compare_dataset_ids(&EvolutionZeroShotValidationRequest {
                warmup_secs: Some(5),
                validation_secs: Some(10),
                dataset_ids: vec!["room-b-2".to_string(), "room-b-3".to_string()],
                benchmark_summary_path: Some(summary_path.to_string_lossy().to_string()),
                candidate_model_id: None,
                candidate_sona_profile: None,
                auto_apply: None,
                auto_rollback_if_regressed: None,
                notes: None,
            })
            .await
            .expect("dataset ids");

        assert_eq!(
            dataset_ids,
            vec![
                "room-b-1".to_string(),
                "room-b-2".to_string(),
                "room-b-3".to_string()
            ]
        );
        assert_eq!(
            benchmark_summary_path.as_deref(),
            Some(summary_path.to_string_lossy().as_ref())
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn zero_shot_compare_summary_should_compute_improvement_delta() {
        let summary = build_zero_shot_compare_summary(
            &["room-b-1".to_string(), "room-b-2".to_string()],
            Some("/tmp/validation-summary.json".to_string()),
            "global_base_v1",
            "geometry_candidate_v1",
            None,
            Some("scene_a_geometry"),
            &json!({
                "sample_count": 24,
                "summary": {
                    "metrics": {
                        "buckets": {
                            "unseen_room_zero_shot": 72.0
                        }
                    }
                }
            }),
            &json!({
                "sample_count": 24,
                "summary": {
                    "metrics": {
                        "buckets": {
                            "unseen_room_zero_shot": 49.0
                        }
                    }
                }
            }),
            Some("synthetic zero-shot compare".to_string()),
        );

        assert_eq!(
            summary.get("passed").and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            summary
                .get("improvement_delta")
                .and_then(|value| value.as_f64())
                .map(|value| value.round() as i64),
            Some(23)
        );
        assert_eq!(
            summary
                .pointer("/sample_counts/base")
                .and_then(|value| value.as_u64()),
            Some(24)
        );
        assert_eq!(
            summary
                .get("candidate_sona_profile")
                .and_then(|value| value.as_str()),
            Some("scene_a_geometry")
        );
    }

    #[test]
    fn zero_shot_eval_helpers_should_accept_recording_cross_domain_summary_shape() {
        let response = json!({
            "status": "ok",
            "summary": {
                "bucket_metrics": {
                    "unseen_room_zero_shot": 3.724904216622559
                },
                "sample_counts": {
                    "unseen_room_zero_shot": 4214
                }
            }
        });

        assert_eq!(
            super::zero_shot_eval_metric(&response),
            Some(3.724904216622559)
        );
        assert_eq!(super::zero_shot_eval_sample_count(&response), 4214);
    }

    #[test]
    fn zero_shot_candidate_sona_profile_should_resolve_from_geometry_version() {
        let session = default_session("scene_a".to_string(), "Scene A".to_string());
        let geometry = EvolutionSceneGeometryResponse {
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            exists: true,
            summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("scene_a_geometry".to_string()),
                ap_count: 4,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(1),
            },
            geometry: None,
        };
        let req = EvolutionZeroShotValidationRequest::default();

        let resolved = resolve_zero_shot_candidate_sona_profile(
            &req,
            &session,
            &geometry,
            &["default".to_string(), "scene_a_geometry".to_string()],
        );

        assert_eq!(resolved.as_deref(), Some("scene_a_geometry"));
    }

    #[test]
    fn zero_shot_compare_can_bypass_geometry_blocker_for_explicit_profile() {
        let validation = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-scene_a-2".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 2,
            status: "blocked".to_string(),
            message: "当前 scene 还没有保存 geometry。".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "preflight_only".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: false,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("scene_a_base".to_string()),
            base_model_id: Some("scene_a_global_base".to_string()),
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: None,
                ap_count: 0,
                stereo_defined: false,
                phone_defined: false,
                updated_at_ms: None,
            },
            blockers: vec!["当前 scene 还没有保存 geometry。".to_string()],
            compare_status: "blocked".to_string(),
            compare_summary: None,
            promotion_gate_status: "preview_only".to_string(),
            promotion_gate_reason: None,
            applied_at_ms: None,
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };

        assert!(super::zero_shot_compare_can_bypass_geometry_blocker(
            &validation,
            &["recording_a".to_string()],
            Some("scene_a_geometry"),
        ));
        assert!(!super::zero_shot_compare_can_bypass_geometry_blocker(
            &validation,
            &["recording_a".to_string()],
            None,
        ));
    }

    #[test]
    fn zero_shot_base_model_resolution_should_fallback_to_candidate_for_profile_compare() {
        let validation = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-scene_a-2".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 2,
            status: "preflight_ready".to_string(),
            message: "ready".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "recording_compare_requested".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: false,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("scene_a_live".to_string()),
            base_model_id: None,
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: None,
                ap_count: 0,
                stereo_defined: false,
                phone_defined: false,
                updated_at_ms: None,
            },
            blockers: Vec::new(),
            compare_status: "not_wired".to_string(),
            compare_summary: None,
            promotion_gate_status: "preview_only".to_string(),
            promotion_gate_reason: None,
            applied_at_ms: None,
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: None,
            target_space: None,
            scene_id: None,
            base_model_id: None,
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: None,
            raw: json!({}),
        };

        let resolved = super::resolve_zero_shot_base_model_id(
            &validation,
            &model,
            Some(&"scene_a_live".to_string()),
        );

        assert_eq!(resolved.as_deref(), Some("scene_a_live"));
    }

    #[test]
    fn zero_shot_eval_model_hint_should_use_live_alias_for_profiled_live_model() {
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("trained-supervised-20260314_040438".to_string()),
            target_space: None,
            scene_id: None,
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: None,
            raw: json!({}),
        };

        let resolved = super::resolve_zero_shot_eval_model_hint(
            "trained-supervised-20260314_040438",
            Some("scene_a_geometry"),
            &model,
        );

        assert_eq!(resolved, "trained-supervised-live.rvf");
    }

    #[test]
    fn zero_shot_eval_model_hint_should_keep_plain_model_id_without_live_profile() {
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("trained-supervised-20260314_040438".to_string()),
            target_space: None,
            scene_id: None,
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            sona_profiles: vec![],
            active_sona_profile: None,
            raw: json!({}),
        };

        let resolved = super::resolve_zero_shot_eval_model_hint(
            "trained-supervised-20260314_040438",
            Some("scene_a_geometry"),
            &model,
        );

        assert_eq!(resolved, "trained-supervised-20260314_040438");
    }

    #[test]
    fn zero_shot_gate_should_be_eligible_when_compare_passes_and_profile_exists() {
        let mut artifact = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-scene_a-4".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 4,
            status: "shadow_validated".to_string(),
            message: "ok".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "recording_compare".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: true,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("trained-supervised-20260314_040438".to_string()),
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("scene_a_geometry".to_string()),
                ap_count: 1,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(1),
            },
            blockers: Vec::new(),
            compare_status: "measured".to_string(),
            compare_summary: Some(json!({
                "base_model_id": "trained-supervised-20260314_040438",
                "candidate_model_id": "trained-supervised-20260314_040438",
                "candidate_sona_profile": "scene_a_geometry",
                "base_metric": 4.2,
                "geometry_conditioned_metric": 4.0,
                "improvement_delta": 0.2,
                "passed": true
            })),
            promotion_gate_status: "preview_only".to_string(),
            promotion_gate_reason: None,
            applied_at_ms: None,
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("trained-supervised-20260314_040438".to_string()),
            target_space: None,
            scene_id: None,
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: None,
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        super::sync_zero_shot_promotion_gate(&mut artifact, &model);

        assert_eq!(artifact.promotion_gate_status, "eligible_for_apply");
        assert_eq!(artifact.status, "shadow_validated");
    }

    #[test]
    fn zero_shot_gate_should_mark_promoted_when_profile_is_active() {
        let mut artifact = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-scene_a-5".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 5,
            status: "shadow_validated".to_string(),
            message: "ok".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "recording_compare".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: true,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("trained-supervised-20260314_040438".to_string()),
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("scene_a_geometry".to_string()),
                ap_count: 1,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(1),
            },
            blockers: Vec::new(),
            compare_status: "measured".to_string(),
            compare_summary: Some(json!({
                "base_model_id": "trained-supervised-20260314_040438",
                "candidate_model_id": "trained-supervised-20260314_040438",
                "candidate_sona_profile": "scene_a_geometry",
                "base_metric": 4.2,
                "geometry_conditioned_metric": 4.0,
                "improvement_delta": 0.2,
                "passed": true
            })),
            promotion_gate_status: "eligible_for_apply".to_string(),
            promotion_gate_reason: None,
            applied_at_ms: Some(10),
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("trained-supervised-20260314_040438".to_string()),
            target_space: None,
            scene_id: None,
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: Some("scene_a_geometry".to_string()),
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        super::sync_zero_shot_promotion_gate(&mut artifact, &model);

        assert_eq!(artifact.promotion_gate_status, "promoted");
        assert_eq!(artifact.status, "promoted");
    }

    #[test]
    fn zero_shot_auto_policy_should_apply_when_gate_passes_and_delta_meets_threshold() {
        let artifact = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-auto-apply".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 6,
            status: "shadow_validated".to_string(),
            message: "ok".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "recording_compare".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: true,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("trained-supervised-live".to_string()),
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("scene_a_geometry".to_string()),
                ap_count: 1,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(1),
            },
            blockers: Vec::new(),
            compare_status: "measured".to_string(),
            compare_summary: Some(json!({
                "base_model_id": "trained-supervised-20260314_040438",
                "candidate_model_id": "trained-supervised-20260314_040438",
                "candidate_sona_profile": "scene_a_geometry",
                "base_metric": 3.78,
                "geometry_conditioned_metric": 3.72,
                "improvement_delta": 0.06,
                "passed": true
            })),
            promotion_gate_status: "eligible_for_apply".to_string(),
            promotion_gate_reason: None,
            applied_at_ms: None,
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("trained-supervised-live".to_string()),
            target_space: None,
            scene_id: None,
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: None,
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let decision = super::decide_zero_shot_auto_policy(&artifact, &model, true, true, 0.05);

        assert_eq!(decision.status, "apply_pending");
        assert!(decision.reason.contains("0.060"));
        assert_eq!(
            decision.action,
            Some(super::ZeroShotAutoPolicyAction::Apply(
                "scene_a_geometry".to_string()
            ))
        );
    }

    #[test]
    fn zero_shot_auto_policy_should_skip_below_threshold() {
        let artifact = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-auto-skip".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 7,
            status: "shadow_validated".to_string(),
            message: "ok".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "recording_compare".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: true,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("trained-supervised-live".to_string()),
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("scene_a_geometry".to_string()),
                ap_count: 1,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(1),
            },
            blockers: Vec::new(),
            compare_status: "measured".to_string(),
            compare_summary: Some(json!({
                "base_model_id": "trained-supervised-20260314_040438",
                "candidate_model_id": "trained-supervised-20260314_040438",
                "candidate_sona_profile": "scene_a_geometry",
                "base_metric": 3.78,
                "geometry_conditioned_metric": 3.74,
                "improvement_delta": 0.04,
                "passed": true
            })),
            promotion_gate_status: "eligible_for_apply".to_string(),
            promotion_gate_reason: None,
            applied_at_ms: None,
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("trained-supervised-live".to_string()),
            target_space: None,
            scene_id: None,
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: None,
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let decision = super::decide_zero_shot_auto_policy(&artifact, &model, true, true, 0.05);

        assert_eq!(decision.status, "skipped_below_threshold");
        assert!(decision.reason.contains("0.040"));
        assert_eq!(decision.action, None);
    }

    #[test]
    fn zero_shot_auto_policy_should_request_rollback_when_active_profile_regresses() {
        let artifact = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-auto-rollback".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 8,
            status: "shadow_regressed".to_string(),
            message: "regressed".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "recording_compare".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: true,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("trained-supervised-live".to_string()),
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("scene_a_geometry".to_string()),
                ap_count: 1,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(1),
            },
            blockers: Vec::new(),
            compare_status: "measured".to_string(),
            compare_summary: Some(json!({
                "base_model_id": "trained-supervised-20260314_040438",
                "candidate_model_id": "trained-supervised-20260314_040438",
                "candidate_sona_profile": "scene_a_geometry",
                "base_metric": 3.78,
                "geometry_conditioned_metric": 3.84,
                "improvement_delta": -0.06,
                "passed": false
            })),
            promotion_gate_status: "blocked_regressed".to_string(),
            promotion_gate_reason: Some(
                "geometry-conditioned path 还没有在 shadow compare 中稳定优于 base。".to_string(),
            ),
            applied_at_ms: Some(10),
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("trained-supervised-live".to_string()),
            target_space: None,
            scene_id: None,
            base_model_id: Some("trained-supervised-20260314_040438".to_string()),
            global_parent_model_id: None,
            best_pck: None,
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "geometry-conditioned".to_string(),
            sona_profiles: vec!["scene_a_geometry".to_string()],
            active_sona_profile: Some("scene_a_geometry".to_string()),
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let decision = super::decide_zero_shot_auto_policy(&artifact, &model, false, true, 0.05);

        assert_eq!(decision.status, "rollback_pending");
        assert!(decision.reason.contains("-0.060"));
        assert_eq!(
            decision.action,
            Some(super::ZeroShotAutoPolicyAction::Rollback(
                "scene_a_geometry".to_string()
            ))
        );
    }

    #[test]
    fn zero_shot_validation_compare_finalize_should_mark_shadow_validated() {
        let mut artifact = super::EvolutionZeroShotValidationArtifact {
            validation_id: "zero-shot-scene_a-3".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            validated_at_ms: 3,
            status: "preflight_ready".to_string(),
            message: "preflight".to_string(),
            adaptation_mode: "geometry-conditioned".to_string(),
            validation_mode: "preflight_only".to_string(),
            warmup_secs: 10,
            validation_secs: 15,
            geometry_ready: true,
            model_ready: true,
            readiness_ready: true,
            live_model_id: Some("geometry_candidate_v1".to_string()),
            base_model_id: Some("global_base_v1".to_string()),
            geometry_summary: EvolutionSceneGeometrySummary {
                coordinate_frame_version: Some("chek-room-v1".to_string()),
                ap_count: 1,
                stereo_defined: true,
                phone_defined: false,
                updated_at_ms: Some(123),
            },
            blockers: Vec::new(),
            compare_status: "not_wired".to_string(),
            compare_summary: None,
            promotion_gate_status: "preview_only".to_string(),
            promotion_gate_reason: None,
            applied_at_ms: None,
            rolled_back_at_ms: None,
            auto_policy_status: "manual_only".to_string(),
            auto_policy_reason: None,
            auto_apply_min_improvement_mm: None,
            auto_applied_at_ms: None,
            auto_rolled_back_at_ms: None,
        };

        finalize_zero_shot_validation_with_compare(
            &mut artifact,
            json!({
                "base_metric": 72.0,
                "geometry_conditioned_metric": 49.0,
                "improvement_delta": 23.0,
                "passed": true
            }),
        );

        assert_eq!(artifact.status, "shadow_validated");
        assert_eq!(artifact.compare_status, "measured");
        assert_eq!(artifact.validation_mode, "recording_compare");
        assert!(artifact.message.contains("优于 base"));
    }

    #[test]
    fn few_shot_candidate_artifact_should_bind_scene_and_base_model() {
        let mut session = default_session("scene_a".to_string(), "Scene A".to_string());
        session.capture_session_id = Some("capture-1".to_string());
        let step = session
            .steps
            .iter_mut()
            .find(|step| step.code == "pose_idle_front_02")
            .expect("pose step");
        step.status = "recorded".to_string();
        step.recording_id = Some("dataset-1".to_string());
        let eligible_steps = vec![&session.steps[2]];
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("global_base_v1".to_string()),
            target_space: Some("wifi-pose".to_string()),
            scene_id: Some("scene_a".to_string()),
            base_model_id: Some("global_parent".to_string()),
            global_parent_model_id: None,
            best_pck: Some(0.81),
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "none".to_string(),
            sona_profiles: Vec::new(),
            active_sona_profile: None,
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let artifact = build_few_shot_candidate_artifact(
            "few-shot-1".to_string(),
            1,
            &session,
            &eligible_steps,
            vec!["dataset-1".to_string()],
            &model,
            "scene_a-few-shot-1".to_string(),
            8,
            30,
            0.7,
        )
        .expect("few-shot artifact");

        assert_eq!(artifact.status, "running");
        assert_eq!(artifact.base_model_id, "global_base_v1");
        assert_eq!(artifact.source_dataset_ids, vec!["dataset-1".to_string()]);
        assert_eq!(artifact.promotion_gate_status, "candidate_only");
    }

    #[test]
    fn few_shot_gate_should_block_without_cross_domain_summary() {
        let mut session = default_session("scene_a".to_string(), "Scene A".to_string());
        session.capture_session_id = Some("capture-1".to_string());
        let step = session
            .steps
            .iter_mut()
            .find(|step| step.code == "pose_idle_front_02")
            .expect("pose step");
        step.status = "recorded".to_string();
        step.recording_id = Some("dataset-1".to_string());
        let eligible_steps = vec![&session.steps[2]];
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("global_base_v1".to_string()),
            target_space: Some("wifi-pose".to_string()),
            scene_id: Some("scene_a".to_string()),
            base_model_id: Some("global_parent".to_string()),
            global_parent_model_id: None,
            best_pck: Some(0.81),
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "none".to_string(),
            sona_profiles: Vec::new(),
            active_sona_profile: None,
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let mut artifact = build_few_shot_candidate_artifact(
            "few-shot-1".to_string(),
            1,
            &session,
            &eligible_steps,
            vec!["dataset-1".to_string()],
            &model,
            "scene_a-few-shot-1".to_string(),
            8,
            30,
            0.7,
        )
        .expect("few-shot artifact");
        artifact.candidate_model_id = Some("trained-lora-1".to_string());
        artifact.after_metrics = Some(json!({
            "candidate_model_id": "trained-lora-1",
            "best_pck": 0.86,
            "best_epoch": 12,
        }));
        artifact.evaluator_summary = build_few_shot_evaluator_summary(&artifact);
        artifact.cross_domain_summary = Some(build_missing_few_shot_cross_domain_summary());

        sync_few_shot_promotion_gate(&mut artifact, &session);

        assert_eq!(
            artifact.promotion_gate_status,
            "blocked_missing_cross_domain_eval"
        );
        assert!(artifact
            .promotion_gate_reason
            .as_deref()
            .unwrap_or_default()
            .contains("cross-domain evaluator summary"));
    }

    #[test]
    fn few_shot_gate_should_be_eligible_when_cross_domain_report_passes() {
        let mut session = default_session("scene_a".to_string(), "Scene A".to_string());
        session.capture_session_id = Some("capture-1".to_string());
        let step = session
            .steps
            .iter_mut()
            .find(|step| step.code == "pose_idle_front_02")
            .expect("pose step");
        step.status = "recorded".to_string();
        step.recording_id = Some("dataset-1".to_string());
        let eligible_steps = vec![&session.steps[2]];
        let model = EvolutionModelSummary {
            loaded: true,
            status: "loaded".to_string(),
            model_id: Some("global_base_v1".to_string()),
            target_space: Some("wifi-pose".to_string()),
            scene_id: Some("scene_a".to_string()),
            base_model_id: Some("global_parent".to_string()),
            global_parent_model_id: None,
            best_pck: Some(0.81),
            scene_history_used: false,
            scene_history_sample_count: 0,
            adaptation_mode: "none".to_string(),
            sona_profiles: Vec::new(),
            active_sona_profile: None,
            lora_profiles: Vec::new(),
            active_lora_profile: None,
            raw: json!({}),
        };

        let mut artifact = build_few_shot_candidate_artifact(
            "few-shot-2".to_string(),
            2,
            &session,
            &eligible_steps,
            vec!["dataset-1".to_string()],
            &model,
            "scene_a-few-shot-2".to_string(),
            8,
            30,
            0.7,
        )
        .expect("few-shot artifact");
        artifact.candidate_model_id = Some("trained-lora-2".to_string());
        artifact.after_metrics = Some(json!({
            "candidate_model_id": "trained-lora-2",
            "best_pck": 0.88,
            "best_epoch": 14,
        }));
        artifact.evaluator_summary = build_few_shot_evaluator_summary(&artifact);
        artifact.cross_domain_summary = Some(json!({
            "metric_name": "mpjpe_mm",
            "passed": true,
            "buckets": {
                "in_domain": 38.0,
                "unseen_room_zero_shot": 72.0,
                "unseen_room_few_shot": 49.0,
                "cross_hardware": 58.0
            }
        }));

        sync_few_shot_promotion_gate(&mut artifact, &session);

        assert_eq!(artifact.promotion_gate_status, "eligible_for_apply");
    }

    #[test]
    fn few_shot_gate_should_mark_promoted_for_applied_candidate() {
        let mut session = default_session("scene_a".to_string(), "Scene A".to_string());
        session.latest_applied_few_shot_candidate_id = Some("few-shot-3".to_string());
        let mut artifact = EvolutionFewShotCandidateArtifact {
            calibration_id: "few-shot-3".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            source_capture_session_id: Some("capture-1".to_string()),
            requested_at_ms: 3,
            completed_at_ms: Some(5),
            previewed_at_ms: Some(6),
            rolled_back_at_ms: None,
            applied_at_ms: Some(7),
            status: "active".to_string(),
            runtime_path: "train_lora_v1_candidate_model".to_string(),
            profile_name: "scene_a-few-shot-3".to_string(),
            base_model_id: "global_base_v1".to_string(),
            candidate_model_id: Some("trained-lora-3".to_string()),
            candidate_model_created_at: Some("2026-03-14T12:00:00Z".to_string()),
            source_dataset_ids: vec!["dataset-1".to_string()],
            selected_recordings: Vec::new(),
            min_recording_quality: 0.7,
            quality_gate: super::EvolutionTrainingQualityGate {
                eligible_dataset_count: 1,
                eligible_empty_count: 1,
                eligible_pose_count: 1,
                skipped_rerecord_dataset_ids: Vec::new(),
                skipped_rerecord_step_codes: Vec::new(),
            },
            rank: 8,
            epochs: 30,
            best_pck: Some(0.88),
            best_epoch: Some(14),
            before_metrics: Some(json!({ "best_pck": 0.81 })),
            after_metrics: Some(json!({ "best_pck": 0.88 })),
            evaluator_summary: Some(json!({ "best_pck_delta": 0.07 })),
            cross_domain_summary: Some(json!({ "passed": true })),
            promotion_gate_status: "eligible_for_apply".to_string(),
            promotion_gate_reason: Some(
                "cross-domain evaluator 已通过，可以人工升配为默认 live 路径。".to_string(),
            ),
        };

        sync_few_shot_promotion_gate(&mut artifact, &session);

        assert_eq!(artifact.promotion_gate_status, "promoted");
        assert_eq!(artifact.promotion_gate_reason, None);
    }

    #[test]
    fn few_shot_cross_domain_summary_should_compute_gate_from_samples() {
        let summary =
            build_few_shot_cross_domain_summary_from_samples(&EvolutionFewShotEvaluateRequest {
                calibration_id: Some("few-shot-4".to_string()),
                hardware_type: Some("wifi_multistatic_v1".to_string()),
                notes: Some("synthetic evaluator smoke test".to_string()),
                samples: vec![
                    EvolutionFewShotEvaluationSample {
                        bucket: "in_domain".to_string(),
                        predicted: vec![0.0, 0.0, 0.0],
                        ground_truth: vec![10.0, 0.0, 0.0],
                    },
                    EvolutionFewShotEvaluationSample {
                        bucket: "unseen_room_zero_shot".to_string(),
                        predicted: vec![0.0, 0.0, 0.0],
                        ground_truth: vec![20.0, 0.0, 0.0],
                    },
                    EvolutionFewShotEvaluationSample {
                        bucket: "unseen_room_few_shot".to_string(),
                        predicted: vec![0.0, 0.0, 0.0],
                        ground_truth: vec![1.0, 0.0, 0.0],
                    },
                ],
            })
            .expect("summary");

        assert_eq!(
            summary.get("metric_name").and_then(|value| value.as_str()),
            Some("mpjpe_mm")
        );
        assert_eq!(
            summary.get("passed").and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            summary
                .get("few_shot_improvement_delta")
                .and_then(|value| value.as_f64())
                .map(|value| value.round() as i64),
            Some(19)
        );
    }

    #[test]
    fn few_shot_cross_domain_summary_should_fail_when_zero_shot_bucket_missing() {
        let summary =
            build_few_shot_cross_domain_summary_from_samples(&EvolutionFewShotEvaluateRequest {
                calibration_id: Some("few-shot-5".to_string()),
                hardware_type: None,
                notes: None,
                samples: vec![
                    EvolutionFewShotEvaluationSample {
                        bucket: "in_domain".to_string(),
                        predicted: vec![0.0, 0.0, 0.0],
                        ground_truth: vec![4.0, 0.0, 0.0],
                    },
                    EvolutionFewShotEvaluationSample {
                        bucket: "unseen_room_few_shot".to_string(),
                        predicted: vec![0.0, 0.0, 0.0],
                        ground_truth: vec![2.0, 0.0, 0.0],
                    },
                ],
            })
            .expect("summary");

        assert_eq!(
            summary.get("passed").and_then(|value| value.as_bool()),
            Some(false)
        );
        assert_eq!(
            summary
                .pointer("/gate/checks/1/ok")
                .and_then(|value| value.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn benchmark_summary_recording_ids_should_dedupe_and_skip_empty_values() {
        let summary = BenchmarkSummary {
            steps: vec![
                super::BenchmarkSummaryStep {
                    recording_id: Some(" pre-1 ".to_string()),
                },
                super::BenchmarkSummaryStep {
                    recording_id: Some("pre-1".to_string()),
                },
                super::BenchmarkSummaryStep {
                    recording_id: Some(" ".to_string()),
                },
                super::BenchmarkSummaryStep {
                    recording_id: Some("pre-2".to_string()),
                },
                super::BenchmarkSummaryStep { recording_id: None },
            ],
        };

        assert_eq!(
            benchmark_summary_recording_ids(&summary),
            vec!["pre-1".to_string(), "pre-2".to_string()]
        );
    }

    #[tokio::test]
    async fn discover_latest_benchmark_summary_paths_should_pick_latest_pre_and_post() {
        let root = unique_temp_dir("benchmark-discovery");
        fs::create_dir_all(&root).expect("create root");

        let first_preapply = write_benchmark_summary(
            &root,
            "wifi-pose-benchmark-preapply-20260315-004605",
            &["pre-1"],
        );
        std::thread::sleep(Duration::from_millis(20));
        let latest_preapply = write_benchmark_summary(
            &root,
            "wifi-pose-benchmark-preapply-20260315-004745",
            &["pre-2"],
        );
        std::thread::sleep(Duration::from_millis(20));
        let latest_postapply = write_benchmark_summary(
            &root,
            "wifi-pose-benchmark-postapply-20260315-005010",
            &["post-1"],
        );
        write_benchmark_summary(&root, "not-a-benchmark-run", &["ignore-me"]);

        let (preapply_summary_path, postapply_summary_path) =
            discover_latest_benchmark_summary_paths(&root).await;

        assert_eq!(
            preapply_summary_path.as_deref(),
            Some(latest_preapply.to_string_lossy().as_ref())
        );
        assert_eq!(
            postapply_summary_path.as_deref(),
            Some(latest_postapply.to_string_lossy().as_ref())
        );
        assert_ne!(
            preapply_summary_path.as_deref(),
            Some(first_preapply.to_string_lossy().as_ref())
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn few_shot_recording_eval_body_from_benchmark_summaries_should_map_pre_and_post() {
        let root = unique_temp_dir("benchmark-request");
        fs::create_dir_all(&root).expect("create root");
        let zero_shot_summary = write_benchmark_summary(
            &root,
            "wifi-pose-benchmark-preapply-20260315-010000",
            &[" zero-1 ", "zero-1", "zero-2"],
        );
        let few_shot_summary = write_benchmark_summary(
            &root,
            "wifi-pose-benchmark-postapply-20260315-010500",
            &["few-1", "few-2"],
        );

        let report = EvolutionFewShotCandidateArtifact {
            calibration_id: "few-shot-13".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            source_capture_session_id: None,
            requested_at_ms: 1,
            completed_at_ms: None,
            previewed_at_ms: None,
            rolled_back_at_ms: None,
            applied_at_ms: None,
            status: "candidate".to_string(),
            runtime_path: "/tmp/few-shot-13.json".to_string(),
            profile_name: "scene_a-few-shot-13".to_string(),
            base_model_id: "global_base_v1".to_string(),
            candidate_model_id: Some("trained-lora-13".to_string()),
            candidate_model_created_at: None,
            source_dataset_ids: vec!["dataset-1".to_string()],
            selected_recordings: Vec::new(),
            min_recording_quality: 0.7,
            quality_gate: Default::default(),
            rank: 8,
            epochs: 30,
            best_pck: Some(0.81),
            best_epoch: Some(7),
            before_metrics: None,
            after_metrics: None,
            evaluator_summary: None,
            cross_domain_summary: None,
            promotion_gate_status: "candidate_only".to_string(),
            promotion_gate_reason: None,
        };

        let payload = build_few_shot_recording_eval_body_from_benchmark_summaries(
            &EvolutionFewShotEvaluateBenchmarkSummariesRequest {
                calibration_id: Some("few-shot-13".to_string()),
                in_domain_summary_path: None,
                zero_shot_summary_path: zero_shot_summary.to_string_lossy().to_string(),
                few_shot_summary_path: few_shot_summary.to_string_lossy().to_string(),
                cross_hardware_summary_path: None,
                notes: Some("from benchmark summary".to_string()),
                producer: None,
                include_samples: false,
            },
            &report,
        )
        .await
        .expect("payload");

        assert_eq!(
            payload
                .pointer("/buckets/0/bucket")
                .and_then(|value| value.as_str()),
            Some("unseen_room_zero_shot")
        );
        assert_eq!(
            payload
                .pointer("/buckets/0/model_id")
                .and_then(|value| value.as_str()),
            Some("global_base_v1")
        );
        assert_eq!(
            payload
                .pointer("/buckets/0/dataset_ids")
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(2)
        );
        assert_eq!(
            payload
                .pointer("/buckets/1/bucket")
                .and_then(|value| value.as_str()),
            Some("unseen_room_few_shot")
        );
        assert_eq!(
            payload
                .pointer("/buckets/1/model_id")
                .and_then(|value| value.as_str()),
            Some("trained-lora-13")
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn few_shot_recording_eval_body_should_default_model_ids_from_candidate() {
        let report = EvolutionFewShotCandidateArtifact {
            calibration_id: "few-shot-11".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            source_capture_session_id: None,
            requested_at_ms: 1,
            completed_at_ms: None,
            previewed_at_ms: None,
            rolled_back_at_ms: None,
            applied_at_ms: None,
            status: "candidate".to_string(),
            runtime_path: "/tmp/few-shot-11.json".to_string(),
            profile_name: "scene_a-few-shot-11".to_string(),
            base_model_id: "global_base_v1".to_string(),
            candidate_model_id: Some("trained-lora-11".to_string()),
            candidate_model_created_at: None,
            source_dataset_ids: vec!["dataset-1".to_string()],
            selected_recordings: Vec::new(),
            min_recording_quality: 0.7,
            quality_gate: Default::default(),
            rank: 8,
            epochs: 30,
            best_pck: Some(0.81),
            best_epoch: Some(7),
            before_metrics: None,
            after_metrics: None,
            evaluator_summary: None,
            cross_domain_summary: None,
            promotion_gate_status: "candidate_only".to_string(),
            promotion_gate_reason: None,
        };

        let payload = build_few_shot_recording_eval_body(
            &EvolutionFewShotEvaluateRecordingsRequest {
                calibration_id: None,
                notes: Some("from ui".to_string()),
                producer: None,
                include_samples: false,
                buckets: vec![
                    EvolutionFewShotRecordingEvalBucketRequest {
                        bucket: "in_domain".to_string(),
                        dataset_ids: vec![" in-1 ".to_string(), "in-1".to_string()],
                        model_id: None,
                    },
                    EvolutionFewShotRecordingEvalBucketRequest {
                        bucket: "unseen_room_few_shot".to_string(),
                        dataset_ids: vec!["few-1".to_string()],
                        model_id: None,
                    },
                ],
            },
            &report,
        )
        .expect("payload");

        assert_eq!(
            payload
                .get("calibration_id")
                .and_then(|value| value.as_str()),
            Some("few-shot-11")
        );
        assert_eq!(
            payload
                .pointer("/buckets/0/model_id")
                .and_then(|value| value.as_str()),
            Some("global_base_v1")
        );
        assert_eq!(
            payload
                .pointer("/buckets/1/model_id")
                .and_then(|value| value.as_str()),
            Some("trained-lora-11")
        );
        assert_eq!(
            payload
                .pointer("/buckets/0/dataset_ids")
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn few_shot_recording_eval_body_should_reject_empty_datasets() {
        let report = EvolutionFewShotCandidateArtifact {
            calibration_id: "few-shot-12".to_string(),
            scene_id: "scene_a".to_string(),
            scene_name: "Scene A".to_string(),
            source_capture_session_id: None,
            requested_at_ms: 1,
            completed_at_ms: None,
            previewed_at_ms: None,
            rolled_back_at_ms: None,
            applied_at_ms: None,
            status: "candidate".to_string(),
            runtime_path: "/tmp/few-shot-12.json".to_string(),
            profile_name: "scene_a-few-shot-12".to_string(),
            base_model_id: "global_base_v1".to_string(),
            candidate_model_id: Some("trained-lora-12".to_string()),
            candidate_model_created_at: None,
            source_dataset_ids: vec!["dataset-1".to_string()],
            selected_recordings: Vec::new(),
            min_recording_quality: 0.7,
            quality_gate: Default::default(),
            rank: 8,
            epochs: 30,
            best_pck: Some(0.81),
            best_epoch: Some(7),
            before_metrics: None,
            after_metrics: None,
            evaluator_summary: None,
            cross_domain_summary: None,
            promotion_gate_status: "candidate_only".to_string(),
            promotion_gate_reason: None,
        };

        let error = build_few_shot_recording_eval_body(
            &EvolutionFewShotEvaluateRecordingsRequest {
                calibration_id: None,
                notes: None,
                producer: None,
                include_samples: false,
                buckets: vec![EvolutionFewShotRecordingEvalBucketRequest {
                    bucket: "unseen_room_zero_shot".to_string(),
                    dataset_ids: vec![" ".to_string()],
                    model_id: None,
                }],
            },
            &report,
        )
        .expect_err("should fail");

        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.contains("dataset_ids"));
    }

    #[test]
    fn few_shot_report_request_from_value_should_accept_wrapped_cross_domain_summary() {
        let req = build_few_shot_report_request_from_value(&json!({
            "calibration_id": "few-shot-6",
            "cross_domain_summary": {
                "metric_name": "mpjpe_mm",
                "buckets": {
                    "in_domain": 38.0,
                    "unseen_room_zero_shot": 72.0,
                    "unseen_room_few_shot": 49.0,
                    "cross_hardware": 58.0
                },
                "domain_gap_ratio": 1.289,
                "adaptation_speedup": 1.47,
                "hardware_type": "wifi_multistatic_v1",
                "notes": "offline evaluator artifact",
                "gate": {
                    "passed": true
                }
            }
        }))
        .expect("wrapped report request");

        assert_eq!(req.calibration_id.as_deref(), Some("few-shot-6"));
        assert_eq!(req.metric_name.as_deref(), Some("mpjpe_mm"));
        assert_eq!(req.in_domain_metric, Some(38.0));
        assert_eq!(req.unseen_room_zero_shot_metric, Some(72.0));
        assert_eq!(req.unseen_room_few_shot_metric, Some(49.0));
        assert_eq!(req.cross_hardware_metric, Some(58.0));
        assert_eq!(req.domain_gap_ratio, Some(1.289));
        assert_eq!(req.adaptation_speedup, Some(1.47));
        assert_eq!(req.hardware_type.as_deref(), Some("wifi_multistatic_v1"));
        assert_eq!(req.notes.as_deref(), Some("offline evaluator artifact"));
        assert!(req.passed);
    }

    #[test]
    fn few_shot_imported_summary_should_annotate_wrapped_report_artifact() {
        let (calibration_id, summary) = build_few_shot_cross_domain_summary_from_imported_artifact(
            &json!({
                "calibration_id": "few-shot-7",
                "schema_version": "few-shot-eval/v1",
                "cross_domain_summary": {
                    "metric_name": "mpjpe_mm",
                    "buckets": {
                        "in_domain": 35.0,
                        "unseen_room_zero_shot": 68.0,
                        "unseen_room_few_shot": 44.0
                    },
                    "cross_domain_metric": 56.0,
                    "domain_gap_ratio": 1.257,
                    "adaptation_speedup": 1.54,
                    "passed": true,
                    "gate": {
                        "passed": true,
                        "checks": [
                            {
                                "key": "domain_gap_ratio",
                                "ok": true
                            }
                        ]
                    }
                }
            }),
            Some("/tmp/few-shot-report.json".to_string()),
            "/runtime/environment_evolution_few_shot_imports/few-shot-7-123.json".to_string(),
            123,
        )
        .expect("imported wrapped report");

        assert_eq!(calibration_id.as_deref(), Some("few-shot-7"));
        assert_eq!(
            summary.get("source").and_then(|value| value.as_str()),
            Some("imported_report_v1")
        );
        assert_eq!(
            summary.get("passed").and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            summary
                .get("cross_domain_metric")
                .and_then(|value| value.as_f64()),
            Some(56.0)
        );
        assert_eq!(
            summary
                .pointer("/gate/checks/0/key")
                .and_then(|value| value.as_str()),
            Some("domain_gap_ratio")
        );
        assert_eq!(
            summary
                .pointer("/imported_artifact/source_path")
                .and_then(|value| value.as_str()),
            Some("/tmp/few-shot-report.json")
        );
        assert_eq!(
            summary
                .pointer("/imported_artifact/snapshot_path")
                .and_then(|value| value.as_str()),
            Some("/runtime/environment_evolution_few_shot_imports/few-shot-7-123.json")
        );
        assert_eq!(
            summary
                .pointer("/imported_artifact/schema_version")
                .and_then(|value| value.as_str()),
            Some("few-shot-eval/v1")
        );
    }

    #[test]
    fn few_shot_imported_summary_should_accept_sample_artifact() {
        let (calibration_id, summary) = build_few_shot_cross_domain_summary_from_imported_artifact(
            &json!({
                "calibration_id": "few-shot-8",
                "schema_version": "few-shot-eval-samples/v1",
                "hardware_type": "wifi_multistatic_v1",
                "samples": [
                    {
                        "bucket": "in_domain",
                        "predicted": [0.0, 0.0, 0.0],
                        "ground_truth": [30.0, 0.0, 0.0]
                    },
                    {
                        "bucket": "unseen_room_zero_shot",
                        "predicted": [0.0, 0.0, 0.0],
                        "ground_truth": [60.0, 0.0, 0.0]
                    },
                    {
                        "bucket": "unseen_room_few_shot",
                        "predicted": [0.0, 0.0, 0.0],
                        "ground_truth": [2.0, 0.0, 0.0]
                    },
                    {
                        "bucket": "cross_hardware",
                        "predicted": [0.0, 0.0, 0.0],
                        "ground_truth": [60.0, 0.0, 0.0]
                    }
                ]
            }),
            None,
            "/runtime/environment_evolution_few_shot_imports/few-shot-8-456.json".to_string(),
            456,
        )
        .expect("imported sample artifact");

        assert_eq!(calibration_id.as_deref(), Some("few-shot-8"));
        assert_eq!(
            summary.get("source").and_then(|value| value.as_str()),
            Some("cross_domain_evaluator_v1")
        );
        assert_eq!(
            summary.get("passed").and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            summary
                .pointer("/imported_artifact/source_path")
                .and_then(|value| value.as_str()),
            None
        );
        assert_eq!(
            summary
                .pointer("/imported_artifact/snapshot_path")
                .and_then(|value| value.as_str()),
            Some("/runtime/environment_evolution_few_shot_imports/few-shot-8-456.json")
        );
        assert_eq!(
            summary
                .pointer("/imported_artifact/schema_version")
                .and_then(|value| value.as_str()),
            Some("few-shot-eval-samples/v1")
        );
    }

    #[test]
    fn few_shot_artifact_match_should_prefer_top_level_calibration_id() {
        assert!(few_shot_artifact_matches_calibration_id(
            &json!({
                "calibration_id": "few-shot-9",
                "cross_domain_summary": {
                    "passed": true
                }
            }),
            std::path::Path::new("/tmp/not-related-name.json"),
            "few-shot-9"
        ));
    }

    #[test]
    fn few_shot_artifact_match_should_fallback_to_filename() {
        assert!(few_shot_artifact_matches_calibration_id(
            &json!({
                "cross_domain_summary": {
                    "passed": true
                }
            }),
            std::path::Path::new("/tmp/few-shot-10-offline-report.json"),
            "few-shot-10"
        ));
    }

    #[test]
    fn summarize_model_info_should_mark_trained_lora_model_as_lora_adapted() {
        let summary = summarize_model_info(
            json!({
                "status": "loaded",
                "container": {
                    "manifest": { "model_id": "trained-lora-20260314_120000" },
                    "metadata": {
                        "training": {
                            "type": "lora",
                            "target_space": "wifi-pose",
                            "base_model_id": "global_base_v1",
                            "scene_id": "scene_a"
                        }
                    }
                }
            }),
            &ExternalLoraProfilesState::default(),
            &ExternalSonaProfilesState::default(),
        );

        assert_eq!(summary.adaptation_mode, "lora-adapted");
        assert_eq!(summary.base_model_id.as_deref(), Some("global_base_v1"));
        assert_eq!(summary.scene_id.as_deref(), Some("scene_a"));
    }

    #[test]
    fn summarize_model_info_should_mark_active_sona_profile_as_geometry_conditioned() {
        let summary = summarize_model_info(
            json!({
                "status": "loaded",
                "container": {
                    "manifest": { "model_id": "trained-supervised-geometry-ready" },
                    "metadata": {
                        "training": {
                            "target_space": "wifi-pose",
                            "base_model_id": "global_base_v1",
                            "scene_id": "scene_a"
                        }
                    }
                }
            }),
            &ExternalLoraProfilesState::default(),
            &ExternalSonaProfilesState {
                profiles: vec!["scene_a_geometry".to_string()],
                active: Some("scene_a_geometry".to_string()),
            },
        );

        assert_eq!(summary.adaptation_mode, "geometry-conditioned");
        assert_eq!(
            summary.active_sona_profile.as_deref(),
            Some("scene_a_geometry")
        );
    }

    #[test]
    fn build_readiness_should_treat_shared_frame_device_pose_as_phone_ready() {
        let vision = VisionSnapshot {
            fresh: true,
            device_pose: Some(VisionDevicePose {
                position_m: [0.0, 0.0, 0.0],
                target_space: "ios_arkit_world_frame".to_string(),
                rotation_deg: Some([0.0, 0.0, 0.0]),
                right_vector: Some([1.0, 0.0, 0.0]),
                up_vector: Some([0.0, 1.0, 0.0]),
                forward_vector: Some([0.0, 0.0, 1.0]),
                timestamp_ns: Some(1),
                tracking_state: Some("normal".to_string()),
                world_mapping_status: Some("mapped".to_string()),
                source: "ios_arkit_world_transform".to_string(),
            }),
            ..VisionSnapshot::default()
        };
        let readiness = build_readiness(
            &vision,
            &StereoSnapshot::default(),
            &WifiPoseSnapshot::default(),
            &CsiSnapshot::default(),
            &default_operator_snapshot(),
            false,
        );
        assert!(readiness.phone_ready);
    }

    #[test]
    fn build_readiness_should_not_count_attitude_only_phone_pose_as_ready() {
        let vision = VisionSnapshot {
            fresh: true,
            hand_kpts_2d: vec![[0.5, 0.5]],
            device_pose: Some(VisionDevicePose {
                position_m: [0.0, 0.0, 0.0],
                target_space: "device_motion_reference_frame".to_string(),
                rotation_deg: Some([0.0, 0.0, 0.0]),
                right_vector: Some([1.0, 0.0, 0.0]),
                up_vector: Some([0.0, 1.0, 0.0]),
                forward_vector: Some([0.0, 0.0, 1.0]),
                timestamp_ns: Some(1),
                tracking_state: None,
                world_mapping_status: None,
                source: "ios_device_attitude".to_string(),
            }),
            ..VisionSnapshot::default()
        };
        let readiness = build_readiness(
            &vision,
            &StereoSnapshot::default(),
            &WifiPoseSnapshot::default(),
            &CsiSnapshot::default(),
            &default_operator_snapshot(),
            false,
        );
        assert!(!readiness.phone_ready);
        assert!(
            readiness
                .issues
                .iter()
                .any(|issue| issue.contains("shared/world-frame device_pose"))
        );
    }

    #[test]
    fn build_readiness_should_not_emit_phone_hand_alignment_issue_without_visible_hands() {
        let vision = VisionSnapshot {
            fresh: true,
            device_pose: Some(VisionDevicePose {
                position_m: [0.0, 0.0, 0.0],
                target_space: "ios_arkit_world_frame".to_string(),
                rotation_deg: Some([0.0, 0.0, 0.0]),
                right_vector: Some([1.0, 0.0, 0.0]),
                up_vector: Some([0.0, 1.0, 0.0]),
                forward_vector: Some([0.0, 0.0, 1.0]),
                timestamp_ns: Some(1),
                tracking_state: Some("normal".to_string()),
                world_mapping_status: Some("mapped".to_string()),
                source: "ios_arkit_world_transform".to_string(),
            }),
            ..VisionSnapshot::default()
        };
        let readiness = build_readiness(
            &vision,
            &StereoSnapshot::default(),
            &WifiPoseSnapshot::default(),
            &CsiSnapshot::default(),
            &default_operator_snapshot(),
            false,
        );
        assert!(readiness.phone_ready);
        assert!(
            !readiness
                .issues
                .iter()
                .any(|issue| issue.contains("手机手部还没和身体对齐"))
        );
    }

    #[test]
    fn phase_tracking_should_not_require_wifi_ready() {
        let readiness = EvolutionReadiness {
            csi_ready: true,
            stereo_ready: true,
            wifi_ready: false,
            phone_ready: false,
            robot_ready: false,
            teacher_ready: true,
            room_empty_ready: true,
            full_multimodal_ready: false,
            pose_capture_ready: false,
            anchor_source: "stereo+wifi_prior".to_string(),
            selected_operator_track_id: Some("stereo-person-1".to_string()),
            iphone_visible_hand_count: 0,
            hand_match_count: 0,
            hand_match_score: 0.0,
            left_wrist_gap_m: None,
            right_wrist_gap_m: None,
            suggested_quality_score: 0.64,
            issues: Vec::new(),
        };
        assert!(phase_tracking_ready(&readiness));
    }

    #[test]
    fn update_step_runtimes_advances_pose_phase_when_wifi_is_down() {
        let mut session = default_session("scene_a".to_string(), "Scene A".to_string());
        let step = session
            .steps
            .iter_mut()
            .find(|step| step.code == "pose_turn_lr_01")
            .expect("pose_turn_lr_01 step");
        step.status = "recording".to_string();
        sync_step_runtimes(&mut session);
        let runtime = step_runtime_mut(&mut session, "pose_turn_lr_01").expect("runtime");
        runtime.phase_hold_started_at_ms = Some(0);

        let readiness = EvolutionReadiness {
            csi_ready: true,
            stereo_ready: true,
            wifi_ready: false,
            phone_ready: false,
            robot_ready: false,
            teacher_ready: true,
            room_empty_ready: true,
            full_multimodal_ready: false,
            pose_capture_ready: false,
            anchor_source: "stereo+wifi_prior".to_string(),
            selected_operator_track_id: Some("stereo-person-1".to_string()),
            iphone_visible_hand_count: 0,
            hand_match_count: 0,
            hand_match_score: 0.0,
            left_wrist_gap_m: None,
            right_wrist_gap_m: None,
            suggested_quality_score: 0.64,
            issues: Vec::new(),
        };
        let recording = EvolutionRecordingSummary {
            active: true,
            session_id: Some("recording-1".to_string()),
            session_name: Some("recording-1".to_string()),
            capture_session_id: Some("capture-1".to_string()),
            step_code: Some("pose_turn_lr_01".to_string()),
            frame_count: 42,
            duration_secs: Some(20),
            raw: json!({}),
        };
        let operator = OperatorSnapshot {
            estimate: OperatorEstimate::default(),
            fresh: true,
        };
        let stereo = StereoSnapshot {
            fresh: true,
            body_layout: BodyKeypointLayout::CocoBody17,
            body_kpts_3d: sample_turn_left_body(),
            ..StereoSnapshot::default()
        };

        update_step_runtimes(&mut session, &recording, &readiness, &operator, &stereo);

        let runtime = session
            .step_runtimes
            .iter()
            .find(|runtime| runtime.step_code == "pose_turn_lr_01")
            .expect("runtime after update");
        assert_eq!(runtime.phase_index, 1);
        assert_eq!(runtime.max_phase_index, 1);
    }

    #[test]
    fn phase_reached_should_support_right_first_turns() {
        let mut runtime = EvolutionStepRuntimeState {
            step_code: "pose_turn_lr_01".to_string(),
            phase_total: 4,
            ..EvolutionStepRuntimeState::default()
        };
        let right = body_metrics_from_body(&sample_turn_right_body()).expect("right turn metrics");
        assert!(phase_reached("pose_turn_lr_01", 0, &right, &mut runtime));
        assert_eq!(runtime.lead_side.as_deref(), Some("right"));

        let front = body_metrics_from_body(&sample_front_body()).expect("front metrics");
        assert!(phase_reached("pose_turn_lr_01", 1, &front, &mut runtime));

        let left = body_metrics_from_body(&sample_turn_left_body()).expect("left turn metrics");
        assert!(phase_reached("pose_turn_lr_01", 2, &left, &mut runtime));
    }

    #[test]
    fn phase_reached_should_not_treat_rest_pose_as_reach() {
        let rest = body_metrics_from_body(&sample_front_body()).expect("rest metrics");
        let mut runtime = EvolutionStepRuntimeState {
            step_code: "pose_reach_lr_01".to_string(),
            phase_total: 4,
            ..EvolutionStepRuntimeState::default()
        };
        seed_runtime_baseline(&mut runtime, &rest);

        assert!(!phase_reached("pose_reach_lr_01", 0, &rest, &mut runtime));

        let right_reach =
            body_metrics_from_body(&sample_reach_right_body()).expect("right reach metrics");
        assert!(phase_reached(
            "pose_reach_lr_01",
            0,
            &right_reach,
            &mut runtime
        ));
        assert_eq!(runtime.lead_side.as_deref(), Some("right"));
    }

    #[test]
    fn training_phase_failed_should_detect_failed_prefixes() {
        assert!(training_phase_failed("failed"));
        assert!(training_phase_failed("failed_non_finite_loss"));
        assert!(training_phase_failed(" Failed_Base_Model_Rebase "));
        assert!(!training_phase_failed("training"));
        assert!(!training_phase_failed("completed"));
    }
}
