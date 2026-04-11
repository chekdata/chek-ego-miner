use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use image::codecs::gif::{GifEncoder, Repeat};
use image::imageops::FilterType;
use image::{Delay, Frame};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::config::Config;
use crate::path_safety;
use crate::protocol::version_guard::ProtocolVersionInfo;
use crate::recorder::upload_queue;

/// 轻量 recorder：把关键 JSONL 事件落盘，便于回放/对账/验收。
///
/// 目录结构参考 PRD：
/// - `/data/ruview/session/<session_id>/manifest.json`
/// - `demo_capture_bundle.json`
/// - `labels/labels.jsonl`
/// - `raw/iphone/*.jsonl`
/// - `fused/fusion_state.jsonl`
/// - `teleop/teleop_frame.jsonl`
/// - `chunks/chunk_state.jsonl`
/// - `clips/<...>/clip_manifest.json`
pub struct SessionRecorder {
    data_dir: PathBuf,
    inner: Mutex<Option<ActiveSession>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionRepairResult {
    pub session_id: String,
    pub trip_id: String,
    pub base_dir: String,
    pub migrated_paths: Vec<String>,
    pub backfilled_pose_imu_lines: u64,
    pub backfilled_depth_index_lines: u64,
    pub generated_edge_frames: bool,
    pub quality_status: Option<String>,
    pub ready_for_upload: bool,
    pub unrecoverable_gaps: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct SessionContextUpdate {
    pub capture_device_id: Option<String>,
    pub operator_id: Option<String>,
    pub task_id: Option<String>,
    pub task_ids: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CsiPacketMeta {
    pub device_id: Option<String>,
    pub node_id: Option<u8>,
    pub sequence: Option<u32>,
    pub freq_raw: Option<u32>,
    pub rssi: Option<i8>,
    pub noise_floor: Option<i8>,
    pub snr: Option<f32>,
    pub source_time_ns: Option<u64>,
    pub header_version: Option<u8>,
    pub clock_domain: Option<String>,
    pub time_sync_status: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ClipSampleStats {
    pub sample_total: u64,
    pub pass_samples: u64,
    pub pass_rate_percent: f64,
    pub locatable_samples: u64,
    pub playable_samples: u64,
    pub label_consistent_samples: u64,
    pub index_complete_samples: u64,
}

#[derive(Debug)]
struct SemanticRuntimeState {
    next_keyframe_index: u64,
    next_event_index: u64,
    next_segment_index: u64,
    last_keyframe_edge_time_ns: Option<u64>,
    last_camera_mode: String,
    current_segment: Option<SemanticSegmentAccumulator>,
    preview_manifest: PreviewManifestState,
    pending_jobs: VecDeque<PendingSemanticInference>,
}

#[derive(Clone, Debug, Default)]
struct SemanticSegmentAccumulator {
    index: u64,
    start_edge_time_ns: u64,
    start_source_time_ns: u64,
    end_edge_time_ns: u64,
    end_source_time_ns: u64,
    start_frame_id: u64,
    end_frame_id: u64,
    camera_mode: String,
    event_ids: Vec<String>,
    keyframe_ids: Vec<String>,
    keyframe_relpaths: Vec<String>,
    captions: Vec<String>,
    tags: Vec<String>,
    objects: Vec<String>,
    actions: Vec<String>,
    model_ids: Vec<String>,
    inference_sources: Vec<String>,
    latencies_ms: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PreviewManifestState {
    status: String,
    vlm_status: String,
    degraded_reasons: Vec<PreviewDegradedReason>,
    keyframes: Vec<PreviewKeyframeRecord>,
    clips: Vec<PreviewClipRecord>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PreviewDegradedReason {
    stage: String,
    detail: String,
    recorded_unix_ms: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PreviewKeyframeRecord {
    id: String,
    relpath: String,
    frame_id: u64,
    source_time_ns: u64,
    edge_time_ns: u64,
    camera_mode: String,
    sample_reasons: Vec<String>,
    caption: String,
    #[serde(default)]
    objects: Vec<String>,
    tags: Vec<String>,
    action_guess: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    inference_source: Option<String>,
    #[serde(default)]
    latency_ms: Option<f64>,
    #[serde(default)]
    inference_status: String,
    embedding_relpath: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PreviewClipRecord {
    id: String,
    relpath: String,
    mime_type: String,
    status: String,
    start_edge_time_ns: u64,
    end_edge_time_ns: u64,
    keyframe_ids: Vec<String>,
    caption: String,
    #[serde(default)]
    objects: Vec<String>,
    tags: Vec<String>,
    action_guess: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    inference_source: Option<String>,
    #[serde(default)]
    latency_ms: Option<f64>,
    embedding_relpath: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PreviewManifestFile {
    #[serde(rename = "type")]
    ty: String,
    schema_version: String,
    trip_id: String,
    session_id: String,
    generated_unix_ms: u64,
    runtime_profile: String,
    upload_policy_mode: String,
    raw_residency_default: String,
    preview_residency_default: String,
    preview_generation_enabled: bool,
    vlm_indexing_enabled: bool,
    model_id: String,
    fallback_model_id: String,
    prompt_version: String,
    vlm_sidecar_base: String,
    vlm_live_interval_ms: u64,
    vlm_event_trigger_enabled: bool,
    vlm_event_trigger_camera_mode_change_enabled: bool,
    vlm_inference_timeout_ms: u64,
    vlm_auto_fallback_latency_ms: u64,
    vlm_auto_fallback_cooldown_ms: u64,
    vlm_max_consecutive_failures: u32,
    status: String,
    vlm_status: String,
    degraded_reasons: Vec<PreviewDegradedReason>,
    keyframes: Vec<PreviewKeyframeRecord>,
    clips: Vec<PreviewClipRecord>,
}

#[derive(Clone, Debug)]
struct SemanticInferenceMetadata {
    preview_record_index: usize,
    event_id: String,
    keyframe_id: String,
    keyframe_relpath: String,
    frame_id: u64,
    source_time_ns: u64,
    edge_time_ns: u64,
    camera_mode: String,
    sample_reasons: Vec<String>,
    roll_segment_before_event: bool,
    fallback_caption: String,
    fallback_tags: Vec<String>,
    fallback_objects: Vec<String>,
    fallback_action_guess: String,
    started_unix_ms: u64,
}

#[derive(Debug)]
struct PendingSemanticInference {
    meta: SemanticInferenceMetadata,
    handle: JoinHandle<Result<SemanticInferenceResult, String>>,
}

#[derive(Clone, Debug, Default)]
struct SemanticInferenceResult {
    caption: String,
    tags: Vec<String>,
    objects: Vec<String>,
    action_guess: String,
    model_id: String,
    inference_source: String,
    latency_ms: Option<f64>,
    degraded_reasons: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct VlmSidecarResponse {
    ok: bool,
    #[serde(default)]
    caption: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    objects: Vec<String>,
    #[serde(default)]
    action_guess: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    inference_source: Option<String>,
    #[serde(default)]
    latency_ms: Option<f64>,
    #[serde(default)]
    degraded_reasons: Vec<String>,
    #[serde(default)]
    fallback_active: bool,
    #[serde(default)]
    error: Option<String>,
}

impl Default for SemanticRuntimeState {
    fn default() -> Self {
        Self {
            next_keyframe_index: 0,
            next_event_index: 0,
            next_segment_index: 0,
            last_keyframe_edge_time_ns: None,
            last_camera_mode: String::new(),
            current_segment: None,
            preview_manifest: PreviewManifestState::default(),
            pending_jobs: VecDeque::new(),
        }
    }
}

impl SessionRecorder {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            inner: Mutex::new(None),
        }
    }

    pub async fn repair_existing_sessions(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<Vec<SessionRepairResult>, String> {
        let session_root = self.data_dir.join("session");
        let mut dir = tokio::fs::read_dir(&session_root)
            .await
            .map_err(|e| format!("读取 session 根目录失败: {} ({e})", session_root.display()))?;
        let mut results = Vec::new();
        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|e| format!("遍历 session 根目录失败: {} ({e})", session_root.display()))?
        {
            let entry_path = entry.path();
            let file_type = entry.file_type().await.map_err(|e| {
                format!("读取 session 目录类型失败: {} ({e})", entry_path.display())
            })?;
            if !file_type.is_dir() {
                continue;
            }
            if let Err(error) = path_safety::ensure_session_dir_path(&entry_path) {
                warn!(path=%entry_path.display(), error=%error, "skipping invalid session directory during repair");
                continue;
            }
            results.push(Self::repair_existing_session_dir(&entry_path, protocol, cfg).await?);
        }
        results.sort_by(|a, b| a.session_id.cmp(&b.session_id));
        Ok(results)
    }

    pub async fn ensure_session(
        &self,
        trip_id: &str,
        session_id: &str,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<(), String> {
        let mut guard = self.inner.lock().await;
        Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
    }

    async fn repair_existing_session_dir(
        base_dir: &Path,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<SessionRepairResult, String> {
        path_safety::ensure_session_dir_path(base_dir)?;
        let session_id = base_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("unknown")
            .to_string();
        let mut migrated_paths = migrate_legacy_iphone_layout(base_dir).await?;
        let (trip_id, created_unix_ms, session_context) =
            load_existing_session_identity(base_dir, &session_id).await?;
        let generated_edge_frames =
            ensure_edge_frames_snapshot(base_dir, &trip_id, &session_id, cfg).await?;
        if generated_edge_frames {
            migrated_paths.push("generated calibration/edge_frames.json".to_string());
        }
        let (backfilled_pose_imu_lines, backfilled_depth_index_lines, backfill_notes) =
            backfill_phone_inputs_from_capture_pose(base_dir, &trip_id, &session_id).await?;
        migrated_paths.extend(backfill_notes);

        let mut active = ActiveSession::open_existing(
            base_dir.to_path_buf(),
            trip_id.clone(),
            session_id.clone(),
            created_unix_ms,
            session_context,
            cfg,
        )
        .await?;
        active.refresh_demo_bundle(protocol, cfg).await?;
        active
            .refresh_session_metadata_artifacts(protocol, cfg)
            .await?;

        let quality_path = base_dir.join("qa").join("local_quality_report.json");
        let quality_value = tokio::fs::read_to_string(&quality_path)
            .await
            .ok()
            .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok());
        let quality_status = quality_value
            .as_ref()
            .and_then(|value| value.get("status"))
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let ready_for_upload = quality_value
            .as_ref()
            .and_then(|value| value.get("ready_for_upload"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        let mut unrecoverable_gaps = Vec::new();
        if active.lines.raw_iphone_pose_imu == 0 {
            unrecoverable_gaps.push("raw/iphone/wide/pose_imu.jsonl".to_string());
        }
        if active.lines.raw_iphone_depth == 0 {
            unrecoverable_gaps.push("raw/iphone/wide/depth/index.jsonl".to_string());
        }
        if active.lines.raw_robot_state == 0 {
            unrecoverable_gaps.push("raw/robot/state.jsonl".to_string());
        }

        Ok(SessionRepairResult {
            session_id,
            trip_id,
            base_dir: base_dir.display().to_string(),
            migrated_paths,
            backfilled_pose_imu_lines,
            backfilled_depth_index_lines,
            generated_edge_frames,
            quality_status,
            ready_for_upload,
            unrecoverable_gaps,
        })
    }

    pub async fn stop_if_match(&self, trip_id: &str, session_id: &str, cfg: &Config) {
        let mut guard = self.inner.lock().await;
        if guard
            .as_ref()
            .is_some_and(|s| s.trip_id == trip_id && s.session_id == session_id)
        {
            if let Some(active) = guard.as_mut() {
                let _ = active.flush_pending_semantic_jobs(cfg, true).await;
                let _ = active.finalize_current_semantic_segment(cfg).await;
                if let Err(error) = active.write_preview_manifest(cfg).await {
                    warn!(error=%error, trip_id, session_id, "flush preview manifest on stop failed");
                }
                if let Err(error) = active.flush_pending_csi_chunk(cfg).await {
                    warn!(error=%error, trip_id, session_id, "flush pending csi chunk on stop failed");
                }
            }
            *guard = None;
        }
    }

    pub async fn clip_sample_stats(
        &self,
        trip_id: &str,
        session_id: &str,
        sample_total: usize,
    ) -> Option<ClipSampleStats> {
        let guard = self.inner.lock().await;
        let active = guard.as_ref()?;
        if active.trip_id != trip_id || active.session_id != session_id {
            return None;
        }
        Some(active.clip_sample_stats(sample_total))
    }

    async fn ensure_session_locked(
        data_dir: &Path,
        guard: &mut Option<ActiveSession>,
        trip_id: &str,
        session_id: &str,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<(), String> {
        let safe_session_id = session_id.trim();
        if trip_id.trim().is_empty() || safe_session_id.is_empty() {
            return Err("session recorder requires non-empty trip_id/session_id".to_string());
        }
        if matches!(safe_session_id, "." | "..")
            || safe_session_id
                .chars()
                .any(|ch| matches!(ch, '/' | '\\' | '\0' | '\n' | '\r'))
        {
            return Err(format!("invalid session_id: {safe_session_id}"));
        }
        if let Some(active) = guard.as_ref() {
            if active.trip_id == trip_id && active.session_id == safe_session_id {
                return Ok(());
            }
        }
        if let Some(active) = guard.as_mut() {
            let _ = active.flush_pending_semantic_jobs(cfg, true).await;
            let _ = active.finalize_current_semantic_segment(cfg).await;
            if let Err(error) = active.write_preview_manifest(cfg).await {
                warn!(error=%error, trip_id=%active.trip_id, session_id=%active.session_id, "flush preview manifest before session switch failed");
            }
            active.flush_pending_csi_chunk(cfg).await?;
        }
        *guard = Some(
            ActiveSession::start(data_dir, trip_id, safe_session_id, protocol, cfg).await?,
        );
        Ok(())
    }

    pub async fn record_label_event(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        v: &serde_json::Value,
    ) {
        let Some(trip_id) = v.get("trip_id").and_then(|x| x.as_str()) else {
            return;
        };
        let Some(session_id) = v.get("session_id").and_then(|x| x.as_str()) else {
            return;
        };
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        let label_line = match ActiveSession::append_jsonl_counted(
            &mut active.labels,
            &mut active.lines.labels,
            v,
        )
        .await
        {
            Ok(n) => n,
            Err(e) => {
                warn!(error=%e, "record label_event failed");
                return;
            }
        };
        active.maybe_update_clips(v, label_line).await;
        let context_changed = active.apply_session_context_update(SessionContextUpdate {
            capture_device_id: None,
            operator_id: v
                .get("operator_id")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
            task_id: v
                .get("task_id")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
            task_ids: Vec::new(),
        });
        if context_changed {
            if let Err(e) = active.refresh_demo_bundle(protocol, cfg).await {
                warn!(error=%e, "refresh demo_capture_bundle after label_event context failed");
            }
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after label_event context failed");
            }
        }
    }

    pub async fn update_session_context(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        update: SessionContextUpdate,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if active.trip_id != trip_id || active.session_id != session_id {
            return;
        }
        if !active.apply_session_context_update(update) {
            return;
        }
        if let Err(e) = active.refresh_demo_bundle(protocol, cfg).await {
            warn!(error=%e, "refresh demo_capture_bundle after session context failed");
        }
        if let Err(e) = active
            .refresh_session_metadata_artifacts(protocol, cfg)
            .await
        {
            warn!(error=%e, "refresh session metadata artifacts after session context failed");
        }
    }

    pub async fn record_capture_pose(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        v: &serde_json::Value,
    ) {
        let Some(trip_id) = v.get("trip_id").and_then(|x| x.as_str()) else {
            return;
        };
        let Some(session_id) = v.get("session_id").and_then(|x| x.as_str()) else {
            return;
        };
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.iphone_kpts,
            &mut active.lines.raw_iphone,
            v,
        )
        .await
        {
            warn!(error=%e, "record capture_pose failed");
            return;
        }
        if let Err(e) = ActiveSession::append_jsonl(&mut active.offline_iphone_pose_v2, v).await {
            warn!(error=%e, "record offline iphone_pose_v2 mirror failed");
            return;
        }
        let mut refresh_metadata = active.lines.raw_iphone == 1;
        refresh_metadata = refresh_metadata
            || active.apply_session_context_update(SessionContextUpdate {
                capture_device_id: v
                    .get("device_id")
                    .and_then(|value| value.as_str())
                    .map(ToOwned::to_owned),
                operator_id: None,
                task_id: None,
                task_ids: Vec::new(),
            });
        if let Some(snapshot) = active.build_iphone_calibration_snapshot(v) {
            if let Err(e) = ActiveSession::write_json_pretty(
                active
                    .base_dir
                    .join("calibration")
                    .join("iphone_capture.json"),
                &snapshot,
            )
            .await
            {
                warn!(error=%e, "write iphone calibration snapshot failed");
            } else {
                refresh_metadata = refresh_metadata || !active.has_iphone_calibration;
                active.has_iphone_calibration = true;
            }
        }
        if refresh_metadata {
            if let Err(e) = active.refresh_demo_bundle(protocol, cfg).await {
                warn!(error=%e, "refresh demo_capture_bundle after first capture_pose failed");
            }
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after capture_pose failed");
            }
        }
    }

    pub async fn record_phone_vision_input(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
        primary_image_bytes: Option<Vec<u8>>,
        depth_bytes: Option<&[u8]>,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };

        let pose_imu_line = match ActiveSession::append_jsonl_counted(
            &mut active.iphone_pose_imu,
            &mut active.lines.raw_iphone_pose_imu,
            v,
        )
        .await
        {
            Ok(line) => line,
            Err(e) => {
                warn!(error=%e, "record phone pose_imu failed");
                return;
            }
        };

        let mut refresh_metadata = pose_imu_line == 1;
        refresh_metadata = refresh_metadata
            || active.apply_session_context_update(SessionContextUpdate {
                capture_device_id: v
                    .get("device_id")
                    .and_then(|value| value.as_str())
                    .map(ToOwned::to_owned),
                operator_id: None,
                task_id: None,
                task_ids: Vec::new(),
            });

        if let Some(camera_calibration) = v.get("camera_calibration") {
            let calibration_snapshot = serde_json::json!({
                "type": "sensor_calibration_snapshot",
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "sensor_kind": "iphone_capture",
                "sensor_id": v.get("device_id").and_then(|value| value.as_str()).unwrap_or("iphone_capture"),
                "edge_time_ns": v.get("edge_time_ns").and_then(|value| value.as_u64()).unwrap_or(0),
                "sensor_frame": "iphone_capture_frame",
                "intrinsics": camera_calibration,
                "notes": "phone_vision_input calibration snapshot",
            });
            if let Err(e) = ActiveSession::write_json_pretty(
                active
                    .base_dir
                    .join("calibration")
                    .join("iphone_capture.json"),
                &calibration_snapshot,
            )
            .await
            {
                warn!(error=%e, "write iphone calibration snapshot from phone vision input failed");
            } else {
                refresh_metadata = refresh_metadata || !active.has_iphone_calibration;
                active.has_iphone_calibration = true;
            }
        }

        if let Some(relpath) = v.get("depth_relpath").and_then(|value| value.as_str()) {
            if let Some(bytes) = depth_bytes {
                if let Err(e) =
                    ActiveSession::write_binary(active.base_dir.join(relpath), bytes).await
                {
                    warn!(error=%e, relpath, "record raw iphone depth frame failed");
                } else {
                    let depth_event = serde_json::json!({
                        "type": "iphone_depth_frame",
                        "schema_version": "1.0.0",
                        "trip_id": trip_id,
                        "session_id": session_id,
                        "device_id": v.get("device_id").and_then(|value| value.as_str()).unwrap_or(""),
                        "source_time_ns": v.get("source_time_ns").and_then(|value| value.as_u64()).unwrap_or(0),
                        "edge_time_ns": v.get("edge_time_ns").and_then(|value| value.as_u64()).unwrap_or(0),
                        "frame_id": v.get("frame_id").and_then(|value| value.as_u64()).unwrap_or(0),
                        "depth_w": v.get("depth_w").and_then(|value| value.as_u64()).unwrap_or(0),
                        "depth_h": v.get("depth_h").and_then(|value| value.as_u64()).unwrap_or(0),
                        "file_relpath": relpath,
                        "file_bytes": bytes.len(),
                        "storage_format": "f32le",
                    });
                    match ActiveSession::append_jsonl_counted(
                        &mut active.iphone_depth_index,
                        &mut active.lines.raw_iphone_depth,
                        &depth_event,
                    )
                    .await
                    {
                        Ok(line) => {
                            refresh_metadata = refresh_metadata || line == 1;
                        }
                        Err(e) => {
                            warn!(error=%e, "record raw iphone depth index failed");
                        }
                    }
                }
            }
        }

        let semantic_outputs_changed = active
            .record_phone_semantic_outputs(cfg, v, primary_image_bytes.as_deref())
            .await;
        refresh_metadata = refresh_metadata || semantic_outputs_changed;

        if refresh_metadata {
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(
                    error=%e,
                    "refresh session metadata artifacts after phone vision input failed"
                );
            }
        }
    }

    pub async fn record_fusion_state(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.fused_state,
            &mut active.lines.fused_state,
            v,
        )
        .await
        {
            warn!(error=%e, "record fusion_state failed");
        } else {
            if let Err(e) =
                ActiveSession::append_jsonl(&mut active.offline_fusion_state_v2, v).await
            {
                warn!(error=%e, "record offline fusion_state_v2 mirror failed");
                return;
            }
            active.maybe_update_clip_quality_from_fusion_state(v);
        }
    }

    pub async fn record_human_demo_pose(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.human_demo_pose,
            &mut active.lines.human_demo_pose,
            v,
        )
        .await
        {
            warn!(error=%e, "record human_demo_pose failed");
            return;
        }
        if let Err(e) = ActiveSession::append_jsonl(&mut active.offline_human_demo_pose_v2, v).await
        {
            warn!(error=%e, "record offline human_demo_pose_v2 mirror failed");
        }
        if active.lines.human_demo_pose == 1 {
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after human_demo_pose failed");
            }
        }
    }

    pub async fn record_teleop_frame(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) =
            ActiveSession::append_jsonl_counted(&mut active.teleop, &mut active.lines.teleop, v)
                .await
        {
            warn!(error=%e, "record teleop_frame failed");
        } else if active.lines.teleop == 1 {
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after teleop_frame failed");
            }
        }
    }

    pub async fn record_robot_state(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        if trip_id.trim().is_empty() || session_id.trim().is_empty() {
            return;
        }
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.robot_state,
            &mut active.lines.raw_robot_state,
            v,
        )
        .await
        {
            warn!(error=%e, "record robot_state failed");
        } else if active.lines.raw_robot_state == 1 {
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after robot_state failed");
            }
        }
    }

    pub async fn record_csi_index(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) =
            ActiveSession::append_jsonl_counted(&mut active.csi_index, &mut active.lines.raw_csi, v)
                .await
        {
            warn!(error=%e, "record csi index failed");
        }
    }

    pub async fn record_csi_packet_bytes(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        edge_time_ns: u64,
        bytes: &[u8],
        meta: CsiPacketMeta,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };

        // 简单 framing：edge_time_ns(u64 LE) + len(u32 LE) + payload(bytes)
        let mut hdr = [0u8; 12];
        hdr[..8].copy_from_slice(&edge_time_ns.to_le_bytes());
        let len_u32 = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        hdr[8..12].copy_from_slice(&len_u32.to_le_bytes());
        if let Err(e) = active.csi_packets.write_all(&hdr).await {
            warn!(error=%e, "record csi packets.bin header failed");
            return;
        }
        if let Err(e) = active.csi_packets.write_all(bytes).await {
            warn!(error=%e, "record csi packets.bin payload failed");
            return;
        }
        let mut framed = hdr.to_vec();
        framed.extend_from_slice(bytes);
        if let Err(e) = active
            .record_csi_chunk_packet(cfg, edge_time_ns, &framed, meta)
            .await
        {
            warn!(error=%e, "record csi chunk aggregation failed");
        }
    }

    pub async fn record_stereo_pose3d(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.stereo_pose3d,
            &mut active.lines.raw_stereo,
            v,
        )
        .await
        {
            warn!(error=%e, "record stereo_pose3d failed");
            return;
        }
        if let Err(e) = ActiveSession::append_jsonl(&mut active.offline_stereo_pose_v2, v).await {
            warn!(error=%e, "record offline stereo_pose_v2 mirror failed");
            return;
        }

        let mut refresh_metadata = active.lines.raw_stereo == 1;
        if let Some(snapshot) = active.build_stereo_calibration_snapshot(v) {
            if let Err(e) = ActiveSession::write_json_pretty(
                active.base_dir.join("calibration").join("stereo_pair.json"),
                &snapshot,
            )
            .await
            {
                warn!(error=%e, "write stereo calibration snapshot failed");
            } else {
                refresh_metadata = refresh_metadata || !active.has_stereo_calibration;
                active.has_stereo_calibration = true;
            }
        }
        if let Err(e) = active.refresh_demo_bundle(protocol, cfg).await {
            warn!(error=%e, "refresh demo_capture_bundle after stereo_pose failed");
        }
        if refresh_metadata {
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after stereo_pose failed");
            }
        }
    }

    pub async fn record_wifi_pose3d(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.wifi_pose3d,
            &mut active.lines.raw_wifi,
            v,
        )
        .await
        {
            warn!(error=%e, "record wifi_pose3d failed");
            return;
        }
        if let Err(e) = ActiveSession::append_jsonl(&mut active.offline_wifi_pose_v2, v).await {
            warn!(error=%e, "record offline wifi_pose_v2 mirror failed");
            return;
        }

        let mut refresh_metadata = active.lines.raw_wifi == 1;
        if let Some(snapshot) = active.build_wifi_calibration_snapshot(v) {
            if let Err(e) = ActiveSession::write_json_pretty(
                active.base_dir.join("calibration").join("wifi_pose.json"),
                &snapshot,
            )
            .await
            {
                warn!(error=%e, "write wifi calibration snapshot failed");
            } else {
                refresh_metadata = refresh_metadata || !active.has_wifi_calibration;
                active.has_wifi_calibration = true;
            }
        }
        if let Err(e) = active.refresh_demo_bundle(protocol, cfg).await {
            warn!(error=%e, "refresh demo_capture_bundle after wifi_pose failed");
        }
        if refresh_metadata {
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after wifi_pose failed");
            }
        }
    }

    pub async fn record_time_sync_sample(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.time_sync_samples,
            &mut active.lines.sync,
            v,
        )
        .await
        {
            warn!(error=%e, "record time_sync_sample failed");
            return;
        }
        if let Err(e) = active.refresh_demo_bundle(protocol, cfg).await {
            warn!(error=%e, "refresh demo_capture_bundle after time_sync failed");
        }
        if active.lines.sync == 1 {
            if let Err(e) = active
                .refresh_session_metadata_artifacts(protocol, cfg)
                .await
            {
                warn!(error=%e, "refresh session metadata artifacts after time_sync failed");
            }
        }
    }

    pub async fn record_frame_correspondence(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        v: &serde_json::Value,
    ) {
        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.frame_correspondence,
            &mut active.lines.frame_correspondence,
            v,
        )
        .await
        {
            warn!(error=%e, "record frame_correspondence failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn record_media_chunk_index(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        device_id: Option<&str>,
        media_scope: Option<&str>,
        media_track: Option<&str>,
        source_kind: Option<&str>,
        clock_domain: Option<&str>,
        chunk_index: u32,
        file_type: Option<&str>,
        file_name: Option<&str>,
        file_relpath: &str,
        file_bytes: u64,
        edge_time_ns: u64,
        upload_edge_time_ns: u64,
        source_time_ns: Option<u64>,
        source_start_time_ns: Option<u64>,
        source_end_time_ns: Option<u64>,
        frame_source_time_ns: Vec<u64>,
        frame_edge_time_ns: Vec<u64>,
        frame_count: Option<u32>,
        frame_rate_hz: Option<f64>,
        media_alignment_kind: &str,
        time_sync_status: &str,
    ) {
        self.record_media_chunk_index_inner(
            protocol,
            cfg,
            trip_id,
            session_id,
            device_id,
            media_scope,
            media_track,
            source_kind,
            clock_domain,
            chunk_index,
            file_type,
            file_name,
            file_relpath,
            file_bytes,
            edge_time_ns,
            upload_edge_time_ns,
            source_time_ns,
            source_start_time_ns,
            source_end_time_ns,
            frame_source_time_ns,
            frame_edge_time_ns,
            frame_count,
            frame_rate_hz,
            media_alignment_kind,
            time_sync_status,
            false,
        )
        .await;
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn record_stereo_media_chunk_index(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        device_id: Option<&str>,
        media_scope: Option<&str>,
        media_track: Option<&str>,
        source_kind: Option<&str>,
        clock_domain: Option<&str>,
        chunk_index: u32,
        file_type: Option<&str>,
        file_name: Option<&str>,
        file_relpath: &str,
        file_bytes: u64,
        edge_time_ns: u64,
        upload_edge_time_ns: u64,
        source_time_ns: Option<u64>,
        source_start_time_ns: Option<u64>,
        source_end_time_ns: Option<u64>,
        frame_source_time_ns: Vec<u64>,
        frame_edge_time_ns: Vec<u64>,
        frame_count: Option<u32>,
        frame_rate_hz: Option<f64>,
        media_alignment_kind: &str,
        time_sync_status: &str,
    ) {
        self.record_media_chunk_index_inner(
            protocol,
            cfg,
            trip_id,
            session_id,
            device_id,
            media_scope,
            media_track,
            source_kind,
            clock_domain,
            chunk_index,
            file_type,
            file_name,
            file_relpath,
            file_bytes,
            edge_time_ns,
            upload_edge_time_ns,
            source_time_ns,
            source_start_time_ns,
            source_end_time_ns,
            frame_source_time_ns,
            frame_edge_time_ns,
            frame_count,
            frame_rate_hz,
            media_alignment_kind,
            time_sync_status,
            true,
        )
        .await;
    }

    #[allow(clippy::too_many_arguments)]
    async fn record_media_chunk_index_inner(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        device_id: Option<&str>,
        media_scope: Option<&str>,
        media_track: Option<&str>,
        source_kind: Option<&str>,
        clock_domain: Option<&str>,
        chunk_index: u32,
        file_type: Option<&str>,
        file_name: Option<&str>,
        file_relpath: &str,
        file_bytes: u64,
        edge_time_ns: u64,
        upload_edge_time_ns: u64,
        source_time_ns: Option<u64>,
        source_start_time_ns: Option<u64>,
        source_end_time_ns: Option<u64>,
        frame_source_time_ns: Vec<u64>,
        frame_edge_time_ns: Vec<u64>,
        frame_count: Option<u32>,
        frame_rate_hz: Option<f64>,
        media_alignment_kind: &str,
        time_sync_status: &str,
        stereo: bool,
    ) {
        let event = serde_json::json!({
            "type": if stereo { "stereo_media_chunk_index" } else { "media_chunk_index" },
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id.unwrap_or(""),
            "media_scope": media_scope.unwrap_or(""),
            "media_track": media_track.unwrap_or(""),
            "source_kind": source_kind.unwrap_or(""),
            "clock_domain": clock_domain.unwrap_or(""),
            "chunk_index": chunk_index,
            "file_type": file_type.unwrap_or(""),
            "file_name": file_name.unwrap_or(""),
            "file_relpath": file_relpath,
            "file_bytes": file_bytes,
            "edge_time_ns": edge_time_ns,
            "upload_edge_time_ns": upload_edge_time_ns,
            "source_time_ns": source_time_ns,
            "source_start_time_ns": source_start_time_ns,
            "source_end_time_ns": source_end_time_ns,
            "frame_source_time_ns": frame_source_time_ns,
            "frame_edge_time_ns": frame_edge_time_ns,
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate_hz,
            "media_alignment_kind": media_alignment_kind,
            "time_sync_status": time_sync_status,
        });

        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        let append_result = if stereo {
            ActiveSession::append_jsonl_counted(
                &mut active.stereo_media_index,
                &mut active.lines.raw_stereo_media,
                &event,
            )
            .await
        } else {
            ActiveSession::append_jsonl_counted(
                &mut active.iphone_media_index,
                &mut active.lines.raw_iphone_media,
                &event,
            )
            .await
        };
        if let Err(e) = append_result {
            warn!(error=%e, stereo, "record media_chunk_index failed");
            return;
        }
        let scope_name = media_scope.unwrap_or(if stereo { "stereo" } else { "iphone" });
        let track_name = media_track.unwrap_or(if stereo { "preview" } else { "main" });
        let storage_track_name = normalized_track_storage_name(scope_name, track_name);
        active
            .seen_media_tracks
            .insert((scope_name.to_string(), track_name.to_string()));
        let track_index_path = active
            .base_dir
            .join("raw")
            .join(scope_name)
            .join(storage_track_name)
            .join("media_index.jsonl");
        if let Some(parent) = track_index_path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                warn!(error=%e, path=%parent.display(), "create media track dir failed");
                return;
            }
        }
        let mut track_index = match open_append(track_index_path.clone()).await {
            Ok(file) => file,
            Err(e) => {
                warn!(error=%e, path=%track_index_path.display(), "open media track index failed");
                return;
            }
        };
        if let Err(e) = ActiveSession::append_jsonl(&mut track_index, &event).await {
            warn!(error=%e, path=%track_index_path.display(), "record track media_chunk_index failed");
            return;
        }
        drop(track_index);
        if let Err(e) = active.refresh_demo_bundle(protocol, cfg).await {
            warn!(error=%e, "refresh demo_capture_bundle after media_chunk_index failed");
            return;
        }
        if let Err(e) = active
            .refresh_session_metadata_artifacts(protocol, cfg)
            .await
        {
            warn!(error=%e, "refresh session metadata artifacts after media_chunk_index failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn record_chunk_state_event(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
        trip_id: &str,
        session_id: &str,
        chunk_index: u32,
        from_state: &str,
        to_state: &str,
        edge_time_ns: u64,
        reason_code: &str,
        reason_message: &str,
    ) {
        let event = serde_json::json!({
            "type": "chunk_state_event",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "chunk_index": chunk_index,
            "from_state": from_state,
            "to_state": to_state,
            "edge_time_ns": edge_time_ns,
            "reason_code": reason_code,
            "reason_message": reason_message,
        });

        let mut guard = self.inner.lock().await;
        if Self::ensure_session_locked(
            &self.data_dir,
            &mut guard,
            trip_id,
            session_id,
            protocol,
            cfg,
        )
        .await
        .is_err()
        {
            return;
        }
        let Some(active) = guard.as_mut() else { return };
        if let Err(e) = ActiveSession::append_jsonl_counted(
            &mut active.chunk_events,
            &mut active.lines.chunks,
            &event,
        )
        .await
        {
            warn!(error=%e, "record chunk_state_event failed");
        }
    }
}

struct ActiveSession {
    trip_id: String,
    session_id: String,
    base_dir: PathBuf,
    created_unix_ms: u64,

    labels: tokio::fs::File,
    iphone_kpts: tokio::fs::File,
    iphone_pose_imu: tokio::fs::File,
    iphone_depth_index: tokio::fs::File,
    iphone_media_index: tokio::fs::File,
    csi_index: tokio::fs::File,
    csi_chunk_index: tokio::fs::File,
    csi_packets: tokio::fs::File,
    stereo_pose3d: tokio::fs::File,
    wifi_pose3d: tokio::fs::File,
    stereo_media_index: tokio::fs::File,
    fused_state: tokio::fs::File,
    human_demo_pose: tokio::fs::File,
    teleop: tokio::fs::File,
    robot_state: tokio::fs::File,
    offline_iphone_pose_v2: tokio::fs::File,
    offline_stereo_pose_v2: tokio::fs::File,
    offline_wifi_pose_v2: tokio::fs::File,
    offline_fusion_state_v2: tokio::fs::File,
    offline_human_demo_pose_v2: tokio::fs::File,
    chunk_events: tokio::fs::File,
    time_sync_samples: tokio::fs::File,
    frame_correspondence: tokio::fs::File,
    next_csi_chunk_index: u64,
    pending_csi_chunk: Option<PendingCsiChunk>,
    seen_media_tracks: HashSet<(String, String)>,

    has_iphone_calibration: bool,
    has_stereo_calibration: bool,
    has_wifi_calibration: bool,
    session_context: SessionCrowdContext,

    lines: LineCounters,
    clip_results: VecDeque<ClipValidationSummary>,

    clip: ClipState,
    semantic: SemanticRuntimeState,
}

#[derive(Debug)]
struct PendingCsiChunk {
    chunk_index: u64,
    segment_start_edge_time_ns: u64,
    segment_end_edge_time_ns: u64,
    segment_start_source_time_ns: Option<u64>,
    segment_end_source_time_ns: Option<u64>,
    bytes: Vec<u8>,
    packet_count: u64,
    node_ids: Vec<u8>,
    freq_raw_values: Vec<u32>,
    sequence_start: Option<u32>,
    sequence_end: Option<u32>,
    last_device_id: Option<String>,
    last_node_id: Option<u8>,
    last_sequence: Option<u32>,
    last_freq_raw: Option<u32>,
    last_rssi: Option<i8>,
    last_noise_floor: Option<i8>,
    last_snr: Option<f32>,
    last_source_time_ns: Option<u64>,
    last_header_version: Option<u8>,
    last_clock_domain: Option<String>,
    clock_domain_mixed: bool,
    last_time_sync_status: Option<String>,
    time_sync_status_mixed: bool,
    rssi_sum: i64,
    rssi_count: u64,
    noise_floor_sum: i64,
    noise_floor_count: u64,
    snr_sum: f64,
    snr_count: u64,
}

impl PendingCsiChunk {
    fn new(chunk_index: u64, edge_time_ns: u64) -> Self {
        Self {
            chunk_index,
            segment_start_edge_time_ns: edge_time_ns,
            segment_end_edge_time_ns: edge_time_ns,
            segment_start_source_time_ns: None,
            segment_end_source_time_ns: None,
            bytes: Vec::new(),
            packet_count: 0,
            node_ids: Vec::new(),
            freq_raw_values: Vec::new(),
            sequence_start: None,
            sequence_end: None,
            last_device_id: None,
            last_node_id: None,
            last_sequence: None,
            last_freq_raw: None,
            last_rssi: None,
            last_noise_floor: None,
            last_snr: None,
            last_source_time_ns: None,
            last_header_version: None,
            last_clock_domain: None,
            clock_domain_mixed: false,
            last_time_sync_status: None,
            time_sync_status_mixed: false,
            rssi_sum: 0,
            rssi_count: 0,
            noise_floor_sum: 0,
            noise_floor_count: 0,
            snr_sum: 0.0,
            snr_count: 0,
        }
    }

    fn push_packet(&mut self, edge_time_ns: u64, framed: &[u8], meta: CsiPacketMeta) {
        if self.packet_count == 0 {
            self.segment_start_edge_time_ns = edge_time_ns;
            self.segment_start_source_time_ns = meta.source_time_ns;
        }
        if self.segment_start_source_time_ns.is_none() && meta.source_time_ns.is_some() {
            self.segment_start_source_time_ns = meta.source_time_ns;
        }
        self.segment_end_edge_time_ns = edge_time_ns;
        if meta.source_time_ns.is_some() {
            self.segment_end_source_time_ns = meta.source_time_ns;
            self.last_source_time_ns = meta.source_time_ns;
        }
        self.bytes.extend_from_slice(framed);
        self.packet_count = self.packet_count.saturating_add(1);

        if let Some(device_id) = meta.device_id {
            self.last_device_id = Some(device_id);
        }
        if let Some(node_id) = meta.node_id {
            self.last_node_id = Some(node_id);
            if !self.node_ids.contains(&node_id) {
                self.node_ids.push(node_id);
            }
        }
        if let Some(freq_raw) = meta.freq_raw {
            self.last_freq_raw = Some(freq_raw);
            if !self.freq_raw_values.contains(&freq_raw) {
                self.freq_raw_values.push(freq_raw);
            }
        }
        if let Some(sequence) = meta.sequence {
            self.last_sequence = Some(sequence);
            self.sequence_start = Some(self.sequence_start.unwrap_or(sequence));
            self.sequence_end = Some(sequence);
        }
        if let Some(rssi) = meta.rssi {
            self.last_rssi = Some(rssi);
            self.rssi_sum += i64::from(rssi);
            self.rssi_count = self.rssi_count.saturating_add(1);
        }
        if let Some(noise_floor) = meta.noise_floor {
            self.last_noise_floor = Some(noise_floor);
            self.noise_floor_sum += i64::from(noise_floor);
            self.noise_floor_count = self.noise_floor_count.saturating_add(1);
        }
        if let Some(snr) = meta.snr {
            self.last_snr = Some(snr);
            self.snr_sum += f64::from(snr);
            self.snr_count = self.snr_count.saturating_add(1);
        }
        if let Some(header_version) = meta.header_version {
            self.last_header_version = Some(header_version);
        }
        if let Some(clock_domain) = meta.clock_domain {
            if let Some(existing) = self.last_clock_domain.as_deref() {
                if existing != clock_domain {
                    self.clock_domain_mixed = true;
                }
            } else {
                self.last_clock_domain = Some(clock_domain.clone());
            }
            if self.last_clock_domain.is_none() {
                self.last_clock_domain = Some(clock_domain);
            }
        }
        if let Some(time_sync_status) = meta.time_sync_status {
            if let Some(existing) = self.last_time_sync_status.as_deref() {
                if existing != time_sync_status {
                    self.time_sync_status_mixed = true;
                }
            } else {
                self.last_time_sync_status = Some(time_sync_status.clone());
            }
            if self.last_time_sync_status.is_none() {
                self.last_time_sync_status = Some(time_sync_status);
            }
        }
    }

    fn should_flush(&self, cfg: &Config) -> bool {
        if self.packet_count == 0 {
            return false;
        }
        let max_packets = cfg.csi_chunk_max_packets.max(1);
        let max_bytes = cfg.csi_chunk_max_bytes.max(1);
        let max_span_ns = cfg.csi_chunk_max_span_ms.saturating_mul(1_000_000);
        self.packet_count >= max_packets
            || u64::try_from(self.bytes.len()).unwrap_or(u64::MAX) >= max_bytes
            || self
                .segment_end_edge_time_ns
                .saturating_sub(self.segment_start_edge_time_ns)
                >= max_span_ns
    }

    fn chunk_relpath(&self, cfg: &Config) -> String {
        let bucket_index = self.chunk_index / cfg.csi_chunk_bucket_size.max(1);
        format!(
            "raw/csi/chunks/{bucket_index:06}/csi__chunk_{:06}.bin",
            self.chunk_index
        )
    }

    fn chunk_event(
        &self,
        trip_id: &str,
        session_id: &str,
        chunk_relpath: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "type": "csi_chunk_index",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "chunk_index": self.chunk_index,
            "file_relpath": chunk_relpath,
            "file_bytes": self.bytes.len(),
            "packet_count": self.packet_count,
            "segment_start_edge_time_ns": self.segment_start_edge_time_ns,
            "segment_end_edge_time_ns": self.segment_end_edge_time_ns,
            "segment_start_source_time_ns": self.segment_start_source_time_ns,
            "segment_end_source_time_ns": self.segment_end_source_time_ns,
            "edge_time_ns": self.segment_end_edge_time_ns,
            "source_time_ns": self.last_source_time_ns,
            "device_id": self.last_device_id,
            "clock_domain": if self.clock_domain_mixed {
                "mixed"
            } else {
                self.last_clock_domain.as_deref().unwrap_or("recv_time_ns")
            },
            "time_sync_status": if self.time_sync_status_mixed {
                "mixed"
            } else {
                self.last_time_sync_status.as_deref().unwrap_or("recv_time_only")
            },
            "header_version": self.last_header_version,
            "node_id": self.last_node_id,
            "sequence": self.last_sequence,
            "freq_raw": self.last_freq_raw,
            "rssi": self.last_rssi,
            "noise_floor": self.last_noise_floor,
            "snr": self.last_snr,
            "node_ids": self.node_ids,
            "freq_raw_values": self.freq_raw_values,
            "sequence_start": self.sequence_start,
            "sequence_end": self.sequence_end,
            "rssi_mean": mean_i64(self.rssi_sum, self.rssi_count),
            "noise_floor_mean": mean_i64(self.noise_floor_sum, self.noise_floor_count),
            "snr_mean": mean_f64(self.snr_sum, self.snr_count),
        })
    }
}

fn mean_i64(sum: i64, count: u64) -> Option<f64> {
    if count == 0 {
        return None;
    }
    Some(sum as f64 / count as f64)
}

fn mean_f64(sum: f64, count: u64) -> Option<f64> {
    if count == 0 {
        return None;
    }
    Some(sum / count as f64)
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct LineCounters {
    labels: u64,
    raw_iphone: u64,
    raw_iphone_pose_imu: u64,
    raw_iphone_depth: u64,
    raw_iphone_media: u64,
    raw_csi: u64,
    raw_stereo: u64,
    raw_wifi: u64,
    raw_stereo_media: u64,
    fused_state: u64,
    human_demo_pose: u64,
    teleop: u64,
    raw_robot_state: u64,
    chunks: u64,
    sync: u64,
    frame_correspondence: u64,
}

#[derive(Default)]
struct ClipState {
    current_scene: String,
    active: Option<ActiveAction>,
    part_counter_by_action: HashMap<String, u32>,
}

#[derive(Clone)]
struct ActiveAction {
    action_id: String,
    action_label: String,
    scene_label: String,
    part_index: u32,
    start_edge_time_ns: u64,
    indices: ClipIndices,
    quality: ClipQualityAgg,
}

#[derive(Clone, Copy, Debug)]
struct ClipIndices {
    labels_start_line: u64,
    raw_iphone_start_line: u64,
    raw_iphone_media_start_line: u64,
    raw_csi_start_line: u64,
    raw_stereo_start_line: u64,
    raw_wifi_start_line: u64,
    raw_stereo_media_start_line: u64,
    fused_start_line: u64,
    human_demo_pose_start_line: u64,
    teleop_start_line: u64,
    chunks_start_line: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ClipQualityAgg {
    n: u64,
    sum_vision: f64,
    sum_csi: f64,
    sum_fused: f64,
    min_fused: f32,
    sum_coherence: f64,
    accept: u64,
    limit: u64,
    freeze: u64,
    estop: u64,
}

impl ClipQualityAgg {
    fn update(&mut self, v: &serde_json::Value) {
        let q = v.get("quality");
        let vision = q
            .and_then(|x| x.get("vision_conf"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let csi = q
            .and_then(|x| x.get("csi_conf"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let fused = q
            .and_then(|x| x.get("fused_conf"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let coherence = q
            .and_then(|x| x.get("coherence"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0);
        let gate_state = q
            .and_then(|x| x.get("gate_state"))
            .and_then(|x| x.as_str())
            .unwrap_or("");

        self.n += 1;
        self.sum_vision += vision;
        self.sum_csi += csi;
        self.sum_fused += fused;
        self.sum_coherence += coherence;
        if self.n == 1 {
            self.min_fused = fused as f32;
        } else {
            self.min_fused = self.min_fused.min(fused as f32);
        }
        match gate_state {
            "accept" => self.accept += 1,
            "limit" => self.limit += 1,
            "freeze" => self.freeze += 1,
            "estop" => self.estop += 1,
            _ => {}
        }
    }
}

#[derive(Serialize)]
struct SessionManifest {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    created_unix_ms: u64,
    generated_unix_ms: u64,
    protocol: ProtocolVersionInfo,
    frames: DemoBundleFrames,
    #[serde(skip_serializing_if = "SessionCrowdContext::is_empty")]
    session_context: SessionCrowdContext,
    artifacts: Vec<SessionManifestArtifact>,
    recorder_state: SessionManifestRecorderState,
}

#[derive(Serialize)]
struct SessionManifestArtifact {
    id: String,
    relpath: String,
    kind: &'static str,
    required: bool,
    exists: bool,
    byte_size: u64,
    line_count: Option<u64>,
    privacy_tier: &'static str,
}

#[derive(Serialize)]
struct SessionManifestRecorderState {
    has_iphone_calibration: bool,
    has_stereo_calibration: bool,
    has_wifi_calibration: bool,
    seen_media_tracks: Vec<String>,
    line_counters: LineCounters,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
struct SessionRuntimeFlags {
    phone_ingest_enabled: bool,
    stereo_enabled: bool,
    wifi_enabled: bool,
    fusion_enabled: bool,
    control_enabled: bool,
    sim_enabled: bool,
    vlm_indexing_enabled: bool,
    preview_generation_enabled: bool,
    crowd_upload_enabled: bool,
}

impl SessionRuntimeFlags {
    fn from_config(cfg: &Config) -> Self {
        let flags = cfg.runtime_feature_flags();
        Self {
            phone_ingest_enabled: flags.phone_ingest_enabled,
            stereo_enabled: flags.stereo_enabled,
            wifi_enabled: flags.wifi_enabled,
            fusion_enabled: flags.fusion_enabled,
            control_enabled: flags.control_enabled,
            sim_enabled: flags.sim_enabled,
            vlm_indexing_enabled: flags.vlm_indexing_enabled,
            preview_generation_enabled: flags.preview_generation_enabled,
            crowd_upload_enabled: cfg.crowd_upload_enabled,
        }
    }

    fn is_empty(&self) -> bool {
        !self.phone_ingest_enabled
            && !self.stereo_enabled
            && !self.wifi_enabled
            && !self.fusion_enabled
            && !self.control_enabled
            && !self.sim_enabled
            && !self.vlm_indexing_enabled
            && !self.preview_generation_enabled
            && !self.crowd_upload_enabled
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct SessionCrowdContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    capture_device_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operator_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    task_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upload_policy_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_residency: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    preview_residency: Option<String>,
    #[serde(default, skip_serializing_if = "SessionRuntimeFlags::is_empty")]
    runtime_flags: SessionRuntimeFlags,
}

impl SessionCrowdContext {
    fn from_config(cfg: &Config) -> Self {
        Self {
            capture_device_id: None,
            operator_id: None,
            task_id: None,
            task_ids: Vec::new(),
            runtime_profile: Some(cfg.runtime_profile_name().to_string()),
            upload_policy_mode: Some(cfg.upload_policy_mode_name().to_string()),
            raw_residency: Some(cfg.raw_residency_default().to_string()),
            preview_residency: Some(cfg.preview_residency_default().to_string()),
            runtime_flags: SessionRuntimeFlags::from_config(cfg),
        }
    }

    fn ensure_runtime_defaults(&mut self, cfg: &Config) {
        if self.runtime_profile.is_none() {
            self.runtime_profile = Some(cfg.runtime_profile_name().to_string());
        }
        if self.upload_policy_mode.is_none() {
            self.upload_policy_mode = Some(cfg.upload_policy_mode_name().to_string());
        }
        if self.raw_residency.is_none() {
            self.raw_residency = Some(cfg.raw_residency_default().to_string());
        }
        if self.preview_residency.is_none() {
            self.preview_residency = Some(cfg.preview_residency_default().to_string());
        }
        if self.runtime_flags.is_empty() {
            self.runtime_flags = SessionRuntimeFlags::from_config(cfg);
        }
    }

    fn merge_missing_from(&mut self, other: &SessionCrowdContext) -> bool {
        let mut changed = false;

        if self.capture_device_id.is_none() && other.capture_device_id.is_some() {
            self.capture_device_id = other.capture_device_id.clone();
            changed = true;
        }
        if self.operator_id.is_none() && other.operator_id.is_some() {
            self.operator_id = other.operator_id.clone();
            changed = true;
        }
        if self.task_id.is_none() && other.task_id.is_some() {
            self.task_id = other.task_id.clone();
            changed = true;
        }
        for task_id in &other.task_ids {
            if !self.task_ids.iter().any(|existing| existing == task_id) {
                self.task_ids.push(task_id.clone());
                changed = true;
            }
        }
        if self.task_id.is_none() {
            if let Some(first_task_id) = self.task_ids.first().cloned() {
                self.task_id = Some(first_task_id);
                changed = true;
            }
        }
        if self.runtime_profile.is_none() && other.runtime_profile.is_some() {
            self.runtime_profile = other.runtime_profile.clone();
            changed = true;
        }
        if self.upload_policy_mode.is_none() && other.upload_policy_mode.is_some() {
            self.upload_policy_mode = other.upload_policy_mode.clone();
            changed = true;
        }
        if self.raw_residency.is_none() && other.raw_residency.is_some() {
            self.raw_residency = other.raw_residency.clone();
            changed = true;
        }
        if self.preview_residency.is_none() && other.preview_residency.is_some() {
            self.preview_residency = other.preview_residency.clone();
            changed = true;
        }
        if self.runtime_flags.is_empty() && !other.runtime_flags.is_empty() {
            self.runtime_flags = other.runtime_flags.clone();
            changed = true;
        }

        changed
    }

    fn is_empty(&self) -> bool {
        self.capture_device_id.is_none()
            && self.operator_id.is_none()
            && self.task_id.is_none()
            && self.task_ids.is_empty()
            && self.runtime_profile.is_none()
            && self.upload_policy_mode.is_none()
            && self.raw_residency.is_none()
            && self.preview_residency.is_none()
            && self.runtime_flags.is_empty()
    }
}

#[derive(Serialize)]
struct DemoBundleFrames {
    operator_frame: String,
    robot_base_frame: String,
    extrinsic_version: String,
}

#[derive(Serialize)]
struct DemoBundleArtifacts {
    manifest: String,
    labels: String,
    capture_pose: String,
    pose_imu: String,
    iphone_depth_index: String,
    iphone_media_index: String,
    iphone_fisheye_media_index: String,
    stereo_pose: String,
    wifi_pose: String,
    stereo_media_index: String,
    csi_index: String,
    csi_chunk_index: String,
    csi_packets: String,
    fusion_state: String,
    human_demo_pose: String,
    teleop_frame: String,
    robot_state: String,
    chunk_state: String,
    time_sync_samples: String,
    frame_correspondence: String,
    offline_manifest: String,
    vlm_events: String,
    vlm_segments: String,
    vlm_embeddings: String,
    preview_manifest: String,
    preview_keyframes: String,
    preview_clips: String,
    local_quality_report: String,
    upload_policy: String,
    upload_manifest: String,
    upload_queue: String,
    upload_receipts: String,
}

#[derive(Serialize)]
struct DemoBundleMediaTrack {
    id: String,
    label: String,
    source: String,
    modality: String,
    media_index: String,
    chunk_dir: String,
    default_visible: bool,
}

#[derive(Clone, Copy)]
struct MediaTrackTemplate {
    scope: &'static str,
    track: &'static str,
    id: &'static str,
    label: &'static str,
    modality: &'static str,
    default_visible: bool,
}

const MEDIA_TRACK_TEMPLATES: [MediaTrackTemplate; 7] = [
    MediaTrackTemplate {
        scope: "iphone",
        track: "main",
        id: "iphone.main",
        label: "iPhone 主预览",
        modality: "video",
        default_visible: true,
    },
    MediaTrackTemplate {
        scope: "iphone",
        track: "aux",
        id: "iphone.aux",
        label: "iPhone 辅视角",
        modality: "video",
        default_visible: false,
    },
    MediaTrackTemplate {
        scope: "iphone",
        track: "depth",
        id: "iphone.depth",
        label: "iPhone 深度",
        modality: "depth",
        default_visible: false,
    },
    MediaTrackTemplate {
        scope: "iphone",
        track: "fisheye",
        id: "iphone.fisheye",
        label: "iPhone 鱼眼",
        modality: "video",
        default_visible: false,
    },
    MediaTrackTemplate {
        scope: "stereo",
        track: "left",
        id: "stereo.left",
        label: "双目左目",
        modality: "video",
        default_visible: false,
    },
    MediaTrackTemplate {
        scope: "stereo",
        track: "right",
        id: "stereo.right",
        label: "双目右目",
        modality: "video",
        default_visible: false,
    },
    MediaTrackTemplate {
        scope: "stereo",
        track: "preview",
        id: "stereo.preview",
        label: "双目预览",
        modality: "video",
        default_visible: true,
    },
];

fn normalized_track_storage_name<'a>(scope: &'a str, track: &'a str) -> &'a str {
    if scope == "iphone" && track == "main" {
        "wide"
    } else {
        track
    }
}

#[derive(Serialize)]
struct DemoCaptureBundle {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    created_unix_ms: u64,
    generated_unix_ms: u64,
    protocol: ProtocolVersionInfo,
    frames: DemoBundleFrames,
    #[serde(skip_serializing_if = "SessionCrowdContext::is_empty")]
    session_context: SessionCrowdContext,
    artifacts: DemoBundleArtifacts,
    media_tracks: Vec<DemoBundleMediaTrack>,
    chunk_dirs: Vec<String>,
    calibration_snapshot_paths: Vec<String>,
}

#[derive(Serialize)]
struct LocalQualityReport {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    generated_unix_ms: u64,
    status: &'static str,
    ready_for_upload: bool,
    score_percent: f32,
    checks: Vec<LocalQualityCheck>,
    missing_artifacts: Vec<String>,
    recommended_actions: Vec<String>,
}

#[derive(Serialize)]
struct LocalQualityCheck {
    id: &'static str,
    ok: bool,
    score: f32,
    detail: String,
}

#[derive(Serialize)]
struct UploadManifest {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    generated_unix_ms: u64,
    upload_policy: UploadPolicy,
    #[serde(skip_serializing_if = "SessionCrowdContext::is_empty")]
    session_context: SessionCrowdContext,
    ready_for_upload: bool,
    artifact_count: usize,
    ready_artifact_count: usize,
    artifacts: Vec<UploadManifestArtifact>,
}

#[derive(Serialize)]
struct UploadPolicySnapshot {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    generated_unix_ms: u64,
    upload_policy: UploadPolicy,
    #[serde(skip_serializing_if = "SessionCrowdContext::is_empty")]
    session_context: SessionCrowdContext,
    privacy_tier_defaults: Vec<&'static str>,
}

#[derive(Serialize)]
struct UploadPolicy {
    mode: &'static str,
    artifact_policy_mode: String,
    runtime_profile: String,
    raw_residency_default: String,
    preview_residency_default: String,
    transport: &'static str,
    required_quality_status: &'static str,
}

#[derive(Serialize)]
struct UploadManifestArtifact {
    id: String,
    relpath: String,
    kind: &'static str,
    category: &'static str,
    required: bool,
    exists: bool,
    byte_size: u64,
    line_count: Option<u64>,
    residency: String,
    upload_state: &'static str,
}

struct ArtifactSpecRecord {
    id: String,
    relpath: String,
    kind: &'static str,
    required: bool,
    exists: bool,
    byte_size: u64,
    line_count: Option<u64>,
    privacy_tier: &'static str,
}

#[derive(Serialize)]
struct OfflineManifestPipeline {
    pipeline_version: &'static str,
    generation_mode: &'static str,
    materialization_state: &'static str,
    notes: &'static str,
}

#[derive(Serialize)]
struct OfflineManifestArtifact {
    id: &'static str,
    relpath: &'static str,
    format: &'static str,
    materialization_state: &'static str,
    generation_mode: &'static str,
    source_relpaths: Vec<&'static str>,
}

#[derive(Serialize)]
struct OfflineManifest {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    generated_unix_ms: u64,
    pipeline: OfflineManifestPipeline,
    artifacts: Vec<OfflineManifestArtifact>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct IndexRange {
    start_line: u64,
    end_line: u64,
}

#[derive(Serialize)]
struct ClipModalityRanges {
    labels: Option<IndexRange>,
    raw_iphone: Option<IndexRange>,
    raw_iphone_media: Option<IndexRange>,
    raw_csi: Option<IndexRange>,
    raw_stereo_pose3d: Option<IndexRange>,
    raw_wifi_pose3d: Option<IndexRange>,
    raw_stereo_media: Option<IndexRange>,
    fused: Option<IndexRange>,
    human_demo_pose: Option<IndexRange>,
    teleop: Option<IndexRange>,
    chunks: Option<IndexRange>,
}

#[derive(Serialize)]
struct ClipQualitySummary {
    sample_count: u64,
    vision_conf_mean: f32,
    csi_conf_mean: f32,
    fused_conf_mean: f32,
    fused_conf_min: f32,
    coherence_mean: f32,
    gate_accept: u64,
    gate_limit: u64,
    gate_freeze: u64,
    gate_estop: u64,
}

#[derive(Serialize)]
struct ClipManifest {
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    action_id: String,
    action_label: String,
    scene_label: String,
    part_index: u32,
    start_edge_time_ns: u64,
    end_edge_time_ns: u64,
    needs_review: bool,
    index_ranges: ClipModalityRanges,
    quality_summary: ClipQualitySummary,
    validation: ClipValidationSummary,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ClipValidationSummary {
    locatable: bool,
    playable: bool,
    label_consistent: bool,
    index_complete: bool,
    pass: bool,
}

impl ActiveSession {
    async fn disk_line_counters(&self) -> Result<LineCounters, String> {
        let labels_dir = self.base_dir.join("labels");
        let raw_iphone_dir = self.base_dir.join("raw").join("iphone");
        let raw_iphone_wide_dir = raw_iphone_dir.join("wide");
        let raw_csi_dir = self.base_dir.join("raw").join("csi");
        let raw_stereo_dir = self.base_dir.join("raw").join("stereo");
        let raw_wifi_dir = self.base_dir.join("raw").join("wifi");
        let raw_robot_dir = self.base_dir.join("raw").join("robot");
        let raw_iphone_depth_dir = raw_iphone_wide_dir.join("depth");
        let fused_dir = self.base_dir.join("fused");
        let teleop_dir = self.base_dir.join("teleop");
        let chunk_dir = self.base_dir.join("chunks");
        let sync_dir = self.base_dir.join("sync");

        Ok(LineCounters {
            labels: count_jsonl_lines(labels_dir.join("labels.jsonl")).await?,
            raw_iphone: count_jsonl_lines(raw_iphone_wide_dir.join("kpts_depth.jsonl")).await?,
            raw_iphone_pose_imu: count_jsonl_lines(raw_iphone_wide_dir.join("pose_imu.jsonl"))
                .await?,
            raw_iphone_depth: count_jsonl_lines(raw_iphone_depth_dir.join("index.jsonl")).await?,
            raw_iphone_media: count_jsonl_lines(raw_iphone_wide_dir.join("media_index.jsonl"))
                .await?,
            raw_csi: count_jsonl_lines(raw_csi_dir.join("index.jsonl")).await?,
            raw_stereo: count_jsonl_lines(raw_stereo_dir.join("pose3d.jsonl")).await?,
            raw_wifi: count_jsonl_lines(raw_wifi_dir.join("pose3d.jsonl")).await?,
            raw_stereo_media: count_jsonl_lines(raw_stereo_dir.join("media_index.jsonl")).await?,
            fused_state: count_jsonl_lines(fused_dir.join("fusion_state.jsonl")).await?,
            human_demo_pose: count_jsonl_lines(fused_dir.join("human_demo_pose.jsonl")).await?,
            teleop: count_jsonl_lines(teleop_dir.join("teleop_frame.jsonl")).await?,
            raw_robot_state: count_jsonl_lines(raw_robot_dir.join("state.jsonl")).await?,
            chunks: count_jsonl_lines(chunk_dir.join("chunk_state.jsonl")).await?,
            sync: count_jsonl_lines(sync_dir.join("time_sync_samples.jsonl")).await?,
            frame_correspondence: count_jsonl_lines(sync_dir.join("frame_correspondence.jsonl"))
                .await?,
        })
    }

    async fn calibration_flags(&self) -> (bool, bool, bool) {
        let base_dir = self.base_dir.clone();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return (false, false, false);
        }
        let calibration_dir = base_dir.join("calibration");
        let has_iphone_calibration =
            tokio::fs::metadata(calibration_dir.join("iphone_capture.json"))
                .await
                .is_ok();
        let has_stereo_calibration = tokio::fs::metadata(calibration_dir.join("stereo_pair.json"))
            .await
            .is_ok();
        let has_wifi_calibration = tokio::fs::metadata(calibration_dir.join("wifi_pose.json"))
            .await
            .is_ok();
        (
            has_iphone_calibration,
            has_stereo_calibration,
            has_wifi_calibration,
        )
    }

    async fn open_existing(
        base_dir: PathBuf,
        trip_id: String,
        session_id: String,
        created_unix_ms: u64,
        session_context: SessionCrowdContext,
        cfg: &Config,
    ) -> Result<Self, String> {
        let labels_dir = base_dir.join("labels");
        let raw_iphone_dir = base_dir.join("raw").join("iphone");
        let raw_iphone_wide_dir = raw_iphone_dir.join("wide");
        let raw_csi_dir = base_dir.join("raw").join("csi");
        let raw_csi_chunks_dir = raw_csi_dir.join("chunks");
        let raw_stereo_dir = base_dir.join("raw").join("stereo");
        let raw_wifi_dir = base_dir.join("raw").join("wifi");
        let raw_robot_dir = base_dir.join("raw").join("robot");
        let raw_iphone_main_dir = raw_iphone_dir.join("main");
        let raw_iphone_aux_dir = raw_iphone_dir.join("aux");
        let raw_iphone_depth_dir = raw_iphone_wide_dir.join("depth");
        let raw_iphone_fisheye_dir = raw_iphone_dir.join("fisheye");
        let raw_iphone_fisheye_chunks_dir = raw_iphone_fisheye_dir.join("chunks");
        let raw_stereo_left_dir = raw_stereo_dir.join("left");
        let raw_stereo_right_dir = raw_stereo_dir.join("right");
        let raw_stereo_preview_dir = raw_stereo_dir.join("preview");
        let fused_dir = base_dir.join("fused");
        let teleop_dir = base_dir.join("teleop");
        let derived_offline_dir = base_dir.join("derived").join("offline");
        let derived_vision_dir = base_dir.join("derived").join("vision");
        let derived_vision_embeddings_dir = derived_vision_dir.join("embeddings");
        let chunk_dir = base_dir.join("chunks");
        let clips_dir = base_dir.join("clips");
        let sync_dir = base_dir.join("sync");
        let calibration_dir = base_dir.join("calibration");
        let preview_dir = base_dir.join("preview");
        let preview_keyframes_dir = preview_dir.join("keyframes");
        let preview_clips_dir = preview_dir.join("clips");
        let qa_dir = base_dir.join("qa");
        let upload_dir = base_dir.join("upload");

        for d in [
            &labels_dir,
            &raw_iphone_dir,
            &raw_iphone_wide_dir,
            &raw_csi_dir,
            &raw_csi_chunks_dir,
            &raw_stereo_dir,
            &raw_wifi_dir,
            &raw_robot_dir,
            &raw_iphone_main_dir,
            &raw_iphone_aux_dir,
            &raw_iphone_depth_dir,
            &raw_iphone_fisheye_dir,
            &raw_iphone_fisheye_chunks_dir,
            &raw_stereo_left_dir,
            &raw_stereo_right_dir,
            &raw_stereo_preview_dir,
            &fused_dir,
            &teleop_dir,
            &derived_offline_dir,
            &derived_vision_dir,
            &derived_vision_embeddings_dir,
            &chunk_dir,
            &clips_dir,
            &sync_dir,
            &calibration_dir,
            &preview_dir,
            &preview_keyframes_dir,
            &preview_clips_dir,
            &qa_dir,
            &upload_dir,
        ] {
            tokio::fs::create_dir_all(d)
                .await
                .map_err(|e| format!("创建目录失败: {} ({e})", d.display()))?;
        }

        let labels = open_append(labels_dir.join("labels.jsonl")).await?;
        let iphone_kpts = open_append(raw_iphone_wide_dir.join("kpts_depth.jsonl")).await?;
        let iphone_pose_imu = open_append(raw_iphone_wide_dir.join("pose_imu.jsonl")).await?;
        let iphone_depth_index = open_append(raw_iphone_depth_dir.join("index.jsonl")).await?;
        let iphone_media_index = open_append(raw_iphone_wide_dir.join("media_index.jsonl")).await?;
        let csi_index = open_append(raw_csi_dir.join("index.jsonl")).await?;
        let csi_chunk_index = open_append(raw_csi_chunks_dir.join("index.jsonl")).await?;
        let csi_packets = open_append_bin(raw_csi_dir.join("packets.bin")).await?;
        let stereo_pose3d = open_append(raw_stereo_dir.join("pose3d.jsonl")).await?;
        let wifi_pose3d = open_append(raw_wifi_dir.join("pose3d.jsonl")).await?;
        let stereo_media_index = open_append(raw_stereo_dir.join("media_index.jsonl")).await?;
        let fused_state = open_append(fused_dir.join("fusion_state.jsonl")).await?;
        let human_demo_pose = open_append(fused_dir.join("human_demo_pose.jsonl")).await?;
        let teleop = open_append(teleop_dir.join("teleop_frame.jsonl")).await?;
        let robot_state = open_append(raw_robot_dir.join("state.jsonl")).await?;
        let offline_iphone_pose_v2 =
            open_append(derived_offline_dir.join("iphone_pose_v2.jsonl")).await?;
        let offline_stereo_pose_v2 =
            open_append(derived_offline_dir.join("stereo_pose_v2.jsonl")).await?;
        let offline_wifi_pose_v2 =
            open_append(derived_offline_dir.join("wifi_pose_v2.jsonl")).await?;
        let offline_fusion_state_v2 =
            open_append(derived_offline_dir.join("fusion_state_v2.jsonl")).await?;
        let offline_human_demo_pose_v2 =
            open_append(derived_offline_dir.join("human_demo_pose_v2.jsonl")).await?;
        let chunk_events = open_append(chunk_dir.join("chunk_state.jsonl")).await?;
        let time_sync_samples = open_append(sync_dir.join("time_sync_samples.jsonl")).await?;
        let frame_correspondence = open_append(sync_dir.join("frame_correspondence.jsonl")).await?;

        let mut seen_media_tracks = HashSet::new();
        for template in MEDIA_TRACK_TEMPLATES {
            let storage_track = normalized_track_storage_name(template.scope, template.track);
            let media_index_path = base_dir
                .join("raw")
                .join(template.scope)
                .join(storage_track)
                .join("media_index.jsonl");
            if tokio::fs::metadata(&media_index_path)
                .await
                .map(|meta| meta.len() > 0)
                .unwrap_or(false)
            {
                seen_media_tracks.insert((template.scope.to_string(), template.track.to_string()));
            }
        }

        let mut session_context = session_context;
        session_context.ensure_runtime_defaults(cfg);
        let semantic = Self::load_semantic_runtime_state(&base_dir, cfg).await?;

        let active = Self {
            trip_id,
            session_id,
            base_dir: base_dir.clone(),
            created_unix_ms,
            labels,
            iphone_kpts,
            iphone_pose_imu,
            iphone_depth_index,
            iphone_media_index,
            csi_index,
            csi_chunk_index,
            csi_packets,
            stereo_pose3d,
            wifi_pose3d,
            stereo_media_index,
            fused_state,
            human_demo_pose,
            teleop,
            robot_state,
            offline_iphone_pose_v2,
            offline_stereo_pose_v2,
            offline_wifi_pose_v2,
            offline_fusion_state_v2,
            offline_human_demo_pose_v2,
            chunk_events,
            time_sync_samples,
            frame_correspondence,
            next_csi_chunk_index: next_csi_chunk_index_from_disk(&base_dir).await?,
            pending_csi_chunk: None,
            seen_media_tracks,
            has_iphone_calibration: tokio::fs::metadata(
                calibration_dir.join("iphone_capture.json"),
            )
            .await
            .is_ok(),
            has_stereo_calibration: tokio::fs::metadata(calibration_dir.join("stereo_pair.json"))
                .await
                .is_ok(),
            has_wifi_calibration: tokio::fs::metadata(calibration_dir.join("wifi_pose.json"))
                .await
                .is_ok(),
            session_context,
            lines: LineCounters {
                labels: count_jsonl_lines(labels_dir.join("labels.jsonl")).await?,
                raw_iphone: count_jsonl_lines(raw_iphone_wide_dir.join("kpts_depth.jsonl")).await?,
                raw_iphone_pose_imu: count_jsonl_lines(raw_iphone_wide_dir.join("pose_imu.jsonl"))
                    .await?,
                raw_iphone_depth: count_jsonl_lines(raw_iphone_depth_dir.join("index.jsonl"))
                    .await?,
                raw_iphone_media: count_jsonl_lines(raw_iphone_wide_dir.join("media_index.jsonl"))
                    .await?,
                raw_csi: count_jsonl_lines(raw_csi_dir.join("index.jsonl")).await?,
                raw_stereo: count_jsonl_lines(raw_stereo_dir.join("pose3d.jsonl")).await?,
                raw_wifi: count_jsonl_lines(raw_wifi_dir.join("pose3d.jsonl")).await?,
                raw_stereo_media: count_jsonl_lines(raw_stereo_dir.join("media_index.jsonl"))
                    .await?,
                fused_state: count_jsonl_lines(fused_dir.join("fusion_state.jsonl")).await?,
                human_demo_pose: count_jsonl_lines(fused_dir.join("human_demo_pose.jsonl")).await?,
                teleop: count_jsonl_lines(teleop_dir.join("teleop_frame.jsonl")).await?,
                raw_robot_state: count_jsonl_lines(raw_robot_dir.join("state.jsonl")).await?,
                chunks: count_jsonl_lines(chunk_dir.join("chunk_state.jsonl")).await?,
                sync: count_jsonl_lines(sync_dir.join("time_sync_samples.jsonl")).await?,
                frame_correspondence: count_jsonl_lines(
                    sync_dir.join("frame_correspondence.jsonl"),
                )
                .await?,
            },
            clip_results: VecDeque::with_capacity(256),
            clip: ClipState::default(),
            semantic,
        };
        active.ensure_semantic_bundle_scaffold(cfg).await?;
        active.write_preview_manifest(cfg).await?;
        Ok(active)
    }

    async fn start(
        data_dir: &Path,
        trip_id: &str,
        session_id: &str,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<Self, String> {
        let session_id = session_id.trim();
        if session_id.is_empty()
            || matches!(session_id, "." | "..")
            || session_id
                .chars()
                .any(|ch| matches!(ch, '/' | '\\' | '\0' | '\n' | '\r'))
        {
            return Err(format!("invalid session_id: {session_id}"));
        }
        let data_dir = data_dir.to_path_buf();
        if data_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 data_dir: {}", data_dir.display()));
        }
        let base_dir = data_dir.join("session").join(session_id);
        let labels_dir = base_dir.join("labels");
        let raw_iphone_dir = base_dir.join("raw").join("iphone");
        let raw_iphone_wide_dir = raw_iphone_dir.join("wide");
        let raw_csi_dir = base_dir.join("raw").join("csi");
        let raw_csi_chunks_dir = raw_csi_dir.join("chunks");
        let raw_stereo_dir = base_dir.join("raw").join("stereo");
        let raw_wifi_dir = base_dir.join("raw").join("wifi");
        let raw_robot_dir = base_dir.join("raw").join("robot");
        let raw_iphone_main_dir = raw_iphone_dir.join("main");
        let raw_iphone_aux_dir = raw_iphone_dir.join("aux");
        let raw_iphone_depth_dir = raw_iphone_wide_dir.join("depth");
        let raw_iphone_fisheye_dir = raw_iphone_dir.join("fisheye");
        let raw_iphone_fisheye_chunks_dir = raw_iphone_fisheye_dir.join("chunks");
        let raw_stereo_left_dir = raw_stereo_dir.join("left");
        let raw_stereo_right_dir = raw_stereo_dir.join("right");
        let raw_stereo_preview_dir = raw_stereo_dir.join("preview");
        let fused_dir = base_dir.join("fused");
        let teleop_dir = base_dir.join("teleop");
        let derived_offline_dir = base_dir.join("derived").join("offline");
        let derived_vision_dir = base_dir.join("derived").join("vision");
        let derived_vision_embeddings_dir = derived_vision_dir.join("embeddings");
        let chunk_dir = base_dir.join("chunks");
        let clips_dir = base_dir.join("clips");
        let sync_dir = base_dir.join("sync");
        let calibration_dir = base_dir.join("calibration");
        let preview_dir = base_dir.join("preview");
        let preview_keyframes_dir = preview_dir.join("keyframes");
        let preview_clips_dir = preview_dir.join("clips");
        let qa_dir = base_dir.join("qa");
        let upload_dir = base_dir.join("upload");

        for d in [
            &labels_dir,
            &raw_iphone_dir,
            &raw_iphone_wide_dir,
            &raw_csi_dir,
            &raw_csi_chunks_dir,
            &raw_stereo_dir,
            &raw_wifi_dir,
            &raw_robot_dir,
            &raw_iphone_main_dir,
            &raw_iphone_aux_dir,
            &raw_iphone_depth_dir,
            &raw_iphone_fisheye_dir,
            &raw_iphone_fisheye_chunks_dir,
            &raw_stereo_left_dir,
            &raw_stereo_right_dir,
            &raw_stereo_preview_dir,
            &fused_dir,
            &teleop_dir,
            &derived_offline_dir,
            &derived_vision_dir,
            &derived_vision_embeddings_dir,
            &chunk_dir,
            &clips_dir,
            &sync_dir,
            &calibration_dir,
            &preview_dir,
            &preview_keyframes_dir,
            &preview_clips_dir,
            &qa_dir,
            &upload_dir,
        ] {
            tokio::fs::create_dir_all(d)
                .await
                .map_err(|e| format!("创建目录失败: {} ({e})", d.display()))?;
        }

        let created_unix_ms = now_unix_ms();

        let labels = open_append(labels_dir.join("labels.jsonl")).await?;
        let iphone_kpts = open_append(raw_iphone_wide_dir.join("kpts_depth.jsonl")).await?;
        let iphone_pose_imu = open_append(raw_iphone_wide_dir.join("pose_imu.jsonl")).await?;
        let iphone_depth_index = open_append(raw_iphone_depth_dir.join("index.jsonl")).await?;
        let iphone_media_index = open_append(raw_iphone_wide_dir.join("media_index.jsonl")).await?;
        let csi_index = open_append(raw_csi_dir.join("index.jsonl")).await?;
        let csi_chunk_index = open_append(raw_csi_chunks_dir.join("index.jsonl")).await?;
        let csi_packets = open_append_bin(raw_csi_dir.join("packets.bin")).await?;
        let stereo_pose3d = open_append(raw_stereo_dir.join("pose3d.jsonl")).await?;
        let wifi_pose3d = open_append(raw_wifi_dir.join("pose3d.jsonl")).await?;
        let stereo_media_index = open_append(raw_stereo_dir.join("media_index.jsonl")).await?;
        let fused_state = open_append(fused_dir.join("fusion_state.jsonl")).await?;
        let human_demo_pose = open_append(fused_dir.join("human_demo_pose.jsonl")).await?;
        let teleop = open_append(teleop_dir.join("teleop_frame.jsonl")).await?;
        let robot_state = open_append(raw_robot_dir.join("state.jsonl")).await?;
        let offline_iphone_pose_v2 =
            open_append(derived_offline_dir.join("iphone_pose_v2.jsonl")).await?;
        let offline_stereo_pose_v2 =
            open_append(derived_offline_dir.join("stereo_pose_v2.jsonl")).await?;
        let offline_wifi_pose_v2 =
            open_append(derived_offline_dir.join("wifi_pose_v2.jsonl")).await?;
        let offline_fusion_state_v2 =
            open_append(derived_offline_dir.join("fusion_state_v2.jsonl")).await?;
        let offline_human_demo_pose_v2 =
            open_append(derived_offline_dir.join("human_demo_pose_v2.jsonl")).await?;
        let chunk_events = open_append(chunk_dir.join("chunk_state.jsonl")).await?;
        let time_sync_samples = open_append(sync_dir.join("time_sync_samples.jsonl")).await?;
        let frame_correspondence = open_append(sync_dir.join("frame_correspondence.jsonl")).await?;

        let edge_frames = serde_json::json!({
            "type": "sensor_calibration_snapshot",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "sensor_kind": "edge_frames",
            "sensor_id": "edge_frames",
            "edge_time_ns": 0u64,
            "operator_frame": cfg.operator_frame,
            "robot_base_frame": cfg.robot_base_frame,
            "extrinsic_version": cfg.extrinsic_version,
            "extrinsic_translation_m": cfg.extrinsic_translation_m,
            "extrinsic_rotation_quat": cfg.extrinsic_rotation_quat,
            "notes": "edge static frame snapshot"
        });
        let edge_frames_path = calibration_dir.join("edge_frames.json");
        Self::write_json_pretty(edge_frames_path, &edge_frames).await?;

        debug!(%trip_id, %session_id, base_dir=%base_dir.display(), "recorder session started");
        let semantic = Self::load_semantic_runtime_state(&base_dir, cfg).await?;

        let mut session = Self {
            trip_id: trip_id.to_string(),
            session_id: session_id.to_string(),
            base_dir,
            created_unix_ms,
            labels,
            iphone_kpts,
            iphone_pose_imu,
            iphone_depth_index,
            iphone_media_index,
            csi_index,
            csi_chunk_index,
            csi_packets,
            stereo_pose3d,
            wifi_pose3d,
            stereo_media_index,
            fused_state,
            human_demo_pose,
            teleop,
            robot_state,
            offline_iphone_pose_v2,
            offline_stereo_pose_v2,
            offline_wifi_pose_v2,
            offline_fusion_state_v2,
            offline_human_demo_pose_v2,
            chunk_events,
            time_sync_samples,
            frame_correspondence,
            next_csi_chunk_index: 0,
            pending_csi_chunk: None,
            seen_media_tracks: HashSet::new(),
            has_iphone_calibration: false,
            has_stereo_calibration: false,
            has_wifi_calibration: false,
            session_context: SessionCrowdContext::from_config(cfg),
            lines: LineCounters::default(),
            clip_results: VecDeque::with_capacity(256),
            clip: ClipState::default(),
            semantic,
        };
        session.ensure_semantic_bundle_scaffold(cfg).await?;
        session.write_preview_manifest(cfg).await?;
        session.refresh_demo_bundle(protocol, cfg).await?;
        session
            .refresh_session_metadata_artifacts(protocol, cfg)
            .await?;
        Ok(session)
    }

    fn apply_session_context_update(&mut self, update: SessionContextUpdate) -> bool {
        let mut changed = false;

        if let Some(device_id) = normalize_non_empty(update.capture_device_id) {
            if self.session_context.capture_device_id.as_deref() != Some(device_id.as_str()) {
                self.session_context.capture_device_id = Some(device_id);
                changed = true;
            }
        }
        if let Some(operator_id) = normalize_non_empty(update.operator_id) {
            if self.session_context.operator_id.as_deref() != Some(operator_id.as_str()) {
                self.session_context.operator_id = Some(operator_id);
                changed = true;
            }
        }
        if let Some(task_id) = normalize_non_empty(update.task_id) {
            if self.session_context.task_id.as_deref() != Some(task_id.as_str()) {
                self.session_context.task_id = Some(task_id.clone());
                changed = true;
            }
            if !self
                .session_context
                .task_ids
                .iter()
                .any(|existing| existing == &task_id)
            {
                self.session_context.task_ids.push(task_id);
                changed = true;
            }
        }
        for task_id in normalize_task_ids(update.task_ids) {
            if !self
                .session_context
                .task_ids
                .iter()
                .any(|existing| existing == &task_id)
            {
                self.session_context.task_ids.push(task_id);
                changed = true;
            }
        }
        if self.session_context.task_id.is_none() {
            if let Some(first_task_id) = self.session_context.task_ids.first().cloned() {
                self.session_context.task_id = Some(first_task_id);
                changed = true;
            }
        }

        changed
    }

    async fn record_csi_chunk_packet(
        &mut self,
        cfg: &Config,
        edge_time_ns: u64,
        framed: &[u8],
        meta: CsiPacketMeta,
    ) -> Result<(), String> {
        let max_span_ns = cfg.csi_chunk_max_span_ms.saturating_mul(1_000_000);
        let should_roll_before_append = self.pending_csi_chunk.as_ref().is_some_and(|chunk| {
            chunk.packet_count > 0
                && edge_time_ns.saturating_sub(chunk.segment_start_edge_time_ns) >= max_span_ns
        });
        if should_roll_before_append {
            self.flush_pending_csi_chunk(cfg).await?;
        }

        if self.pending_csi_chunk.is_none() {
            let chunk_index = self.next_csi_chunk_index;
            self.next_csi_chunk_index = self.next_csi_chunk_index.saturating_add(1);
            self.pending_csi_chunk = Some(PendingCsiChunk::new(chunk_index, edge_time_ns));
        }

        let should_flush_after_append = {
            let chunk = self
                .pending_csi_chunk
                .as_mut()
                .ok_or_else(|| "pending csi chunk missing".to_string())?;
            chunk.push_packet(edge_time_ns, framed, meta);
            chunk.should_flush(cfg)
        };

        if should_flush_after_append {
            self.flush_pending_csi_chunk(cfg).await?;
        }
        Ok(())
    }

    async fn flush_pending_csi_chunk(&mut self, cfg: &Config) -> Result<(), String> {
        let Some(chunk) = self.pending_csi_chunk.take() else {
            return Ok(());
        };
        if chunk.packet_count == 0 || chunk.bytes.is_empty() {
            return Ok(());
        }

        let chunk_relpath = chunk.chunk_relpath(cfg);
        let chunk_path = self.base_dir.join(&chunk_relpath);
        if let Some(parent) = chunk_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("创建 CSI chunk 目录失败: {} ({e})", parent.display()))?;
        }
        tokio::fs::write(&chunk_path, &chunk.bytes)
            .await
            .map_err(|e| format!("写入 CSI chunk 失败: {} ({e})", chunk_path.display()))?;
        let chunk_event = chunk.chunk_event(&self.trip_id, &self.session_id, &chunk_relpath);
        Self::append_jsonl(&mut self.csi_chunk_index, &chunk_event).await?;
        Ok(())
    }

    async fn append_jsonl(f: &mut tokio::fs::File, v: &serde_json::Value) -> Result<(), String> {
        let mut line = serde_json::to_vec(v).map_err(|e| e.to_string())?;
        line.push(b'\n');
        f.write_all(&line)
            .await
            .map_err(|e| format!("jsonl 写入失败: {e}"))?;
        Ok(())
    }

    async fn append_jsonl_counted(
        f: &mut tokio::fs::File,
        counter: &mut u64,
        v: &serde_json::Value,
    ) -> Result<u64, String> {
        Self::append_jsonl(f, v).await?;
        *counter = counter.saturating_add(1);
        Ok(*counter)
    }

    async fn write_json_pretty(path: PathBuf, v: &serde_json::Value) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?;
        tokio::fs::write(&path, bytes)
            .await
            .map_err(|e| format!("json 写入失败: {} ({e})", path.display()))
    }

    async fn write_binary(path: PathBuf, bytes: &[u8]) -> Result<(), String> {
        tokio::fs::write(&path, bytes)
            .await
            .map_err(|e| format!("二进制写入失败: {} ({e})", path.display()))
    }

    async fn append_jsonl_relpath(
        &self,
        relpath: &str,
        value: &serde_json::Value,
    ) -> Result<(), String> {
        let mut file = open_append(self.base_dir.join(relpath)).await?;
        Self::append_jsonl(&mut file, value).await
    }

    async fn load_semantic_runtime_state(
        base_dir: &Path,
        cfg: &Config,
    ) -> Result<SemanticRuntimeState, String> {
        let manifest_path = base_dir.join("preview").join("preview_manifest.json");
        let preview_manifest = match tokio::fs::read_to_string(&manifest_path).await {
            Ok(content) => match serde_json::from_str::<PreviewManifestFile>(&content) {
                Ok(parsed) => PreviewManifestState {
                    status: parsed.status,
                    vlm_status: parsed.vlm_status,
                    degraded_reasons: parsed.degraded_reasons,
                    keyframes: parsed.keyframes,
                    clips: parsed.clips,
                },
                Err(_) => Self::default_preview_manifest_state(cfg),
            },
            Err(_) => Self::default_preview_manifest_state(cfg),
        };
        let next_event_index = count_jsonl_lines(
            base_dir
                .join("derived")
                .join("vision")
                .join("vlm_events.jsonl"),
        )
        .await
        .unwrap_or(0);
        let next_segment_index = count_jsonl_lines(
            base_dir
                .join("derived")
                .join("vision")
                .join("vlm_segments.jsonl"),
        )
        .await
        .unwrap_or(0);
        let next_keyframe_index = preview_manifest.keyframes.len() as u64;
        let last_keyframe_edge_time_ns = preview_manifest
            .keyframes
            .last()
            .map(|item| item.edge_time_ns);
        let last_camera_mode = preview_manifest
            .keyframes
            .last()
            .map(|item| item.camera_mode.clone())
            .unwrap_or_default();
        Ok(SemanticRuntimeState {
            next_keyframe_index,
            next_event_index,
            next_segment_index,
            last_keyframe_edge_time_ns,
            last_camera_mode,
            current_segment: None,
            preview_manifest,
            pending_jobs: VecDeque::new(),
        })
    }

    fn default_preview_manifest_state(cfg: &Config) -> PreviewManifestState {
        PreviewManifestState {
            status: if cfg.preview_generation_enabled {
                "pending_generation".to_string()
            } else {
                "disabled".to_string()
            },
            vlm_status: if cfg.vlm_indexing_enabled {
                "pending_generation".to_string()
            } else {
                "disabled".to_string()
            },
            degraded_reasons: Vec::new(),
            keyframes: Vec::new(),
            clips: Vec::new(),
        }
    }

    fn preview_manifest_snapshot(&self, cfg: &Config) -> PreviewManifestFile {
        PreviewManifestFile {
            ty: "preview_manifest".to_string(),
            schema_version: "1.0.0".to_string(),
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            generated_unix_ms: now_unix_ms(),
            runtime_profile: cfg.runtime_profile_name().to_string(),
            upload_policy_mode: cfg.upload_policy_mode_name().to_string(),
            raw_residency_default: cfg.raw_residency_default().to_string(),
            preview_residency_default: cfg.preview_residency_default().to_string(),
            preview_generation_enabled: cfg.preview_generation_enabled,
            vlm_indexing_enabled: cfg.vlm_indexing_enabled,
            model_id: cfg.vlm_model_id.clone(),
            fallback_model_id: cfg.vlm_fallback_model_id.clone(),
            prompt_version: cfg.vlm_prompt_version.clone(),
            vlm_sidecar_base: cfg.vlm_sidecar_base.clone(),
            vlm_live_interval_ms: cfg.vlm_keyframe_interval_ms,
            vlm_event_trigger_enabled: cfg.vlm_event_trigger_enabled,
            vlm_event_trigger_camera_mode_change_enabled: cfg
                .vlm_event_trigger_camera_mode_change_enabled,
            vlm_inference_timeout_ms: cfg.vlm_inference_timeout_ms,
            vlm_auto_fallback_latency_ms: cfg.vlm_auto_fallback_latency_ms,
            vlm_auto_fallback_cooldown_ms: cfg.vlm_auto_fallback_cooldown_ms,
            vlm_max_consecutive_failures: cfg.vlm_max_consecutive_failures,
            status: self.semantic.preview_manifest.status.clone(),
            vlm_status: self.semantic.preview_manifest.vlm_status.clone(),
            degraded_reasons: self.semantic.preview_manifest.degraded_reasons.clone(),
            keyframes: self.semantic.preview_manifest.keyframes.clone(),
            clips: self.semantic.preview_manifest.clips.clone(),
        }
    }

    async fn write_preview_manifest(&self, cfg: &Config) -> Result<(), String> {
        let value = serde_json::to_value(self.preview_manifest_snapshot(cfg))
            .map_err(|e| format!("preview manifest 序列化失败: {e}"))?;
        Self::write_json_pretty(
            self.base_dir.join("preview").join("preview_manifest.json"),
            &value,
        )
        .await
    }

    fn mark_semantic_degraded(
        &mut self,
        stage: &str,
        detail: impl Into<String>,
        cfg: &Config,
    ) -> bool {
        let detail = detail.into();
        if self
            .semantic
            .preview_manifest
            .degraded_reasons
            .iter()
            .any(|item| item.stage == stage && item.detail == detail)
        {
            return false;
        }
        self.semantic
            .preview_manifest
            .degraded_reasons
            .push(PreviewDegradedReason {
                stage: stage.to_string(),
                detail,
                recorded_unix_ms: now_unix_ms(),
            });
        if cfg.preview_generation_enabled {
            self.semantic.preview_manifest.status = "degraded".to_string();
        }
        if cfg.vlm_indexing_enabled {
            self.semantic.preview_manifest.vlm_status = "degraded".to_string();
        }
        true
    }

    fn refresh_semantic_status(&mut self, cfg: &Config) {
        if cfg.preview_generation_enabled {
            self.semantic.preview_manifest.status =
                if self.semantic.preview_manifest.degraded_reasons.is_empty() {
                    if self.semantic.preview_manifest.keyframes.is_empty() {
                        "pending_generation".to_string()
                    } else {
                        "ready".to_string()
                    }
                } else {
                    "degraded".to_string()
                };
        } else {
            self.semantic.preview_manifest.status = "disabled".to_string();
        }

        if cfg.vlm_indexing_enabled {
            self.semantic.preview_manifest.vlm_status =
                if self.semantic.preview_manifest.degraded_reasons.is_empty() {
                    if self.semantic.next_event_index == 0 && self.semantic.next_segment_index == 0
                    {
                        "pending_generation".to_string()
                    } else {
                        "ready".to_string()
                    }
                } else {
                    "degraded".to_string()
                };
        } else {
            self.semantic.preview_manifest.vlm_status = "disabled".to_string();
        }
    }

    fn semantic_segment_should_roll(
        &self,
        cfg: &Config,
        edge_time_ns: u64,
        camera_mode: &str,
    ) -> bool {
        let Some(segment) = self.semantic.current_segment.as_ref() else {
            return false;
        };
        let segment_window_ns = cfg.vlm_segment_window_ms.saturating_mul(1_000_000);
        edge_time_ns.saturating_sub(segment.start_edge_time_ns) >= segment_window_ns
            || (!segment.camera_mode.is_empty() && segment.camera_mode != camera_mode)
    }

    async fn write_embedding_artifact(
        &self,
        relpath: &str,
        entry_id: &str,
        entry_type: &str,
        source_text: &str,
    ) -> Result<(), String> {
        let embedding = serde_json::json!({
            "type": "semantic_embedding",
            "schema_version": "1.0.0",
            "trip_id": self.trip_id.clone(),
            "session_id": self.session_id.clone(),
            "entry_id": entry_id,
            "entry_type": entry_type,
            "source_text": source_text,
            "vector": deterministic_embedding(source_text),
        });
        Self::write_json_pretty(self.base_dir.join(relpath), &embedding).await
    }

    async fn finalize_current_semantic_segment(&mut self, cfg: &Config) -> bool {
        let Some(segment) = self.semantic.current_segment.take() else {
            return false;
        };
        if segment.keyframe_relpaths.is_empty() {
            return false;
        }

        let dominant_action =
            dominant_string(&segment.actions).unwrap_or_else(|| "steady_capture".to_string());
        let summary_tags = unique_strings(&segment.tags);
        let summary_objects = unique_strings(&segment.objects);
        let segment_model_id = last_non_empty_string(&segment.model_ids);
        let segment_inference_source = last_non_empty_string(&segment.inference_sources);
        let segment_latency_ms = mean_non_empty_f64(&segment.latencies_ms);
        let summary_caption = build_segment_caption(
            &segment.camera_mode,
            segment.keyframe_relpaths.len(),
            segment.start_edge_time_ns,
            segment.end_edge_time_ns,
            &dominant_action,
        );
        let segment_id = format!("segment_{:06}", segment.index);
        let embedding_relpath = if cfg.vlm_indexing_enabled {
            let relpath = format!("derived/vision/embeddings/{segment_id}.json");
            let source_text = format!(
                "{} {} {} {}",
                summary_caption,
                summary_tags.join(" "),
                summary_objects.join(" "),
                dominant_action
            );
            if let Err(error) = self
                .write_embedding_artifact(&relpath, &segment_id, "vlm_segment", &source_text)
                .await
            {
                warn!(error=%error, %segment_id, "write segment embedding failed");
                self.mark_semantic_degraded("segment_embedding", error, cfg);
                None
            } else {
                Some(relpath)
            }
        } else {
            None
        };

        if cfg.vlm_indexing_enabled {
            let segment_value = serde_json::json!({
                "type": "vlm_segment",
                "schema_version": "1.0.0",
                "trip_id": self.trip_id.clone(),
                "session_id": self.session_id.clone(),
                "segment_id": segment_id,
                "start_edge_time_ns": segment.start_edge_time_ns,
                "end_edge_time_ns": segment.end_edge_time_ns,
                "start_source_time_ns": segment.start_source_time_ns,
                "end_source_time_ns": segment.end_source_time_ns,
                "start_frame_id": segment.start_frame_id,
                "end_frame_id": segment.end_frame_id,
                "camera_mode": segment.camera_mode,
                "event_ids": segment.event_ids,
                "keyframe_ids": segment.keyframe_ids,
                "caption": summary_caption,
                "tags": summary_tags,
                "objects": summary_objects,
                "action_guess": dominant_action,
                "model_id": segment_model_id.clone().unwrap_or_else(|| cfg.vlm_model_id.clone()),
                "prompt_version": cfg.vlm_prompt_version.clone(),
                "inference_source": segment_inference_source,
                "latency_ms": segment_latency_ms,
                "embedding_relpath": embedding_relpath,
            });
            if let Err(error) = self
                .append_jsonl_relpath("derived/vision/vlm_segments.jsonl", &segment_value)
                .await
            {
                warn!(error=%error, "append vlm segment failed");
                self.mark_semantic_degraded("vlm_segment_append", error, cfg);
            } else {
            }
        }
        self.semantic.next_segment_index = self.semantic.next_segment_index.saturating_add(1);

        if cfg.preview_generation_enabled {
            let clip_id = format!(
                "preview_clip_{:06}",
                self.semantic.preview_manifest.clips.len()
            );
            let clip_relpath = format!("preview/clips/{clip_id}.gif");
            let clip_result = self
                .write_gif_preview(
                    &clip_relpath,
                    &segment.keyframe_relpaths,
                    cfg.preview_clip_max_frames,
                    cfg.preview_clip_frame_delay_ms,
                )
                .await;
            match clip_result {
                Ok(()) => {
                    self.semantic
                        .preview_manifest
                        .clips
                        .push(PreviewClipRecord {
                            id: clip_id,
                            relpath: clip_relpath,
                            mime_type: "image/gif".to_string(),
                            status: "ready".to_string(),
                            start_edge_time_ns: segment.start_edge_time_ns,
                            end_edge_time_ns: segment.end_edge_time_ns,
                            keyframe_ids: segment.keyframe_ids,
                            caption: summary_caption,
                            objects: unique_strings(&segment.objects),
                            tags: unique_strings(&segment.tags),
                            action_guess: dominant_action,
                            model_id: segment_model_id,
                            inference_source: segment_inference_source,
                            latency_ms: segment_latency_ms,
                            embedding_relpath,
                        });
                }
                Err(error) => {
                    warn!(error=%error, "write preview clip failed");
                    self.mark_semantic_degraded("preview_clip_generation", error, cfg);
                }
            }
        }

        self.refresh_semantic_status(cfg);
        if let Err(error) = self.write_preview_manifest(cfg).await {
            warn!(error=%error, "write preview manifest after segment finalize failed");
        }
        true
    }

    async fn write_gif_preview(
        &self,
        relpath: &str,
        keyframe_relpaths: &[String],
        max_frames: usize,
        frame_delay_ms: u64,
    ) -> Result<(), String> {
        let base_dir = self.base_dir.clone();
        let output_relpath = relpath.to_string();
        let selected_relpaths = select_preview_frames(keyframe_relpaths, max_frames);
        let delay_ms = frame_delay_ms.max(80);
        tokio::task::spawn_blocking(move || -> Result<(), String> {
            if selected_relpaths.is_empty() {
                return Err("没有可用于 preview clip 的 keyframe".to_string());
            }
            let output_path = base_dir.join(&output_relpath);
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!("创建 preview clip 目录失败: {} ({e})", parent.display())
                })?;
            }
            let file = std::fs::File::create(&output_path)
                .map_err(|e| format!("创建 preview clip 失败: {} ({e})", output_path.display()))?;
            let mut encoder = GifEncoder::new(file);
            encoder
                .set_repeat(Repeat::Infinite)
                .map_err(|e| format!("配置 GIF repeat 失败: {e}"))?;

            let mut target_dims: Option<(u32, u32)> = None;
            let mut frames = Vec::new();
            for rel in selected_relpaths {
                let image_path = base_dir.join(&rel);
                let bytes = std::fs::read(&image_path)
                    .map_err(|e| format!("读取 keyframe 失败: {} ({e})", image_path.display()))?;
                let decoded = image::load_from_memory(&bytes).map_err(|e| {
                    format!("解码 keyframe JPEG 失败: {} ({e})", image_path.display())
                })?;
                let rgba = match target_dims {
                    Some((width, height))
                        if decoded.width() != width || decoded.height() != height =>
                    {
                        image::imageops::resize(
                            &decoded.to_rgba8(),
                            width,
                            height,
                            FilterType::Triangle,
                        )
                    }
                    Some(_) => decoded.to_rgba8(),
                    None => {
                        target_dims = Some((decoded.width(), decoded.height()));
                        decoded.to_rgba8()
                    }
                };
                frames.push(Frame::from_parts(
                    rgba,
                    0,
                    0,
                    Delay::from_saturating_duration(Duration::from_millis(delay_ms)),
                ));
            }
            encoder
                .encode_frames(frames.into_iter())
                .map_err(|e| format!("编码 preview GIF 失败: {e}"))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("preview clip 后台任务失败: {e}"))?
    }

    async fn flush_pending_semantic_jobs(
        &mut self,
        cfg: &Config,
        wait_for_completion: bool,
    ) -> bool {
        if self.semantic.pending_jobs.is_empty() {
            return false;
        }

        let mut changed = false;
        let mut remaining = VecDeque::new();
        while let Some(pending) = self.semantic.pending_jobs.pop_front() {
            let PendingSemanticInference { meta, handle } = pending;
            if !wait_for_completion && !handle.is_finished() {
                remaining.push_back(PendingSemanticInference { meta, handle });
                continue;
            }
            let outcome = handle
                .await
                .map_err(|error| format!("vlm sidecar job join failed: {error}"))
                .and_then(|result| result);
            changed |= self
                .materialize_semantic_inference(cfg, meta, outcome)
                .await;
        }
        self.semantic.pending_jobs = remaining;
        if changed {
            self.refresh_semantic_status(cfg);
            if let Err(error) = self.write_preview_manifest(cfg).await {
                warn!(error=%error, "write preview manifest after flushing semantic jobs failed");
            }
        }
        changed
    }

    async fn materialize_semantic_inference(
        &mut self,
        cfg: &Config,
        meta: SemanticInferenceMetadata,
        outcome: Result<SemanticInferenceResult, String>,
    ) -> bool {
        let mut changed = false;
        let result = match outcome {
            Ok(result) => result,
            Err(error) => {
                warn!(
                    error=%error,
                    event_id=%meta.event_id,
                    keyframe_id=%meta.keyframe_id,
                    "vlm sidecar inference failed, falling back to heuristic semantics"
                );
                changed |= self.mark_semantic_degraded("vlm_sidecar_inference", &error, cfg);
                self.build_heuristic_inference_result(&meta, "heuristic_fallback")
            }
        };
        if !result.degraded_reasons.is_empty() {
            for reason in &result.degraded_reasons {
                changed |= self.mark_semantic_degraded("vlm_runtime", reason, cfg);
            }
        }

        if meta.roll_segment_before_event {
            changed |= self.finalize_current_semantic_segment(cfg).await;
        }

        let caption = result.caption.clone();
        let tags = result.tags.clone();
        let objects = result.objects.clone();
        let action_guess = result.action_guess.clone();
        let model_id = result.model_id.clone();
        let inference_source = result.inference_source.clone();
        let latency_ms = result.latency_ms;
        let degraded_reasons = result.degraded_reasons.clone();

        let embedding_relpath = if cfg.vlm_indexing_enabled {
            let relpath = format!("derived/vision/embeddings/{}.json", meta.event_id);
            let source_text = format!(
                "{} {} {} {}",
                caption,
                tags.join(" "),
                objects.join(" "),
                action_guess
            );
            if let Err(error) = self
                .write_embedding_artifact(&relpath, &meta.event_id, "vlm_event", &source_text)
                .await
            {
                warn!(error=%error, event_id=%meta.event_id, "write event embedding failed");
                changed |= self.mark_semantic_degraded("event_embedding", error, cfg);
                None
            } else {
                Some(relpath)
            }
        } else {
            None
        };

        if cfg.vlm_indexing_enabled {
            let event_value = serde_json::json!({
                "type": "vlm_event",
                "schema_version": "1.0.0",
                "trip_id": self.trip_id.clone(),
                "session_id": self.session_id.clone(),
                "event_id": meta.event_id,
                "frame_id": meta.frame_id,
                "source_time_ns": meta.source_time_ns,
                "edge_time_ns": meta.edge_time_ns,
                "camera_mode": meta.camera_mode,
                "frame_relpath": meta.keyframe_relpath,
                "model_id": model_id,
                "prompt_version": cfg.vlm_prompt_version.clone(),
                "caption": caption,
                "tags": tags,
                "objects": objects,
                "action_guess": action_guess,
                "sample_reasons": meta.sample_reasons,
                "embedding_relpath": embedding_relpath,
                "latency_ms": latency_ms,
                "inference_source": inference_source,
                "degraded_reasons": degraded_reasons,
                "started_unix_ms": meta.started_unix_ms,
            });
            if let Err(error) = self
                .append_jsonl_relpath("derived/vision/vlm_events.jsonl", &event_value)
                .await
            {
                warn!(error=%error, event_id=%meta.event_id, "append vlm event failed");
                changed |= self.mark_semantic_degraded("vlm_event_append", error, cfg);
            } else {
                changed = true;
            }
        }

        self.semantic.next_event_index = self.semantic.next_event_index.saturating_add(1);

        let inference_status =
            semantic_inference_status(&result.inference_source, result.latency_ms);
        if let Some(preview_record) = self
            .semantic
            .preview_manifest
            .keyframes
            .get_mut(meta.preview_record_index)
        {
            preview_record.caption = result.caption.clone();
            preview_record.tags = result.tags.clone();
            preview_record.objects = result.objects.clone();
            preview_record.action_guess = result.action_guess.clone();
            preview_record.model_id = Some(result.model_id.clone());
            preview_record.inference_source = Some(result.inference_source.clone());
            preview_record.latency_ms = result.latency_ms;
            preview_record.embedding_relpath = embedding_relpath.clone();
            preview_record.inference_status = inference_status;
            changed = true;
        }

        if self.semantic.current_segment.is_none() {
            self.semantic.current_segment = Some(SemanticSegmentAccumulator {
                index: self.semantic.next_segment_index,
                start_edge_time_ns: meta.edge_time_ns,
                start_source_time_ns: meta.source_time_ns,
                end_edge_time_ns: meta.edge_time_ns,
                end_source_time_ns: meta.source_time_ns,
                start_frame_id: meta.frame_id,
                end_frame_id: meta.frame_id,
                camera_mode: meta.camera_mode.clone(),
                ..SemanticSegmentAccumulator::default()
            });
        }
        let segment = self
            .semantic
            .current_segment
            .as_mut()
            .expect("semantic segment must exist");
        segment.end_edge_time_ns = meta.edge_time_ns;
        segment.end_source_time_ns = meta.source_time_ns;
        segment.end_frame_id = meta.frame_id;
        segment.event_ids.push(meta.event_id);
        segment.keyframe_ids.push(meta.keyframe_id);
        segment.keyframe_relpaths.push(meta.keyframe_relpath);
        segment.captions.push(result.caption);
        segment.tags.extend(result.tags);
        segment.objects.extend(result.objects);
        segment.actions.push(result.action_guess);
        segment.model_ids.push(result.model_id);
        segment.inference_sources.push(result.inference_source);
        if let Some(latency_ms) = result.latency_ms {
            segment.latencies_ms.push(latency_ms);
        }

        changed
    }

    fn build_heuristic_inference_result(
        &self,
        meta: &SemanticInferenceMetadata,
        inference_source: &str,
    ) -> SemanticInferenceResult {
        SemanticInferenceResult {
            caption: meta.fallback_caption.clone(),
            tags: meta.fallback_tags.clone(),
            objects: meta.fallback_objects.clone(),
            action_guess: meta.fallback_action_guess.clone(),
            model_id: if inference_source == "heuristic_direct" {
                "edge_semantic_heuristic_direct".to_string()
            } else {
                "edge_semantic_heuristic_fallback".to_string()
            },
            inference_source: inference_source.to_string(),
            latency_ms: Some(now_unix_ms().saturating_sub(meta.started_unix_ms) as f64),
            degraded_reasons: if inference_source == "heuristic_direct" {
                Vec::new()
            } else {
                vec!["sidecar_unavailable_using_heuristic_fallback".to_string()]
            },
        }
    }

    fn queue_semantic_inference(
        &mut self,
        cfg: &Config,
        meta: SemanticInferenceMetadata,
        primary_image_bytes: &[u8],
    ) {
        let cfg = cfg.clone();
        let image_bytes = primary_image_bytes.to_vec();
        let frame_id = meta.frame_id;
        let source_time_ns = meta.source_time_ns;
        let edge_time_ns = meta.edge_time_ns;
        let camera_mode = meta.camera_mode.clone();
        let sample_reasons = meta.sample_reasons.clone();
        let handle = tokio::spawn(async move {
            request_vlm_sidecar_inference(
                cfg,
                image_bytes,
                frame_id,
                source_time_ns,
                edge_time_ns,
                camera_mode,
                sample_reasons,
            )
            .await
        });
        self.semantic
            .pending_jobs
            .push_back(PendingSemanticInference { meta, handle });
    }

    async fn record_phone_semantic_outputs(
        &mut self,
        cfg: &Config,
        v: &serde_json::Value,
        primary_image_bytes: Option<&[u8]>,
    ) -> bool {
        if !cfg.vlm_indexing_enabled && !cfg.preview_generation_enabled {
            return false;
        }

        let mut changed = self.flush_pending_semantic_jobs(cfg, false).await;
        if primary_image_bytes.is_none() {
            changed |= self.mark_semantic_degraded(
                "primary_image_missing",
                "preview/vlm requested but no decoded primary image bytes were forwarded",
                cfg,
            );
            if let Some(error) = v
                .get("primary_image_decode_error")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                changed |= self.mark_semantic_degraded("primary_image_decode", error, cfg);
                self.refresh_semantic_status(cfg);
                if let Err(write_error) = self.write_preview_manifest(cfg).await {
                    warn!(error=%write_error, "write degraded preview manifest failed");
                }
            }
            return changed;
        }

        let frame_id = v
            .get("frame_id")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let source_time_ns = v
            .get("source_time_ns")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let edge_time_ns = v
            .get("edge_time_ns")
            .and_then(|value| value.as_u64())
            .unwrap_or(source_time_ns);
        let camera_mode = v
            .get("camera_mode")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .trim()
            .to_string();
        let action_guess = semantic_action_guess(v);

        let first_sample = self.semantic.last_keyframe_edge_time_ns.is_none();
        let interval_ns = cfg.vlm_keyframe_interval_ms.saturating_mul(1_000_000);
        let fixed_due = self
            .semantic
            .last_keyframe_edge_time_ns
            .map(|last| edge_time_ns.saturating_sub(last) >= interval_ns)
            .unwrap_or(true);
        let camera_mode_change_due = cfg.vlm_event_trigger_enabled
            && cfg.vlm_event_trigger_camera_mode_change_enabled
            && !first_sample
            && self.semantic.last_camera_mode != camera_mode;
        let action_change_due = cfg.vlm_event_trigger_enabled
            && self
                .semantic
                .current_segment
                .as_ref()
                .and_then(|segment| segment.actions.last())
                .is_some_and(|last_action| last_action != &action_guess);
        let event_due = first_sample || camera_mode_change_due || action_change_due;
        if !fixed_due && !event_due {
            return false;
        }

        let keyframe_id = format!("keyframe_{:06}", self.semantic.next_keyframe_index);
        let event_id = format!("event_{:06}", self.semantic.next_event_index);
        let keyframe_relpath = format!(
            "preview/keyframes/{keyframe_id}__{:020}.jpg",
            source_time_ns
        );
        if let Err(error) = Self::write_binary(
            self.base_dir.join(&keyframe_relpath),
            primary_image_bytes.unwrap_or_default(),
        )
        .await
        {
            warn!(error=%error, "write preview keyframe failed");
            changed |= self.mark_semantic_degraded("preview_keyframe_write", error, cfg);
            self.refresh_semantic_status(cfg);
            if let Err(write_error) = self.write_preview_manifest(cfg).await {
                warn!(error=%write_error, "write degraded preview manifest failed");
            }
            return changed;
        }

        let sample_reasons = {
            let mut reasons = Vec::new();
            if fixed_due {
                reasons.push("fixed_interval".to_string());
            }
            if first_sample {
                reasons.push("session_start".to_string());
            }
            if event_due {
                reasons.push("event_trigger".to_string());
            }
            if camera_mode_change_due {
                reasons.push("camera_mode_change".to_string());
            }
            if action_change_due {
                reasons.push("action_change".to_string());
            }
            reasons
        };
        let tags = semantic_tags(v, &camera_mode, &action_guess);
        let objects = semantic_objects(v);
        let caption = build_event_caption(&camera_mode, v, &action_guess);

        let preview_record = PreviewKeyframeRecord {
            id: keyframe_id.clone(),
            relpath: keyframe_relpath.clone(),
            frame_id,
            source_time_ns,
            edge_time_ns,
            camera_mode: camera_mode.clone(),
            sample_reasons: sample_reasons.clone(),
            caption: caption.clone(),
            objects: objects.clone(),
            tags: tags.clone(),
            action_guess: action_guess.clone(),
            model_id: None,
            inference_source: None,
            latency_ms: None,
            inference_status: if cfg.vlm_sidecar_base.trim().is_empty() {
                "heuristic_direct".to_string()
            } else {
                "pending_sidecar".to_string()
            },
            embedding_relpath: None,
        };
        let preview_record_index = self.semantic.preview_manifest.keyframes.len();
        self.semantic
            .preview_manifest
            .keyframes
            .push(preview_record);

        let roll_segment_before_event =
            self.semantic_segment_should_roll(cfg, edge_time_ns, &camera_mode);
        self.semantic.next_keyframe_index = self.semantic.next_keyframe_index.saturating_add(1);
        self.semantic.last_keyframe_edge_time_ns = Some(edge_time_ns);
        self.semantic.last_camera_mode = camera_mode.clone();
        let meta = SemanticInferenceMetadata {
            preview_record_index,
            event_id,
            keyframe_id,
            keyframe_relpath,
            frame_id,
            source_time_ns,
            edge_time_ns,
            camera_mode,
            sample_reasons,
            roll_segment_before_event,
            fallback_caption: caption,
            fallback_tags: tags,
            fallback_objects: objects,
            fallback_action_guess: action_guess,
            started_unix_ms: now_unix_ms(),
        };
        if cfg.vlm_sidecar_base.trim().is_empty() {
            self.materialize_semantic_inference(
                cfg,
                meta.clone(),
                Ok(self.build_heuristic_inference_result(&meta, "heuristic_direct")),
            )
            .await;
        } else if let Some(primary_image_bytes) = primary_image_bytes {
            self.queue_semantic_inference(cfg, meta, primary_image_bytes);
        }
        self.refresh_semantic_status(cfg);
        if let Err(error) = self.write_preview_manifest(cfg).await {
            warn!(error=%error, "write preview manifest after keyframe failed");
        }
        true
    }

    async fn ensure_semantic_bundle_scaffold(&self, cfg: &Config) -> Result<(), String> {
        let base_dir = self.base_dir.clone();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        let derived_vision_dir = base_dir.join("derived").join("vision");
        let derived_vision_embeddings_dir = derived_vision_dir.join("embeddings");
        let preview_dir = base_dir.join("preview");
        let preview_keyframes_dir = preview_dir.join("keyframes");
        let preview_clips_dir = preview_dir.join("clips");
        for dir in [
            &derived_vision_dir,
            &derived_vision_embeddings_dir,
            &preview_dir,
            &preview_keyframes_dir,
            &preview_clips_dir,
        ] {
            tokio::fs::create_dir_all(dir)
                .await
                .map_err(|e| format!("创建语义/预览目录失败: {} ({e})", dir.display()))?;
        }

        open_append(derived_vision_dir.join("vlm_events.jsonl")).await?;
        open_append(derived_vision_dir.join("vlm_segments.jsonl")).await?;
        if tokio::fs::metadata(preview_dir.join("preview_manifest.json"))
            .await
            .is_err()
        {
            self.write_preview_manifest(cfg).await?;
        }
        Ok(())
    }

    async fn refresh_offline_manifest(&self) -> Result<(), String> {
        let base_dir = self.base_dir.as_path();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        let manifest = OfflineManifest {
            ty: "offline_manifest",
            schema_version: "1.0.0",
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            generated_unix_ms: now_unix_ms(),
            pipeline: OfflineManifestPipeline {
                pipeline_version: "offline-bootstrap-v0",
                generation_mode: "bootstrap_live_mirror",
                materialization_state: "ready_for_recompute",
                notes: "当前 derived/offline 先镜像实时事实流，为 Isaac/数字孪生离线重算提供稳定入口；后续可由真实离线任务原位覆盖为 *_v2 正式产物。",
            },
            artifacts: vec![
                OfflineManifestArtifact {
                    id: "iphone_pose_v2",
                    relpath: "derived/offline/iphone_pose_v2.jsonl",
                    format: "jsonl",
                    materialization_state: "mirrored_from_online",
                    generation_mode: "bootstrap_live_mirror",
                    source_relpaths: vec!["raw/iphone/wide/kpts_depth.jsonl"],
                },
                OfflineManifestArtifact {
                    id: "stereo_pose_v2",
                    relpath: "derived/offline/stereo_pose_v2.jsonl",
                    format: "jsonl",
                    materialization_state: "mirrored_from_online",
                    generation_mode: "bootstrap_live_mirror",
                    source_relpaths: vec!["raw/stereo/pose3d.jsonl"],
                },
                OfflineManifestArtifact {
                    id: "wifi_pose_v2",
                    relpath: "derived/offline/wifi_pose_v2.jsonl",
                    format: "jsonl",
                    materialization_state: "mirrored_from_online",
                    generation_mode: "bootstrap_live_mirror",
                    source_relpaths: vec!["raw/wifi/pose3d.jsonl"],
                },
                OfflineManifestArtifact {
                    id: "fusion_state_v2",
                    relpath: "derived/offline/fusion_state_v2.jsonl",
                    format: "jsonl",
                    materialization_state: "mirrored_from_online",
                    generation_mode: "bootstrap_live_mirror",
                    source_relpaths: vec!["fused/fusion_state.jsonl"],
                },
                OfflineManifestArtifact {
                    id: "human_demo_pose_v2",
                    relpath: "derived/offline/human_demo_pose_v2.jsonl",
                    format: "jsonl",
                    materialization_state: "mirrored_from_online",
                    generation_mode: "bootstrap_live_mirror",
                    source_relpaths: vec!["fused/human_demo_pose.jsonl"],
                },
            ],
        };
        let value = serde_json::to_value(manifest).map_err(|e| e.to_string())?;
        Self::write_json_pretty(
            base_dir
                .join("derived")
                .join("offline")
                .join("offline_manifest.json"),
            &value,
        )
        .await
    }

    async fn refresh_demo_bundle(
        &mut self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<(), String> {
        let base_dir = self.base_dir.clone();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        self.hydrate_session_context_from_disk(cfg).await?;
        self.refresh_offline_manifest().await?;
        let mut calibration_snapshot_paths = vec!["calibration/edge_frames.json".to_string()];
        if self.has_iphone_calibration {
            calibration_snapshot_paths.push("calibration/iphone_capture.json".to_string());
        }
        if tokio::fs::metadata(
            base_dir
                .join("calibration")
                .join("iphone_fisheye.json"),
        )
        .await
        .is_ok()
        {
            calibration_snapshot_paths.push("calibration/iphone_fisheye.json".to_string());
        }
        if self.has_stereo_calibration {
            calibration_snapshot_paths.push("calibration/stereo_pair.json".to_string());
        }
        if self.has_wifi_calibration {
            calibration_snapshot_paths.push("calibration/wifi_pose.json".to_string());
        }
        let media_tracks = self.collect_media_tracks().await?;
        let chunk_dirs = media_tracks
            .iter()
            .map(|track| track.chunk_dir.clone())
            .collect::<Vec<_>>();

        let bundle = DemoCaptureBundle {
            ty: "demo_capture_bundle",
            schema_version: "1.0.0",
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            created_unix_ms: self.created_unix_ms,
            generated_unix_ms: now_unix_ms(),
            protocol: protocol.clone(),
            frames: DemoBundleFrames {
                operator_frame: cfg.operator_frame.clone(),
                robot_base_frame: cfg.robot_base_frame.clone(),
                extrinsic_version: cfg.extrinsic_version.clone(),
            },
            session_context: self.session_context.clone(),
            artifacts: DemoBundleArtifacts {
                manifest: "manifest.json".to_string(),
                labels: "labels/labels.jsonl".to_string(),
                capture_pose: "raw/iphone/wide/kpts_depth.jsonl".to_string(),
                pose_imu: "raw/iphone/wide/pose_imu.jsonl".to_string(),
                iphone_depth_index: "raw/iphone/wide/depth/index.jsonl".to_string(),
                iphone_media_index: "raw/iphone/wide/media_index.jsonl".to_string(),
                iphone_fisheye_media_index: "raw/iphone/fisheye/media_index.jsonl".to_string(),
                stereo_pose: "raw/stereo/pose3d.jsonl".to_string(),
                wifi_pose: "raw/wifi/pose3d.jsonl".to_string(),
                stereo_media_index: "raw/stereo/media_index.jsonl".to_string(),
                csi_index: "raw/csi/index.jsonl".to_string(),
                csi_chunk_index: "raw/csi/chunks/index.jsonl".to_string(),
                csi_packets: "raw/csi/packets.bin".to_string(),
                fusion_state: "fused/fusion_state.jsonl".to_string(),
                human_demo_pose: "fused/human_demo_pose.jsonl".to_string(),
                teleop_frame: "teleop/teleop_frame.jsonl".to_string(),
                robot_state: "raw/robot/state.jsonl".to_string(),
                chunk_state: "chunks/chunk_state.jsonl".to_string(),
                time_sync_samples: "sync/time_sync_samples.jsonl".to_string(),
                frame_correspondence: "sync/frame_correspondence.jsonl".to_string(),
                offline_manifest: "derived/offline/offline_manifest.json".to_string(),
                vlm_events: "derived/vision/vlm_events.jsonl".to_string(),
                vlm_segments: "derived/vision/vlm_segments.jsonl".to_string(),
                vlm_embeddings: "derived/vision/embeddings".to_string(),
                preview_manifest: "preview/preview_manifest.json".to_string(),
                preview_keyframes: "preview/keyframes".to_string(),
                preview_clips: "preview/clips".to_string(),
                local_quality_report: "qa/local_quality_report.json".to_string(),
                upload_policy: "qa/upload_policy.json".to_string(),
                upload_manifest: "upload/upload_manifest.json".to_string(),
                upload_queue: "upload/upload_queue.json".to_string(),
                upload_receipts: "upload/upload_receipts.jsonl".to_string(),
            },
            media_tracks,
            chunk_dirs: {
                let mut dirs = chunk_dirs;
                dirs.push("raw/csi/chunks".to_string());
                dirs
            },
            calibration_snapshot_paths,
        };
        let value = serde_json::to_value(bundle).map_err(|e| e.to_string())?;
        Self::write_json_pretty(self.base_dir.join("demo_capture_bundle.json"), &value).await
    }

    async fn refresh_session_metadata_artifacts(
        &mut self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<(), String> {
        self.hydrate_session_context_from_disk(cfg).await?;
        self.ensure_semantic_bundle_scaffold(cfg).await?;
        self.refresh_local_quality_report(cfg).await?;
        self.refresh_upload_policy(cfg).await?;
        self.refresh_upload_manifest(cfg).await?;
        upload_queue::refresh_upload_queue(&self.base_dir).await?;
        self.refresh_session_manifest(protocol, cfg).await
    }

    async fn refresh_session_manifest(
        &self,
        protocol: &ProtocolVersionInfo,
        cfg: &Config,
    ) -> Result<(), String> {
        let artifacts = self.collect_session_manifest_artifacts().await?;
        let line_counters = self.disk_line_counters().await?;
        let (has_iphone_calibration, has_stereo_calibration, has_wifi_calibration) =
            self.calibration_flags().await;
        let mut seen_media_tracks = self
            .seen_media_tracks
            .iter()
            .map(|(scope, track)| format!("{scope}.{track}"))
            .collect::<Vec<_>>();
        seen_media_tracks.sort();
        let manifest = SessionManifest {
            ty: "session_manifest",
            schema_version: "2.0.0",
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            created_unix_ms: self.created_unix_ms,
            generated_unix_ms: now_unix_ms(),
            protocol: protocol.clone(),
            frames: DemoBundleFrames {
                operator_frame: cfg.operator_frame.clone(),
                robot_base_frame: cfg.robot_base_frame.clone(),
                extrinsic_version: cfg.extrinsic_version.clone(),
            },
            session_context: self.session_context.clone(),
            artifacts,
            recorder_state: SessionManifestRecorderState {
                has_iphone_calibration,
                has_stereo_calibration,
                has_wifi_calibration,
                seen_media_tracks,
                line_counters,
            },
        };
        let value = serde_json::to_value(manifest).map_err(|e| e.to_string())?;
        Self::write_json_pretty(self.base_dir.join("manifest.json"), &value).await
    }

    async fn refresh_local_quality_report(&self, cfg: &Config) -> Result<(), String> {
        let checks = self.build_local_quality_checks(cfg).await?;
        let total = checks.len().max(1) as f32;
        let passed = checks.iter().filter(|check| check.ok).count() as f32;
        let score_percent = ((passed / total) * 100.0 * 100.0).round() / 100.0;

        let missing_artifacts = checks
            .iter()
            .filter(|check| !check.ok)
            .map(|check| check.id.to_string())
            .collect::<Vec<_>>();
        let recommended_actions = missing_artifacts
            .iter()
            .map(|missing| match missing.as_str() {
                "capture_pose_present" => "补录手机主链 capture pose 与深度事实。".to_string(),
                "iphone_calibration_present" => {
                    "确保手机主链 calibration snapshot 已落盘。".to_string()
                }
                "time_sync_present" => {
                    "补采 time/sync 样本，避免多模态时间轴不可对齐。".to_string()
                }
                "human_demo_pose_present" => {
                    "检查 fused/human_demo_pose 是否稳定输出。".to_string()
                }
                "teleop_frame_present" => {
                    "检查 teleop frame 输出，确保机器人 target 已落盘。".to_string()
                }
                "stereo_pose_present" => {
                    "建议补齐双目 pose，提高 whole-body 几何质量。".to_string()
                }
                "wifi_or_csi_present" => "建议补齐 Wi-Fi pose/CSI，保留多源观测。".to_string(),
                "fisheye_track_present" => {
                    "建议开启超广角连续辅路；若机型只支持快照辅路，至少提高抽帧频率并确认正式 fisheye 轨已落盘。".to_string()
                }
                "media_tracks_present" => "建议补录至少一条可回放媒体轨。".to_string(),
                _ => format!("检查 `{missing}` 对应的采集链路。"),
            })
            .collect::<Vec<_>>();

        let core_missing = [
            "capture_pose_present",
            "iphone_calibration_present",
            "time_sync_present",
            "human_demo_pose_present",
            "teleop_frame_present",
        ]
        .iter()
        .any(|id| missing_artifacts.iter().any(|missing| missing == id));
        let optional_missing = missing_artifacts.iter().any(|id| {
            ![
                "capture_pose_present",
                "iphone_calibration_present",
                "time_sync_present",
                "human_demo_pose_present",
                "teleop_frame_present",
            ]
            .contains(&id.as_str())
        });
        let status = if core_missing {
            "reject_local"
        } else if optional_missing {
            "retry_recommended"
        } else {
            "pass"
        };

        let report = LocalQualityReport {
            ty: "local_quality_report",
            schema_version: "1.0.0",
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            generated_unix_ms: now_unix_ms(),
            status,
            ready_for_upload: status != "reject_local",
            score_percent,
            checks,
            missing_artifacts,
            recommended_actions,
        };
        let value = serde_json::to_value(report).map_err(|e| e.to_string())?;
        Self::write_json_pretty(
            self.base_dir.join("qa").join("local_quality_report.json"),
            &value,
        )
        .await
    }

    async fn refresh_upload_manifest(&self, cfg: &Config) -> Result<(), String> {
        let base_dir = self.base_dir.as_path();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        let quality_path = base_dir.join("qa").join("local_quality_report.json");
        let quality_status = match tokio::fs::read_to_string(&quality_path).await {
            Ok(content) => serde_json::from_str::<serde_json::Value>(&content)
                .ok()
                .and_then(|value| {
                    value
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "pending".to_string()),
            Err(_) => "pending".to_string(),
        };

        let mut artifacts = self.collect_upload_artifacts(cfg).await?;
        artifacts.sort_by(|a, b| a.relpath.cmp(&b.relpath));
        let ready_artifact_count = artifacts.iter().filter(|artifact| artifact.exists).count();
        let manifest = UploadManifest {
            ty: "upload_manifest",
            schema_version: "1.0.0",
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            generated_unix_ms: now_unix_ms(),
            upload_policy: self.build_upload_policy(cfg),
            session_context: self.session_context.clone(),
            ready_for_upload: matches!(quality_status.as_str(), "pass" | "retry_recommended"),
            artifact_count: artifacts.len(),
            ready_artifact_count,
            artifacts,
        };
        let value = serde_json::to_value(manifest).map_err(|e| e.to_string())?;
        Self::write_json_pretty(
            base_dir.join("upload").join("upload_manifest.json"),
            &value,
        )
        .await
    }

    async fn refresh_upload_policy(&self, cfg: &Config) -> Result<(), String> {
        let base_dir = self.base_dir.as_path();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        let snapshot = UploadPolicySnapshot {
            ty: "upload_policy",
            schema_version: "1.0.0",
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            generated_unix_ms: now_unix_ms(),
            upload_policy: self.build_upload_policy(cfg),
            session_context: self.session_context.clone(),
            privacy_tier_defaults: vec![
                "mandatory_structured",
                "consented_raw",
                "restricted",
                "derived_semantic",
                "preview_derivative",
            ],
        };
        let value = serde_json::to_value(snapshot).map_err(|e| e.to_string())?;
        Self::write_json_pretty(base_dir.join("qa").join("upload_policy.json"), &value).await
    }

    async fn hydrate_session_context_from_disk(&mut self, cfg: &Config) -> Result<(), String> {
        let persisted = load_persisted_session_context(&self.base_dir).await?;
        self.session_context.merge_missing_from(&persisted);
        self.session_context.ensure_runtime_defaults(cfg);
        Ok(())
    }

    async fn build_local_quality_checks(
        &self,
        cfg: &Config,
    ) -> Result<Vec<LocalQualityCheck>, String> {
        let base_dir = self.base_dir.clone();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        let session_root = PathBuf::from(&cfg.data_dir).join("session");
        let canonical_session_root = tokio::fs::canonicalize(&session_root)
            .await
            .map_err(|e| format!("解析 session 根目录失败: {} ({e})", session_root.display()))?;
        let canonical_base_dir = tokio::fs::canonicalize(&base_dir)
            .await
            .map_err(|e| format!("解析 session 目录失败: {} ({e})", base_dir.display()))?;
        if !canonical_base_dir.starts_with(&canonical_session_root) {
            return Err(format!(
                "非法 session 目录: {} 不在 {} 下",
                canonical_base_dir.display(),
                canonical_session_root.display()
            ));
        }
        let line_counters = self.disk_line_counters().await?;
        let media_tracks_present = !self.collect_media_tracks().await?.is_empty();
        let csi_packets_path = canonical_base_dir.join("raw").join("csi").join("packets.bin");
        let csi_packets_bytes = tokio::fs::metadata(&csi_packets_path)
            .await
            .map(|meta| meta.len())
            .unwrap_or(0);
        let (has_iphone_calibration, _, _) = self.calibration_flags().await;
        let csi_index_summary =
            summarize_csi_index(canonical_base_dir.join("raw").join("csi").join("index.jsonl"))
                .await?;
        let fisheye_summary = summarize_media_index_frames(
            canonical_base_dir
                .join("raw")
                .join("iphone")
                .join("fisheye")
                .join("media_index.jsonl"),
        )
        .await?;
        let wifi_or_csi_present = line_counters.raw_wifi > 0
            || (line_counters.raw_csi > 0
                && csi_packets_bytes > 0
                && csi_index_summary.max_node_count > 0);
        let fisheye_track_present = fisheye_summary.frame_count >= 10;
        Ok(vec![
            LocalQualityCheck {
                id: "capture_pose_present",
                ok: !cfg.phone_ingest_enabled || line_counters.raw_iphone > 0,
                score: if !cfg.phone_ingest_enabled || line_counters.raw_iphone > 0 {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "raw/iphone/wide/kpts_depth.jsonl 行数={}{}",
                    line_counters.raw_iphone,
                    if cfg.phone_ingest_enabled {
                        ""
                    } else {
                        "（phone ingest disabled by runtime profile）"
                    }
                ),
            },
            LocalQualityCheck {
                id: "pose_imu_present",
                ok: !cfg.phone_ingest_enabled || line_counters.raw_iphone_pose_imu > 0,
                score: if !cfg.phone_ingest_enabled || line_counters.raw_iphone_pose_imu > 0 {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "raw/iphone/wide/pose_imu.jsonl 行数={}",
                    line_counters.raw_iphone_pose_imu
                ),
            },
            LocalQualityCheck {
                id: "raw_depth_present",
                ok: !cfg.phone_ingest_enabled || line_counters.raw_iphone_depth > 0,
                score: if !cfg.phone_ingest_enabled || line_counters.raw_iphone_depth > 0 {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "raw/iphone/wide/depth/index.jsonl 行数={}",
                    line_counters.raw_iphone_depth
                ),
            },
            LocalQualityCheck {
                id: "iphone_calibration_present",
                ok: !cfg.phone_ingest_enabled || has_iphone_calibration,
                score: if !cfg.phone_ingest_enabled || has_iphone_calibration {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "calibration/iphone_capture.json {}{}",
                    if has_iphone_calibration {
                        "已生成"
                    } else {
                        "缺失"
                    },
                    if cfg.phone_ingest_enabled {
                        ""
                    } else {
                        "（phone ingest disabled by runtime profile）"
                    }
                ),
            },
            LocalQualityCheck {
                id: "time_sync_present",
                ok: !(cfg.fusion_enabled || cfg.stereo_enabled || cfg.wifi_enabled || cfg.control_enabled)
                    || line_counters.sync > 0,
                score: if !(cfg.fusion_enabled
                    || cfg.stereo_enabled
                    || cfg.wifi_enabled
                    || cfg.control_enabled)
                    || line_counters.sync > 0
                {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "sync/time_sync_samples.jsonl 行数={}{}",
                    line_counters.sync,
                    if cfg.fusion_enabled || cfg.stereo_enabled || cfg.wifi_enabled || cfg.control_enabled {
                        ""
                    } else {
                        "（time sync not required in raw_capture_only profile）"
                    }
                ),
            },
            LocalQualityCheck {
                id: "human_demo_pose_present",
                ok: !cfg.fusion_enabled || line_counters.human_demo_pose > 0,
                score: if !cfg.fusion_enabled || line_counters.human_demo_pose > 0 {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "fused/human_demo_pose.jsonl 行数={}{}",
                    line_counters.human_demo_pose,
                    if cfg.fusion_enabled {
                        ""
                    } else {
                        "（fusion disabled by runtime profile）"
                    }
                ),
            },
            LocalQualityCheck {
                id: "teleop_frame_present",
                ok: !(cfg.fusion_enabled || cfg.control_enabled) || line_counters.teleop > 0,
                score: if !(cfg.fusion_enabled || cfg.control_enabled) || line_counters.teleop > 0 {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "teleop/teleop_frame.jsonl 行数={}{}",
                    line_counters.teleop,
                    if cfg.fusion_enabled || cfg.control_enabled {
                        ""
                    } else {
                        "（teleop publisher disabled by runtime profile）"
                    }
                ),
            },
            LocalQualityCheck {
                id: "robot_state_present",
                ok: line_counters.raw_robot_state > 0,
                score: if line_counters.raw_robot_state > 0 {
                    1.0
                } else {
                    0.0
                },
                detail: format!("raw/robot/state.jsonl 行数={}", line_counters.raw_robot_state),
            },
            LocalQualityCheck {
                id: "stereo_pose_present",
                ok: !cfg.stereo_enabled || line_counters.raw_stereo > 0,
                score: if !cfg.stereo_enabled || line_counters.raw_stereo > 0 {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "raw/stereo/pose3d.jsonl 行数={}{}",
                    line_counters.raw_stereo,
                    if cfg.stereo_enabled {
                        ""
                    } else {
                        "（stereo disabled by runtime profile）"
                    }
                ),
            },
            LocalQualityCheck {
                id: "wifi_or_csi_present",
                ok: !cfg.wifi_enabled || wifi_or_csi_present,
                score: if !cfg.wifi_enabled || wifi_or_csi_present {
                    1.0
                } else {
                    0.0
                },
                detail: format!(
                    "raw/wifi/pose3d 行数={}，raw/csi/index 行数={}，raw/csi/packets.bin bytes={}，CSI 有效节点行数={}，最大 node_count={}{}",
                    line_counters.raw_wifi,
                    line_counters.raw_csi,
                    csi_packets_bytes,
                    csi_index_summary.rows_with_nodes,
                    csi_index_summary.max_node_count,
                    if cfg.wifi_enabled {
                        ""
                    } else {
                        "（wifi disabled by runtime profile）"
                    }
                ),
            },
            LocalQualityCheck {
                id: "fisheye_track_present",
                ok: fisheye_track_present,
                score: if fisheye_track_present { 1.0 } else { 0.0 },
                detail: format!(
                    "raw/iphone/fisheye/media_index 行数={}，累计 frame_count={}",
                    fisheye_summary.record_count, fisheye_summary.frame_count
                ),
            },
            LocalQualityCheck {
                id: "media_tracks_present",
                ok: media_tracks_present,
                score: if media_tracks_present { 1.0 } else { 0.0 },
                detail: if media_tracks_present {
                    "至少一条媒体轨可回放".to_string()
                } else {
                    "当前没有可回放媒体轨".to_string()
                },
            },
        ])
    }

    async fn collect_session_manifest_artifacts(
        &self,
    ) -> Result<Vec<SessionManifestArtifact>, String> {
        let mut artifacts = Vec::new();
        for spec in self.base_artifact_specs().await? {
            artifacts.push(SessionManifestArtifact {
                id: spec.id,
                relpath: spec.relpath,
                kind: spec.kind,
                required: spec.required,
                exists: spec.exists,
                byte_size: spec.byte_size,
                line_count: spec.line_count,
                privacy_tier: spec.privacy_tier,
            });
        }
        Ok(artifacts)
    }

    async fn collect_upload_artifacts(
        &self,
        cfg: &Config,
    ) -> Result<Vec<UploadManifestArtifact>, String> {
        let mut artifacts = Vec::new();
        for spec in self
            .base_artifact_specs()
            .await?
            .into_iter()
            .filter(|spec| {
                !["upload_manifest", "upload_queue", "upload_receipts"].contains(&spec.id.as_str())
                    && self.artifact_allowed_for_upload(cfg, spec.privacy_tier)
            })
        {
            artifacts.push(UploadManifestArtifact {
                id: spec.id,
                relpath: spec.relpath,
                kind: spec.kind,
                category: spec.privacy_tier,
                required: spec.required,
                exists: spec.exists,
                byte_size: spec.byte_size,
                line_count: spec.line_count,
                residency: self.artifact_residency(cfg, spec.privacy_tier).to_string(),
                upload_state: if spec.exists { "ready" } else { "pending" },
            });
        }
        Ok(artifacts)
    }

    fn build_upload_policy(&self, cfg: &Config) -> UploadPolicy {
        UploadPolicy {
            mode: "edge_crowd_upload_v1",
            artifact_policy_mode: cfg.upload_policy_mode_name().to_string(),
            runtime_profile: cfg.runtime_profile_name().to_string(),
            raw_residency_default: cfg.raw_residency_default().to_string(),
            preview_residency_default: cfg.preview_residency_default().to_string(),
            transport: "artifact_chunked_upload",
            required_quality_status: "pass_or_retry_recommended",
        }
    }

    fn artifact_allowed_for_upload(&self, cfg: &Config, privacy_tier: &str) -> bool {
        match cfg.crowd_upload_policy_mode {
            crate::config::CrowdUploadPolicyMode::MetadataOnly => {
                matches!(privacy_tier, "mandatory_structured" | "derived_semantic")
            }
            crate::config::CrowdUploadPolicyMode::MetadataPlusPreview => matches!(
                privacy_tier,
                "mandatory_structured" | "derived_semantic" | "preview_derivative"
            ),
            crate::config::CrowdUploadPolicyMode::FullRawMirror => true,
        }
    }

    fn artifact_residency(&self, cfg: &Config, privacy_tier: &str) -> &'static str {
        match cfg.crowd_upload_policy_mode {
            crate::config::CrowdUploadPolicyMode::MetadataOnly => match privacy_tier {
                "mandatory_structured" | "derived_semantic" => "cloud_mirrored",
                "preview_derivative" => "edge_only",
                _ => "edge_only",
            },
            crate::config::CrowdUploadPolicyMode::MetadataPlusPreview => match privacy_tier {
                "preview_derivative" => "cloud_preview_only",
                "mandatory_structured" | "derived_semantic" => "cloud_mirrored",
                _ => "edge_only",
            },
            crate::config::CrowdUploadPolicyMode::FullRawMirror => "cloud_mirrored",
        }
    }

    async fn base_artifact_specs(&self) -> Result<Vec<ArtifactSpecRecord>, String> {
        let line_counters = self.disk_line_counters().await?;
        let media_tracks = self.collect_media_tracks().await?;
        let mut specs = vec![
            self.artifact_spec(
                "manifest",
                "manifest.json",
                "file",
                true,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "demo_capture_bundle",
                "demo_capture_bundle.json",
                "file",
                true,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "labels",
                "labels/labels.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.labels),
            )
            .await?,
            self.artifact_spec(
                "capture_pose",
                "raw/iphone/wide/kpts_depth.jsonl",
                "file",
                true,
                "mandatory_structured",
                Some(line_counters.raw_iphone),
            )
            .await?,
            self.artifact_spec(
                "pose_imu",
                "raw/iphone/wide/pose_imu.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_iphone_pose_imu),
            )
            .await?,
            self.artifact_spec(
                "iphone_depth_index",
                "raw/iphone/wide/depth/index.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_iphone_depth),
            )
            .await?,
            self.artifact_spec(
                "iphone_depth_frames",
                "raw/iphone/wide/depth",
                "directory",
                false,
                "consented_raw",
                None,
            )
            .await?,
            self.artifact_spec(
                "iphone_media_index",
                "raw/iphone/wide/media_index.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_iphone_media),
            )
            .await?,
            self.artifact_spec(
                "iphone_fisheye_media_index",
                "raw/iphone/fisheye/media_index.jsonl",
                "file",
                false,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "iphone_fisheye_chunks",
                "raw/iphone/fisheye/chunks",
                "directory",
                false,
                "consented_raw",
                None,
            )
            .await?,
            self.artifact_spec(
                "stereo_pose",
                "raw/stereo/pose3d.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_stereo),
            )
            .await?,
            self.artifact_spec(
                "wifi_pose",
                "raw/wifi/pose3d.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_wifi),
            )
            .await?,
            self.artifact_spec(
                "stereo_media_index",
                "raw/stereo/media_index.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_stereo_media),
            )
            .await?,
            self.artifact_spec(
                "csi_index",
                "raw/csi/index.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_csi),
            )
            .await?,
            self.artifact_spec(
                "csi_chunk_index",
                "raw/csi/chunks/index.jsonl",
                "file",
                false,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "csi_packets",
                "raw/csi/packets.bin",
                "file",
                false,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "fusion_state",
                "fused/fusion_state.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.fused_state),
            )
            .await?,
            self.artifact_spec(
                "human_demo_pose",
                "fused/human_demo_pose.jsonl",
                "file",
                true,
                "mandatory_structured",
                Some(line_counters.human_demo_pose),
            )
            .await?,
            self.artifact_spec(
                "teleop_frame",
                "teleop/teleop_frame.jsonl",
                "file",
                true,
                "mandatory_structured",
                Some(line_counters.teleop),
            )
            .await?,
            self.artifact_spec(
                "robot_state",
                "raw/robot/state.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.raw_robot_state),
            )
            .await?,
            self.artifact_spec(
                "chunk_state",
                "chunks/chunk_state.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.chunks),
            )
            .await?,
            self.artifact_spec(
                "time_sync_samples",
                "sync/time_sync_samples.jsonl",
                "file",
                true,
                "mandatory_structured",
                Some(line_counters.sync),
            )
            .await?,
            self.artifact_spec(
                "frame_correspondence",
                "sync/frame_correspondence.jsonl",
                "file",
                false,
                "mandatory_structured",
                Some(line_counters.frame_correspondence),
            )
            .await?,
            self.artifact_spec(
                "offline_manifest",
                "derived/offline/offline_manifest.json",
                "file",
                false,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "vlm_events",
                "derived/vision/vlm_events.jsonl",
                "file",
                false,
                "derived_semantic",
                None,
            )
            .await?,
            self.artifact_spec(
                "vlm_segments",
                "derived/vision/vlm_segments.jsonl",
                "file",
                false,
                "derived_semantic",
                None,
            )
            .await?,
            self.artifact_spec(
                "vlm_embeddings",
                "derived/vision/embeddings",
                "directory",
                false,
                "derived_semantic",
                None,
            )
            .await?,
            self.artifact_spec(
                "preview_manifest",
                "preview/preview_manifest.json",
                "file",
                false,
                "preview_derivative",
                None,
            )
            .await?,
            self.artifact_spec(
                "preview_keyframes",
                "preview/keyframes",
                "directory",
                false,
                "preview_derivative",
                None,
            )
            .await?,
            self.artifact_spec(
                "preview_clips",
                "preview/clips",
                "directory",
                false,
                "preview_derivative",
                None,
            )
            .await?,
            self.artifact_spec(
                "local_quality_report",
                "qa/local_quality_report.json",
                "file",
                true,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "upload_policy",
                "qa/upload_policy.json",
                "file",
                true,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "upload_manifest",
                "upload/upload_manifest.json",
                "file",
                true,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "upload_queue",
                "upload/upload_queue.json",
                "file",
                false,
                "restricted",
                None,
            )
            .await?,
            self.artifact_spec(
                "upload_receipts",
                "upload/upload_receipts.jsonl",
                "file",
                false,
                "restricted",
                None,
            )
            .await?,
            self.artifact_spec(
                "edge_frames",
                "calibration/edge_frames.json",
                "file",
                true,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "iphone_calibration",
                "calibration/iphone_capture.json",
                "file",
                self.lines.raw_iphone > 0,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "iphone_fisheye_calibration",
                "calibration/iphone_fisheye.json",
                "file",
                false,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "stereo_calibration",
                "calibration/stereo_pair.json",
                "file",
                false,
                "mandatory_structured",
                None,
            )
            .await?,
            self.artifact_spec(
                "wifi_calibration",
                "calibration/wifi_pose.json",
                "file",
                false,
                "mandatory_structured",
                None,
            )
            .await?,
        ];
        for track in media_tracks {
            specs.push(
                self.artifact_spec(
                    &format!("media_track_index.{}", track.id),
                    &track.media_index,
                    "file",
                    false,
                    "mandatory_structured",
                    None,
                )
                .await?,
            );
            specs.push(
                self.artifact_spec(
                    &format!("media_track_chunks.{}", track.id),
                    &track.chunk_dir,
                    "directory",
                    false,
                    "consented_raw",
                    None,
                )
                .await?,
            );
        }
        Ok(specs)
    }

    async fn artifact_spec(
        &self,
        id: &str,
        relpath: &str,
        kind: &'static str,
        required: bool,
        privacy_tier: &'static str,
        line_count: Option<u64>,
    ) -> Result<ArtifactSpecRecord, String> {
        let base_dir = self.base_dir.as_path();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        let relpath_path = Path::new(relpath);
        let mut safe_relpath = PathBuf::new();
        for component in relpath_path.components() {
            match component {
                Component::Normal(part) => safe_relpath.push(part),
                Component::CurDir
                | Component::ParentDir
                | Component::RootDir
                | Component::Prefix(_) => return Err("artifact relpath 必须是受限的相对路径".to_string()),
            }
        }
        let path = base_dir.join(&safe_relpath);
        let metadata = tokio::fs::metadata(&path).await.ok();
        Ok(ArtifactSpecRecord {
            id: id.to_string(),
            relpath: relpath.to_string(),
            kind,
            required,
            exists: metadata.is_some(),
            byte_size: metadata.map(|meta| meta.len()).unwrap_or(0),
            line_count,
            privacy_tier,
        })
    }

    async fn collect_media_tracks(&self) -> Result<Vec<DemoBundleMediaTrack>, String> {
        let base_dir = self.base_dir.as_path();
        if base_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(format!("非法 session 目录: {}", base_dir.display()));
        }
        let mut tracks = Vec::new();
        for template in MEDIA_TRACK_TEMPLATES {
            let storage_track = normalized_track_storage_name(template.scope, template.track);
            let media_index_relpath =
                format!("raw/{}/{}/media_index.jsonl", template.scope, storage_track);
            let media_index_path = base_dir.join(&media_index_relpath);
            let include_track = self
                .seen_media_tracks
                .contains(&(template.scope.to_string(), template.track.to_string()))
                || tokio::fs::metadata(&media_index_path)
                    .await
                    .map(|meta| meta.len() > 0)
                    .unwrap_or(false);
            if !include_track {
                continue;
            }
            tracks.push(DemoBundleMediaTrack {
                id: template.id.to_string(),
                label: template.label.to_string(),
                source: template.scope.to_string(),
                modality: template.modality.to_string(),
                media_index: media_index_relpath,
                chunk_dir: format!("raw/{}/{}/chunks", template.scope, storage_track),
                default_visible: template.default_visible,
            });
        }
        Ok(tracks)
    }

    fn build_iphone_calibration_snapshot(
        &self,
        v: &serde_json::Value,
    ) -> Option<serde_json::Value> {
        let calibration = v.get("camera")?.get("calibration")?;
        Some(serde_json::json!({
            "type": "sensor_calibration_snapshot",
            "schema_version": "1.0.0",
            "trip_id": self.trip_id.clone(),
            "session_id": self.session_id.clone(),
            "sensor_kind": "iphone_capture",
            "sensor_id": v.get("device_id").and_then(|x| x.as_str()).unwrap_or("iphone_capture"),
            "edge_time_ns": v.get("edge_time_ns").and_then(|x| x.as_u64()).unwrap_or(0),
            "sensor_frame": "iphone_capture_frame",
            "intrinsics": calibration,
            "extrinsic_version": v
                .get("camera")
                .and_then(|x| x.get("extrinsic_version"))
                .and_then(|x| x.as_str())
                .unwrap_or(""),
            "notes": v
                .get("capture_profile")
                .map(|profile| profile.to_string())
                .unwrap_or_default()
        }))
    }

    fn build_stereo_calibration_snapshot(
        &self,
        v: &serde_json::Value,
    ) -> Option<serde_json::Value> {
        let calibration = v.get("calibration")?;
        Some(serde_json::json!({
            "type": "sensor_calibration_snapshot",
            "schema_version": "1.0.0",
            "trip_id": self.trip_id.clone(),
            "session_id": self.session_id.clone(),
            "sensor_kind": "stereo_pair",
            "sensor_id": v.get("device_id").and_then(|x| x.as_str()).unwrap_or("stereo_pair"),
            "edge_time_ns": v.get("edge_time_ns").and_then(|x| x.as_u64()).unwrap_or(0),
            "sensor_frame": calibration.get("sensor_frame").and_then(|x| x.as_str()).unwrap_or("stereo_pair_frame"),
            "operator_frame": calibration.get("operator_frame").and_then(|x| x.as_str()).unwrap_or(""),
            "extrinsic_version": calibration.get("extrinsic_version").and_then(|x| x.as_str()).unwrap_or(""),
            "left_intrinsics": calibration.get("left_intrinsics").cloned().unwrap_or(serde_json::Value::Null),
            "right_intrinsics": calibration.get("right_intrinsics").cloned().unwrap_or(serde_json::Value::Null),
            "notes": "stereo ingest calibration snapshot"
        }))
    }

    fn build_wifi_calibration_snapshot(&self, v: &serde_json::Value) -> Option<serde_json::Value> {
        let calibration = v.get("calibration")?;
        let mut snapshot = serde_json::json!({
            "type": "sensor_calibration_snapshot",
            "schema_version": "1.0.0",
            "trip_id": self.trip_id.clone(),
            "session_id": self.session_id.clone(),
            "sensor_kind": "wifi_pose",
            "sensor_id": v.get("device_id").and_then(|x| x.as_str()).unwrap_or("wifi_pose"),
            "edge_time_ns": v.get("edge_time_ns").and_then(|x| x.as_u64()).unwrap_or(0),
            "sensor_frame": calibration.get("sensor_frame").and_then(|x| x.as_str()).unwrap_or("wifi_pose_frame"),
            "operator_frame": calibration.get("operator_frame").and_then(|x| x.as_str()).unwrap_or(""),
            "extrinsic_version": calibration.get("extrinsic_version").and_then(|x| x.as_str()).unwrap_or(""),
            "notes": calibration.get("notes").and_then(|x| x.as_str()).unwrap_or("wifi pose ingest calibration snapshot"),
        });
        if let Some(translation) = calibration.get("extrinsic_translation_m").cloned() {
            snapshot["extrinsic_translation_m"] = translation;
        }
        if let Some(rotation) = calibration.get("extrinsic_rotation_quat").cloned() {
            snapshot["extrinsic_rotation_quat"] = rotation;
        }
        Some(snapshot)
    }

    fn clip_sample_stats(&self, sample_total: usize) -> ClipSampleStats {
        let n = self.clip_results.len().min(sample_total.max(1));
        if n == 0 {
            return ClipSampleStats::default();
        }
        let recent = self
            .clip_results
            .iter()
            .rev()
            .take(n)
            .copied()
            .collect::<Vec<_>>();
        let pass = recent.iter().filter(|item| item.pass).count() as u64;
        let locatable = recent.iter().filter(|item| item.locatable).count() as u64;
        let playable = recent.iter().filter(|item| item.playable).count() as u64;
        let label_consistent = recent.iter().filter(|item| item.label_consistent).count() as u64;
        let index_complete = recent.iter().filter(|item| item.index_complete).count() as u64;
        ClipSampleStats {
            sample_total: n as u64,
            pass_samples: pass,
            pass_rate_percent: (pass as f64 / n as f64) * 100.0,
            locatable_samples: locatable,
            playable_samples: playable,
            label_consistent_samples: label_consistent,
            index_complete_samples: index_complete,
        }
    }

    fn maybe_update_clip_quality_from_fusion_state(&mut self, v: &serde_json::Value) {
        let Some(active) = self.clip.active.as_mut() else {
            return;
        };
        active.quality.update(v);
    }

    async fn maybe_update_clips(&mut self, v: &serde_json::Value, label_line: u64) {
        let Some(ev) = v.get("event").and_then(|x| x.as_str()) else {
            return;
        };
        let edge_time_ns = v.get("edge_time_ns").and_then(|x| x.as_u64()).unwrap_or(0);
        match ev {
            "scene_switch" => {
                let new_scene = v
                    .get("scene_label")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if new_scene.is_empty() {
                    return;
                }
                // 若动作跨场景：按 scene_switch 切分为多片段（PRD 13.3）。
                if let Some(active) = self.clip.active.clone() {
                    let needs_review = false;
                    // scene_switch 本身更适合作为“新片段”的开始标签，因此旧片段 label_end 取上一行。
                    let end_label_line = label_line.saturating_sub(1);
                    self.emit_clip_manifest(&active, edge_time_ns, end_label_line, needs_review)
                        .await;
                    // 开启同 action_id 的新分段（scene_label 变更）。
                    let part_index = self.next_part_index(&active.action_id);
                    self.clip.active = Some(ActiveAction {
                        action_id: active.action_id.clone(),
                        action_label: active.action_label.clone(),
                        scene_label: new_scene.clone(),
                        part_index,
                        start_edge_time_ns: edge_time_ns,
                        indices: self.make_clip_indices(label_line),
                        quality: ClipQualityAgg::default(),
                    });
                }
                self.clip.current_scene = new_scene;
            }
            "action_start" => {
                let action_id = v
                    .get("action_id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if action_id.is_empty() {
                    return;
                }
                let action_label = v
                    .get("action_label")
                    .and_then(|x| x.as_str())
                    .unwrap_or("unknown")
                    .trim()
                    .to_string();
                let scene_label = v
                    .get("scene_label")
                    .and_then(|x| x.as_str())
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| {
                        if self.clip.current_scene.trim().is_empty() {
                            "unknown".to_string()
                        } else {
                            self.clip.current_scene.clone()
                        }
                    });

                // 防御：如果上一段动作未结束，先落盘并标记 needs_review。
                if let Some(prev) = self.clip.active.clone() {
                    let end_label_line = label_line.saturating_sub(1);
                    self.emit_clip_manifest(&prev, edge_time_ns, end_label_line, true)
                        .await;
                }

                let part_index = self.next_part_index(&action_id);
                self.clip.active = Some(ActiveAction {
                    action_id,
                    action_label,
                    scene_label,
                    part_index,
                    start_edge_time_ns: edge_time_ns,
                    indices: self.make_clip_indices(label_line),
                    quality: ClipQualityAgg::default(),
                });
            }
            "action_end" => {
                let action_id = v
                    .get("action_id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let Some(active) = self.clip.active.take() else {
                    // 没有 start 就来了 end：标记需要人工复核。
                    if !action_id.is_empty() {
                        let part_index = self.next_part_index(&action_id);
                        let a = ActiveAction {
                            action_id,
                            action_label: "unknown".to_string(),
                            scene_label: if self.clip.current_scene.trim().is_empty() {
                                "unknown".to_string()
                            } else {
                                self.clip.current_scene.clone()
                            },
                            part_index,
                            start_edge_time_ns: edge_time_ns,
                            indices: self.make_clip_indices(label_line),
                            quality: ClipQualityAgg::default(),
                        };
                        self.emit_clip_manifest(&a, edge_time_ns, label_line, true)
                            .await;
                    }
                    return;
                };
                let needs_review = !action_id.is_empty() && action_id != active.action_id;
                self.emit_clip_manifest(&active, edge_time_ns, label_line, needs_review)
                    .await;
            }
            _ => {}
        }
    }

    fn make_clip_indices(&self, labels_start_line: u64) -> ClipIndices {
        ClipIndices {
            labels_start_line,
            raw_iphone_start_line: self.lines.raw_iphone.saturating_add(1),
            raw_iphone_media_start_line: self.lines.raw_iphone_media.saturating_add(1),
            raw_csi_start_line: self.lines.raw_csi.saturating_add(1),
            raw_stereo_start_line: self.lines.raw_stereo.saturating_add(1),
            raw_wifi_start_line: self.lines.raw_wifi.saturating_add(1),
            raw_stereo_media_start_line: self.lines.raw_stereo_media.saturating_add(1),
            fused_start_line: self.lines.fused_state.saturating_add(1),
            human_demo_pose_start_line: self.lines.human_demo_pose.saturating_add(1),
            teleop_start_line: self.lines.teleop.saturating_add(1),
            chunks_start_line: self.lines.chunks.saturating_add(1),
        }
    }

    fn next_part_index(&mut self, action_id: &str) -> u32 {
        let v = self
            .clip
            .part_counter_by_action
            .entry(action_id.to_string())
            .or_insert(0);
        *v += 1;
        *v
    }

    async fn emit_clip_manifest(
        &mut self,
        active: &ActiveAction,
        end_edge_time_ns: u64,
        label_end_line: u64,
        needs_review: bool,
    ) {
        let dir_name = format!(
            "{}_{}_{}__part{}",
            sanitize_component(&active.action_id),
            sanitize_component(&active.action_label),
            sanitize_component(&active.scene_label),
            active.part_index
        );
        let clip_dir = self.base_dir.join("clips").join(dir_name);
        if let Err(e) = tokio::fs::create_dir_all(&clip_dir).await {
            warn!(error=%e, path=%clip_dir.display(), "create clip dir failed");
            return;
        }

        let labels_range = range_opt(active.indices.labels_start_line, label_end_line);
        let raw_iphone = range_opt(active.indices.raw_iphone_start_line, self.lines.raw_iphone);
        let raw_iphone_media = range_opt(
            active.indices.raw_iphone_media_start_line,
            self.lines.raw_iphone_media,
        );
        let raw_csi = range_opt(active.indices.raw_csi_start_line, self.lines.raw_csi);
        let raw_stereo = range_opt(active.indices.raw_stereo_start_line, self.lines.raw_stereo);
        let raw_wifi = range_opt(active.indices.raw_wifi_start_line, self.lines.raw_wifi);
        let raw_stereo_media = range_opt(
            active.indices.raw_stereo_media_start_line,
            self.lines.raw_stereo_media,
        );
        let fused = range_opt(active.indices.fused_start_line, self.lines.fused_state);
        let human_demo_pose = range_opt(
            active.indices.human_demo_pose_start_line,
            self.lines.human_demo_pose,
        );
        let teleop = range_opt(active.indices.teleop_start_line, self.lines.teleop);
        let chunks = range_opt(active.indices.chunks_start_line, self.lines.chunks);

        let quality = if active.quality.n == 0 {
            ClipQualitySummary {
                sample_count: 0,
                vision_conf_mean: 0.0,
                csi_conf_mean: 0.0,
                fused_conf_mean: 0.0,
                fused_conf_min: 0.0,
                coherence_mean: 0.0,
                gate_accept: 0,
                gate_limit: 0,
                gate_freeze: 0,
                gate_estop: 0,
            }
        } else {
            let denom = active.quality.n as f64;
            ClipQualitySummary {
                sample_count: active.quality.n,
                vision_conf_mean: (active.quality.sum_vision / denom) as f32,
                csi_conf_mean: (active.quality.sum_csi / denom) as f32,
                fused_conf_mean: (active.quality.sum_fused / denom) as f32,
                fused_conf_min: active.quality.min_fused,
                coherence_mean: (active.quality.sum_coherence / denom) as f32,
                gate_accept: active.quality.accept,
                gate_limit: active.quality.limit,
                gate_freeze: active.quality.freeze,
                gate_estop: active.quality.estop,
            }
        };

        let mut needs_review = needs_review;
        // PRD 13.4：切片内至少存在一条融合输出（fusion_state）。
        if fused.is_none() {
            needs_review = true;
        }

        let index_ranges = ClipModalityRanges {
            labels: labels_range,
            raw_iphone,
            raw_iphone_media,
            raw_csi,
            raw_stereo_pose3d: raw_stereo,
            raw_wifi_pose3d: raw_wifi,
            raw_stereo_media,
            fused,
            human_demo_pose,
            teleop,
            chunks,
        };
        let validation = self
            .validate_clip_manifest(active, needs_review, &index_ranges)
            .await;

        if self.clip_results.len() >= 256 {
            self.clip_results.pop_front();
        }
        self.clip_results.push_back(validation);

        let manifest = ClipManifest {
            schema_version: "1.0.0",
            trip_id: self.trip_id.clone(),
            session_id: self.session_id.clone(),
            action_id: active.action_id.clone(),
            action_label: active.action_label.clone(),
            scene_label: active.scene_label.clone(),
            part_index: active.part_index,
            start_edge_time_ns: active.start_edge_time_ns,
            end_edge_time_ns,
            needs_review,
            index_ranges,
            quality_summary: quality,
            validation,
        };
        let path = clip_dir.join("clip_manifest.json");
        let bytes = match serde_json::to_vec_pretty(&manifest) {
            Ok(v) => v,
            Err(e) => {
                warn!(error=%e, "serialize clip_manifest failed");
                return;
            }
        };
        if let Err(e) = tokio::fs::write(&path, bytes).await {
            warn!(error=%e, path=%path.display(), "write clip_manifest failed");
        }
    }

    async fn validate_clip_manifest(
        &self,
        active: &ActiveAction,
        needs_review: bool,
        index_ranges: &ClipModalityRanges,
    ) -> ClipValidationSummary {
        let locatable = self.clip_is_locatable(index_ranges);
        let index_complete = self.clip_indexes_complete(index_ranges);
        let label_consistent = match index_ranges.labels {
            Some(range) => self.clip_labels_match(active, range).await,
            None => false,
        };
        let playable = if locatable {
            self.clip_is_playable(index_ranges).await
        } else {
            false
        };
        ClipValidationSummary {
            locatable,
            playable,
            label_consistent,
            index_complete,
            pass: !needs_review && locatable && playable && label_consistent && index_complete,
        }
    }

    fn clip_is_locatable(&self, index_ranges: &ClipModalityRanges) -> bool {
        self.range_within(index_ranges.labels, self.lines.labels)
            && self.range_within(index_ranges.raw_iphone, self.lines.raw_iphone)
            && self.range_within(index_ranges.raw_iphone_media, self.lines.raw_iphone_media)
            && self.range_within(index_ranges.raw_csi, self.lines.raw_csi)
            && self.range_within(index_ranges.raw_stereo_pose3d, self.lines.raw_stereo)
            && self.range_within(index_ranges.raw_wifi_pose3d, self.lines.raw_wifi)
            && self.range_within(index_ranges.raw_stereo_media, self.lines.raw_stereo_media)
            && self.range_within(index_ranges.fused, self.lines.fused_state)
            && self.range_within(index_ranges.human_demo_pose, self.lines.human_demo_pose)
            && self.range_within(index_ranges.teleop, self.lines.teleop)
            && self.range_within(index_ranges.chunks, self.lines.chunks)
    }

    fn clip_indexes_complete(&self, index_ranges: &ClipModalityRanges) -> bool {
        let has_replay_core = index_ranges.fused.is_some()
            || index_ranges.human_demo_pose.is_some()
            || index_ranges.teleop.is_some();
        index_ranges.labels.is_some() && has_replay_core
    }

    async fn clip_is_playable(&self, index_ranges: &ClipModalityRanges) -> bool {
        let has_replay_core = index_ranges.fused.is_some()
            || index_ranges.human_demo_pose.is_some()
            || index_ranges.teleop.is_some();
        if !has_replay_core {
            return false;
        }
        if let Some(range) = index_ranges.raw_iphone_media {
            if !self
                .media_files_exist_any(
                    &[
                        self.base_dir
                            .join("raw")
                            .join("iphone")
                            .join("wide")
                            .join("media_index.jsonl"),
                        self.base_dir
                            .join("raw")
                            .join("iphone")
                            .join("media_index.jsonl"),
                    ],
                    range,
                )
                .await
            {
                return false;
            }
        }
        if let Some(range) = index_ranges.raw_stereo_media {
            if !self
                .media_files_exist_any(
                    &[
                        self.base_dir
                            .join("raw")
                            .join("stereo")
                            .join("preview")
                            .join("media_index.jsonl"),
                        self.base_dir
                            .join("raw")
                            .join("stereo")
                            .join("media_index.jsonl"),
                    ],
                    range,
                )
                .await
            {
                return false;
            }
        }
        true
    }

    async fn clip_labels_match(&self, active: &ActiveAction, range: IndexRange) -> bool {
        let path = self.base_dir.join("labels").join("labels.jsonl");
        let Ok(events) = read_jsonl_range(&path, range).await else {
            return false;
        };
        if events.is_empty() {
            return false;
        }

        let mut has_action_event = false;
        for event in events {
            if let Some(action_id) = event.get("action_id").and_then(|value| value.as_str()) {
                if !action_id.trim().is_empty() && action_id != active.action_id {
                    return false;
                }
            }
            if let Some(action_label) = event.get("action_label").and_then(|value| value.as_str()) {
                if !action_label.trim().is_empty() && action_label != active.action_label {
                    return false;
                }
            }
            if let Some(scene_label) = event.get("scene_label").and_then(|value| value.as_str()) {
                if !scene_label.trim().is_empty() && scene_label != active.scene_label {
                    return false;
                }
            }
            if matches!(
                event.get("event").and_then(|value| value.as_str()),
                Some("action_start" | "action_end")
            ) {
                has_action_event = true;
            }
        }
        has_action_event
    }

    async fn media_files_exist(&self, media_index_path: PathBuf, range: IndexRange) -> bool {
        let Ok(entries) = read_jsonl_range(&media_index_path, range).await else {
            return false;
        };
        if entries.is_empty() {
            return false;
        }
        for entry in entries {
            let Some(relpath) = entry.get("file_relpath").and_then(|value| value.as_str()) else {
                return false;
            };
            if relpath.trim().is_empty() {
                return false;
            }
            if tokio::fs::metadata(self.base_dir.join(relpath))
                .await
                .is_err()
            {
                return false;
            }
        }
        true
    }

    async fn media_files_exist_any(
        &self,
        media_index_paths: &[PathBuf],
        range: IndexRange,
    ) -> bool {
        for media_index_path in media_index_paths {
            if self
                .media_files_exist(media_index_path.clone(), range)
                .await
            {
                return true;
            }
        }
        false
    }

    fn range_within(&self, range: Option<IndexRange>, total_lines: u64) -> bool {
        match range {
            Some(range) => {
                range.start_line > 0
                    && range.end_line >= range.start_line
                    && range.end_line <= total_lines
            }
            None => true,
        }
    }
}

fn array_triplet_f32(value: Option<&serde_json::Value>) -> Option<[f32; 3]> {
    let array = value?.as_array()?;
    if array.len() != 3 {
        return None;
    }
    Some([
        array.first()?.as_f64()? as f32,
        array.get(1)?.as_f64()? as f32,
        array.get(2)?.as_f64()? as f32,
    ])
}

fn vector_norm(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn semantic_action_guess(v: &serde_json::Value) -> String {
    let accel = array_triplet_f32(v.get("imu").and_then(|value| value.get("accel")));
    let gyro = array_triplet_f32(v.get("imu").and_then(|value| value.get("gyro")));
    let accel_norm = accel.map(vector_norm).unwrap_or(0.0);
    let gyro_norm = gyro.map(vector_norm).unwrap_or(0.0);
    if accel_norm > 1.8 || gyro_norm > 1.2 {
        "moving_phone".to_string()
    } else if accel_norm > 0.6 || gyro_norm > 0.4 {
        "reframing_scene".to_string()
    } else {
        "steady_capture".to_string()
    }
}

fn semantic_tags(v: &serde_json::Value, camera_mode: &str, action_guess: &str) -> Vec<String> {
    let mut tags = vec![
        "phone_main_view".to_string(),
        sanitize_component(camera_mode),
        sanitize_component(action_guess),
    ];
    if v.get("camera_has_depth").and_then(|value| value.as_bool()) == Some(true) {
        tags.push("rgbd".to_string());
    } else {
        tags.push("rgb_only".to_string());
    }
    if v.get("operator_track_id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
    {
        tags.push("operator_tracked".to_string());
    }
    unique_strings(&tags)
}

fn semantic_objects(v: &serde_json::Value) -> Vec<String> {
    let mut objects = vec!["phone_view".to_string()];
    if v.get("camera_has_depth").and_then(|value| value.as_bool()) == Some(true) {
        objects.push("depth_map".to_string());
    }
    if v.get("device_pose").is_some() {
        objects.push("device_pose".to_string());
    }
    if v.get("operator_track_id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
    {
        objects.push("wearer".to_string());
    }
    unique_strings(&objects)
}

fn build_event_caption(camera_mode: &str, v: &serde_json::Value, action_guess: &str) -> String {
    let has_depth = v.get("camera_has_depth").and_then(|value| value.as_bool()) == Some(true);
    let modality = if has_depth { "RGB-D" } else { "RGB" };
    format!(
        "Phone {} {} keyframe while the operator is {}.",
        camera_mode.trim(),
        modality,
        action_guess.replace('_', " ")
    )
}

fn build_segment_caption(
    camera_mode: &str,
    keyframe_count: usize,
    start_edge_time_ns: u64,
    end_edge_time_ns: u64,
    action_guess: &str,
) -> String {
    let duration_s = end_edge_time_ns.saturating_sub(start_edge_time_ns) as f64 / 1_000_000_000.0;
    format!(
        "{} sampled keyframes from phone {} view over {:.1}s, dominated by {}.",
        keyframe_count,
        camera_mode.trim(),
        duration_s,
        action_guess.replace('_', " ")
    )
}

fn select_preview_frames(keyframe_relpaths: &[String], max_frames: usize) -> Vec<String> {
    if keyframe_relpaths.is_empty() {
        return Vec::new();
    }
    let limit = max_frames.max(1);
    if keyframe_relpaths.len() <= limit {
        return keyframe_relpaths.to_vec();
    }
    let stride = (keyframe_relpaths.len() as f64 / limit as f64).ceil() as usize;
    let mut selected = Vec::new();
    for relpath in keyframe_relpaths.iter().step_by(stride.max(1)) {
        selected.push(relpath.clone());
        if selected.len() >= limit {
            break;
        }
    }
    if selected.len() < limit {
        if let Some(last) = keyframe_relpaths.last() {
            if selected.last() != Some(last) {
                selected.push(last.clone());
            }
        }
    }
    selected
}

fn unique_strings(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let normalized = value.trim();
        if normalized.is_empty() {
            continue;
        }
        if seen.insert(normalized.to_string()) {
            out.push(normalized.to_string());
        }
    }
    out
}

fn dominant_string(values: &[String]) -> Option<String> {
    let mut counts = HashMap::<String, usize>::new();
    for value in values {
        let normalized = value.trim();
        if normalized.is_empty() {
            continue;
        }
        *counts.entry(normalized.to_string()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
        .map(|item| item.0)
}

fn last_non_empty_string(values: &[String]) -> Option<String> {
    values
        .iter()
        .rev()
        .find_map(|value| normalize_non_empty(Some(value.clone())))
}

fn mean_non_empty_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum = values.iter().copied().sum::<f64>();
    Some(sum / (values.len() as f64))
}

fn semantic_inference_status(inference_source: &str, latency_ms: Option<f64>) -> String {
    if inference_source.trim().starts_with("heuristic") {
        "heuristic".to_string()
    } else if inference_source.trim().contains("fallback") {
        "ready_fallback".to_string()
    } else if latency_ms.is_some() {
        "ready".to_string()
    } else {
        "pending".to_string()
    }
}

fn semantic_action_guess_from_caption(caption: &str) -> String {
    let lowered = caption.to_ascii_lowercase();
    if lowered.contains("reach") || lowered.contains("grab") || lowered.contains("pick") {
        "reaching_object".to_string()
    } else if lowered.contains("hold") || lowered.contains("carry") {
        "holding_object".to_string()
    } else if lowered.contains("walk") || lowered.contains("move") {
        "moving_phone".to_string()
    } else {
        "steady_capture".to_string()
    }
}

fn semantic_keywords_from_caption(caption: &str) -> (Vec<String>, Vec<String>) {
    let lowered = caption.to_ascii_lowercase();
    let mut tags = Vec::new();
    let mut objects = Vec::new();
    for (needle, tag) in [
        ("person", "person_visible"),
        ("people", "person_visible"),
        ("hand", "hand_visible"),
        ("hands", "hand_visible"),
        ("desk", "desk_scene"),
        ("table", "desk_scene"),
        ("room", "indoor_scene"),
        ("office", "indoor_scene"),
    ] {
        if lowered.contains(needle) {
            tags.push(tag.to_string());
        }
    }
    for object in [
        "hand", "phone", "mug", "cup", "bottle", "laptop", "keyboard", "desk", "table", "chair",
        "door", "screen",
    ] {
        if lowered.contains(object) {
            objects.push(object.to_string());
        }
    }
    (unique_strings(&tags), unique_strings(&objects))
}

fn deterministic_embedding(text: &str) -> Vec<f32> {
    let mut accum = [0u32; 16];
    for (idx, byte) in text.as_bytes().iter().enumerate() {
        let slot = idx % accum.len();
        let mixed = u32::from(*byte)
            .wrapping_mul((idx as u32).wrapping_add(17))
            .rotate_left((idx % 13) as u32);
        accum[slot] = accum[slot].wrapping_add(mixed);
    }
    accum
        .iter()
        .map(|value| (*value as f32) / (u32::MAX as f32))
        .collect()
}

async fn request_vlm_sidecar_inference(
    cfg: Config,
    image_bytes: Vec<u8>,
    frame_id: u64,
    source_time_ns: u64,
    edge_time_ns: u64,
    camera_mode: String,
    sample_reasons: Vec<String>,
) -> Result<SemanticInferenceResult, String> {
    let base = cfg
        .vlm_sidecar_base
        .trim()
        .trim_end_matches('/')
        .to_string();
    if base.is_empty() {
        return Err("EDGE_VLM_SIDECAR_BASE empty".to_string());
    }

    let request_body = serde_json::json!({
        "frame_id": frame_id,
        "source_time_ns": source_time_ns,
        "edge_time_ns": edge_time_ns,
        "camera_mode": camera_mode,
        "sample_reasons": sample_reasons,
        "prompt_version": cfg.vlm_prompt_version,
        "image_jpeg_b64": base64::engine::general_purpose::STANDARD.encode(image_bytes),
    });
    let http_client = reqwest::Client::new();
    let response = tokio::time::timeout(
        Duration::from_millis(cfg.vlm_inference_timeout_ms.max(1)),
        http_client
            .post(format!("{base}/infer"))
            .json(&request_body)
            .send(),
    )
    .await
    .map_err(|_| {
        format!(
            "vlm sidecar timeout after {}ms",
            cfg.vlm_inference_timeout_ms
        )
    })?
    .map_err(|error| format!("vlm sidecar request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("vlm sidecar returned HTTP {}", response.status()));
    }
    let body = response
        .json::<VlmSidecarResponse>()
        .await
        .map_err(|error| format!("parse vlm sidecar response failed: {error}"))?;
    if !body.ok {
        return Err(body
            .error
            .unwrap_or_else(|| "vlm sidecar returned ok=false".to_string()));
    }

    let caption = normalize_non_empty(body.caption)
        .unwrap_or_else(|| "Scene observed from the phone main view.".to_string());
    let (derived_tags, derived_objects) = semantic_keywords_from_caption(&caption);
    let mut degraded_reasons = unique_strings(&body.degraded_reasons);
    if body.fallback_active {
        degraded_reasons.push("sidecar_model_fallback_active".to_string());
    }
    Ok(SemanticInferenceResult {
        caption: caption.clone(),
        tags: if body.tags.is_empty() {
            derived_tags
        } else {
            unique_strings(&body.tags)
        },
        objects: if body.objects.is_empty() {
            derived_objects
        } else {
            unique_strings(&body.objects)
        },
        action_guess: normalize_non_empty(body.action_guess)
            .unwrap_or_else(|| semantic_action_guess_from_caption(&caption)),
        model_id: normalize_non_empty(body.model_id).unwrap_or(cfg.vlm_model_id),
        inference_source: normalize_non_empty(body.inference_source)
            .unwrap_or_else(|| "vlm_sidecar".to_string()),
        latency_ms: body.latency_ms,
        degraded_reasons,
    })
}

async fn open_append(path: PathBuf) -> Result<tokio::fs::File, String> {
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err("append path 不能包含 . 或 .. 路径段".to_string());
    }
    tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| format!("打开文件失败: {} ({e})", path.display()))
}

async fn open_append_bin(path: PathBuf) -> Result<tokio::fs::File, String> {
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err("append binary path 不能包含 . 或 .. 路径段".to_string());
    }
    tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| format!("打开二进制文件失败: {} ({e})", path.display()))
}

async fn count_jsonl_lines(path: PathBuf) -> Result<u64, String> {
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err("jsonl path 不能包含 . 或 .. 路径段".to_string());
    }
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => Ok(content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count() as u64),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(format!("统计 JSONL 行数失败: {} ({error})", path.display())),
    }
}

#[derive(Default)]
struct CsiIndexSummary {
    rows_with_nodes: u64,
    max_node_count: usize,
}

async fn summarize_csi_index(path: PathBuf) -> Result<CsiIndexSummary, String> {
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err("csi index path 不能包含 . 或 .. 路径段".to_string());
    }
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CsiIndexSummary::default())
        }
        Err(error) => return Err(format!("读取 CSI 索引失败: {} ({error})", path.display())),
    };

    let mut summary = CsiIndexSummary::default();
    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(trimmed).map_err(|error| {
            format!(
                "解析 CSI 索引失败: {}:{} ({error})",
                path.display(),
                line_no + 1
            )
        })?;
        let node_count = value
            .get("node_count")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;
        if node_count > 0 {
            summary.rows_with_nodes = summary.rows_with_nodes.saturating_add(1);
        }
        summary.max_node_count = summary.max_node_count.max(node_count);
    }
    Ok(summary)
}

#[derive(Default)]
struct MediaIndexSummary {
    record_count: u64,
    frame_count: u64,
}

async fn summarize_media_index_frames(path: PathBuf) -> Result<MediaIndexSummary, String> {
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err("media index path 不能包含 . 或 .. 路径段".to_string());
    }
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(MediaIndexSummary::default())
        }
        Err(error) => return Err(format!("读取媒体索引失败: {} ({error})", path.display())),
    };

    let mut summary = MediaIndexSummary::default();
    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(trimmed).map_err(|error| {
            format!(
                "解析媒体索引失败: {}:{} ({error})",
                path.display(),
                line_no + 1
            )
        })?;
        summary.record_count = summary.record_count.saturating_add(1);
        summary.frame_count = summary.frame_count.saturating_add(
            value
                .get("frame_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0),
        );
    }
    Ok(summary)
}

async fn next_csi_chunk_index_from_disk(base_dir: &Path) -> Result<u64, String> {
    let path = base_dir
        .join("raw")
        .join("csi")
        .join("chunks")
        .join("index.jsonl");
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => {
            return Err(format!(
                "读取 CSI chunk 索引失败: {} ({error})",
                path.display()
            ))
        }
    };
    let mut next = 0u64;
    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(trimmed).map_err(|error| {
            format!(
                "解析 CSI chunk 索引失败: {}:{} ({error})",
                path.display(),
                line_no + 1
            )
        })?;
        let chunk_index = value
            .get("chunk_index")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        next = next.max(chunk_index.saturating_add(1));
    }
    Ok(next)
}

async fn migrate_legacy_iphone_layout(base_dir: &Path) -> Result<Vec<String>, String> {
    let raw_iphone_dir = base_dir.join("raw").join("iphone");
    let raw_iphone_wide_dir = raw_iphone_dir.join("wide");
    tokio::fs::create_dir_all(&raw_iphone_wide_dir)
        .await
        .map_err(|e| {
            format!(
                "创建 iPhone wide 目录失败: {} ({e})",
                raw_iphone_wide_dir.display()
            )
        })?;

    let mut notes = Vec::new();
    for relpath in [
        "kpts_depth.jsonl",
        "pose_imu.jsonl",
        "media_index.jsonl",
        "depth",
        "chunks",
    ] {
        let old_path = raw_iphone_dir.join(relpath);
        let new_path = raw_iphone_wide_dir.join(relpath);
        if !tokio::fs::try_exists(&old_path)
            .await
            .map_err(|e| format!("检查旧 iPhone 路径失败: {} ({e})", old_path.display()))?
        {
            continue;
        }
        if tokio::fs::try_exists(&new_path)
            .await
            .map_err(|e| format!("检查新 iPhone 路径失败: {} ({e})", new_path.display()))?
        {
            continue;
        }
        tokio::fs::rename(&old_path, &new_path).await.map_err(|e| {
            format!(
                "迁移旧 iPhone 路径失败: {} -> {} ({e})",
                old_path.display(),
                new_path.display()
            )
        })?;
        notes.push(format!(
            "migrated {} -> {}",
            old_path
                .strip_prefix(base_dir)
                .unwrap_or(&old_path)
                .display(),
            new_path
                .strip_prefix(base_dir)
                .unwrap_or(&new_path)
                .display()
        ));
    }
    Ok(notes)
}

async fn load_existing_session_identity(
    base_dir: &Path,
    fallback_session_id: &str,
) -> Result<(String, u64, SessionCrowdContext), String> {
    for relpath in ["manifest.json", "demo_capture_bundle.json"] {
        let path = base_dir.join(relpath);
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(format!(
                    "读取 session 元数据失败: {} ({error})",
                    path.display()
                ))
            }
        };
        let value = serde_json::from_str::<serde_json::Value>(&content)
            .map_err(|e| format!("解析 session 元数据失败: {} ({e})", path.display()))?;
        let trip_id = value
            .get("trip_id")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(fallback_session_id)
            .to_string();
        let created_unix_ms = value
            .get("created_unix_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or_else(now_unix_ms);
        let context = value
            .get("session_context")
            .map(parse_session_context)
            .unwrap_or_default();
        return Ok((trip_id, created_unix_ms, context));
    }

    let capture_pose_path = base_dir
        .join("raw")
        .join("iphone")
        .join("wide")
        .join("kpts_depth.jsonl");
    if let Ok(content) = tokio::fs::read_to_string(&capture_pose_path).await {
        if let Some(first) = content.lines().find(|line| !line.trim().is_empty()) {
            let value = serde_json::from_str::<serde_json::Value>(first).map_err(|e| {
                format!(
                    "解析 capture_pose 失败: {} ({e})",
                    capture_pose_path.display()
                )
            })?;
            let trip_id = value
                .get("trip_id")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(fallback_session_id)
                .to_string();
            return Ok((trip_id, now_unix_ms(), SessionCrowdContext::default()));
        }
    }

    Ok((
        fallback_session_id.to_string(),
        now_unix_ms(),
        SessionCrowdContext::default(),
    ))
}

async fn load_persisted_session_context(base_dir: &Path) -> Result<SessionCrowdContext, String> {
    if base_dir
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(format!("非法 session 目录: {}", base_dir.display()));
    }
    let parent_name = base_dir
        .parent()
        .and_then(|path| path.file_name())
        .and_then(|value| value.to_str());
    if parent_name != Some("session") {
        return Err(format!("非法 session 目录: {}", base_dir.display()));
    }
    let session_id = base_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("非法 session 目录: {}", base_dir.display()))?;
    if session_id.is_empty()
        || matches!(session_id, "." | "..")
        || session_id
            .chars()
            .any(|ch| matches!(ch, '/' | '\\' | '\0' | '\n' | '\r'))
    {
        return Err(format!("非法 session 目录: {}", base_dir.display()));
    }
    let mut merged = SessionCrowdContext::default();
    for relpath in [
        "upload/upload_manifest.json",
        "manifest.json",
        "demo_capture_bundle.json",
    ] {
        let path = base_dir.join(relpath);
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(format!(
                    "读取 session_context 元数据失败: {} ({error})",
                    path.display()
                ))
            }
        };
        let value = serde_json::from_str::<serde_json::Value>(&content)
            .map_err(|e| format!("解析 session_context 元数据失败: {} ({e})", path.display()))?;
        if let Some(context) = value.get("session_context").map(parse_session_context) {
            merged.merge_missing_from(&context);
        }
    }
    Ok(merged)
}

fn parse_session_context(value: &serde_json::Value) -> SessionCrowdContext {
    SessionCrowdContext {
        capture_device_id: normalize_non_empty(
            value
                .get("capture_device_id")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
        ),
        operator_id: normalize_non_empty(
            value
                .get("operator_id")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
        ),
        task_id: normalize_non_empty(
            value
                .get("task_id")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
        ),
        task_ids: normalize_task_ids(
            value
                .get("task_ids")
                .and_then(|value| value.as_array())
                .into_iter()
                .flatten()
                .filter_map(|item| item.as_str().map(ToOwned::to_owned))
                .collect(),
        ),
        runtime_profile: normalize_non_empty(
            value
                .get("runtime_profile")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
        ),
        upload_policy_mode: normalize_non_empty(
            value
                .get("upload_policy_mode")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
        ),
        raw_residency: normalize_non_empty(
            value
                .get("raw_residency")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
        ),
        preview_residency: normalize_non_empty(
            value
                .get("preview_residency")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned),
        ),
        runtime_flags: value
            .get("runtime_flags")
            .cloned()
            .and_then(|value| serde_json::from_value::<SessionRuntimeFlags>(value).ok())
            .unwrap_or_default(),
    }
}

async fn ensure_edge_frames_snapshot(
    base_dir: &Path,
    trip_id: &str,
    session_id: &str,
    cfg: &Config,
) -> Result<bool, String> {
    let path = base_dir.join("calibration").join("edge_frames.json");
    if tokio::fs::try_exists(&path)
        .await
        .map_err(|e| format!("检查 edge_frames 快照失败: {} ({e})", path.display()))?
    {
        return Ok(false);
    }
    let value = serde_json::json!({
        "type": "sensor_calibration_snapshot",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "sensor_kind": "edge_frames",
        "sensor_id": "edge_frames",
        "edge_time_ns": 0u64,
        "operator_frame": cfg.operator_frame,
        "robot_base_frame": cfg.robot_base_frame,
        "extrinsic_version": cfg.extrinsic_version,
        "extrinsic_translation_m": cfg.extrinsic_translation_m,
        "extrinsic_rotation_quat": cfg.extrinsic_rotation_quat,
        "notes": "edge static frame snapshot"
    });
    ActiveSession::write_json_pretty(path, &value).await?;
    Ok(true)
}

async fn backfill_phone_inputs_from_capture_pose(
    base_dir: &Path,
    trip_id: &str,
    session_id: &str,
) -> Result<(u64, u64, Vec<String>), String> {
    let capture_pose_path = base_dir
        .join("raw")
        .join("iphone")
        .join("wide")
        .join("kpts_depth.jsonl");
    let content = match tokio::fs::read_to_string(&capture_pose_path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((0, 0, Vec::new()))
        }
        Err(error) => {
            return Err(format!(
                "读取 capture_pose 失败: {} ({error})",
                capture_pose_path.display()
            ))
        }
    };

    let pose_imu_path = base_dir
        .join("raw")
        .join("iphone")
        .join("wide")
        .join("pose_imu.jsonl");
    let depth_index_path = base_dir
        .join("raw")
        .join("iphone")
        .join("wide")
        .join("depth")
        .join("index.jsonl");
    let has_pose_imu = count_jsonl_lines(pose_imu_path.clone()).await? > 0;
    let has_depth_index = count_jsonl_lines(depth_index_path.clone()).await? > 0;
    let mut pose_lines = Vec::new();
    let mut depth_lines = Vec::new();
    let mut notes = Vec::new();

    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(trimmed).map_err(|error| {
            format!(
                "解析 capture_pose 失败: {}:{} ({error})",
                capture_pose_path.display(),
                line_no + 1
            )
        })?;

        if !has_pose_imu
            && (value.get("device_pose").is_some()
                || value.get("imu").is_some()
                || value
                    .get("camera")
                    .and_then(|camera| camera.get("calibration"))
                    .is_some())
        {
            let camera = value.get("camera");
            pose_lines.push(serde_json::json!({
                "type": "phone_vision_input",
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "device_id": value.get("device_id").and_then(|v| v.as_str()).unwrap_or(""),
                "operator_track_id": value.get("operator_track_id").cloned().unwrap_or(serde_json::Value::Null),
                "source_time_ns": value.get("source_time_ns").and_then(|v| v.as_u64()).unwrap_or(0),
                "recv_time_ns": value.get("recv_time_ns").and_then(|v| v.as_u64()).unwrap_or(0),
                "edge_time_ns": value.get("edge_time_ns").and_then(|v| v.as_u64()).unwrap_or(0),
                "frame_id": value.get("frame_id").and_then(|v| v.as_u64()).unwrap_or(0),
                "camera_mode": camera.and_then(|v| v.get("mode")).cloned().unwrap_or(serde_json::Value::Null),
                "image_w": camera.and_then(|v| v.get("image_w")).cloned().unwrap_or(serde_json::Value::Null),
                "image_h": camera.and_then(|v| v.get("image_h")).cloned().unwrap_or(serde_json::Value::Null),
                "sensor_image_w": camera.and_then(|v| v.get("sensor_image_w")).cloned().unwrap_or(serde_json::Value::Null),
                "sensor_image_h": camera.and_then(|v| v.get("sensor_image_h")).cloned().unwrap_or(serde_json::Value::Null),
                "normalized_was_rotated_right": camera.and_then(|v| v.get("normalized_was_rotated_right")).cloned().unwrap_or(serde_json::Value::Null),
                "camera_has_depth": camera.and_then(|v| v.get("has_depth")).cloned().unwrap_or(serde_json::Value::Null),
                "camera_calibration": camera.and_then(|v| v.get("calibration")).cloned().unwrap_or(serde_json::Value::Null),
                "device_pose": value.get("device_pose").cloned().unwrap_or(serde_json::Value::Null),
                "imu": value.get("imu").cloned().unwrap_or(serde_json::Value::Null),
                "depth_w": value.get("depth_w").cloned().unwrap_or(serde_json::Value::Null),
                "depth_h": value.get("depth_h").cloned().unwrap_or(serde_json::Value::Null),
                "depth_relpath": value.get("depth_relpath").cloned().unwrap_or(serde_json::Value::Null),
                "depth_payload_decode_error": serde_json::Value::Null,
            }));
        }

        if !has_depth_index {
            if let Some(relpath) = value.get("depth_relpath").and_then(|v| v.as_str()) {
                let relpath = relpath.trim();
                if !relpath.is_empty()
                    && tokio::fs::try_exists(base_dir.join(relpath))
                        .await
                        .map_err(|e| {
                            format!(
                                "检查 depth frame 失败: {} ({e})",
                                base_dir.join(relpath).display()
                            )
                        })?
                {
                    let file_bytes = tokio::fs::metadata(base_dir.join(relpath))
                        .await
                        .map(|meta| meta.len())
                        .unwrap_or(0);
                    depth_lines.push(serde_json::json!({
                        "type": "iphone_depth_frame",
                        "schema_version": "1.0.0",
                        "trip_id": trip_id,
                        "session_id": session_id,
                        "device_id": value.get("device_id").and_then(|v| v.as_str()).unwrap_or(""),
                        "source_time_ns": value.get("source_time_ns").and_then(|v| v.as_u64()).unwrap_or(0),
                        "edge_time_ns": value.get("edge_time_ns").and_then(|v| v.as_u64()).unwrap_or(0),
                        "frame_id": value.get("frame_id").and_then(|v| v.as_u64()).unwrap_or(0),
                        "depth_w": value.get("depth_w").and_then(|v| v.as_u64()).unwrap_or(0),
                        "depth_h": value.get("depth_h").and_then(|v| v.as_u64()).unwrap_or(0),
                        "file_relpath": relpath,
                        "file_bytes": file_bytes,
                        "storage_format": "f32le",
                    }));
                }
            }
        }
    }

    let mut backfilled_pose_imu_lines = 0u64;
    if !has_pose_imu && !pose_lines.is_empty() {
        let payload = pose_lines
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("编码 pose_imu 回填失败: {e}"))?
            .join("\n");
        let bytes = format!("{payload}\n");
        tokio::fs::write(&pose_imu_path, bytes)
            .await
            .map_err(|e| format!("写入 pose_imu 回填失败: {} ({e})", pose_imu_path.display()))?;
        backfilled_pose_imu_lines = pose_lines.len() as u64;
        notes.push(format!(
            "backfilled {} lines into {}",
            backfilled_pose_imu_lines,
            pose_imu_path
                .strip_prefix(base_dir)
                .unwrap_or(&pose_imu_path)
                .display()
        ));
    }

    let mut backfilled_depth_index_lines = 0u64;
    if !has_depth_index && !depth_lines.is_empty() {
        let payload = depth_lines
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("编码 depth index 回填失败: {e}"))?
            .join("\n");
        let bytes = format!("{payload}\n");
        tokio::fs::write(&depth_index_path, bytes)
            .await
            .map_err(|e| {
                format!(
                    "写入 depth index 回填失败: {} ({e})",
                    depth_index_path.display()
                )
            })?;
        backfilled_depth_index_lines = depth_lines.len() as u64;
        notes.push(format!(
            "backfilled {} lines into {}",
            backfilled_depth_index_lines,
            depth_index_path
                .strip_prefix(base_dir)
                .unwrap_or(&depth_index_path)
                .display()
        ));
    }

    Ok((
        backfilled_pose_imu_lines,
        backfilled_depth_index_lines,
        notes,
    ))
}

fn normalize_non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty())
}

fn normalize_task_ids(task_ids: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for task_id in task_ids
        .into_iter()
        .map(|task_id| task_id.trim().to_string())
        .filter(|task_id| !task_id.is_empty())
    {
        if seen.insert(task_id.clone()) {
            normalized.push(task_id);
        }
    }
    normalized
}

fn sanitize_component(input: &str) -> String {
    let raw = input.trim();
    if raw.is_empty() {
        return "unknown".to_string();
    }
    raw.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | ' ' | '\t' | '\n' | '\r' => '_',
            _ => c,
        })
        .collect()
}

fn range_opt(start_line: u64, end_line: u64) -> Option<IndexRange> {
    if start_line == 0 || end_line == 0 || end_line < start_line {
        return None;
    }
    Some(IndexRange {
        start_line,
        end_line,
    })
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

async fn read_jsonl_range(
    path: &Path,
    range: IndexRange,
) -> Result<Vec<serde_json::Value>, String> {
    let raw = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| format!("读取 JSONL 失败: {} ({e})", path.display()))?;
    let start = range.start_line as usize;
    let end = range.end_line as usize;
    let mut out = Vec::new();
    for (line_no, line) in raw.lines().enumerate() {
        let one_based = line_no + 1;
        if one_based < start {
            continue;
        }
        if one_based > end {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(trimmed)
            .map_err(|e| format!("解析 JSONL 失败: {}:{one_based} ({e})", path.display()))?;
        out.push(value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::SessionRecorder;
    use crate::config::Config;
    use crate::protocol::version_guard::ProtocolVersionInfo;

    #[tokio::test]
    async fn repair_existing_sessions_should_migrate_legacy_iphone_layout_and_rebuild_metadata() {
        let unique = format!(
            "edge-recorder-repair-{}-{}",
            std::process::id(),
            super::now_unix_ms()
        );
        let data_dir = std::env::temp_dir().join(unique);
        let session_dir = data_dir.join("session").join("sess-legacy-001");
        let raw_iphone_dir = session_dir.join("raw").join("iphone");
        let fused_dir = session_dir.join("fused");
        let teleop_dir = session_dir.join("teleop");
        let sync_dir = session_dir.join("sync");
        let calibration_dir = session_dir.join("calibration");

        tokio::fs::create_dir_all(raw_iphone_dir.join("chunks").join("000001"))
            .await
            .expect("iphone chunks dir");
        tokio::fs::create_dir_all(&fused_dir)
            .await
            .expect("fused dir");
        tokio::fs::create_dir_all(&teleop_dir)
            .await
            .expect("teleop dir");
        tokio::fs::create_dir_all(&sync_dir)
            .await
            .expect("sync dir");
        tokio::fs::create_dir_all(&calibration_dir)
            .await
            .expect("calibration dir");

        let capture_pose_line = serde_json::json!({
            "type": "capture_pose_packet",
            "schema_version": "1.0.0",
            "trip_id": "trip-legacy-001",
            "session_id": "sess-legacy-001",
            "device_id": "ios-legacy-001",
            "source_time_ns": 1u64,
            "recv_time_ns": 2u64,
            "edge_time_ns": 3u64,
            "frame_id": 1u64,
            "body_layout": "pico_body_24",
            "hand_layout": "pico_hand_26",
            "body_kpts_3d": [[0.0, 0.1, 0.8]],
            "hand_kpts_3d": [[0.0, 0.1, 0.8]],
            "camera": {
                "mode": "teleop_phone_back",
                "image_w": 1280,
                "image_h": 720,
                "has_depth": true,
                "calibration": {
                    "fx": 1.0,
                    "fy": 1.0
                }
            },
            "device_pose": {
                "source": "ios_arkit",
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
            },
            "imu": {
                "gyro_radps": [0.0, 0.0, 0.0],
                "accel_mps2": [0.0, 0.0, 0.0]
            }
        });
        tokio::fs::write(
            raw_iphone_dir.join("kpts_depth.jsonl"),
            format!("{}\n", serde_json::to_string(&capture_pose_line).unwrap()),
        )
        .await
        .expect("legacy kpts");
        tokio::fs::write(
            raw_iphone_dir.join("media_index.jsonl"),
            "{\"type\":\"media_chunk_index\",\"track\":\"main\"}\n",
        )
        .await
        .expect("legacy media index");
        tokio::fs::write(
            fused_dir.join("human_demo_pose.jsonl"),
            "{\"type\":\"human_demo_pose_packet\"}\n",
        )
        .await
        .expect("human demo pose");
        tokio::fs::write(
            teleop_dir.join("teleop_frame.jsonl"),
            "{\"type\":\"teleop_frame_v1\"}\n",
        )
        .await
        .expect("teleop frame");
        tokio::fs::write(
            sync_dir.join("time_sync_samples.jsonl"),
            "{\"type\":\"time_sync_sample\"}\n",
        )
        .await
        .expect("time sync");
        tokio::fs::write(
            calibration_dir.join("iphone_capture.json"),
            "{\"type\":\"sensor_calibration_snapshot\"}\n",
        )
        .await
        .expect("iphone calibration");
        tokio::fs::write(
            session_dir.join("manifest.json"),
            serde_json::json!({
                "type": "session_manifest",
                "schema_version": "1.0.0",
                "trip_id": "trip-legacy-001",
                "session_id": "sess-legacy-001",
                "created_unix_ms": 1234u64,
                "session_context": {
                    "capture_device_id": "edge-legacy-001"
                }
            })
            .to_string(),
        )
        .await
        .expect("legacy manifest");

        let recorder = SessionRecorder::new(data_dir.to_string_lossy().to_string());
        let protocol = ProtocolVersionInfo {
            name: "teleop-protocol".to_string(),
            version: "1.13.0".to_string(),
            schema_sha256: "test".to_string(),
        };
        let cfg = Config::from_env().expect("config");

        let results = recorder
            .repair_existing_sessions(&protocol, &cfg)
            .await
            .expect("repair existing sessions");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "sess-legacy-001");
        assert_eq!(results[0].trip_id, "trip-legacy-001");
        assert!(results[0].backfilled_pose_imu_lines > 0);
        assert!(results[0].quality_status.is_some());

        assert!(
            tokio::fs::try_exists(raw_iphone_dir.join("wide").join("kpts_depth.jsonl"))
                .await
                .expect("wide kpts exists")
        );
        assert!(
            tokio::fs::try_exists(raw_iphone_dir.join("wide").join("pose_imu.jsonl"))
                .await
                .expect("wide pose imu exists")
        );
        assert!(
            tokio::fs::try_exists(session_dir.join("qa").join("upload_policy.json"))
                .await
                .expect("upload policy exists")
        );
        assert!(
            tokio::fs::try_exists(session_dir.join("upload").join("upload_manifest.json"))
                .await
                .expect("upload manifest exists")
        );
        assert!(
            tokio::fs::try_exists(session_dir.join("upload").join("upload_queue.json"))
                .await
                .expect("upload queue exists")
        );

        let quality: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(session_dir.join("qa").join("local_quality_report.json"))
                .await
                .expect("quality report"),
        )
        .expect("quality json");
        assert_eq!(
            quality.get("status").and_then(|value| value.as_str()),
            Some("retry_recommended")
        );

        tokio::fs::remove_dir_all(&data_dir)
            .await
            .expect("cleanup temp data dir");
    }

    #[tokio::test]
    async fn refresh_metadata_should_preserve_persisted_operator_and_task_context() {
        let unique = format!(
            "edge-recorder-context-preserve-{}-{}",
            std::process::id(),
            super::now_unix_ms()
        );
        let data_dir = std::env::temp_dir().join(unique);
        let protocol = ProtocolVersionInfo {
            name: "teleop-protocol".to_string(),
            version: "1.13.0".to_string(),
            schema_sha256: "test".to_string(),
        };
        let cfg = Config::from_env().expect("config");

        let mut active = super::ActiveSession::start(
            &data_dir,
            "trip-preserve-001",
            "sess-preserve-001",
            &protocol,
            &cfg,
        )
        .await
        .expect("start active session");

        active.session_context.capture_device_id = Some("edge-preserve-001".to_string());
        active.session_context.operator_id = Some("operator-preserve-001".to_string());
        active.session_context.task_id = Some("task-preserve-001".to_string());
        active.session_context.task_ids = vec!["task-preserve-001".to_string()];
        active
            .refresh_demo_bundle(&protocol, &cfg)
            .await
            .expect("write demo bundle");
        active
            .refresh_session_metadata_artifacts(&protocol, &cfg)
            .await
            .expect("write metadata artifacts");

        active.session_context.operator_id = None;
        active.session_context.task_id = None;
        active.session_context.task_ids.clear();
        active
            .refresh_session_metadata_artifacts(&protocol, &cfg)
            .await
            .expect("rebuild metadata artifacts");

        let upload_manifest: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(
                data_dir
                    .join("session")
                    .join("sess-preserve-001")
                    .join("upload")
                    .join("upload_manifest.json"),
            )
            .await
            .expect("upload manifest"),
        )
        .expect("upload manifest json");
        let session_context = upload_manifest
            .get("session_context")
            .expect("session_context present");
        assert_eq!(
            session_context
                .get("operator_id")
                .and_then(|value| value.as_str()),
            Some("operator-preserve-001")
        );
        assert_eq!(
            session_context
                .get("task_id")
                .and_then(|value| value.as_str()),
            Some("task-preserve-001")
        );

        tokio::fs::remove_dir_all(&data_dir)
            .await
            .expect("cleanup temp data dir");
    }
}
