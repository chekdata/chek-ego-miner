use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Deserialize;
use serde_json::{json, Value};
use tar::Builder;
use tokio::time::MissedTickBehavior;
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::recorder::upload_queue::{self, UploadReceiptInput};
use crate::AppState;

#[derive(Debug, Deserialize)]
struct UploadQueueFile {
    trip_id: String,
    session_id: String,
    ready_for_upload: bool,
    entries: Vec<UploadQueueEntry>,
}

#[derive(Debug, Deserialize)]
struct UploadQueueEntry {
    asset_id: String,
    relpath: String,
    kind: String,
    category: String,
    required: bool,
    exists: bool,
    byte_size: u64,
    status: String,
    retry_count: u32,
    last_receipt_unix_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct UploadManifestFile {
    trip_id: String,
    session_id: String,
    ready_for_upload: bool,
    upload_policy: Option<UploadPolicySnapshot>,
    session_context: Option<UploadManifestSessionContext>,
    artifacts: Vec<UploadManifestFileArtifact>,
}

#[derive(Debug, Clone, Deserialize)]
struct UploadManifestFileArtifact {
    id: String,
    relpath: String,
    kind: String,
    category: String,
    required: bool,
    exists: bool,
    byte_size: u64,
    line_count: Option<u64>,
    residency: Option<String>,
    upload_state: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
struct UploadPolicySnapshot {
    artifact_policy_mode: Option<String>,
    runtime_profile: Option<String>,
    raw_residency_default: Option<String>,
    preview_residency_default: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
struct UploadManifestSessionContext {
    capture_device_id: Option<String>,
    operator_id: Option<String>,
    task_id: Option<String>,
    #[serde(default)]
    task_ids: Vec<String>,
    runtime_profile: Option<String>,
    upload_policy_mode: Option<String>,
    raw_residency: Option<String>,
    preview_residency: Option<String>,
}

#[derive(Debug)]
struct LoadedUploadManifest {
    raw_value: Value,
    manifest: UploadManifestFile,
    by_asset_id: HashMap<String, UploadManifestFileArtifact>,
}

#[derive(Debug, Default, Deserialize)]
struct LocalQualityReportSummary {
    status: Option<String>,
    score_percent: Option<f64>,
    #[serde(default)]
    missing_artifacts: Vec<String>,
    #[serde(default)]
    recommended_actions: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RemoteUploadResponse {
    remote_object_key: Option<String>,
    remote_upload_id: Option<String>,
    error: Option<RemoteUploadError>,
}

#[derive(Debug, Deserialize)]
struct RemoteUploadError {
    message: Option<String>,
}

struct PreparedUploadPayload {
    bytes: Vec<u8>,
    content_type: &'static str,
    upload_encoding: Option<&'static str>,
}

#[derive(Debug, Deserialize)]
struct ControlPlaneArtifactResponse {
    upload_target: ControlPlaneUploadTarget,
}

#[derive(Debug, Deserialize)]
struct ControlPlaneUploadTarget {
    storage_key: String,
    transport: ControlPlaneUploadTransport,
}

#[derive(Debug, Deserialize)]
struct ControlPlaneUploadTransport {
    mode: String,
    method: String,
    url: String,
    auth_strategy: Option<String>,
    #[serde(default)]
    supports_multipart: bool,
    file_field: Option<String>,
    metadata_field: Option<String>,
    #[serde(default)]
    scope_token_required: bool,
    scope_token_header: Option<String>,
    storage_key_header: Option<String>,
    headers: Option<HashMap<String, String>>,
}

pub async fn run_crowd_upload_worker(state: AppState) {
    let control_base_url = state
        .config
        .crowd_upload_control_base_url
        .trim()
        .trim_end_matches('/')
        .to_string();
    let artifact_url = state.config.crowd_upload_artifact_url.trim().to_string();
    if control_base_url.is_empty() && artifact_url.is_empty() {
        warn!(
            "crowd upload enabled but EDGE_CROWD_UPLOAD_CONTROL_BASE_URL / EDGE_CROWD_UPLOAD_ARTIFACT_URL are empty; uploader disabled"
        );
        return;
    }

    let poll = Duration::from_millis(state.config.crowd_upload_poll_ms.max(250));
    let mut ticker = tokio::time::interval(poll);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    info!(
        upload_backend=%if control_base_url.is_empty() { "direct_artifact_url" } else { "control_plane" },
        control_base_url=%control_base_url,
        artifact_url=%artifact_url,
        poll_ms=state.config.crowd_upload_poll_ms,
        max_retry_count=state.config.crowd_upload_max_retry_count,
        "crowd upload worker started"
    );

    loop {
        ticker.tick().await;
        if let Err(error) = run_upload_cycle(&state, &control_base_url, &artifact_url).await {
            warn!(error=%error, "crowd upload worker cycle failed");
        }
    }
}

async fn run_upload_cycle(
    state: &AppState,
    control_base_url: &str,
    artifact_url: &str,
) -> Result<(), String> {
    let session_root = Path::new(&state.config.data_dir).join("session");
    if !tokio::fs::try_exists(&session_root)
        .await
        .map_err(|e| format!("检查 session 目录失败: {} ({e})", session_root.display()))?
    {
        return Ok(());
    }

    let session_dirs = collect_session_dirs(&session_root).await?;
    for base_dir in session_dirs {
        if let Err(error) =
            process_session_uploads(state, control_base_url, artifact_url, &base_dir).await
        {
            warn!(session_dir=%base_dir.display(), error=%error, "session upload processing failed");
        }
    }
    Ok(())
}

async fn collect_session_dirs(session_root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut entries = tokio::fs::read_dir(session_root)
        .await
        .map_err(|e| format!("读取 session 目录失败: {} ({e})", session_root.display()))?;
    let mut session_dirs = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| format!("遍历 session 目录失败: {} ({e})", session_root.display()))?
    {
        let file_type = entry.file_type().await.map_err(|e| {
            format!(
                "读取 session file_type 失败: {} ({e})",
                entry.path().display()
            )
        })?;
        if file_type.is_dir() {
            session_dirs.push(entry.path());
        }
    }
    session_dirs.sort();
    Ok(session_dirs)
}

async fn load_upload_manifest(base_dir: &Path) -> Result<LoadedUploadManifest, String> {
    let path = base_dir.join("upload").join("upload_manifest.json");
    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("读取 upload_manifest.json 失败: {} ({e})", path.display()))?;
    let raw_value: Value = serde_json::from_str(&content)
        .map_err(|e| format!("解析 upload_manifest.json 失败: {e}"))?;
    let manifest: UploadManifestFile = serde_json::from_value(raw_value.clone())
        .map_err(|e| format!("解析 upload_manifest 结构失败: {e}"))?;
    let by_asset_id = manifest
        .artifacts
        .iter()
        .cloned()
        .map(|artifact| (artifact.id.clone(), artifact))
        .collect::<HashMap<_, _>>();
    Ok(LoadedUploadManifest {
        raw_value,
        manifest,
        by_asset_id,
    })
}

async fn load_local_quality_report(base_dir: &Path) -> Result<LocalQualityReportSummary, String> {
    let path = base_dir.join("qa").join("local_quality_report.json");
    if !tokio::fs::try_exists(&path).await.map_err(|e| {
        format!(
            "检查 local_quality_report.json 失败: {} ({e})",
            path.display()
        )
    })? {
        return Ok(LocalQualityReportSummary::default());
    }
    let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
        format!(
            "读取 local_quality_report.json 失败: {} ({e})",
            path.display()
        )
    })?;
    serde_json::from_str::<LocalQualityReportSummary>(&content)
        .map_err(|e| format!("解析 local_quality_report.json 失败: {e}"))
}

async fn process_session_uploads(
    state: &AppState,
    control_base_url: &str,
    artifact_url: &str,
    base_dir: &Path,
) -> Result<(), String> {
    let queue_value = match upload_queue::load_or_refresh_upload_queue(base_dir).await {
        Ok(queue) => queue,
        Err(error) => {
            if error.contains("upload_manifest.json") {
                return Ok(());
            }
            return Err(error);
        }
    };
    let queue: UploadQueueFile =
        serde_json::from_value(queue_value).map_err(|e| format!("解析 upload_queue 失败: {e}"))?;
    if !queue.ready_for_upload {
        return Ok(());
    }
    let loaded_manifest = load_upload_manifest(base_dir).await?;
    if !loaded_manifest.manifest.ready_for_upload {
        return Ok(());
    }
    let local_quality = load_local_quality_report(base_dir).await?;

    if !control_base_url.trim().is_empty() {
        sync_session_to_control_plane(
            state,
            control_base_url,
            &queue,
            &loaded_manifest,
            &local_quality,
        )
        .await?;
    }

    for entry in &queue.entries {
        if !should_attempt_upload(
            entry,
            state.config.crowd_upload_max_retry_count,
            state.config.crowd_upload_uploading_stale_ms,
        ) {
            continue;
        }
        let result = if !control_base_url.trim().is_empty() {
            upload_entry_via_control_plane(
                state,
                control_base_url,
                base_dir,
                &queue,
                &loaded_manifest,
                entry,
            )
            .await
        } else {
            upload_entry_direct(state, artifact_url, base_dir, &queue, entry).await
        };
        if let Err(error) = result {
            warn!(
                trip_id=%queue.trip_id,
                session_id=%queue.session_id,
                asset_id=%entry.asset_id,
                error=%error,
                "asset upload failed"
            );
        }
    }
    Ok(())
}

async fn sync_session_to_control_plane(
    state: &AppState,
    control_base_url: &str,
    queue: &UploadQueueFile,
    loaded_manifest: &LoadedUploadManifest,
    local_quality: &LocalQualityReportSummary,
) -> Result<(), String> {
    let session_context = loaded_manifest
        .manifest
        .session_context
        .clone()
        .unwrap_or_default();
    let upload_policy = loaded_manifest
        .manifest
        .upload_policy
        .clone()
        .unwrap_or_default();
    let quality_status = normalize_non_empty(local_quality.status.as_deref())
        .unwrap_or_else(|| "pending".to_string());
    let auto_quality_score = local_quality.score_percent.unwrap_or(0.0).clamp(0.0, 100.0);
    let review_risk_score = ((100.0 - auto_quality_score) / 100.0).clamp(0.0, 1.0);
    let required_ready = queue
        .entries
        .iter()
        .filter(|entry| entry.required)
        .all(|entry| entry.status == "acked");
    let ready_for_review =
        matches!(quality_status.as_str(), "pass" | "retry_recommended") && required_ready;
    let session_status = if required_ready {
        "uploaded"
    } else {
        "uploading"
    };
    let total_size_bytes = loaded_manifest
        .manifest
        .artifacts
        .iter()
        .map(|artifact| artifact.byte_size)
        .sum::<u64>();
    let payload = json!({
        "trip_id": queue.trip_id,
        "session_id": queue.session_id,
        "operator_id": session_context.operator_id.clone().unwrap_or_default(),
        "capture_device_id": session_context.capture_device_id.clone().unwrap_or_default(),
        "task_id": session_context.task_id.clone().unwrap_or_default(),
        "status": session_status,
        "quality_status": quality_status,
        "privacy_level": "restricted",
        "upload_policy": resolve_first_non_empty([
            session_context.upload_policy_mode.as_deref(),
            upload_policy.artifact_policy_mode.as_deref(),
            Some(state.config.upload_policy_mode_name()),
        ]).unwrap_or_else(|| "metadata_only".to_string()),
        "runtime_profile": resolve_first_non_empty([
            session_context.runtime_profile.as_deref(),
            upload_policy.runtime_profile.as_deref(),
            Some(state.config.runtime_profile_name()),
        ]).unwrap_or_default(),
        "raw_residency": resolve_first_non_empty([
            session_context.raw_residency.as_deref(),
            upload_policy.raw_residency_default.as_deref(),
            Some(state.config.raw_residency_default()),
        ]).unwrap_or_default(),
        "preview_residency": resolve_first_non_empty([
            session_context.preview_residency.as_deref(),
            upload_policy.preview_residency_default.as_deref(),
            Some(state.config.preview_residency_default()),
        ]).unwrap_or_default(),
        "ready_for_review": ready_for_review,
        "auto_quality_score": auto_quality_score,
        "review_risk_score": review_risk_score,
        "total_size_bytes": total_size_bytes,
        "artifact_count": loaded_manifest.manifest.artifacts.len(),
        "metadata": {
            "upload_manifest": loaded_manifest.raw_value.clone(),
            "local_quality_report_summary": {
                "status": local_quality.status.clone().unwrap_or_default(),
                "missing_artifacts": local_quality.missing_artifacts,
                "recommended_actions": local_quality.recommended_actions,
            },
            "session_context_summary": {
                "capture_device_id": session_context.capture_device_id,
                "operator_id": session_context.operator_id,
                "task_id": session_context.task_id,
                "task_ids": session_context.task_ids,
            },
            "upload_contract": {
                "source": "edge_orchestrator_control_plane_uploader",
                "mode": "session_artifact_receipt",
            }
        }
    });
    let response =
        control_plane_request(state, "POST", control_base_url, "/v1/edge/sessions/upsert")
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("控制面 upsert session 请求失败: {e}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| format!("读取控制面 upsert session 响应失败: {e}"))?;
        return Err(format!(
            "控制面 upsert session 响应失败: status={} body={}",
            status.as_u16(),
            summarize_response_body(&body)
        ));
    }
    Ok(())
}

fn should_attempt_upload(
    entry: &UploadQueueEntry,
    max_retry_count: u32,
    uploading_stale_ms: u64,
) -> bool {
    if !entry.exists {
        return false;
    }
    match entry.status.as_str() {
        "queued" => true,
        "failed" => entry.retry_count < max_retry_count,
        "uploading" => entry
            .last_receipt_unix_ms
            .is_some_and(|ts| now_unix_ms().saturating_sub(ts) >= uploading_stale_ms),
        _ => false,
    }
}

async fn upload_entry_via_control_plane(
    state: &AppState,
    control_base_url: &str,
    base_dir: &Path,
    queue: &UploadQueueFile,
    loaded_manifest: &LoadedUploadManifest,
    entry: &UploadQueueEntry,
) -> Result<(), String> {
    let artifact_path = base_dir.join(&entry.relpath);
    if !tokio::fs::try_exists(&artifact_path)
        .await
        .map_err(|e| format!("检查上传资源失败: {} ({e})", artifact_path.display()))?
    {
        append_failed_receipt(
            state,
            base_dir,
            queue,
            entry,
            format!("上传资源缺失: {}", artifact_path.display()),
        )
        .await?;
        return Ok(());
    }

    upload_queue::append_upload_receipt_and_refresh(
        base_dir,
        UploadReceiptInput {
            trip_id: queue.trip_id.clone(),
            session_id: queue.session_id.clone(),
            asset_id: entry.asset_id.clone(),
            status: "uploading".to_string(),
            receipt_source: state.config.crowd_upload_receipt_source.clone(),
            remote_object_key: None,
            remote_upload_id: None,
            last_error: None,
        },
    )
    .await?;

    let prepared = match prepare_upload_payload(&artifact_path, entry).await {
        Ok(prepared) => prepared,
        Err(error) => {
            append_failed_receipt(
                state,
                base_dir,
                queue,
                entry,
                format!("准备上传资源失败: {} ({error})", artifact_path.display()),
            )
            .await?;
            return Ok(());
        }
    };

    let upload_target = match declare_control_plane_artifact(
        state,
        control_base_url,
        queue,
        loaded_manifest,
        entry,
    )
    .await
    {
        Ok(upload_target) => upload_target,
        Err(error) => {
            append_failed_receipt(state, base_dir, queue, entry, error).await?;
            return Ok(());
        }
    };

    let upload_result =
        send_transport_upload_request(state, queue, entry, &prepared, &upload_target).await;
    let remote_response = match upload_result {
        Ok(response) => response,
        Err(error) => {
            append_failed_receipt(state, base_dir, queue, entry, error).await?;
            return Ok(());
        }
    };

    let storage_key = if !upload_target.storage_key.trim().is_empty() {
        upload_target.storage_key.clone()
    } else {
        remote_response
            .remote_object_key
            .clone()
            .unwrap_or_default()
    };
    let receipt_metadata = json!({
        "remote_upload_id": remote_response.remote_upload_id.clone(),
        "upload_encoding": prepared.upload_encoding,
        "source_byte_size": entry.byte_size,
    });
    if let Err(error) = post_control_plane_receipt(
        state,
        control_base_url,
        queue,
        entry,
        prepared.bytes.len() as u64,
        &storage_key,
        receipt_metadata,
    )
    .await
    {
        append_failed_receipt(state, base_dir, queue, entry, error).await?;
        return Ok(());
    }

    upload_queue::append_upload_receipt_and_refresh(
        base_dir,
        UploadReceiptInput {
            trip_id: queue.trip_id.clone(),
            session_id: queue.session_id.clone(),
            asset_id: entry.asset_id.clone(),
            status: "acked".to_string(),
            receipt_source: state.config.crowd_upload_receipt_source.clone(),
            remote_object_key: Some(storage_key),
            remote_upload_id: remote_response.remote_upload_id,
            last_error: remote_response.error.and_then(|error| error.message),
        },
    )
    .await?;
    metrics::counter!("crowd_upload_asset_acked_count").increment(1);
    Ok(())
}

async fn upload_entry_direct(
    state: &AppState,
    artifact_url: &str,
    base_dir: &Path,
    queue: &UploadQueueFile,
    entry: &UploadQueueEntry,
) -> Result<(), String> {
    let artifact_path = base_dir.join(&entry.relpath);
    if !tokio::fs::try_exists(&artifact_path)
        .await
        .map_err(|e| format!("检查上传资源失败: {} ({e})", artifact_path.display()))?
    {
        append_failed_receipt(
            state,
            base_dir,
            queue,
            entry,
            format!("上传资源缺失: {}", artifact_path.display()),
        )
        .await?;
        return Ok(());
    }

    upload_queue::append_upload_receipt_and_refresh(
        base_dir,
        UploadReceiptInput {
            trip_id: queue.trip_id.clone(),
            session_id: queue.session_id.clone(),
            asset_id: entry.asset_id.clone(),
            status: "uploading".to_string(),
            receipt_source: state.config.crowd_upload_receipt_source.clone(),
            remote_object_key: None,
            remote_upload_id: None,
            last_error: None,
        },
    )
    .await?;

    let prepared = match prepare_upload_payload(&artifact_path, entry).await {
        Ok(prepared) => prepared,
        Err(error) => {
            append_failed_receipt(
                state,
                base_dir,
                queue,
                entry,
                format!("准备上传资源失败: {} ({error})", artifact_path.display()),
            )
            .await?;
            return Ok(());
        }
    };

    let mut request = state
        .http_client
        .post(artifact_url)
        .header("content-type", prepared.content_type)
        .header("x-chek-trip-id", &queue.trip_id)
        .header("x-chek-session-id", &queue.session_id)
        .header("x-chek-asset-id", &entry.asset_id)
        .header("x-chek-relpath", &entry.relpath)
        .header("x-chek-kind", &entry.kind)
        .header("x-chek-category", &entry.category)
        .header("x-chek-required", entry.required.to_string())
        .header("x-chek-byte-size", prepared.bytes.len().to_string())
        .header("x-chek-source-byte-size", entry.byte_size.to_string())
        .header("x-chek-retry-count", entry.retry_count.to_string())
        .body(prepared.bytes);
    if let Some(upload_encoding) = prepared.upload_encoding {
        request = request.header("x-chek-upload-encoding", upload_encoding);
    }
    if let Some(token) = state.config.crowd_upload_token.as_deref() {
        request = request.bearer_auth(token);
    }

    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            append_failed_receipt(
                state,
                base_dir,
                queue,
                entry,
                format!("上传请求失败: {error}"),
            )
            .await?;
            return Ok(());
        }
    };

    let status = response.status();
    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| format!("读取上传响应失败: {e}"))?;
    if !status.is_success() {
        append_failed_receipt(
            state,
            base_dir,
            queue,
            entry,
            format!(
                "上传响应失败: status={} body={}",
                status.as_u16(),
                summarize_response_body(&response_bytes)
            ),
        )
        .await?;
        return Ok(());
    }

    let parsed = serde_json::from_slice::<RemoteUploadResponse>(&response_bytes).ok();
    let last_error = parsed
        .as_ref()
        .and_then(|body| body.error.as_ref())
        .and_then(|error| error.message.clone());

    upload_queue::append_upload_receipt_and_refresh(
        base_dir,
        UploadReceiptInput {
            trip_id: queue.trip_id.clone(),
            session_id: queue.session_id.clone(),
            asset_id: entry.asset_id.clone(),
            status: "acked".to_string(),
            receipt_source: state.config.crowd_upload_receipt_source.clone(),
            remote_object_key: parsed
                .as_ref()
                .and_then(|body| body.remote_object_key.clone()),
            remote_upload_id: parsed
                .as_ref()
                .and_then(|body| body.remote_upload_id.clone()),
            last_error,
        },
    )
    .await?;
    metrics::counter!("crowd_upload_asset_acked_count").increment(1);
    Ok(())
}

async fn declare_control_plane_artifact(
    state: &AppState,
    control_base_url: &str,
    queue: &UploadQueueFile,
    loaded_manifest: &LoadedUploadManifest,
    entry: &UploadQueueEntry,
) -> Result<ControlPlaneUploadTarget, String> {
    let manifest_artifact = loaded_manifest
        .by_asset_id
        .get(&entry.asset_id)
        .ok_or_else(|| format!("upload_manifest 缺少 asset_id={}", entry.asset_id))?;
    let preview_status = if entry.category == "preview_derivative" && entry.exists {
        "preview_ready"
    } else {
        "not_ready"
    };
    let payload = json!({
        "asset_id": entry.asset_id,
        "relpath": entry.relpath,
        "kind": manifest_artifact.kind,
        "category": manifest_artifact.category,
        "required": manifest_artifact.required,
        "status": if entry.status == "acked" { "stored" } else { "declared" },
        "residency_status": manifest_artifact
            .residency
            .clone()
            .unwrap_or_else(|| "cloud_mirrored".to_string()),
        "preview_status": preview_status,
        "delivery_status": "awaiting_entitlement",
        "declared_size_bytes": manifest_artifact.byte_size,
        "stored_size_bytes": if entry.status == "acked" { manifest_artifact.byte_size } else { 0 },
        "metadata": {
            "line_count": manifest_artifact.line_count,
            "exists": manifest_artifact.exists,
            "upload_state": manifest_artifact.upload_state,
            "source_relpath": manifest_artifact.relpath,
        }
    });
    let response = control_plane_request(
        state,
        "POST",
        control_base_url,
        &format!("/v1/edge/sessions/{}/artifacts", queue.session_id),
    )
    .json(&payload)
    .send()
    .await
    .map_err(|e| format!("控制面声明 artifact 请求失败: {e}"))?;
    let status = response.status();
    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| format!("读取控制面声明 artifact 响应失败: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "控制面声明 artifact 响应失败: status={} body={}",
            status.as_u16(),
            summarize_response_body(&response_bytes)
        ));
    }
    let parsed = serde_json::from_slice::<ControlPlaneArtifactResponse>(&response_bytes)
        .map_err(|e| format!("解析控制面 artifact upload_target 失败: {e}"))?;
    Ok(parsed.upload_target)
}

async fn send_transport_upload_request(
    state: &AppState,
    queue: &UploadQueueFile,
    entry: &UploadQueueEntry,
    prepared: &PreparedUploadPayload,
    upload_target: &ControlPlaneUploadTarget,
) -> Result<RemoteUploadResponse, String> {
    let transport = &upload_target.transport;
    let upload_url = if transport.url.trim().is_empty() {
        state.config.crowd_upload_artifact_url.trim().to_string()
    } else {
        transport.url.trim().to_string()
    };
    if upload_url.is_empty() {
        return Err(
            "control plane transport.url 与 EDGE_CROWD_UPLOAD_ARTIFACT_URL 均为空".to_string(),
        );
    }

    let method = reqwest::Method::from_bytes(transport.method.trim().to_uppercase().as_bytes())
        .map_err(|e| format!("解析 transport.method 失败: {e}"))?;

    let mut headers = transport.headers.clone().unwrap_or_default();
    headers
        .entry("X-Chek-Trip-Id".to_string())
        .or_insert_with(|| queue.trip_id.clone());
    headers
        .entry("X-Chek-Session-Id".to_string())
        .or_insert_with(|| queue.session_id.clone());
    headers
        .entry("X-Chek-Asset-Id".to_string())
        .or_insert_with(|| entry.asset_id.clone());
    headers
        .entry("X-Chek-Relpath".to_string())
        .or_insert_with(|| entry.relpath.clone());
    headers
        .entry("X-Chek-Kind".to_string())
        .or_insert_with(|| entry.kind.clone());
    headers
        .entry("X-Chek-Category".to_string())
        .or_insert_with(|| entry.category.clone());
    headers
        .entry("X-Chek-Required".to_string())
        .or_insert_with(|| entry.required.to_string());
    headers
        .entry("X-Chek-Byte-Size".to_string())
        .or_insert_with(|| prepared.bytes.len().to_string());
    headers
        .entry("X-Chek-Source-Byte-Size".to_string())
        .or_insert_with(|| entry.byte_size.to_string());
    headers
        .entry("X-Chek-Retry-Count".to_string())
        .or_insert_with(|| entry.retry_count.to_string());
    if !upload_target.storage_key.trim().is_empty() {
        let header_name = transport
            .storage_key_header
            .clone()
            .unwrap_or_else(|| "X-Chek-Storage-Key".to_string());
        headers
            .entry(header_name)
            .or_insert_with(|| upload_target.storage_key.clone());
    }

    let scope_token = state
        .config
        .crowd_upload_scope_token
        .as_deref()
        .filter(|value| !value.is_empty());
    let bearer_token = state
        .config
        .crowd_upload_token
        .as_deref()
        .filter(|value| !value.is_empty());

    let request = if transport.mode.trim() == "multipart_post" || transport.supports_multipart {
        let file_field = transport
            .file_field
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("file");
        let metadata_field = transport
            .metadata_field
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("metadata");
        let metadata = json!({
            "trip_id": queue.trip_id,
            "session_id": queue.session_id,
            "asset_id": entry.asset_id,
            "relpath": entry.relpath,
            "kind": entry.kind,
            "category": entry.category,
            "required": entry.required,
            "storage_key": upload_target.storage_key,
            "upload_encoding": prepared.upload_encoding,
        });
        let part = reqwest::multipart::Part::bytes(prepared.bytes.clone())
            .file_name(upload_file_name(entry, prepared))
            .mime_str(prepared.content_type)
            .map_err(|e| format!("构造 multipart 上传 part 失败: {e}"))?;
        let form = reqwest::multipart::Form::new()
            .part(file_field.to_string(), part)
            .text(metadata_field.to_string(), metadata.to_string());
        let request = state
            .http_client
            .request(method, &upload_url)
            .multipart(form);
        let request = apply_transport_headers(request, &headers, prepared.upload_encoding)?;
        apply_transport_auth(request, transport, scope_token, bearer_token)?
    } else {
        let request = state
            .http_client
            .request(method, &upload_url)
            .header("content-type", prepared.content_type)
            .body(prepared.bytes.clone());
        let request = apply_transport_headers(request, &headers, prepared.upload_encoding)?;
        apply_transport_auth(request, transport, scope_token, bearer_token)?
    };

    let response = request
        .send()
        .await
        .map_err(|e| format!("按 transport 上传 artifact 失败: {e}"))?;
    let status = response.status();
    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| format!("读取 transport 上传响应失败: {e}"))?;
    if !status.is_success() {
        return Err(format!(
            "transport 上传响应失败: status={} body={}",
            status.as_u16(),
            summarize_response_body(&response_bytes)
        ));
    }
    Ok(
        serde_json::from_slice::<RemoteUploadResponse>(&response_bytes).unwrap_or(
            RemoteUploadResponse {
                remote_object_key: Some(upload_target.storage_key.clone()),
                remote_upload_id: None,
                error: None,
            },
        ),
    )
}

async fn post_control_plane_receipt(
    state: &AppState,
    control_base_url: &str,
    queue: &UploadQueueFile,
    entry: &UploadQueueEntry,
    stored_size_bytes: u64,
    storage_key: &str,
    metadata: Value,
) -> Result<(), String> {
    let payload = json!({
        "asset_id": entry.asset_id,
        "status": "acked",
        "stored_size_bytes": stored_size_bytes,
        "storage_key": storage_key,
        "metadata": metadata,
    });
    let response = control_plane_request(
        state,
        "POST",
        control_base_url,
        &format!("/v1/edge/sessions/{}/receipts", queue.session_id),
    )
    .json(&payload)
    .send()
    .await
    .map_err(|e| format!("控制面写入 receipt 请求失败: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response
            .bytes()
            .await
            .map_err(|e| format!("读取控制面 receipt 响应失败: {e}"))?;
        return Err(format!(
            "控制面写入 receipt 响应失败: status={} body={}",
            status.as_u16(),
            summarize_response_body(&body)
        ));
    }
    Ok(())
}

async fn prepare_upload_payload(
    artifact_path: &Path,
    entry: &UploadQueueEntry,
) -> Result<PreparedUploadPayload, String> {
    match entry.kind.as_str() {
        "directory" => {
            let artifact_path = artifact_path.to_path_buf();
            let relpath = entry.relpath.clone();
            tokio::task::spawn_blocking(move || {
                package_directory_as_tar_gz(&artifact_path, &relpath)
            })
            .await
            .map_err(|e| format!("目录打包任务失败: {e}"))?
        }
        _ => tokio::fs::read(artifact_path)
            .await
            .map(|bytes| PreparedUploadPayload {
                bytes,
                content_type: "application/octet-stream",
                upload_encoding: None,
            })
            .map_err(|e| format!("读取文件失败: {e}")),
    }
}

fn package_directory_as_tar_gz(
    artifact_path: &Path,
    relpath: &str,
) -> Result<PreparedUploadPayload, String> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    {
        let mut builder = Builder::new(&mut encoder);
        let root = artifact_path;
        let root_display = relpath.trim_matches('/').to_string();
        for entry in WalkDir::new(root).sort_by_file_name() {
            let entry = entry.map_err(|e| format!("遍历目录失败: {e}"))?;
            let path = entry.path();
            let relative = path
                .strip_prefix(root)
                .map_err(|e| format!("计算目录相对路径失败: {e}"))?;
            let archive_path = if relative.as_os_str().is_empty() {
                std::path::PathBuf::from(&root_display)
            } else {
                std::path::PathBuf::from(&root_display).join(relative)
            };
            if entry.file_type().is_dir() {
                builder
                    .append_dir(archive_path, path)
                    .map_err(|e| format!("写入目录归档失败: {e}"))?;
            } else if entry.file_type().is_file() {
                builder
                    .append_path_with_name(path, archive_path)
                    .map_err(|e| format!("写入文件归档失败: {e}"))?;
            }
        }
        builder
            .finish()
            .map_err(|e| format!("完成目录归档失败: {e}"))?;
    }
    let bytes = encoder
        .finish()
        .map_err(|e| format!("完成 gzip 压缩失败: {e}"))?;
    Ok(PreparedUploadPayload {
        bytes,
        content_type: "application/gzip",
        upload_encoding: Some("tar_gz_directory"),
    })
}

fn control_plane_request(
    state: &AppState,
    method: &str,
    control_base_url: &str,
    path: &str,
) -> reqwest::RequestBuilder {
    let base = control_base_url.trim_end_matches('/');
    let path = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    let url = format!("{base}{path}");
    let method = reqwest::Method::from_bytes(method.as_bytes()).unwrap_or(reqwest::Method::POST);
    let mut request = state.http_client.request(method, url);
    if let Some(token) = state
        .config
        .crowd_upload_token
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        request = request.bearer_auth(token);
    }
    request
}

fn apply_transport_headers(
    mut request: reqwest::RequestBuilder,
    headers: &HashMap<String, String>,
    upload_encoding: Option<&str>,
) -> Result<reqwest::RequestBuilder, String> {
    for (key, value) in headers {
        if key.trim().is_empty() || value.trim().is_empty() {
            continue;
        }
        request = request.header(key, value);
    }
    if let Some(upload_encoding) = upload_encoding {
        request = request.header("x-chek-upload-encoding", upload_encoding);
    }
    Ok(request)
}

fn apply_transport_auth(
    mut request: reqwest::RequestBuilder,
    transport: &ControlPlaneUploadTransport,
    scope_token: Option<&str>,
    bearer_token: Option<&str>,
) -> Result<reqwest::RequestBuilder, String> {
    if transport.scope_token_required {
        let token = scope_token
            .or(bearer_token)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                "transport 需要 scope token，但 EDGE_CROWD_UPLOAD_SCOPE_TOKEN 未配置".to_string()
            })?;
        let header_name = transport
            .scope_token_header
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("X-Chek-Upload-Scope-Token");
        request = request.header(header_name, token);
    }
    if matches!(
        transport
            .auth_strategy
            .as_deref()
            .map(str::trim)
            .unwrap_or("bearer"),
        "" | "bearer"
    ) {
        if let Some(token) = bearer_token.filter(|value| !value.trim().is_empty()) {
            request = request.bearer_auth(token);
        }
    }
    Ok(request)
}

fn upload_file_name(entry: &UploadQueueEntry, prepared: &PreparedUploadPayload) -> String {
    let base = Path::new(&entry.relpath)
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("artifact.bin");
    if prepared.upload_encoding == Some("tar_gz_directory") && !base.ends_with(".tar.gz") {
        format!("{base}.tar.gz")
    } else {
        base.to_string()
    }
}

fn normalize_non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

fn resolve_first_non_empty<'a>(
    candidates: impl IntoIterator<Item = Option<&'a str>>,
) -> Option<String> {
    for candidate in candidates {
        if let Some(value) = normalize_non_empty(candidate) {
            return Some(value);
        }
    }
    None
}

async fn append_failed_receipt(
    state: &AppState,
    base_dir: &Path,
    queue: &UploadQueueFile,
    entry: &UploadQueueEntry,
    last_error: String,
) -> Result<(), String> {
    metrics::counter!("crowd_upload_asset_failed_count").increment(1);
    upload_queue::append_upload_receipt_and_refresh(
        base_dir,
        UploadReceiptInput {
            trip_id: queue.trip_id.clone(),
            session_id: queue.session_id.clone(),
            asset_id: entry.asset_id.clone(),
            status: "failed".to_string(),
            receipt_source: state.config.crowd_upload_receipt_source.clone(),
            remote_object_key: None,
            remote_upload_id: None,
            last_error: Some(last_error),
        },
    )
    .await
}

fn summarize_response_body(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "<empty>".to_string();
    }
    let body = String::from_utf8_lossy(bytes).trim().to_string();
    if body.len() > 240 {
        format!("{}...", &body[..240])
    } else {
        body
    }
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::GzDecoder;
    use tar::Archive;

    use super::{package_directory_as_tar_gz, should_attempt_upload, UploadQueueEntry};

    #[test]
    fn should_retry_failed_entries_before_reaching_limit() {
        let entry = UploadQueueEntry {
            asset_id: "artifact-1".to_string(),
            relpath: "raw/demo.jsonl".to_string(),
            kind: "jsonl".to_string(),
            category: "raw".to_string(),
            required: true,
            exists: true,
            byte_size: 128,
            status: "failed".to_string(),
            retry_count: 2,
            last_receipt_unix_ms: Some(1),
        };
        assert!(should_attempt_upload(&entry, 3, 60_000));
        assert!(!should_attempt_upload(&entry, 2, 60_000));
    }

    #[test]
    fn should_package_directory_entries_as_tar_gz() {
        let temp_root =
            std::env::temp_dir().join(format!("edge-uploader-dir-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_root);
        std::fs::create_dir_all(temp_root.join("nested")).unwrap();
        std::fs::write(temp_root.join("nested").join("demo.txt"), b"hello").unwrap();

        let prepared = package_directory_as_tar_gz(&temp_root, "preview/keyframes").unwrap();
        assert_eq!(prepared.content_type, "application/gzip");
        assert_eq!(prepared.upload_encoding, Some("tar_gz_directory"));

        let mut archive = Archive::new(GzDecoder::new(prepared.bytes.as_slice()));
        let mut found = false;
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().to_string_lossy().to_string();
            if path.ends_with("preview/keyframes/nested/demo.txt") {
                let mut content = String::new();
                entry.read_to_string(&mut content).unwrap();
                assert_eq!(content, "hello");
                found = true;
            }
        }
        assert!(
            found,
            "expected packaged archive to include nested/demo.txt"
        );
        let _ = std::fs::remove_dir_all(&temp_root);
    }
}
