use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sysinfo::Disks;
use tokio::io::AsyncWriteExt;
use walkdir::WalkDir;

use crate::path_safety;
use crate::AppState;

const PRESSURE_WARNING_FREE_RATIO: f64 = 0.20;
const PRESSURE_CRITICAL_FREE_RATIO: f64 = 0.10;
const PROTECTED_POOL_BUDGET_RATIO: f64 = 0.20;
const MIN_PROTECTED_POOL_BUDGET_BYTES: u64 = 1_073_741_824;
const DEFAULT_SWEEP_MAX_SESSIONS: usize = 1;
const MAX_SWEEP_MAX_SESSIONS: usize = 100;

pub fn public_router(state: AppState) -> Router {
    Router::new()
        .route("/storage/status", get(get_storage_status))
        .with_state(state)
}

pub fn protected_router(state: AppState) -> Router {
    Router::new()
        .route("/storage/sessions", get(get_storage_sessions))
        .route("/storage/sweeps/dry-run", post(post_storage_sweep_dry_run))
        .route("/storage/sweeps/apply", post(post_storage_sweep_apply))
        .route(
            "/storage/consumption-receipt",
            post(post_storage_consumption_receipt),
        )
        .with_state(state)
}

#[derive(Debug, Default, Deserialize)]
struct LocalQualityReportSummary {
    status: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct UploadManifestArtifact {
    #[serde(default)]
    byte_size: u64,
    #[serde(default)]
    exists: bool,
}

#[derive(Debug, Default, Deserialize)]
struct UploadManifestFile {
    #[serde(default)]
    trip_id: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    ready_for_upload: bool,
    #[serde(default)]
    artifacts: Vec<UploadManifestArtifact>,
}

#[derive(Debug, Default, Deserialize)]
struct UploadQueueEntry {
    #[serde(default)]
    status: String,
}

#[derive(Debug, Default, Deserialize)]
struct UploadQueueFile {
    #[serde(default)]
    ready_for_upload: bool,
    #[serde(default)]
    entries: Vec<UploadQueueEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredConsumptionReceipt {
    session_id: String,
    signal_kind: String,
    note: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recorded_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    generated_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    consumed_at: Option<u64>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CleanupStateFile {
    last_run_at: Option<u64>,
    #[serde(default)]
    last_reclaimed_bytes: u64,
    #[serde(default)]
    last_error: Option<String>,
}

#[derive(Debug, Clone)]
struct SessionStorageInventory {
    session_id: String,
    #[allow(dead_code)]
    trip_id: String,
    base_dir: PathBuf,
    byte_size: u64,
    modified_unix_ms: u64,
    qa_status: String,
    upload_queue_status: String,
    last_cleanup_skipped_reason: String,
    local_storage_class: String,
    storage_pool: String,
}

#[derive(Debug, Deserialize)]
struct StorageConsumptionReceiptRequest {
    session_id: String,
    signal_kind: String,
    #[serde(default)]
    note: String,
}

#[derive(Debug, Default, Deserialize)]
struct StorageSweepRequest {
    #[serde(default)]
    target_reclaim_bytes: Option<u64>,
    #[serde(default)]
    max_sessions: Option<usize>,
}

async fn get_storage_status(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let inventory = load_storage_inventory(&state)
        .await
        .map_err(internal_error)?;
    let cleanup_state = load_cleanup_state(&state).await.unwrap_or_default();
    let active_session_id = active_session_id(&state);
    let rolling_sessions = inventory
        .iter()
        .filter(|session| session.storage_pool == "rolling")
        .collect::<Vec<_>>();
    let protected_sessions = inventory
        .iter()
        .filter(|session| session.storage_pool == "protected")
        .collect::<Vec<_>>();
    let rolling_used_bytes = rolling_sessions
        .iter()
        .map(|session| session.byte_size)
        .sum::<u64>();
    let protected_used_bytes = protected_sessions
        .iter()
        .map(|session| session.byte_size)
        .sum::<u64>();
    let cleanup_candidates = cleanup_candidates(&inventory, &StorageSweepRequest::default());
    let evictable_bytes = cleanup_candidates
        .iter()
        .map(|session| session.byte_size)
        .sum::<u64>();
    let current_blockers = inventory
        .iter()
        .filter_map(|session| {
            if session.last_cleanup_skipped_reason.is_empty() {
                None
            } else {
                Some(format!(
                    "{}:{}",
                    session.last_cleanup_skipped_reason, session.session_id
                ))
            }
        })
        .collect::<Vec<_>>();
    let (disk_total_bytes, disk_free_bytes) =
        disk_capacity_for_path(Path::new(&state.config.data_dir));
    let disk_used_bytes = disk_total_bytes.saturating_sub(disk_free_bytes);
    let pressure_level = disk_pressure_level(disk_total_bytes, disk_free_bytes);
    let protected_budget_bytes = protected_pool_budget_bytes(disk_total_bytes);

    Ok(Json(serde_json::json!({
        "pressure_level": pressure_level,
        "disk_used_bytes": disk_used_bytes,
        "disk_free_bytes": disk_free_bytes,
        "disk_total_bytes": disk_total_bytes,
        "rolling_pool": {
            "used_bytes": rolling_used_bytes,
            "evictable_bytes": evictable_bytes,
            "session_count": rolling_sessions.len(),
            "status": rolling_pool_status(&pressure_level, rolling_used_bytes),
        },
        "protected_pool": {
            "used_bytes": protected_used_bytes,
            "budget_bytes": protected_budget_bytes,
            "session_count": protected_sessions.len(),
            "status": protected_pool_status(protected_used_bytes, protected_budget_bytes),
        },
        "cleanup": {
            "auto_sweep_enabled": true,
            "last_run_at": cleanup_state.last_run_at,
            "last_reclaimed_bytes": cleanup_state.last_reclaimed_bytes,
            "dry_run_selected_sessions": cleanup_candidates.iter().map(|session| session.session_id.clone()).collect::<Vec<_>>(),
            "current_blockers": current_blockers,
            "last_error": cleanup_state.last_error,
        },
        "session_summary": {
            "active_session_id": active_session_id,
        }
    })))
}

async fn get_storage_sessions(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let inventory = load_storage_inventory(&state)
        .await
        .map_err(internal_error)?;
    Ok(Json(serde_json::json!({
        "sessions": inventory.iter().map(|session| serde_json::json!({
            "session_id": session.session_id,
            "local_storage_class": session.local_storage_class,
            "storage_pool": session.storage_pool,
            "byte_size": session.byte_size,
            "qa_status": session.qa_status,
            "upload_queue_status": session.upload_queue_status,
            "last_cleanup_skipped_reason": if session.last_cleanup_skipped_reason.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::String(session.last_cleanup_skipped_reason.clone())
            },
        })).collect::<Vec<_>>()
    })))
}

async fn post_storage_sweep_dry_run(
    State(state): State<AppState>,
    Json(body): Json<StorageSweepRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let inventory = load_storage_inventory(&state)
        .await
        .map_err(internal_error)?;
    let candidates = cleanup_candidates(&inventory, &body);
    Ok(Json(serde_json::json!({
        "selected_session_count": candidates.len(),
        "selected_bytes": candidates.iter().map(|session| session.byte_size).sum::<u64>(),
        "selected_session_ids": candidates.iter().map(|session| session.session_id.clone()).collect::<Vec<_>>(),
        "policy": {
            "max_sessions": sweep_max_sessions(&body),
            "target_reclaim_bytes": body.target_reclaim_bytes.unwrap_or(0),
            "requires_upload_queue_status": "acked",
        },
    })))
}

async fn post_storage_sweep_apply(
    State(state): State<AppState>,
    Json(body): Json<StorageSweepRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let inventory = load_storage_inventory(&state)
        .await
        .map_err(internal_error)?;
    let candidates = cleanup_candidates(&inventory, &body);
    let selected_session_count = candidates.len();
    let selected_session_ids = candidates
        .iter()
        .map(|session| session.session_id.clone())
        .collect::<Vec<_>>();
    let mut applied_reclaimed_bytes: u64 = 0;
    let mut last_error_messages: Vec<String> = Vec::new();

    for session in &candidates {
        match tokio::fs::remove_dir_all(&session.base_dir).await {
            Ok(()) => {
                applied_reclaimed_bytes = applied_reclaimed_bytes.saturating_add(session.byte_size);
            }
            Err(error) => {
                let message = format!(
                    "删除 session 目录失败: {} ({error})",
                    session.base_dir.display()
                );
                last_error_messages.push(message);
            }
        }
    }

    let cleanup_state = CleanupStateFile {
        last_run_at: Some(now_unix_ms()),
        last_reclaimed_bytes: applied_reclaimed_bytes,
        last_error: if last_error_messages.is_empty() {
            None
        } else {
            Some(last_error_messages.join(" | "))
        },
    };
    write_cleanup_state(&state, &cleanup_state)
        .await
        .map_err(internal_error)?;

    Ok(Json(serde_json::json!({
        "selected_session_count": selected_session_count,
        "applied_reclaimed_bytes": applied_reclaimed_bytes,
        "selected_session_ids": selected_session_ids,
        "policy": {
            "max_sessions": sweep_max_sessions(&body),
            "target_reclaim_bytes": body.target_reclaim_bytes.unwrap_or(0),
            "requires_upload_queue_status": "acked",
        },
    })))
}

async fn post_storage_consumption_receipt(
    State(state): State<AppState>,
    Json(req): Json<StorageConsumptionReceiptRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if req.session_id.trim().is_empty() || req.signal_kind.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err(
                "missing_required_fields",
                "session_id / signal_kind 不能为空".to_string(),
            )),
        ));
    }

    let session_id =
        path_safety::validate_path_component(&req.session_id, "session_id").map_err(|message| {
            (
                StatusCode::BAD_REQUEST,
                Json(err("invalid_session_id", message)),
            )
        })?;
    let session_dir = path_safety::session_base_dir(Path::new(&state.config.data_dir), &session_id)
        .map_err(|message| {
            (
                StatusCode::BAD_REQUEST,
                Json(err("invalid_session_id", message)),
            )
        })?;
    let exists = tokio::fs::try_exists(&session_dir).await.map_err(|error| {
        internal_error(format!(
            "检查 session 目录失败: {} ({error})",
            session_dir.display()
        ))
    })?;
    if !exists {
        return Err((
            StatusCode::NOT_FOUND,
            Json(err(
                "session_not_found",
                format!("session 不存在: {}", session_id),
            )),
        ));
    }

    let receipt = StoredConsumptionReceipt {
        session_id,
        signal_kind: req.signal_kind.trim().to_string(),
        note: req.note.trim().to_string(),
        recorded_unix_ms: Some(now_unix_ms()),
        generated_unix_ms: None,
        consumed_at: None,
    };
    append_consumption_receipt(&state, &receipt)
        .await
        .map_err(internal_error)?;

    let signal_kind = normalize_non_empty(&receipt.signal_kind).unwrap_or_default();
    let protected = is_protected_signal_kind(&signal_kind);

    Ok(Json(serde_json::json!({
        "ok": true,
        "session": {
            "session_id": receipt.session_id,
            "storage_pool": if protected { "protected" } else { "rolling" },
            "local_storage_class": if protected {
                signal_kind.clone()
            } else {
                "rolling_candidate".to_string()
            },
            "last_cleanup_skipped_reason": if protected {
                signal_kind.clone()
            } else {
                String::new()
            },
        },
        "receipt": receipt,
    })))
}

async fn load_storage_inventory(state: &AppState) -> Result<Vec<SessionStorageInventory>, String> {
    let session_dirs = collect_session_dirs(state).await?;
    let latest_receipts = load_latest_consumption_receipts(state).await?;
    let active_session_id = active_session_id(state);
    let mut inventory = Vec::new();

    for base_dir in session_dirs {
        let upload_manifest =
            load_optional_json::<UploadManifestFile>(&base_dir, "upload/upload_manifest.json")
                .await?;
        let upload_queue =
            load_optional_json::<UploadQueueFile>(&base_dir, "upload/upload_queue.json").await?;
        let local_quality = load_optional_json::<LocalQualityReportSummary>(
            &base_dir,
            "qa/local_quality_report.json",
        )
        .await?;
        let fallback_session_id = base_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_string();
        let session_id = upload_manifest
            .as_ref()
            .and_then(|manifest| normalize_non_empty(&manifest.session_id))
            .unwrap_or_else(|| fallback_session_id.clone());
        let trip_id = upload_manifest
            .as_ref()
            .and_then(|manifest| normalize_non_empty(&manifest.trip_id))
            .unwrap_or_else(|| session_id.clone());
        let byte_size = upload_manifest
            .as_ref()
            .map(upload_manifest_byte_size)
            .filter(|value| *value > 0)
            .unwrap_or_else(|| dir_size_bytes_sync(&base_dir));
        let modified_unix_ms = metadata_modified_unix_ms(&base_dir).await.unwrap_or(0);
        let qa_status = local_quality
            .as_ref()
            .and_then(|report| normalize_non_empty_option(report.status.as_ref()))
            .unwrap_or_else(|| "pending".to_string());
        let upload_queue_status = upload_queue
            .as_ref()
            .map(upload_queue_status)
            .unwrap_or_else(|| default_upload_queue_status(upload_manifest.as_ref()));
        let latest_receipt = latest_receipts.get(&session_id);
        let signal_kind = latest_receipt
            .and_then(|receipt| normalize_non_empty(&receipt.signal_kind))
            .unwrap_or_default();
        let protected_by_receipt = is_protected_signal_kind(&signal_kind);
        let is_active = !active_session_id.is_empty() && session_id == active_session_id;
        let (storage_pool, local_storage_class, last_cleanup_skipped_reason) = if is_active {
            (
                "protected".to_string(),
                if protected_by_receipt {
                    format!("active_{}", signal_kind)
                } else {
                    "active_session".to_string()
                },
                "active_session".to_string(),
            )
        } else if protected_by_receipt {
            (
                "protected".to_string(),
                signal_kind.clone(),
                signal_kind.clone(),
            )
        } else {
            (
                "rolling".to_string(),
                "rolling_candidate".to_string(),
                String::new(),
            )
        };

        inventory.push(SessionStorageInventory {
            session_id,
            trip_id,
            base_dir,
            byte_size,
            modified_unix_ms,
            qa_status,
            upload_queue_status,
            last_cleanup_skipped_reason,
            local_storage_class,
            storage_pool,
        });
    }

    inventory.sort_by(|left, right| {
        right
            .modified_unix_ms
            .cmp(&left.modified_unix_ms)
            .then_with(|| right.session_id.cmp(&left.session_id))
    });
    Ok(inventory)
}

fn cleanup_candidates<'a>(
    inventory: &'a [SessionStorageInventory],
    request: &StorageSweepRequest,
) -> Vec<&'a SessionStorageInventory> {
    let mut candidates = inventory
        .iter()
        .filter(|session| is_cleanup_candidate(session))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.modified_unix_ms
            .cmp(&right.modified_unix_ms)
            .then_with(|| left.session_id.cmp(&right.session_id))
    });
    let max_sessions = sweep_max_sessions(request);
    let target_reclaim_bytes = request.target_reclaim_bytes.unwrap_or(0);
    let mut selected = Vec::new();
    let mut selected_bytes = 0_u64;
    for session in candidates {
        if selected.len() >= max_sessions {
            break;
        }
        selected_bytes = selected_bytes.saturating_add(session.byte_size);
        selected.push(session);
        if target_reclaim_bytes > 0 && selected_bytes >= target_reclaim_bytes {
            break;
        }
    }
    selected
}

fn sweep_max_sessions(request: &StorageSweepRequest) -> usize {
    request
        .max_sessions
        .unwrap_or(DEFAULT_SWEEP_MAX_SESSIONS)
        .clamp(1, MAX_SWEEP_MAX_SESSIONS)
}

fn is_cleanup_candidate(session: &SessionStorageInventory) -> bool {
    session.storage_pool == "rolling" && session.upload_queue_status == "acked"
}

async fn collect_session_dirs(state: &AppState) -> Result<Vec<PathBuf>, String> {
    let session_root = canonical_data_child_dir(state, "session").await?;
    let exists = tokio::fs::try_exists(&session_root)
        .await
        .map_err(|error| {
            format!(
                "检查 session 目录失败: {} ({error})",
                session_root.display()
            )
        })?;
    if !exists {
        return Ok(Vec::new());
    }

    let mut entries = tokio::fs::read_dir(&session_root).await.map_err(|error| {
        format!(
            "读取 session 目录失败: {} ({error})",
            session_root.display()
        )
    })?;
    let mut session_dirs = Vec::new();
    while let Some(entry) = entries.next_entry().await.map_err(|error| {
        format!(
            "遍历 session 目录失败: {} ({error})",
            session_root.display()
        )
    })? {
        let file_type = entry.file_type().await.map_err(|error| {
            format!(
                "读取 session 目录类型失败: {} ({error})",
                entry.path().display()
            )
        })?;
        if file_type.is_dir() {
            let file_name = entry
                .file_name()
                .to_str()
                .ok_or_else(|| "session 目录名必须是 UTF-8".to_string())?
                .to_string();
            let safe_session_id = path_safety::validate_path_component(&file_name, "session_id")?;
            let candidate = session_root.join(safe_session_id);
            let canonical_candidate =
                tokio::fs::canonicalize(&candidate).await.map_err(|error| {
                    format!("解析 session 目录失败: {} ({error})", candidate.display())
                })?;
            if !canonical_candidate.starts_with(&session_root) {
                return Err(format!(
                    "session 目录越界: {} not under {}",
                    canonical_candidate.display(),
                    session_root.display()
                ));
            }
            session_dirs.push(canonical_candidate);
        }
    }
    session_dirs.sort();
    Ok(session_dirs)
}

async fn load_optional_json<T>(trusted_root: &Path, relpath: &str) -> Result<Option<T>, String>
where
    T: DeserializeOwned,
{
    let Some(path) = optional_child_file_path(trusted_root, relpath).await? else {
        return Ok(None);
    };
    let exists = tokio::fs::try_exists(&path)
        .await
        .map_err(|error| format!("检查文件是否存在失败: {} ({error})", path.display()))?;
    if !exists {
        return Ok(None);
    }
    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|error| format!("读取文件失败: {} ({error})", path.display()))?;
    match serde_json::from_str::<T>(&content) {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            tracing::warn!(
                error = %error,
                path = %path.display(),
                "storage compat: 忽略无法解析的 JSON 文件"
            );
            Ok(None)
        }
    }
}

fn upload_manifest_byte_size(manifest: &UploadManifestFile) -> u64 {
    manifest
        .artifacts
        .iter()
        .filter(|artifact| artifact.exists || artifact.byte_size > 0)
        .map(|artifact| artifact.byte_size)
        .sum()
}

fn upload_queue_status(queue: &UploadQueueFile) -> String {
    if queue.entries.iter().any(|entry| entry.status == "failed") {
        "failed".to_string()
    } else if !queue.ready_for_upload {
        "blocked".to_string()
    } else if queue.entries.iter().all(|entry| entry.status == "acked") && !queue.entries.is_empty()
    {
        "acked".to_string()
    } else if queue
        .entries
        .iter()
        .any(|entry| entry.status == "uploading")
    {
        "uploading".to_string()
    } else if queue.entries.iter().any(|entry| entry.status == "queued") {
        "queued".to_string()
    } else if queue
        .entries
        .iter()
        .any(|entry| entry.status == "pending_artifact")
    {
        "pending_artifact".to_string()
    } else if queue.entries.is_empty() {
        "queued".to_string()
    } else {
        "pending".to_string()
    }
}

fn default_upload_queue_status(upload_manifest: Option<&UploadManifestFile>) -> String {
    if upload_manifest
        .map(|manifest| manifest.ready_for_upload)
        .unwrap_or(false)
    {
        "queued".to_string()
    } else {
        "blocked".to_string()
    }
}

async fn metadata_modified_unix_ms(path: &Path) -> Option<u64> {
    let metadata = tokio::fs::metadata(path).await.ok()?;
    let modified = metadata.modified().ok()?;
    system_time_to_unix_ms(modified)
}

fn dir_size_bytes_sync(path: &Path) -> u64 {
    WalkDir::new(path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| entry.metadata().ok())
        .map(|metadata| metadata.len())
        .sum()
}

fn disk_capacity_for_path(path: &Path) -> (u64, u64) {
    let path = match std::fs::canonicalize(path) {
        Ok(canonical) => canonical,
        Err(_) => path.to_path_buf(),
    };
    tokio::task::block_in_place(|| {
        let disks = Disks::new_with_refreshed_list();
        let mut best_match: Option<(usize, u64, u64)> = None;
        for disk in disks.iter() {
            let mount_point = disk.mount_point();
            if !path.starts_with(mount_point) {
                continue;
            }
            let match_len = mount_point.as_os_str().len();
            let candidate = (match_len, disk.total_space(), disk.available_space());
            if best_match
                .as_ref()
                .map(|(best_len, _, _)| match_len > *best_len)
                .unwrap_or(true)
            {
                best_match = Some(candidate);
            }
        }
        best_match
            .map(|(_, total_space, available_space)| (total_space, available_space))
            .unwrap_or((0, 0))
    })
}

fn disk_pressure_level(total_bytes: u64, free_bytes: u64) -> &'static str {
    if total_bytes == 0 {
        return "unknown";
    }
    let free_ratio = free_bytes as f64 / total_bytes as f64;
    if free_ratio <= PRESSURE_CRITICAL_FREE_RATIO {
        "critical"
    } else if free_ratio <= PRESSURE_WARNING_FREE_RATIO {
        "warning"
    } else {
        "healthy"
    }
}

fn rolling_pool_status(pressure_level: &str, used_bytes: u64) -> &'static str {
    if used_bytes == 0 {
        "tracked"
    } else if pressure_level == "critical" {
        "over_hard_limit"
    } else if pressure_level == "warning" {
        "over_soft_limit"
    } else {
        "evictable_inventory_available"
    }
}

fn protected_pool_budget_bytes(total_bytes: u64) -> u64 {
    (((total_bytes as f64) * PROTECTED_POOL_BUDGET_RATIO) as u64)
        .max(MIN_PROTECTED_POOL_BUDGET_BYTES)
}

fn protected_pool_status(used_bytes: u64, budget_bytes: u64) -> &'static str {
    if budget_bytes > 0 && used_bytes > budget_bytes {
        "over_budget"
    } else {
        "tracked"
    }
}

fn active_session_id(state: &AppState) -> String {
    normalize_non_empty(&state.session.snapshot().session_id).unwrap_or_default()
}

async fn canonical_data_dir(state: &AppState) -> Result<PathBuf, String> {
    let data_dir = Path::new(&state.config.data_dir);
    let canonical = tokio::fs::canonicalize(data_dir)
        .await
        .map_err(|error| format!("解析 data_dir 失败: {} ({error})", data_dir.display()))?;
    path_safety::ensure_no_relative_escape(&canonical, "data_dir")?;
    Ok(canonical)
}

async fn canonical_data_child_dir(state: &AppState, child: &str) -> Result<PathBuf, String> {
    let data_dir = canonical_data_dir(state).await?;
    let safe_child = path_safety::validate_path_component(child, "data_child")?;
    Ok(data_dir.join(safe_child))
}

async fn optional_child_file_path(root: &Path, relpath: &str) -> Result<Option<PathBuf>, String> {
    let rel = path_safety::validate_relative_path(relpath, "storage relpath")?;
    let path = root.join(rel);
    let parent = path
        .parent()
        .ok_or_else(|| format!("文件路径缺少父目录: {}", path.display()))?;
    let parent_exists = tokio::fs::try_exists(parent)
        .await
        .map_err(|error| format!("检查父目录是否存在失败: {} ({error})", parent.display()))?;
    if !parent_exists {
        return Ok(None);
    }
    let canonical_root = match tokio::fs::canonicalize(root).await {
        Ok(path) => path,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => {
            return Err(format!("解析根目录失败: {} ({error})", root.display()));
        }
    };
    let canonical_parent = match tokio::fs::canonicalize(parent).await {
        Ok(path) => path,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => {
            return Err(format!("解析父目录失败: {} ({error})", parent.display()));
        }
    };
    if !canonical_parent.starts_with(&canonical_root) {
        return Err(format!(
            "文件父目录越界: {} not under {}",
            canonical_parent.display(),
            canonical_root.display()
        ));
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("文件路径缺少文件名: {}", path.display()))?;
    Ok(Some(canonical_parent.join(file_name)))
}

fn child_file_path(root: &Path, filename: &str) -> Result<PathBuf, String> {
    let safe_filename = path_safety::validate_path_component(filename, "filename")?;
    Ok(root.join(safe_filename))
}

async fn canonical_storage_state_dir(state: &AppState) -> Result<PathBuf, String> {
    canonical_data_child_dir(state, "storage").await
}

async fn load_latest_consumption_receipts(
    state: &AppState,
) -> Result<HashMap<String, StoredConsumptionReceipt>, String> {
    let storage_state_dir = canonical_storage_state_dir(state).await?;
    let Some(path) =
        optional_child_file_path(&storage_state_dir, "consumption_receipts.jsonl").await?
    else {
        return Ok(HashMap::new());
    };
    let exists = tokio::fs::try_exists(&path).await.map_err(|error| {
        format!(
            "检查 consumption receipts 失败: {} ({error})",
            path.display()
        )
    })?;
    if !exists {
        return Ok(HashMap::new());
    }
    let content = tokio::fs::read_to_string(&path).await.map_err(|error| {
        format!(
            "读取 consumption receipts 失败: {} ({error})",
            path.display()
        )
    })?;
    let mut receipts = HashMap::new();
    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<StoredConsumptionReceipt>(line) {
            Ok(receipt) => {
                receipts.insert(receipt.session_id.clone(), receipt);
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    line_index = index.saturating_add(1),
                    "storage compat: 忽略无法解析的 consumption receipt 行"
                );
                continue;
            }
        }
    }
    Ok(receipts)
}

async fn append_consumption_receipt(
    state: &AppState,
    receipt: &StoredConsumptionReceipt,
) -> Result<(), String> {
    let storage_state_dir = canonical_storage_state_dir(state).await?;
    tokio::fs::create_dir_all(&storage_state_dir)
        .await
        .map_err(|error| {
            format!(
                "创建 storage state 目录失败: {} ({error})",
                storage_state_dir.display()
            )
        })?;
    let canonical_storage_state_dir =
        tokio::fs::canonicalize(&storage_state_dir)
            .await
            .map_err(|error| {
                format!(
                    "解析 storage state 目录失败: {} ({error})",
                    storage_state_dir.display()
                )
            })?;
    let path = child_file_path(&canonical_storage_state_dir, "consumption_receipts.jsonl")?;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|error| {
            format!(
                "打开 consumption receipts 失败: {} ({error})",
                path.display()
            )
        })?;
    let mut line = serde_json::to_vec(receipt).map_err(|error| error.to_string())?;
    line.push(b'\n');
    file.write_all(&line).await.map_err(|error| {
        format!(
            "写入 consumption receipt 失败: {} ({error})",
            path.display()
        )
    })
}

async fn load_cleanup_state(state: &AppState) -> Result<CleanupStateFile, String> {
    let storage_state_dir = canonical_storage_state_dir(state).await?;
    Ok(
        load_optional_json::<CleanupStateFile>(&storage_state_dir, "cleanup_state.json")
            .await?
            .unwrap_or_default(),
    )
}

async fn write_cleanup_state(app_state: &AppState, state: &CleanupStateFile) -> Result<(), String> {
    let storage_state_dir = canonical_storage_state_dir(app_state).await?;
    tokio::fs::create_dir_all(&storage_state_dir)
        .await
        .map_err(|error| {
            format!(
                "创建 storage state 目录失败: {} ({error})",
                storage_state_dir.display()
            )
        })?;
    let canonical_storage_state_dir =
        tokio::fs::canonicalize(&storage_state_dir)
            .await
            .map_err(|error| {
                format!(
                    "解析 storage state 目录失败: {} ({error})",
                    storage_state_dir.display()
                )
            })?;
    let path = child_file_path(&canonical_storage_state_dir, "cleanup_state.json")?;
    let bytes = serde_json::to_vec_pretty(state).map_err(|error| error.to_string())?;
    tokio::fs::write(&path, bytes)
        .await
        .map_err(|error| format!("写入 cleanup state 失败: {} ({error})", path.display()))
}

fn normalize_non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn normalize_non_empty_option(value: Option<&String>) -> Option<String> {
    value.and_then(|value| normalize_non_empty(value))
}

fn system_time_to_unix_ms(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis() as u64)
}

fn now_unix_ms() -> u64 {
    system_time_to_unix_ms(SystemTime::now()).unwrap_or(0)
}

fn is_protected_signal_kind(signal_kind: &str) -> bool {
    matches!(signal_kind, "manual_hold" | "protected_hold")
}

fn err(code: &str, message: String) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "error": {
            "code": code,
            "message": message,
        }
    })
}

fn internal_error(message: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(err("storage_compat_failed", message)),
    )
}
