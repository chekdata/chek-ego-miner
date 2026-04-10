use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone)]
pub struct UploadReceiptInput {
    pub trip_id: String,
    pub session_id: String,
    pub asset_id: String,
    pub status: String,
    pub receipt_source: String,
    pub remote_object_key: Option<String>,
    pub remote_upload_id: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UploadManifestFile {
    trip_id: String,
    session_id: String,
    ready_for_upload: bool,
    artifacts: Vec<UploadManifestFileArtifact>,
}

#[derive(Debug, Deserialize)]
struct UploadManifestFileArtifact {
    id: String,
    relpath: String,
    kind: String,
    category: String,
    required: bool,
    exists: bool,
    byte_size: u64,
    line_count: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredUploadReceipt {
    #[serde(rename = "type")]
    ty: String,
    schema_version: String,
    trip_id: String,
    session_id: String,
    queue_id: String,
    asset_id: String,
    status: String,
    receipt_source: String,
    remote_object_key: Option<String>,
    remote_upload_id: Option<String>,
    last_error: Option<String>,
    recorded_unix_ms: u64,
}

#[derive(Debug, Serialize)]
struct UploadQueueSnapshot {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    trip_id: String,
    session_id: String,
    generated_unix_ms: u64,
    source_upload_manifest: &'static str,
    ready_for_upload: bool,
    status: &'static str,
    summary: UploadQueueSummary,
    entries: Vec<UploadQueueEntry>,
}

#[derive(Debug, Serialize)]
struct UploadQueueSummary {
    total_entries: usize,
    queued_count: usize,
    uploading_count: usize,
    acked_count: usize,
    failed_count: usize,
    blocked_count: usize,
    pending_artifact_count: usize,
}

#[derive(Debug, Serialize)]
struct UploadQueueEntry {
    queue_id: String,
    asset_type: &'static str,
    asset_id: String,
    relpath: String,
    kind: String,
    category: String,
    required: bool,
    exists: bool,
    byte_size: u64,
    line_count: Option<u64>,
    status: &'static str,
    retry_count: u32,
    last_error: Option<String>,
    remote_object_key: Option<String>,
    remote_upload_id: Option<String>,
    last_receipt_unix_ms: Option<u64>,
}

#[derive(Debug, Default)]
struct ReceiptAggregate {
    status: Option<&'static str>,
    retry_count: u32,
    last_error: Option<String>,
    remote_object_key: Option<String>,
    remote_upload_id: Option<String>,
    last_receipt_unix_ms: Option<u64>,
}

pub fn is_valid_receipt_status(status: &str) -> bool {
    matches!(status, "uploading" | "acked" | "failed")
}

pub async fn refresh_upload_queue(base_dir: &Path) -> Result<(), String> {
    let upload_dir = upload_dir(base_dir);
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|e| format!("创建 upload 目录失败: {} ({e})", upload_dir.display()))?;

    let manifest = read_upload_manifest(base_dir).await?;
    let receipt_aggregates = read_receipt_aggregates(base_dir).await?;

    let mut entries = Vec::new();
    let mut summary = UploadQueueSummary {
        total_entries: manifest.artifacts.len(),
        queued_count: 0,
        uploading_count: 0,
        acked_count: 0,
        failed_count: 0,
        blocked_count: 0,
        pending_artifact_count: 0,
    };

    for artifact in manifest.artifacts {
        let aggregate = receipt_aggregates.get(&artifact.id);
        let status = aggregate
            .and_then(|value| value.status)
            .unwrap_or_else(|| default_entry_status(manifest.ready_for_upload, artifact.exists));
        increment_summary(&mut summary, status);
        entries.push(UploadQueueEntry {
            queue_id: queue_id_for(&manifest.session_id, &artifact.id),
            asset_type: asset_type_for(&artifact.relpath),
            asset_id: artifact.id,
            relpath: artifact.relpath,
            kind: artifact.kind,
            category: artifact.category,
            required: artifact.required,
            exists: artifact.exists,
            byte_size: artifact.byte_size,
            line_count: artifact.line_count,
            status,
            retry_count: aggregate.map(|value| value.retry_count).unwrap_or(0),
            last_error: aggregate.and_then(|value| value.last_error.clone()),
            remote_object_key: aggregate.and_then(|value| value.remote_object_key.clone()),
            remote_upload_id: aggregate.and_then(|value| value.remote_upload_id.clone()),
            last_receipt_unix_ms: aggregate.and_then(|value| value.last_receipt_unix_ms),
        });
    }
    entries.sort_by(|a, b| a.relpath.cmp(&b.relpath));

    let queue = UploadQueueSnapshot {
        ty: "upload_queue",
        schema_version: "1.0.0",
        trip_id: manifest.trip_id,
        session_id: manifest.session_id,
        generated_unix_ms: now_unix_ms(),
        source_upload_manifest: "upload/upload_manifest.json",
        ready_for_upload: manifest.ready_for_upload,
        status: queue_status(manifest.ready_for_upload, &summary),
        summary,
        entries,
    };
    let bytes = serde_json::to_vec_pretty(&queue).map_err(|e| e.to_string())?;
    tokio::fs::write(upload_dir.join("upload_queue.json"), bytes)
        .await
        .map_err(|e| format!("写入 upload_queue.json 失败: {e}"))
}

pub async fn append_upload_receipt_and_refresh(
    base_dir: &Path,
    input: UploadReceiptInput,
) -> Result<(), String> {
    let upload_dir = upload_dir(base_dir);
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|e| format!("创建 upload 目录失败: {} ({e})", upload_dir.display()))?;

    let manifest = read_upload_manifest(base_dir).await?;
    if manifest.trip_id != input.trip_id || manifest.session_id != input.session_id {
        return Err(format!(
            "upload receipt trip/session 不匹配: expected {}/{} got {}/{}",
            manifest.trip_id, manifest.session_id, input.trip_id, input.session_id
        ));
    }
    if !manifest
        .artifacts
        .iter()
        .any(|artifact| artifact.id == input.asset_id)
    {
        return Err(format!(
            "upload receipt asset_id 不存在: {}",
            input.asset_id
        ));
    }

    let receipt = StoredUploadReceipt {
        ty: "upload_receipt".to_string(),
        schema_version: "1.0.0".to_string(),
        trip_id: input.trip_id,
        session_id: input.session_id.clone(),
        queue_id: queue_id_for(&input.session_id, &input.asset_id),
        asset_id: input.asset_id,
        status: input.status,
        receipt_source: input.receipt_source,
        remote_object_key: input.remote_object_key,
        remote_upload_id: input.remote_upload_id,
        last_error: input.last_error,
        recorded_unix_ms: now_unix_ms(),
    };

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(upload_dir.join("upload_receipts.jsonl"))
        .await
        .map_err(|e| format!("打开 upload_receipts.jsonl 失败: {e}"))?;
    let mut line = serde_json::to_vec(&receipt).map_err(|e| e.to_string())?;
    line.push(b'\n');
    file.write_all(&line)
        .await
        .map_err(|e| format!("写入 upload_receipts.jsonl 失败: {e}"))?;

    refresh_upload_queue(base_dir).await
}

pub async fn load_or_refresh_upload_queue(base_dir: &Path) -> Result<serde_json::Value, String> {
    let path = upload_dir(base_dir).join("upload_queue.json");
    if !tokio::fs::try_exists(&path)
        .await
        .map_err(|e| format!("检查 upload_queue.json 是否存在失败: {e}"))?
    {
        refresh_upload_queue(base_dir).await?;
    }
    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("读取 upload_queue.json 失败: {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("解析 upload_queue.json 失败: {e}"))
}

fn upload_dir(base_dir: &Path) -> PathBuf {
    base_dir.join("upload")
}

async fn read_upload_manifest(base_dir: &Path) -> Result<UploadManifestFile, String> {
    let manifest_path = upload_dir(base_dir).join("upload_manifest.json");
    let content = tokio::fs::read_to_string(&manifest_path)
        .await
        .map_err(|e| {
            format!(
                "读取 upload_manifest.json 失败: {} ({e})",
                manifest_path.display()
            )
        })?;
    serde_json::from_str::<UploadManifestFile>(&content)
        .map_err(|e| format!("解析 upload_manifest.json 失败: {e}"))
}

async fn read_receipt_aggregates(
    base_dir: &Path,
) -> Result<HashMap<String, ReceiptAggregate>, String> {
    let path = upload_dir(base_dir).join("upload_receipts.jsonl");
    if !tokio::fs::try_exists(&path)
        .await
        .map_err(|e| format!("检查 upload_receipts.jsonl 是否存在失败: {e}"))?
    {
        return Ok(HashMap::new());
    }

    let content = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("读取 upload_receipts.jsonl 失败: {e}"))?;
    let mut aggregates = HashMap::new();
    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let receipt = serde_json::from_str::<StoredUploadReceipt>(line).map_err(|e| {
            format!(
                "解析 upload_receipts.jsonl 第 {} 行失败: {e}",
                index.saturating_add(1)
            )
        })?;
        let aggregate = aggregates
            .entry(receipt.asset_id)
            .or_insert_with(ReceiptAggregate::default);
        if receipt.status == "failed" {
            aggregate.retry_count = aggregate.retry_count.saturating_add(1);
        }
        aggregate.status = status_literal(&receipt.status);
        aggregate.last_error = receipt.last_error;
        aggregate.remote_object_key = receipt.remote_object_key;
        aggregate.remote_upload_id = receipt.remote_upload_id;
        aggregate.last_receipt_unix_ms = Some(receipt.recorded_unix_ms);
    }
    Ok(aggregates)
}

fn queue_id_for(session_id: &str, asset_id: &str) -> String {
    format!("{session_id}:{asset_id}")
}

fn asset_type_for(relpath: &str) -> &'static str {
    if relpath.ends_with("manifest.json") || relpath.contains("quality_report") {
        "manifest"
    } else {
        "recording"
    }
}

fn default_entry_status(ready_for_upload: bool, exists: bool) -> &'static str {
    if !ready_for_upload {
        "blocked"
    } else if !exists {
        "pending_artifact"
    } else {
        "queued"
    }
}

fn status_literal(status: &str) -> Option<&'static str> {
    match status {
        "uploading" => Some("uploading"),
        "acked" => Some("acked"),
        "failed" => Some("failed"),
        "queued" => Some("queued"),
        "blocked" => Some("blocked"),
        "pending_artifact" => Some("pending_artifact"),
        _ => None,
    }
}

fn increment_summary(summary: &mut UploadQueueSummary, status: &str) {
    match status {
        "queued" => summary.queued_count = summary.queued_count.saturating_add(1),
        "uploading" => summary.uploading_count = summary.uploading_count.saturating_add(1),
        "acked" => summary.acked_count = summary.acked_count.saturating_add(1),
        "failed" => summary.failed_count = summary.failed_count.saturating_add(1),
        "blocked" => summary.blocked_count = summary.blocked_count.saturating_add(1),
        "pending_artifact" => {
            summary.pending_artifact_count = summary.pending_artifact_count.saturating_add(1)
        }
        _ => {}
    }
}

fn queue_status(ready_for_upload: bool, summary: &UploadQueueSummary) -> &'static str {
    if summary.total_entries == 0 {
        return "empty";
    }
    if !ready_for_upload {
        return "blocked";
    }
    if summary.failed_count > 0 {
        return "failed";
    }
    if summary.uploading_count > 0 {
        return "uploading";
    }
    if summary.pending_artifact_count > 0 {
        return "queued";
    }
    if summary.queued_count == 0 && summary.acked_count > 0 {
        return "acked";
    }
    "queued"
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
