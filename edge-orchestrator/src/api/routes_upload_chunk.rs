use std::path::{Path, PathBuf};

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tracing::{debug, warn};

use crate::path_safety;
use crate::ws::types::ChunkAckPacket;
use crate::AppState;

#[derive(Debug, Deserialize)]
struct UploadFrameMatchMetadata {
    pub device_id: Option<String>,
    pub media_scope: Option<String>,
    pub media_track: Option<String>,
    pub source_kind: Option<String>,
    pub frame_index: Option<u32>,
    pub frame_source_time_ns: Option<u64>,
    pub frame_edge_time_ns: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct UploadFrameCorrespondenceMetadata {
    pub frame_index: Option<u32>,
    pub frame_source_time_ns: Option<u64>,
    pub frame_edge_time_ns: Option<u64>,
    pub matches: Option<Vec<UploadFrameMatchMetadata>>,
}

#[derive(Debug, Deserialize)]
struct UploadChunkMetadata {
    pub trip_id: String,
    pub session_id: String,
    #[allow(dead_code)]
    pub device_id: Option<String>,
    pub media_scope: Option<String>,
    pub media_track: Option<String>,
    pub source_kind: Option<String>,
    pub clock_domain: Option<String>,
    pub chunk_index: u32,
    pub file_type: Option<String>,
    pub file_name: Option<String>,
    pub source_time_ns: Option<u64>,
    pub source_start_time_ns: Option<u64>,
    pub source_end_time_ns: Option<u64>,
    pub frame_source_time_ns: Option<Vec<u64>>,
    pub frame_count: Option<u32>,
    pub frame_rate_hz: Option<f64>,
    pub frame_correspondence: Option<Vec<UploadFrameCorrespondenceMetadata>>,
    pub camera_calibration: Option<Value>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        // iOS/Android 现网协议兼容：沿用旧路径（multipart：file + metadata）。
        // 这里表示 phone -> edge 的本地原始数据摄取，不是 backend-app legacy 云端 crowd-data 主链。
        .route("/common_task/upload_chunk", post(upload_chunk))
        // PRD 语义更明确的别名（同实现）。
        .route("/chunk/upload", post(upload_chunk))
        .with_state(state)
}

async fn upload_chunk(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    // 允许 metadata/file 的字段顺序不固定：先把 file 落到临时目录，再按 metadata 决定最终落盘位置。
    let tmp_dir = Path::new(&state.config.data_dir).join("tmp");
    if let Err(e) = tokio::fs::create_dir_all(&tmp_dir).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(err("mkdir_fail", format!("创建临时目录失败: {e}"))),
        ));
    }

    let mut metadata_raw: Option<String> = None;
    let mut tmp_path: Option<PathBuf> = None;
    let mut orig_filename: Option<String> = None;
    let mut file_bytes: u64 = 0;

    while let Ok(Some(mut field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "metadata" => match field.text().await {
                Ok(s) => metadata_raw = Some(s),
                Err(e) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(err("bad_metadata", format!("metadata 读取失败: {e}"))),
                    ));
                }
            },
            "file" => {
                orig_filename = field.file_name().map(|s| s.to_string());
                let tmp_name = format!(
                    "upload_{}.bin",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                );
                let path = tmp_dir.join(tmp_name);
                let mut f = match tokio::fs::File::create(&path).await {
                    Ok(v) => v,
                    Err(e) => {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(err("tmp_create_fail", format!("创建临时文件失败: {e}"))),
                        ));
                    }
                };
                while let Ok(Some(chunk)) = field.chunk().await {
                    file_bytes += chunk.len() as u64;
                    if let Err(e) = f.write_all(&chunk).await {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(err("tmp_write_fail", format!("写入临时文件失败: {e}"))),
                        ));
                    }
                }
                tmp_path = Some(path);
            }
            _ => {
                // 忽略未知字段，避免未来扩展破坏兼容性。
                while let Ok(Some(_)) = field.chunk().await {}
            }
        }
    }

    let Some(metadata_raw) = metadata_raw else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err(
                "missing_metadata",
                "缺少 multipart 字段 metadata".to_string(),
            )),
        ));
    };
    let Some(tmp_path) = tmp_path else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err("missing_file", "缺少 multipart 字段 file".to_string())),
        ));
    };

    let mut meta = match serde_json::from_str::<UploadChunkMetadata>(&metadata_raw) {
        Ok(v) => v,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(err(
                    "bad_metadata_json",
                    format!("metadata JSON 解析失败: {e}"),
                )),
            ));
        }
    };
    if meta.trip_id.trim().is_empty() || meta.session_id.trim().is_empty() {
        let snap = state.session.snapshot();
        if meta.trip_id.trim().is_empty() {
            meta.trip_id = snap.trip_id;
        }
        if meta.session_id.trim().is_empty() {
            meta.session_id = snap.session_id;
        }
    }
    if meta.trip_id.trim().is_empty() || meta.session_id.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err(
                "invalid_trip_or_session",
                "trip_id/session_id 不能为空".to_string(),
            )),
        ));
    }

    // 最终落盘位置（与 PRD 目录结构对齐：/data/ruview/session/<session_id>/raw/<scope>/<track>/...）
    let media_scope = match meta.media_scope.as_deref() {
        Some("stereo") => "stereo",
        _ => "iphone",
    };
    let media_track = match resolve_media_track(media_scope, meta.media_track.as_deref()) {
        Ok(track) => track,
        Err((code, message)) => {
            return Err((StatusCode::BAD_REQUEST, Json(err(code, message))));
        }
    };
    let storage_track = storage_media_track(media_scope, media_track);
    let session_dir =
        match path_safety::session_base_dir(Path::new(&state.config.data_dir), &meta.session_id) {
            Ok(path) => path,
            Err(message) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(err("invalid_session_id", message)),
                ));
            }
        };
    let chunk_dir = session_dir
        .join("raw")
        .join(media_scope)
        .join(storage_track)
        .join("chunks")
        .join(format!("{:06}", meta.chunk_index));
    if let Err(e) = tokio::fs::create_dir_all(&chunk_dir).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(err("mkdir_fail", format!("创建 chunk 目录失败: {e}"))),
        ));
    }

    let file_type = sanitize_file_component(meta.file_type.as_deref().unwrap_or("bin"));
    let base_name = sanitize_filename(
        meta.file_name
            .as_deref()
            .or(orig_filename.as_deref())
            .unwrap_or("upload.bin"),
    );
    let dst_path = chunk_dir.join(format!("{file_type}__{base_name}"));
    if dst_path.exists() {
        // 幂等/重试：允许覆盖，避免“重复上传就 409”导致队列卡死。
        if let Err(e) = tokio::fs::remove_file(&dst_path).await {
            warn!(error=%e, path=%dst_path.display(), "failed to remove existing upload file");
        }
    }
    if let Err(e) = tokio::fs::rename(&tmp_path, &dst_path).await {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(err(
                "store_fail",
                format!("落盘失败: {e} (dst={})", dst_path.display()),
            )),
        ));
    }

    if media_scope == "iphone" && storage_track == "fisheye" {
        if let Some(camera_calibration) = meta.camera_calibration.as_ref() {
            let calibration_path = session_dir.join("calibration").join("iphone_fisheye.json");
            let calibration_snapshot = serde_json::json!({
                "type": "sensor_calibration_snapshot",
                "sensor": "iphone_fisheye",
                "trip_id": meta.trip_id,
                "session_id": meta.session_id,
                "sensor_frame": "iphone_fisheye_camera",
                "notes": "upload_chunk fisheye calibration snapshot",
                "camera_calibration": camera_calibration,
            });
            if let Some(parent) = calibration_path.parent() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    warn!(error=%e, path=%parent.display(), "create fisheye calibration dir failed");
                }
            }
            match serde_json::to_vec_pretty(&calibration_snapshot) {
                Ok(raw) => {
                    if let Err(e) = tokio::fs::write(&calibration_path, raw).await {
                        warn!(error=%e, path=%calibration_path.display(), "write fisheye calibration snapshot failed");
                    }
                }
                Err(e) => {
                    warn!(error=%e, "encode fisheye calibration snapshot failed");
                }
            }
        }
    }

    let upload_edge_time_ns = state.gate.edge_time_ns();
    let frame_source_time_ns = meta.frame_source_time_ns.clone().unwrap_or_default();
    let frame_edge_time_ns = meta
        .device_id
        .as_deref()
        .and_then(|device_id| {
            frame_source_time_ns
                .iter()
                .map(|source_time_ns| {
                    state
                        .gate
                        .estimate_source_time_to_edge_historical(device_id, *source_time_ns)
                })
                .collect::<Option<Vec<u64>>>()
        })
        .unwrap_or_default();
    let source_time_ns = meta
        .source_time_ns
        .or_else(|| frame_source_time_ns.first().copied());
    let source_start_time_ns = meta
        .source_start_time_ns
        .or_else(|| frame_source_time_ns.first().copied())
        .or(source_time_ns);
    let source_end_time_ns = meta
        .source_end_time_ns
        .or_else(|| frame_source_time_ns.last().copied())
        .or(source_time_ns);
    let segment_start_edge_time_ns = frame_edge_time_ns.first().copied().or_else(|| {
        meta.device_id.as_deref().and_then(|device_id| {
            source_start_time_ns.and_then(|ts| {
                state
                    .gate
                    .estimate_source_time_to_edge_historical(device_id, ts)
            })
        })
    });
    let segment_end_edge_time_ns = frame_edge_time_ns.last().copied().or_else(|| {
        meta.device_id.as_deref().and_then(|device_id| {
            source_end_time_ns.and_then(|ts| {
                state
                    .gate
                    .estimate_source_time_to_edge_historical(device_id, ts)
            })
        })
    });
    let edge_time_ns = match (segment_start_edge_time_ns, segment_end_edge_time_ns) {
        (Some(start), Some(end)) if end >= start => start + (end - start) / 2,
        (Some(ts), _) | (_, Some(ts)) => ts,
        _ => upload_edge_time_ns,
    };
    let media_alignment_kind = if !frame_source_time_ns.is_empty()
        && frame_source_time_ns.len() == frame_edge_time_ns.len()
    {
        "frame_level_indexed"
    } else {
        "chunk_level"
    };
    let time_sync_status = if !frame_source_time_ns.is_empty()
        && frame_source_time_ns.len() == frame_edge_time_ns.len()
    {
        "frame_indexed"
    } else if segment_start_edge_time_ns.is_some() || segment_end_edge_time_ns.is_some() {
        "segment_estimated"
    } else {
        "unmapped"
    };
    let relpath = dst_path
        .strip_prefix(&session_dir)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| dst_path.to_string_lossy().to_string());
    if media_scope == "stereo" {
        state
            .recorder
            .record_stereo_media_chunk_index(
                &state.protocol,
                &state.config,
                &meta.trip_id,
                &meta.session_id,
                meta.device_id.as_deref(),
                Some(media_scope),
                Some(media_track),
                meta.source_kind.as_deref(),
                meta.clock_domain.as_deref(),
                meta.chunk_index,
                meta.file_type.as_deref(),
                meta.file_name.as_deref().or(orig_filename.as_deref()),
                &relpath,
                file_bytes,
                edge_time_ns,
                upload_edge_time_ns,
                source_time_ns,
                source_start_time_ns,
                source_end_time_ns,
                frame_source_time_ns.clone(),
                frame_edge_time_ns.clone(),
                meta.frame_count,
                meta.frame_rate_hz,
                media_alignment_kind,
                time_sync_status,
            )
            .await;
    } else {
        state
            .recorder
            .record_media_chunk_index(
                &state.protocol,
                &state.config,
                &meta.trip_id,
                &meta.session_id,
                meta.device_id.as_deref(),
                Some(media_scope),
                Some(media_track),
                meta.source_kind.as_deref(),
                meta.clock_domain.as_deref(),
                meta.chunk_index,
                meta.file_type.as_deref(),
                meta.file_name.as_deref().or(orig_filename.as_deref()),
                &relpath,
                file_bytes,
                edge_time_ns,
                upload_edge_time_ns,
                source_time_ns,
                source_start_time_ns,
                source_end_time_ns,
                frame_source_time_ns.clone(),
                frame_edge_time_ns.clone(),
                meta.frame_count,
                meta.frame_rate_hz,
                media_alignment_kind,
                time_sync_status,
            )
            .await;
    }
    if let Some(correspondences) = meta.frame_correspondence.as_ref() {
        for correspondence in correspondences {
            let anchor_frame_index = correspondence.frame_index;
            let anchor_source_time_ns = correspondence
                .frame_source_time_ns
                .or_else(|| resolve_indexed_u64(anchor_frame_index, &frame_source_time_ns));
            let anchor_edge_time_ns = correspondence
                .frame_edge_time_ns
                .or_else(|| resolve_indexed_u64(anchor_frame_index, &frame_edge_time_ns))
                .or_else(|| {
                    meta.device_id.as_deref().and_then(|device_id| {
                        anchor_source_time_ns.and_then(|ts| {
                            state
                                .gate
                                .estimate_source_time_to_edge_historical(device_id, ts)
                        })
                    })
                })
                .unwrap_or(edge_time_ns);
            let matches = correspondence
                .matches
                .as_ref()
                .map(|items| {
                    items
                        .iter()
                        .map(|item| {
                            let match_scope = item.media_scope.as_deref().unwrap_or(media_scope);
                            let match_track =
                                resolve_media_track(match_scope, item.media_track.as_deref())
                                    .unwrap_or(item.media_track.as_deref().unwrap_or(
                                        if match_scope == "stereo" {
                                            "preview"
                                        } else {
                                            "main"
                                        },
                                    ));
                            let match_source_time_ns = item.frame_source_time_ns;
                            let match_edge_time_ns = item.frame_edge_time_ns.or_else(|| {
                                item.device_id.as_deref().and_then(|device_id| {
                                    match_source_time_ns.and_then(|ts| {
                                        state
                                            .gate
                                            .estimate_source_time_to_edge_historical(device_id, ts)
                                    })
                                })
                            });
                            serde_json::json!({
                                "device_id": item.device_id.clone().unwrap_or_default(),
                                "media_scope": match_scope,
                                "media_track": match_track,
                                "source_kind": item.source_kind.clone().unwrap_or_default(),
                                "frame_index": item.frame_index,
                                "frame_source_time_ns": match_source_time_ns,
                                "frame_edge_time_ns": match_edge_time_ns,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let event = serde_json::json!({
                "type": "frame_correspondence",
                "schema_version": "1.0.0",
                "trip_id": meta.trip_id.clone(),
                "session_id": meta.session_id.clone(),
                "source_kind": meta.source_kind.clone().unwrap_or_default(),
                "clock_domain": meta.clock_domain.clone().unwrap_or_default(),
                "anchor": {
                    "device_id": meta.device_id.clone().unwrap_or_default(),
                    "media_scope": media_scope,
                    "media_track": media_track,
                    "chunk_index": meta.chunk_index,
                    "frame_index": anchor_frame_index,
                    "frame_source_time_ns": anchor_source_time_ns,
                    "frame_edge_time_ns": anchor_edge_time_ns,
                },
                "matches": matches,
                "edge_time_ns": anchor_edge_time_ns,
            });
            state
                .recorder
                .record_frame_correspondence(
                    &state.protocol,
                    &state.config,
                    &meta.trip_id,
                    &meta.session_id,
                    &event,
                )
                .await;
        }
    }

    metrics::counter!("upload_chunk_count", "file_type" => file_type.clone(), "media_scope" => media_scope.to_string()).increment(1);
    metrics::counter!("upload_chunk_bytes_total").increment(file_bytes);

    // chunk 状态机：收到并完成落盘后推进到 stored；若满足完整条件则下发一次 ack。
    let should_ack = if media_scope == "iphone" {
        state.chunk_sm.note_file_stored(
            &meta.trip_id,
            &meta.session_id,
            meta.chunk_index,
            meta.file_type.as_deref(),
            &state.config.upload_required_file_types,
        )
    } else {
        false
    };

    if should_ack {
        let pkt = ChunkAckPacket {
            ty: "chunk_ack_packet",
            schema_version: "1.0.0",
            session_id: meta.session_id.clone(),
            trip_id: meta.trip_id.clone(),
            chunk_index: meta.chunk_index,
            status: "stored".to_string(),
            edge_time_ns: upload_edge_time_ns,
        };
        let subs = state.chunk_ack_tx.send(pkt);
        state
            .chunk_sm
            .mark_acked(&meta.trip_id, &meta.session_id, meta.chunk_index);
        metrics::counter!("chunk_ack_emit_count").increment(1);
        debug!(
            subscribers = subs.unwrap_or(0),
            chunk_index = meta.chunk_index,
            "emit chunk_ack_packet(stored)"
        );

        // 事件落盘：received -> stored -> acked
        state
            .recorder
            .record_chunk_state_event(
                &state.protocol,
                &state.config,
                &meta.trip_id,
                &meta.session_id,
                meta.chunk_index,
                "received",
                "stored",
                upload_edge_time_ns,
                "",
                "",
            )
            .await;
        state
            .recorder
            .record_chunk_state_event(
                &state.protocol,
                &state.config,
                &meta.trip_id,
                &meta.session_id,
                meta.chunk_index,
                "stored",
                "acked",
                upload_edge_time_ns,
                "",
                "",
            )
            .await;
    }

    Ok(Json(serde_json::json!({
        "ok": true,
        "trip_id": meta.trip_id,
        "session_id": meta.session_id,
        "chunk_index": meta.chunk_index,
        "file_type": file_type,
        "media_scope": media_scope,
        "media_track": media_track,
        "stored_path": dst_path.display().to_string(),
    })))
}

fn sanitize_filename(input: &str) -> String {
    let raw = Path::new(input)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("upload.bin")
        .trim();
    if raw.is_empty() {
        return "upload.bin".to_string();
    }
    raw.chars()
        .map(|c| match c {
            '/' | '\\' | ':' => '_',
            _ => c,
        })
        .collect()
}

fn sanitize_file_component(input: &str) -> String {
    let raw = input.trim();
    if raw.is_empty() {
        return "bin".to_string();
    }
    let sanitized: String = raw
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '\0' | '\n' | '\r' => '_',
            _ => c,
        })
        .collect();
    if sanitized.trim_matches('.').is_empty() {
        "bin".to_string()
    } else {
        sanitized
    }
}

fn resolve_media_track<'a>(
    media_scope: &str,
    media_track: Option<&'a str>,
) -> Result<&'a str, (&'static str, String)> {
    match media_scope {
        "iphone" => match media_track.unwrap_or("main") {
            "main" | "aux" | "depth" | "fisheye" => Ok(media_track.unwrap_or("main")),
            other => Err((
                "invalid_media_track",
                format!("iphone media_track 不支持: {other}"),
            )),
        },
        "stereo" => match media_track.unwrap_or("preview") {
            "left" | "right" | "preview" => Ok(media_track.unwrap_or("preview")),
            other => Err((
                "invalid_media_track",
                format!("stereo media_track 不支持: {other}"),
            )),
        },
        other => Err((
            "invalid_media_scope",
            format!("不支持的 media_scope: {other}"),
        )),
    }
}

fn storage_media_track<'a>(media_scope: &'a str, media_track: &'a str) -> &'a str {
    if media_scope == "iphone" {
        match media_track {
            "main" => "wide",
            // Backward compatibility: older iPhone auxiliary uploads were stored as `aux`
            // even when they semantically represented the ultrawide/fisheye side track.
            "aux" => "fisheye",
            _ => media_track,
        }
    } else {
        media_track
    }
}

fn resolve_indexed_u64(index: Option<u32>, values: &[u64]) -> Option<u64> {
    let index = usize::try_from(index?).ok()?;
    values.get(index).copied()
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
