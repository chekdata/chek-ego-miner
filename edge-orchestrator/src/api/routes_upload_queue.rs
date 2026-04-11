use std::path::Path;

use axum::extract::{Query, State};
use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;

use crate::path_safety;
use crate::recorder::upload_queue::{self, UploadReceiptInput};
use crate::AppState;

#[derive(Debug, Deserialize)]
struct UploadQueueQuery {
    session_id: String,
}

#[derive(Debug, Deserialize)]
struct UploadReceiptRequest {
    trip_id: String,
    session_id: String,
    asset_id: String,
    status: String,
    receipt_source: Option<String>,
    remote_object_key: Option<String>,
    remote_upload_id: Option<String>,
    last_error: Option<String>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/upload/queue", get(get_upload_queue))
        .route("/upload/receipt", post(post_upload_receipt))
        .with_state(state)
}

async fn get_upload_queue(
    State(state): State<AppState>,
    Query(query): Query<UploadQueueQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if query.session_id.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err("missing_session_id", "session_id 不能为空".to_string())),
        ));
    }
    let base_dir = session_base_dir(&state, &query.session_id).map_err(|message| {
        (
            StatusCode::BAD_REQUEST,
            Json(err("invalid_session_id", message)),
        )
    })?;
    match upload_queue::load_or_refresh_upload_queue(&base_dir).await {
        Ok(queue) => Ok(Json(queue)),
        Err(message) => Err((
            StatusCode::NOT_FOUND,
            Json(err("upload_queue_unavailable", message)),
        )),
    }
}

async fn post_upload_receipt(
    State(state): State<AppState>,
    Json(req): Json<UploadReceiptRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if req.trip_id.trim().is_empty()
        || req.session_id.trim().is_empty()
        || req.asset_id.trim().is_empty()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err(
                "missing_required_fields",
                "trip_id/session_id/asset_id 不能为空".to_string(),
            )),
        ));
    }
    if !upload_queue::is_valid_receipt_status(req.status.trim()) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err(
                "invalid_upload_status",
                format!("upload receipt status 不支持: {}", req.status),
            )),
        ));
    }
    let base_dir = session_base_dir(&state, &req.session_id).map_err(|message| {
        (
            StatusCode::BAD_REQUEST,
            Json(err("invalid_session_id", message)),
        )
    })?;
    let receipt = UploadReceiptInput {
        trip_id: req.trip_id,
        session_id: req.session_id,
        asset_id: req.asset_id,
        status: req.status,
        receipt_source: req
            .receipt_source
            .unwrap_or_else(|| "edge_uploader".to_string()),
        remote_object_key: req.remote_object_key,
        remote_upload_id: req.remote_upload_id,
        last_error: req.last_error,
    };

    if let Err(message) = upload_queue::append_upload_receipt_and_refresh(&base_dir, receipt).await
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(err("upload_receipt_failed", message)),
        ));
    }

    match upload_queue::load_or_refresh_upload_queue(&base_dir).await {
        Ok(queue) => Ok(Json(serde_json::json!({
            "ok": true,
            "queue": queue,
        }))),
        Err(message) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(err("upload_queue_refresh_failed", message)),
        )),
    }
}

fn session_base_dir(state: &AppState, session_id: &str) -> Result<std::path::PathBuf, String> {
    path_safety::session_base_dir(Path::new(&state.config.data_dir), session_id)
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
