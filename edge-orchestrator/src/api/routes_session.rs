use axum::extract::State;
use axum::{routing::post, Json, Router};
use serde::{Deserialize, Serialize};

use crate::recorder::session_recorder::SessionContextUpdate;
use crate::AppState;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct SessionStartRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub device_id: String,
    pub operator_id: Option<String>,
    pub task_id: Option<String>,
    pub task_ids: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct SessionStopRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub reason: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ModeSetRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub mode: String,
}

#[derive(Serialize)]
pub struct OkResponse {
    ok: bool,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/session/start", post(start))
        .route("/session/stop", post(stop))
        .route("/session/pause", post(pause))
        .route("/session/resume", post(resume))
        .route("/mode/set", post(set_mode))
        .with_state(state)
}

async fn start(
    State(state): State<AppState>,
    Json(req): Json<SessionStartRequest>,
) -> Json<OkResponse> {
    state.session.set_active(req.trip_id, req.session_id);
    state
        .session
        .set_mode(state.config.default_session_mode().to_string());
    let session = state.session.snapshot();
    let recorder = state.recorder.clone();
    let protocol = state.protocol.clone();
    let config = state.config.clone();
    let trip_id = session.trip_id.clone();
    let session_id = session.session_id.clone();
    let context_update = SessionContextUpdate {
        capture_device_id: Some(req.device_id),
        operator_id: req.operator_id,
        task_id: req.task_id,
        task_ids: req.task_ids.unwrap_or_default(),
    };
    let _ = recorder
        .ensure_session(&trip_id, &session_id, &protocol, &config)
        .await;
    recorder
        .update_session_context(&protocol, &config, &trip_id, &session_id, context_update)
        .await;
    metrics::counter!("session_start_count").increment(1);
    Json(OkResponse { ok: true })
}

async fn stop(
    State(state): State<AppState>,
    Json(req): Json<SessionStopRequest>,
) -> Json<OkResponse> {
    state.session.clear_if_match(&req.trip_id, &req.session_id);
    state.gate.disarm("session_stop");
    let recorder = state.recorder.clone();
    let config = state.config.clone();
    let trip_id = req.trip_id.clone();
    let session_id = req.session_id.clone();
    tokio::spawn(async move {
        recorder.stop_if_match(&trip_id, &session_id, &config).await;
    });
    metrics::counter!("session_stop_count").increment(1);
    Json(OkResponse { ok: true })
}

async fn pause(
    State(state): State<AppState>,
    Json(_req): Json<SessionStopRequest>,
) -> Json<OkResponse> {
    state.gate.disarm("session_pause");
    metrics::counter!("session_pause_count").increment(1);
    Json(OkResponse { ok: true })
}

async fn resume(
    State(_state): State<AppState>,
    Json(_req): Json<SessionStopRequest>,
) -> Json<OkResponse> {
    metrics::counter!("session_resume_count").increment(1);
    Json(OkResponse { ok: true })
}

async fn set_mode(
    State(state): State<AppState>,
    Json(req): Json<ModeSetRequest>,
) -> Json<serde_json::Value> {
    match req.mode.as_str() {
        "vision_only" | "csi_only" | "fused" => {}
        _ => {
            return Json(serde_json::json!({
                "ok": false,
                "error": {
                    "code": "invalid_mode",
                    "message": "mode 必须为 vision_only | csi_only | fused"
                }
            }));
        }
    }

    state.session.set_active(req.trip_id, req.session_id);
    state.session.set_mode(req.mode);
    metrics::counter!("mode_set_count").increment(1);
    Json(serde_json::json!({"ok": true}))
}
