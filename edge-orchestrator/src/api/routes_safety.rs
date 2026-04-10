use axum::extract::State;
use axum::{routing::post, Json, Router};
use serde::Deserialize;

use crate::AppState;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct EstopRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub reason: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ReleaseRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct SafetyConfigRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub limit_conf_threshold: Option<f32>,
    pub freeze_conf_threshold: Option<f32>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/safety/estop", post(estop))
        .route("/safety/release", post(release))
        .route("/safety/config", post(config))
        .with_state(state)
}

async fn estop(
    State(state): State<AppState>,
    Json(req): Json<EstopRequest>,
) -> Json<serde_json::Value> {
    let reason = req.reason.unwrap_or_else(|| "estop".to_string());
    state.gate.set_estop(&reason);
    metrics::counter!("safety_estop_count").increment(1);
    Json(serde_json::json!({"ok": true}))
}

async fn release(
    State(state): State<AppState>,
    Json(_req): Json<ReleaseRequest>,
) -> Json<serde_json::Value> {
    state.gate.release_estop();
    metrics::counter!("safety_release_count").increment(1);
    Json(serde_json::json!({"ok": true}))
}

async fn config(
    State(state): State<AppState>,
    Json(req): Json<SafetyConfigRequest>,
) -> Json<serde_json::Value> {
    state
        .gate
        .set_quality_thresholds(req.limit_conf_threshold, req.freeze_conf_threshold);
    metrics::counter!("safety_config_count").increment(1);
    Json(serde_json::json!({"ok": true}))
}
