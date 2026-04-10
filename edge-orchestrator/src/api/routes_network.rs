use axum::extract::State;
use axum::http::StatusCode;
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command;

use crate::AppState;

#[derive(Deserialize)]
pub struct UplinkSwitchRequest {
    pub target: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/network/uplink", get(get_uplink_status))
        .route("/network/uplink/select", post(select_uplink))
        .with_state(state)
}

async fn get_uplink_status(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    run_uplink_manager(&state, &["status"]).await
}

async fn select_uplink(
    State(state): State<AppState>,
    Json(req): Json<UplinkSwitchRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match req.target.as_str() {
        "ethernet" | "cellular" => {
            run_uplink_manager(&state, &["switch", "--target", &req.target]).await
        }
        _ => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "code": "invalid_uplink_target",
                "message": format!("不支持的 uplink target: {}", req.target),
            })),
        )),
    }
}

async fn run_uplink_manager(
    state: &AppState,
    args: &[&str],
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let output = Command::new("python3")
        .arg(&state.config.uplink_manager_path)
        .args(args)
        .output()
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "code": "uplink_manager_spawn_failed",
                    "message": error.to_string(),
                })),
            )
        })?;

    if !output.status.success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "code": "uplink_manager_failed",
                "status": output.status.code(),
                "stdout": String::from_utf8_lossy(&output.stdout).trim(),
                "stderr": String::from_utf8_lossy(&output.stderr).trim(),
            })),
        ));
    }

    let payload = serde_json::from_slice::<Value>(&output.stdout).map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "code": "uplink_manager_invalid_json",
                "message": error.to_string(),
                "stdout": String::from_utf8_lossy(&output.stdout).trim(),
            })),
        )
    })?;

    Ok(Json(payload))
}
