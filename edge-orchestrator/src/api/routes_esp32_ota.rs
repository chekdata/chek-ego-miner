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
pub struct Esp32OtaDeployRequest {
    pub targets: Option<Vec<String>>,
    pub image_path: Option<String>,
    pub online_only: Option<bool>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/devices/esp32/ota/status", get(get_status))
        .route("/devices/esp32/ota/deploy", post(deploy))
        .with_state(state)
}

async fn get_status(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mac_prefixes = state.config.esp32_ota_mac_prefixes.join(",");
    run_esp32_ota_manager(
        &state,
        &[
            "status",
            "--leases-path",
            &state.config.esp32_ota_leases_path,
            "--wifi-if",
            &state.config.esp32_ota_wifi_if,
            "--mac-prefixes",
            &mac_prefixes,
        ],
    )
    .await
}

async fn deploy(
    State(state): State<AppState>,
    Json(req): Json<Esp32OtaDeployRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let image_path = req
        .image_path
        .unwrap_or_else(|| state.config.esp32_firmware_image_path.clone());
    let mac_prefixes = state.config.esp32_ota_mac_prefixes.join(",");
    let mut args = vec![
        "deploy",
        "--image",
        image_path.as_str(),
        "--leases-path",
        state.config.esp32_ota_leases_path.as_str(),
        "--wifi-if",
        state.config.esp32_ota_wifi_if.as_str(),
        "--mac-prefixes",
        mac_prefixes.as_str(),
    ];

    if req.online_only.unwrap_or(true) {
        args.push("--online-only");
    }

    let target_list = req.targets.unwrap_or_default();
    let target_csv = if target_list.is_empty() {
        None
    } else {
        Some(target_list.join(","))
    };
    if let Some(targets) = target_csv.as_deref() {
        args.push("--targets");
        args.push(targets);
    }

    run_esp32_ota_manager(&state, &args).await
}

async fn run_esp32_ota_manager(
    state: &AppState,
    args: &[&str],
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let output = Command::new("python3")
        .arg(&state.config.esp32_ota_manager_path)
        .args(args)
        .output()
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "code": "esp32_ota_manager_spawn_failed",
                    "message": error.to_string(),
                })),
            )
        })?;

    if !output.status.success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({
                "code": "esp32_ota_manager_failed",
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
                "code": "esp32_ota_manager_invalid_json",
                "message": error.to_string(),
                "stdout": String::from_utf8_lossy(&output.stdout).trim(),
            })),
        )
    })?;

    Ok(Json(payload))
}
