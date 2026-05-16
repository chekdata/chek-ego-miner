use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use crate::token_authority::{PairingExchangeRequest, PairingPublicUrls, PairingRevokeRequest};
use crate::AppState;

pub fn public_router(state: AppState) -> Router {
    Router::new()
        .route("/pairing/envelope", get(get_pairing_envelope))
        .route("/pairing/exchange", post(post_pairing_exchange))
        .route("/devices.json", get(get_devices_json))
        .with_state(state)
}

pub fn protected_router(state: AppState) -> Router {
    Router::new()
        .route("/pairing/revoke", post(post_pairing_revoke))
        .with_state(state)
}

async fn get_pairing_envelope(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let authority = state
        .token_authority
        .as_ref()
        .ok_or_else(authority_disabled)?;
    let envelope = authority
        .issue_pairing_envelope(
            state.config.pairing_profile_id.clone(),
            PairingPublicUrls {
                edge_base_url: pairing_url_or_default(
                    &state.config.pairing_edge_base_url,
                    "http",
                    &state.config.http_addr,
                ),
                edge_ws_url: pairing_url_or_default(
                    &state.config.pairing_edge_ws_url,
                    "ws",
                    &state.config.ws_addr,
                ),
                status_ui_url: pairing_status_ui_url(&state),
            },
            state.config.pairing_ttl_sec,
            state.config.pairing_operator_hint.clone(),
            state.config.pairing_scene_hint.clone(),
        )
        .await
        .map_err(internal_error)?;
    Ok(Json(json!(envelope)))
}

async fn post_pairing_exchange(
    State(state): State<AppState>,
    Json(req): Json<PairingExchangeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let authority = state
        .token_authority
        .as_ref()
        .ok_or_else(authority_disabled)?;
    let (device, scoped_upload_token, expires_unix_ms) = authority
        .exchange_pairing_challenge(req, state.config.pairing_upload_token_ttl_sec)
        .await
        .map_err(pairing_error)?;
    Ok(Json(json!({
        "ok": true,
        "device": device,
        "scoped_upload_token": scoped_upload_token,
        "token_type": "chek_edge_pairing_bearer",
        "expires_unix_ms": expires_unix_ms,
        "edge_base_url": pairing_url_or_default(
            &state.config.pairing_edge_base_url,
            "http",
            &state.config.http_addr,
        ),
        "status_ui_url": pairing_status_ui_url(&state),
    })))
}

async fn get_devices_json(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let authority = state
        .token_authority
        .as_ref()
        .ok_or_else(authority_disabled)?;
    let devices = authority
        .public_devices(Some(state.config.pairing_profile_id.clone()))
        .await
        .map_err(internal_error)?;
    Ok(Json(json!({
        "generated_unix_ms": now_unix_ms(),
        "profile_id": state.config.pairing_profile_id,
        "devices": devices,
        "active_pairing_challenges": null,
        "authority": {
            "kind": "sqlite",
        },
    })))
}

async fn post_pairing_revoke(
    State(state): State<AppState>,
    Json(req): Json<PairingRevokeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let authority = state
        .token_authority
        .as_ref()
        .ok_or_else(authority_disabled)?;
    let revoked_count = authority.revoke(req).await.map_err(pairing_error)?;
    Ok(Json(json!({
        "ok": true,
        "revoked_count": revoked_count,
    })))
}

fn authority_disabled() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "ok": false,
            "error": {
                "code": "sqlite_token_authority_disabled",
                "message": "EDGE_UPLOAD_TOKEN_AUTHORITY_DB_PATH 未配置，当前仍处于 JSON registry pilot 模式",
            }
        })),
    )
}

fn pairing_error(message: String) -> (StatusCode, Json<serde_json::Value>) {
    let status = match message.as_str() {
        "unknown_or_expired_pairing_challenge" => StatusCode::NOT_FOUND,
        "pairing_code_mismatch" => StatusCode::FORBIDDEN,
        "pairing_challenge_already_used" => StatusCode::CONFLICT,
        _ => StatusCode::BAD_REQUEST,
    };
    (
        status,
        Json(json!({
            "ok": false,
            "error": {
                "code": message,
                "message": "pairing authority request failed",
            }
        })),
    )
}

fn internal_error(message: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "ok": false,
            "error": {
                "code": "internal_error",
                "message": message,
            }
        })),
    )
}

fn pairing_url_or_default(configured: &str, scheme: &str, addr: &str) -> String {
    let configured = configured.trim().trim_end_matches('/');
    if !configured.is_empty() {
        return configured.to_string();
    }
    let host_port = addr
        .strip_prefix("0.0.0.0:")
        .map(|port| format!("127.0.0.1:{port}"))
        .unwrap_or_else(|| addr.to_string());
    format!("{scheme}://{host_port}")
}

fn pairing_status_ui_url(state: &AppState) -> String {
    let configured = state
        .config
        .pairing_status_ui_url
        .trim()
        .trim_end_matches('/');
    if !configured.is_empty() {
        return configured.to_string();
    }
    format!(
        "{}/#/capture",
        pairing_url_or_default(
            &state.config.pairing_edge_base_url,
            "http",
            &state.config.http_addr
        )
    )
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}
