use axum::extract::State;
use axum::http::header::AUTHORIZATION;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use ring::digest;
use serde::Serialize;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppState;

#[derive(Clone, Debug, Default)]
pub struct UploadAuthContext {
    pub auth_kind: String,
    pub device_id: Option<String>,
    pub device_name: Option<String>,
    pub login_identity: Option<String>,
    pub profile_id: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody {
    code: String,
    message: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    ok: bool,
    error: ErrorBody,
}

fn unauthorized(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            ok: false,
            error: ErrorBody {
                code: "unauthorized".to_string(),
                message: message.to_string(),
            },
        }),
    )
        .into_response()
}

pub(crate) fn extract_bearer_token(headers: &axum::http::HeaderMap) -> Option<String> {
    let raw = headers.get(AUTHORIZATION)?.to_str().ok()?;
    let raw = raw.trim();
    let bearer_prefix = "Bearer ";
    if raw.len() <= bearer_prefix.len() {
        return None;
    }
    if !raw.starts_with(bearer_prefix) && !raw.starts_with("bearer ") {
        return None;
    }
    Some(raw[bearer_prefix.len()..].trim().to_string())
}

fn is_upload_chunk_path(path: &str) -> bool {
    matches!(
        path,
        "/common_task/upload_chunk"
            | "/chunk/upload"
            | "/edge/common_task/upload_chunk"
            | "/edge/chunk/upload"
    )
}

fn is_public_path(path: &str) -> bool {
    path == "/"
        || path == "/index.html"
        || path == "/health"
        || path.starts_with("/assets/")
        || path == "/sensing"
        || path.starts_with("/sensing/")
        || path == "/sim-control"
        || path.starts_with("/sim-control/")
        || path == "/replay"
        || path.starts_with("/replay/")
}

pub(crate) async fn upload_bearer_authorized(
    state: &AppState,
    headers: &axum::http::HeaderMap,
    device_id: Option<&str>,
) -> Result<(), String> {
    upload_bearer_context(state, headers, device_id)
        .await
        .map(|_| ())
}

pub(crate) async fn upload_bearer_context(
    state: &AppState,
    headers: &axum::http::HeaderMap,
    device_id: Option<&str>,
) -> Result<UploadAuthContext, String> {
    let has_edge_token = state
        .config
        .edge_token
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    let has_scoped_authority = state.token_authority.is_some()
        || state
            .config
            .upload_token_registry_path
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty());
    let has_registry = state
        .config
        .upload_token_registry_path
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());

    if !has_edge_token && !has_scoped_authority {
        return Ok(UploadAuthContext {
            auth_kind: "disabled".to_string(),
            ..UploadAuthContext::default()
        });
    }

    let Some(actual) = extract_bearer_token(headers) else {
        return Err("缺少 Authorization: Bearer <edge_token 或 scoped_upload_token>".to_string());
    };

    if state
        .config
        .edge_token
        .as_deref()
        .is_some_and(|expected| actual == expected)
    {
        return Ok(UploadAuthContext {
            auth_kind: "edge_token".to_string(),
            device_id: device_id
                .and_then(normalize_non_empty)
                .map(ToOwned::to_owned),
            ..UploadAuthContext::default()
        });
    }

    if let Some(authority) = state.token_authority.as_ref() {
        match authority.validate_upload_token(&actual, device_id).await {
            Ok(context) => return Ok(context),
            Err(error) if has_registry && error == "scoped_upload_token 不匹配" => {}
            Err(error) => return Err(error),
        }
    }

    let Some(registry_path) = state.config.upload_token_registry_path.as_deref() else {
        return Err("edge_token 不匹配".to_string());
    };

    scoped_upload_token_context(registry_path, &actual, device_id).await
}

async fn scoped_upload_token_context(
    registry_path: &str,
    token: &str,
    device_id: Option<&str>,
) -> Result<UploadAuthContext, String> {
    let expected_hash = sha256_hex(token);
    let registry = tokio::fs::read_to_string(registry_path)
        .await
        .map_err(|error| format!("读取上传 token registry 失败: {error}"))?;
    let payload: Value = serde_json::from_str(&registry)
        .map_err(|error| format!("解析上传 token registry 失败: {error}"))?;
    let devices = registry_devices(&payload);
    let normalized_device_id = device_id.and_then(normalize_non_empty);
    let now_ms = now_unix_ms();

    for item in devices {
        if let Some(expected_device_id) = normalized_device_id {
            let item_device_id = item
                .get("device_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or("");
            if item_device_id != expected_device_id {
                continue;
            }
        }
        let Some(stored_hash) = item
            .get("upload_token_sha256")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        if !stored_hash.eq_ignore_ascii_case(&expected_hash) {
            continue;
        }
        let expires_ms = item
            .get("token_expires_unix_ms")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        if expires_ms <= now_ms {
            return Err("scoped_upload_token 已过期".to_string());
        }
        if item
            .get("upload_token_status")
            .and_then(Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("expired"))
        {
            return Err("scoped_upload_token 已标记过期".to_string());
        }
        return Ok(UploadAuthContext {
            auth_kind: "scoped_upload_token".to_string(),
            device_id: string_field(item, "device_id"),
            device_name: string_field(item, "device_name"),
            login_identity: string_field(item, "login_identity"),
            profile_id: string_field(item, "profile_id"),
        });
    }

    if normalized_device_id.is_some() {
        Err("scoped_upload_token 与 device_id 不匹配".to_string())
    } else {
        Err("scoped_upload_token 不匹配".to_string())
    }
}

fn string_field(item: &Value, key: &str) -> Option<String> {
    item.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn registry_devices(payload: &Value) -> Vec<&Value> {
    match payload.get("devices") {
        Some(Value::Array(items)) => items.iter().filter(|item| item.is_object()).collect(),
        Some(Value::Object(map)) => map.values().filter(|item| item.is_object()).collect(),
        _ => Vec::new(),
    }
}

fn sha256_hex(value: &str) -> String {
    let digest = digest::digest(&digest::SHA256, value.as_bytes());
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{:02x}", *byte))
        .collect()
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn normalize_non_empty(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// HTTP 鉴权中间件：工作站静态入口、公共代理和 `/health` 公开，其余控制面默认要求 `Authorization: Bearer <EDGE_TOKEN>`。
pub async fn require_http_auth(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let Some(expected) = state.config.edge_token.as_deref() else {
        return next.run(req).await;
    };

    let path = req.uri().path();
    if is_public_path(path) {
        return next.run(req).await;
    }
    if state.config.metrics_public && path == "/metrics" {
        return next.run(req).await;
    }
    if is_upload_chunk_path(path) {
        if let Err(message) = upload_bearer_authorized(&state, req.headers(), None).await {
            return unauthorized(&message);
        }
        return next.run(req).await;
    }

    let Some(actual) = extract_bearer_token(req.headers()) else {
        return unauthorized("缺少 Authorization: Bearer <edge_token>");
    };
    if actual != expected {
        return unauthorized("edge_token 不匹配");
    }

    next.run(req).await
}

/// WS 鉴权：允许走 `Authorization: Bearer <edge_token>` 或 `?token=<edge_token>`。
pub fn ws_authorized(
    state: &AppState,
    token_query: Option<&str>,
    headers: &axum::http::HeaderMap,
) -> bool {
    let Some(expected) = state.config.edge_token.as_deref() else {
        return true;
    };

    if let Some(actual) = extract_bearer_token(headers) {
        return actual == expected;
    }
    token_query.is_some_and(|t| !t.is_empty() && t == expected)
}

pub fn ws_unauthorized_response() -> Response {
    unauthorized("WS 未鉴权：请携带 Authorization: Bearer <edge_token> 或 ?token=<edge_token>")
}
