use axum::extract::State;
use axum::http::header::AUTHORIZATION;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

use crate::AppState;

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

fn extract_bearer_token(headers: &axum::http::HeaderMap) -> Option<String> {
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
