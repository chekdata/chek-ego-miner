use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use axum::body::{to_bytes, Body};
use axum::extract::State;
use axum::http::{header, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{routing::any, Json, Router};
use serde_json::{json, Map, Value};
use tokio::process::Command;

use crate::calibration::IphoneStereoExtrinsic;
use crate::path_safety;
use crate::sensing::{StereoTrackedPersonSnapshot, VisionDevicePose};
use crate::{AppState, SimTrackingCarrySnapshot};

const MAX_PROXY_BODY_BYTES: usize = 64 * 1024 * 1024;
const DEPTH_CHUNK_MAGIC: &[u8; 8] = b"CHEKDEP1";
const DEPTH_PREVIEW_MAX_DIM: usize = 48;
const SIM_TRACKING_TRANSIENT_CARRY_MS: u64 = 2_500;
const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", any(serve_index))
        .route("/index.html", any(serve_index))
        .route("/assets/*path", any(serve_asset))
        .route("/observatory.html", any(serve_dist_file))
        .route("/observatory/*path", any(serve_dist_file))
        .route("/stereo-preview.jpg", any(serve_stereo_preview))
        .route("/stereo-watchdog.json", any(serve_stereo_watchdog))
        .route("/live-preview.json", any(serve_live_preview_summary))
        .route("/teleop/latest.json", any(serve_latest_teleop))
        .route("/retarget/latest.json", any(serve_latest_retarget))
        .route("/live-preview/file/*path", any(serve_live_preview_file))
        .route("/sensing", any(proxy_sensing))
        .route("/sensing/*path", any(proxy_sensing))
        .route("/sim-control", any(proxy_sim_control))
        .route("/sim-control/*path", any(proxy_sim_control))
        .route("/replay", any(proxy_replay))
        .route("/replay/*path", any(proxy_replay))
        .with_state(state)
}

pub fn resolve_ui_dist_dir(state: &AppState) -> Option<PathBuf> {
    let dist_dir = PathBuf::from(state.config.ui_dist_dir.trim());
    let index = dist_dir.join("index.html");
    if index.exists() && index.is_file() {
        Some(dist_dir)
    } else {
        None
    }
}

async fn serve_index(State(state): State<AppState>, request: Request<Body>) -> Response {
    let Some(dist_dir) = resolve_ui_dist_dir(&state) else {
        return (StatusCode::NOT_FOUND, "工作站 dist 未找到").into_response();
    };
    let index_path = dist_dir.join("index.html");
    serve_file(index_path, request.method() == Method::HEAD, false).await
}

async fn serve_asset(State(state): State<AppState>, request: Request<Body>) -> Response {
    serve_dist_path(state, request, true).await
}

async fn serve_dist_file(State(state): State<AppState>, request: Request<Body>) -> Response {
    serve_dist_path(state, request, false).await
}

async fn serve_dist_path(state: AppState, request: Request<Body>, cache_assets: bool) -> Response {
    let Some(dist_dir) = resolve_ui_dist_dir(&state) else {
        return (StatusCode::NOT_FOUND, "工作站 dist 未找到").into_response();
    };
    let relative = request.uri().path().trim_start_matches('/');
    let Some(asset_path) = safe_asset_path(&dist_dir, relative) else {
        return (StatusCode::NOT_FOUND, "静态资源不存在").into_response();
    };
    serve_file(asset_path, request.method() == Method::HEAD, cache_assets).await
}

async fn serve_stereo_preview(State(state): State<AppState>, request: Request<Body>) -> Response {
    let preview_path = PathBuf::from(state.config.stereo_preview_path.trim());
    serve_file(preview_path, request.method() == Method::HEAD, false).await
}

async fn serve_stereo_watchdog(State(state): State<AppState>) -> Response {
    Json(build_stereo_watchdog_payload(&state).await).into_response()
}

async fn serve_live_preview_summary(State(state): State<AppState>) -> Response {
    Json(build_live_preview_payload(&state).await).into_response()
}

async fn serve_latest_teleop(State(state): State<AppState>) -> Response {
    let current_session = state.session.snapshot();
    let edge_time_ns_now = state.gate.edge_time_ns();
    let payload = match state.teleop_latest.lock() {
        Ok(guard) => match guard.as_ref() {
            Some(frame)
                if teleop_frame_is_current(
                    &current_session.trip_id,
                    &current_session.session_id,
                    edge_time_ns_now,
                    frame,
                    state.config.teleop_publish_hz.max(1),
                ) =>
            {
                json!({
                    "available": true,
                    "teleop_frame": frame,
                })
            }
            Some(_) => json!({
                "available": false,
                "reason": "stale_teleop_frame",
            }),
            None => json!({
                "available": false,
                "reason": "no_teleop_frame",
            }),
        },
        Err(_) => json!({
            "available": false,
            "reason": "teleop_frame_lock_failed",
        }),
    };
    Json(payload).into_response()
}

async fn serve_latest_retarget(State(state): State<AppState>) -> Response {
    let current_session = state.session.snapshot();
    let edge_time_ns_now = state.gate.edge_time_ns();
    let live_session = build_live_session_context(
        &state,
        &current_session.trip_id,
        &current_session.session_id,
    );
    let operator = state.operator.snapshot(
        state
            .config
            .vision_stale_ms
            .max(state.config.stereo_stale_ms),
    );
    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let wifi_pose = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let iphone_stereo_calibration = state.iphone_stereo_calibration.snapshot();
    let resolved_target = resolve_target_human_state(
        live_session.as_ref(),
        &operator,
        &vision,
        &stereo,
        &wifi_pose,
        iphone_stereo_calibration.as_ref(),
        edge_time_ns_now,
    );
    let payload = match state.retarget_latest.lock() {
        Ok(guard) => match guard.as_ref() {
            Some(reference)
                if retarget_reference_is_current(
                    &current_session.trip_id,
                    &current_session.session_id,
                    edge_time_ns_now,
                    reference,
                    state.config.teleop_publish_hz.max(1),
                ) =>
            {
                let mut reference = reference.clone();
                if current_session.session_id == reference.source_session_id {
                    if let Some(target_person_id) = resolved_target.target_person_id.clone() {
                        reference.target_person_id = target_person_id;
                    }
                }
                json!({
                    "available": true,
                    "retarget_reference": reference,
                })
            }
            Some(_) => json!({
                "available": false,
                "reason": "stale_retarget_reference",
            }),
            None => json!({
                "available": false,
                "reason": "no_retarget_reference",
            }),
        },
        Err(_) => json!({
            "available": false,
            "reason": "retarget_reference_lock_failed",
        }),
    };
    Json(payload).into_response()
}

fn teleop_frame_is_current(
    current_trip_id: &str,
    current_session_id: &str,
    edge_time_ns_now: u64,
    frame: &crate::ws::types::TeleopFrameV1,
    publish_hz: u32,
) -> bool {
    if current_session_id.trim().is_empty() {
        return false;
    }
    if frame.session_id != current_session_id {
        return false;
    }
    if !current_trip_id.trim().is_empty() && frame.trip_id != current_trip_id {
        return false;
    }
    cached_edge_time_is_fresh(edge_time_ns_now, frame.edge_time_ns, publish_hz)
}

fn retarget_reference_is_current(
    current_trip_id: &str,
    current_session_id: &str,
    edge_time_ns_now: u64,
    reference: &crate::ws::types::RetargetReferenceV1,
    publish_hz: u32,
) -> bool {
    if current_session_id.trim().is_empty() {
        return false;
    }
    if reference.session_id != current_session_id
        || reference.source_session_id != current_session_id
    {
        return false;
    }
    if !current_trip_id.trim().is_empty() && reference.trip_id != current_trip_id {
        return false;
    }
    cached_edge_time_is_fresh_with_floor(edge_time_ns_now, reference.edge_time_ns, publish_hz, 8000)
}

fn cached_edge_time_is_fresh(edge_time_ns_now: u64, edge_time_ns: u64, publish_hz: u32) -> bool {
    cached_edge_time_is_fresh_with_floor(edge_time_ns_now, edge_time_ns, publish_hz, 800)
}

fn cached_edge_time_is_fresh_with_floor(
    edge_time_ns_now: u64,
    edge_time_ns: u64,
    publish_hz: u32,
    stale_floor_ms: u64,
) -> bool {
    if edge_time_ns == 0 || edge_time_ns_now < edge_time_ns {
        return false;
    }
    let period_ms = (1000 / publish_hz.max(1)) as u64;
    let stale_after_ms = period_ms.saturating_mul(8).max(stale_floor_ms);
    let age_ms = (edge_time_ns_now - edge_time_ns) / 1_000_000;
    age_ms <= stale_after_ms
}

async fn serve_live_preview_file(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Response {
    let live_session = resolve_live_session(&state);
    let Some(context) = live_session.context else {
        return (StatusCode::NOT_FOUND, "当前没有活跃 session").into_response();
    };
    let relative = request
        .uri()
        .path()
        .trim_start_matches("/live-preview/file/");
    let Some(path) = safe_asset_path(&context.session_dir, relative) else {
        return (StatusCode::NOT_FOUND, "预览文件不存在").into_response();
    };
    serve_file(path, request.method() == Method::HEAD, false).await
}

async fn proxy_sensing(State(state): State<AppState>, request: Request<Body>) -> Response {
    proxy_request(
        &state.http_client,
        state.config.sensing_proxy_base.trim(),
        "/sensing",
        request,
        "sensing",
    )
    .await
}

async fn proxy_sim_control(State(state): State<AppState>, request: Request<Body>) -> Response {
    proxy_request(
        &state.http_client,
        state.config.sim_control_proxy_base.trim(),
        "/sim-control",
        request,
        "sim-control",
    )
    .await
}

async fn proxy_replay(State(state): State<AppState>, request: Request<Body>) -> Response {
    proxy_request(
        &state.http_client,
        state.config.replay_proxy_base.trim(),
        "/replay",
        request,
        "replay",
    )
    .await
}

async fn proxy_request(
    client: &reqwest::Client,
    upstream_base: &str,
    prefix: &str,
    request: Request<Body>,
    target_name: &str,
) -> Response {
    if upstream_base.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("{target_name} upstream 未配置"),
        )
            .into_response();
    }

    let (parts, body) = request.into_parts();
    let path = parts.uri.path();
    let stripped_path = path.strip_prefix(prefix).unwrap_or(path);
    let normalized_path = if stripped_path.is_empty() {
        "/"
    } else {
        stripped_path
    };
    let mut upstream_url = format!("{}{}", upstream_base.trim_end_matches('/'), normalized_path);
    if let Some(query) = parts.uri.query() {
        upstream_url.push('?');
        upstream_url.push_str(query);
    }

    let payload = match to_bytes(body, MAX_PROXY_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, format!("读取请求体失败: {error}")).into_response();
        }
    };

    let mut upstream_request = client.request(parts.method, upstream_url);
    for (name, value) in &parts.headers {
        let header_name = name.as_str().to_ascii_lowercase();
        if HOP_BY_HOP_HEADERS.contains(&header_name.as_str()) {
            continue;
        }
        upstream_request = upstream_request.header(name, value);
    }
    if !payload.is_empty() {
        upstream_request = upstream_request.body(payload.to_vec());
    }

    match upstream_request.send().await {
        Ok(upstream_response) => {
            let status = upstream_response.status();
            let headers = upstream_response.headers().clone();
            let body = match upstream_response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    return (
                        StatusCode::BAD_GATEWAY,
                        format!("{target_name} 响应读取失败: {error}"),
                    )
                        .into_response();
                }
            };

            let mut response_builder = Response::builder().status(status);
            for (name, value) in &headers {
                let header_name = name.as_str().to_ascii_lowercase();
                if HOP_BY_HOP_HEADERS.contains(&header_name.as_str()) {
                    continue;
                }
                response_builder = response_builder.header(name, value);
            }
            response_builder
                .body(Body::from(body))
                .unwrap_or_else(|error| {
                    (
                        StatusCode::BAD_GATEWAY,
                        format!("{target_name} 响应构造失败: {error}"),
                    )
                        .into_response()
                })
        }
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            format!("{target_name} upstream 不可用: {error}"),
        )
            .into_response(),
    }
}

fn safe_asset_path(dist_dir: &Path, relative: &str) -> Option<PathBuf> {
    let relative_path = Path::new(relative);
    if relative_path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return None;
    }

    let candidate = dist_dir.join(relative_path);
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
struct LiveSessionContext {
    trip_id: String,
    session_id: String,
    session_dir: PathBuf,
}

#[derive(Clone, Debug)]
struct LiveSessionResolution {
    context: Option<LiveSessionContext>,
    source: &'static str,
    reason: Option<String>,
    phone_ingress_effective_trip_id: String,
    phone_ingress_effective_session_id: String,
    vision_metrics_trip_id: String,
    vision_metrics_session_id: String,
}

const SESSION_RESOLUTION_FALLBACK_MAX_AGE_MS: u64 = 15_000;
const STARTUP_STEREO_ASSOCIATION_GRACE_MS: u64 = 20_000;
const RUNTIME_STEREO_ASSOCIATION_GAP_HOLD_MS: u64 = 6_000;

fn build_live_session_context(
    state: &AppState,
    trip_id: &str,
    session_id: &str,
) -> Option<LiveSessionContext> {
    let session_id = path_safety::validate_path_component(session_id, "session_id").ok()?;
    let trip_id = if trip_id.trim().is_empty() {
        session_id.clone()
    } else {
        trip_id.trim().to_string()
    };
    let session_root = PathBuf::from(state.config.data_dir.trim()).join("session");
    let session_dir = session_root.join(&session_id);
    Some(LiveSessionContext {
        trip_id,
        session_id,
        session_dir,
    })
}

fn edge_time_recent(
    edge_time_now_ns: u64,
    sample_edge_time_ns: Option<u64>,
    max_age_ms: u64,
) -> bool {
    let Some(sample_edge_time_ns) = sample_edge_time_ns else {
        return false;
    };
    if edge_time_now_ns == 0 || sample_edge_time_ns == 0 || edge_time_now_ns < sample_edge_time_ns {
        return false;
    }
    let age_ms = (edge_time_now_ns - sample_edge_time_ns) / 1_000_000;
    age_ms <= max_age_ms
}

fn edge_time_within_ms(edge_time_now_ns: u64, max_age_ms: u64) -> bool {
    edge_time_now_ns > 0 && edge_time_now_ns <= max_age_ms.saturating_mul(1_000_000)
}

fn resolve_live_session(state: &AppState) -> LiveSessionResolution {
    let snapshot = state.session.snapshot();
    if let Some(context) =
        build_live_session_context(state, &snapshot.trip_id, &snapshot.session_id)
    {
        return LiveSessionResolution {
            context: Some(context),
            source: "session_store",
            reason: None,
            phone_ingress_effective_trip_id: String::new(),
            phone_ingress_effective_session_id: String::new(),
            vision_metrics_trip_id: String::new(),
            vision_metrics_session_id: String::new(),
        };
    }

    let edge_time_now_ns = state.gate.edge_time_ns();
    let phone_ingress = state.phone_ingress_status.snapshot();
    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let phone_effective_trip_id = phone_ingress.effective_trip_id.trim().to_string();
    let phone_effective_session_id = phone_ingress.effective_session_id.trim().to_string();
    let vision_metrics_trip_id = vision.metrics_trip_id.trim().to_string();
    let vision_metrics_session_id = vision.metrics_session_id.trim().to_string();

    let phone_fallback_recent = edge_time_recent(
        edge_time_now_ns,
        phone_ingress.last_attempt_edge_time_ns,
        SESSION_RESOLUTION_FALLBACK_MAX_AGE_MS,
    );
    if phone_fallback_recent && !phone_effective_session_id.is_empty() {
        if let Some(context) =
            build_live_session_context(state, &phone_effective_trip_id, &phone_effective_session_id)
        {
            return LiveSessionResolution {
                context: Some(context),
                source: "phone_ingress_effective",
                reason: Some(
                    "session store 为空，回退到最近一次 phone ingress effective session"
                        .to_string(),
                ),
                phone_ingress_effective_trip_id: phone_effective_trip_id,
                phone_ingress_effective_session_id: phone_effective_session_id,
                vision_metrics_trip_id,
                vision_metrics_session_id,
            };
        }
    }

    let vision_fallback_recent = !vision_metrics_session_id.is_empty()
        && (vision.fresh
            || vision.body_3d_recent
            || vision.hand_3d_recent
            || edge_time_recent(
                edge_time_now_ns,
                Some(vision.last_edge_time_ns),
                SESSION_RESOLUTION_FALLBACK_MAX_AGE_MS,
            ));
    if vision_fallback_recent {
        if let Some(context) =
            build_live_session_context(state, &vision_metrics_trip_id, &vision_metrics_session_id)
        {
            return LiveSessionResolution {
                context: Some(context),
                source: "vision_metrics",
                reason: Some(
                    "session store 为空，回退到最近一次 vision metrics session".to_string(),
                ),
                phone_ingress_effective_trip_id: phone_effective_trip_id,
                phone_ingress_effective_session_id: phone_effective_session_id,
                vision_metrics_trip_id,
                vision_metrics_session_id,
            };
        }
    }

    let reason = if !phone_effective_session_id.is_empty() {
        Some("session store 为空，且最近一次 phone ingress effective session 已过期".to_string())
    } else if !vision_metrics_session_id.is_empty() {
        Some("session store 为空，且最近一次 vision metrics session 已过期".to_string())
    } else {
        Some("当前还没有 active session，也没有可回退的 recent phone/vision session".to_string())
    };

    LiveSessionResolution {
        context: None,
        source: "none",
        reason,
        phone_ingress_effective_trip_id: phone_effective_trip_id,
        phone_ingress_effective_session_id: phone_effective_session_id,
        vision_metrics_trip_id,
        vision_metrics_session_id,
    }
}

async fn build_live_preview_payload(state: &AppState) -> Value {
    let operator = state.operator.snapshot(state.config.operator_hold_ms);
    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let wifi_pose = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let csi = state.csi.snapshot(state.config.csi_stale_ms);
    let phone_ingress = state.phone_ingress_status.snapshot();
    let association_hint_clients = state.association_hint_clients.snapshot();
    let fusion_stream_clients = state.fusion_stream_clients.snapshot();
    let iphone_stereo_calibration = state.iphone_stereo_calibration.snapshot();
    let live_session_resolution = resolve_live_session(state);
    let edge_time_now_ns = state.gate.edge_time_ns();
    let live_session = live_session_resolution.context.as_ref();
    let session_dir = live_session.map(|context| &context.session_dir);
    let main = build_live_media_track(session_dir, "iphone", "main").await;
    let aux = build_live_media_track(session_dir, "iphone", "aux").await;
    let depth = build_live_media_track(session_dir, "iphone", "depth").await;
    let fisheye = build_live_media_track(session_dir, "iphone", "fisheye").await;
    let task_semantics = build_live_task_semantics(session_dir).await;
    let vlm_summary = build_live_vlm_summary(
        session_dir,
        state.config.vlm_indexing_enabled,
        state.config.preview_generation_enabled,
    )
    .await;
    let target_human_state = build_target_human_state_payload(
        live_session,
        &operator,
        &vision,
        &stereo,
        &wifi_pose,
        &phone_ingress,
        iphone_stereo_calibration.as_ref(),
        edge_time_now_ns,
    );
    let scene_state = build_scene_state_payload(
        live_session,
        &operator,
        &vision,
        &stereo,
        &wifi_pose,
        &csi,
        iphone_stereo_calibration.as_ref(),
        edge_time_now_ns,
    );
    let source_session_id = target_human_state
        .get("source_session_id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty());
    let target_person_id = target_human_state
        .get("target_person_id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty());
    let target_edge_time_ns = target_human_state
        .get("source_edge_time_ns")
        .and_then(|value| value.as_u64());
    let control_deadman = build_control_deadman_payload(state);
    let sim_tracking_state = build_sim_tracking_state_payload(
        state,
        source_session_id,
        target_person_id,
        target_edge_time_ns,
    )
    .await;

    serde_json::json!({
        "trip_id": live_session.as_ref().map(|context| context.trip_id.clone()).unwrap_or_default(),
        "session_id": live_session.as_ref().map(|context| context.session_id.clone()).unwrap_or_default(),
        "runtime_profile": state.config.runtime_profile_name(),
        "upload_policy_mode": state.config.upload_policy_mode_name(),
        "raw_residency_default": state.config.raw_residency_default(),
        "preview_residency_default": state.config.preview_residency_default(),
        "feature_flags": state.config.runtime_feature_flags(),
        "crowd_upload_enabled": state.config.crowd_upload_enabled,
        "session_resolution": {
            "source": live_session_resolution.source,
            "reason": live_session_resolution.reason,
            "phone_ingress_effective_trip_id": non_empty_string_value(&live_session_resolution.phone_ingress_effective_trip_id),
            "phone_ingress_effective_session_id": non_empty_string_value(&live_session_resolution.phone_ingress_effective_session_id),
            "vision_metrics_trip_id": non_empty_string_value(&live_session_resolution.vision_metrics_trip_id),
            "vision_metrics_session_id": non_empty_string_value(&live_session_resolution.vision_metrics_session_id),
        },
        "target_human_state": target_human_state,
        "scene_state": scene_state,
        "client_origins": build_client_origins_payload(&association_hint_clients, &fusion_stream_clients),
        "control_deadman": control_deadman,
        "tracks": [main, aux, depth, fisheye],
        "sim_tracking_state": sim_tracking_state,
        "task_semantics": task_semantics,
        "vlm_summary": vlm_summary,
    })
}

fn build_client_origins_payload(
    association_hint: &crate::control::gate::AssociationHintClientSnapshot,
    fusion_stream: &crate::control::gate::FusionClientSnapshot,
) -> Value {
    json!({
        "association_hint": {
            "request_count": association_hint.request_count,
            "last_request_edge_time_ns": association_hint.last_request_edge_time_ns,
            "client_addr": non_empty_string_value(&association_hint.client_addr),
            "forwarded_for": non_empty_string_value(&association_hint.forwarded_for),
            "user_agent": non_empty_string_value(&association_hint.user_agent),
        },
        "fusion_stream": {
            "active_count": fusion_stream.active_count,
            "total_connections": fusion_stream.total_connections,
            "last_connect_edge_time_ns": fusion_stream.last_connect_edge_time_ns,
            "last_disconnect_edge_time_ns": fusion_stream.last_disconnect_edge_time_ns,
            "last_client_addr": non_empty_string_value(&fusion_stream.last_client_addr),
            "last_forwarded_for": non_empty_string_value(&fusion_stream.last_forwarded_for),
            "last_user_agent": non_empty_string_value(&fusion_stream.last_user_agent),
            "last_transport": non_empty_string_value(&fusion_stream.last_transport),
            "last_format": non_empty_string_value(&fusion_stream.last_format),
            "last_compression": non_empty_string_value(&fusion_stream.last_compression),
            "last_operator_debug": fusion_stream.last_operator_debug,
            "active_clients": fusion_stream.active_clients.iter().map(|client| json!({
                "connection_id": client.connection_id,
                "connected_edge_time_ns": client.connected_edge_time_ns,
                "client_addr": non_empty_string_value(&client.client_addr),
                "forwarded_for": non_empty_string_value(&client.forwarded_for),
                "user_agent": non_empty_string_value(&client.user_agent),
                "transport": non_empty_string_value(&client.transport),
                "format": non_empty_string_value(&client.format),
                "compression": non_empty_string_value(&client.compression),
                "operator_debug": client.operator_debug,
            })).collect::<Vec<_>>(),
        }
    })
}

struct ResolvedTargetHumanState {
    has_target: bool,
    fresh: bool,
    target_person_id: Option<String>,
    source: Option<String>,
    source_edge_time_ns: Option<u64>,
    body_kpts_3d: Vec<[f32; 3]>,
    hand_kpts_3d: Vec<[f32; 3]>,
    body_3d_source: String,
    hand_3d_source: String,
    association_anchor_source: String,
    uses_operator_estimate: bool,
    iphone_track_id: Option<String>,
    stereo_track_id: Option<String>,
    wifi_track_id: Option<String>,
    association_confidence: f32,
}

fn target_authority_source(has_target: bool) -> Option<&'static str> {
    has_target.then_some("phone_session")
}

fn non_empty_string_value(value: &str) -> Value {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Value::Null
    } else {
        Value::String(trimmed.to_string())
    }
}

fn build_control_deadman_payload(state: &AppState) -> Value {
    let session = state.session.snapshot();
    let deadman = state.gate.deadman_snapshot();
    let session_matches_active = !session.trip_id.is_empty()
        && !session.session_id.is_empty()
        && session.trip_id == deadman.keepalive.last_trip_id
        && session.session_id == deadman.keepalive.last_session_id;

    json!({
        "enabled": deadman.enabled,
        "timeout_ms": deadman.timeout_ms,
        "link_ok": deadman.link_ok,
        "pressed": deadman.pressed,
        "keepalive": {
            "last_age_ms": deadman.keepalive.last_age_ms,
            "last_edge_time_ns": deadman.keepalive.last_edge_time_ns,
            "last_source_time_ns": deadman.keepalive.last_source_time_ns,
            "last_seq": deadman.keepalive.last_seq,
            "last_trip_id": non_empty_string_value(&deadman.keepalive.last_trip_id),
            "last_session_id": non_empty_string_value(&deadman.keepalive.last_session_id),
            "last_device_id": non_empty_string_value(&deadman.keepalive.last_device_id),
            "packets_in_last_5s": deadman.keepalive.packets_in_last_5s,
            "approx_rate_hz_5s": deadman.keepalive.approx_rate_hz_5s,
            "session_matches_active": session_matches_active,
        }
    })
}

fn target_association_source(
    has_target: bool,
    association_anchor_source: &str,
    stereo_track_id: Option<&str>,
    hand_match_score: f32,
    iphone_visible_hand_count: usize,
) -> Option<&'static str> {
    if !has_target {
        return None;
    }
    if stereo_track_id.is_some() {
        if association_anchor_source.contains("iphone_hand")
            || (iphone_visible_hand_count > 0 && hand_match_score > 0.0)
        {
            return Some("phone_hand_ego_to_stereo_track");
        }
        return Some("phone_session_to_stereo_track");
    }
    Some("phone_direct")
}

fn target_association_basis(
    has_target: bool,
    association_anchor_source: &str,
    stereo_track_id: Option<&str>,
    wifi_track_id: Option<&str>,
    iphone_visible_hand_count: usize,
    hand_match_score: f32,
    left_wrist_gap_m: Option<f32>,
    right_wrist_gap_m: Option<f32>,
) -> Vec<String> {
    if !has_target {
        return Vec::new();
    }
    let mut basis = vec!["phone_session".to_string()];
    if iphone_visible_hand_count > 0 {
        basis.push("phone_hand_visible".to_string());
    } else if association_anchor_source.contains("iphone_hand") {
        basis.push("phone_hand_track_sticky".to_string());
    }
    if stereo_track_id.is_some() {
        basis.push("stereo_track_linked".to_string());
    }
    if wifi_track_id.is_some() {
        basis.push("wifi_track_linked".to_string());
    }
    if hand_match_score > 0.0 {
        basis.push("hand_motion_correlation".to_string());
    }
    if left_wrist_gap_m.is_some() || right_wrist_gap_m.is_some() {
        basis.push("wrist_alignment".to_string());
    }
    basis
}

struct PhoneAuthoritativeState {
    session_aligned: bool,
    recent: bool,
    track_id: Option<String>,
}

fn vision_has_authoritative_points(vision: &crate::sensing::VisionSnapshot) -> bool {
    !vision.body_kpts_2d.is_empty()
        || !vision.hand_kpts_2d.is_empty()
        || !vision.body_kpts_3d.is_empty()
        || !vision.hand_kpts_3d.is_empty()
}

fn stereo_derived_phone_geometry_source(source: &str) -> bool {
    let trimmed = source.trim();
    !trimmed.is_empty()
        && trimmed != "none"
        && (trimmed.contains("edge_depth_reprojected") || trimmed.contains("stereo"))
}

fn resolve_phone_authoritative_state(
    live_session: Option<&LiveSessionContext>,
    vision: &crate::sensing::VisionSnapshot,
) -> PhoneAuthoritativeState {
    let phone_track = vision
        .operator_track_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| Some("primary_operator".to_string()));
    let phone_has_points = vision_has_authoritative_points(vision);
    let phone_session_matches = live_session.is_some_and(|context| {
        let metrics_session_id = vision.metrics_session_id.trim();
        !metrics_session_id.is_empty() && metrics_session_id == context.session_id.as_str()
    });
    let phone_execution_mode_matches = live_session.is_some()
        && vision.execution_mode.trim() == "edge_authoritative_phone_vision"
        && phone_has_points;
    let phone_session_aligned = phone_session_matches || phone_execution_mode_matches;

    PhoneAuthoritativeState {
        session_aligned: phone_session_aligned,
        recent: phone_has_points
            && (vision.fresh || vision.body_3d_recent || vision.hand_3d_recent),
        track_id: phone_track,
    }
}

fn resolve_target_human_state(
    live_session: Option<&LiveSessionContext>,
    operator: &crate::operator::OperatorSnapshot,
    vision: &crate::sensing::VisionSnapshot,
    stereo: &crate::sensing::StereoSnapshot,
    wifi_pose: &crate::sensing::WifiPoseSnapshot,
    iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>,
    edge_time_now_ns: u64,
) -> ResolvedTargetHumanState {
    let Some(live_session) = live_session else {
        return ResolvedTargetHumanState {
            has_target: false,
            fresh: false,
            target_person_id: None,
            source: None,
            source_edge_time_ns: None,
            body_kpts_3d: Vec::new(),
            hand_kpts_3d: Vec::new(),
            body_3d_source: "none".to_string(),
            hand_3d_source: "none".to_string(),
            association_anchor_source: String::new(),
            uses_operator_estimate: false,
            iphone_track_id: None,
            stereo_track_id: None,
            wifi_track_id: None,
            association_confidence: 0.0,
        };
    };

    let phone = resolve_phone_authoritative_state(Some(live_session), vision);
    let estimate = &operator.estimate;
    let device_pose_stereo_candidate_track_id = resolve_device_pose_stereo_candidate(
        vision.device_pose.as_ref(),
        stereo,
        iphone_stereo_calibration,
    );
    let associated_stereo_track_id = estimate
        .association
        .stereo_operator_track_id
        .clone()
        .or_else(|| {
            estimate
                .association
                .anchor_source
                .contains("stereo")
                .then(|| estimate.association.selected_operator_track_id.clone())
                .flatten()
        })
        .or_else(|| {
            stereo
                .operator_track_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .or_else(|| {
            (stereo.persons.len() == 1)
                .then(|| stereo.persons.first())
                .flatten()
                .and_then(|person| person.operator_track_id.as_deref())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .or(device_pose_stereo_candidate_track_id.clone());
    let matched_stereo_person = associated_stereo_track_id.as_ref().and_then(|track_id| {
        stereo
            .persons
            .iter()
            .find(|person| person.operator_track_id.as_deref() == Some(track_id.as_str()))
    });
    let phone_body_has_valid_geometry = vision.body_kpts_3d.iter().copied().any(valid_point_3d);
    let phone_has_valid_geometry =
        phone_body_has_valid_geometry || vision.hand_kpts_3d.iter().copied().any(valid_point_3d);
    let phone_has_stereo_derived_geometry =
        stereo_derived_phone_geometry_source(&vision.body_3d_source)
            || stereo_derived_phone_geometry_source(&vision.hand_3d_source);
    let has_phone_session_stereo_hint = estimate.association.iphone_operator_track_id.is_some()
        || estimate.association.anchor_source.contains("iphone_hand")
        || device_pose_stereo_candidate_track_id.is_some()
        || estimate.association.stereo_operator_track_id.is_some()
        || (estimate.association.anchor_source.contains("stereo")
            && estimate.association.selected_operator_track_id.is_some());
    let stereo_startup_warmup_hold = phone.session_aligned
        && phone.recent
        && associated_stereo_track_id.is_none()
        && !stereo.fresh
        && edge_time_within_ms(edge_time_now_ns, STARTUP_STEREO_ASSOCIATION_GRACE_MS);
    let runtime_stereo_association_gap_hold = phone.session_aligned
        && phone.recent
        && associated_stereo_track_id.is_none()
        && (has_phone_session_stereo_hint || phone_has_stereo_derived_geometry)
        && phone_has_valid_geometry
        && edge_time_recent(
            edge_time_now_ns,
            Some(
                estimate
                    .updated_edge_time_ns
                    .max(stereo.last_edge_time_ns)
                    .max(vision.last_edge_time_ns),
            ),
            RUNTIME_STEREO_ASSOCIATION_GAP_HOLD_MS,
        );
    let phone_associated_to_stereo = phone.session_aligned
        && phone.recent
        && (!estimate.raw_pose.body_kpts_3d.is_empty()
            || matched_stereo_person.is_some()
            || phone_has_valid_geometry)
        && associated_stereo_track_id.is_some()
        && (has_phone_session_stereo_hint || phone_has_stereo_derived_geometry);
    if phone_associated_to_stereo {
        let operator_source_edge_time_ns = estimate
            .raw_pose
            .source_edge_time_ns
            .max(estimate.updated_edge_time_ns);
        let operator_body_source_edge_time_ns = crate::operator::part_source_edge_time_ns(
            estimate.fusion_breakdown.body_source,
            stereo.last_edge_time_ns,
            vision.last_edge_time_ns,
            wifi_pose.last_edge_time_ns,
            operator_source_edge_time_ns,
        );
        let operator_hand_source_edge_time_ns = crate::operator::part_source_edge_time_ns(
            estimate.fusion_breakdown.hand_source,
            stereo.last_edge_time_ns,
            vision.last_edge_time_ns,
            wifi_pose.last_edge_time_ns,
            operator_source_edge_time_ns,
        );
        let operator_body_selected = !estimate.raw_pose.body_kpts_3d.is_empty();
        let stereo_hand_has_valid_geometry = matched_stereo_person
            .map(|person| person.hand_kpts_3d.iter().copied().any(valid_point_3d))
            .unwrap_or(false);
        let operator_hand_selected = estimate
            .operator_state
            .hand_kpts_3d
            .iter()
            .copied()
            .any(valid_point_3d)
            && estimate.fusion_breakdown.hand_source != crate::operator::OperatorPartSource::None;
        let stereo_body_selected = !operator_body_selected && matched_stereo_person.is_some();
        let vision_body_selected = !operator_body_selected
            && matched_stereo_person.is_none()
            && phone_body_has_valid_geometry;
        let vision_hand_selected = !operator_hand_selected
            && !stereo_hand_has_valid_geometry
            && vision.hand_kpts_3d.iter().copied().any(valid_point_3d);
        let hand_kpts_3d = if estimate
            .operator_state
            .hand_kpts_3d
            .iter()
            .copied()
            .any(valid_point_3d)
        {
            estimate.operator_state.hand_kpts_3d.clone()
        } else if let Some(person) = matched_stereo_person
            .filter(|person| person.hand_kpts_3d.iter().copied().any(valid_point_3d))
        {
            person.hand_kpts_3d.clone()
        } else if vision.hand_kpts_3d.iter().copied().any(valid_point_3d) {
            vision.hand_kpts_3d.clone()
        } else {
            Vec::new()
        };
        let body_kpts_3d = if !estimate.raw_pose.body_kpts_3d.is_empty() {
            estimate.raw_pose.body_kpts_3d.clone()
        } else if let Some(person) = matched_stereo_person {
            person.body_kpts_3d.clone()
        } else if phone_body_has_valid_geometry {
            vision.body_kpts_3d.clone()
        } else {
            Vec::new()
        };
        let body_source = if !estimate.raw_pose.body_kpts_3d.is_empty() {
            estimate.fusion_breakdown.body_source.as_str().to_string()
        } else if matched_stereo_person.is_some() {
            "stereo".to_string()
        } else if phone_body_has_valid_geometry {
            if vision.body_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                vision.body_3d_source.trim().to_string()
            }
        } else {
            "none".to_string()
        };
        let hand_source = if hand_kpts_3d.is_empty() {
            "none".to_string()
        } else if estimate
            .operator_state
            .hand_kpts_3d
            .iter()
            .copied()
            .any(valid_point_3d)
            && estimate.fusion_breakdown.hand_source != crate::operator::OperatorPartSource::None
        {
            estimate.fusion_breakdown.hand_source.as_str().to_string()
        } else if stereo_hand_has_valid_geometry {
            "stereo".to_string()
        } else if vision.hand_3d_source.trim().is_empty() {
            "phone_edge_vision".to_string()
        } else {
            vision.hand_3d_source.trim().to_string()
        };
        let source_edge_time_ns = [
            operator_body_selected.then_some(operator_body_source_edge_time_ns),
            operator_hand_selected.then_some(operator_hand_source_edge_time_ns),
            stereo_body_selected.then_some(stereo.last_edge_time_ns),
            stereo_hand_has_valid_geometry.then_some(stereo.last_edge_time_ns),
            vision_body_selected.then_some(vision.last_edge_time_ns),
            vision_hand_selected.then_some(vision.last_edge_time_ns),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or_else(|| {
            operator_source_edge_time_ns
                .max(vision.last_edge_time_ns)
                .max(stereo.last_edge_time_ns)
        });
        let target_person_id = phone
            .track_id
            .clone()
            .or_else(|| Some("primary_operator".to_string()));
        return ResolvedTargetHumanState {
            has_target: true,
            fresh: vision.fresh || operator.fresh || stereo.fresh,
            target_person_id,
            source: Some("phone_session_associated_stereo".to_string()),
            source_edge_time_ns: Some(source_edge_time_ns),
            body_kpts_3d,
            hand_kpts_3d,
            body_3d_source: if estimate.raw_pose.body_kpts_3d.is_empty() {
                body_source
            } else {
                body_source
            },
            hand_3d_source: hand_source,
            association_anchor_source: if device_pose_stereo_candidate_track_id.is_some()
                && estimate.association.anchor_source.trim().is_empty()
            {
                "iphone_device_pose_to_stereo".to_string()
            } else {
                estimate.association.anchor_source.to_string()
            },
            uses_operator_estimate: !estimate.raw_pose.body_kpts_3d.is_empty(),
            iphone_track_id: phone.track_id.clone(),
            stereo_track_id: associated_stereo_track_id,
            wifi_track_id: estimate
                .association
                .wifi_operator_track_id
                .clone()
                .or_else(|| {
                    wifi_pose
                        .operator_track_id
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                }),
            association_confidence: estimate
                .association
                .hand_match_score
                .max(
                    matched_stereo_person
                        .map(|person| person.stereo_confidence)
                        .unwrap_or(0.0),
                )
                .max(vision.hand_conf)
                .max(vision.body_conf),
        };
    }

    if stereo_startup_warmup_hold {
        return ResolvedTargetHumanState {
            has_target: true,
            fresh: phone.recent,
            target_person_id: phone.track_id.clone(),
            source: Some("phone_session_associated_stereo".to_string()),
            source_edge_time_ns: Some(vision.last_edge_time_ns),
            body_kpts_3d: vision.body_kpts_3d.clone(),
            hand_kpts_3d: vision.hand_kpts_3d.clone(),
            body_3d_source: if vision.body_kpts_3d.is_empty() {
                "none".to_string()
            } else if vision.body_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                vision.body_3d_source.trim().to_string()
            },
            hand_3d_source: if vision.hand_kpts_3d.is_empty() {
                "none".to_string()
            } else if vision.hand_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                vision.hand_3d_source.trim().to_string()
            },
            association_anchor_source: "stereo_startup_warmup_hold".to_string(),
            uses_operator_estimate: false,
            iphone_track_id: phone.track_id.clone(),
            stereo_track_id: None,
            wifi_track_id: wifi_pose
                .operator_track_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            association_confidence: vision
                .vision_conf
                .max(vision.body_conf)
                .max(vision.hand_conf),
        };
    }

    if runtime_stereo_association_gap_hold {
        return ResolvedTargetHumanState {
            has_target: true,
            fresh: phone.recent,
            target_person_id: phone
                .track_id
                .clone()
                .or_else(|| Some("primary_operator".to_string())),
            source: Some("phone_session_associated_stereo".to_string()),
            source_edge_time_ns: Some(
                estimate
                    .updated_edge_time_ns
                    .max(stereo.last_edge_time_ns)
                    .max(vision.last_edge_time_ns),
            ),
            body_kpts_3d: vision.body_kpts_3d.clone(),
            hand_kpts_3d: vision.hand_kpts_3d.clone(),
            body_3d_source: if vision.body_kpts_3d.is_empty() {
                "none".to_string()
            } else if vision.body_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                vision.body_3d_source.trim().to_string()
            },
            hand_3d_source: if vision.hand_kpts_3d.is_empty() {
                "none".to_string()
            } else if vision.hand_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                vision.hand_3d_source.trim().to_string()
            },
            association_anchor_source: "runtime_stereo_association_gap_hold".to_string(),
            uses_operator_estimate: false,
            iphone_track_id: phone.track_id.clone(),
            stereo_track_id: None,
            wifi_track_id: wifi_pose
                .operator_track_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            association_confidence: vision
                .vision_conf
                .max(vision.body_conf)
                .max(vision.hand_conf),
        };
    }

    if phone.session_aligned && phone.recent {
        return ResolvedTargetHumanState {
            has_target: true,
            fresh: phone.recent,
            target_person_id: phone.track_id.clone(),
            source: Some("phone_edge_vision".to_string()),
            source_edge_time_ns: Some(vision.last_edge_time_ns),
            body_kpts_3d: vision.body_kpts_3d.clone(),
            hand_kpts_3d: vision.hand_kpts_3d.clone(),
            body_3d_source: if vision.body_kpts_3d.is_empty() {
                "none".to_string()
            } else if vision.body_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                vision.body_3d_source.trim().to_string()
            },
            hand_3d_source: if vision.hand_kpts_3d.is_empty() {
                "none".to_string()
            } else if vision.hand_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                vision.hand_3d_source.trim().to_string()
            },
            association_anchor_source: "phone_edge_vision_direct".to_string(),
            uses_operator_estimate: false,
            iphone_track_id: phone.track_id.clone(),
            stereo_track_id: stereo
                .operator_track_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            wifi_track_id: wifi_pose
                .operator_track_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            association_confidence: vision
                .vision_conf
                .max(vision.body_conf)
                .max(vision.hand_conf),
        };
    }

    ResolvedTargetHumanState {
        has_target: false,
        fresh: false,
        target_person_id: None,
        source: None,
        source_edge_time_ns: None,
        body_kpts_3d: Vec::new(),
        hand_kpts_3d: Vec::new(),
        body_3d_source: "none".to_string(),
        hand_3d_source: "none".to_string(),
        association_anchor_source: String::new(),
        uses_operator_estimate: false,
        iphone_track_id: None,
        stereo_track_id: None,
        wifi_track_id: None,
        association_confidence: 0.0,
    }
}

fn build_target_human_state_payload(
    live_session: Option<&LiveSessionContext>,
    operator: &crate::operator::OperatorSnapshot,
    vision: &crate::sensing::VisionSnapshot,
    stereo: &crate::sensing::StereoSnapshot,
    wifi_pose: &crate::sensing::WifiPoseSnapshot,
    phone_ingress: &crate::control::gate::PhoneIngressStatusSnapshot,
    iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>,
    edge_time_now_ns: u64,
) -> Value {
    let estimate = &operator.estimate;
    let has_active_session = live_session.is_some();
    let phone = resolve_phone_authoritative_state(live_session, vision);
    let resolved = resolve_target_human_state(
        live_session,
        operator,
        vision,
        stereo,
        wifi_pose,
        iphone_stereo_calibration,
        edge_time_now_ns,
    );
    let has_target = resolved.has_target;
    let body_joint_count = resolved.body_kpts_3d.len();
    let (left_hand_point_count, right_hand_point_count) =
        if has_target && !resolved.uses_operator_estimate {
            let left = if vision.left_hand_fresh_3d {
                vision.left_hand_kpts_3d.len()
            } else {
                0
            };
            let right = if vision.right_hand_fresh_3d {
                vision.right_hand_kpts_3d.len()
            } else {
                0
            };
            (left, right)
        } else {
            let left = resolved.hand_kpts_3d.len().min(21);
            let right = resolved.hand_kpts_3d.len().saturating_sub(21).min(21);
            (left, right)
        };
    let hand_point_count = left_hand_point_count + right_hand_point_count;
    let phone_sensor_available = vision.fresh || vision.body_3d_recent || vision.hand_3d_recent;
    let target_person_id = resolved.target_person_id.clone().unwrap_or_default();
    let selected_stereo_track_id = resolved
        .stereo_track_id
        .clone()
        .or_else(|| estimate.association.selected_operator_track_id.clone())
        .or_else(|| {
            stereo
                .operator_track_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        });
    let iphone_track_id = resolved
        .iphone_track_id
        .clone()
        .or_else(|| estimate.association.iphone_operator_track_id.clone());
    let wifi_track_id = resolved
        .wifi_track_id
        .clone()
        .or_else(|| estimate.association.wifi_operator_track_id.clone());
    let authority_source = target_authority_source(has_target);
    let association_source = target_association_source(
        has_target,
        &resolved.association_anchor_source,
        selected_stereo_track_id.as_deref(),
        estimate.association.hand_match_score,
        estimate.association.iphone_visible_hand_count,
    );
    let association_basis = target_association_basis(
        has_target,
        &resolved.association_anchor_source,
        selected_stereo_track_id.as_deref(),
        wifi_track_id.as_deref(),
        estimate.association.iphone_visible_hand_count,
        estimate.association.hand_match_score,
        estimate.association.left_wrist_gap_m,
        estimate.association.right_wrist_gap_m,
    );
    let body_geometry_source = if has_target {
        resolved.body_3d_source.clone()
    } else {
        "none".to_string()
    };
    let hand_geometry_source = if has_target {
        resolved.hand_3d_source.clone()
    } else {
        "none".to_string()
    };
    let target_ready = has_target && body_joint_count > 0;

    serde_json::json!({
        "source_session_id": live_session.map(|context| context.session_id.clone()),
        "source_trip_id": live_session.map(|context| context.trip_id.clone()),
        "target_person_id": if has_target { Value::String(target_person_id.clone()) } else { Value::Null },
        "authority_source": serde_json::to_value(authority_source).unwrap_or(Value::Null),
        "association_source": serde_json::to_value(association_source).unwrap_or(Value::Null),
        "body_geometry_source": Value::String(body_geometry_source.clone()),
        "hand_geometry_source": Value::String(hand_geometry_source.clone()),
        "fresh": if has_target { Value::Bool(resolved.fresh) } else { Value::Bool(false) },
        "has_target": has_target,
        "ready": Value::Bool(target_ready),
        "source": if has_target { serde_json::to_value(&resolved.source).unwrap_or(Value::Null) } else { Value::Null },
        "edge_time_ns": if has_target { serde_json::to_value(&resolved.source_edge_time_ns).unwrap_or(Value::Null) } else { Value::Null },
        "source_edge_time_ns": if has_target { serde_json::to_value(&resolved.source_edge_time_ns).unwrap_or(Value::Null) } else { Value::Null },
        "target_track_ids": {
            "iphone_track_id": if has_target { serde_json::to_value(&iphone_track_id).unwrap_or(Value::Null) } else { Value::Null },
            "stereo_track_id": if has_target { serde_json::to_value(&selected_stereo_track_id).unwrap_or(Value::Null) } else { Value::Null },
            "wifi_track_id": if has_target { serde_json::to_value(&wifi_track_id).unwrap_or(Value::Null) } else { Value::Null },
        },
        "association_confidence": if has_target { Value::from(resolved.association_confidence) } else { Value::from(0.0) },
        "body_joint_count": if has_target { Value::from(body_joint_count) } else { Value::from(0) },
        "hand_point_count": if has_target { Value::from(hand_point_count) } else { Value::from(0) },
        "left_hand_point_count": if has_target { Value::from(left_hand_point_count) } else { Value::from(0) },
        "right_hand_point_count": if has_target { Value::from(right_hand_point_count) } else { Value::from(0) },
        "body_kpts_3d": if has_target { serde_json::to_value(&resolved.body_kpts_3d).unwrap_or(Value::Array(vec![])) } else { Value::Array(vec![]) },
        "hand_kpts_3d": if has_target { serde_json::to_value(&resolved.hand_kpts_3d).unwrap_or(Value::Array(vec![])) } else { Value::Array(vec![]) },
        "body_3d_source": Value::String(body_geometry_source.clone()),
        "hand_3d_source": Value::String(hand_geometry_source.clone()),
        "source_breakdown": {
            "iphone": {
                "available": phone.session_aligned && phone_sensor_available,
                "body_joint_count": vision.body_kpts_3d.len(),
                "hand_point_count": vision.hand_kpts_3d.len(),
                "body_source": vision.body_3d_source.clone(),
                "hand_source": vision.hand_3d_source.clone(),
                "body_3d_recent": vision.body_3d_recent,
                "hand_3d_recent": vision.hand_3d_recent,
                "operator_track_id": vision.operator_track_id.clone(),
                "metrics_session_id": vision.metrics_session_id.clone(),
                "ingress": {
                    "status": Value::String(phone_ingress.status.clone()),
                    "error_code": non_empty_string_value(&phone_ingress.error_code),
                    "message": non_empty_string_value(&phone_ingress.message),
                    "in_flight": Value::Bool(phone_ingress.in_flight),
                    "attempt_edge_time_ns": serde_json::to_value(phone_ingress.last_attempt_edge_time_ns).unwrap_or(Value::Null),
                    "success_edge_time_ns": serde_json::to_value(phone_ingress.last_success_edge_time_ns).unwrap_or(Value::Null),
                    "frame_id": serde_json::to_value(phone_ingress.last_frame_id).unwrap_or(Value::Null),
                    "request_trip_id": non_empty_string_value(&phone_ingress.request_trip_id),
                    "request_session_id": non_empty_string_value(&phone_ingress.request_session_id),
                    "effective_trip_id": non_empty_string_value(&phone_ingress.effective_trip_id),
                    "effective_session_id": non_empty_string_value(&phone_ingress.effective_session_id),
                    "device_id": non_empty_string_value(&phone_ingress.device_id),
                    "operator_track_id": non_empty_string_value(&phone_ingress.operator_track_id),
                    "camera_mode": non_empty_string_value(&phone_ingress.camera_mode),
                    "camera_has_depth": serde_json::to_value(phone_ingress.camera_has_depth).unwrap_or(Value::Null),
                    "accepted": serde_json::to_value(phone_ingress.accepted).unwrap_or(Value::Null),
                },
            },
            "stereo": {
                "available": stereo.fresh,
                "body_joint_count": stereo.body_kpts_3d.len(),
                "hand_point_count": stereo.hand_kpts_3d.len(),
            },
            "wifi": {
                "available": wifi_pose.fresh,
                "body_joint_count": wifi_pose.body_kpts_3d.len(),
                "confidence": wifi_pose.body_confidence,
            },
            "aux_support": {
                "present": vision.aux_snapshot_present,
                "state": vision.aux_support_state.clone(),
                "body_points_3d_filled": vision.aux_body_points_3d_filled,
                "hand_points_3d_filled": vision.aux_hand_points_3d_filled,
            },
        },
        "retarget_ready": Value::Bool(has_target && body_joint_count > 0),
        "left_hand_curls": if has_target && resolved.uses_operator_estimate { serde_json::to_value(&estimate.left_hand_curls).unwrap_or(Value::Null) } else { Value::Null },
        "right_hand_curls": if has_target && resolved.uses_operator_estimate { serde_json::to_value(&estimate.right_hand_curls).unwrap_or(Value::Null) } else { Value::Null },
        "execution_mode": if has_active_session { Value::String(vision.execution_mode.clone()) } else { Value::String(String::new()) },
        "camera_mode": if has_active_session { Value::String(vision.camera_mode.clone()) } else { Value::String(String::new()) },
        "camera_has_depth": vision.camera_has_depth,
        "aux_snapshot_present": has_active_session && vision.aux_snapshot_present,
        "aux_body_points_2d_valid": if has_active_session { Value::from(vision.aux_body_points_2d_valid) } else { Value::from(0) },
        "aux_hand_points_2d_valid": if has_active_session { Value::from(vision.aux_hand_points_2d_valid) } else { Value::from(0) },
        "aux_body_points_3d_filled": if has_active_session { Value::from(vision.aux_body_points_3d_filled) } else { Value::from(0) },
        "aux_hand_points_3d_filled": if has_active_session { Value::from(vision.aux_hand_points_3d_filled) } else { Value::from(0) },
        "aux_support_state": if has_active_session { Value::String(vision.aux_support_state.clone()) } else { Value::String(String::new()) },
        "depth_z_mean_m": vision.depth_z_mean_m,
        "image_w": vision.image_w,
        "image_h": vision.image_h,
        "latency_ms": vision.iphone_to_edge_latency_ms,
        "association": {
            "anchor_source": if has_target { Value::String(resolved.association_anchor_source.clone()) } else { Value::String(String::new()) },
            "authority_source": serde_json::to_value(authority_source).unwrap_or(Value::Null),
            "association_source": serde_json::to_value(association_source).unwrap_or(Value::Null),
            "association_basis": serde_json::to_value(association_basis).unwrap_or(Value::Array(vec![])),
            "selected_operator_track_id": if has_target { serde_json::to_value(&selected_stereo_track_id).unwrap_or(Value::Null) } else { Value::Null },
            "stereo_operator_track_id": if has_target { serde_json::to_value(&selected_stereo_track_id).unwrap_or(Value::Null) } else { Value::Null },
            "wifi_operator_track_id": if has_target { serde_json::to_value(&wifi_track_id).unwrap_or(Value::Null) } else { Value::Null },
            "iphone_operator_track_id": if has_target { serde_json::to_value(&iphone_track_id).unwrap_or(Value::Null) } else { Value::Null },
            "hand_match_score": if has_target { Value::from(estimate.association.hand_match_score) } else { Value::Null },
            "wifi_association_score": if has_target { Value::from(estimate.association.wifi_association_score) } else { Value::Null },
            "left_wrist_gap_m": if has_target { serde_json::to_value(estimate.association.left_wrist_gap_m).unwrap_or(Value::Null) } else { Value::Null },
            "right_wrist_gap_m": if has_target { serde_json::to_value(estimate.association.right_wrist_gap_m).unwrap_or(Value::Null) } else { Value::Null },
        },
        "motion_state": {
            "root_pos_m": if has_target && resolved.uses_operator_estimate { serde_json::to_value(&estimate.motion_state.root_pos_m).unwrap_or(Value::Null) } else { Value::Array(vec![Value::Null, Value::Null, Value::Null]) },
            "root_vel_mps": if has_target && resolved.uses_operator_estimate { serde_json::to_value(&estimate.motion_state.root_vel_mps).unwrap_or(Value::Null) } else { Value::Array(vec![Value::Null, Value::Null, Value::Null]) },
            "heading_yaw_rad": if has_target && resolved.uses_operator_estimate { Value::from(estimate.motion_state.heading_yaw_rad) } else { Value::Null },
            "motion_phase": if has_target && resolved.uses_operator_estimate { Value::from(estimate.motion_state.motion_phase) } else { Value::Null },
            "body_presence_conf": if has_target && resolved.uses_operator_estimate { Value::from(estimate.motion_state.body_presence_conf) } else { Value::from(0.0) },
            "wearer_confidence": if has_target && resolved.uses_operator_estimate { Value::from(estimate.motion_state.wearer_confidence) } else { Value::from(0.0) },
            "smoother_mode": if has_target && resolved.uses_operator_estimate { Value::String(estimate.motion_state.smoother_mode.as_str().to_string()) } else { Value::String("none".to_string()) },
            "updated_edge_time_ns": if has_target && resolved.uses_operator_estimate { Value::from(estimate.motion_state.updated_edge_time_ns) } else { Value::Null },
        },
    })
}

fn build_scene_state_payload(
    live_session: Option<&LiveSessionContext>,
    operator: &crate::operator::OperatorSnapshot,
    vision: &crate::sensing::VisionSnapshot,
    stereo: &crate::sensing::StereoSnapshot,
    wifi_pose: &crate::sensing::WifiPoseSnapshot,
    csi: &crate::sensing::CsiSnapshot,
    iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>,
    edge_time_now_ns: u64,
) -> Value {
    let has_active_session = live_session.is_some();
    let phone_sensor_available = vision.fresh || vision.body_3d_recent || vision.hand_3d_recent;
    let resolved = resolve_target_human_state(
        live_session,
        operator,
        vision,
        stereo,
        wifi_pose,
        iphone_stereo_calibration,
        edge_time_now_ns,
    );
    let highlight_available = has_active_session && resolved.has_target;
    let target_person_id = resolved.target_person_id.clone().unwrap_or_default();
    serde_json::json!({
        "source_session_id": live_session.map(|context| context.session_id.clone()),
        "source_trip_id": live_session.map(|context| context.trip_id.clone()),
        "highlight_target_session_id": live_session.map(|context| context.session_id.clone()),
        "highlight_target_person_id": if highlight_available { Value::String(target_person_id) } else { Value::Null },
        "highlight_available": highlight_available,
        "highlight_body_kpts_3d": if highlight_available { serde_json::to_value(&resolved.body_kpts_3d).unwrap_or(Value::Array(vec![])) } else { Value::Array(vec![]) },
        "highlight_hand_kpts_3d": if highlight_available { serde_json::to_value(&resolved.hand_kpts_3d).unwrap_or(Value::Array(vec![])) } else { Value::Array(vec![]) },
        "highlight_source": if highlight_available { serde_json::to_value(&resolved.source).unwrap_or(Value::Null) } else { Value::String("none".to_string()) },
        "highlight_authority_source": if highlight_available { Value::String("phone_session".to_string()) } else { Value::Null },
        "highlight_body_geometry_source": if highlight_available { Value::String(resolved.body_3d_source.clone()) } else { Value::String("none".to_string()) },
        "highlight_hand_geometry_source": if highlight_available { Value::String(resolved.hand_3d_source.clone()) } else { Value::String("none".to_string()) },
        "highlight_updated_edge_time_ns": if highlight_available { serde_json::to_value(&resolved.source_edge_time_ns).unwrap_or(Value::Null) } else { Value::Null },
        "tracked_people": {
            "target": if highlight_available {
                serde_json::json!({
                    "person_id": resolved.target_person_id,
                    "body_joint_count": resolved.body_kpts_3d.len(),
                    "hand_point_count": resolved.hand_kpts_3d.len(),
                    "source": resolved.source,
                    "authority_source": "phone_session",
                    "body_geometry_source": resolved.body_3d_source,
                    "hand_geometry_source": resolved.hand_3d_source,
                })
            } else {
                Value::Null
            },
            "stereo_people": stereo.persons.iter().map(|person| {
                serde_json::json!({
                    "track_id": person.operator_track_id,
                    "body_joint_count": person.body_kpts_3d.len(),
                    "hand_point_count": person.hand_kpts_3d.len(),
                    "confidence": person.stereo_confidence,
                    "body_space": stereo.body_space,
                    "hand_space": stereo.hand_space,
                    "body_kpts_3d": person.body_kpts_3d,
                    "hand_kpts_3d": person.hand_kpts_3d,
                })
            }).collect::<Vec<_>>(),
        },
        "sensor_poses": {
            "phone": serde_json::to_value(&vision.device_pose).unwrap_or(Value::Null),
        },
        "sensors": {
            "phone": {
                "available": phone_sensor_available,
                "execution_mode": vision.execution_mode,
                "camera_mode": vision.camera_mode,
                "camera_has_depth": vision.camera_has_depth,
                "body_joint_count": vision.body_kpts_3d.len(),
                "hand_point_count": vision.hand_kpts_3d.len(),
                "operator_track_id": vision.operator_track_id,
                "edge_time_ns": vision.last_edge_time_ns,
            },
            "stereo": {
                "available": stereo.fresh,
                "person_count": stereo.persons.len(),
                "operator_track_id": stereo.operator_track_id,
                "body_joint_count": stereo.body_kpts_3d.len(),
                "hand_point_count": stereo.hand_kpts_3d.len(),
                "edge_time_ns": stereo.last_edge_time_ns,
            },
            "wifi": {
                "available": wifi_pose.fresh,
                "operator_track_id": wifi_pose.operator_track_id,
                "body_joint_count": wifi_pose.body_kpts_3d.len(),
                "confidence": wifi_pose.body_confidence,
                "edge_time_ns": wifi_pose.last_edge_time_ns,
            },
            "csi": {
                "available": csi.fresh,
                "node_count": csi.node_count,
                "drop_rate": csi.drop_rate,
                "snr_mean": csi.snr_mean,
                "confidence": csi.csi_conf,
            },
        },
    })
}

fn resolve_device_pose_stereo_candidate(
    device_pose: Option<&VisionDevicePose>,
    stereo: &crate::sensing::StereoSnapshot,
    iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>,
) -> Option<String> {
    const MAX_DEVICE_POSE_TRACK_DISTANCE_M: f32 = 1.85;
    const MIN_DEVICE_POSE_TRACK_MARGIN_M: f32 = 0.2;

    let anchor = resolve_device_pose_stereo_anchor(device_pose, iphone_stereo_calibration)?;
    let mut scored = stereo
        .persons
        .iter()
        .filter_map(|person| {
            let track_id = person.operator_track_id.as_ref()?;
            let center = stereo_person_body_center(person)?;
            Some((track_id.clone(), dist3_3d(center, anchor)))
        })
        .collect::<Vec<_>>();
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let (track_id, best_distance) = scored.first()?.clone();
    if best_distance > MAX_DEVICE_POSE_TRACK_DISTANCE_M {
        return None;
    }
    let margin_m = scored
        .get(1)
        .map(|(_, distance)| distance - best_distance)
        .unwrap_or(MAX_DEVICE_POSE_TRACK_DISTANCE_M);
    if margin_m < MIN_DEVICE_POSE_TRACK_MARGIN_M {
        return None;
    }
    Some(track_id)
}

fn resolve_device_pose_stereo_anchor(
    device_pose: Option<&VisionDevicePose>,
    iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>,
) -> Option<[f32; 3]> {
    let pose = device_pose?;
    if position_is_useful_3d(pose.position_m) && pose.target_space.trim() == "stereo_pair_frame" {
        return Some(pose.position_m);
    }
    let calibration = iphone_stereo_calibration?;
    if position_is_useful_3d(pose.position_m) {
        let point = calibration.apply_point(pose.position_m);
        return valid_point_3d(point).then_some(point);
    }
    let point = calibration.extrinsic_translation_m;
    valid_point_3d(point).then_some(point)
}

fn stereo_person_body_center(person: &StereoTrackedPersonSnapshot) -> Option<[f32; 3]> {
    const TORSO_INDICES: [usize; 4] = [5, 6, 11, 12];
    let valid = TORSO_INDICES
        .iter()
        .filter_map(|index| {
            person
                .body_kpts_3d
                .get(*index)
                .copied()
                .filter(|point| valid_point_3d(*point))
        })
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return None;
    }
    let count = valid.len() as f32;
    Some([
        valid.iter().map(|point| point[0]).sum::<f32>() / count,
        valid.iter().map(|point| point[1]).sum::<f32>() / count,
        valid.iter().map(|point| point[2]).sum::<f32>() / count,
    ])
}

fn valid_point_3d(point: [f32; 3]) -> bool {
    point.iter().all(|value| value.is_finite()) && point.iter().any(|value| value.abs() > 1e-6)
}

fn position_is_useful_3d(point: [f32; 3]) -> bool {
    valid_point_3d(point) && point.iter().any(|value| value.abs() > 1e-5)
}

fn dist3_3d(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

async fn build_sim_tracking_state_payload(
    state: &AppState,
    target_session_id: Option<&str>,
    target_person_id: Option<&str>,
    target_edge_time_ns: Option<u64>,
) -> Value {
    let edge_time_now_ns = state.gate.edge_time_ns();
    let base = state
        .config
        .isaac_runtime_base_url
        .trim()
        .trim_end_matches('/');
    if base.is_empty() {
        return serde_json::json!({
            "available": false,
            "ready": false,
            "mode": "unknown",
            "label": "未配置 Isaac host",
            "detail": "EDGE_ISAAC_RUNTIME_BASE_URL 未配置，无法探测独立 RTX 主机上的仿真运行链。",
            "control_state": "disarmed",
            "drive_mode": "unavailable",
            "is_demo_source": false,
            "source_kind": "unconfigured",
            "source_session_id": Value::Null,
            "source_person_id": Value::Null,
            "source_edge_time_ns": Value::Null,
            "controller_kind": "unknown",
            "controller_status": "offline",
            "retarget_status": "unconfigured",
            "latency_from_target_human_ms": Value::Null,
            "matches_target_human_state": false,
            "matches_target_human_session": false,
            "reason_mismatch": "isaac_runtime_base_url_unconfigured",
        });
    }

    let wbc_url = format!("{base}:8094/action/latest");
    if let Some(payload) = fetch_runtime_action(&state.http_client, &wbc_url).await {
        let summary = summarize_runtime_payload(
            "wbc",
            "Grounded WBC",
            &wbc_url,
            payload,
            target_session_id,
            target_person_id,
            target_edge_time_ns,
        );
        return maybe_apply_sim_tracking_carry(
            state,
            summary,
            target_session_id,
            target_person_id,
            edge_time_now_ns,
        );
    }

    let beyondmimic_url = format!("{base}:8093/action/latest");
    if let Some(payload) = fetch_runtime_action(&state.http_client, &beyondmimic_url).await {
        let summary = summarize_runtime_payload(
            "beyondmimic",
            "BeyondMimic",
            &beyondmimic_url,
            payload,
            target_session_id,
            target_person_id,
            target_edge_time_ns,
        );
        return maybe_apply_sim_tracking_carry(
            state,
            summary,
            target_session_id,
            target_person_id,
            edge_time_now_ns,
        );
    }

    maybe_apply_sim_tracking_carry(
        state,
        serde_json::json!({
            "available": false,
            "ready": false,
            "mode": "offline",
            "label": "仿真链离线",
            "detail": "当前没有从 RTX Isaac host 探测到可读的 8094(WBC) 或 8093(BeyondMimic) 动作源。",
            "host_base_url": base,
            "control_state": "disarmed",
            "drive_mode": "offline",
            "is_demo_source": false,
            "source_kind": "offline",
            "source_session_id": Value::Null,
            "source_person_id": Value::Null,
            "source_edge_time_ns": Value::Null,
            "controller_kind": "unknown",
            "controller_status": "offline",
            "retarget_status": "offline",
            "latency_from_target_human_ms": Value::Null,
            "matches_target_human_state": false,
            "matches_target_human_session": false,
            "reason_mismatch": "sim_runtime_offline",
        }),
        target_session_id,
        target_person_id,
        edge_time_now_ns,
    )
}

async fn fetch_runtime_action(client: &reqwest::Client, url: &str) -> Option<Value> {
    let response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json::<Value>().await.ok()
}

fn normalized_target_id(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|item| !item.is_empty())
}

fn sim_tracking_payload_matches_target(
    payload: &Value,
    target_session_id: Option<&str>,
    target_person_id: Option<&str>,
) -> bool {
    let Some(target_session_id) = normalized_target_id(target_session_id) else {
        return false;
    };
    let Some(target_person_id) = normalized_target_id(target_person_id) else {
        return false;
    };
    payload
        .get("source_session_id")
        .and_then(Value::as_str)
        .is_some_and(|value| value == target_session_id)
        && payload
            .get("source_person_id")
            .and_then(Value::as_str)
            .is_some_and(|value| value == target_person_id)
}

fn sim_tracking_payload_is_human_session_ready(
    payload: &Value,
    target_session_id: Option<&str>,
    target_person_id: Option<&str>,
) -> bool {
    payload
        .get("available")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && payload
            .get("ready")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        && payload
            .get("control_state")
            .and_then(Value::as_str)
            .is_some_and(|value| value == "armed")
        && payload
            .get("drive_mode")
            .and_then(Value::as_str)
            .is_some_and(|value| value == "human_session_driven")
        && sim_tracking_payload_matches_target(payload, target_session_id, target_person_id)
}

fn sim_tracking_degrade_reason(
    payload: &Value,
    target_session_id: Option<&str>,
    target_person_id: Option<&str>,
) -> Option<&'static str> {
    if !payload
        .get("available")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some("runtime_unavailable");
    }
    if !sim_tracking_payload_matches_target(payload, target_session_id, target_person_id) {
        return Some("runtime_target_mismatch");
    }
    if payload
        .get("control_state")
        .and_then(Value::as_str)
        .is_some_and(|value| value != "armed")
    {
        return Some("runtime_control_state_degraded");
    }
    if payload
        .get("drive_mode")
        .and_then(Value::as_str)
        .is_some_and(|value| value != "human_session_driven")
    {
        return Some("runtime_drive_mode_degraded");
    }
    if !payload
        .get("ready")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some("runtime_not_ready");
    }
    None
}

fn carried_sim_tracking_payload(
    snapshot: &SimTrackingCarrySnapshot,
    observed_payload: &Value,
    carry_reason: &str,
) -> Value {
    let mut payload = snapshot.payload.clone();
    if let Some(object) = payload.as_object_mut() {
        let detail = object
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        object.insert("carried_forward".to_string(), Value::Bool(true));
        object.insert(
            "carry_reason".to_string(),
            Value::String(carry_reason.to_string()),
        );
        object.insert(
            "observed_control_state".to_string(),
            observed_payload
                .get("control_state")
                .cloned()
                .unwrap_or(Value::Null),
        );
        object.insert(
            "observed_motion_reason".to_string(),
            observed_payload
                .get("motion_reason")
                .cloned()
                .unwrap_or(Value::Null),
        );
        object.insert(
            "observed_drive_mode".to_string(),
            observed_payload
                .get("drive_mode")
                .cloned()
                .unwrap_or(Value::Null),
        );
        object.insert(
            "observed_ready".to_string(),
            observed_payload
                .get("ready")
                .cloned()
                .unwrap_or(Value::Null),
        );
        object.insert(
            "detail".to_string(),
            Value::String(if detail.is_empty() {
                format!("短时沿用最近一次已对齐人体主链的 host 动作帧，原因：{carry_reason}。")
            } else {
                format!("{detail} 当前处于短时平滑 carry，原因：{carry_reason}。")
            }),
        );
    }
    payload
}

fn maybe_apply_sim_tracking_carry(
    state: &AppState,
    payload: Value,
    target_session_id: Option<&str>,
    target_person_id: Option<&str>,
    edge_time_now_ns: u64,
) -> Value {
    let Ok(mut guard) = state.sim_tracking_latest.lock() else {
        return payload;
    };

    if sim_tracking_payload_is_human_session_ready(&payload, target_session_id, target_person_id) {
        *guard = Some(SimTrackingCarrySnapshot {
            payload: payload.clone(),
            target_session_id: normalized_target_id(target_session_id).map(str::to_string),
            target_person_id: normalized_target_id(target_person_id).map(str::to_string),
            updated_edge_time_ns: edge_time_now_ns,
        });
        return payload;
    }

    let Some(carry_reason) =
        sim_tracking_degrade_reason(&payload, target_session_id, target_person_id)
    else {
        return payload;
    };
    let Some(snapshot) = guard.as_ref() else {
        return payload;
    };
    if snapshot.target_session_id.as_deref() != normalized_target_id(target_session_id)
        || snapshot.target_person_id.as_deref() != normalized_target_id(target_person_id)
    {
        return payload;
    }
    if edge_time_now_ns.saturating_sub(snapshot.updated_edge_time_ns)
        > SIM_TRACKING_TRANSIENT_CARRY_MS * 1_000_000
    {
        return payload;
    }
    if !sim_tracking_payload_is_human_session_ready(
        &snapshot.payload,
        target_session_id,
        target_person_id,
    ) {
        return payload;
    }
    carried_sim_tracking_payload(snapshot, &payload, carry_reason)
}

fn summarize_runtime_payload(
    mode: &str,
    label: &str,
    source_url: &str,
    payload: Value,
    target_session_id: Option<&str>,
    target_person_id: Option<&str>,
    target_edge_time_ns: Option<u64>,
) -> Value {
    let motion_allowed = payload
        .get("motion_allowed")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let control_state = payload
        .get("control_state")
        .and_then(|value| value.as_str())
        .unwrap_or(if motion_allowed { "armed" } else { "disarmed" })
        .to_string();
    let motion_reason = payload
        .get("motion_reason")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();
    let motion_mode = payload
        .get("motion_mode")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();
    let waist_dim = payload
        .get("waist_q_target")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let leg_dim = payload
        .get("leg_q_target")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let arm_dim = payload
        .get("arm_q_target")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let left_hand_dim = payload
        .get("left_hand_target")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let right_hand_dim = payload
        .get("right_hand_target")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let hand_dim = left_hand_dim.max(right_hand_dim);
    let source_session_id = payload
        .get("source_session_id")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    let source_kind = payload
        .get("source_kind")
        .and_then(|value| value.as_str())
        .unwrap_or("independent_runtime")
        .trim()
        .to_string();
    let source_person_id = payload
        .get("source_person_id")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    let source_edge_time_ns = payload
        .get("source_edge_time_ns")
        .and_then(|value| value.as_u64());
    let controller_kind = payload
        .get("controller_kind")
        .and_then(|value| value.as_str())
        .unwrap_or(mode);
    let matches_target_human_session = target_session_id
        .zip(source_session_id.as_deref())
        .is_some_and(|(target, source)| !target.is_empty() && target == source);
    let matches_target_human_person = target_person_id
        .zip(source_person_id.as_deref())
        .is_some_and(|(target, source)| !target.is_empty() && target == source);
    let matches_target_human_state = matches_target_human_session
        && target_person_id.is_some()
        && source_person_id.is_some()
        && matches_target_human_person;
    let is_demo_source = matches!(source_kind.as_str(), "demo" | "independent_runtime");
    let drive_mode = if is_demo_source {
        "demo"
    } else if matches_target_human_state {
        "human_session_driven"
    } else {
        "unbound"
    };
    let ready = drive_mode == "human_session_driven"
        && motion_allowed
        && (waist_dim > 0 || leg_dim > 0 || arm_dim > 0 || hand_dim > 0);
    let controller_status = payload
        .get("controller_status")
        .and_then(|value| value.as_str())
        .unwrap_or(if ready { "ready" } else { "waiting" });
    let retarget_status = payload
        .get("retarget_status")
        .and_then(|value| value.as_str())
        .unwrap_or(if ready { "active" } else { "pending" });
    let latency_from_target_human_ms =
        target_edge_time_ns
            .zip(source_edge_time_ns)
            .map(|(target, source)| {
                ((source as i128 - target as i128).unsigned_abs() as f64) / 1_000_000.0
            });
    let reason_mismatch = if is_demo_source {
        Value::String("demo_source_not_human_session".to_string())
    } else if matches_target_human_session {
        if target_person_id.is_some() && source_person_id.is_some() && !matches_target_human_person
        {
            Value::String("sim_runtime_bound_to_different_person".to_string())
        } else {
            Value::Null
        }
    } else if target_session_id.is_none_or(str::is_empty) {
        Value::String("no_active_target_human_session".to_string())
    } else if source_session_id.is_none() {
        Value::String("sim_runtime_not_bound_to_current_human_session".to_string())
    } else {
        Value::String("sim_runtime_bound_to_different_session".to_string())
    };
    let runtime_label = if is_demo_source {
        format!("{label} Demo")
    } else {
        label.to_string()
    };
    let runtime_detail = if is_demo_source {
        format!("{label} 当前输出的是开发演示动作源，不代表当前人体主链。")
    } else if ready {
        format!("{label} 已对齐当前人体主链，当前动作维度 waist={waist_dim} leg={leg_dim} arm={arm_dim} hand={hand_dim}。")
    } else if matches_target_human_state {
        format!("{label} 已绑定当前人体主链，但当前动作还没放行。")
    } else if matches_target_human_session {
        format!("{label} 已绑定当前 session，但当前动作源的人物口径还没完全对齐。")
    } else {
        format!("{label} 已连上动作源，但当前不是由当前人体主链驱动。")
    };

    serde_json::json!({
        "available": true,
        "ready": ready,
        "mode": mode,
        "label": runtime_label,
        "detail": runtime_detail,
        "source_url": source_url,
        "control_state": control_state,
        "motion_allowed": motion_allowed,
        "motion_reason": motion_reason,
        "motion_mode": motion_mode,
        "waist_dim": waist_dim,
        "leg_dim": leg_dim,
        "arm_dim": arm_dim,
        "hand_dim": hand_dim,
        "drive_mode": drive_mode,
        "is_demo_source": is_demo_source,
        "source_kind": source_kind,
        "source_session_id": source_session_id,
        "source_person_id": source_person_id,
        "source_edge_time_ns": source_edge_time_ns,
        "controller_kind": controller_kind,
        "controller_status": controller_status,
        "retarget_status": retarget_status,
        "latency_from_target_human_ms": latency_from_target_human_ms,
        "matches_target_human_state": matches_target_human_state,
        "matches_target_human_session": matches_target_human_session,
        "reason_mismatch": reason_mismatch,
    })
}

async fn build_live_media_track(session_dir: Option<&PathBuf>, scope: &str, track: &str) -> Value {
    let id = format!("{scope}.{track}");
    let label = match (scope, track) {
        ("iphone", "main") => "手机主路",
        ("iphone", "aux") => "手机辅路",
        ("iphone", "depth") => "手机深度",
        ("iphone", "fisheye") => "手机鱼眼",
        _ => track,
    };
    let kind = if track == "depth" { "depth" } else { "video" };

    let Some(session_dir) = session_dir else {
        return serde_json::json!({
            "id": id,
            "label": label,
            "kind": kind,
            "status": "waiting_session",
        });
    };

    let storage_track = if scope == "iphone" && track == "main" {
        "wide"
    } else {
        track
    };
    let media_index_path = session_dir
        .join("raw")
        .join(scope)
        .join(storage_track)
        .join("media_index.jsonl");
    let Some((record, status_detail)) =
        resolve_live_media_record(session_dir, &media_index_path, scope, track).await
    else {
        return serde_json::json!({
            "id": id,
            "label": label,
            "kind": kind,
            "status": "waiting_track",
        });
    };

    let file_relpath = record
        .get("file_relpath")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();
    let file_exists = safe_asset_path(session_dir, &file_relpath).is_some();
    let preview_url = if kind == "video" && file_exists && !file_relpath.is_empty() {
        Some(format!("/live-preview/file/{file_relpath}"))
    } else {
        None
    };
    let depth_preview = if kind == "depth" && file_exists && !file_relpath.is_empty() {
        safe_asset_path(session_dir, &file_relpath)
            .and_then(|path| load_depth_chunk_preview(&path))
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    };

    serde_json::json!({
        "id": id,
        "label": label,
        "kind": kind,
        "status": if file_exists { "ready" } else { "missing_file" },
        "status_detail": status_detail,
        "media_scope": record.get("media_scope").and_then(|value| value.as_str()).unwrap_or(scope),
        "media_track": record.get("media_track").and_then(|value| value.as_str()).unwrap_or(track),
        "file_relpath": file_relpath,
        "preview_url": preview_url,
        "file_bytes": record.get("file_bytes").and_then(|value| value.as_u64()),
        "edge_time_ns": record.get("edge_time_ns").and_then(|value| value.as_u64()),
        "upload_edge_time_ns": record.get("upload_edge_time_ns").and_then(|value| value.as_u64()),
        "frame_count": record.get("frame_count").and_then(|value| value.as_u64()),
        "frame_rate_hz": record.get("frame_rate_hz").and_then(|value| value.as_f64()),
        "time_sync_status": record.get("time_sync_status").and_then(|value| value.as_str()),
        "source_start_time_ns": record.get("source_start_time_ns").and_then(|value| value.as_u64()),
        "source_end_time_ns": record.get("source_end_time_ns").and_then(|value| value.as_u64()),
        "depth_preview": depth_preview,
    })
}

async fn build_live_task_semantics(session_dir: Option<&PathBuf>) -> Value {
    let Some(session_dir) = session_dir else {
        return serde_json::json!({
            "available": false,
            "main_task": "等待活跃 session",
            "sub_task": "等待标注",
            "current_interaction": "等待标注",
            "related_objects": [],
            "timeline": [],
            "events": [],
        });
    };
    let labels_path = session_dir.join("labels").join("labels.jsonl");
    let records = read_recent_jsonl_values(&labels_path, 8).await;
    let Some(record) = records.last() else {
        return serde_json::json!({
            "available": false,
            "main_task": "当前 session 尚未写入任务语义",
            "sub_task": "等待 action/scene 标注",
            "current_interaction": "等待 label_event_packet",
            "related_objects": [],
            "timeline": [],
            "events": [],
        });
    };

    let scene_label =
        non_empty_string(record.get("scene_label")).unwrap_or_else(|| "当前采集".to_string());
    let action_label =
        non_empty_string(record.get("action_label")).unwrap_or_else(|| "未标注子任务".to_string());
    let event_name = non_empty_string(record.get("event")).unwrap_or_else(|| "unknown".to_string());
    let interaction = match event_name.as_str() {
        "action_start" => "动作开始",
        "action_end" => "动作结束",
        "scene_switch" => "场景切换",
        "quality_mark" => "质量标记",
        _ => event_name.as_str(),
    }
    .to_string();
    let related_objects = extract_related_objects(&record);
    let mut timeline = Vec::new();
    for item in &records {
        let scene =
            non_empty_string(item.get("scene_label")).unwrap_or_else(|| "当前采集".to_string());
        let action = non_empty_string(item.get("action_label"))
            .unwrap_or_else(|| "未标注子任务".to_string());
        let event = non_empty_string(item.get("event")).unwrap_or_else(|| "unknown".to_string());
        let summary = format!("{scene} · {action} · {event}");
        if !timeline.iter().any(|existing| existing == &summary) {
            timeline.push(summary);
        }
    }

    let events = records
        .iter()
        .rev()
        .map(|item| {
            let scene =
                non_empty_string(item.get("scene_label")).unwrap_or_else(|| "当前采集".to_string());
            let action = non_empty_string(item.get("action_label"))
                .unwrap_or_else(|| "未标注子任务".to_string());
            let event =
                non_empty_string(item.get("event")).unwrap_or_else(|| "unknown".to_string());
            let interaction = match event.as_str() {
                "action_start" => "动作开始",
                "action_end" => "动作结束",
                "scene_switch" => "场景切换",
                "quality_mark" => "质量标记",
                _ => event.as_str(),
            }
            .to_string();
            serde_json::json!({
                "event": event,
                "scene_label": scene,
                "action_label": action,
                "interaction": interaction,
                "edge_time_ns": item.get("edge_time_ns").and_then(|value| value.as_u64()),
                "related_objects": extract_related_objects(item),
                "quality_mark": non_empty_string(item.get("quality_mark")),
            })
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "available": true,
        "main_task": scene_label,
        "sub_task": action_label,
        "current_interaction": interaction,
        "related_objects": related_objects,
        "timeline": timeline,
        "events": events,
    })
}

async fn build_live_vlm_summary(
    session_dir: Option<&PathBuf>,
    vlm_enabled: bool,
    preview_enabled: bool,
) -> Value {
    let empty_status = if vlm_enabled {
        "waiting_session"
    } else {
        "disabled"
    };
    let empty_preview_status = if preview_enabled {
        "waiting_session"
    } else {
        "disabled"
    };
    let Some(session_dir) = session_dir else {
        return serde_json::json!({
            "enabled": vlm_enabled,
            "available": false,
            "status": empty_status,
            "preview_status": empty_preview_status,
            "degraded_reasons": [],
            "latest_tags": [],
            "latest_objects": [],
            "recent_events": [],
            "recent_segments": [],
        });
    };

    let manifest_path = session_dir.join("preview").join("preview_manifest.json");
    let manifest = match tokio::fs::read_to_string(&manifest_path).await {
        Ok(content) => serde_json::from_str::<Value>(&content).ok(),
        Err(_) => None,
    };
    let status = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_status"))
        .and_then(|value| value.as_str())
        .unwrap_or(if vlm_enabled {
            "waiting_manifest"
        } else {
            "disabled"
        })
        .to_string();
    let preview_status = manifest
        .as_ref()
        .and_then(|value| value.get("status"))
        .and_then(|value| value.as_str())
        .unwrap_or(if preview_enabled {
            "waiting_manifest"
        } else {
            "disabled"
        })
        .to_string();
    let enabled = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_indexing_enabled"))
        .and_then(|value| value.as_bool())
        .unwrap_or(vlm_enabled);
    let configured_model_id = manifest
        .as_ref()
        .and_then(|value| non_empty_string(value.get("model_id")));
    let fallback_model_id = manifest
        .as_ref()
        .and_then(|value| non_empty_string(value.get("fallback_model_id")));
    let prompt_version = manifest
        .as_ref()
        .and_then(|value| non_empty_string(value.get("prompt_version")));
    let sidecar_base = manifest
        .as_ref()
        .and_then(|value| non_empty_string(value.get("vlm_sidecar_base")));
    let live_interval_ms = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_live_interval_ms"))
        .and_then(|value| value.as_u64());
    let event_trigger_enabled = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_event_trigger_enabled"))
        .and_then(|value| value.as_bool());
    let event_trigger_camera_mode_change_enabled = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_event_trigger_camera_mode_change_enabled"))
        .and_then(|value| value.as_bool());
    let inference_timeout_ms = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_inference_timeout_ms"))
        .and_then(|value| value.as_u64());
    let auto_fallback_latency_ms = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_auto_fallback_latency_ms"))
        .and_then(|value| value.as_u64());
    let auto_fallback_cooldown_ms = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_auto_fallback_cooldown_ms"))
        .and_then(|value| value.as_u64());
    let max_consecutive_failures = manifest
        .as_ref()
        .and_then(|value| value.get("vlm_max_consecutive_failures"))
        .and_then(|value| value.as_u64());
    let degraded_reasons = manifest
        .as_ref()
        .and_then(|value| value.get("degraded_reasons"))
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let stage = non_empty_string(item.get("stage"));
                    let detail = non_empty_string(item.get("detail"));
                    match (stage, detail) {
                        (Some(stage), Some(detail)) => Some(format!("{stage}: {detail}")),
                        (Some(stage), None) => Some(stage),
                        (None, Some(detail)) => Some(detail),
                        (None, None) => None,
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let latest_keyframe = manifest
        .as_ref()
        .and_then(|value| value.get("keyframes"))
        .and_then(|value| value.as_array())
        .and_then(|items| {
            items.iter().rev().find_map(|item| {
                let relpath = non_empty_string(item.get("relpath"))?;
                let preview_url = build_live_preview_file_url(session_dir, &relpath)?;
                Some(serde_json::json!({
                    "id": non_empty_string(item.get("id")),
                    "preview_url": preview_url,
                    "caption": non_empty_string(item.get("caption")),
                    "tags": collect_string_list(item.get("tags")),
                    "action_guess": non_empty_string(item.get("action_guess")),
                    "edge_time_ns": item.get("edge_time_ns").and_then(|value| value.as_u64()),
                }))
            })
        })
        .unwrap_or(Value::Null);
    let latest_clip = manifest
        .as_ref()
        .and_then(|value| value.get("clips"))
        .and_then(|value| value.as_array())
        .and_then(|items| {
            items.iter().rev().find_map(|item| {
                let relpath = non_empty_string(item.get("relpath"))?;
                let preview_url = build_live_preview_file_url(session_dir, &relpath)?;
                Some(serde_json::json!({
                    "id": non_empty_string(item.get("id")),
                    "preview_url": preview_url,
                    "caption": non_empty_string(item.get("caption")),
                    "tags": collect_string_list(item.get("tags")),
                    "action_guess": non_empty_string(item.get("action_guess")),
                    "start_edge_time_ns": item.get("start_edge_time_ns").and_then(|value| value.as_u64()),
                    "end_edge_time_ns": item.get("end_edge_time_ns").and_then(|value| value.as_u64()),
                }))
            })
        })
        .unwrap_or(Value::Null);

    let recent_event_records = read_recent_jsonl_values(
        &session_dir
            .join("derived")
            .join("vision")
            .join("vlm_events.jsonl"),
        6,
    )
    .await;
    let recent_events = recent_event_records
        .iter()
        .rev()
        .map(|item| {
            let preview_url = non_empty_string(item.get("frame_relpath"))
                .and_then(|relpath| build_live_preview_file_url(session_dir, &relpath));
            serde_json::json!({
                "event_id": non_empty_string(item.get("event_id")),
                "edge_time_ns": item.get("edge_time_ns").and_then(|value| value.as_u64()),
                "camera_mode": non_empty_string(item.get("camera_mode")),
                "caption": non_empty_string(item.get("caption")),
                "action_guess": non_empty_string(item.get("action_guess")),
                "tags": collect_string_list(item.get("tags")),
                "objects": collect_string_list(item.get("objects")),
                "preview_url": preview_url,
                "sample_reasons": collect_string_list(item.get("sample_reasons")),
                "model_id": non_empty_string(item.get("model_id")),
                "prompt_version": non_empty_string(item.get("prompt_version")),
                "inference_source": non_empty_string(item.get("inference_source")),
                "latency_ms": item.get("latency_ms").and_then(|value| value.as_f64()),
            })
        })
        .collect::<Vec<_>>();
    let recent_segment_records = read_recent_jsonl_values(
        &session_dir
            .join("derived")
            .join("vision")
            .join("vlm_segments.jsonl"),
        4,
    )
    .await;
    let recent_segments = recent_segment_records
        .iter()
        .rev()
        .map(|item| {
            serde_json::json!({
                "segment_id": non_empty_string(item.get("segment_id")),
                "start_edge_time_ns": item.get("start_edge_time_ns").and_then(|value| value.as_u64()),
                "end_edge_time_ns": item.get("end_edge_time_ns").and_then(|value| value.as_u64()),
                "camera_mode": non_empty_string(item.get("camera_mode")),
                "caption": non_empty_string(item.get("caption")),
                "action_guess": non_empty_string(item.get("action_guess")),
                "tags": collect_string_list(item.get("tags")),
                "objects": collect_string_list(item.get("objects")),
                "keyframe_count": item
                    .get("keyframe_ids")
                    .and_then(|value| value.as_array())
                    .map(|items| items.len()),
                "event_count": item
                    .get("event_ids")
                    .and_then(|value| value.as_array())
                    .map(|items| items.len()),
                "model_id": non_empty_string(item.get("model_id")),
                "prompt_version": non_empty_string(item.get("prompt_version")),
                "inference_source": non_empty_string(item.get("inference_source")),
                "latency_ms": item.get("latency_ms").and_then(|value| value.as_f64()),
            })
        })
        .collect::<Vec<_>>();

    let latest_event = recent_event_records.last();
    let latest_segment = recent_segment_records.last();
    let model_id = latest_event
        .and_then(|item| non_empty_string(item.get("model_id")))
        .or_else(|| latest_segment.and_then(|item| non_empty_string(item.get("model_id"))))
        .or(configured_model_id);
    let latest_caption = latest_event
        .and_then(|item| non_empty_string(item.get("caption")))
        .or_else(|| latest_segment.and_then(|item| non_empty_string(item.get("caption"))))
        .or_else(|| {
            latest_clip
                .get("caption")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        })
        .or_else(|| {
            latest_keyframe
                .get("caption")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        });
    let latest_action_guess = latest_event
        .and_then(|item| non_empty_string(item.get("action_guess")))
        .or_else(|| latest_segment.and_then(|item| non_empty_string(item.get("action_guess"))))
        .or_else(|| {
            latest_clip
                .get("action_guess")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        })
        .or_else(|| {
            latest_keyframe
                .get("action_guess")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        });
    let latest_tags = {
        let event_tags = latest_event
            .map(|item| collect_string_list(item.get("tags")))
            .unwrap_or_default();
        if !event_tags.is_empty() {
            event_tags
        } else {
            let segment_tags = latest_segment
                .map(|item| collect_string_list(item.get("tags")))
                .unwrap_or_default();
            if !segment_tags.is_empty() {
                segment_tags
            } else {
                collect_string_list(latest_keyframe.get("tags"))
            }
        }
    };
    let latest_objects = {
        let event_objects = latest_event
            .map(|item| collect_string_list(item.get("objects")))
            .unwrap_or_default();
        if !event_objects.is_empty() {
            event_objects
        } else {
            latest_segment
                .map(|item| collect_string_list(item.get("objects")))
                .unwrap_or_default()
        }
    };
    let latest_inference_source = latest_event
        .and_then(|item| non_empty_string(item.get("inference_source")))
        .or_else(|| latest_segment.and_then(|item| non_empty_string(item.get("inference_source"))))
        .or_else(|| {
            latest_keyframe
                .get("inference_source")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        });
    let latest_latency_ms = latest_event
        .and_then(|item| item.get("latency_ms"))
        .and_then(|value| value.as_f64())
        .or_else(|| {
            latest_segment
                .and_then(|item| item.get("latency_ms"))
                .and_then(|value| value.as_f64())
        })
        .or_else(|| {
            latest_keyframe
                .get("latency_ms")
                .and_then(|value| value.as_f64())
        });
    let available = !recent_events.is_empty()
        || !recent_segments.is_empty()
        || !latest_keyframe.is_null()
        || !latest_clip.is_null();

    serde_json::json!({
        "enabled": enabled,
        "available": available,
        "status": status,
        "preview_status": preview_status,
        "model_id": model_id,
        "fallback_model_id": fallback_model_id,
        "prompt_version": prompt_version,
        "sidecar_base": sidecar_base,
        "live_interval_ms": live_interval_ms,
        "event_trigger_enabled": event_trigger_enabled,
        "event_trigger_camera_mode_change_enabled": event_trigger_camera_mode_change_enabled,
        "inference_timeout_ms": inference_timeout_ms,
        "auto_fallback_latency_ms": auto_fallback_latency_ms,
        "auto_fallback_cooldown_ms": auto_fallback_cooldown_ms,
        "max_consecutive_failures": max_consecutive_failures,
        "degraded_reasons": degraded_reasons,
        "latest_caption": latest_caption,
        "latest_action_guess": latest_action_guess,
        "latest_tags": latest_tags,
        "latest_objects": latest_objects,
        "latest_inference_source": latest_inference_source,
        "latest_latency_ms": latest_latency_ms,
        "latest_keyframe": latest_keyframe,
        "latest_clip": latest_clip,
        "recent_events": recent_events,
        "recent_segments": recent_segments,
    })
}

fn non_empty_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(|item| item.as_str())
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
}

fn collect_string_list(value: Option<&Value>) -> Vec<String> {
    let mut values = value
        .and_then(|item| item.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    values.sort();
    values.dedup();
    values
}

fn build_live_preview_file_url(session_dir: &Path, relpath: &str) -> Option<String> {
    let relpath = relpath.trim();
    if relpath.is_empty() {
        return None;
    }
    safe_asset_path(session_dir, relpath).map(|_| format!("/live-preview/file/{relpath}"))
}

fn extract_related_objects(record: &Value) -> Vec<String> {
    let mut values = Vec::new();
    for key in ["objects", "object_ids", "object_labels"] {
        if let Some(items) = record.get(key).and_then(|value| value.as_array()) {
            for item in items {
                if let Some(text) = item.as_str().map(str::trim).filter(|text| !text.is_empty()) {
                    values.push(text.to_string());
                }
            }
        }
    }
    for key in ["object_id", "object_label", "target_object"] {
        if let Some(text) = non_empty_string(record.get(key)) {
            values.push(text);
        }
    }
    values.sort();
    values.dedup();
    values
}

async fn read_last_jsonl_value(path: &Path) -> Option<Value> {
    let bytes = tokio::fs::read(path).await.ok()?;
    let end = bytes.iter().rposition(|byte| !byte.is_ascii_whitespace())?;
    let trimmed = &bytes[..=end];
    let line_start = trimmed
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map(|index| index + 1)
        .unwrap_or(0);
    serde_json::from_slice(&trimmed[line_start..]).ok()
}

async fn read_recent_jsonl_values(path: &Path, limit: usize) -> Vec<Value> {
    if limit == 0 {
        return Vec::new();
    }
    let Ok(bytes) = tokio::fs::read(path).await else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut values = text
        .lines()
        .rev()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            serde_json::from_str::<Value>(trimmed).ok()
        })
        .take(limit)
        .collect::<Vec<_>>();
    values.reverse();
    values
}

async fn resolve_live_media_record(
    session_dir: &Path,
    media_index_path: &Path,
    scope: &str,
    track: &str,
) -> Option<(Value, Option<String>)> {
    if let Some(record) = read_last_jsonl_value(media_index_path).await {
        let file_relpath = record
            .get("file_relpath")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if !file_relpath.is_empty() && safe_asset_path(session_dir, file_relpath).is_some() {
            return Some((record, None));
        }
    }

    for record in read_recent_jsonl_values(media_index_path, 512)
        .await
        .into_iter()
        .rev()
    {
        let file_relpath = record
            .get("file_relpath")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if !file_relpath.is_empty() && safe_asset_path(session_dir, file_relpath).is_some() {
            return Some((
                record,
                Some("索引最新分片不可读，已回退到最近可用的真实分片。".to_string()),
            ));
        }
    }

    let Some(file_relpath) =
        find_latest_existing_track_file_relpath(session_dir, scope, track).await
    else {
        return None;
    };

    Some((
        serde_json::json!({
            "media_scope": scope,
            "media_track": track,
            "file_relpath": file_relpath,
        }),
        Some("索引不可用，已直接读取当前 session 最近落盘的真实分片。".to_string()),
    ))
}

async fn find_latest_existing_track_file_relpath(
    session_dir: &Path,
    scope: &str,
    track: &str,
) -> Option<String> {
    let chunk_root = session_dir
        .join("raw")
        .join(scope)
        .join(track)
        .join("chunks");
    let mut buckets = tokio::fs::read_dir(&chunk_root).await.ok()?;
    let mut latest: Option<(SystemTime, String)> = None;

    loop {
        let Some(bucket_entry) = buckets.next_entry().await.ok()? else {
            break;
        };
        let Ok(file_type) = bucket_entry.file_type().await else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let bucket_name = bucket_entry.file_name().to_string_lossy().to_string();
        let mut files = match tokio::fs::read_dir(bucket_entry.path()).await {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        loop {
            let Some(file_entry) = files.next_entry().await.ok()? else {
                break;
            };
            let Ok(file_type) = file_entry.file_type().await else {
                continue;
            };
            if !file_type.is_file() {
                continue;
            }
            let modified = file_entry
                .metadata()
                .await
                .ok()
                .and_then(|meta| meta.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let relpath = format!(
                "raw/{scope}/{track}/chunks/{bucket_name}/{}",
                file_entry.file_name().to_string_lossy()
            );
            match &latest {
                Some((best_time, _)) if modified <= *best_time => {}
                _ => latest = Some((modified, relpath)),
            }
        }
    }

    latest.map(|(_, relpath)| relpath)
}

fn load_depth_chunk_preview(path: &Path) -> Option<Value> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 12 || &data[..8] != DEPTH_CHUNK_MAGIC {
        return None;
    }
    let frame_count = read_u32_le(&data, 8)? as usize;
    if frame_count == 0 {
        return None;
    }
    let mut offset = 12usize;
    let mut selected: Option<(u64, usize, usize, Vec<u16>)> = None;
    for _ in 0..frame_count {
        let source_time_ns = read_u64_le(&data, offset)?;
        offset += 8;
        let width = read_u32_le(&data, offset)? as usize;
        offset += 4;
        let height = read_u32_le(&data, offset)? as usize;
        offset += 4;
        let payload_len = read_u32_le(&data, offset)? as usize;
        offset += 4;
        if offset + payload_len > data.len() {
            return None;
        }
        if width == 0 || height == 0 || payload_len < width * height * 2 {
            offset += payload_len;
            continue;
        }
        let mut values = Vec::with_capacity(width * height);
        for chunk in data[offset..offset + width * height * 2].chunks_exact(2) {
            values.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        selected = Some((source_time_ns, width, height, values));
        offset += payload_len;
    }
    let (source_time_ns, width, height, values) = selected?;
    let mut valid_total = 0usize;
    let mut depth_sum = 0u64;
    let mut min_depth: Option<u16> = None;
    let mut max_depth: Option<u16> = None;
    for depth in &values {
        if *depth == 0 || *depth == u16::MAX {
            continue;
        }
        valid_total += 1;
        depth_sum = depth_sum.saturating_add(*depth as u64);
        min_depth = Some(min_depth.map_or(*depth, |current| current.min(*depth)));
        max_depth = Some(max_depth.map_or(*depth, |current| current.max(*depth)));
    }
    let step_x = ((width as f64 / DEPTH_PREVIEW_MAX_DIM as f64).ceil() as usize).max(1);
    let step_y = ((height as f64 / DEPTH_PREVIEW_MAX_DIM as f64).ceil() as usize).max(1);
    let mut sampled_rows: Vec<Vec<Option<u16>>> = Vec::new();
    let mut y = 0usize;
    while y < height {
        let end_y = (y + step_y).min(height);
        let mut row = Vec::new();
        let mut x = 0usize;
        while x < width {
            let end_x = (x + step_x).min(width);
            let mut sample_sum = 0u64;
            let mut sample_count = 0u64;
            for yy in y..end_y {
                let row_offset = yy * width;
                for xx in x..end_x {
                    let depth = values[row_offset + xx];
                    if depth == 0 || depth == u16::MAX {
                        continue;
                    }
                    sample_sum = sample_sum.saturating_add(depth as u64);
                    sample_count = sample_count.saturating_add(1);
                }
            }
            row.push(if sample_count > 0 {
                Some((sample_sum as f64 / sample_count as f64).round() as u16)
            } else {
                None
            });
            x += step_x;
        }
        sampled_rows.push(row);
        y += step_y;
    }
    Some(serde_json::json!({
        "width": width,
        "height": height,
        "sampled_width": sampled_rows.first().map(|row| row.len()).unwrap_or(0),
        "sampled_height": sampled_rows.len(),
        "source_time_ns": source_time_ns,
        "selected_frame_index": frame_count.saturating_sub(1),
        "min_depth_mm": min_depth,
        "max_depth_mm": max_depth,
        "mean_depth_mm": if valid_total > 0 { Some((depth_sum as f64 / valid_total as f64 * 10.0).round() / 10.0) } else { None::<f64> },
        "valid_pixel_count": valid_total,
        "coverage_ratio": if width * height > 0 { valid_total as f64 / (width * height) as f64 } else { 0.0 },
        "sampled_depth_mm": sampled_rows,
    }))
}

fn read_u32_le(data: &[u8], offset: usize) -> Option<u32> {
    let bytes: [u8; 4] = data.get(offset..offset + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn read_u64_le(data: &[u8], offset: usize) -> Option<u64> {
    let bytes: [u8; 8] = data.get(offset..offset + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

pub(crate) async fn build_stereo_watchdog_payload(state: &AppState) -> Value {
    let status_path = PathBuf::from(state.config.stereo_watchdog_status_path.trim());
    let preview_path = PathBuf::from(state.config.stereo_preview_path.trim());
    let left_frame_path = PathBuf::from(state.config.stereo_left_frame_path.trim());
    let right_frame_path = PathBuf::from(state.config.stereo_right_frame_path.trim());
    let stereo_service = systemd_properties("chek-edge-stereo.service").await;
    let watchdog_service = systemd_properties("chek-edge-stereo-watchdog.service").await;
    let mut payload = match tokio::fs::read_to_string(&status_path).await {
        Ok(content) => {
            serde_json::from_str::<Value>(&content).unwrap_or_else(|_| Value::Object(Map::new()))
        }
        Err(_) => Value::Object(Map::new()),
    };

    if !payload.is_object() {
        payload = Value::Object(Map::new());
    }

    let preview_age = file_age_sec(&preview_path);
    let left_frame_age = file_age_sec(&left_frame_path);
    let right_frame_age = file_age_sec(&right_frame_path);
    let freshest_frame = [
        ("preview", preview_age),
        ("left", left_frame_age),
        ("right", right_frame_age),
    ]
    .into_iter()
    .filter_map(|(source, age)| age.map(|value| (source, value)))
    .min_by(|(_, left), (_, right)| left.total_cmp(right));
    let freshest_frame_source = freshest_frame.map(|(source, _)| source);
    let freshest_frame_age = freshest_frame.map(|(_, age)| age);
    let fallback_frame_fresh = freshest_frame_age.map(|age| age <= 2.5).unwrap_or(false);
    let object = payload
        .as_object_mut()
        .expect("payload normalized to object");
    object.insert(
        "status_path".to_string(),
        Value::String(status_path.display().to_string()),
    );
    object.insert(
        "preview_path".to_string(),
        Value::String(preview_path.display().to_string()),
    );
    object.insert(
        "left_frame_path".to_string(),
        Value::String(left_frame_path.display().to_string()),
    );
    object.insert(
        "right_frame_path".to_string(),
        Value::String(right_frame_path.display().to_string()),
    );
    object.insert(
        "stereo_service_state".to_string(),
        Value::String(systemd_property_value(
            &stereo_service,
            "ActiveState",
            "unknown",
        )),
    );
    object.insert(
        "stereo_service_substate".to_string(),
        Value::String(systemd_property_value(
            &stereo_service,
            "SubState",
            "unknown",
        )),
    );
    object.insert(
        "watchdog_service_state".to_string(),
        Value::String(systemd_property_value(
            &watchdog_service,
            "ActiveState",
            "unknown",
        )),
    );
    object.insert(
        "watchdog_service_substate".to_string(),
        Value::String(systemd_property_value(
            &watchdog_service,
            "SubState",
            "unknown",
        )),
    );
    object.insert(
        "watchdog_enabled".to_string(),
        Value::Bool(matches!(
            systemd_property_value(&watchdog_service, "ActiveState", "unknown").as_str(),
            "active" | "activating"
        )),
    );
    if let Some(age) = preview_age {
        object.insert("preview_age_sec".to_string(), Value::from(age));
    } else {
        object.insert("preview_age_sec".to_string(), Value::Null);
    }
    if let Some(age) = left_frame_age {
        object.insert("left_frame_age_sec".to_string(), Value::from(age));
    } else {
        object.insert("left_frame_age_sec".to_string(), Value::Null);
    }
    if let Some(age) = right_frame_age {
        object.insert("right_frame_age_sec".to_string(), Value::from(age));
    } else {
        object.insert("right_frame_age_sec".to_string(), Value::Null);
    }
    if let Some(age) = freshest_frame_age {
        object.insert("freshest_frame_age_sec".to_string(), Value::from(age));
    } else {
        object.insert("freshest_frame_age_sec".to_string(), Value::Null);
    }
    if let Some(source) = freshest_frame_source {
        object.insert(
            "freshest_frame_source".to_string(),
            Value::String(source.to_string()),
        );
    } else {
        object.insert("freshest_frame_source".to_string(), Value::Null);
    }
    object
        .entry("healthy".to_string())
        .or_insert_with(|| Value::Bool(fallback_frame_fresh));
    object.entry("status".to_string()).or_insert_with(|| {
        Value::String(if fallback_frame_fresh {
            "healthy".to_string()
        } else {
            "unknown".to_string()
        })
    });
    payload
}

fn file_age_sec(path: &Path) -> Option<f64> {
    let modified = path.metadata().ok()?.modified().ok()?;
    let elapsed = SystemTime::now().duration_since(modified).ok()?;
    Some(elapsed.as_secs_f64())
}

async fn systemd_properties(service_name: &str) -> Map<String, Value> {
    let output = match Command::new("systemctl")
        .args(["show", service_name, "-p", "ActiveState", "-p", "SubState"])
        .output()
        .await
    {
        Ok(output) if output.status.success() => output,
        _ => return Map::new(),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut props = Map::new();
    for line in stdout.lines() {
        if let Some((key, value)) = line.split_once('=') {
            props.insert(key.to_string(), Value::String(value.to_string()));
        }
    }
    props
}

fn systemd_property_value(props: &Map<String, Value>, key: &str, fallback: &str) -> String {
    props
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or(fallback)
        .to_string()
}

async fn serve_file(path: PathBuf, head_only: bool, cacheable: bool) -> Response {
    let payload = match tokio::fs::read(&path).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (StatusCode::NOT_FOUND, format!("静态资源读取失败: {error}")).into_response();
        }
    };

    let builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type_for(&path))
        .header(
            header::CACHE_CONTROL,
            if cacheable {
                "public, max-age=3600"
            } else {
                "no-store"
            },
        )
        .header(header::CONTENT_LENGTH, payload.len().to_string());

    if head_only {
        return builder.body(Body::empty()).unwrap_or_else(|_| {
            (StatusCode::INTERNAL_SERVER_ERROR, "静态响应构造失败").into_response()
        });
    }

    builder
        .body(Body::from(payload))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "静态响应构造失败").into_response())
}

fn content_type_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
    {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "json" | "map" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "mp4" => "video/mp4",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::{OperatorEstimate, OperatorSnapshot};
    use crate::sensing::{
        StereoSnapshot, StereoTrackedPersonSnapshot, VisionSnapshot, WifiPoseDiagnostics,
        WifiPoseSnapshot,
    };
    use crate::SimTrackingCarrySnapshot;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_test_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("chek-workstation-{prefix}-{unique}"))
    }

    #[tokio::test]
    async fn live_vlm_summary_exposes_preview_and_recent_entries() {
        let session_dir = unique_test_dir("vlm-summary");
        fs::create_dir_all(session_dir.join("preview")).unwrap();
        fs::create_dir_all(session_dir.join("preview").join("keyframes")).unwrap();
        fs::create_dir_all(session_dir.join("preview").join("clips")).unwrap();
        fs::create_dir_all(session_dir.join("derived").join("vision")).unwrap();

        fs::write(
            session_dir
                .join("preview")
                .join("keyframes")
                .join("keyframe_000001.jpg"),
            b"jpeg",
        )
        .unwrap();
        fs::write(
            session_dir
                .join("preview")
                .join("clips")
                .join("preview_clip_000001.gif"),
            b"gif",
        )
        .unwrap();
        fs::write(
            session_dir.join("preview").join("preview_manifest.json"),
            serde_json::json!({
                "type": "preview_manifest",
                "schema_version": "1.0.0",
                "trip_id": "trip-1",
                "session_id": "session-1",
                "generated_unix_ms": 1,
                "runtime_profile": "capture_plus_vlm",
                "upload_policy_mode": "metadata_plus_preview",
                "raw_residency_default": "edge_only",
                "preview_residency_default": "cloud_preview_only",
                "preview_generation_enabled": true,
                "vlm_indexing_enabled": true,
                "model_id": "smolvlm2-500m",
                "prompt_version": "prompt-v1",
                "status": "ready",
                "vlm_status": "ready",
                "degraded_reasons": [],
                "keyframes": [{
                    "id": "keyframe_000001",
                    "relpath": "preview/keyframes/keyframe_000001.jpg",
                    "frame_id": 11,
                    "source_time_ns": 12,
                    "edge_time_ns": 13,
                    "camera_mode": "wide",
                    "sample_reasons": ["fixed_interval"],
                    "caption": "operator reaches toward mug",
                    "tags": ["reach", "mug"],
                    "action_guess": "reaching",
                    "embedding_relpath": "derived/vision/embeddings/event_000001.json"
                }],
                "clips": [{
                    "id": "preview_clip_000001",
                    "relpath": "preview/clips/preview_clip_000001.gif",
                    "mime_type": "image/gif",
                    "status": "ready",
                    "start_edge_time_ns": 10,
                    "end_edge_time_ns": 20,
                    "keyframe_ids": ["keyframe_000001"],
                    "caption": "reach segment around mug",
                    "tags": ["reach", "mug"],
                    "action_guess": "reaching",
                    "embedding_relpath": "derived/vision/embeddings/segment_000001.json"
                }]
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            session_dir
                .join("derived")
                .join("vision")
                .join("vlm_events.jsonl"),
            format!(
                "{}\n",
                serde_json::json!({
                    "event_id": "event_000001",
                    "edge_time_ns": 13,
                    "camera_mode": "wide",
                    "frame_relpath": "preview/keyframes/keyframe_000001.jpg",
                    "caption": "operator reaches toward mug",
                    "action_guess": "reaching",
                    "tags": ["reach", "mug"],
                    "objects": ["mug"],
                    "sample_reasons": ["fixed_interval"],
                    "model_id": "smolvlm2-500m",
                    "prompt_version": "prompt-v1",
                })
            ),
        )
        .unwrap();
        fs::write(
            session_dir
                .join("derived")
                .join("vision")
                .join("vlm_segments.jsonl"),
            format!(
                "{}\n",
                serde_json::json!({
                    "segment_id": "segment_000001",
                    "start_edge_time_ns": 10,
                    "end_edge_time_ns": 20,
                    "camera_mode": "wide",
                    "caption": "reach segment around mug",
                    "action_guess": "reaching",
                    "tags": ["reach", "mug"],
                    "objects": ["mug"],
                    "event_ids": ["event_000001"],
                    "keyframe_ids": ["keyframe_000001"],
                    "model_id": "smolvlm2-500m",
                    "prompt_version": "prompt-v1",
                })
            ),
        )
        .unwrap();

        let summary = build_live_vlm_summary(Some(&session_dir), true, true).await;

        assert_eq!(
            summary.get("status").and_then(|value| value.as_str()),
            Some("ready")
        );
        assert_eq!(
            summary.get("model_id").and_then(|value| value.as_str()),
            Some("smolvlm2-500m")
        );
        assert_eq!(
            summary
                .get("latest_caption")
                .and_then(|value| value.as_str()),
            Some("operator reaches toward mug")
        );
        assert_eq!(
            summary
                .get("latest_keyframe")
                .and_then(|value| value.get("preview_url"))
                .and_then(|value| value.as_str()),
            Some("/live-preview/file/preview/keyframes/keyframe_000001.jpg")
        );
        assert_eq!(
            summary
                .get("recent_events")
                .and_then(|value| value.as_array())
                .map(|items| items.len()),
            Some(1)
        );
        assert_eq!(
            summary
                .get("recent_segments")
                .and_then(|value| value.as_array())
                .map(|items| items.len()),
            Some(1)
        );

        let _ = fs::remove_dir_all(&session_dir);
    }

    #[test]
    fn two_d_only_phone_pose_keeps_phone_session_associated_stereo_target() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_2d: vec![[0.1, 0.2]; 17],
            body_conf: 0.92,
            ..VisionSnapshot::default()
        };

        let mut estimate = OperatorEstimate::default();
        estimate.association.selected_operator_track_id = Some("stereo-person-1".to_string());
        estimate.association.stereo_operator_track_id = Some("stereo-person-1".to_string());
        estimate.association.iphone_operator_track_id = Some("primary_operator".to_string());

        let operator = OperatorSnapshot {
            estimate,
            fresh: false,
        };

        let stereo = StereoSnapshot {
            fresh: true,
            operator_track_id: Some("stereo-person-1".to_string()),
            persons: vec![StereoTrackedPersonSnapshot {
                operator_track_id: Some("stereo-person-1".to_string()),
                stereo_confidence: 0.83,
                body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
                ..StereoTrackedPersonSnapshot::default()
            }],
            ..StereoSnapshot::default()
        };

        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            0,
        );

        assert!(resolved.has_target);
        assert_eq!(
            resolved.source.as_deref(),
            Some("phone_session_associated_stereo")
        );
        assert_eq!(resolved.stereo_track_id.as_deref(), Some("stereo-person-1"));
        assert_eq!(
            resolved.target_person_id.as_deref(),
            Some("primary_operator")
        );
        assert_eq!(resolved.body_kpts_3d.len(), 17);
    }

    #[test]
    fn single_stereo_person_fallback_keeps_phone_session_associated_stereo_target() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
            hand_kpts_3d: vec![[0.1, 0.2, 1.0]; 42],
            ..VisionSnapshot::default()
        };

        let mut estimate = OperatorEstimate::default();
        estimate.association.anchor_source = "iphone_hand";
        estimate.association.iphone_operator_track_id = Some("primary_operator".to_string());

        let operator = OperatorSnapshot {
            estimate,
            fresh: false,
        };

        let stereo = StereoSnapshot {
            fresh: true,
            operator_track_id: None,
            persons: vec![StereoTrackedPersonSnapshot {
                operator_track_id: Some("stereo-person-1".to_string()),
                stereo_confidence: 0.77,
                body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
                ..StereoTrackedPersonSnapshot::default()
            }],
            ..StereoSnapshot::default()
        };

        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            0,
        );

        assert!(resolved.has_target);
        assert_eq!(
            resolved.source.as_deref(),
            Some("phone_session_associated_stereo")
        );
        assert_eq!(resolved.stereo_track_id.as_deref(), Some("stereo-person-1"));
    }

    #[test]
    fn stereo_hand_fills_missing_phone_hand_geometry() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
            hand_kpts_3d: Vec::new(),
            ..VisionSnapshot::default()
        };

        let mut estimate = OperatorEstimate::default();
        estimate.association.anchor_source = "iphone_hand";
        estimate.association.selected_operator_track_id = Some("stereo-person-1".to_string());
        estimate.association.stereo_operator_track_id = Some("stereo-person-1".to_string());
        estimate.association.iphone_operator_track_id = Some("primary_operator".to_string());

        let operator = OperatorSnapshot {
            estimate,
            fresh: false,
        };

        let stereo = StereoSnapshot {
            fresh: true,
            operator_track_id: Some("stereo-person-1".to_string()),
            persons: vec![StereoTrackedPersonSnapshot {
                operator_track_id: Some("stereo-person-1".to_string()),
                stereo_confidence: 0.83,
                body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
                hand_kpts_3d: vec![[0.1, 0.2, 1.0]; 42],
                ..StereoTrackedPersonSnapshot::default()
            }],
            ..StereoSnapshot::default()
        };

        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            0,
        );

        assert!(resolved.has_target);
        assert_eq!(
            resolved.source.as_deref(),
            Some("phone_session_associated_stereo")
        );
        assert_eq!(resolved.hand_3d_source, "stereo");
        assert_eq!(resolved.hand_kpts_3d.len(), 42);
    }

    #[test]
    fn startup_stereo_gap_keeps_phone_session_associated_stereo_target() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
            hand_kpts_3d: vec![[0.1, 0.2, 1.0]; 42],
            body_3d_source: "edge_depth_reprojected_hand_context_hold".to_string(),
            hand_3d_source: "edge_depth_reprojected".to_string(),
            last_edge_time_ns: 5_000_000_000,
            ..VisionSnapshot::default()
        };

        let operator = OperatorSnapshot {
            estimate: OperatorEstimate::default(),
            fresh: false,
        };
        let stereo = StereoSnapshot::default();
        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            5_000_000_000,
        );

        assert!(resolved.has_target);
        assert_eq!(
            resolved.source.as_deref(),
            Some("phone_session_associated_stereo")
        );
        assert_eq!(
            resolved.association_anchor_source,
            "stereo_startup_warmup_hold"
        );
        assert_eq!(
            resolved.body_3d_source,
            "edge_depth_reprojected_hand_context_hold"
        );
        assert_eq!(resolved.hand_3d_source, "edge_depth_reprojected");
    }

    #[test]
    fn startup_stereo_gap_grace_expiry_falls_back_to_phone_edge_vision() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
            hand_kpts_3d: vec![[0.1, 0.2, 1.0]; 42],
            body_3d_source: "edge_depth_reprojected_hand_context_hold".to_string(),
            hand_3d_source: "edge_depth_reprojected".to_string(),
            last_edge_time_ns: 25_000_000_000,
            ..VisionSnapshot::default()
        };

        let operator = OperatorSnapshot {
            estimate: OperatorEstimate::default(),
            fresh: false,
        };
        let stereo = StereoSnapshot::default();
        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            25_000_000_000,
        );

        assert!(resolved.has_target);
        assert_eq!(
            resolved.source.as_deref(),
            Some("phone_session_associated_stereo")
        );
        assert_eq!(
            resolved.association_anchor_source,
            "runtime_stereo_association_gap_hold"
        );
    }

    #[test]
    fn runtime_stereo_association_gap_hold_keeps_phone_session_associated_stereo_target() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
            hand_kpts_3d: vec![[0.1, 0.2, 1.0]; 42],
            body_3d_source: "edge_depth_reprojected".to_string(),
            hand_3d_source: "edge_depth_reprojected".to_string(),
            last_edge_time_ns: 20_500_000_000,
            ..VisionSnapshot::default()
        };

        let mut estimate = OperatorEstimate::default();
        estimate.association.anchor_source = "iphone_hand";
        estimate.association.iphone_operator_track_id = Some("primary_operator".to_string());
        estimate.updated_edge_time_ns = 20_400_000_000;

        let operator = OperatorSnapshot {
            estimate,
            fresh: false,
        };
        let stereo = StereoSnapshot {
            fresh: false,
            last_edge_time_ns: 20_300_000_000,
            ..StereoSnapshot::default()
        };
        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            21_000_000_000,
        );

        assert!(resolved.has_target);
        assert_eq!(
            resolved.source.as_deref(),
            Some("phone_session_associated_stereo")
        );
        assert_eq!(
            resolved.association_anchor_source,
            "runtime_stereo_association_gap_hold"
        );
        assert_eq!(resolved.body_3d_source, "edge_depth_reprojected");
    }

    #[test]
    fn runtime_stereo_association_gap_hold_accepts_stereo_derived_phone_geometry_without_hint() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
            hand_kpts_3d: vec![[0.1, 0.2, 1.0]; 42],
            body_3d_source: "edge_depth_reprojected_hand_context_hold".to_string(),
            hand_3d_source: "edge_depth_reprojected".to_string(),
            last_edge_time_ns: 30_500_000_000,
            ..VisionSnapshot::default()
        };

        let operator = OperatorSnapshot {
            estimate: OperatorEstimate {
                updated_edge_time_ns: 30_400_000_000,
                ..OperatorEstimate::default()
            },
            fresh: false,
        };
        let stereo = StereoSnapshot {
            fresh: false,
            last_edge_time_ns: 30_300_000_000,
            ..StereoSnapshot::default()
        };
        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            31_000_000_000,
        );

        assert!(resolved.has_target);
        assert_eq!(
            resolved.source.as_deref(),
            Some("phone_session_associated_stereo")
        );
        assert_eq!(
            resolved.association_anchor_source,
            "runtime_stereo_association_gap_hold"
        );
        assert_eq!(
            resolved.body_3d_source,
            "edge_depth_reprojected_hand_context_hold"
        );
        assert_eq!(resolved.hand_3d_source, "edge_depth_reprojected");
    }

    #[test]
    fn runtime_stereo_association_gap_hold_expiry_falls_back_to_phone_edge_vision() {
        let live_session = LiveSessionContext {
            trip_id: "trip-1".to_string(),
            session_id: "session-1".to_string(),
            session_dir: PathBuf::from("/tmp/session-1"),
        };

        let vision = VisionSnapshot {
            fresh: true,
            execution_mode: "edge_authoritative_phone_vision".to_string(),
            metrics_session_id: "session-1".to_string(),
            operator_track_id: Some("primary_operator".to_string()),
            body_kpts_3d: vec![[0.1, 0.2, 1.0]; 17],
            hand_kpts_3d: vec![[0.1, 0.2, 1.0]; 42],
            body_3d_source: "edge_depth_reprojected".to_string(),
            hand_3d_source: "edge_depth_reprojected".to_string(),
            last_edge_time_ns: 20_000_000_000,
            ..VisionSnapshot::default()
        };

        let mut estimate = OperatorEstimate::default();
        estimate.association.anchor_source = "iphone_hand";
        estimate.association.iphone_operator_track_id = Some("primary_operator".to_string());
        estimate.updated_edge_time_ns = 19_900_000_000;

        let operator = OperatorSnapshot {
            estimate,
            fresh: false,
        };
        let stereo = StereoSnapshot {
            fresh: false,
            last_edge_time_ns: 19_800_000_000,
            ..StereoSnapshot::default()
        };
        let wifi_pose = WifiPoseSnapshot {
            diagnostics: WifiPoseDiagnostics::default(),
            ..WifiPoseSnapshot::default()
        };

        let resolved = resolve_target_human_state(
            Some(&live_session),
            &operator,
            &vision,
            &stereo,
            &wifi_pose,
            None,
            28_000_000_000,
        );

        assert!(resolved.has_target);
        assert_eq!(resolved.source.as_deref(), Some("phone_edge_vision"));
    }

    #[test]
    fn sim_tracking_payload_ready_only_when_matching_current_human_session() {
        let payload = json!({
            "available": true,
            "ready": true,
            "control_state": "armed",
            "drive_mode": "human_session_driven",
            "source_session_id": "session-1",
            "source_person_id": "primary_operator",
        });

        assert!(sim_tracking_payload_is_human_session_ready(
            &payload,
            Some("session-1"),
            Some("primary_operator"),
        ));
        assert!(!sim_tracking_payload_is_human_session_ready(
            &payload,
            Some("session-2"),
            Some("primary_operator"),
        ));
    }

    #[test]
    fn carried_sim_tracking_payload_marks_observed_degraded_runtime() {
        let snapshot = SimTrackingCarrySnapshot {
            payload: json!({
                "available": true,
                "ready": true,
                "detail": "Grounded WBC 已对齐当前人体主链。",
                "control_state": "armed",
                "drive_mode": "human_session_driven",
                "source_session_id": "session-1",
                "source_person_id": "primary_operator",
            }),
            target_session_id: Some("session-1".to_string()),
            target_person_id: Some("primary_operator".to_string()),
            updated_edge_time_ns: 1,
        };
        let observed = json!({
            "available": true,
            "ready": false,
            "control_state": "disarmed",
            "motion_reason": "retarget_reference_disarmed",
            "drive_mode": "unbound",
        });

        let carried =
            carried_sim_tracking_payload(&snapshot, &observed, "runtime_control_state_degraded");

        assert_eq!(carried.get("carried_forward"), Some(&Value::Bool(true)));
        assert_eq!(
            carried.get("observed_control_state"),
            Some(&Value::String("disarmed".to_string()))
        );
        assert_eq!(
            carried.get("carry_reason"),
            Some(&Value::String("runtime_control_state_degraded".to_string()))
        );
    }
}
