use axum::extract::State;
use axum::response::IntoResponse;
use axum::{routing::get, Json, Router};
use serde::Serialize;
use serde_json::Value;

use crate::AppState;

#[derive(Serialize)]
struct HealthResponse<'a> {
    status: &'a str,
    protocol: crate::protocol::version_guard::ProtocolVersionInfo,
    bridge: BridgeHealth,
    stereo: StereoHealth,
}

#[derive(Serialize)]
struct BridgeHealth {
    unitree_ready: bool,
    leap_ready: bool,
    lan_control_ok: bool,
}

#[derive(Serialize)]
struct StereoHealth {
    available: bool,
    fresh: bool,
    body_count: usize,
    operator_track_id: Option<String>,
    service_state: String,
    watchdog_service_state: String,
    watchdog_enabled: bool,
    watchdog_status: String,
    healthy: bool,
    preview_age_sec: Option<f64>,
    left_frame_age_sec: Option<f64>,
    right_frame_age_sec: Option<f64>,
    freshest_frame_age_sec: Option<f64>,
    freshest_frame_source: Option<String>,
    reasons: Vec<String>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let bridge = state.bridge_store.snapshot(state.config.bridge_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let stereo_watchdog = super::routes_workstation::build_stereo_watchdog_payload(&state).await;
    Json(HealthResponse {
        status: "ok",
        protocol: state.protocol.clone(),
        bridge: BridgeHealth {
            unitree_ready: bridge.unitree_ready,
            leap_ready: bridge.leap_ready,
            lan_control_ok: bridge.lan_control_ok,
        },
        stereo: build_stereo_health(
            stereo.fresh,
            stereo.body_kpts_3d.len(),
            stereo.operator_track_id.clone(),
            &stereo_watchdog,
        ),
    })
}

async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics_handle.render()
}

fn json_string(payload: &Value, key: &str, fallback: &str) -> String {
    payload
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or(fallback)
        .to_string()
}

fn json_bool(payload: &Value, key: &str, fallback: bool) -> bool {
    payload
        .get(key)
        .and_then(|value| value.as_bool())
        .unwrap_or(fallback)
}

fn json_string_vec(payload: &Value, key: &str) -> Vec<String> {
    payload
        .get(key)
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(|text| text.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn build_stereo_health(
    fresh: bool,
    body_count: usize,
    operator_track_id: Option<String>,
    stereo_watchdog: &Value,
) -> StereoHealth {
    let available = fresh && body_count > 0;
    let service_state = json_string(
        stereo_watchdog,
        "stereo_service_state",
        if fresh { "active" } else { "unknown" },
    );
    let watchdog_service_state = json_string(stereo_watchdog, "watchdog_service_state", "unknown");
    let watchdog_enabled = json_bool(stereo_watchdog, "watchdog_enabled", false);
    let watchdog_status = json_string(stereo_watchdog, "status", "unknown");
    let watchdog_healthy = json_bool(stereo_watchdog, "healthy", available);
    let preview_age_sec = stereo_watchdog
        .get("preview_age_sec")
        .and_then(|value| value.as_f64());
    let left_frame_age_sec = stereo_watchdog
        .get("left_frame_age_sec")
        .and_then(|value| value.as_f64());
    let right_frame_age_sec = stereo_watchdog
        .get("right_frame_age_sec")
        .and_then(|value| value.as_f64());
    let freshest_frame_age_sec = stereo_watchdog
        .get("freshest_frame_age_sec")
        .and_then(|value| value.as_f64());
    let freshest_frame_source = stereo_watchdog
        .get("freshest_frame_source")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    let debug_frame_fresh = freshest_frame_age_sec
        .or(preview_age_sec)
        .map(|age| age <= 2.5)
        .unwrap_or(false);
    let mut reasons = json_string_vec(stereo_watchdog, "reasons");
    if !fresh && !debug_frame_fresh {
        reasons.push("stereo_snapshot_stale".to_string());
    }
    if body_count == 0 {
        reasons.push("stereo_body_missing".to_string());
    }
    if service_state != "active" && service_state != "activating" {
        reasons.push(format!("stereo_service_{service_state}"));
    }
    if watchdog_service_state != "active" && watchdog_service_state != "activating" {
        reasons.push(format!("stereo_watchdog_{watchdog_service_state}"));
    }
    if !watchdog_enabled {
        reasons.push("stereo_watchdog_disabled".to_string());
    }
    reasons.sort();
    reasons.dedup();

    StereoHealth {
        available,
        fresh,
        body_count,
        operator_track_id,
        service_state,
        watchdog_service_state,
        watchdog_enabled,
        watchdog_status,
        healthy: watchdog_healthy && available,
        preview_age_sec,
        left_frame_age_sec,
        right_frame_age_sec,
        freshest_frame_age_sec,
        freshest_frame_source,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_string_falls_back_when_missing() {
        let payload = json!({"status": "healthy"});
        assert_eq!(json_string(&payload, "status", "unknown"), "healthy");
        assert_eq!(json_string(&payload, "missing", "unknown"), "unknown");
    }

    #[test]
    fn stereo_health_marks_missing_body_as_unhealthy() {
        let payload = json!({
            "healthy": true,
            "status": "healthy",
            "stereo_service_state": "active",
            "watchdog_service_state": "active",
            "watchdog_enabled": true,
            "reasons": []
        });
        let stereo = build_stereo_health(false, 0, None, &payload);
        assert!(!stereo.healthy);
        assert!(!stereo.available);
        assert!(stereo
            .reasons
            .contains(&"stereo_snapshot_stale".to_string()));
        assert!(stereo.reasons.contains(&"stereo_body_missing".to_string()));
    }
}
