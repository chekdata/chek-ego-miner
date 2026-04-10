use axum::extract::State;
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Serialize)]
struct TimeResponse {
    edge_time_ns: u64,
}

#[derive(Serialize)]
struct TimeSyncDeviceResponse {
    device_id: String,
    source_kind: String,
    clock_offset_ns: i64,
    last_rtt_ns: u64,
    rtt_ok_ms: u64,
    age_ms: u64,
    fresh: bool,
    rtt_ok: bool,
}

#[derive(Serialize)]
struct TimeSyncSummaryResponse {
    ok: bool,
    ok_window_ms: u64,
    rtt_ok_ms: u64,
    device_count: usize,
    fresh_device_count: usize,
    fresh_ok_device_count: usize,
    stale_device_count: usize,
    last_any_age_ms: Option<u64>,
    last_any_rtt_ns: Option<u64>,
    devices: Vec<TimeSyncDeviceResponse>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct TimeSyncRequest {
    pub schema_version: String,
    #[serde(default)]
    pub trip_id: String,
    #[serde(default)]
    pub session_id: String,
    pub device_id: String,
    pub source_kind: Option<String>,
    pub clock_domain: Option<String>,
    pub clock_offset_ns: i64,
    pub rtt_ns: u64,
    pub sample_count: u32,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/time", get(get_time))
        .route("/time/sync/current", get(get_time_sync_current))
        .route("/time/sync", post(post_time_sync))
        .with_state(state)
}

async fn get_time(State(state): State<AppState>) -> Json<TimeResponse> {
    Json(TimeResponse {
        edge_time_ns: state.gate.edge_time_ns(),
    })
}

async fn get_time_sync_current(State(state): State<AppState>) -> Json<TimeSyncSummaryResponse> {
    let summary = state.gate.time_sync_summary();
    Json(TimeSyncSummaryResponse {
        ok: state.gate.time_sync_ok(
            state.config.time_sync_ok_window_ms,
            state.config.time_sync_rtt_ok_ms,
        ),
        ok_window_ms: state.config.time_sync_ok_window_ms,
        rtt_ok_ms: state.config.time_sync_rtt_ok_ms,
        device_count: summary.device_count,
        fresh_device_count: summary.fresh_device_count,
        fresh_ok_device_count: summary.fresh_ok_device_count,
        stale_device_count: summary.stale_device_count,
        last_any_age_ms: summary.last_any_age_ms,
        last_any_rtt_ns: summary.last_any_rtt_ns,
        devices: summary
            .devices
            .into_iter()
            .map(|device| TimeSyncDeviceResponse {
                device_id: device.device_id,
                source_kind: device.source_kind,
                clock_offset_ns: device.clock_offset_ns,
                last_rtt_ns: device.last_rtt_ns,
                rtt_ok_ms: device.rtt_ok_ms,
                age_ms: device.age_ms,
                fresh: device.fresh,
                rtt_ok: device.rtt_ok,
            })
            .collect(),
    })
}

async fn post_time_sync(
    State(state): State<AppState>,
    Json(req): Json<TimeSyncRequest>,
) -> Json<serde_json::Value> {
    let edge_time_ns = state.gate.edge_time_ns();
    let source_kind = req
        .source_kind
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let clock_domain = req
        .clock_domain
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    state.gate.record_time_sync(
        req.device_id.clone(),
        source_kind.clone(),
        req.clock_offset_ns,
        req.rtt_ns,
        req.sample_count,
    );
    let current_session = state.session.snapshot();
    let (trip_id, session_id, used_active_session) =
        if !req.trip_id.trim().is_empty() && !req.session_id.trim().is_empty() {
            (req.trip_id.clone(), req.session_id.clone(), false)
        } else if !current_session.trip_id.trim().is_empty()
            && !current_session.session_id.trim().is_empty()
        {
            (
                current_session.trip_id.clone(),
                current_session.session_id.clone(),
                true,
            )
        } else {
            (String::new(), String::new(), false)
        };
    let recorded = !trip_id.is_empty() && !session_id.is_empty();
    if recorded {
        let sample = serde_json::json!({
            "type": "time_sync_sample",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": req.device_id,
            "source_kind": source_kind,
            "clock_domain": clock_domain,
            "clock_offset_ns": req.clock_offset_ns,
            "rtt_ns": req.rtt_ns,
            "sample_count": req.sample_count,
            "edge_time_ns": edge_time_ns,
            "used_active_session": used_active_session,
            "recorded_unix_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        });
        let recorder = state.recorder.clone();
        let protocol = state.protocol.clone();
        let config = state.config.clone();
        let trip_id_for_record = sample
            .get("trip_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let session_id_for_record = sample
            .get("session_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        tokio::spawn(async move {
            recorder
                .record_time_sync_sample(
                    &protocol,
                    &config,
                    &trip_id_for_record,
                    &session_id_for_record,
                    &sample,
                )
                .await;
        });
    }
    metrics::counter!("time_sync_count").increment(1);
    Json(serde_json::json!({
        "ok": true,
        "recorded": recorded,
        "used_active_session": used_active_session,
        "edge_time_ns": edge_time_ns
    }))
}
