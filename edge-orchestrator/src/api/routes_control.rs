use axum::extract::State;
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::recorder::session_recorder::SessionContextUpdate;
use crate::AppState;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ArmRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub robot_type: String,
    pub end_effector_type: String,
    pub operator_id: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct DisarmRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub reason: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct SetControlProfileRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub teleop_enabled: bool,
    pub body_control_enabled: bool,
    pub hand_control_enabled: bool,
}

#[derive(Serialize)]
pub struct ControlStateResponse {
    pub schema_version: &'static str,
    pub trip_id: String,
    pub session_id: String,
    pub state: String,
    pub reason: String,
    pub runtime_profile: String,
    pub upload_policy_mode: String,
    pub raw_residency: String,
    pub preview_residency: String,
    pub feature_flags: crate::config::RuntimeFeatureFlags,
    pub crowd_upload_enabled: bool,
    pub preflight: Preflight,
    pub deadman: DeadmanState,
    pub operation_profile: OperationProfile,
    pub phone_capture_commands: PhoneCaptureCommands,
}

#[derive(Serialize)]
pub struct Preflight {
    pub unitree_bridge_ready: bool,
    pub leap_bridge_ready: bool,
    pub time_sync_ok: bool,
    pub extrinsic_ok: bool,
    pub lan_control_ok: bool,
}

#[derive(Serialize)]
pub struct DeadmanState {
    pub enabled: bool,
    pub timeout_ms: u64,
    pub link_ok: bool,
    pub pressed: bool,
    pub keepalive: DeadmanKeepaliveState,
}

#[derive(Serialize)]
pub struct DeadmanKeepaliveState {
    pub last_age_ms: Option<u64>,
    pub last_edge_time_ns: Option<u64>,
    pub last_source_time_ns: Option<u64>,
    pub last_seq: Option<u64>,
    pub last_trip_id: String,
    pub last_session_id: String,
    pub last_device_id: String,
    pub packets_in_last_5s: u64,
    pub approx_rate_hz_5s: f32,
    pub session_matches_active: bool,
}

#[derive(Serialize)]
pub struct OperationProfile {
    pub teleop_enabled: bool,
    pub body_control_enabled: bool,
    pub hand_control_enabled: bool,
}

#[derive(Serialize)]
pub struct PhoneCaptureCommands {
    pub aux_snapshot_request_seq: u64,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct RequestAuxSnapshotRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub reason: Option<String>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/control/state", get(get_state))
        .route("/control/arm", post(arm))
        .route("/control/disarm", post(disarm))
        .route("/control/keepalive", post(keepalive))
        .route("/control/profile", post(set_profile))
        .route(
            "/control/phone_capture/request_aux_snapshot",
            post(request_aux_snapshot),
        )
        .with_state(state)
}

async fn get_state(State(state): State<AppState>) -> Json<ControlStateResponse> {
    Json(build_control_state_response(&state))
}

async fn arm(
    State(state): State<AppState>,
    Json(req): Json<ArmRequest>,
) -> Json<ControlStateResponse> {
    if !state.config.control_enabled {
        warn!(
            trip_id = %req.trip_id,
            session_id = %req.session_id,
            runtime_profile = state.config.runtime_profile_name(),
            "arm ignored because control is disabled by runtime profile"
        );
        state.session.set_active(req.trip_id, req.session_id);
        state.gate.disarm("control_disabled");
        return Json(build_control_state_response(&state));
    }
    state
        .session
        .set_active(req.trip_id.clone(), req.session_id.clone());
    state
        .recorder
        .update_session_context(
            &state.protocol,
            &state.config,
            &req.trip_id,
            &req.session_id,
            SessionContextUpdate {
                capture_device_id: None,
                operator_id: Some(req.operator_id.clone()),
                task_id: None,
                task_ids: Vec::new(),
                runtime_profile: None,
                upload_policy_mode: None,
                raw_residency: None,
                preview_residency: None,
            },
        )
        .await;
    state
        .gate
        .set_robot_hints(&req.robot_type, &req.end_effector_type);
    let preflight = compute_preflight(&state);
    let deadman = state.gate.deadman_snapshot();

    let fail_reason = compute_arm_preflight_fail_reason(&req, &preflight, &deadman);
    if fail_reason.is_none() {
        state.gate.arm();
        metrics::counter!("control_arm_count").increment(1);
    } else {
        let reason = fail_reason.unwrap_or(crate::reason::REASON_UNKNOWN);
        warn!(reason, "arm preflight failed");
        state.gate.fault(reason);
        metrics::counter!("control_arm_fail_count", "reason" => reason).increment(1);
    }

    Json(build_control_state_response(&state))
}

async fn disarm(
    State(state): State<AppState>,
    Json(req): Json<DisarmRequest>,
) -> Json<ControlStateResponse> {
    state.gate.disarm(&req.reason);
    metrics::counter!("control_disarm_count").increment(1);
    Json(build_control_state_response(&state))
}

async fn keepalive(
    State(state): State<AppState>,
    Json(req): Json<crate::control::keepalive::ControlKeepalivePacket>,
) -> Json<ControlStateResponse> {
    state
        .session
        .set_active(req.trip_id.clone(), req.session_id.clone());
    state.gate.ingest_keepalive(
        &req.trip_id,
        &req.session_id,
        &req.device_id,
        req.source_time_ns,
        req.seq,
        req.deadman_pressed,
    );
    metrics::counter!("control_keepalive_http_count").increment(1);
    Json(build_control_state_response(&state))
}

async fn set_profile(
    State(state): State<AppState>,
    Json(req): Json<SetControlProfileRequest>,
) -> Json<ControlStateResponse> {
    let teleop_enabled = state.config.control_enabled && req.teleop_enabled;
    let body_control_enabled = teleop_enabled && req.body_control_enabled;
    let hand_control_enabled = teleop_enabled && req.hand_control_enabled;
    info!(
        trip_id = %req.trip_id,
        session_id = %req.session_id,
        teleop_enabled,
        body_control_enabled,
        hand_control_enabled,
        "control profile updated"
    );
    state.session.set_active(req.trip_id, req.session_id);
    state
        .session
        .set_control_profile(teleop_enabled, body_control_enabled, hand_control_enabled);
    if !teleop_enabled {
        state.gate.disarm("capture_only");
    }
    metrics::counter!("control_profile_set_count").increment(1);
    Json(build_control_state_response(&state))
}

async fn request_aux_snapshot(
    State(state): State<AppState>,
    Json(req): Json<RequestAuxSnapshotRequest>,
) -> Json<ControlStateResponse> {
    state.session.set_active(req.trip_id, req.session_id);
    let seq = state.phone_capture_commands.request_aux_snapshot();
    metrics::counter!("control_phone_aux_snapshot_request_count").increment(1);
    tracing::info!(seq, reason=?req.reason, "queued manual aux snapshot request");
    Json(build_control_state_response(&state))
}

fn build_control_state_response(state: &AppState) -> ControlStateResponse {
    let session = state.session.snapshot();
    let gate = state.gate.snapshot();
    let preflight = compute_preflight(state);
    let deadman = state.gate.deadman_snapshot();
    let phone_capture_commands = state.phone_capture_commands.snapshot();
    let deadman_session_matches_active = !session.trip_id.is_empty()
        && !session.session_id.is_empty()
        && session.trip_id == deadman.keepalive.last_trip_id
        && session.session_id == deadman.keepalive.last_session_id;

    ControlStateResponse {
        schema_version: "1.0.0",
        trip_id: session.trip_id,
        session_id: session.session_id,
        state: gate.state,
        reason: gate.reason,
        runtime_profile: session.runtime_profile,
        upload_policy_mode: session.upload_policy_mode,
        raw_residency: session.raw_residency,
        preview_residency: session.preview_residency,
        feature_flags: session.feature_flags,
        crowd_upload_enabled: session.crowd_upload_enabled,
        preflight,
        deadman: DeadmanState {
            enabled: deadman.enabled,
            timeout_ms: deadman.timeout_ms,
            link_ok: deadman.link_ok,
            pressed: deadman.pressed,
            keepalive: DeadmanKeepaliveState {
                last_age_ms: deadman.keepalive.last_age_ms,
                last_edge_time_ns: deadman.keepalive.last_edge_time_ns,
                last_source_time_ns: deadman.keepalive.last_source_time_ns,
                last_seq: deadman.keepalive.last_seq,
                last_trip_id: deadman.keepalive.last_trip_id,
                last_session_id: deadman.keepalive.last_session_id,
                last_device_id: deadman.keepalive.last_device_id,
                packets_in_last_5s: deadman.keepalive.packets_in_last_5s,
                approx_rate_hz_5s: deadman.keepalive.approx_rate_hz_5s,
                session_matches_active: deadman_session_matches_active,
            },
        },
        operation_profile: OperationProfile {
            teleop_enabled: session.teleop_enabled,
            body_control_enabled: session.body_control_enabled,
            hand_control_enabled: session.hand_control_enabled,
        },
        phone_capture_commands: PhoneCaptureCommands {
            aux_snapshot_request_seq: phone_capture_commands.aux_snapshot_request_seq,
        },
    }
}

fn compute_preflight(state: &AppState) -> Preflight {
    let bridge = state.bridge_store.snapshot(state.config.bridge_stale_ms);

    Preflight {
        unitree_bridge_ready: bridge.unitree_ready,
        leap_bridge_ready: bridge.leap_ready,
        time_sync_ok: state.gate.time_sync_ok(
            state.config.time_sync_ok_window_ms,
            state.config.time_sync_rtt_ok_ms,
        ),
        extrinsic_ok: state.gate.extrinsic_ok(),
        lan_control_ok: bridge.lan_control_ok,
    }
}

fn compute_arm_preflight_fail_reason(
    req: &ArmRequest,
    preflight: &Preflight,
    deadman: &crate::control::gate::DeadmanSnapshot,
) -> Option<&'static str> {
    if !preflight.unitree_bridge_ready {
        return Some(crate::reason::REASON_UNITREE_BRIDGE_UNREADY);
    }
    if req.end_effector_type == "LEAP_V2" && !preflight.leap_bridge_ready {
        return Some(crate::reason::REASON_LEAP_BRIDGE_UNREADY);
    }
    if !preflight.time_sync_ok {
        return Some(crate::reason::REASON_TIME_SYNC_UNREADY);
    }
    if !preflight.extrinsic_ok {
        return Some(crate::reason::REASON_EXTRINSIC_UNREADY);
    }
    if !preflight.lan_control_ok {
        return Some(crate::reason::REASON_LAN_CONTROL_UNREADY);
    }
    if deadman.enabled && !deadman.link_ok {
        return Some(crate::reason::REASON_DEADMAN_LINK_UNREADY);
    }
    None
}
