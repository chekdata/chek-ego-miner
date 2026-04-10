use std::net::SocketAddr;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{ConnectInfo, Query, State, WebSocketUpgrade};
use axum::http::HeaderMap;
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::control::keepalive::ControlKeepalivePacket;
use crate::operator::{OperatorPartSource, CANONICAL_BODY_FRAME, OPERATOR_FRAME};
use crate::ws::transport::{CompressionMode, TransportEncoder, TransportMode, TransportOptions};
use crate::ws::types::{
    FusionAssociationDebugView, FusionFusedDebugView, FusionMotionStateDebugView,
    FusionOperatorDebug, FusionSourceDebugView, FusionStatePacket, FusionTrackedPersonDebugView,
};
use crate::AppState;

#[derive(Deserialize)]
struct WsAuthQuery {
    token: Option<String>,
    format: Option<String>,
    transport: Option<String>,
    compression: Option<String>,
    debug_views: Option<String>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/stream/fusion", get(ws_upgrade))
        .with_state(state)
}

async fn ws_upgrade(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    Query(q): Query<WsAuthQuery>,
) -> axum::response::Response {
    if !crate::auth::ws_authorized(&state, q.token.as_deref(), &headers) {
        return crate::auth::ws_unauthorized_response();
    }
    let use_cbor = q
        .format
        .as_deref()
        .map(|s| s.eq_ignore_ascii_case("cbor"))
        .unwrap_or(false);
    let transport = TransportMode::parse(q.transport.as_deref());
    let compression = CompressionMode::parse(q.compression.as_deref());
    let include_operator_debug = q
        .debug_views
        .as_deref()
        .map(|value| matches!(value, "1" | "true" | "operator" | "operator_pose" | "all"))
        .unwrap_or(false);
    let client = crate::control::gate::FusionClientConnectionUpdate {
        edge_time_ns: state.gate.edge_time_ns(),
        client_addr: client_addr.to_string(),
        forwarded_for: extract_header_text(&headers, "x-forwarded-for"),
        user_agent: extract_header_text(&headers, "user-agent"),
        transport: transport_mode_label(transport).to_string(),
        format: if use_cbor {
            "cbor".to_string()
        } else {
            "json".to_string()
        },
        compression: compression_mode_label(compression).to_string(),
        operator_debug: include_operator_debug,
    };
    ws.on_upgrade(move |socket| {
        handle_socket(
            state,
            socket,
            TransportOptions {
                use_cbor,
                mode: transport,
                compression,
            },
            include_operator_debug,
            client,
        )
    })
}

async fn handle_socket(
    state: AppState,
    mut socket: WebSocket,
    transport: TransportOptions,
    include_operator_debug: bool,
    client: crate::control::gate::FusionClientConnectionUpdate,
) {
    let connection_id = state.fusion_stream_clients.record_connect(client);
    let mut rx_state = state.fusion_state_tx.subscribe();
    let mut rx_ack = state.chunk_ack_tx.subscribe();
    let mut fusion_encoder = TransportEncoder::new(transport, "fusion_state_packet", "1.0.0", true);
    let mut ack_encoder = TransportEncoder::new(transport, "chunk_ack_packet", "1.0.0", false);
    loop {
        tokio::select! {
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(txt))) => handle_incoming_text(&state, &txt).await,
                    Some(Ok(Message::Binary(bin))) => handle_incoming_binary(&state, &bin).await,
                    Some(Ok(Message::Ping(p))) => {
                        let _ = socket.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {},
                    Some(Err(_)) => break,
                }
            }
            pkt = rx_state.recv() => {
                match pkt {
                    Ok(packet) => {
                        let packet = if include_operator_debug {
                            with_operator_debug(&state, packet)
                        } else {
                            packet
                        };
                        let msg = match fusion_encoder.encode_packet(packet.fusion_seq, &packet) {
                            Ok(message) => message,
                            Err(error) => {
                                warn!(error = %error, "fusion_state_packet 序列化失败");
                                continue;
                            }
                        };
                        if socket.send(msg).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        metrics::counter!("fusion_state_lagged_count").increment(1);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            ack = rx_ack.recv() => {
                match ack {
                    Ok(packet) => {
                        let msg = match ack_encoder.encode_packet(packet.edge_time_ns, &packet) {
                            Ok(message) => message,
                            Err(error) => {
                                warn!(error = %error, "chunk_ack_packet 序列化失败");
                                continue;
                            }
                        };
                        if socket.send(msg).await.is_err() {
                            break;
                        }
                        metrics::counter!("chunk_ack_packet_sent_count").increment(1);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        metrics::counter!("chunk_ack_lagged_count").increment(1);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
    state
        .fusion_stream_clients
        .record_disconnect(connection_id, state.gate.edge_time_ns());
}

fn extract_header_text(headers: &HeaderMap, key: &str) -> String {
    headers
        .get(key)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_default()
        .to_string()
}

fn transport_mode_label(mode: TransportMode) -> &'static str {
    match mode {
        TransportMode::Full => "full",
        TransportMode::Delta => "delta",
    }
}

fn compression_mode_label(mode: CompressionMode) -> &'static str {
    match mode {
        CompressionMode::None => "none",
        CompressionMode::Gzip => "gzip",
    }
}

fn with_operator_debug(state: &AppState, mut packet: FusionStatePacket) -> FusionStatePacket {
    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let wifi_pose = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let operator = state.operator.snapshot(state.config.operator_hold_ms);
    let phone_pose_processing_enabled = state.config.phone_vision_processing_enabled;
    let expose_phone_pose_points = vision.fresh && phone_pose_processing_enabled;

    let vision_body_kpts_3d = if expose_phone_pose_points {
        vision.body_kpts_3d.clone()
    } else {
        Vec::new()
    };
    let vision_hand_kpts_3d = if expose_phone_pose_points {
        vision.hand_kpts_3d.clone()
    } else {
        Vec::new()
    };
    let iphone_capture = FusionSourceDebugView {
        available: vision.fresh
            && (vision.device_pose.is_some()
                || !vision_body_kpts_3d.is_empty()
                || !vision_hand_kpts_3d.is_empty()),
        fresh: vision.fresh,
        operator_track_id: vision.operator_track_id.clone().unwrap_or_default(),
        edge_time_ns: vision.last_edge_time_ns,
        recv_time_ns: vision.last_recv_time_ns,
        body_layout: vision.body_layout.as_str().to_string(),
        hand_layout: vision.hand_layout.as_str().to_string(),
        body_space: "operator_frame".to_string(),
        hand_space: "operator_frame".to_string(),
        canonical_body_layout: "coco_body_17".to_string(),
        canonical_hand_layout: "mediapipe_hand_21".to_string(),
        raw_body_count: vision_body_kpts_3d.len(),
        raw_hand_count: vision_hand_kpts_3d.len(),
        body_kpts_3d: crate::operator::canonical_body_points_3d(
            &vision_body_kpts_3d,
            vision.body_layout,
        ),
        hand_kpts_3d: crate::operator::canonical_hand_points_3d(
            &vision_hand_kpts_3d,
            vision.hand_layout,
        ),
        left_body_kpts_2d: Vec::new(),
        right_body_kpts_2d: Vec::new(),
        confidence: Some(vision.vision_conf),
        body_conf: Some(vision.body_conf),
        hand_conf: Some(vision.hand_conf),
        depth_z_mean_m: vision.depth_z_mean_m,
        execution_mode: if vision.fresh && !phone_pose_processing_enabled {
            Some("device_pose_passthrough".to_string())
        } else {
            (vision.fresh && !vision.execution_mode.is_empty())
                .then(|| vision.execution_mode.clone())
        },
        aux_snapshot_present: vision.aux_snapshot_present.then_some(true),
        aux_body_points_2d_valid: (vision.aux_body_points_2d_valid > 0)
            .then_some(vision.aux_body_points_2d_valid),
        aux_hand_points_2d_valid: (vision.aux_hand_points_2d_valid > 0)
            .then_some(vision.aux_hand_points_2d_valid),
        aux_body_points_3d_filled: (vision.aux_body_points_3d_filled > 0)
            .then_some(vision.aux_body_points_3d_filled),
        aux_hand_points_3d_filled: (vision.aux_hand_points_3d_filled > 0)
            .then_some(vision.aux_hand_points_3d_filled),
        aux_support_state: (!vision.aux_support_state.is_empty())
            .then(|| vision.aux_support_state.clone()),
        device_pose: vision.fresh.then(|| vision.device_pose.clone()).flatten(),
        imu: vision.fresh.then(|| vision.imu.clone()).flatten(),
        lifecycle_state: String::new(),
        coherence_gate_decision: String::new(),
        target_space: OPERATOR_FRAME.to_string(),
        selection_reason: String::new(),
        source_tag_left: String::new(),
        source_tag_right: String::new(),
        hand_hint_gap_m: None,
        continuity_gap_m: None,
        persons: Vec::new(),
    };

    let stereo_pair = FusionSourceDebugView {
        available: !stereo.body_kpts_3d.is_empty() || !stereo.hand_kpts_3d.is_empty(),
        fresh: stereo.fresh,
        operator_track_id: stereo.operator_track_id.clone().unwrap_or_default(),
        edge_time_ns: stereo.last_edge_time_ns,
        recv_time_ns: stereo.last_recv_time_ns,
        body_layout: stereo.body_layout.as_str().to_string(),
        hand_layout: stereo.hand_layout.as_str().to_string(),
        body_space: stereo.body_space.clone(),
        hand_space: stereo.hand_space.clone(),
        canonical_body_layout: "coco_body_17".to_string(),
        canonical_hand_layout: "mediapipe_hand_21".to_string(),
        raw_body_count: stereo.body_kpts_3d.len(),
        raw_hand_count: stereo.hand_kpts_3d.len(),
        body_kpts_3d: crate::operator::canonical_body_points_3d(
            &stereo.body_kpts_3d,
            stereo.body_layout,
        ),
        hand_kpts_3d: crate::operator::canonical_hand_points_3d(
            &stereo.hand_kpts_3d,
            stereo.hand_layout,
        ),
        left_body_kpts_2d: stereo.left_body_kpts_2d.clone(),
        right_body_kpts_2d: stereo.right_body_kpts_2d.clone(),
        confidence: Some(stereo.stereo_confidence),
        body_conf: None,
        hand_conf: None,
        depth_z_mean_m: None,
        execution_mode: None,
        aux_snapshot_present: None,
        aux_body_points_2d_valid: None,
        aux_hand_points_2d_valid: None,
        aux_body_points_3d_filled: None,
        aux_hand_points_3d_filled: None,
        aux_support_state: None,
        device_pose: None,
        imu: None,
        lifecycle_state: String::new(),
        coherence_gate_decision: String::new(),
        target_space: if stereo.body_space.is_empty() {
            crate::operator::STEREO_PAIR_FRAME.to_string()
        } else {
            stereo.body_space.clone()
        },
        selection_reason: stereo
            .persons
            .first()
            .map(|person| person.selection_reason.clone())
            .unwrap_or_default(),
        source_tag_left: stereo
            .persons
            .first()
            .map(|person| person.source_tag_left.clone())
            .unwrap_or_default(),
        source_tag_right: stereo
            .persons
            .first()
            .map(|person| person.source_tag_right.clone())
            .unwrap_or_default(),
        hand_hint_gap_m: stereo
            .persons
            .first()
            .and_then(|person| person.hand_hint_gap_m),
        continuity_gap_m: stereo
            .persons
            .first()
            .and_then(|person| person.continuity_gap_m),
        persons: stereo
            .persons
            .iter()
            .map(|person| FusionTrackedPersonDebugView {
                operator_track_id: person.operator_track_id.clone().unwrap_or_default(),
                confidence: person.stereo_confidence,
                body_layout: stereo.body_layout.as_str().to_string(),
                hand_layout: stereo.hand_layout.as_str().to_string(),
                body_space: if stereo.body_space.is_empty() {
                    crate::operator::STEREO_PAIR_FRAME.to_string()
                } else {
                    stereo.body_space.clone()
                },
                hand_space: if stereo.hand_space.is_empty() {
                    if stereo.body_space.is_empty() {
                        crate::operator::STEREO_PAIR_FRAME.to_string()
                    } else {
                        stereo.body_space.clone()
                    }
                } else {
                    stereo.hand_space.clone()
                },
                body_kpts_3d: crate::operator::canonical_body_points_3d(
                    &person.body_kpts_3d,
                    stereo.body_layout,
                ),
                hand_kpts_3d: crate::operator::canonical_hand_points_3d(
                    &person.hand_kpts_3d,
                    stereo.hand_layout,
                ),
                left_body_kpts_2d: person.left_body_kpts_2d.clone(),
                right_body_kpts_2d: person.right_body_kpts_2d.clone(),
                selection_reason: person.selection_reason.clone(),
                source_tag_left: person.source_tag_left.clone(),
                source_tag_right: person.source_tag_right.clone(),
                hand_hint_gap_m: person.hand_hint_gap_m,
                continuity_gap_m: person.continuity_gap_m,
            })
            .collect(),
    };

    let wifi_canonical = crate::operator::stabilize_wifi_canonical_body_points(
        &crate::operator::canonical_body_points_3d(&wifi_pose.body_kpts_3d, wifi_pose.body_layout),
    );
    let wifi_body_debug_points = wifi_canonical;
    let wifi_body_debug_space = normalized_wifi_body_space(&wifi_pose.body_space).to_string();

    let wifi_view = FusionSourceDebugView {
        available: !wifi_pose.body_kpts_3d.is_empty(),
        fresh: wifi_pose.fresh,
        operator_track_id: wifi_pose.operator_track_id.clone().unwrap_or_default(),
        edge_time_ns: wifi_pose.last_edge_time_ns,
        recv_time_ns: wifi_pose.last_recv_time_ns,
        body_layout: wifi_pose.body_layout.as_str().to_string(),
        hand_layout: "unknown".to_string(),
        body_space: wifi_body_debug_space.clone(),
        hand_space: "unknown".to_string(),
        canonical_body_layout: "coco_body_17".to_string(),
        canonical_hand_layout: "mediapipe_hand_21".to_string(),
        raw_body_count: wifi_pose.body_kpts_3d.len(),
        raw_hand_count: 0,
        body_kpts_3d: wifi_body_debug_points,
        hand_kpts_3d: Vec::new(),
        left_body_kpts_2d: Vec::new(),
        right_body_kpts_2d: Vec::new(),
        confidence: Some(wifi_pose.body_confidence),
        body_conf: Some(wifi_pose.body_confidence),
        hand_conf: None,
        depth_z_mean_m: None,
        execution_mode: None,
        aux_snapshot_present: None,
        aux_body_points_2d_valid: None,
        aux_hand_points_2d_valid: None,
        aux_body_points_3d_filled: None,
        aux_hand_points_3d_filled: None,
        aux_support_state: None,
        device_pose: None,
        imu: None,
        lifecycle_state: wifi_pose.diagnostics.lifecycle_state.clone(),
        coherence_gate_decision: wifi_pose.diagnostics.coherence_gate_decision.clone(),
        target_space: if wifi_pose.diagnostics.target_space.is_empty() {
            normalized_wifi_body_space(&wifi_pose.body_space).to_string()
        } else {
            wifi_pose.diagnostics.target_space.clone()
        },
        selection_reason: String::new(),
        source_tag_left: String::new(),
        source_tag_right: String::new(),
        hand_hint_gap_m: None,
        continuity_gap_m: None,
        persons: Vec::new(),
    };

    let fused_pose = FusionFusedDebugView {
        available: !operator.estimate.operator_state.body_kpts_3d.is_empty()
            || !operator.estimate.operator_state.hand_kpts_3d.is_empty(),
        fresh: operator.fresh,
        selected_source: operator.estimate.source.as_str().to_string(),
        body_source: operator
            .estimate
            .fusion_breakdown
            .body_source
            .as_str()
            .to_string(),
        hand_source: operator
            .estimate
            .fusion_breakdown
            .hand_source
            .as_str()
            .to_string(),
        body_space: fused_body_space(
            operator.estimate.fusion_breakdown.body_source,
            &stereo.body_space,
            &wifi_body_debug_space,
        )
        .to_string(),
        hand_space: fused_hand_space(&operator.estimate.source, &stereo.hand_space).to_string(),
        raw_source_edge_time_ns: operator.estimate.raw_pose.source_edge_time_ns,
        raw_body_layout: operator.estimate.raw_pose.body_layout.as_str().to_string(),
        raw_hand_layout: operator.estimate.raw_pose.hand_layout.as_str().to_string(),
        canonical_body_layout: "coco_body_17".to_string(),
        canonical_hand_layout: "mediapipe_hand_21".to_string(),
        raw_body_count: operator.estimate.raw_pose.body_kpts_3d.len(),
        raw_hand_count: operator.estimate.raw_pose.hand_kpts_3d.len(),
        stereo_body_joint_count: operator.estimate.fusion_breakdown.stereo_body_joint_count,
        vision_body_joint_count: operator.estimate.fusion_breakdown.vision_body_joint_count,
        wifi_body_joint_count: operator.estimate.fusion_breakdown.wifi_body_joint_count,
        blended_body_joint_count: operator.estimate.fusion_breakdown.blended_body_joint_count,
        stereo_hand_point_count: operator.estimate.fusion_breakdown.stereo_hand_point_count,
        vision_hand_point_count: operator.estimate.fusion_breakdown.vision_hand_point_count,
        wifi_hand_point_count: operator.estimate.fusion_breakdown.wifi_hand_point_count,
        blended_hand_point_count: operator.estimate.fusion_breakdown.blended_hand_point_count,
        body_kpts_3d: operator.estimate.operator_state.body_kpts_3d.clone(),
        hand_kpts_3d: operator.estimate.operator_state.hand_kpts_3d.clone(),
    };

    let association = FusionAssociationDebugView {
        selected_operator_track_id: operator
            .estimate
            .association
            .selected_operator_track_id
            .clone()
            .unwrap_or_default(),
        anchor_source: operator.estimate.association.anchor_source.to_string(),
        stereo_operator_track_id: operator
            .estimate
            .association
            .stereo_operator_track_id
            .clone()
            .unwrap_or_default(),
        wifi_operator_track_id: operator
            .estimate
            .association
            .wifi_operator_track_id
            .clone()
            .unwrap_or_default(),
        iphone_operator_track_id: operator
            .estimate
            .association
            .iphone_operator_track_id
            .clone()
            .unwrap_or_default(),
        wifi_anchor_eligible: operator.estimate.association.wifi_anchor_eligible,
        wifi_lifecycle_state: operator.estimate.association.wifi_lifecycle_state.clone(),
        wifi_coherence_gate_decision: operator
            .estimate
            .association
            .wifi_coherence_gate_decision
            .clone(),
        iphone_visible_hand_count: operator.estimate.association.iphone_visible_hand_count,
        hand_match_count: operator.estimate.association.hand_match_count,
        hand_match_score: operator.estimate.association.hand_match_score,
        left_wrist_gap_m: operator.estimate.association.left_wrist_gap_m,
        right_wrist_gap_m: operator.estimate.association.right_wrist_gap_m,
    };
    let motion_state = FusionMotionStateDebugView {
        root_pos_m: operator.estimate.motion_state.root_pos_m,
        root_vel_mps: operator.estimate.motion_state.root_vel_mps,
        root_std_m: operator.estimate.motion_state.root_std_m,
        heading_yaw_rad: operator.estimate.motion_state.heading_yaw_rad,
        heading_rate_radps: operator.estimate.motion_state.heading_rate_radps,
        heading_std_rad: operator.estimate.motion_state.heading_std_rad,
        motion_phase: operator.estimate.motion_state.motion_phase,
        body_presence_conf: operator.estimate.motion_state.body_presence_conf,
        csi_prior_reliability: operator.estimate.motion_state.csi_prior_reliability,
        wearer_confidence: operator.estimate.motion_state.wearer_confidence,
        stereo_track_id: operator
            .estimate
            .motion_state
            .stereo_track_id
            .clone()
            .unwrap_or_default(),
        last_good_stereo_time_ns: operator.estimate.motion_state.last_good_stereo_time_ns,
        last_good_csi_time_ns: operator.estimate.motion_state.last_good_csi_time_ns,
        stereo_measurement_used: operator.estimate.motion_state.stereo_measurement_used,
        csi_measurement_used: operator.estimate.motion_state.csi_measurement_used,
        accepted_stereo_observations: operator.estimate.motion_state.accepted_stereo_observations,
        accepted_csi_observations: operator.estimate.motion_state.accepted_csi_observations,
        rejected_stereo_observations: operator.estimate.motion_state.rejected_stereo_observations,
        rejected_csi_observations: operator.estimate.motion_state.rejected_csi_observations,
        smoother_mode: operator
            .estimate
            .motion_state
            .smoother_mode
            .as_str()
            .to_string(),
        updated_edge_time_ns: operator.estimate.motion_state.updated_edge_time_ns,
    };

    packet.operator_debug = Some(FusionOperatorDebug {
        iphone_capture,
        stereo_pair,
        wifi_pose: wifi_view,
        fused_pose,
        association,
        motion_state,
    });
    packet
}

fn fused_body_space<'a>(
    body_source: OperatorPartSource,
    stereo_body_space: &'a str,
    wifi_body_space: &'a str,
) -> &'a str {
    let wifi_body_space = if wifi_body_space.is_empty() {
        CANONICAL_BODY_FRAME
    } else {
        wifi_body_space
    };
    match body_source {
        OperatorPartSource::Stereo
        | OperatorPartSource::FusedStereoVision3d
        | OperatorPartSource::FusedStereoVision2dProjected => stereo_body_space,
        OperatorPartSource::WifiPose3d => wifi_body_space,
        OperatorPartSource::FusedMultiSource3d => {
            if stereo_body_space == wifi_body_space
                || (stereo_body_space != CANONICAL_BODY_FRAME
                    && wifi_body_space == CANONICAL_BODY_FRAME)
            {
                stereo_body_space
            } else {
                "mixed_frame"
            }
        }
        OperatorPartSource::Vision3d | OperatorPartSource::Vision2dProjected => OPERATOR_FRAME,
        OperatorPartSource::None => {
            if !stereo_body_space.is_empty() {
                stereo_body_space
            } else {
                wifi_body_space
            }
        }
    }
}

fn fused_hand_space<'a>(
    source: &crate::operator::OperatorSource,
    stereo_hand_space: &'a str,
) -> &'a str {
    match source {
        crate::operator::OperatorSource::Stereo
        | crate::operator::OperatorSource::FusedStereoVision3d
        | crate::operator::OperatorSource::FusedStereoVision2dProjected => stereo_hand_space,
        _ => "operator_frame",
    }
}

fn normalized_wifi_body_space(body_space: &str) -> &str {
    if body_space.is_empty() || body_space == OPERATOR_FRAME {
        CANONICAL_BODY_FRAME
    } else {
        body_space
    }
}

fn is_ios_phone_capture_packet(v: &serde_json::Value) -> bool {
    if v.get("platform")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("ios"))
    {
        return true;
    }
    if v.get("camera")
        .and_then(|value| value.get("mode"))
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.starts_with("teleop_phone_"))
    {
        return true;
    }
    v.get("device_pose")
        .and_then(|value| value.get("source"))
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.starts_with("ios_") || value.contains("arkit"))
}

fn sanitize_phone_capture_packet_for_passthrough(v: &mut serde_json::Value) {
    let Some(obj) = v.as_object_mut() else {
        return;
    };
    obj.insert("body_kpts_2d".to_string(), serde_json::json!([]));
    obj.insert("hand_kpts_2d".to_string(), serde_json::json!([]));
    obj.insert("body_kpts_3d".to_string(), serde_json::json!([]));
    obj.insert("hand_kpts_3d".to_string(), serde_json::json!([]));

    let confidence = obj
        .entry("confidence".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(confidence_obj) = confidence.as_object_mut() {
        confidence_obj.insert("body".to_string(), serde_json::json!(0.0));
        confidence_obj.insert("hand".to_string(), serde_json::json!(0.0));
    }

    let capture_profile = obj
        .entry("capture_profile".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(profile_obj) = capture_profile.as_object_mut() {
        profile_obj.insert("body_3d_source".to_string(), serde_json::json!("none"));
        profile_obj.insert("hand_3d_source".to_string(), serde_json::json!("none"));
        profile_obj.insert(
            "execution_mode".to_string(),
            serde_json::json!("device_pose_passthrough"),
        );
        profile_obj.insert(
            "vision_processing_enabled".to_string(),
            serde_json::json!(false),
        );
    }
}

async fn handle_incoming_text(state: &AppState, txt: &str) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(txt) else {
        metrics::counter!("fusion_ws_invalid_json_count").increment(1);
        return;
    };
    let recv_time_ns = state.gate.edge_time_ns();
    handle_incoming_value(state, v, recv_time_ns).await;
}

async fn handle_incoming_binary(state: &AppState, bin: &[u8]) {
    let Ok(v) = serde_cbor::from_slice::<serde_json::Value>(bin) else {
        metrics::counter!("fusion_ws_invalid_cbor_count").increment(1);
        return;
    };
    let recv_time_ns = state.gate.edge_time_ns();
    handle_incoming_value(state, v, recv_time_ns).await;
}

async fn handle_incoming_value(state: &AppState, mut v: serde_json::Value, recv_time_ns: u64) {
    let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
        "control_keepalive_packet" => match serde_json::from_value::<ControlKeepalivePacket>(v) {
            Ok(pkt) => {
                state.gate.ingest_keepalive(
                    &pkt.trip_id,
                    &pkt.session_id,
                    &pkt.device_id,
                    pkt.source_time_ns,
                    pkt.seq,
                    pkt.deadman_pressed,
                );
            }
            Err(e) => {
                debug!(error=%e, "control_keepalive_packet 无效");
                metrics::counter!("deadman_keepalive_invalid_count").increment(1);
            }
        },
        "capture_pose_packet" => {
            metrics::counter!("capture_pose_packet_count").increment(1);
            let trip_id = v
                .get("trip_id")
                .and_then(|x| x.as_str())
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .map(ToOwned::to_owned);
            let session_id = v
                .get("session_id")
                .and_then(|x| x.as_str())
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .map(ToOwned::to_owned);
            if let (Some(trip_id), Some(session_id)) = (trip_id, session_id) {
                state.session.set_active(trip_id, session_id);
            }
            let device_id = v
                .get("device_id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let source_time_ns = v.get("source_time_ns").and_then(|x| x.as_u64());
            let edge_time_ns = match source_time_ns {
                Some(st) if !device_id.trim().is_empty() => {
                    state
                        .gate
                        .map_source_time_to_edge(device_id.as_str(), st, recv_time_ns)
                        .0
                }
                _ => recv_time_ns,
            };

            if let Some(obj) = v.as_object_mut() {
                obj.insert("recv_time_ns".to_string(), serde_json::json!(recv_time_ns));
                obj.insert("edge_time_ns".to_string(), serde_json::json!(edge_time_ns));
            }
            if !state.config.phone_vision_processing_enabled && is_ios_phone_capture_packet(&v) {
                sanitize_phone_capture_packet_for_passthrough(&mut v);
            }
            let accepted_capture_pose_packet =
                state
                    .vision
                    .ingest_capture_pose_json(&v, edge_time_ns, recv_time_ns);
            if accepted_capture_pose_packet {
                state
                    .recorder
                    .record_capture_pose(&state.protocol, &state.config, &v)
                    .await;
            } else {
                metrics::counter!("capture_pose_packet_ignored_count").increment(1);
            }
        }
        "label_event_packet" => {
            metrics::counter!("label_event_packet_count").increment(1);
            let trip_id = v
                .get("trip_id")
                .and_then(|x| x.as_str())
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .map(ToOwned::to_owned);
            let session_id = v
                .get("session_id")
                .and_then(|x| x.as_str())
                .map(str::trim)
                .filter(|x| !x.is_empty())
                .map(ToOwned::to_owned);
            if let (Some(trip_id), Some(session_id)) = (trip_id, session_id) {
                state.session.set_active(trip_id, session_id);
            }
            let device_id = v
                .get("device_id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let source_time_ns = v.get("source_time_ns").and_then(|x| x.as_u64());
            let edge_time_ns = match source_time_ns {
                Some(st) if !device_id.trim().is_empty() => {
                    state
                        .gate
                        .map_source_time_to_edge(device_id.as_str(), st, recv_time_ns)
                        .0
                }
                _ => recv_time_ns,
            };

            if let Some(obj) = v.as_object_mut() {
                obj.insert("recv_time_ns".to_string(), serde_json::json!(recv_time_ns));
                obj.insert("edge_time_ns".to_string(), serde_json::json!(edge_time_ns));
            }
            state
                .recorder
                .record_label_event(&state.protocol, &state.config, &v)
                .await;
        }
        _ => {
            metrics::counter!("fusion_ws_unknown_packet_count").increment(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_phone_capture_packet_for_passthrough_clears_pose_points() {
        let mut packet = serde_json::json!({
            "platform": "ios",
            "camera": { "mode": "teleop_phone_back_ultrawide" },
            "device_pose": { "source": "ios_arkit_world_transform" },
            "body_kpts_2d": [[1.0, 2.0]],
            "hand_kpts_2d": [[3.0, 4.0]],
            "body_kpts_3d": [[5.0, 6.0, 7.0]],
            "hand_kpts_3d": [[8.0, 9.0, 10.0]],
            "confidence": { "body": 0.9, "hand": 0.8 },
            "capture_profile": {
                "body_3d_source": "edge_depth_reprojected",
                "hand_3d_source": "edge_depth_reprojected",
                "execution_mode": "edge_authoritative_phone_vision"
            }
        });

        sanitize_phone_capture_packet_for_passthrough(&mut packet);

        assert_eq!(packet.get("body_kpts_2d"), Some(&serde_json::json!([])));
        assert_eq!(packet.get("hand_kpts_2d"), Some(&serde_json::json!([])));
        assert_eq!(packet.get("body_kpts_3d"), Some(&serde_json::json!([])));
        assert_eq!(packet.get("hand_kpts_3d"), Some(&serde_json::json!([])));
        assert_eq!(
            packet.pointer("/capture_profile/execution_mode"),
            Some(&serde_json::json!("device_pose_passthrough"))
        );
        assert_eq!(
            packet.pointer("/capture_profile/vision_processing_enabled"),
            Some(&serde_json::json!(false))
        );
        assert_eq!(
            packet.pointer("/capture_profile/body_3d_source"),
            Some(&serde_json::json!("none"))
        );
        assert_eq!(
            packet.pointer("/capture_profile/hand_3d_source"),
            Some(&serde_json::json!("none"))
        );
        assert_eq!(
            packet.pointer("/confidence/body"),
            Some(&serde_json::json!(0.0))
        );
        assert_eq!(
            packet.pointer("/confidence/hand"),
            Some(&serde_json::json!(0.0))
        );
    }

    #[test]
    fn is_ios_phone_capture_packet_matches_ios_markers() {
        assert!(is_ios_phone_capture_packet(&serde_json::json!({
            "platform": "ios"
        })));
        assert!(is_ios_phone_capture_packet(&serde_json::json!({
            "camera": { "mode": "teleop_phone_back_ultrawide" }
        })));
        assert!(is_ios_phone_capture_packet(&serde_json::json!({
            "device_pose": { "source": "ios_arkit_world_transform" }
        })));
        assert!(!is_ios_phone_capture_packet(&serde_json::json!({
            "platform": "simulator",
            "camera": { "mode": "debug_camera" }
        })));
    }
}
