use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::http::HeaderMap;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::ws::transport::{CompressionMode, TransportEncoder, TransportMode, TransportOptions};
use crate::ws::types::BridgeStatePacket;
use crate::{AppState, RobotStateRecordRequest};

#[derive(Deserialize)]
struct WsAuthQuery {
    token: Option<String>,
    format: Option<String>,
    transport: Option<String>,
    compression: Option<String>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/stream/teleop", get(ws_upgrade))
        .with_state(state)
}

async fn ws_upgrade(
    State(state): State<AppState>,
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
    ws.on_upgrade(move |socket| {
        handle_socket(
            state,
            socket,
            TransportOptions {
                use_cbor,
                mode: transport,
                compression,
            },
        )
    })
}

async fn handle_socket(state: AppState, socket: WebSocket, transport: TransportOptions) {
    let mut encoder = TransportEncoder::new(transport, "teleop_frame_v1", "teleop_frame_v1", true);
    let mut rx = state.teleop_tx.subscribe();
    let (mut write, mut read) = socket.split();
    let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<Message>();
    let writer = tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                outbound = outbound_rx.recv() => {
                    let Some(outbound) = outbound else {
                        return;
                    };
                    if write.send(outbound).await.is_err() {
                        return;
                    }
                }
                frame = rx.recv() => {
                    match frame {
                        Ok(frame) => {
                            let msg = match encoder.encode_packet(frame.edge_time_ns, &frame) {
                                Ok(message) => message,
                                Err(error) => {
                                    warn!(error = %error, "teleop_frame_v1 序列化失败");
                                    continue;
                                }
                            };
                            if write.send(msg).await.is_err() {
                                return;
                            }
                            metrics::counter!("teleop_frame_sent_count").increment(1);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                            metrics::counter!("teleop_frame_lagged_count").increment(1);
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                    }
                }
            }
        }
    });
    let mut writer = writer;
    loop {
        tokio::select! {
            biased;
            _ = &mut writer => break,
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(txt))) => handle_bridge_state(&state, &txt).await,
                    Some(Ok(Message::Binary(bin))) => handle_bridge_state_bin(&state, &bin).await,
                    Some(Ok(Message::Ping(p))) => {
                        let _ = outbound_tx.send(Message::Pong(p));
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {},
                    Some(Err(_)) => break,
                }
            }
        }
    }
    writer.abort();
}

async fn handle_bridge_state(state: &AppState, txt: &str) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(txt) else {
        return;
    };
    let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if ty != "bridge_state_packet" {
        return;
    }

    match serde_json::from_value::<BridgeStatePacket>(v) {
        Ok(pkt) => process_bridge_state_packet(state, pkt),
        Err(e) => {
            debug!(error=%e, "bridge_state_packet 无效");
            metrics::counter!("bridge_state_packet_invalid_count").increment(1);
        }
    }
}

async fn handle_bridge_state_bin(state: &AppState, bin: &[u8]) {
    let Ok(v) = serde_cbor::from_slice::<serde_json::Value>(bin) else {
        return;
    };
    let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if ty != "bridge_state_packet" {
        return;
    }

    match serde_json::from_value::<BridgeStatePacket>(v) {
        Ok(pkt) => process_bridge_state_packet(state, pkt),
        Err(e) => {
            debug!(error=%e, "bridge_state_packet 无效");
            metrics::counter!("bridge_state_packet_invalid_count").increment(1);
        }
    }
}

fn process_bridge_state_packet(state: &AppState, pkt: BridgeStatePacket) {
    state.bridge_store.update_bridge(
        &pkt.bridge_id,
        pkt.is_ready,
        &pkt.fault_code,
        &pkt.fault_message,
        pkt.last_command_edge_time_ns,
    );
    if !pkt.is_ready {
        metrics::counter!("bridge_state_packet_not_ready_count").increment(1);
    }
    let event = serde_json::json!({
        "type": "robot_state_packet",
        "schema_version": pkt.schema_version.clone(),
        "bridge_id": pkt.bridge_id.clone(),
        "trip_id": pkt.trip_id.clone(),
        "session_id": pkt.session_id.clone(),
        "robot_type": pkt.robot_type.clone(),
        "end_effector_type": pkt.end_effector_type.clone(),
        "edge_time_ns": pkt.edge_time_ns,
        "is_ready": pkt.is_ready,
        "fault_code": pkt.fault_code.clone(),
        "fault_message": pkt.fault_message.clone(),
        "last_command_edge_time_ns": pkt.last_command_edge_time_ns,
        "control_state": pkt.control_state.clone(),
        "safety_state": pkt.safety_state.clone(),
        "body_control_enabled": pkt.body_control_enabled,
        "hand_control_enabled": pkt.hand_control_enabled,
        "arm_q_commanded": pkt.arm_q_commanded.clone(),
        "arm_tau_commanded": pkt.arm_tau_commanded.clone(),
        "left_hand_q_commanded": pkt.left_hand_q_commanded.clone(),
        "right_hand_q_commanded": pkt.right_hand_q_commanded.clone(),
    });
    match state
        .robot_state_record_tx
        .try_send(RobotStateRecordRequest {
            trip_id: pkt.trip_id,
            session_id: pkt.session_id,
            payload: event,
        }) {
        Ok(()) => {}
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            metrics::counter!("bridge_state_record_dropped_count", "reason" => "channel_full")
                .increment(1);
        }
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
            metrics::counter!("bridge_state_record_dropped_count", "reason" => "channel_closed")
                .increment(1);
        }
    }
    metrics::counter!("bridge_state_packet_count").increment(1);
}
