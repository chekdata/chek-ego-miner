use serde::Serialize;

use crate::bridge::types::BridgeStatePacket;

#[derive(Clone, Debug)]
pub struct BridgeReadiness {
    pub is_ready: bool,
    pub fault_code: String,
    pub fault_message: String,
}

impl BridgeReadiness {
    pub fn ok() -> Self {
        Self {
            is_ready: true,
            fault_code: "".to_string(),
            fault_message: "".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BridgeHealthSnapshot {
    pub bridge_id: String,
    pub is_ready: bool,
    pub fault_code: String,
    pub fault_message: String,
    pub last_command_edge_time_ns: u64,
}

pub struct BridgeStateInput<'a> {
    pub bridge_id: &'a str,
    pub trip_id: &'a str,
    pub session_id: &'a str,
    pub robot_type: &'a str,
    pub end_effector_type: &'a str,
    pub edge_time_ns: u64,
    pub last_command_edge_time_ns: u64,
    pub control_state: Option<&'a str>,
    pub safety_state: Option<&'a str>,
    pub body_control_enabled: Option<bool>,
    pub hand_control_enabled: Option<bool>,
    pub arm_q_commanded: Option<&'a [f32]>,
    pub arm_tau_commanded: Option<&'a [f32]>,
    pub left_hand_q_commanded: Option<&'a [f32]>,
    pub right_hand_q_commanded: Option<&'a [f32]>,
}

pub fn build_bridge_state_packet(
    input: BridgeStateInput<'_>,
    readiness: &BridgeReadiness,
) -> BridgeStatePacket {
    BridgeStatePacket {
        ty: "bridge_state_packet",
        schema_version: "1.0.0",
        bridge_id: input.bridge_id.to_string(),
        trip_id: input.trip_id.to_string(),
        session_id: input.session_id.to_string(),
        robot_type: input.robot_type.to_string(),
        end_effector_type: input.end_effector_type.to_string(),
        edge_time_ns: input.edge_time_ns,
        is_ready: readiness.is_ready,
        fault_code: readiness.fault_code.clone(),
        fault_message: readiness.fault_message.clone(),
        last_command_edge_time_ns: input.last_command_edge_time_ns,
        control_state: input.control_state.map(str::to_string),
        safety_state: input.safety_state.map(str::to_string),
        body_control_enabled: input.body_control_enabled,
        hand_control_enabled: input.hand_control_enabled,
        arm_q_commanded: input.arm_q_commanded.map(|v| v.to_vec()),
        arm_tau_commanded: input.arm_tau_commanded.map(|v| v.to_vec()),
        left_hand_q_commanded: input.left_hand_q_commanded.map(|v| v.to_vec()),
        right_hand_q_commanded: input.right_hand_q_commanded.map(|v| v.to_vec()),
    }
}
