use serde::{Deserialize, Serialize};

/// `teleop_frame_v1`（最小版，按 PRD 15.2/15.4 可选字段裁剪）。
///
/// 注意：此处只包含 LEAP bridge 关心的字段；未列出的字段由上游/其他 bridge 处理。
#[derive(Clone, Debug, Deserialize)]
pub struct TeleopFrameV1 {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub robot_type: String,
    pub end_effector_type: String,
    pub edge_time_ns: u64,
    pub control_state: String,
    pub safety_state: String,
    pub teleop_enabled: Option<bool>,
    pub body_control_enabled: Option<bool>,
    pub hand_control_enabled: Option<bool>,

    pub hand_joint_layout: Option<String>,
    pub hand_target_layout: Option<String>,
    pub left_hand_joints: Option<Vec<f32>>,
    pub right_hand_joints: Option<Vec<f32>>,
    pub left_hand_target: Option<Vec<f32>>,
    pub right_hand_target: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandSide {
    Left,
    Right,
}

/// 单手目标帧（用于配对与 stale 策略）。
#[derive(Clone, Debug)]
pub struct HandTargetFrame {
    pub side: HandSide,
    pub edge_time_ns: u64,
    pub target: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairingDegrade {
    Normal,
    Hold,
    Freeze,
}

/// 双手配对后的输出（可能包含退化状态）。
#[derive(Clone, Debug)]
pub struct PairedHandFrame {
    pub left: HandTargetFrame,
    pub right: HandTargetFrame,
    pub delta_ns: u64,
    pub degrade: PairingDegrade,
}

/// LEAP 最终下发命令（左右手）。
#[derive(Clone, Debug)]
pub struct LeapCommandFrame {
    pub edge_time_ns: u64,
    pub left_cmd: Vec<f32>,
    pub right_cmd: Vec<f32>,
}

/// 回传给 edge 的 bridge 状态（PRD 15.6）。
#[derive(Clone, Debug, Serialize)]
pub struct BridgeStatePacket {
    #[serde(rename = "type")]
    pub ty: &'static str,
    pub schema_version: &'static str,
    pub bridge_id: String,
    pub trip_id: String,
    pub session_id: String,
    pub robot_type: String,
    pub end_effector_type: String,
    pub edge_time_ns: u64,
    pub is_ready: bool,
    pub fault_code: String,
    pub fault_message: String,
    pub last_command_edge_time_ns: u64,
}
