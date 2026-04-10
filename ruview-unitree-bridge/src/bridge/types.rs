use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct Pose {
    pub pos: [f32; 3],
    pub quat: [f32; 4],
}

#[derive(Clone, Debug, Deserialize)]
pub struct TeleopQuality {
    pub source_mode: String,
    pub fused_conf: f32,
    pub vision_conf: Option<f32>,
    pub csi_conf: Option<f32>,
}

/// `teleop_frame_v1`（最小版，按 PRD 15.2 可选字段裁剪）。
///
/// 注意：此处只包含 Unitree bridge 关心的字段；未列出的字段由上游/其他 bridge 处理。
#[derive(Clone, Debug, Deserialize)]
pub struct TeleopFrameV1 {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub robot_type: String,
    pub end_effector_type: String,
    pub edge_time_ns: u64,
    pub operator_frame: String,
    pub robot_base_frame: String,
    pub extrinsic_version: String,
    pub control_state: String,
    pub teleop_enabled: Option<bool>,
    pub body_control_enabled: Option<bool>,
    pub hand_control_enabled: Option<bool>,
    pub left_wrist_pose: Pose,
    pub right_wrist_pose: Pose,
    pub quality: TeleopQuality,
    pub safety_state: String,

    pub arm_q_target: Option<Vec<f32>>,
    pub arm_tauff_target: Option<Vec<f32>>,

    pub left_hand_joints: Option<Vec<f32>>,
    pub right_hand_joints: Option<Vec<f32>>,
    pub left_hand_target: Option<Vec<f32>>,
    pub right_hand_target: Option<Vec<f32>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_control_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand_control_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_q_commanded: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_tau_commanded: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_hand_q_commanded: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_hand_q_commanded: Option<Vec<f32>>,
}
