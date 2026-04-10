use serde::{Deserialize, Serialize};

use crate::sensing::{VisionDevicePose, VisionImuSample};

#[derive(Clone, Debug, Serialize)]
pub struct ChunkAckPacket {
    #[serde(rename = "type")]
    pub ty: &'static str,
    pub schema_version: &'static str,
    pub session_id: String,
    pub trip_id: String,
    pub chunk_index: u32,
    pub status: String,
    pub edge_time_ns: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionStatePacket {
    #[serde(rename = "type")]
    pub ty: &'static str,
    pub schema_version: &'static str,
    pub trip_id: String,
    pub session_id: String,
    pub fusion_seq: u64,
    pub edge_time_ns: u64,
    pub operator_state: OperatorState,
    pub quality: FusionQuality,
    pub safety: FusionSafety,
    pub control: FusionControl,
    pub latency_ms: LatencyMs,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator_debug: Option<FusionOperatorDebug>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionOperatorDebug {
    pub iphone_capture: FusionSourceDebugView,
    pub stereo_pair: FusionSourceDebugView,
    pub wifi_pose: FusionSourceDebugView,
    pub fused_pose: FusionFusedDebugView,
    pub association: FusionAssociationDebugView,
    pub motion_state: FusionMotionStateDebugView,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionSourceDebugView {
    pub available: bool,
    pub fresh: bool,
    pub operator_track_id: String,
    pub edge_time_ns: u64,
    pub recv_time_ns: u64,
    pub body_layout: String,
    pub hand_layout: String,
    pub body_space: String,
    pub hand_space: String,
    pub canonical_body_layout: String,
    pub canonical_hand_layout: String,
    pub raw_body_count: usize,
    pub raw_hand_count: usize,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub left_body_kpts_2d: Vec<[f32; 2]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub right_body_kpts_2d: Vec<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_z_mean_m: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_snapshot_present: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_body_points_2d_valid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_hand_points_2d_valid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_body_points_3d_filled: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_hand_points_3d_filled: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aux_support_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_pose: Option<VisionDevicePose>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub imu: Option<VisionImuSample>,
    pub lifecycle_state: String,
    pub coherence_gate_decision: String,
    pub target_space: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub selection_reason: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub source_tag_left: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub source_tag_right: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand_hint_gap_m: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuity_gap_m: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub persons: Vec<FusionTrackedPersonDebugView>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionTrackedPersonDebugView {
    pub operator_track_id: String,
    pub confidence: f32,
    pub body_layout: String,
    pub hand_layout: String,
    pub body_space: String,
    pub hand_space: String,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub left_body_kpts_2d: Vec<[f32; 2]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub right_body_kpts_2d: Vec<[f32; 2]>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub selection_reason: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub source_tag_left: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub source_tag_right: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand_hint_gap_m: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuity_gap_m: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionFusedDebugView {
    pub available: bool,
    pub fresh: bool,
    pub selected_source: String,
    pub body_source: String,
    pub hand_source: String,
    pub body_space: String,
    pub hand_space: String,
    pub raw_source_edge_time_ns: u64,
    pub raw_body_layout: String,
    pub raw_hand_layout: String,
    pub canonical_body_layout: String,
    pub canonical_hand_layout: String,
    pub raw_body_count: usize,
    pub raw_hand_count: usize,
    pub stereo_body_joint_count: usize,
    pub vision_body_joint_count: usize,
    pub wifi_body_joint_count: usize,
    pub blended_body_joint_count: usize,
    pub stereo_hand_point_count: usize,
    pub vision_hand_point_count: usize,
    pub wifi_hand_point_count: usize,
    pub blended_hand_point_count: usize,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionAssociationDebugView {
    pub selected_operator_track_id: String,
    pub anchor_source: String,
    pub stereo_operator_track_id: String,
    pub wifi_operator_track_id: String,
    pub iphone_operator_track_id: String,
    pub wifi_anchor_eligible: bool,
    pub wifi_lifecycle_state: String,
    pub wifi_coherence_gate_decision: String,
    pub iphone_visible_hand_count: usize,
    pub hand_match_count: usize,
    pub hand_match_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_wrist_gap_m: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_wrist_gap_m: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionMotionStateDebugView {
    pub root_pos_m: [f32; 3],
    pub root_vel_mps: [f32; 3],
    pub root_std_m: f32,
    pub heading_yaw_rad: f32,
    pub heading_rate_radps: f32,
    pub heading_std_rad: f32,
    pub motion_phase: f32,
    pub body_presence_conf: f32,
    pub csi_prior_reliability: f32,
    pub wearer_confidence: f32,
    pub stereo_track_id: String,
    pub last_good_stereo_time_ns: u64,
    pub last_good_csi_time_ns: u64,
    pub stereo_measurement_used: bool,
    pub csi_measurement_used: bool,
    pub accepted_stereo_observations: u32,
    pub accepted_csi_observations: u32,
    pub rejected_stereo_observations: u32,
    pub rejected_csi_observations: u32,
    pub smoother_mode: String,
    pub updated_edge_time_ns: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HumanDemoPosePacket {
    #[serde(rename = "type")]
    pub ty: &'static str,
    pub schema_version: &'static str,
    pub trip_id: String,
    pub session_id: String,
    pub fusion_seq: u64,
    pub edge_time_ns: u64,
    pub selected_source: String,
    pub raw_pose: HumanDemoRawPose,
    pub canonical_pose: HumanDemoCanonicalPose,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_debug: Option<HumanDemoFusionDebug>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HumanDemoRawPose {
    pub source_edge_time_ns: u64,
    pub body_layout: String,
    pub hand_layout: String,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HumanDemoCanonicalPose {
    pub body_layout: String,
    pub hand_layout: String,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    pub end_effector_pose: EndEffectorPose,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_hand_joints: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_hand_joints: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_hand_curls: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_hand_curls: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct HumanDemoFusionDebug {
    pub body_source: String,
    pub hand_source: String,
    pub stereo_body_joint_count: usize,
    pub vision_body_joint_count: usize,
    pub wifi_body_joint_count: usize,
    pub blended_body_joint_count: usize,
    pub stereo_hand_point_count: usize,
    pub vision_hand_point_count: usize,
    pub wifi_hand_point_count: usize,
    pub blended_hand_point_count: usize,
    pub motion_root_pos_m: [f32; 3],
    pub motion_root_vel_mps: [f32; 3],
    pub motion_heading_yaw_rad: f32,
    pub motion_heading_rate_radps: f32,
    pub motion_phase: f32,
    pub motion_body_presence_conf: f32,
    pub motion_csi_prior_reliability: f32,
    pub motion_wearer_confidence: f32,
    pub motion_stereo_track_id: String,
    pub motion_last_good_stereo_time_ns: u64,
    pub motion_last_good_csi_time_ns: u64,
    pub motion_stereo_measurement_used: bool,
    pub motion_csi_measurement_used: bool,
    pub motion_smoother_mode: String,
    pub motion_updated_edge_time_ns: u64,
}

impl HumanDemoFusionDebug {
    pub fn into_option(self) -> Option<Self> {
        if self.body_source.is_empty()
            && self.hand_source.is_empty()
            && self.stereo_body_joint_count == 0
            && self.vision_body_joint_count == 0
            && self.wifi_body_joint_count == 0
            && self.blended_body_joint_count == 0
            && self.stereo_hand_point_count == 0
            && self.vision_hand_point_count == 0
            && self.wifi_hand_point_count == 0
            && self.blended_hand_point_count == 0
            && self.motion_updated_edge_time_ns == 0
        {
            None
        } else {
            Some(self)
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OperatorState {
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    pub end_effector_pose: EndEffectorPose,
}

#[derive(Clone, Debug, Serialize)]
pub struct EndEffectorPose {
    pub left: Pose,
    pub right: Pose,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionQuality {
    pub source_mode: String,
    pub vision_conf: f32,
    pub csi_conf: f32,
    pub fused_conf: f32,
    pub coherence: f32,
    pub gate_state: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionSafety {
    pub state: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionControl {
    pub state: String,
    pub reason: String,
    pub bridge_ready: bool,
    pub deadman: DeadmanState,
}

#[derive(Clone, Debug, Serialize)]
pub struct DeadmanState {
    pub enabled: bool,
    pub timeout_ms: u64,
    pub link_ok: bool,
    pub pressed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LatencyMs {
    pub iphone_to_edge: f32,
    pub csi_to_edge: f32,
    pub fusion_compute: f32,
    pub edge_to_robot: f32,
    pub e2e: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct TeleopFrameV1 {
    pub schema_version: &'static str,
    pub trip_id: String,
    pub session_id: String,
    pub robot_type: String,
    pub end_effector_type: String,
    pub edge_time_ns: u64,
    pub operator_frame: String,
    pub robot_base_frame: String,
    pub extrinsic_version: String,
    pub control_state: String,
    pub teleop_enabled: bool,
    pub body_control_enabled: bool,
    pub hand_control_enabled: bool,
    pub left_wrist_pose: Pose,
    pub right_wrist_pose: Pose,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand_joint_layout: Option<TeleopHandJointLayout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand_target_layout: Option<TeleopHandTargetLayout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_hand_joints: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_hand_joints: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_hand_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_hand_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waist_joint_layout: Option<TeleopWaistJointLayout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leg_joint_layout: Option<TeleopLegJointLayout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waist_q_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leg_q_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_q_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_tauff_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_debug: Option<TeleopTargetDebug>,
    pub quality: TeleopQuality,
    pub safety_state: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetargetReferenceV1 {
    pub schema_version: &'static str,
    pub trip_id: String,
    pub session_id: String,
    pub source_session_id: String,
    pub target_person_id: String,
    pub source_kind: &'static str,
    pub source_edge_time_ns: u64,
    pub robot_type: String,
    pub end_effector_type: String,
    pub edge_time_ns: u64,
    pub control_state: String,
    pub body_control_enabled: bool,
    pub hand_control_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hand_target_layout: Option<TeleopHandTargetLayout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waist_joint_layout: Option<TeleopWaistJointLayout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leg_joint_layout: Option<TeleopLegJointLayout>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waist_q_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leg_q_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_q_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_hand_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_hand_target: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_debug: Option<TeleopTargetDebug>,
    pub quality: TeleopQuality,
    pub retarget_status: &'static str,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TeleopTargetDebug {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_hand_target_source: Option<TeleopHandTargetSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_hand_target_source: Option<TeleopHandTargetSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_q_target_source: Option<TeleopArmTargetSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower_body_q_target_source: Option<TeleopLowerBodyTargetSource>,
}

impl TeleopTargetDebug {
    pub fn into_option(self) -> Option<Self> {
        if self.left_hand_target_source.is_none()
            && self.right_hand_target_source.is_none()
            && self.arm_q_target_source.is_none()
            && self.lower_body_q_target_source.is_none()
        {
            None
        } else {
            Some(self)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TeleopHandTargetSource {
    AnatomicalJoints,
    Curl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TeleopHandJointLayout {
    #[serde(rename = "anatomical_joint_16")]
    AnatomicalJoint16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TeleopHandTargetLayout {
    #[serde(rename = "anatomical_target_16")]
    AnatomicalTarget16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TeleopWaistJointLayout {
    #[serde(rename = "unitree_g1_waist_1")]
    UnitreeG1Waist1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TeleopLegJointLayout {
    #[serde(rename = "unitree_g1_leg_6x2")]
    UnitreeG1Leg6x2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TeleopArmTargetSource {
    BodyAnchor,
    WristPoseIk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TeleopLowerBodyTargetSource {
    StandingBalance,
    BodyAnchor,
}

#[derive(Clone, Debug, Serialize)]
pub struct Pose {
    pub pos: [f32; 3],
    pub quat: [f32; 4],
}

#[derive(Clone, Debug, Serialize)]
pub struct TeleopQuality {
    pub source_mode: String,
    pub fused_conf: f32,
    pub vision_conf: f32,
    pub csi_conf: f32,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct BridgeStatePacket {
    #[serde(rename = "type")]
    pub ty: String,
    pub schema_version: String,
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
    #[serde(default)]
    pub control_state: Option<String>,
    #[serde(default)]
    pub safety_state: Option<String>,
    #[serde(default)]
    pub body_control_enabled: Option<bool>,
    #[serde(default)]
    pub hand_control_enabled: Option<bool>,
    #[serde(default)]
    pub arm_q_commanded: Option<Vec<f32>>,
    #[serde(default)]
    pub arm_tau_commanded: Option<Vec<f32>>,
    #[serde(default)]
    pub left_hand_q_commanded: Option<Vec<f32>>,
    #[serde(default)]
    pub right_hand_q_commanded: Option<Vec<f32>>,
}
