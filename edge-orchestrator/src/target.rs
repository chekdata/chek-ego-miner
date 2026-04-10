use std::collections::HashMap;
use std::f32::consts::PI;

use nalgebra::{SMatrix, SVector, Unit, UnitQuaternion, Vector3};
use tracing::{debug, warn};

use crate::ws::types::{
    Pose, TeleopArmTargetSource, TeleopHandTargetSource, TeleopLowerBodyTargetSource,
    TeleopTargetDebug,
};

pub const HAND_JOINT_TARGET_DIM: usize = 16;
pub type HandJointTarget = [f32; HAND_JOINT_TARGET_DIM];
pub const G1_WAIST_Q_TARGET_DIM: usize = 1;
pub const G1_LEG_Q_TARGET_DIM: usize = 12;

const THUMB_YAW_MAX_RAD: f32 = 0.60;
const THUMB_PITCH_MAX_RAD: f32 = 1.20;
const THUMB_MCP_MAX_RAD: f32 = 1.30;
const THUMB_IP_MAX_RAD: f32 = 1.35;
const FINGER_MCP_MAX_RAD: f32 = 1.25;
const FINGER_PIP_MAX_RAD: f32 = 1.60;
const FINGER_DIP_MAX_RAD: f32 = 1.35;
const MIN_SEGMENT_NORM: f32 = 1.0e-5;
const COCO_LEFT_SHOULDER: usize = 5;
const COCO_RIGHT_SHOULDER: usize = 6;
const COCO_LEFT_HIP: usize = 11;
const COCO_RIGHT_HIP: usize = 12;
const COCO_LEFT_KNEE: usize = 13;
const COCO_RIGHT_KNEE: usize = 14;
const COCO_LEFT_ANKLE: usize = 15;
const COCO_RIGHT_ANKLE: usize = 16;
const G1_HIP_PITCH_TARGET_LIMIT_RAD: f32 = 0.95;
const G1_HIP_ROLL_TARGET_LIMIT_RAD: f32 = 0.45;
const G1_HIP_YAW_TARGET_LIMIT_RAD: f32 = 0.45;
const G1_KNEE_TARGET_LIMIT_RAD: f32 = 1.40;
const G1_ANKLE_PITCH_TARGET_LIMIT_RAD: f32 = 0.55;
const G1_ANKLE_ROLL_TARGET_LIMIT_RAD: f32 = 0.25;
const HAND_TARGET_CARRY_NS: u64 = 3_000_000_000;

pub struct PrecomputedTargets {
    pub left_hand_target: Option<Vec<f32>>,
    pub right_hand_target: Option<Vec<f32>>,
    pub waist_q_target: Option<Vec<f32>>,
    pub leg_q_target: Option<Vec<f32>>,
    pub arm_q_target: Option<Vec<f32>>,
    pub target_debug: TeleopTargetDebug,
}

pub struct TeleopTargetPrecomputer {
    arm_ik: Option<UnitreeArmIk>,
    last_arm_q_target: Option<Vec<f32>>,
    last_arm_q_target_source: Option<TeleopArmTargetSource>,
    last_wrist_pose_pair: Option<(Pose, Pose)>,
    last_left_hand_target: Option<Vec<f32>>,
    last_right_hand_target: Option<Vec<f32>>,
    last_left_hand_target_edge_time_ns: Option<u64>,
    last_right_hand_target_edge_time_ns: Option<u64>,
    last_scope: Option<(String, String, String)>,
}

impl TeleopTargetPrecomputer {
    pub fn new() -> Self {
        let arm_ik = match UnitreeArmIk::new() {
            Ok(value) => Some(value),
            Err(error) => {
                warn!(error = %error, "初始化 arm_q_target 预计算器失败");
                None
            }
        };
        Self {
            arm_ik,
            last_arm_q_target: None,
            last_arm_q_target_source: None,
            last_wrist_pose_pair: None,
            last_left_hand_target: None,
            last_right_hand_target: None,
            last_left_hand_target_edge_time_ns: None,
            last_right_hand_target_edge_time_ns: None,
            last_scope: None,
        }
    }

    pub fn compute(
        &mut self,
        trip_id: &str,
        session_id: &str,
        robot_type: &str,
        edge_time_ns: u64,
        body_kpts_robot: &[[f32; 3]],
        left_wrist_pose: &Pose,
        right_wrist_pose: &Pose,
        left_hand_joints: Option<HandJointTarget>,
        right_hand_joints: Option<HandJointTarget>,
    ) -> PrecomputedTargets {
        self.reset_scope_if_needed(trip_id, session_id, robot_type);

        let left_hand_target = self.resolve_hand_target_with_carry(
            hand_target_from_joints(left_hand_joints),
            edge_time_ns,
            true,
        );
        let right_hand_target = self.resolve_hand_target_with_carry(
            hand_target_from_joints(right_hand_joints),
            edge_time_ns,
            false,
        );
        let (waist_q_target, leg_q_target, lower_body_q_target_source) =
            compute_lower_body_q_target(robot_type, body_kpts_robot);
        let (arm_q_target, arm_q_target_source) = self.compute_arm_q_target(
            robot_type,
            body_kpts_robot,
            left_wrist_pose,
            right_wrist_pose,
        );

        PrecomputedTargets {
            left_hand_target,
            right_hand_target,
            waist_q_target,
            leg_q_target,
            arm_q_target,
            target_debug: TeleopTargetDebug {
                left_hand_target_source: left_hand_joints
                    .filter(|values| values.iter().all(|value| value.is_finite()))
                    .map(|_| TeleopHandTargetSource::AnatomicalJoints),
                right_hand_target_source: right_hand_joints
                    .filter(|values| values.iter().all(|value| value.is_finite()))
                    .map(|_| TeleopHandTargetSource::AnatomicalJoints),
                lower_body_q_target_source,
                arm_q_target_source,
            },
        }
    }

    fn reset_scope_if_needed(&mut self, trip_id: &str, session_id: &str, robot_type: &str) {
        let next_scope = (
            trip_id.to_string(),
            session_id.to_string(),
            robot_type.to_string(),
        );
        if self.last_scope.as_ref() != Some(&next_scope) {
            self.last_scope = Some(next_scope);
            self.last_arm_q_target = None;
            self.last_arm_q_target_source = None;
            self.last_wrist_pose_pair = None;
            self.last_left_hand_target = None;
            self.last_right_hand_target = None;
            self.last_left_hand_target_edge_time_ns = None;
            self.last_right_hand_target_edge_time_ns = None;
        }
    }

    fn resolve_hand_target_with_carry(
        &mut self,
        current: Option<Vec<f32>>,
        edge_time_ns: u64,
        is_left: bool,
    ) -> Option<Vec<f32>> {
        let (last_target, last_edge_time_ns) = if is_left {
            (
                &mut self.last_left_hand_target,
                &mut self.last_left_hand_target_edge_time_ns,
            )
        } else {
            (
                &mut self.last_right_hand_target,
                &mut self.last_right_hand_target_edge_time_ns,
            )
        };

        if let Some(target) = current {
            *last_target = Some(target.clone());
            *last_edge_time_ns = Some(edge_time_ns);
            return Some(target);
        }

        let Some(cached_target) = last_target.clone() else {
            return None;
        };
        let Some(cached_edge_time_ns) = *last_edge_time_ns else {
            *last_target = None;
            return None;
        };
        if edge_time_ns.saturating_sub(cached_edge_time_ns) <= HAND_TARGET_CARRY_NS {
            return Some(cached_target);
        }
        *last_target = None;
        *last_edge_time_ns = None;
        None
    }

    fn compute_arm_q_target(
        &mut self,
        robot_type: &str,
        body_kpts_robot: &[[f32; 3]],
        left_wrist_pose: &Pose,
        right_wrist_pose: &Pose,
    ) -> (Option<Vec<f32>>, Option<TeleopArmTargetSource>) {
        let robot_type = canonical_robot_type(robot_type);
        if poses_are_uninitialized(left_wrist_pose, right_wrist_pose) {
            return (None, None);
        }

        if let (Some(last_q), Some((last_left, last_right))) = (
            self.last_arm_q_target.as_ref(),
            self.last_wrist_pose_pair.as_ref(),
        ) {
            if same_pose(last_left, left_wrist_pose) && same_pose(last_right, right_wrist_pose) {
                metrics::counter!("teleop_arm_target_cache_hit_count").increment(1);
                return (last_q.clone().into(), self.last_arm_q_target_source);
            }
        }

        let Some(arm_ik) = self.arm_ik.as_ref() else {
            metrics::counter!(
                "teleop_arm_target_precompute_failed_count",
                "reason" => crate::reason::REASON_IK_UNIMPLEMENTED
            )
            .increment(1);
            return (None, None);
        };

        if let Some(target) = arm_ik.approximate_arm_q_target_from_body(
            robot_type,
            body_kpts_robot,
            left_wrist_pose,
            right_wrist_pose,
        ) {
            metrics::counter!("teleop_arm_target_precomputed_count", "mode" => "body_anchored")
                .increment(1);
            self.last_wrist_pose_pair = Some((left_wrist_pose.clone(), right_wrist_pose.clone()));
            self.last_arm_q_target = Some(target.clone());
            self.last_arm_q_target_source = Some(TeleopArmTargetSource::BodyAnchor);
            return (Some(target), self.last_arm_q_target_source);
        }

        if !body_kpts_robot.is_empty() {
            if let Some(target) = arm_ik.approximate_arm_q_target_from_wrist_positions(
                robot_type,
                left_wrist_pose,
                right_wrist_pose,
            ) {
                metrics::counter!(
                    "teleop_arm_target_precomputed_count",
                    "mode" => "body_anchor_wrist_fallback"
                )
                .increment(1);
                self.last_wrist_pose_pair =
                    Some((left_wrist_pose.clone(), right_wrist_pose.clone()));
                self.last_arm_q_target = Some(target.clone());
                self.last_arm_q_target_source = Some(TeleopArmTargetSource::BodyAnchor);
                return (Some(target), self.last_arm_q_target_source);
            }
        }

        let (left_target, right_target) = arm_ik.retarget_wrist_targets(
            robot_type,
            body_kpts_robot,
            left_wrist_pose,
            right_wrist_pose,
        );

        match arm_ik.solve_arm_q_target(
            robot_type,
            &left_target,
            &right_target,
            self.last_arm_q_target.as_deref(),
            Some(14),
        ) {
            Ok(target) => {
                metrics::counter!("teleop_arm_target_precomputed_count").increment(1);
                self.last_wrist_pose_pair =
                    Some((left_wrist_pose.clone(), right_wrist_pose.clone()));
                self.last_arm_q_target = Some(target.clone());
                self.last_arm_q_target_source = Some(TeleopArmTargetSource::WristPoseIk);
                (Some(target), self.last_arm_q_target_source)
            }
            Err(reason) => {
                metrics::counter!(
                    "teleop_arm_target_precompute_failed_count",
                    "reason" => reason
                )
                .increment(1);
                debug!(reason, robot_type, "arm_q_target 预计算失败");
                (None, None)
            }
        }
    }
}

pub fn hand_target_from_joints(joints: Option<HandJointTarget>) -> Option<Vec<f32>> {
    let joints = joints?;
    let ranges = hand_joint_target_limits();
    let mut out = Vec::with_capacity(HAND_JOINT_TARGET_DIM);
    for (index, value) in joints.into_iter().enumerate() {
        if !value.is_finite() {
            metrics::counter!(
                "teleop_hand_target_precompute_failed_count",
                "reason" => crate::reason::REASON_NAN_OR_INF
            )
            .increment(1);
            return None;
        }
        let (min_value, max_value) = ranges[index];
        out.push(value.clamp(min_value, max_value));
    }
    metrics::counter!("teleop_hand_target_precomputed_count").increment(1);
    Some(out)
}

pub fn hand_joint_target_from_hand_kpts_3d(
    hand_kpts_3d: &[[f32; 3]],
    base: usize,
    _is_left: bool,
) -> Option<HandJointTarget> {
    if hand_kpts_3d.len() < base + 21 {
        return None;
    }

    let wrist = vec3_at(hand_kpts_3d, base)?;
    let thumb_cmc = vec3_at(hand_kpts_3d, base + 1)?;
    let thumb_mcp = vec3_at(hand_kpts_3d, base + 2)?;
    let thumb_ip = vec3_at(hand_kpts_3d, base + 3)?;
    let thumb_tip = vec3_at(hand_kpts_3d, base + 4)?;
    let index_mcp = vec3_at(hand_kpts_3d, base + 5)?;
    let index_pip = vec3_at(hand_kpts_3d, base + 6)?;
    let index_dip = vec3_at(hand_kpts_3d, base + 7)?;
    let index_tip = vec3_at(hand_kpts_3d, base + 8)?;
    let middle_mcp = vec3_at(hand_kpts_3d, base + 9)?;
    let middle_pip = vec3_at(hand_kpts_3d, base + 10)?;
    let middle_dip = vec3_at(hand_kpts_3d, base + 11)?;
    let middle_tip = vec3_at(hand_kpts_3d, base + 12)?;
    let ring_mcp = vec3_at(hand_kpts_3d, base + 13)?;
    let ring_pip = vec3_at(hand_kpts_3d, base + 14)?;
    let ring_dip = vec3_at(hand_kpts_3d, base + 15)?;
    let ring_tip = vec3_at(hand_kpts_3d, base + 16)?;
    let pinky_mcp = vec3_at(hand_kpts_3d, base + 17)?;
    let pinky_pip = vec3_at(hand_kpts_3d, base + 18)?;
    let pinky_dip = vec3_at(hand_kpts_3d, base + 19)?;
    let pinky_tip = vec3_at(hand_kpts_3d, base + 20)?;

    let palm_forward = normalize_vec(middle_mcp - wrist)?;
    let palm_across = normalize_vec(index_mcp - pinky_mcp)?;
    let palm_normal = normalize_vec(palm_forward.cross(&palm_across))?;
    let thumb_axis = normalize_vec(thumb_mcp - thumb_cmc)?;
    let thumb_plane_axis = normalize_vec(thumb_axis - palm_normal * thumb_axis.dot(&palm_normal))?;

    let thumb_yaw = palm_forward
        .cross(&thumb_plane_axis)
        .norm()
        .atan2(palm_forward.dot(&thumb_plane_axis).max(-1.0))
        .clamp(0.0, THUMB_YAW_MAX_RAD);

    let thumb_pitch = flexion_angle(wrist, thumb_cmc, thumb_mcp, THUMB_PITCH_MAX_RAD)?;
    let thumb_mcp_flex = flexion_angle(thumb_cmc, thumb_mcp, thumb_ip, THUMB_MCP_MAX_RAD)?;
    let thumb_ip_flex = flexion_angle(thumb_mcp, thumb_ip, thumb_tip, THUMB_IP_MAX_RAD)?;

    let index_mcp_flex = flexion_angle(wrist, index_mcp, index_pip, FINGER_MCP_MAX_RAD)?;
    let index_pip_flex = flexion_angle(index_mcp, index_pip, index_dip, FINGER_PIP_MAX_RAD)?;
    let index_dip_flex = flexion_angle(index_pip, index_dip, index_tip, FINGER_DIP_MAX_RAD)?;

    let middle_mcp_flex = flexion_angle(wrist, middle_mcp, middle_pip, FINGER_MCP_MAX_RAD)?;
    let middle_pip_flex = flexion_angle(middle_mcp, middle_pip, middle_dip, FINGER_PIP_MAX_RAD)?;
    let middle_dip_flex = flexion_angle(middle_pip, middle_dip, middle_tip, FINGER_DIP_MAX_RAD)?;

    let ring_mcp_flex = flexion_angle(wrist, ring_mcp, ring_pip, FINGER_MCP_MAX_RAD)?;
    let ring_pip_flex = flexion_angle(ring_mcp, ring_pip, ring_dip, FINGER_PIP_MAX_RAD)?;
    let ring_dip_flex = flexion_angle(ring_pip, ring_dip, ring_tip, FINGER_DIP_MAX_RAD)?;

    let pinky_mcp_flex = flexion_angle(wrist, pinky_mcp, pinky_pip, FINGER_MCP_MAX_RAD)?;
    let pinky_pip_flex = flexion_angle(pinky_mcp, pinky_pip, pinky_dip, FINGER_PIP_MAX_RAD)?;
    let pinky_dip_flex = flexion_angle(pinky_pip, pinky_dip, pinky_tip, FINGER_DIP_MAX_RAD)?;

    let out = [
        thumb_yaw,
        thumb_pitch,
        thumb_mcp_flex,
        thumb_ip_flex,
        index_mcp_flex,
        index_pip_flex,
        index_dip_flex,
        middle_mcp_flex,
        middle_pip_flex,
        middle_dip_flex,
        ring_mcp_flex,
        ring_pip_flex,
        ring_dip_flex,
        pinky_mcp_flex,
        pinky_pip_flex,
        pinky_dip_flex,
    ];
    if out.iter().all(|value| value.is_finite()) {
        Some(out)
    } else {
        None
    }
}

pub fn hand_curls_from_hand_joint_target(joints: &HandJointTarget) -> [f32; 5] {
    let mut out = Vec::with_capacity(5);
    let ranges = [
        THUMB_PITCH_MAX_RAD + THUMB_MCP_MAX_RAD + THUMB_IP_MAX_RAD,
        FINGER_MCP_MAX_RAD + FINGER_PIP_MAX_RAD + FINGER_DIP_MAX_RAD,
        FINGER_MCP_MAX_RAD + FINGER_PIP_MAX_RAD + FINGER_DIP_MAX_RAD,
        FINGER_MCP_MAX_RAD + FINGER_PIP_MAX_RAD + FINGER_DIP_MAX_RAD,
        FINGER_MCP_MAX_RAD + FINGER_PIP_MAX_RAD + FINGER_DIP_MAX_RAD,
    ];
    let sums = [
        joints[1] + joints[2] + joints[3],
        joints[4] + joints[5] + joints[6],
        joints[7] + joints[8] + joints[9],
        joints[10] + joints[11] + joints[12],
        joints[13] + joints[14] + joints[15],
    ];
    for (value, max_value) in sums.into_iter().zip(ranges) {
        out.push((value / max_value).clamp(0.0, 1.0));
    }
    [out[0], out[1], out[2], out[3], out[4]]
}

pub fn hand_joint_target_from_curls(curls: [f32; 5]) -> HandJointTarget {
    let thumb_total =
        (curls[0].clamp(0.0, 1.0)) * (THUMB_PITCH_MAX_RAD + THUMB_MCP_MAX_RAD + THUMB_IP_MAX_RAD);
    let finger_total = |index: usize| {
        (curls[index].clamp(0.0, 1.0))
            * (FINGER_MCP_MAX_RAD + FINGER_PIP_MAX_RAD + FINGER_DIP_MAX_RAD)
    };
    let split_thumb = |total: f32| {
        let denom = THUMB_PITCH_MAX_RAD + THUMB_MCP_MAX_RAD + THUMB_IP_MAX_RAD;
        let ratio = if denom > 1e-6 { total / denom } else { 0.0 };
        [
            THUMB_PITCH_MAX_RAD * ratio,
            THUMB_MCP_MAX_RAD * ratio,
            THUMB_IP_MAX_RAD * ratio,
        ]
    };
    let split_finger = |total: f32| {
        let denom = FINGER_MCP_MAX_RAD + FINGER_PIP_MAX_RAD + FINGER_DIP_MAX_RAD;
        let ratio = if denom > 1e-6 { total / denom } else { 0.0 };
        [
            FINGER_MCP_MAX_RAD * ratio,
            FINGER_PIP_MAX_RAD * ratio,
            FINGER_DIP_MAX_RAD * ratio,
        ]
    };
    let thumb = split_thumb(thumb_total);
    let index = split_finger(finger_total(1));
    let middle = split_finger(finger_total(2));
    let ring = split_finger(finger_total(3));
    let pinky = split_finger(finger_total(4));
    [
        0.0, thumb[0], thumb[1], thumb[2], index[0], index[1], index[2], middle[0], middle[1],
        middle[2], ring[0], ring[1], ring[2], pinky[0], pinky[1], pinky[2],
    ]
}

fn compute_lower_body_q_target(
    robot_type: &str,
    body_kpts_robot: &[[f32; 3]],
) -> (
    Option<Vec<f32>>,
    Option<Vec<f32>>,
    Option<TeleopLowerBodyTargetSource>,
) {
    let robot_type = canonical_robot_type(robot_type);
    if !matches!(robot_type, "G1_29" | "G1_23") {
        return (None, None, None);
    }

    let default_waist = vec![0.0; G1_WAIST_Q_TARGET_DIM];
    let mut leg_target = vec![0.0; G1_LEG_Q_TARGET_DIM];
    let left_leg = estimate_g1_leg_q_target(body_kpts_robot, true);
    let right_leg = estimate_g1_leg_q_target(body_kpts_robot, false);
    let mut source = TeleopLowerBodyTargetSource::StandingBalance;

    match (left_leg, right_leg) {
        (Some(left_values), Some(right_values)) => {
            leg_target[0..6].copy_from_slice(&left_values);
            leg_target[6..12].copy_from_slice(&right_values);
            source = TeleopLowerBodyTargetSource::BodyAnchor;
        }
        _ => {
            source = TeleopLowerBodyTargetSource::StandingBalance;
        }
    }

    (Some(default_waist), Some(leg_target), Some(source))
}

pub fn canonical_robot_type(robot_type: &str) -> &str {
    match robot_type {
        "unitree_g1" | "unitree_g1_29" | "g1" | "g1_29" => "G1_29",
        "unitree_g1_23" | "g1_23" => "G1_23",
        other => other,
    }
}

fn estimate_g1_leg_q_target(body_kpts_robot: &[[f32; 3]], is_left: bool) -> Option<[f32; 6]> {
    let (hip_index, knee_index, ankle_index) = if is_left {
        (COCO_LEFT_HIP, COCO_LEFT_KNEE, COCO_LEFT_ANKLE)
    } else {
        (COCO_RIGHT_HIP, COCO_RIGHT_KNEE, COCO_RIGHT_ANKLE)
    };

    let hip = vec3_at(body_kpts_robot, hip_index)?;
    let knee = vec3_at(body_kpts_robot, knee_index)?;
    let ankle = vec3_at(body_kpts_robot, ankle_index)?;
    let femur = normalize_vec(knee - hip)?;
    let shin = normalize_vec(ankle - knee)?;
    let torso = estimate_torso_axis(body_kpts_robot).unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));

    let hip_pitch = femur.x.atan2((-femur.z).max(1.0e-3)).clamp(
        -G1_HIP_PITCH_TARGET_LIMIT_RAD,
        G1_HIP_PITCH_TARGET_LIMIT_RAD,
    );
    let hip_roll = femur
        .y
        .atan2((-femur.z).max(1.0e-3))
        .clamp(-G1_HIP_ROLL_TARGET_LIMIT_RAD, G1_HIP_ROLL_TARGET_LIMIT_RAD);
    let hip_yaw = 0.5
        * torso
            .y
            .atan2(torso.x.abs().max(1.0e-3))
            .clamp(-G1_HIP_YAW_TARGET_LIMIT_RAD, G1_HIP_YAW_TARGET_LIMIT_RAD);
    let knee = angle_between_unit(femur, shin)
        .unwrap_or(0.0)
        .clamp(0.0, G1_KNEE_TARGET_LIMIT_RAD);
    let ankle_pitch = (-(hip_pitch + knee) * 0.55).clamp(
        -G1_ANKLE_PITCH_TARGET_LIMIT_RAD,
        G1_ANKLE_PITCH_TARGET_LIMIT_RAD,
    );
    let ankle_roll = (-hip_roll * 0.85).clamp(
        -G1_ANKLE_ROLL_TARGET_LIMIT_RAD,
        G1_ANKLE_ROLL_TARGET_LIMIT_RAD,
    );

    Some([hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll])
}

fn estimate_torso_axis(body_kpts_robot: &[[f32; 3]]) -> Option<Vector3<f32>> {
    let left_shoulder = vec3_at(body_kpts_robot, COCO_LEFT_SHOULDER)?;
    let right_shoulder = vec3_at(body_kpts_robot, COCO_RIGHT_SHOULDER)?;
    let left_hip = vec3_at(body_kpts_robot, COCO_LEFT_HIP)?;
    let right_hip = vec3_at(body_kpts_robot, COCO_RIGHT_HIP)?;
    let shoulder_center = (left_shoulder + right_shoulder) * 0.5;
    let hip_center = (left_hip + right_hip) * 0.5;
    normalize_vec(shoulder_center - hip_center)
}

fn angle_between_unit(a: Vector3<f32>, b: Vector3<f32>) -> Option<f32> {
    if !a.iter().all(|value| value.is_finite()) || !b.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(a.dot(&b).clamp(-1.0, 1.0).acos())
}

fn hand_joint_target_limits() -> [(f32, f32); HAND_JOINT_TARGET_DIM] {
    [
        (0.0, THUMB_YAW_MAX_RAD),
        (0.0, THUMB_PITCH_MAX_RAD),
        (0.0, THUMB_MCP_MAX_RAD),
        (0.0, THUMB_IP_MAX_RAD),
        (0.0, FINGER_MCP_MAX_RAD),
        (0.0, FINGER_PIP_MAX_RAD),
        (0.0, FINGER_DIP_MAX_RAD),
        (0.0, FINGER_MCP_MAX_RAD),
        (0.0, FINGER_PIP_MAX_RAD),
        (0.0, FINGER_DIP_MAX_RAD),
        (0.0, FINGER_MCP_MAX_RAD),
        (0.0, FINGER_PIP_MAX_RAD),
        (0.0, FINGER_DIP_MAX_RAD),
        (0.0, FINGER_MCP_MAX_RAD),
        (0.0, FINGER_PIP_MAX_RAD),
        (0.0, FINGER_DIP_MAX_RAD),
    ]
}

fn vec3_at(points: &[[f32; 3]], index: usize) -> Option<Vector3<f32>> {
    let point = points.get(index)?;
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    // Phone/stereo fusion uses [0,0,0] placeholders for missing joints. Treat them
    // as absent so lower-body retarget does not hallucinate extreme leg targets from
    // synthetic origin points.
    if point.iter().all(|value| value.abs() <= 1.0e-6) {
        return None;
    }
    Some(Vector3::new(point[0], point[1], point[2]))
}

fn normalize_vec(value: Vector3<f32>) -> Option<Vector3<f32>> {
    let norm = value.norm();
    if !norm.is_finite() || norm <= MIN_SEGMENT_NORM {
        return None;
    }
    Some(value / norm)
}

fn flexion_angle(
    parent: Vector3<f32>,
    joint: Vector3<f32>,
    child: Vector3<f32>,
    max_angle: f32,
) -> Option<f32> {
    let parent_dir = normalize_vec(parent - joint)?;
    let child_dir = normalize_vec(child - joint)?;
    let inner = parent_dir.dot(&child_dir).clamp(-1.0, 1.0).acos();
    Some((PI - inner).clamp(0.0, max_angle))
}

fn same_pose(left: &Pose, right: &Pose) -> bool {
    left.pos
        .iter()
        .zip(right.pos.iter())
        .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1.0e-6)
        && left
            .quat
            .iter()
            .zip(right.quat.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1.0e-6)
}

fn poses_are_uninitialized(left: &Pose, right: &Pose) -> bool {
    fn is_default_pose(pose: &Pose) -> bool {
        same_pose(
            pose,
            &Pose {
                pos: [0.0, 0.0, 0.0],
                quat: [0.0, 0.0, 0.0, 1.0],
            },
        )
    }
    is_default_pose(left) && is_default_pose(right)
}

struct UnitreeArmIk {
    g1_23: Option<ArmModel>,
    g1_29: ArmModel,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArmSide {
    Left,
    Right,
}

impl UnitreeArmIk {
    fn approximate_arm_q_target_from_body(
        &self,
        robot_type: &str,
        body_kpts_robot: &[[f32; 3]],
        left_wrist_pose: &Pose,
        right_wrist_pose: &Pose,
    ) -> Option<Vec<f32>> {
        let model = self.model(robot_type)?;
        let (left_target, right_target) =
            body_anchored_wrist_targets(model, body_kpts_robot, left_wrist_pose, right_wrist_pose)?;
        let left_q = model.left.approximate_solve_position_only(Vector3::new(
            left_target.pos[0],
            left_target.pos[1],
            left_target.pos[2],
        ))?;
        let right_q = model.right.approximate_solve_position_only(Vector3::new(
            right_target.pos[0],
            right_target.pos[1],
            right_target.pos[2],
        ))?;

        let mut out = Vec::with_capacity(14);
        out.extend_from_slice(&left_q);
        out.extend_from_slice(&right_q);
        Some(out)
    }

    fn approximate_arm_q_target_from_wrist_positions(
        &self,
        robot_type: &str,
        left_wrist_pose: &Pose,
        right_wrist_pose: &Pose,
    ) -> Option<Vec<f32>> {
        let model = self.model(robot_type)?;
        let left_q = model.left.approximate_solve_position_only(Vector3::new(
            left_wrist_pose.pos[0],
            left_wrist_pose.pos[1],
            left_wrist_pose.pos[2],
        ))?;
        let right_q = model.right.approximate_solve_position_only(Vector3::new(
            right_wrist_pose.pos[0],
            right_wrist_pose.pos[1],
            right_wrist_pose.pos[2],
        ))?;

        let mut out = Vec::with_capacity(14);
        out.extend_from_slice(&left_q);
        out.extend_from_slice(&right_q);
        Some(out)
    }

    fn new() -> Result<Self, String> {
        let g1_29 = ArmModel::from_urdf_str(URDF_G1_BODY29_HAND14)?;
        let g1_23 = ArmModel::from_urdf_str(URDF_G1_BODY23).ok();
        Ok(Self { g1_23, g1_29 })
    }

    fn model(&self, robot_type: &str) -> Option<&ArmModel> {
        match robot_type {
            "G1_23" => self.g1_23.as_ref(),
            "G1_29" => Some(&self.g1_29),
            _ => None,
        }
    }

    #[cfg(test)]
    fn fk_wrist_pose(&self, robot_type: &str, side: ArmSide, q: &[f32; 7]) -> Result<Pose, String> {
        let model = self
            .model(robot_type)
            .ok_or_else(|| format!("不支持的 robot_type: {robot_type}"))?;
        let (pos, rot) = match side {
            ArmSide::Left => model.left.fk(q),
            ArmSide::Right => model.right.fk(q),
        };
        Ok(Pose {
            pos: [pos.x, pos.y, pos.z],
            quat: [rot.i, rot.j, rot.k, rot.w],
        })
    }

    fn solve_arm_q_target(
        &self,
        robot_type: &str,
        left_target: &Pose,
        right_target: &Pose,
        seed: Option<&[f32]>,
        expected_len: Option<usize>,
    ) -> Result<Vec<f32>, &'static str> {
        let expected = expected_len.unwrap_or(14);
        if expected != 14 {
            return Err(crate::reason::REASON_IK_JOINT_LEN_UNSUPPORTED);
        }

        let model = self
            .model(robot_type)
            .ok_or(crate::reason::REASON_IK_UNSUPPORTED_ROBOT)?;

        let seed_left = seed
            .filter(|values| values.len() >= 7)
            .map(|values| &values[0..7])
            .map(slice_to_array7)
            .unwrap_or([0.0; 7]);
        let seed_right = seed
            .filter(|values| values.len() >= 14)
            .map(|values| &values[7..14])
            .map(slice_to_array7)
            .unwrap_or([0.0; 7]);

        let left_des = pose_to_target(left_target)?;
        let right_des = pose_to_target(right_target)?;

        let left_q = solve_with_candidate_seeds(
            &model.left,
            left_des.0,
            left_des.1,
            &candidate_seeds(seed_left, left_target.pos[1]),
        )
        .map_err(|_| crate::reason::REASON_IK_FAILED)?;
        let right_q = solve_with_candidate_seeds(
            &model.right,
            right_des.0,
            right_des.1,
            &candidate_seeds(seed_right, right_target.pos[1]),
        )
        .map_err(|_| crate::reason::REASON_IK_FAILED)?;

        let mut out = Vec::with_capacity(14);
        out.extend_from_slice(&left_q);
        out.extend_from_slice(&right_q);
        Ok(out)
    }
}

fn slice_to_array7(values: &[f32]) -> [f32; 7] {
    let mut out = [0.0f32; 7];
    for (index, value) in values.iter().take(7).copied().enumerate() {
        out[index] = value;
    }
    out
}

fn solve_with_candidate_seeds(
    chain: &KinematicChain,
    target_pos: Vector3<f32>,
    target_rot: UnitQuaternion<f32>,
    seeds: &[[f32; 7]],
) -> Result<[f32; 7], String> {
    let mut last_error = "IK did not converge".to_string();
    for seed in seeds {
        match chain.solve(target_pos, target_rot, seed) {
            Ok(q) => return Ok(q),
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

fn candidate_seeds(primary: [f32; 7], lateral_y: f32) -> Vec<[f32; 7]> {
    let lateral_sign = if lateral_y >= 0.0 { 1.0 } else { -1.0 };
    vec![
        primary,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.35 * lateral_sign, 0.0, 0.75, 0.0, 0.0, 0.0],
        [0.60, 0.45 * lateral_sign, 0.0, 1.20, 0.0, 0.0, 0.0],
        [-0.20, 0.35 * lateral_sign, 0.0, 1.00, 0.0, 0.0, 0.0],
    ]
}

fn pose_to_target(pose: &Pose) -> Result<(Vector3<f32>, UnitQuaternion<f32>), &'static str> {
    if !pose.pos.iter().all(|value| value.is_finite())
        || !pose.quat.iter().all(|value| value.is_finite())
    {
        return Err(crate::reason::REASON_NAN_OR_INF);
    }
    let pos = Vector3::new(pose.pos[0], pose.pos[1], pose.pos[2]);
    let rot = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        pose.quat[3],
        pose.quat[0],
        pose.quat[1],
        pose.quat[2],
    ));
    Ok((pos, rot))
}

struct ArmModel {
    left_shoulder_pos: Vector3<f32>,
    right_shoulder_pos: Vector3<f32>,
    left: KinematicChain,
    right: KinematicChain,
}

impl ArmModel {
    fn from_urdf_str(urdf: &str) -> Result<Self, String> {
        let robot = urdf_rs::read_from_string(urdf).map_err(|error| error.to_string())?;
        let by_child: HashMap<String, urdf_rs::Joint> = robot
            .joints
            .into_iter()
            .map(|joint| (joint.child.link.clone(), joint))
            .collect();

        let left = KinematicChain::from_chain("torso_link", "left_wrist_yaw_link", &by_child)?;
        let right = KinematicChain::from_chain("torso_link", "right_wrist_yaw_link", &by_child)?;
        let left_shoulder_pos = left.shoulder_pos();
        let right_shoulder_pos = right.shoulder_pos();
        Ok(Self {
            left_shoulder_pos,
            right_shoulder_pos,
            left,
            right,
        })
    }
}

#[derive(Clone)]
struct JointKin {
    origin_t: Vector3<f32>,
    origin_r: UnitQuaternion<f32>,
    axis: Vector3<f32>,
    lower: f32,
    upper: f32,
}

struct KinematicChain {
    joints: Vec<JointKin>,
    tool_translation: Vector3<f32>,
    upper_arm_len: f32,
    forearm_len: f32,
}

type FkFramesResult = (
    Vector3<f32>,
    UnitQuaternion<f32>,
    [Vector3<f32>; 7],
    [Vector3<f32>; 7],
);

impl KinematicChain {
    fn from_chain(
        base_link: &str,
        end_link: &str,
        by_child: &HashMap<String, urdf_rs::Joint>,
    ) -> Result<Self, String> {
        let mut joints_rev = Vec::new();
        let mut current = end_link.to_string();
        while current != base_link {
            let joint = by_child
                .get(&current)
                .ok_or_else(|| {
                    format!("URDF 缺少 child={current} 的 joint（base={base_link} end={end_link}）")
                })?
                .clone();
            joints_rev.push(joint);
            current = joints_rev.last().expect("just pushed").parent.link.clone();
        }
        joints_rev.reverse();

        let mut joints = Vec::new();
        for joint in joints_rev {
            if joint.joint_type != urdf_rs::JointType::Revolute {
                return Err(format!("joint 不是 revolute: {}", joint.name));
            }

            let xyz = joint.origin.xyz.0;
            let rpy = joint.origin.rpy.0;
            let origin_t = Vector3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32);
            let origin_r =
                UnitQuaternion::from_euler_angles(rpy[0] as f32, rpy[1] as f32, rpy[2] as f32);

            let axis_xyz = joint.axis.xyz.0;
            let axis = Vector3::new(axis_xyz[0] as f32, axis_xyz[1] as f32, axis_xyz[2] as f32);
            let axis = if axis.norm_squared() > 0.0 {
                axis.normalize()
            } else {
                Vector3::x()
            };

            joints.push(JointKin {
                origin_t,
                origin_r,
                axis,
                lower: joint.limit.lower as f32,
                upper: joint.limit.upper as f32,
            });
        }

        if joints.len() != 7 {
            return Err(format!("机械臂链路关节数不是 7，而是 {}", joints.len()));
        }

        let mut chain = Self {
            joints,
            tool_translation: Vector3::new(0.05, 0.0, 0.0),
            upper_arm_len: 0.0,
            forearm_len: 0.0,
        };
        let (tool_pos, _, joint_pos, _) = chain.fk_frames(&[0.0; 7]);
        chain.upper_arm_len = (joint_pos[3] - chain.shoulder_pos()).norm().max(1.0e-3);
        chain.forearm_len = (tool_pos - joint_pos[3]).norm().max(1.0e-3);
        Ok(chain)
    }

    #[cfg(test)]
    fn fk(&self, q: &[f32; 7]) -> (Vector3<f32>, UnitQuaternion<f32>) {
        let (pos, rot, _, _) = self.fk_frames(q);
        (pos, rot)
    }

    fn fk_frames(&self, q: &[f32; 7]) -> FkFramesResult {
        let mut pos = Vector3::new(0.0, 0.0, 0.0);
        let mut rot = UnitQuaternion::identity();
        let mut joint_pos = [Vector3::new(0.0, 0.0, 0.0); 7];
        let mut joint_axis_world = [Vector3::new(0.0, 0.0, 1.0); 7];

        for (index, joint) in self.joints.iter().enumerate() {
            pos += rot.transform_vector(&joint.origin_t);
            rot *= joint.origin_r;
            joint_pos[index] = pos;
            joint_axis_world[index] = rot.transform_vector(&joint.axis);
            let delta = UnitQuaternion::from_axis_angle(&Unit::new_normalize(joint.axis), q[index]);
            rot *= delta;
        }

        let tool_pos = pos + rot.transform_vector(&self.tool_translation);
        (tool_pos, rot, joint_pos, joint_axis_world)
    }

    fn shoulder_pos(&self) -> Vector3<f32> {
        self.joints
            .first()
            .map(|joint| joint.origin_t)
            .unwrap_or_else(Vector3::zeros)
    }

    fn solve(
        &self,
        target_pos: Vector3<f32>,
        target_rot: UnitQuaternion<f32>,
        seed: &[f32; 7],
    ) -> Result<[f32; 7], String> {
        let mut q = *seed;
        let lambda = 1.0e-2f32;
        let step = 0.6f32;

        for _ in 0..96 {
            let (pos, rot, joint_pos, joint_axis_world) = self.fk_frames(&q);

            let pos_err = target_pos - pos;
            let rot_err_q = target_rot * rot.inverse();
            let rot_err = rot_err_q.scaled_axis();

            if pos_err.norm() < 1.0e-3 && rot_err.norm() < 2.0e-2 {
                return Ok(q);
            }

            let mut jac = SMatrix::<f32, 6, 7>::zeros();
            for index in 0..7 {
                let axis = joint_axis_world[index];
                let lever = pos - joint_pos[index];
                let linear = axis.cross(&lever);
                jac[(0, index)] = linear.x;
                jac[(1, index)] = linear.y;
                jac[(2, index)] = linear.z;
                jac[(3, index)] = axis.x;
                jac[(4, index)] = axis.y;
                jac[(5, index)] = axis.z;
            }

            let error = SVector::<f32, 6>::new(
                pos_err.x, pos_err.y, pos_err.z, rot_err.x, rot_err.y, rot_err.z,
            );

            let jj_t = jac * jac.transpose();
            let damped = jj_t + SMatrix::<f32, 6, 6>::identity() * (lambda * lambda);
            let Some(inv) = damped.try_inverse() else {
                return Err("IK 矩阵求逆失败".to_string());
            };

            let delta = jac.transpose() * (inv * error) * step;
            for index in 0..7 {
                q[index] += delta[index];
                let joint = &self.joints[index];
                q[index] = q[index].clamp(joint.lower, joint.upper);
            }
        }

        Err("IK did not converge".to_string())
    }

    fn approximate_solve_position_only(&self, target_pos: Vector3<f32>) -> Option<[f32; 7]> {
        let shoulder = self.shoulder_pos();
        let delta = target_pos - shoulder;
        if !delta.iter().all(|value| value.is_finite()) {
            return None;
        }

        let forward = delta.x.max(1.0e-3);
        let lateral = delta.y;
        let vertical = delta.z;
        let reach = delta.norm();
        let max_reach = (self.upper_arm_len + self.forearm_len - 1.0e-3).max(1.0e-3);
        let min_reach = (self.upper_arm_len - self.forearm_len).abs() + 1.0e-3;
        let reach = reach.clamp(min_reach, max_reach);

        let cos_elbow = ((self.upper_arm_len * self.upper_arm_len
            + self.forearm_len * self.forearm_len
            - reach * reach)
            / (2.0 * self.upper_arm_len * self.forearm_len))
            .clamp(-1.0, 1.0);
        let elbow = std::f32::consts::PI - cos_elbow.acos();
        let shoulder_pitch = (-vertical).atan2(forward);
        let shoulder_roll =
            lateral.atan2((forward * forward + vertical * vertical).sqrt().max(1.0e-3));

        let mut q = [0.0f32; 7];
        q[0] = shoulder_pitch;
        q[1] = shoulder_roll;
        q[2] = 0.0;
        q[3] = elbow;
        q[4] = 0.0;
        q[5] = 0.0;
        q[6] = 0.0;

        for (index, joint) in self.joints.iter().enumerate() {
            q[index] = q[index].clamp(joint.lower, joint.upper);
        }
        Some(q)
    }
}

impl UnitreeArmIk {
    fn retarget_wrist_targets(
        &self,
        robot_type: &str,
        body_kpts_robot: &[[f32; 3]],
        left_wrist_pose: &Pose,
        right_wrist_pose: &Pose,
    ) -> (Pose, Pose) {
        let Some(model) = self.model(robot_type) else {
            return (left_wrist_pose.clone(), right_wrist_pose.clone());
        };
        retarget_wrist_targets_from_body(model, body_kpts_robot, left_wrist_pose, right_wrist_pose)
    }
}

fn retarget_wrist_targets_from_body(
    model: &ArmModel,
    body_kpts_robot: &[[f32; 3]],
    left_wrist_pose: &Pose,
    right_wrist_pose: &Pose,
) -> (Pose, Pose) {
    body_anchored_wrist_targets(model, body_kpts_robot, left_wrist_pose, right_wrist_pose)
        .unwrap_or_else(|| (left_wrist_pose.clone(), right_wrist_pose.clone()))
}

fn body_anchored_wrist_targets(
    model: &ArmModel,
    body_kpts_robot: &[[f32; 3]],
    left_wrist_pose: &Pose,
    right_wrist_pose: &Pose,
) -> Option<(Pose, Pose)> {
    let left_shoulder = body_kpts_robot.get(5).copied()?;
    let right_shoulder = body_kpts_robot.get(6).copied()?;
    if !is_finite3(left_shoulder) || !is_finite3(right_shoulder) {
        return None;
    }

    let human_width = norm3(sub3(right_shoulder, left_shoulder));
    if !human_width.is_finite() || human_width <= 1.0e-4 {
        return None;
    }

    let robot_width = (model.right_shoulder_pos - model.left_shoulder_pos).norm();
    if !robot_width.is_finite() || robot_width <= 1.0e-4 {
        return None;
    }

    let scale = robot_width / human_width;
    let left_rel = operator_rel_to_robot_arm_rel(sub3(left_wrist_pose.pos, left_shoulder));
    let right_rel = operator_rel_to_robot_arm_rel(sub3(right_wrist_pose.pos, right_shoulder));

    let left_target_pos = add_vec3(
        model.left_shoulder_pos,
        scale_vec3(Vector3::new(left_rel[0], left_rel[1], left_rel[2]), scale),
    );
    let right_target_pos = add_vec3(
        model.right_shoulder_pos,
        scale_vec3(
            Vector3::new(right_rel[0], right_rel[1], right_rel[2]),
            scale,
        ),
    );

    Some((
        Pose {
            pos: [left_target_pos.x, left_target_pos.y, left_target_pos.z],
            quat: [0.0, 0.0, 0.0, 1.0],
        },
        Pose {
            pos: [right_target_pos.x, right_target_pos.y, right_target_pos.z],
            quat: [0.0, 0.0, 0.0, 1.0],
        },
    ))
}

fn add_vec3(a: Vector3<f32>, b: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

fn scale_vec3(v: Vector3<f32>, scale: f32) -> Vector3<f32> {
    Vector3::new(v.x * scale, v.y * scale, v.z * scale)
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn is_finite3(v: [f32; 3]) -> bool {
    v.iter().all(|value| value.is_finite())
}

fn operator_rel_to_robot_arm_rel(relative: [f32; 3]) -> [f32; 3] {
    [relative[2], -relative[0], relative[1]]
}

const URDF_G1_BODY23: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/urdf/g1_body23.urdf"
));

const URDF_G1_BODY29_HAND14: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/urdf/g1_body29_hand14.urdf"
));

#[cfg(test)]
mod tests {
    use super::{
        hand_joint_target_from_hand_kpts_3d, hand_target_from_joints, ArmSide,
        TeleopTargetPrecomputer, UnitreeArmIk, HAND_JOINT_TARGET_DIM,
    };
    use crate::ws::types::{
        TeleopArmTargetSource, TeleopHandTargetSource, TeleopLowerBodyTargetSource,
    };

    #[test]
    fn hand_target_should_clamp_anatomical_joint_vector() {
        let target = hand_target_from_joints(Some([
            1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ]))
        .expect("target");
        assert_eq!(target.len(), HAND_JOINT_TARGET_DIM);
        assert!(target[0] <= 0.60);
        assert!(target[1] <= 1.20);
        assert!(target[5] <= 1.60);
    }

    #[test]
    fn hand_joint_target_should_be_derived_from_canonical_hand_points() {
        let target = hand_joint_target_from_hand_kpts_3d(&build_open_hand_points(), 0, true)
            .expect("joint target");
        assert_eq!(target.len(), HAND_JOINT_TARGET_DIM);
        assert!(target.iter().all(|value| value.is_finite()));
        assert!(target[0] >= 0.0);
        assert!(target[1] >= 0.0);
    }

    #[test]
    fn arm_target_should_be_precomputed_for_reachable_g1_29_pose() {
        let ik = UnitreeArmIk::new().expect("ik init");
        let q0 = [0.0f32; 7];
        let left = ik
            .fk_wrist_pose("G1_29", ArmSide::Left, &q0)
            .expect("fk left");
        let right = ik
            .fk_wrist_pose("G1_29", ArmSide::Right, &q0)
            .expect("fk right");

        let mut precomputer = TeleopTargetPrecomputer::new();
        let targets = precomputer.compute(
            "trip-test-001",
            "sess-test-001",
            "G1_29",
            1,
            &[],
            &left,
            &right,
            Some([
                0.30, 0.20, 0.10, 0.05, 0.24, 0.48, 0.32, 0.21, 0.44, 0.30, 0.18, 0.40, 0.27, 0.15,
                0.36, 0.22,
            ]),
            Some([
                0.28, 0.18, 0.09, 0.04, 0.22, 0.46, 0.31, 0.20, 0.42, 0.29, 0.16, 0.38, 0.26, 0.14,
                0.34, 0.21,
            ]),
        );

        assert_eq!(
            targets.left_hand_target.as_ref().map(Vec::len),
            Some(HAND_JOINT_TARGET_DIM)
        );
        assert_eq!(
            targets.target_debug.left_hand_target_source,
            Some(TeleopHandTargetSource::AnatomicalJoints)
        );
        assert_eq!(
            targets.right_hand_target.as_ref().map(Vec::len),
            Some(HAND_JOINT_TARGET_DIM)
        );
        assert_eq!(
            targets.target_debug.right_hand_target_source,
            Some(TeleopHandTargetSource::AnatomicalJoints)
        );
        let arm_q_target = targets.arm_q_target.expect("arm q target");
        assert_eq!(arm_q_target.len(), 14);
        assert!(arm_q_target.iter().all(|value| value.is_finite()));
        assert_eq!(
            targets.target_debug.arm_q_target_source,
            Some(TeleopArmTargetSource::WristPoseIk)
        );
    }

    fn build_open_hand_points() -> Vec<[f32; 3]> {
        let wrist = [-0.132, 0.304, 0.853];
        vec![
            wrist,
            offset(wrist, -0.010, 0.020, 0.000),
            offset(wrist, -0.022, 0.040, 0.000),
            offset(wrist, -0.034, 0.061, 0.000),
            offset(wrist, -0.046, 0.080, 0.000),
            offset(wrist, -0.018, 0.040, 0.000),
            offset(wrist, -0.020, 0.058, 0.000),
            offset(wrist, -0.024, 0.081, 0.000),
            offset(wrist, -0.028, 0.102, 0.000),
            offset(wrist, -0.004, 0.043, 0.000),
            offset(wrist, -0.005, 0.064, 0.000),
            offset(wrist, -0.005, 0.088, 0.000),
            offset(wrist, -0.005, 0.112, 0.000),
            offset(wrist, 0.012, 0.038, 0.000),
            offset(wrist, 0.014, 0.054, 0.000),
            offset(wrist, 0.018, 0.078, 0.000),
            offset(wrist, 0.022, 0.101, 0.000),
            offset(wrist, 0.028, 0.028, 0.000),
            offset(wrist, 0.033, 0.039, 0.000),
            offset(wrist, 0.037, 0.063, 0.000),
            offset(wrist, 0.041, 0.087, 0.000),
        ]
    }

    fn offset(base: [f32; 3], dx: f32, dy: f32, dz: f32) -> [f32; 3] {
        [base[0] + dx, base[1] + dy, base[2] + dz]
    }

    #[test]
    fn arm_target_should_be_precomputed_from_body_anchored_wrist_pose() {
        let body_operator = vec![
            [0.0, 0.36, 0.80],
            [-0.02, 0.37, 0.80],
            [0.02, 0.37, 0.80],
            [-0.03, 0.36, 0.80],
            [0.03, 0.36, 0.80],
            [-0.08, 0.22, 0.82],
            [0.08, 0.22, 0.82],
            [-0.11, 0.25, 0.84],
            [0.11, 0.23, 0.84],
            [-0.132, 0.304, 0.853],
            [0.132, 0.209, 0.853],
            [-0.05, 0.06, 0.80],
            [0.05, 0.06, 0.80],
            [-0.05, -0.18, 0.79],
            [0.05, -0.18, 0.79],
            [-0.05, -0.44, 0.78],
            [0.05, -0.44, 0.78],
        ];
        let left = super::Pose {
            pos: [-0.132, 0.304, 0.853],
            quat: [0.0, 0.0, -0.17126086, 0.98522574],
        };
        let right = super::Pose {
            pos: [0.132, 0.209, 0.853],
            quat: [0.0, 0.0, 0.17126101, 0.98522574],
        };
        let body = body_operator;

        let ik = UnitreeArmIk::new().expect("ik init");
        let (left_target, right_target) = ik.retarget_wrist_targets("G1_29", &body, &left, &right);
        assert!(left_target.pos.iter().all(|value| value.is_finite()));
        assert!(right_target.pos.iter().all(|value| value.is_finite()));

        let mut precomputer = TeleopTargetPrecomputer::new();
        let targets = precomputer.compute(
            "trip-test-002",
            "sess-test-002",
            "G1_29",
            1,
            &body,
            &left,
            &right,
            None,
            None,
        );
        assert!(targets.arm_q_target.is_some());
        assert_eq!(
            targets.target_debug.arm_q_target_source,
            Some(TeleopArmTargetSource::BodyAnchor)
        );
    }

    #[test]
    fn arm_target_cache_should_preserve_generation_source() {
        let body = vec![
            [0.0, 0.36, 0.80],
            [-0.02, 0.37, 0.80],
            [0.02, 0.37, 0.80],
            [-0.03, 0.36, 0.80],
            [0.03, 0.36, 0.80],
            [-0.08, 0.22, 0.82],
            [0.08, 0.22, 0.82],
            [-0.11, 0.25, 0.84],
            [0.11, 0.23, 0.84],
            [-0.132, 0.304, 0.853],
            [0.132, 0.209, 0.853],
            [-0.05, 0.06, 0.80],
            [0.05, 0.06, 0.80],
            [-0.05, -0.18, 0.79],
            [0.05, -0.18, 0.79],
            [-0.05, -0.44, 0.78],
            [0.05, -0.44, 0.78],
        ];
        let left = super::Pose {
            pos: [-0.132, 0.304, 0.853],
            quat: [0.0, 0.0, -0.17126086, 0.98522574],
        };
        let right = super::Pose {
            pos: [0.132, 0.209, 0.853],
            quat: [0.0, 0.0, 0.17126101, 0.98522574],
        };

        let mut precomputer = TeleopTargetPrecomputer::new();
        let first = precomputer.compute(
            "trip-test-003",
            "sess-test-003",
            "G1_29",
            1,
            &body,
            &left,
            &right,
            None,
            None,
        );
        let second = precomputer.compute(
            "trip-test-003",
            "sess-test-003",
            "G1_29",
            2,
            &body,
            &left,
            &right,
            None,
            None,
        );

        assert_eq!(first.arm_q_target, second.arm_q_target);
        assert_eq!(
            second.target_debug.arm_q_target_source,
            Some(TeleopArmTargetSource::BodyAnchor)
        );
    }

    #[test]
    fn arm_target_should_fallback_to_body_anchor_when_body_is_partial() {
        let partial_body = vec![[0.0, 0.0, 0.0]; 17];
        let left = super::Pose {
            pos: [-0.132, 0.304, 0.853],
            quat: [0.0, 0.0, -0.17126086, 0.98522574],
        };
        let right = super::Pose {
            pos: [0.132, 0.209, 0.853],
            quat: [0.0, 0.0, 0.17126101, 0.98522574],
        };

        let mut precomputer = TeleopTargetPrecomputer::new();
        let targets = precomputer.compute(
            "trip-test-004",
            "sess-test-004",
            "G1_29",
            1,
            &partial_body,
            &left,
            &right,
            None,
            None,
        );

        assert!(targets.arm_q_target.is_some());
        assert_eq!(
            targets.target_debug.arm_q_target_source,
            Some(TeleopArmTargetSource::BodyAnchor)
        );
    }

    #[test]
    fn lower_body_target_should_be_precomputed_for_g1_pose() {
        let body = vec![
            [0.0, 0.36, 0.80],
            [-0.02, 0.37, 0.80],
            [0.02, 0.37, 0.80],
            [-0.03, 0.36, 0.80],
            [0.03, 0.36, 0.80],
            [-0.08, 0.22, 0.82],
            [0.08, 0.22, 0.82],
            [-0.11, 0.25, 0.84],
            [0.11, 0.23, 0.84],
            [-0.132, 0.304, 0.853],
            [0.132, 0.209, 0.853],
            [-0.05, 0.06, 0.80],
            [0.05, 0.06, 0.80],
            [-0.06, -0.18, 0.74],
            [0.06, -0.18, 0.74],
            [-0.05, -0.44, 0.72],
            [0.05, -0.44, 0.72],
        ];
        let left = super::Pose {
            pos: [-0.132, 0.304, 0.853],
            quat: [0.0, 0.0, -0.17126086, 0.98522574],
        };
        let right = super::Pose {
            pos: [0.132, 0.209, 0.853],
            quat: [0.0, 0.0, 0.17126101, 0.98522574],
        };

        let mut precomputer = TeleopTargetPrecomputer::new();
        let targets = precomputer.compute(
            "trip-test-004",
            "sess-test-004",
            "G1_29",
            1,
            &body,
            &left,
            &right,
            None,
            None,
        );

        assert_eq!(targets.waist_q_target.as_ref().map(Vec::len), Some(1));
        assert_eq!(targets.leg_q_target.as_ref().map(Vec::len), Some(12));
        assert!(targets
            .leg_q_target
            .as_ref()
            .expect("leg")
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(
            targets.target_debug.lower_body_q_target_source,
            Some(TeleopLowerBodyTargetSource::BodyAnchor)
        );
    }
}
