use thiserror::Error;

use crate::bridge::types::Pose;
use crate::bridge::types::TeleopFrameV1;
use crate::reason;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("JSON 解析失败: {0}")]
    InvalidJson(String),
    #[error("协议字段无效: {0}")]
    InvalidField(&'static str),
    #[error("维度不匹配: {0}")]
    DimensionMismatch(&'static str),
    #[error("数值非法（NaN/Inf）: {0}")]
    NanOrInf(&'static str),
}

#[derive(Clone, Debug)]
pub struct ParsedTeleopFrame {
    pub trip_id: String,
    pub session_id: String,
    pub robot_type: String,
    pub end_effector_type: String,
    pub edge_time_ns: u64,
    pub control_state: String,
    pub safety_state: String,
    pub body_control_enabled: bool,

    pub left_wrist_pose: Pose,
    pub right_wrist_pose: Pose,

    pub arm_q_target: Option<Vec<f32>>,
    pub arm_tauff_target: Option<Vec<f32>>,
    pub left_hand_target: Option<Vec<f32>>,
    pub right_hand_target: Option<Vec<f32>>,
}

pub fn parse_teleop_frame_v1_json(
    raw: &str,
    expected_arm_joint_len: Option<usize>,
    expected_dex_joint_len: Option<usize>,
) -> Result<ParsedTeleopFrame, ParseError> {
    let frame = serde_json::from_str::<TeleopFrameV1>(raw)
        .map_err(|e| ParseError::InvalidJson(e.to_string()))?;
    parse_teleop_frame_v1(frame, expected_arm_joint_len, expected_dex_joint_len)
}

pub fn parse_teleop_frame_v1(
    frame: TeleopFrameV1,
    expected_arm_joint_len: Option<usize>,
    expected_dex_joint_len: Option<usize>,
) -> Result<ParsedTeleopFrame, ParseError> {
    if frame.schema_version != "teleop_frame_v1" {
        return Err(ParseError::InvalidField(reason::REASON_SCHEMA_INVALID));
    }
    if frame.trip_id.trim().is_empty() || frame.session_id.trim().is_empty() {
        return Err(ParseError::InvalidField("trip_id/session_id 不能为空"));
    }
    if frame.control_state != "armed" && frame.control_state != "disarmed" {
        return Err(ParseError::InvalidField(
            "control_state 必须为 armed/disarmed",
        ));
    }
    if !matches!(
        frame.safety_state.as_str(),
        "normal" | "limit" | "freeze" | "estop"
    ) {
        return Err(ParseError::InvalidField("safety_state 非法"));
    }

    if !all_finite_pose(&frame.left_wrist_pose.pos, &frame.left_wrist_pose.quat)
        || !all_finite_pose(&frame.right_wrist_pose.pos, &frame.right_wrist_pose.quat)
    {
        return Err(ParseError::NanOrInf(reason::REASON_NAN_OR_INF));
    }

    let arm_q_target = frame.arm_q_target.filter(|v| !v.is_empty());
    let arm_tauff_target = frame.arm_tauff_target.filter(|v| !v.is_empty());
    let left_hand_target = select_hand_target(&frame.left_hand_target, &frame.left_hand_joints);
    let right_hand_target = select_hand_target(&frame.right_hand_target, &frame.right_hand_joints);

    if let Some(len) = expected_arm_joint_len {
        if let Some(v) = arm_q_target.as_ref() {
            if v.len() != len {
                return Err(ParseError::DimensionMismatch(
                    reason::REASON_DIMENSION_MISMATCH,
                ));
            }
        }
        if let Some(v) = arm_tauff_target.as_ref() {
            if v.len() != len {
                return Err(ParseError::DimensionMismatch(
                    reason::REASON_DIMENSION_MISMATCH,
                ));
            }
        }
    }
    if let Some(len) = expected_dex_joint_len {
        if let Some(v) = left_hand_target.as_ref() {
            if v.len() != len {
                return Err(ParseError::DimensionMismatch(
                    reason::REASON_DIMENSION_MISMATCH,
                ));
            }
        }
        if let Some(v) = right_hand_target.as_ref() {
            if v.len() != len {
                return Err(ParseError::DimensionMismatch(
                    reason::REASON_DIMENSION_MISMATCH,
                ));
            }
        }
    }

    if let Some(v) = arm_q_target.as_ref() {
        if !all_finite(v) {
            return Err(ParseError::NanOrInf(reason::REASON_NAN_OR_INF));
        }
    }
    if let Some(v) = arm_tauff_target.as_ref() {
        if !all_finite(v) {
            return Err(ParseError::NanOrInf(reason::REASON_NAN_OR_INF));
        }
    }
    if let Some(v) = left_hand_target.as_ref() {
        if !all_finite(v) {
            return Err(ParseError::NanOrInf(reason::REASON_NAN_OR_INF));
        }
    }
    if let Some(v) = right_hand_target.as_ref() {
        if !all_finite(v) {
            return Err(ParseError::NanOrInf(reason::REASON_NAN_OR_INF));
        }
    }

    if let (Some(l), Some(r)) = (left_hand_target.as_ref(), right_hand_target.as_ref()) {
        if l.len() != r.len() {
            return Err(ParseError::DimensionMismatch(
                reason::REASON_DIMENSION_MISMATCH,
            ));
        }
    }

    Ok(ParsedTeleopFrame {
        trip_id: frame.trip_id,
        session_id: frame.session_id,
        robot_type: frame.robot_type,
        end_effector_type: frame.end_effector_type,
        edge_time_ns: frame.edge_time_ns,
        control_state: frame.control_state,
        safety_state: frame.safety_state,
        body_control_enabled: frame.body_control_enabled.unwrap_or(true),
        left_wrist_pose: frame.left_wrist_pose,
        right_wrist_pose: frame.right_wrist_pose,
        arm_q_target,
        arm_tauff_target,
        left_hand_target,
        right_hand_target,
    })
}

fn select_hand_target(target: &Option<Vec<f32>>, joints: &Option<Vec<f32>>) -> Option<Vec<f32>> {
    target
        .clone()
        .or_else(|| joints.clone())
        .filter(|v| !v.is_empty())
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn all_finite_pose(pos: &[f32; 3], quat: &[f32; 4]) -> bool {
    pos.iter().all(|x| x.is_finite()) && quat.iter().all(|x| x.is_finite())
}
