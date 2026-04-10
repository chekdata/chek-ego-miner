use thiserror::Error;

use crate::bridge::types::{HandSide, HandTargetFrame, TeleopFrameV1};
use crate::reason;

const HAND_JOINT_LAYOUT_ANATOMICAL_16: &str = "anatomical_joint_16";
const HAND_TARGET_LAYOUT_ANATOMICAL_16: &str = "anatomical_target_16";

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
pub struct ParsedHandTargets {
    pub trip_id: String,
    pub session_id: String,
    pub robot_type: String,
    pub end_effector_type: String,
    pub edge_time_ns: u64,
    pub control_state: String,
    pub safety_state: String,
    pub hand_control_enabled: bool,
    pub left: Option<HandTargetFrame>,
    pub right: Option<HandTargetFrame>,
}

impl ParsedHandTargets {
    pub fn is_for_leap(&self) -> bool {
        self.end_effector_type == "LEAP_V2"
    }
}

pub fn parse_teleop_frame_v1_json(
    raw: &str,
    expected_joint_len: Option<usize>,
) -> Result<ParsedHandTargets, ParseError> {
    let frame = serde_json::from_str::<TeleopFrameV1>(raw)
        .map_err(|e| ParseError::InvalidJson(e.to_string()))?;
    parse_teleop_frame_v1(frame, expected_joint_len)
}

pub fn parse_teleop_frame_v1(
    frame: TeleopFrameV1,
    expected_joint_len: Option<usize>,
) -> Result<ParsedHandTargets, ParseError> {
    if frame.schema_version != "teleop_frame_v1" {
        return Err(ParseError::InvalidField(reason::REASON_SCHEMA_INVALID));
    }
    if frame.trip_id.trim().is_empty() || frame.session_id.trim().is_empty() {
        return Err(ParseError::InvalidField("trip_id/session_id 不能为空"));
    }

    let left_vec = select_hand_vec(&frame, HandSide::Left)?;
    let right_vec = select_hand_vec(&frame, HandSide::Right)?;

    if let (Some(l), Some(r)) = (left_vec.as_ref(), right_vec.as_ref()) {
        if l.len() != r.len() {
            return Err(ParseError::DimensionMismatch(
                reason::REASON_DIMENSION_MISMATCH,
            ));
        }
    }

    if let Some(len) = expected_joint_len {
        if let Some(v) = left_vec.as_ref() {
            if v.len() != len {
                return Err(ParseError::DimensionMismatch(
                    reason::REASON_DIMENSION_MISMATCH,
                ));
            }
        }
        if let Some(v) = right_vec.as_ref() {
            if v.len() != len {
                return Err(ParseError::DimensionMismatch(
                    reason::REASON_DIMENSION_MISMATCH,
                ));
            }
        }
    }

    if let Some(v) = left_vec.as_ref() {
        if !all_finite(v) {
            return Err(ParseError::NanOrInf(reason::REASON_NAN_OR_INF));
        }
    }
    if let Some(v) = right_vec.as_ref() {
        if !all_finite(v) {
            return Err(ParseError::NanOrInf(reason::REASON_NAN_OR_INF));
        }
    }

    Ok(ParsedHandTargets {
        trip_id: frame.trip_id,
        session_id: frame.session_id,
        robot_type: frame.robot_type,
        end_effector_type: frame.end_effector_type,
        edge_time_ns: frame.edge_time_ns,
        control_state: frame.control_state,
        safety_state: frame.safety_state,
        hand_control_enabled: frame.hand_control_enabled.unwrap_or(true),
        left: left_vec.map(|target| HandTargetFrame {
            side: HandSide::Left,
            edge_time_ns: frame.edge_time_ns,
            target,
        }),
        right: right_vec.map(|target| HandTargetFrame {
            side: HandSide::Right,
            edge_time_ns: frame.edge_time_ns,
            target,
        }),
    })
}

fn select_hand_vec(frame: &TeleopFrameV1, side: HandSide) -> Result<Option<Vec<f32>>, ParseError> {
    match side {
        HandSide::Left => {
            if let Some(target) = frame.left_hand_target.as_ref().filter(|v| !v.is_empty()) {
                validate_hand_target_layout(frame.hand_target_layout.as_deref(), target.len())?;
                return Ok(Some(target.clone()));
            }
            if let Some(joints) = frame.left_hand_joints.as_ref().filter(|v| !v.is_empty()) {
                validate_hand_joint_layout(frame.hand_joint_layout.as_deref(), joints.len())?;
                return Ok(Some(joints.clone()));
            }
            Ok(None)
        }
        HandSide::Right => {
            if let Some(target) = frame.right_hand_target.as_ref().filter(|v| !v.is_empty()) {
                validate_hand_target_layout(frame.hand_target_layout.as_deref(), target.len())?;
                return Ok(Some(target.clone()));
            }
            if let Some(joints) = frame.right_hand_joints.as_ref().filter(|v| !v.is_empty()) {
                validate_hand_joint_layout(frame.hand_joint_layout.as_deref(), joints.len())?;
                return Ok(Some(joints.clone()));
            }
            Ok(None)
        }
    }
}

fn validate_hand_target_layout(layout: Option<&str>, len: usize) -> Result<(), ParseError> {
    match layout {
        Some(HAND_TARGET_LAYOUT_ANATOMICAL_16) => validate_exact_len(len, 16),
        Some(_) => Err(ParseError::InvalidField("hand_target_layout 非法")),
        None => validate_legacy_hand_len(len),
    }
}

fn validate_hand_joint_layout(layout: Option<&str>, len: usize) -> Result<(), ParseError> {
    match layout {
        Some(HAND_JOINT_LAYOUT_ANATOMICAL_16) => validate_exact_len(len, 16),
        Some(_) => Err(ParseError::InvalidField("hand_joint_layout 非法")),
        None => validate_legacy_hand_len(len),
    }
}

fn validate_exact_len(actual: usize, expected: usize) -> Result<(), ParseError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ParseError::DimensionMismatch(
            reason::REASON_DIMENSION_MISMATCH,
        ))
    }
}

fn validate_legacy_hand_len(len: usize) -> Result<(), ParseError> {
    if matches!(len, 5 | 16) {
        Ok(())
    } else {
        Err(ParseError::DimensionMismatch(
            reason::REASON_DIMENSION_MISMATCH,
        ))
    }
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}
