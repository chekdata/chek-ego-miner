use std::f32::consts::PI;

use thiserror::Error;

use crate::bridge::parser::ParsedTeleopFrame;
use crate::bridge::validator::{DexKind, EndpointGuard};
use crate::dds::unitree_client::{ArmCommand, DexCommand};
use crate::reason;

#[derive(Clone, Debug)]
pub struct MapperConfig {
    pub arm_joint_min: Vec<f32>,
    pub arm_joint_max: Vec<f32>,
    pub dex_joint_min: Vec<f32>,
    pub dex_joint_max: Vec<f32>,
}

impl MapperConfig {
    pub fn new(
        arm_joint_min: Vec<f32>,
        arm_joint_max: Vec<f32>,
        dex_joint_min: Vec<f32>,
        dex_joint_max: Vec<f32>,
    ) -> Self {
        Self {
            arm_joint_min,
            arm_joint_max,
            dex_joint_min,
            dex_joint_max,
        }
    }
}

#[derive(Debug, Error)]
pub enum MapError {
    #[error("映射失败: {0}")]
    Failed(&'static str),
}

#[derive(Clone, Debug)]
pub struct MappedCommands {
    pub arm: ArmCommand,
    pub dex3_left: Option<DexCommand>,
    pub dex3_right: Option<DexCommand>,
    pub dex1_left: Option<DexCommand>,
    pub dex1_right: Option<DexCommand>,
}

pub fn map_commands(
    frame: &ParsedTeleopFrame,
    guard: &EndpointGuard,
    cfg: &MapperConfig,
) -> Result<MappedCommands, MapError> {
    let arm_q = frame
        .arm_q_target
        .clone()
        .ok_or(MapError::Failed(reason::REASON_ARM_TARGET_MISSING))?;
    let arm_tau = frame.arm_tauff_target.clone();
    let arm = ArmCommand {
        edge_time_ns: frame.edge_time_ns,
        q: clamp_vec(&arm_q, &cfg.arm_joint_min, &cfg.arm_joint_max),
        tau: arm_tau.map(|v| clamp_vec(&v, &cfg.arm_joint_min, &cfg.arm_joint_max)),
    };

    // 手部：仅当 end_effector_type 显式允许时才映射到对应 Dex family。
    let mut dex3_left = None;
    let mut dex3_right = None;
    let mut dex1_left = None;
    let mut dex1_right = None;

    if let (Some(l), Some(r)) = (
        frame.left_hand_target.as_ref(),
        frame.right_hand_target.as_ref(),
    ) {
        if guard.allow_dex(DexKind::Dex3) {
            dex3_left = Some(DexCommand {
                edge_time_ns: frame.edge_time_ns,
                q: clamp_vec(l, &cfg.dex_joint_min, &cfg.dex_joint_max),
            });
            dex3_right = Some(DexCommand {
                edge_time_ns: frame.edge_time_ns,
                q: clamp_vec(r, &cfg.dex_joint_min, &cfg.dex_joint_max),
            });
        } else if guard.allow_dex(DexKind::Dex1) {
            dex1_left = Some(DexCommand {
                edge_time_ns: frame.edge_time_ns,
                q: clamp_vec(l, &cfg.dex_joint_min, &cfg.dex_joint_max),
            });
            dex1_right = Some(DexCommand {
                edge_time_ns: frame.edge_time_ns,
                q: clamp_vec(r, &cfg.dex_joint_min, &cfg.dex_joint_max),
            });
        } else {
            // LEAP_V2/NONE：禁止 Dex，下游手部由其他 bridge 处理或不下发。
        }
    }

    Ok(MappedCommands {
        arm,
        dex3_left,
        dex3_right,
        dex1_left,
        dex1_right,
    })
}

pub fn clamp_vec(v: &[f32], min_v: &[f32], max_v: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len());
    for (idx, raw) in v.iter().copied().enumerate() {
        let (lo, hi) = if !min_v.is_empty() && idx < min_v.len() && idx < max_v.len() {
            (min_v[idx], max_v[idx])
        } else {
            (-PI, PI)
        };
        out.push(raw.clamp(lo, hi));
    }
    out
}
