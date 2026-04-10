use std::f32::consts::PI;

use crate::bridge::types::{HandSide, LeapCommandFrame, PairedHandFrame, PairingDegrade};
use crate::reason;

pub const LEAP_COMMAND_DIM: usize = 16;

#[derive(Clone, Debug)]
pub struct SideRetargetCalibration {
    pub joint_scale: Vec<f32>,
    pub joint_offset: Vec<f32>,
}

impl SideRetargetCalibration {
    pub fn new(joint_scale: Vec<f32>, joint_offset: Vec<f32>) -> Self {
        Self {
            joint_scale,
            joint_offset,
        }
    }

    pub fn inactive() -> Self {
        Self {
            joint_scale: Vec::new(),
            joint_offset: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.joint_scale
            .iter()
            .any(|value| (*value - 1.0).abs() > 1.0e-6)
            || self.joint_offset.iter().any(|value| value.abs() > 1.0e-6)
    }

    pub fn joint_count(&self) -> usize {
        self.joint_scale.len().max(self.joint_offset.len())
    }

    pub fn non_default_scale_count(&self) -> usize {
        self.joint_scale
            .iter()
            .filter(|value| (**value - 1.0).abs() > 1.0e-6)
            .count()
    }

    pub fn non_zero_offset_count(&self) -> usize {
        self.joint_offset
            .iter()
            .filter(|value| value.abs() > 1.0e-6)
            .count()
    }

    pub fn apply(&self, raw: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(raw.len());
        for (idx, value) in raw.iter().copied().enumerate() {
            let scale = calibration_scale_at(&self.joint_scale, idx);
            let offset = calibration_offset_at(&self.joint_offset, idx);
            out.push(value * scale + offset);
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct RetargetConfig {
    pub joint_min: Vec<f32>,
    pub joint_max: Vec<f32>,
    pub left_calibration: SideRetargetCalibration,
    pub right_calibration: SideRetargetCalibration,
}

impl RetargetConfig {
    pub fn new(joint_min: Vec<f32>, joint_max: Vec<f32>) -> Self {
        Self {
            joint_min,
            joint_max,
            left_calibration: SideRetargetCalibration::inactive(),
            right_calibration: SideRetargetCalibration::inactive(),
        }
    }

    pub fn with_side_calibration(
        mut self,
        side: HandSide,
        calibration: SideRetargetCalibration,
    ) -> Self {
        match side {
            HandSide::Left => self.left_calibration = calibration,
            HandSide::Right => self.right_calibration = calibration,
        }
        self
    }

    pub fn calibration_for(&self, side: HandSide) -> &SideRetargetCalibration {
        match side {
            HandSide::Left => &self.left_calibration,
            HandSide::Right => &self.right_calibration,
        }
    }
}

/// 将左右手 target 统一映射为 LEAP Hand v2 的关节目标（默认 16 维）并做限幅。
///
/// 兼容两种上游输入：
/// - **len=16**：视为 Edge 生成的 16 维解剖关节目标
///   （`thumb_yaw, thumb_pitch, thumb_mcp, thumb_ip, index/middle/ring/pinky 的 mcp/pip/dip`），
///   在 bridge 内转换为 LEAP Hand v2 命令并做限幅。
/// - **len=5**：视为 5 指 curl（thumb,index,middle,ring,pinky，范围建议 [0,1]），
///   映射为 16 维关节角目标（用于“无 LEAP 实机”的方案 B 联调）。
///
/// 若维度不在 {5,16}，返回 `dimension_mismatch`（拒绝本帧，避免下游 SDK 维度崩溃）。
pub fn retarget_paired(
    paired: &PairedHandFrame,
    cfg: &RetargetConfig,
) -> Result<LeapCommandFrame, &'static str> {
    let left = retarget_one(
        paired.left.side,
        &paired.left.target,
        cfg.calibration_for(paired.left.side),
    )?;
    let right = retarget_one(
        paired.right.side,
        &paired.right.target,
        cfg.calibration_for(paired.right.side),
    )?;

    Ok(LeapCommandFrame {
        edge_time_ns: paired.left.edge_time_ns.max(paired.right.edge_time_ns),
        left_cmd: clamp_vec(&left, cfg),
        right_cmd: clamp_vec(&right, cfg),
    })
}

pub fn clamp_vec(v: &[f32], cfg: &RetargetConfig) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len());
    for (idx, raw) in v.iter().copied().enumerate() {
        let (min_v, max_v) = joint_limit_at(idx, cfg);
        out.push(raw.clamp(min_v, max_v));
    }
    out
}

fn joint_limit_at(idx: usize, cfg: &RetargetConfig) -> (f32, f32) {
    if !cfg.joint_min.is_empty() && idx < cfg.joint_min.len() && idx < cfg.joint_max.len() {
        return (cfg.joint_min[idx], cfg.joint_max[idx]);
    }
    (-PI, PI)
}

fn retarget_one(
    side: HandSide,
    target: &[f32],
    calibration: &SideRetargetCalibration,
) -> Result<Vec<f32>, &'static str> {
    match target.len() {
        LEAP_COMMAND_DIM => Ok(calibration.apply(&anatomical16_to_leap16(side, target))),
        5 => {
            let mut c = [0.0f32; 5];
            for (i, x) in target.iter().take(5).copied().enumerate() {
                c[i] = x.clamp(0.0, 1.0);
            }
            Ok(calibration.apply(&curls5_to_leap16(c)))
        }
        _ => Err(reason::REASON_DIMENSION_MISMATCH),
    }
}

fn anatomical16_to_leap16(side: HandSide, target: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(16);
    let thumb_abduction = target[0].clamp(0.0, PI / 3.0);
    let thumb_yaw = match side {
        HandSide::Left => -thumb_abduction,
        HandSide::Right => thumb_abduction,
    };
    out.push(thumb_yaw);
    out.push(target[1].clamp(0.0, 1.2));
    out.push(target[2].clamp(0.0, 1.3));
    out.push(target[3].clamp(0.0, 1.35));
    for value in target.iter().skip(4).take(12) {
        out.push(value.clamp(0.0, PI / 2.0));
    }
    out
}

/// 5 指 curl -> 16 关节角（弧度）。
///
/// 关节顺序约定（16 维）：
/// - thumb: yaw, pitch, mcp, ip（4）
/// - index/middle/ring/pinky: mcp, pip, dip（各 3，共 12）
///
/// 说明：这里的目标是“可联调 + 输出维度稳定”。真实 LEAP 关节排序/机械结构可能不同，
/// 现场接入真实 SDK 时应以厂家/URDF/驱动定义为准。
fn curls5_to_leap16(c: [f32; 5]) -> [f32; 16] {
    let thumb = c[0];
    let index = c[1];
    let middle = c[2];
    let ring = c[3];
    let pinky = c[4];

    // 经验映射：curl 越大 -> 屈曲越大。
    // thumb yaw/pitch 使用较小幅度，避免“强制内扣”。
    let thumb_yaw = -0.35 * thumb; // adduction（负号仅为约定，实际以硬件为准）
    let thumb_pitch = 0.9 * thumb;
    let thumb_mcp = 1.2 * thumb;
    let thumb_ip = 1.0 * thumb;

    // 其余四指：MCP/PIP/DIP 分配不同比例（PIP 最大）。
    fn finger3(curl: f32) -> [f32; 3] {
        [1.1 * curl, 1.4 * curl, 1.0 * curl]
    }
    let i = finger3(index);
    let m = finger3(middle);
    let r = finger3(ring);
    let p = finger3(pinky);

    [
        thumb_yaw,
        thumb_pitch,
        thumb_mcp,
        thumb_ip, // 0..3
        i[0],
        i[1],
        i[2], // 4..6
        m[0],
        m[1],
        m[2], // 7..9
        r[0],
        r[1],
        r[2], // 10..12
        p[0],
        p[1],
        p[2], // 13..15
    ]
}

fn calibration_scale_at(values: &[f32], idx: usize) -> f32 {
    values.get(idx).copied().unwrap_or(1.0)
}

fn calibration_offset_at(values: &[f32], idx: usize) -> f32 {
    values.get(idx).copied().unwrap_or(0.0)
}

pub fn degrade_reason(degrade: PairingDegrade) -> &'static str {
    match degrade {
        PairingDegrade::Normal => crate::reason::REASON_OK,
        PairingDegrade::Hold => crate::reason::REASON_HAND_STALE_HOLD,
        PairingDegrade::Freeze => crate::reason::REASON_HAND_STALE_FREEZE,
    }
}
