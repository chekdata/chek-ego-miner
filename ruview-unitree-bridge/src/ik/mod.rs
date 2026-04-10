use std::collections::HashMap;

use nalgebra::{SMatrix, SVector, Unit, UnitQuaternion, Vector3};

use crate::bridge::types::Pose;
use crate::reason;

/// 机械臂 IK（MVP）：用于在 `arm_q_target` 缺失时，按 `left/right_wrist_pose` 回退求解关节角。
///
/// 说明：
/// - 首版只覆盖 Unitree G1（Body23/Body29），并以 URDF 为事实源。
/// - 当前实现采用阻尼最小二乘（DLS）的数值 IK：稳定、实现简单，但不是最优/最快。
/// - 目标 pose 约定在 URDF base（`torso_link`）坐标系下；外参转换由上游完成。
pub struct UnitreeIk {
    g1_23: Option<ArmModel>,
    g1_29: ArmModel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArmSide {
    Left,
    Right,
}

impl UnitreeIk {
    pub fn new() -> Result<Self, String> {
        let g1_29 = ArmModel::from_urdf_str(URDF_G1_BODY29_HAND14)?;
        // 注意：teleops-reference 的 `g1_body23.urdf` 默认只有 5DOF（无 wrist_pitch/yaw）。
        // 为避免阻塞主路径，这里把 G1_23 的 IK 视为“可选能力”，构建失败不影响启动。
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

    pub fn fk_wrist_pose(
        &self,
        robot_type: &str,
        side: ArmSide,
        q: &[f32; 7],
    ) -> Result<Pose, String> {
        let model = self
            .model(robot_type)
            .ok_or_else(|| format!("不支持的 robot_type: {robot_type}"))?;
        let (p, r) = match side {
            ArmSide::Left => model.left.fk(q),
            ArmSide::Right => model.right.fk(q),
        };
        Ok(Pose {
            pos: [p.x, p.y, p.z],
            quat: [r.i, r.j, r.k, r.w],
        })
    }

    /// 基于 wrist pose 求解 `arm_q_target`（默认输出：左臂 7 + 右臂 7 = 14）。
    pub fn solve_arm_q_target(
        &self,
        robot_type: &str,
        left_target: &Pose,
        right_target: &Pose,
        seed: Option<&[f32]>,
        expected_len: Option<usize>,
    ) -> Result<Vec<f32>, &'static str> {
        let expected = expected_len.unwrap_or(14);
        if expected != 14 {
            return Err(reason::REASON_IK_JOINT_LEN_UNSUPPORTED);
        }

        let model = self
            .model(robot_type)
            .ok_or(reason::REASON_IK_UNSUPPORTED_ROBOT)?;

        let seed_left = seed
            .filter(|s| s.len() >= 7)
            .map(|s| &s[0..7])
            .map(slice_to_array7)
            .unwrap_or([0.0; 7]);
        let seed_right = seed
            .filter(|s| s.len() >= 14)
            .map(|s| &s[7..14])
            .map(slice_to_array7)
            .unwrap_or([0.0; 7]);

        let left_des = pose_to_target(left_target)?;
        let right_des = pose_to_target(right_target)?;

        let left_q = model
            .left
            .solve(left_des.0, left_des.1, &seed_left)
            .map_err(|_| reason::REASON_IK_FAILED)?;
        let right_q = model
            .right
            .solve(right_des.0, right_des.1, &seed_right)
            .map_err(|_| reason::REASON_IK_FAILED)?;

        let mut out: Vec<f32> = Vec::with_capacity(14);
        out.extend_from_slice(&left_q);
        out.extend_from_slice(&right_q);
        Ok(out)
    }
}

fn slice_to_array7(v: &[f32]) -> [f32; 7] {
    let mut out = [0.0f32; 7];
    for (i, x) in v.iter().take(7).copied().enumerate() {
        out[i] = x;
    }
    out
}

fn pose_to_target(p: &Pose) -> Result<(Vector3<f32>, UnitQuaternion<f32>), &'static str> {
    let pos = Vector3::new(p.pos[0], p.pos[1], p.pos[2]);
    // PRD 示例使用 [x,y,z,w]；这里按该约定解析，并做归一化。
    let (x, y, z, w) = (p.quat[0], p.quat[1], p.quat[2], p.quat[3]);
    if !x.is_finite() || !y.is_finite() || !z.is_finite() || !w.is_finite() {
        return Err(reason::REASON_NAN_OR_INF);
    }
    let rot = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(w, x, y, z));
    Ok((pos, rot))
}

struct ArmModel {
    left: KinematicChain,
    right: KinematicChain,
}

impl ArmModel {
    fn from_urdf_str(urdf: &str) -> Result<Self, String> {
        let robot = urdf_rs::read_from_string(urdf).map_err(|e| e.to_string())?;
        let by_child: HashMap<String, urdf_rs::Joint> = robot
            .joints
            .into_iter()
            .map(|j| (j.child.link.clone(), j))
            .collect();

        let base = "torso_link";
        let left_end = "left_wrist_yaw_link";
        let right_end = "right_wrist_yaw_link";

        let left = KinematicChain::from_chain(base, left_end, &by_child)?;
        let right = KinematicChain::from_chain(base, right_end, &by_child)?;
        Ok(Self { left, right })
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
    joints: Vec<JointKin>, // 7 DOF（G1 手臂）
    tool_translation: Vector3<f32>,
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
        let mut joints_rev: Vec<urdf_rs::Joint> = Vec::new();
        let mut cur = end_link.to_string();
        while cur != base_link {
            let j = by_child
                .get(&cur)
                .ok_or_else(|| {
                    format!("URDF 缺少 child={cur} 的 joint（base={base_link} end={end_link}）")
                })?
                .clone();
            joints_rev.push(j);
            cur = joints_rev.last().expect("just pushed").parent.link.clone();
        }
        joints_rev.reverse();

        let mut joints: Vec<JointKin> = Vec::new();
        for j in joints_rev {
            if j.joint_type != urdf_rs::JointType::Revolute {
                return Err(format!("joint 不是 revolute: {}", j.name));
            }
            let xyz = j.origin.xyz.0;
            let rpy = j.origin.rpy.0;
            let origin_t = Vector3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32);
            let origin_r =
                UnitQuaternion::from_euler_angles(rpy[0] as f32, rpy[1] as f32, rpy[2] as f32);

            let a = j.axis.xyz.0;
            let axis = Vector3::new(a[0] as f32, a[1] as f32, a[2] as f32);
            let axis = if axis.norm_squared() > 0.0 {
                axis.normalize()
            } else {
                Vector3::x()
            };

            let (lower, upper) = (j.limit.lower as f32, j.limit.upper as f32);

            joints.push(JointKin {
                origin_t,
                origin_r,
                axis,
                lower,
                upper,
            });
        }

        if joints.len() != 7 {
            return Err(format!("arm joint 数量不是 7：got={}", joints.len()));
        }

        Ok(Self {
            joints,
            tool_translation: Vector3::new(0.05, 0.0, 0.0),
        })
    }

    fn fk(&self, q: &[f32; 7]) -> (Vector3<f32>, UnitQuaternion<f32>) {
        let mut p = Vector3::zeros();
        let mut r = UnitQuaternion::identity();
        for (idx, j) in self.joints.iter().enumerate() {
            p += r.transform_vector(&j.origin_t);
            r *= j.origin_r;
            let rot = UnitQuaternion::from_axis_angle(&Unit::new_normalize(j.axis), q[idx]);
            r *= rot;
        }
        (p, r)
    }

    fn fk_with_frames(&self, q: &[f32; 7]) -> FkFramesResult {
        let mut p = Vector3::zeros();
        let mut r = UnitQuaternion::identity();
        let mut joint_pos = [Vector3::zeros(); 7];
        let mut joint_axis = [Vector3::zeros(); 7];

        for (idx, j) in self.joints.iter().enumerate() {
            p += r.transform_vector(&j.origin_t);
            r *= j.origin_r;
            let axis_world = r.transform_vector(&j.axis);
            joint_pos[idx] = p;
            joint_axis[idx] = axis_world;
            let rot = UnitQuaternion::from_axis_angle(&Unit::new_normalize(j.axis), q[idx]);
            r *= rot;
        }
        let tool_pos = p + r.transform_vector(&self.tool_translation);
        (tool_pos, r, joint_pos, joint_axis)
    }

    fn solve(
        &self,
        target_p: Vector3<f32>,
        target_r: UnitQuaternion<f32>,
        seed: &[f32; 7],
    ) -> Result<[f32; 7], String> {
        let mut q = *seed;

        // 参数：MVP 先取“能跑通+稳定”的默认值，后续可做 config。
        let max_iter = 60;
        let lambda = 0.15f32; // 阻尼
        let alpha = 0.8f32; // 步长
        let pos_tol = 0.01f32; // 1cm
        let rot_tol = 0.25f32; // ~14deg

        for _ in 0..max_iter {
            let (p, r, joint_pos, joint_axis) = self.fk_with_frames(&q);
            let pos_err = target_p - p;

            let rot_err_q = target_r * r.inverse();
            let rot_vec = match rot_err_q.axis_angle() {
                Some((axis, angle)) => axis.into_inner() * angle,
                None => Vector3::zeros(),
            };

            if pos_err.norm() < pos_tol && rot_vec.norm() < rot_tol {
                return Ok(q);
            }

            let mut e = SVector::<f32, 6>::zeros();
            e[0] = pos_err.x;
            e[1] = pos_err.y;
            e[2] = pos_err.z;
            e[3] = rot_vec.x;
            e[4] = rot_vec.y;
            e[5] = rot_vec.z;

            let mut jmat = SMatrix::<f32, 6, 7>::zeros();
            for i in 0..7 {
                let a = joint_axis[i];
                let jp = joint_pos[i];
                let jv = a.cross(&(p - jp));
                jmat[(0, i)] = jv.x;
                jmat[(1, i)] = jv.y;
                jmat[(2, i)] = jv.z;
                jmat[(3, i)] = a.x;
                jmat[(4, i)] = a.y;
                jmat[(5, i)] = a.z;
            }

            let jj_t = jmat * jmat.transpose();
            let a = jj_t + SMatrix::<f32, 6, 6>::identity() * (lambda * lambda);
            let Some(inv) = a.try_inverse() else {
                return Err("DLS inverse failed".to_string());
            };
            let y = inv * e;
            let dq = jmat.transpose() * y;

            for i in 0..7 {
                q[i] = (q[i] + dq[i] * alpha).clamp(self.joints[i].lower, self.joints[i].upper);
            }
        }

        Err("IK did not converge".to_string())
    }
}

const URDF_G1_BODY23: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/urdf/g1_body23.urdf"
));

const URDF_G1_BODY29_HAND14: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/urdf/g1_body29_hand14.urdf"
));
