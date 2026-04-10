use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::calibration::{transform_points_3d, IphoneStereoExtrinsic, WifiStereoExtrinsic};
use crate::config::Config;
use crate::sensing::{
    BodyKeypointLayout, CsiSnapshot, HandKeypointLayout, StereoSnapshot, VisionSnapshot,
    WifiPoseDiagnostics, WifiPoseSnapshot,
};
use crate::ws::types::{EndEffectorPose, OperatorState, Pose};

const PICO_BODY24_TO_COCO17: [usize; 17] =
    [15, 15, 15, 15, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8];
const PICO_HAND26_TO_MEDIAPIPE21: [usize; 21] = [
    1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25,
];
const BODY_BLEND_MAX_GAP_M: f32 = 0.15;
const HAND_BLEND_MAX_GAP_M: f32 = 0.06;
const BODY_STEREO_WEIGHT_BIAS: f32 = 1.10;
const HAND_STEREO_WEIGHT_BIAS: f32 = 1.15;
const PROJECTED_VISION_WEIGHT_SCALE: f32 = 0.55;
const MIN_SOURCE_WEIGHT: f32 = 0.05;
const BODY_REFINEMENT_SPAN_TOLERANCE_RATIO: f32 = 0.16;
const BODY_REFINEMENT_SEGMENT_TOLERANCE_RATIO: f32 = 0.18;
const BODY_REFINEMENT_SYMMETRY_RATIO: f32 = 1.30;
const BODY_REFINEMENT_SHOULDER_SPAN_MIN_M: f32 = 0.10;
const BODY_REFINEMENT_SHOULDER_SPAN_MAX_M: f32 = 0.72;
const BODY_REFINEMENT_HIP_SPAN_MIN_M: f32 = 0.08;
const BODY_REFINEMENT_HIP_SPAN_MAX_M: f32 = 0.55;
const BODY_REFINEMENT_HEAD_OFFSET_MIN_M: f32 = 0.03;
const BODY_REFINEMENT_HEAD_OFFSET_MAX_M: f32 = 0.28;
const BODY_REFINEMENT_UPPER_ARM_MIN_M: f32 = 0.05;
const BODY_REFINEMENT_UPPER_ARM_MAX_M: f32 = 0.30;
const BODY_REFINEMENT_FOREARM_MIN_M: f32 = 0.03;
const BODY_REFINEMENT_FOREARM_MAX_M: f32 = 0.22;
const BODY_REFINEMENT_THIGH_MIN_M: f32 = 0.12;
const BODY_REFINEMENT_THIGH_MAX_M: f32 = 0.62;
const BODY_REFINEMENT_CALF_MIN_M: f32 = 0.12;
const BODY_REFINEMENT_CALF_MAX_M: f32 = 0.58;
const IPHONE_HAND_MATCH_TRUST_THRESHOLD: f32 = 0.15;
const IPHONE_HAND_MATCH_MAX_GAP_M: f32 = 0.60;
const IPHONE_HAND_SHOULDER_FALLBACK_MAX_GAP_M: f32 = 0.45;
const IPHONE_HAND_FUSION_MIN_SCORE: f32 = 0.45;
const IPHONE_HAND_FUSION_MAX_GAP_M: f32 = 0.35;
const IPHONE_HAND_ALIGNMENT_UPDATE_MAX_GAP_M: f32 = 1.20;
const IPHONE_HAND_ALIGNMENT_ALPHA: f32 = 0.35;
const IPHONE_HAND_ALIGNMENT_MAX_OFFSET_M: f32 = 1.50;
const IPHONE_HAND_MATCH_STICKY_MIN_SCORE: f32 = 0.65;
const IPHONE_HAND_MATCH_STICKY_MAX_GAP_M: f32 = 0.22;
const IPHONE_HAND_MATCH_STICKY_DECAY: f32 = 0.88;
const IPHONE_WEARER_TRACK_STICKY_MIN_SCORE: f32 = 0.72;
const WIFI_SELECTED_TRACK_STICKY_MIN_SCORE: f32 = 0.50;
const WIFI_SELECTED_TRACK_STICKY_MIN_LAYOUT_SCORE: f32 = 0.70;
const WIFI_BODY_MIN_PRESENT_JOINTS: usize = 5;
const WIFI_BODY_MAX_ABS_COORD_M: f32 = 8.0;
const WIFI_BODY_MAX_AXIS_SPAN_M: f32 = 4.0;
const WIFI_BODY_MAX_DIAGONAL_SPAN_M: f32 = 5.0;
const MOTION_STATE_LIVE_ROOT_MAX_OFFSET_M: f32 = 0.10;
const MOTION_STATE_HOLD_TRANSLATION_MAX_M: f32 = 0.18;
const MOTION_STATE_VELOCITY_DAMPING: f32 = 0.82;
const MOTION_STATE_PRESENCE_DECAY: f32 = 0.94;
const MOTION_STATE_PRESENCE_MIN_FOR_HOLD: f32 = 0.15;
const MOTION_STATE_FIXED_LAG_NS: u64 = 350_000_000;
const MOTION_STATE_HISTORY_RETENTION_NS: u64 = 1_500_000_000;
const MOTION_STATE_STATE_HISTORY_RETENTION_NS: u64 = 2_500_000_000;
const MOTION_STATE_STEREO_LIVE_MAX_AGE_NS: u64 = 180_000_000;
const MOTION_STATE_PREDICTED_ROOT_STD_M: f32 = 0.14;
const MOTION_STATE_PREDICTED_HEADING_STD_RAD: f32 = 0.30;
const MOTION_STATE_STEREO_ROOT_STD_M: f32 = 0.035;
const MOTION_STATE_STEREO_HEADING_STD_RAD: f32 = 0.18;
const MOTION_STATE_CSI_ROOT_STD_M: f32 = 0.28;
const MOTION_STATE_CSI_HEADING_STD_RAD: f32 = 0.62;
const MOTION_STATE_ROOT_PROCESS_NOISE_MPS: f32 = 0.18;
const MOTION_STATE_HEADING_PROCESS_NOISE_RADPS: f32 = 0.42;
const MOTION_STATE_MIN_ROOT_STD_M: f32 = 0.025;
const MOTION_STATE_MAX_ROOT_STD_M: f32 = 1.25;
const MOTION_STATE_MIN_HEADING_STD_RAD: f32 = 0.08;
const MOTION_STATE_MAX_HEADING_STD_RAD: f32 = 1.60;
const MOTION_STATE_STEREO_ROOT_GATE_SIGMA: f32 = 4.0;
const MOTION_STATE_STEREO_HEADING_GATE_SIGMA: f32 = 4.0;
const MOTION_STATE_CSI_ROOT_GATE_SIGMA: f32 = 2.8;
const MOTION_STATE_CSI_HEADING_GATE_SIGMA: f32 = 2.6;
const IPHONE_WEARER_TRACK_STICKY_HOLD_NS: u64 = 1_500_000_000;

pub const OPERATOR_FRAME: &str = "operator_frame";
pub const STEREO_PAIR_FRAME: &str = "stereo_pair_frame";
pub const CANONICAL_BODY_FRAME: &str = "canonical_body_frame";

fn wifi_tracking_anchor_eligible(diagnostics: Option<&WifiPoseDiagnostics>) -> bool {
    let Some(diag) = diagnostics else {
        return true;
    };
    let lifecycle = diag.lifecycle_state.trim().to_ascii_lowercase();
    let gate = diag.coherence_gate_decision.trim().to_ascii_lowercase();
    let lifecycle_ok = lifecycle.is_empty() || lifecycle == "active";
    let gate_ok = gate.is_empty() || gate == "accept";
    lifecycle_ok && gate_ok
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperatorSource {
    Stereo,
    Vision3d,
    Vision2dProjected,
    WifiPose3d,
    FusedMultiSource3d,
    FusedStereoVision3d,
    FusedStereoVision2dProjected,
    Hold,
    None,
}

impl OperatorSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stereo => "stereo",
            Self::Vision3d => "vision_3d",
            Self::Vision2dProjected => "vision_2d_projected",
            Self::WifiPose3d => "wifi_pose_3d",
            Self::FusedMultiSource3d => "fused_multi_source_3d",
            Self::FusedStereoVision3d => "fused_stereo_vision_3d",
            Self::FusedStereoVision2dProjected => "fused_stereo_vision_2d_projected",
            Self::Hold => "hold",
            Self::None => "none",
        }
    }

    fn as_part_source(self) -> OperatorPartSource {
        match self {
            Self::Stereo => OperatorPartSource::Stereo,
            Self::Vision3d => OperatorPartSource::Vision3d,
            Self::Vision2dProjected => OperatorPartSource::Vision2dProjected,
            Self::WifiPose3d => OperatorPartSource::WifiPose3d,
            Self::FusedMultiSource3d => OperatorPartSource::FusedMultiSource3d,
            Self::FusedStereoVision3d => OperatorPartSource::FusedStereoVision3d,
            Self::FusedStereoVision2dProjected => OperatorPartSource::FusedStereoVision2dProjected,
            Self::Hold | Self::None => OperatorPartSource::None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OperatorPartSource {
    Stereo,
    Vision3d,
    Vision2dProjected,
    WifiPose3d,
    FusedMultiSource3d,
    FusedStereoVision3d,
    FusedStereoVision2dProjected,
    #[default]
    None,
}

impl OperatorPartSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stereo => "stereo",
            Self::Vision3d => "vision_3d",
            Self::Vision2dProjected => "vision_2d_projected",
            Self::WifiPose3d => "wifi_pose_3d",
            Self::FusedMultiSource3d => "fused_multi_source_3d",
            Self::FusedStereoVision3d => "fused_stereo_vision_3d",
            Self::FusedStereoVision2dProjected => "fused_stereo_vision_2d_projected",
            Self::None => "none",
        }
    }
}

pub fn part_source_edge_time_ns(
    part_source: OperatorPartSource,
    stereo_edge_time_ns: u64,
    vision_edge_time_ns: u64,
    wifi_edge_time_ns: u64,
    fallback_edge_time_ns: u64,
) -> u64 {
    match part_source {
        OperatorPartSource::Stereo => stereo_edge_time_ns.max(fallback_edge_time_ns),
        OperatorPartSource::Vision3d | OperatorPartSource::Vision2dProjected => {
            vision_edge_time_ns.max(fallback_edge_time_ns)
        }
        OperatorPartSource::WifiPose3d => wifi_edge_time_ns.max(fallback_edge_time_ns),
        OperatorPartSource::FusedStereoVision3d
        | OperatorPartSource::FusedStereoVision2dProjected => stereo_edge_time_ns
            .max(vision_edge_time_ns)
            .max(fallback_edge_time_ns),
        OperatorPartSource::FusedMultiSource3d => stereo_edge_time_ns
            .max(vision_edge_time_ns)
            .max(wifi_edge_time_ns)
            .max(fallback_edge_time_ns),
        OperatorPartSource::None => fallback_edge_time_ns,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OperatorSmootherMode {
    StereoLive,
    FixedLagBlend,
    HeldWithCsiPrior,
    CsiPriorOnly,
    #[default]
    Degraded,
}

impl OperatorSmootherMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StereoLive => "stereo_live",
            Self::FixedLagBlend => "fixed_lag_blend",
            Self::HeldWithCsiPrior => "held_with_csi_prior",
            Self::CsiPriorOnly => "csi_prior_only",
            Self::Degraded => "degraded",
        }
    }
}

#[derive(Clone, Debug, Default)]
struct CsiPriorObservation {
    edge_time_ns: u64,
    operator_track_id: Option<String>,
    reliability: f32,
    motion_phase: f32,
    root_zone_center_m: Option<[f32; 3]>,
    heading_yaw_rad: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MotionObservationKind {
    StereoGeometry,
    CsiPrior,
}

#[derive(Clone, Debug)]
struct MotionStateObservation {
    edge_time_ns: u64,
    kind: MotionObservationKind,
    root_pos_m: Option<[f32; 3]>,
    heading_yaw_rad: Option<f32>,
    reliability: f32,
    motion_phase: f32,
    track_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct OperatorMotionState {
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
    pub stereo_track_id: Option<String>,
    pub last_good_stereo_time_ns: u64,
    pub last_good_csi_time_ns: u64,
    pub stereo_measurement_used: bool,
    pub csi_measurement_used: bool,
    pub accepted_stereo_observations: u32,
    pub accepted_csi_observations: u32,
    pub rejected_stereo_observations: u32,
    pub rejected_csi_observations: u32,
    pub smoother_mode: OperatorSmootherMode,
    pub updated_edge_time_ns: u64,
}

impl Default for OperatorMotionState {
    fn default() -> Self {
        Self {
            root_pos_m: [0.0, 0.0, 0.0],
            root_vel_mps: [0.0, 0.0, 0.0],
            root_std_m: MOTION_STATE_PREDICTED_ROOT_STD_M * 3.0,
            heading_yaw_rad: 0.0,
            heading_rate_radps: 0.0,
            heading_std_rad: MOTION_STATE_PREDICTED_HEADING_STD_RAD * 3.0,
            motion_phase: 0.0,
            body_presence_conf: 0.0,
            csi_prior_reliability: 0.0,
            wearer_confidence: 0.0,
            stereo_track_id: None,
            last_good_stereo_time_ns: 0,
            last_good_csi_time_ns: 0,
            stereo_measurement_used: false,
            csi_measurement_used: false,
            accepted_stereo_observations: 0,
            accepted_csi_observations: 0,
            rejected_stereo_observations: 0,
            rejected_csi_observations: 0,
            smoother_mode: OperatorSmootherMode::Degraded,
            updated_edge_time_ns: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OperatorRawPose {
    pub source_edge_time_ns: u64,
    pub body_layout: BodyKeypointLayout,
    pub hand_layout: HandKeypointLayout,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub struct OperatorEstimate {
    pub source: OperatorSource,
    pub operator_state: OperatorState,
    pub raw_pose: OperatorRawPose,
    /// 手部“抓取”强度（5 指），范围建议 [0, 1]；用于 bridge retarget 的简化输入。
    pub left_hand_curls: Option<[f32; 5]>,
    pub right_hand_curls: Option<[f32; 5]>,
    pub fusion_breakdown: OperatorFusionBreakdown,
    pub association: OperatorAssociationDebug,
    pub motion_state: OperatorMotionState,
    pub updated_edge_time_ns: u64,
}

impl Default for OperatorEstimate {
    fn default() -> Self {
        Self {
            source: OperatorSource::None,
            operator_state: OperatorState {
                body_kpts_3d: Vec::new(),
                hand_kpts_3d: Vec::new(),
                end_effector_pose: EndEffectorPose {
                    left: Pose {
                        pos: [0.0, 0.0, 0.0],
                        quat: [0.0, 0.0, 0.0, 1.0],
                    },
                    right: Pose {
                        pos: [0.0, 0.0, 0.0],
                        quat: [0.0, 0.0, 0.0, 1.0],
                    },
                },
            },
            raw_pose: OperatorRawPose::default(),
            left_hand_curls: None,
            right_hand_curls: None,
            fusion_breakdown: OperatorFusionBreakdown::default(),
            association: OperatorAssociationDebug::default(),
            motion_state: OperatorMotionState::default(),
            updated_edge_time_ns: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OperatorFusionBreakdown {
    pub body_source: OperatorPartSource,
    pub hand_source: OperatorPartSource,
    pub stereo_body_joint_count: usize,
    pub vision_body_joint_count: usize,
    pub wifi_body_joint_count: usize,
    pub blended_body_joint_count: usize,
    pub stereo_hand_point_count: usize,
    pub vision_hand_point_count: usize,
    pub wifi_hand_point_count: usize,
    pub blended_hand_point_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct OperatorAssociationDebug {
    pub selected_operator_track_id: Option<String>,
    pub anchor_source: &'static str,
    pub stereo_operator_track_id: Option<String>,
    pub wifi_operator_track_id: Option<String>,
    pub iphone_operator_track_id: Option<String>,
    pub wifi_anchor_eligible: bool,
    pub wifi_lifecycle_state: String,
    pub wifi_coherence_gate_decision: String,
    pub iphone_visible_hand_count: usize,
    pub hand_match_count: usize,
    pub hand_match_score: f32,
    pub wifi_association_score: f32,
    pub wifi_layout_score: f32,
    pub wifi_zone_score: f32,
    pub wifi_motion_energy: f32,
    pub wifi_doppler_hz: f32,
    pub wifi_signal_quality: f32,
    pub wifi_zone_summary_reliable: bool,
    pub left_wrist_gap_m: Option<f32>,
    pub right_wrist_gap_m: Option<f32>,
}

#[derive(Clone, Debug)]
struct OperatorSourcePose {
    source: OperatorSource,
    raw_pose: OperatorRawPose,
    operator_track_id: Option<String>,
    body_space: String,
    hand_space: String,
    hand_geometry_trusted: bool,
    left_hand_fresh: bool,
    right_hand_fresh: bool,
    canonical_body_kpts_3d: Vec<[f32; 3]>,
    canonical_hand_kpts_3d: Vec<[f32; 3]>,
    left_hand_curls: Option<[f32; 5]>,
    right_hand_curls: Option<[f32; 5]>,
    body_weight: f32,
    hand_weight: f32,
    wifi_diagnostics: Option<WifiPoseDiagnostics>,
}

#[derive(Clone, Debug)]
pub struct OperatorSnapshot {
    pub estimate: OperatorEstimate,
    pub fresh: bool,
}

#[derive(Default)]
struct OperatorInner {
    estimate: OperatorEstimate,
    motion_state: OperatorMotionState,
    motion_observations: VecDeque<MotionStateObservation>,
    motion_state_history: VecDeque<OperatorMotionState>,
    track_motion_observations: HashMap<String, VecDeque<MotionStateObservation>>,
    track_motion_states: HashMap<String, OperatorMotionState>,
    track_motion_state_histories: HashMap<String, VecDeque<OperatorMotionState>>,
    last_update_at: Option<Instant>,
    iphone_hand_alignment_offset: Option<[f32; 3]>,
    iphone_hand_alignment_track_id: Option<String>,
    last_confirmed_wearer_track_id: Option<String>,
    last_confirmed_wearer_edge_time_ns: u64,
}

/// OperatorState 缓存与“补盲策略”：
///
/// - 当 stereo / vision 同时 fresh 时，先统一 canonical 骨架，再按关节级几何融合
/// - 仅当某一路缺失或 stale 时，才退回单一路 source
/// - vision 仍允许 2D->3D 投影，但只作为显式 `vision_2d_projected` 观测参与融合
/// - vision/stereo 都 stale 时，若 CSI 仍活跃，允许在 `OPERATOR_HOLD_MS` 内 hold 上一帧
#[derive(Default)]
pub struct OperatorStore {
    inner: RwLock<OperatorInner>,
}

impl OperatorStore {
    pub fn tick(
        &self,
        cfg: &Config,
        now_edge_time_ns: u64,
        vision: &VisionSnapshot,
        stereo: &StereoSnapshot,
        wifi_pose: &WifiPoseSnapshot,
        csi: &CsiSnapshot,
        iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>,
        wifi_stereo_calibration: Option<&WifiStereoExtrinsic>,
    ) -> OperatorEstimate {
        let now = Instant::now();
        let mut inner = self.inner.write().expect("operator lock poisoned");
        let previous_motion_state =
            (inner.motion_state.updated_edge_time_ns > 0).then_some(inner.motion_state.clone());
        let previous_selected_track_id = inner
            .estimate
            .association
            .selected_operator_track_id
            .clone();

        let stereo_pose = source_pose_from_stereo(stereo);
        let mut vision_pose = source_pose_from_vision(vision, cfg, iphone_stereo_calibration);
        if let (Some(offset), Some(offset_track_id)) = (
            inner.iphone_hand_alignment_offset,
            inner.iphone_hand_alignment_track_id.as_deref(),
        ) {
            let current_stereo_track_id = stereo_pose
                .as_ref()
                .and_then(|source| source.operator_track_id.as_deref());
            let should_apply_offset = current_stereo_track_id == Some(offset_track_id)
                || previous_selected_track_id.as_deref() == Some(offset_track_id);
            if should_apply_offset {
                vision_pose = vision_pose
                    .as_ref()
                    .map(|source| apply_vision_hand_alignment_offset(source, offset));
            }
        }
        let wifi_pose = source_pose_from_wifi(wifi_pose, wifi_stereo_calibration);
        let csi_prior = derive_csi_prior_observation(wifi_pose.as_ref(), csi, now_edge_time_ns);

        if let Some(est) = fuse_operator_sources(
            now_edge_time_ns,
            stereo_pose.as_ref(),
            vision_pose.as_ref(),
            wifi_pose.as_ref(),
        ) {
            let est = stabilize_operator_estimate(
                est,
                Some(&inner.estimate),
                inner.last_update_at.is_some_and(|t| {
                    now.duration_since(t) <= Duration::from_millis(cfg.operator_hold_ms)
                }),
            );
            let active_stereo_track_id = resolve_active_stereo_track_id(
                Some(&est),
                stereo_pose.as_ref(),
                previous_selected_track_id.as_deref(),
            );
            let motion_observations = build_motion_state_observations(
                Some(&est),
                stereo_pose.as_ref(),
                csi_prior.as_ref(),
                now_edge_time_ns,
            );
            record_motion_state_observations(
                &mut inner.motion_observations,
                motion_observations.clone(),
                now_edge_time_ns,
            );
            record_track_motion_state_observations(
                &mut inner.track_motion_observations,
                active_stereo_track_id.as_deref(),
                &motion_observations,
                now_edge_time_ns,
            );
            let (
                scoped_previous_motion_state,
                scoped_motion_state_history,
                scoped_motion_observations,
            ) = scoped_motion_state_context(&inner, active_stereo_track_id.as_deref());
            let motion_state = optimize_operator_motion_state(
                scoped_previous_motion_state
                    .as_ref()
                    .or(previous_motion_state.as_ref()),
                &scoped_motion_state_history,
                if scoped_motion_observations.is_empty() {
                    &inner.motion_observations
                } else {
                    &scoped_motion_observations
                },
                now_edge_time_ns,
                false,
            );
            let mut est = apply_live_motion_state_smoothing(est, &motion_state);
            est.motion_state = motion_state.clone();
            apply_confirmed_wearer_sticky(&mut inner, &mut est, now_edge_time_ns);
            update_iphone_hand_alignment_state(
                &mut inner,
                stereo_pose.as_ref(),
                vision_pose.as_ref(),
                &est,
            );
            record_motion_state_snapshot(
                &mut inner.motion_state_history,
                &motion_state,
                now_edge_time_ns,
            );
            {
                let OperatorInner {
                    track_motion_states,
                    track_motion_state_histories,
                    ..
                } = &mut *inner;
                record_track_motion_state_snapshot(
                    track_motion_states,
                    track_motion_state_histories,
                    &motion_state,
                    now_edge_time_ns,
                );
            }
            inner.motion_state = motion_state;
            inner.estimate = est.clone();
            inner.last_update_at = Some(now);
            return est;
        }

        // 3) hold（CSI 仍活跃时允许短时补盲）
        let can_hold = (csi_prior.is_some()
            || csi.fresh
            || previous_motion_state.as_ref().is_some_and(|state| {
                state.body_presence_conf >= MOTION_STATE_PRESENCE_MIN_FOR_HOLD
            }))
            && inner.last_update_at.is_some_and(|t| {
                now.duration_since(t) <= Duration::from_millis(cfg.operator_hold_ms)
            })
            && inner.estimate.source != OperatorSource::None;
        if can_hold {
            let active_stereo_track_id = resolve_active_stereo_track_id(
                Some(&inner.estimate),
                stereo_pose.as_ref(),
                previous_selected_track_id.as_deref(),
            );
            let motion_observations = build_motion_state_observations(
                None,
                stereo_pose.as_ref(),
                csi_prior.as_ref(),
                now_edge_time_ns,
            );
            record_motion_state_observations(
                &mut inner.motion_observations,
                motion_observations.clone(),
                now_edge_time_ns,
            );
            record_track_motion_state_observations(
                &mut inner.track_motion_observations,
                active_stereo_track_id.as_deref(),
                &motion_observations,
                now_edge_time_ns,
            );
            let (
                scoped_previous_motion_state,
                scoped_motion_state_history,
                scoped_motion_observations,
            ) = scoped_motion_state_context(&inner, active_stereo_track_id.as_deref());
            let motion_state = optimize_operator_motion_state(
                scoped_previous_motion_state
                    .as_ref()
                    .or(previous_motion_state.as_ref()),
                &scoped_motion_state_history,
                if scoped_motion_observations.is_empty() {
                    &inner.motion_observations
                } else {
                    &scoped_motion_observations
                },
                now_edge_time_ns,
                true,
            );
            let mut est = hold_estimate_with_motion_prior(
                &inner.estimate,
                previous_motion_state.as_ref(),
                &motion_state,
                now_edge_time_ns,
            );
            est.motion_state = motion_state.clone();
            apply_confirmed_wearer_sticky(&mut inner, &mut est, now_edge_time_ns);
            record_motion_state_snapshot(
                &mut inner.motion_state_history,
                &motion_state,
                now_edge_time_ns,
            );
            {
                let OperatorInner {
                    track_motion_states,
                    track_motion_state_histories,
                    ..
                } = &mut *inner;
                record_track_motion_state_snapshot(
                    track_motion_states,
                    track_motion_state_histories,
                    &motion_state,
                    now_edge_time_ns,
                );
            }
            inner.motion_state = motion_state;
            return est;
        }

        // 4) none：清空输出（避免继续输出陈旧姿态）
        record_motion_state_observations(
            &mut inner.motion_observations,
            build_motion_state_observations(None, None, csi_prior.as_ref(), now_edge_time_ns),
            now_edge_time_ns,
        );
        let motion_state = optimize_operator_motion_state(
            previous_motion_state.as_ref(),
            &inner.motion_state_history,
            &inner.motion_observations,
            now_edge_time_ns,
            false,
        );
        let mut est = OperatorEstimate::default();
        est.motion_state = motion_state.clone();
        est.updated_edge_time_ns = now_edge_time_ns;
        apply_confirmed_wearer_sticky(&mut inner, &mut est, now_edge_time_ns);
        record_motion_state_snapshot(
            &mut inner.motion_state_history,
            &motion_state,
            now_edge_time_ns,
        );
        {
            let OperatorInner {
                track_motion_states,
                track_motion_state_histories,
                ..
            } = &mut *inner;
            record_track_motion_state_snapshot(
                track_motion_states,
                track_motion_state_histories,
                &motion_state,
                now_edge_time_ns,
            );
        }
        inner.motion_state = motion_state;
        inner.estimate = est.clone();
        inner.last_update_at = Some(now);
        est
    }

    pub fn snapshot(&self, stale_ms: u64) -> OperatorSnapshot {
        let now = Instant::now();
        let inner = self.inner.read().expect("operator lock poisoned");
        let fresh = inner
            .last_update_at
            .is_some_and(|t| now.duration_since(t) <= Duration::from_millis(stale_ms));
        OperatorSnapshot {
            estimate: inner.estimate.clone(),
            fresh,
        }
    }
}

fn source_pose_from_wifi(
    wifi_pose: &WifiPoseSnapshot,
    wifi_stereo_calibration: Option<&WifiStereoExtrinsic>,
) -> Option<OperatorSourcePose> {
    if !wifi_pose.fresh || wifi_pose.body_kpts_3d.is_empty() {
        return None;
    }

    let body_space = normalize_wifi_body_space(&wifi_pose.body_space, wifi_stereo_calibration);
    let body_kpts_3d = transform_wifi_body_points(
        &wifi_pose.body_kpts_3d,
        &body_space,
        wifi_stereo_calibration,
    );

    let raw_pose = OperatorRawPose {
        source_edge_time_ns: wifi_pose.last_edge_time_ns,
        body_layout: wifi_pose.body_layout,
        hand_layout: HandKeypointLayout::Unknown,
        body_kpts_3d,
        hand_kpts_3d: Vec::new(),
    };
    let canonical_body_kpts_3d = stabilize_wifi_canonical_body_points(
        &canonicalize_body_points_3d(&raw_pose.body_kpts_3d, raw_pose.body_layout),
    );
    if !wifi_body_points_are_plausible(&canonical_body_kpts_3d) {
        return None;
    }

    Some(OperatorSourcePose {
        source: OperatorSource::WifiPose3d,
        raw_pose,
        operator_track_id: wifi_pose.operator_track_id.clone(),
        body_space,
        hand_space: String::new(),
        hand_geometry_trusted: false,
        left_hand_fresh: false,
        right_hand_fresh: false,
        canonical_body_kpts_3d,
        canonical_hand_kpts_3d: Vec::new(),
        left_hand_curls: None,
        right_hand_curls: None,
        body_weight: weighted_source_confidence(wifi_pose.body_confidence, 1.0),
        hand_weight: 0.0,
        wifi_diagnostics: Some(wifi_pose.diagnostics.clone()),
    })
}

fn wifi_body_points_are_plausible(points: &[[f32; 3]]) -> bool {
    let Some((min_point, max_point, present_count, max_abs_coord)) = point_cloud_bounds_3d(points)
    else {
        return false;
    };
    if present_count < WIFI_BODY_MIN_PRESENT_JOINTS {
        return false;
    }

    let span = sub(max_point, min_point);
    let max_axis_span = span[0].max(span[1]).max(span[2]);

    max_abs_coord <= WIFI_BODY_MAX_ABS_COORD_M
        && max_axis_span <= WIFI_BODY_MAX_AXIS_SPAN_M
        && norm(span) <= WIFI_BODY_MAX_DIAGONAL_SPAN_M
}

fn point_cloud_bounds_3d(points: &[[f32; 3]]) -> Option<([f32; 3], [f32; 3], usize, f32)> {
    let mut min_point = [f32::INFINITY; 3];
    let mut max_point = [f32::NEG_INFINITY; 3];
    let mut present_count = 0usize;
    let mut max_abs_coord = 0.0f32;

    for point in points.iter().copied().filter(is_present_point3) {
        present_count += 1;
        for axis in 0..3 {
            min_point[axis] = min_point[axis].min(point[axis]);
            max_point[axis] = max_point[axis].max(point[axis]);
            max_abs_coord = max_abs_coord.max(point[axis].abs());
        }
    }

    (present_count > 0).then_some((min_point, max_point, present_count, max_abs_coord))
}

fn transform_wifi_body_points(
    points: &[[f32; 3]],
    body_space: &str,
    wifi_stereo_calibration: Option<&WifiStereoExtrinsic>,
) -> Vec<[f32; 3]> {
    if is_canonical_body_space(body_space) {
        points.to_vec()
    } else {
        transform_points_3d(points, wifi_stereo_calibration)
    }
}

fn normalize_wifi_body_space(
    body_space: &str,
    wifi_stereo_calibration: Option<&WifiStereoExtrinsic>,
) -> String {
    if body_space.is_empty() || (body_space == OPERATOR_FRAME && wifi_stereo_calibration.is_none())
    {
        CANONICAL_BODY_FRAME.to_string()
    } else {
        body_space.to_string()
    }
}

fn is_canonical_body_space(body_space: &str) -> bool {
    let body_space = if body_space.is_empty() {
        CANONICAL_BODY_FRAME
    } else {
        body_space
    };
    body_space == CANONICAL_BODY_FRAME
}

fn spaces_are_geometry_compatible(lhs: &str, rhs: &str) -> bool {
    let lhs = if lhs.is_empty() {
        CANONICAL_BODY_FRAME
    } else {
        lhs
    };
    let rhs = if rhs.is_empty() {
        CANONICAL_BODY_FRAME
    } else {
        rhs
    };
    lhs == rhs
}

fn body_can_anchor_iphone_hands(body_space: &str) -> bool {
    !is_canonical_body_space(body_space)
}

fn iphone_hand_space(iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>) -> String {
    match iphone_stereo_calibration {
        Some(calibration) if calibration.target_frame == OPERATOR_FRAME => {
            // Legacy iphone-stereo calibrations were solved against stereo body points
            // but labeled as operator_frame. Treat them as stereo_pair_frame so
            // association/fusion compares geometry in the correct shared frame.
            STEREO_PAIR_FRAME.to_string()
        }
        Some(calibration) if !calibration.target_frame.is_empty() => {
            calibration.target_frame.clone()
        }
        _ => OPERATOR_FRAME.to_string(),
    }
}

fn vision_hand_geometry_is_trusted(vision: &VisionSnapshot) -> bool {
    let source = vision.hand_3d_source.trim();
    if source.is_empty() || source == "none" {
        return false;
    }
    !matches!(source, "depth_reprojected" | "edge_depth_reprojected")
}

fn source_pose_from_stereo(stereo: &StereoSnapshot) -> Option<OperatorSourcePose> {
    if !stereo.fresh || (stereo.body_kpts_3d.is_empty() && stereo.hand_kpts_3d.is_empty()) {
        return None;
    }

    let body_space = if stereo.body_space.is_empty() {
        STEREO_PAIR_FRAME.to_string()
    } else {
        stereo.body_space.clone()
    };
    let hand_space = if stereo.hand_space.is_empty() {
        body_space.clone()
    } else {
        stereo.hand_space.clone()
    };

    let raw_pose = OperatorRawPose {
        source_edge_time_ns: stereo.last_edge_time_ns,
        body_layout: stereo.body_layout,
        hand_layout: stereo.hand_layout,
        body_kpts_3d: stereo.body_kpts_3d.clone(),
        hand_kpts_3d: stereo.hand_kpts_3d.clone(),
    };
    let canonical_body_kpts_3d =
        canonicalize_body_points_3d(&raw_pose.body_kpts_3d, raw_pose.body_layout);
    let canonical_hand_kpts_3d =
        canonicalize_hand_points_3d(&raw_pose.hand_kpts_3d, raw_pose.hand_layout);

    Some(OperatorSourcePose {
        source: OperatorSource::Stereo,
        raw_pose,
        operator_track_id: stereo.operator_track_id.clone(),
        body_space,
        hand_space,
        hand_geometry_trusted: true,
        left_hand_fresh: canonical_hand_kpts_3d.get(0).is_some_and(is_present_point3),
        right_hand_fresh: canonical_hand_kpts_3d
            .get(21)
            .is_some_and(is_present_point3),
        left_hand_curls: finger_curls_from_hand_kpts_3d(&canonical_hand_kpts_3d, 0),
        right_hand_curls: finger_curls_from_hand_kpts_3d(&canonical_hand_kpts_3d, 21),
        canonical_body_kpts_3d,
        canonical_hand_kpts_3d,
        body_weight: weighted_source_confidence(stereo.stereo_confidence, BODY_STEREO_WEIGHT_BIAS),
        hand_weight: weighted_source_confidence(stereo.stereo_confidence, HAND_STEREO_WEIGHT_BIAS),
        wifi_diagnostics: None,
    })
}

fn source_pose_from_vision(
    vision: &VisionSnapshot,
    cfg: &Config,
    iphone_stereo_calibration: Option<&IphoneStereoExtrinsic>,
) -> Option<OperatorSourcePose> {
    if !cfg.phone_vision_processing_enabled {
        return None;
    }
    if !vision.fresh {
        return None;
    }
    let body_space = iphone_hand_space(iphone_stereo_calibration);
    let hand_space = body_space.clone();
    let hand_geometry_trusted = vision_hand_geometry_is_trusted(vision);

    if !vision.body_kpts_3d.is_empty() || !vision.hand_kpts_3d.is_empty() {
        let body_kpts_3d = transform_points_3d(&vision.body_kpts_3d, iphone_stereo_calibration);
        let hand_kpts_3d = transform_points_3d(&vision.hand_kpts_3d, iphone_stereo_calibration);
        let canonical_body_kpts_3d = canonicalize_body_points_3d(&body_kpts_3d, vision.body_layout);
        let canonical_hand_kpts_3d = canonicalize_hand_points_3d(&hand_kpts_3d, vision.hand_layout);
        if canonical_body_kpts_3d.iter().any(is_present_point3)
            || canonical_hand_kpts_3d.iter().any(is_present_point3)
        {
            let left_hand_fresh = vision.left_hand_fresh_3d
                || canonical_hand_has_points(&canonical_hand_kpts_3d, true);
            let right_hand_fresh = vision.right_hand_fresh_3d
                || canonical_hand_has_points(&canonical_hand_kpts_3d, false);
            return Some(OperatorSourcePose {
                source: OperatorSource::Vision3d,
                raw_pose: OperatorRawPose {
                    source_edge_time_ns: vision.last_edge_time_ns,
                    body_layout: vision.body_layout,
                    hand_layout: vision.hand_layout,
                    body_kpts_3d,
                    hand_kpts_3d,
                },
                operator_track_id: vision.operator_track_id.clone(),
                body_space,
                hand_space,
                hand_geometry_trusted,
                left_hand_fresh,
                right_hand_fresh,
                left_hand_curls: finger_curls_from_hand_kpts_3d(&canonical_hand_kpts_3d, 0),
                right_hand_curls: finger_curls_from_hand_kpts_3d(&canonical_hand_kpts_3d, 21),
                canonical_body_kpts_3d,
                canonical_hand_kpts_3d,
                body_weight: weighted_source_confidence(vision.body_conf, 1.0),
                hand_weight: weighted_source_confidence(vision.hand_conf, 1.0),
                wifi_diagnostics: None,
            });
        }
    }

    let hand_2d_coverage = point_coverage_2d(&vision.hand_kpts_2d);
    if hand_2d_coverage <= 0.0 {
        return None;
    }

    let hand3d = project_2d_kpts_to_3d(
        &vision.hand_kpts_2d,
        vision.image_w,
        vision.image_h,
        vision.depth_z_mean_m,
        cfg,
    );
    let hand_kpts_3d = transform_points_3d(&hand3d, iphone_stereo_calibration);
    let canonical_hand_kpts_3d = canonicalize_hand_points_3d(&hand_kpts_3d, vision.hand_layout);
    if !canonical_hand_kpts_3d.iter().any(is_present_point3) {
        return None;
    }
    let raw_pose = OperatorRawPose {
        source_edge_time_ns: vision.last_edge_time_ns,
        body_layout: BodyKeypointLayout::Unknown,
        hand_layout: vision.hand_layout,
        body_kpts_3d: Vec::new(),
        hand_kpts_3d,
    };
    let hand2d_norm = normalize_2d_kpts(&vision.hand_kpts_2d, vision.image_w, vision.image_h);
    let hand2d_canonical = canonicalize_hand_points_2d(&hand2d_norm, vision.hand_layout);
    let left_hand_fresh =
        vision.left_hand_fresh_3d || canonical_hand_has_points(&canonical_hand_kpts_3d, true);
    let right_hand_fresh =
        vision.right_hand_fresh_3d || canonical_hand_has_points(&canonical_hand_kpts_3d, false);

    Some(OperatorSourcePose {
        source: OperatorSource::Vision2dProjected,
        raw_pose,
        operator_track_id: vision.operator_track_id.clone(),
        body_space: String::new(),
        hand_space,
        hand_geometry_trusted: false,
        left_hand_fresh,
        right_hand_fresh,
        left_hand_curls: finger_curls_from_hand_kpts_2d(&hand2d_canonical, 0),
        right_hand_curls: finger_curls_from_hand_kpts_2d(&hand2d_canonical, 21),
        canonical_body_kpts_3d: Vec::new(),
        canonical_hand_kpts_3d,
        body_weight: 0.0,
        hand_weight: weighted_source_confidence(vision.hand_conf, PROJECTED_VISION_WEIGHT_SCALE)
            * hand_2d_coverage,
        wifi_diagnostics: None,
    })
}

fn weighted_source_confidence(conf: f32, scale: f32) -> f32 {
    (conf.clamp(0.0, 1.0) * scale).max(MIN_SOURCE_WEIGHT)
}

fn derive_csi_prior_observation(
    wifi: Option<&OperatorSourcePose>,
    csi: &CsiSnapshot,
    now_edge_time_ns: u64,
) -> Option<CsiPriorObservation> {
    let wifi_diag = wifi.and_then(|source| source.wifi_diagnostics.as_ref());
    let layout_score = wifi_diag.map(|diag| diag.layout_score).unwrap_or(0.0);
    let signal_quality = wifi_diag.map(|diag| diag.signal_quality).unwrap_or(0.0);
    let zone_score = wifi_diag.map(|diag| diag.zone_score).unwrap_or(0.0);
    let zone_summary_reliable = wifi_diag
        .map(|diag| diag.zone_summary_reliable)
        .unwrap_or(false);
    let motion_energy = wifi_diag.map(|diag| diag.motion_energy).unwrap_or(0.0);
    let doppler_hz = wifi_diag.map(|diag| diag.doppler_hz).unwrap_or(0.0);

    let mut reliability = if csi.fresh {
        (csi.csi_conf.clamp(0.0, 1.0) * 0.45
            + layout_score.clamp(0.0, 1.0) * 0.25
            + signal_quality.clamp(0.0, 1.0) * 0.15
            + zone_score.clamp(0.0, 1.0) * if zone_summary_reliable { 0.15 } else { 0.05 })
        .clamp(0.0, 1.0)
    } else if wifi_diag.is_some() {
        (layout_score.clamp(0.0, 1.0) * 0.5
            + signal_quality.clamp(0.0, 1.0) * 0.3
            + zone_score.clamp(0.0, 1.0) * if zone_summary_reliable { 0.2 } else { 0.0 })
        .clamp(0.0, 1.0)
    } else {
        0.0
    };

    let root_zone_center_m = wifi.and_then(|source| {
        (!is_canonical_body_space(&source.body_space))
            .then(|| estimate_body_root(&source.canonical_body_kpts_3d))
            .flatten()
    });
    let heading_yaw_rad = wifi.and_then(|source| {
        (!is_canonical_body_space(&source.body_space))
            .then(|| estimate_body_heading_yaw(&source.canonical_body_kpts_3d))
            .flatten()
    });

    if root_zone_center_m.is_none() && heading_yaw_rad.is_none() {
        reliability *= 0.75;
    }
    if reliability <= 0.05 {
        return None;
    }

    Some(CsiPriorObservation {
        edge_time_ns: now_edge_time_ns,
        operator_track_id: wifi.and_then(|source| source.operator_track_id.clone()),
        reliability,
        motion_phase: compute_motion_phase(motion_energy, doppler_hz),
        root_zone_center_m,
        heading_yaw_rad,
    })
}

fn build_motion_state_observations(
    estimate: Option<&OperatorEstimate>,
    stereo: Option<&OperatorSourcePose>,
    csi_prior: Option<&CsiPriorObservation>,
    now_edge_time_ns: u64,
) -> Vec<MotionStateObservation> {
    let mut observations = Vec::with_capacity(2);
    let geometry_measurement = estimate
        .filter(|current| {
            current.source != OperatorSource::Hold
                && !current.operator_state.body_kpts_3d.is_empty()
                && current.fusion_breakdown.body_source != OperatorPartSource::None
        })
        .and_then(|current| {
            Some((
                estimate_body_root(&current.operator_state.body_kpts_3d)?,
                estimate_body_heading_yaw(&current.operator_state.body_kpts_3d),
                current
                    .association
                    .selected_operator_track_id
                    .clone()
                    .or_else(|| current.association.stereo_operator_track_id.clone()),
            ))
        })
        .or_else(|| {
            stereo.and_then(|source| {
                Some((
                    estimate_body_root(&source.canonical_body_kpts_3d)?,
                    estimate_body_heading_yaw(&source.canonical_body_kpts_3d),
                    source.operator_track_id.clone(),
                ))
            })
        });

    if let Some((root_pos_m, heading_yaw_rad, track_id)) = geometry_measurement {
        observations.push(MotionStateObservation {
            edge_time_ns: now_edge_time_ns,
            kind: MotionObservationKind::StereoGeometry,
            root_pos_m: Some(root_pos_m),
            heading_yaw_rad,
            reliability: 1.0,
            motion_phase: csi_prior.map(|prior| prior.motion_phase).unwrap_or(0.0),
            track_id,
        });
    }

    if let Some(prior) = csi_prior {
        if prior.root_zone_center_m.is_some() || prior.heading_yaw_rad.is_some() {
            observations.push(MotionStateObservation {
                edge_time_ns: prior.edge_time_ns,
                kind: MotionObservationKind::CsiPrior,
                root_pos_m: prior.root_zone_center_m,
                heading_yaw_rad: prior.heading_yaw_rad,
                reliability: prior.reliability.clamp(0.0, 1.0),
                motion_phase: prior.motion_phase,
                track_id: prior.operator_track_id.clone(),
            });
        }
    }

    observations
}

fn record_motion_state_observations(
    history: &mut VecDeque<MotionStateObservation>,
    observations: Vec<MotionStateObservation>,
    now_edge_time_ns: u64,
) {
    for observation in observations {
        history.push_back(observation);
    }
    let window_start_ns = now_edge_time_ns.saturating_sub(MOTION_STATE_HISTORY_RETENTION_NS);
    while history
        .front()
        .is_some_and(|observation| observation.edge_time_ns < window_start_ns)
    {
        history.pop_front();
    }
}

fn record_track_motion_state_observations(
    track_histories: &mut HashMap<String, VecDeque<MotionStateObservation>>,
    active_track_id: Option<&str>,
    observations: &[MotionStateObservation],
    now_edge_time_ns: u64,
) {
    let Some(track_id) = active_track_id else {
        prune_track_motion_observation_cache(track_histories, now_edge_time_ns);
        return;
    };
    let history = track_histories.entry(track_id.to_string()).or_default();
    for observation in observations {
        history.push_back(observation.clone());
    }
    let window_start_ns = now_edge_time_ns.saturating_sub(MOTION_STATE_HISTORY_RETENTION_NS);
    while history
        .front()
        .is_some_and(|observation| observation.edge_time_ns < window_start_ns)
    {
        history.pop_front();
    }
    prune_track_motion_observation_cache(track_histories, now_edge_time_ns);
}

fn prune_track_motion_observation_cache(
    track_histories: &mut HashMap<String, VecDeque<MotionStateObservation>>,
    now_edge_time_ns: u64,
) {
    let cutoff_ns = now_edge_time_ns.saturating_sub(MOTION_STATE_HISTORY_RETENTION_NS);
    track_histories.retain(|_, history| {
        while history
            .front()
            .is_some_and(|observation| observation.edge_time_ns < cutoff_ns)
        {
            history.pop_front();
        }
        !history.is_empty()
    });
}

fn record_motion_state_snapshot(
    history: &mut VecDeque<OperatorMotionState>,
    state: &OperatorMotionState,
    now_edge_time_ns: u64,
) {
    history.push_back(state.clone());
    let window_start_ns = now_edge_time_ns.saturating_sub(MOTION_STATE_STATE_HISTORY_RETENTION_NS);
    while history
        .front()
        .is_some_and(|snapshot| snapshot.updated_edge_time_ns < window_start_ns)
    {
        history.pop_front();
    }
}

fn scoped_motion_state_context(
    inner: &OperatorInner,
    active_track_id: Option<&str>,
) -> (
    Option<OperatorMotionState>,
    VecDeque<OperatorMotionState>,
    VecDeque<MotionStateObservation>,
) {
    let scoped_previous = active_track_id
        .and_then(|track_id| inner.track_motion_states.get(track_id))
        .cloned();
    let scoped_state_history = active_track_id
        .and_then(|track_id| inner.track_motion_state_histories.get(track_id))
        .cloned()
        .unwrap_or_default();
    let scoped_observation_history = active_track_id
        .and_then(|track_id| inner.track_motion_observations.get(track_id))
        .cloned()
        .unwrap_or_default();
    (
        scoped_previous,
        scoped_state_history,
        scoped_observation_history,
    )
}

fn resolve_active_stereo_track_id(
    estimate: Option<&OperatorEstimate>,
    stereo: Option<&OperatorSourcePose>,
    previous_selected_track_id: Option<&str>,
) -> Option<String> {
    estimate
        .and_then(|current| current.association.stereo_operator_track_id.clone())
        .or_else(|| stereo.and_then(|source| source.operator_track_id.clone()))
        .or_else(|| previous_selected_track_id.map(|track_id| track_id.to_string()))
}

fn record_track_motion_state_snapshot(
    track_states: &mut HashMap<String, OperatorMotionState>,
    track_histories: &mut HashMap<String, VecDeque<OperatorMotionState>>,
    state: &OperatorMotionState,
    now_edge_time_ns: u64,
) {
    let Some(track_id) = state.stereo_track_id.clone() else {
        prune_track_motion_state_cache(track_states, track_histories, now_edge_time_ns);
        return;
    };
    track_states.insert(track_id.clone(), state.clone());
    let history = track_histories.entry(track_id).or_default();
    record_motion_state_snapshot(history, state, now_edge_time_ns);
    prune_track_motion_state_cache(track_states, track_histories, now_edge_time_ns);
}

fn prune_track_motion_state_cache(
    track_states: &mut HashMap<String, OperatorMotionState>,
    track_histories: &mut HashMap<String, VecDeque<OperatorMotionState>>,
    now_edge_time_ns: u64,
) {
    let cutoff_ns = now_edge_time_ns.saturating_sub(MOTION_STATE_STATE_HISTORY_RETENTION_NS);
    track_states.retain(|_, state| state.updated_edge_time_ns >= cutoff_ns);
    track_histories.retain(|track_id, history| {
        while history
            .front()
            .is_some_and(|snapshot| snapshot.updated_edge_time_ns < cutoff_ns)
        {
            history.pop_front();
        }
        !history.is_empty() || track_states.contains_key(track_id)
    });
}

fn observation_recency_weight(edge_time_ns: u64, now_edge_time_ns: u64) -> f32 {
    let age_ns = now_edge_time_ns.saturating_sub(edge_time_ns);
    if age_ns >= MOTION_STATE_FIXED_LAG_NS {
        return 0.0;
    }
    let age_ratio = age_ns as f32 / MOTION_STATE_FIXED_LAG_NS as f32;
    (1.0 - age_ratio).clamp(0.1, 1.0)
}

fn sticky_iphone_anchor_source(anchor_source: &'static str) -> &'static str {
    match anchor_source {
        "wifi+stereo" => "wifi+stereo+iphone_hand",
        "stereo+wifi_prior" => "stereo+iphone_hand+wifi_prior",
        "stereo" => "stereo+iphone_hand",
        "wifi" => "wifi+iphone_hand",
        "wifi_prior" => "iphone_hand+wifi_prior",
        "none" => "iphone_hand",
        other => other,
    }
}

fn motion_observation_std(
    observation: &MotionStateObservation,
    recency_weight: f32,
) -> (f32, f32, f32, f32) {
    let (base_root_std, base_heading_std, root_gate_sigma, heading_gate_sigma) =
        match observation.kind {
            MotionObservationKind::StereoGeometry => (
                MOTION_STATE_STEREO_ROOT_STD_M,
                MOTION_STATE_STEREO_HEADING_STD_RAD,
                MOTION_STATE_STEREO_ROOT_GATE_SIGMA,
                MOTION_STATE_STEREO_HEADING_GATE_SIGMA,
            ),
            MotionObservationKind::CsiPrior => (
                MOTION_STATE_CSI_ROOT_STD_M,
                MOTION_STATE_CSI_HEADING_STD_RAD,
                MOTION_STATE_CSI_ROOT_GATE_SIGMA,
                MOTION_STATE_CSI_HEADING_GATE_SIGMA,
            ),
        };
    let effective_reliability = (observation.reliability * recency_weight).clamp(0.05, 1.0);
    let reliability_scale = effective_reliability.sqrt();
    (
        (base_root_std / reliability_scale).clamp(base_root_std, MOTION_STATE_MAX_ROOT_STD_M),
        (base_heading_std / reliability_scale)
            .clamp(base_heading_std, MOTION_STATE_MAX_HEADING_STD_RAD),
        root_gate_sigma,
        heading_gate_sigma,
    )
}

fn propagate_motion_state(state: &mut OperatorMotionState, target_edge_time_ns: u64) {
    if target_edge_time_ns <= state.updated_edge_time_ns {
        return;
    }
    let dt_s = (target_edge_time_ns.saturating_sub(state.updated_edge_time_ns) as f32 * 1e-9)
        .clamp(0.0, 0.5);
    if dt_s <= f32::EPSILON {
        state.updated_edge_time_ns = target_edge_time_ns;
        return;
    }
    state.root_pos_m = add3(state.root_pos_m, scale3(state.root_vel_mps, dt_s));
    state.heading_yaw_rad = wrap_angle_rad(state.heading_yaw_rad + state.heading_rate_radps * dt_s);
    state.root_vel_mps = scale3(state.root_vel_mps, MOTION_STATE_VELOCITY_DAMPING);
    state.heading_rate_radps *= MOTION_STATE_VELOCITY_DAMPING;
    state.root_std_m = ((state.root_std_m * state.root_std_m)
        + (MOTION_STATE_ROOT_PROCESS_NOISE_MPS * dt_s).powi(2))
    .sqrt()
    .clamp(MOTION_STATE_MIN_ROOT_STD_M, MOTION_STATE_MAX_ROOT_STD_M);
    state.heading_std_rad = ((state.heading_std_rad * state.heading_std_rad)
        + (MOTION_STATE_HEADING_PROCESS_NOISE_RADPS * dt_s).powi(2))
    .sqrt()
    .clamp(
        MOTION_STATE_MIN_HEADING_STD_RAD,
        MOTION_STATE_MAX_HEADING_STD_RAD,
    );
    state.updated_edge_time_ns = target_edge_time_ns;
}

fn kalman_gain(prior_std: f32, measurement_std: f32) -> f32 {
    let prior_var = (prior_std * prior_std).max(1e-6);
    let measurement_var = (measurement_std * measurement_std).max(1e-6);
    (prior_var / (prior_var + measurement_var)).clamp(0.0, 0.95)
}

fn root_observation_accepted(
    state: &OperatorMotionState,
    root_pos_m: [f32; 3],
    measurement_std: f32,
    gate_sigma: f32,
) -> bool {
    if state.body_presence_conf <= f32::EPSILON || state.updated_edge_time_ns == 0 {
        return true;
    }
    let innovation = dist3(root_pos_m, state.root_pos_m);
    let combined_std = (state.root_std_m * state.root_std_m + measurement_std * measurement_std)
        .sqrt()
        .max(MOTION_STATE_MIN_ROOT_STD_M);
    innovation <= (gate_sigma * combined_std * 1.732_050_8).max(0.08)
}

fn heading_observation_accepted(
    state: &OperatorMotionState,
    heading_yaw_rad: f32,
    measurement_std: f32,
    gate_sigma: f32,
) -> bool {
    if state.body_presence_conf <= f32::EPSILON || state.updated_edge_time_ns == 0 {
        return true;
    }
    let innovation = wrap_angle_rad(heading_yaw_rad - state.heading_yaw_rad).abs();
    let combined_std = (state.heading_std_rad * state.heading_std_rad
        + measurement_std * measurement_std)
        .sqrt()
        .max(MOTION_STATE_MIN_HEADING_STD_RAD);
    innovation <= (gate_sigma * combined_std).max(0.12)
}

fn apply_root_measurement(
    state: &mut OperatorMotionState,
    root_pos_m: [f32; 3],
    measurement_std: f32,
) {
    let gain = kalman_gain(state.root_std_m, measurement_std);
    let innovation = sub(root_pos_m, state.root_pos_m);
    state.root_pos_m = add3(state.root_pos_m, scale3(innovation, gain));
    state.root_std_m = (((1.0 - gain).max(0.05)) * state.root_std_m * state.root_std_m)
        .sqrt()
        .clamp(MOTION_STATE_MIN_ROOT_STD_M, MOTION_STATE_MAX_ROOT_STD_M);
}

fn apply_heading_measurement(
    state: &mut OperatorMotionState,
    heading_yaw_rad: f32,
    measurement_std: f32,
) {
    let gain = kalman_gain(state.heading_std_rad, measurement_std);
    let innovation = wrap_angle_rad(heading_yaw_rad - state.heading_yaw_rad);
    state.heading_yaw_rad = wrap_angle_rad(state.heading_yaw_rad + innovation * gain);
    state.heading_std_rad =
        (((1.0 - gain).max(0.05)) * state.heading_std_rad * state.heading_std_rad)
            .sqrt()
            .clamp(
                MOTION_STATE_MIN_HEADING_STD_RAD,
                MOTION_STATE_MAX_HEADING_STD_RAD,
            );
}

fn optimize_operator_motion_state(
    previous: Option<&OperatorMotionState>,
    state_history: &VecDeque<OperatorMotionState>,
    history: &VecDeque<MotionStateObservation>,
    now_edge_time_ns: u64,
    hold_active: bool,
) -> OperatorMotionState {
    let window_start_ns = now_edge_time_ns.saturating_sub(MOTION_STATE_FIXED_LAG_NS);
    let mut recent_observations: Vec<MotionStateObservation> = history
        .iter()
        .filter(|observation| observation.edge_time_ns >= window_start_ns)
        .cloned()
        .collect();
    recent_observations.sort_by_key(|observation| observation.edge_time_ns);
    let replay_anchor_time_ns = recent_observations
        .first()
        .map(|observation| observation.edge_time_ns)
        .unwrap_or(window_start_ns);
    let desired_stereo_track_id = recent_observations
        .iter()
        .rev()
        .find(|observation| observation.kind == MotionObservationKind::StereoGeometry)
        .and_then(|observation| observation.track_id.clone());
    let mut state = select_motion_state_anchor(
        previous,
        state_history,
        replay_anchor_time_ns,
        desired_stereo_track_id.as_deref(),
    );
    if state.updated_edge_time_ns == 0 {
        state.updated_edge_time_ns = replay_anchor_time_ns;
    } else if state.updated_edge_time_ns < replay_anchor_time_ns {
        propagate_motion_state(&mut state, replay_anchor_time_ns);
    }
    if desired_stereo_track_id.is_some()
        && state.stereo_track_id.is_some()
        && state.stereo_track_id.as_deref() != desired_stereo_track_id.as_deref()
    {
        state = reset_motion_state_for_track_change(&state, replay_anchor_time_ns);
    }

    let output_reference = previous.cloned().unwrap_or_else(|| state.clone());
    let previous_root = output_reference.root_pos_m;
    let previous_heading = output_reference.heading_yaw_rad;
    let predicted_presence_conf = previous
        .map(|prev| prev.body_presence_conf * MOTION_STATE_PRESENCE_DECAY)
        .unwrap_or(0.0);

    let mut has_current_stereo = false;
    let mut accepted_stereo = 0u32;
    let mut accepted_csi = 0u32;
    let mut rejected_stereo = 0u32;
    let mut rejected_csi = 0u32;
    let mut latest_stereo_time_ns = previous
        .map(|prev| prev.last_good_stereo_time_ns)
        .unwrap_or(0);
    let mut latest_csi_time_ns = previous.map(|prev| prev.last_good_csi_time_ns).unwrap_or(0);
    let mut latest_stereo_track_id = previous.and_then(|prev| prev.stereo_track_id.clone());
    let mut max_csi_reliability = 0.0f32;

    for observation in recent_observations.iter() {
        if observation.edge_time_ns > state.updated_edge_time_ns {
            propagate_motion_state(&mut state, observation.edge_time_ns);
        }
        let recency_weight = observation_recency_weight(observation.edge_time_ns, now_edge_time_ns);
        if recency_weight <= f32::EPSILON {
            continue;
        }
        let (root_std, heading_std, root_gate_sigma, heading_gate_sigma) =
            motion_observation_std(observation, recency_weight);
        let mut accepted_observation = false;
        if let Some(root_pos_m) = observation.root_pos_m {
            if root_observation_accepted(&state, root_pos_m, root_std, root_gate_sigma) {
                apply_root_measurement(&mut state, root_pos_m, root_std);
                accepted_observation = true;
            } else {
                match observation.kind {
                    MotionObservationKind::StereoGeometry => rejected_stereo += 1,
                    MotionObservationKind::CsiPrior => rejected_csi += 1,
                }
            }
        }
        if let Some(heading_yaw_rad) = observation.heading_yaw_rad {
            if heading_observation_accepted(
                &state,
                heading_yaw_rad,
                heading_std,
                heading_gate_sigma,
            ) {
                apply_heading_measurement(&mut state, heading_yaw_rad, heading_std);
                accepted_observation = true;
            } else {
                match observation.kind {
                    MotionObservationKind::StereoGeometry => rejected_stereo += 1,
                    MotionObservationKind::CsiPrior => rejected_csi += 1,
                }
            }
        }
        let phase_gain = (observation.reliability * recency_weight * 0.55).clamp(0.05, 0.75);
        state.motion_phase =
            state.motion_phase + (observation.motion_phase - state.motion_phase) * phase_gain;

        if accepted_observation {
            match observation.kind {
                MotionObservationKind::StereoGeometry => {
                    accepted_stereo += 1;
                    latest_stereo_time_ns = observation.edge_time_ns;
                    latest_stereo_track_id = observation
                        .track_id
                        .clone()
                        .or_else(|| latest_stereo_track_id.clone());
                    has_current_stereo = now_edge_time_ns.saturating_sub(observation.edge_time_ns)
                        <= MOTION_STATE_STEREO_LIVE_MAX_AGE_NS;
                }
                MotionObservationKind::CsiPrior => {
                    accepted_csi += 1;
                    latest_csi_time_ns = observation.edge_time_ns;
                    max_csi_reliability = max_csi_reliability.max(observation.reliability);
                }
            }
        } else if observation.kind == MotionObservationKind::CsiPrior {
            max_csi_reliability = max_csi_reliability.max(observation.reliability * 0.85);
        }
    }

    propagate_motion_state(&mut state, now_edge_time_ns);

    let dt_s = previous
        .map(|prev| now_edge_time_ns.saturating_sub(prev.updated_edge_time_ns) as f32 * 1e-9)
        .unwrap_or(0.0)
        .clamp(0.0, 0.5);
    if dt_s > 1e-4 {
        state.root_vel_mps = scale3(sub(state.root_pos_m, previous_root), 1.0 / dt_s);
        state.heading_rate_radps = wrap_angle_rad(state.heading_yaw_rad - previous_heading) / dt_s;
    }

    state.stereo_measurement_used = accepted_stereo > 0;
    state.csi_measurement_used = accepted_csi > 0;
    state.csi_prior_reliability = max_csi_reliability;
    state.accepted_stereo_observations = accepted_stereo;
    state.accepted_csi_observations = accepted_csi;
    state.rejected_stereo_observations = rejected_stereo;
    state.rejected_csi_observations = rejected_csi;
    state.body_presence_conf = if has_current_stereo {
        0.95
    } else if accepted_stereo > 0 {
        predicted_presence_conf.max(0.72)
    } else if accepted_csi > 0 {
        predicted_presence_conf.max((max_csi_reliability * 0.8).clamp(0.0, 0.82))
    } else {
        predicted_presence_conf
    };
    state.wearer_confidence = previous
        .map(|prev| prev.wearer_confidence * 0.96)
        .unwrap_or(0.0);
    state.last_good_stereo_time_ns = latest_stereo_time_ns;
    state.last_good_csi_time_ns = latest_csi_time_ns;
    state.stereo_track_id = latest_stereo_track_id
        .or_else(|| previous.and_then(|prev| prev.stereo_track_id.clone()))
        .or_else(|| {
            recent_observations
                .iter()
                .rev()
                .find_map(|observation| observation.track_id.clone())
        });
    state.smoother_mode = if has_current_stereo {
        OperatorSmootherMode::StereoLive
    } else if accepted_stereo > 0 && accepted_csi > 0 {
        OperatorSmootherMode::HeldWithCsiPrior
    } else if accepted_stereo > 0 || hold_active {
        OperatorSmootherMode::FixedLagBlend
    } else if accepted_csi > 0 {
        OperatorSmootherMode::CsiPriorOnly
    } else {
        OperatorSmootherMode::Degraded
    };
    state
}

fn select_motion_state_anchor(
    previous: Option<&OperatorMotionState>,
    state_history: &VecDeque<OperatorMotionState>,
    replay_anchor_time_ns: u64,
    desired_stereo_track_id: Option<&str>,
) -> OperatorMotionState {
    state_history
        .iter()
        .rev()
        .find(|snapshot| {
            snapshot.updated_edge_time_ns <= replay_anchor_time_ns
                && track_matches_anchor(snapshot, desired_stereo_track_id)
        })
        .cloned()
        .or_else(|| {
            previous
                .filter(|snapshot| track_matches_anchor(snapshot, desired_stereo_track_id))
                .cloned()
        })
        .or_else(|| {
            state_history
                .iter()
                .rev()
                .find(|snapshot| snapshot.updated_edge_time_ns <= replay_anchor_time_ns)
                .cloned()
        })
        .or_else(|| previous.cloned())
        .unwrap_or_default()
}

fn track_matches_anchor(
    snapshot: &OperatorMotionState,
    desired_stereo_track_id: Option<&str>,
) -> bool {
    match desired_stereo_track_id {
        Some(track_id) => snapshot.stereo_track_id.as_deref() == Some(track_id),
        None => true,
    }
}

fn reset_motion_state_for_track_change(
    anchor: &OperatorMotionState,
    replay_anchor_time_ns: u64,
) -> OperatorMotionState {
    let mut reset = OperatorMotionState {
        updated_edge_time_ns: replay_anchor_time_ns,
        motion_phase: anchor.motion_phase,
        csi_prior_reliability: anchor.csi_prior_reliability * 0.5,
        wearer_confidence: anchor.wearer_confidence * 0.5,
        ..OperatorMotionState::default()
    };
    reset.root_std_m = (anchor.root_std_m * 1.8).clamp(
        MOTION_STATE_PREDICTED_ROOT_STD_M,
        MOTION_STATE_MAX_ROOT_STD_M,
    );
    reset.heading_std_rad = (anchor.heading_std_rad * 1.8).clamp(
        MOTION_STATE_PREDICTED_HEADING_STD_RAD,
        MOTION_STATE_MAX_HEADING_STD_RAD,
    );
    reset
}

#[cfg(test)]
fn update_operator_motion_state(
    previous: Option<&OperatorMotionState>,
    state_history: &VecDeque<OperatorMotionState>,
    estimate: Option<&OperatorEstimate>,
    stereo: Option<&OperatorSourcePose>,
    csi_prior: Option<&CsiPriorObservation>,
    now_edge_time_ns: u64,
) -> OperatorMotionState {
    let observations =
        build_motion_state_observations(estimate, stereo, csi_prior, now_edge_time_ns);
    let mut history = VecDeque::new();
    record_motion_state_observations(&mut history, observations, now_edge_time_ns);
    let mut state = optimize_operator_motion_state(
        previous,
        state_history,
        &history,
        now_edge_time_ns,
        estimate.is_some_and(|current| current.source == OperatorSource::Hold),
    );
    state.wearer_confidence = estimate
        .map(|current| current.association.hand_match_score.clamp(0.0, 1.0))
        .unwrap_or_else(|| {
            previous
                .map(|prev| prev.wearer_confidence * 0.96)
                .unwrap_or(0.0)
        });
    state
}

fn apply_live_motion_state_smoothing(
    mut estimate: OperatorEstimate,
    motion_state: &OperatorMotionState,
) -> OperatorEstimate {
    if estimate.operator_state.body_kpts_3d.is_empty() {
        return estimate;
    }
    let Some(current_root) = estimate_body_root(&estimate.operator_state.body_kpts_3d) else {
        return estimate;
    };
    let delta = sub(motion_state.root_pos_m, current_root);
    let delta_norm = norm(delta);
    if !delta_norm.is_finite() || delta_norm <= 1e-5 {
        return estimate;
    }
    let limited_delta = if delta_norm > MOTION_STATE_LIVE_ROOT_MAX_OFFSET_M {
        scale3(delta, MOTION_STATE_LIVE_ROOT_MAX_OFFSET_M / delta_norm)
    } else {
        delta
    };
    apply_translation_to_estimate(&mut estimate, limited_delta);
    estimate
}

fn hold_estimate_with_motion_prior(
    previous_estimate: &OperatorEstimate,
    previous_motion_state: Option<&OperatorMotionState>,
    motion_state: &OperatorMotionState,
    now_edge_time_ns: u64,
) -> OperatorEstimate {
    let mut estimate = previous_estimate.clone();
    estimate.source = OperatorSource::Hold;
    estimate.updated_edge_time_ns = now_edge_time_ns;

    let translation = previous_motion_state
        .map(|previous| sub(motion_state.root_pos_m, previous.root_pos_m))
        .filter(|delta| norm(*delta).is_finite())
        .unwrap_or([0.0, 0.0, 0.0]);
    let translation_norm = norm(translation);
    if translation_norm > 1e-5 {
        let limited = if translation_norm > MOTION_STATE_HOLD_TRANSLATION_MAX_M {
            scale3(
                translation,
                MOTION_STATE_HOLD_TRANSLATION_MAX_M / translation_norm,
            )
        } else {
            translation
        };
        apply_translation_to_estimate(&mut estimate, limited);
    }

    estimate
}

fn apply_translation_to_estimate(estimate: &mut OperatorEstimate, delta: [f32; 3]) {
    if !delta[0].is_finite() || !delta[1].is_finite() || !delta[2].is_finite() {
        return;
    }
    translate_points_in_place(&mut estimate.operator_state.body_kpts_3d, delta);
    translate_points_in_place(&mut estimate.operator_state.hand_kpts_3d, delta);
    translate_points_in_place(&mut estimate.raw_pose.body_kpts_3d, delta);
    translate_points_in_place(&mut estimate.raw_pose.hand_kpts_3d, delta);
    estimate.operator_state.end_effector_pose.left.pos =
        add3(estimate.operator_state.end_effector_pose.left.pos, delta);
    estimate.operator_state.end_effector_pose.right.pos =
        add3(estimate.operator_state.end_effector_pose.right.pos, delta);
}

fn translate_points_in_place(points: &mut [[f32; 3]], delta: [f32; 3]) {
    for point in points.iter_mut().filter(|point| is_present_point3(point)) {
        *point = add3(*point, delta);
    }
}

fn estimate_body_root(points: &[[f32; 3]]) -> Option<[f32; 3]> {
    let mut anchors = Vec::new();
    if let (Some(left_hip), Some(right_hip)) = (
        points
            .get(COCO_LEFT_HIP_INDEX)
            .copied()
            .filter(is_present_point3),
        points
            .get(COCO_RIGHT_HIP_INDEX)
            .copied()
            .filter(is_present_point3),
    ) {
        anchors.push(midpoint3(left_hip, right_hip));
    }
    if let (Some(left_shoulder), Some(right_shoulder)) = (
        points
            .get(COCO_LEFT_SHOULDER_INDEX)
            .copied()
            .filter(is_present_point3),
        points
            .get(COCO_RIGHT_SHOULDER_INDEX)
            .copied()
            .filter(is_present_point3),
    ) {
        anchors.push(midpoint3(left_shoulder, right_shoulder));
    }
    if anchors.is_empty() {
        anchors.extend(points.iter().copied().filter(is_present_point3));
    }
    mean_point3(&anchors)
}

fn estimate_body_heading_yaw(points: &[[f32; 3]]) -> Option<f32> {
    let left = points
        .get(COCO_LEFT_SHOULDER_INDEX)
        .copied()
        .filter(is_present_point3)
        .or_else(|| {
            points
                .get(COCO_LEFT_HIP_INDEX)
                .copied()
                .filter(is_present_point3)
        })?;
    let right = points
        .get(COCO_RIGHT_SHOULDER_INDEX)
        .copied()
        .filter(is_present_point3)
        .or_else(|| {
            points
                .get(COCO_RIGHT_HIP_INDEX)
                .copied()
                .filter(is_present_point3)
        })?;
    let shoulder_center = midpoint3(left, right);
    let across = sub(right, left);
    let across_horizontal = normalize([across[0], 0.0, across[2]]);
    if !is_finite3(across_horizontal) {
        return None;
    }
    let mut forward = normalize(cross([0.0, 1.0, 0.0], across_horizontal));
    if let Some(nose) = points
        .get(COCO_NOSE_INDEX)
        .copied()
        .filter(is_present_point3)
    {
        let nose_dir = normalize([
            nose[0] - shoulder_center[0],
            0.0,
            nose[2] - shoulder_center[2],
        ]);
        if is_finite3(nose_dir) && dot3(forward, nose_dir) < 0.0 {
            forward = scale3(forward, -1.0);
        }
    }
    if !is_finite3(forward) {
        return None;
    }
    Some(forward[0].atan2(forward[2]))
}

fn compute_motion_phase(motion_energy: f32, doppler_hz: f32) -> f32 {
    let energy_phase = (motion_energy / 6.0).clamp(0.0, 1.0);
    let doppler_phase = (doppler_hz.abs() / 1.5).clamp(0.0, 1.0);
    (energy_phase * 0.65 + doppler_phase * 0.35).clamp(0.0, 1.0)
}

fn midpoint3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    scale3(add3(a, b), 0.5)
}

fn mean_point3(points: &[[f32; 3]]) -> Option<[f32; 3]> {
    let mut sum = [0.0, 0.0, 0.0];
    let mut count = 0usize;
    for point in points.iter().copied().filter(is_present_point3) {
        sum = add3(sum, point);
        count += 1;
    }
    (count > 0).then_some(scale3(sum, 1.0 / count as f32))
}

fn wrap_angle_rad(angle: f32) -> f32 {
    let mut angle = angle;
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn point_coverage_2d(points: &[[f32; 2]]) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    let present = points
        .iter()
        .filter(|point| is_present_point2(point))
        .count();
    (present as f32 / points.len() as f32).clamp(0.0, 1.0)
}

fn fuse_operator_sources(
    now_edge_time_ns: u64,
    stereo: Option<&OperatorSourcePose>,
    vision: Option<&OperatorSourcePose>,
    wifi: Option<&OperatorSourcePose>,
) -> Option<OperatorEstimate> {
    match (stereo, vision, wifi) {
        (Some(stereo), Some(vision), Some(wifi)) => Some(build_fused_estimate_with_wifi(
            now_edge_time_ns,
            stereo,
            vision,
            wifi,
        )),
        (Some(stereo), Some(vision), None) => {
            Some(build_fused_estimate(now_edge_time_ns, stereo, vision))
        }
        (Some(stereo), None, Some(wifi)) => Some(build_source_plus_wifi_estimate(
            now_edge_time_ns,
            stereo,
            wifi,
        )),
        (None, Some(vision), Some(wifi)) => Some(build_wifi_body_with_vision_hands_estimate(
            now_edge_time_ns,
            vision,
            wifi,
        )),
        (Some(source), None, None) | (None, Some(source), None) => {
            Some(build_single_source_estimate(now_edge_time_ns, source))
        }
        (None, None, Some(wifi)) => Some(build_wifi_prior_only_estimate(now_edge_time_ns, wifi)),
        (None, None, None) => None,
    }
}

fn build_single_source_estimate(
    now_edge_time_ns: u64,
    source: &OperatorSourcePose,
) -> OperatorEstimate {
    let (operator_state, left_hand_curls, right_hand_curls) = build_from_canonical_3d(
        source.canonical_body_kpts_3d.clone(),
        source.canonical_hand_kpts_3d.clone(),
    );
    let is_stereo = source.source == OperatorSource::Stereo;
    let fusion_breakdown = OperatorFusionBreakdown {
        body_source: if source.canonical_body_kpts_3d.is_empty() {
            OperatorPartSource::None
        } else {
            source.source.as_part_source()
        },
        hand_source: if source.canonical_hand_kpts_3d.is_empty() {
            OperatorPartSource::None
        } else {
            source.source.as_part_source()
        },
        stereo_body_joint_count: if is_stereo {
            source.canonical_body_kpts_3d.len()
        } else {
            0
        },
        vision_body_joint_count: if matches!(
            source.source,
            OperatorSource::Vision3d | OperatorSource::Vision2dProjected
        ) {
            source.canonical_body_kpts_3d.len()
        } else {
            0
        },
        wifi_body_joint_count: if source.source == OperatorSource::WifiPose3d {
            source.canonical_body_kpts_3d.len()
        } else {
            0
        },
        blended_body_joint_count: 0,
        stereo_hand_point_count: if is_stereo {
            source.canonical_hand_kpts_3d.len()
        } else {
            0
        },
        vision_hand_point_count: if matches!(
            source.source,
            OperatorSource::Vision3d | OperatorSource::Vision2dProjected
        ) {
            source.canonical_hand_kpts_3d.len()
        } else {
            0
        },
        wifi_hand_point_count: if source.source == OperatorSource::WifiPose3d {
            source.canonical_hand_kpts_3d.len()
        } else {
            0
        },
        blended_hand_point_count: 0,
    };

    let (stereo_source, vision_source, wifi_source, wifi_diagnostics) = match source.source {
        OperatorSource::Stereo => (Some(source), None, None, None),
        OperatorSource::Vision3d | OperatorSource::Vision2dProjected => {
            (None, Some(source), None, None)
        }
        OperatorSource::WifiPose3d => (None, None, Some(source), source.wifi_diagnostics.as_ref()),
        _ => (Some(source), None, None, None),
    };

    OperatorEstimate {
        source: source.source,
        operator_state,
        raw_pose: source.raw_pose.clone(),
        left_hand_curls: left_hand_curls.or(source.left_hand_curls),
        right_hand_curls: right_hand_curls.or(source.right_hand_curls),
        fusion_breakdown,
        association: derive_operator_association(
            stereo_source,
            vision_source,
            wifi_source,
            wifi_diagnostics,
            &fusion_breakdown,
        ),
        motion_state: OperatorMotionState::default(),
        updated_edge_time_ns: now_edge_time_ns,
    }
}

fn attach_wifi_context(
    mut estimate: OperatorEstimate,
    now_edge_time_ns: u64,
    stereo: Option<&OperatorSourcePose>,
    vision: Option<&OperatorSourcePose>,
    wifi: &OperatorSourcePose,
    extra_vision_hand_points: usize,
) -> OperatorEstimate {
    estimate.fusion_breakdown.vision_hand_point_count = estimate
        .fusion_breakdown
        .vision_hand_point_count
        .max(extra_vision_hand_points);
    estimate.association = derive_operator_association(
        stereo,
        vision,
        Some(wifi),
        wifi.wifi_diagnostics.as_ref(),
        &estimate.fusion_breakdown,
    );
    estimate.updated_edge_time_ns = now_edge_time_ns;
    estimate
}

fn build_wifi_prior_only_estimate(
    now_edge_time_ns: u64,
    wifi: &OperatorSourcePose,
) -> OperatorEstimate {
    let mut estimate = OperatorEstimate {
        source: OperatorSource::None,
        updated_edge_time_ns: now_edge_time_ns,
        ..OperatorEstimate::default()
    };
    estimate.association = derive_operator_association(
        None,
        None,
        Some(wifi),
        wifi.wifi_diagnostics.as_ref(),
        &estimate.fusion_breakdown,
    );
    estimate
}

fn build_wifi_body_with_vision_hands_estimate(
    now_edge_time_ns: u64,
    vision: &OperatorSourcePose,
    wifi: &OperatorSourcePose,
) -> OperatorEstimate {
    let estimate = build_single_source_estimate(now_edge_time_ns, vision);
    attach_wifi_context(
        estimate,
        now_edge_time_ns,
        None,
        Some(vision),
        wifi,
        vision.canonical_hand_kpts_3d.len(),
    )
}

fn build_fused_estimate(
    now_edge_time_ns: u64,
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
) -> OperatorEstimate {
    let body_result = fuse_canonical_points(
        &stereo.canonical_body_kpts_3d,
        &vision.canonical_body_kpts_3d,
        stereo.body_weight,
        vision.body_weight,
        BODY_BLEND_MAX_GAP_M,
    );
    let empty_hand_points: &[[f32; 3]] = &[];
    let can_fuse_vision_hands = vision_hands_are_fusion_safe(stereo, vision);
    let stereo_has_hand_points = stereo.canonical_hand_kpts_3d.iter().any(is_present_point3);
    let allow_vision_hand_fallback = !can_fuse_vision_hands
        && !stereo_has_hand_points
        && vision.canonical_hand_kpts_3d.iter().any(is_present_point3);
    let vision_hand_points = if can_fuse_vision_hands {
        vision.canonical_hand_kpts_3d.as_slice()
    } else if allow_vision_hand_fallback {
        // For sim/telemetry we still want articulated hand targets when stereo body is present
        // but stereo has no hand points at all. In that case global wrist alignment is unreliable,
        // but the phone-side relative finger geometry is still useful.
        vision.canonical_hand_kpts_3d.as_slice()
    } else {
        empty_hand_points
    };
    let hand_result = fuse_canonical_points(
        &stereo.canonical_hand_kpts_3d,
        vision_hand_points,
        stereo.hand_weight,
        if can_fuse_vision_hands || allow_vision_hand_fallback {
            vision.hand_weight
        } else {
            0.0
        },
        HAND_BLEND_MAX_GAP_M,
    );
    let refined_body_points = refine_fused_body_geometry(
        &stereo.canonical_body_kpts_3d,
        &vision.canonical_body_kpts_3d,
        &body_result.points,
    );
    let (operator_state, left_hand_curls_3d, right_hand_curls_3d) =
        build_from_canonical_3d(refined_body_points, hand_result.points.clone());
    let fusion_breakdown = OperatorFusionBreakdown {
        body_source: derive_part_source(vision, &body_result),
        hand_source: derive_part_source(vision, &hand_result),
        stereo_body_joint_count: body_result.stereo_count,
        vision_body_joint_count: body_result.vision_count,
        wifi_body_joint_count: 0,
        blended_body_joint_count: body_result.blended_count,
        stereo_hand_point_count: hand_result.stereo_count,
        vision_hand_point_count: hand_result.vision_count,
        wifi_hand_point_count: 0,
        blended_hand_point_count: hand_result.blended_count,
    };
    let raw_pose = derive_raw_pose(stereo, vision, &fusion_breakdown);

    OperatorEstimate {
        source: derive_operator_source(stereo, vision, &fusion_breakdown),
        operator_state,
        raw_pose,
        left_hand_curls: left_hand_curls_3d.or_else(|| {
            select_hand_curl_fallback(stereo.left_hand_curls, vision.left_hand_curls, &hand_result)
        }),
        right_hand_curls: right_hand_curls_3d.or_else(|| {
            select_hand_curl_fallback(
                stereo.right_hand_curls,
                vision.right_hand_curls,
                &hand_result,
            )
        }),
        fusion_breakdown,
        association: derive_operator_association(
            Some(stereo),
            Some(vision),
            None,
            None,
            &fusion_breakdown,
        ),
        motion_state: OperatorMotionState::default(),
        updated_edge_time_ns: now_edge_time_ns,
    }
}

fn build_source_plus_wifi_estimate(
    now_edge_time_ns: u64,
    primary: &OperatorSourcePose,
    wifi: &OperatorSourcePose,
) -> OperatorEstimate {
    let estimate = build_single_source_estimate(now_edge_time_ns, primary);
    attach_wifi_context(estimate, now_edge_time_ns, Some(primary), None, wifi, 0)
}

fn build_fused_estimate_with_wifi(
    now_edge_time_ns: u64,
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    wifi: &OperatorSourcePose,
) -> OperatorEstimate {
    let base = build_fused_estimate(now_edge_time_ns, stereo, vision);
    attach_wifi_context(base, now_edge_time_ns, Some(stereo), Some(vision), wifi, 0)
}

fn refine_fused_body_geometry(
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    fused_points: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    let mut refined = fused_points.to_vec();
    if refined.len() <= COCO_RIGHT_ANKLE_INDEX {
        return refined;
    }

    refine_lateral_pair_span(
        &mut refined,
        stereo_points,
        vision_points,
        COCO_LEFT_SHOULDER_INDEX,
        COCO_RIGHT_SHOULDER_INDEX,
        BODY_REFINEMENT_SHOULDER_SPAN_MIN_M,
        BODY_REFINEMENT_SHOULDER_SPAN_MAX_M,
    );
    refine_lateral_pair_span(
        &mut refined,
        stereo_points,
        vision_points,
        COCO_LEFT_HIP_INDEX,
        COCO_RIGHT_HIP_INDEX,
        BODY_REFINEMENT_HIP_SPAN_MIN_M,
        BODY_REFINEMENT_HIP_SPAN_MAX_M,
    );
    refine_bilateral_segments(
        &mut refined,
        stereo_points,
        vision_points,
        (COCO_LEFT_SHOULDER_INDEX, COCO_LEFT_ELBOW_INDEX),
        (COCO_RIGHT_SHOULDER_INDEX, COCO_RIGHT_ELBOW_INDEX),
        BODY_REFINEMENT_UPPER_ARM_MIN_M,
        BODY_REFINEMENT_UPPER_ARM_MAX_M,
    );
    refine_bilateral_segments(
        &mut refined,
        stereo_points,
        vision_points,
        (COCO_LEFT_ELBOW_INDEX, COCO_LEFT_WRIST_INDEX),
        (COCO_RIGHT_ELBOW_INDEX, COCO_RIGHT_WRIST_INDEX),
        BODY_REFINEMENT_FOREARM_MIN_M,
        BODY_REFINEMENT_FOREARM_MAX_M,
    );
    refine_bilateral_segments(
        &mut refined,
        stereo_points,
        vision_points,
        (COCO_LEFT_HIP_INDEX, COCO_LEFT_KNEE_INDEX),
        (COCO_RIGHT_HIP_INDEX, COCO_RIGHT_KNEE_INDEX),
        BODY_REFINEMENT_THIGH_MIN_M,
        BODY_REFINEMENT_THIGH_MAX_M,
    );
    refine_bilateral_segments(
        &mut refined,
        stereo_points,
        vision_points,
        (COCO_LEFT_KNEE_INDEX, COCO_LEFT_ANKLE_INDEX),
        (COCO_RIGHT_KNEE_INDEX, COCO_RIGHT_ANKLE_INDEX),
        BODY_REFINEMENT_CALF_MIN_M,
        BODY_REFINEMENT_CALF_MAX_M,
    );
    refine_head_anchor(&mut refined, stereo_points, vision_points);

    refined
}

fn refine_lateral_pair_span(
    points: &mut [[f32; 3]],
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    left_index: usize,
    right_index: usize,
    min_length_m: f32,
    max_length_m: f32,
) {
    let (Some(left), Some(right)) = (
        present_point(points, left_index),
        present_point(points, right_index),
    ) else {
        return;
    };
    let current_length = dist3(left, right);
    let Some(target_length) = reference_pair_length(
        stereo_points,
        vision_points,
        points,
        left_index,
        right_index,
        min_length_m,
        max_length_m,
    ) else {
        return;
    };
    if !length_needs_refinement(
        current_length,
        target_length,
        min_length_m,
        max_length_m,
        BODY_REFINEMENT_SPAN_TOLERANCE_RATIO,
    ) {
        return;
    }
    let Some(direction) = choose_pair_direction(
        stereo_points,
        vision_points,
        points,
        left_index,
        right_index,
    ) else {
        return;
    };
    let center = midpoint3(left, right);
    let half = scale3(direction, target_length * 0.5);
    points[left_index] = add3(center, half);
    points[right_index] = sub(center, half);
}

fn refine_bilateral_segments(
    points: &mut [[f32; 3]],
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    left_segment: (usize, usize),
    right_segment: (usize, usize),
    min_length_m: f32,
    max_length_m: f32,
) {
    let left_target = reference_pair_length(
        stereo_points,
        vision_points,
        points,
        left_segment.0,
        left_segment.1,
        min_length_m,
        max_length_m,
    );
    let right_target = reference_pair_length(
        stereo_points,
        vision_points,
        points,
        right_segment.0,
        right_segment.1,
        min_length_m,
        max_length_m,
    );
    let (left_target, right_target) =
        harmonize_bilateral_targets(left_target, right_target, min_length_m, max_length_m);
    refine_segment_length(
        points,
        stereo_points,
        vision_points,
        left_segment.0,
        left_segment.1,
        left_target,
        min_length_m,
        max_length_m,
    );
    refine_segment_length(
        points,
        stereo_points,
        vision_points,
        right_segment.0,
        right_segment.1,
        right_target,
        min_length_m,
        max_length_m,
    );
}

fn refine_segment_length(
    points: &mut [[f32; 3]],
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    proximal_index: usize,
    distal_index: usize,
    target_length: Option<f32>,
    min_length_m: f32,
    max_length_m: f32,
) {
    let Some(target_length) = target_length else {
        return;
    };
    let (Some(proximal), Some(distal)) = (
        present_point(points, proximal_index),
        present_point(points, distal_index),
    ) else {
        return;
    };
    let current_length = dist3(proximal, distal);
    if !length_needs_refinement(
        current_length,
        target_length,
        min_length_m,
        max_length_m,
        BODY_REFINEMENT_SEGMENT_TOLERANCE_RATIO,
    ) {
        return;
    }
    let Some(direction) = choose_segment_direction(
        stereo_points,
        vision_points,
        points,
        proximal_index,
        distal_index,
    ) else {
        return;
    };
    points[distal_index] = add3(proximal, scale3(direction, target_length));
}

fn refine_head_anchor(
    points: &mut [[f32; 3]],
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
) {
    let Some(nose) = present_point(points, COCO_NOSE_INDEX) else {
        return;
    };
    let Some(shoulder_center) =
        pair_center(points, COCO_LEFT_SHOULDER_INDEX, COCO_RIGHT_SHOULDER_INDEX)
    else {
        return;
    };
    let current_length = dist3(nose, shoulder_center);
    let target_length = reference_head_offset_length(stereo_points, vision_points, points)
        .unwrap_or((BODY_REFINEMENT_HEAD_OFFSET_MIN_M + BODY_REFINEMENT_HEAD_OFFSET_MAX_M) * 0.5);
    let current_direction = normalize(sub(nose, shoulder_center));
    let source_direction =
        choose_head_direction(stereo_points, vision_points, points).unwrap_or([0.0, 1.0, 0.0]);
    let direction = if is_finite3(current_direction) && current_direction[1] >= 0.15 {
        current_direction
    } else {
        source_direction
    };
    if !length_needs_refinement(
        current_length,
        target_length,
        BODY_REFINEMENT_HEAD_OFFSET_MIN_M,
        BODY_REFINEMENT_HEAD_OFFSET_MAX_M,
        BODY_REFINEMENT_SEGMENT_TOLERANCE_RATIO,
    ) && nose[1] >= shoulder_center[1]
    {
        return;
    }
    points[COCO_NOSE_INDEX] = add3(shoulder_center, scale3(direction, target_length));
}

fn pair_center(points: &[[f32; 3]], a: usize, b: usize) -> Option<[f32; 3]> {
    Some(midpoint3(
        present_point(points, a)?,
        present_point(points, b)?,
    ))
}

fn reference_head_offset_length(
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    fused_points: &[[f32; 3]],
) -> Option<f32> {
    let stereo = pair_center(
        stereo_points,
        COCO_LEFT_SHOULDER_INDEX,
        COCO_RIGHT_SHOULDER_INDEX,
    )
    .zip(present_point(stereo_points, COCO_NOSE_INDEX))
    .map(|(center, nose)| dist3(center, nose));
    let vision = pair_center(
        vision_points,
        COCO_LEFT_SHOULDER_INDEX,
        COCO_RIGHT_SHOULDER_INDEX,
    )
    .zip(present_point(vision_points, COCO_NOSE_INDEX))
    .map(|(center, nose)| dist3(center, nose));
    let fused = pair_center(
        fused_points,
        COCO_LEFT_SHOULDER_INDEX,
        COCO_RIGHT_SHOULDER_INDEX,
    )
    .zip(present_point(fused_points, COCO_NOSE_INDEX))
    .map(|(center, nose)| dist3(center, nose));
    select_reference_length(
        stereo,
        vision,
        fused,
        BODY_REFINEMENT_HEAD_OFFSET_MIN_M,
        BODY_REFINEMENT_HEAD_OFFSET_MAX_M,
    )
}

fn choose_head_direction(
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    fused_points: &[[f32; 3]],
) -> Option<[f32; 3]> {
    [
        (
            pair_center(
                stereo_points,
                COCO_LEFT_SHOULDER_INDEX,
                COCO_RIGHT_SHOULDER_INDEX,
            ),
            present_point(stereo_points, COCO_NOSE_INDEX),
        ),
        (
            pair_center(
                vision_points,
                COCO_LEFT_SHOULDER_INDEX,
                COCO_RIGHT_SHOULDER_INDEX,
            ),
            present_point(vision_points, COCO_NOSE_INDEX),
        ),
        (
            pair_center(
                fused_points,
                COCO_LEFT_SHOULDER_INDEX,
                COCO_RIGHT_SHOULDER_INDEX,
            ),
            present_point(fused_points, COCO_NOSE_INDEX),
        ),
    ]
    .into_iter()
    .find_map(|(center, nose)| {
        let center = center?;
        let nose = nose?;
        let direction = normalize(sub(nose, center));
        (is_finite3(direction) && direction[1] >= 0.15).then_some(direction)
    })
}

fn reference_pair_length(
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    fused_points: &[[f32; 3]],
    first_index: usize,
    second_index: usize,
    min_length_m: f32,
    max_length_m: f32,
) -> Option<f32> {
    let stereo = pair_length(stereo_points, first_index, second_index);
    let vision = pair_length(vision_points, first_index, second_index);
    let fused = pair_length(fused_points, first_index, second_index);
    select_reference_length(stereo, vision, fused, min_length_m, max_length_m)
}

fn pair_length(points: &[[f32; 3]], first_index: usize, second_index: usize) -> Option<f32> {
    Some(dist3(
        present_point(points, first_index)?,
        present_point(points, second_index)?,
    ))
}

fn select_reference_length(
    stereo_length: Option<f32>,
    vision_length: Option<f32>,
    fused_length: Option<f32>,
    min_length_m: f32,
    max_length_m: f32,
) -> Option<f32> {
    [stereo_length, vision_length, fused_length]
        .into_iter()
        .flatten()
        .find(|length| is_plausible_length(*length, min_length_m, max_length_m))
}

fn harmonize_bilateral_targets(
    left_target: Option<f32>,
    right_target: Option<f32>,
    min_length_m: f32,
    max_length_m: f32,
) -> (Option<f32>, Option<f32>) {
    match (left_target, right_target) {
        (Some(left), Some(right)) => {
            if left.max(right) / left.min(right).max(1e-3) > BODY_REFINEMENT_SYMMETRY_RATIO {
                let mean = ((left + right) * 0.5).clamp(min_length_m, max_length_m);
                (Some(mean), Some(mean))
            } else {
                (Some(left), Some(right))
            }
        }
        (Some(left), None) => (Some(left), Some(left)),
        (None, Some(right)) => (Some(right), Some(right)),
        (None, None) => (None, None),
    }
}

fn length_needs_refinement(
    current_length: f32,
    target_length: f32,
    min_length_m: f32,
    max_length_m: f32,
    tolerance_ratio: f32,
) -> bool {
    !is_plausible_length(current_length, min_length_m, max_length_m)
        || (current_length - target_length).abs() > target_length * tolerance_ratio
}

fn is_plausible_length(length: f32, min_length_m: f32, max_length_m: f32) -> bool {
    length.is_finite() && length >= min_length_m && length <= max_length_m
}

fn choose_pair_direction(
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    fused_points: &[[f32; 3]],
    left_index: usize,
    right_index: usize,
) -> Option<[f32; 3]> {
    [stereo_points, vision_points, fused_points]
        .into_iter()
        .find_map(|points| {
            let left = present_point(points, left_index)?;
            let right = present_point(points, right_index)?;
            let direction = normalize(sub(left, right));
            is_finite3(direction).then_some(direction)
        })
}

fn choose_segment_direction(
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    fused_points: &[[f32; 3]],
    proximal_index: usize,
    distal_index: usize,
) -> Option<[f32; 3]> {
    [stereo_points, vision_points, fused_points]
        .into_iter()
        .find_map(|points| {
            let proximal = present_point(points, proximal_index)?;
            let distal = present_point(points, distal_index)?;
            let direction = normalize(sub(distal, proximal));
            is_finite3(direction).then_some(direction)
        })
}

#[derive(Clone, Debug, Default)]
struct FusedPointCloud {
    points: Vec<[f32; 3]>,
    stereo_count: usize,
    vision_count: usize,
    blended_count: usize,
}

fn fuse_canonical_points(
    stereo_points: &[[f32; 3]],
    vision_points: &[[f32; 3]],
    stereo_weight: f32,
    vision_weight: f32,
    max_gap_m: f32,
) -> FusedPointCloud {
    let len = stereo_points.len().max(vision_points.len());
    let mut result = FusedPointCloud {
        points: Vec::with_capacity(len),
        ..FusedPointCloud::default()
    };

    for index in 0..len {
        let stereo_point = stereo_points.get(index).copied().filter(is_present_point3);
        let vision_point = vision_points.get(index).copied().filter(is_present_point3);
        match (stereo_point, vision_point) {
            (Some(stereo_point), Some(vision_point)) => {
                if dist3(stereo_point, vision_point) <= max_gap_m {
                    let total_weight = (stereo_weight + vision_weight).max(MIN_SOURCE_WEIGHT);
                    let inv_total = 1.0 / total_weight;
                    result.points.push([
                        (stereo_point[0] * stereo_weight + vision_point[0] * vision_weight)
                            * inv_total,
                        (stereo_point[1] * stereo_weight + vision_point[1] * vision_weight)
                            * inv_total,
                        (stereo_point[2] * stereo_weight + vision_point[2] * vision_weight)
                            * inv_total,
                    ]);
                    result.stereo_count += 1;
                    result.vision_count += 1;
                    result.blended_count += 1;
                } else if stereo_weight >= vision_weight {
                    result.points.push(stereo_point);
                    result.stereo_count += 1;
                } else {
                    result.points.push(vision_point);
                    result.vision_count += 1;
                }
            }
            (Some(stereo_point), None) => {
                result.points.push(stereo_point);
                result.stereo_count += 1;
            }
            (None, Some(vision_point)) => {
                result.points.push(vision_point);
                result.vision_count += 1;
            }
            (None, None) => result.points.push([0.0, 0.0, 0.0]),
        }
    }

    result
}

fn derive_part_source(vision: &OperatorSourcePose, result: &FusedPointCloud) -> OperatorPartSource {
    if result.stereo_count > 0 && result.vision_count > 0 {
        match vision.source {
            OperatorSource::Vision2dProjected => OperatorPartSource::FusedStereoVision2dProjected,
            _ => OperatorPartSource::FusedStereoVision3d,
        }
    } else if result.stereo_count > 0 {
        OperatorPartSource::Stereo
    } else if result.vision_count > 0 {
        vision.source.as_part_source()
    } else {
        OperatorPartSource::None
    }
}

fn derive_operator_source(
    _stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    breakdown: &OperatorFusionBreakdown,
) -> OperatorSource {
    let stereo_total = breakdown.stereo_body_joint_count + breakdown.stereo_hand_point_count;
    let vision_total = breakdown.vision_body_joint_count + breakdown.vision_hand_point_count;
    match (stereo_total > 0, vision_total > 0) {
        (true, true) => match vision.source {
            OperatorSource::Vision2dProjected => OperatorSource::FusedStereoVision2dProjected,
            _ => OperatorSource::FusedStereoVision3d,
        },
        (true, false) => OperatorSource::Stereo,
        (false, true) => vision.source,
        (false, false) => OperatorSource::None,
    }
}

fn derive_raw_pose(
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    breakdown: &OperatorFusionBreakdown,
) -> OperatorRawPose {
    let primary_body = primary_raw_source(
        breakdown.stereo_body_joint_count,
        breakdown.vision_body_joint_count,
        stereo,
        vision,
    );
    let primary_hand = primary_raw_source(
        breakdown.stereo_hand_point_count,
        breakdown.vision_hand_point_count,
        stereo,
        vision,
    );

    OperatorRawPose {
        source_edge_time_ns: primary_body
            .iter()
            .chain(primary_hand.iter())
            .map(|source| source.raw_pose.source_edge_time_ns)
            .max()
            .unwrap_or(0),
        body_layout: primary_body
            .map(|source| source.raw_pose.body_layout)
            .unwrap_or_default(),
        hand_layout: primary_hand
            .map(|source| source.raw_pose.hand_layout)
            .unwrap_or_default(),
        body_kpts_3d: primary_body
            .map(|source| source.raw_pose.body_kpts_3d.clone())
            .unwrap_or_default(),
        hand_kpts_3d: primary_hand
            .map(|source| source.raw_pose.hand_kpts_3d.clone())
            .unwrap_or_default(),
    }
}

fn derive_operator_association(
    stereo: Option<&OperatorSourcePose>,
    vision: Option<&OperatorSourcePose>,
    wifi: Option<&OperatorSourcePose>,
    wifi_diagnostics: Option<&WifiPoseDiagnostics>,
    breakdown: &OperatorFusionBreakdown,
) -> OperatorAssociationDebug {
    let stereo_operator_track_id = stereo.and_then(|source| source.operator_track_id.clone());
    let wifi_operator_track_id = wifi.and_then(|source| source.operator_track_id.clone());
    let iphone_source_track_id = vision.and_then(|source| source.operator_track_id.clone());
    let wifi_prior_joint_count = wifi
        .map(|source| source.canonical_body_kpts_3d.len())
        .unwrap_or(0);
    let wifi_prior_available = wifi_prior_joint_count > 0;
    let hand_match_body_source = stereo
        .filter(|source| body_can_anchor_iphone_hands(&source.body_space))
        .or_else(|| wifi.filter(|source| body_can_anchor_iphone_hands(&source.body_space)));
    let (
        hand_match_score,
        left_wrist_gap_m,
        right_wrist_gap_m,
        iphone_visible_hand_count,
        hand_match_count,
    ) = match_body_with_iphone_hands(hand_match_body_source, vision);
    let iphone_has_hand_points = iphone_visible_hand_count > 0;
    let iphone_trusted_hands = iphone_has_hand_points
        && (breakdown.stereo_body_joint_count == 0
            || hand_match_score > IPHONE_HAND_MATCH_TRUST_THRESHOLD);
    let wifi_association_score = compute_wifi_association_score(
        stereo_operator_track_id.as_deref(),
        wifi_operator_track_id.as_deref(),
        wifi_diagnostics,
    );
    let wifi_anchor_eligible = wifi_tracking_anchor_eligible(wifi_diagnostics);
    let wifi_trusted =
        wifi_prior_available && wifi_association_score >= 0.18 && wifi_anchor_eligible;
    let wifi_anchor_only = breakdown.wifi_body_joint_count > 0
        && breakdown.stereo_body_joint_count == 0
        && wifi_anchor_eligible;

    let selected_operator_track_id = if breakdown.stereo_body_joint_count > 0 {
        stereo_operator_track_id
            .clone()
            .or_else(|| {
                iphone_source_track_id
                    .clone()
                    .filter(|_| iphone_trusted_hands)
            })
            .or_else(|| wifi_operator_track_id.clone().filter(|_| wifi_trusted))
            .or_else(|| stereo_operator_track_id.clone())
    } else if breakdown.wifi_body_joint_count > 0 {
        wifi_operator_track_id
            .clone()
            .filter(|_| wifi_trusted || wifi_anchor_only)
            .or_else(|| stereo_operator_track_id.clone())
            .or_else(|| {
                iphone_source_track_id
                    .clone()
                    .filter(|_| iphone_trusted_hands)
            })
    } else if iphone_trusted_hands {
        wifi_operator_track_id
            .clone()
            .filter(|_| wifi_trusted)
            .or_else(|| stereo_operator_track_id.clone())
            .or_else(|| iphone_source_track_id.clone())
    } else if wifi_trusted {
        wifi_operator_track_id
            .clone()
            .or_else(|| stereo_operator_track_id.clone())
            .or_else(|| {
                iphone_source_track_id
                    .clone()
                    .filter(|_| iphone_trusted_hands)
            })
    } else if breakdown.stereo_body_joint_count > 0 {
        stereo_operator_track_id
            .clone()
            .or_else(|| wifi_operator_track_id.clone().filter(|_| wifi_trusted))
            .or_else(|| {
                iphone_source_track_id
                    .clone()
                    .filter(|_| iphone_trusted_hands)
            })
    } else {
        stereo_operator_track_id.clone().or_else(|| {
            iphone_source_track_id
                .clone()
                .filter(|_| iphone_trusted_hands)
        })
    };

    let iphone_operator_track_id = if iphone_trusted_hands {
        selected_operator_track_id
            .clone()
            .or_else(|| stereo_operator_track_id.clone())
            .or_else(|| wifi_operator_track_id.clone())
            .or_else(|| iphone_source_track_id.clone())
    } else {
        None
    };

    let anchor_source = if breakdown.wifi_body_joint_count > 0 {
        if breakdown.stereo_body_joint_count > 0 {
            if wifi_trusted {
                if iphone_trusted_hands {
                    "wifi+stereo+iphone_hand"
                } else {
                    "wifi+stereo"
                }
            } else if iphone_trusted_hands {
                "stereo+iphone_hand"
            } else {
                "stereo"
            }
        } else if iphone_trusted_hands {
            "wifi+iphone_hand"
        } else if wifi_trusted || wifi_anchor_only {
            "wifi"
        } else {
            "none"
        }
    } else if breakdown.stereo_body_joint_count > 0 {
        if iphone_trusted_hands && wifi_trusted {
            "stereo+iphone_hand+wifi_prior"
        } else if iphone_trusted_hands {
            "stereo+iphone_hand"
        } else if wifi_trusted {
            "stereo+wifi_prior"
        } else {
            "stereo"
        }
    } else if iphone_trusted_hands {
        if wifi_trusted {
            "iphone_hand+wifi_prior"
        } else {
            "iphone_hand"
        }
    } else if wifi_trusted {
        "wifi_prior"
    } else {
        "none"
    };

    OperatorAssociationDebug {
        selected_operator_track_id,
        anchor_source,
        stereo_operator_track_id,
        wifi_operator_track_id,
        iphone_operator_track_id,
        wifi_anchor_eligible,
        wifi_lifecycle_state: wifi_diagnostics
            .map(|diag| diag.lifecycle_state.clone())
            .unwrap_or_default(),
        wifi_coherence_gate_decision: wifi_diagnostics
            .map(|diag| diag.coherence_gate_decision.clone())
            .unwrap_or_default(),
        iphone_visible_hand_count,
        hand_match_count,
        hand_match_score,
        wifi_association_score,
        wifi_layout_score: wifi_diagnostics
            .map(|diag| diag.layout_score)
            .unwrap_or(0.0),
        wifi_zone_score: wifi_diagnostics.map(|diag| diag.zone_score).unwrap_or(0.0),
        wifi_motion_energy: wifi_diagnostics
            .map(|diag| diag.motion_energy)
            .unwrap_or(0.0),
        wifi_doppler_hz: wifi_diagnostics.map(|diag| diag.doppler_hz).unwrap_or(0.0),
        wifi_signal_quality: wifi_diagnostics
            .map(|diag| diag.signal_quality)
            .unwrap_or(0.0),
        wifi_zone_summary_reliable: wifi_diagnostics
            .map(|diag| diag.zone_summary_reliable)
            .unwrap_or(false),
        left_wrist_gap_m,
        right_wrist_gap_m,
    }
}

fn stabilize_operator_estimate(
    mut estimate: OperatorEstimate,
    previous: Option<&OperatorEstimate>,
    previous_recent: bool,
) -> OperatorEstimate {
    let Some(previous) = previous else {
        return estimate;
    };
    if !previous_recent {
        return estimate;
    }
    let Some(previous_selected_track_id) = previous.association.selected_operator_track_id.as_ref()
    else {
        return estimate;
    };

    let current_wifi_only = estimate.fusion_breakdown.wifi_body_joint_count > 0
        && estimate.fusion_breakdown.stereo_body_joint_count == 0;
    let previous_had_stereo_identity = previous.fusion_breakdown.stereo_body_joint_count > 0
        || previous.association.anchor_source.contains("stereo");
    let current_wifi_confident = estimate.association.wifi_association_score
        >= WIFI_SELECTED_TRACK_STICKY_MIN_SCORE
        && estimate.association.wifi_layout_score >= WIFI_SELECTED_TRACK_STICKY_MIN_LAYOUT_SCORE
        && estimate.association.wifi_anchor_eligible;

    if current_wifi_only && previous_had_stereo_identity && current_wifi_confident {
        estimate.association.selected_operator_track_id = Some(previous_selected_track_id.clone());
    }

    let current_selection_degraded_to_non_stereo = estimate
        .association
        .selected_operator_track_id
        .as_deref()
        .is_none_or(|track_id| !track_id.starts_with("stereo-"));
    let current_has_no_confirmed_anchor =
        matches!(estimate.association.anchor_source, "none" | "wifi");

    if previous_had_stereo_identity
        && current_selection_degraded_to_non_stereo
        && current_has_no_confirmed_anchor
    {
        estimate.association.selected_operator_track_id = Some(previous_selected_track_id.clone());
    }

    let current_selected_track_matches_previous =
        estimate.association.selected_operator_track_id.as_ref()
            == Some(previous_selected_track_id);
    let previous_wearer_track_id = previous.association.iphone_operator_track_id.as_ref();
    let previous_best_gap = [
        previous.association.left_wrist_gap_m,
        previous.association.right_wrist_gap_m,
    ]
    .into_iter()
    .flatten()
    .fold(f32::INFINITY, f32::min);
    let previous_hand_match_stable = previous.association.hand_match_count > 0
        && previous.association.hand_match_score >= IPHONE_HAND_MATCH_STICKY_MIN_SCORE
        && previous_best_gap <= IPHONE_HAND_MATCH_STICKY_MAX_GAP_M;
    let previous_wearer_track_stable = previous_hand_match_stable
        && previous.association.hand_match_score >= IPHONE_WEARER_TRACK_STICKY_MIN_SCORE
        && previous_wearer_track_id.is_some();
    let current_has_visible_iphone_hand = estimate.association.iphone_visible_hand_count > 0;
    let current_has_stereo = estimate.fusion_breakdown.stereo_body_joint_count > 0
        || estimate.association.anchor_source.contains("stereo");
    let current_has_body_anchor = current_has_stereo
        || estimate.fusion_breakdown.wifi_body_joint_count > 0
        || estimate.association.anchor_source.contains("wifi");
    let current_degraded_to_iphone_only_track = estimate.association.anchor_source == "iphone_hand"
        && current_has_visible_iphone_hand
        && estimate.association.selected_operator_track_id.as_deref() == Some("primary_operator");
    let current_matches_previous_wearer = previous_wearer_track_id.is_some_and(|wearer_track_id| {
        [
            estimate.association.selected_operator_track_id.as_deref(),
            estimate.association.stereo_operator_track_id.as_deref(),
            estimate.association.wifi_operator_track_id.as_deref(),
            estimate.association.iphone_operator_track_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|track_id| track_id == wearer_track_id)
    });
    let current_hand_match_unstable = estimate.association.hand_match_count == 0
        || estimate.association.hand_match_score < previous.association.hand_match_score * 0.5;
    let current_iphone_temporarily_missing = estimate.association.iphone_visible_hand_count == 0
        && estimate.association.hand_match_count == 0
        && estimate.association.hand_match_score <= f32::EPSILON;

    if previous_wearer_track_stable && current_has_body_anchor && current_matches_previous_wearer {
        estimate.association.selected_operator_track_id = previous_wearer_track_id.cloned();
        if estimate.association.iphone_operator_track_id.is_none() {
            estimate.association.iphone_operator_track_id = previous_wearer_track_id.cloned();
        }
    } else if previous_wearer_track_stable
        && current_has_body_anchor
        && current_degraded_to_iphone_only_track
    {
        estimate.association.selected_operator_track_id = previous_wearer_track_id.cloned();
        estimate.association.iphone_operator_track_id = previous_wearer_track_id.cloned();
        if estimate.association.stereo_operator_track_id.is_none() {
            estimate.association.stereo_operator_track_id =
                previous.association.stereo_operator_track_id.clone();
        }
        estimate.association.hand_match_score = (previous.association.hand_match_score
            * IPHONE_HAND_MATCH_STICKY_DECAY)
            .clamp(0.0, 1.0);
        estimate.association.left_wrist_gap_m = previous.association.left_wrist_gap_m;
        estimate.association.right_wrist_gap_m = previous.association.right_wrist_gap_m;
        estimate.association.hand_match_count = previous.association.hand_match_count;
        estimate.association.anchor_source = previous.association.anchor_source;
    }

    if previous_hand_match_stable
        && current_has_visible_iphone_hand
        && current_has_stereo
        && current_selected_track_matches_previous
        && current_hand_match_unstable
    {
        estimate.association.hand_match_score = (previous.association.hand_match_score
            * IPHONE_HAND_MATCH_STICKY_DECAY)
            .clamp(0.0, 1.0);
        estimate.association.left_wrist_gap_m = previous.association.left_wrist_gap_m;
        estimate.association.right_wrist_gap_m = previous.association.right_wrist_gap_m;
        estimate.association.hand_match_count = previous.association.hand_match_count;
        estimate.association.iphone_operator_track_id =
            previous.association.iphone_operator_track_id.clone();
        estimate.association.anchor_source =
            sticky_iphone_anchor_source(estimate.association.anchor_source);
    } else if previous_hand_match_stable
        && current_iphone_temporarily_missing
        && current_has_body_anchor
        && current_selected_track_matches_previous
    {
        estimate.association.hand_match_score = (previous.association.hand_match_score
            * IPHONE_HAND_MATCH_STICKY_DECAY)
            .clamp(0.0, 1.0);
        estimate.association.left_wrist_gap_m = previous.association.left_wrist_gap_m;
        estimate.association.right_wrist_gap_m = previous.association.right_wrist_gap_m;
        estimate.association.hand_match_count = previous.association.hand_match_count;
        estimate.association.iphone_operator_track_id =
            previous.association.iphone_operator_track_id.clone();
        estimate.association.anchor_source =
            sticky_iphone_anchor_source(estimate.association.anchor_source);
    }

    let current_body_missing = estimate.operator_state.body_kpts_3d.is_empty();
    let previous_body_available = !previous.operator_state.body_kpts_3d.is_empty();
    let current_hands_missing = !estimate
        .operator_state
        .hand_kpts_3d
        .iter()
        .any(is_present_point3);
    let previous_hands_available = previous
        .operator_state
        .hand_kpts_3d
        .iter()
        .any(is_present_point3);
    let previous_wearer_track_or_selected = previous
        .association
        .iphone_operator_track_id
        .as_ref()
        .or(previous.association.selected_operator_track_id.as_ref());
    let current_matches_previous_visible_body =
        previous_wearer_track_or_selected.is_some_and(|track_id| {
            [
                estimate.association.selected_operator_track_id.as_deref(),
                estimate.association.stereo_operator_track_id.as_deref(),
                estimate.association.wifi_operator_track_id.as_deref(),
                estimate.association.iphone_operator_track_id.as_deref(),
            ]
            .into_iter()
            .flatten()
            .any(|current_track_id| current_track_id == track_id)
        });

    if current_body_missing
        && previous_body_available
        && current_has_body_anchor
        && current_matches_previous_visible_body
    {
        estimate.operator_state.body_kpts_3d = previous.operator_state.body_kpts_3d.clone();
        estimate.fusion_breakdown.body_source = previous.fusion_breakdown.body_source;
        if estimate.raw_pose.body_kpts_3d.is_empty() {
            estimate.raw_pose.body_kpts_3d = previous.raw_pose.body_kpts_3d.clone();
            estimate.raw_pose.body_layout = previous.raw_pose.body_layout;
        }
        if estimate.source == OperatorSource::None {
            estimate.source = OperatorSource::Hold;
        }
    }

    if previous_hand_match_stable
        && current_hands_missing
        && previous_hands_available
        && current_has_body_anchor
        && current_matches_previous_visible_body
    {
        estimate.operator_state.hand_kpts_3d = previous.operator_state.hand_kpts_3d.clone();
        estimate.left_hand_curls = estimate.left_hand_curls.or(previous.left_hand_curls);
        estimate.right_hand_curls = estimate.right_hand_curls.or(previous.right_hand_curls);
        if estimate.raw_pose.hand_kpts_3d.is_empty() {
            estimate.raw_pose.hand_kpts_3d = previous.raw_pose.hand_kpts_3d.clone();
            estimate.raw_pose.hand_layout = previous.raw_pose.hand_layout;
        }
        if estimate.fusion_breakdown.hand_source == OperatorPartSource::None {
            estimate.fusion_breakdown.hand_source = previous.fusion_breakdown.hand_source;
            estimate.fusion_breakdown.stereo_hand_point_count =
                previous.fusion_breakdown.stereo_hand_point_count;
            estimate.fusion_breakdown.vision_hand_point_count =
                previous.fusion_breakdown.vision_hand_point_count;
            estimate.fusion_breakdown.wifi_hand_point_count =
                previous.fusion_breakdown.wifi_hand_point_count;
            estimate.fusion_breakdown.blended_hand_point_count =
                previous.fusion_breakdown.blended_hand_point_count;
        }
        estimate.association.anchor_source =
            sticky_iphone_anchor_source(estimate.association.anchor_source);
        if estimate.association.iphone_operator_track_id.is_none() {
            estimate.association.iphone_operator_track_id =
                previous.association.iphone_operator_track_id.clone();
        }
    }

    estimate
}

fn apply_confirmed_wearer_sticky(
    inner: &mut OperatorInner,
    estimate: &mut OperatorEstimate,
    now_edge_time_ns: u64,
) {
    let confirmed_track_id = estimate
        .association
        .iphone_operator_track_id
        .clone()
        .filter(|track_id| {
            !track_id.is_empty()
                && estimate.association.hand_match_count > 0
                && estimate.association.hand_match_score >= IPHONE_WEARER_TRACK_STICKY_MIN_SCORE
        });
    if let Some(track_id) = confirmed_track_id {
        inner.last_confirmed_wearer_track_id = Some(track_id);
        inner.last_confirmed_wearer_edge_time_ns = now_edge_time_ns;
        return;
    }

    let Some(held_track_id) = inner.last_confirmed_wearer_track_id.clone() else {
        return;
    };
    if now_edge_time_ns.saturating_sub(inner.last_confirmed_wearer_edge_time_ns)
        > IPHONE_WEARER_TRACK_STICKY_HOLD_NS
    {
        inner.last_confirmed_wearer_track_id = None;
        inner.last_confirmed_wearer_edge_time_ns = 0;
        return;
    }

    let track_still_matches = [
        estimate.association.selected_operator_track_id.as_deref(),
        estimate.association.stereo_operator_track_id.as_deref(),
        estimate.association.wifi_operator_track_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|track_id| track_id == held_track_id);
    if !track_still_matches {
        return;
    }

    if estimate.association.iphone_operator_track_id.is_none() {
        estimate.association.iphone_operator_track_id = Some(held_track_id);
        estimate.association.anchor_source =
            sticky_iphone_anchor_source(estimate.association.anchor_source);
        estimate.association.hand_match_score = estimate
            .association
            .hand_match_score
            .max(IPHONE_WEARER_TRACK_STICKY_MIN_SCORE * 0.82);
    }
}

fn compute_wifi_association_score(
    stereo_track_id: Option<&str>,
    wifi_track_id: Option<&str>,
    diagnostics: Option<&WifiPoseDiagnostics>,
) -> f32 {
    let Some(diag) = diagnostics else {
        return 0.0;
    };

    let track_score = match (stereo_track_id, wifi_track_id) {
        (Some(stereo), Some(wifi)) if !stereo.is_empty() && stereo == wifi => 0.26,
        (Some(stereo), Some(wifi)) if !stereo.is_empty() && !wifi.is_empty() => 0.12,
        (None, Some(_)) => 0.08,
        _ => 0.0,
    };
    let layout_conf = diag.layout_score.clamp(0.0, 1.0);
    let zone_conf = diag.zone_score.clamp(0.0, 1.0);
    let signal_conf = diag.signal_quality.clamp(0.0, 1.0);

    let layout_score = layout_conf * 0.16;
    let signal_score = signal_conf * 0.14;
    let motion_score = (diag.motion_energy / 8.0).clamp(0.0, 1.0) * 0.10;
    let doppler_score = (diag.doppler_hz.abs() / 0.8).clamp(0.0, 1.0) * 0.08;
    let zone_score = if diag.zone_summary_reliable {
        zone_conf * 0.08
    } else {
        0.0
    };
    let stream_score = (diag.stream_fps / 10.0).clamp(0.0, 1.0) * 0.04;
    let vital_score = diag
        .vital_signal_quality
        .map(|value| value.clamp(0.0, 1.0) * 0.04)
        .unwrap_or(0.0);
    let multi_node_bonus = if diag.layout_node_count >= 4 {
        layout_conf.min(signal_conf) * 0.05
    } else if diag.layout_node_count == 3 {
        layout_conf.min(signal_conf) * 0.02
    } else {
        0.0
    };
    let coherent_bonus = if diag.zone_summary_reliable {
        layout_conf.min(zone_conf).min(signal_conf) * 0.05
    } else {
        0.0
    };

    (track_score
        + layout_score
        + signal_score
        + motion_score
        + doppler_score
        + zone_score
        + stream_score
        + vital_score
        + multi_node_bonus
        + coherent_bonus)
        .clamp(0.0, 1.0)
}

fn match_body_with_iphone_hands(
    body: Option<&OperatorSourcePose>,
    vision: Option<&OperatorSourcePose>,
) -> (f32, Option<f32>, Option<f32>, usize, usize) {
    let iphone_visible_hand_count = vision.map(count_visible_iphone_hands).unwrap_or(0);
    let Some(body) = body else {
        return (0.0, None, None, iphone_visible_hand_count, 0);
    };
    let Some(vision) = vision else {
        return (0.0, None, None, 0, 0);
    };
    if !vision.hand_geometry_trusted {
        return (0.0, None, None, iphone_visible_hand_count, 0);
    }
    if !vision.hand_space.is_empty()
        && !spaces_are_geometry_compatible(&body.body_space, &vision.hand_space)
    {
        return (0.0, None, None, iphone_visible_hand_count, 0);
    }
    let direct = evaluate_hand_match_assignment(body, vision, false);
    let mirrored = evaluate_hand_match_assignment(body, vision, true);
    let best = if direct.preferred_over(&mirrored) {
        direct
    } else {
        mirrored
    };
    let trusted = best.trusted_gaps();
    if trusted.is_empty() {
        return (
            0.0,
            best.left_gap_value(),
            best.right_gap_value(),
            iphone_visible_hand_count,
            0,
        );
    }
    let mean_gap = trusted.iter().sum::<f32>() / trusted.len() as f32;
    let score = if trusted.len() == 1 && best.single_gap_used_shoulder_fallback() {
        (1.0 - (mean_gap / IPHONE_HAND_SHOULDER_FALLBACK_MAX_GAP_M)).clamp(0.0, 1.0)
    } else {
        (1.0 - (mean_gap / IPHONE_HAND_MATCH_MAX_GAP_M)).clamp(0.0, 1.0)
    };
    (
        score,
        best.left_gap_value(),
        best.right_gap_value(),
        iphone_visible_hand_count,
        trusted.len(),
    )
}

fn vision_hands_are_fusion_safe(stereo: &OperatorSourcePose, vision: &OperatorSourcePose) -> bool {
    let (score, left_gap_m, right_gap_m, visible_count, matched_count) =
        match_body_with_iphone_hands(Some(stereo), Some(vision));
    if visible_count == 0 || matched_count == 0 || score < IPHONE_HAND_FUSION_MIN_SCORE {
        return false;
    }

    let best_gap_m = [left_gap_m, right_gap_m]
        .into_iter()
        .flatten()
        .fold(f32::INFINITY, f32::min);
    best_gap_m.is_finite() && best_gap_m <= IPHONE_HAND_FUSION_MAX_GAP_M
}

fn apply_vision_hand_alignment_offset(
    source: &OperatorSourcePose,
    offset: [f32; 3],
) -> OperatorSourcePose {
    if !offset.iter().all(|value| value.is_finite()) {
        return source.clone();
    }
    if offset.iter().all(|value| value.abs() <= 1e-6) {
        return source.clone();
    }
    let mut adjusted = source.clone();
    for point in &mut adjusted.raw_pose.hand_kpts_3d {
        if is_present_point3(point) {
            *point = add3(*point, offset);
        }
    }
    for point in &mut adjusted.canonical_hand_kpts_3d {
        if is_present_point3(point) {
            *point = add3(*point, offset);
        }
    }
    adjusted
}

fn estimate_iphone_hand_alignment_delta(
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
) -> Option<[f32; 3]> {
    if !vision.hand_geometry_trusted {
        return None;
    }
    if !spaces_are_geometry_compatible(&stereo.body_space, &vision.hand_space) {
        return None;
    }
    let direct = estimate_hand_alignment_delta_assignment(stereo, vision, false);
    let mirrored = estimate_hand_alignment_delta_assignment(stereo, vision, true);
    let best = if direct.preferred_over(&mirrored) {
        direct
    } else {
        mirrored
    };
    best.mean_delta()
}

#[derive(Clone, Debug, Default)]
struct HandAlignmentDeltaCandidate {
    deltas: Vec<[f32; 3]>,
    gaps: Vec<f32>,
}

impl HandAlignmentDeltaCandidate {
    fn preferred_over(&self, other: &Self) -> bool {
        match self.deltas.len().cmp(&other.deltas.len()) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => self.mean_gap() <= other.mean_gap(),
        }
    }

    fn mean_gap(&self) -> f32 {
        if self.gaps.is_empty() {
            return f32::INFINITY;
        }
        self.gaps.iter().sum::<f32>() / self.gaps.len() as f32
    }

    fn mean_delta(&self) -> Option<[f32; 3]> {
        if self.deltas.is_empty() {
            return None;
        }
        let sum = self.deltas.iter().copied().fold([0.0, 0.0, 0.0], add3);
        Some(scale3(sum, 1.0 / self.deltas.len() as f32))
    }
}

fn estimate_hand_alignment_delta_assignment(
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    mirrored: bool,
) -> HandAlignmentDeltaCandidate {
    let (left_vision_is_left, right_vision_is_left) = if mirrored {
        (false, true)
    } else {
        (true, false)
    };
    let mut deltas = Vec::new();
    let mut gaps = Vec::new();
    for (stereo_is_left, vision_is_left) in
        [(true, left_vision_is_left), (false, right_vision_is_left)]
    {
        if !vision_hand_is_fresh(vision, vision_is_left) {
            continue;
        }
        let Some(stereo_wrist) = body_wrist_from_canonical(stereo, stereo_is_left) else {
            continue;
        };
        let Some(iphone_wrist) = hand_wrist_from_canonical(vision, vision_is_left) else {
            continue;
        };
        let gap_m = dist3(stereo_wrist, iphone_wrist);
        if gap_m <= IPHONE_HAND_ALIGNMENT_UPDATE_MAX_GAP_M {
            deltas.push(sub(stereo_wrist, iphone_wrist));
            gaps.push(gap_m);
        }
    }
    HandAlignmentDeltaCandidate { deltas, gaps }
}

fn clamp_offset_norm(offset: [f32; 3], max_norm_m: f32) -> [f32; 3] {
    let norm = dist3(offset, [0.0, 0.0, 0.0]);
    if !norm.is_finite() || norm <= max_norm_m {
        return offset;
    }
    scale3(offset, max_norm_m / norm)
}

fn update_iphone_hand_alignment_state(
    inner: &mut OperatorInner,
    stereo: Option<&OperatorSourcePose>,
    vision: Option<&OperatorSourcePose>,
    estimate: &OperatorEstimate,
) {
    let selected_track_id = estimate.association.selected_operator_track_id.clone();
    let stereo_track_id = stereo.and_then(|source| source.operator_track_id.clone());
    if selected_track_id.is_none()
        || stereo_track_id.is_none()
        || selected_track_id != stereo_track_id
    {
        if inner.iphone_hand_alignment_track_id.as_ref() != selected_track_id.as_ref() {
            inner.iphone_hand_alignment_offset = None;
            inner.iphone_hand_alignment_track_id = selected_track_id;
        }
        return;
    }

    if let (Some(stereo), Some(vision)) = (stereo, vision) {
        if let Some(delta) = estimate_iphone_hand_alignment_delta(stereo, vision) {
            let updated = match inner.iphone_hand_alignment_offset {
                Some(previous) => add3(previous, scale3(delta, IPHONE_HAND_ALIGNMENT_ALPHA)),
                None => delta,
            };
            inner.iphone_hand_alignment_offset = Some(clamp_offset_norm(
                updated,
                IPHONE_HAND_ALIGNMENT_MAX_OFFSET_M,
            ));
            inner.iphone_hand_alignment_track_id = selected_track_id;
            return;
        }
    }

    inner.iphone_hand_alignment_track_id = selected_track_id;
}

fn count_visible_iphone_hands(source: &OperatorSourcePose) -> usize {
    usize::from(source.left_hand_fresh && hand_wrist_from_canonical(source, true).is_some())
        + usize::from(source.right_hand_fresh && hand_wrist_from_canonical(source, false).is_some())
}

fn canonical_hand_has_points(points: &[[f32; 3]], is_left: bool) -> bool {
    if is_left {
        points.get(0).is_some_and(is_present_point3)
    } else {
        points.get(21).is_some_and(is_present_point3)
    }
}

fn vision_hand_is_fresh(source: &OperatorSourcePose, is_left: bool) -> bool {
    if is_left {
        source.left_hand_fresh
    } else {
        source.right_hand_fresh
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct HandMatchGap {
    gap_m: f32,
    used_shoulder_fallback: bool,
    trusted: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct HandMatchCandidate {
    left_gap: Option<HandMatchGap>,
    right_gap: Option<HandMatchGap>,
}

impl HandMatchCandidate {
    fn trusted_count(&self) -> usize {
        usize::from(self.left_gap.is_some_and(|gap| gap.trusted))
            + usize::from(self.right_gap.is_some_and(|gap| gap.trusted))
    }

    fn mean_gap(&self) -> Option<f32> {
        let valid = self.valid_gaps();
        (!valid.is_empty()).then(|| valid.iter().sum::<f32>() / valid.len() as f32)
    }

    fn mean_trusted_gap(&self) -> Option<f32> {
        let trusted = self.trusted_gaps();
        (!trusted.is_empty()).then(|| trusted.iter().sum::<f32>() / trusted.len() as f32)
    }

    fn valid_gaps(&self) -> Vec<f32> {
        [self.left_gap, self.right_gap]
            .into_iter()
            .flatten()
            .map(|gap| gap.gap_m)
            .collect()
    }

    fn trusted_gaps(&self) -> Vec<f32> {
        [self.left_gap, self.right_gap]
            .into_iter()
            .flatten()
            .filter(|gap| gap.trusted)
            .map(|gap| gap.gap_m)
            .collect()
    }

    fn left_gap_value(&self) -> Option<f32> {
        self.left_gap.map(|gap| gap.gap_m)
    }

    fn right_gap_value(&self) -> Option<f32> {
        self.right_gap.map(|gap| gap.gap_m)
    }

    fn single_gap_used_shoulder_fallback(&self) -> bool {
        if self.trusted_count() != 1 {
            return false;
        }
        self.left_gap
            .filter(|gap| gap.trusted)
            .or(self.right_gap.filter(|gap| gap.trusted))
            .map(|gap| gap.used_shoulder_fallback)
            .unwrap_or(false)
    }

    fn preferred_over(&self, other: &Self) -> bool {
        match self.trusted_count().cmp(&other.trusted_count()) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => {
                self.mean_trusted_gap()
                    .or_else(|| self.mean_gap())
                    .unwrap_or(f32::INFINITY)
                    <= other
                        .mean_trusted_gap()
                        .or_else(|| other.mean_gap())
                        .unwrap_or(f32::INFINITY)
            }
        }
    }
}

fn evaluate_hand_match_assignment(
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    mirrored: bool,
) -> HandMatchCandidate {
    let (left_vision_is_left, right_vision_is_left) = if mirrored {
        (false, true)
    } else {
        (true, false)
    };

    HandMatchCandidate {
        left_gap: match_gap_with_shoulder_fallback(stereo, vision, true, left_vision_is_left),
        right_gap: match_gap_with_shoulder_fallback(stereo, vision, false, right_vision_is_left),
    }
}

fn match_gap_with_shoulder_fallback(
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    stereo_is_left: bool,
    vision_is_left: bool,
) -> Option<HandMatchGap> {
    wrist_gap_m(stereo, vision, stereo_is_left, vision_is_left)
        .map(|gap_m| HandMatchGap {
            gap_m,
            used_shoulder_fallback: false,
            trusted: gap_m <= IPHONE_HAND_MATCH_MAX_GAP_M,
        })
        .or_else(|| {
            shoulder_gap_m(stereo, vision, stereo_is_left, vision_is_left).map(|gap_m| {
                HandMatchGap {
                    gap_m,
                    used_shoulder_fallback: true,
                    trusted: gap_m <= IPHONE_HAND_SHOULDER_FALLBACK_MAX_GAP_M,
                }
            })
        })
}

fn wrist_gap_m(
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    stereo_is_left: bool,
    vision_is_left: bool,
) -> Option<f32> {
    if !vision_hand_is_fresh(vision, vision_is_left) {
        return None;
    }
    let stereo_wrist = body_wrist_from_canonical(stereo, stereo_is_left)?;
    let iphone_wrist = hand_wrist_from_canonical(vision, vision_is_left)?;
    Some(dist3(stereo_wrist, iphone_wrist))
}

fn body_wrist_from_canonical(source: &OperatorSourcePose, is_left: bool) -> Option<[f32; 3]> {
    let body_index = if is_left { 9 } else { 10 };
    source
        .canonical_body_kpts_3d
        .get(body_index)
        .copied()
        .filter(is_present_point3)
}

fn shoulder_gap_m(
    stereo: &OperatorSourcePose,
    vision: &OperatorSourcePose,
    stereo_is_left: bool,
    vision_is_left: bool,
) -> Option<f32> {
    if !vision_hand_is_fresh(vision, vision_is_left) {
        return None;
    }
    let stereo_shoulder = body_shoulder_from_canonical(stereo, stereo_is_left)?;
    let iphone_wrist = hand_wrist_from_canonical(vision, vision_is_left)?;
    Some(dist3(stereo_shoulder, iphone_wrist))
}

fn body_shoulder_from_canonical(source: &OperatorSourcePose, is_left: bool) -> Option<[f32; 3]> {
    let body_index = if is_left { 5 } else { 6 };
    source
        .canonical_body_kpts_3d
        .get(body_index)
        .copied()
        .filter(is_present_point3)
}

fn hand_wrist_from_canonical(source: &OperatorSourcePose, is_left: bool) -> Option<[f32; 3]> {
    let hand_index = if is_left { 0 } else { 21 };
    source
        .canonical_hand_kpts_3d
        .get(hand_index)
        .copied()
        .filter(is_present_point3)
}

fn primary_raw_source<'a>(
    stereo_count: usize,
    vision_count: usize,
    stereo: &'a OperatorSourcePose,
    vision: &'a OperatorSourcePose,
) -> Option<&'a OperatorSourcePose> {
    match (stereo_count > 0, vision_count > 0) {
        (true, true) if stereo_count >= vision_count => Some(stereo),
        (true, true) => Some(vision),
        (true, false) => Some(stereo),
        (false, true) => Some(vision),
        (false, false) => None,
    }
}

fn select_hand_curl_fallback(
    stereo: Option<[f32; 5]>,
    vision: Option<[f32; 5]>,
    hand_result: &FusedPointCloud,
) -> Option<[f32; 5]> {
    match (hand_result.stereo_count > 0, hand_result.vision_count > 0) {
        (true, true) if hand_result.stereo_count >= hand_result.vision_count => stereo.or(vision),
        (true, true) => vision.or(stereo),
        (true, false) => stereo,
        (false, true) => vision,
        (false, false) => None,
    }
}

fn is_present_point2(point: &[f32; 2]) -> bool {
    point[0].is_finite() && point[1].is_finite() && (point[0].abs() > 1e-6 || point[1].abs() > 1e-6)
}

fn is_present_point3(point: &[f32; 3]) -> bool {
    point[0].is_finite()
        && point[1].is_finite()
        && point[2].is_finite()
        && (point[0].abs() > 1e-6 || point[1].abs() > 1e-6 || point[2].abs() > 1e-6)
}

#[cfg(test)]
fn build_from_3d(
    body_kpts_3d: Vec<[f32; 3]>,
    hand_kpts_3d: Vec<[f32; 3]>,
    body_layout: BodyKeypointLayout,
    hand_layout: HandKeypointLayout,
    _edge_time_ns: u64,
) -> (OperatorState, Option<[f32; 5]>, Option<[f32; 5]>) {
    let body_kpts_3d = canonicalize_body_points_3d(&body_kpts_3d, body_layout);
    let hand_kpts_3d = canonicalize_hand_points_3d(&hand_kpts_3d, hand_layout);
    build_from_canonical_3d(body_kpts_3d, hand_kpts_3d)
}

fn build_from_canonical_3d(
    body_kpts_3d: Vec<[f32; 3]>,
    hand_kpts_3d: Vec<[f32; 3]>,
) -> (OperatorState, Option<[f32; 5]>, Option<[f32; 5]>) {
    let left_pose = hand_pose_from_kpts_3d(&hand_kpts_3d, 0, true);
    let right_pose = hand_pose_from_kpts_3d(&hand_kpts_3d, 21, false);
    let left_pose = if pose_is_uninitialized(&left_pose) {
        body_wrist_pose_from_body_kpts_3d(&body_kpts_3d, true).unwrap_or(left_pose)
    } else {
        left_pose
    };
    let right_pose = if pose_is_uninitialized(&right_pose) {
        body_wrist_pose_from_body_kpts_3d(&body_kpts_3d, false).unwrap_or(right_pose)
    } else {
        right_pose
    };

    let left_curls = finger_curls_from_hand_kpts_3d(&hand_kpts_3d, 0);
    let right_curls = finger_curls_from_hand_kpts_3d(&hand_kpts_3d, 21);

    (
        OperatorState {
            body_kpts_3d,
            hand_kpts_3d,
            end_effector_pose: EndEffectorPose {
                left: left_pose,
                right: right_pose,
            },
        },
        left_curls,
        right_curls,
    )
}

pub fn operator_state_from_canonical_3d(
    body_kpts_3d: Vec<[f32; 3]>,
    hand_kpts_3d: Vec<[f32; 3]>,
) -> (OperatorState, Option<[f32; 5]>, Option<[f32; 5]>) {
    build_from_canonical_3d(body_kpts_3d, hand_kpts_3d)
}

pub fn hand_curls_from_vision_2d(
    hand_kpts_2d: &[[f32; 2]],
    hand_layout: HandKeypointLayout,
    image_w: Option<u32>,
    image_h: Option<u32>,
    is_left: bool,
) -> Option<[f32; 5]> {
    if hand_kpts_2d.is_empty() {
        return None;
    }
    let hand2d_norm = normalize_2d_kpts(hand_kpts_2d, image_w, image_h);
    let hand2d_canonical = canonicalize_hand_points_2d(&hand2d_norm, hand_layout);
    let base = if hand2d_canonical.len() >= 42 && !is_left {
        21
    } else {
        0
    };
    finger_curls_from_hand_kpts_2d(&hand2d_canonical, base)
}

fn pose_is_uninitialized(pose: &Pose) -> bool {
    pose.pos == [0.0, 0.0, 0.0] && pose.quat == [0.0, 0.0, 0.0, 1.0]
}

fn body_wrist_pose_from_body_kpts_3d(body_kpts_3d: &[[f32; 3]], is_left: bool) -> Option<Pose> {
    let wrist_index = if is_left { 9 } else { 10 };
    let elbow_index = if is_left { 7 } else { 8 };
    let same_shoulder_index = if is_left { 5 } else { 6 };
    let other_shoulder_index = if is_left { 6 } else { 5 };

    let wrist = *body_kpts_3d.get(wrist_index)?;
    let elbow = *body_kpts_3d.get(elbow_index)?;
    let same_shoulder = *body_kpts_3d.get(same_shoulder_index)?;
    let other_shoulder = *body_kpts_3d.get(other_shoulder_index)?;
    if !is_finite3(wrist)
        || !is_finite3(elbow)
        || !is_finite3(same_shoulder)
        || !is_finite3(other_shoulder)
    {
        return None;
    }

    let forearm = normalize(sub(wrist, elbow));
    let shoulder_axis = normalize(sub(other_shoulder, same_shoulder));
    let normal = normalize(cross(shoulder_axis, forearm));
    let lateral = normalize(cross(forearm, normal));
    if !is_finite3(forearm) || !is_finite3(lateral) || !is_finite3(normal) {
        return Some(Pose {
            pos: wrist,
            quat: [0.0, 0.0, 0.0, 1.0],
        });
    }

    let rot = [
        [lateral[0], forearm[0], normal[0]],
        [lateral[1], forearm[1], normal[1]],
        [lateral[2], forearm[2], normal[2]],
    ];
    Some(Pose {
        pos: wrist,
        quat: quat_from_rot(rot),
    })
}

#[cfg(test)]
fn canonicalize_body_points_2d(kpts: &[[f32; 2]], layout: BodyKeypointLayout) -> Vec<[f32; 2]> {
    canonicalize_body_points(kpts, layout)
}

pub(crate) fn canonicalize_body_points_3d(
    kpts: &[[f32; 3]],
    layout: BodyKeypointLayout,
) -> Vec<[f32; 3]> {
    canonicalize_body_points(kpts, layout)
}

pub fn canonical_body_points_3d(kpts: &[[f32; 3]], layout: BodyKeypointLayout) -> Vec<[f32; 3]> {
    canonicalize_body_points_3d(kpts, layout)
}

const COCO_NOSE_INDEX: usize = 0;
const COCO_LEFT_SHOULDER_INDEX: usize = 5;
const COCO_RIGHT_SHOULDER_INDEX: usize = 6;
const COCO_LEFT_ELBOW_INDEX: usize = 7;
const COCO_RIGHT_ELBOW_INDEX: usize = 8;
const COCO_LEFT_WRIST_INDEX: usize = 9;
const COCO_RIGHT_WRIST_INDEX: usize = 10;
const COCO_LEFT_HIP_INDEX: usize = 11;
const COCO_RIGHT_HIP_INDEX: usize = 12;
const COCO_LEFT_KNEE_INDEX: usize = 13;
const COCO_RIGHT_KNEE_INDEX: usize = 14;
const COCO_LEFT_ANKLE_INDEX: usize = 15;
const COCO_RIGHT_ANKLE_INDEX: usize = 16;

fn canonicalize_body_points<const D: usize>(
    kpts: &[[f32; D]],
    layout: BodyKeypointLayout,
) -> Vec<[f32; D]> {
    match layout {
        BodyKeypointLayout::PicoBody24 if kpts.len() == 24 => {
            remap_points(kpts, &PICO_BODY24_TO_COCO17)
        }
        _ => kpts.to_vec(),
    }
}

fn present_point(points: &[[f32; 3]], index: usize) -> Option<[f32; 3]> {
    points.get(index).copied().filter(is_present_point3)
}

fn mean_axis(points: &[[f32; 3]], indices: &[usize], axis: usize) -> Option<f32> {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for index in indices {
        if let Some(point) = present_point(points, *index) {
            sum += point[axis];
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f32)
}

pub(crate) fn stabilize_wifi_canonical_body_points(points: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut stabilized = points.to_vec();
    if stabilized.is_empty() {
        return stabilized;
    }

    let left_shoulder = present_point(&stabilized, COCO_LEFT_SHOULDER_INDEX);
    let right_shoulder = present_point(&stabilized, COCO_RIGHT_SHOULDER_INDEX);
    let left_hip = present_point(&stabilized, COCO_LEFT_HIP_INDEX);
    let right_hip = present_point(&stabilized, COCO_RIGHT_HIP_INDEX);
    let nose = present_point(&stabilized, COCO_NOSE_INDEX);

    let shoulders_mean_y = mean_axis(
        &stabilized,
        &[COCO_LEFT_SHOULDER_INDEX, COCO_RIGHT_SHOULDER_INDEX],
        1,
    );
    let hips_mean_y = mean_axis(&stabilized, &[COCO_LEFT_HIP_INDEX, COCO_RIGHT_HIP_INDEX], 1);

    let flip_x = match (left_shoulder, right_shoulder) {
        (Some(left), Some(right)) => left[0] < right[0],
        _ => match (left_hip, right_hip) {
            (Some(left), Some(right)) => left[0] < right[0],
            _ => false,
        },
    };

    let flip_y = match (shoulders_mean_y, hips_mean_y) {
        (Some(shoulders), Some(hips)) => shoulders > hips,
        _ => match nose {
            Some(nose_point) => hips_mean_y.is_some_and(|hips| nose_point[1] > hips),
            None => false,
        },
    };

    if flip_x || flip_y {
        for point in &mut stabilized {
            if !is_present_point3(point) {
                continue;
            }
            if flip_x {
                point[0] = -point[0];
            }
            if flip_y {
                point[1] = -point[1];
            }
        }
    }

    stabilized
}

fn canonicalize_hand_points_2d(kpts: &[[f32; 2]], layout: HandKeypointLayout) -> Vec<[f32; 2]> {
    canonicalize_hand_points(kpts, layout)
}

pub(crate) fn canonicalize_hand_points_3d(
    kpts: &[[f32; 3]],
    layout: HandKeypointLayout,
) -> Vec<[f32; 3]> {
    canonicalize_hand_points(kpts, layout)
}

pub fn canonical_hand_points_3d(kpts: &[[f32; 3]], layout: HandKeypointLayout) -> Vec<[f32; 3]> {
    canonicalize_hand_points_3d(kpts, layout)
}

fn canonicalize_hand_points<const D: usize>(
    kpts: &[[f32; D]],
    layout: HandKeypointLayout,
) -> Vec<[f32; D]> {
    match layout {
        HandKeypointLayout::PicoHand26 => remap_hand_points(kpts, 26, &PICO_HAND26_TO_MEDIAPIPE21),
        _ => kpts.to_vec(),
    }
}

fn remap_points<const D: usize>(kpts: &[[f32; D]], mapping: &[usize]) -> Vec<[f32; D]> {
    if mapping.iter().any(|index| *index >= kpts.len()) {
        return kpts.to_vec();
    }
    mapping.iter().map(|index| kpts[*index]).collect()
}

fn remap_hand_points<const D: usize>(
    kpts: &[[f32; D]],
    source_hand_len: usize,
    mapping: &[usize],
) -> Vec<[f32; D]> {
    if kpts.is_empty() {
        return Vec::new();
    }
    if kpts.len() % source_hand_len != 0 {
        return kpts.to_vec();
    }

    let mut out = Vec::with_capacity((kpts.len() / source_hand_len) * mapping.len());
    for chunk in kpts.chunks_exact(source_hand_len) {
        if mapping.iter().any(|index| *index >= chunk.len()) {
            return kpts.to_vec();
        }
        out.extend(mapping.iter().map(|index| chunk[*index]));
    }
    out
}

fn normalize_2d_kpts(
    kpts: &[[f32; 2]],
    image_w: Option<u32>,
    image_h: Option<u32>,
) -> Vec<[f32; 2]> {
    if kpts.is_empty() {
        return Vec::new();
    }

    let mut max_abs = 0.0f32;
    for [x, y] in kpts {
        max_abs = max_abs.max(x.abs()).max(y.abs());
    }

    // 经验规则：
    // - <=1.5：大概率是归一化坐标（0..1 或 -1..1），直接使用
    // - 否则：当作像素坐标，优先按 image_w/h 归一化
    if max_abs <= 1.5 {
        return kpts.to_vec();
    }

    if let (Some(w), Some(h)) = (image_w, image_h) {
        let wf = (w as f32).max(1.0);
        let hf = (h as f32).max(1.0);
        return kpts
            .iter()
            .map(|[x, y]| [(*x / wf).clamp(0.0, 1.0), (*y / hf).clamp(0.0, 1.0)])
            .collect();
    }

    // fallback：按 max_abs 做粗归一化（仅用于联调兜底）
    let denom = max_abs.max(1.0);
    kpts.iter()
        .map(|[x, y]| [(*x / denom).clamp(0.0, 1.0), (*y / denom).clamp(0.0, 1.0)])
        .collect()
}

fn project_2d_kpts_to_3d(
    kpts_2d: &[[f32; 2]],
    image_w: Option<u32>,
    image_h: Option<u32>,
    depth_z_mean_m: Option<f32>,
    cfg: &Config,
) -> Vec<[f32; 3]> {
    let norm = normalize_2d_kpts(kpts_2d, image_w, image_h);
    let z = depth_z_mean_m
        .filter(|x| x.is_finite() && *x > 0.0)
        .unwrap_or(cfg.vision_proj_z_base_m)
        .max(0.0);
    norm.iter()
        .map(|[x, y]| {
            if !is_present_point2(&[*x, *y]) {
                return [0.0, 0.0, 0.0];
            }
            let xm = (x - 0.5) * cfg.vision_proj_x_span_m;
            let ym = (0.5 - y) * cfg.vision_proj_y_span_m;
            [xm, ym, z]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::config::Config;

    use super::{
        add3, apply_confirmed_wearer_sticky, build_from_3d, build_fused_estimate,
        build_fused_estimate_with_wifi, build_motion_state_observations,
        build_single_source_estimate, build_source_plus_wifi_estimate,
        build_wifi_body_with_vision_hands_estimate, canonicalize_body_points_2d,
        canonicalize_body_points_3d, canonicalize_hand_points_2d, canonicalize_hand_points_3d,
        compute_wifi_association_score, derive_csi_prior_observation, derive_operator_association,
        derive_part_source, dist3, estimate_body_root, fuse_canonical_points,
        fuse_operator_sources, hold_estimate_with_motion_prior, optimize_operator_motion_state,
        point_coverage_2d, record_motion_state_observations,
        record_track_motion_state_observations, record_track_motion_state_snapshot,
        refine_fused_body_geometry, resolve_active_stereo_track_id, scoped_motion_state_context,
        source_pose_from_vision, source_pose_from_wifi, stabilize_operator_estimate,
        stabilize_wifi_canonical_body_points, sub, update_operator_motion_state, wrap_angle_rad,
        BodyKeypointLayout, CsiSnapshot, FusedPointCloud, HandKeypointLayout,
        MotionObservationKind, MotionStateObservation, OperatorEstimate, OperatorFusionBreakdown,
        OperatorInner, OperatorMotionState, OperatorPartSource, OperatorRawPose,
        OperatorSmootherMode, OperatorSource, OperatorSourcePose, VisionSnapshot,
        WifiPoseDiagnostics, WifiPoseSnapshot, BODY_BLEND_MAX_GAP_M, CANONICAL_BODY_FRAME,
        COCO_LEFT_ELBOW_INDEX, COCO_LEFT_HIP_INDEX, COCO_LEFT_KNEE_INDEX, COCO_LEFT_SHOULDER_INDEX,
        COCO_LEFT_WRIST_INDEX, COCO_RIGHT_ELBOW_INDEX, COCO_RIGHT_HIP_INDEX, COCO_RIGHT_KNEE_INDEX,
        COCO_RIGHT_SHOULDER_INDEX, COCO_RIGHT_WRIST_INDEX, OPERATOR_FRAME,
        PROJECTED_VISION_WEIGHT_SCALE, STEREO_PAIR_FRAME,
    };
    use std::collections::VecDeque;

    #[test]
    fn pico_body_24_should_map_to_coco_17_indices() {
        let input: Vec<[f32; 3]> = (0..24).map(|index| [index as f32, 0.0, 0.0]).collect();
        let output = canonicalize_body_points_3d(&input, BodyKeypointLayout::PicoBody24);

        assert_eq!(output.len(), 17);
        assert_eq!(output[5], [16.0, 0.0, 0.0]);
        assert_eq!(output[6], [17.0, 0.0, 0.0]);
        assert_eq!(output[9], [20.0, 0.0, 0.0]);
        assert_eq!(output[10], [21.0, 0.0, 0.0]);
        assert_eq!(output[11], [1.0, 0.0, 0.0]);
        assert_eq!(output[12], [2.0, 0.0, 0.0]);
    }

    #[test]
    fn pico_hand_26_should_map_to_mediapipe_21_indices() {
        let input: Vec<[f32; 3]> = (0..26).map(|index| [index as f32, 0.0, 0.0]).collect();
        let output = canonicalize_hand_points_3d(&input, HandKeypointLayout::PicoHand26);

        assert_eq!(output.len(), 21);
        assert_eq!(output[0], [1.0, 0.0, 0.0]);
        assert_eq!(output[1], [2.0, 0.0, 0.0]);
        assert_eq!(output[5], [7.0, 0.0, 0.0]);
        assert_eq!(output[8], [10.0, 0.0, 0.0]);
        assert_eq!(output[17], [22.0, 0.0, 0.0]);
        assert_eq!(output[20], [25.0, 0.0, 0.0]);
    }

    #[test]
    fn pico_hand_52_should_map_dual_hands_without_crossing_offsets() {
        let input: Vec<[f32; 2]> = (0..52).map(|index| [index as f32, 0.0]).collect();
        let output = canonicalize_hand_points_2d(&input, HandKeypointLayout::PicoHand26);

        assert_eq!(output.len(), 42);
        assert_eq!(output[0], [1.0, 0.0]);
        assert_eq!(output[20], [25.0, 0.0]);
        assert_eq!(output[21], [27.0, 0.0]);
        assert_eq!(output[41], [51.0, 0.0]);
    }

    #[test]
    fn build_from_pico_layout_should_emit_canonical_targets() {
        let body = build_pico_body_points();
        let hand = build_pico_dual_hand_points();

        let (state, left_curls, right_curls) = build_from_3d(
            body,
            hand,
            BodyKeypointLayout::PicoBody24,
            HandKeypointLayout::PicoHand26,
            0,
        );

        assert_eq!(state.body_kpts_3d.len(), 17);
        assert_eq!(state.hand_kpts_3d.len(), 42);
        assert!(left_curls.is_some());
        assert!(right_curls.is_some());
        assert_ne!(state.end_effector_pose.left.pos, [0.0, 0.0, 0.0]);
        assert_ne!(state.end_effector_pose.right.pos, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn build_from_body_only_should_derive_wrist_pose_from_body_keypoints() {
        let body = build_pico_body_points();

        let (state, left_curls, right_curls) = build_from_3d(
            body,
            Vec::new(),
            BodyKeypointLayout::PicoBody24,
            HandKeypointLayout::Unknown,
            0,
        );

        assert_eq!(state.body_kpts_3d.len(), 17);
        assert!(state.hand_kpts_3d.is_empty());
        assert!(left_curls.is_none());
        assert!(right_curls.is_none());
        assert_eq!(state.end_effector_pose.left.pos, state.body_kpts_3d[9]);
        assert_eq!(state.end_effector_pose.right.pos, state.body_kpts_3d[10]);
        assert_ne!(state.end_effector_pose.left.quat, [0.0, 0.0, 0.0, 1.0]);
        assert_ne!(state.end_effector_pose.right.quat, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn pico_body_24_2d_should_map_to_coco_17_indices() {
        let input: Vec<[f32; 2]> = (0..24).map(|index| [index as f32, 0.0]).collect();
        let output = canonicalize_body_points_2d(&input, BodyKeypointLayout::PicoBody24);

        assert_eq!(output.len(), 17);
        assert_eq!(output[5], [16.0, 0.0]);
        assert_eq!(output[6], [17.0, 0.0]);
    }

    #[test]
    fn fuse_canonical_points_should_fill_missing_indices_from_vision() {
        let stereo = vec![[0.05, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let vision = vec![[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.0, 0.0, 0.0]];

        let fused = fuse_canonical_points(&stereo, &vision, 1.0, 0.8, 0.2);

        assert_eq!(fused.points.len(), 3);
        assert!(fused.points[0][0] > 0.0 && fused.points[0][0] < 0.1);
        assert!(fused.points[1][0] > 1.0 && fused.points[1][0] < 1.1);
        assert_eq!(fused.points[2], [2.0, 0.0, 0.0]);
        assert_eq!(fused.stereo_count, 2);
        assert_eq!(fused.vision_count, 3);
        assert_eq!(fused.blended_count, 2);
    }

    #[test]
    fn fuse_canonical_points_should_ignore_zero_placeholders() {
        let stereo = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let vision = vec![[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]];

        let fused = fuse_canonical_points(&stereo, &vision, 1.0, 0.8, 0.2);

        assert_eq!(fused.points.len(), 2);
        assert_eq!(fused.points[0], [0.0, 0.0, 0.0]);
        assert!(fused.points[1][0] > 1.0 && fused.points[1][0] < 1.1);
        assert_eq!(fused.stereo_count, 1);
        assert_eq!(fused.vision_count, 1);
        assert_eq!(fused.blended_count, 1);
    }

    #[test]
    fn refine_fused_body_geometry_should_pull_stretched_limbs_back_toward_stereo_lengths() {
        let stereo =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        let mut vision = stereo.clone();
        vision[COCO_LEFT_SHOULDER_INDEX][0] += 0.12;
        vision[COCO_LEFT_ELBOW_INDEX][0] += 0.12;
        vision[COCO_LEFT_WRIST_INDEX][0] += 0.14;
        vision[COCO_RIGHT_SHOULDER_INDEX][0] -= 0.12;
        vision[COCO_RIGHT_ELBOW_INDEX][0] -= 0.12;
        vision[COCO_RIGHT_WRIST_INDEX][0] -= 0.14;
        vision[COCO_LEFT_KNEE_INDEX][2] += 0.11;
        vision[COCO_RIGHT_KNEE_INDEX][2] += 0.11;

        let fused = fuse_canonical_points(&stereo, &vision, 1.0, 0.9, BODY_BLEND_MAX_GAP_M);
        let refined = refine_fused_body_geometry(&stereo, &vision, &fused.points);

        let stereo_shoulder_span = dist3(
            stereo[COCO_LEFT_SHOULDER_INDEX],
            stereo[COCO_RIGHT_SHOULDER_INDEX],
        );
        let fused_shoulder_span = dist3(
            fused.points[COCO_LEFT_SHOULDER_INDEX],
            fused.points[COCO_RIGHT_SHOULDER_INDEX],
        );
        let refined_shoulder_span = dist3(
            refined[COCO_LEFT_SHOULDER_INDEX],
            refined[COCO_RIGHT_SHOULDER_INDEX],
        );
        assert!(
            (refined_shoulder_span - stereo_shoulder_span).abs()
                < (fused_shoulder_span - stereo_shoulder_span).abs()
        );
    }

    #[test]
    fn build_fused_estimate_should_preserve_plausible_body_geometry() {
        let canonical =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.canonical_body_kpts_3d = canonical.clone();

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.canonical_body_kpts_3d = canonical.clone();
        vision.canonical_body_kpts_3d[COCO_LEFT_WRIST_INDEX][0] += 0.01;
        vision.canonical_body_kpts_3d[COCO_RIGHT_WRIST_INDEX][0] -= 0.01;
        vision.canonical_body_kpts_3d[COCO_LEFT_KNEE_INDEX][2] += 0.01;
        vision.canonical_body_kpts_3d[COCO_RIGHT_KNEE_INDEX][2] += 0.01;

        let estimate = build_fused_estimate(456, &stereo, &vision);

        let body = &estimate.operator_state.body_kpts_3d;
        assert_eq!(body.len(), 17);
        assert!(
            (dist3(
                body[COCO_LEFT_SHOULDER_INDEX],
                body[COCO_RIGHT_SHOULDER_INDEX]
            ) - dist3(
                canonical[COCO_LEFT_SHOULDER_INDEX],
                canonical[COCO_RIGHT_SHOULDER_INDEX],
            ))
            .abs()
                < 0.03
        );
        assert!(
            (dist3(body[COCO_LEFT_ELBOW_INDEX], body[COCO_LEFT_WRIST_INDEX])
                - dist3(
                    canonical[COCO_LEFT_ELBOW_INDEX],
                    canonical[COCO_LEFT_WRIST_INDEX],
                ))
            .abs()
                < 0.03
        );
        assert!(
            (dist3(body[COCO_LEFT_HIP_INDEX], body[COCO_LEFT_KNEE_INDEX])
                - dist3(
                    canonical[COCO_LEFT_HIP_INDEX],
                    canonical[COCO_LEFT_KNEE_INDEX]
                ))
            .abs()
                < 0.03
        );
    }

    #[test]
    fn derive_part_source_should_report_joint_level_mixed_source() {
        let stereo = test_source_pose(OperatorSource::Stereo);
        let vision = test_source_pose(OperatorSource::Vision3d);
        let result = FusedPointCloud {
            points: vec![[0.0, 0.0, 0.0]],
            stereo_count: 6,
            vision_count: 3,
            blended_count: 2,
        };

        let source = derive_part_source(&vision, &result);

        assert_eq!(source.as_str(), "fused_stereo_vision_3d");
        assert_eq!(stereo.source.as_str(), "stereo");
    }

    #[test]
    fn point_coverage_2d_should_treat_zero_points_as_missing() {
        let points = vec![[0.0, 0.0], [0.6, 0.4], [0.0, 0.0], [0.3, 0.2]];

        let coverage = point_coverage_2d(&points);

        assert!((coverage - 0.5).abs() < 1e-6);
    }

    #[test]
    fn source_pose_from_vision_should_ignore_all_zero_projected_hands() {
        let cfg = Config::from_env().expect("config");
        let vision = VisionSnapshot {
            fresh: true,
            hand_conf: 0.82,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            hand_kpts_2d: vec![[0.0, 0.0]; 42],
            image_w: Some(1280),
            image_h: Some(720),
            ..VisionSnapshot::default()
        };

        let source = source_pose_from_vision(&vision, &cfg, None);

        assert!(source.is_none());
    }

    #[test]
    fn source_pose_from_vision_should_downweight_single_hand_projected_input() {
        let cfg = Config::from_env().expect("config");
        let mut hand_kpts_2d = vec![[0.0, 0.0]; 42];
        for (index, point) in hand_kpts_2d.iter_mut().take(21).enumerate() {
            *point = [0.55 + index as f32 * 0.005, 0.35 + index as f32 * 0.004];
        }
        let vision = VisionSnapshot {
            fresh: true,
            hand_conf: 1.0,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            hand_kpts_2d,
            image_w: Some(1280),
            image_h: Some(720),
            ..VisionSnapshot::default()
        };

        let source = source_pose_from_vision(&vision, &cfg, None).expect("vision source");

        assert_eq!(source.source, OperatorSource::Vision2dProjected);
        assert!((source.hand_weight - (PROJECTED_VISION_WEIGHT_SCALE * 0.5)).abs() < 1e-6);
        assert!(source.canonical_body_kpts_3d.is_empty());
        assert_eq!(source.canonical_hand_kpts_3d.len(), 42);
        assert_eq!(source.canonical_hand_kpts_3d[21], [0.0, 0.0, 0.0]);
        assert!(source.canonical_hand_kpts_3d[0][2] > 0.0);
    }

    #[test]
    fn source_pose_from_vision_should_drop_body_points_and_keep_hands_only() {
        let cfg = Config::from_env().expect("config");
        let vision = VisionSnapshot {
            fresh: true,
            body_conf: 0.95,
            hand_conf: 0.81,
            body_layout: BodyKeypointLayout::CocoBody17,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            body_kpts_3d: vec![[0.1, 0.2, 0.8]; 17],
            hand_kpts_3d: vec![[0.2, 0.1, 0.6]; 42],
            ..VisionSnapshot::default()
        };

        let source = source_pose_from_vision(&vision, &cfg, None).expect("vision source");

        assert_eq!(source.source, OperatorSource::Vision3d);
        assert_eq!(source.canonical_body_kpts_3d.len(), 17);
        assert_eq!(source.raw_pose.body_layout, BodyKeypointLayout::CocoBody17);
        assert_eq!(source.raw_pose.body_kpts_3d.len(), 17);
        assert_eq!(source.canonical_hand_kpts_3d.len(), 42);
    }

    #[test]
    fn source_pose_from_vision_marks_depth_reprojected_hands_untrusted_for_geometry() {
        let cfg = Config::from_env().expect("config");
        let vision = VisionSnapshot {
            fresh: true,
            hand_conf: 0.81,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            hand_kpts_3d: vec![[0.2, 0.1, 0.6]; 42],
            hand_3d_source: "depth_reprojected".to_string(),
            execution_mode: "local_recognize".to_string(),
            ..VisionSnapshot::default()
        };

        let source = source_pose_from_vision(&vision, &cfg, None).expect("vision source");

        assert_eq!(source.source, OperatorSource::Vision3d);
        assert!(!source.hand_geometry_trusted);
    }

    #[test]
    fn source_pose_from_vision_should_treat_present_3d_hands_as_fresh_even_if_flags_are_false() {
        let cfg = Config::from_env().expect("config");
        let vision = VisionSnapshot {
            fresh: true,
            hand_conf: 0.81,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            hand_kpts_3d: build_pico_dual_hand_points(),
            left_hand_fresh_3d: false,
            right_hand_fresh_3d: false,
            ..VisionSnapshot::default()
        };

        let source = source_pose_from_vision(&vision, &cfg, None).expect("vision source");

        assert_eq!(source.source, OperatorSource::Vision3d);
        assert!(source.left_hand_fresh);
        assert!(source.right_hand_fresh);
    }

    #[test]
    fn source_pose_from_vision_should_treat_present_projected_hands_as_fresh_even_if_flags_are_false(
    ) {
        let cfg = Config::from_env().expect("config");
        let mut hand_kpts_2d = vec![[0.0, 0.0]; 42];
        for (index, point) in hand_kpts_2d.iter_mut().take(42).enumerate() {
            let hand_offset = if index < 21 { 0.0 } else { 0.18 };
            *point = [
                0.32 + hand_offset + (index % 21) as f32 * 0.006,
                0.28 + (index % 21) as f32 * 0.004,
            ];
        }
        let vision = VisionSnapshot {
            fresh: true,
            hand_conf: 0.88,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            hand_kpts_2d,
            image_w: Some(1280),
            image_h: Some(720),
            depth_z_mean_m: Some(0.78),
            left_hand_fresh_3d: false,
            right_hand_fresh_3d: false,
            ..VisionSnapshot::default()
        };

        let source = source_pose_from_vision(&vision, &cfg, None).expect("vision source");

        assert_eq!(source.source, OperatorSource::Vision2dProjected);
        assert!(source.left_hand_fresh);
        assert!(source.right_hand_fresh);
    }

    #[test]
    fn source_pose_from_wifi_should_keep_meter_scale_body_points() {
        let wifi = WifiPoseSnapshot {
            fresh: true,
            body_confidence: 0.82,
            operator_track_id: Some("wifi-main".to_string()),
            body_kpts_3d: build_pico_body_points(),
            body_layout: BodyKeypointLayout::PicoBody24,
            last_edge_time_ns: 42,
            ..WifiPoseSnapshot::default()
        };

        let source = source_pose_from_wifi(&wifi, None).expect("wifi source");

        assert_eq!(source.source, OperatorSource::WifiPose3d);
        assert_eq!(source.canonical_body_kpts_3d.len(), 17);
        assert_eq!(source.operator_track_id.as_deref(), Some("wifi-main"));
    }

    #[test]
    fn stabilize_wifi_canonical_body_points_should_unflip_inverted_pose() {
        let mut points =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        for point in &mut points {
            point[0] = -point[0];
            point[1] = -point[1];
        }

        let stabilized = stabilize_wifi_canonical_body_points(&points);
        let left_shoulder = stabilized[COCO_LEFT_SHOULDER_INDEX];
        let right_shoulder = stabilized[COCO_RIGHT_SHOULDER_INDEX];
        let shoulders_mean_y = (left_shoulder[1] + right_shoulder[1]) * 0.5;
        let hips_mean_y =
            (stabilized[COCO_LEFT_HIP_INDEX][1] + stabilized[COCO_RIGHT_HIP_INDEX][1]) * 0.5;

        assert!(left_shoulder[0] > right_shoulder[0]);
        assert!(shoulders_mean_y < hips_mean_y);
    }

    #[test]
    fn source_pose_from_wifi_should_drop_implausible_large_scale_body_points() {
        let wifi = WifiPoseSnapshot {
            fresh: true,
            body_confidence: 0.82,
            operator_track_id: Some("wifi-main".to_string()),
            body_kpts_3d: build_pico_body_points()
                .into_iter()
                .map(|point| {
                    [
                        point[0] * 150_000.0,
                        point[1] * 150_000.0,
                        point[2] * 150_000.0,
                    ]
                })
                .collect(),
            body_layout: BodyKeypointLayout::PicoBody24,
            last_edge_time_ns: 42,
            ..WifiPoseSnapshot::default()
        };

        assert!(source_pose_from_wifi(&wifi, None).is_none());
    }

    #[test]
    fn derive_csi_prior_observation_should_use_reliable_wifi_geometry_hint() {
        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.body_space = STEREO_PAIR_FRAME.to_string();
        wifi.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.93,
            zone_score: 0.82,
            zone_summary_reliable: true,
            motion_energy: 4.6,
            doppler_hz: 0.58,
            signal_quality: 0.88,
            vital_signal_quality: Some(0.72),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: STEREO_PAIR_FRAME.to_string(),
        });

        let prior = derive_csi_prior_observation(
            Some(&wifi),
            &CsiSnapshot {
                fresh: true,
                csi_conf: 0.91,
                ..CsiSnapshot::default()
            },
            123,
        )
        .expect("csi prior");

        assert!(prior.reliability > 0.7);
        assert!(prior.root_zone_center_m.is_some());
        assert!(prior.heading_yaw_rad.is_some());
        assert!(prior.motion_phase > 0.4);
        assert_eq!(prior.operator_track_id.as_deref(), Some("wifi-main"));
    }

    #[test]
    fn update_operator_motion_state_should_follow_stereo_geometry() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let estimate = build_single_source_estimate(1_000_000_000, &stereo);
        let previous = OperatorMotionState {
            root_pos_m: [0.6, -0.2, 1.1],
            root_vel_mps: [0.0, 0.0, 0.0],
            heading_yaw_rad: 1.4,
            updated_edge_time_ns: 900_000_000,
            ..OperatorMotionState::default()
        };
        let expected_root = estimate_body_root(&estimate.operator_state.body_kpts_3d).unwrap();

        let state = update_operator_motion_state(
            Some(&previous),
            &VecDeque::new(),
            Some(&estimate),
            Some(&stereo),
            None,
            1_000_000_000,
        );

        assert!(state.stereo_measurement_used);
        assert_eq!(state.smoother_mode, OperatorSmootherMode::StereoLive);
        assert_eq!(state.stereo_track_id.as_deref(), Some("stereo-main"));
        assert_eq!(state.last_good_stereo_time_ns, 1_000_000_000);
        assert!(
            dist3(state.root_pos_m, expected_root) < dist3(previous.root_pos_m, expected_root),
            "motion state root should move toward stereo measurement"
        );
    }

    #[test]
    fn hold_estimate_with_motion_prior_should_translate_previous_body() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        let previous_estimate = build_single_source_estimate(100, &stereo);
        let previous_root = estimate_body_root(&previous_estimate.operator_state.body_kpts_3d)
            .expect("previous root");
        let previous_motion = OperatorMotionState {
            root_pos_m: previous_root,
            updated_edge_time_ns: 100,
            body_presence_conf: 0.9,
            ..OperatorMotionState::default()
        };
        let current_motion = OperatorMotionState {
            root_pos_m: add3(previous_root, [0.05, 0.0, 0.0]),
            updated_edge_time_ns: 120,
            body_presence_conf: 0.8,
            smoother_mode: OperatorSmootherMode::HeldWithCsiPrior,
            ..previous_motion.clone()
        };

        let held = hold_estimate_with_motion_prior(
            &previous_estimate,
            Some(&previous_motion),
            &current_motion,
            120,
        );
        let held_root = estimate_body_root(&held.operator_state.body_kpts_3d).expect("held root");

        assert_eq!(held.source, OperatorSource::Hold);
        assert!((held_root[0] - (previous_root[0] + 0.05)).abs() < 1e-3);
    }

    #[test]
    fn optimize_operator_motion_state_should_use_recent_window_during_stereo_dropout() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        let estimate = build_single_source_estimate(1_000_000_000, &stereo);
        let csi_prior = derive_csi_prior_observation(
            Some(&test_wifi_pose_for_motion("wifi-main", [0.58, 0.0, 0.92])),
            &CsiSnapshot {
                fresh: true,
                csi_conf: 0.82,
                ..CsiSnapshot::default()
            },
            1_160_000_000,
        )
        .expect("csi prior");

        let previous = OperatorMotionState {
            root_pos_m: [0.52, 0.0, 0.9],
            root_vel_mps: [0.03, 0.0, 0.02],
            heading_yaw_rad: 0.15,
            heading_rate_radps: 0.05,
            body_presence_conf: 0.88,
            updated_edge_time_ns: 1_100_000_000,
            ..OperatorMotionState::default()
        };

        let mut history = VecDeque::new();
        record_motion_state_observations(
            &mut history,
            build_motion_state_observations(Some(&estimate), Some(&stereo), None, 1_000_000_000),
            1_000_000_000,
        );
        record_motion_state_observations(
            &mut history,
            build_motion_state_observations(None, None, Some(&csi_prior), 1_160_000_000),
            1_160_000_000,
        );

        let state = optimize_operator_motion_state(
            Some(&previous),
            &VecDeque::new(),
            &history,
            1_200_000_000,
            true,
        );
        assert_eq!(state.smoother_mode, OperatorSmootherMode::HeldWithCsiPrior);
        assert!(state.stereo_measurement_used);
        assert!(state.csi_measurement_used);
        assert_eq!(state.stereo_track_id.as_deref(), Some("stereo-main"));
        assert!(state.body_presence_conf >= 0.72);
        assert_eq!(state.last_good_stereo_time_ns, 1_000_000_000);
        assert_eq!(state.last_good_csi_time_ns, 1_160_000_000);
        assert!(state.csi_prior_reliability > 0.4);
    }

    #[test]
    fn optimize_operator_motion_state_should_fall_back_to_fixed_lag_blend_without_csi() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        let estimate = build_single_source_estimate(1_000_000_000, &stereo);
        let previous = OperatorMotionState {
            root_pos_m: [0.52, 0.0, 0.9],
            heading_yaw_rad: 0.1,
            body_presence_conf: 0.82,
            updated_edge_time_ns: 1_100_000_000,
            ..OperatorMotionState::default()
        };

        let mut history = VecDeque::new();
        record_motion_state_observations(
            &mut history,
            build_motion_state_observations(Some(&estimate), Some(&stereo), None, 1_000_000_000),
            1_000_000_000,
        );

        let state = optimize_operator_motion_state(
            Some(&previous),
            &VecDeque::new(),
            &history,
            1_200_000_000,
            true,
        );

        assert_eq!(state.smoother_mode, OperatorSmootherMode::FixedLagBlend);
        assert!(state.stereo_measurement_used);
        assert!(!state.csi_measurement_used);
        assert_eq!(state.stereo_track_id.as_deref(), Some("stereo-main"));
    }

    #[test]
    fn optimize_operator_motion_state_should_reject_large_csi_outlier() {
        let previous = OperatorMotionState {
            root_pos_m: [0.55, 0.0, 0.9],
            root_vel_mps: [0.02, 0.0, 0.01],
            root_std_m: 0.08,
            heading_yaw_rad: 0.12,
            heading_std_rad: 0.2,
            body_presence_conf: 0.9,
            updated_edge_time_ns: 1_000_000_000,
            ..OperatorMotionState::default()
        };
        let mut history = VecDeque::new();
        record_motion_state_observations(
            &mut history,
            vec![MotionStateObservation {
                edge_time_ns: 1_140_000_000,
                kind: MotionObservationKind::CsiPrior,
                root_pos_m: Some([4.8, 0.0, 4.2]),
                heading_yaw_rad: Some(2.2),
                reliability: 0.92,
                motion_phase: 0.4,
                track_id: Some("wifi-main".to_string()),
            }],
            1_140_000_000,
        );

        let state = optimize_operator_motion_state(
            Some(&previous),
            &VecDeque::new(),
            &history,
            1_200_000_000,
            true,
        );

        assert!(!state.csi_measurement_used);
        assert!(state.rejected_csi_observations > 0);
        assert!(dist3(state.root_pos_m, previous.root_pos_m) < 0.35);
        assert!(wrap_angle_rad(state.heading_yaw_rad - previous.heading_yaw_rad).abs() < 0.5);
    }

    #[test]
    fn optimize_operator_motion_state_should_replay_from_recent_state_anchor_for_delayed_csi() {
        let previous = OperatorMotionState {
            root_pos_m: [0.0, 0.0, 0.0],
            root_std_m: 0.05,
            heading_yaw_rad: 0.0,
            heading_std_rad: 0.18,
            body_presence_conf: 0.92,
            updated_edge_time_ns: 1_000_000_000,
            ..OperatorMotionState::default()
        };
        let recent_snapshot = OperatorMotionState {
            root_pos_m: [0.98, 0.0, 0.86],
            root_std_m: 0.05,
            heading_yaw_rad: 0.14,
            heading_std_rad: 0.18,
            body_presence_conf: 0.94,
            updated_edge_time_ns: 1_150_000_000,
            stereo_track_id: Some("stereo-main".to_string()),
            ..OperatorMotionState::default()
        };
        let mut state_history = VecDeque::new();
        state_history.push_back(recent_snapshot);
        let mut history = VecDeque::new();
        record_motion_state_observations(
            &mut history,
            vec![MotionStateObservation {
                edge_time_ns: 1_180_000_000,
                kind: MotionObservationKind::CsiPrior,
                root_pos_m: Some([1.05, 0.0, 0.9]),
                heading_yaw_rad: Some(0.18),
                reliability: 0.92,
                motion_phase: 0.45,
                track_id: Some("wifi-main".to_string()),
            }],
            1_180_000_000,
        );

        let state = optimize_operator_motion_state(
            Some(&previous),
            &state_history,
            &history,
            1_200_000_000,
            true,
        );

        assert!(state.csi_measurement_used);
        assert!(dist3(state.root_pos_m, [1.05, 0.0, 0.9]) < 0.25);
        assert!(dist3(state.root_pos_m, previous.root_pos_m) > 0.5);
    }

    #[test]
    fn optimize_operator_motion_state_should_reset_anchor_when_stereo_track_changes() {
        let previous = OperatorMotionState {
            root_pos_m: [0.92, 0.0, 0.9],
            root_std_m: 0.04,
            heading_yaw_rad: 0.0,
            heading_std_rad: 0.16,
            body_presence_conf: 0.96,
            updated_edge_time_ns: 1_000_000_000,
            stereo_track_id: Some("stereo-a".to_string()),
            ..OperatorMotionState::default()
        };
        let mut state_history = VecDeque::new();
        state_history.push_back(previous.clone());
        let mut history = VecDeque::new();
        record_motion_state_observations(
            &mut history,
            vec![MotionStateObservation {
                edge_time_ns: 1_180_000_000,
                kind: MotionObservationKind::StereoGeometry,
                root_pos_m: Some([-0.85, 0.0, 0.88]),
                heading_yaw_rad: Some(-0.24),
                reliability: 1.0,
                motion_phase: 0.2,
                track_id: Some("stereo-b".to_string()),
            }],
            1_180_000_000,
        );

        let state = optimize_operator_motion_state(
            Some(&previous),
            &state_history,
            &history,
            1_200_000_000,
            false,
        );

        assert!(state.stereo_measurement_used);
        assert_eq!(state.stereo_track_id.as_deref(), Some("stereo-b"));
        assert!(dist3(state.root_pos_m, [-0.85, 0.0, 0.88]) < 0.2);
        assert!(state.rejected_stereo_observations == 0);
    }

    #[test]
    fn scoped_motion_state_context_should_return_track_local_state_when_available() {
        let mut inner = OperatorInner::default();
        let scoped = OperatorMotionState {
            root_pos_m: [0.8, 0.0, 0.9],
            updated_edge_time_ns: 100,
            stereo_track_id: Some("stereo-b".to_string()),
            ..OperatorMotionState::default()
        };
        record_track_motion_state_snapshot(
            &mut inner.track_motion_states,
            &mut inner.track_motion_state_histories,
            &scoped,
            100,
        );

        let (previous, history, observations) =
            scoped_motion_state_context(&inner, Some("stereo-b"));

        assert_eq!(
            previous.unwrap().stereo_track_id.as_deref(),
            Some("stereo-b")
        );
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].stereo_track_id.as_deref(), Some("stereo-b"));
        assert!(observations.is_empty());
    }

    #[test]
    fn resolve_active_stereo_track_id_should_prefer_estimate_then_stereo_then_previous() {
        let mut estimate = OperatorEstimate::default();
        estimate.association.stereo_operator_track_id = Some("stereo-est".to_string());
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-live".to_string());

        assert_eq!(
            resolve_active_stereo_track_id(Some(&estimate), Some(&stereo), Some("stereo-prev"))
                .as_deref(),
            Some("stereo-est")
        );
        assert_eq!(
            resolve_active_stereo_track_id(None, Some(&stereo), Some("stereo-prev")).as_deref(),
            Some("stereo-live")
        );
        assert_eq!(
            resolve_active_stereo_track_id(None, None, Some("stereo-prev")).as_deref(),
            Some("stereo-prev")
        );
    }

    #[test]
    fn record_track_motion_state_observations_should_keep_track_local_window() {
        let mut track_histories = std::collections::HashMap::new();
        let observations = vec![MotionStateObservation {
            edge_time_ns: 100,
            kind: MotionObservationKind::StereoGeometry,
            root_pos_m: Some([0.8, 0.0, 0.9]),
            heading_yaw_rad: Some(0.1),
            reliability: 1.0,
            motion_phase: 0.25,
            track_id: Some("stereo-b".to_string()),
        }];

        record_track_motion_state_observations(
            &mut track_histories,
            Some("stereo-b"),
            &observations,
            100,
        );

        let history = track_histories.get("stereo-b").expect("track history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].track_id.as_deref(), Some("stereo-b"));
    }

    #[test]
    fn apply_confirmed_wearer_sticky_should_hold_recent_confirmed_track() {
        let mut inner = OperatorInner {
            last_confirmed_wearer_track_id: Some("stereo-wearer".to_string()),
            last_confirmed_wearer_edge_time_ns: 900,
            ..OperatorInner::default()
        };
        let mut estimate = OperatorEstimate::default();
        estimate.association.selected_operator_track_id = Some("stereo-wearer".to_string());
        estimate.association.stereo_operator_track_id = Some("stereo-wearer".to_string());
        estimate.association.anchor_source = "stereo";

        apply_confirmed_wearer_sticky(&mut inner, &mut estimate, 1_500_000_000);

        assert_eq!(
            estimate.association.iphone_operator_track_id.as_deref(),
            Some("stereo-wearer")
        );
        assert_eq!(estimate.association.anchor_source, "stereo+iphone_hand");
        assert!(estimate.association.hand_match_score >= 0.5);
    }

    #[test]
    fn build_source_plus_wifi_estimate_should_emit_multi_source_body_and_keep_primary_hands() {
        let mut primary = test_source_pose(OperatorSource::Stereo);
        primary.raw_pose = OperatorRawPose {
            source_edge_time_ns: 11,
            body_layout: BodyKeypointLayout::CocoBody17,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            body_kpts_3d: build_pico_body_points(),
            hand_kpts_3d: build_pico_dual_hand_points(),
        };
        primary.operator_track_id = Some("stereo-main".to_string());
        primary.canonical_body_kpts_3d = canonicalize_body_points_3d(
            &primary.raw_pose.body_kpts_3d,
            BodyKeypointLayout::PicoBody24,
        );
        primary.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &primary.raw_pose.hand_kpts_3d,
            HandKeypointLayout::PicoHand26,
        );

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.body_space = STEREO_PAIR_FRAME.to_string();
        wifi.raw_pose = OperatorRawPose {
            source_edge_time_ns: 12,
            body_layout: BodyKeypointLayout::PicoBody24,
            hand_layout: HandKeypointLayout::Unknown,
            body_kpts_3d: build_pico_body_points(),
            hand_kpts_3d: Vec::new(),
        };
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.canonical_body_kpts_3d = canonicalize_body_points_3d(
            &wifi.raw_pose.body_kpts_3d,
            BodyKeypointLayout::PicoBody24,
        );
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let estimate = build_source_plus_wifi_estimate(123, &primary, &wifi);

        assert_eq!(estimate.source.as_str(), "stereo");
        assert_eq!(estimate.fusion_breakdown.body_source.as_str(), "stereo");
        assert_eq!(estimate.fusion_breakdown.hand_source.as_str(), "stereo");
        assert_eq!(estimate.fusion_breakdown.wifi_body_joint_count, 0);
        assert_eq!(estimate.fusion_breakdown.stereo_hand_point_count, 42);
        assert_eq!(
            estimate.association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(estimate.association.anchor_source, "stereo+wifi_prior");
        assert_eq!(
            estimate.association.wifi_operator_track_id.as_deref(),
            Some("wifi-main")
        );
        assert_eq!(estimate.operator_state.hand_kpts_3d.len(), 42);
    }

    #[test]
    fn build_fused_estimate_with_wifi_should_report_all_three_sources() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.raw_pose = OperatorRawPose {
            source_edge_time_ns: 11,
            body_layout: BodyKeypointLayout::PicoBody24,
            hand_layout: HandKeypointLayout::PicoHand26,
            body_kpts_3d: build_pico_body_points(),
            hand_kpts_3d: build_pico_dual_hand_points(),
        };
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
            &stereo.raw_pose.body_kpts_3d,
            BodyKeypointLayout::PicoBody24,
        );
        stereo.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &stereo.raw_pose.hand_kpts_3d,
            HandKeypointLayout::PicoHand26,
        );

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.raw_pose = OperatorRawPose {
            source_edge_time_ns: 12,
            body_layout: BodyKeypointLayout::Unknown,
            hand_layout: HandKeypointLayout::MediapipeHand21,
            body_kpts_3d: Vec::new(),
            hand_kpts_3d: stereo.canonical_hand_kpts_3d.clone(),
        };
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.hand_space = STEREO_PAIR_FRAME.to_string();
        vision.canonical_hand_kpts_3d = stereo.canonical_hand_kpts_3d.clone();

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.body_space = STEREO_PAIR_FRAME.to_string();
        wifi.raw_pose = OperatorRawPose {
            source_edge_time_ns: 13,
            body_layout: BodyKeypointLayout::PicoBody24,
            hand_layout: HandKeypointLayout::Unknown,
            body_kpts_3d: build_pico_body_points(),
            hand_kpts_3d: Vec::new(),
        };
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.canonical_body_kpts_3d = canonicalize_body_points_3d(
            &wifi.raw_pose.body_kpts_3d,
            BodyKeypointLayout::PicoBody24,
        );
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let estimate = build_fused_estimate_with_wifi(456, &stereo, &vision, &wifi);

        assert_eq!(estimate.source.as_str(), "fused_stereo_vision_3d");
        assert_eq!(estimate.fusion_breakdown.body_source.as_str(), "stereo");
        assert_eq!(
            estimate.fusion_breakdown.hand_source.as_str(),
            "fused_stereo_vision_3d"
        );
        assert_eq!(estimate.fusion_breakdown.wifi_body_joint_count, 0);
        assert_eq!(
            estimate.association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(
            estimate.association.anchor_source,
            "stereo+iphone_hand+wifi_prior"
        );
        assert_eq!(
            estimate.association.iphone_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(
            estimate.association.wifi_operator_track_id.as_deref(),
            Some("wifi-main")
        );
        assert!(estimate.association.hand_match_score > 0.0);
    }

    #[test]
    fn fuse_operator_sources_should_emit_wifi_only_body_estimate() {
        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let estimate =
            fuse_operator_sources(777, None, None, Some(&wifi)).expect("wifi-only estimate");

        assert_eq!(estimate.source.as_str(), "none");
        assert_eq!(estimate.fusion_breakdown.body_source.as_str(), "none");
        assert_eq!(estimate.fusion_breakdown.wifi_body_joint_count, 0);
        assert_eq!(estimate.fusion_breakdown.hand_source.as_str(), "none");
        assert!(estimate.operator_state.body_kpts_3d.is_empty());
        assert_eq!(estimate.association.anchor_source, "none");
    }

    #[test]
    fn wifi_body_with_vision_hands_should_keep_body_and_hand_output() {
        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.canonical_body_kpts_3d = Vec::new();
        vision.raw_pose.body_kpts_3d = Vec::new();
        vision.raw_pose.body_layout = BodyKeypointLayout::Unknown;
        vision.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );
        vision.raw_pose.hand_layout = HandKeypointLayout::PicoHand26;
        vision.raw_pose.hand_kpts_3d = build_pico_dual_hand_points();

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.body_space = OPERATOR_FRAME.to_string();
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let estimate = build_wifi_body_with_vision_hands_estimate(321, &vision, &wifi);

        assert_eq!(estimate.source.as_str(), "vision_3d");
        assert!(estimate.operator_state.body_kpts_3d.is_empty());
        assert_eq!(estimate.operator_state.hand_kpts_3d.len(), 42);
        assert_eq!(estimate.fusion_breakdown.body_source.as_str(), "none");
        assert_eq!(estimate.fusion_breakdown.hand_source.as_str(), "vision_3d");
        assert_eq!(estimate.fusion_breakdown.wifi_body_joint_count, 0);
        assert_eq!(estimate.association.anchor_source, "iphone_hand+wifi_prior");
        assert_eq!(
            estimate.association.selected_operator_track_id.as_deref(),
            Some("wifi-main")
        );
    }

    #[test]
    fn derive_operator_association_should_use_stereo_body_and_iphone_hands() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.hand_space = STEREO_PAIR_FRAME.to_string();
        vision.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            vision_hand_point_count: 42,
            ..OperatorFusionBreakdown::default()
        };
        let association =
            derive_operator_association(Some(&stereo), Some(&vision), None, None, &breakdown);

        assert_eq!(
            association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(association.anchor_source, "stereo+iphone_hand");
        assert_eq!(
            association.iphone_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert!(association.hand_match_score > 0.0);
        assert!(association.left_wrist_gap_m.is_some());
        assert!(association.right_wrist_gap_m.is_some());
    }

    #[test]
    fn build_fused_estimate_should_drop_vision_hands_when_hand_match_is_poor() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        stereo.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );
        stereo.raw_pose.hand_layout = HandKeypointLayout::PicoHand26;
        stereo.raw_pose.hand_kpts_3d = build_pico_dual_hand_points();

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.canonical_hand_kpts_3d = build_pico_dual_hand_points()
            .into_iter()
            .map(|point| [point[0] + 1.0, point[1], point[2]])
            .collect();

        let estimate = build_fused_estimate(456, &stereo, &vision);

        assert_eq!(estimate.source, OperatorSource::Stereo);
        assert_eq!(
            estimate.fusion_breakdown.body_source,
            OperatorPartSource::Stereo
        );
        assert_eq!(
            estimate.fusion_breakdown.hand_source,
            OperatorPartSource::Stereo
        );
        assert_eq!(estimate.fusion_breakdown.vision_hand_point_count, 0);
        assert_eq!(estimate.association.anchor_source, "stereo");
        assert_eq!(estimate.association.hand_match_score, 0.0);
    }

    #[test]
    fn build_fused_estimate_should_keep_vision_hands_when_stereo_has_no_hands() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.canonical_hand_kpts_3d = build_pico_dual_hand_points()
            .into_iter()
            .map(|point| [point[0] + 1.0, point[1], point[2]])
            .collect();
        vision.raw_pose.hand_layout = HandKeypointLayout::PicoHand26;
        vision.raw_pose.hand_kpts_3d = build_pico_dual_hand_points()
            .into_iter()
            .map(|point| [point[0] + 1.0, point[1], point[2]])
            .collect();

        let estimate = build_fused_estimate(457, &stereo, &vision);

        assert_eq!(
            estimate.fusion_breakdown.hand_source,
            OperatorPartSource::Vision3d
        );
        assert_eq!(estimate.fusion_breakdown.stereo_hand_point_count, 0);
        assert!(estimate.fusion_breakdown.vision_hand_point_count > 0);
        assert_eq!(estimate.association.anchor_source, "stereo");
        assert_eq!(estimate.association.hand_match_score, 0.0);
        assert_eq!(estimate.operator_state.hand_kpts_3d.len(), 52);
    }

    #[test]
    fn build_fused_estimate_should_keep_vision_hands_when_hand_match_is_good() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );
        vision.raw_pose.hand_layout = HandKeypointLayout::PicoHand26;
        vision.raw_pose.hand_kpts_3d = build_pico_dual_hand_points();
        vision.hand_space = STEREO_PAIR_FRAME.to_string();

        let estimate = build_fused_estimate(789, &stereo, &vision);

        assert_eq!(estimate.source, OperatorSource::FusedStereoVision3d);
        assert_eq!(
            estimate.fusion_breakdown.hand_source,
            OperatorPartSource::Vision3d
        );
        assert!(estimate.fusion_breakdown.vision_hand_point_count > 0);
        assert_eq!(estimate.association.anchor_source, "stereo+iphone_hand");
        assert!(estimate.association.hand_match_score > 0.0);
    }

    #[test]
    fn derive_operator_association_should_not_match_iphone_hands_against_canonical_wifi_body() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        wifi.body_space = CANONICAL_BODY_FRAME.to_string();

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.hand_space = STEREO_PAIR_FRAME.to_string();
        vision.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            wifi_body_joint_count: 17,
            vision_hand_point_count: 42,
            ..OperatorFusionBreakdown::default()
        };
        let association = derive_operator_association(
            Some(&stereo),
            Some(&vision),
            Some(&wifi),
            None,
            &breakdown,
        );

        assert_eq!(association.anchor_source, "stereo+iphone_hand");
        assert!(association.hand_match_score > 0.0);
    }

    #[test]
    fn build_source_plus_wifi_estimate_should_keep_stereo_geometry_when_frames_differ() {
        let mut primary = test_source_pose(OperatorSource::Stereo);
        primary.operator_track_id = Some("stereo-main".to_string());
        primary.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        primary.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        wifi.body_space = CANONICAL_BODY_FRAME.to_string();
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let estimate = build_source_plus_wifi_estimate(123, &primary, &wifi);

        assert_eq!(estimate.source, OperatorSource::Stereo);
        assert_eq!(
            estimate.fusion_breakdown.body_source,
            OperatorPartSource::Stereo
        );
        assert_eq!(estimate.fusion_breakdown.wifi_body_joint_count, 0);
        assert_eq!(estimate.association.anchor_source, "stereo+wifi_prior");
        assert_eq!(
            estimate.association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(estimate.operator_state.body_kpts_3d.len(), 17);
    }

    #[test]
    fn derive_operator_association_should_not_promote_predict_only_wifi_over_stereo() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "lost".to_string(),
            coherence_gate_decision: "PredictOnly".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            wifi_body_joint_count: 17,
            ..OperatorFusionBreakdown::default()
        };
        let association = derive_operator_association(
            Some(&stereo),
            None,
            Some(&wifi),
            wifi.wifi_diagnostics.as_ref(),
            &breakdown,
        );

        assert!(!association.wifi_anchor_eligible);
        assert_eq!(association.anchor_source, "stereo");
        assert_eq!(
            association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
    }

    #[test]
    fn derive_operator_association_should_fallback_to_shoulder_when_stereo_wrist_missing() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        stereo.canonical_body_kpts_3d[9] = [0.0, 0.0, 0.0];

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.hand_space = STEREO_PAIR_FRAME.to_string();
        vision.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            vision_hand_point_count: 42,
            ..OperatorFusionBreakdown::default()
        };
        let association =
            derive_operator_association(Some(&stereo), Some(&vision), None, None, &breakdown);

        assert_eq!(association.anchor_source, "stereo+iphone_hand");
        assert_eq!(
            association.iphone_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert!(association.hand_match_score > 0.0);
        assert!(association.left_wrist_gap_m.is_some());
    }

    #[test]
    fn derive_operator_association_should_allow_single_visible_mirrored_hand() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.body_space = STEREO_PAIR_FRAME.to_string();
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let canonical_hands = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );
        let mut mirrored_single_hand = vec![[0.0, 0.0, 0.0]; 42];
        mirrored_single_hand[21..42].copy_from_slice(&canonical_hands[..21]);

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.hand_space = STEREO_PAIR_FRAME.to_string();
        vision.left_hand_fresh = false;
        vision.right_hand_fresh = true;
        vision.canonical_hand_kpts_3d = mirrored_single_hand;

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            vision_hand_point_count: 21,
            ..OperatorFusionBreakdown::default()
        };
        let association =
            derive_operator_association(Some(&stereo), Some(&vision), None, None, &breakdown);

        assert_eq!(association.anchor_source, "stereo+iphone_hand");
        assert!(association.hand_match_score >= 0.45);
        assert!(association.left_wrist_gap_m.is_some());
        assert!(association.right_wrist_gap_m.is_none());
    }

    #[test]
    fn derive_operator_association_should_keep_good_hand_when_other_hand_is_far() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.body_space = STEREO_PAIR_FRAME.to_string();
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.hand_space = STEREO_PAIR_FRAME.to_string();
        vision.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );
        vision.canonical_hand_kpts_3d[21] = [3.5, -2.0, 5.0];

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            vision_hand_point_count: 42,
            ..OperatorFusionBreakdown::default()
        };
        let association =
            derive_operator_association(Some(&stereo), Some(&vision), None, None, &breakdown);

        assert!(association.hand_match_score > 0.0);
        assert_eq!(association.hand_match_count, 1);
        assert!(association.left_wrist_gap_m.is_some());
        assert!(association.right_wrist_gap_m.is_some());
    }

    #[test]
    fn derive_operator_association_should_report_visible_iphone_hand_without_body_match() {
        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.hand_space = STEREO_PAIR_FRAME.to_string();
        vision.left_hand_fresh = false;
        vision.right_hand_fresh = true;
        vision.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
            &build_pico_dual_hand_points(),
            HandKeypointLayout::PicoHand26,
        );

        let breakdown = OperatorFusionBreakdown {
            vision_hand_point_count: 21,
            ..OperatorFusionBreakdown::default()
        };
        let association = derive_operator_association(None, Some(&vision), None, None, &breakdown);

        assert_eq!(association.iphone_visible_hand_count, 1);
        assert_eq!(association.hand_match_count, 0);
        assert_eq!(association.hand_match_score, 0.0);
        assert_eq!(association.anchor_source, "iphone_hand");
    }

    #[test]
    fn derive_operator_association_should_not_claim_iphone_anchor_without_hand_match() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.canonical_hand_kpts_3d = build_pico_dual_hand_points()
            .into_iter()
            .map(|point| [point[0] + 1.0, point[1], point[2]])
            .collect();

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            vision_hand_point_count: 42,
            ..OperatorFusionBreakdown::default()
        };
        let association = derive_operator_association(
            Some(&stereo),
            Some(&vision),
            Some(&wifi),
            wifi.wifi_diagnostics.as_ref(),
            &breakdown,
        );

        assert_eq!(association.anchor_source, "stereo+wifi_prior");
        assert_eq!(
            association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(association.iphone_operator_track_id.as_deref(), None);
        assert_eq!(association.hand_match_score, 0.0);
    }

    #[test]
    fn stabilize_operator_estimate_should_hold_previous_stereo_track_for_wifi_only_fallback() {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-main".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo
        });
        previous.association.anchor_source = "wifi+stereo";
        previous.association.selected_operator_track_id = Some("stereo-main".to_string());
        previous.fusion_breakdown.stereo_body_joint_count = 17;

        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.raw_pose.body_kpts_3d = vec![[0.0, 0.0, 1.0]; 17];
        wifi.canonical_body_kpts_3d = vec![[0.0, 0.0, 1.0]; 17];
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let current = build_single_source_estimate(101, &wifi);
        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert_eq!(stabilized.association.anchor_source, "wifi");
        assert_eq!(
            stabilized.association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
    }

    #[test]
    fn stabilize_operator_estimate_should_hold_previous_hand_match_for_same_stereo_track() {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-main".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo
        });
        previous.association.anchor_source = "wifi+stereo+iphone_hand";
        previous.association.selected_operator_track_id = Some("stereo-main".to_string());
        previous.association.iphone_visible_hand_count = 1;
        previous.association.hand_match_count = 1;
        previous.association.hand_match_score = 0.82;
        previous.association.left_wrist_gap_m = Some(0.18);
        previous.association.right_wrist_gap_m = None;
        previous.fusion_breakdown.stereo_body_joint_count = 17;

        let mut current = previous.clone();
        current.association.anchor_source = "wifi+stereo";
        current.association.iphone_visible_hand_count = 1;
        current.association.hand_match_count = 0;
        current.association.hand_match_score = 0.0;
        current.association.left_wrist_gap_m = Some(0.71);
        current.association.right_wrist_gap_m = None;

        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert!(stabilized.association.hand_match_score > 0.70);
        assert_eq!(stabilized.association.hand_match_count, 1);
        assert_eq!(stabilized.association.left_wrist_gap_m, Some(0.18));
        assert_eq!(
            stabilized.association.anchor_source,
            "wifi+stereo+iphone_hand"
        );
    }

    #[test]
    fn stabilize_operator_estimate_should_hold_previous_hand_match_when_iphone_temporarily_missing()
    {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-main".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo
        });
        previous.association.anchor_source = "wifi+stereo+iphone_hand";
        previous.association.selected_operator_track_id = Some("stereo-main".to_string());
        previous.association.iphone_operator_track_id = Some("stereo-main".to_string());
        previous.association.iphone_visible_hand_count = 1;
        previous.association.hand_match_count = 1;
        previous.association.hand_match_score = 0.86;
        previous.association.right_wrist_gap_m = Some(0.12);
        previous.fusion_breakdown.stereo_body_joint_count = 17;
        previous.fusion_breakdown.wifi_body_joint_count = 17;

        let mut current = previous.clone();
        current.association.anchor_source = "wifi+stereo";
        current.association.iphone_operator_track_id = None;
        current.association.iphone_visible_hand_count = 0;
        current.association.hand_match_count = 0;
        current.association.hand_match_score = 0.0;
        current.association.left_wrist_gap_m = None;
        current.association.right_wrist_gap_m = None;

        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert!(stabilized.association.hand_match_score > 0.75);
        assert_eq!(stabilized.association.hand_match_count, 1);
        assert_eq!(
            stabilized.association.iphone_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(stabilized.association.right_wrist_gap_m, Some(0.12));
        assert_eq!(
            stabilized.association.anchor_source,
            "wifi+stereo+iphone_hand"
        );
    }

    #[test]
    fn stabilize_operator_estimate_should_hold_previous_hand_points_when_current_hands_drop() {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-main".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo.canonical_hand_kpts_3d = canonicalize_hand_points_3d(
                &build_pico_dual_hand_points(),
                HandKeypointLayout::PicoHand26,
            );
            stereo.raw_pose.hand_layout = HandKeypointLayout::PicoHand26;
            stereo.raw_pose.hand_kpts_3d = build_pico_dual_hand_points();
            stereo
        });
        previous.association.anchor_source = "wifi+stereo+iphone_hand";
        previous.association.selected_operator_track_id = Some("stereo-main".to_string());
        previous.association.stereo_operator_track_id = Some("stereo-main".to_string());
        previous.association.iphone_operator_track_id = Some("stereo-main".to_string());
        previous.association.iphone_visible_hand_count = 1;
        previous.association.hand_match_count = 1;
        previous.association.hand_match_score = 0.88;
        previous.association.left_wrist_gap_m = Some(0.10);
        previous.fusion_breakdown.stereo_body_joint_count = 17;
        previous.fusion_breakdown.wifi_body_joint_count = 17;
        previous.fusion_breakdown.hand_source = OperatorPartSource::Stereo;
        previous.fusion_breakdown.stereo_hand_point_count = 42;

        let mut current = previous.clone();
        current.operator_state.hand_kpts_3d.clear();
        current.raw_pose.hand_kpts_3d.clear();
        current.left_hand_curls = None;
        current.right_hand_curls = None;
        current.fusion_breakdown.hand_source = OperatorPartSource::None;
        current.fusion_breakdown.stereo_hand_point_count = 0;
        current.fusion_breakdown.vision_hand_point_count = 0;
        current.fusion_breakdown.wifi_hand_point_count = 0;
        current.fusion_breakdown.blended_hand_point_count = 0;
        current.association.anchor_source = "wifi+stereo";
        current.association.iphone_operator_track_id = None;
        current.association.iphone_visible_hand_count = 0;
        current.association.hand_match_count = 0;
        current.association.hand_match_score = 0.0;
        current.association.left_wrist_gap_m = None;
        current.association.right_wrist_gap_m = None;

        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert_eq!(stabilized.operator_state.hand_kpts_3d.len(), 42);
        assert_eq!(stabilized.raw_pose.hand_kpts_3d.len(), 52);
        assert!(stabilized.left_hand_curls.is_some());
        assert!(stabilized.right_hand_curls.is_some());
        assert_eq!(
            stabilized.fusion_breakdown.hand_source,
            OperatorPartSource::Stereo
        );
        assert_eq!(stabilized.fusion_breakdown.stereo_hand_point_count, 42);
        assert_eq!(
            stabilized.association.anchor_source,
            "wifi+stereo+iphone_hand"
        );
    }

    #[test]
    fn stabilize_operator_estimate_should_prefer_previous_wearer_track_when_body_anchor_matches() {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-wearer".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo
        });
        previous.association.anchor_source = "wifi+stereo+iphone_hand";
        previous.association.selected_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.stereo_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.wifi_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.iphone_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.iphone_visible_hand_count = 1;
        previous.association.hand_match_count = 1;
        previous.association.hand_match_score = 0.91;
        previous.association.left_wrist_gap_m = Some(0.08);
        previous.fusion_breakdown.stereo_body_joint_count = 17;
        previous.fusion_breakdown.wifi_body_joint_count = 17;

        let mut current = previous.clone();
        current.association.anchor_source = "wifi+stereo";
        current.association.selected_operator_track_id = Some("stereo-other".to_string());
        current.association.stereo_operator_track_id = Some("stereo-wearer".to_string());
        current.association.wifi_operator_track_id = Some("stereo-wearer".to_string());
        current.association.iphone_operator_track_id = None;
        current.association.iphone_visible_hand_count = 0;
        current.association.hand_match_count = 0;
        current.association.hand_match_score = 0.0;
        current.association.left_wrist_gap_m = None;
        current.association.right_wrist_gap_m = None;

        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert_eq!(
            stabilized.association.selected_operator_track_id.as_deref(),
            Some("stereo-wearer")
        );
        assert_eq!(
            stabilized.association.iphone_operator_track_id.as_deref(),
            Some("stereo-wearer")
        );
    }

    #[test]
    fn stabilize_operator_estimate_should_hold_previous_wearer_track_when_stereo_temporarily_degrades_to_iphone_only(
    ) {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-wearer".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo
        });
        previous.association.anchor_source = "stereo+iphone_hand";
        previous.association.selected_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.stereo_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.wifi_operator_track_id = Some("wifi-track-1".to_string());
        previous.association.iphone_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.iphone_visible_hand_count = 1;
        previous.association.hand_match_count = 1;
        previous.association.hand_match_score = 0.98;
        previous.association.right_wrist_gap_m = Some(0.0005);
        previous.fusion_breakdown.stereo_body_joint_count = 17;
        previous.fusion_breakdown.wifi_body_joint_count = 17;

        let mut current = previous.clone();
        current.association.anchor_source = "iphone_hand";
        current.association.selected_operator_track_id = Some("primary_operator".to_string());
        current.association.stereo_operator_track_id = None;
        current.association.iphone_operator_track_id = Some("primary_operator".to_string());
        current.association.hand_match_count = 0;
        current.association.hand_match_score = 0.0;
        current.association.left_wrist_gap_m = None;
        current.association.right_wrist_gap_m = None;
        current.fusion_breakdown.stereo_body_joint_count = 0;

        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert_eq!(stabilized.association.anchor_source, "stereo+iphone_hand");
        assert_eq!(
            stabilized.association.selected_operator_track_id.as_deref(),
            Some("stereo-wearer")
        );
        assert_eq!(
            stabilized.association.stereo_operator_track_id.as_deref(),
            Some("stereo-wearer")
        );
        assert_eq!(
            stabilized.association.iphone_operator_track_id.as_deref(),
            Some("stereo-wearer")
        );
        assert_eq!(stabilized.association.hand_match_count, 1);
        assert!(stabilized.association.hand_match_score > 0.8);
    }

    #[test]
    fn stabilize_operator_estimate_should_hold_previous_wearer_body_when_current_body_missing() {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-wearer".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo
        });
        previous.association.anchor_source = "stereo+iphone_hand";
        previous.association.selected_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.stereo_operator_track_id = Some("stereo-wearer".to_string());
        previous.association.iphone_operator_track_id = Some("stereo-wearer".to_string());
        previous.fusion_breakdown.body_source = OperatorPartSource::Stereo;
        previous.raw_pose.body_kpts_3d = build_pico_body_points();

        let mut current = previous.clone();
        current.operator_state.body_kpts_3d.clear();
        current.raw_pose.body_kpts_3d.clear();
        current.association.anchor_source = "stereo";
        current.association.iphone_operator_track_id = None;

        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert_eq!(stabilized.operator_state.body_kpts_3d.len(), 17);
        assert_eq!(stabilized.raw_pose.body_kpts_3d.len(), 24);
        assert_eq!(
            stabilized.fusion_breakdown.body_source,
            OperatorPartSource::Stereo
        );
        assert_eq!(
            stabilized.association.selected_operator_track_id.as_deref(),
            Some("stereo-wearer")
        );
    }

    #[test]
    fn derive_operator_association_should_drop_iphone_only_anchor_when_hands_do_not_match() {
        let mut stereo = test_source_pose(OperatorSource::Stereo);
        stereo.operator_track_id = Some("stereo-main".to_string());
        stereo.canonical_body_kpts_3d =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);

        let mut vision = test_source_pose(OperatorSource::Vision3d);
        vision.operator_track_id = Some("iphone-main".to_string());
        vision.canonical_hand_kpts_3d = build_pico_dual_hand_points()
            .into_iter()
            .map(|point| [point[0], point[1] - 1.0, point[2]])
            .collect();

        let breakdown = OperatorFusionBreakdown {
            stereo_body_joint_count: 17,
            vision_hand_point_count: 42,
            ..OperatorFusionBreakdown::default()
        };
        let association =
            derive_operator_association(Some(&stereo), Some(&vision), None, None, &breakdown);

        assert_eq!(association.anchor_source, "stereo");
        assert_eq!(
            association.selected_operator_track_id.as_deref(),
            Some("stereo-main")
        );
        assert_eq!(association.iphone_operator_track_id.as_deref(), None);
        assert_eq!(association.hand_match_score, 0.0);
    }

    #[test]
    fn wifi_association_score_should_reward_multi_node_layout_and_zone_coherence() {
        let weak = WifiPoseDiagnostics {
            layout_node_count: 1,
            layout_score: 0.18,
            zone_score: 0.0,
            zone_summary_reliable: false,
            motion_energy: 1.4,
            doppler_hz: 0.12,
            signal_quality: 0.42,
            vital_signal_quality: None,
            stream_fps: 6.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        };
        let strong = WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.92,
            zone_score: 0.88,
            zone_summary_reliable: true,
            motion_energy: 4.8,
            doppler_hz: 0.56,
            signal_quality: 0.91,
            vital_signal_quality: Some(0.74),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        };

        let weak_score =
            compute_wifi_association_score(Some("stereo-main"), Some("wifi-main"), Some(&weak));
        let strong_score =
            compute_wifi_association_score(Some("stereo-main"), Some("stereo-main"), Some(&strong));

        assert!(strong_score > weak_score + 0.30);
        assert!(strong_score > 0.80);
    }

    #[test]
    fn wifi_association_score_should_not_take_zone_bonus_when_summary_unreliable() {
        let unreliable = WifiPoseDiagnostics {
            layout_node_count: 4,
            layout_score: 0.85,
            zone_score: 0.95,
            zone_summary_reliable: false,
            motion_energy: 3.0,
            doppler_hz: 0.4,
            signal_quality: 0.88,
            vital_signal_quality: Some(0.7),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        };
        let reliable = WifiPoseDiagnostics {
            zone_summary_reliable: true,
            ..unreliable.clone()
        };

        let unreliable_score = compute_wifi_association_score(
            Some("stereo-main"),
            Some("stereo-main"),
            Some(&unreliable),
        );
        let reliable_score = compute_wifi_association_score(
            Some("stereo-main"),
            Some("stereo-main"),
            Some(&reliable),
        );

        assert!(reliable_score > unreliable_score);
        assert!(reliable_score - unreliable_score >= 0.08);
    }

    #[test]
    fn wifi_single_source_estimate_should_preserve_association_diagnostics() {
        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.raw_pose.body_kpts_3d = vec![[0.0, 0.0, 1.0]; 17];
        wifi.canonical_body_kpts_3d = vec![[0.0, 0.0, 1.0]; 17];
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 3,
            layout_score: 0.84,
            zone_score: 0.95,
            zone_summary_reliable: true,
            motion_energy: 6.0,
            doppler_hz: 0.7,
            signal_quality: 0.65,
            vital_signal_quality: Some(0.4),
            stream_fps: 10.0,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let estimate = build_single_source_estimate(42, &wifi);

        assert_eq!(estimate.association.anchor_source, "wifi");
        assert!(estimate.association.wifi_association_score > 0.18);
        assert!(estimate.association.wifi_layout_score > 0.8);
        assert!(estimate.association.wifi_zone_summary_reliable);
        assert_eq!(
            estimate.association.selected_operator_track_id.as_deref(),
            Some("wifi-main")
        );
    }

    #[test]
    fn derive_operator_association_should_not_select_wifi_track_when_no_anchor_is_available() {
        let mut wifi = test_source_pose(OperatorSource::WifiPose3d);
        wifi.operator_track_id = Some("wifi-main".to_string());
        wifi.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 1,
            layout_score: 0.05,
            zone_score: 0.0,
            zone_summary_reliable: false,
            motion_energy: 0.2,
            doppler_hz: 0.0,
            signal_quality: 0.1,
            vital_signal_quality: None,
            stream_fps: 2.0,
            lifecycle_state: "idle".to_string(),
            coherence_gate_decision: "Reject".to_string(),
            target_space: CANONICAL_BODY_FRAME.to_string(),
        });

        let association = derive_operator_association(
            None,
            None,
            Some(&wifi),
            wifi.wifi_diagnostics.as_ref(),
            &OperatorFusionBreakdown::default(),
        );

        assert_eq!(association.anchor_source, "none");
        assert_eq!(association.selected_operator_track_id, None);
        assert_eq!(
            association.wifi_operator_track_id.as_deref(),
            Some("wifi-main")
        );
    }

    #[test]
    fn stabilize_operator_estimate_should_hold_previous_stereo_track_when_current_selection_degrades_to_wifi(
    ) {
        let mut previous = build_single_source_estimate(100, &{
            let mut stereo = test_source_pose(OperatorSource::Stereo);
            stereo.operator_track_id = Some("stereo-person-7".to_string());
            stereo.canonical_body_kpts_3d = canonicalize_body_points_3d(
                &build_pico_body_points(),
                BodyKeypointLayout::PicoBody24,
            );
            stereo
        });
        previous.association.anchor_source = "stereo";
        previous.association.selected_operator_track_id = Some("stereo-person-7".to_string());
        previous.association.stereo_operator_track_id = Some("stereo-person-7".to_string());
        previous.fusion_breakdown.stereo_body_joint_count = 17;

        let mut current = super::OperatorEstimate::default();
        current.association.anchor_source = "none";
        current.association.selected_operator_track_id = Some("wifi-track-1".to_string());
        current.association.wifi_operator_track_id = Some("wifi-track-1".to_string());

        let stabilized = stabilize_operator_estimate(current, Some(&previous), true);

        assert_eq!(
            stabilized.association.selected_operator_track_id.as_deref(),
            Some("stereo-person-7")
        );
    }

    fn build_pico_body_points() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.04, 0.80],
            [-0.05, 0.06, 0.80],
            [0.05, 0.06, 0.80],
            [0.0, 0.11, 0.81],
            [-0.05, -0.18, 0.79],
            [0.05, -0.18, 0.79],
            [0.0, 0.18, 0.82],
            [-0.05, -0.44, 0.78],
            [0.05, -0.44, 0.78],
            [0.0, 0.28, 0.83],
            [-0.05, -0.48, 0.83],
            [0.05, -0.48, 0.83],
            [0.0, 0.34, 0.82],
            [-0.04, 0.30, 0.82],
            [0.04, 0.30, 0.82],
            [0.0, 0.39, 0.82],
            [-0.08, 0.22, 0.82],
            [0.08, 0.22, 0.82],
            [-0.11, 0.25, 0.84],
            [0.11, 0.23, 0.84],
            [-0.132, 0.304, 0.853],
            [0.132, 0.209, 0.853],
            [-0.145, 0.33, 0.86],
            [0.145, 0.23, 0.86],
        ]
    }

    fn build_pico_dual_hand_points() -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        out.extend(build_pico_hand_points([-0.132, 0.304, 0.853], true));
        out.extend(build_pico_hand_points([0.132, 0.209, 0.853], false));
        out
    }

    fn build_pico_hand_points(wrist: [f32; 3], left: bool) -> Vec<[f32; 3]> {
        let toward_thumb = if left { -1.0 } else { 1.0 };
        let toward_pinky = if left { -1.0 } else { 1.0 };

        vec![
            offset(wrist, 0.0, 0.030, 0.0),
            wrist,
            offset(wrist, 0.010 * toward_thumb, 0.020, 0.0),
            offset(wrist, 0.022 * toward_thumb, 0.040, 0.0),
            offset(wrist, 0.034 * toward_thumb, 0.061, 0.0),
            offset(wrist, 0.046 * toward_thumb, 0.080, 0.0),
            offset(wrist, 0.018 * toward_thumb, 0.040, 0.0),
            offset(wrist, 0.020 * toward_thumb, 0.058, 0.0),
            offset(wrist, 0.024 * toward_thumb, 0.081, 0.0),
            offset(wrist, 0.028 * toward_thumb, 0.102, 0.0),
            offset(wrist, 0.031 * toward_thumb, 0.122, 0.0),
            offset(wrist, 0.004 * toward_thumb, 0.043, 0.0),
            offset(wrist, 0.005 * toward_thumb, 0.064, 0.0),
            offset(wrist, 0.005 * toward_thumb, 0.088, 0.0),
            offset(wrist, 0.005 * toward_thumb, 0.112, 0.0),
            offset(wrist, 0.005 * toward_thumb, 0.133, 0.0),
            offset(wrist, -0.012 * toward_thumb, 0.038, 0.0),
            offset(wrist, -0.014 * toward_thumb, 0.054, 0.0),
            offset(wrist, -0.018 * toward_thumb, 0.078, 0.0),
            offset(wrist, -0.022 * toward_thumb, 0.101, 0.0),
            offset(wrist, -0.026 * toward_thumb, 0.121, 0.0),
            offset(wrist, -0.028 * toward_pinky, 0.028, 0.0),
            offset(wrist, -0.033 * toward_pinky, 0.039, 0.0),
            offset(wrist, -0.037 * toward_pinky, 0.063, 0.0),
            offset(wrist, -0.041 * toward_pinky, 0.087, 0.0),
            offset(wrist, -0.045 * toward_pinky, 0.109, 0.0),
        ]
    }

    fn offset(base: [f32; 3], dx: f32, dy: f32, dz: f32) -> [f32; 3] {
        [base[0] + dx, base[1] + dy, base[2] + dz]
    }

    fn test_source_pose(source: OperatorSource) -> OperatorSourcePose {
        OperatorSourcePose {
            source,
            raw_pose: OperatorRawPose::default(),
            operator_track_id: None,
            body_space: match source {
                OperatorSource::Stereo => STEREO_PAIR_FRAME.to_string(),
                OperatorSource::WifiPose3d => CANONICAL_BODY_FRAME.to_string(),
                _ => String::new(),
            },
            hand_space: match source {
                OperatorSource::Stereo => STEREO_PAIR_FRAME.to_string(),
                OperatorSource::Vision3d | OperatorSource::Vision2dProjected => {
                    OPERATOR_FRAME.to_string()
                }
                _ => String::new(),
            },
            hand_geometry_trusted: source == OperatorSource::Vision3d,
            left_hand_fresh: true,
            right_hand_fresh: true,
            canonical_body_kpts_3d: Vec::new(),
            canonical_hand_kpts_3d: Vec::new(),
            left_hand_curls: None,
            right_hand_curls: None,
            body_weight: 1.0,
            hand_weight: 1.0,
            wifi_diagnostics: None,
        }
    }

    fn test_wifi_pose_for_motion(track_id: &str, hip_center: [f32; 3]) -> OperatorSourcePose {
        let mut source = test_source_pose(OperatorSource::WifiPose3d);
        let body =
            canonicalize_body_points_3d(&build_pico_body_points(), BodyKeypointLayout::PicoBody24);
        let root = estimate_body_root(&body).expect("root");
        let delta = sub(hip_center, root);
        source.operator_track_id = Some(track_id.to_string());
        source.body_space = STEREO_PAIR_FRAME.to_string();
        source.canonical_body_kpts_3d = body.into_iter().map(|point| add3(point, delta)).collect();
        source.wifi_diagnostics = Some(WifiPoseDiagnostics {
            layout_node_count: 3,
            layout_score: 0.92,
            zone_score: 0.84,
            motion_energy: 3.0,
            doppler_hz: 0.65,
            signal_quality: 0.9,
            vital_signal_quality: Some(0.78),
            stream_fps: 10.0,
            zone_summary_reliable: true,
            lifecycle_state: "active".to_string(),
            coherence_gate_decision: "Accept".to_string(),
            target_space: STEREO_PAIR_FRAME.to_string(),
        });
        source
    }
}
fn hand_pose_from_kpts_3d(kpts: &[[f32; 3]], base: usize, is_left: bool) -> Pose {
    let Some(wrist) = kpts.get(base) else {
        return Pose {
            pos: [0.0, 0.0, 0.0],
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    };
    if !is_present_point3(wrist) {
        return Pose {
            pos: [0.0, 0.0, 0.0],
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    }

    let pos = *wrist;
    let Some(index_mcp) = kpts.get(base + 5) else {
        return Pose {
            pos,
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    };
    let Some(middle_mcp) = kpts.get(base + 9) else {
        return Pose {
            pos,
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    };
    let Some(ring_mcp) = kpts.get(base + 13) else {
        return Pose {
            pos,
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    };
    let Some(pinky_mcp) = kpts.get(base + 17) else {
        return Pose {
            pos,
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    };
    if !is_present_point3(index_mcp)
        || !is_present_point3(middle_mcp)
        || !is_present_point3(ring_mcp)
        || !is_present_point3(pinky_mcp)
    {
        return Pose {
            pos: [0.0, 0.0, 0.0],
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    }

    let palm_forward = normalize(scale3(
        add3(
            add3(sub(*index_mcp, pos), sub(*middle_mcp, pos)),
            add3(sub(*ring_mcp, pos), sub(*pinky_mcp, pos)),
        ),
        0.25,
    ));
    let palm_lateral = normalize(if is_left {
        sub(*pinky_mcp, *index_mcp)
    } else {
        sub(*index_mcp, *pinky_mcp)
    });
    let z_axis = normalize(cross(palm_lateral, palm_forward));
    let x_axis = palm_lateral;
    let y_axis = normalize(cross(z_axis, x_axis));
    if !is_finite3(x_axis) || !is_finite3(y_axis) || !is_finite3(z_axis) {
        return Pose {
            pos,
            quat: [0.0, 0.0, 0.0, 1.0],
        };
    }

    // 以列向量构造旋转矩阵：R = [x y z]
    let r00 = x_axis[0];
    let r10 = x_axis[1];
    let r20 = x_axis[2];
    let r01 = y_axis[0];
    let r11 = y_axis[1];
    let r21 = y_axis[2];
    let r02 = z_axis[0];
    let r12 = z_axis[1];
    let r22 = z_axis[2];

    let rot = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]];
    let quat = quat_from_rot(rot);
    Pose { pos, quat }
}

fn finger_curls_from_hand_kpts_2d(kpts: &[[f32; 2]], base: usize) -> Option<[f32; 5]> {
    if kpts.len() < base + 21 {
        return None;
    }
    for index in [
        base,
        base + 4,
        base + 5,
        base + 8,
        base + 12,
        base + 16,
        base + 17,
        base + 20,
    ] {
        if !kpts.get(index).is_some_and(is_present_point2) {
            return None;
        }
    }
    let wrist = kpts[base];
    let index_mcp = kpts[base + 5];
    let pinky_mcp = kpts[base + 17];
    let palm_w = dist2(index_mcp, pinky_mcp).max(1e-6);

    let thumb_tip = kpts[base + 4];
    let index_tip = kpts[base + 8];
    let middle_tip = kpts[base + 12];
    let ring_tip = kpts[base + 16];
    let pinky_tip = kpts[base + 20];

    Some([
        curl_from_tip(wrist, thumb_tip, palm_w),
        curl_from_tip(wrist, index_tip, palm_w),
        curl_from_tip(wrist, middle_tip, palm_w),
        curl_from_tip(wrist, ring_tip, palm_w),
        curl_from_tip(wrist, pinky_tip, palm_w),
    ])
}

fn finger_curls_from_hand_kpts_3d(kpts: &[[f32; 3]], base: usize) -> Option<[f32; 5]> {
    if kpts.len() < base + 21 {
        return None;
    }
    for index in [
        base,
        base + 4,
        base + 5,
        base + 8,
        base + 12,
        base + 16,
        base + 17,
        base + 20,
    ] {
        if !kpts.get(index).is_some_and(is_present_point3) {
            return None;
        }
    }
    let wrist = kpts[base];
    let index_mcp = kpts[base + 5];
    let pinky_mcp = kpts[base + 17];
    let palm_w = dist3(index_mcp, pinky_mcp).max(1e-6);

    let thumb_tip = kpts[base + 4];
    let index_tip = kpts[base + 8];
    let middle_tip = kpts[base + 12];
    let ring_tip = kpts[base + 16];
    let pinky_tip = kpts[base + 20];

    Some([
        curl_from_tip3(wrist, thumb_tip, palm_w),
        curl_from_tip3(wrist, index_tip, palm_w),
        curl_from_tip3(wrist, middle_tip, palm_w),
        curl_from_tip3(wrist, ring_tip, palm_w),
        curl_from_tip3(wrist, pinky_tip, palm_w),
    ])
}

fn curl_from_tip(wrist: [f32; 2], tip: [f32; 2], palm_w: f32) -> f32 {
    // open hand：tip 距 wrist 大；fist：tip 距 wrist 小
    let d = dist2(wrist, tip);
    // 经验尺度：用 palm_w * 3.0 归一化，落在 0..1
    (1.0 - (d / (palm_w * 3.0))).clamp(0.0, 1.0)
}

fn curl_from_tip3(wrist: [f32; 3], tip: [f32; 3], palm_w: f32) -> f32 {
    let d = dist3(wrist, tip);
    (1.0 - (d / (palm_w * 3.0))).clamp(0.0, 1.0)
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f32; 3], factor: f32) -> [f32; 3] {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = norm(v);
    if !n.is_finite() || n <= 1e-6 {
        [f32::NAN, f32::NAN, f32::NAN]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn is_finite3(v: [f32; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

/// 旋转矩阵 -> 四元数（x,y,z,w），行主序输入。
fn quat_from_rot(r: [[f32; 3]; 3]) -> [f32; 4] {
    let (r00, r01, r02) = (r[0][0], r[0][1], r[0][2]);
    let (r10, r11, r12) = (r[1][0], r[1][1], r[1][2]);
    let (r20, r21, r22) = (r[2][0], r[2][1], r[2][2]);

    let trace = r00 + r11 + r22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        let w = 0.25 * s;
        let x = (r21 - r12) / s;
        let y = (r02 - r20) / s;
        let z = (r10 - r01) / s;
        return normalize_quat([x, y, z, w]);
    }
    if r00 > r11 && r00 > r22 {
        let s = (1.0 + r00 - r11 - r22).sqrt() * 2.0;
        let w = (r21 - r12) / s;
        let x = 0.25 * s;
        let y = (r01 + r10) / s;
        let z = (r02 + r20) / s;
        return normalize_quat([x, y, z, w]);
    }
    if r11 > r22 {
        let s = (1.0 + r11 - r00 - r22).sqrt() * 2.0;
        let w = (r02 - r20) / s;
        let x = (r01 + r10) / s;
        let y = 0.25 * s;
        let z = (r12 + r21) / s;
        return normalize_quat([x, y, z, w]);
    }
    let s = (1.0 + r22 - r00 - r11).sqrt() * 2.0;
    let w = (r10 - r01) / s;
    let x = (r02 + r20) / s;
    let y = (r12 + r21) / s;
    let z = 0.25 * s;
    normalize_quat([x, y, z, w])
}

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !n.is_finite() || n <= 1e-6 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// 将 operator_frame 下的 pose 映射到 robot_base_frame（使用 Config 外参）。
pub fn transform_pose_operator_to_robot(pose: &Pose, cfg: &Config) -> Pose {
    let q = normalize_quat(cfg.extrinsic_rotation_quat);
    let t = cfg.extrinsic_translation_m;
    let pos_r = quat_rotate_vec3(q, pose.pos);
    let pos = [pos_r[0] + t[0], pos_r[1] + t[1], pos_r[2] + t[2]];
    let quat = normalize_quat(quat_mul(q, pose.quat));
    Pose { pos, quat }
}

pub fn transform_point_operator_to_robot(point: [f32; 3], cfg: &Config) -> [f32; 3] {
    let q = normalize_quat(cfg.extrinsic_rotation_quat);
    let t = cfg.extrinsic_translation_m;
    let pos_r = quat_rotate_vec3(q, point);
    [pos_r[0] + t[0], pos_r[1] + t[1], pos_r[2] + t[2]]
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // (x,y,z,w)
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

fn quat_rotate_vec3(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    // v' = q * (v,0) * conj(q)
    let qv = [v[0], v[1], v[2], 0.0];
    let r = quat_mul(quat_mul(q, qv), quat_conj(q));
    [r[0], r[1], r[2]]
}
