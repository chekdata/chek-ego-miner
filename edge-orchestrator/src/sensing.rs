use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::Value;
use tokio::net::UdpSocket;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, info, warn};

use crate::recorder::session_recorder::CsiPacketMeta;

const ADR018_CSI_MAGIC_V1: u32 = 0xC5110001;
const ADR018_CSI_MAGIC_V2: u32 = 0xC5110005;
const ADR018_CSI_HEADER_SIZE_V1: usize = 20;
const ADR018_CSI_HEADER_SIZE_V2: usize = 28;

#[cfg(test)]
const VISION_PART_HOLD_MS: u64 = 40;
#[cfg(not(test))]
const VISION_PART_HOLD_MS: u64 = 6_000;

#[cfg(test)]
const VISION_BODY_3D_HOLD_MS: u64 = 120;
#[cfg(not(test))]
const VISION_BODY_3D_HOLD_MS: u64 = 10_000;

#[cfg(test)]
const VISION_SINGLE_HAND_SIDE_HOLD_MS: u64 = 120;
#[cfg(not(test))]
const VISION_SINGLE_HAND_SIDE_HOLD_MS: u64 = 6_000;

const VISION_HAND_3D_OUTLIER_NEAREST_NEIGHBOR_MAX_M: f32 = 0.18;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BodyKeypointLayout {
    #[default]
    Unknown,
    CocoBody17,
    PicoBody24,
}

impl BodyKeypointLayout {
    pub fn resolve(declared: Option<&str>, counts: &[usize]) -> Self {
        Self::parse(declared).unwrap_or_else(|| Self::infer(counts))
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::CocoBody17 => "coco_body_17",
            Self::PicoBody24 => "pico_body_24",
        }
    }

    fn parse(value: Option<&str>) -> Option<Self> {
        match value {
            Some("coco_body_17") => Some(Self::CocoBody17),
            Some("pico_body_24") => Some(Self::PicoBody24),
            _ => None,
        }
    }

    fn infer(counts: &[usize]) -> Self {
        for count in counts {
            match *count {
                17 => return Self::CocoBody17,
                24 => return Self::PicoBody24,
                _ => {}
            }
        }
        Self::Unknown
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HandKeypointLayout {
    #[default]
    Unknown,
    MediapipeHand21,
    PicoHand26,
}

impl HandKeypointLayout {
    pub fn resolve(declared: Option<&str>, counts: &[usize]) -> Self {
        Self::parse(declared).unwrap_or_else(|| Self::infer(counts))
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::MediapipeHand21 => "mediapipe_hand_21",
            Self::PicoHand26 => "pico_hand_26",
        }
    }

    fn parse(value: Option<&str>) -> Option<Self> {
        match value {
            Some("mediapipe_hand_21") => Some(Self::MediapipeHand21),
            Some("pico_hand_26") => Some(Self::PicoHand26),
            _ => None,
        }
    }

    fn infer(counts: &[usize]) -> Self {
        for count in counts {
            match *count {
                21 | 42 => return Self::MediapipeHand21,
                26 | 52 => return Self::PicoHand26,
                _ => {}
            }
        }
        Self::Unknown
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandSide {
    Left,
    Right,
}

fn hand_points_per_side(layout: HandKeypointLayout) -> Option<usize> {
    match layout {
        HandKeypointLayout::MediapipeHand21 => Some(21),
        HandKeypointLayout::PicoHand26 => Some(26),
        HandKeypointLayout::Unknown => None,
    }
}

fn valid_point2(point: &[f32; 2]) -> bool {
    point[0].is_finite() && point[1].is_finite() && (point[0].abs() > 1e-6 || point[1].abs() > 1e-6)
}

fn valid_point3(point: &[f32; 3]) -> bool {
    point[0].is_finite()
        && point[1].is_finite()
        && point[2].is_finite()
        && (point[0].abs() > 1e-6 || point[1].abs() > 1e-6 || point[2].abs() > 1e-6)
}

fn split_hand_points<const D: usize>(
    points: &[[f32; D]],
    layout: HandKeypointLayout,
) -> Vec<Vec<[f32; D]>> {
    let Some(hand_len) = hand_points_per_side(layout) else {
        return if points.is_empty() {
            Vec::new()
        } else {
            vec![points.to_vec()]
        };
    };
    if points.len() % hand_len != 0 {
        return vec![points.to_vec()];
    }
    points
        .chunks_exact(hand_len)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn zero_points<const D: usize>(count: usize) -> Vec<[f32; D]> {
    vec![[0.0; D]; count]
}

fn combine_hand_points<const D: usize>(
    left: &[[f32; D]],
    right: &[[f32; D]],
    layout: HandKeypointLayout,
) -> Vec<[f32; D]> {
    let Some(hand_len) = hand_points_per_side(layout) else {
        if left.is_empty() {
            return right.to_vec();
        }
        let mut combined = left.to_vec();
        combined.extend_from_slice(right);
        return combined;
    };

    let left_present = left.len() == hand_len;
    let right_present = right.len() == hand_len;
    if (!left.is_empty() && !left_present) || (!right.is_empty() && !right_present) {
        let mut combined = left.to_vec();
        combined.extend_from_slice(right);
        return combined;
    }

    match (left_present, right_present) {
        (false, false) => Vec::new(),
        (true, false) => left.to_vec(),
        (false, true) => {
            let mut combined = zero_points::<D>(hand_len);
            combined.extend_from_slice(right);
            combined
        }
        (true, true) => {
            let mut combined = Vec::with_capacity(hand_len * 2);
            combined.extend_from_slice(left);
            combined.extend_from_slice(right);
            combined
        }
    }
}

fn raw_body_wrist_index(layout: BodyKeypointLayout, is_left: bool) -> Option<usize> {
    match layout {
        BodyKeypointLayout::CocoBody17 => Some(if is_left { 9 } else { 10 }),
        BodyKeypointLayout::PicoBody24 => Some(if is_left { 20 } else { 21 }),
        BodyKeypointLayout::Unknown => None,
    }
}

fn body_wrist_2d(
    points: &[[f32; 2]],
    layout: BodyKeypointLayout,
    side: HandSide,
) -> Option<[f32; 2]> {
    let index = raw_body_wrist_index(layout, side == HandSide::Left)?;
    points.get(index).copied().filter(valid_point2)
}

fn body_wrist_3d(
    points: &[[f32; 3]],
    layout: BodyKeypointLayout,
    side: HandSide,
) -> Option<[f32; 3]> {
    let index = raw_body_wrist_index(layout, side == HandSide::Left)?;
    points.get(index).copied().filter(valid_point3)
}

fn hand_wrist_2d(points: &[[f32; 2]]) -> Option<[f32; 2]> {
    points.first().copied().filter(valid_point2)
}

fn hand_wrist_3d(points: &[[f32; 3]]) -> Option<[f32; 3]> {
    points.first().copied().filter(valid_point3)
}

fn dist2_sq(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn dist3_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn filter_hand_point_outliers(points: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let valid_indices: Vec<usize> = points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| valid_point3(point).then_some(index))
        .collect();
    if valid_indices.len() < 4 {
        return points.to_vec();
    }

    let mut filtered = points.to_vec();
    for &index in &valid_indices {
        let point = points[index];
        let mut nearest_sq = f32::INFINITY;
        for &other_index in &valid_indices {
            if other_index == index {
                continue;
            }
            nearest_sq = nearest_sq.min(dist3_sq(point, points[other_index]));
        }
        if nearest_sq.is_finite()
            && nearest_sq
                > VISION_HAND_3D_OUTLIER_NEAREST_NEIGHBOR_MAX_M
                    * VISION_HAND_3D_OUTLIER_NEAREST_NEIGHBOR_MAX_M
        {
            filtered[index] = [0.0, 0.0, 0.0];
        }
    }
    filtered
}

/// 视觉输入的轻量快照（用于融合与可观测）。
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct VisionDevicePose {
    pub position_m: [f32; 3],
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub target_space: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation_deg: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_vector: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub up_vector: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forward_vector: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracking_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub world_mapping_status: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct VisionImuSample {
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
}

impl<'de> Deserialize<'de> for VisionDevicePose {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        parse_device_pose(Some(&value))
            .ok_or_else(|| de::Error::custom("invalid device_pose payload"))
    }
}

impl<'de> Deserialize<'de> for VisionImuSample {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        parse_imu(Some(&value)).ok_or_else(|| de::Error::custom("invalid imu payload"))
    }
}

#[derive(Clone, Debug, Default)]
pub struct VisionSnapshot {
    pub vision_conf: f32,
    pub body_conf: f32,
    pub hand_conf: f32,
    pub device_id: Option<String>,
    pub platform: String,
    pub operator_track_id: Option<String>,
    pub operator_track_sample_count: u64,
    pub operator_track_switch_count: u64,
    pub metrics_trip_id: String,
    pub metrics_session_id: String,
    pub timeout_count: u64,
    /// 2D 关键点（通常是归一化坐标；由 Capture Client 产生）。
    pub body_kpts_2d: Vec<[f32; 2]>,
    pub hand_kpts_2d: Vec<[f32; 2]>,
    pub left_hand_kpts_2d: Vec<[f32; 2]>,
    pub right_hand_kpts_2d: Vec<[f32; 2]>,
    /// 可选 3D 关键点（若 Capture Client 侧已能提供 3D，可直接上传）。
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    pub left_hand_kpts_3d: Vec<[f32; 3]>,
    pub right_hand_kpts_3d: Vec<[f32; 3]>,
    pub body_3d_recent: bool,
    pub hand_3d_recent: bool,
    pub left_hand_fresh_3d: bool,
    pub right_hand_fresh_3d: bool,
    pub body_layout: BodyKeypointLayout,
    pub hand_layout: HandKeypointLayout,
    pub body_3d_source: String,
    pub hand_3d_source: String,
    pub execution_mode: String,
    pub aux_snapshot_present: bool,
    pub aux_body_points_2d_valid: u32,
    pub aux_hand_points_2d_valid: u32,
    pub aux_body_points_3d_filled: u32,
    pub aux_hand_points_3d_filled: u32,
    pub aux_support_state: String,
    pub device_class: String,
    pub camera_mode: String,
    pub camera_has_depth: bool,
    /// 深度摘要（MVP：用于 2D->3D 的粗略投影）。
    pub depth_z_mean_m: Option<f32>,
    /// 图像尺寸（若上传 2D 像素坐标，edge 可据此归一化）。
    pub image_w: Option<u32>,
    pub image_h: Option<u32>,
    pub device_pose: Option<VisionDevicePose>,
    pub imu: Option<VisionImuSample>,
    pub last_edge_time_ns: u64,
    pub last_recv_time_ns: u64,
    pub iphone_to_edge_latency_ms: f32,
    pub last_at: Option<Instant>,
    pub fresh: bool,
}

#[derive(Default)]
struct VisionInner {
    body_conf: f32,
    hand_conf: f32,
    vision_conf: f32,
    device_id: Option<String>,
    platform: String,
    operator_track_id: Option<String>,
    operator_track_sample_count: u64,
    operator_track_switch_count: u64,
    last_counted_operator_track_id: Option<String>,
    metrics_trip_id: String,
    metrics_session_id: String,
    timeout_count: u64,
    timeout_armed: bool,
    body_kpts_2d: Vec<[f32; 2]>,
    hand_kpts_2d: Vec<[f32; 2]>,
    left_hand_kpts_2d: Vec<[f32; 2]>,
    right_hand_kpts_2d: Vec<[f32; 2]>,
    body_kpts_3d: Vec<[f32; 3]>,
    hand_kpts_3d: Vec<[f32; 3]>,
    left_hand_kpts_3d: Vec<[f32; 3]>,
    right_hand_kpts_3d: Vec<[f32; 3]>,
    body_layout: BodyKeypointLayout,
    hand_layout: HandKeypointLayout,
    body_3d_source: String,
    hand_3d_source: String,
    execution_mode: String,
    aux_snapshot_present: bool,
    aux_body_points_2d_valid: u32,
    aux_hand_points_2d_valid: u32,
    aux_body_points_3d_filled: u32,
    aux_hand_points_3d_filled: u32,
    aux_support_state: String,
    device_class: String,
    camera_mode: String,
    camera_has_depth: bool,
    depth_z_mean_m: Option<f32>,
    image_w: Option<u32>,
    image_h: Option<u32>,
    device_pose: Option<VisionDevicePose>,
    imu: Option<VisionImuSample>,
    last_edge_time_ns: u64,
    last_recv_time_ns: u64,
    iphone_to_edge_latency_ms: f32,
    last_at: Option<Instant>,
    body_last_points_at: Option<Instant>,
    body_last_3d_points_at: Option<Instant>,
    left_hand_last_points_at: Option<Instant>,
    right_hand_last_points_at: Option<Instant>,
    left_hand_last_3d_points_at: Option<Instant>,
    right_hand_last_3d_points_at: Option<Instant>,
    left_hand_last_3d_edge_time_ns: Option<u64>,
    right_hand_last_3d_edge_time_ns: Option<u64>,
}

/// 视觉输入缓存：由 `/stream/fusion` 上行 `capture_pose_packet`
/// 或 `POST /ingest/phone_vision_frame` 转写后的 `capture_pose_packet` 更新。
pub struct VisionStore {
    inner: RwLock<VisionInner>,
    allow_simulated_capture: bool,
}

impl Default for VisionStore {
    fn default() -> Self {
        Self::new(false)
    }
}

fn is_simulated_vision_source(device_id: &str, platform: &str) -> bool {
    let normalized_device_id = device_id.trim().to_ascii_lowercase();
    let normalized_platform = platform.trim().to_ascii_lowercase();
    normalized_platform == "simulated_capture"
        || normalized_device_id.starts_with("capture-sim-")
        || normalized_device_id.starts_with("sim-")
        || normalized_device_id.contains("operator-debug-sim")
}

fn side_has_hand_points(inner: &VisionInner, side: HandSide) -> bool {
    match side {
        HandSide::Left => {
            !inner.left_hand_kpts_2d.is_empty() || !inner.left_hand_kpts_3d.is_empty()
        }
        HandSide::Right => {
            !inner.right_hand_kpts_2d.is_empty() || !inner.right_hand_kpts_3d.is_empty()
        }
    }
}

fn side_has_hand_points_3d(inner: &VisionInner, side: HandSide) -> bool {
    match side {
        HandSide::Left => inner.left_hand_kpts_3d.iter().any(valid_point3),
        HandSide::Right => inner.right_hand_kpts_3d.iter().any(valid_point3),
    }
}

fn side_last_points_at(inner: &VisionInner, side: HandSide) -> Option<Instant> {
    match side {
        HandSide::Left => inner.left_hand_last_points_at,
        HandSide::Right => inner.right_hand_last_points_at,
    }
}

fn side_last_points_3d_at(inner: &VisionInner, side: HandSide) -> Option<Instant> {
    match side {
        HandSide::Left => inner.left_hand_last_3d_points_at,
        HandSide::Right => inner.right_hand_last_3d_points_at,
    }
}

fn side_hand_points_3d<'a>(inner: &'a VisionInner, side: HandSide) -> &'a [[f32; 3]] {
    match side {
        HandSide::Left => &inner.left_hand_kpts_3d,
        HandSide::Right => &inner.right_hand_kpts_3d,
    }
}

fn merge_sparse_hand_points_3d(
    inner: &VisionInner,
    side: HandSide,
    points_3d: &[[f32; 3]],
    now: Instant,
    hold_window: Duration,
) -> Vec<[f32; 3]> {
    let cache_recent = side_last_points_3d_at(inner, side)
        .is_some_and(|ts| now.duration_since(ts) <= hold_window)
        && side_has_hand_points_3d(inner, side);
    if !cache_recent {
        return filter_hand_point_outliers(points_3d);
    }

    let cached = side_hand_points_3d(inner, side);
    if points_3d.is_empty() {
        return cached.to_vec();
    }
    if cached.len() != points_3d.len() {
        return points_3d.to_vec();
    }

    let merged = points_3d
        .iter()
        .zip(cached.iter())
        .map(|(incoming, previous)| {
            if valid_point3(incoming) {
                *incoming
            } else {
                *previous
            }
        })
        .collect::<Vec<_>>();
    filter_hand_point_outliers(&merged)
}

fn side_assignment_cost(
    inner: &VisionInner,
    side: HandSide,
    points_2d: &[[f32; 2]],
    points_3d: &[[f32; 3]],
) -> Option<f32> {
    let mut total = 0.0f32;
    let mut weight = 0.0f32;

    if let Some(wrist) = hand_wrist_3d(points_3d) {
        if let Some(body_wrist) = body_wrist_3d(&inner.body_kpts_3d, inner.body_layout, side) {
            total += dist3_sq(wrist, body_wrist);
            weight += 1.0;
        }
        let cache = match side {
            HandSide::Left => hand_wrist_3d(&inner.left_hand_kpts_3d),
            HandSide::Right => hand_wrist_3d(&inner.right_hand_kpts_3d),
        };
        if let Some(cache_wrist) = cache {
            total += dist3_sq(wrist, cache_wrist) * 0.35;
            weight += 0.35;
        }
    }

    if let Some(wrist) = hand_wrist_2d(points_2d) {
        if let Some(body_wrist) = body_wrist_2d(&inner.body_kpts_2d, inner.body_layout, side) {
            total += dist2_sq(wrist, body_wrist);
            weight += 1.0;
        }
        let cache = match side {
            HandSide::Left => hand_wrist_2d(&inner.left_hand_kpts_2d),
            HandSide::Right => hand_wrist_2d(&inner.right_hand_kpts_2d),
        };
        if let Some(cache_wrist) = cache {
            total += dist2_sq(wrist, cache_wrist) * 0.35;
            weight += 0.35;
        }
    }

    if weight > 0.0 {
        Some(total / weight)
    } else {
        None
    }
}

fn infer_single_hand_side(
    inner: &VisionInner,
    points_2d: &[[f32; 2]],
    points_3d: &[[f32; 3]],
) -> HandSide {
    let left_cost = side_assignment_cost(inner, HandSide::Left, points_2d, points_3d);
    let right_cost = side_assignment_cost(inner, HandSide::Right, points_2d, points_3d);
    match (left_cost, right_cost) {
        (Some(left), Some(right)) => {
            if left <= right {
                HandSide::Left
            } else {
                HandSide::Right
            }
        }
        (Some(_), None) => HandSide::Left,
        (None, Some(_)) => HandSide::Right,
        (None, None) => match (
            side_has_hand_points(inner, HandSide::Left),
            side_has_hand_points(inner, HandSide::Right),
        ) {
            (false, true) => HandSide::Left,
            (true, false) => HandSide::Right,
            _ => HandSide::Left,
        },
    }
}

fn dual_hand_assignment_cost(
    inner: &VisionInner,
    first_side: HandSide,
    first_points_2d: &[[f32; 2]],
    first_points_3d: &[[f32; 3]],
    second_side: HandSide,
    second_points_2d: &[[f32; 2]],
    second_points_3d: &[[f32; 3]],
) -> Option<f32> {
    let first = side_assignment_cost(inner, first_side, first_points_2d, first_points_3d);
    let second = side_assignment_cost(inner, second_side, second_points_2d, second_points_3d);
    match (first, second) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn infer_dual_hand_sides(
    inner: &VisionInner,
    first_points_2d: &[[f32; 2]],
    first_points_3d: &[[f32; 3]],
    second_points_2d: &[[f32; 2]],
    second_points_3d: &[[f32; 3]],
) -> [HandSide; 2] {
    let direct = dual_hand_assignment_cost(
        inner,
        HandSide::Left,
        first_points_2d,
        first_points_3d,
        HandSide::Right,
        second_points_2d,
        second_points_3d,
    );
    let swapped = dual_hand_assignment_cost(
        inner,
        HandSide::Right,
        first_points_2d,
        first_points_3d,
        HandSide::Left,
        second_points_2d,
        second_points_3d,
    );
    match (direct, swapped) {
        (Some(direct_cost), Some(swapped_cost)) if swapped_cost < direct_cost => {
            [HandSide::Right, HandSide::Left]
        }
        (None, Some(_)) => [HandSide::Right, HandSide::Left],
        _ => [HandSide::Left, HandSide::Right],
    }
}

fn set_hand_side(
    inner: &mut VisionInner,
    side: HandSide,
    points_2d: Vec<[f32; 2]>,
    points_3d: Vec<[f32; 3]>,
    now: Instant,
    edge_time_ns: u64,
    saw_fresh_3d_points: bool,
) {
    match side {
        HandSide::Left => {
            inner.left_hand_kpts_2d = points_2d;
            inner.left_hand_kpts_3d = points_3d;
            inner.left_hand_last_points_at = Some(now);
            if saw_fresh_3d_points {
                inner.left_hand_last_3d_points_at = Some(now);
                inner.left_hand_last_3d_edge_time_ns = Some(edge_time_ns);
            }
        }
        HandSide::Right => {
            inner.right_hand_kpts_2d = points_2d;
            inner.right_hand_kpts_3d = points_3d;
            inner.right_hand_last_points_at = Some(now);
            if saw_fresh_3d_points {
                inner.right_hand_last_3d_points_at = Some(now);
                inner.right_hand_last_3d_edge_time_ns = Some(edge_time_ns);
            }
        }
    }
}

fn clear_hand_side(inner: &mut VisionInner, side: HandSide) {
    match side {
        HandSide::Left => {
            inner.left_hand_kpts_2d.clear();
            inner.left_hand_kpts_3d.clear();
            inner.left_hand_last_points_at = None;
            inner.left_hand_last_3d_points_at = None;
            inner.left_hand_last_3d_edge_time_ns = None;
        }
        HandSide::Right => {
            inner.right_hand_kpts_2d.clear();
            inner.right_hand_kpts_3d.clear();
            inner.right_hand_last_points_at = None;
            inner.right_hand_last_3d_points_at = None;
            inner.right_hand_last_3d_edge_time_ns = None;
        }
    }
}

fn refresh_combined_hand_points(inner: &mut VisionInner) {
    inner.hand_kpts_2d = combine_hand_points(
        &inner.left_hand_kpts_2d,
        &inner.right_hand_kpts_2d,
        inner.hand_layout,
    );
    inner.hand_kpts_3d = combine_hand_points(
        &inner.left_hand_kpts_3d,
        &inner.right_hand_kpts_3d,
        inner.hand_layout,
    );
}

impl VisionStore {
    pub fn new(allow_simulated_capture: bool) -> Self {
        Self {
            inner: RwLock::new(VisionInner::default()),
            allow_simulated_capture,
        }
    }

    pub fn ingest_capture_pose_json(
        &self,
        v: &Value,
        edge_time_ns: u64,
        recv_time_ns: u64,
    ) -> bool {
        // PRD 示例：confidence.body / confidence.hand
        let body = v
            .get("confidence")
            .and_then(|c| c.get("body"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as f32;
        let hand = v
            .get("confidence")
            .and_then(|c| c.get("hand"))
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as f32;
        let body_kpts_2d = parse_vec2_array(v.get("body_kpts_2d"));
        let hand_kpts_2d = parse_vec2_array(v.get("hand_kpts_2d"));
        let body_kpts_3d = parse_vec3_array(v.get("body_kpts_3d"));
        let hand_kpts_3d = parse_vec3_array(v.get("hand_kpts_3d"));
        let has_body_points = !body_kpts_2d.is_empty() || !body_kpts_3d.is_empty();
        let has_hand_points = !hand_kpts_2d.is_empty() || !hand_kpts_3d.is_empty();
        let metrics_trip_id = v
            .get("trip_id")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .unwrap_or_default()
            .to_string();
        let metrics_session_id = v
            .get("session_id")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .unwrap_or_default()
            .to_string();
        let operator_track_id = v
            .get("operator_track_id")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(ToOwned::to_owned);
        let body_layout = BodyKeypointLayout::resolve(
            v.get("body_layout").and_then(|x| x.as_str()),
            &[body_kpts_3d.len(), body_kpts_2d.len()],
        );
        let hand_layout = HandKeypointLayout::resolve(
            v.get("hand_layout").and_then(|x| x.as_str()),
            &[hand_kpts_3d.len(), hand_kpts_2d.len()],
        );
        let depth_z_mean_m = v
            .get("depth_summary")
            .and_then(|d| d.get("z_mean_m"))
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .filter(|x| x.is_finite());
        let capture_profile = v.get("capture_profile");
        let body_3d_source = capture_profile
            .and_then(|profile| profile.get("body_3d_source"))
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let hand_3d_source = capture_profile
            .and_then(|profile| profile.get("hand_3d_source"))
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let execution_mode = capture_profile
            .and_then(|profile| profile.get("execution_mode"))
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let vision_processing_enabled = capture_profile
            .and_then(|profile| profile.get("vision_processing_enabled"))
            .and_then(|x| x.as_bool())
            .unwrap_or(true);
        let aux_snapshot_present = capture_profile
            .and_then(|profile| profile.get("aux_snapshot_present"))
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let aux_body_points_2d_valid = capture_profile
            .and_then(|profile| profile.get("aux_body_points_2d_valid"))
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok())
            .unwrap_or(0);
        let aux_hand_points_2d_valid = capture_profile
            .and_then(|profile| profile.get("aux_hand_points_2d_valid"))
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok())
            .unwrap_or(0);
        let aux_body_points_3d_filled = capture_profile
            .and_then(|profile| profile.get("aux_body_points_3d_filled"))
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok())
            .unwrap_or(0);
        let aux_hand_points_3d_filled = capture_profile
            .and_then(|profile| profile.get("aux_hand_points_3d_filled"))
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok())
            .unwrap_or(0);
        let aux_support_state = capture_profile
            .and_then(|profile| profile.get("aux_support_state"))
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let image_w = v
            .get("camera")
            .and_then(|c| c.get("image_w"))
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok());
        let image_h = v
            .get("camera")
            .and_then(|c| c.get("image_h"))
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok());
        let camera_mode = v
            .get("camera")
            .and_then(|c| c.get("mode"))
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let camera_has_depth = v
            .get("camera")
            .and_then(|c| c.get("has_depth"))
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let device_id = v
            .get("device_id")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(ToOwned::to_owned);
        let platform = v
            .get("platform")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .unwrap_or_default()
            .to_string();
        let device_class = v
            .get("device_class")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let device_pose = parse_device_pose(v.get("device_pose"));
        let imu = parse_imu(v.get("imu"));

        let now = Instant::now();
        let mut inner = self.inner.write().expect("vision lock poisoned");
        let incoming_is_simulated = device_id.as_deref().is_some_and(|incoming_device_id| {
            is_simulated_vision_source(incoming_device_id, platform.as_str())
        }) || (!platform.is_empty()
            && is_simulated_vision_source("", platform.as_str()));
        if incoming_is_simulated && !self.allow_simulated_capture {
            return false;
        }
        let current_real_vision_is_fresh = inner.last_at.is_some_and(|last_at| {
            let current_device_id = inner.device_id.as_deref().unwrap_or_default();
            !is_simulated_vision_source(current_device_id, inner.platform.as_str())
                && now.duration_since(last_at) <= Duration::from_millis(10_000)
        });
        if incoming_is_simulated && current_real_vision_is_fresh {
            return false;
        }
        if !metrics_trip_id.is_empty()
            && !metrics_session_id.is_empty()
            && (inner.metrics_trip_id != metrics_trip_id
                || inner.metrics_session_id != metrics_session_id)
        {
            inner.operator_track_sample_count = 0;
            inner.operator_track_switch_count = 0;
            inner.last_counted_operator_track_id = None;
            inner.metrics_trip_id = metrics_trip_id.clone();
            inner.metrics_session_id = metrics_session_id.clone();
            inner.timeout_count = 0;
            inner.timeout_armed = false;
        }
        let hold_window = Duration::from_millis(VISION_PART_HOLD_MS);
        let body_3d_hold_window = Duration::from_millis(VISION_BODY_3D_HOLD_MS);
        let single_hand_side_hold_window = Duration::from_millis(VISION_SINGLE_HAND_SIDE_HOLD_MS);
        let explicit_passthrough =
            !vision_processing_enabled || execution_mode == "device_pose_passthrough";
        let keep_body_points = !explicit_passthrough
            && !has_body_points
            && inner
                .body_last_points_at
                .is_some_and(|ts| now.duration_since(ts) <= hold_window)
            && (!inner.body_kpts_2d.is_empty() || !inner.body_kpts_3d.is_empty());
        let incoming_body_has_3d = body_kpts_3d.iter().any(valid_point3);
        let incoming_hand_has_3d = hand_kpts_3d.iter().any(valid_point3);
        let keep_body_3d_points = !explicit_passthrough
            && !incoming_body_has_3d
            && inner
                .body_last_3d_points_at
                .is_some_and(|ts| now.duration_since(ts) <= body_3d_hold_window)
            && inner.body_kpts_3d.iter().any(valid_point3);
        let keep_left_hand_points = !explicit_passthrough
            && side_last_points_at(&inner, HandSide::Left)
                .is_some_and(|ts| now.duration_since(ts) <= hold_window)
            && side_has_hand_points(&inner, HandSide::Left);
        let keep_right_hand_points = !explicit_passthrough
            && side_last_points_at(&inner, HandSide::Right)
                .is_some_and(|ts| now.duration_since(ts) <= hold_window)
            && side_has_hand_points(&inner, HandSide::Right);
        let keep_left_hand_side_for_single_hand = !explicit_passthrough
            && side_last_points_at(&inner, HandSide::Left)
                .is_some_and(|ts| now.duration_since(ts) <= single_hand_side_hold_window)
            && side_has_hand_points(&inner, HandSide::Left);
        let keep_right_hand_side_for_single_hand = !explicit_passthrough
            && side_last_points_at(&inner, HandSide::Right)
                .is_some_and(|ts| now.duration_since(ts) <= single_hand_side_hold_window)
            && side_has_hand_points(&inner, HandSide::Right);

        if has_body_points {
            inner.body_kpts_2d = body_kpts_2d;
            inner.body_layout = body_layout;
            inner.body_conf = body.clamp(0.0, 1.0);
            inner.body_last_points_at = Some(now);
            if incoming_body_has_3d {
                inner.body_kpts_3d = body_kpts_3d;
                inner.body_last_3d_points_at = Some(now);
            } else if !keep_body_3d_points {
                inner.body_kpts_3d.clear();
                inner.body_last_3d_points_at = None;
            }
        } else if !keep_body_points && !keep_body_3d_points {
            inner.body_kpts_2d.clear();
            inner.body_kpts_3d.clear();
            inner.body_layout = body_layout;
            inner.body_conf = 0.0;
            inner.body_last_points_at = None;
            inner.body_last_3d_points_at = None;
        } else if !keep_body_points {
            inner.body_kpts_2d.clear();
            inner.body_layout = body_layout;
            inner.body_last_points_at = None;
        }

        if has_hand_points {
            inner.hand_layout = hand_layout;
            inner.hand_conf = hand.clamp(0.0, 1.0);
            let hand_chunks_2d = split_hand_points(&hand_kpts_2d, hand_layout);
            let hand_chunks_3d = split_hand_points(&hand_kpts_3d, hand_layout);
            let chunk_count = hand_chunks_2d.len().max(hand_chunks_3d.len());
            let dual_hand_sides = if chunk_count == 2 {
                Some(infer_dual_hand_sides(
                    &inner,
                    hand_chunks_2d.get(0).map(Vec::as_slice).unwrap_or(&[]),
                    hand_chunks_3d.get(0).map(Vec::as_slice).unwrap_or(&[]),
                    hand_chunks_2d.get(1).map(Vec::as_slice).unwrap_or(&[]),
                    hand_chunks_3d.get(1).map(Vec::as_slice).unwrap_or(&[]),
                ))
            } else {
                None
            };

            let mut seen_left = false;
            let mut seen_right = false;
            for index in 0..chunk_count {
                let points_2d = hand_chunks_2d.get(index).cloned().unwrap_or_default();
                let incoming_points_3d = hand_chunks_3d.get(index).cloned().unwrap_or_default();
                if points_2d.is_empty() && incoming_points_3d.is_empty() {
                    continue;
                }
                let side = if chunk_count == 1 {
                    infer_single_hand_side(&inner, &points_2d, &incoming_points_3d)
                } else {
                    dual_hand_sides
                        .as_ref()
                        .and_then(|sides| sides.get(index))
                        .copied()
                        .unwrap_or(if index == 0 {
                            HandSide::Left
                        } else {
                            HandSide::Right
                        })
                };
                let saw_fresh_3d_points = incoming_points_3d.iter().any(valid_point3);
                let points_3d = merge_sparse_hand_points_3d(
                    &inner,
                    side,
                    &incoming_points_3d,
                    now,
                    hold_window,
                );
                set_hand_side(
                    &mut inner,
                    side,
                    points_2d,
                    points_3d,
                    now,
                    edge_time_ns,
                    saw_fresh_3d_points,
                );
                match side {
                    HandSide::Left => seen_left = true,
                    HandSide::Right => seen_right = true,
                }
            }

            if !seen_left && !keep_left_hand_side_for_single_hand {
                clear_hand_side(&mut inner, HandSide::Left);
            }
            if !seen_right && !keep_right_hand_side_for_single_hand {
                clear_hand_side(&mut inner, HandSide::Right);
            }
        } else {
            if !keep_left_hand_points {
                clear_hand_side(&mut inner, HandSide::Left);
            }
            if !keep_right_hand_points {
                clear_hand_side(&mut inner, HandSide::Right);
            }
            if !side_has_hand_points(&inner, HandSide::Left)
                && !side_has_hand_points(&inner, HandSide::Right)
            {
                inner.hand_layout = hand_layout;
                inner.hand_conf = 0.0;
            }
        }

        refresh_combined_hand_points(&mut inner);

        let effective_body = !inner.body_kpts_2d.is_empty() || !inner.body_kpts_3d.is_empty();
        let effective_hand = !inner.hand_kpts_2d.is_empty() || !inner.hand_kpts_3d.is_empty();
        inner.vision_conf = match (effective_body, effective_hand) {
            (true, true) => inner.body_conf.min(inner.hand_conf),
            (true, false) => inner.body_conf,
            (false, true) => inner.hand_conf,
            (false, false) => 0.0,
        };
        inner.device_id = device_id;
        inner.platform = platform;
        if let Some(track_id) = operator_track_id.as_deref() {
            inner.operator_track_sample_count = inner.operator_track_sample_count.saturating_add(1);
            if inner
                .last_counted_operator_track_id
                .as_deref()
                .is_some_and(|previous| previous != track_id)
            {
                inner.operator_track_switch_count =
                    inner.operator_track_switch_count.saturating_add(1);
            }
            inner.last_counted_operator_track_id = Some(track_id.to_string());
        }
        inner.operator_track_id = operator_track_id;
        inner.depth_z_mean_m = depth_z_mean_m;
        if incoming_body_has_3d {
            inner.body_3d_source = if body_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                body_3d_source
            };
        } else if !keep_body_3d_points {
            inner.body_3d_source = body_3d_source;
        }
        let effective_hand_has_3d = inner.hand_kpts_3d.iter().any(valid_point3);
        if incoming_hand_has_3d {
            inner.hand_3d_source = if hand_3d_source.trim().is_empty() {
                "phone_edge_vision".to_string()
            } else {
                hand_3d_source
            };
        } else if explicit_passthrough || !effective_hand_has_3d {
            inner.hand_3d_source = hand_3d_source;
        }
        inner.execution_mode = execution_mode;
        inner.aux_snapshot_present = aux_snapshot_present;
        inner.aux_body_points_2d_valid = aux_body_points_2d_valid;
        inner.aux_hand_points_2d_valid = aux_hand_points_2d_valid;
        inner.aux_body_points_3d_filled = aux_body_points_3d_filled;
        inner.aux_hand_points_3d_filled = aux_hand_points_3d_filled;
        inner.aux_support_state = aux_support_state;
        inner.device_class = device_class;
        inner.camera_mode = camera_mode;
        inner.camera_has_depth = camera_has_depth;
        inner.image_w = image_w;
        inner.image_h = image_h;
        inner.device_pose = device_pose;
        inner.imu = imu;
        inner.last_edge_time_ns = edge_time_ns;
        inner.last_recv_time_ns = recv_time_ns;
        inner.iphone_to_edge_latency_ms =
            ((recv_time_ns.saturating_sub(edge_time_ns)) as f32) / 1_000_000.0;
        inner.last_at = Some(now);
        inner.timeout_armed = true;
        true
    }

    pub fn snapshot(&self, stale_ms: u64) -> VisionSnapshot {
        let now = Instant::now();
        let mut inner = self.inner.write().expect("vision lock poisoned");
        let fresh = inner
            .last_at
            .is_some_and(|t| now.duration_since(t) <= Duration::from_millis(stale_ms));
        let body_3d_recent = inner.body_last_3d_points_at.is_some_and(|t| {
            now.duration_since(t) <= Duration::from_millis(VISION_BODY_3D_HOLD_MS)
        }) && inner.body_kpts_3d.iter().any(valid_point3);
        let hand_3d_recent = inner.left_hand_last_3d_points_at.is_some_and(|t| {
            now.duration_since(t) <= Duration::from_millis(VISION_SINGLE_HAND_SIDE_HOLD_MS)
        }) && !inner.left_hand_kpts_3d.is_empty()
            || inner.right_hand_last_3d_points_at.is_some_and(|t| {
                now.duration_since(t) <= Duration::from_millis(VISION_SINGLE_HAND_SIDE_HOLD_MS)
            }) && !inner.right_hand_kpts_3d.is_empty();
        if !fresh && inner.last_at.is_some() && inner.timeout_armed {
            inner.timeout_count = inner.timeout_count.saturating_add(1);
            inner.timeout_armed = false;
        }
        VisionSnapshot {
            vision_conf: inner.vision_conf,
            body_conf: inner.body_conf,
            hand_conf: inner.hand_conf,
            device_id: inner.device_id.clone(),
            platform: inner.platform.clone(),
            operator_track_id: inner.operator_track_id.clone(),
            operator_track_sample_count: inner.operator_track_sample_count,
            operator_track_switch_count: inner.operator_track_switch_count,
            metrics_trip_id: inner.metrics_trip_id.clone(),
            metrics_session_id: inner.metrics_session_id.clone(),
            timeout_count: inner.timeout_count,
            body_kpts_2d: inner.body_kpts_2d.clone(),
            hand_kpts_2d: inner.hand_kpts_2d.clone(),
            left_hand_kpts_2d: inner.left_hand_kpts_2d.clone(),
            right_hand_kpts_2d: inner.right_hand_kpts_2d.clone(),
            body_kpts_3d: inner.body_kpts_3d.clone(),
            hand_kpts_3d: inner.hand_kpts_3d.clone(),
            left_hand_kpts_3d: inner.left_hand_kpts_3d.clone(),
            right_hand_kpts_3d: inner.right_hand_kpts_3d.clone(),
            body_3d_recent,
            hand_3d_recent,
            left_hand_fresh_3d: fresh
                && inner
                    .left_hand_last_3d_edge_time_ns
                    .is_some_and(|ts| ts == inner.last_edge_time_ns)
                && side_has_hand_points_3d(&inner, HandSide::Left),
            right_hand_fresh_3d: fresh
                && inner
                    .right_hand_last_3d_edge_time_ns
                    .is_some_and(|ts| ts == inner.last_edge_time_ns)
                && side_has_hand_points_3d(&inner, HandSide::Right),
            body_layout: inner.body_layout,
            hand_layout: inner.hand_layout,
            body_3d_source: inner.body_3d_source.clone(),
            hand_3d_source: inner.hand_3d_source.clone(),
            execution_mode: inner.execution_mode.clone(),
            aux_snapshot_present: inner.aux_snapshot_present,
            aux_body_points_2d_valid: inner.aux_body_points_2d_valid,
            aux_hand_points_2d_valid: inner.aux_hand_points_2d_valid,
            aux_body_points_3d_filled: inner.aux_body_points_3d_filled,
            aux_hand_points_3d_filled: inner.aux_hand_points_3d_filled,
            aux_support_state: inner.aux_support_state.clone(),
            device_class: inner.device_class.clone(),
            camera_mode: inner.camera_mode.clone(),
            camera_has_depth: inner.camera_has_depth,
            depth_z_mean_m: inner.depth_z_mean_m,
            image_w: inner.image_w,
            image_h: inner.image_h,
            device_pose: inner.device_pose.clone(),
            imu: inner.imu.clone(),
            last_edge_time_ns: inner.last_edge_time_ns,
            last_recv_time_ns: inner.last_recv_time_ns,
            iphone_to_edge_latency_ms: inner.iphone_to_edge_latency_ms,
            last_at: inner.last_at,
            fresh,
        }
    }

    pub fn reset_identity_metrics(&self) {
        let mut inner = self.inner.write().expect("vision lock poisoned");
        inner.operator_track_sample_count = 0;
        inner.operator_track_switch_count = 0;
        inner.last_counted_operator_track_id = None;
        inner.timeout_count = 0;
        inner.timeout_armed = false;
    }
}

fn parse_vec2_array(v: Option<&Value>) -> Vec<[f32; 2]> {
    let Some(arr) = v.and_then(|x| x.as_array()) else {
        return Vec::new();
    };
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(p) = item.as_array() else { continue };
        if p.len() < 2 {
            continue;
        };
        let x = p[0].as_f64().unwrap_or(f64::NAN) as f32;
        let y = p[1].as_f64().unwrap_or(f64::NAN) as f32;
        if x.is_finite() && y.is_finite() {
            out.push([x, y]);
        }
    }
    out
}

fn parse_vec3_array(v: Option<&Value>) -> Vec<[f32; 3]> {
    let Some(arr) = v.and_then(|x| x.as_array()) else {
        return Vec::new();
    };
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(p) = item.as_array() else { continue };
        if p.len() < 3 {
            continue;
        };
        let x = p[0].as_f64().unwrap_or(f64::NAN) as f32;
        let y = p[1].as_f64().unwrap_or(f64::NAN) as f32;
        let z = p[2].as_f64().unwrap_or(f64::NAN) as f32;
        if x.is_finite() && y.is_finite() && z.is_finite() {
            out.push([x, y, z]);
        }
    }
    out
}

fn parse_vec3_value(v: Option<&Value>) -> Option<[f32; 3]> {
    let arr = v?.as_array()?;
    if arr.len() < 3 {
        return None;
    }
    let x = arr[0].as_f64().unwrap_or(f64::NAN) as f32;
    let y = arr[1].as_f64().unwrap_or(f64::NAN) as f32;
    let z = arr[2].as_f64().unwrap_or(f64::NAN) as f32;
    if x.is_finite() && y.is_finite() && z.is_finite() {
        Some([x, y, z])
    } else {
        None
    }
}

fn parse_vec4_value(v: Option<&Value>) -> Option<[f32; 4]> {
    let arr = v?.as_array()?;
    if arr.len() != 4 {
        return None;
    }
    let x = arr.first()?.as_f64()? as f32;
    let y = arr.get(1)?.as_f64()? as f32;
    let z = arr.get(2)?.as_f64()? as f32;
    let w = arr.get(3)?.as_f64()? as f32;
    if x.is_finite() && y.is_finite() && z.is_finite() && w.is_finite() {
        Some([x, y, z, w])
    } else {
        None
    }
}

fn first_object_value<'a>(
    obj: &'a serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<&'a Value> {
    keys.iter().find_map(|key| obj.get(*key))
}

fn parse_transform_columns(v: Option<&Value>) -> Option<[[f32; 4]; 4]> {
    let value = v?;
    if let Some(obj) = value.as_object() {
        return parse_transform_columns(
            obj.get("columns")
                .or_else(|| obj.get("matrix"))
                .or_else(|| obj.get("transform")),
        );
    }
    let items = value.as_array()?;
    if items.len() == 4 {
        let c0 = parse_vec4_value(items.first())?;
        let c1 = parse_vec4_value(items.get(1))?;
        let c2 = parse_vec4_value(items.get(2))?;
        let c3 = parse_vec4_value(items.get(3))?;
        return Some([c0, c1, c2, c3]);
    }
    if items.len() == 16 {
        let mut columns = [[0.0; 4]; 4];
        for column in 0..4 {
            for row in 0..4 {
                columns[column][row] = items.get(column * 4 + row)?.as_f64()? as f32;
            }
        }
        return Some(columns);
    }
    None
}

fn parse_device_pose(v: Option<&Value>) -> Option<VisionDevicePose> {
    let obj = v?.as_object()?;
    let transform_columns = first_object_value(
        obj,
        &[
            "transform_columns",
            "transformColumns",
            "world_transform_columns",
            "worldTransformColumns",
            "transform",
            "world_transform",
            "worldTransform",
            "camera_transform",
            "cameraTransform",
        ],
    )
    .and_then(|value| parse_transform_columns(Some(value)));
    let position_m = first_object_value(
        obj,
        &[
            "position_m",
            "positionM",
            "position",
            "translation_m",
            "translationM",
            "translation",
        ],
    )
    .and_then(|value| parse_vec3_value(Some(value)))
    .or_else(|| transform_columns.map(|columns| [columns[3][0], columns[3][1], columns[3][2]]))?;
    Some(VisionDevicePose {
        position_m,
        target_space: first_object_value(
            obj,
            &[
                "target_space",
                "targetSpace",
                "space",
                "frame",
                "reference_frame",
                "referenceFrame",
            ],
        )
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string(),
        rotation_deg: first_object_value(
            obj,
            &["rotation_deg", "rotationDeg", "euler_deg", "eulerDeg"],
        )
        .and_then(|value| parse_vec3_value(Some(value))),
        right_vector: first_object_value(
            obj,
            &["right_vector", "rightVector", "basis_right", "basisRight"],
        )
        .and_then(|value| parse_vec3_value(Some(value)))
        .or_else(|| transform_columns.map(|columns| [columns[0][0], columns[0][1], columns[0][2]])),
        up_vector: first_object_value(obj, &["up_vector", "upVector", "basis_up", "basisUp"])
            .and_then(|value| parse_vec3_value(Some(value)))
            .or_else(|| {
                transform_columns.map(|columns| [columns[1][0], columns[1][1], columns[1][2]])
            }),
        forward_vector: first_object_value(
            obj,
            &[
                "forward_vector",
                "forwardVector",
                "basis_forward",
                "basisForward",
                "front_vector",
                "frontVector",
            ],
        )
        .and_then(|value| parse_vec3_value(Some(value)))
        .or_else(|| transform_columns.map(|columns| [columns[2][0], columns[2][1], columns[2][2]])),
        timestamp_ns: first_object_value(obj, &["timestamp_ns", "timestampNs"])
            .and_then(|value| value.as_u64()),
        tracking_state: first_object_value(obj, &["tracking_state", "trackingState"])
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        world_mapping_status: first_object_value(
            obj,
            &["world_mapping_status", "worldMappingStatus"],
        )
        .and_then(|value| value.as_str())
        .map(|value| value.to_string()),
        source: first_object_value(obj, &["source", "pose_source", "poseSource", "provider"])
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
    })
}

fn parse_imu(v: Option<&Value>) -> Option<VisionImuSample> {
    let obj = v?.as_object()?;
    let accel = first_object_value(
        obj,
        &[
            "accel",
            "accel_mps2",
            "accelMps2",
            "user_accel_mps2",
            "userAccelMps2",
        ],
    )
    .and_then(|value| parse_vec3_value(Some(value)));
    let gyro = first_object_value(
        obj,
        &[
            "gyro",
            "gyro_rps",
            "gyroRps",
            "rotation_rate_rps",
            "rotationRateRps",
        ],
    )
    .and_then(|value| parse_vec3_value(Some(value)));
    if accel.is_none() && gyro.is_none() {
        return None;
    }
    Some(VisionImuSample {
        accel: accel.unwrap_or([0.0, 0.0, 0.0]),
        gyro: gyro.unwrap_or([0.0, 0.0, 0.0]),
    })
}

#[cfg(test)]
mod vision_tests {
    use std::thread;
    use std::time::Duration;

    use serde_json::json;

    use crate::operator::STEREO_PAIR_FRAME;

    use super::{
        BodyKeypointLayout, HandKeypointLayout, StereoStore, StereoTrackedPersonSnapshot,
        VisionStore, VISION_PART_HOLD_MS, VISION_SINGLE_HAND_SIDE_HOLD_MS,
    };

    fn body_with_wrist_refs() -> Vec<[f32; 3]> {
        let mut body = vec![[0.0, 0.0, 0.0]; 17];
        body[9] = [-0.35, 0.0, 1.0];
        body[10] = [0.35, 0.0, 1.0];
        body
    }

    fn hand_points(base_x: f32) -> Vec<[f32; 3]> {
        (0..21)
            .map(|index| {
                [
                    base_x + index as f32 * 0.01,
                    0.05 + index as f32 * 0.002,
                    1.0,
                ]
            })
            .collect()
    }

    fn sparse_hand_points(base_x: f32, keep: &[usize]) -> Vec<[f32; 3]> {
        hand_points(base_x)
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                if keep.contains(&index) {
                    point
                } else {
                    [0.0, 0.0, 0.0]
                }
            })
            .collect()
    }

    fn outlier_hand_points(base_x: f32, outlier_index: usize, outlier: [f32; 3]) -> Vec<[f32; 3]> {
        hand_points(base_x)
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                if index == outlier_index {
                    outlier
                } else {
                    point
                }
            })
            .collect()
    }

    fn assert_point3_close(actual: [f32; 3], expected: [f32; 3]) {
        for index in 0..3 {
            assert!((actual[index] - expected[index]).abs() < 1e-5);
        }
    }

    #[test]
    fn hand_only_capture_keeps_nonzero_vision_conf() {
        let store = VisionStore::default();
        let hand_kpts_2d: Vec<[f32; 2]> = (0..42).map(|_| [0.1, 0.2]).collect();
        let hand_kpts_3d: Vec<[f32; 3]> = (0..42).map(|_| [0.1, 0.2, 0.3]).collect();
        let payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.82
            },
            "body_kpts_2d": [],
            "body_kpts_3d": [],
            "hand_kpts_2d": hand_kpts_2d,
            "hand_kpts_3d": hand_kpts_3d
        });

        store.ingest_capture_pose_json(&payload, 10, 20);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.body_conf, 0.0);
        assert_eq!(snapshot.hand_conf, 0.82);
        assert_eq!(snapshot.vision_conf, 0.82);
    }

    #[test]
    fn full_body_and_hand_capture_uses_min_confidence() {
        let store = VisionStore::default();
        let body_kpts_2d: Vec<[f32; 2]> = (0..17).map(|_| [0.1, 0.2]).collect();
        let body_kpts_3d: Vec<[f32; 3]> = (0..17).map(|_| [0.1, 0.2, 0.3]).collect();
        let hand_kpts_2d: Vec<[f32; 2]> = (0..42).map(|_| [0.1, 0.2]).collect();
        let hand_kpts_3d: Vec<[f32; 3]> = (0..42).map(|_| [0.1, 0.2, 0.3]).collect();
        let payload = json!({
            "confidence": {
                "body": 0.61,
                "hand": 0.84
            },
            "body_kpts_2d": body_kpts_2d,
            "body_kpts_3d": body_kpts_3d,
            "hand_kpts_2d": hand_kpts_2d,
            "hand_kpts_3d": hand_kpts_3d
        });

        store.ingest_capture_pose_json(&payload, 10, 20);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.body_conf, 0.61);
        assert_eq!(snapshot.hand_conf, 0.84);
        assert_eq!(snapshot.vision_conf, 0.61);
    }

    #[test]
    fn short_hand_drop_keeps_recent_hand_points() {
        let store = VisionStore::default();
        let hand_kpts_2d: Vec<[f32; 2]> = (0..42).map(|index| [0.1 + index as f32, 0.2]).collect();
        let hand_kpts_3d: Vec<[f32; 3]> = (0..42)
            .map(|index| [0.1 + index as f32 * 0.01, 0.2, 0.3])
            .collect();
        let payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.91
            },
            "body_kpts_2d": [],
            "body_kpts_3d": [],
            "hand_kpts_2d": hand_kpts_2d,
            "hand_kpts_3d": hand_kpts_3d
        });
        store.ingest_capture_pose_json(&payload, 10, 20);

        let drop_payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.0
            },
            "body_kpts_2d": [],
            "body_kpts_3d": [],
            "hand_kpts_2d": [],
            "hand_kpts_3d": []
        });
        store.ingest_capture_pose_json(&drop_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_conf, 0.91);
        assert_eq!(snapshot.vision_conf, 0.91);
    }

    #[test]
    fn short_hand_drop_keeps_recent_hand_source() {
        let store = VisionStore::default();
        let payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.91
            },
            "body_kpts_2d": [],
            "body_kpts_3d": [],
            "hand_kpts_2d": (0..42).map(|index| vec![0.1 + index as f32, 0.2]).collect::<Vec<_>>(),
            "hand_kpts_3d": (0..42).map(|index| vec![0.1 + index as f32 * 0.01, 0.2, 0.3]).collect::<Vec<_>>(),
            "capture_profile": {
                "hand_3d_source": "edge_depth_reprojected"
            }
        });
        store.ingest_capture_pose_json(&payload, 10, 20);

        let drop_payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.0
            },
            "body_kpts_2d": [],
            "body_kpts_3d": [],
            "hand_kpts_2d": [],
            "hand_kpts_3d": [],
            "capture_profile": {
                "hand_3d_source": "none"
            }
        });
        store.ingest_capture_pose_json(&drop_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_3d_source, "edge_depth_reprojected");
    }

    #[test]
    fn explicit_device_pose_passthrough_clears_cached_body_and_hand_points() {
        let store = VisionStore::default();
        let payload = json!({
            "confidence": {
                "body": 0.61,
                "hand": 0.84
            },
            "body_kpts_2d": (0..17).map(|_| vec![0.1, 0.2]).collect::<Vec<_>>(),
            "body_kpts_3d": (0..17).map(|_| vec![0.1, 0.2, 0.3]).collect::<Vec<_>>(),
            "hand_kpts_2d": (0..42).map(|_| vec![0.1, 0.2]).collect::<Vec<_>>(),
            "hand_kpts_3d": (0..42).map(|_| vec![0.1, 0.2, 0.3]).collect::<Vec<_>>()
        });
        store.ingest_capture_pose_json(&payload, 10, 20);

        let passthrough_payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.0
            },
            "body_kpts_2d": [],
            "body_kpts_3d": [],
            "hand_kpts_2d": [],
            "hand_kpts_3d": [],
            "capture_profile": {
                "execution_mode": "device_pose_passthrough",
                "vision_processing_enabled": false,
                "body_3d_source": "none",
                "hand_3d_source": "none"
            }
        });
        store.ingest_capture_pose_json(&passthrough_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert!(snapshot.body_kpts_2d.is_empty());
        assert!(snapshot.body_kpts_3d.is_empty());
        assert!(snapshot.hand_kpts_2d.is_empty());
        assert!(snapshot.hand_kpts_3d.is_empty());
        assert_eq!(snapshot.body_conf, 0.0);
        assert_eq!(snapshot.hand_conf, 0.0);
        assert_eq!(snapshot.vision_conf, 0.0);
        assert_eq!(snapshot.execution_mode, "device_pose_passthrough");
        assert_eq!(snapshot.hand_3d_source, "none");
    }

    #[test]
    fn right_hand_only_frame_is_slotted_to_right_side() {
        let store = VisionStore::default();
        let payload = json!({
            "confidence": {
                "body": 0.73,
                "hand": 0.88
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": hand_points(0.45)
        });

        store.ingest_capture_pose_json(&payload, 10, 20);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_kpts_3d[0], [0.0, 0.0, 0.0]);
        assert_eq!(snapshot.hand_kpts_3d[21], [0.45, 0.05, 1.0]);
    }

    #[test]
    fn single_hand_update_keeps_other_recent_hand_side() {
        let store = VisionStore::default();
        let mut both_hands = hand_points(-0.45);
        both_hands.extend(hand_points(0.45));
        let initial_payload = json!({
            "confidence": {
                "body": 0.71,
                "hand": 0.92
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": both_hands
        });
        store.ingest_capture_pose_json(&initial_payload, 10, 20);

        let update_payload = json!({
            "confidence": {
                "body": 0.70,
                "hand": 0.81
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": hand_points(0.60)
        });
        store.ingest_capture_pose_json(&update_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_kpts_3d[0], [-0.45, 0.05, 1.0]);
        assert_eq!(snapshot.hand_kpts_3d[21], [0.60, 0.05, 1.0]);
        assert_eq!(snapshot.hand_conf, 0.81);
    }

    #[test]
    fn repeated_single_hand_frames_keep_other_side_beyond_full_drop_window() {
        let store = VisionStore::default();
        let mut both_hands = hand_points(-0.45);
        both_hands.extend(hand_points(0.45));
        let initial_payload = json!({
            "confidence": {
                "body": 0.71,
                "hand": 0.92
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": both_hands
        });
        store.ingest_capture_pose_json(&initial_payload, 10, 20);

        thread::sleep(Duration::from_millis(VISION_PART_HOLD_MS + 20));

        let update_payload = json!({
            "confidence": {
                "body": 0.70,
                "hand": 0.81
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": hand_points(0.60)
        });
        store.ingest_capture_pose_json(&update_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_kpts_3d[0], [-0.45, 0.05, 1.0]);
        assert_eq!(snapshot.hand_kpts_3d[21], [0.60, 0.05, 1.0]);
    }

    #[test]
    fn repeated_single_hand_frames_eventually_clear_missing_side() {
        let store = VisionStore::default();
        let mut both_hands = hand_points(-0.45);
        both_hands.extend(hand_points(0.45));
        let initial_payload = json!({
            "confidence": {
                "body": 0.71,
                "hand": 0.92
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": both_hands
        });
        store.ingest_capture_pose_json(&initial_payload, 10, 20);

        thread::sleep(Duration::from_millis(VISION_SINGLE_HAND_SIDE_HOLD_MS + 20));

        let update_payload = json!({
            "confidence": {
                "body": 0.70,
                "hand": 0.81
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": hand_points(0.60)
        });
        store.ingest_capture_pose_json(&update_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_kpts_3d[0], [0.0, 0.0, 0.0]);
        assert_eq!(snapshot.hand_kpts_3d[21], [0.60, 0.05, 1.0]);
    }

    #[test]
    fn sparse_hand_3d_update_backfills_missing_joints_from_recent_cache() {
        let store = VisionStore::default();
        let mut both_hands = hand_points(-0.45);
        both_hands.extend(hand_points(0.45));
        let initial_payload = json!({
            "confidence": {
                "body": 0.71,
                "hand": 0.92
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": both_hands
        });
        store.ingest_capture_pose_json(&initial_payload, 10, 20);

        let mut updated_hands = sparse_hand_points(-0.30, &[0, 5, 9]);
        updated_hands.extend(hand_points(0.60));
        let update_payload = json!({
            "confidence": {
                "body": 0.70,
                "hand": 0.81
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": updated_hands
        });
        store.ingest_capture_pose_json(&update_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_point3_close(snapshot.hand_kpts_3d[0], [-0.30, 0.05, 1.0]);
        assert_point3_close(snapshot.hand_kpts_3d[5], [-0.25, 0.06, 1.0]);
        assert_point3_close(snapshot.hand_kpts_3d[8], [-0.37, 0.066, 1.0]);
        assert_point3_close(snapshot.hand_kpts_3d[17], [-0.28, 0.08400001, 1.0]);
        assert_point3_close(snapshot.hand_kpts_3d[21], [0.60, 0.05, 1.0]);
    }

    #[test]
    fn isolated_hand_point_outlier_is_dropped() {
        let store = VisionStore::default();
        let payload = json!({
            "confidence": {
                "body": 0.73,
                "hand": 0.88
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": outlier_hand_points(0.45, 8, [2.5, 2.5, 4.0])
        });

        store.ingest_capture_pose_json(&payload, 10, 20);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_kpts_3d[29], [0.0, 0.0, 0.0]);
        assert_eq!(snapshot.hand_kpts_3d[21], [0.45, 0.05, 1.0]);
    }

    #[test]
    fn reversed_dual_hand_frame_is_reassigned_using_body_wrist_refs() {
        let store = VisionStore::default();
        let mut reversed_hands = hand_points(0.45);
        reversed_hands.extend(hand_points(-0.45));
        let payload = json!({
            "confidence": {
                "body": 0.73,
                "hand": 0.88
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": reversed_hands
        });

        store.ingest_capture_pose_json(&payload, 10, 20);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_kpts_3d[0], [-0.45, 0.05, 1.0]);
        assert_eq!(snapshot.hand_kpts_3d[21], [0.45, 0.05, 1.0]);
    }

    #[test]
    fn reversed_dual_hand_frame_is_reassigned_using_recent_cache_without_body() {
        let store = VisionStore::default();
        let mut initial_hands = hand_points(-0.45);
        initial_hands.extend(hand_points(0.45));
        let initial_payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.92
            },
            "body_kpts_3d": [],
            "hand_kpts_3d": initial_hands
        });
        store.ingest_capture_pose_json(&initial_payload, 10, 20);

        let mut reversed_hands = hand_points(0.60);
        reversed_hands.extend(hand_points(-0.30));
        let update_payload = json!({
            "confidence": {
                "body": 0.0,
                "hand": 0.81
            },
            "body_kpts_3d": [],
            "hand_kpts_3d": reversed_hands
        });
        store.ingest_capture_pose_json(&update_payload, 30, 40);
        let snapshot = store.snapshot(1000);

        assert_eq!(snapshot.hand_kpts_3d.len(), 42);
        assert_eq!(snapshot.hand_kpts_3d[0], [-0.30, 0.05, 1.0]);
        assert_eq!(snapshot.hand_kpts_3d[21], [0.60, 0.05, 1.0]);
    }

    #[test]
    fn simulated_capture_is_ignored_while_fresh_ios_stream_exists() {
        let store = VisionStore::new(true);
        let real_payload = json!({
            "device_id": "iphone-real-001",
            "platform": "ios",
            "operator_track_id": "primary_operator",
            "confidence": {
                "body": 0.74,
                "hand": 0.88
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": hand_points(-0.45)
        });
        assert!(store.ingest_capture_pose_json(&real_payload, 10, 20));

        let simulated_payload = json!({
            "device_id": "capture-sim-local-001",
            "platform": "simulated_capture",
            "operator_track_id": "sim_primary_operator",
            "confidence": {
                "body": 0.96,
                "hand": 0.96
            },
            "body_kpts_3d": vec![[9.0, 9.0, 9.0]; 17],
            "hand_kpts_3d": vec![[9.0, 9.0, 9.0]; 42]
        });
        assert!(!store.ingest_capture_pose_json(&simulated_payload, 30, 40));

        let snapshot = store.snapshot(1000);
        assert_eq!(snapshot.device_id.as_deref(), Some("iphone-real-001"));
        assert_eq!(snapshot.platform, "ios");
        assert_eq!(
            snapshot.operator_track_id.as_deref(),
            Some("primary_operator")
        );
        assert_eq!(snapshot.hand_kpts_3d[0], [-0.45, 0.05, 1.0]);
    }

    #[test]
    fn simulated_capture_is_rejected_by_default_when_no_real_ios_stream_exists() {
        let store = VisionStore::default();
        let simulated_payload = json!({
            "device_id": "capture-sim-local-001",
            "platform": "simulated_capture",
            "operator_track_id": "sim_primary_operator",
            "confidence": {
                "body": 0.96,
                "hand": 0.96
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": hand_points(0.45)
        });

        assert!(!store.ingest_capture_pose_json(&simulated_payload, 10, 20));

        let snapshot = store.snapshot(1000);
        assert_eq!(snapshot.device_id, None);
        assert_eq!(snapshot.platform, "");
    }

    #[test]
    fn simulated_capture_is_allowed_when_enabled_and_no_real_ios_stream_exists() {
        let store = VisionStore::new(true);
        let simulated_payload = json!({
            "device_id": "capture-sim-local-001",
            "platform": "simulated_capture",
            "operator_track_id": "sim_primary_operator",
            "confidence": {
                "body": 0.96,
                "hand": 0.96
            },
            "body_kpts_3d": body_with_wrist_refs(),
            "hand_kpts_3d": hand_points(0.45)
        });

        assert!(store.ingest_capture_pose_json(&simulated_payload, 10, 20));

        let snapshot = store.snapshot(1000);
        assert_eq!(snapshot.device_id.as_deref(), Some("capture-sim-local-001"));
        assert_eq!(snapshot.platform, "simulated_capture");
        assert_eq!(
            snapshot.operator_track_id.as_deref(),
            Some("sim_primary_operator")
        );
    }

    #[test]
    fn device_pose_and_imu_are_preserved_in_snapshot() {
        let store = VisionStore::default();
        let payload = json!({
            "device_id": "iphone-real-001",
            "platform": "ios",
            "confidence": {
                "body": 0.0,
                "hand": 0.82
            },
            "hand_kpts_3d": hand_points(0.45),
            "device_pose": {
                "position_m": [1.0, 1.5, 2.0],
                "rotation_deg": [0.0, 90.0, 0.0],
                "right_vector": [1.0, 0.0, 0.0],
                "up_vector": [0.0, 1.0, 0.0],
                "forward_vector": [0.0, 0.0, 1.0],
                "source": "ios_device_pose"
            },
            "imu": {
                "accel": [0.1, 0.2, 0.3],
                "gyro": [0.4, 0.5, 0.6]
            }
        });

        assert!(store.ingest_capture_pose_json(&payload, 10, 20));
        let snapshot = store.snapshot(1000);

        assert_eq!(
            snapshot.device_pose.as_ref().map(|pose| pose.position_m),
            Some([1.0, 1.5, 2.0])
        );
        assert_eq!(
            snapshot
                .device_pose
                .as_ref()
                .and_then(|pose| pose.forward_vector),
            Some([0.0, 0.0, 1.0])
        );
        assert_eq!(
            snapshot.imu.as_ref().map(|imu| imu.accel),
            Some([0.1, 0.2, 0.3])
        );
        assert_eq!(
            snapshot.imu.as_ref().map(|imu| imu.gyro),
            Some([0.4, 0.5, 0.6])
        );
    }

    #[test]
    fn device_pose_and_imu_accept_ios_style_aliases() {
        let store = VisionStore::default();
        let payload = json!({
            "device_id": "iphone-real-001",
            "platform": "ios",
            "confidence": {
                "body": 0.0,
                "hand": 0.82
            },
            "hand_kpts_3d": hand_points(0.45),
            "device_pose": {
                "position": [1.0, 1.5, 2.0],
                "targetSpace": "room_world_frame",
                "rotationDeg": [0.0, 90.0, 0.0],
                "basisRight": [1.0, 0.0, 0.0],
                "basisUp": [0.0, 1.0, 0.0],
                "basisForward": [0.0, 0.0, 1.0],
                "poseSource": "ios_device_pose"
            },
            "imu": {
                "userAccelMps2": [0.1, 0.2, 0.3],
                "rotationRateRps": [0.4, 0.5, 0.6]
            }
        });

        assert!(store.ingest_capture_pose_json(&payload, 10, 20));
        let snapshot = store.snapshot(1000);

        assert_eq!(
            snapshot.device_pose.as_ref().map(|pose| pose.position_m),
            Some([1.0, 1.5, 2.0])
        );
        assert_eq!(
            snapshot
                .device_pose
                .as_ref()
                .map(|pose| pose.target_space.as_str()),
            Some("room_world_frame")
        );
        assert_eq!(
            snapshot.imu.as_ref().map(|imu| imu.accel),
            Some([0.1, 0.2, 0.3])
        );
        assert_eq!(
            snapshot.imu.as_ref().map(|imu| imu.gyro),
            Some([0.4, 0.5, 0.6])
        );
    }

    #[test]
    fn device_pose_accepts_transform_columns_payload() {
        let store = VisionStore::default();
        let payload = json!({
            "device_id": "iphone-real-001",
            "platform": "ios",
            "confidence": {
                "body": 0.0,
                "hand": 0.82
            },
            "hand_kpts_3d": hand_points(0.45),
            "device_pose": {
                "worldTransform": {
                    "columns": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.3, 1.2, 2.4, 1.0]
                    ]
                },
                "targetSpace": "stereo_pair_frame",
                "source": "arkit_world_transform"
            }
        });

        assert!(store.ingest_capture_pose_json(&payload, 10, 20));
        let snapshot = store.snapshot(1000);

        assert_eq!(
            snapshot.device_pose.as_ref().map(|pose| pose.position_m),
            Some([0.3, 1.2, 2.4])
        );
        assert_eq!(
            snapshot
                .device_pose
                .as_ref()
                .and_then(|pose| pose.forward_vector),
            Some([0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn stereo_store_should_preserve_multiple_persons_and_primary_track() {
        let store = StereoStore::default();
        let persons = vec![
            StereoTrackedPersonSnapshot {
                operator_track_id: Some("stereo-person-1".to_string()),
                stereo_confidence: 0.92,
                body_kpts_3d: vec![[0.1, 1.0, 2.0]; 17],
                hand_kpts_3d: Vec::new(),
                left_body_kpts_2d: vec![[0.1, 0.2]; 17],
                right_body_kpts_2d: vec![[0.2, 0.2]; 17],
                selection_reason: "geometry_score".to_string(),
                source_tag_left: "full_frame".to_string(),
                source_tag_right: "full_frame".to_string(),
                hand_hint_gap_m: None,
                continuity_gap_m: Some(0.12),
            },
            StereoTrackedPersonSnapshot {
                operator_track_id: Some("stereo-person-2".to_string()),
                stereo_confidence: 0.77,
                body_kpts_3d: vec![[1.1, 1.0, 2.0]; 17],
                hand_kpts_3d: Vec::new(),
                left_body_kpts_2d: vec![[0.5, 0.2]; 17],
                right_body_kpts_2d: vec![[0.6, 0.2]; 17],
                selection_reason: "track_persistence".to_string(),
                source_tag_left: "low_roi_left".to_string(),
                source_tag_right: "low_roi_right".to_string(),
                hand_hint_gap_m: None,
                continuity_gap_m: None,
            },
        ];

        assert!(store.ingest_pose3d(
            Some("stereo-uvc-001".to_string()),
            None,
            persons[0].body_kpts_3d.clone(),
            persons[0].hand_kpts_3d.clone(),
            persons[0].left_body_kpts_2d.clone(),
            persons[0].right_body_kpts_2d.clone(),
            BodyKeypointLayout::CocoBody17,
            HandKeypointLayout::Unknown,
            STEREO_PAIR_FRAME.to_string(),
            STEREO_PAIR_FRAME.to_string(),
            persons.clone(),
            persons[0].operator_track_id.clone(),
            persons[0].stereo_confidence,
            10,
            20,
        ));

        let snapshot = store.snapshot(1000);
        assert_eq!(snapshot.persons.len(), 2);
        assert_eq!(
            snapshot.operator_track_id.as_deref(),
            Some("stereo-person-1")
        );
        assert_eq!(
            snapshot.persons[1].operator_track_id.as_deref(),
            Some("stereo-person-2")
        );
        assert_eq!(snapshot.body_kpts_3d[0], [0.1, 1.0, 2.0]);
        assert_eq!(snapshot.persons[1].body_kpts_3d[0], [1.1, 1.0, 2.0]);
    }
}

/// 双目输入的轻量快照（用于融合与可观测）。
#[derive(Clone, Debug, Default)]
pub struct StereoTrackedPersonSnapshot {
    pub operator_track_id: Option<String>,
    pub stereo_confidence: f32,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    pub left_body_kpts_2d: Vec<[f32; 2]>,
    pub right_body_kpts_2d: Vec<[f32; 2]>,
    pub selection_reason: String,
    pub source_tag_left: String,
    pub source_tag_right: String,
    pub hand_hint_gap_m: Option<f32>,
    pub continuity_gap_m: Option<f32>,
}

/// 双目输入的轻量快照（用于融合与可观测）。
#[derive(Clone, Debug, Default)]
pub struct StereoSnapshot {
    pub device_id: Option<String>,
    pub stereo_confidence: f32,
    pub operator_track_id: Option<String>,
    pub calibration: Option<Value>,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub hand_kpts_3d: Vec<[f32; 3]>,
    pub left_body_kpts_2d: Vec<[f32; 2]>,
    pub right_body_kpts_2d: Vec<[f32; 2]>,
    pub body_layout: BodyKeypointLayout,
    pub hand_layout: HandKeypointLayout,
    pub body_space: String,
    pub hand_space: String,
    pub persons: Vec<StereoTrackedPersonSnapshot>,
    pub last_edge_time_ns: u64,
    pub last_recv_time_ns: u64,
    pub last_at: Option<Instant>,
    pub fresh: bool,
}

#[derive(Default)]
struct StereoInner {
    device_id: Option<String>,
    stereo_confidence: f32,
    operator_track_id: Option<String>,
    calibration: Option<Value>,
    body_kpts_3d: Vec<[f32; 3]>,
    hand_kpts_3d: Vec<[f32; 3]>,
    left_body_kpts_2d: Vec<[f32; 2]>,
    right_body_kpts_2d: Vec<[f32; 2]>,
    body_layout: BodyKeypointLayout,
    hand_layout: HandKeypointLayout,
    body_space: String,
    hand_space: String,
    persons: Vec<StereoTrackedPersonSnapshot>,
    last_edge_time_ns: u64,
    last_recv_time_ns: u64,
    last_at: Option<Instant>,
}

/// 双目输入缓存：由 edge 本地 ingest 更新（HTTP/SDK/ROS2/gRPC 任选其一）。
#[derive(Default)]
pub struct StereoStore {
    inner: RwLock<StereoInner>,
}

fn is_simulated_stereo_device(device_id: &str) -> bool {
    let normalized = device_id.trim().to_ascii_lowercase();
    normalized.starts_with("capture-sim-")
        || normalized.starts_with("sim-")
        || normalized.contains("operator-debug-sim")
}

impl StereoStore {
    pub fn ingest_pose3d(
        &self,
        device_id: Option<String>,
        calibration: Option<Value>,
        body_kpts_3d: Vec<[f32; 3]>,
        hand_kpts_3d: Vec<[f32; 3]>,
        left_body_kpts_2d: Vec<[f32; 2]>,
        right_body_kpts_2d: Vec<[f32; 2]>,
        body_layout: BodyKeypointLayout,
        hand_layout: HandKeypointLayout,
        body_space: String,
        hand_space: String,
        persons: Vec<StereoTrackedPersonSnapshot>,
        operator_track_id: Option<String>,
        stereo_confidence: f32,
        edge_time_ns: u64,
        recv_time_ns: u64,
    ) -> bool {
        let incoming_device_id = device_id
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let incoming_is_sim = incoming_device_id
            .as_deref()
            .is_some_and(is_simulated_stereo_device);
        let mut inner = self.inner.write().expect("stereo lock poisoned");
        let now = Instant::now();
        let current_real_stereo_is_fresh = inner
            .last_at
            .zip(inner.device_id.as_deref())
            .is_some_and(|(last_at, current_device_id)| {
                !is_simulated_stereo_device(current_device_id)
                    && now.duration_since(last_at) <= Duration::from_millis(10_000)
            });
        if incoming_is_sim && current_real_stereo_is_fresh {
            return false;
        }

        let normalized_persons = if persons.is_empty() {
            vec![StereoTrackedPersonSnapshot {
                operator_track_id: operator_track_id.clone(),
                stereo_confidence: stereo_confidence.clamp(0.0, 1.0),
                body_kpts_3d: body_kpts_3d.clone(),
                hand_kpts_3d: hand_kpts_3d.clone(),
                left_body_kpts_2d: left_body_kpts_2d.clone(),
                right_body_kpts_2d: right_body_kpts_2d.clone(),
                selection_reason: String::new(),
                source_tag_left: String::new(),
                source_tag_right: String::new(),
                hand_hint_gap_m: None,
                continuity_gap_m: None,
            }]
        } else {
            persons
        };
        let primary_person = normalized_persons.first().cloned().unwrap_or_default();

        inner.device_id = incoming_device_id;
        inner.stereo_confidence = primary_person.stereo_confidence.clamp(0.0, 1.0);
        inner.operator_track_id = primary_person.operator_track_id.clone();
        inner.calibration = calibration;
        inner.body_kpts_3d = primary_person.body_kpts_3d.clone();
        inner.hand_kpts_3d = primary_person.hand_kpts_3d.clone();
        inner.left_body_kpts_2d = primary_person.left_body_kpts_2d.clone();
        inner.right_body_kpts_2d = primary_person.right_body_kpts_2d.clone();
        inner.body_layout = body_layout;
        inner.hand_layout = hand_layout;
        inner.body_space = body_space;
        inner.hand_space = hand_space;
        inner.persons = normalized_persons;
        inner.last_edge_time_ns = edge_time_ns;
        inner.last_recv_time_ns = recv_time_ns;
        inner.last_at = Some(now);
        true
    }

    pub fn snapshot(&self, stale_ms: u64) -> StereoSnapshot {
        let now = Instant::now();
        let inner = self.inner.read().expect("stereo lock poisoned");
        let fresh = inner
            .last_at
            .is_some_and(|t| now.duration_since(t) <= Duration::from_millis(stale_ms));
        StereoSnapshot {
            device_id: inner.device_id.clone(),
            stereo_confidence: inner.stereo_confidence,
            operator_track_id: inner.operator_track_id.clone(),
            calibration: inner.calibration.clone(),
            body_kpts_3d: inner.body_kpts_3d.clone(),
            hand_kpts_3d: inner.hand_kpts_3d.clone(),
            left_body_kpts_2d: inner.left_body_kpts_2d.clone(),
            right_body_kpts_2d: inner.right_body_kpts_2d.clone(),
            body_layout: inner.body_layout,
            hand_layout: inner.hand_layout,
            body_space: inner.body_space.clone(),
            hand_space: inner.hand_space.clone(),
            persons: inner.persons.clone(),
            last_edge_time_ns: inner.last_edge_time_ns,
            last_recv_time_ns: inner.last_recv_time_ns,
            last_at: inner.last_at,
            fresh,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WifiPoseSnapshot {
    pub body_confidence: f32,
    pub operator_track_id: Option<String>,
    pub body_kpts_3d: Vec<[f32; 3]>,
    pub body_layout: BodyKeypointLayout,
    pub body_space: String,
    pub diagnostics: WifiPoseDiagnostics,
    pub last_edge_time_ns: u64,
    pub last_recv_time_ns: u64,
    pub last_at: Option<Instant>,
    pub fresh: bool,
}

#[derive(Clone, Debug, Default)]
pub struct WifiPoseDiagnostics {
    pub layout_node_count: usize,
    pub layout_score: f32,
    pub zone_score: f32,
    pub zone_summary_reliable: bool,
    pub motion_energy: f32,
    pub doppler_hz: f32,
    pub signal_quality: f32,
    pub vital_signal_quality: Option<f32>,
    pub stream_fps: f32,
    pub lifecycle_state: String,
    pub coherence_gate_decision: String,
    pub target_space: String,
}

#[derive(Default)]
struct WifiPoseInner {
    body_confidence: f32,
    operator_track_id: Option<String>,
    body_kpts_3d: Vec<[f32; 3]>,
    body_layout: BodyKeypointLayout,
    body_space: String,
    diagnostics: WifiPoseDiagnostics,
    last_edge_time_ns: u64,
    last_recv_time_ns: u64,
    last_at: Option<Instant>,
}

#[derive(Default)]
pub struct WifiPoseStore {
    inner: RwLock<WifiPoseInner>,
}

impl WifiPoseStore {
    pub fn ingest_pose3d(
        &self,
        body_kpts_3d: Vec<[f32; 3]>,
        body_layout: BodyKeypointLayout,
        body_space: String,
        operator_track_id: Option<String>,
        body_confidence: f32,
        diagnostics: WifiPoseDiagnostics,
        edge_time_ns: u64,
        recv_time_ns: u64,
    ) {
        let mut inner = self.inner.write().expect("wifi pose lock poisoned");
        inner.body_confidence = body_confidence.clamp(0.0, 1.0);
        inner.operator_track_id = operator_track_id;
        inner.body_kpts_3d = body_kpts_3d;
        inner.body_layout = body_layout;
        inner.body_space = body_space;
        inner.diagnostics = diagnostics;
        inner.last_edge_time_ns = edge_time_ns;
        inner.last_recv_time_ns = recv_time_ns;
        inner.last_at = Some(Instant::now());
    }

    pub fn snapshot(&self, stale_ms: u64) -> WifiPoseSnapshot {
        let now = Instant::now();
        let inner = self.inner.read().expect("wifi pose lock poisoned");
        let fresh = inner
            .last_at
            .is_some_and(|t| now.duration_since(t) <= Duration::from_millis(stale_ms));
        WifiPoseSnapshot {
            body_confidence: inner.body_confidence,
            operator_track_id: inner.operator_track_id.clone(),
            body_kpts_3d: inner.body_kpts_3d.clone(),
            body_layout: inner.body_layout,
            body_space: inner.body_space.clone(),
            diagnostics: inner.diagnostics.clone(),
            last_edge_time_ns: inner.last_edge_time_ns,
            last_recv_time_ns: inner.last_recv_time_ns,
            last_at: inner.last_at,
            fresh,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TrackedWifiPoseResponse {
    tracked_person: Option<TrackedWifiPerson>,
}

#[derive(Debug, Deserialize)]
struct TrackedWifiPerson {
    track_id: Option<String>,
    target_space: Option<String>,
    body_kpts_3d: Option<Vec<[f32; 3]>>,
    body_layout: Option<String>,
    canonical_body_space: Option<String>,
    canonical_body_kpts_3d: Option<Vec<[f32; 3]>>,
    person_confidence: Option<f32>,
    lifecycle_state: Option<String>,
    coherence_gate_decision: Option<String>,
}

pub async fn run_wifi_tracked_pose_poll(state: crate::AppState) {
    if !state.config.wifi_tracked_pose_direct_enabled {
        info!("direct tracked Wi-Fi pose polling disabled");
        return;
    }

    let tracked_url = format!(
        "{}/api/v1/pose/tracked",
        state.config.sensing_proxy_base.trim_end_matches('/')
    );
    let mut ticker = interval(Duration::from_millis(
        state.config.wifi_tracked_pose_poll_ms.max(20),
    ));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    info!(tracked_url = %tracked_url, "direct tracked Wi-Fi pose polling started");
    loop {
        ticker.tick().await;

        let response = match state.http_client.get(&tracked_url).send().await {
            Ok(response) => response,
            Err(error) => {
                warn!(tracked_url = %tracked_url, %error, "failed to fetch tracked Wi-Fi pose");
                continue;
            }
        };

        if !response.status().is_success() {
            warn!(
                tracked_url = %tracked_url,
                status = %response.status(),
                "tracked Wi-Fi pose endpoint returned non-success status"
            );
            continue;
        }

        let payload = match response.json::<TrackedWifiPoseResponse>().await {
            Ok(payload) => payload,
            Err(error) => {
                warn!(tracked_url = %tracked_url, %error, "failed to decode tracked Wi-Fi pose");
                continue;
            }
        };

        let Some(tracked) = payload.tracked_person else {
            continue;
        };

        let body_kpts_3d = tracked
            .canonical_body_kpts_3d
            .or(tracked.body_kpts_3d)
            .unwrap_or_default();
        if body_kpts_3d.is_empty() {
            continue;
        }

        let body_layout =
            BodyKeypointLayout::resolve(tracked.body_layout.as_deref(), &[body_kpts_3d.len()]);
        let tracked_target_space = tracked.target_space.clone();
        let body_space = tracked
            .canonical_body_space
            .or(tracked.target_space)
            .unwrap_or_else(|| "canonical_body_frame".to_string());
        let body_kpts_3d_for_record = body_kpts_3d.clone();
        let diagnostics = WifiPoseDiagnostics {
            lifecycle_state: tracked
                .lifecycle_state
                .unwrap_or_else(|| "unknown".to_string()),
            coherence_gate_decision: tracked
                .coherence_gate_decision
                .unwrap_or_else(|| "Unknown".to_string()),
            target_space: tracked_target_space.unwrap_or_else(|| body_space.clone()),
            ..WifiPoseDiagnostics::default()
        };
        let edge_time_ns = state.gate.edge_time_ns();

        state.wifi_pose.ingest_pose3d(
            body_kpts_3d,
            body_layout,
            body_space.clone(),
            tracked.track_id.clone(),
            tracked.person_confidence.unwrap_or(0.0).clamp(0.0, 1.0),
            diagnostics.clone(),
            edge_time_ns,
            edge_time_ns,
        );

        let session = state.session.snapshot();
        if !session.trip_id.trim().is_empty() && !session.session_id.trim().is_empty() {
            let record = serde_json::json!({
                "type": "wifi_pose_packet",
                "schema_version": "1.0.0",
                "trip_id": session.trip_id,
                "session_id": session.session_id,
                "device_id": "wifi-tracked-direct",
                "operator_track_id": tracked.track_id,
                "source_time_ns": edge_time_ns,
                "recv_time_ns": edge_time_ns,
                "edge_time_ns": edge_time_ns,
                "body_layout": body_layout.as_str(),
                "body_space": body_space,
                "body_kpts_3d": body_kpts_3d_for_record,
                "body_confidence": tracked.person_confidence.unwrap_or(0.0).clamp(0.0, 1.0),
                "source_label": "wifi_densepose_tracked_direct_poll",
                "diagnostics": {
                    "layout_node_count": diagnostics.layout_node_count,
                    "layout_score": diagnostics.layout_score,
                    "zone_score": diagnostics.zone_score,
                    "zone_summary_reliable": diagnostics.zone_summary_reliable,
                    "motion_energy": diagnostics.motion_energy,
                    "doppler_hz": diagnostics.doppler_hz,
                    "signal_quality": diagnostics.signal_quality,
                    "vital_signal_quality": diagnostics.vital_signal_quality,
                    "stream_fps": diagnostics.stream_fps,
                    "lifecycle_state": diagnostics.lifecycle_state,
                    "coherence_gate_decision": diagnostics.coherence_gate_decision,
                    "target_space": diagnostics.target_space,
                },
            });
            state
                .recorder
                .record_wifi_pose3d(
                    &state.protocol,
                    &state.config,
                    &session.trip_id,
                    &session.session_id,
                    &record,
                )
                .await;
        }
    }
}

/// CSI 输入的轻量快照（用于融合与可观测）。
#[derive(Clone, Debug, Default)]
pub struct CsiSnapshot {
    pub node_count: usize,
    pub drop_rate: f32,
    pub snr_mean: f32,
    pub csi_conf: f32,
    pub last_at: Option<Instant>,
    pub fresh: bool,
}

const CSI_RECENT_WINDOW_MS_MIN: u64 = 500;
const CSI_FULL_COVERAGE_NODES: f32 = 4.0;
const CSI_TARGET_PACKET_RATE_HZ_PER_NODE: f32 = 20.0;
const CSI_SEQUENCE_RESET_GUARD: u32 = 1024;

#[derive(Debug, Clone, Copy)]
struct RecentCsiPacket {
    sequence: u32,
    freq_raw: u32,
    snr: f32,
    at: Instant,
}

#[derive(Debug, Clone)]
struct NodeState {
    last_seq: u32,
    recv: u64,
    dropped: u64,
    last_snr: f32,
    last_at: Instant,
    recent: VecDeque<RecentCsiPacket>,
}

#[derive(Default)]
struct CsiInner {
    nodes: HashMap<u8, NodeState>,
    last_any_at: Option<Instant>,
}

/// CSI 输入缓存：由 UDP 5005（ADR-018）更新。
#[derive(Default)]
pub struct CsiStore {
    inner: RwLock<CsiInner>,
}

impl CsiStore {
    pub fn ingest_adr018_packet(&self, data: &[u8]) {
        let Some(h) = parse_adr018_header(data) else {
            return;
        };
        let now = Instant::now();
        let mut inner = self.inner.write().expect("csi lock poisoned");
        inner.last_any_at = Some(now);
        match inner.nodes.get_mut(&h.node_id) {
            Some(s) => {
                s.recv += 1;
                if h.sequence < s.last_seq
                    && s.last_seq.saturating_sub(h.sequence) > CSI_SEQUENCE_RESET_GUARD
                {
                    s.recent.clear();
                } else {
                    let expected = s.last_seq.wrapping_add(1);
                    if h.sequence > expected {
                        let gap = h.sequence - expected;
                        s.dropped += gap as u64;
                    }
                }
                s.last_seq = h.sequence;
                s.last_snr = h.snr;
                s.last_at = now;
                s.recent.push_back(RecentCsiPacket {
                    sequence: h.sequence,
                    freq_raw: h.freq_raw,
                    snr: h.snr,
                    at: now,
                });
            }
            None => {
                inner.nodes.insert(
                    h.node_id,
                    NodeState {
                        last_seq: h.sequence,
                        recv: 1,
                        dropped: 0,
                        last_snr: h.snr,
                        last_at: now,
                        recent: VecDeque::from([RecentCsiPacket {
                            sequence: h.sequence,
                            freq_raw: h.freq_raw,
                            snr: h.snr,
                            at: now,
                        }]),
                    },
                );
            }
        }
    }

    pub fn snapshot(&self, stale_ms: u64) -> CsiSnapshot {
        let now = Instant::now();
        let recent_window = Duration::from_millis(stale_ms.max(CSI_RECENT_WINDOW_MS_MIN));
        let mut inner = self.inner.write().expect("csi lock poisoned");
        let fresh = inner
            .last_any_at
            .is_some_and(|t| now.duration_since(t) <= Duration::from_millis(stale_ms));

        let mut recv: u64 = 0;
        let mut dropped: u64 = 0;
        let mut snr_sum = 0.0_f32;
        let mut rate_quality_sum = 0.0_f32;
        let mut active_nodes = 0usize;

        for s in inner.nodes.values_mut() {
            while s
                .recent
                .front()
                .is_some_and(|pkt| now.duration_since(pkt.at) > recent_window)
            {
                s.recent.pop_front();
            }

            if s.recent.is_empty() {
                continue;
            }

            active_nodes += 1;
            recv += s.recent.len() as u64;

            let mut node_dropped: u64 = 0;
            let mut node_snr_sum = 0.0_f32;
            let mut prev_seq: Option<u32> = None;
            let mut last_freq_raw: Option<u32> = None;
            let mut freq_switches: u64 = 0;
            for pkt in &s.recent {
                node_snr_sum += pkt.snr;
                if let Some(prev) = prev_seq {
                    if pkt.sequence > prev.wrapping_add(1) {
                        node_dropped += (pkt.sequence - prev.wrapping_add(1)) as u64;
                    }
                }
                if let Some(prev_freq) = last_freq_raw {
                    if prev_freq != pkt.freq_raw {
                        freq_switches += 1;
                    }
                }
                prev_seq = Some(pkt.sequence);
                last_freq_raw = Some(pkt.freq_raw);
            }
            dropped += node_dropped;

            let node_packet_rate_hz = s.recent.len() as f32 / recent_window.as_secs_f32();
            rate_quality_sum +=
                (node_packet_rate_hz / CSI_TARGET_PACKET_RATE_HZ_PER_NODE).clamp(0.0, 1.0);

            let node_snr_mean = node_snr_sum / s.recent.len() as f32;
            snr_sum += node_snr_mean;

            if freq_switches > 0 {
                metrics::gauge!("csi_freq_switch_count")
                    .set(freq_switches as f64 / s.recent.len().max(1) as f64);
            }
        }

        let denom = (recv + dropped).max(1) as f32;
        let drop_rate = (dropped as f32 / denom).clamp(0.0, 1.0);

        let node_count = active_nodes;
        let snr_mean = if node_count == 0 {
            0.0
        } else {
            snr_sum / node_count as f32
        };

        // 这里对齐 RuView 的“窗口化观测”语义：
        // - 使用最近窗口内的活跃节点数，而不是进程生命周期累计节点
        // - 使用最近窗口内的包密度，而不是把 CSI callback 的 sequence gap 直接当成运输丢包
        // - SNR 仍作为一个保守质量因子
        let coverage = (node_count as f32 / CSI_FULL_COVERAGE_NODES).clamp(0.0, 1.0);
        let density = if node_count == 0 {
            0.0
        } else {
            rate_quality_sum / node_count as f32
        };
        let snr_norm = ((snr_mean - 10.0) / 30.0).clamp(0.0, 1.0);
        let csi_conf = if fresh {
            (coverage * density * snr_norm).clamp(0.0, 1.0)
        } else {
            0.0
        };

        CsiSnapshot {
            node_count,
            drop_rate,
            snr_mean,
            csi_conf,
            last_at: inner.last_any_at,
            fresh,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CsiStore;
    use std::thread;
    use std::time::Duration;

    fn build_test_packet_v1(node_id: u8, sequence: u32, rssi: i8, noise_floor: i8) -> Vec<u8> {
        let mut buf = vec![0u8; super::ADR018_CSI_HEADER_SIZE_V1 + 8];
        buf[0..4].copy_from_slice(&super::ADR018_CSI_MAGIC_V1.to_le_bytes());
        buf[4] = node_id;
        buf[5] = 1;
        buf[6..8].copy_from_slice(&2u16.to_le_bytes());
        buf[8..12].copy_from_slice(&2437u32.to_le_bytes());
        buf[12..16].copy_from_slice(&sequence.to_le_bytes());
        buf[16] = rssi as u8;
        buf[17] = noise_floor as u8;
        buf
    }

    fn build_test_packet_v2(
        node_id: u8,
        sequence: u32,
        rssi: i8,
        noise_floor: i8,
        source_time_ns: u64,
    ) -> Vec<u8> {
        let mut buf = vec![0u8; super::ADR018_CSI_HEADER_SIZE_V2 + 8];
        buf[0..4].copy_from_slice(&super::ADR018_CSI_MAGIC_V2.to_le_bytes());
        buf[4] = node_id;
        buf[5] = 1;
        buf[6..8].copy_from_slice(&2u16.to_le_bytes());
        buf[8..12].copy_from_slice(&2437u32.to_le_bytes());
        buf[12..16].copy_from_slice(&sequence.to_le_bytes());
        buf[16] = rssi as u8;
        buf[17] = noise_floor as u8;
        buf[20..28].copy_from_slice(&source_time_ns.to_le_bytes());
        buf
    }

    #[test]
    fn csi_snapshot_uses_recent_window_instead_of_lifetime_gap_history() {
        let store = CsiStore::default();

        store.ingest_adr018_packet(&build_test_packet_v1(1, 1, -42, -92));
        store.ingest_adr018_packet(&build_test_packet_v1(1, 10_000, -42, -92));
        let poisoned = store.snapshot(300);
        assert!(
            poisoned.csi_conf < 0.1,
            "sparse recent packets should keep csi_conf low"
        );

        thread::sleep(Duration::from_millis(650));

        for seq in 20_000..20_012 {
            store.ingest_adr018_packet(&build_test_packet_v1(1, seq, -42, -92));
        }
        let recovered = store.snapshot(300);

        assert_eq!(recovered.node_count, 1);
        assert!(
            recovered.drop_rate < 0.1,
            "recent window should discard stale gap history"
        );
        assert!(
            recovered.csi_conf > 0.2,
            "recent dense packets should yield non-trivial csi_conf"
        );
    }

    #[test]
    fn csi_snapshot_saturates_coverage_at_four_recent_nodes() {
        let store = CsiStore::default();

        for node_id in 1..=4 {
            for seq in 0..8 {
                store.ingest_adr018_packet(&build_test_packet_v1(node_id, seq, -45, -92));
            }
        }

        let snap = store.snapshot(300);
        assert_eq!(snap.node_count, 4);
        assert!(
            snap.csi_conf > 0.4,
            "four active nodes should no longer be capped as 4/6 coverage"
        );
    }

    #[test]
    fn csi_header_v2_exposes_source_timestamp() {
        let packet = build_test_packet_v2(3, 9, -48, -90, 1_234_567_890);
        let parsed = super::parse_adr018_header(&packet).expect("v2 packet should parse");
        assert_eq!(parsed.node_id, 3);
        assert_eq!(parsed.sequence, 9);
        assert_eq!(parsed.source_time_ns, Some(1_234_567_890));
        assert_eq!(parsed.header_version, 2);
        assert_eq!(parsed.header_size, super::ADR018_CSI_HEADER_SIZE_V2);
    }
}

struct Adr018Header {
    magic: u32,
    node_id: u8,
    sequence: u32,
    freq_raw: u32,
    rssi: i8,
    noise_floor: i8,
    snr: f32,
    header_size: usize,
    header_version: u8,
    source_time_ns: Option<u64>,
}

fn parse_adr018_header(data: &[u8]) -> Option<Adr018Header> {
    // ADR-018 raw CSI:
    // v1: magic(u32) + node_id(u8) + antennas(u8) + subcarriers(u16) + freq(u32) + seq(u32) + rssi(i8) + noise(i8) + reserved(u16)
    // v2: above fields + source_time_ns(u64) at [20..28]
    if data.len() < ADR018_CSI_HEADER_SIZE_V1 {
        return None;
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let (header_size, header_version, source_time_ns) = match magic {
        ADR018_CSI_MAGIC_V1 => (ADR018_CSI_HEADER_SIZE_V1, 1, None),
        ADR018_CSI_MAGIC_V2 => {
            if data.len() < ADR018_CSI_HEADER_SIZE_V2 {
                return None;
            }
            let source_time_ns = u64::from_le_bytes([
                data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27],
            ]);
            (ADR018_CSI_HEADER_SIZE_V2, 2, Some(source_time_ns))
        }
        _ => return None,
    };
    let node_id = data[4];
    let freq_raw = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let sequence = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let rssi = data[16] as i8;
    let noise_floor = data[17] as i8;
    let snr = (rssi as f32 - noise_floor as f32).max(0.0);
    Some(Adr018Header {
        magic,
        node_id,
        sequence,
        freq_raw,
        rssi,
        noise_floor,
        snr,
        header_size,
        header_version,
        source_time_ns,
    })
}

/// CSI UDP 接入任务（ADR-018）。失败时会重试（用于现场“网卡重启/端口被占用”场景）。
pub async fn run_csi_udp_ingest(state: crate::AppState) {
    let store = state.csi.clone();
    let bind_addr = state.config.csi_udp_bind.clone();
    let mirror_addr = state.config.csi_udp_mirror_addr.clone();
    let stale_ms = state.config.csi_stale_ms;

    #[derive(Debug)]
    struct RecordItem {
        trip_id: String,
        session_id: String,
        edge_time_ns: u64,
        index: serde_json::Value,
        bytes: Vec<u8>,
        packet_meta: CsiPacketMeta,
    }

    // 落盘异步队列：避免阻塞 UDP recv loop
    let (tx, mut rx) = tokio::sync::mpsc::channel::<RecordItem>(2048);
    {
        let recorder = state.recorder.clone();
        let protocol = state.protocol.clone();
        let cfg = state.config.clone();
        tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                recorder
                    .record_csi_index(
                        &protocol,
                        &cfg,
                        &item.trip_id,
                        &item.session_id,
                        &item.index,
                    )
                    .await;
                recorder
                    .record_csi_packet_bytes(
                        &protocol,
                        &cfg,
                        &item.trip_id,
                        &item.session_id,
                        item.edge_time_ns,
                        &item.bytes,
                        item.packet_meta,
                    )
                    .await;
            }
        });
    }

    loop {
        match UdpSocket::bind(&bind_addr).await {
            Ok(sock) => {
                info!(%bind_addr, mirror_addr=?mirror_addr, "CSI UDP ingest started");
                let mut buf = vec![0u8; 4096];
                loop {
                    match sock.recv_from(&mut buf).await {
                        Ok((n, _src)) => {
                            let data = &buf[..n];
                            store.ingest_adr018_packet(data);
                            if let Some(addr) = mirror_addr.as_deref() {
                                if let Err(e) = sock.send_to(data, addr).await {
                                    warn!(error=%e, mirror_addr=%addr, "CSI UDP mirror send failed");
                                }
                            }

                            // 轻量指标：drop_rate/node_count/snr/csi_conf
                            let snap = store.snapshot(stale_ms);
                            metrics::gauge!("csi_node_count").set(snap.node_count as f64);
                            metrics::gauge!("drop_rate_csi").set(snap.drop_rate as f64);
                            metrics::gauge!("csi_snr_mean").set(snap.snr_mean as f64);
                            metrics::gauge!("csi_conf").set(snap.csi_conf as f64);

                            // 会话内落盘（raw/csi/index.jsonl + packets.bin）
                            if let Some(h) = parse_adr018_header(data) {
                                let session = state.session.snapshot();
                                let trip_id = session.trip_id.clone();
                                let session_id = session.session_id.clone();
                                if !trip_id.is_empty() && !session_id.is_empty() {
                                    let recv_time_ns = state.gate.edge_time_ns();
                                    let device_id = format!("csi-node-{:02}", h.node_id);
                                    let (edge_time_ns, clock_domain, time_sync_status) =
                                        if let Some(source_time_ns) = h.source_time_ns {
                                            let (mapped_edge_time_ns, degraded) =
                                                state.gate.map_source_time_to_edge(
                                                    &device_id,
                                                    source_time_ns,
                                                    recv_time_ns,
                                                );
                                            (
                                                mapped_edge_time_ns,
                                                "esp32_boot_ns",
                                                if degraded {
                                                    "degraded_to_recv_time"
                                                } else {
                                                    "mapped_to_edge_time"
                                                },
                                            )
                                        } else {
                                            (recv_time_ns, "recv_time_ns", "recv_time_only")
                                        };
                                    let index = serde_json::json!({
                                        "type": "csi_packet_index",
                                        "schema_version": "1.0.0",
                                        "trip_id": trip_id,
                                        "session_id": session_id,
                                        "device_id": device_id.clone(),
                                        "edge_time_ns": edge_time_ns,
                                        "recv_time_ns": recv_time_ns,
                                        "source_time_ns": h.source_time_ns,
                                        "clock_domain": clock_domain,
                                        "time_sync_status": time_sync_status,
                                        "header_magic": h.magic,
                                        "header_size": h.header_size,
                                        "header_version": h.header_version,
                                        "node_id": h.node_id,
                                        "sequence": h.sequence,
                                        "freq_raw": h.freq_raw,
                                        "rssi": h.rssi,
                                        "noise_floor": h.noise_floor,
                                        "snr": h.snr,
                                        "payload_len": n.saturating_sub(h.header_size),
                                    });

                                    if tx
                                        .try_send(RecordItem {
                                            trip_id: index["trip_id"]
                                                .as_str()
                                                .unwrap_or_default()
                                                .to_string(),
                                            session_id: index["session_id"]
                                                .as_str()
                                                .unwrap_or_default()
                                                .to_string(),
                                            edge_time_ns,
                                            index,
                                            bytes: data.to_vec(),
                                            packet_meta: CsiPacketMeta {
                                                device_id: Some(device_id),
                                                node_id: Some(h.node_id),
                                                sequence: Some(h.sequence),
                                                freq_raw: Some(h.freq_raw),
                                                rssi: Some(h.rssi),
                                                noise_floor: Some(h.noise_floor),
                                                snr: Some(h.snr),
                                                source_time_ns: h.source_time_ns,
                                                header_version: Some(h.header_version),
                                                clock_domain: Some(clock_domain.to_string()),
                                                time_sync_status: Some(
                                                    time_sync_status.to_string(),
                                                ),
                                            },
                                        })
                                        .is_err()
                                    {
                                        metrics::counter!("csi_record_queue_full_drop_count")
                                            .increment(1);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!(error=%e, "CSI UDP recv failed, restart in 1s");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                warn!(error=%e, %bind_addr, "CSI UDP bind failed, retry in 2s");
            }
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
        // 重新开始下一轮 bind
        debug!(%bind_addr, "CSI UDP ingest retrying");
    }
}
