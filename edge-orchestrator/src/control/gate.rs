use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use std::sync::RwLock;

use crate::config::Config;

#[derive(Clone, Debug)]
pub struct GateSnapshot {
    pub state: String,
    pub reason: String,
    pub robot_type_hint: String,
    pub end_effector_type_hint: String,
}

struct GateInner {
    state: ControlState,
    reason: String,
    safety_state: SafetyState,
    safety_reason: String,
    quality_safety_state: SafetyState,
    quality_safety_reason: String,
    motion_safety_state: SafetyState,
    motion_safety_reason: String,
    robot_type_hint: String,
    end_effector_type_hint: String,

    deadman: DeadmanState,
    time_sync: TimeSyncStore,

    started_at: Instant,

    // 融合质量门控阈值（可在线调参）
    limit_conf_threshold: f32,
    freeze_conf_threshold: f32,
}

#[derive(Clone, Debug)]
struct TimeSyncEntry {
    source_kind: String,
    clock_offset_ns: i64,
    last_rtt_ns: u64,
    last_update_at: Instant,
}

#[derive(Clone, Debug, Default)]
struct TimeSyncStore {
    last_any_at: Option<Instant>,
    // rtt_ns 仅用于健康判断（time_sync_ok）；时钟偏移用于 source_time_ns -> edge_time_ns 映射。
    last_any_rtt_ns: Option<u64>,
    by_device: HashMap<String, TimeSyncEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlState {
    Disarmed,
    Armed,
    Fault,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafetyState {
    Normal,
    Limit,
    Freeze,
    Estop,
}

#[derive(Clone, Debug)]
struct DeadmanState {
    enabled: bool,
    timeout_ms: u64,
    last_keepalive_at: Option<Instant>,
    last_keepalive_edge_time_ns: Option<u64>,
    last_keepalive_source_time_ns: Option<u64>,
    last_keepalive_seq: Option<u64>,
    last_keepalive_trip_id: String,
    last_keepalive_session_id: String,
    last_keepalive_device_id: String,
    recent_keepalive_ats: VecDeque<Instant>,
    pressed: bool,
}

impl DeadmanState {
    fn link_ok(&self, now: Instant) -> bool {
        match self.last_keepalive_at {
            Some(t) => now.duration_since(t) <= Duration::from_millis(self.timeout_ms),
            None => false,
        }
    }

    fn trim_recent_keepalive_history(&mut self, now: Instant) {
        const DEADMAN_RECENT_WINDOW_MS: u64 = 5_000;
        let recent_window = Duration::from_millis(DEADMAN_RECENT_WINDOW_MS);
        while self
            .recent_keepalive_ats
            .front()
            .is_some_and(|at| now.duration_since(*at) > recent_window)
        {
            self.recent_keepalive_ats.pop_front();
        }
    }
}

pub struct Gate {
    cfg: Config,
    inner: RwLock<GateInner>,
}

#[derive(Clone, Debug)]
pub struct DeadmanSnapshot {
    pub enabled: bool,
    pub timeout_ms: u64,
    pub link_ok: bool,
    pub pressed: bool,
    pub keepalive: DeadmanKeepaliveSnapshot,
}

#[derive(Clone, Debug, Default)]
pub struct DeadmanKeepaliveSnapshot {
    pub last_age_ms: Option<u64>,
    pub last_edge_time_ns: Option<u64>,
    pub last_source_time_ns: Option<u64>,
    pub last_seq: Option<u64>,
    pub last_trip_id: String,
    pub last_session_id: String,
    pub last_device_id: String,
    pub packets_in_last_5s: u64,
    pub approx_rate_hz_5s: f32,
}

#[derive(Clone, Debug)]
pub struct TimeSyncDeviceSummary {
    pub device_id: String,
    pub source_kind: String,
    pub clock_offset_ns: i64,
    pub last_rtt_ns: u64,
    pub rtt_ok_ms: u64,
    pub age_ms: u64,
    pub fresh: bool,
    pub rtt_ok: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TimeSyncSummarySnapshot {
    pub last_any_age_ms: Option<u64>,
    pub last_any_rtt_ns: Option<u64>,
    pub device_count: usize,
    pub fresh_device_count: usize,
    pub fresh_ok_device_count: usize,
    pub stale_device_count: usize,
    pub devices: Vec<TimeSyncDeviceSummary>,
}

impl Gate {
    pub fn new(cfg: Config) -> Self {
        let now = Instant::now();
        Self {
            cfg: cfg.clone(),
            inner: RwLock::new(GateInner {
                state: ControlState::Disarmed,
                reason: "".to_string(),
                safety_state: SafetyState::Normal,
                safety_reason: "".to_string(),
                quality_safety_state: SafetyState::Normal,
                quality_safety_reason: "".to_string(),
                motion_safety_state: SafetyState::Normal,
                motion_safety_reason: "".to_string(),
                robot_type_hint: cfg.default_robot_type.clone(),
                end_effector_type_hint: cfg.default_end_effector_type.clone(),
                deadman: DeadmanState {
                    enabled: cfg.deadman_enabled_default,
                    timeout_ms: cfg.deadman_timeout_ms,
                    last_keepalive_at: None,
                    last_keepalive_edge_time_ns: None,
                    last_keepalive_source_time_ns: None,
                    last_keepalive_seq: None,
                    last_keepalive_trip_id: String::new(),
                    last_keepalive_session_id: String::new(),
                    last_keepalive_device_id: String::new(),
                    recent_keepalive_ats: VecDeque::new(),
                    pressed: false,
                },
                time_sync: TimeSyncStore::default(),
                started_at: now,

                // PRD 默认：fused_conf < 0.50 => limit；< 0.35 => freeze
                limit_conf_threshold: 0.50,
                freeze_conf_threshold: 0.35,
            }),
        }
    }

    pub fn edge_time_ns(&self) -> u64 {
        // 单调时钟：进程启动以来的纳秒计数（仅用于对齐与排序，不是系统墙钟时间）。
        let inner = self.inner.read().expect("gate lock poisoned");
        inner.started_at.elapsed().as_nanos() as u64
    }

    pub fn snapshot(&self) -> GateSnapshot {
        let inner = self.inner.read().expect("gate lock poisoned");
        GateSnapshot {
            state: match inner.state {
                ControlState::Disarmed => "disarmed",
                ControlState::Armed => "armed",
                ControlState::Fault => "fault",
            }
            .to_string(),
            reason: inner.reason.clone(),
            robot_type_hint: inner.robot_type_hint.clone(),
            end_effector_type_hint: inner.end_effector_type_hint.clone(),
        }
    }

    pub fn deadman_snapshot(&self) -> DeadmanSnapshot {
        let now = Instant::now();
        let inner = self.inner.read().expect("gate lock poisoned");
        let packets_in_last_5s = inner
            .deadman
            .recent_keepalive_ats
            .iter()
            .filter(|at| now.duration_since(**at) <= Duration::from_secs(5))
            .count() as u64;
        DeadmanSnapshot {
            enabled: inner.deadman.enabled,
            timeout_ms: inner.deadman.timeout_ms,
            link_ok: inner.deadman.link_ok(now),
            pressed: inner.deadman.pressed,
            keepalive: DeadmanKeepaliveSnapshot {
                last_age_ms: inner
                    .deadman
                    .last_keepalive_at
                    .map(|at| now.duration_since(at).as_millis() as u64),
                last_edge_time_ns: inner.deadman.last_keepalive_edge_time_ns,
                last_source_time_ns: inner.deadman.last_keepalive_source_time_ns,
                last_seq: inner.deadman.last_keepalive_seq,
                last_trip_id: inner.deadman.last_keepalive_trip_id.clone(),
                last_session_id: inner.deadman.last_keepalive_session_id.clone(),
                last_device_id: inner.deadman.last_keepalive_device_id.clone(),
                packets_in_last_5s,
                approx_rate_hz_5s: packets_in_last_5s as f32 / 5.0,
            },
        }
    }

    pub fn ingest_keepalive(
        &self,
        trip_id: &str,
        session_id: &str,
        device_id: &str,
        source_time_ns: u64,
        seq: u64,
        pressed: bool,
    ) {
        let now = Instant::now();
        let mut inner = self.inner.write().expect("gate lock poisoned");
        inner.deadman.last_keepalive_at = Some(now);
        inner.deadman.last_keepalive_edge_time_ns =
            Some(inner.started_at.elapsed().as_nanos() as u64);
        inner.deadman.last_keepalive_source_time_ns = Some(source_time_ns);
        inner.deadman.last_keepalive_seq = Some(seq);
        inner.deadman.last_keepalive_trip_id = trip_id.trim().to_string();
        inner.deadman.last_keepalive_session_id = session_id.trim().to_string();
        inner.deadman.last_keepalive_device_id = device_id.trim().to_string();
        inner.deadman.recent_keepalive_ats.push_back(now);
        inner.deadman.trim_recent_keepalive_history(now);
        inner.deadman.pressed = pressed;
        metrics::counter!("deadman_keepalive_count").increment(1);
    }

    pub fn arm(&self) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        inner.state = ControlState::Armed;
        inner.reason.clear();
    }

    pub fn set_robot_hints(&self, robot_type: &str, end_effector_type: &str) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        inner.robot_type_hint = robot_type.to_string();
        inner.end_effector_type_hint = end_effector_type.to_string();
    }

    pub fn disarm(&self, reason: &str) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        inner.state = ControlState::Disarmed;
        inner.reason = reason.to_string();
    }

    pub fn fault(&self, reason: &str) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        inner.state = ControlState::Fault;
        inner.reason = reason.to_string();
    }

    pub fn set_estop(&self, reason: &str) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        inner.safety_state = SafetyState::Estop;
        inner.safety_reason = reason.to_string();
        inner.quality_safety_state = SafetyState::Estop;
        inner.quality_safety_reason = reason.to_string();
        inner.motion_safety_state = SafetyState::Estop;
        inner.motion_safety_reason = reason.to_string();
        inner.state = ControlState::Fault;
        inner.reason = reason.to_string();
    }

    pub fn release_estop(&self) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        inner.safety_state = SafetyState::Normal;
        inner.safety_reason.clear();
        inner.quality_safety_state = SafetyState::Normal;
        inner.quality_safety_reason.clear();
        inner.motion_safety_state = SafetyState::Normal;
        inner.motion_safety_reason.clear();
        // 解除急停后保持“未使能”状态，避免“解除急停即动作”。
        inner.state = ControlState::Disarmed;
        inner.reason = "release".to_string();
    }

    pub fn safety_state(&self) -> SafetyState {
        let inner = self.inner.read().expect("gate lock poisoned");
        inner.safety_state
    }

    pub fn safety_reason(&self) -> String {
        let inner = self.inner.read().expect("gate lock poisoned");
        inner.safety_reason.clone()
    }

    pub fn quality_thresholds(&self) -> (f32, f32) {
        let inner = self.inner.read().expect("gate lock poisoned");
        (inner.limit_conf_threshold, inner.freeze_conf_threshold)
    }

    pub fn set_quality_thresholds(&self, limit: Option<f32>, freeze: Option<f32>) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        if let Some(v) = limit {
            if v.is_finite() && (0.0..=1.0).contains(&v) {
                inner.limit_conf_threshold = v;
            }
        }
        if let Some(v) = freeze {
            if v.is_finite() && (0.0..=1.0).contains(&v) {
                inner.freeze_conf_threshold = v;
            }
        }
    }

    /// 由融合质量驱动的 safety_state（不影响 control_state）。
    pub fn update_safety_from_fused_conf(&self, fused_conf: f32) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        if inner.safety_state == SafetyState::Estop {
            return;
        }
        let fc = fused_conf.clamp(0.0, 1.0);
        let desired = if fc < inner.freeze_conf_threshold {
            SafetyState::Freeze
        } else if fc < inner.limit_conf_threshold {
            SafetyState::Limit
        } else {
            SafetyState::Normal
        };
        inner.quality_safety_state = desired;
        inner.quality_safety_reason = if desired == SafetyState::Normal {
            "".to_string()
        } else {
            "fused_conf".to_string()
        };
        recompute_safety(&mut inner);
    }

    /// 由“速度/加速度/jerk”驱动的 safety_state（与融合质量并行；最终取更严重者）。
    pub fn update_safety_from_motion(&self, desired: SafetyState, reason: &str) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        if inner.safety_state == SafetyState::Estop {
            return;
        }
        inner.motion_safety_state = desired;
        inner.motion_safety_reason = if desired == SafetyState::Normal {
            "".to_string()
        } else {
            reason.to_string()
        };
        recompute_safety(&mut inner);
    }

    pub fn should_emit_motion(&self) -> bool {
        let now = Instant::now();
        let inner = self.inner.read().expect("gate lock poisoned");
        if inner.safety_state == SafetyState::Estop {
            return false;
        }
        if inner.state != ControlState::Armed {
            return false;
        }
        if inner.deadman.enabled {
            return inner.deadman.link_ok(now) && inner.deadman.pressed;
        }
        true
    }

    pub fn record_time_sync(
        &self,
        device_id: String,
        source_kind: String,
        clock_offset_ns: i64,
        rtt_ns: u64,
        _sample_count: u32,
    ) {
        let mut inner = self.inner.write().expect("gate lock poisoned");
        let now = Instant::now();
        inner.time_sync.last_any_at = Some(now);
        inner.time_sync.last_any_rtt_ns = Some(rtt_ns);
        if !device_id.trim().is_empty() {
            inner.time_sync.by_device.insert(
                device_id,
                TimeSyncEntry {
                    source_kind,
                    clock_offset_ns,
                    last_rtt_ns: rtt_ns,
                    last_update_at: now,
                },
            );
        }
    }

    pub fn time_sync_ok(&self, ok_window_ms: u64, rtt_ok_ms: u64) -> bool {
        let inner = self.inner.read().expect("gate lock poisoned");
        let now = Instant::now();
        let fresh_cutoff = Duration::from_millis(ok_window_ms);
        inner.time_sync.by_device.values().any(|entry| {
            let rtt_budget_ms = per_source_rtt_ok_ms(&entry.source_kind, rtt_ok_ms);
            now.duration_since(entry.last_update_at) <= fresh_cutoff
                && entry.last_rtt_ns <= rtt_budget_ms * 1_000_000
        })
    }

    pub fn time_sync_summary(&self) -> TimeSyncSummarySnapshot {
        let now = Instant::now();
        let inner = self.inner.read().expect("gate lock poisoned");
        let fresh_cutoff = Duration::from_millis(self.cfg.time_sync_ok_window_ms);
        let mut devices = inner
            .time_sync
            .by_device
            .iter()
            .map(|(device_id, entry)| {
                let rtt_ok_ms =
                    per_source_rtt_ok_ms(&entry.source_kind, self.cfg.time_sync_rtt_ok_ms);
                TimeSyncDeviceSummary {
                    device_id: device_id.clone(),
                    source_kind: entry.source_kind.clone(),
                    clock_offset_ns: entry.clock_offset_ns,
                    last_rtt_ns: entry.last_rtt_ns,
                    rtt_ok_ms,
                    age_ms: now.duration_since(entry.last_update_at).as_millis() as u64,
                    fresh: now.duration_since(entry.last_update_at) <= fresh_cutoff,
                    rtt_ok: entry.last_rtt_ns <= rtt_ok_ms * 1_000_000,
                }
            })
            .collect::<Vec<_>>();
        devices.sort_by(|left, right| left.device_id.cmp(&right.device_id));
        let fresh_device_count = devices.iter().filter(|device| device.fresh).count();
        let fresh_ok_device_count = devices
            .iter()
            .filter(|device| device.fresh && device.rtt_ok)
            .count();
        TimeSyncSummarySnapshot {
            last_any_age_ms: inner
                .time_sync
                .last_any_at
                .map(|at| now.duration_since(at).as_millis() as u64),
            last_any_rtt_ns: inner.time_sync.last_any_rtt_ns,
            device_count: devices.len(),
            fresh_device_count,
            fresh_ok_device_count,
            stale_device_count: devices.len().saturating_sub(fresh_device_count),
            devices,
        }
    }

    /// 将 `source_time_ns` 映射到 `edge_time_ns`（PRD 3.3）。
    ///
    /// 返回：(edge_time_ns, degraded)
    ///
    /// degraded=true 表示：
    /// - 当前没有可用的时钟偏移；或
    /// - 估计值与 recv_time_ns 差异过大，已降级使用 recv_time_ns。
    pub fn map_source_time_to_edge(
        &self,
        device_id: &str,
        source_time_ns: u64,
        recv_time_ns: u64,
    ) -> (u64, bool) {
        const MAX_ABS_DIFF_NS: i128 = 100_000_000; // 100ms（PRD 3.3.3）

        let inner = self.inner.read().expect("gate lock poisoned");
        let Some(entry) = inner.time_sync.by_device.get(device_id) else {
            metrics::counter!("time_sync_degraded_count", "reason" => "missing").increment(1);
            return (recv_time_ns, true);
        };

        // stale：按 time_sync_ok_window_ms 做同口径约束
        if Instant::now().duration_since(entry.last_update_at)
            > Duration::from_millis(self.cfg.time_sync_ok_window_ms)
        {
            metrics::counter!("time_sync_degraded_count", "reason" => "stale").increment(1);
            return (recv_time_ns, true);
        }

        let est = source_time_ns as i128 + entry.clock_offset_ns as i128;
        if est < 0 || est > u64::MAX as i128 {
            metrics::counter!("time_sync_degraded_count", "reason" => "overflow").increment(1);
            return (recv_time_ns, true);
        }
        let est_u64 = est as u64;
        let diff = (est_u64 as i128 - recv_time_ns as i128).abs();
        if diff > MAX_ABS_DIFF_NS {
            metrics::counter!("time_sync_degraded_count", "reason" => "abs_diff").increment(1);
            return (recv_time_ns, true);
        }
        (est_u64, false)
    }

    /// 将历史媒体分片里的 `source_time_ns` 估算到 `edge_time_ns`。
    ///
    /// 与 `map_source_time_to_edge` 的区别：
    /// - 不再拿 `recv_time_ns` 做 100ms 以内的新鲜度约束；
    /// - 仍然要求 `time_sync` 记录存在且未过期；
    /// - 仅用于 chunk 内逐帧时间戳索引，不应用于实时控制包。
    pub fn estimate_source_time_to_edge_historical(
        &self,
        device_id: &str,
        source_time_ns: u64,
    ) -> Option<u64> {
        let inner = self.inner.read().expect("gate lock poisoned");
        let entry = inner.time_sync.by_device.get(device_id)?;
        if Instant::now().duration_since(entry.last_update_at)
            > Duration::from_millis(self.cfg.time_sync_ok_window_ms)
        {
            return None;
        }
        let est = source_time_ns as i128 + entry.clock_offset_ns as i128;
        if !(0..=u64::MAX as i128).contains(&est) {
            return None;
        }
        Some(est as u64)
    }

    pub fn extrinsic_ok(&self) -> bool {
        !self.cfg.extrinsic_version.is_empty()
    }
}

fn per_source_rtt_ok_ms(source_kind: &str, default_rtt_ok_ms: u64) -> u64 {
    match source_kind {
        "stereo_pair" | "stereo_camera" => default_rtt_ok_ms,
        "wifi_csi_node" => default_rtt_ok_ms.max(50),
        "iphone_capture" | "device_pose" | "phone_capture" => default_rtt_ok_ms.max(30),
        _ => default_rtt_ok_ms,
    }
}

fn severity(s: SafetyState) -> u8 {
    match s {
        SafetyState::Normal => 0,
        SafetyState::Limit => 1,
        SafetyState::Freeze => 2,
        SafetyState::Estop => 3,
    }
}

fn recompute_safety(inner: &mut GateInner) {
    if inner.safety_state == SafetyState::Estop {
        return;
    }

    let mut desired = inner.quality_safety_state;
    let mut reason = inner.quality_safety_reason.clone();

    if severity(inner.motion_safety_state) > severity(desired) {
        desired = inner.motion_safety_state;
        reason = inner.motion_safety_reason.clone();
    } else if severity(inner.motion_safety_state) == severity(desired)
        && desired != SafetyState::Normal
        && reason.trim().is_empty()
    {
        reason = inner.motion_safety_reason.clone();
    }

    if desired == SafetyState::Normal {
        reason.clear();
    }

    if desired != inner.safety_state {
        inner.safety_state = desired;
        inner.safety_reason = reason;
        match desired {
            SafetyState::Normal => metrics::counter!("safety_normal_count").increment(1),
            SafetyState::Limit => metrics::counter!("safety_limit_count").increment(1),
            SafetyState::Freeze => metrics::counter!("freeze_count").increment(1),
            SafetyState::Estop => metrics::counter!("estop_count").increment(1),
        }
    } else {
        inner.safety_reason = reason;
    }
}

#[derive(Default)]
pub struct BridgeStore {
    inner: RwLock<BridgeStoreInner>,
}

#[derive(Default)]
struct BridgeStoreInner {
    unitree: Option<BridgeSnapshotInner>,
    leap: Option<BridgeSnapshotInner>,
    last_any_at: Option<Instant>,
}

#[derive(Clone)]
struct BridgeSnapshotInner {
    is_ready: bool,
    last_update_at: Instant,
    last_command_edge_time_ns: u64,
    fault_code: String,
    fault_message: String,
}

pub struct BridgeSnapshot {
    pub unitree_ready: bool,
    pub leap_ready: bool,
    pub lan_control_ok: bool,
    pub unitree_last_command_edge_time_ns: u64,
    pub leap_last_command_edge_time_ns: u64,
    pub unitree_fault_code: String,
    pub unitree_fault_message: String,
    pub leap_fault_code: String,
    pub leap_fault_message: String,
}

impl BridgeStore {
    pub fn update_bridge(
        &self,
        bridge_id: &str,
        is_ready: bool,
        fault_code: &str,
        fault_message: &str,
        last_command_edge_time_ns: u64,
    ) {
        let now = Instant::now();
        let mut inner = self.inner.write().expect("bridge store lock poisoned");
        inner.last_any_at = Some(now);
        let snap = BridgeSnapshotInner {
            is_ready,
            last_update_at: now,
            last_command_edge_time_ns,
            fault_code: fault_code.to_string(),
            fault_message: fault_message.to_string(),
        };
        if bridge_id.contains("leap") {
            inner.leap = Some(snap);
        } else {
            inner.unitree = Some(snap);
        }
    }

    pub fn snapshot(&self, stale_ms: u64) -> BridgeSnapshot {
        let now = Instant::now();
        let inner = self.inner.read().expect("bridge store lock poisoned");
        let stale = Duration::from_millis(stale_ms);

        let (unitree_ready, unitree_last_cmd, unitree_fault_code, unitree_fault_message) = inner
            .unitree
            .as_ref()
            .map(|b| {
                let fresh = now.duration_since(b.last_update_at) <= stale;
                (
                    b.is_ready && fresh,
                    if fresh {
                        b.last_command_edge_time_ns
                    } else {
                        0
                    },
                    b.fault_code.clone(),
                    b.fault_message.clone(),
                )
            })
            .unwrap_or((false, 0, "".to_string(), "".to_string()));
        let (leap_ready, leap_last_cmd, leap_fault_code, leap_fault_message) = inner
            .leap
            .as_ref()
            .map(|b| {
                let fresh = now.duration_since(b.last_update_at) <= stale;
                (
                    b.is_ready && fresh,
                    if fresh {
                        b.last_command_edge_time_ns
                    } else {
                        0
                    },
                    b.fault_code.clone(),
                    b.fault_message.clone(),
                )
            })
            .unwrap_or((false, 0, "".to_string(), "".to_string()));
        let lan_control_ok = inner
            .last_any_at
            .map(|t| now.duration_since(t) <= stale)
            .unwrap_or(false);

        BridgeSnapshot {
            unitree_ready,
            leap_ready,
            lan_control_ok,
            unitree_last_command_edge_time_ns: unitree_last_cmd,
            leap_last_command_edge_time_ns: leap_last_cmd,
            unitree_fault_code,
            unitree_fault_message,
            leap_fault_code,
            leap_fault_message,
        }
    }
}

pub struct SessionStore {
    default_snapshot: SessionSnapshot,
    inner: RwLock<SessionSnapshot>,
}

impl Default for SessionStore {
    fn default() -> Self {
        let default_snapshot = SessionSnapshot::default();
        Self {
            default_snapshot: default_snapshot.clone(),
            inner: RwLock::new(default_snapshot),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionSnapshot {
    pub trip_id: String,
    pub session_id: String,
    pub mode: String,
    pub runtime_profile: String,
    pub upload_policy_mode: String,
    pub raw_residency: String,
    pub preview_residency: String,
    pub feature_flags: crate::config::RuntimeFeatureFlags,
    pub crowd_upload_enabled: bool,
    pub teleop_enabled: bool,
    pub body_control_enabled: bool,
    pub hand_control_enabled: bool,
}

impl Default for SessionSnapshot {
    fn default() -> Self {
        Self {
            trip_id: String::new(),
            session_id: String::new(),
            mode: String::new(),
            runtime_profile: String::new(),
            upload_policy_mode: String::new(),
            raw_residency: String::new(),
            preview_residency: String::new(),
            feature_flags: crate::config::RuntimeFeatureFlags::default(),
            crowd_upload_enabled: false,
            teleop_enabled: false,
            body_control_enabled: true,
            hand_control_enabled: true,
        }
    }
}

impl SessionStore {
    pub fn new(cfg: &Config) -> Self {
        let teleop_enabled = cfg.control_enabled
            && matches!(
                cfg.runtime_profile,
                crate::config::EdgeRuntimeProfile::TeleopFullstack
            );
        let default_snapshot = SessionSnapshot {
            trip_id: String::new(),
            session_id: String::new(),
            mode: cfg.default_session_mode().to_string(),
            runtime_profile: cfg.runtime_profile_name().to_string(),
            upload_policy_mode: cfg.upload_policy_mode_name().to_string(),
            raw_residency: cfg.raw_residency_default().to_string(),
            preview_residency: cfg.preview_residency_default().to_string(),
            feature_flags: cfg.runtime_feature_flags(),
            crowd_upload_enabled: cfg.crowd_upload_enabled,
            teleop_enabled,
            body_control_enabled: teleop_enabled,
            hand_control_enabled: teleop_enabled,
        };
        Self {
            default_snapshot: default_snapshot.clone(),
            inner: RwLock::new(default_snapshot),
        }
    }

    pub fn set_active(&self, trip_id: String, session_id: String) {
        let mut inner = self.inner.write().expect("session store lock poisoned");
        if !inner.trip_id.trim().is_empty()
            && !inner.session_id.trim().is_empty()
            && !is_local_debug_session(&inner.trip_id, &inner.session_id)
            && is_local_debug_session(&trip_id, &session_id)
        {
            return;
        }
        inner.trip_id = trip_id;
        inner.session_id = session_id;
    }

    pub fn set_mode(&self, mode: String) {
        let mut inner = self.inner.write().expect("session store lock poisoned");
        inner.mode = mode;
    }

    pub fn set_control_profile(
        &self,
        teleop_enabled: bool,
        body_control_enabled: bool,
        hand_control_enabled: bool,
    ) {
        let mut inner = self.inner.write().expect("session store lock poisoned");
        inner.teleop_enabled = teleop_enabled;
        inner.body_control_enabled = body_control_enabled;
        inner.hand_control_enabled = hand_control_enabled;
    }

    pub fn clear_if_match(&self, trip_id: &str, session_id: &str) {
        let mut inner = self.inner.write().expect("session store lock poisoned");
        if inner.trip_id == trip_id && inner.session_id == session_id {
            *inner = self.default_snapshot.clone();
        }
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        self.inner
            .read()
            .expect("session store lock poisoned")
            .clone()
    }
}

pub struct PhoneCaptureCommandStore {
    inner: RwLock<PhoneCaptureCommandSnapshot>,
}

impl Default for PhoneCaptureCommandStore {
    fn default() -> Self {
        Self {
            inner: RwLock::new(PhoneCaptureCommandSnapshot::default()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PhoneCaptureCommandSnapshot {
    pub aux_snapshot_request_seq: u64,
}

impl PhoneCaptureCommandStore {
    pub fn request_aux_snapshot(&self) -> u64 {
        let mut inner = self
            .inner
            .write()
            .expect("phone capture command store lock poisoned");
        inner.aux_snapshot_request_seq = inner.aux_snapshot_request_seq.saturating_add(1);
        inner.aux_snapshot_request_seq
    }

    pub fn snapshot(&self) -> PhoneCaptureCommandSnapshot {
        self.inner
            .read()
            .expect("phone capture command store lock poisoned")
            .clone()
    }
}

pub struct PhoneIngressStatusStore {
    inner: RwLock<PhoneIngressStatusSnapshot>,
}

impl Default for PhoneIngressStatusStore {
    fn default() -> Self {
        Self {
            inner: RwLock::new(PhoneIngressStatusSnapshot::default()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhoneIngressStatusSnapshot {
    pub status: String,
    pub error_code: String,
    pub message: String,
    pub in_flight: bool,
    pub last_attempt_edge_time_ns: Option<u64>,
    pub last_success_edge_time_ns: Option<u64>,
    pub last_frame_id: Option<u64>,
    pub request_trip_id: String,
    pub request_session_id: String,
    pub effective_trip_id: String,
    pub effective_session_id: String,
    pub device_id: String,
    pub operator_track_id: String,
    pub camera_mode: String,
    pub camera_has_depth: Option<bool>,
    pub accepted: Option<bool>,
}

impl Default for PhoneIngressStatusSnapshot {
    fn default() -> Self {
        Self {
            status: "never_seen".to_string(),
            error_code: String::new(),
            message: String::new(),
            in_flight: false,
            last_attempt_edge_time_ns: None,
            last_success_edge_time_ns: None,
            last_frame_id: None,
            request_trip_id: String::new(),
            request_session_id: String::new(),
            effective_trip_id: String::new(),
            effective_session_id: String::new(),
            device_id: String::new(),
            operator_track_id: String::new(),
            camera_mode: String::new(),
            camera_has_depth: None,
            accepted: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PhoneIngressStatusUpdate {
    pub edge_time_ns: u64,
    pub frame_id: Option<u64>,
    pub request_trip_id: String,
    pub request_session_id: String,
    pub effective_trip_id: String,
    pub effective_session_id: String,
    pub device_id: String,
    pub operator_track_id: String,
    pub camera_mode: String,
    pub camera_has_depth: Option<bool>,
}

impl PhoneIngressStatusStore {
    const ACCEPTED_STATUS_GRACE_NS: u64 = 1_000_000_000;

    fn apply_common(inner: &mut PhoneIngressStatusSnapshot, update: &PhoneIngressStatusUpdate) {
        inner.last_attempt_edge_time_ns = Some(update.edge_time_ns);
        inner.last_frame_id = update.frame_id;
        inner.request_trip_id = update.request_trip_id.clone();
        inner.request_session_id = update.request_session_id.clone();
        inner.effective_trip_id = update.effective_trip_id.clone();
        inner.effective_session_id = update.effective_session_id.clone();
        inner.device_id = update.device_id.clone();
        inner.operator_track_id = update.operator_track_id.clone();
        inner.camera_mode = update.camera_mode.clone();
        inner.camera_has_depth = update.camera_has_depth;
    }

    pub fn record_attempt(&self, update: PhoneIngressStatusUpdate) {
        let mut inner = self
            .inner
            .write()
            .expect("phone ingress status store lock poisoned");
        Self::apply_common(&mut inner, &update);
        inner.in_flight = true;
        let preserve_recent_accept = inner.accepted == Some(true)
            && inner
                .last_success_edge_time_ns
                .map(|last_success| {
                    update.edge_time_ns.saturating_sub(last_success)
                        <= Self::ACCEPTED_STATUS_GRACE_NS
                })
                .unwrap_or(false);
        if preserve_recent_accept {
            inner.status = "accepted".to_string();
            inner.error_code.clear();
            if inner.message.trim().is_empty() {
                inner.message = "capture_pose_packet accepted for recorder".to_string();
            }
            inner.accepted = Some(true);
        } else {
            inner.status = "processing".to_string();
            inner.error_code.clear();
            inner.message = "phone ingress request received".to_string();
            inner.accepted = None;
        }
    }

    pub fn record_accepted(&self, update: PhoneIngressStatusUpdate, message: &str) {
        let mut inner = self
            .inner
            .write()
            .expect("phone ingress status store lock poisoned");
        Self::apply_common(&mut inner, &update);
        inner.status = "accepted".to_string();
        inner.error_code.clear();
        inner.message = message.to_string();
        inner.in_flight = false;
        inner.last_success_edge_time_ns = Some(update.edge_time_ns);
        inner.accepted = Some(true);
    }

    pub fn record_rejected(
        &self,
        update: PhoneIngressStatusUpdate,
        error_code: &str,
        message: &str,
    ) {
        let mut inner = self
            .inner
            .write()
            .expect("phone ingress status store lock poisoned");
        Self::apply_common(&mut inner, &update);
        inner.status = "rejected".to_string();
        inner.error_code = error_code.to_string();
        inner.message = message.to_string();
        inner.in_flight = false;
        inner.accepted = Some(false);
    }

    pub fn record_error(
        &self,
        update: PhoneIngressStatusUpdate,
        error_code: &str,
        message: String,
    ) {
        let mut inner = self
            .inner
            .write()
            .expect("phone ingress status store lock poisoned");
        Self::apply_common(&mut inner, &update);
        inner.status = "error".to_string();
        inner.error_code = error_code.to_string();
        inner.message = message;
        inner.in_flight = false;
        inner.accepted = None;
    }

    pub fn snapshot(&self) -> PhoneIngressStatusSnapshot {
        self.inner
            .read()
            .expect("phone ingress status store lock poisoned")
            .clone()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClientOriginUpdate {
    pub edge_time_ns: u64,
    pub client_addr: String,
    pub forwarded_for: String,
    pub user_agent: String,
}

pub struct AssociationHintClientStore {
    inner: RwLock<AssociationHintClientSnapshot>,
}

impl Default for AssociationHintClientStore {
    fn default() -> Self {
        Self {
            inner: RwLock::new(AssociationHintClientSnapshot::default()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AssociationHintClientSnapshot {
    pub request_count: u64,
    pub last_request_edge_time_ns: Option<u64>,
    pub client_addr: String,
    pub forwarded_for: String,
    pub user_agent: String,
}

impl Default for AssociationHintClientSnapshot {
    fn default() -> Self {
        Self {
            request_count: 0,
            last_request_edge_time_ns: None,
            client_addr: String::new(),
            forwarded_for: String::new(),
            user_agent: String::new(),
        }
    }
}

impl AssociationHintClientStore {
    pub fn record_request(&self, update: ClientOriginUpdate) {
        let mut inner = self
            .inner
            .write()
            .expect("association hint client store lock poisoned");
        inner.request_count = inner.request_count.saturating_add(1);
        inner.last_request_edge_time_ns = Some(update.edge_time_ns);
        inner.client_addr = update.client_addr;
        inner.forwarded_for = update.forwarded_for;
        inner.user_agent = update.user_agent;
    }

    pub fn snapshot(&self) -> AssociationHintClientSnapshot {
        self.inner
            .read()
            .expect("association hint client store lock poisoned")
            .clone()
    }
}

#[derive(Clone, Debug, Default)]
pub struct FusionClientConnectionUpdate {
    pub edge_time_ns: u64,
    pub client_addr: String,
    pub forwarded_for: String,
    pub user_agent: String,
    pub transport: String,
    pub format: String,
    pub compression: String,
    pub operator_debug: bool,
}

#[derive(Clone, Debug)]
pub struct FusionClientConnectionSnapshot {
    pub connection_id: u64,
    pub connected_edge_time_ns: u64,
    pub client_addr: String,
    pub forwarded_for: String,
    pub user_agent: String,
    pub transport: String,
    pub format: String,
    pub compression: String,
    pub operator_debug: bool,
}

impl Default for FusionClientConnectionSnapshot {
    fn default() -> Self {
        Self {
            connection_id: 0,
            connected_edge_time_ns: 0,
            client_addr: String::new(),
            forwarded_for: String::new(),
            user_agent: String::new(),
            transport: String::new(),
            format: String::new(),
            compression: String::new(),
            operator_debug: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FusionClientSnapshot {
    pub active_count: usize,
    pub total_connections: u64,
    pub last_connect_edge_time_ns: Option<u64>,
    pub last_disconnect_edge_time_ns: Option<u64>,
    pub last_client_addr: String,
    pub last_forwarded_for: String,
    pub last_user_agent: String,
    pub last_transport: String,
    pub last_format: String,
    pub last_compression: String,
    pub last_operator_debug: Option<bool>,
    pub active_clients: Vec<FusionClientConnectionSnapshot>,
}

impl Default for FusionClientSnapshot {
    fn default() -> Self {
        Self {
            active_count: 0,
            total_connections: 0,
            last_connect_edge_time_ns: None,
            last_disconnect_edge_time_ns: None,
            last_client_addr: String::new(),
            last_forwarded_for: String::new(),
            last_user_agent: String::new(),
            last_transport: String::new(),
            last_format: String::new(),
            last_compression: String::new(),
            last_operator_debug: None,
            active_clients: Vec::new(),
        }
    }
}

#[derive(Default)]
struct FusionClientStoreInner {
    snapshot: FusionClientSnapshot,
    active_clients: HashMap<u64, FusionClientConnectionSnapshot>,
}

pub struct FusionClientStore {
    next_connection_id: AtomicU64,
    inner: RwLock<FusionClientStoreInner>,
}

impl Default for FusionClientStore {
    fn default() -> Self {
        Self {
            next_connection_id: AtomicU64::new(1),
            inner: RwLock::new(FusionClientStoreInner::default()),
        }
    }
}

impl FusionClientStore {
    pub fn record_connect(&self, update: FusionClientConnectionUpdate) -> u64 {
        let connection_id = self.next_connection_id.fetch_add(1, Ordering::Relaxed);
        let client = FusionClientConnectionSnapshot {
            connection_id,
            connected_edge_time_ns: update.edge_time_ns,
            client_addr: update.client_addr.clone(),
            forwarded_for: update.forwarded_for.clone(),
            user_agent: update.user_agent.clone(),
            transport: update.transport.clone(),
            format: update.format.clone(),
            compression: update.compression.clone(),
            operator_debug: update.operator_debug,
        };
        let mut inner = self
            .inner
            .write()
            .expect("fusion client store lock poisoned");
        inner.snapshot.total_connections = inner.snapshot.total_connections.saturating_add(1);
        inner.snapshot.last_connect_edge_time_ns = Some(update.edge_time_ns);
        inner.snapshot.last_client_addr = update.client_addr;
        inner.snapshot.last_forwarded_for = update.forwarded_for;
        inner.snapshot.last_user_agent = update.user_agent;
        inner.snapshot.last_transport = update.transport;
        inner.snapshot.last_format = update.format;
        inner.snapshot.last_compression = update.compression;
        inner.snapshot.last_operator_debug = Some(update.operator_debug);
        inner.active_clients.insert(connection_id, client);
        sync_fusion_client_snapshot(&mut inner);
        connection_id
    }

    pub fn record_disconnect(&self, connection_id: u64, edge_time_ns: u64) {
        let mut inner = self
            .inner
            .write()
            .expect("fusion client store lock poisoned");
        if inner.active_clients.remove(&connection_id).is_some() {
            inner.snapshot.last_disconnect_edge_time_ns = Some(edge_time_ns);
            sync_fusion_client_snapshot(&mut inner);
        }
    }

    pub fn snapshot(&self) -> FusionClientSnapshot {
        self.inner
            .read()
            .expect("fusion client store lock poisoned")
            .snapshot
            .clone()
    }
}

fn sync_fusion_client_snapshot(inner: &mut FusionClientStoreInner) {
    let mut active_clients = inner.active_clients.values().cloned().collect::<Vec<_>>();
    active_clients.sort_by(|left, right| {
        right
            .connected_edge_time_ns
            .cmp(&left.connected_edge_time_ns)
            .then_with(|| right.connection_id.cmp(&left.connection_id))
    });
    inner.snapshot.active_count = active_clients.len();
    inner.snapshot.active_clients = active_clients;
}

fn is_local_debug_session(trip_id: &str, session_id: &str) -> bool {
    let trip_lower = trip_id.trim().to_ascii_lowercase();
    let session_lower = session_id.trim().to_ascii_lowercase();
    trip_lower.contains("local-debug")
        || session_lower.contains("local-debug")
        || trip_lower.starts_with("trip-local")
        || session_lower.starts_with("sess-local")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_sync_ok_uses_fresh_devices_instead_of_last_any_sample() {
        let mut cfg = Config::from_env().expect("config");
        cfg.time_sync_ok_window_ms = 30;
        cfg.time_sync_rtt_ok_ms = 20;
        let gate = Gate::new(cfg.clone());

        gate.record_time_sync(
            "stale-device".to_string(),
            "stereo_pair".to_string(),
            0,
            1_000_000,
            1,
        );
        std::thread::sleep(Duration::from_millis(40));
        assert!(!gate.time_sync_ok(cfg.time_sync_ok_window_ms, cfg.time_sync_rtt_ok_ms));

        gate.record_time_sync(
            "fresh-device".to_string(),
            "stereo_pair".to_string(),
            42,
            2_000_000,
            5,
        );
        assert!(gate.time_sync_ok(cfg.time_sync_ok_window_ms, cfg.time_sync_rtt_ok_ms));
    }

    #[test]
    fn time_sync_summary_marks_freshness_and_rtt_health_per_device() {
        let mut cfg = Config::from_env().expect("config");
        cfg.time_sync_ok_window_ms = 50;
        cfg.time_sync_rtt_ok_ms = 20;
        let gate = Gate::new(cfg.clone());

        gate.record_time_sync(
            "stale-device".to_string(),
            "stereo_pair".to_string(),
            11,
            1_000_000,
            1,
        );
        std::thread::sleep(Duration::from_millis(60));
        gate.record_time_sync(
            "fresh-high-rtt".to_string(),
            "iphone_capture".to_string(),
            22,
            40_000_000,
            1,
        );
        gate.record_time_sync(
            "fresh-ok".to_string(),
            "stereo_pair".to_string(),
            33,
            5_000_000,
            3,
        );

        let summary = gate.time_sync_summary();
        assert_eq!(summary.device_count, 3);
        assert_eq!(summary.fresh_device_count, 2);
        assert_eq!(summary.fresh_ok_device_count, 1);
        assert_eq!(summary.stale_device_count, 1);

        let stale = summary
            .devices
            .iter()
            .find(|device| device.device_id == "stale-device")
            .expect("stale-device");
        assert!(!stale.fresh);
        assert!(stale.rtt_ok);

        let high_rtt = summary
            .devices
            .iter()
            .find(|device| device.device_id == "fresh-high-rtt")
            .expect("fresh-high-rtt");
        assert!(high_rtt.fresh);
        assert!(!high_rtt.rtt_ok);
        assert_eq!(high_rtt.rtt_ok_ms, 30);

        let ok = summary
            .devices
            .iter()
            .find(|device| device.device_id == "fresh-ok")
            .expect("fresh-ok");
        assert!(ok.fresh);
        assert!(ok.rtt_ok);
    }

    #[test]
    fn deadman_snapshot_reports_recent_keepalive_metadata() {
        let gate = Gate::new(Config::from_env().expect("config"));

        gate.ingest_keepalive("trip-123", "sess-123", "device-123", 42, 7, true);

        let snapshot = gate.deadman_snapshot();
        assert!(snapshot.link_ok);
        assert!(snapshot.pressed);
        assert_eq!(snapshot.keepalive.last_seq, Some(7));
        assert_eq!(snapshot.keepalive.last_source_time_ns, Some(42));
        assert_eq!(snapshot.keepalive.last_trip_id, "trip-123");
        assert_eq!(snapshot.keepalive.last_session_id, "sess-123");
        assert_eq!(snapshot.keepalive.last_device_id, "device-123");
        assert!(snapshot.keepalive.last_age_ms.is_some());
        assert!(snapshot.keepalive.last_edge_time_ns.is_some());
        assert!(snapshot.keepalive.packets_in_last_5s >= 1);
        assert!(snapshot.keepalive.approx_rate_hz_5s > 0.0);
    }
}
