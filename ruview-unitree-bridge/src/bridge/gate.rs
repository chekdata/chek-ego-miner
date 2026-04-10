use serde::Serialize;
use tokio::time::{Duration, Instant};

use crate::bridge::parser::ParsedTeleopFrame;
use crate::reason;

#[derive(Clone, Copy, Debug)]
pub struct GateConfig {
    pub keepalive_timeout_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafetyState {
    Normal,
    Limit,
    Freeze,
    Estop,
}

impl SafetyState {
    pub fn parse(s: &str) -> Self {
        match s {
            "limit" => Self::Limit,
            "freeze" => Self::Freeze,
            "estop" => Self::Estop,
            _ => Self::Normal,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GateSnapshot {
    pub control_state: String,
    pub safety_state: String,
    pub keepalive_ok: bool,
    pub reason: String,
    pub last_teleop_edge_time_ns: u64,
}

/// 安全门控（MVP）：
///
/// - `control_state == "armed"` 才允许输出
/// - keepalive 超时保护（基于“收到有效遥操帧”的本地时间）
/// - `safety_state == "estop"` 直接阻断
pub struct Gate {
    cfg: GateConfig,
    last_frame_at: Option<Instant>,
    last_edge_time_ns: u64,
    control_state: String,
    safety_state: String,
    body_control_enabled: bool,
}

impl Gate {
    pub fn new(cfg: GateConfig) -> Self {
        Self {
            cfg,
            last_frame_at: None,
            last_edge_time_ns: 0,
            control_state: "disarmed".to_string(),
            safety_state: "normal".to_string(),
            body_control_enabled: true,
        }
    }

    pub fn ingest(&mut self, parsed: &ParsedTeleopFrame) {
        self.last_frame_at = Some(Instant::now());
        self.last_edge_time_ns = parsed.edge_time_ns;
        self.control_state = parsed.control_state.clone();
        self.safety_state = parsed.safety_state.clone();
        self.body_control_enabled = parsed.body_control_enabled;
    }

    pub fn safety_state(&self) -> SafetyState {
        SafetyState::parse(&self.safety_state)
    }

    /// 决定“是否允许下发会导致运动的命令”。
    pub fn allow_motion(&self, now: Instant) -> (bool, String) {
        // 1) estop 最高优先级
        if self.safety_state() == SafetyState::Estop {
            return (false, reason::REASON_SAFETY_ESTOP.to_string());
        }

        // 2) armed 门控
        if self.control_state != "armed" {
            return (false, reason::REASON_DISARMED.to_string());
        }

        if !self.body_control_enabled {
            return (false, "body_control_disabled".to_string());
        }

        // 3) keepalive
        let Some(at) = self.last_frame_at else {
            return (false, reason::REASON_KEEPALIVE_TIMEOUT.to_string());
        };
        let ok = now.duration_since(at) <= Duration::from_millis(self.cfg.keepalive_timeout_ms);
        if !ok {
            return (false, reason::REASON_KEEPALIVE_TIMEOUT.to_string());
        }

        (true, reason::REASON_OK.to_string())
    }

    pub fn snapshot(&self, now: Instant) -> GateSnapshot {
        let keepalive_ok = match self.last_frame_at {
            Some(at) => {
                now.duration_since(at) <= Duration::from_millis(self.cfg.keepalive_timeout_ms)
            }
            None => false,
        };

        let (_, reason) = self.allow_motion(now);
        GateSnapshot {
            control_state: self.control_state.clone(),
            safety_state: self.safety_state.clone(),
            keepalive_ok,
            reason,
            last_teleop_edge_time_ns: self.last_edge_time_ns,
        }
    }
}
