use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::{mpsc, watch};
use tokio::time::{Duration, Instant};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

use crate::bridge::gate::{Gate, GateConfig, GateSnapshot};
use crate::bridge::pairing::{PairingConfig, PairingEngine};
use crate::bridge::parser::{parse_teleop_frame_v1_json, ParsedHandTargets};
use crate::bridge::retarget::{
    degrade_reason, retarget_paired, RetargetConfig, SideRetargetCalibration,
};
use crate::bridge::types::BridgeStatePacket;
use crate::config::Config;
use crate::leap::client::{LeapClient, MockLeapClient};
use crate::protocol::version_guard::ProtocolVersionInfo;
use crate::reason;
use crate::state::hardware_state::{HardwareReadiness, LeapHardwareState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    EdgeWs,
    Demo,
    Manual,
}

impl InputMode {
    fn from_config(cfg: &Config) -> Self {
        if cfg.edge_teleop_ws_url.is_some() {
            Self::EdgeWs
        } else {
            Self::Demo
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HealthSnapshot {
    pub bridge_id: String,
    pub protocol: ProtocolVersionInfo,
    pub gate: GateSnapshot,
    pub is_ready: bool,
    pub fault_code: String,
    pub fault_message: String,
    pub last_command_edge_time_ns: u64,
    pub left_retarget_calibrated: bool,
    pub right_retarget_calibrated: bool,
    pub left_retarget_joint_count: usize,
    pub right_retarget_joint_count: usize,
    pub left_retarget_non_default_scale_count: usize,
    pub right_retarget_non_default_scale_count: usize,
    pub left_retarget_non_zero_offset_count: usize,
    pub right_retarget_non_zero_offset_count: usize,
}

struct SharedState {
    health: HealthSnapshot,
}

const BRIDGE_STATE_HEARTBEAT_MS: u64 = 250;

#[derive(Clone)]
pub struct RunnerHandle {
    shared: Arc<RwLock<SharedState>>,
    teleop_tx: mpsc::Sender<String>,
    shutdown_tx: watch::Sender<bool>,
}

impl RunnerHandle {
    pub fn health_snapshot(&self) -> HealthSnapshot {
        let guard = self.shared.read().expect("health lock poisoned");
        guard.health.clone()
    }

    pub async fn inject_teleop_json(&self, raw: impl Into<String>) -> Result<(), String> {
        self.teleop_tx
            .send(raw.into())
            .await
            .map_err(|_| "teleop 注入失败：runner 已退出".to_string())
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
}

pub struct BridgeRunner {
    cfg: Config,
    protocol: ProtocolVersionInfo,
    input_mode: InputMode,

    teleop_rx: mpsc::Receiver<String>,
    teleop_tx: mpsc::Sender<String>,
    state_json_tx: watch::Sender<String>,
    state_json_rx: watch::Receiver<String>,
    shutdown_rx: watch::Receiver<bool>,

    shared: Arc<RwLock<SharedState>>,

    leap: Arc<dyn LeapClient>,
    pairing: PairingEngine,
    gate: Gate,
    retarget_cfg: RetargetConfig,
    hw: LeapHardwareState,

    last_parsed: Option<ParsedHandTargets>,
    last_command_edge_time_ns: u64,
    last_state_packet_json: Option<String>,
    last_state_packet_sent_at: Option<Instant>,
}

impl BridgeRunner {
    fn build_health_snapshot(
        &self,
        gate: GateSnapshot,
        is_ready: bool,
        fault_code: String,
        fault_message: String,
    ) -> HealthSnapshot {
        HealthSnapshot {
            bridge_id: self.cfg.bridge_id.clone(),
            protocol: self.protocol.clone(),
            gate,
            is_ready,
            fault_code,
            fault_message,
            last_command_edge_time_ns: self.last_command_edge_time_ns,
            left_retarget_calibrated: self.retarget_cfg.left_calibration.is_active(),
            right_retarget_calibrated: self.retarget_cfg.right_calibration.is_active(),
            left_retarget_joint_count: self.retarget_cfg.left_calibration.joint_count(),
            right_retarget_joint_count: self.retarget_cfg.right_calibration.joint_count(),
            left_retarget_non_default_scale_count: self
                .retarget_cfg
                .left_calibration
                .non_default_scale_count(),
            right_retarget_non_default_scale_count: self
                .retarget_cfg
                .right_calibration
                .non_default_scale_count(),
            left_retarget_non_zero_offset_count: self
                .retarget_cfg
                .left_calibration
                .non_zero_offset_count(),
            right_retarget_non_zero_offset_count: self
                .retarget_cfg
                .right_calibration
                .non_zero_offset_count(),
        }
    }

    pub fn new(cfg: Config, protocol: ProtocolVersionInfo) -> (Self, RunnerHandle) {
        let leap = Arc::new(MockLeapClient::new());
        let mode = InputMode::from_config(&cfg);
        Self::new_with_mode_and_client(cfg, protocol, mode, leap)
    }

    pub fn new_manual(
        cfg: Config,
        protocol: ProtocolVersionInfo,
        leap: Arc<dyn LeapClient>,
    ) -> (Self, RunnerHandle) {
        Self::new_with_mode_and_client(cfg, protocol, InputMode::Manual, leap)
    }

    fn new_with_mode_and_client(
        cfg: Config,
        protocol: ProtocolVersionInfo,
        input_mode: InputMode,
        leap: Arc<dyn LeapClient>,
    ) -> (Self, RunnerHandle) {
        let (teleop_tx, teleop_rx) = mpsc::channel::<String>(1024);
        let (state_json_tx, state_json_rx) = watch::channel::<String>(String::new());
        let (shutdown_tx, shutdown_rx) = watch::channel::<bool>(false);

        let gate = Gate::new(GateConfig {
            keepalive_timeout_ms: cfg.keepalive_timeout_ms,
        });

        let pairing = PairingEngine::new(PairingConfig {
            pairing_window_ms: cfg.pairing_window_ms,
            hold_timeout_ms: cfg.hold_timeout_ms,
            freeze_timeout_ms: cfg.freeze_timeout_ms,
        });

        let retarget_cfg = RetargetConfig::new(cfg.joint_min.clone(), cfg.joint_max.clone())
            .with_side_calibration(
                crate::bridge::types::HandSide::Left,
                SideRetargetCalibration::new(
                    cfg.left_joint_scale.clone(),
                    cfg.left_joint_offset.clone(),
                ),
            )
            .with_side_calibration(
                crate::bridge::types::HandSide::Right,
                SideRetargetCalibration::new(
                    cfg.right_joint_scale.clone(),
                    cfg.right_joint_offset.clone(),
                ),
            );
        let hw = LeapHardwareState::new(cfg.max_temperature_c);

        let runner = Self {
            cfg: cfg.clone(),
            protocol,
            input_mode,
            teleop_rx,
            teleop_tx: teleop_tx.clone(),
            state_json_tx,
            state_json_rx,
            shutdown_rx: shutdown_rx.clone(),
            shared: Arc::new(RwLock::new(SharedState {
                health: HealthSnapshot {
                    bridge_id: String::new(),
                    protocol: ProtocolVersionInfo {
                        name: String::new(),
                        version: String::new(),
                        schema_sha256: String::new(),
                    },
                    gate: GateSnapshot {
                        control_state: String::new(),
                        safety_state: String::new(),
                        keepalive_ok: false,
                        reason: String::new(),
                        last_teleop_edge_time_ns: 0,
                    },
                    is_ready: false,
                    fault_code: String::new(),
                    fault_message: String::new(),
                    last_command_edge_time_ns: 0,
                    left_retarget_calibrated: false,
                    right_retarget_calibrated: false,
                    left_retarget_joint_count: 0,
                    right_retarget_joint_count: 0,
                    left_retarget_non_default_scale_count: 0,
                    right_retarget_non_default_scale_count: 0,
                    left_retarget_non_zero_offset_count: 0,
                    right_retarget_non_zero_offset_count: 0,
                },
            })),
            leap,
            pairing,
            gate,
            retarget_cfg,
            hw,
            last_parsed: None,
            last_command_edge_time_ns: 0,
            last_state_packet_json: None,
            last_state_packet_sent_at: None,
        };
        let initial_health = runner.build_health_snapshot(
            GateSnapshot {
                control_state: "disarmed".to_string(),
                safety_state: "normal".to_string(),
                keepalive_ok: false,
                reason: reason::REASON_UNKNOWN.to_string(),
                last_teleop_edge_time_ns: 0,
            },
            false,
            reason::REASON_UNKNOWN.to_string(),
            "未收到 teleop 帧".to_string(),
        );
        {
            let mut guard = runner.shared.write().expect("health lock poisoned");
            guard.health = initial_health;
        }
        let shared = runner.shared.clone();

        (
            runner,
            RunnerHandle {
                shared,
                teleop_tx,
                shutdown_tx,
            },
        )
    }

    pub async fn run(mut self) {
        match self.input_mode {
            InputMode::EdgeWs => {
                if let Some(url) = self.cfg.edge_teleop_ws_url.clone() {
                    let token = self.cfg.edge_token.clone();
                    let teleop_tx = self.teleop_tx.clone();
                    let state_rx = self.state_json_rx.clone();
                    let mut shutdown = self.shutdown_rx.clone();
                    tokio::spawn(async move {
                        run_edge_ws(url, token, teleop_tx, state_rx, &mut shutdown).await;
                    });
                }
            }
            InputMode::Demo => {
                let teleop_tx = self.teleop_tx.clone();
                let mut shutdown = self.shutdown_rx.clone();
                tokio::spawn(async move { run_demo_source(teleop_tx, &mut shutdown).await });
            }
            InputMode::Manual => {}
        }

        let tick_ns = 1_000_000_000u64 / self.cfg.publish_hz as u64;
        let mut interval = tokio::time::interval(Duration::from_nanos(tick_ns));

        loop {
            if *self.shutdown_rx.borrow() {
                break;
            }

            tokio::select! {
                biased;
                _ = interval.tick() => {
                    self.on_tick().await;
                }
                msg = self.teleop_rx.recv() => {
                    let Some(raw) = msg else {
                        break;
                    };
                    self.on_teleop_raw(&raw);
                }
                _ = self.shutdown_rx.changed() => {}
            }
        }
    }

    fn on_teleop_raw(&mut self, raw: &str) {
        let parsed = match parse_teleop_frame_v1_json(raw, self.cfg.expected_joint_len) {
            Ok(p) => p,
            Err(e) => {
                debug!(error=%e, "teleop_frame_v1 parse failed");
                metrics::counter!("teleop_frame_parse_failed_count").increment(1);
                return;
            }
        };

        if !parsed.is_for_leap() {
            metrics::counter!("teleop_frame_skipped_not_leap_count").increment(1);
            return;
        }

        if let Some(last) = self.last_parsed.as_ref() {
            if parsed.edge_time_ns <= last.edge_time_ns {
                metrics::counter!("teleop_frame_dropped_non_monotonic_count").increment(1);
                debug!(
                    last_edge_time_ns = last.edge_time_ns,
                    edge_time_ns = parsed.edge_time_ns,
                    "drop non-monotonic teleop frame"
                );
                return;
            }
        }

        self.gate.ingest(&parsed);
        self.pairing.ingest(&parsed);
        self.last_parsed = Some(parsed);
        metrics::counter!("teleop_frame_ingested_count").increment(1);
    }

    async fn on_tick(&mut self) {
        let now = Instant::now();
        let gate_snap = self.gate.snapshot(now);

        // 1) 硬件状态（用于 is_ready 与阻断）
        let hardware = match self.leap.get_status().await {
            Ok(s) => self.hw.update(s),
            Err(e) => HardwareReadiness {
                is_ready: false,
                fault_code: reason::REASON_HARDWARE_ERROR.to_string(),
                fault_message: format!("读取硬件状态失败: {e}"),
            },
        };

        // 2) motion gate（不影响 is_ready；只影响“是否下发命令”）
        let (gate_ok, gate_reason) = self.gate.should_emit_motion(now);

        // 3) 配对与 retarget
        let mut published = false;
        let mut reject_reason = String::new();
        if gate_ok && hardware.is_ready {
            if let Some(paired) = self.pairing.pair() {
                // freeze 策略：pairing 进入 Freeze 时，不下发新目标（保持 last_command）
                if paired.degrade == crate::bridge::types::PairingDegrade::Freeze {
                    reject_reason = degrade_reason(paired.degrade).to_string();
                    metrics::counter!("leap_publish_suppressed_freeze_count").increment(1);
                } else if gate_snap.safety_state == "freeze" {
                    reject_reason = "freeze".to_string();
                    metrics::counter!("leap_publish_suppressed_safety_freeze_count").increment(1);
                } else {
                    match retarget_paired(&paired, &self.retarget_cfg) {
                        Ok(cmd) => match self.leap.publish(&cmd).await {
                            Ok(()) => {
                                published = true;
                                self.last_command_edge_time_ns = cmd.edge_time_ns;
                                metrics::counter!("leap_publish_count").increment(1);
                            }
                            Err(e) => {
                                warn!(error=%e, "LEAP publish failed");
                                reject_reason = reason::REASON_HARDWARE_ERROR.to_string();
                                metrics::counter!("leap_publish_failed_count").increment(1);
                            }
                        },
                        Err(code) => {
                            reject_reason = code.to_string();
                            metrics::counter!("leap_retarget_failed_count", "reason" => code)
                                .increment(1);
                        }
                    };
                }

                // pairing 的退化原因（hold/freeze）作为观测指标
                let dr = degrade_reason(paired.degrade);
                metrics::counter!("pairing_degrade_count", "reason" => dr).increment(1);
            } else {
                reject_reason = reason::REASON_UNKNOWN.to_string();
            }
        } else {
            reject_reason = gate_reason.clone();
        }

        if !published {
            metrics::counter!("leap_publish_reject_count", "reason" => reject_reason.clone())
                .increment(1);
        }

        // 4) 组装并上报 bridge_state_packet（注意：is_ready 主要表达“硬件/服务 readiness”，
        // 不应把 disarmed/keepalive 等操作侧门控当成硬件故障上报给 edge）。
        let (trip_id, session_id, robot_type, end_effector_type, edge_time_ns) =
            match self.last_parsed.as_ref() {
                Some(p) => (
                    p.trip_id.clone(),
                    p.session_id.clone(),
                    p.robot_type.clone(),
                    p.end_effector_type.clone(),
                    p.edge_time_ns,
                ),
                None => (
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "LEAP_V2".to_string(),
                    0,
                ),
            };

        let state = BridgeStatePacket {
            ty: "bridge_state_packet",
            schema_version: "1.0.0",
            bridge_id: self.cfg.bridge_id.clone(),
            trip_id,
            session_id,
            robot_type,
            end_effector_type,
            edge_time_ns,
            is_ready: hardware.is_ready,
            fault_code: hardware.fault_code.clone(),
            fault_message: hardware.fault_message.clone(),
            last_command_edge_time_ns: self.last_command_edge_time_ns,
        };

        if let Ok(txt) = serde_json::to_string(&state) {
            self.publish_bridge_state_packet(now, txt);
        }

        // 5) health snapshot（HTTP /health 用）
        {
            let mut guard = self.shared.write().expect("health lock poisoned");
            guard.health = self.build_health_snapshot(
                gate_snap,
                hardware.is_ready,
                hardware.fault_code,
                hardware.fault_message,
            );
        }
    }

    fn publish_bridge_state_packet(&mut self, now: Instant, payload: String) {
        let changed = self.last_state_packet_json.as_ref() != Some(&payload);
        let heartbeat_due = self
            .last_state_packet_sent_at
            .map(|sent_at| {
                now.duration_since(sent_at) >= Duration::from_millis(BRIDGE_STATE_HEARTBEAT_MS)
            })
            .unwrap_or(true);
        if !changed && !heartbeat_due {
            return;
        }
        let _ = self.state_json_tx.send(payload.clone());
        self.last_state_packet_json = Some(payload);
        self.last_state_packet_sent_at = Some(now);
    }
}

async fn run_demo_source(teleop_tx: mpsc::Sender<String>, shutdown: &mut watch::Receiver<bool>) {
    let mut seq: u64 = 0;
    let mut interval = tokio::time::interval(Duration::from_millis(10));
    loop {
        if *shutdown.borrow() {
            break;
        }
        tokio::select! {
            _ = interval.tick() => {
                seq = seq.wrapping_add(1);
                let edge_time_ns = seq * 10_000_000; // 10ms
                let left: Vec<f32> = vec![0.1; 16];
                let right: Vec<f32> = vec![0.1; 16];
                let frame = serde_json::json!({
                    "schema_version": "teleop_frame_v1",
                    "trip_id": "demo-trip",
                    "session_id": "demo-session",
                    "robot_type": "G1_29",
                    "end_effector_type": "LEAP_V2",
                    "edge_time_ns": edge_time_ns,
                    "control_state": "armed",
                    "safety_state": "normal",
                    "hand_joint_layout": "anatomical_joint_16",
                    "hand_target_layout": "anatomical_target_16",
                    "left_hand_target": left,
                    "right_hand_target": right
                });
                let _ = teleop_tx.send(frame.to_string()).await;
            }
            _ = shutdown.changed() => {}
        }
    }
}

async fn run_edge_ws(
    base_url: String,
    token: Option<String>,
    teleop_tx: mpsc::Sender<String>,
    state_rx: watch::Receiver<String>,
    shutdown: &mut watch::Receiver<bool>,
) {
    let url = build_edge_ws_url(&base_url, token.as_deref());
    loop {
        if *shutdown.borrow() {
            break;
        }

        info!(url=%url, "连接 edge /stream/teleop");
        let (ws, _resp) = match connect_async(url.as_str()).await {
            Ok(v) => v,
            Err(e) => {
                warn!(error=%e, "edge ws 连接失败，2s 后重试");
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(2)) => {},
                    _ = shutdown.changed() => {},
                }
                continue;
            }
        };

        let (write, mut read) = ws.split();
        let (outbound_tx, mut outbound_rx) = mpsc::unbounded_channel::<Message>();
        let mut writer_state_rx = state_rx.clone();
        let mut writer_shutdown = shutdown.clone();
        let writer = tokio::spawn(async move {
            let mut write = write;
            loop {
                if *writer_shutdown.borrow() {
                    let _ = write.send(Message::Close(None)).await;
                    return;
                }

                tokio::select! {
                    biased;
                    msg = outbound_rx.recv() => {
                        let Some(msg) = msg else {
                            return;
                        };
                        if let Err(e) = write.send(msg).await {
                            warn!(error=%e, "edge ws 发送错误");
                            return;
                        }
                    }
                    changed = writer_state_rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                        let txt = writer_state_rx.borrow().clone();
                        if txt.trim().is_empty() {
                            continue;
                        }
                        if let Err(e) = write.send(Message::Text(txt)).await {
                            warn!(error=%e, "edge ws 发送错误");
                            return;
                        }
                    }
                    _ = writer_shutdown.changed() => {}
                }
            }
        });
        loop {
            if *shutdown.borrow() {
                writer.abort();
                return;
            }

            tokio::select! {
                biased;
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(txt))) => {
                            let _ = teleop_tx.send(txt).await;
                        }
                        Some(Ok(Message::Binary(bin))) => {
                            match serde_cbor::from_slice::<serde_json::Value>(&bin) {
                                Ok(v) => {
                                    let _ = teleop_tx.send(v.to_string()).await;
                                }
                                Err(_) => {
                                    metrics::counter!("edge_ws_invalid_cbor_count").increment(1);
                                }
                            }
                        }
                        Some(Ok(Message::Ping(p))) => {
                            let _ = outbound_tx.send(Message::Pong(p));
                        }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Ok(_)) => {},
                        Some(Err(e)) => {
                            warn!(error=%e, "edge ws 接收错误");
                            break;
                        }
                    }
                }
                _ = shutdown.changed() => {}
            }
        }
        writer.abort();
    }
}

fn build_edge_ws_url(base: &str, token: Option<&str>) -> String {
    let Some(token) = token.filter(|t| !t.trim().is_empty()) else {
        return base.to_string();
    };
    if base.contains('?') {
        format!("{base}&token={token}")
    } else {
        format!("{base}?token={token}")
    }
}

pub fn http_addr(cfg: &Config) -> Result<SocketAddr, String> {
    cfg.http_addr
        .parse::<SocketAddr>()
        .map_err(|e| format!("HTTP_ADDR 无效: {} ({e})", cfg.http_addr))
}
