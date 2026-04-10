use std::sync::{Arc, RwLock};

use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::{mpsc, watch};
use tokio::time::{Duration, Instant};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

use crate::bridge::gate::{Gate, GateConfig, GateSnapshot, SafetyState};
use crate::bridge::mapper::{map_commands, MapError, MappedCommands, MapperConfig};
use crate::bridge::parser::{parse_teleop_frame_v1_json, ParsedTeleopFrame};
use crate::bridge::validator::{DexKind, EndpointGuard};
use crate::config::Config;
use crate::dds::unitree_client::{MockUnitreeClient, UnitreeClient};
use crate::ik::UnitreeIk;
use crate::protocol::version_guard::ProtocolVersionInfo;
use crate::reason;
use crate::state::bridge_state::{build_bridge_state_packet, BridgeReadiness, BridgeStateInput};

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

    dds: Arc<dyn UnitreeClient>,
    gate: Gate,
    mapper_cfg: MapperConfig,
    ik: Option<UnitreeIk>,

    last_parsed: Option<ParsedTeleopFrame>,
    last_command: Option<MappedCommands>,
    last_command_edge_time_ns: u64,
    last_ingested_edge_time_ns: Option<u64>,
    last_state_packet_json: Option<String>,
    last_state_packet_sent_at: Option<Instant>,

    readiness: BridgeReadiness,
}

impl BridgeRunner {
    pub fn new(cfg: Config, protocol: ProtocolVersionInfo) -> (Self, RunnerHandle) {
        let dds = Arc::new(MockUnitreeClient::new());
        let mode = InputMode::from_config(&cfg);
        Self::new_with_mode_and_client(cfg, protocol, mode, dds)
    }

    pub fn new_manual(
        cfg: Config,
        protocol: ProtocolVersionInfo,
        dds: Arc<dyn UnitreeClient>,
    ) -> (Self, RunnerHandle) {
        Self::new_with_mode_and_client(cfg, protocol, InputMode::Manual, dds)
    }

    fn new_with_mode_and_client(
        cfg: Config,
        protocol: ProtocolVersionInfo,
        input_mode: InputMode,
        dds: Arc<dyn UnitreeClient>,
    ) -> (Self, RunnerHandle) {
        let (teleop_tx, teleop_rx) = mpsc::channel::<String>(1024);
        let (state_json_tx, state_json_rx) = watch::channel::<String>(String::new());
        let (shutdown_tx, shutdown_rx) = watch::channel::<bool>(false);

        let gate = Gate::new(GateConfig {
            keepalive_timeout_ms: cfg.keepalive_timeout_ms,
        });
        let mapper_cfg = MapperConfig::new(
            cfg.arm_joint_min.clone(),
            cfg.arm_joint_max.clone(),
            cfg.dex_joint_min.clone(),
            cfg.dex_joint_max.clone(),
        );

        let ik = match UnitreeIk::new() {
            Ok(v) => Some(v),
            Err(e) => {
                warn!(error=%e, "初始化 IK 失败（将仅支持 arm_q_target 直通）");
                None
            }
        };

        let readiness = BridgeReadiness {
            is_ready: true,
            fault_code: "".to_string(),
            fault_message: "".to_string(),
        };

        let shared = Arc::new(RwLock::new(SharedState {
            health: HealthSnapshot {
                bridge_id: cfg.bridge_id.clone(),
                protocol: protocol.clone(),
                gate: GateSnapshot {
                    control_state: "disarmed".to_string(),
                    safety_state: "normal".to_string(),
                    keepalive_ok: false,
                    reason: reason::REASON_UNKNOWN.to_string(),
                    last_teleop_edge_time_ns: 0,
                },
                is_ready: true,
                fault_code: "".to_string(),
                fault_message: "".to_string(),
                last_command_edge_time_ns: 0,
            },
        }));

        (
            Self {
                cfg: cfg.clone(),
                protocol,
                input_mode,
                teleop_rx,
                teleop_tx: teleop_tx.clone(),
                state_json_tx,
                state_json_rx,
                shutdown_rx: shutdown_rx.clone(),
                shared: shared.clone(),
                dds,
                gate,
                mapper_cfg,
                ik,
                last_parsed: None,
                last_command: None,
                last_command_edge_time_ns: 0,
                last_ingested_edge_time_ns: None,
                last_state_packet_json: None,
                last_state_packet_sent_at: None,
                readiness,
            },
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
                _ = interval.tick() => self.on_tick().await,
                msg = self.teleop_rx.recv() => {
                    let Some(raw) = msg else { break; };
                    self.on_teleop_raw(&raw);
                }
                _ = self.shutdown_rx.changed() => {}
            }
        }
    }

    fn on_teleop_raw(&mut self, raw: &str) {
        let parsed = match parse_teleop_frame_v1_json(
            raw,
            self.cfg.expected_arm_joint_len,
            self.cfg.expected_dex_joint_len,
        ) {
            Ok(p) => p,
            Err(e) => {
                debug!(error=%e, "teleop_frame_v1 parse failed");
                metrics::counter!("teleop_frame_parse_failed_count").increment(1);
                return;
            }
        };

        if let Some(last) = self.last_ingested_edge_time_ns {
            if parsed.edge_time_ns <= last {
                metrics::counter!("teleop_frame_dropped_non_monotonic_count").increment(1);
                debug!(
                    last_edge_time_ns = last,
                    edge_time_ns = parsed.edge_time_ns,
                    "drop non-monotonic teleop frame"
                );
                return;
            }
        }
        self.last_ingested_edge_time_ns = Some(parsed.edge_time_ns);

        self.gate.ingest(&parsed);
        self.last_parsed = Some(parsed);
        metrics::counter!("teleop_frame_ingested_count").increment(1);
    }

    async fn on_tick(&mut self) {
        let now = Instant::now();
        let gate_snap = self.gate.snapshot(now);
        let safety = self.gate.safety_state();
        let (gate_ok, gate_reason) = self.gate.allow_motion(now);

        let mut published = false;
        let mut reject_reason = reason::REASON_UNKNOWN.to_string();

        if gate_ok {
            match safety {
                SafetyState::Freeze => {
                    // freeze：复用上一帧命令（若存在）；否则不发布
                    if let Some(cmd) = self.last_command.clone() {
                        match self.publish_dds(&cmd).await {
                            Ok(()) => {
                                published = true;
                                self.readiness = BridgeReadiness::ok();
                                self.last_command_edge_time_ns = cmd.arm.edge_time_ns;
                                metrics::counter!("unitree_publish_count").increment(1);
                            }
                            Err(e) => {
                                warn!(error=%e, "DDS publish failed");
                                self.readiness = BridgeReadiness {
                                    is_ready: false,
                                    fault_code: reason::REASON_DDS_PUBLISH_FAILED.to_string(),
                                    fault_message: e.to_string(),
                                };
                                reject_reason = reason::REASON_DDS_PUBLISH_FAILED.to_string();
                                metrics::counter!("unitree_publish_failed_count").increment(1);
                            }
                        }
                    } else {
                        reject_reason = "freeze_no_last".to_string();
                    }
                }
                SafetyState::Limit | SafetyState::Normal => {
                    if let Some(frame) = self.last_parsed.as_ref() {
                        let mut frame_for_map = frame.clone();
                        let mut ik_failed: Option<&'static str> = None;
                        if frame_for_map.arm_q_target.is_none() {
                            let seed = self.last_command.as_ref().map(|c| c.arm.q.as_slice());
                            ik_failed = match self.ik.as_ref() {
                                Some(ik) => match ik.solve_arm_q_target(
                                    frame_for_map.robot_type.as_str(),
                                    &frame_for_map.left_wrist_pose,
                                    &frame_for_map.right_wrist_pose,
                                    seed,
                                    self.cfg.expected_arm_joint_len,
                                ) {
                                    Ok(q) => {
                                        frame_for_map.arm_q_target = Some(q);
                                        None
                                    }
                                    Err(code) => Some(code),
                                },
                                None => Some(reason::REASON_IK_UNIMPLEMENTED),
                            };
                            if let Some(code) = ik_failed {
                                reject_reason = code.to_string();
                                metrics::counter!("unitree_ik_failed_count", "reason" => code)
                                    .increment(1);
                            }
                        }

                        let guard = EndpointGuard::new(frame.end_effector_type.clone());
                        let mapped = if ik_failed.is_some() {
                            None
                        } else {
                            match map_commands(&frame_for_map, &guard, &self.mapper_cfg) {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    let MapError::Failed(code) = e;
                                    reject_reason = code.to_string();
                                    metrics::counter!("unitree_map_failed_count", "reason" => code)
                                        .increment(1);
                                    None
                                }
                            }
                        };

                        if let Some(mapped) = mapped {
                            // endpoint guard：显式拒绝不允许的 Dex publish（防呆）
                            if (mapped.dex3_left.is_some() || mapped.dex3_right.is_some())
                                && guard.ensure_dex_allowed(DexKind::Dex3).is_err()
                            {
                                metrics::counter!(
                                    "endpoint_guard_denied_count",
                                    "kind" => "dex3"
                                )
                                .increment(1);
                            }
                            if (mapped.dex1_left.is_some() || mapped.dex1_right.is_some())
                                && guard.ensure_dex_allowed(DexKind::Dex1).is_err()
                            {
                                metrics::counter!(
                                    "endpoint_guard_denied_count",
                                    "kind" => "dex1"
                                )
                                .increment(1);
                            }

                            match self.publish_dds(&mapped).await {
                                Ok(()) => {
                                    published = true;
                                    self.readiness = BridgeReadiness::ok();
                                    self.last_command = Some(mapped.clone());
                                    self.last_command_edge_time_ns = mapped.arm.edge_time_ns;
                                    metrics::counter!("unitree_publish_count").increment(1);
                                }
                                Err(e) => {
                                    warn!(error=%e, "DDS publish failed");
                                    self.readiness = BridgeReadiness {
                                        is_ready: false,
                                        fault_code: reason::REASON_DDS_PUBLISH_FAILED.to_string(),
                                        fault_message: e.to_string(),
                                    };
                                    reject_reason = reason::REASON_DDS_PUBLISH_FAILED.to_string();
                                    metrics::counter!("unitree_publish_failed_count").increment(1);
                                }
                            }
                        }
                    } else {
                        reject_reason = reason::REASON_UNKNOWN.to_string();
                    }
                }
                SafetyState::Estop => {
                    reject_reason = reason::REASON_SAFETY_ESTOP.to_string();
                }
            }
        } else {
            reject_reason = gate_reason.clone();
        }

        if !published {
            metrics::counter!("unitree_publish_reject_count", "reason" => reject_reason.clone())
                .increment(1);
        }

        self.update_state_packets(&gate_snap);
    }

    async fn publish_dds(
        &self,
        mapped: &MappedCommands,
    ) -> Result<(), crate::dds::unitree_client::DdsPublishError> {
        self.dds.publish_arm(&mapped.arm).await?;
        if let Some(cmd) = mapped.dex3_left.as_ref() {
            self.dds.publish_dex3_left(cmd).await?;
        }
        if let Some(cmd) = mapped.dex3_right.as_ref() {
            self.dds.publish_dex3_right(cmd).await?;
        }
        if let Some(cmd) = mapped.dex1_left.as_ref() {
            self.dds.publish_dex1_left(cmd).await?;
        }
        if let Some(cmd) = mapped.dex1_right.as_ref() {
            self.dds.publish_dex1_right(cmd).await?;
        }
        Ok(())
    }

    fn update_state_packets(&mut self, gate_snap: &GateSnapshot) {
        let now = Instant::now();
        // 组装并上报 bridge_state_packet：
        // - is_ready 主要表达“DDS/服务 readiness”，不应把 disarmed/keepalive 当成硬故障上报
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
                    "".to_string(),
                    0,
                ),
            };

        let pkt = build_bridge_state_packet(
            BridgeStateInput {
                bridge_id: &self.cfg.bridge_id,
                trip_id: &trip_id,
                session_id: &session_id,
                robot_type: &robot_type,
                end_effector_type: &end_effector_type,
                edge_time_ns,
                last_command_edge_time_ns: self.last_command_edge_time_ns,
                control_state: Some(&gate_snap.control_state),
                safety_state: Some(&gate_snap.safety_state),
                body_control_enabled: self
                    .last_parsed
                    .as_ref()
                    .map(|parsed| parsed.body_control_enabled),
                hand_control_enabled: self.last_parsed.as_ref().map(|parsed| {
                    parsed.left_hand_target.is_some() || parsed.right_hand_target.is_some()
                }),
                arm_q_commanded: self.last_command.as_ref().map(|cmd| cmd.arm.q.as_slice()),
                arm_tau_commanded: self
                    .last_command
                    .as_ref()
                    .and_then(|cmd| cmd.arm.tau.as_deref()),
                left_hand_q_commanded: self
                    .last_command
                    .as_ref()
                    .and_then(|cmd| cmd.dex3_left.as_ref().or(cmd.dex1_left.as_ref()))
                    .map(|cmd| cmd.q.as_slice()),
                right_hand_q_commanded: self
                    .last_command
                    .as_ref()
                    .and_then(|cmd| cmd.dex3_right.as_ref().or(cmd.dex1_right.as_ref()))
                    .map(|cmd| cmd.q.as_slice()),
            },
            &self.readiness,
        );
        if let Ok(txt) = serde_json::to_string(&pkt) {
            self.publish_bridge_state_packet(now, txt);
        }

        {
            let mut guard = self.shared.write().expect("health lock poisoned");
            guard.health = HealthSnapshot {
                bridge_id: self.cfg.bridge_id.clone(),
                protocol: self.protocol.clone(),
                gate: gate_snap.clone(),
                is_ready: self.readiness.is_ready,
                fault_code: self.readiness.fault_code.clone(),
                fault_message: self.readiness.fault_message.clone(),
                last_command_edge_time_ns: self.last_command_edge_time_ns,
            };
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
                let frame = serde_json::json!({
                    "schema_version": "teleop_frame_v1",
                    "trip_id": "demo-trip",
                    "session_id": "demo-session",
                    "robot_type": "G1_29",
                    "end_effector_type": "LEAP_V2",
                    "edge_time_ns": edge_time_ns,
                    "operator_frame": "op",
                    "robot_base_frame": "base",
                    "extrinsic_version": "v1",
                    "control_state": "armed",
                    "left_wrist_pose": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
                    "right_wrist_pose": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
                    "quality": { "source_mode": "fused", "fused_conf": 1.0, "vision_conf": 0.0, "csi_conf": 0.0 },
                    "safety_state": "normal",
                    "arm_q_target": [0.0, 0.1, 0.2],
                    "arm_tauff_target": [0.0, 0.0, 0.0]
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
