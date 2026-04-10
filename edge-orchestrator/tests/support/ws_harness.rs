#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio::sync::{watch, Mutex};
use tokio_tungstenite::tungstenite::Message;

use super::TestServer;

#[derive(Clone, Copy, Debug)]
struct KeepaliveControl {
    enabled: bool,
    pressed: bool,
}

pub struct WsHarness {
    shutdown_tx: watch::Sender<bool>,
    keepalive_tx: watch::Sender<KeepaliveControl>,

    teleop_frame: Arc<Mutex<Option<serde_json::Value>>>,
    #[allow(dead_code)]
    fusion_state: Arc<Mutex<Option<serde_json::Value>>>,
    chunk_ack: Arc<Mutex<Option<serde_json::Value>>>,

    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl WsHarness {
    pub async fn connect(
        server: &TestServer,
        trip_id: &str,
        session_id: &str,
    ) -> anyhow::Result<Self> {
        let fusion_url = format!(
            "{}/stream/fusion?token={}",
            server.ws_base, server.edge_token
        );
        let teleop_url = format!(
            "{}/stream/teleop?token={}",
            server.ws_base, server.edge_token
        );

        let (fusion_ws, _) = tokio_tungstenite::connect_async(fusion_url)
            .await
            .context("连接 /stream/fusion 失败")?;
        let (teleop_ws, _) = tokio_tungstenite::connect_async(teleop_url)
            .await
            .context("连接 /stream/teleop 失败")?;

        let (fusion_write, fusion_read) = fusion_ws.split();
        let (teleop_write, teleop_read) = teleop_ws.split();

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (keepalive_tx, keepalive_rx) = watch::channel(KeepaliveControl {
            enabled: true,
            pressed: false,
        });

        let teleop_frame: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let fusion_state: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let chunk_ack: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));

        let tasks: Vec<tokio::task::JoinHandle<()>> = vec![
            tokio::spawn(run_keepalive_sender(
                fusion_write,
                shutdown_rx.clone(),
                keepalive_rx,
                trip_id.to_string(),
                session_id.to_string(),
            )),
            tokio::spawn(run_fusion_receiver(
                fusion_read,
                shutdown_rx.clone(),
                fusion_state.clone(),
                chunk_ack.clone(),
            )),
            tokio::spawn(run_bridge_state_sender(
                teleop_write,
                shutdown_rx.clone(),
                trip_id.to_string(),
                session_id.to_string(),
            )),
            tokio::spawn(run_teleop_receiver(
                teleop_read,
                shutdown_rx,
                teleop_frame.clone(),
            )),
        ];

        Ok(Self {
            shutdown_tx,
            keepalive_tx,
            teleop_frame,
            fusion_state,
            chunk_ack,
            tasks,
        })
    }

    pub fn set_keepalive_enabled(&self, enabled: bool) {
        self.keepalive_tx.send_modify(|v| v.enabled = enabled);
    }

    pub fn set_deadman_pressed(&self, pressed: bool) {
        self.keepalive_tx.send_modify(|v| v.pressed = pressed);
    }

    pub async fn latest_teleop_control_state(&self) -> Option<String> {
        let frame = self.teleop_frame.lock().await;
        frame
            .as_ref()
            .and_then(|f| f.get("control_state"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    pub async fn latest_teleop_frame(&self) -> Option<serde_json::Value> {
        self.teleop_frame.lock().await.clone()
    }

    pub async fn wait_teleop_control_state(
        &self,
        expected: &str,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if tokio::time::Instant::now() > deadline {
                let got = self.latest_teleop_control_state().await;
                anyhow::bail!("等待 teleop control_state 超时：expected={expected} got={got:?}");
            }
            if self.latest_teleop_control_state().await.as_deref() == Some(expected) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    pub async fn wait_chunk_ack_stored(
        &self,
        chunk_index: u32,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if tokio::time::Instant::now() > deadline {
                let got = self.chunk_ack.lock().await.clone();
                anyhow::bail!("等待 chunk_ack(stored) 超时：chunk_index={chunk_index} got={got:?}");
            }
            let got = self.chunk_ack.lock().await.clone();
            if let Some(v) = got {
                let idx_ok =
                    v.get("chunk_index").and_then(|x| x.as_u64()) == Some(chunk_index as u64);
                let status_ok = v.get("status").and_then(|x| x.as_str()) == Some("stored");
                if idx_ok && status_ok {
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

impl Drop for WsHarness {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(true);
        for t in &self.tasks {
            t.abort();
        }
    }
}

async fn run_keepalive_sender<
    W: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
>(
    mut write: W,
    shutdown_rx: watch::Receiver<bool>,
    keepalive_rx: watch::Receiver<KeepaliveControl>,
    trip_id: String,
    session_id: String,
) {
    let mut seq: u64 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        let ctl = *keepalive_rx.borrow();
        if ctl.enabled {
            seq += 1;
            let packet = json!({
                "type": "control_keepalive_packet",
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "device_id": "client-test-001",
                "source_time_ns": 0,
                "seq": seq,
                "deadman_pressed": ctl.pressed,
            });
            let _ = write.send(Message::Text(packet.to_string())).await;
        }

        tokio::time::sleep(Duration::from_millis(50)).await; // 20Hz
    }
}

async fn run_bridge_state_sender<
    W: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
>(
    mut write: W,
    shutdown_rx: watch::Receiver<bool>,
    trip_id: String,
    session_id: String,
) {
    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        let base = json!({
            "type": "bridge_state_packet",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "robot_type": "G1_29",
            "end_effector_type": "LEAP_V2",
            "edge_time_ns": 0,
            "is_ready": true,
            "fault_code": "",
            "fault_message": "",
            "last_command_edge_time_ns": 0,
        });

        let unitree = {
            let mut v = base.clone();
            v["bridge_id"] = json!("unitree-bridge-01");
            v
        };
        let leap = {
            let mut v = base.clone();
            v["bridge_id"] = json!("leap-bridge-01");
            v
        };

        let _ = write.send(Message::Text(unitree.to_string())).await;
        let _ = write.send(Message::Text(leap.to_string())).await;

        tokio::time::sleep(Duration::from_millis(100)).await; // 10Hz
    }
}

async fn run_fusion_receiver<
    R: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
>(
    mut read: R,
    shutdown_rx: watch::Receiver<bool>,
    fusion_state: Arc<Mutex<Option<serde_json::Value>>>,
    chunk_ack: Arc<Mutex<Option<serde_json::Value>>>,
) {
    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        let msg = match read.next().await {
            Some(Ok(m)) => m,
            Some(Err(_)) => return,
            None => return,
        };
        let Message::Text(txt) = msg else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
            continue;
        };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("fusion_state_packet") => {
                *fusion_state.lock().await = Some(v);
            }
            Some("chunk_ack_packet") => {
                *chunk_ack.lock().await = Some(v);
            }
            _ => {}
        }
    }
}

async fn run_teleop_receiver<
    R: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
>(
    mut read: R,
    shutdown_rx: watch::Receiver<bool>,
    teleop_frame: Arc<Mutex<Option<serde_json::Value>>>,
) {
    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        let msg = match read.next().await {
            Some(Ok(m)) => m,
            Some(Err(_)) => return,
            None => return,
        };
        let Message::Text(txt) = msg else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
            continue;
        };
        if v.get("schema_version").and_then(|t| t.as_str()) == Some("teleop_frame_v1") {
            *teleop_frame.lock().await = Some(v);
        }
    }
}
