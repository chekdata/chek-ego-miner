use crate::support;

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use support::ws_harness::WsHarness;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn fusion_stress_does_not_break_teleop_stream() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-isolation-001";
    let session_id = "sess-isolation-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-test-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    let ws = WsHarness::connect(&server, trip_id, session_id).await?;

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-test-001",
            "clock_offset_ns": 0,
            "rtt_ns": 1_000_000,
            "sample_count": 8,
        }))
        .send()
        .await?
        .error_for_status()?;

    // 等待 preflight 就绪
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待 preflight 就绪超时");
        }
        let state = client
            .get(format!("{}/control/state", server.http_base))
            .bearer_auth(&server.edge_token)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;
        let preflight = state.get("preflight").cloned().unwrap_or_default();
        let deadman = state.get("deadman").cloned().unwrap_or_default();
        let ok = preflight
            .get("unitree_bridge_ready")
            .and_then(|v| v.as_bool())
            == Some(true)
            && preflight.get("leap_bridge_ready").and_then(|v| v.as_bool()) == Some(true)
            && preflight.get("time_sync_ok").and_then(|v| v.as_bool()) == Some(true)
            && preflight.get("extrinsic_ok").and_then(|v| v.as_bool()) == Some(true)
            && preflight.get("lan_control_ok").and_then(|v| v.as_bool()) == Some(true)
            && deadman.get("link_ok").and_then(|v| v.as_bool()) == Some(true);
        if ok {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // arm
    let arm = client
        .post(format!("{}/control/arm", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "robot_type": "G1_29",
            "end_effector_type": "LEAP_V2",
            "operator_id": "op-test-001",
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(arm.get("state").and_then(|v| v.as_str()), Some("armed"));

    // teleop 读端（单独开一个只读连接，用于统计收帧稳定性）
    let teleop_url = format!(
        "{}/stream/teleop?token={}",
        server.ws_base, server.edge_token
    );
    let (teleop_ws, _) = tokio_tungstenite::connect_async(teleop_url).await?;
    let (_teleop_write, mut teleop_read) = teleop_ws.split();

    // fusion flood 连接
    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let (fusion_ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;
    let (mut fusion_write, mut fusion_read) = fusion_ws.split();

    let payload = "x".repeat(64 * 1024);
    let flood_msg = serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "client-flood-001",
        "source_time_ns": 0,
        "payload": payload,
    })
    .to_string();

    let flood_done = Arc::new(Mutex::new(false));
    let flood_done_rx = flood_done.clone();
    let flood_sender = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(5)); // ~200 msg/s
        for _ in 0..600 {
            tick.tick().await;
            if fusion_write
                .send(Message::Text(flood_msg.clone()))
                .await
                .is_err()
            {
                break;
            }
        }
        *flood_done.lock().await = true;
    });

    let flood_receiver = tokio::spawn(async move {
        while let Some(Ok(_)) = fusion_read.next().await {
            if *flood_done_rx.lock().await {
                break;
            }
        }
    });

    // 统计 teleop 收帧间隔
    let mut times: Vec<tokio::time::Instant> = Vec::new();
    let collect_deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < collect_deadline {
        let next = match tokio::time::timeout(Duration::from_millis(200), teleop_read.next()).await
        {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(Ok(msg)) = next else { continue };
        let Message::Text(txt) = msg else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
            continue;
        };
        if v.get("schema_version").and_then(|s| s.as_str()) == Some("teleop_frame_v1") {
            times.push(tokio::time::Instant::now());
        }
    }

    let _ = flood_sender.await;
    let _ = flood_receiver.await;

    // 计算 p95 帧间隔（越小越好）。目标 50Hz（20ms），测试阈值给更宽松的上限避免环境抖动误报。
    assert!(
        times.len() >= 60,
        "teleop 收帧过少（len={}），可能被 fusion 压力打断",
        times.len()
    );
    let mut intervals_ms: Vec<f64> = times
        .windows(2)
        .map(|w| (w[1] - w[0]).as_secs_f64() * 1000.0)
        .collect();
    intervals_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_idx = ((intervals_ms.len() as f64) * 0.95).floor() as usize;
    let p95 = intervals_ms
        .get(p95_idx.min(intervals_ms.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0);

    assert!(
        p95 <= 100.0,
        "teleop 帧间隔 p95 过大：p95_ms={p95:.2}（len={}）",
        intervals_ms.len()
    );

    // 确保仍处于 armed（避免被压力触发 fault 而停止发帧）
    let state = client
        .get(format!("{}/control/state", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_ne!(state.get("state").and_then(|v| v.as_str()), Some("fault"));

    // 避免未使用变量告警（ws 需要保持存活到此处）
    let _ = ws.latest_teleop_control_state().await;

    Ok(())
}
