use std::time::Duration;

use futures_util::SinkExt;
use tokio_tungstenite::tungstenite::Message;

use crate::support;

#[tokio::test]
async fn session_metrics_should_report_id_switch_and_vision_timeout() -> anyhow::Result<()> {
    let server = support::TestServer::spawn_with_env([("VISION_STALE_MS", "400")]).await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-metrics-001";
    let session_id = "sess-metrics-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-metrics-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;

    ws.send(Message::Text(
        capture_pose_packet(trip_id, session_id, 1, "track-a").to_string(),
    ))
    .await?;
    tokio::time::sleep(Duration::from_millis(120)).await;
    ws.send(Message::Text(
        capture_pose_packet(trip_id, session_id, 2, "track-b").to_string(),
    ))
    .await?;

    tokio::time::sleep(Duration::from_millis(1_300)).await;

    let metrics = wait_for_metrics(&client, &server.http_base, &server.edge_token).await?;

    assert_eq!(metric_value(&metrics, "id_switch_rate"), Some(100.0));
    assert!(metric_value(&metrics, "vision_timeout_count").unwrap_or_default() >= 1.0);
    assert!(metric_value(&metrics, "id_switch_count").unwrap_or_default() >= 1.0);
    assert!(metrics.contains("host_cpu_usage_percent"));
    assert!(metrics.contains("host_memory_used_percent"));
    Ok(())
}

async fn wait_for_metrics(
    client: &reqwest::Client,
    http_base: &str,
    edge_token: &str,
) -> anyhow::Result<String> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    loop {
        let metrics = client
            .get(format!("{http_base}/metrics"))
            .bearer_auth(edge_token)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        if metric_value(&metrics, "id_switch_rate") == Some(100.0)
            && metric_value(&metrics, "vision_timeout_count").unwrap_or_default() >= 1.0
            && metric_value(&metrics, "id_switch_count").unwrap_or_default() >= 1.0
        {
            return Ok(metrics);
        }
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待指标稳定超时: {metrics}");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn capture_pose_packet(
    trip_id: &str,
    session_id: &str,
    frame_id: u64,
    operator_track_id: &str,
) -> serde_json::Value {
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "client-metrics-001",
        "device_class": "B",
        "platform": "ios",
        "operator_track_id": operator_track_id,
        "source_time_ns": 1_770_001_000_000_000_000u64 + frame_id,
        "frame_id": frame_id,
        "confidence": {
            "body": 0.95,
            "hand": 0.94
        },
        "body_layout": "coco_body_17",
        "hand_layout": "mediapipe_hand_21",
        "body_kpts_3d": [[0.0, 0.2, 0.8], [0.0, 0.1, 0.8]],
        "hand_kpts_3d": [[0.1, 0.0, 0.7], [0.12, 0.02, 0.72]]
    })
}

fn metric_value(metrics: &str, name: &str) -> Option<f64> {
    metrics.lines().find_map(|line| {
        let trimmed = line.trim();
        if !trimmed.starts_with(name) {
            return None;
        }
        trimmed
            .split_whitespace()
            .last()
            .and_then(|value| value.parse::<f64>().ok())
    })
}
