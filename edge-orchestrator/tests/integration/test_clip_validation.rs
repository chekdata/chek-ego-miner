use std::path::PathBuf;
use std::time::Duration;

use futures_util::SinkExt;
use tokio_tungstenite::tungstenite::Message;

use crate::support;
use support::ws_harness::WsHarness;

#[tokio::test]
async fn clip_manifest_should_emit_full_validation_summary() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-clip-validation-001";
    let session_id = "sess-clip-validation-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-clip-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    let _harness = WsHarness::connect(&server, trip_id, session_id).await?;
    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;

    ws.send(Message::Text(
        serde_json::json!({
            "type": "label_event_packet",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-clip-001",
            "source_time_ns": 1_770_003_000_000_000_000u64,
            "seq": 1u64,
            "event": "action_start",
            "action_id": "act-clip-001",
            "action_label": "pick",
            "scene_label": "warehouse_line_a"
        })
        .to_string(),
    ))
    .await?;

    for frame_id in 0..4u64 {
        ws.send(Message::Text(
            capture_pose_packet(trip_id, session_id, frame_id).to_string(),
        ))
        .await?;
        tokio::time::sleep(Duration::from_millis(120)).await;
    }

    wait_for_file_contains(
        &server
            .data_dir
            .join("session")
            .join(session_id)
            .join("fused")
            .join("fusion_state.jsonl"),
        "\"fusion_seq\":",
    )
    .await?;

    ws.send(Message::Text(
        serde_json::json!({
            "type": "label_event_packet",
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-clip-001",
            "source_time_ns": 1_770_003_000_500_000_000u64,
            "seq": 2u64,
            "event": "action_end",
            "action_id": "act-clip-001",
            "action_label": "pick",
            "scene_label": "warehouse_line_a"
        })
        .to_string(),
    ))
    .await?;

    let clip_manifest = server
        .data_dir
        .join("session")
        .join(session_id)
        .join("clips")
        .join("act-clip-001_pick_warehouse_line_a__part1")
        .join("clip_manifest.json");
    wait_for_file(&clip_manifest).await?;

    let manifest: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(&clip_manifest).await?)?;
    let validation = manifest.get("validation").cloned().unwrap_or_default();
    assert_eq!(
        validation
            .get("locatable")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        validation.get("playable").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        validation
            .get("label_consistent")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        validation
            .get("index_complete")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        validation.get("pass").and_then(|value| value.as_bool()),
        Some(true)
    );
    Ok(())
}

fn capture_pose_packet(trip_id: &str, session_id: &str, frame_id: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "client-clip-001",
        "device_class": "B",
        "platform": "ios",
        "operator_track_id": "primary_operator",
        "source_time_ns": 1_770_003_100_000_000_000u64 + frame_id,
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

async fn wait_for_file(path: &PathBuf) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待文件超时: {}", path.display());
        }
        if tokio::fs::metadata(path).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_file_contains(path: &PathBuf, needle: &str) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待文件内容超时: {} -> {}", path.display(), needle);
        }
        if let Ok(raw) = tokio::fs::read_to_string(path).await {
            if raw.contains(needle) {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
