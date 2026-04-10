use std::time::Duration;

use base64::Engine;
use flate2::read::GzDecoder;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::support;

#[tokio::test]
async fn fusion_ws_should_support_delta_and_gzip_transport() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-transport-001";
    let session_id = "sess-transport-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-transport-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}&transport=delta&compression=gzip",
        server.ws_base, server.edge_token
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;

    ws.send(Message::Text(
        capture_pose_packet(trip_id, session_id, 1).to_string(),
    ))
    .await?;
    tokio::time::sleep(Duration::from_millis(120)).await;
    ws.send(Message::Text(
        capture_pose_packet(trip_id, session_id, 2).to_string(),
    ))
    .await?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    let mut saw_full = false;
    let mut saw_delta = false;
    while tokio::time::Instant::now() < deadline {
        let msg = match tokio::time::timeout(Duration::from_millis(250), ws.next()).await {
            Ok(Some(Ok(message))) => message,
            Ok(Some(Err(error))) => return Err(error.into()),
            Ok(None) => anyhow::bail!("fusion ws closed"),
            Err(_) => continue,
        };
        let Message::Text(text) = msg else {
            continue;
        };
        let packet: serde_json::Value = serde_json::from_str(&text)?;
        if packet.get("type").and_then(|value| value.as_str()) != Some("stream_transport_packet") {
            continue;
        }
        if packet.get("stream").and_then(|value| value.as_str()) != Some("fusion_state_packet") {
            continue;
        }
        assert_eq!(
            packet.get("compression").and_then(|value| value.as_str()),
            Some("gzip")
        );
        let payload_bytes = packet
            .get("payload_bytes")
            .and_then(|value| value.as_str())
            .expect("payload_bytes missing");
        let decoded = base64::engine::general_purpose::STANDARD.decode(payload_bytes)?;
        let payload = inflate_json(&decoded)?;
        assert!(payload.is_object());

        match packet.get("encoding").and_then(|value| value.as_str()) {
            Some("full") => {
                saw_full = true;
                assert!(payload.get("type").is_some());
            }
            Some("delta") => {
                saw_delta = true;
                assert!(packet
                    .get("base_sequence")
                    .and_then(|value| value.as_u64())
                    .is_some());
            }
            _ => {}
        }

        if saw_full && saw_delta {
            return Ok(());
        }
    }

    anyhow::bail!("等待 delta/gzip 传输帧超时")
}

fn capture_pose_packet(trip_id: &str, session_id: &str, frame_id: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "client-transport-001",
        "device_class": "B",
        "platform": "ios",
        "operator_track_id": "primary_operator",
        "source_time_ns": 1_770_002_000_000_000_000u64 + frame_id,
        "frame_id": frame_id,
        "confidence": {
            "body": 0.95,
            "hand": 0.94
        },
        "body_layout": "coco_body_17",
        "hand_layout": "mediapipe_hand_21",
        "body_kpts_3d": [
            [0.0, 0.2, 0.8],
            [0.0, 0.1 + frame_id as f64 * 0.01, 0.8]
        ],
        "hand_kpts_3d": [[0.1, 0.0, 0.7], [0.12, 0.02, 0.72]]
    })
}

fn inflate_json(bytes: &[u8]) -> anyhow::Result<serde_json::Value> {
    let mut decoder = GzDecoder::new(bytes);
    let mut out = String::new();
    use std::io::Read;
    decoder.read_to_string(&mut out)?;
    Ok(serde_json::from_str(&out)?)
}
