use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::support;
use support::ws_harness::WsHarness;

#[tokio::test]
async fn fusion_stream_should_expose_operator_debug_views_when_requested() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-fusion-debug-001";
    let session_id = "sess-fusion-debug-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-debug-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    let _ws = WsHarness::connect(&server, trip_id, session_id).await?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}&debug_views=operator",
        server.ws_base, server.edge_token
    );
    let (fusion_ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;
    let (mut write, mut read) = fusion_ws.split();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    let mut seq = 0u64;
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待 operator_debug 视图超时");
        }

        write
            .send(Message::Text(
                capture_pose_packet(trip_id, session_id, seq).to_string(),
            ))
            .await?;
        client
            .post(format!("{}/ingest/stereo_pose", server.http_base))
            .bearer_auth(&server.edge_token)
            .json(&serde_json::json!({
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "device_id": "stereo-debug-001",
                "source_time_ns": 1_770_000_001_000_000_000u64 + seq,
                "left_frame_id": seq,
                "right_frame_id": seq + 1,
                "body_layout": "pico_body_24",
                "hand_layout": "pico_hand_26",
                "body_kpts_3d": build_pico_body_points(),
                "hand_kpts_3d": build_pico_dual_hand_points(),
                "stereo_confidence": 0.95
            }))
            .send()
            .await?
            .error_for_status()?;
        seq += 2;

        let msg = match tokio::time::timeout(Duration::from_millis(250), read.next()).await {
            Ok(Some(Ok(message))) => message,
            Ok(Some(Err(error))) => return Err(error.into()),
            Ok(None) => anyhow::bail!("fusion ws closed"),
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(40)).await;
                continue;
            }
        };
        let Message::Text(text) = msg else {
            continue;
        };
        let Ok(packet) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if packet.get("type").and_then(|value| value.as_str()) != Some("fusion_state_packet") {
            continue;
        }

        let Some(operator_debug) = packet.get("operator_debug") else {
            continue;
        };
        let iphone = operator_debug
            .get("iphone_capture")
            .cloned()
            .unwrap_or_default();
        let stereo = operator_debug
            .get("stereo_pair")
            .cloned()
            .unwrap_or_default();
        let fused = operator_debug
            .get("fused_pose")
            .cloned()
            .unwrap_or_default();
        let motion_state = operator_debug
            .get("motion_state")
            .cloned()
            .unwrap_or_default();

        if iphone.get("available").and_then(|value| value.as_bool()) != Some(true) {
            continue;
        }
        if stereo.get("available").and_then(|value| value.as_bool()) != Some(true) {
            continue;
        }
        assert_eq!(
            iphone.get("body_layout").and_then(|value| value.as_str()),
            Some("pico_body_24")
        );
        assert_eq!(
            iphone
                .get("raw_body_count")
                .and_then(|value| value.as_u64()),
            Some(24)
        );
        assert_eq!(
            iphone
                .get("body_kpts_3d")
                .and_then(|value| value.as_array())
                .map(|value| value.len()),
            Some(17)
        );
        assert_eq!(
            stereo.get("hand_layout").and_then(|value| value.as_str()),
            Some("pico_hand_26")
        );
        assert_eq!(
            stereo
                .get("raw_hand_count")
                .and_then(|value| value.as_u64()),
            Some(52)
        );
        assert_eq!(
            fused
                .get("raw_hand_layout")
                .and_then(|value| value.as_str()),
            Some("pico_hand_26")
        );
        assert_eq!(
            fused.get("body_source").and_then(|value| value.as_str()),
            Some("fused_stereo_vision_3d")
        );
        assert_eq!(
            fused.get("hand_source").and_then(|value| value.as_str()),
            Some("stereo")
        );
        assert_eq!(
            fused
                .get("selected_source")
                .and_then(|value| value.as_str()),
            Some("fused_stereo_vision_3d")
        );
        assert_eq!(
            fused
                .get("stereo_body_joint_count")
                .and_then(|value| value.as_u64()),
            Some(17)
        );
        assert_eq!(
            fused
                .get("vision_hand_point_count")
                .and_then(|value| value.as_u64()),
            Some(0)
        );
        assert_eq!(
            fused
                .get("hand_kpts_3d")
                .and_then(|value| value.as_array())
                .map(|value| value.len()),
            Some(42)
        );
        assert_eq!(
            motion_state
                .get("smoother_mode")
                .and_then(|value| value.as_str()),
            Some("stereo_live")
        );
        assert_eq!(
            motion_state
                .get("stereo_measurement_used")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert!(
            motion_state
                .get("updated_edge_time_ns")
                .and_then(|value| value.as_u64())
                .unwrap_or_default()
                > 0
        );
        return Ok(());
    }
}

#[tokio::test]
async fn fusion_stream_should_mix_body_and_hand_from_different_sources() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-fusion-mixed-001";
    let session_id = "sess-fusion-mixed-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-debug-002",
        }))
        .send()
        .await?
        .error_for_status()?;

    let _ws = WsHarness::connect(&server, trip_id, session_id).await?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}&debug_views=operator",
        server.ws_base, server.edge_token
    );
    let (fusion_ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;
    let (mut write, mut read) = fusion_ws.split();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待 mixed operator_debug 视图超时");
        }

        write
            .send(Message::Text(
                serde_json::json!({
                    "type": "capture_pose_packet",
                    "schema_version": "1.0.0",
                    "trip_id": trip_id,
                    "session_id": session_id,
                    "device_id": "client-debug-002",
                    "source_time_ns": 1_770_000_010_000_000_000u64,
                    "seq": 1u64,
                    "body_layout": "pico_body_24",
                    "hand_layout": "pico_hand_26",
                    "confidence": { "body": 0.96, "hand": 0.93 },
                    "body_kpts_3d": build_pico_body_points(),
                    "hand_kpts_3d": []
                })
                .to_string(),
            ))
            .await?;
        client
            .post(format!("{}/ingest/stereo_pose", server.http_base))
            .bearer_auth(&server.edge_token)
            .json(&serde_json::json!({
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "device_id": "stereo-debug-002",
                "source_time_ns": 1_770_000_011_000_000_000u64,
                "left_frame_id": 10u64,
                "right_frame_id": 11u64,
                "body_layout": "pico_body_24",
                "hand_layout": "pico_hand_26",
                "body_kpts_3d": [],
                "hand_kpts_3d": build_pico_dual_hand_points(),
                "stereo_confidence": 0.95
            }))
            .send()
            .await?
            .error_for_status()?;

        let msg = match tokio::time::timeout(Duration::from_millis(250), read.next()).await {
            Ok(Some(Ok(message))) => message,
            Ok(Some(Err(error))) => return Err(error.into()),
            Ok(None) => anyhow::bail!("fusion ws closed"),
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(40)).await;
                continue;
            }
        };
        let Message::Text(text) = msg else {
            continue;
        };
        let Ok(packet) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if packet.get("type").and_then(|value| value.as_str()) != Some("fusion_state_packet") {
            continue;
        }
        let Some(fused) = packet
            .get("operator_debug")
            .and_then(|value| value.get("fused_pose"))
        else {
            continue;
        };
        assert_eq!(
            fused
                .get("selected_source")
                .and_then(|value| value.as_str()),
            Some("fused_stereo_vision_3d")
        );
        assert_eq!(
            fused.get("body_source").and_then(|value| value.as_str()),
            Some("vision_3d")
        );
        assert_eq!(
            fused.get("hand_source").and_then(|value| value.as_str()),
            Some("stereo")
        );
        assert_eq!(
            fused
                .get("vision_body_joint_count")
                .and_then(|value| value.as_u64()),
            Some(17)
        );
        assert_eq!(
            fused
                .get("stereo_hand_point_count")
                .and_then(|value| value.as_u64()),
            Some(42)
        );
        return Ok(());
    }
}

fn capture_pose_packet(trip_id: &str, session_id: &str, seq: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "client-debug-001",
        "source_time_ns": 1_770_000_000_000_000_000u64 + seq,
        "seq": seq,
        "body_layout": "pico_body_24",
        "hand_layout": "pico_hand_26",
        "confidence": {
            "body": 0.96,
            "hand": 0.93
        },
        "body_kpts_3d": build_pico_body_points(),
        "hand_kpts_3d": build_pico_dual_hand_points(),
    })
}

fn build_pico_body_points() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.04, 0.80],
        [-0.05, 0.06, 0.80],
        [0.05, 0.06, 0.80],
        [0.0, 0.11, 0.81],
        [-0.05, -0.18, 0.79],
        [0.05, -0.18, 0.79],
        [0.0, 0.18, 0.82],
        [-0.05, -0.44, 0.78],
        [0.05, -0.44, 0.78],
        [0.0, 0.28, 0.83],
        [-0.05, -0.48, 0.83],
        [0.05, -0.48, 0.83],
        [0.0, 0.34, 0.82],
        [-0.04, 0.30, 0.82],
        [0.04, 0.30, 0.82],
        [0.0, 0.39, 0.82],
        [-0.08, 0.22, 0.82],
        [0.08, 0.22, 0.82],
        [-0.11, 0.25, 0.84],
        [0.11, 0.23, 0.84],
        [-0.132, 0.304, 0.853],
        [0.132, 0.209, 0.853],
        [-0.145, 0.33, 0.86],
        [0.145, 0.23, 0.86],
    ]
}

fn build_pico_dual_hand_points() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    out.extend(build_pico_hand_points([-0.132, 0.304, 0.853], true));
    out.extend(build_pico_hand_points([0.132, 0.209, 0.853], false));
    out
}

fn build_pico_hand_points(wrist: [f32; 3], left: bool) -> Vec<[f32; 3]> {
    let toward_thumb = if left { -1.0 } else { 1.0 };
    let toward_pinky = if left { -1.0 } else { 1.0 };

    vec![
        offset(wrist, 0.0, 0.030, 0.0),
        wrist,
        offset(wrist, 0.010 * toward_thumb, 0.020, 0.0),
        offset(wrist, 0.022 * toward_thumb, 0.040, 0.0),
        offset(wrist, 0.034 * toward_thumb, 0.061, 0.0),
        offset(wrist, 0.046 * toward_thumb, 0.080, 0.0),
        offset(wrist, 0.018 * toward_thumb, 0.040, 0.0),
        offset(wrist, 0.020 * toward_thumb, 0.058, 0.0),
        offset(wrist, 0.024 * toward_thumb, 0.081, 0.0),
        offset(wrist, 0.028 * toward_thumb, 0.102, 0.0),
        offset(wrist, 0.031 * toward_thumb, 0.122, 0.0),
        offset(wrist, 0.004 * toward_thumb, 0.043, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.064, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.088, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.112, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.133, 0.0),
        offset(wrist, -0.012 * toward_thumb, 0.038, 0.0),
        offset(wrist, -0.014 * toward_thumb, 0.054, 0.0),
        offset(wrist, -0.018 * toward_thumb, 0.078, 0.0),
        offset(wrist, -0.022 * toward_thumb, 0.101, 0.0),
        offset(wrist, -0.026 * toward_thumb, 0.121, 0.0),
        offset(wrist, -0.028 * toward_pinky, 0.028, 0.0),
        offset(wrist, -0.033 * toward_pinky, 0.039, 0.0),
        offset(wrist, -0.037 * toward_pinky, 0.063, 0.0),
        offset(wrist, -0.041 * toward_pinky, 0.087, 0.0),
        offset(wrist, -0.045 * toward_pinky, 0.109, 0.0),
    ]
}

fn offset(base: [f32; 3], dx: f32, dy: f32, dz: f32) -> [f32; 3] {
    [base[0] + dx, base[1] + dy, base[2] + dz]
}
