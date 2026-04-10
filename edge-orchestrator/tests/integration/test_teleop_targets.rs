use std::time::Duration;

use futures_util::SinkExt;
use tokio_tungstenite::tungstenite::Message;

use crate::support;
use support::ws_harness::WsHarness;

#[tokio::test]
async fn teleop_frame_should_include_precomputed_targets() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-target-001";
    let session_id = "sess-target-001";

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

    support::enable_full_teleop_profile(&client, &server, trip_id, session_id).await?;

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

    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let (mut fusion_ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    let mut seq = 0u64;
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!(
                "等待 precomputed target 超时，latest={:?}",
                ws.latest_teleop_frame().await
            );
        }

        fusion_ws
            .send(Message::Text(
                capture_pose_packet(trip_id, session_id, seq).to_string(),
            ))
            .await?;
        seq += 1;

        if let Some(frame) = ws.latest_teleop_frame().await {
            if has_precomputed_targets(&frame) {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
}

fn has_precomputed_targets(frame: &serde_json::Value) -> bool {
    frame
        .get("hand_joint_layout")
        .and_then(|value| value.as_str())
        == Some("anatomical_joint_16")
        && frame
            .get("hand_target_layout")
            .and_then(|value| value.as_str())
            == Some("anatomical_target_16")
        && matches!(extract_f32_array(frame, "left_hand_target"), Some(values) if values.len() == 16)
        && matches!(extract_f32_array(frame, "right_hand_target"), Some(values) if values.len() == 16)
        && matches!(extract_f32_array(frame, "left_hand_joints"), Some(values) if values.len() == 16)
        && matches!(extract_f32_array(frame, "right_hand_joints"), Some(values) if values.len() == 16)
        && matches!(extract_f32_array(frame, "arm_q_target"), Some(values) if values.len() == 14)
        && extract_target_source(frame, "left_hand_target_source") == Some("anatomical_joints")
        && extract_target_source(frame, "right_hand_target_source") == Some("anatomical_joints")
        && extract_target_source(frame, "arm_q_target_source") == Some("body_anchor")
}

fn extract_f32_array(frame: &serde_json::Value, field: &str) -> Option<Vec<f32>> {
    let values = frame.get(field)?.as_array()?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let value = value.as_f64()? as f32;
        if !value.is_finite() {
            return None;
        }
        out.push(value);
    }
    Some(out)
}

fn extract_target_source<'a>(frame: &'a serde_json::Value, field: &str) -> Option<&'a str> {
    frame.get("target_debug")?.get(field)?.as_str()
}

fn capture_pose_packet(trip_id: &str, session_id: &str, seq: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "client-target-001",
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

#[tokio::test]
async fn stereo_pose_ingest_should_include_precomputed_targets_for_pico_layouts(
) -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-stereo-target-001";
    let session_id = "sess-stereo-target-001";

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

    support::enable_full_teleop_profile(&client, &server, trip_id, session_id).await?;

    let ws = WsHarness::connect(&server, trip_id, session_id).await?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    let mut seq = 0u64;
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!(
                "等待 stereo precomputed target 超时，latest={:?}",
                ws.latest_teleop_frame().await
            );
        }

        client
            .post(format!("{}/ingest/stereo_pose", server.http_base))
            .bearer_auth(&server.edge_token)
            .json(&serde_json::json!({
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "device_id": "stereo-test-001",
                "source_time_ns": 1_770_000_000_100_000_000u64 + seq,
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

        if let Some(frame) = ws.latest_teleop_frame().await {
            if has_precomputed_targets(&frame) {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
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
