use std::time::Duration;

use futures_util::StreamExt;

use crate::support;
use support::ws_harness::WsHarness;

#[tokio::test]
async fn wifi_pose_ingest_should_surface_prior_debug_without_driving_body_anchor_targets(
) -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-wifi-pose-001";
    let session_id = "sess-wifi-pose-001";
    let device_id = "wifi-pose-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
        }))
        .send()
        .await?
        .error_for_status()?;

    support::enable_full_teleop_profile(&client, &server, trip_id, session_id).await?;

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "source_kind": "wifi_pose",
            "clock_domain": "wifi_uptime_ns",
            "clock_offset_ns": 0,
            "rtt_ns": 1_100_000,
            "sample_count": 6
        }))
        .send()
        .await?
        .error_for_status()?;

    let teleop_ws = WsHarness::connect(&server, trip_id, session_id).await?;
    let fusion_url = format!(
        "{}/stream/fusion?token={}&debug_views=operator",
        server.ws_base, server.edge_token
    );
    let (mut fusion_ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!(
                "等待 wifi pose 先验链路超时，latest teleop={:?}",
                teleop_ws.latest_teleop_frame().await
            );
        }

        client
            .post(format!("{}/ingest/wifi_pose", server.http_base))
            .bearer_auth(&server.edge_token)
            .json(&wifi_pose_packet(trip_id, session_id, device_id))
            .send()
            .await?
            .error_for_status()?;

        let teleop_ready = teleop_ws
            .latest_teleop_frame()
            .await
            .is_some_and(|frame| has_wifi_prior_without_body_anchor_targets(&frame));
        let fusion_ready = wait_for_wifi_fusion_debug(&mut fusion_ws).await?;
        if teleop_ready && fusion_ready {
            return Ok(());
        }

        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_wifi_fusion_debug(
    fusion_ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> anyhow::Result<bool> {
    let timeout = tokio::time::sleep(Duration::from_millis(250));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            _ = &mut timeout => return Ok(false),
            msg = fusion_ws.next() => {
                let Some(msg) = msg else {
                    anyhow::bail!("fusion ws closed");
                };
                let msg = msg?;
                let tokio_tungstenite::tungstenite::Message::Text(text) = msg else {
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
                let Some(wifi_pose) = operator_debug.get("wifi_pose") else {
                    continue;
                };
                let Some(fused_pose) = operator_debug.get("fused_pose") else {
                    continue;
                };
                if wifi_pose.get("available").and_then(|value| value.as_bool()) != Some(true) {
                    continue;
                }
                return Ok(
                    fused_pose.get("selected_source").and_then(|value| value.as_str()) == Some("none")
                        && fused_pose.get("body_source").and_then(|value| value.as_str()) == Some("none")
                        && fused_pose.get("hand_source").and_then(|value| value.as_str()) == Some("none")
                        && fused_pose.get("wifi_body_joint_count").and_then(|value| value.as_u64()) == Some(0)
                        && fused_pose.get("wifi_hand_point_count").and_then(|value| value.as_u64()) == Some(0)
                        && wifi_pose.get("raw_body_count").and_then(|value| value.as_u64()) == Some(24)
                );
            }
        }
    }
}

fn has_wifi_prior_without_body_anchor_targets(frame: &serde_json::Value) -> bool {
    frame.get("arm_q_target").is_none()
        && missing_or_null(frame.get("left_hand_target"))
        && missing_or_null(frame.get("right_hand_target"))
        && frame
            .get("target_debug")
            .and_then(|value| value.get("arm_q_target_source"))
            .is_none()
        && frame
            .get("target_debug")
            .and_then(|value| value.get("left_hand_target_source"))
            .is_none()
        && frame
            .get("target_debug")
            .and_then(|value| value.get("right_hand_target_source"))
            .is_none()
}

fn missing_or_null(value: Option<&serde_json::Value>) -> bool {
    value.is_none() || value.is_some_and(serde_json::Value::is_null)
}

fn wifi_pose_packet(trip_id: &str, session_id: &str, device_id: &str) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": device_id,
        "operator_track_id": "wifi-operator-main",
        "source_time_ns": 1_770_000_200_000_000_000u64,
        "body_layout": "pico_body_24",
        "body_space": "operator_frame",
        "body_kpts_3d": build_pico_body_points(),
        "body_confidence": 0.78,
        "source_label": "wifi_densepose",
        "person_id": 0,
        "total_persons": 1,
        "raw_body_layout": "pico_body_24",
        "raw_body_space": "wifi_pose_frame",
        "raw_body_kpts_3d": build_pico_body_points(),
        "calibration": {
            "sensor_frame": "wifi_pose_frame",
            "operator_frame": "operator_frame",
            "extrinsic_version": "wifi-ext-001",
            "extrinsic_translation_m": [0.12, 0.0, 0.05],
            "extrinsic_rotation_quat": [0.0, 0.0, 0.0, 1.0],
            "notes": "wifi pose integration test"
        }
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
