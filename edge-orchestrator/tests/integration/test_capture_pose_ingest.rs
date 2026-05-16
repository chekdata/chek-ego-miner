use std::time::Duration;

use crate::support;

#[tokio::test]
async fn capture_pose_http_ingest_should_record_phone_pose_artifacts() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-capture-pose-http-001";
    let session_id = "sess-capture-pose-http-001";
    let device_id = "iphone-capture-pose-http-001";

    let response = client
        .post(format!("{}/ingest/capture_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&capture_pose_packet(trip_id, session_id, device_id))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    assert_eq!(
        response.get("ok").and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        response.get("accepted").and_then(|value| value.as_bool()),
        Some(true)
    );

    let kpts_path = server
        .data_dir
        .join("session")
        .join(session_id)
        .join("raw")
        .join("iphone")
        .join("wide")
        .join("kpts_depth.jsonl");
    let calibration_path = server
        .data_dir
        .join("session")
        .join(session_id)
        .join("calibration")
        .join("iphone_capture.json");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() <= deadline {
        if tokio::fs::metadata(&kpts_path).await.is_ok()
            && tokio::fs::metadata(&calibration_path).await.is_ok()
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let kpts = tokio::fs::read_to_string(&kpts_path).await?;
    assert!(kpts.contains("\"type\":\"capture_pose_packet\""));
    assert!(kpts.contains(device_id));

    let calibration = tokio::fs::read_to_string(&calibration_path).await?;
    assert!(calibration.contains("\"fx_px\""));

    Ok(())
}

#[tokio::test]
async fn capture_pose_http_ingest_should_honor_phone_ingest_runtime_gate() -> anyhow::Result<()> {
    let server = support::TestServer::spawn_with_env([("EDGE_PHONE_INGEST_ENABLED", "0")]).await?;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{}/ingest/capture_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&capture_pose_packet(
            "trip-capture-pose-disabled-001",
            "sess-capture-pose-disabled-001",
            "iphone-capture-pose-disabled-001",
        ))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    assert_eq!(
        response.get("ok").and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        response
            .get("error")
            .and_then(|value| value.get("code"))
            .and_then(|value| value.as_str()),
        Some("phone_ingest_disabled")
    );

    Ok(())
}

fn capture_pose_packet(trip_id: &str, session_id: &str, device_id: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": device_id,
        "device_class": "B",
        "platform": "ios",
        "source_time_ns": 123_456_789u64,
        "frame_id": 7u64,
        "camera": {
            "mode": "teleop_phone_back_wide_depth",
            "has_depth": true,
            "image_w": 1280,
            "image_h": 720,
            "calibration": {
                "fx_px": 930.4,
                "fy_px": 931.0,
                "cx_px": 640.0,
                "cy_px": 360.0,
                "reference_image_w": 1280,
                "reference_image_h": 720
            }
        },
        "device_pose": {
            "position_m": [0.1, 0.2, 0.3],
            "rotation_deg": [1.0, 2.0, 3.0],
            "target_space": "operator_frame",
            "source": "ios_device_attitude"
        },
        "imu": {
            "accel": [0.01, 0.02, 0.03],
            "gyro": [0.11, 0.12, 0.13]
        },
        "body_layout": "coco_body_17",
        "hand_layout": "mediapipe_hand_21",
        "body_kpts_3d": [[0.0, 0.2, 0.8], [0.0, 0.1, 0.8]],
        "hand_kpts_3d": [[0.0, 0.0, 0.0]],
        "confidence": {
            "body": 0.7,
            "hand": 0.1
        }
    })
}
