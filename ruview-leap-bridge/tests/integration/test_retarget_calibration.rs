use std::sync::Arc;

use tokio::time::Duration;

use ruview_leap_bridge::bridge::publisher::BridgeRunner;
use ruview_leap_bridge::config::Config;
use ruview_leap_bridge::leap::client::MockLeapClient;
use ruview_leap_bridge::protocol::version_guard::ProtocolVersionInfo;

fn make_cfg() -> Config {
    let mut left_joint_scale = vec![1.0; 16];
    left_joint_scale[13] = 1.2;
    let mut left_joint_offset = vec![0.0; 16];
    left_joint_offset[0] = 0.05;

    let mut right_joint_scale = vec![1.0; 16];
    right_joint_scale[10] = 0.9;
    let mut right_joint_offset = vec![0.0; 16];
    right_joint_offset[15] = -0.03;

    Config {
        protocol_pin_path: "protocol_pin.json".to_string(),
        edge_teleop_ws_url: None,
        edge_token: None,
        bridge_id: "leap-bridge-test".to_string(),
        publish_hz: 100,
        pairing_window_ms: 20,
        hold_timeout_ms: 200,
        freeze_timeout_ms: 200,
        keepalive_timeout_ms: 200,
        joint_min: vec![],
        joint_max: vec![],
        expected_joint_len: Some(16),
        left_joint_scale,
        left_joint_offset,
        right_joint_scale,
        right_joint_offset,
        max_temperature_c: 70.0,
        http_addr: "127.0.0.1:0".to_string(),
    }
}

fn proto() -> ProtocolVersionInfo {
    ProtocolVersionInfo {
        name: "teleop-protocol".to_string(),
        version: "1.12.0".to_string(),
        schema_sha256: "test".to_string(),
    }
}

fn teleop(edge_time_ns: u64) -> String {
    serde_json::json!({
        "schema_version": "teleop_frame_v1",
        "trip_id": "t",
        "session_id": "s",
        "robot_type": "G1_29",
        "end_effector_type": "LEAP_V2",
        "edge_time_ns": edge_time_ns,
        "control_state": "armed",
        "safety_state": "normal",
        "hand_joint_layout": "anatomical_joint_16",
        "hand_target_layout": "anatomical_target_16",
        "left_hand_target": [0.3, 0.2, 0.1, 0.05, 0.4, 0.6, 0.5, 0.35, 0.58, 0.44, 0.32, 0.55, 0.41, 0.28, 0.52, 0.39],
        "right_hand_target": [0.3, 0.2, 0.1, 0.05, 0.4, 0.6, 0.5, 0.35, 0.58, 0.44, 0.32, 0.55, 0.41, 0.28, 0.52, 0.39]
    })
    .to_string()
}

#[tokio::test(start_paused = true)]
async fn integration_retarget_calibration_should_adjust_published_command() {
    let cfg = make_cfg();
    let mock = Arc::new(MockLeapClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle.inject_teleop_json(teleop(0)).await.unwrap();
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;

    let published = mock.take_published();
    assert!(!published.is_empty(), "should publish calibrated command");
    let frame = published.last().expect("published frame");
    assert!((frame.left_cmd[0] - (-0.25)).abs() < 1.0e-6);
    assert!((frame.left_cmd[13] - 0.33600003).abs() < 1.0e-6);
    assert!((frame.right_cmd[10] - 0.288).abs() < 1.0e-6);
    assert!((frame.right_cmd[15] - 0.36).abs() < 1.0e-6);

    let health = handle.health_snapshot();
    assert!(health.left_retarget_calibrated);
    assert!(health.right_retarget_calibrated);
    assert_eq!(health.left_retarget_joint_count, 16);
    assert_eq!(health.right_retarget_joint_count, 16);
    assert_eq!(health.left_retarget_non_default_scale_count, 1);
    assert_eq!(health.right_retarget_non_default_scale_count, 1);
    assert_eq!(health.left_retarget_non_zero_offset_count, 1);
    assert_eq!(health.right_retarget_non_zero_offset_count, 1);

    handle.shutdown();
    let _ = task.await;
}
