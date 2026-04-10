use std::sync::Arc;

use tokio::time::Duration;

use ruview_leap_bridge::bridge::publisher::BridgeRunner;
use ruview_leap_bridge::config::Config;
use ruview_leap_bridge::leap::client::MockLeapClient;
use ruview_leap_bridge::protocol::version_guard::ProtocolVersionInfo;

fn make_cfg() -> Config {
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
        left_joint_scale: vec![],
        left_joint_offset: vec![],
        right_joint_scale: vec![],
        right_joint_offset: vec![],
        max_temperature_c: 70.0,
        http_addr: "127.0.0.1:0".to_string(),
    }
}

fn proto() -> ProtocolVersionInfo {
    ProtocolVersionInfo {
        name: "teleop-protocol".to_string(),
        version: "1.0.0".to_string(),
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
        "left_hand_target": [0.12, 0.18, 0.22, 0.14, 0.24, 0.58, 0.31, 0.20, 0.55, 0.30, 0.18, 0.52, 0.28, 0.16, 0.48, 0.26],
        "right_hand_target": [0.10, 0.17, 0.20, 0.13, 0.22, 0.56, 0.30, 0.19, 0.53, 0.29, 0.17, 0.50, 0.27, 0.15, 0.46, 0.25]
    })
    .to_string()
}

#[tokio::test(start_paused = true)]
async fn integration_keepalive_timeout_blocks_publish() {
    let cfg = make_cfg();
    let mock = Arc::new(MockLeapClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle.inject_teleop_json(teleop(0)).await.unwrap();

    // 先推进一段时间，确保至少发生一次 publish
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    let before = mock.published_count();
    assert!(before > 0, "should publish before keepalive timeout");

    // 不再注入帧，推进到 keepalive 超时后，publish 不再增长
    tokio::time::advance(Duration::from_millis(250)).await;
    tokio::task::yield_now().await;
    let after = mock.published_count();
    assert_eq!(after, before, "publish should stop after keepalive timeout");

    handle.shutdown();
    let _ = task.await;
}
