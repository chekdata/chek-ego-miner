use std::sync::Arc;

use tokio::time::Duration;

use ruview_unitree_bridge::bridge::publisher::BridgeRunner;
use ruview_unitree_bridge::config::Config;
use ruview_unitree_bridge::dds::unitree_client::MockUnitreeClient;
use ruview_unitree_bridge::protocol::version_guard::ProtocolVersionInfo;
use ruview_unitree_bridge::reason;

fn make_cfg() -> Config {
    Config {
        protocol_pin_path: "protocol_pin.json".to_string(),
        edge_teleop_ws_url: None,
        edge_token: None,
        bridge_id: "unitree-bridge-test".to_string(),
        publish_hz: 100,
        keepalive_timeout_ms: 200,
        expected_arm_joint_len: Some(3),
        expected_dex_joint_len: None,
        arm_joint_min: vec![],
        arm_joint_max: vec![],
        dex_joint_min: vec![],
        dex_joint_max: vec![],
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
        "operator_frame": "op",
        "robot_base_frame": "base",
        "extrinsic_version": "v1",
        "control_state": "armed",
        "left_wrist_pose": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
        "right_wrist_pose": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
        "quality": { "source_mode": "fused", "fused_conf": 1.0, "vision_conf": 0.0, "csi_conf": 0.0 },
        "safety_state": "normal",
        "arm_q_target": [0.0, 0.1, 0.2],
        "arm_tauff_target": [0.0, 0.0, 0.0]
    })
    .to_string()
}

#[tokio::test(start_paused = true)]
async fn integration_dds_publish_failure_flips_unready() {
    let cfg = make_cfg();
    let mock = Arc::new(MockUnitreeClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle.inject_teleop_json(teleop(0)).await.unwrap();
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    assert!(mock.published_count() > 0);

    mock.set_fail_publish(true);
    tokio::time::advance(Duration::from_millis(20)).await;
    tokio::task::yield_now().await;
    let snap = handle.health_snapshot();
    assert!(!snap.is_ready);
    assert_eq!(snap.fault_code, reason::REASON_DDS_PUBLISH_FAILED);

    handle.shutdown();
    let _ = task.await;
}
