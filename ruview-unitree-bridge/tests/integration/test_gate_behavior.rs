use std::f32::consts::PI;
use std::sync::Arc;

use tokio::time::Duration;

use ruview_unitree_bridge::bridge::publisher::BridgeRunner;
use ruview_unitree_bridge::config::Config;
use ruview_unitree_bridge::dds::unitree_client::{MockUnitreeClient, PublishedMessage};
use ruview_unitree_bridge::protocol::version_guard::ProtocolVersionInfo;

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

fn teleop(edge_time_ns: u64, control_state: &str, safety_state: &str) -> String {
    teleop_with_arm_q(edge_time_ns, control_state, safety_state, [0.0, 0.1, 0.2])
}

fn teleop_with_arm_q(
    edge_time_ns: u64,
    control_state: &str,
    safety_state: &str,
    arm_q_target: [f32; 3],
) -> String {
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
        "control_state": control_state,
        "left_wrist_pose": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
        "right_wrist_pose": { "pos": [0.0,0.0,0.0], "quat": [1.0,0.0,0.0,0.0] },
        "quality": { "source_mode": "fused", "fused_conf": 1.0, "vision_conf": 0.0, "csi_conf": 0.0 },
        "safety_state": safety_state,
        "arm_q_target": arm_q_target,
        "arm_tauff_target": [0.0, 0.0, 0.0]
    })
    .to_string()
}

fn take_last_arm_q(published: Vec<PublishedMessage>) -> Vec<f32> {
    published
        .into_iter()
        .rev()
        .find_map(|m| match m {
            PublishedMessage::Arm(cmd) => Some(cmd.q),
            _ => None,
        })
        .unwrap_or_default()
}

#[tokio::test(start_paused = true)]
async fn integration_disarmed_should_not_publish() {
    let cfg = make_cfg();
    let mock = Arc::new(MockUnitreeClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle
        .inject_teleop_json(teleop(0, "disarmed", "normal"))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    assert_eq!(mock.published_count(), 0);

    handle.shutdown();
    let _ = task.await;
}

#[tokio::test(start_paused = true)]
async fn integration_armed_should_publish_then_timeout_blocks() {
    let cfg = make_cfg();
    let mock = Arc::new(MockUnitreeClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle
        .inject_teleop_json(teleop(0, "armed", "normal"))
        .await
        .unwrap();

    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    let before = mock.published_count();
    assert!(before > 0);

    tokio::time::advance(Duration::from_millis(250)).await;
    tokio::task::yield_now().await;
    let after = mock.published_count();
    assert_eq!(after, before, "keepalive timeout 后不应继续增长");

    handle.shutdown();
    let _ = task.await;
}

#[tokio::test(start_paused = true)]
async fn integration_estop_should_block() {
    let cfg = make_cfg();
    let mock = Arc::new(MockUnitreeClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle
        .inject_teleop_json(teleop(0, "armed", "estop"))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    assert_eq!(mock.published_count(), 0);

    handle.shutdown();
    let _ = task.await;
}

#[tokio::test(start_paused = true)]
async fn integration_freeze_should_hold_last_command() {
    let cfg = make_cfg();
    let mock = Arc::new(MockUnitreeClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle
        .inject_teleop_json(teleop_with_arm_q(0, "armed", "normal", [0.0, 0.1, 0.2]))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    let prev = take_last_arm_q(mock.take_published());
    assert_eq!(prev, vec![0.0, 0.1, 0.2]);

    // freeze 帧携带新的 arm_q_target，但不允许更新目标，应继续输出上一帧的 q
    handle
        .inject_teleop_json(teleop_with_arm_q(
            10_000_000,
            "armed",
            "freeze",
            [9.0, 9.0, 9.0],
        ))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    let now_q = take_last_arm_q(mock.take_published());
    assert_eq!(now_q, prev);

    handle.shutdown();
    let _ = task.await;
}

#[tokio::test(start_paused = true)]
async fn integration_limit_should_clamp_joint_targets() {
    let cfg = make_cfg();
    let mock = Arc::new(MockUnitreeClient::new());
    let (runner, handle) = BridgeRunner::new_manual(cfg, proto(), mock.clone());
    let task = tokio::spawn(async move { runner.run().await });

    handle
        .inject_teleop_json(teleop_with_arm_q(0, "armed", "limit", [100.0, 0.1, -100.0]))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_millis(30)).await;
    tokio::task::yield_now().await;
    let q = take_last_arm_q(mock.take_published());
    assert_eq!(q, vec![PI, 0.1, -PI]);

    handle.shutdown();
    let _ = task.await;
}
