#[path = "../support/mod.rs"]
mod support;

use std::time::Duration;

use support::ws_harness::WsHarness;

#[tokio::test]
async fn contract_control_arm_disarm() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-contract-001";
    let session_id = "sess-contract-001";

    // 未鉴权应拒绝（除 /health 外）
    let resp = client
        .get(format!("{}/control/state", server.http_base))
        .send()
        .await?;
    assert_eq!(resp.status(), 401);

    // 开会话（不等于 arm）
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

    // 启动 WS（bridge ready + keepalive）
    let _ws = WsHarness::connect(&server, trip_id, session_id).await?;

    // time/sync
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

    // 等待 preflight 就绪
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待 preflight 就绪超时");
        }

        let state = client
            .get(format!("{}/control/state", server.http_base))
            .bearer_auth(&server.edge_token)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let preflight = state.get("preflight").cloned().unwrap_or_default();
        let deadman = state.get("deadman").cloned().unwrap_or_default();
        let ok = preflight
            .get("unitree_bridge_ready")
            .and_then(|v| v.as_bool())
            == Some(true)
            && preflight.get("leap_bridge_ready").and_then(|v| v.as_bool()) == Some(true)
            && preflight.get("time_sync_ok").and_then(|v| v.as_bool()) == Some(true)
            && preflight.get("extrinsic_ok").and_then(|v| v.as_bool()) == Some(true)
            && preflight.get("lan_control_ok").and_then(|v| v.as_bool()) == Some(true)
            && deadman.get("link_ok").and_then(|v| v.as_bool()) == Some(true);

        if ok {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // arm
    let arm = client
        .post(format!("{}/control/arm", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "robot_type": "G1_29",
            "end_effector_type": "LEAP_V2",
            "operator_id": "op-test-001",
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(arm.get("state").and_then(|v| v.as_str()), Some("armed"));

    // disarm（幂等）
    let disarm = client
        .post(format!("{}/control/disarm", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "reason": "operator_disarm",
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        disarm.get("state").and_then(|v| v.as_str()),
        Some("disarmed")
    );

    Ok(())
}

#[tokio::test]
async fn contract_control_state_exposes_runtime_profile_and_feature_flags() -> anyhow::Result<()> {
    let server = support::TestServer::spawn_with_env(vec![
        (
            "EDGE_RUNTIME_PROFILE".to_string(),
            "capture_plus_vlm".to_string(),
        ),
        (
            "EDGE_CROWD_UPLOAD_POLICY_MODE".to_string(),
            "metadata_only".to_string(),
        ),
    ])
    .await?;
    let client = reqwest::Client::new();

    let state = client
        .get(format!("{}/control/state", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    assert_eq!(
        state
            .get("runtime_profile")
            .and_then(|value| value.as_str()),
        Some("capture_plus_vlm")
    );
    assert_eq!(
        state
            .get("upload_policy_mode")
            .and_then(|value| value.as_str()),
        Some("metadata_only")
    );
    assert_eq!(
        state
            .get("feature_flags")
            .and_then(|value| value.get("vlm_indexing_enabled"))
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        state
            .get("feature_flags")
            .and_then(|value| value.get("control_enabled"))
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    Ok(())
}

#[tokio::test]
async fn contract_safety_estop_release() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-safety-001";
    let session_id = "sess-safety-001";

    let estop_reason = "test_estop";
    client
        .post(format!("{}/safety/estop", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "reason": estop_reason,
        }))
        .send()
        .await?
        .error_for_status()?;

    let state_after_estop = client
        .get(format!("{}/control/state", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        state_after_estop.get("state").and_then(|v| v.as_str()),
        Some("fault")
    );
    assert_eq!(
        state_after_estop.get("reason").and_then(|v| v.as_str()),
        Some(estop_reason)
    );

    client
        .post(format!("{}/safety/release", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
        }))
        .send()
        .await?
        .error_for_status()?;

    let state_after_release = client
        .get(format!("{}/control/state", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        state_after_release.get("state").and_then(|v| v.as_str()),
        Some("disarmed")
    );
    assert_eq!(
        state_after_release.get("reason").and_then(|v| v.as_str()),
        Some("release")
    );

    Ok(())
}

#[tokio::test]
async fn contract_ws_routes_available() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let teleop_url = format!(
        "{}/stream/teleop?token={}",
        server.ws_base, server.edge_token
    );

    let (_fusion_ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;
    let (_teleop_ws, _) = tokio_tungstenite::connect_async(teleop_url).await?;
    Ok(())
}
