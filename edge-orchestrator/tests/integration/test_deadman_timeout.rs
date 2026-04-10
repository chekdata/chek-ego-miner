use crate::support;

use std::time::Duration;

use support::ws_harness::WsHarness;

#[tokio::test]
async fn deadman_timeout_enters_fault() -> anyhow::Result<()> {
    let server = support::TestServer::spawn_with_env(vec![("DEADMAN_TIMEOUT_MS", "200")]).await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-deadman-001";
    let session_id = "sess-deadman-001";

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

    // 等待 deadman.link_ok=true
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待 deadman.link_ok 超时");
        }
        let state = client
            .get(format!("{}/control/state", server.http_base))
            .bearer_auth(&server.edge_token)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;
        if state
            .get("deadman")
            .and_then(|d| d.get("link_ok"))
            .and_then(|v| v.as_bool())
            == Some(true)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
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

    // 停止 keepalive，等待进入 fault（默认 200ms）
    ws.set_keepalive_enabled(false);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待进入 fault 超时");
        }
        let state = client
            .get(format!("{}/control/state", server.http_base))
            .bearer_auth(&server.edge_token)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;
        if state.get("state").and_then(|v| v.as_str()) == Some("fault") {
            assert_eq!(
                state.get("reason").and_then(|v| v.as_str()),
                Some("deadman_timeout")
            );
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    Ok(())
}
