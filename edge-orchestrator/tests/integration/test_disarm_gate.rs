use crate::support;

use std::time::Duration;

use support::ws_harness::WsHarness;

#[tokio::test]
async fn disarm_stops_teleop_output() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-disarm-001";
    let session_id = "sess-disarm-001";

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

    // arm（仅要求 link_ok，不要求 pressed）
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

    // Deadman 按住后应输出 armed
    ws.set_deadman_pressed(true);
    ws.wait_teleop_control_state("armed", Duration::from_secs(3))
        .await?;

    // disarm 后应回到 disarmed
    client
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
        .error_for_status()?;

    ws.wait_teleop_control_state("disarmed", Duration::from_secs(3))
        .await?;

    Ok(())
}
