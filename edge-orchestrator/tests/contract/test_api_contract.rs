#[path = "../support/mod.rs"]
mod support;

use std::time::Duration;
use std::{fs, path::Path};

use support::ws_harness::WsHarness;

#[tokio::test]
async fn contract_edge_live_preview_alias_matches_root() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let root = client
        .get(format!("{}/live-preview.json", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    let alias = client
        .get(format!("{}/edge/live-preview.json", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    assert_eq!(alias, root);
    assert!(alias.get("vlm_summary").is_some());

    Ok(())
}

#[tokio::test]
async fn contract_storage_status_and_sessions_compatibility() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": "trip-storage-active-001",
            "session_id": "sess-storage-active-001",
            "device_id": "device-storage-active-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-active-001",
        "sess-storage-active-001",
        2_048,
        "pass",
        true,
        "queued",
    )?;
    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-held-001",
        "sess-storage-held-001",
        4_096,
        "retry_recommended",
        true,
        "queued",
    )?;
    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-rolling-001",
        "sess-storage-rolling-001",
        8_192,
        "pass",
        true,
        "acked",
    )?;
    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-rolling-002",
        "sess-storage-rolling-002",
        16_384,
        "pass",
        true,
        "acked",
    )?;

    client
        .post(format!(
            "{}/edge/storage/consumption-receipt",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "session_id": "sess-storage-held-001",
            "signal_kind": "manual_hold",
            "note": "engineering_manual_hold",
        }))
        .send()
        .await?
        .error_for_status()?;

    let status = client
        .get(format!("{}/storage/status", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    let status_alias = client
        .get(format!("{}/edge/storage/status", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        status
            .get("session_summary")
            .and_then(|value| value.get("active_session_id"))
            .and_then(|value| value.as_str()),
        Some("sess-storage-active-001")
    );
    assert_eq!(
        status
            .get("rolling_pool")
            .and_then(|value| value.get("session_count"))
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        status
            .get("rolling_pool")
            .and_then(|value| value.get("evictable_bytes"))
            .and_then(|value| value.as_u64()),
        Some(24_576)
    );
    assert_eq!(
        status
            .get("protected_pool")
            .and_then(|value| value.get("session_count"))
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        status_alias
            .get("session_summary")
            .and_then(|value| value.get("active_session_id"))
            .and_then(|value| value.as_str()),
        Some("sess-storage-active-001")
    );
    assert_eq!(
        status_alias
            .get("rolling_pool")
            .and_then(|value| value.get("session_count"))
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        status_alias
            .get("rolling_pool")
            .and_then(|value| value.get("evictable_bytes"))
            .and_then(|value| value.as_u64()),
        Some(24_576)
    );
    assert_eq!(
        status_alias
            .get("protected_pool")
            .and_then(|value| value.get("session_count"))
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    let blockers = status
        .get("cleanup")
        .and_then(|value| value.get("current_blockers"))
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let alias_blockers = status_alias
        .get("cleanup")
        .and_then(|value| value.get("current_blockers"))
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(blockers
        .iter()
        .any(|value| { value.as_str() == Some("active_session:sess-storage-active-001") }));
    assert!(blockers
        .iter()
        .any(|value| { value.as_str() == Some("manual_hold:sess-storage-held-001") }));
    assert_eq!(alias_blockers, blockers);

    let sessions = client
        .get(format!("{}/edge/storage/sessions", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    let items = sessions
        .get("sessions")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let active = session_item(&items, "sess-storage-active-001").expect("active session item");
    let held = session_item(&items, "sess-storage-held-001").expect("held session item");
    let rolling = session_item(&items, "sess-storage-rolling-001").expect("rolling session item");
    let rolling_extra =
        session_item(&items, "sess-storage-rolling-002").expect("extra rolling session item");
    assert_eq!(
        active.get("storage_pool").and_then(|value| value.as_str()),
        Some("protected")
    );
    assert_eq!(
        held.get("last_cleanup_skipped_reason")
            .and_then(|value| value.as_str()),
        Some("manual_hold")
    );
    assert_eq!(
        rolling.get("storage_pool").and_then(|value| value.as_str()),
        Some("rolling")
    );
    assert_eq!(
        rolling
            .get("upload_queue_status")
            .and_then(|value| value.as_str()),
        Some("acked")
    );
    assert_eq!(
        rolling_extra
            .get("upload_queue_status")
            .and_then(|value| value.as_str()),
        Some("acked")
    );

    Ok(())
}

#[tokio::test]
async fn contract_storage_cleanup_apply_reclaims_only_rolling_sessions() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": "trip-storage-active-apply-001",
            "session_id": "sess-storage-active-apply-001",
            "device_id": "device-storage-active-apply-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-active-apply-001",
        "sess-storage-active-apply-001",
        1_024,
        "pass",
        true,
        "queued",
    )?;
    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-held-apply-001",
        "sess-storage-held-apply-001",
        2_048,
        "pass",
        true,
        "queued",
    )?;
    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-rolling-apply-001",
        "sess-storage-rolling-apply-001",
        4_096,
        "pass",
        true,
        "acked",
    )?;
    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-queued-apply-001",
        "sess-storage-queued-apply-001",
        8_192,
        "pass",
        true,
        "queued",
    )?;

    client
        .post(format!(
            "{}/edge/storage/consumption-receipt",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "session_id": "sess-storage-held-apply-001",
            "signal_kind": "manual_hold",
            "note": "engineering_manual_hold",
        }))
        .send()
        .await?
        .error_for_status()?;

    let dry_run = client
        .post(format!("{}/edge/storage/sweeps/dry-run", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({}))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        dry_run
            .get("selected_session_count")
            .and_then(|value| value.as_u64()),
        Some(1)
    );
    assert_eq!(
        dry_run
            .get("selected_bytes")
            .and_then(|value| value.as_u64()),
        Some(4_096)
    );

    let apply = client
        .post(format!("{}/edge/storage/sweeps/apply", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({}))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        apply
            .get("selected_session_count")
            .and_then(|value| value.as_u64()),
        Some(1)
    );
    assert_eq!(
        apply
            .get("applied_reclaimed_bytes")
            .and_then(|value| value.as_u64()),
        Some(4_096)
    );

    assert!(server
        .data_dir
        .join("session")
        .join("sess-storage-active-apply-001")
        .exists());
    assert!(server
        .data_dir
        .join("session")
        .join("sess-storage-held-apply-001")
        .exists());
    assert!(!server
        .data_dir
        .join("session")
        .join("sess-storage-rolling-apply-001")
        .exists());
    assert!(server
        .data_dir
        .join("session")
        .join("sess-storage-queued-apply-001")
        .exists());

    let status = client
        .get(format!("{}/storage/status", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        status
            .get("cleanup")
            .and_then(|value| value.get("last_reclaimed_bytes"))
            .and_then(|value| value.as_u64()),
        Some(4_096)
    );
    assert_eq!(
        status
            .get("rolling_pool")
            .and_then(|value| value.get("session_count"))
            .and_then(|value| value.as_u64()),
        Some(1)
    );

    Ok(())
}

#[tokio::test]
async fn contract_storage_cleanup_apply_keeps_session_started_after_dry_run() -> anyhow::Result<()>
{
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();
    let session_id = "sess-storage-race-apply-001";

    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-race-apply-001",
        session_id,
        4_096,
        "pass",
        true,
        "acked",
    )?;

    let dry_run = client
        .post(format!("{}/edge/storage/sweeps/dry-run", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({}))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        dry_run
            .get("selected_session_ids")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .and_then(|value| value.as_str()),
        Some(session_id)
    );

    let storage_state_dir = server.data_dir.join("storage");
    fs::create_dir_all(&storage_state_dir)?;
    fs::write(
        storage_state_dir.join("cleanup_state.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "last_run_at": 123,
            "last_reclaimed_bytes": 999,
            "last_error": "previous delete failure",
        }))?,
    )?;

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": "trip-storage-race-apply-001",
            "session_id": session_id,
            "device_id": "device-storage-race-apply-001",
        }))
        .send()
        .await?
        .error_for_status()?;

    let apply = client
        .post(format!("{}/edge/storage/sweeps/apply", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({}))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        apply
            .get("applied_session_count")
            .and_then(|value| value.as_u64()),
        Some(0)
    );
    assert!(server.data_dir.join("session").join(session_id).exists());

    let status = client
        .get(format!("{}/storage/status", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        status
            .get("cleanup")
            .and_then(|value| value.get("last_reclaimed_bytes"))
            .and_then(|value| value.as_u64()),
        Some(999)
    );
    assert_eq!(
        status
            .get("cleanup")
            .and_then(|value| value.get("last_error"))
            .and_then(|value| value.as_str()),
        Some("previous delete failure")
    );

    Ok(())
}

#[tokio::test]
async fn contract_storage_cleanup_apply_keeps_session_held_after_dry_run() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();
    let session_id = "sess-storage-receipt-race-001";

    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-receipt-race-001",
        session_id,
        4_096,
        "pass",
        true,
        "acked",
    )?;

    let dry_run = client
        .post(format!("{}/edge/storage/sweeps/dry-run", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({}))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        dry_run
            .get("selected_session_ids")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .and_then(|value| value.as_str()),
        Some(session_id)
    );

    client
        .post(format!(
            "{}/edge/storage/consumption-receipt",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "session_id": session_id,
            "signal_kind": "manual_hold",
            "note": "operator protected after dry run",
        }))
        .send()
        .await?
        .error_for_status()?;

    let apply = client
        .post(format!("{}/edge/storage/sweeps/apply", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({}))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        apply
            .get("applied_session_count")
            .and_then(|value| value.as_u64()),
        Some(0)
    );
    assert!(server.data_dir.join("session").join(session_id).exists());

    Ok(())
}

#[tokio::test]
async fn contract_storage_compat_accepts_legacy_consumption_receipts() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-legacy-held-001",
        "sess-storage-legacy-held-001",
        2_048,
        "pass",
        true,
        "queued",
    )?;
    seed_storage_session_fixture(
        &server.data_dir,
        "trip-storage-legacy-release-001",
        "sess-storage-legacy-release-001",
        4_096,
        "pass",
        true,
        "queued",
    )?;
    seed_legacy_consumption_receipts(&server.data_dir)?;

    let status = client
        .get(format!("{}/storage/status", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        status
            .get("protected_pool")
            .and_then(|value| value.get("session_count"))
            .and_then(|value| value.as_u64()),
        Some(1)
    );
    assert_eq!(
        status
            .get("rolling_pool")
            .and_then(|value| value.get("session_count"))
            .and_then(|value| value.as_u64()),
        Some(1)
    );

    let sessions = client
        .get(format!("{}/edge/storage/sessions", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    let items = sessions
        .get("sessions")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let held = session_item(&items, "sess-storage-legacy-held-001").expect("legacy held session");
    let released =
        session_item(&items, "sess-storage-legacy-release-001").expect("legacy released session");
    assert_eq!(
        held.get("storage_pool").and_then(|value| value.as_str()),
        Some("protected")
    );
    assert_eq!(
        released
            .get("storage_pool")
            .and_then(|value| value.as_str()),
        Some("rolling")
    );

    Ok(())
}

#[tokio::test]
async fn contract_storage_compat_rejects_traversal_session_ids() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let response = client
        .post(format!(
            "{}/edge/storage/consumption-receipt",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "session_id": "../outside",
            "signal_kind": "manual_hold",
            "note": "must-not-write-outside-session-root",
        }))
        .send()
        .await?;

    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(!server
        .data_dir
        .join("storage")
        .join("consumption_receipts.jsonl")
        .exists());
    assert!(!server.data_dir.join("outside").exists());

    Ok(())
}

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

fn seed_storage_session_fixture(
    data_dir: &Path,
    trip_id: &str,
    session_id: &str,
    byte_size: usize,
    qa_status: &str,
    ready_for_upload: bool,
    queue_status: &str,
) -> anyhow::Result<()> {
    let session_dir = data_dir.join("session").join(session_id);
    fs::create_dir_all(session_dir.join("raw"))?;
    fs::create_dir_all(session_dir.join("qa"))?;
    fs::create_dir_all(session_dir.join("upload"))?;

    fs::write(
        session_dir.join("raw").join("payload.bin"),
        vec![b'x'; byte_size],
    )?;
    fs::write(
        session_dir.join("qa").join("local_quality_report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "status": qa_status,
        }))?,
    )?;
    fs::write(
        session_dir.join("upload").join("upload_manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "ready_for_upload": ready_for_upload,
            "artifacts": [{
                "id": "payload",
                "relpath": "raw/payload.bin",
                "required": true,
                "exists": true,
                "byte_size": byte_size,
            }],
        }))?,
    )?;
    fs::write(
        session_dir.join("upload").join("upload_queue.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "ready_for_upload": ready_for_upload,
            "entries": [{
                "status": queue_status,
            }],
        }))?,
    )?;
    Ok(())
}

fn session_item<'a>(
    items: &'a [serde_json::Value],
    session_id: &str,
) -> Option<&'a serde_json::Value> {
    items
        .iter()
        .find(|item| item.get("session_id").and_then(|value| value.as_str()) == Some(session_id))
}

fn seed_legacy_consumption_receipts(data_dir: &Path) -> anyhow::Result<()> {
    let storage_dir = data_dir.join("storage");
    fs::create_dir_all(&storage_dir)?;
    fs::write(
        storage_dir.join("consumption_receipts.jsonl"),
        [
            serde_json::json!({
                "type": "storage_consumption_receipt",
                "schema_version": "1.0.0",
                "generated_unix_ms": 111,
                "session_id": "sess-storage-legacy-held-001",
                "signal_kind": "manual_hold",
                "consumed_at": 111,
                "note": "legacy-hold",
            })
            .to_string(),
            serde_json::json!({
                "type": "storage_consumption_receipt",
                "schema_version": "1.0.0",
                "generated_unix_ms": 222,
                "session_id": "sess-storage-legacy-release-001",
                "signal_kind": "manual_release",
                "consumed_at": 222,
                "note": "legacy-release",
            })
            .to_string(),
        ]
        .join("\n")
            + "\n",
    )?;
    Ok(())
}
