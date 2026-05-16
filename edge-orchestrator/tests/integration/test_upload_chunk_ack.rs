use crate::support;

use axum::extract::State;
use axum::routing::patch;
use axum::{Json, Router};
use ring::digest;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::multipart;

use support::ws_harness::WsHarness;

fn sha256_hex(value: &str) -> String {
    let digest = digest::digest(&digest::SHA256, value.as_bytes());
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{:02x}", *byte))
        .collect()
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

async fn assert_session_identity_artifacts(
    base_dir: &std::path::Path,
    device_id: &str,
    login_identity: &str,
    device_name: &str,
    profile_id: &str,
) -> anyhow::Result<()> {
    for relpath in [
        "manifest.json",
        "demo_capture_bundle.json",
        "upload/upload_manifest.json",
    ] {
        let path = base_dir.join(relpath);
        support::wait_for_file(path.clone()).await?;
        let value: serde_json::Value = serde_json::from_slice(&tokio::fs::read(path).await?)?;
        let session_context = value
            .get("session_context")
            .and_then(|value| value.as_object())
            .ok_or_else(|| anyhow::anyhow!("missing session_context in {relpath}"))?;
        assert_eq!(
            session_context
                .get("capture_device_id")
                .and_then(|value| value.as_str()),
            Some(device_id),
            "{relpath} capture_device_id"
        );
        assert_eq!(
            session_context
                .get("login_identity")
                .and_then(|value| value.as_str()),
            Some(login_identity),
            "{relpath} login_identity"
        );
        assert_eq!(
            session_context
                .get("device_name")
                .and_then(|value| value.as_str()),
            Some(device_name),
            "{relpath} device_name"
        );
        assert_eq!(
            session_context
                .get("pairing_profile_id")
                .and_then(|value| value.as_str()),
            Some(profile_id),
            "{relpath} pairing_profile_id"
        );
        assert_eq!(
            session_context
                .get("upload_auth_kind")
                .and_then(|value| value.as_str()),
            Some("scoped_upload_token"),
            "{relpath} upload_auth_kind"
        );
    }
    Ok(())
}

#[tokio::test]
async fn upload_chunk_triggers_ack_and_cleaned_flow() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-upload-001";
    let session_id = "sess-upload-001";
    let chunk_index: u32 = 0;

    // 订阅 /stream/fusion 以接收 chunk_ack_packet。
    let ws = WsHarness::connect(&server, trip_id, session_id).await?;

    // 1) 上传 CSV（此时不应 ack —— 默认要求 csv+det 都齐才 ack）。
    {
        let meta = serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-test-001",
            "chunk_index": chunk_index,
            "file_type": "csv",
            "file_name": "chunk_0.csv"
        });
        let form = multipart::Form::new()
            .part(
                "file",
                multipart::Part::bytes(b"csv-bytes".to_vec()).file_name("chunk_0.csv"),
            )
            .text("metadata", meta.to_string());

        client
            .post(format!("{}/common_task/upload_chunk", server.http_base))
            .bearer_auth(&server.edge_token)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;
    }

    // 给一点时间，确保没有“提前 ack”。
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 2) 上传 DET（此时满足 csv+det 完整条件，应收到一次 chunk_ack(stored)）。
    {
        let meta = serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-test-001",
            "chunk_index": chunk_index,
            "file_type": "det",
            "file_name": "chunk_0.det"
        });
        let form = multipart::Form::new()
            .part(
                "file",
                multipart::Part::bytes(b"det-bytes".to_vec()).file_name("chunk_0.det"),
            )
            .text("metadata", meta.to_string());

        client
            .post(format!("{}/common_task/upload_chunk", server.http_base))
            .bearer_auth(&server.edge_token)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;
    }

    ws.wait_chunk_ack_stored(chunk_index, Duration::from_secs(3))
        .await?;

    // 3) Capture Client 回执 cleaned（幂等）。
    client
        .post(format!("{}/chunk/cleaned", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "chunk_index": chunk_index,
            "device_id": "client-test-001",
            "source_time_ns": 0,
        }))
        .send()
        .await?
        .error_for_status()?;

    let state: serde_json::Value = support::authed_get_json(
        &client,
        &server,
        &format!(
            "/chunk/state?trip_id={trip_id}&session_id={session_id}&chunk_index={chunk_index}"
        ),
    )
    .await?;

    assert_eq!(state.get("state").and_then(|v| v.as_str()), Some("cleaned"));

    Ok(())
}

#[tokio::test]
async fn upload_chunk_ack_updates_workstation_device_status_when_configured() -> anyhow::Result<()>
{
    let status_updates = Arc::new(tokio::sync::Mutex::new(Vec::<serde_json::Value>::new()));
    let status_app = Router::new()
        .route(
            "/devices/status",
            patch(
                |State(status_updates): State<Arc<tokio::sync::Mutex<Vec<serde_json::Value>>>>,
                 Json(payload): Json<serde_json::Value>| async move {
                    status_updates.lock().await.push(payload);
                    Json(serde_json::json!({"ok": true}))
                },
            ),
        )
        .with_state(status_updates.clone());
    let status_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let status_addr = status_listener.local_addr()?;
    let status_server = tokio::spawn(async move {
        let _ = axum::serve(status_listener, status_app).await;
    });

    let server = support::TestServer::spawn_with_env(vec![(
        "EDGE_WORKSTATION_DEVICE_STATUS_URL".to_string(),
        format!("http://127.0.0.1:{}/devices/status", status_addr.port()),
    )])
    .await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-upload-status-001";
    let session_id = "sess-upload-status-001";
    let device_id = "iphone-status-001";
    let chunk_index: u32 = 7;

    for file_type in ["csv", "det"] {
        let meta = serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "chunk_index": chunk_index,
            "file_type": file_type,
            "file_name": format!("chunk_7.{file_type}")
        });
        let form = multipart::Form::new()
            .part(
                "file",
                multipart::Part::bytes(format!("{file_type}-bytes").into_bytes())
                    .file_name(format!("chunk_7.{file_type}")),
            )
            .text("metadata", meta.to_string());

        client
            .post(format!("{}/common_task/upload_chunk", server.http_base))
            .bearer_auth(&server.edge_token)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;
    }

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    let update = loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!(
                "等待 workstation device status update 超时: {:?}",
                status_updates.lock().await.clone()
            );
        }
        if let Some(payload) = status_updates.lock().await.last().cloned() {
            break payload;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    assert_eq!(
        update.get("device_id").and_then(|value| value.as_str()),
        Some(device_id)
    );
    assert_eq!(
        update.get("session_id").and_then(|value| value.as_str()),
        Some(session_id)
    );
    assert_eq!(
        update.get("ingest_status").and_then(|value| value.as_str()),
        Some("acked")
    );
    assert_eq!(
        update
            .get("upload_queue_depth")
            .and_then(|value| value.as_u64()),
        Some(0)
    );
    assert_eq!(
        update
            .get("last_ack")
            .and_then(|value| value.get("chunk_index"))
            .and_then(|value| value.as_u64()),
        Some(chunk_index as u64)
    );

    status_server.abort();
    Ok(())
}

#[tokio::test]
async fn upload_chunk_ack_updates_workstation_device_status_for_iphone_raw_media(
) -> anyhow::Result<()> {
    let status_updates = Arc::new(tokio::sync::Mutex::new(Vec::<serde_json::Value>::new()));
    let status_app = Router::new()
        .route(
            "/devices/status",
            patch(
                |State(status_updates): State<Arc<tokio::sync::Mutex<Vec<serde_json::Value>>>>,
                 Json(payload): Json<serde_json::Value>| async move {
                    status_updates.lock().await.push(payload);
                    Json(serde_json::json!({"ok": true}))
                },
            ),
        )
        .with_state(status_updates.clone());
    let status_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let status_addr = status_listener.local_addr()?;
    let status_server = tokio::spawn(async move {
        let _ = axum::serve(status_listener, status_app).await;
    });

    let server = support::TestServer::spawn_with_env(vec![(
        "EDGE_WORKSTATION_DEVICE_STATUS_URL".to_string(),
        format!("http://127.0.0.1:{}/devices/status", status_addr.port()),
    )])
    .await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-iphone-raw-media-001";
    let session_id = "sess-iphone-raw-media-001";
    let device_id = "iphone-raw-media-001";
    let chunk_index: u32 = 3;

    for (file_type, media_track, file_name) in [
        ("video", "main", "iphone_main_000003.mp4"),
        ("depth16", "depth", "iphone_depth_000003.bin"),
    ] {
        let meta = serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "chunk_index": chunk_index,
            "file_type": file_type,
            "file_name": file_name,
            "media_scope": "iphone",
            "media_track": media_track
        });
        let form = multipart::Form::new()
            .part(
                "file",
                multipart::Part::bytes(format!("{file_type}-bytes").into_bytes())
                    .file_name(file_name.to_string()),
            )
            .text("metadata", meta.to_string());

        client
            .post(format!("{}/common_task/upload_chunk", server.http_base))
            .bearer_auth(&server.edge_token)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;
    }

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    let update = loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!(
                "等待 iPhone raw media workstation device status update 超时: {:?}",
                status_updates.lock().await.clone()
            );
        }
        if let Some(payload) = status_updates.lock().await.last().cloned() {
            break payload;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    assert_eq!(
        update.get("device_id").and_then(|value| value.as_str()),
        Some(device_id)
    );
    assert_eq!(
        update.get("session_id").and_then(|value| value.as_str()),
        Some(session_id)
    );
    assert_eq!(
        update.get("ingest_status").and_then(|value| value.as_str()),
        Some("acked")
    );
    assert_eq!(
        update
            .get("last_ack")
            .and_then(|value| value.get("chunk_index"))
            .and_then(|value| value.as_u64()),
        Some(chunk_index as u64)
    );

    status_server.abort();
    Ok(())
}

#[tokio::test]
async fn upload_chunk_accepts_pairing_scoped_token_bound_to_device() -> anyhow::Result<()> {
    let scoped_token = "scoped-upload-token-test-001";
    let device_id = "iphone-scoped-token-001";
    let login_identity = "scoped@example.com";
    let device_name = "Scoped Token iPhone";
    let profile_id = "ego_wide_rgbd_multi_iphone_v1";
    let registry_path = std::env::temp_dir().join(format!(
        "edge-upload-token-registry-{}-{}.json",
        std::process::id(),
        now_unix_ms()
    ));
    std::fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "version": "1.0",
            "profile_id": "ego_wide_rgbd_multi_iphone_v1",
            "updated_unix_ms": now_unix_ms(),
            "devices": [{
                "device_id": device_id,
                "device_name": device_name,
                "login_identity": login_identity,
                "profile_id": profile_id,
                "upload_token_sha256": sha256_hex(scoped_token),
                "upload_token_status": "issued_by_workstation_pairing_endpoint",
                "token_expires_unix_ms": now_unix_ms() + 3_600_000,
            }]
        }))?,
    )?;
    let server = support::TestServer::spawn_with_env(vec![(
        "EDGE_UPLOAD_TOKEN_REGISTRY_PATH".to_string(),
        registry_path.to_string_lossy().to_string(),
    )])
    .await?;
    let client = reqwest::Client::new();

    let meta = serde_json::json!({
        "trip_id": "trip-scoped-token-001",
        "session_id": "sess-scoped-token-001",
        "device_id": device_id,
        "chunk_index": 0,
        "file_type": "csv",
        "file_name": "chunk_0.csv"
    });
    let form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(b"csv-bytes".to_vec()).file_name("chunk_0.csv"),
        )
        .text("metadata", meta.to_string());

    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(scoped_token)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server
        .data_dir
        .join("session")
        .join("sess-scoped-token-001");
    assert_session_identity_artifacts(
        &base_dir,
        device_id,
        login_identity,
        device_name,
        profile_id,
    )
    .await?;

    let wrong_device_meta = serde_json::json!({
        "trip_id": "trip-scoped-token-001",
        "session_id": "sess-scoped-token-001",
        "device_id": "iphone-not-paired",
        "chunk_index": 1,
        "file_type": "csv",
        "file_name": "chunk_1.csv"
    });
    let wrong_device_form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(b"csv-bytes".to_vec()).file_name("chunk_1.csv"),
        )
        .text("metadata", wrong_device_meta.to_string());
    let rejected = client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(scoped_token)
        .multipart(wrong_device_form)
        .send()
        .await?;
    assert_eq!(rejected.status(), reqwest::StatusCode::UNAUTHORIZED);

    let _ = std::fs::remove_file(registry_path);
    Ok(())
}

#[tokio::test]
async fn sqlite_pairing_authority_exchanges_validates_and_revokes_tokens() -> anyhow::Result<()> {
    let authority_path = std::env::temp_dir().join(format!(
        "edge-upload-token-authority-{}-{}.sqlite3",
        std::process::id(),
        now_unix_ms()
    ));
    let server = support::TestServer::spawn_with_env(vec![
        (
            "EDGE_UPLOAD_TOKEN_AUTHORITY_DB_PATH".to_string(),
            authority_path.to_string_lossy().to_string(),
        ),
        (
            "EDGE_PAIRING_EDGE_BASE_URL".to_string(),
            "http://127.0.0.1:3010/edge".to_string(),
        ),
        (
            "EDGE_PAIRING_EDGE_WS_URL".to_string(),
            "ws://127.0.0.1:8765/stream/fusion".to_string(),
        ),
        (
            "EDGE_PAIRING_STATUS_UI_URL".to_string(),
            "http://127.0.0.1:3010/#/capture".to_string(),
        ),
    ])
    .await?;
    let client = reqwest::Client::new();

    let envelope = client
        .get(format!("{}/pairing/envelope", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        envelope.get("type").and_then(|value| value.as_str()),
        Some("chek_ego_edge_pairing")
    );

    let device_id = "iphone-sqlite-authority-001";
    let exchange = client
        .post(format!("{}/pairing/exchange", server.http_base))
        .json(&serde_json::json!({
            "pairing_challenge": envelope
                .get("pairing_challenge")
                .and_then(|value| value.as_str())
                .unwrap_or_default(),
            "pairing_code": envelope
                .get("pairing_code")
                .and_then(|value| value.as_str())
                .unwrap_or_default(),
            "device_id": device_id,
            "device_name": "SQLite Authority iPhone",
            "login_identity": "sqlite@example.com"
        }))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    let scoped_token = exchange
        .get("scoped_upload_token")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing scoped_upload_token"))?
        .to_string();
    assert!(!scoped_token.is_empty());

    let devices = client
        .get(format!("{}/devices.json", server.http_base))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        devices
            .pointer("/authority/kind")
            .and_then(|value| value.as_str()),
        Some("sqlite")
    );
    assert_eq!(
        devices
            .get("devices")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .and_then(|item| item.get("device_id"))
            .and_then(|value| value.as_str()),
        Some(device_id)
    );
    assert!(
        !serde_json::to_string(&devices)?.contains(&scoped_token),
        "devices.json must not expose raw scoped token"
    );
    assert!(
        !serde_json::to_string(&devices)?.contains(&sha256_hex(&scoped_token)),
        "devices.json must not expose scoped token hashes"
    );

    for file_type in ["csv", "det"] {
        let meta = serde_json::json!({
            "trip_id": "trip-sqlite-authority-001",
            "session_id": "sess-sqlite-authority-001",
            "device_id": device_id,
            "chunk_index": 0,
            "file_type": file_type,
            "file_name": format!("chunk_0.{file_type}")
        });
        let form = multipart::Form::new()
            .part(
                "file",
                multipart::Part::bytes(format!("{file_type}-bytes").into_bytes())
                    .file_name(format!("chunk_0.{file_type}")),
            )
            .text("metadata", meta.to_string());
        client
            .post(format!("{}/common_task/upload_chunk", server.http_base))
            .bearer_auth(&scoped_token)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;
    }

    let devices_after_ack = {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            let payload = client
                .get(format!("{}/devices.json", server.http_base))
                .send()
                .await?
                .error_for_status()?
                .json::<serde_json::Value>()
                .await?;
            let device = payload
                .get("devices")
                .and_then(|value| value.as_array())
                .and_then(|items| {
                    items.iter().find(|item| {
                        item.get("device_id").and_then(|value| value.as_str()) == Some(device_id)
                    })
                })
                .cloned();
            if let Some(device) = device {
                if device.get("session_id").and_then(|value| value.as_str())
                    == Some("sess-sqlite-authority-001")
                {
                    break payload;
                }
            }
            if tokio::time::Instant::now() > deadline {
                anyhow::bail!("等待 SQLite authority device ack status 超时: {payload}");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    let authority_device = devices_after_ack
        .get("devices")
        .and_then(|value| value.as_array())
        .and_then(|items| {
            items.iter().find(|item| {
                item.get("device_id").and_then(|value| value.as_str()) == Some(device_id)
            })
        })
        .ok_or_else(|| anyhow::anyhow!("missing authority device after ack"))?;
    assert_eq!(
        authority_device
            .get("upload_queue_depth")
            .and_then(|value| value.as_u64()),
        Some(0)
    );
    assert_eq!(
        authority_device
            .get("ingest_status")
            .and_then(|value| value.as_str()),
        Some("acked")
    );
    assert_eq!(
        authority_device
            .get("last_ack")
            .and_then(|value| value.get("chunk_index"))
            .and_then(|value| value.as_u64()),
        Some(0)
    );

    let wrong_device_form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(b"csv-bytes".to_vec()).file_name("chunk_1.csv"),
        )
        .text(
            "metadata",
            serde_json::json!({
                "trip_id": "trip-sqlite-authority-001",
                "session_id": "sess-sqlite-authority-001",
                "device_id": "iphone-not-bound-to-token",
                "chunk_index": 1,
                "file_type": "csv",
                "file_name": "chunk_1.csv"
            })
            .to_string(),
        );
    let wrong_device = client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&scoped_token)
        .multipart(wrong_device_form)
        .send()
        .await?;
    assert_eq!(wrong_device.status(), reqwest::StatusCode::UNAUTHORIZED);

    let revoke = client
        .post(format!("{}/pairing/revoke", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({ "device_id": device_id }))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        revoke.get("revoked_count").and_then(|value| value.as_u64()),
        Some(1)
    );

    let revoked_form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(b"csv-bytes".to_vec()).file_name("chunk_2.csv"),
        )
        .text(
            "metadata",
            serde_json::json!({
                "trip_id": "trip-sqlite-authority-001",
                "session_id": "sess-sqlite-authority-001",
                "device_id": device_id,
                "chunk_index": 2,
                "file_type": "csv",
                "file_name": "chunk_2.csv"
            })
            .to_string(),
        );
    let revoked = client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&scoped_token)
        .multipart(revoked_form)
        .send()
        .await?;
    assert_eq!(revoked.status(), reqwest::StatusCode::UNAUTHORIZED);

    Ok(())
}

#[tokio::test]
async fn upload_chunk_accepts_multiple_pairing_scoped_tokens_concurrently() -> anyhow::Result<()> {
    let devices = [
        (
            "scoped-upload-token-a",
            "iphone-scoped-a",
            "alice@example.com",
            "Alice iPhone",
            "sess-scoped-a",
        ),
        (
            "scoped-upload-token-b",
            "iphone-scoped-b",
            "bob@example.com",
            "Bob iPhone",
            "sess-scoped-b",
        ),
    ];
    let registry_path = std::env::temp_dir().join(format!(
        "edge-upload-token-registry-multi-{}-{}.json",
        std::process::id(),
        now_unix_ms()
    ));
    std::fs::write(
        &registry_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "version": "1.0",
            "profile_id": "ego_wide_rgbd_multi_iphone_v1",
            "updated_unix_ms": now_unix_ms(),
            "devices": devices
                .iter()
                .map(|(token, device_id, login_identity, device_name, _session_id)| {
                    serde_json::json!({
                        "device_id": device_id,
                        "device_name": device_name,
                        "login_identity": login_identity,
                        "profile_id": "ego_wide_rgbd_multi_iphone_v1",
                        "upload_token_sha256": sha256_hex(token),
                        "upload_token_status": "issued_by_workstation_pairing_endpoint",
                        "token_expires_unix_ms": now_unix_ms() + 3_600_000,
                    })
                })
                .collect::<Vec<_>>(),
        }))?,
    )?;
    let server = support::TestServer::spawn_with_env(vec![(
        "EDGE_UPLOAD_TOKEN_REGISTRY_PATH".to_string(),
        registry_path.to_string_lossy().to_string(),
    )])
    .await?;
    let client = reqwest::Client::new();

    let mut uploads = Vec::new();
    for (token, device_id, _login_identity, _device_name, session_id) in devices {
        let client = client.clone();
        let upload_url = format!("{}/common_task/upload_chunk", server.http_base);
        uploads.push(tokio::spawn(async move {
            let meta = serde_json::json!({
                "trip_id": "trip-scoped-multi",
                "session_id": session_id,
                "device_id": device_id,
                "chunk_index": 0,
                "file_type": "csv",
                "file_name": format!("{device_id}.csv")
            });
            let form = multipart::Form::new()
                .part(
                    "file",
                    multipart::Part::bytes(format!("{device_id}-bytes").into_bytes())
                        .file_name(format!("{device_id}.csv")),
                )
                .text("metadata", meta.to_string());
            let response = client
                .post(upload_url)
                .bearer_auth(token)
                .multipart(form)
                .send()
                .await?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                anyhow::bail!("upload failed for {device_id}: status={status} body={body}");
            }
            anyhow::Ok(())
        }));
    }

    for upload in uploads {
        upload.await??;
    }

    for (_token, device_id, login_identity, device_name, session_id) in devices {
        let base_dir = server.data_dir.join("session").join(session_id);
        assert_session_identity_artifacts(
            &base_dir,
            device_id,
            login_identity,
            device_name,
            "ego_wide_rgbd_multi_iphone_v1",
        )
        .await?;
    }

    let _ = std::fs::remove_file(registry_path);
    Ok(())
}
