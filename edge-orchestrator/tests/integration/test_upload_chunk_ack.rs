use crate::support;

use std::time::Duration;

use reqwest::multipart;

use support::ws_harness::WsHarness;

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

    let state = client
        .get(format!(
            "{}/chunk/state?trip_id={trip_id}&session_id={session_id}&chunk_index={chunk_index}",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    assert_eq!(state.get("state").and_then(|v| v.as_str()), Some("cleaned"));

    Ok(())
}
