use crate::support;

use std::time::Duration;

use futures_util::StreamExt;
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn fusion_ws_supports_cbor_binary_frames() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;

    // 订阅 /stream/fusion，并声明 format=cbor（P2：二进制化）。
    let url = format!(
        "{}/stream/fusion?token={}&format=cbor",
        server.ws_base, server.edge_token
    );
    let (ws, _) = tokio_tungstenite::connect_async(url).await?;
    let (_write, mut read) = ws.split();

    // 触发一次 chunk_ack：先上传 csv，再上传 det（默认 required=csv,det）。
    let client = reqwest::Client::new();
    let trip_id = "trip-cbor-001";
    let session_id = "sess-cbor-001";
    let chunk_index: u32 = 12;

    async fn upload_one(
        client: &reqwest::Client,
        server: &support::TestServer,
        trip_id: &str,
        session_id: &str,
        chunk_index: u32,
        file_type: &str,
    ) -> anyhow::Result<()> {
        let meta = serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "client-test-001",
            "chunk_index": chunk_index,
            "file_type": file_type,
            "file_name": format!("{file_type}.bin"),
        });
        let form = reqwest::multipart::Form::new()
            .text("metadata", meta.to_string())
            .part(
                "file",
                reqwest::multipart::Part::bytes(vec![1u8, 2, 3, 4]).file_name("blob.bin"),
            );
        client
            .post(format!("{}/common_task/upload_chunk", server.http_base))
            .bearer_auth(&server.edge_token)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    upload_one(&client, &server, trip_id, session_id, chunk_index, "csv").await?;
    upload_one(&client, &server, trip_id, session_id, chunk_index, "det").await?;

    // 读取 WS：应该收到 chunk_ack_packet（二进制 CBOR）。
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待 chunk_ack_packet(cbor) 超时");
        }
        let msg = match tokio::time::timeout(Duration::from_millis(200), read.next()).await {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(e))) => return Err(e.into()),
            Ok(None) => anyhow::bail!("ws closed"),
            Err(_) => continue,
        };

        let bin = match msg {
            Message::Binary(b) => b,
            Message::Text(_) => continue, // format=cbor 时不期望，但保持兼容
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Close(_) => anyhow::bail!("ws closed"),
            _ => continue,
        };
        let Ok(v) = serde_cbor::from_slice::<serde_json::Value>(&bin) else {
            continue;
        };
        if v.get("type").and_then(|t| t.as_str()) != Some("chunk_ack_packet") {
            continue;
        }
        let idx_ok = v.get("chunk_index").and_then(|x| x.as_u64()) == Some(chunk_index as u64);
        let status_ok = v.get("status").and_then(|x| x.as_str()) == Some("stored");
        if idx_ok && status_ok {
            return Ok(());
        }
    }
}
