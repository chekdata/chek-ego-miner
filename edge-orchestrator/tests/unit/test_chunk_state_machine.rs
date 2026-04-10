#[path = "../support/mod.rs"]
mod support;

#[tokio::test]
async fn chunk_cleaned_state_is_queryable() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-chunk-001";
    let session_id = "sess-chunk-001";
    let chunk_index = 12;

    let state0 = client
        .get(format!(
            "{}/chunk/state?trip_id={trip_id}&session_id={session_id}&chunk_index={chunk_index}",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        state0.get("state").and_then(|v| v.as_str()),
        Some("received")
    );

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

    let state1 = client
        .get(format!(
            "{}/chunk/state?trip_id={trip_id}&session_id={session_id}&chunk_index={chunk_index}",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
    assert_eq!(
        state1.get("state").and_then(|v| v.as_str()),
        Some("cleaned")
    );

    Ok(())
}
