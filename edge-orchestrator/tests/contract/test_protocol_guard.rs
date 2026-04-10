use std::fs;
use std::process::Stdio;
use std::time::Duration;

#[tokio::test]
async fn protocol_guard_rejects_incompatible_pin() -> anyhow::Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let exe = env!("CARGO_BIN_EXE_edge-orchestrator");

    let http_port = pick_free_port()?;
    let mut ws_port = pick_free_port()?;
    while ws_port == http_port {
        ws_port = pick_free_port()?;
    }

    let pin_path = std::env::temp_dir().join(format!(
        "edge_orchestrator_protocol_pin_invalid_{}_{}.json",
        std::process::id(),
        http_port
    ));
    fs::write(
        &pin_path,
        serde_json::json!({
            "name": "teleop-protocol",
            "version": "1.0.0",
            "schema_sha256": "deadbeef",
            "compat_min": "2.0.0",
            "compat_max": "3.0.0"
        })
        .to_string(),
    )?;

    let mut child = tokio::process::Command::new(exe)
        .current_dir(manifest_dir)
        .env("EDGE_HTTP_ADDR", format!("127.0.0.1:{http_port}"))
        .env("EDGE_WS_ADDR", format!("127.0.0.1:{ws_port}"))
        .env("EDGE_TOKEN", "edge-token-test-001")
        .env("EXTRINSIC_VERSION", "ext-test-0.1.0")
        .env("TELEOP_PROTOCOL_PIN_PATH", &pin_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    let status = tokio::time::timeout(Duration::from_secs(5), child.wait()).await??;
    assert!(!status.success(), "协议钉住不兼容时应启动失败");

    let _ = fs::remove_file(&pin_path);
    Ok(())
}

fn pick_free_port() -> anyhow::Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}
