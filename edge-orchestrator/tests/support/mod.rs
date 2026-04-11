use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, Context};
use serde::de::DeserializeOwned;

pub mod ws_harness;

pub struct TestServer {
    pub http_base: String,
    #[allow(dead_code)]
    pub ws_base: String,
    #[allow(dead_code)]
    pub csi_addr: String,
    pub edge_token: String,
    pub data_dir: PathBuf,
    child: Child,
}

impl TestServer {
    pub async fn spawn() -> anyhow::Result<Self> {
        Self::spawn_with_env(Vec::<(String, String)>::new()).await
    }

    pub async fn spawn_with_env<I, K, V>(extra_env: I) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<std::ffi::OsStr>,
        V: AsRef<std::ffi::OsStr>,
    {
        let http_port = pick_free_port()?;
        let mut ws_port = pick_free_port()?;
        while ws_port == http_port {
            ws_port = pick_free_port()?;
        }
        let csi_port = pick_free_udp_port()?;

        let http_base = format!("http://127.0.0.1:{http_port}");
        let ws_base = format!("ws://127.0.0.1:{ws_port}");
        let csi_addr = format!("127.0.0.1:{csi_port}");
        let edge_token = format!("edge-token-test-{http_port}");
        let data_dir =
            std::env::temp_dir().join(format!("edge-orchestrator-test-{http_port}-{ws_port}"));
        std::fs::create_dir_all(&data_dir).context("创建测试 data_dir 失败")?;

        let exe = env!("CARGO_BIN_EXE_edge-orchestrator");
        let manifest_dir = env!("CARGO_MANIFEST_DIR");

        let mut command = Command::new(exe);
        command
            .current_dir(manifest_dir)
            .env("EDGE_HTTP_ADDR", format!("127.0.0.1:{http_port}"))
            .env("EDGE_WS_ADDR", format!("127.0.0.1:{ws_port}"))
            .env("EDGE_DATA_DIR", &data_dir)
            .env("CSI_UDP_BIND", &csi_addr)
            .env("CSI_CHUNK_MAX_PACKETS", "3")
            .env("CSI_CHUNK_MAX_SPAN_MS", "2000")
            .env("CSI_CHUNK_MAX_BYTES", "1048576")
            .env("EDGE_TOKEN", &edge_token)
            .env("EXTRINSIC_VERSION", "ext-test-0.1.0")
            .env("RUST_LOG", "info")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (key, value) in extra_env {
            command.env(key, value);
        }

        let child = command
            .spawn()
            .with_context(|| format!("启动 edge-orchestrator 失败：{exe}"))?;

        wait_health_ok(&http_base).await?;

        Ok(Self {
            http_base,
            ws_base,
            csi_addr,
            edge_token,
            data_dir,
            child,
        })
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }

    pub fn endpoint(&self, path: &str) -> anyhow::Result<String> {
        if !path.starts_with('/') {
            return Err(anyhow!("测试请求路径必须以 / 开头: {path}"));
        }
        if path.contains('\r') || path.contains('\n') {
            return Err(anyhow!("测试请求路径不能包含换行符"));
        }
        let url = reqwest::Url::parse(&self.http_base)
            .with_context(|| format!("解析测试 http_base 失败: {}", self.http_base))?;
        let host = url
            .host_str()
            .ok_or_else(|| anyhow!("测试 http_base 缺少主机: {}", self.http_base))?;
        let is_loopback = host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<std::net::IpAddr>()
                .map(|ip| ip.is_loopback())
                .unwrap_or(false);
        if url.scheme() != "http" || !is_loopback {
            return Err(anyhow!(
                "测试 http_base 只允许使用 loopback http 地址: {}",
                self.http_base
            ));
        }
        Ok(format!("{}{}", self.http_base, path))
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.kill();
        let _ = std::fs::remove_dir_all(&self.data_dir);
    }
}

#[allow(dead_code)]
pub async fn enable_full_teleop_profile(
    client: &reqwest::Client,
    server: &TestServer,
    trip_id: &str,
    session_id: &str,
) -> anyhow::Result<()> {
    client
        .post(format!("{}/control/profile", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "teleop_enabled": true,
            "body_control_enabled": true,
            "hand_control_enabled": true,
        }))
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub async fn authed_get_json<T: DeserializeOwned>(
    client: &reqwest::Client,
    server: &TestServer,
    path: &str,
) -> anyhow::Result<T> {
    client
        .get(server.endpoint(path)?)
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .error_for_status()?
        .json::<T>()
        .await
        .map_err(Into::into)
}

fn pick_free_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("绑定临时端口失败")?;
    let port = listener.local_addr().context("读取临时端口失败")?.port();
    Ok(port)
}

fn pick_free_udp_port() -> anyhow::Result<u16> {
    let sock = std::net::UdpSocket::bind(("127.0.0.1", 0)).context("绑定临时 UDP 端口失败")?;
    let port = sock.local_addr().context("读取临时 UDP 端口失败")?.port();
    Ok(port)
}

async fn wait_health_ok(http_base: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if tokio::time::Instant::now() > deadline {
            return Err(anyhow!("等待 /health 超时：{http_base}"));
        }

        if let Ok(resp) = client.get(format!("{http_base}/health")).send().await {
            if let Ok(v) = resp.json::<serde_json::Value>().await {
                if v.get("status").and_then(|s| s.as_str()) == Some("ok") {
                    return Ok(());
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
