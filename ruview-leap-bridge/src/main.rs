use std::net::SocketAddr;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use metrics_exporter_prometheus::PrometheusBuilder;
use tokio::signal;
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

use ruview_leap_bridge::bridge::publisher::{BridgeRunner, RunnerHandle};
use ruview_leap_bridge::config::Config;
use ruview_leap_bridge::protocol::version_guard::{ProtocolPin, ProtocolVersionInfo};

#[derive(Clone)]
struct HttpState {
    handle: RunnerHandle,
    metrics: metrics_exporter_prometheus::PrometheusHandle,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config = Config::load()?;
    let metrics_handle = install_metrics_recorder()?;

    let protocol_pin = ProtocolPin::load_from_path(&config.protocol_pin_path)?;
    let protocol: ProtocolVersionInfo = protocol_pin.validate_and_to_info()?;
    ruview_leap_bridge::protocol::version_guard::emit_protocol_metrics(&protocol);

    let (runner, handle): (BridgeRunner, RunnerHandle) =
        BridgeRunner::new(config.clone(), protocol);
    tokio::spawn(async move { runner.run().await });

    // 健康与指标（MVP）
    let http_addr: SocketAddr = config.http_addr.parse()?;
    let app = Router::new()
        .route("/health", get(get_health))
        .route("/metrics", get(get_metrics))
        .with_state(HttpState {
            handle,
            metrics: metrics_handle,
        })
        .layer(TraceLayer::new_for_http());

    info!(%http_addr, "ruview-leap-bridge starting");
    axum::serve(tokio::net::TcpListener::bind(http_addr).await?, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .json()
        .init();
}

fn install_metrics_recorder() -> anyhow::Result<metrics_exporter_prometheus::PrometheusHandle> {
    Ok(PrometheusBuilder::new().install_recorder()?)
}

async fn get_health(
    State(state): State<HttpState>,
) -> Json<ruview_leap_bridge::bridge::publisher::HealthSnapshot> {
    Json(state.handle.health_snapshot())
}

async fn get_metrics(State(state): State<HttpState>) -> String {
    state.metrics.render()
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install terminate handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
