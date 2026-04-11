#![recursion_limit = "256"]

mod api;
mod auth;
mod calibration;
mod config;
mod control;
mod host_metrics;
mod operator;
mod path_safety;
mod protocol;
mod reason;
mod recorder;
mod sensing;
mod target;
mod upload_worker;
mod ws;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::Router;
use metrics_exporter_prometheus::PrometheusBuilder;
use serde_json::Value;
use tokio::process::Command;
use tokio::signal;
use tokio::sync::{broadcast, mpsc};
use tokio::time::{sleep, Duration};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::config::Config;
use crate::protocol::version_guard::{ProtocolPin, ProtocolVersionInfo};
use crate::ws::types::{ChunkAckPacket, FusionStatePacket, RetargetReferenceV1, TeleopFrameV1};

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub protocol: ProtocolVersionInfo,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    pub http_client: reqwest::Client,

    pub gate: Arc<control::gate::Gate>,
    pub chunk_sm: Arc<recorder::chunk_state_machine::ChunkStateMachine>,
    pub recorder: Arc<recorder::session_recorder::SessionRecorder>,
    pub robot_state_record_tx: mpsc::Sender<RobotStateRecordRequest>,
    pub bridge_store: Arc<control::gate::BridgeStore>,
    pub session: Arc<control::gate::SessionStore>,
    pub phone_capture_commands: Arc<control::gate::PhoneCaptureCommandStore>,
    pub phone_ingress_status: Arc<control::gate::PhoneIngressStatusStore>,
    pub association_hint_clients: Arc<control::gate::AssociationHintClientStore>,
    pub fusion_stream_clients: Arc<control::gate::FusionClientStore>,
    pub vision: Arc<sensing::VisionStore>,
    pub stereo: Arc<sensing::StereoStore>,
    pub wifi_pose: Arc<sensing::WifiPoseStore>,
    pub csi: Arc<sensing::CsiStore>,
    pub operator: Arc<operator::OperatorStore>,
    pub iphone_stereo_calibration: Arc<calibration::IphoneStereoCalibrationStore>,
    pub wifi_stereo_calibration: Arc<calibration::WifiStereoCalibrationStore>,

    pub fusion_state_tx: broadcast::Sender<FusionStatePacket>,
    pub chunk_ack_tx: broadcast::Sender<ChunkAckPacket>,
    pub teleop_tx: broadcast::Sender<TeleopFrameV1>,
    pub teleop_latest: Arc<Mutex<Option<TeleopFrameV1>>>,
    pub retarget_latest: Arc<Mutex<Option<RetargetReferenceV1>>>,
    pub sim_tracking_latest: Arc<Mutex<Option<SimTrackingCarrySnapshot>>>,
}

#[derive(Clone, Debug)]
pub struct SimTrackingCarrySnapshot {
    pub payload: Value,
    pub target_session_id: Option<String>,
    pub target_person_id: Option<String>,
    pub updated_edge_time_ns: u64,
}

#[derive(Clone, Debug)]
pub struct RobotStateRecordRequest {
    pub trip_id: String,
    pub session_id: String,
    pub payload: Value,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let config = Config::from_env()?;
    let protocol_pin_path = resolve_manifest_relative_path(&config.protocol_pin_path);
    let protocol_pin = ProtocolPin::load_from_path(&protocol_pin_path)?;
    let protocol = protocol_pin.validate_and_to_info()?;

    if args.iter().any(|arg| arg == "--repair-existing-sessions") {
        let recorder = recorder::session_recorder::SessionRecorder::new(config.data_dir.clone());
        let results = recorder
            .repair_existing_sessions(&protocol, &config)
            .await
            .map_err(anyhow::Error::msg)?;
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    let metrics_handle = install_metrics_recorder()?;
    let http_client = reqwest::Client::builder().build()?;

    let gate = Arc::new(control::gate::Gate::new(config.clone()));
    let chunk_sm = Arc::new(recorder::chunk_state_machine::ChunkStateMachine::default());
    let recorder = Arc::new(recorder::session_recorder::SessionRecorder::new(
        config.data_dir.clone(),
    ));
    let (robot_state_record_tx, robot_state_record_rx) = mpsc::channel(512);
    let bridge_store = Arc::new(control::gate::BridgeStore::default());
    let session = Arc::new(control::gate::SessionStore::new(&config));
    let phone_capture_commands = Arc::new(control::gate::PhoneCaptureCommandStore::default());
    let phone_ingress_status = Arc::new(control::gate::PhoneIngressStatusStore::default());
    let association_hint_clients = Arc::new(control::gate::AssociationHintClientStore::default());
    let fusion_stream_clients = Arc::new(control::gate::FusionClientStore::default());
    let vision = Arc::new(sensing::VisionStore::new(config.allow_simulated_capture));
    let stereo = Arc::new(sensing::StereoStore::default());
    let wifi_pose = Arc::new(sensing::WifiPoseStore::default());
    let csi = Arc::new(sensing::CsiStore::default());
    let operator = Arc::new(operator::OperatorStore::default());
    let iphone_stereo_calibration = Arc::new(calibration::IphoneStereoCalibrationStore::new(
        config.iphone_stereo_extrinsic_path.clone(),
    ));
    let wifi_stereo_calibration = Arc::new(calibration::WifiStereoCalibrationStore::new(
        config.wifi_stereo_extrinsic_path.clone(),
    ));

    let (fusion_state_tx, _) = broadcast::channel(config.fusion_broadcast_capacity);
    let (chunk_ack_tx, _) = broadcast::channel(config.chunk_ack_broadcast_capacity);
    let (teleop_tx, _) = broadcast::channel(config.teleop_broadcast_capacity);
    let teleop_latest = Arc::new(Mutex::new(None));
    let retarget_latest = Arc::new(Mutex::new(None));
    let sim_tracking_latest = Arc::new(Mutex::new(None));

    let state = AppState {
        config: config.clone(),
        protocol: protocol.clone(),
        metrics_handle,
        http_client: http_client.clone(),
        gate,
        chunk_sm,
        recorder,
        robot_state_record_tx,
        bridge_store,
        session,
        phone_capture_commands,
        phone_ingress_status,
        association_hint_clients,
        fusion_stream_clients,
        vision,
        stereo,
        wifi_pose,
        csi,
        operator,
        iphone_stereo_calibration,
        wifi_stereo_calibration,
        fusion_state_tx,
        chunk_ack_tx,
        teleop_tx,
        teleop_latest,
        retarget_latest,
        sim_tracking_latest,
    };

    tokio::spawn(run_robot_state_record_worker(
        state.recorder.clone(),
        state.protocol.clone(),
        state.config.clone(),
        robot_state_record_rx,
    ));

    protocol::version_guard::emit_protocol_metrics(&protocol);

    // 后台任务（看门狗 + 周期发布）。
    if config.control_enabled {
        let state = state.clone();
        tokio::spawn(async move { control::tasks::run_deadman_watchdog(state).await });
    }
    if config.fusion_enabled {
        let state = state.clone();
        tokio::spawn(async move { ws::tasks::run_fusion_state_publisher(state).await });
    }
    if config.fusion_enabled || config.control_enabled {
        let state = state.clone();
        tokio::spawn(async move { ws::tasks::run_teleop_publisher(state).await });
    }
    {
        let state = state.clone();
        tokio::spawn(async move { sensing::run_csi_udp_ingest(state).await });
    }
    {
        let state = state.clone();
        tokio::spawn(async move { sensing::run_wifi_tracked_pose_poll(state).await });
    }
    if config.crowd_upload_enabled {
        let state = state.clone();
        tokio::spawn(async move { upload_worker::run_crowd_upload_worker(state).await });
    }
    maybe_spawn_phone_vision_service(&config, &http_client).await;
    maybe_spawn_vlm_sidecar(&config, &http_client).await;

    let http_addr: SocketAddr = config.http_addr.parse()?;
    let ws_addr: SocketAddr = config.ws_addr.parse()?;

    let http_app: Router = api::http_router(state.clone());
    let ws_app: Router = ws::ws_router(state.clone());

    info!(%http_addr, %ws_addr, "edge-orchestrator starting");

    let http_server = axum::serve(
        tokio::net::TcpListener::bind(http_addr).await?,
        http_app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal());
    let ws_server = axum::serve(
        tokio::net::TcpListener::bind(ws_addr).await?,
        ws_app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal());

    tokio::select! {
        res = http_server => { res?; }
        res = ws_server => { res?; }
    }

    Ok(())
}

async fn maybe_spawn_phone_vision_service(config: &Config, http_client: &reqwest::Client) {
    if !config.phone_vision_service_autostart {
        return;
    }

    let base = config.phone_vision_service_base.trim();
    if !(base.starts_with("http://127.0.0.1:") || base.starts_with("http://localhost:")) {
        info!(service_base=%base, "skip autostart for non-local phone vision service");
        return;
    }

    let mut command = Command::new(&config.phone_vision_python_bin);
    command
        .arg(&config.phone_vision_service_path)
        .env(
            "EDGE_PHONE_VISION_SERVICE_BASE",
            &config.phone_vision_service_base,
        )
        .kill_on_drop(false);

    match command.spawn() {
        Ok(_) => {
            info!(
                service_base=%config.phone_vision_service_base,
                python_bin=%config.phone_vision_python_bin,
                script=%config.phone_vision_service_path,
                "phone vision service autostarted"
            );
            wait_for_phone_vision_service_ready(config, http_client).await;
        }
        Err(error) => warn!(
            service_base=%config.phone_vision_service_base,
            python_bin=%config.phone_vision_python_bin,
            script=%config.phone_vision_service_path,
            error=%error,
            "failed to autostart phone vision service"
        ),
    }
}

fn resolve_manifest_relative_path(path: &str) -> String {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() || candidate.exists() {
        return candidate.to_string_lossy().to_string();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(candidate)
        .to_string_lossy()
        .to_string()
}

async fn wait_for_phone_vision_service_ready(config: &Config, http_client: &reqwest::Client) {
    let health_url = format!(
        "{}/health",
        config.phone_vision_service_base.trim_end_matches('/')
    );
    for attempt in 1..=20 {
        match http_client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                info!(health_url=%health_url, attempt, "phone vision service ready");
                return;
            }
            Ok(response) => {
                warn!(
                    health_url=%health_url,
                    attempt,
                    status=%response.status(),
                    "phone vision service health returned non-success during warmup"
                );
            }
            Err(error) => {
                warn!(
                    health_url=%health_url,
                    attempt,
                    error=%error,
                    "phone vision service not ready yet"
                );
            }
        }
        sleep(Duration::from_millis(250)).await;
    }
    warn!(health_url=%health_url, "phone vision service warmup timed out");
}

async fn maybe_spawn_vlm_sidecar(config: &Config, http_client: &reqwest::Client) {
    if !config.vlm_indexing_enabled || !config.vlm_sidecar_autostart {
        return;
    }

    let base = config.vlm_sidecar_base.trim();
    if base.is_empty() {
        info!("skip vlm sidecar autostart because EDGE_VLM_SIDECAR_BASE is empty");
        return;
    }
    if !(base.starts_with("http://127.0.0.1:") || base.starts_with("http://localhost:")) {
        info!(service_base=%base, "skip autostart for non-local vlm sidecar");
        return;
    }

    let mut command = Command::new(&config.vlm_sidecar_python_bin);
    command
        .arg(&config.vlm_sidecar_path)
        .env("EDGE_VLM_SIDECAR_BASE", &config.vlm_sidecar_base)
        .env("EDGE_VLM_MODEL_ID", &config.vlm_model_id)
        .env("EDGE_VLM_FALLBACK_MODEL_ID", &config.vlm_fallback_model_id)
        .env(
            "EDGE_VLM_PRIMARY_MODEL_PATH",
            &config.vlm_primary_model_path,
        )
        .env(
            "EDGE_VLM_FALLBACK_MODEL_PATH",
            &config.vlm_fallback_model_path,
        )
        .env("EDGE_VLM_RUNTIME_DEVICE", &config.vlm_runtime_device)
        .env(
            "EDGE_VLM_EDGE_LONGEST_SIDE_PX",
            config.vlm_edge_longest_side_px.to_string(),
        )
        .env(
            "EDGE_VLM_EDGE_IMAGE_SEQ_LEN",
            config.vlm_edge_image_seq_len.to_string(),
        )
        .env(
            "EDGE_VLM_DISABLE_IMAGE_SPLITTING",
            if config.vlm_disable_image_splitting {
                "1"
            } else {
                "0"
            },
        )
        .env(
            "EDGE_VLM_AUTO_FALLBACK_LATENCY_MS",
            config.vlm_auto_fallback_latency_ms.to_string(),
        )
        .env(
            "EDGE_VLM_AUTO_FALLBACK_COOLDOWN_MS",
            config.vlm_auto_fallback_cooldown_ms.to_string(),
        )
        .env(
            "EDGE_VLM_MAX_CONSECUTIVE_FAILURES",
            config.vlm_max_consecutive_failures.to_string(),
        )
        .kill_on_drop(false);

    match command.spawn() {
        Ok(_) => {
            info!(
                service_base=%config.vlm_sidecar_base,
                python_bin=%config.vlm_sidecar_python_bin,
                script=%config.vlm_sidecar_path,
                model_id=%config.vlm_model_id,
                fallback_model_id=%config.vlm_fallback_model_id,
                "vlm sidecar autostarted"
            );
            wait_for_vlm_sidecar_ready(config, http_client).await;
        }
        Err(error) => warn!(
            service_base=%config.vlm_sidecar_base,
            python_bin=%config.vlm_sidecar_python_bin,
            script=%config.vlm_sidecar_path,
            error=%error,
            "failed to autostart vlm sidecar"
        ),
    }
}

async fn wait_for_vlm_sidecar_ready(config: &Config, http_client: &reqwest::Client) {
    let health_url = format!("{}/health", config.vlm_sidecar_base.trim_end_matches('/'));
    for attempt in 1..=20 {
        match http_client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                info!(health_url=%health_url, attempt, "vlm sidecar ready");
                return;
            }
            Ok(response) => {
                warn!(
                    health_url=%health_url,
                    attempt,
                    status=%response.status(),
                    "vlm sidecar health returned non-success during warmup"
                );
            }
            Err(error) => {
                warn!(
                    health_url=%health_url,
                    attempt,
                    error=%error,
                    "vlm sidecar not ready yet"
                );
            }
        }
        sleep(Duration::from_millis(250)).await;
    }
    warn!(health_url=%health_url, "vlm sidecar warmup timed out");
}

async fn run_robot_state_record_worker(
    recorder: Arc<recorder::session_recorder::SessionRecorder>,
    protocol: ProtocolVersionInfo,
    config: Config,
    mut rx: mpsc::Receiver<RobotStateRecordRequest>,
) {
    while let Some(request) = rx.recv().await {
        recorder
            .record_robot_state(
                &protocol,
                &config,
                &request.trip_id,
                &request.session_id,
                &request.payload,
            )
            .await;
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .json()
        .init();
}

fn install_metrics_recorder() -> anyhow::Result<metrics_exporter_prometheus::PrometheusHandle> {
    let builder = PrometheusBuilder::new();
    let handle = builder.install_recorder()?;
    Ok(handle)
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

    info!("shutdown signal received");
}
