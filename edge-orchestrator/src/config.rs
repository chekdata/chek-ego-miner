use std::env;

use crate::path_safety;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid int for {key}: {value}")]
    Int { key: String, value: String },
    #[error("invalid float for {key}: {value}")]
    Float { key: String, value: String },
    #[error("invalid float list for {key}: {value}")]
    FloatList { key: String, value: String },
    #[error("invalid value for {key}: {value}")]
    Value { key: String, value: String },
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeFeatureFlags {
    pub phone_ingest_enabled: bool,
    pub stereo_enabled: bool,
    pub wifi_enabled: bool,
    pub fusion_enabled: bool,
    pub control_enabled: bool,
    pub sim_enabled: bool,
    pub vlm_indexing_enabled: bool,
    pub preview_generation_enabled: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeRuntimeProfile {
    RawCaptureOnly,
    CapturePlusFacts,
    CapturePlusVlm,
    TeleopFullstack,
}

impl EdgeRuntimeProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RawCaptureOnly => "raw_capture_only",
            Self::CapturePlusFacts => "capture_plus_facts",
            Self::CapturePlusVlm => "capture_plus_vlm",
            Self::TeleopFullstack => "teleop_fullstack",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "raw_capture_only" => Some(Self::RawCaptureOnly),
            "capture_plus_facts" => Some(Self::CapturePlusFacts),
            "capture_plus_vlm" => Some(Self::CapturePlusVlm),
            "teleop_fullstack" => Some(Self::TeleopFullstack),
            _ => None,
        }
    }

    fn default_feature_flags(self) -> RuntimeFeatureFlags {
        match self {
            Self::RawCaptureOnly => RuntimeFeatureFlags {
                phone_ingest_enabled: true,
                stereo_enabled: false,
                wifi_enabled: false,
                fusion_enabled: false,
                control_enabled: false,
                sim_enabled: false,
                vlm_indexing_enabled: false,
                preview_generation_enabled: false,
            },
            Self::CapturePlusFacts => RuntimeFeatureFlags {
                phone_ingest_enabled: true,
                stereo_enabled: false,
                wifi_enabled: false,
                fusion_enabled: true,
                control_enabled: false,
                sim_enabled: false,
                vlm_indexing_enabled: false,
                preview_generation_enabled: false,
            },
            Self::CapturePlusVlm => RuntimeFeatureFlags {
                phone_ingest_enabled: true,
                stereo_enabled: false,
                wifi_enabled: false,
                fusion_enabled: true,
                control_enabled: false,
                sim_enabled: false,
                vlm_indexing_enabled: true,
                preview_generation_enabled: true,
            },
            Self::TeleopFullstack => RuntimeFeatureFlags {
                phone_ingest_enabled: true,
                stereo_enabled: true,
                wifi_enabled: false,
                fusion_enabled: true,
                control_enabled: true,
                sim_enabled: false,
                vlm_indexing_enabled: false,
                preview_generation_enabled: false,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CrowdUploadPolicyMode {
    MetadataOnly,
    MetadataPlusPreview,
    FullRawMirror,
}

impl CrowdUploadPolicyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MetadataOnly => "metadata_only",
            Self::MetadataPlusPreview => "metadata_plus_preview",
            Self::FullRawMirror => "full_raw_mirror",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "metadata_only" => Some(Self::MetadataOnly),
            "metadata_plus_preview" => Some(Self::MetadataPlusPreview),
            "full_raw_mirror" => Some(Self::FullRawMirror),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub http_addr: String,
    pub ws_addr: String,

    /// 数据落盘根目录（会话目录会落到 `<data_dir>/session/<session_id>/...`）。
    pub data_dir: String,

    /// 最小鉴权 token（对应 PRD 的 `edge_token`）。未配置时不启用鉴权（仅建议用于本机开发）。
    pub edge_token: Option<String>,
    /// 是否允许 Prometheus `/metrics` 在未鉴权情况下访问（默认 false）。
    pub metrics_public: bool,

    pub protocol_pin_path: String,

    pub operator_frame: String,
    pub robot_base_frame: String,
    pub extrinsic_version: String,
    pub default_robot_type: String,
    pub default_end_effector_type: String,

    pub deadman_enabled_default: bool,
    pub deadman_timeout_ms: u64,
    pub time_sync_ok_window_ms: u64,
    pub time_sync_rtt_ok_ms: u64,

    /// CSI UDP 监听地址（ADR-018，默认 `0.0.0.0:5005`）。
    pub csi_udp_bind: String,
    /// 可选 CSI UDP 镜像地址。配置后会把收到的原始 CSI 包转发到该地址，便于 Wi‑Fi pose 推理服务并行消费。
    pub csi_udp_mirror_addr: Option<String>,
    /// CSI 输入“新鲜度”窗口：超过该时间未收到 CSI，则 `csi_conf=0`。
    pub csi_stale_ms: u64,
    /// CSI raw chunk 的每目录文件数上限；到达后切换到下一个 bucket 目录。
    pub csi_chunk_bucket_size: u64,
    /// 单个 CSI raw chunk 的最大包数；超过后立即滚下一个 chunk。
    pub csi_chunk_max_packets: u64,
    /// 单个 CSI raw chunk 的最大文件大小（bytes）；超过后立即滚下一个 chunk。
    pub csi_chunk_max_bytes: u64,
    /// 单个 CSI raw chunk 的最大时间跨度（ms）；超过后立即滚下一个 chunk。
    pub csi_chunk_max_span_ms: u64,
    /// 视觉输入“新鲜度”窗口：超过该时间未收到 `capture_pose_packet`，则 `vision_conf=0`。
    pub vision_stale_ms: u64,
    /// 是否允许 simulated capture 进入 live 视觉链路。默认 false，避免仿真流污染现场关联。
    pub allow_simulated_capture: bool,
    /// 双目输入“新鲜度”窗口：超过该时间未收到双目 pose，则 stereo 视为不新鲜。
    pub stereo_stale_ms: u64,
    /// Wi‑Fi 骨骼输入“新鲜度”窗口：超过该时间未收到 `wifi_pose_packet`，则 Wi‑Fi pose 视为不新鲜。
    pub wifi_pose_stale_ms: u64,
    /// 是否让 edge 直接轮询 sensing-server 的稳定 tracked Wi‑Fi pose contract。
    pub wifi_tracked_pose_direct_enabled: bool,
    /// 直接轮询 tracked Wi‑Fi pose 的周期（ms）。
    pub wifi_tracked_pose_poll_ms: u64,
    /// 视觉断流时的“CSI 补盲 hold 窗口”：在 CSI 仍新鲜的前提下，允许继续回显/输出上一帧姿态。
    pub operator_hold_ms: u64,

    /// 2D->3D 粗投影参数（MVP：用于无深度/无双目场景下的调试回显）。
    pub vision_proj_x_span_m: f32,
    pub vision_proj_y_span_m: f32,
    pub vision_proj_z_base_m: f32,

    /// `operator_frame -> robot_base_frame` 外参（用于 teleop wrist pose 输出）。
    pub extrinsic_translation_m: [f32; 3],
    pub extrinsic_rotation_quat: [f32; 4],

    /// 速度/加速度/jerk 门控（用于 limit/freeze；单位：m/s, m/s^2, m/s^3）。
    pub motion_max_speed_m_s: f32,
    pub motion_max_accel_m_s2: f32,
    pub motion_max_jerk_m_s3: f32,

    pub fusion_publish_hz: u32,
    pub teleop_publish_hz: u32,

    pub fusion_broadcast_capacity: usize,
    pub teleop_broadcast_capacity: usize,
    pub chunk_ack_broadcast_capacity: usize,

    pub bridge_stale_ms: u64,

    /// `upload_chunk`（multipart）按 `chunk_index` 对账时，需要哪些 file_type 才认为该 chunk 完整并下发 `chunk_ack(stored)`。
    /// - 为空：任意单文件上传即认为完整（更通用，但可能导致“先 ack 后上传第二个文件”）。
    /// - 默认：`csv,det`（对齐现有 iOS chunk uploader 的两文件模型）。
    pub upload_required_file_types: Vec<String>,

    /// `acked -> cleaned` SLA（用于 `ack_cleanup_rate` 统计口径，PRD 默认 30s）。
    pub chunk_acked_to_cleaned_sla_ms: u64,

    /// 回放抽检样本数（用于 `replay_clip_sample_pass_rate`，PRD 默认 10）。
    pub replay_sample_total: usize,
    pub uplink_manager_path: String,
    pub esp32_ota_manager_path: String,
    pub esp32_firmware_image_path: String,
    pub esp32_ota_leases_path: String,
    pub esp32_ota_wifi_if: String,
    pub esp32_ota_mac_prefixes: Vec<String>,
    pub iphone_stereo_extrinsic_path: String,
    pub wifi_stereo_extrinsic_path: String,
    pub stereo_runtime_calibration_path: String,
    pub stereo_calibration_solver_path: String,
    pub stereo_calibration_python_bin: String,
    pub ui_dist_dir: String,
    pub stereo_preview_path: String,
    pub stereo_left_frame_path: String,
    pub stereo_right_frame_path: String,
    pub stereo_watchdog_status_path: String,
    pub sensing_proxy_base: String,
    pub sim_control_proxy_base: String,
    pub replay_proxy_base: String,
    pub isaac_runtime_base_url: String,
    pub runtime_profile: EdgeRuntimeProfile,
    pub phone_ingest_enabled: bool,
    pub stereo_enabled: bool,
    pub wifi_enabled: bool,
    pub fusion_enabled: bool,
    pub control_enabled: bool,
    pub sim_enabled: bool,
    pub vlm_indexing_enabled: bool,
    pub preview_generation_enabled: bool,
    pub vlm_model_id: String,
    pub vlm_fallback_model_id: String,
    pub vlm_prompt_version: String,
    pub vlm_keyframe_interval_ms: u64,
    pub vlm_segment_window_ms: u64,
    pub vlm_event_trigger_enabled: bool,
    pub vlm_event_trigger_camera_mode_change_enabled: bool,
    pub vlm_sidecar_base: String,
    pub vlm_sidecar_path: String,
    pub vlm_sidecar_python_bin: String,
    pub vlm_sidecar_autostart: bool,
    pub vlm_primary_model_path: String,
    pub vlm_fallback_model_path: String,
    pub vlm_runtime_device: String,
    pub vlm_edge_longest_side_px: u32,
    pub vlm_edge_image_seq_len: u32,
    pub vlm_disable_image_splitting: bool,
    pub vlm_inference_timeout_ms: u64,
    pub vlm_auto_fallback_latency_ms: u64,
    pub vlm_auto_fallback_cooldown_ms: u64,
    pub vlm_max_consecutive_failures: u32,
    pub preview_clip_max_frames: usize,
    pub preview_clip_frame_delay_ms: u64,
    pub crowd_upload_enabled: bool,
    pub crowd_upload_policy_mode: CrowdUploadPolicyMode,
    pub crowd_upload_poll_ms: u64,
    pub crowd_upload_control_base_url: String,
    pub crowd_upload_artifact_url: String,
    pub crowd_upload_token: Option<String>,
    pub crowd_upload_scope_token: Option<String>,
    pub crowd_upload_max_retry_count: u32,
    pub crowd_upload_uploading_stale_ms: u64,
    pub crowd_upload_receipt_source: String,
    pub phone_vision_service_base: String,
    pub phone_vision_service_path: String,
    pub phone_vision_python_bin: String,
    pub phone_vision_service_autostart: bool,
    pub phone_vision_processing_enabled: bool,
    pub zero_shot_auto_apply_enabled: bool,
    pub zero_shot_auto_apply_min_improvement_mm: f32,
    pub zero_shot_auto_rollback_enabled: bool,
    pub few_shot_evaluator_inbox_dir: String,
    pub few_shot_benchmark_capture_dir: String,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let data_dir = env::var("EDGE_DATA_DIR").unwrap_or_else(|_| "./data/ruview".to_string());
        let runtime_profile =
            env_runtime_profile("EDGE_RUNTIME_PROFILE", EdgeRuntimeProfile::TeleopFullstack)?;
        let crowd_upload_policy_mode = env_upload_policy_mode(
            "EDGE_CROWD_UPLOAD_POLICY_MODE",
            CrowdUploadPolicyMode::FullRawMirror,
        )?;
        let default_flags = runtime_profile.default_feature_flags();
        let preview_generation_default = default_flags.preview_generation_enabled
            || matches!(
                crowd_upload_policy_mode,
                CrowdUploadPolicyMode::MetadataPlusPreview
            );
        Ok(Self {
            http_addr: env::var("EDGE_HTTP_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string()),
            ws_addr: env::var("EDGE_WS_ADDR").unwrap_or_else(|_| "0.0.0.0:8765".to_string()),

            data_dir: data_dir.clone(),

            edge_token: env::var("EDGE_TOKEN").ok().filter(|v| !v.is_empty()),
            metrics_public: env_bool("METRICS_PUBLIC", false),

            protocol_pin_path: env::var("TELEOP_PROTOCOL_PIN_PATH")
                .unwrap_or_else(|_| "protocol_pin.json".to_string()),

            operator_frame: env::var("OPERATOR_FRAME")
                .unwrap_or_else(|_| "operator_frame".to_string()),
            robot_base_frame: env::var("ROBOT_BASE_FRAME")
                .unwrap_or_else(|_| "robot_base_frame".to_string()),
            extrinsic_version: env::var("EXTRINSIC_VERSION").unwrap_or_else(|_| "".to_string()),
            default_robot_type: env::var("DEFAULT_ROBOT_TYPE")
                .unwrap_or_else(|_| "G1_29".to_string()),
            default_end_effector_type: env::var("DEFAULT_END_EFFECTOR_TYPE")
                .unwrap_or_else(|_| "LEAP_V2".to_string()),

            deadman_enabled_default: env_bool("DEADMAN_ENABLED", true),
            deadman_timeout_ms: env_u64("DEADMAN_TIMEOUT_MS", 3000)?,
            time_sync_ok_window_ms: env_u64("TIME_SYNC_OK_WINDOW_MS", 5_000)?,
            time_sync_rtt_ok_ms: env_u64("TIME_SYNC_RTT_OK_MS", 20)?,

            csi_udp_bind: env::var("CSI_UDP_BIND").unwrap_or_else(|_| "0.0.0.0:5005".to_string()),
            csi_udp_mirror_addr: env::var("CSI_UDP_MIRROR_ADDR")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            csi_stale_ms: env_u64("CSI_STALE_MS", 500)?,
            csi_chunk_bucket_size: env_u64("CSI_CHUNK_BUCKET_SIZE", 1000)?,
            csi_chunk_max_packets: env_u64("CSI_CHUNK_MAX_PACKETS", 64)?,
            csi_chunk_max_bytes: env_u64("CSI_CHUNK_MAX_BYTES", 256 * 1024)?,
            csi_chunk_max_span_ms: env_u64("CSI_CHUNK_MAX_SPAN_MS", 250)?,
            vision_stale_ms: env_u64("VISION_STALE_MS", 6_000)?,
            allow_simulated_capture: env_bool("EDGE_ALLOW_SIMULATED_CAPTURE", false),
            stereo_stale_ms: env_u64("STEREO_STALE_MS", 3_000)?,
            wifi_pose_stale_ms: env_u64("WIFI_POSE_STALE_MS", 300)?,
            wifi_tracked_pose_direct_enabled: env_bool("WIFI_TRACKED_POSE_DIRECT_ENABLED", false),
            wifi_tracked_pose_poll_ms: env_u64("WIFI_TRACKED_POSE_POLL_MS", 100)?,
            operator_hold_ms: env_u64("OPERATOR_HOLD_MS", 200)?,

            vision_proj_x_span_m: env_f32("VISION_PROJ_X_SPAN_M", 0.8)?,
            vision_proj_y_span_m: env_f32("VISION_PROJ_Y_SPAN_M", 1.6)?,
            vision_proj_z_base_m: env_f32("VISION_PROJ_Z_BASE_M", 0.8)?,

            extrinsic_translation_m: env_f32_3("EXTRINSIC_TRANSLATION_M", [0.0, 0.0, 0.0])?,
            extrinsic_rotation_quat: env_f32_4("EXTRINSIC_ROTATION_QUAT", [0.0, 0.0, 0.0, 1.0])?,

            motion_max_speed_m_s: env_f32("MOTION_MAX_SPEED_M_S", 2.4)?,
            motion_max_accel_m_s2: env_f32("MOTION_MAX_ACCEL_M_S2", 20.0)?,
            motion_max_jerk_m_s3: env_f32("MOTION_MAX_JERK_M_S3", 800.0)?,

            fusion_publish_hz: env_u32("FUSION_PUBLISH_HZ", 10)?,
            teleop_publish_hz: env_u32("TELEOP_PUBLISH_HZ", 50)?,

            fusion_broadcast_capacity: env_usize("FUSION_BROADCAST_CAPACITY", 128)?,
            teleop_broadcast_capacity: env_usize("TELEOP_BROADCAST_CAPACITY", 256)?,
            chunk_ack_broadcast_capacity: env_usize("CHUNK_ACK_BROADCAST_CAPACITY", 128)?,

            bridge_stale_ms: env_u64("BRIDGE_STALE_MS", 8_000)?,

            upload_required_file_types: env_csv("UPLOAD_REQUIRED_FILE_TYPES", "csv,det"),

            chunk_acked_to_cleaned_sla_ms: env_u64("CHUNK_ACKED_TO_CLEANED_SLA_MS", 30_000)?,

            replay_sample_total: env_usize("REPLAY_SAMPLE_TOTAL", 10)?,
            uplink_manager_path: env::var("UPLINK_MANAGER_PATH")
                .unwrap_or_else(|_| "../chek-edge-debug/scripts/uplink_manager.py".to_string()),
            esp32_ota_manager_path: env::var("ESP32_OTA_MANAGER_PATH")
                .unwrap_or_else(|_| "../chek-edge-debug/scripts/esp32_ota_manager.py".to_string()),
            esp32_firmware_image_path: env::var("ESP32_FIRMWARE_IMAGE_PATH").unwrap_or_else(|_| {
                "../RuView/firmware/esp32-csi-node/build/esp32-csi-node.bin".to_string()
            }),
            esp32_ota_leases_path: env::var("ESP32_OTA_LEASES_PATH")
                .unwrap_or_else(|_| "/var/lib/misc/dnsmasq.leases".to_string()),
            esp32_ota_wifi_if: env::var("ESP32_OTA_WIFI_IF")
                .unwrap_or_else(|_| "wlP1p1s0".to_string()),
            esp32_ota_mac_prefixes: env_csv(
                "ESP32_OTA_MAC_PREFIXES",
                "e8:3d:c1,1c:db:d4,3c:0f:02,24:6f:28,30:c6:f7,7c:df:a1,84:f7:03,cc:db:a7,ac:15:18",
            ),
            iphone_stereo_extrinsic_path: path_safety::validate_fs_path(
                &env::var("IPHONE_STEREO_EXTRINSIC_PATH").unwrap_or_else(|_| {
                    format!("{}/runtime/iphone_stereo_extrinsic.json", data_dir)
                }),
                "IPHONE_STEREO_EXTRINSIC_PATH",
            )
            .map_err(|value| ConfigError::Value {
                key: "IPHONE_STEREO_EXTRINSIC_PATH".to_string(),
                value,
            })?,
            wifi_stereo_extrinsic_path: path_safety::validate_fs_path(
                &env::var("WIFI_STEREO_EXTRINSIC_PATH")
                    .unwrap_or_else(|_| format!("{}/runtime/wifi_stereo_extrinsic.json", data_dir)),
                "WIFI_STEREO_EXTRINSIC_PATH",
            )
            .map_err(|value| ConfigError::Value {
                key: "WIFI_STEREO_EXTRINSIC_PATH".to_string(),
                value,
            })?,
            stereo_runtime_calibration_path: path_safety::validate_fs_path(
                &env::var("STEREO_RUNTIME_CALIBRATION_PATH").unwrap_or_else(|_| {
                    format!("{}/runtime/stereo_pair_calibration.json", data_dir)
                }),
                "STEREO_RUNTIME_CALIBRATION_PATH",
            )
            .map_err(|value| ConfigError::Value {
                key: "STEREO_RUNTIME_CALIBRATION_PATH".to_string(),
                value,
            })?,
            stereo_calibration_solver_path: env::var("EDGE_STEREO_CALIBRATION_SOLVER_PATH")
                .unwrap_or_else(|_| "../scripts/stereo_calibration_solver.py".to_string()),
            stereo_calibration_python_bin: env::var("EDGE_STEREO_CALIBRATION_PYTHON_BIN")
                .unwrap_or_else(|_| "python3".to_string()),
            ui_dist_dir: env::var("EDGE_UI_DIST_DIR")
                .unwrap_or_else(|_| "../RuView/ui-react/dist".to_string()),
            stereo_preview_path: env::var("EDGE_STEREO_PREVIEW_PATH")
                .unwrap_or_else(|_| "/tmp/stereo-uvc-preview.jpg".to_string()),
            stereo_left_frame_path: env::var("EDGE_STEREO_LEFT_FRAME_PATH")
                .unwrap_or_else(|_| "/tmp/stereo-uvc-left.jpg".to_string()),
            stereo_right_frame_path: env::var("EDGE_STEREO_RIGHT_FRAME_PATH")
                .unwrap_or_else(|_| "/tmp/stereo-uvc-right.jpg".to_string()),
            stereo_watchdog_status_path: env::var("EDGE_STEREO_WATCHDOG_STATUS_PATH")
                .unwrap_or_else(|_| "/run/chek-edge/stereo-watchdog.json".to_string()),
            sensing_proxy_base: env::var("EDGE_SENSING_PROXY_BASE")
                .unwrap_or_else(|_| "http://127.0.0.1:18080".to_string()),
            sim_control_proxy_base: env::var("EDGE_SIM_CONTROL_PROXY_BASE")
                .unwrap_or_else(|_| "http://127.0.0.1:3011".to_string()),
            replay_proxy_base: env::var("EDGE_REPLAY_PROXY_BASE")
                .unwrap_or_else(|_| "http://127.0.0.1:3020".to_string()),
            isaac_runtime_base_url: env::var("EDGE_ISAAC_RUNTIME_BASE_URL").unwrap_or_default(),
            runtime_profile,
            phone_ingest_enabled: env_bool(
                "EDGE_PHONE_INGEST_ENABLED",
                default_flags.phone_ingest_enabled,
            ),
            stereo_enabled: env_bool("EDGE_STEREO_ENABLED", default_flags.stereo_enabled),
            wifi_enabled: env_bool("EDGE_WIFI_ENABLED", default_flags.wifi_enabled),
            fusion_enabled: env_bool("EDGE_FUSION_ENABLED", default_flags.fusion_enabled),
            control_enabled: env_bool("EDGE_CONTROL_ENABLED", default_flags.control_enabled),
            sim_enabled: env_bool("EDGE_SIM_ENABLED", default_flags.sim_enabled),
            vlm_indexing_enabled: env_bool(
                "EDGE_VLM_INDEXING_ENABLED",
                default_flags.vlm_indexing_enabled,
            ),
            preview_generation_enabled: env_bool(
                "EDGE_PREVIEW_GENERATION_ENABLED",
                preview_generation_default,
            ),
            vlm_model_id: env::var("EDGE_VLM_MODEL_ID").unwrap_or_else(|_| {
                if default_flags.vlm_indexing_enabled {
                    "SmolVLM2-500M".to_string()
                } else {
                    "edge_semantic_disabled".to_string()
                }
            }),
            vlm_fallback_model_id: env::var("EDGE_VLM_FALLBACK_MODEL_ID")
                .unwrap_or_else(|_| "SmolVLM2-256M".to_string()),
            vlm_prompt_version: env::var("EDGE_VLM_PROMPT_VERSION")
                .unwrap_or_else(|_| "edge_semantic_v1".to_string()),
            vlm_keyframe_interval_ms: env_u64("EDGE_VLM_KEYFRAME_INTERVAL_MS", 3_000)?,
            vlm_segment_window_ms: env_u64("EDGE_VLM_SEGMENT_WINDOW_MS", 15_000)?,
            vlm_event_trigger_enabled: env_bool("EDGE_VLM_EVENT_TRIGGER_ENABLED", true),
            vlm_event_trigger_camera_mode_change_enabled: env_bool(
                "EDGE_VLM_EVENT_TRIGGER_CAMERA_MODE_CHANGE_ENABLED",
                true,
            ),
            vlm_sidecar_base: env::var("EDGE_VLM_SIDECAR_BASE").unwrap_or_else(|_| "".to_string()),
            vlm_sidecar_path: env::var("EDGE_VLM_SIDECAR_PATH")
                .unwrap_or_else(|_| "../scripts/edge_vlm_sidecar.py".to_string()),
            vlm_sidecar_python_bin: env::var("EDGE_VLM_SIDECAR_PYTHON_BIN")
                .unwrap_or_else(|_| "python3".to_string()),
            vlm_sidecar_autostart: env_bool("EDGE_VLM_SIDECAR_AUTOSTART", false),
            vlm_primary_model_path: env::var("EDGE_VLM_PRIMARY_MODEL_PATH")
                .unwrap_or_else(|_| "".to_string()),
            vlm_fallback_model_path: env::var("EDGE_VLM_FALLBACK_MODEL_PATH")
                .unwrap_or_else(|_| "".to_string()),
            vlm_runtime_device: env::var("EDGE_VLM_RUNTIME_DEVICE")
                .unwrap_or_else(|_| "auto".to_string()),
            vlm_edge_longest_side_px: env_u32("EDGE_VLM_EDGE_LONGEST_SIDE_PX", 256)?,
            vlm_edge_image_seq_len: env_u32("EDGE_VLM_EDGE_IMAGE_SEQ_LEN", 16)?,
            vlm_disable_image_splitting: env_bool("EDGE_VLM_DISABLE_IMAGE_SPLITTING", true),
            vlm_inference_timeout_ms: env_u64("EDGE_VLM_INFERENCE_TIMEOUT_MS", 3_500)?,
            vlm_auto_fallback_latency_ms: env_u64("EDGE_VLM_AUTO_FALLBACK_LATENCY_MS", 2_200)?,
            vlm_auto_fallback_cooldown_ms: env_u64("EDGE_VLM_AUTO_FALLBACK_COOLDOWN_MS", 60_000)?,
            vlm_max_consecutive_failures: env_u32("EDGE_VLM_MAX_CONSECUTIVE_FAILURES", 2)?,
            preview_clip_max_frames: env_usize("EDGE_PREVIEW_CLIP_MAX_FRAMES", 8)?,
            preview_clip_frame_delay_ms: env_u64("EDGE_PREVIEW_CLIP_FRAME_DELAY_MS", 400)?,
            crowd_upload_enabled: env_bool("EDGE_CROWD_UPLOAD_ENABLED", false),
            crowd_upload_policy_mode,
            crowd_upload_poll_ms: env_u64("EDGE_CROWD_UPLOAD_POLL_MS", 5_000)?,
            crowd_upload_control_base_url: env::var("EDGE_CROWD_UPLOAD_CONTROL_BASE_URL")
                .unwrap_or_else(|_| "".to_string()),
            crowd_upload_artifact_url: env::var("EDGE_CROWD_UPLOAD_ARTIFACT_URL")
                .unwrap_or_else(|_| "".to_string()),
            crowd_upload_token: env::var("EDGE_CROWD_UPLOAD_TOKEN")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            crowd_upload_scope_token: env::var("EDGE_CROWD_UPLOAD_SCOPE_TOKEN")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            crowd_upload_max_retry_count: env_u32("EDGE_CROWD_UPLOAD_MAX_RETRY_COUNT", 5)?,
            crowd_upload_uploading_stale_ms: env_u64(
                "EDGE_CROWD_UPLOAD_UPLOADING_STALE_MS",
                60_000,
            )?,
            crowd_upload_receipt_source: env::var("EDGE_CROWD_UPLOAD_RECEIPT_SOURCE")
                .unwrap_or_else(|_| "edge_uploader_worker".to_string()),
            phone_vision_service_base: env::var("EDGE_PHONE_VISION_SERVICE_BASE")
                .unwrap_or_else(|_| "http://127.0.0.1:3031".to_string()),
            phone_vision_service_path: env::var("EDGE_PHONE_VISION_SERVICE_PATH")
                .unwrap_or_else(|_| "../scripts/edge_phone_vision_service.py".to_string()),
            phone_vision_python_bin: env::var("EDGE_PHONE_VISION_PYTHON_BIN")
                .unwrap_or_else(|_| "python3".to_string()),
            phone_vision_service_autostart: env_bool("EDGE_PHONE_VISION_SERVICE_AUTOSTART", false),
            phone_vision_processing_enabled: env_bool("EDGE_PHONE_VISION_PROCESSING_ENABLED", true),
            zero_shot_auto_apply_enabled: env_bool("EDGE_ZERO_SHOT_AUTO_APPLY_ENABLED", false),
            zero_shot_auto_apply_min_improvement_mm: env_f32(
                "EDGE_ZERO_SHOT_AUTO_APPLY_MIN_IMPROVEMENT_MM",
                0.0,
            )?,
            zero_shot_auto_rollback_enabled: env_bool(
                "EDGE_ZERO_SHOT_AUTO_ROLLBACK_ENABLED",
                false,
            ),
            few_shot_evaluator_inbox_dir: env::var("EDGE_FEW_SHOT_EVAL_INBOX_DIR").unwrap_or_else(
                |_| format!("{}/runtime/environment_evolution_few_shot_inbox", data_dir),
            ),
            few_shot_benchmark_capture_dir: env::var("EDGE_FEW_SHOT_BENCHMARK_CAPTURE_DIR")
                .unwrap_or_else(|_| "../chek-edge-debug/runtime-captures".to_string()),
        })
    }

    pub fn runtime_profile_name(&self) -> &'static str {
        self.runtime_profile.as_str()
    }

    pub fn upload_policy_mode_name(&self) -> &'static str {
        self.crowd_upload_policy_mode.as_str()
    }

    pub fn runtime_feature_flags(&self) -> RuntimeFeatureFlags {
        RuntimeFeatureFlags {
            phone_ingest_enabled: self.phone_ingest_enabled,
            stereo_enabled: self.stereo_enabled,
            wifi_enabled: self.wifi_enabled,
            fusion_enabled: self.fusion_enabled,
            control_enabled: self.control_enabled,
            sim_enabled: self.sim_enabled,
            vlm_indexing_enabled: self.vlm_indexing_enabled,
            preview_generation_enabled: self.preview_generation_enabled,
        }
    }

    pub fn default_session_mode(&self) -> &'static str {
        if self.fusion_enabled {
            "fused"
        } else if self.wifi_enabled {
            "csi_only"
        } else {
            "vision_only"
        }
    }

    pub fn raw_residency_default(&self) -> &'static str {
        match self.crowd_upload_policy_mode {
            CrowdUploadPolicyMode::MetadataOnly | CrowdUploadPolicyMode::MetadataPlusPreview => {
                "edge_only"
            }
            CrowdUploadPolicyMode::FullRawMirror => "cloud_mirrored",
        }
    }

    pub fn preview_residency_default(&self) -> &'static str {
        if !self.preview_generation_enabled {
            return "unavailable";
        }
        match self.crowd_upload_policy_mode {
            CrowdUploadPolicyMode::MetadataOnly => "edge_only",
            CrowdUploadPolicyMode::MetadataPlusPreview => "cloud_preview_only",
            CrowdUploadPolicyMode::FullRawMirror => "cloud_mirrored",
        }
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    match env::var(key) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

fn env_runtime_profile(
    key: &str,
    default: EdgeRuntimeProfile,
) -> Result<EdgeRuntimeProfile, ConfigError> {
    match env::var(key) {
        Ok(v) => EdgeRuntimeProfile::parse(&v).ok_or_else(|| ConfigError::Value {
            key: key.to_string(),
            value: v,
        }),
        Err(_) => Ok(default),
    }
}

fn env_upload_policy_mode(
    key: &str,
    default: CrowdUploadPolicyMode,
) -> Result<CrowdUploadPolicyMode, ConfigError> {
    match env::var(key) {
        Ok(v) => CrowdUploadPolicyMode::parse(&v).ok_or_else(|| ConfigError::Value {
            key: key.to_string(),
            value: v,
        }),
        Err(_) => Ok(default),
    }
}

fn env_u64(key: &str, default: u64) -> Result<u64, ConfigError> {
    match env::var(key) {
        Ok(v) => v.parse::<u64>().map_err(|_| ConfigError::Int {
            key: key.to_string(),
            value: v,
        }),
        Err(_) => Ok(default),
    }
}

fn env_f32(key: &str, default: f32) -> Result<f32, ConfigError> {
    match env::var(key) {
        Ok(v) => v.parse::<f32>().map_err(|_| ConfigError::Float {
            key: key.to_string(),
            value: v,
        }),
        Err(_) => Ok(default),
    }
}

fn env_u32(key: &str, default: u32) -> Result<u32, ConfigError> {
    match env::var(key) {
        Ok(v) => v.parse::<u32>().map_err(|_| ConfigError::Int {
            key: key.to_string(),
            value: v,
        }),
        Err(_) => Ok(default),
    }
}

fn env_usize(key: &str, default: usize) -> Result<usize, ConfigError> {
    match env::var(key) {
        Ok(v) => v.parse::<usize>().map_err(|_| ConfigError::Int {
            key: key.to_string(),
            value: v,
        }),
        Err(_) => Ok(default),
    }
}

fn env_f32_list<const N: usize>(key: &str, default: [f32; N]) -> Result<[f32; N], ConfigError> {
    let raw = match env::var(key) {
        Ok(v) => v,
        Err(_) => return Ok(default),
    };
    let parts: Vec<&str> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != N {
        return Err(ConfigError::FloatList {
            key: key.to_string(),
            value: raw,
        });
    }
    let mut out = [0.0f32; N];
    for (idx, p) in parts.into_iter().enumerate() {
        let v = p.parse::<f32>().map_err(|_| ConfigError::FloatList {
            key: key.to_string(),
            value: raw.clone(),
        })?;
        out[idx] = v;
    }
    Ok(out)
}

fn env_f32_3(key: &str, default: [f32; 3]) -> Result<[f32; 3], ConfigError> {
    env_f32_list::<3>(key, default)
}

fn env_f32_4(key: &str, default: [f32; 4]) -> Result<[f32; 4], ConfigError> {
    env_f32_list::<4>(key, default)
}

fn env_csv(key: &str, default_csv: &str) -> Vec<String> {
    let raw = env::var(key).unwrap_or_else(|_| default_csv.to_string());
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}
