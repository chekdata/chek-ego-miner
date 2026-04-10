use std::env;
use std::fs;

use serde::Deserialize;
use thiserror::Error;

use crate::bridge::retarget::LEAP_COMMAND_DIM;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("读取配置文件失败: {0}")]
    ReadFile(String),
    #[error("解析 TOML 失败: {0}")]
    InvalidToml(String),
    #[error("无效配置: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Deserialize)]
pub struct FileConfig {
    pub edge_teleop_ws_url: Option<String>,
    pub edge_token: Option<String>,
    pub bridge_id: Option<String>,

    pub publish_hz: Option<u32>,
    pub pairing_window_ms: Option<u64>,
    pub hold_timeout_ms: Option<u64>,
    pub freeze_timeout_ms: Option<u64>,
    pub keepalive_timeout_ms: Option<u64>,

    pub joint_min: Option<Vec<f32>>,
    pub joint_max: Option<Vec<f32>>,
    pub expected_joint_len: Option<usize>,
    pub left_joint_scale: Option<Vec<f32>>,
    pub left_joint_offset: Option<Vec<f32>>,
    pub right_joint_scale: Option<Vec<f32>>,
    pub right_joint_offset: Option<Vec<f32>>,

    pub max_temperature_c: Option<f32>,
    pub http_addr: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub protocol_pin_path: String,

    pub edge_teleop_ws_url: Option<String>,
    pub edge_token: Option<String>,
    pub bridge_id: String,

    pub publish_hz: u32,
    pub pairing_window_ms: u64,
    pub hold_timeout_ms: u64,
    pub freeze_timeout_ms: u64,
    pub keepalive_timeout_ms: u64,

    pub joint_min: Vec<f32>,
    pub joint_max: Vec<f32>,
    pub expected_joint_len: Option<usize>,
    pub left_joint_scale: Vec<f32>,
    pub left_joint_offset: Vec<f32>,
    pub right_joint_scale: Vec<f32>,
    pub right_joint_offset: Vec<f32>,

    pub max_temperature_c: f32,

    pub http_addr: String,
}

impl Config {
    pub fn load() -> Result<Self, ConfigError> {
        let config_path =
            env::var("LEAP_CONFIG_PATH").unwrap_or_else(|_| "config/leap.toml".to_string());
        let protocol_pin_path = env::var("TELEOP_PROTOCOL_PIN_PATH")
            .unwrap_or_else(|_| "protocol_pin.json".to_string());

        let raw = fs::read_to_string(&config_path)
            .map_err(|e| ConfigError::ReadFile(format!("{config_path}: {e}")))?;
        let file_cfg: FileConfig =
            toml::from_str(&raw).map_err(|e| ConfigError::InvalidToml(e.to_string()))?;

        let edge_teleop_ws_url = sanitize_opt(env_override_opt(
            "EDGE_TELEOP_WS_URL",
            file_cfg.edge_teleop_ws_url,
        ));
        let edge_token = sanitize_opt(env_override_opt("EDGE_TOKEN", file_cfg.edge_token));
        let bridge_id = env::var("BRIDGE_ID")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or(file_cfg.bridge_id)
            .unwrap_or_else(|| "leap-bridge-01".to_string());

        let publish_hz = env_u32("PUBLISH_HZ").unwrap_or(file_cfg.publish_hz.unwrap_or(100));
        let pairing_window_ms =
            env_u64("PAIRING_WINDOW_MS").unwrap_or(file_cfg.pairing_window_ms.unwrap_or(20));
        let hold_timeout_ms =
            env_u64("HOLD_TIMEOUT_MS").unwrap_or(file_cfg.hold_timeout_ms.unwrap_or(200));
        let freeze_timeout_ms =
            env_u64("FREEZE_TIMEOUT_MS").unwrap_or(file_cfg.freeze_timeout_ms.unwrap_or(200));
        let keepalive_timeout_ms =
            env_u64("KEEPALIVE_TIMEOUT_MS").unwrap_or(file_cfg.keepalive_timeout_ms.unwrap_or(200));

        if publish_hz == 0 {
            return Err(ConfigError::Invalid("publish_hz 必须大于 0".to_string()));
        }
        if pairing_window_ms == 0 {
            return Err(ConfigError::Invalid(
                "pairing_window_ms 必须大于 0".to_string(),
            ));
        }
        if hold_timeout_ms == 0 || freeze_timeout_ms == 0 || keepalive_timeout_ms == 0 {
            return Err(ConfigError::Invalid(
                "hold/freeze/keepalive timeout 必须大于 0".to_string(),
            ));
        }

        let expected_joint_len = env_usize("EXPECTED_JOINT_LEN")
            .map(|v| if v == 0 { None } else { Some(v) })
            .unwrap_or_else(|| match file_cfg.expected_joint_len {
                Some(0) => None,
                Some(v) => Some(v),
                None => Some(16),
            });

        let joint_min = file_cfg.joint_min.unwrap_or_default();
        let joint_max = file_cfg.joint_max.unwrap_or_default();
        let left_joint_scale = file_cfg.left_joint_scale.unwrap_or_default();
        let left_joint_offset = file_cfg.left_joint_offset.unwrap_or_default();
        let right_joint_scale = file_cfg.right_joint_scale.unwrap_or_default();
        let right_joint_offset = file_cfg.right_joint_offset.unwrap_or_default();
        if (!joint_min.is_empty() || !joint_max.is_empty()) && joint_min.len() != joint_max.len() {
            return Err(ConfigError::Invalid(
                "joint_min 与 joint_max 长度必须一致（或都为空）".to_string(),
            ));
        }
        if let Some(len) = expected_joint_len {
            if !joint_min.is_empty() && joint_min.len() != len {
                return Err(ConfigError::Invalid(
                    "expected_joint_len 与 joint_min 长度不一致".to_string(),
                ));
            }
            if !joint_max.is_empty() && joint_max.len() != len {
                return Err(ConfigError::Invalid(
                    "expected_joint_len 与 joint_max 长度不一致".to_string(),
                ));
            }
        }
        validate_calibration_vec("left_joint_scale", &left_joint_scale)?;
        validate_calibration_vec("left_joint_offset", &left_joint_offset)?;
        validate_calibration_vec("right_joint_scale", &right_joint_scale)?;
        validate_calibration_vec("right_joint_offset", &right_joint_offset)?;

        let max_temperature_c =
            env_f32("MAX_TEMPERATURE_C").unwrap_or(file_cfg.max_temperature_c.unwrap_or(70.0));
        let http_addr = env::var("HTTP_ADDR")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or(file_cfg.http_addr)
            .unwrap_or_else(|| "0.0.0.0:8090".to_string());

        Ok(Self {
            protocol_pin_path,
            edge_teleop_ws_url,
            edge_token,
            bridge_id,
            publish_hz,
            pairing_window_ms,
            hold_timeout_ms,
            freeze_timeout_ms,
            keepalive_timeout_ms,
            joint_min,
            joint_max,
            expected_joint_len,
            left_joint_scale,
            left_joint_offset,
            right_joint_scale,
            right_joint_offset,
            max_temperature_c,
            http_addr,
        })
    }
}

fn sanitize_opt(v: Option<String>) -> Option<String> {
    v.and_then(|s| {
        let t = s.trim().to_string();
        if t.is_empty() {
            None
        } else {
            Some(t)
        }
    })
}

fn env_override_opt(key: &str, fallback: Option<String>) -> Option<String> {
    env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or(fallback)
}

fn env_u32(key: &str) -> Option<u32> {
    env::var(key).ok().and_then(|v| v.parse::<u32>().ok())
}

fn env_u64(key: &str) -> Option<u64> {
    env::var(key).ok().and_then(|v| v.parse::<u64>().ok())
}

fn env_usize(key: &str) -> Option<usize> {
    env::var(key).ok().and_then(|v| v.parse::<usize>().ok())
}

fn env_f32(key: &str) -> Option<f32> {
    env::var(key).ok().and_then(|v| v.parse::<f32>().ok())
}

fn validate_calibration_vec(name: &str, values: &[f32]) -> Result<(), ConfigError> {
    if !values.is_empty() && values.len() != LEAP_COMMAND_DIM {
        return Err(ConfigError::Invalid(format!(
            "{name} 长度必须为 {LEAP_COMMAND_DIM}（或留空）"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ConfigError::Invalid(format!("{name} 不能包含 NaN/Inf")));
    }
    Ok(())
}
