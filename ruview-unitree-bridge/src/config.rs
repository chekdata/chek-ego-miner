use std::env;
use std::fs;

use serde::Deserialize;
use thiserror::Error;

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
    pub keepalive_timeout_ms: Option<u64>,

    pub expected_arm_joint_len: Option<usize>,
    pub expected_dex_joint_len: Option<usize>,

    pub arm_joint_min: Option<Vec<f32>>,
    pub arm_joint_max: Option<Vec<f32>>,
    pub dex_joint_min: Option<Vec<f32>>,
    pub dex_joint_max: Option<Vec<f32>>,

    pub http_addr: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub protocol_pin_path: String,

    pub edge_teleop_ws_url: Option<String>,
    pub edge_token: Option<String>,
    pub bridge_id: String,

    pub publish_hz: u32,
    pub keepalive_timeout_ms: u64,

    pub expected_arm_joint_len: Option<usize>,
    pub expected_dex_joint_len: Option<usize>,

    pub arm_joint_min: Vec<f32>,
    pub arm_joint_max: Vec<f32>,
    pub dex_joint_min: Vec<f32>,
    pub dex_joint_max: Vec<f32>,

    pub http_addr: String,
}

impl Config {
    pub fn load() -> Result<Self, ConfigError> {
        let config_path =
            env::var("UNITREE_CONFIG_PATH").unwrap_or_else(|_| "config/unitree.toml".to_string());
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
            .unwrap_or_else(|| "unitree-bridge-01".to_string());

        let publish_hz = env_u32("PUBLISH_HZ").unwrap_or(file_cfg.publish_hz.unwrap_or(100));
        let keepalive_timeout_ms =
            env_u64("KEEPALIVE_TIMEOUT_MS").unwrap_or(file_cfg.keepalive_timeout_ms.unwrap_or(200));

        if publish_hz == 0 {
            return Err(ConfigError::Invalid("publish_hz 必须大于 0".to_string()));
        }
        if keepalive_timeout_ms == 0 {
            return Err(ConfigError::Invalid(
                "keepalive_timeout_ms 必须大于 0".to_string(),
            ));
        }

        let expected_arm_joint_len = env_usize("EXPECTED_ARM_JOINT_LEN")
            .map(|v| if v == 0 { None } else { Some(v) })
            .unwrap_or_else(|| match file_cfg.expected_arm_joint_len.unwrap_or(0) {
                0 => None,
                v => Some(v),
            });
        let expected_dex_joint_len = env_usize("EXPECTED_DEX_JOINT_LEN")
            .map(|v| if v == 0 { None } else { Some(v) })
            .unwrap_or_else(|| match file_cfg.expected_dex_joint_len.unwrap_or(0) {
                0 => None,
                v => Some(v),
            });

        let arm_joint_min = file_cfg.arm_joint_min.unwrap_or_default();
        let arm_joint_max = file_cfg.arm_joint_max.unwrap_or_default();
        if (!arm_joint_min.is_empty() || !arm_joint_max.is_empty())
            && arm_joint_min.len() != arm_joint_max.len()
        {
            return Err(ConfigError::Invalid(
                "arm_joint_min 与 arm_joint_max 长度必须一致（或都为空）".to_string(),
            ));
        }
        let dex_joint_min = file_cfg.dex_joint_min.unwrap_or_default();
        let dex_joint_max = file_cfg.dex_joint_max.unwrap_or_default();
        if (!dex_joint_min.is_empty() || !dex_joint_max.is_empty())
            && dex_joint_min.len() != dex_joint_max.len()
        {
            return Err(ConfigError::Invalid(
                "dex_joint_min 与 dex_joint_max 长度必须一致（或都为空）".to_string(),
            ));
        }

        if let Some(len) = expected_arm_joint_len {
            if !arm_joint_min.is_empty() && arm_joint_min.len() != len {
                return Err(ConfigError::Invalid(
                    "expected_arm_joint_len 与 arm_joint_min 长度不一致".to_string(),
                ));
            }
            if !arm_joint_max.is_empty() && arm_joint_max.len() != len {
                return Err(ConfigError::Invalid(
                    "expected_arm_joint_len 与 arm_joint_max 长度不一致".to_string(),
                ));
            }
        }
        if let Some(len) = expected_dex_joint_len {
            if !dex_joint_min.is_empty() && dex_joint_min.len() != len {
                return Err(ConfigError::Invalid(
                    "expected_dex_joint_len 与 dex_joint_min 长度不一致".to_string(),
                ));
            }
            if !dex_joint_max.is_empty() && dex_joint_max.len() != len {
                return Err(ConfigError::Invalid(
                    "expected_dex_joint_len 与 dex_joint_max 长度不一致".to_string(),
                ));
            }
        }

        let http_addr = env::var("HTTP_ADDR")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or(file_cfg.http_addr)
            .unwrap_or_else(|| "0.0.0.0:8091".to_string());

        Ok(Self {
            protocol_pin_path,
            edge_teleop_ws_url,
            edge_token,
            bridge_id,
            publish_hz,
            keepalive_timeout_ms,
            expected_arm_joint_len,
            expected_dex_joint_len,
            arm_joint_min,
            arm_joint_max,
            dex_joint_min,
            dex_joint_max,
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
