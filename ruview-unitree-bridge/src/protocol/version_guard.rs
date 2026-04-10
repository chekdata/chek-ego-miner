use std::fs;
use std::path::Path;

use semver::Version;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtocolGuardError {
    #[error("读取协议钉住文件失败: {0}")]
    ReadFailed(String),
    #[error("协议钉住 JSON 无效: {0}")]
    InvalidJson(String),
    #[error("协议钉住 semver 无效: {0}")]
    InvalidSemver(String),
    #[error("协议版本 {version} 不在兼容区间 [{compat_min}, {compat_max}]")]
    Incompatible {
        version: String,
        compat_min: String,
        compat_max: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct ProtocolVersionInfo {
    pub name: String,
    pub version: String,
    pub schema_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProtocolPin {
    pub name: String,
    pub version: String,
    pub schema_sha256: String,
    pub compat_min: String,
    pub compat_max: String,
}

impl ProtocolPin {
    pub fn load_from_path(path: &str) -> Result<Self, ProtocolGuardError> {
        let raw = fs::read_to_string(Path::new(path))
            .map_err(|e| ProtocolGuardError::ReadFailed(format!("{path}: {e}")))?;
        serde_json::from_str::<Self>(&raw)
            .map_err(|e| ProtocolGuardError::InvalidJson(e.to_string()))
    }

    pub fn validate_and_to_info(&self) -> Result<ProtocolVersionInfo, ProtocolGuardError> {
        let v = Version::parse(&self.version)
            .map_err(|e| ProtocolGuardError::InvalidSemver(e.to_string()))?;
        let min = Version::parse(&self.compat_min)
            .map_err(|e| ProtocolGuardError::InvalidSemver(e.to_string()))?;
        let max = Version::parse(&self.compat_max)
            .map_err(|e| ProtocolGuardError::InvalidSemver(e.to_string()))?;

        if v < min || v > max {
            return Err(ProtocolGuardError::Incompatible {
                version: self.version.clone(),
                compat_min: self.compat_min.clone(),
                compat_max: self.compat_max.clone(),
            });
        }

        Ok(ProtocolVersionInfo {
            name: self.name.clone(),
            version: self.version.clone(),
            schema_sha256: self.schema_sha256.clone(),
        })
    }
}

pub fn emit_protocol_metrics(info: &ProtocolVersionInfo) {
    metrics::gauge!(
        "teleop_protocol_info",
        "name" => info.name.clone(),
        "version" => info.version.clone()
    )
    .set(1.0);
}
