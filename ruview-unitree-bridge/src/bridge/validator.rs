use thiserror::Error;

use crate::reason;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DexKind {
    Dex3,
    Dex1,
}

#[derive(Debug, Error)]
pub enum GuardError {
    #[error("end_effector_type 防呆拒绝: {0}")]
    Deny(&'static str),
}

#[derive(Clone, Debug)]
pub struct EndpointGuard {
    pub end_effector_type: String,
}

impl EndpointGuard {
    pub fn new(end_effector_type: impl Into<String>) -> Self {
        Self {
            end_effector_type: end_effector_type.into(),
        }
    }

    pub fn allow_dex(&self, kind: DexKind) -> bool {
        matches!(
            (self.end_effector_type.as_str(), kind),
            ("DEX3", DexKind::Dex3) | ("DEX1", DexKind::Dex1)
        )
    }

    pub fn ensure_dex_allowed(&self, kind: DexKind) -> Result<(), GuardError> {
        if self.allow_dex(kind) {
            return Ok(());
        }
        Err(GuardError::Deny(reason::REASON_ENDPOINT_GUARD_DENY))
    }
}
