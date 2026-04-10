use crate::bridge::parser::ParsedHandTargets;
use crate::bridge::types::{HandTargetFrame, PairedHandFrame, PairingDegrade};

#[derive(Clone, Copy, Debug)]
pub struct PairingConfig {
    pub pairing_window_ms: u64,
    pub hold_timeout_ms: u64,
    pub freeze_timeout_ms: u64,
}

pub struct PairingEngine {
    cfg: PairingConfig,
    left: Option<HandTargetFrame>,
    right: Option<HandTargetFrame>,
}

impl PairingEngine {
    pub fn new(cfg: PairingConfig) -> Self {
        Self {
            cfg,
            left: None,
            right: None,
        }
    }

    pub fn ingest(&mut self, parsed: &ParsedHandTargets) {
        if let Some(left) = parsed.left.clone() {
            self.left = Some(left);
        }
        if let Some(right) = parsed.right.clone() {
            self.right = Some(right);
        }
    }

    pub fn pair(&self) -> Option<PairedHandFrame> {
        let left = self.left.clone()?;
        let right = self.right.clone()?;

        let delta_ns = left.edge_time_ns.abs_diff(right.edge_time_ns);
        let window_ns = self.cfg.pairing_window_ms.saturating_mul(1_000_000);
        if delta_ns <= window_ns {
            return Some(PairedHandFrame {
                left,
                right,
                delta_ns,
                degrade: PairingDegrade::Normal,
            });
        }

        // 超出窗口：按 stale 时长决定 hold/freeze
        let stale_ms = delta_ns / 1_000_000;
        let degrade = if stale_ms < self.cfg.hold_timeout_ms {
            PairingDegrade::Hold
        } else if stale_ms >= self.cfg.freeze_timeout_ms {
            PairingDegrade::Freeze
        } else {
            PairingDegrade::Hold
        };

        Some(PairedHandFrame {
            left,
            right,
            delta_ns,
            degrade,
        })
    }
}
