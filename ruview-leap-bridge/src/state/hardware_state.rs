use crate::leap::client::{LeapDeviceStatus, LeapHandStatus};
use crate::reason;

/// bridge 侧硬件 readiness 评估结果（用于上报与门控）。
#[derive(Clone, Debug)]
pub struct HardwareReadiness {
    pub is_ready: bool,
    pub fault_code: String,
    pub fault_message: String,
}

impl HardwareReadiness {
    pub fn ok() -> Self {
        Self {
            is_ready: true,
            fault_code: "".to_string(),
            fault_message: "".to_string(),
        }
    }
}

/// LEAP 硬件状态模型（MVP：只做“当前快照 -> readiness”）。
#[derive(Clone, Debug)]
pub struct LeapHardwareState {
    max_temperature_c: f32,
    last: LeapDeviceStatus,
}

impl LeapHardwareState {
    pub fn new(max_temperature_c: f32) -> Self {
        Self {
            max_temperature_c,
            last: LeapDeviceStatus::default(),
        }
    }

    pub fn update(&mut self, status: LeapDeviceStatus) -> HardwareReadiness {
        self.last = status.clone();
        evaluate_readiness(&status, self.max_temperature_c)
    }

    pub fn last(&self) -> &LeapDeviceStatus {
        &self.last
    }
}

fn evaluate_readiness(status: &LeapDeviceStatus, max_temperature_c: f32) -> HardwareReadiness {
    if !status.left.online || !status.right.online {
        let mut msg = Vec::new();
        if !status.left.online {
            msg.push("left offline");
        }
        if !status.right.online {
            msg.push("right offline");
        }
        return HardwareReadiness {
            is_ready: false,
            fault_code: reason::REASON_HARDWARE_OFFLINE.to_string(),
            fault_message: msg.join(", "),
        };
    }

    if status.left.temperature_c > max_temperature_c
        || status.right.temperature_c > max_temperature_c
    {
        let mut msg = Vec::new();
        if status.left.temperature_c > max_temperature_c {
            msg.push(format!("left overtemp {:.1}C", status.left.temperature_c));
        }
        if status.right.temperature_c > max_temperature_c {
            msg.push(format!("right overtemp {:.1}C", status.right.temperature_c));
        }
        return HardwareReadiness {
            is_ready: false,
            fault_code: reason::REASON_HARDWARE_OVERTEMP.to_string(),
            fault_message: msg.join(", "),
        };
    }

    if has_error(&status.left) || has_error(&status.right) {
        let mut msg = Vec::new();
        if let Some(code) = status.left.error_code.as_deref().filter(|c| !c.is_empty()) {
            msg.push(format!("left error {code}"));
        }
        if let Some(code) = status.right.error_code.as_deref().filter(|c| !c.is_empty()) {
            msg.push(format!("right error {code}"));
        }
        return HardwareReadiness {
            is_ready: false,
            fault_code: reason::REASON_HARDWARE_ERROR.to_string(),
            fault_message: msg.join(", "),
        };
    }

    HardwareReadiness::ok()
}

fn has_error(hand: &LeapHandStatus) -> bool {
    hand.error_code
        .as_deref()
        .map(|c| !c.trim().is_empty())
        .unwrap_or(false)
}
