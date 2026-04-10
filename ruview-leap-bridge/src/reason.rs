/// 统一的 reason code（用于故障、拒绝原因、以及上报 bridge_state_packet）。
///
/// 约定：
/// - 使用 `snake_case`
/// - 便于日志、指标 label 与联调排障
pub const REASON_OK: &str = "ok";
pub const REASON_UNKNOWN: &str = "unknown";

pub const REASON_PROTOCOL_INVALID: &str = "protocol_invalid";
pub const REASON_SCHEMA_INVALID: &str = "schema_invalid";
pub const REASON_DIMENSION_MISMATCH: &str = "dimension_mismatch";
pub const REASON_NAN_OR_INF: &str = "nan_or_inf";

pub const REASON_DISARMED: &str = "disarmed";
pub const REASON_KEEPALIVE_TIMEOUT: &str = "keepalive_timeout";
pub const REASON_SAFETY_ESTOP: &str = "safety_estop";

pub const REASON_HAND_STALE_HOLD: &str = "hand_stale_hold";
pub const REASON_HAND_STALE_FREEZE: &str = "hand_stale_freeze";
pub const REASON_PAIRING_WINDOW_EXCEEDED: &str = "pairing_window_exceeded";

pub const REASON_HARDWARE_OFFLINE: &str = "hardware_offline";
pub const REASON_HARDWARE_OVERTEMP: &str = "hardware_overtemp";
pub const REASON_HARDWARE_ERROR: &str = "hardware_error";
