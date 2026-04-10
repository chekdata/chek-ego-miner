/// 统一的 reason code（用于 `fault` 与门控拒绝原因）。
///
/// 约定：
/// - 使用 `snake_case`
/// - 用于日志字段、Prometheus label、以及 `/control/state.reason`
pub const REASON_UNKNOWN: &str = "unknown";

pub const REASON_DEADMAN_TIMEOUT: &str = "deadman_timeout";
pub const REASON_DEADMAN_RELEASED: &str = "deadman_released";
pub const REASON_DEADMAN_LINK_UNREADY: &str = "deadman_link_unready";

pub const REASON_BRIDGE_UNREADY: &str = "bridge_unready";
pub const REASON_UNITREE_BRIDGE_UNREADY: &str = "unitree_bridge_unready";
pub const REASON_LEAP_BRIDGE_UNREADY: &str = "leap_bridge_unready";

pub const REASON_TIME_SYNC_UNREADY: &str = "time_sync_unready";
pub const REASON_EXTRINSIC_UNREADY: &str = "extrinsic_unready";
pub const REASON_LAN_CONTROL_UNREADY: &str = "lan_control_unready";

pub const REASON_NAN_OR_INF: &str = "nan_or_inf";
pub const REASON_IK_UNIMPLEMENTED: &str = "ik_unimplemented";
pub const REASON_IK_UNSUPPORTED_ROBOT: &str = "ik_unsupported_robot";
pub const REASON_IK_JOINT_LEN_UNSUPPORTED: &str = "ik_joint_len_unsupported";
pub const REASON_IK_FAILED: &str = "ik_failed";
