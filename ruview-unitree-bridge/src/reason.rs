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
pub const REASON_NON_MONOTONIC_TIMESTAMP: &str = "non_monotonic_timestamp";

pub const REASON_DISARMED: &str = "disarmed";
pub const REASON_KEEPALIVE_TIMEOUT: &str = "keepalive_timeout";
pub const REASON_SAFETY_ESTOP: &str = "safety_estop";

pub const REASON_ENDPOINT_GUARD_DENY: &str = "endpoint_guard_deny";

pub const REASON_ARM_TARGET_MISSING: &str = "arm_target_missing";
pub const REASON_IK_UNIMPLEMENTED: &str = "ik_unimplemented";
pub const REASON_IK_UNSUPPORTED_ROBOT: &str = "ik_unsupported_robot";
pub const REASON_IK_JOINT_LEN_UNSUPPORTED: &str = "ik_joint_len_unsupported";
pub const REASON_IK_FAILED: &str = "ik_failed";

pub const REASON_DDS_PUBLISH_FAILED: &str = "dds_publish_failed";
