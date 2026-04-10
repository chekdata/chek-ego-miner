// 合并运行 `tests/integration/*` 下的用例（Cargo 仅会把 `tests/*.rs` 视为集成测试入口）。

mod support;

#[path = "integration/test_deadman_timeout.rs"]
mod test_deadman_timeout;

#[path = "integration/test_demo_capture_bundle.rs"]
mod test_demo_capture_bundle;

#[path = "integration/test_clip_validation.rs"]
mod test_clip_validation;

#[path = "integration/test_disarm_gate.rs"]
mod test_disarm_gate;

#[path = "integration/test_fusion_debug_views.rs"]
mod test_fusion_debug_views;

#[path = "integration/test_stream_isolation.rs"]
mod test_stream_isolation;

#[path = "integration/test_session_metrics.rs"]
mod test_session_metrics;

#[path = "integration/test_teleop_targets.rs"]
mod test_teleop_targets;

#[path = "integration/test_upload_chunk_ack.rs"]
mod test_upload_chunk_ack;

#[path = "integration/test_wifi_pose_ingest.rs"]
mod test_wifi_pose_ingest;

#[path = "integration/test_ws_cbor_format.rs"]
mod test_ws_cbor_format;

#[path = "integration/test_ws_transport_envelope.rs"]
mod test_ws_transport_envelope;
