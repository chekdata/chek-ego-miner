use ruview_unitree_bridge::bridge::parser::parse_teleop_frame_v1_json;

#[test]
fn contract_parse_valid_teleop_frame_v1_ok() {
    let raw = r#"
    {
      "schema_version": "teleop_frame_v1",
      "trip_id": "trip-20260304-001",
      "session_id": "sess-20260304-001",
      "robot_type": "G1_29",
      "end_effector_type": "LEAP_V2",
      "edge_time_ns": 123456789,
      "operator_frame": "op",
      "robot_base_frame": "base",
      "extrinsic_version": "v1",
      "control_state": "armed",
      "left_wrist_pose": { "pos": [0,0,0], "quat": [1,0,0,0] },
      "right_wrist_pose": { "pos": [0,0,0], "quat": [1,0,0,0] },
      "quality": { "source_mode": "fused", "fused_conf": 1.0, "vision_conf": 0.0, "csi_conf": 0.0 },
      "safety_state": "normal",

      "arm_q_target": [0.0, 0.1, 0.2],
      "arm_tauff_target": [0.0, 0.0, 0.0]
    }
    "#;

    let parsed = parse_teleop_frame_v1_json(raw, Some(3), None).expect("should parse");
    assert_eq!(parsed.trip_id, "trip-20260304-001");
    assert_eq!(parsed.control_state, "armed");
}
