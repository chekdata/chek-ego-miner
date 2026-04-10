use ruview_leap_bridge::bridge::parser::parse_teleop_frame_v1_json;

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
      "control_state": "armed",
      "safety_state": "normal",
      "hand_joint_layout": "anatomical_joint_16",
      "hand_target_layout": "anatomical_target_16",
      "left_hand_target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
      "right_hand_target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],

      "operator_frame": "op",
      "robot_base_frame": "base",
      "extrinsic_version": "v1",
      "left_wrist_pose": { "pos": [0,0,0], "quat": [1,0,0,0] },
      "right_wrist_pose": { "pos": [0,0,0], "quat": [1,0,0,0] },
      "quality": { "source_mode": "fused", "fused_conf": 1.0, "vision_conf": 0.0, "csi_conf": 0.0 }
    }
    "#;

    let parsed = parse_teleop_frame_v1_json(raw, Some(16)).expect("should parse");
    assert_eq!(parsed.trip_id, "trip-20260304-001");
    assert!(parsed.is_for_leap());
    assert!(parsed.left.is_some());
    assert!(parsed.right.is_some());
}

#[test]
fn contract_parse_legacy_curl5_without_layout_should_still_parse_when_overridden() {
    let raw = r#"
    {
      "schema_version": "teleop_frame_v1",
      "trip_id": "trip-legacy-001",
      "session_id": "sess-legacy-001",
      "robot_type": "G1_29",
      "end_effector_type": "LEAP_V2",
      "edge_time_ns": 123456789,
      "control_state": "armed",
      "safety_state": "normal",
      "left_hand_target": [0.1, 0.2, 0.3, 0.4, 0.5],
      "right_hand_target": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    "#;

    let parsed = parse_teleop_frame_v1_json(raw, Some(5)).expect("legacy curl should parse");
    assert!(parsed.left.is_some());
    assert!(parsed.right.is_some());
}

#[test]
fn contract_missing_trip_id_should_fail() {
    let raw = r#"
    {
      "schema_version": "teleop_frame_v1",
      "trip_id": "",
      "session_id": "sess-1",
      "robot_type": "G1_29",
      "end_effector_type": "LEAP_V2",
      "edge_time_ns": 1,
      "control_state": "armed",
      "safety_state": "normal",
      "hand_target_layout": "anatomical_target_16",
      "left_hand_target": [0.1],
      "right_hand_target": [0.1]
    }
    "#;
    let err = parse_teleop_frame_v1_json(raw, Some(16)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("trip_id") || msg.contains("schema") || msg.contains("协议"),
        "{msg}"
    );
}
