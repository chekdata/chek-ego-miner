use ruview_leap_bridge::bridge::retarget::{
    retarget_paired, RetargetConfig, SideRetargetCalibration,
};
use ruview_leap_bridge::bridge::types::{HandSide, HandTargetFrame, PairedHandFrame, PairingDegrade};
use ruview_leap_bridge::reason;

fn paired_with_targets(left: Vec<f32>, right: Vec<f32>) -> PairedHandFrame {
    PairedHandFrame {
        left: HandTargetFrame {
            side: HandSide::Left,
            edge_time_ns: 10,
            target: left,
        },
        right: HandTargetFrame {
            side: HandSide::Right,
            edge_time_ns: 12,
            target: right,
        },
        delta_ns: 2,
        degrade: PairingDegrade::Normal,
    }
}

#[test]
fn unit_retarget_curls5_should_expand_to_leap16() {
    let cfg = RetargetConfig::new(vec![], vec![]);
    let paired = paired_with_targets(vec![1.0, 0.5, 0.0, 0.25, 0.75], vec![0.0; 5]);
    let cmd = retarget_paired(&paired, &cfg).expect("retarget should succeed");
    assert_eq!(cmd.left_cmd.len(), 16);
    assert_eq!(cmd.right_cmd.len(), 16);

    // thumb_yaw should be negative when thumb curl > 0
    assert!(cmd.left_cmd[0] < 0.0);
}

#[test]
fn unit_retarget_anatomical16_should_apply_side_specific_thumb_yaw() {
    let cfg = RetargetConfig::new(vec![], vec![]);
    let paired = paired_with_targets(
        vec![0.3, 0.2, 0.1, 0.05, 0.4, 0.6, 0.5, 0.35, 0.58, 0.44, 0.32, 0.55, 0.41, 0.28, 0.52, 0.39],
        vec![0.3, 0.2, 0.1, 0.05, 0.4, 0.6, 0.5, 0.35, 0.58, 0.44, 0.32, 0.55, 0.41, 0.28, 0.52, 0.39],
    );
    let cmd = retarget_paired(&paired, &cfg).expect("retarget should succeed");
    assert_eq!(cmd.left_cmd.len(), 16);
    assert_eq!(cmd.right_cmd.len(), 16);
    assert_eq!(cmd.left_cmd[0], -0.3);
    assert_eq!(cmd.right_cmd[0], 0.3);
}

#[test]
fn unit_retarget_unknown_len_should_fail() {
    let cfg = RetargetConfig::new(vec![], vec![]);
    let paired = paired_with_targets(vec![0.1; 7], vec![0.1; 7]);
    let err = retarget_paired(&paired, &cfg).unwrap_err();
    assert_eq!(err, reason::REASON_DIMENSION_MISMATCH);
}

#[test]
fn unit_retarget_anatomical16_should_apply_side_calibration_profile() {
    let left_scale = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.0, 1.0,
    ];
    let left_offset = vec![
        0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let right_scale = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let right_offset = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.03,
    ];
    let cfg = RetargetConfig::new(vec![], vec![])
        .with_side_calibration(
            HandSide::Left,
            SideRetargetCalibration::new(left_scale, left_offset),
        )
        .with_side_calibration(
            HandSide::Right,
            SideRetargetCalibration::new(right_scale, right_offset),
        );
    let paired = paired_with_targets(
        vec![0.3, 0.2, 0.1, 0.05, 0.4, 0.6, 0.5, 0.35, 0.58, 0.44, 0.32, 0.55, 0.41, 0.28, 0.52, 0.39],
        vec![0.3, 0.2, 0.1, 0.05, 0.4, 0.6, 0.5, 0.35, 0.58, 0.44, 0.32, 0.55, 0.41, 0.28, 0.52, 0.39],
    );
    let cmd = retarget_paired(&paired, &cfg).expect("retarget should succeed");
    assert_eq!(cmd.left_cmd[0], -0.25);
    assert_eq!(cmd.left_cmd[13], 0.33600003);
    assert_eq!(cmd.right_cmd[10], 0.288);
    assert_eq!(cmd.right_cmd[15], 0.36);
}
