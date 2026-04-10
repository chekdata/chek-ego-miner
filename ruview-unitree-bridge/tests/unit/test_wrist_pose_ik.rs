use ruview_unitree_bridge::ik::{ArmSide, UnitreeIk};

#[test]
fn unit_ik_fk_roundtrip_should_converge() {
    let ik = UnitreeIk::new().expect("IK init should succeed");
    let robot_type = "G1_29";

    let q_left = [0.0f32; 7];
    let q_right = [0.0f32; 7];
    let left = ik
        .fk_wrist_pose(robot_type, ArmSide::Left, &q_left)
        .expect("fk left");
    let right = ik
        .fk_wrist_pose(robot_type, ArmSide::Right, &q_right)
        .expect("fk right");

    let seed: Vec<f32> = [q_left.as_slice(), q_right.as_slice()].concat();
    let solved = ik
        .solve_arm_q_target(robot_type, &left, &right, Some(&seed), Some(14))
        .expect("solve should succeed");

    assert_eq!(solved.len(), 14);
    assert!(solved.iter().all(|x| x.is_finite()));
}
