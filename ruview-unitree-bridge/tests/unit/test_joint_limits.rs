use ruview_unitree_bridge::bridge::mapper::clamp_vec;

#[test]
fn unit_joint_limit_clamp_should_work() {
    let min_v = vec![-1.0, -0.5];
    let max_v = vec![1.0, 0.5];
    let v = vec![-2.0, 2.0];
    let out = clamp_vec(&v, &min_v, &max_v);
    assert_eq!(out, vec![-1.0, 0.5]);
}
