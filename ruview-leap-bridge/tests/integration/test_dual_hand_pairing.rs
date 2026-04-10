use ruview_leap_bridge::bridge::pairing::{PairingConfig, PairingEngine};
use ruview_leap_bridge::bridge::types::{HandSide, HandTargetFrame, PairingDegrade};

fn hand(side: HandSide, edge_time_ns: u64) -> HandTargetFrame {
    HandTargetFrame {
        side,
        edge_time_ns,
        target: vec![0.1, 0.2, 0.3],
    }
}

#[test]
fn integration_pairing_window_normal_within_20ms() {
    let mut engine = PairingEngine::new(PairingConfig {
        pairing_window_ms: 20,
        hold_timeout_ms: 200,
        freeze_timeout_ms: 200,
    });

    let parsed = ruview_leap_bridge::bridge::parser::ParsedHandTargets {
        trip_id: "t".to_string(),
        session_id: "s".to_string(),
        robot_type: "G1_29".to_string(),
        end_effector_type: "LEAP_V2".to_string(),
        edge_time_ns: 0,
        control_state: "armed".to_string(),
        safety_state: "normal".to_string(),
        hand_control_enabled: true,
        left: Some(hand(HandSide::Left, 0)),
        right: Some(hand(HandSide::Right, 10_000_000)), // 10ms
    };

    engine.ingest(&parsed);
    let paired = engine.pair().expect("should pair");
    assert_eq!(paired.degrade, PairingDegrade::Normal);
    assert!(paired.delta_ns <= 20_000_000);
}

#[test]
fn integration_pairing_degrade_hold_then_freeze() {
    let mut engine = PairingEngine::new(PairingConfig {
        pairing_window_ms: 20,
        hold_timeout_ms: 200,
        freeze_timeout_ms: 200,
    });

    // 初始：右手停留在 0ms
    let base = ruview_leap_bridge::bridge::parser::ParsedHandTargets {
        trip_id: "t".to_string(),
        session_id: "s".to_string(),
        robot_type: "G1_29".to_string(),
        end_effector_type: "LEAP_V2".to_string(),
        edge_time_ns: 0,
        control_state: "armed".to_string(),
        safety_state: "normal".to_string(),
        hand_control_enabled: true,
        left: Some(hand(HandSide::Left, 0)),
        right: Some(hand(HandSide::Right, 0)),
    };
    engine.ingest(&base);

    // 左手 50ms，超过 20ms 窗口，但 <200ms，进入 Hold
    let left_50ms = ruview_leap_bridge::bridge::parser::ParsedHandTargets {
        left: Some(hand(HandSide::Left, 50_000_000)),
        right: None,
        edge_time_ns: 50_000_000,
        ..base.clone()
    };
    engine.ingest(&left_50ms);
    let paired = engine.pair().expect("should pair");
    assert_eq!(paired.degrade, PairingDegrade::Hold);

    // 左手 250ms，>=200ms，进入 Freeze
    let left_250ms = ruview_leap_bridge::bridge::parser::ParsedHandTargets {
        left: Some(hand(HandSide::Left, 250_000_000)),
        right: None,
        edge_time_ns: 250_000_000,
        ..base
    };
    engine.ingest(&left_250ms);
    let paired = engine.pair().expect("should pair");
    assert_eq!(paired.degrade, PairingDegrade::Freeze);
}
