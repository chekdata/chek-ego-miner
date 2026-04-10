use std::time::Duration;

use tokio::time::MissedTickBehavior;
use tracing::warn;

use crate::AppState;

const BRIDGE_UNREADY_FAULT_GRACE_MS: u64 = 1_500;
const TRANSIENT_FAULT_AUTO_RECOVER_GRACE_MS: u64 = 250;

pub async fn run_deadman_watchdog(state: AppState) {
    let tick = Duration::from_millis(20);
    let mut ticker = tokio::time::interval(tick);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

    // 现场经验：在高负载/调试环境下，单次调度抖动可能导致“瞬时 not-ok”。
    // 为避免误报，这里对 deadman/bridge 的 fault 触发加一个很小的“连续不健康窗口”（不改变语义，只降低抖动）。
    let mut deadman_unhealthy_since: Option<tokio::time::Instant> = None;
    let mut bridge_unready_since: Option<tokio::time::Instant> = None;
    let mut transient_fault_recovered_since: Option<tokio::time::Instant> = None;

    loop {
        ticker.tick().await;

        let now = tokio::time::Instant::now();

        let deadman = state.gate.deadman_snapshot();
        let snap = state.gate.snapshot();
        let bridge = state.bridge_store.snapshot(state.config.bridge_stale_ms);
        let mut bridge_ready = bridge.unitree_ready;
        if snap.end_effector_type_hint == "LEAP_V2" {
            bridge_ready = bridge_ready && bridge.leap_ready;
        }
        let deadman_motion_ok = !deadman.enabled || (deadman.link_ok && deadman.pressed);

        if snap.state == "fault"
            && matches!(
                snap.reason.as_str(),
                crate::reason::REASON_DEADMAN_TIMEOUT | crate::reason::REASON_BRIDGE_UNREADY
            )
        {
            let session = state.session.snapshot();
            let session_active = session.teleop_enabled
                && !session.trip_id.is_empty()
                && !session.session_id.is_empty();
            if session_active && deadman_motion_ok && bridge_ready {
                transient_fault_recovered_since.get_or_insert(now);
                if now.duration_since(transient_fault_recovered_since.unwrap())
                    >= Duration::from_millis(TRANSIENT_FAULT_AUTO_RECOVER_GRACE_MS)
                {
                    warn!(reason = %snap.reason, "transient fault recovered -> re-arm");
                    state.gate.arm();
                    metrics::counter!("control_fault_auto_recover_count").increment(1);
                    deadman_unhealthy_since = None;
                    bridge_unready_since = None;
                    transient_fault_recovered_since = None;
                }
            } else {
                transient_fault_recovered_since = None;
            }
            continue;
        }

        transient_fault_recovered_since = None;

        if snap.state != "armed" {
            deadman_unhealthy_since = None;
            bridge_unready_since = None;
            continue;
        }

        if deadman.enabled && !deadman.link_ok {
            deadman_unhealthy_since.get_or_insert(now);
            if now.duration_since(deadman_unhealthy_since.unwrap()) >= Duration::from_millis(40) {
                warn!(
                    reason = crate::reason::REASON_DEADMAN_TIMEOUT,
                    "deadman timeout -> fault"
                );
                state.gate.fault(crate::reason::REASON_DEADMAN_TIMEOUT);
                metrics::counter!("deadman_timeout_count").increment(1);
            }
        } else {
            deadman_unhealthy_since = None;
        }

        if !bridge_ready {
            bridge_unready_since.get_or_insert(now);
            if now.duration_since(bridge_unready_since.unwrap())
                >= Duration::from_millis(BRIDGE_UNREADY_FAULT_GRACE_MS)
            {
                warn!(
                    reason = crate::reason::REASON_BRIDGE_UNREADY,
                    "bridge became unready -> fault"
                );
                state.gate.fault(crate::reason::REASON_BRIDGE_UNREADY);
                metrics::counter!("bridge_unready_fault_count").increment(1);
            }
        } else {
            bridge_unready_since = None;
        }
    }
}
