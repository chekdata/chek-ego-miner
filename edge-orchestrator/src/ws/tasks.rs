use std::time::Duration;

use futures_util::FutureExt;
use tokio::time::MissedTickBehavior;
use tracing::{error, warn};

use crate::calibration::transform_points_3d;
use crate::host_metrics::HostMetricsCollector;
use crate::ws::types::{
    DeadmanState, FusionControl, FusionQuality, FusionSafety, FusionStatePacket,
    HumanDemoCanonicalPose, HumanDemoFusionDebug, HumanDemoPosePacket, HumanDemoRawPose, LatencyMs,
    Pose, RetargetReferenceV1, TeleopFrameV1, TeleopHandJointLayout, TeleopHandTargetLayout,
    TeleopLegJointLayout, TeleopQuality, TeleopWaistJointLayout,
};
use crate::{operator, target, target::TeleopTargetPrecomputer, AppState};

fn valid_point_3d(point: [f32; 3]) -> bool {
    point.iter().all(|value| value.is_finite()) && point.iter().any(|value| value.abs() > 1e-5)
}

fn points_have_valid_3d(points: &[[f32; 3]]) -> bool {
    points.iter().copied().any(valid_point_3d)
}

fn resolve_selected_body_source_edge_time_ns(
    operator_body_selected: bool,
    matched_stereo_person_available: bool,
    vision_body_available: bool,
    operator_body_source_edge_time_ns: u64,
    stereo_edge_time_ns: u64,
    vision_edge_time_ns: u64,
    operator_source_edge_time_ns: u64,
) -> u64 {
    if operator_body_selected {
        operator_body_source_edge_time_ns
    } else if matched_stereo_person_available {
        stereo_edge_time_ns.max(operator_source_edge_time_ns)
    } else if vision_body_available {
        vision_edge_time_ns.max(operator_source_edge_time_ns)
    } else {
        operator_source_edge_time_ns
            .max(stereo_edge_time_ns)
            .max(vision_edge_time_ns)
    }
}

fn resolve_selected_hand_source_edge_time_ns(
    operator_hand_selected: bool,
    phone_hand_authoritative_selected: bool,
    operator_hand_source_edge_time_ns: u64,
    vision_edge_time_ns: u64,
) -> u64 {
    if phone_hand_authoritative_selected {
        vision_edge_time_ns.max(operator_hand_source_edge_time_ns)
    } else if operator_hand_selected {
        operator_hand_source_edge_time_ns
    } else {
        0
    }
}

fn resolve_retarget_target_person_id(
    phone_session_aligned: bool,
    vision_operator_track_id: Option<&str>,
    associated_stereo_track_id: Option<&str>,
    iphone_operator_track_id: Option<&str>,
    selected_operator_track_id: Option<&str>,
) -> String {
    if phone_session_aligned {
        vision_operator_track_id
            .filter(|value| !value.trim().is_empty())
            .or(associated_stereo_track_id.filter(|value| !value.trim().is_empty()))
            .or(iphone_operator_track_id.filter(|value| !value.trim().is_empty()))
            .filter(|value| !value.trim().is_empty())
            .or(selected_operator_track_id.filter(|value| !value.trim().is_empty()))
            .unwrap_or("primary_operator")
            .to_string()
    } else {
        selected_operator_track_id.unwrap_or_default().to_string()
    }
}

pub async fn run_fusion_state_publisher(state: AppState) {
    let interval = Duration::from_millis((1000 / state.config.fusion_publish_hz.max(1)) as u64);
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut seq: u64 = 0;

    // 质量/性能指标（滚动窗口）
    let expected_interval_ms = (1000.0 / state.config.fusion_publish_hz.max(1) as f32).max(1.0);
    let mut last_tick_at: Option<std::time::Instant> = None;
    let mut fps_window_started_at = std::time::Instant::now();
    let mut fps_frames: u32 = 0;
    let mut jitter_samples_ms: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(600);
    let mut e2e_samples_ms: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(600);
    let mut iphone_to_edge_samples_ms: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(600);
    let mut root_jitter_samples_m: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(600);
    let mut heading_jitter_samples_deg: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(600);
    let mut previous_root_pos_m: Option<[f32; 3]> = None;
    let mut previous_heading_yaw_rad: Option<f32> = None;
    let mut previous_selected_operator_track_id: Option<String> = None;
    let mut hold_started_at_edge_time_ns: Option<u64> = None;
    let mut stereo_dropout_started_edge_time_ns: Option<u64> = None;

    let mut next_session_metrics_at = std::time::Instant::now();
    let mut current_metrics_session: Option<(String, String)> = None;
    let mut reported_vision_timeout_count: u64 = 0;
    let mut reported_operator_track_switch_count: u64 = 0;
    let mut host_metrics = HostMetricsCollector::new();

    loop {
        ticker.tick().await;
        let tick_started_at = std::time::Instant::now();
        seq += 1;

        let session = state.session.snapshot();
        let trip_id = session.trip_id.clone();
        let session_id = session.session_id.clone();
        let mode = session.mode.clone();
        let next_metrics_session = if trip_id.is_empty() || session_id.is_empty() {
            None
        } else {
            Some((trip_id.clone(), session_id.clone()))
        };
        if current_metrics_session != next_metrics_session {
            current_metrics_session = next_metrics_session;
            reported_vision_timeout_count = 0;
            reported_operator_track_switch_count = 0;
            metrics::gauge!("id_switch_rate").set(0.0);
        }
        let vision = state.vision.snapshot(state.config.vision_stale_ms);
        let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
        let wifi_pose = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
        let csi = state.csi.snapshot(state.config.csi_stale_ms);
        let vision_conf = effective_vision_conf(&vision, &stereo);
        let vision_fresh = vision.fresh || stereo.fresh;
        let csi_conf = if csi.fresh { csi.csi_conf } else { 0.0 };
        let (source_mode, fused_conf, coherence) =
            compute_fused_quality(&mode, vision_conf, csi_conf, vision_fresh, csi.fresh);

        // safety_state 由融合质量驱动（estop 优先）
        state.gate.update_safety_from_fused_conf(fused_conf);

        let edge_time_ns_now = state.gate.edge_time_ns();
        let iphone_stereo_calibration = state.iphone_stereo_calibration.snapshot();
        let wifi_stereo_calibration = state.wifi_stereo_calibration.snapshot();
        let op = state.operator.tick(
            &state.config,
            edge_time_ns_now,
            &vision,
            &stereo,
            &wifi_pose,
            &csi,
            iphone_stereo_calibration.as_ref(),
            wifi_stereo_calibration.as_ref(),
        );
        if let Some(previous_track_id) = previous_selected_operator_track_id.as_deref() {
            if let Some(current_track_id) = op.association.selected_operator_track_id.as_deref() {
                if current_track_id != previous_track_id {
                    metrics::counter!("selected_operator_switch_count").increment(1);
                }
            }
        }
        previous_selected_operator_track_id = op.association.selected_operator_track_id.clone();

        if op.motion_state.updated_edge_time_ns > 0 {
            if let Some(previous_root) = previous_root_pos_m {
                let jitter_m = norm3([
                    op.motion_state.root_pos_m[0] - previous_root[0],
                    op.motion_state.root_pos_m[1] - previous_root[1],
                    op.motion_state.root_pos_m[2] - previous_root[2],
                ]);
                if root_jitter_samples_m.len() >= 600 {
                    root_jitter_samples_m.pop_front();
                }
                root_jitter_samples_m.push_back(jitter_m);
            }
            previous_root_pos_m = Some(op.motion_state.root_pos_m);

            if let Some(previous_heading) = previous_heading_yaw_rad {
                let heading_jitter_deg =
                    wrap_angle_delta_rad(op.motion_state.heading_yaw_rad - previous_heading).abs()
                        * 180.0
                        / std::f32::consts::PI;
                if heading_jitter_samples_deg.len() >= 600 {
                    heading_jitter_samples_deg.pop_front();
                }
                heading_jitter_samples_deg.push_back(heading_jitter_deg);
            }
            previous_heading_yaw_rad = Some(op.motion_state.heading_yaw_rad);
        }

        match op.motion_state.smoother_mode {
            operator::OperatorSmootherMode::HeldWithCsiPrior => {
                hold_started_at_edge_time_ns.get_or_insert(edge_time_ns_now);
            }
            _ => {
                if hold_started_at_edge_time_ns.is_some() {
                    metrics::gauge!("hold_duration_ms").set(0.0);
                }
                hold_started_at_edge_time_ns = None;
            }
        }
        if let Some(hold_started_at) = hold_started_at_edge_time_ns {
            let hold_duration_ms =
                (edge_time_ns_now.saturating_sub(hold_started_at) as f64) / 1_000_000.0;
            metrics::gauge!("hold_duration_ms").set(hold_duration_ms);
        }

        if stereo.fresh && op.motion_state.stereo_measurement_used {
            if let Some(dropout_started_at) = stereo_dropout_started_edge_time_ns.take() {
                let recovery_ms =
                    (edge_time_ns_now.saturating_sub(dropout_started_at) as f64) / 1_000_000.0;
                metrics::gauge!("stereo_dropout_recovery_ms").set(recovery_ms);
            }
        } else if stereo_dropout_started_edge_time_ns.is_none() {
            stereo_dropout_started_edge_time_ns = Some(edge_time_ns_now);
        }

        let gate = state.gate.snapshot();
        let deadman = state.gate.deadman_snapshot();
        let bridge = state.bridge_store.snapshot(state.config.bridge_stale_ms);

        emit_health_gauges(&state, &gate, &deadman, &bridge);

        let mut bridge_ready = bridge.unitree_ready;
        if gate.end_effector_type_hint == "LEAP_V2" {
            bridge_ready = bridge_ready && bridge.leap_ready;
        }

        let pkt = FusionStatePacket {
            ty: "fusion_state_packet",
            schema_version: "1.0.0",
            trip_id: trip_id.clone(),
            session_id: session_id.clone(),
            fusion_seq: seq,
            edge_time_ns: edge_time_ns_now,
            operator_state: op.operator_state.clone(),
            quality: FusionQuality {
                source_mode,
                vision_conf,
                csi_conf,
                fused_conf,
                coherence,
                gate_state: match state.gate.safety_state() {
                    crate::control::gate::SafetyState::Normal => "accept",
                    crate::control::gate::SafetyState::Limit => "limit",
                    crate::control::gate::SafetyState::Freeze => "freeze",
                    crate::control::gate::SafetyState::Estop => "estop",
                }
                .to_string(),
            },
            safety: FusionSafety {
                state: match state.gate.safety_state() {
                    crate::control::gate::SafetyState::Normal => "normal",
                    crate::control::gate::SafetyState::Limit => "limit",
                    crate::control::gate::SafetyState::Freeze => "freeze",
                    crate::control::gate::SafetyState::Estop => "estop",
                }
                .to_string(),
                reason: state.gate.safety_reason(),
            },
            control: FusionControl {
                state: gate.state,
                reason: gate.reason,
                bridge_ready,
                deadman: DeadmanState {
                    enabled: deadman.enabled,
                    timeout_ms: deadman.timeout_ms,
                    link_ok: deadman.link_ok,
                    pressed: deadman.pressed,
                },
            },
            latency_ms: LatencyMs {
                iphone_to_edge: if vision.fresh {
                    vision.iphone_to_edge_latency_ms
                } else {
                    0.0
                },
                csi_to_edge: 0.0,
                fusion_compute: 0.0, // 后面会更新为实际耗时
                edge_to_robot: compute_edge_to_robot_ms(edge_time_ns_now, &bridge),
                e2e: 0.0, // 后面会更新（iphone_to_edge + fusion_compute + edge_to_robot）
            },
            operator_debug: None,
        };

        // 轻量“融合计算耗时”（MVP：构包+快照读取耗时）
        let fusion_compute_ms = tick_started_at.elapsed().as_secs_f32() * 1000.0;
        let mut pkt = pkt;
        pkt.latency_ms.fusion_compute = fusion_compute_ms;
        pkt.latency_ms.e2e = pkt.latency_ms.iphone_to_edge
            + pkt.latency_ms.fusion_compute
            + pkt.latency_ms.edge_to_robot;
        let e2e_ms = pkt.latency_ms.e2e;

        let left_hand_joint_target = crate::target::hand_joint_target_from_hand_kpts_3d(
            &op.operator_state.hand_kpts_3d,
            0,
            true,
        );
        let right_hand_joint_target = crate::target::hand_joint_target_from_hand_kpts_3d(
            &op.operator_state.hand_kpts_3d,
            21,
            false,
        );

        let demo_pose_pkt = HumanDemoPosePacket {
            ty: "human_demo_pose_packet",
            schema_version: "1.0.0",
            trip_id: trip_id.clone(),
            session_id: session_id.clone(),
            fusion_seq: seq,
            edge_time_ns: edge_time_ns_now,
            selected_source: op.source.as_str().to_string(),
            raw_pose: HumanDemoRawPose {
                source_edge_time_ns: op.raw_pose.source_edge_time_ns,
                body_layout: op.raw_pose.body_layout.as_str().to_string(),
                hand_layout: op.raw_pose.hand_layout.as_str().to_string(),
                body_kpts_3d: op.raw_pose.body_kpts_3d.clone(),
                hand_kpts_3d: op.raw_pose.hand_kpts_3d.clone(),
            },
            canonical_pose: HumanDemoCanonicalPose {
                body_layout: "coco_body_17".to_string(),
                hand_layout: "mediapipe_hand_21".to_string(),
                body_kpts_3d: op.operator_state.body_kpts_3d.clone(),
                hand_kpts_3d: op.operator_state.hand_kpts_3d.clone(),
                end_effector_pose: op.operator_state.end_effector_pose.clone(),
                left_hand_joints: left_hand_joint_target.map(|values| values.to_vec()),
                right_hand_joints: right_hand_joint_target.map(|values| values.to_vec()),
                left_hand_curls: op
                    .left_hand_curls
                    .or_else(|| {
                        left_hand_joint_target
                            .as_ref()
                            .map(crate::target::hand_curls_from_hand_joint_target)
                    })
                    .map(|values| values.to_vec()),
                right_hand_curls: op
                    .right_hand_curls
                    .or_else(|| {
                        right_hand_joint_target
                            .as_ref()
                            .map(crate::target::hand_curls_from_hand_joint_target)
                    })
                    .map(|values| values.to_vec()),
            },
            fusion_debug: HumanDemoFusionDebug {
                body_source: op.fusion_breakdown.body_source.as_str().to_string(),
                hand_source: op.fusion_breakdown.hand_source.as_str().to_string(),
                stereo_body_joint_count: op.fusion_breakdown.stereo_body_joint_count,
                vision_body_joint_count: op.fusion_breakdown.vision_body_joint_count,
                wifi_body_joint_count: op.fusion_breakdown.wifi_body_joint_count,
                blended_body_joint_count: op.fusion_breakdown.blended_body_joint_count,
                stereo_hand_point_count: op.fusion_breakdown.stereo_hand_point_count,
                vision_hand_point_count: op.fusion_breakdown.vision_hand_point_count,
                wifi_hand_point_count: op.fusion_breakdown.wifi_hand_point_count,
                blended_hand_point_count: op.fusion_breakdown.blended_hand_point_count,
                motion_root_pos_m: op.motion_state.root_pos_m,
                motion_root_vel_mps: op.motion_state.root_vel_mps,
                motion_heading_yaw_rad: op.motion_state.heading_yaw_rad,
                motion_heading_rate_radps: op.motion_state.heading_rate_radps,
                motion_phase: op.motion_state.motion_phase,
                motion_body_presence_conf: op.motion_state.body_presence_conf,
                motion_csi_prior_reliability: op.motion_state.csi_prior_reliability,
                motion_wearer_confidence: op.motion_state.wearer_confidence,
                motion_stereo_track_id: op.motion_state.stereo_track_id.clone().unwrap_or_default(),
                motion_last_good_stereo_time_ns: op.motion_state.last_good_stereo_time_ns,
                motion_last_good_csi_time_ns: op.motion_state.last_good_csi_time_ns,
                motion_stereo_measurement_used: op.motion_state.stereo_measurement_used,
                motion_csi_measurement_used: op.motion_state.csi_measurement_used,
                motion_smoother_mode: op.motion_state.smoother_mode.as_str().to_string(),
                motion_updated_edge_time_ns: op.motion_state.updated_edge_time_ns,
            }
            .into_option(),
        };

        let record = if !trip_id.is_empty() && !session_id.is_empty() {
            serde_json::to_value(&pkt).ok()
        } else {
            None
        };
        let demo_pose_record = if !trip_id.is_empty() && !session_id.is_empty() {
            serde_json::to_value(&demo_pose_pkt).ok()
        } else {
            None
        };
        let _ = state.fusion_state_tx.send(pkt);

        // 落盘（JSONL）
        if let Some(v) = record {
            state
                .recorder
                .record_fusion_state(&state.protocol, &state.config, &trip_id, &session_id, &v)
                .await;
        }
        if let Some(v) = demo_pose_record {
            state
                .recorder
                .record_human_demo_pose(&state.protocol, &state.config, &trip_id, &session_id, &v)
                .await;
        }

        // ----------- 指标（滚动窗口，近似即可）-----------
        let now = std::time::Instant::now();
        fps_frames = fps_frames.saturating_add(1);
        let fps_elapsed = now.duration_since(fps_window_started_at);
        if fps_elapsed >= Duration::from_secs(1) {
            let fps = fps_frames as f64 / fps_elapsed.as_secs_f64().max(1e-6);
            metrics::gauge!("fps_fusion").set(fps);
            fps_frames = 0;
            fps_window_started_at = now;
        }

        if let Some(prev) = last_tick_at {
            let dt_ms = now.duration_since(prev).as_secs_f32() * 1000.0;
            let jitter_ms = (dt_ms - expected_interval_ms).abs();
            if jitter_samples_ms.len() >= 600 {
                jitter_samples_ms.pop_front();
            }
            jitter_samples_ms.push_back(jitter_ms);
        }
        last_tick_at = Some(now);

        if vision.fresh {
            if iphone_to_edge_samples_ms.len() >= 600 {
                iphone_to_edge_samples_ms.pop_front();
            }
            iphone_to_edge_samples_ms.push_back(vision.iphone_to_edge_latency_ms);
        }
        let vision_timeout_delta = vision
            .timeout_count
            .saturating_sub(reported_vision_timeout_count);
        if vision_timeout_delta > 0 {
            metrics::counter!("vision_timeout_count").increment(vision_timeout_delta);
            reported_vision_timeout_count = vision.timeout_count;
        }
        let vision_switch_delta = vision
            .operator_track_switch_count
            .saturating_sub(reported_operator_track_switch_count);
        if vision_switch_delta > 0 {
            metrics::counter!("id_switch_count").increment(vision_switch_delta);
            reported_operator_track_switch_count = vision.operator_track_switch_count;
        }
        if e2e_ms > 0.0 {
            if e2e_samples_ms.len() >= 600 {
                e2e_samples_ms.pop_front();
            }
            e2e_samples_ms.push_back(e2e_ms);
        }

        // 每秒更新一次“会话级指标”（chunk 对账、回放抽检等）
        if now >= next_session_metrics_at {
            next_session_metrics_at = now + Duration::from_secs(1);
            host_metrics.emit();

            // jitter（p95）
            if !jitter_samples_ms.is_empty() {
                let p95 = quantile_ms(&jitter_samples_ms, 0.95);
                metrics::gauge!("jitter_ms").set(p95 as f64);
            }
            if !e2e_samples_ms.is_empty() {
                let p50 = quantile_ms(&e2e_samples_ms, 0.50);
                let p95 = quantile_ms(&e2e_samples_ms, 0.95);
                metrics::gauge!("latency_e2e_p50").set(p50 as f64);
                metrics::gauge!("latency_e2e_p95").set(p95 as f64);
            }
            if !iphone_to_edge_samples_ms.is_empty() {
                let p50 = quantile_ms(&iphone_to_edge_samples_ms, 0.50);
                let p95 = quantile_ms(&iphone_to_edge_samples_ms, 0.95);
                metrics::gauge!("latency_iphone_to_edge_p50").set(p50 as f64);
                metrics::gauge!("latency_iphone_to_edge_p95").set(p95 as f64);
            }
            if !root_jitter_samples_m.is_empty() {
                let p95 = quantile_ms(&root_jitter_samples_m, 0.95);
                metrics::gauge!("root_position_jitter_m").set(p95 as f64);
            }
            if !heading_jitter_samples_deg.is_empty() {
                let p95 = quantile_ms(&heading_jitter_samples_deg, 0.95);
                metrics::gauge!("heading_jitter_deg").set(p95 as f64);
            }
            let id_switch_rate = if vision.operator_track_sample_count <= 1 {
                0.0
            } else {
                (vision.operator_track_switch_count as f64
                    / (vision.operator_track_sample_count - 1) as f64)
                    * 100.0
            };
            metrics::gauge!("id_switch_rate").set(id_switch_rate);

            // chunk 对账指标（PRD 9.1）
            if !trip_id.is_empty() && !session_id.is_empty() {
                let stats = state.chunk_sm.session_stats(
                    &trip_id,
                    &session_id,
                    Duration::from_millis(state.config.chunk_acked_to_cleaned_sla_ms),
                );
                let ack_cleanup_rate = if stats.acked_total == 0 {
                    0.0
                } else {
                    (stats.cleaned_within_sla as f64 / stats.acked_total as f64) * 100.0
                };
                metrics::gauge!("ack_cleanup_rate").set(ack_cleanup_rate);

                let edge_chunk_continuity = if stats.stored_expected == 0 {
                    0.0
                } else {
                    (stats.stored_unique as f64 / stats.stored_expected as f64) * 100.0
                };
                metrics::gauge!("edge_chunk_continuity").set(edge_chunk_continuity);

                // 回放抽检指标（PRD 9.1，MVP：仅检查 clip_manifest needs_review + fusion 索引范围）
                if let Some(s) = state
                    .recorder
                    .clip_sample_stats(&trip_id, &session_id, state.config.replay_sample_total)
                    .await
                {
                    metrics::gauge!("replay_clip_sample_pass_rate").set(s.pass_rate_percent);
                }

                // raw/csi/index.jsonl（MVP：按秒落一条汇总索引，便于回放对齐与现场验收）
                let csi_snap = state.csi.snapshot(state.config.csi_stale_ms);
                let csi_index = serde_json::json!({
                    "type": "csi_index_event",
                    "schema_version": "1.0.0",
                    "trip_id": trip_id,
                    "session_id": session_id,
                    "edge_time_ns": state.gate.edge_time_ns(),
                    "node_count": csi_snap.node_count,
                    "drop_rate": csi_snap.drop_rate,
                    "csi_conf": csi_snap.csi_conf
                });
                state
                    .recorder
                    .record_csi_index(
                        &state.protocol,
                        &state.config,
                        &session.trip_id,
                        &session.session_id,
                        &csi_index,
                    )
                    .await;
            }
        }
    }
}

fn quantile_ms(samples: &std::collections::VecDeque<f32>, q: f32) -> f32 {
    let mut v: Vec<f32> = samples.iter().copied().collect();
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        return 0.0;
    }
    let qq = q.clamp(0.0, 1.0);
    let idx = ((v.len() - 1) as f32 * qq).round() as usize;
    v[idx.min(v.len() - 1)]
}

fn wrap_angle_delta_rad(angle: f32) -> f32 {
    let mut angle = angle;
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

pub async fn run_teleop_publisher(state: AppState) {
    loop {
        let outcome = std::panic::AssertUnwindSafe(run_teleop_publisher_inner(state.clone()))
            .catch_unwind()
            .await;
        match outcome {
            Ok(()) => {
                warn!("teleop publisher exited unexpectedly; restarting");
                metrics::counter!("teleop_publisher_restart_count", "reason" => "returned")
                    .increment(1);
            }
            Err(_) => {
                error!("teleop publisher panicked; restarting");
                metrics::counter!("teleop_publisher_restart_count", "reason" => "panic")
                    .increment(1);
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn run_teleop_publisher_inner(state: AppState) {
    let interval = Duration::from_millis((1000 / state.config.teleop_publish_hz.max(1)) as u64);
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut limiter = MotionLimiter::new(
        state.config.motion_max_speed_m_s,
        state.config.motion_max_accel_m_s2,
        state.config.motion_max_jerk_m_s3,
    );
    let mut target_precomputer = TeleopTargetPrecomputer::new();
    loop {
        ticker.tick().await;

        let session = state.session.snapshot();
        let trip_id = session.trip_id.clone();
        let mut session_id = session.session_id.clone();
        if trip_id.is_empty() || session_id.is_empty() {
            continue;
        }
        let teleop_enabled = session.teleop_enabled;
        let body_control_enabled = teleop_enabled && session.body_control_enabled;
        let hand_control_enabled = teleop_enabled && session.hand_control_enabled;

        let gate = state.gate.snapshot();
        let bridge = state.bridge_store.snapshot(state.config.bridge_stale_ms);
        let mut bridge_ready = bridge.unitree_ready;
        if gate.end_effector_type_hint == "LEAP_V2" {
            bridge_ready = bridge_ready && bridge.leap_ready;
        }

        let edge_time_ns_now = state.gate.edge_time_ns();

        let should_emit = state.gate.should_emit_motion();
        let control_state = if teleop_enabled && should_emit && bridge_ready {
            "armed"
        } else {
            "disarmed"
        };

        if control_state == "armed" {
            metrics::counter!("teleop_motion_emit_count").increment(1);
        } else if gate.state == "armed" {
            let deadman = state.gate.deadman_snapshot();
            let reason = if !bridge_ready {
                crate::reason::REASON_BRIDGE_UNREADY
            } else if deadman.enabled && !deadman.link_ok {
                crate::reason::REASON_DEADMAN_TIMEOUT
            } else if deadman.enabled && !deadman.pressed {
                crate::reason::REASON_DEADMAN_RELEASED
            } else {
                crate::reason::REASON_UNKNOWN
            };
            metrics::counter!("teleop_gate_block_count", "reason" => reason).increment(1);
        }

        let mode = session.mode.clone();
        let vision = state.vision.snapshot(state.config.vision_stale_ms);
        let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
        let csi = state.csi.snapshot(state.config.csi_stale_ms);
        let iphone_stereo_calibration = state.iphone_stereo_calibration.snapshot();
        let vision_conf = effective_vision_conf(&vision, &stereo);
        let vision_fresh = vision.fresh || stereo.fresh;
        let csi_conf = if csi.fresh { csi.csi_conf } else { 0.0 };
        let (source_mode, fused_conf, _coherence) =
            compute_fused_quality(&mode, vision_conf, csi_conf, vision_fresh, csi.fresh);

        // safety_state 由融合质量驱动（estop 优先）
        state.gate.update_safety_from_fused_conf(fused_conf);

        // operator -> wrist pose（operator_frame -> robot_base_frame）
        let op = state.operator.snapshot(
            state
                .config
                .vision_stale_ms
                .max(state.config.stereo_stale_ms),
        );
        let phone_session_matches = {
            let metrics_session_id = vision.metrics_session_id.trim();
            !metrics_session_id.is_empty() && metrics_session_id == session_id.as_str()
        };
        let phone_edge_authoritative_live =
            vision.fresh && vision.execution_mode.trim() == "edge_authoritative_phone_vision";
        let phone_session_aligned = phone_session_matches || phone_edge_authoritative_live;
        let phone_has_points = !vision.body_kpts_3d.is_empty() || !vision.hand_kpts_3d.is_empty();
        let associated_stereo_track_id = op
            .estimate
            .association
            .stereo_operator_track_id
            .clone()
            .or_else(|| {
                op.estimate
                    .association
                    .anchor_source
                    .contains("stereo")
                    .then(|| op.estimate.association.selected_operator_track_id.clone())
                    .flatten()
            });
        let matched_stereo_person = associated_stereo_track_id.as_ref().and_then(|track_id| {
            stereo
                .persons
                .iter()
                .find(|person| person.operator_track_id.as_deref() == Some(track_id.as_str()))
        });
        let phone_associated_stereo_preferred = phone_session_aligned
            && (!op.estimate.raw_pose.body_kpts_3d.is_empty() || matched_stereo_person.is_some())
            && associated_stereo_track_id.is_some();
        if phone_session_matches {
            session_id = vision.metrics_session_id.trim().to_string();
        }
        let phone_left_hand_joint_target = if hand_control_enabled && phone_session_aligned {
            let transformed = transform_points_3d(
                &vision.left_hand_kpts_3d,
                iphone_stereo_calibration.as_ref(),
            );
            let canonical = operator::canonical_hand_points_3d(&transformed, vision.hand_layout);
            target::hand_joint_target_from_hand_kpts_3d(&canonical, 0, true).or_else(|| {
                operator::hand_curls_from_vision_2d(
                    &vision.left_hand_kpts_2d,
                    vision.hand_layout,
                    vision.image_w,
                    vision.image_h,
                    true,
                )
                .map(target::hand_joint_target_from_curls)
            })
        } else {
            None
        };
        let phone_right_hand_joint_target = if hand_control_enabled && phone_session_aligned {
            let transformed = transform_points_3d(
                &vision.right_hand_kpts_3d,
                iphone_stereo_calibration.as_ref(),
            );
            let canonical = operator::canonical_hand_points_3d(&transformed, vision.hand_layout);
            target::hand_joint_target_from_hand_kpts_3d(&canonical, 0, false).or_else(|| {
                operator::hand_curls_from_vision_2d(
                    &vision.right_hand_kpts_2d,
                    vision.hand_layout,
                    vision.image_w,
                    vision.image_h,
                    false,
                )
                .map(target::hand_joint_target_from_curls)
            })
        } else {
            None
        };
        let (selected_operator_state, selected_left_hand_curls, selected_right_hand_curls) =
            if phone_session_aligned && phone_has_points && !phone_associated_stereo_preferred {
                let transformed_body =
                    transform_points_3d(&vision.body_kpts_3d, iphone_stereo_calibration.as_ref());
                let transformed_hands =
                    transform_points_3d(&vision.hand_kpts_3d, iphone_stereo_calibration.as_ref());
                let canonical_body =
                    operator::canonical_body_points_3d(&transformed_body, vision.body_layout);
                let canonical_hands =
                    operator::canonical_hand_points_3d(&transformed_hands, vision.hand_layout);
                let (operator_state, left_curls, right_curls) =
                    operator::operator_state_from_canonical_3d(canonical_body, canonical_hands);
                (operator_state, left_curls, right_curls)
            } else {
                (
                    op.estimate.operator_state.clone(),
                    op.estimate.left_hand_curls,
                    op.estimate.right_hand_curls,
                )
            };
        let selected_left_hand_curls = selected_left_hand_curls.or_else(|| {
            if phone_session_aligned {
                operator::hand_curls_from_vision_2d(
                    &vision.hand_kpts_2d,
                    vision.hand_layout,
                    vision.image_w,
                    vision.image_h,
                    true,
                )
            } else {
                None
            }
        });
        let selected_right_hand_curls = selected_right_hand_curls.or_else(|| {
            if phone_session_aligned {
                operator::hand_curls_from_vision_2d(
                    &vision.hand_kpts_2d,
                    vision.hand_layout,
                    vision.image_w,
                    vision.image_h,
                    false,
                )
            } else {
                None
            }
        });

        let desired_left = operator::transform_pose_operator_to_robot(
            &selected_operator_state.end_effector_pose.left,
            &state.config,
        );
        let desired_right = operator::transform_pose_operator_to_robot(
            &selected_operator_state.end_effector_pose.right,
            &state.config,
        );
        let mut body_kpts_robot = selected_operator_state
            .body_kpts_3d
            .iter()
            .copied()
            .map(|point| operator::transform_point_operator_to_robot(point, &state.config))
            .collect::<Vec<_>>();
        if body_kpts_robot.is_empty() {
            let fallback_body = if let Some(person) = matched_stereo_person {
                Some(person.body_kpts_3d.clone())
            } else if !vision.body_kpts_3d.is_empty() {
                Some(operator::canonical_body_points_3d(
                    &vision.body_kpts_3d,
                    vision.body_layout,
                ))
            } else if !stereo.body_kpts_3d.is_empty() {
                Some(operator::canonical_body_points_3d(
                    &stereo.body_kpts_3d,
                    stereo.body_layout,
                ))
            } else {
                None
            };
            if let Some(points) = fallback_body {
                body_kpts_robot = points
                    .iter()
                    .copied()
                    .map(|point| operator::transform_point_operator_to_robot(point, &state.config))
                    .collect();
            }
        }

        // 速度/加速度/jerk 限制（limit/freeze）；并回写 gate.safety_state（供 App/桥接观测）
        let limited = limiter.apply(edge_time_ns_now, &desired_left, &desired_right);
        state
            .gate
            .update_safety_from_motion(limited.safety_state, limited.reason.as_str());

        let left_hand_joint_target = if hand_control_enabled {
            phone_left_hand_joint_target.or_else(|| {
                target::hand_joint_target_from_hand_kpts_3d(
                    &selected_operator_state.hand_kpts_3d,
                    0,
                    true,
                )
            })
        } else {
            None
        };
        let right_hand_joint_target = if hand_control_enabled {
            phone_right_hand_joint_target.or_else(|| {
                target::hand_joint_target_from_hand_kpts_3d(
                    &selected_operator_state.hand_kpts_3d,
                    21,
                    false,
                )
            })
        } else {
            None
        };
        let left_hand_joint_target = left_hand_joint_target
            .or_else(|| selected_left_hand_curls.map(crate::target::hand_joint_target_from_curls));
        let right_hand_joint_target = right_hand_joint_target
            .or_else(|| selected_right_hand_curls.map(crate::target::hand_joint_target_from_curls));
        let left_hand_joints = if hand_control_enabled {
            left_hand_joint_target.map(|values| values.to_vec())
        } else {
            None
        };
        let right_hand_joints = if hand_control_enabled {
            right_hand_joint_target.map(|values| values.to_vec())
        } else {
            None
        };
        let canonical_robot_type =
            crate::target::canonical_robot_type(gate.robot_type_hint.as_str());
        let precomputed_targets = target_precomputer.compute(
            &trip_id,
            &session_id,
            canonical_robot_type,
            edge_time_ns_now,
            &body_kpts_robot,
            &limited.left,
            &limited.right,
            left_hand_joint_target,
            right_hand_joint_target,
        );
        let target_person_id = resolve_retarget_target_person_id(
            phone_session_aligned,
            vision.operator_track_id.as_deref(),
            associated_stereo_track_id.as_deref(),
            op.estimate.association.iphone_operator_track_id.as_deref(),
            op.estimate
                .association
                .selected_operator_track_id
                .as_deref(),
        );
        let operator_source_edge_time_ns = op
            .estimate
            .raw_pose
            .source_edge_time_ns
            .max(op.estimate.updated_edge_time_ns);
        let operator_body_source_edge_time_ns = operator::part_source_edge_time_ns(
            op.estimate.fusion_breakdown.body_source,
            stereo.last_edge_time_ns,
            vision.last_edge_time_ns,
            0,
            operator_source_edge_time_ns,
        );
        let operator_hand_source_edge_time_ns = operator::part_source_edge_time_ns(
            op.estimate.fusion_breakdown.hand_source,
            stereo.last_edge_time_ns,
            vision.last_edge_time_ns,
            0,
            operator_source_edge_time_ns,
        );
        let phone_hand_authoritative_selected = phone_session_aligned
            && (phone_left_hand_joint_target.is_some()
                || phone_right_hand_joint_target.is_some()
                || selected_left_hand_curls.is_some()
                || selected_right_hand_curls.is_some());
        let operator_body_selected = !op.estimate.raw_pose.body_kpts_3d.is_empty();
        let matched_stereo_person_available = matched_stereo_person.is_some();
        let vision_body_available = !vision.body_kpts_3d.is_empty();
        let operator_hand_selected = points_have_valid_3d(&selected_operator_state.hand_kpts_3d)
            && op.estimate.fusion_breakdown.hand_source
                != crate::operator::OperatorPartSource::None;
        let selected_body_source_edge_time_ns = resolve_selected_body_source_edge_time_ns(
            operator_body_selected,
            matched_stereo_person_available,
            vision_body_available,
            operator_body_source_edge_time_ns,
            stereo.last_edge_time_ns,
            vision.last_edge_time_ns,
            operator_source_edge_time_ns,
        );
        let selected_hand_source_edge_time_ns = resolve_selected_hand_source_edge_time_ns(
            operator_hand_selected,
            phone_hand_authoritative_selected,
            operator_hand_source_edge_time_ns,
            vision.last_edge_time_ns,
        );
        let retarget_source_edge_time_ns = if phone_session_aligned {
            if phone_associated_stereo_preferred {
                selected_body_source_edge_time_ns.max(selected_hand_source_edge_time_ns)
            } else {
                vision.last_edge_time_ns
            }
        } else if op.estimate.raw_pose.source_edge_time_ns > 0 {
            op.estimate.raw_pose.source_edge_time_ns
        } else {
            edge_time_ns_now
        };
        let end_effector_type = gate.end_effector_type_hint.clone();
        let waist_joint_layout =
            if body_control_enabled && precomputed_targets.waist_q_target.is_some() {
                Some(TeleopWaistJointLayout::UnitreeG1Waist1)
            } else {
                None
            };
        let leg_joint_layout = if body_control_enabled && precomputed_targets.leg_q_target.is_some()
        {
            Some(TeleopLegJointLayout::UnitreeG1Leg6x2)
        } else {
            None
        };
        let waist_q_target = if body_control_enabled {
            precomputed_targets.waist_q_target.clone()
        } else {
            None
        };
        let leg_q_target = if body_control_enabled {
            precomputed_targets.leg_q_target.clone()
        } else {
            None
        };
        let arm_q_target = if body_control_enabled {
            precomputed_targets.arm_q_target.clone()
        } else {
            None
        };
        let left_hand_target = if hand_control_enabled {
            precomputed_targets.left_hand_target.clone()
        } else {
            None
        };
        let right_hand_target = if hand_control_enabled {
            precomputed_targets.right_hand_target.clone()
        } else {
            None
        };
        let target_debug = if body_control_enabled || hand_control_enabled {
            precomputed_targets.target_debug.clone().into_option()
        } else {
            None
        };
        let quality = TeleopQuality {
            source_mode: source_mode.clone(),
            fused_conf,
            vision_conf,
            csi_conf,
        };

        let frame = TeleopFrameV1 {
            schema_version: "teleop_frame_v1",
            trip_id: trip_id.clone(),
            session_id: session_id.clone(),
            robot_type: canonical_robot_type.to_string(),
            end_effector_type: end_effector_type.clone(),
            edge_time_ns: edge_time_ns_now,
            operator_frame: state.config.operator_frame.clone(),
            robot_base_frame: state.config.robot_base_frame.clone(),
            extrinsic_version: state.config.extrinsic_version.clone(),
            control_state: control_state.to_string(),
            teleop_enabled,
            body_control_enabled,
            hand_control_enabled,
            left_wrist_pose: if body_control_enabled {
                limited.left
            } else {
                crate::ws::types::Pose {
                    pos: [0.0, 0.0, 0.0],
                    quat: [0.0, 0.0, 0.0, 1.0],
                }
            },
            right_wrist_pose: if body_control_enabled {
                limited.right
            } else {
                crate::ws::types::Pose {
                    pos: [0.0, 0.0, 0.0],
                    quat: [0.0, 0.0, 0.0, 1.0],
                }
            },
            hand_joint_layout: hand_control_enabled
                .then_some(TeleopHandJointLayout::AnatomicalJoint16),
            hand_target_layout: hand_control_enabled
                .then_some(TeleopHandTargetLayout::AnatomicalTarget16),
            left_hand_joints,
            right_hand_joints,
            left_hand_target: left_hand_target.clone(),
            right_hand_target: right_hand_target.clone(),
            waist_joint_layout,
            leg_joint_layout,
            waist_q_target: waist_q_target.clone(),
            leg_q_target: leg_q_target.clone(),
            arm_q_target: arm_q_target.clone(),
            arm_tauff_target: None,
            target_debug: target_debug.clone(),
            quality: quality.clone(),
            safety_state: match state.gate.safety_state() {
                crate::control::gate::SafetyState::Normal => "normal",
                crate::control::gate::SafetyState::Limit => "limit",
                crate::control::gate::SafetyState::Freeze => "freeze",
                crate::control::gate::SafetyState::Estop => "estop",
            }
            .to_string(),
        };
        let retarget_reference = RetargetReferenceV1 {
            schema_version: "retarget_reference_v1",
            trip_id: trip_id.clone(),
            session_id: session_id.clone(),
            source_session_id: session_id.clone(),
            target_person_id,
            source_kind: "target_human_state",
            source_edge_time_ns: retarget_source_edge_time_ns,
            robot_type: canonical_robot_type.to_string(),
            end_effector_type,
            edge_time_ns: edge_time_ns_now,
            control_state: control_state.to_string(),
            body_control_enabled,
            hand_control_enabled,
            hand_target_layout: hand_control_enabled
                .then_some(TeleopHandTargetLayout::AnatomicalTarget16),
            waist_joint_layout,
            leg_joint_layout,
            waist_q_target,
            leg_q_target,
            arm_q_target,
            left_hand_target,
            right_hand_target,
            target_debug,
            quality,
            retarget_status: "retarget_reference_v1",
        };

        let record = if !trip_id.is_empty() && !session_id.is_empty() {
            serde_json::to_value(&frame).ok()
        } else {
            None
        };
        if let Ok(mut latest) = state.teleop_latest.lock() {
            *latest = Some(frame.clone());
        }
        if let Ok(mut latest) = state.retarget_latest.lock() {
            *latest = Some(retarget_reference);
        }
        let _ = state.teleop_tx.send(frame);

        // 落盘（JSONL）
        if let Some(v) = record {
            state
                .recorder
                .record_teleop_frame(&state.protocol, &state.config, &trip_id, &session_id, &v)
                .await;
        }
    }
}

fn emit_health_gauges(
    state: &AppState,
    gate: &crate::control::gate::GateSnapshot,
    deadman: &crate::control::gate::DeadmanSnapshot,
    bridge: &crate::control::gate::BridgeSnapshot,
) {
    metrics::gauge!("bridge_unitree_ready").set(bool_to_f64(bridge.unitree_ready));
    metrics::gauge!("bridge_leap_ready").set(bool_to_f64(bridge.leap_ready));
    metrics::gauge!("bridge_lan_control_ok").set(bool_to_f64(bridge.lan_control_ok));

    metrics::gauge!("deadman_link_ok").set(bool_to_f64(deadman.link_ok));
    metrics::gauge!("deadman_pressed").set(bool_to_f64(deadman.pressed));

    metrics::gauge!("time_sync_ok").set(bool_to_f64(state.gate.time_sync_ok(
        state.config.time_sync_ok_window_ms,
        state.config.time_sync_rtt_ok_ms,
    )));
    metrics::gauge!("extrinsic_ok").set(bool_to_f64(state.gate.extrinsic_ok()));

    metrics::gauge!("control_armed").set(bool_to_f64(gate.state == "armed"));
    metrics::gauge!("control_fault").set(bool_to_f64(gate.state == "fault"));

    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let wifi_pose = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let csi = state.csi.snapshot(state.config.csi_stale_ms);
    metrics::gauge!("vision_conf").set(vision.vision_conf as f64);
    metrics::gauge!("wifi_pose_body_conf").set(wifi_pose.body_confidence as f64);
    metrics::gauge!("wifi_pose_fresh").set(bool_to_f64(wifi_pose.fresh));
    metrics::gauge!("csi_conf").set(csi.csi_conf as f64);
}

fn compute_edge_to_robot_ms(
    edge_time_ns_now: u64,
    bridge: &crate::control::gate::BridgeSnapshot,
) -> f32 {
    let last_cmd = bridge
        .unitree_last_command_edge_time_ns
        .max(bridge.leap_last_command_edge_time_ns);
    if last_cmd == 0 {
        return 0.0;
    }
    (edge_time_ns_now.saturating_sub(last_cmd) as f32) / 1_000_000.0
}

fn bool_to_f64(v: bool) -> f64 {
    if v {
        1.0
    } else {
        0.0
    }
}

struct MotionLimiter {
    max_speed: f32,
    max_accel: f32,
    max_jerk: f32,
    left: Option<MotionSample>,
    right: Option<MotionSample>,
}

#[derive(Clone, Copy)]
struct MotionSample {
    edge_time_ns: u64,
    pos: [f32; 3],
    quat: [f32; 4],
    vel: [f32; 3],
    acc: [f32; 3],
}

struct LimitedPoses {
    left: Pose,
    right: Pose,
    safety_state: crate::control::gate::SafetyState,
    reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MotionGrade {
    Normal,
    Limit,
    Freeze,
}

impl MotionLimiter {
    fn new(max_speed: f32, max_accel: f32, max_jerk: f32) -> Self {
        Self {
            max_speed,
            max_accel,
            max_jerk,
            left: None,
            right: None,
        }
    }

    fn apply(&mut self, edge_time_ns: u64, left: &Pose, right: &Pose) -> LimitedPoses {
        let max_speed = self.max_speed;
        let max_accel = self.max_accel;
        let max_jerk = self.max_jerk;

        let (left_out, left_grade, left_reason) = apply_one(
            max_speed,
            max_accel,
            max_jerk,
            edge_time_ns,
            left,
            &mut self.left,
        );
        let (right_out, right_grade, right_reason) = apply_one(
            max_speed,
            max_accel,
            max_jerk,
            edge_time_ns,
            right,
            &mut self.right,
        );

        let grade = max_grade(left_grade, right_grade);
        let (safety_state, reason) = match grade {
            MotionGrade::Freeze => (
                crate::control::gate::SafetyState::Freeze,
                if left_grade == MotionGrade::Freeze {
                    left_reason
                } else {
                    right_reason
                },
            ),
            MotionGrade::Limit => (
                crate::control::gate::SafetyState::Limit,
                if left_grade == MotionGrade::Limit {
                    left_reason
                } else {
                    right_reason
                },
            ),
            MotionGrade::Normal => (crate::control::gate::SafetyState::Normal, "".to_string()),
        };

        if safety_state == crate::control::gate::SafetyState::Limit {
            metrics::counter!("motion_limit_count").increment(1);
        } else if safety_state == crate::control::gate::SafetyState::Freeze {
            metrics::counter!("motion_freeze_count").increment(1);
        }

        LimitedPoses {
            left: left_out,
            right: right_out,
            safety_state,
            reason,
        }
    }
}

fn apply_one(
    max_speed: f32,
    max_accel: f32,
    max_jerk: f32,
    edge_time_ns: u64,
    desired: &Pose,
    sample: &mut Option<MotionSample>,
) -> (Pose, MotionGrade, String) {
    let prev = match *sample {
        Some(p) => p,
        None => {
            if is_default_pose(desired) {
                return (desired.clone(), MotionGrade::Normal, "".to_string());
            }
            *sample = Some(MotionSample {
                edge_time_ns,
                pos: desired.pos,
                quat: desired.quat,
                vel: [0.0, 0.0, 0.0],
                acc: [0.0, 0.0, 0.0],
            });
            return (desired.clone(), MotionGrade::Normal, "".to_string());
        }
    };

    if is_default_sample(prev) && !is_default_pose(desired) {
        *sample = Some(MotionSample {
            edge_time_ns,
            pos: desired.pos,
            quat: desired.quat,
            vel: [0.0, 0.0, 0.0],
            acc: [0.0, 0.0, 0.0],
        });
        return (desired.clone(), MotionGrade::Normal, "".to_string());
    }

    // “缺手/缺输入”兜底：如果上游给了默认 pose（0,0,0 + 单位四元数），则保持上一帧输出。
    // - 目的：避免单手丢失时，另一只手/手臂被拉回原点导致突变。
    if is_default_pose(desired) {
        return (
            Pose {
                pos: prev.pos,
                quat: prev.quat,
            },
            MotionGrade::Normal,
            "".to_string(),
        );
    }

    let dt_ns = edge_time_ns.saturating_sub(prev.edge_time_ns);
    let dt_s = (dt_ns as f32) / 1_000_000_000.0;
    if !dt_s.is_finite() || dt_s <= 1e-4 {
        return (desired.clone(), MotionGrade::Normal, "".to_string());
    }

    let dp = [
        desired.pos[0] - prev.pos[0],
        desired.pos[1] - prev.pos[1],
        desired.pos[2] - prev.pos[2],
    ];
    let v_des = [dp[0] / dt_s, dp[1] / dt_s, dp[2] / dt_s];

    let mut grade = MotionGrade::Normal;
    let mut reason = String::new();

    // 1) speed clamp
    let mut v = v_des;
    if max_speed.is_finite() && max_speed > 0.0 {
        let sp = norm3(v);
        if sp.is_finite() && sp > max_speed {
            let s = max_speed / sp;
            v = [v[0] * s, v[1] * s, v[2] * s];
            grade = MotionGrade::Limit;
            reason = "motion_speed".to_string();
        }
    }

    // 2) accel clamp
    let mut acc = [
        (v[0] - prev.vel[0]) / dt_s,
        (v[1] - prev.vel[1]) / dt_s,
        (v[2] - prev.vel[2]) / dt_s,
    ];
    if max_accel.is_finite() && max_accel > 0.0 {
        let acc_norm = norm3(acc);
        if acc_norm.is_finite() && acc_norm > max_accel && acc_norm > 1e-6 {
            let s = max_accel / acc_norm;
            acc = [acc[0] * s, acc[1] * s, acc[2] * s];
            v = [
                prev.vel[0] + acc[0] * dt_s,
                prev.vel[1] + acc[1] * dt_s,
                prev.vel[2] + acc[2] * dt_s,
            ];
            grade = MotionGrade::Limit;
            if reason.is_empty() {
                reason = "motion_accel".to_string();
            }
        }
    }

    // 3) jerk clamp
    if max_jerk.is_finite() && max_jerk > 0.0 {
        let dacc = [
            acc[0] - prev.acc[0],
            acc[1] - prev.acc[1],
            acc[2] - prev.acc[2],
        ];
        let dacc_norm = norm3(dacc);
        let dacc_max = max_jerk * dt_s;
        if dacc_norm.is_finite() && dacc_norm > dacc_max && dacc_norm > 1e-6 {
            let s = dacc_max / dacc_norm;
            acc = [
                prev.acc[0] + dacc[0] * s,
                prev.acc[1] + dacc[1] * s,
                prev.acc[2] + dacc[2] * s,
            ];
            v = [
                prev.vel[0] + acc[0] * dt_s,
                prev.vel[1] + acc[1] * dt_s,
                prev.vel[2] + acc[2] * dt_s,
            ];
            grade = MotionGrade::Limit;
            reason = "motion_jerk".to_string();
        }
    }

    let pos = [
        prev.pos[0] + v[0] * dt_s,
        prev.pos[1] + v[1] * dt_s,
        prev.pos[2] + v[2] * dt_s,
    ];
    let out = Pose {
        pos,
        quat: desired.quat,
    };
    *sample = Some(MotionSample {
        edge_time_ns,
        pos,
        quat: desired.quat,
        vel: v,
        acc,
    });
    (out, grade, reason)
}

fn max_grade(a: MotionGrade, b: MotionGrade) -> MotionGrade {
    use MotionGrade::*;
    match (a, b) {
        (Freeze, _) | (_, Freeze) => Freeze,
        (Limit, _) | (_, Limit) => Limit,
        _ => Normal,
    }
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn is_default_pose(p: &Pose) -> bool {
    let pos0 = p.pos[0].abs() <= 1e-6 && p.pos[1].abs() <= 1e-6 && p.pos[2].abs() <= 1e-6;
    let quat_id = (p.quat[0].abs() <= 1e-6)
        && (p.quat[1].abs() <= 1e-6)
        && (p.quat[2].abs() <= 1e-6)
        && ((p.quat[3] - 1.0).abs() <= 1e-6);
    pos0 && quat_id
}

fn is_default_sample(sample: MotionSample) -> bool {
    is_default_pose(&Pose {
        pos: sample.pos,
        quat: sample.quat,
    })
}

fn effective_vision_conf(
    vision: &crate::sensing::VisionSnapshot,
    stereo: &crate::sensing::StereoSnapshot,
) -> f32 {
    let vision_conf = if vision.fresh {
        vision.vision_conf
    } else {
        0.0
    };
    let stereo_conf =
        if stereo.fresh && (!stereo.body_kpts_3d.is_empty() || !stereo.hand_kpts_3d.is_empty()) {
            stereo.stereo_confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
    vision_conf.max(stereo_conf)
}

fn compute_fused_quality(
    mode: &str,
    vision_conf: f32,
    csi_conf: f32,
    vision_fresh: bool,
    csi_fresh: bool,
) -> (String, f32, f32) {
    match mode {
        "vision_only" => ("vision".to_string(), vision_conf, 0.0),
        "csi_only" => ("csi".to_string(), csi_conf, 0.0),
        _ => {
            if vision_fresh && csi_fresh {
                let fused = (vision_conf * 0.7 + csi_conf * 0.3).clamp(0.0, 1.0);
                let coherence = (1.0 - (vision_conf - csi_conf).abs()).clamp(0.0, 1.0);
                ("fused".to_string(), fused, coherence)
            } else if vision_fresh {
                ("vision".to_string(), vision_conf, 0.0)
            } else if csi_fresh {
                ("csi".to_string(), csi_conf, 0.0)
            } else {
                ("fused".to_string(), 0.0, 0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_one, effective_vision_conf, resolve_retarget_target_person_id,
        resolve_selected_body_source_edge_time_ns, resolve_selected_hand_source_edge_time_ns,
        MotionGrade, MotionSample,
    };
    use crate::sensing::{StereoSnapshot, VisionSnapshot};
    use crate::ws::types::Pose;

    #[test]
    fn motion_limiter_should_not_seed_history_from_default_pose() {
        let mut sample = None;
        let default_pose = Pose {
            pos: [0.0, 0.0, 0.0],
            quat: [0.0, 0.0, 0.0, 1.0],
        };
        let desired_pose = Pose {
            pos: [0.18, 0.24, 1.09],
            quat: [0.01, -0.02, 0.11, 0.99],
        };

        let (_, first_grade, _) = apply_one(1.5, 12.0, 250.0, 0, &default_pose, &mut sample);
        assert_eq!(first_grade, MotionGrade::Normal);
        assert!(sample.is_none(), "默认 pose 不应成为 motion history");

        let (out, second_grade, _) =
            apply_one(1.5, 12.0, 250.0, 20_000_000, &desired_pose, &mut sample);
        assert_eq!(second_grade, MotionGrade::Normal);
        assert_eq!(out.pos, desired_pose.pos);
        assert_eq!(out.quat, desired_pose.quat);

        let stored = sample.expect("真实 pose 应写入 motion history");
        assert_eq!(stored.pos, desired_pose.pos);
        assert_eq!(stored.quat, desired_pose.quat);
    }

    #[test]
    fn motion_limiter_should_reset_legacy_default_sample_before_first_real_pose() {
        let mut sample = Some(MotionSample {
            edge_time_ns: 0,
            pos: [0.0, 0.0, 0.0],
            quat: [0.0, 0.0, 0.0, 1.0],
            vel: [0.0, 0.0, 0.0],
            acc: [0.0, 0.0, 0.0],
        });
        let desired_pose = Pose {
            pos: [-0.16, 0.21, 0.96],
            quat: [0.0, 0.0, 0.13052619, 0.9914449],
        };

        let (out, grade, _) = apply_one(1.5, 12.0, 250.0, 20_000_000, &desired_pose, &mut sample);
        assert_eq!(grade, MotionGrade::Normal);
        assert_eq!(out.pos, desired_pose.pos);
        assert_eq!(out.quat, desired_pose.quat);
    }

    #[test]
    fn motion_limiter_should_limit_jerk_without_getting_stuck_in_freeze_loop() {
        let mut sample = Some(MotionSample {
            edge_time_ns: 0,
            pos: [0.0, 0.0, 0.8],
            quat: [0.0, 0.0, 0.0, 1.0],
            vel: [0.0, 0.0, 0.0],
            acc: [0.0, 0.0, 0.0],
        });
        let desired_pose = Pose {
            pos: [0.18, 0.28, 0.84],
            quat: [0.0, 0.0, 0.0, 1.0],
        };

        let (first_out, first_grade, first_reason) =
            apply_one(1.5, 12.0, 250.0, 20_000_000, &desired_pose, &mut sample);
        assert_eq!(first_grade, MotionGrade::Limit);
        assert_eq!(first_reason, "motion_jerk");
        assert_ne!(first_out.pos, [0.0, 0.0, 0.8]);

        let (second_out, second_grade, second_reason) =
            apply_one(1.5, 12.0, 250.0, 40_000_000, &desired_pose, &mut sample);
        assert_ne!(second_grade, MotionGrade::Freeze);
        assert!(matches!(
            second_grade,
            MotionGrade::Normal | MotionGrade::Limit
        ));
        assert!(!second_reason.is_empty());
        assert!(second_out.pos[0] > first_out.pos[0]);
        assert!(second_out.pos[1] > first_out.pos[1]);
    }

    #[test]
    fn stereo_body_only_frames_contribute_vision_conf() {
        let stereo = StereoSnapshot {
            stereo_confidence: 0.62,
            body_kpts_3d: vec![[0.0, 0.0, 1.0]; 17],
            fresh: true,
            ..Default::default()
        };
        let vision = VisionSnapshot::default();
        assert!((effective_vision_conf(&vision, &stereo) - 0.62).abs() < f32::EPSILON);
    }

    #[test]
    fn selected_body_source_prefers_associated_stereo_over_phone_fallback() {
        let edge_time_ns = resolve_selected_body_source_edge_time_ns(
            false, true, true, 1_000, 6_000, 4_000, 2_000,
        );
        assert_eq!(edge_time_ns, 6_000);
    }

    #[test]
    fn selected_hand_source_uses_phone_authoritative_time_when_consumed() {
        let edge_time_ns = resolve_selected_hand_source_edge_time_ns(true, true, 5_000, 9_000);
        assert_eq!(edge_time_ns, 9_000);
    }

    #[test]
    fn retarget_target_person_id_prefers_phone_track_when_phone_session_aligned() {
        let target_person_id = resolve_retarget_target_person_id(
            true,
            Some("sim_primary_operator"),
            Some("sim_primary_operator"),
            Some("sim_primary_operator"),
            Some("stereo-main"),
        );
        assert_eq!(target_person_id, "sim_primary_operator");
    }
}
