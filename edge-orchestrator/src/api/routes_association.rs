use std::net::SocketAddr;

use axum::extract::{ConnectInfo, State};
use axum::http::HeaderMap;
use axum::{routing::get, Json, Router};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::calibration::transform_points_3d;
use crate::calibration::IphoneStereoExtrinsic;
use crate::operator::{
    canonicalize_body_points_3d, canonicalize_hand_points_3d, stabilize_wifi_canonical_body_points,
    OperatorPartSource, OperatorSource, CANONICAL_BODY_FRAME, OPERATOR_FRAME, STEREO_PAIR_FRAME,
};
use crate::sensing::{StereoTrackedPersonSnapshot, VisionDevicePose, VisionSnapshot};
use crate::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/association/hint", get(get_association_hint))
        .route("/association/teacher", get(get_association_teacher))
        .with_state(state)
}

fn phone_hand_geometry_is_hint_trustworthy(vision: &VisionSnapshot) -> bool {
    let source = vision.hand_3d_source.trim();
    if source.is_empty() || source == "none" {
        return false;
    }
    !matches!(source, "depth_reprojected" | "edge_depth_reprojected")
}

async fn get_association_hint(
    State(state): State<AppState>,
    ConnectInfo(client_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let edge_time_ns = state.gate.edge_time_ns();
    state
        .association_hint_clients
        .record_request(crate::control::gate::ClientOriginUpdate {
            edge_time_ns,
            client_addr: client_addr.to_string(),
            forwarded_for: extract_header_text(&headers, "x-forwarded-for"),
            user_agent: extract_header_text(&headers, "user-agent"),
        });
    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let wifi = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let operator = state.operator.snapshot(state.config.operator_hold_ms);
    let iphone_calibration = state.iphone_stereo_calibration.snapshot();
    let wifi_calibration = state.wifi_stereo_calibration.snapshot();

    let phone_pose_processing_enabled = state.config.phone_vision_processing_enabled;
    let expose_phone_pose_points = vision.fresh && phone_pose_processing_enabled;
    let expose_phone_hand_geometry =
        expose_phone_pose_points && phone_hand_geometry_is_hint_trustworthy(&vision);
    let vision_hand = canonicalize_hand_points_3d(
        &transform_points_3d(&vision.hand_kpts_3d, iphone_calibration.as_ref()),
        vision.hand_layout,
    );
    let vision_body_kpts_2d = if expose_phone_pose_points {
        vision.body_kpts_2d.clone()
    } else {
        Vec::new()
    };
    let vision_hand_kpts_2d = if expose_phone_pose_points {
        vision.hand_kpts_2d.clone()
    } else {
        Vec::new()
    };
    let vision_body_kpts_3d = if expose_phone_pose_points {
        vision.body_kpts_3d.clone()
    } else {
        Vec::new()
    };
    let vision_hand_kpts_3d = if expose_phone_pose_points {
        vision.hand_kpts_3d.clone()
    } else {
        Vec::new()
    };
    let raw_hand_points_2d_valid = valid_point_count_2d(&vision_hand_kpts_2d);
    let raw_body_points_2d_valid = valid_point_count_2d(&vision_body_kpts_2d);
    let left_hand_points_2d_valid = hand_valid_point_count_2d(&vision_hand_kpts_2d, true);
    let right_hand_points_2d_valid = hand_valid_point_count_2d(&vision_hand_kpts_2d, false);
    let raw_hand_points_3d_valid = valid_point_count_3d(&vision_hand_kpts_3d);
    let raw_body_points_3d_valid = valid_point_count_3d(&vision_body_kpts_3d);
    let left_hand_points_3d_valid = hand_valid_point_count_3d(&vision_hand_kpts_3d, true);
    let right_hand_points_3d_valid = hand_valid_point_count_3d(&vision_hand_kpts_3d, false);
    let iphone_available_3d = !vision_hand_kpts_3d.is_empty() || !vision_body_kpts_3d.is_empty();
    let iphone_available_2d = raw_hand_points_2d_valid > 0 || raw_body_points_2d_valid > 0;
    let stereo_body = canonicalize_body_points_3d(&stereo.body_kpts_3d, stereo.body_layout);
    let wifi_body_space = normalized_wifi_body_space(&wifi.body_space, wifi_calibration.is_some());
    let wifi_body_points = if wifi_body_space == CANONICAL_BODY_FRAME {
        wifi.body_kpts_3d.clone()
    } else {
        transform_points_3d(&wifi.body_kpts_3d, wifi_calibration.as_ref())
    };
    let wifi_body = stabilize_wifi_canonical_body_points(&canonicalize_body_points_3d(
        &wifi_body_points,
        wifi.body_layout,
    ));
    let wifi_body_debug = wifi_body.clone();
    let wifi_debug_space = wifi_body_space;
    let device_pose_anchor =
        resolve_stereo_device_anchor(vision.device_pose.as_ref(), iphone_calibration.as_ref());
    let device_pose_stereo_candidate = select_device_pose_stereo_candidate(
        &stereo.persons,
        device_pose_anchor.as_ref().map(|(anchor, _)| *anchor),
    );

    Json(json!({
        "edge_time_ns": edge_time_ns,
        "iphone": {
            "available": iphone_available_2d || iphone_available_3d,
            "available_2d": iphone_available_2d,
            "available_3d": iphone_available_3d,
            "fresh": vision.fresh,
            "operator_track_id": vision.operator_track_id,
            "hand_layout": vision.hand_layout.as_str(),
            "device_class": vision.device_class,
            "camera_mode": vision.camera_mode,
            "camera_has_depth": vision.camera_has_depth,
            "body_3d_source": if expose_phone_pose_points { vision.body_3d_source.clone() } else { String::new() },
            "hand_3d_source": if expose_phone_pose_points { vision.hand_3d_source.clone() } else { String::new() },
            "execution_mode": if vision.fresh && !phone_pose_processing_enabled {
                "device_pose_passthrough".to_string()
            } else if expose_phone_pose_points {
                vision.execution_mode.clone()
            } else {
                String::new()
            },
            "aux_snapshot_present": vision.aux_snapshot_present,
            "aux_body_points_2d_valid": vision.aux_body_points_2d_valid,
            "aux_hand_points_2d_valid": vision.aux_hand_points_2d_valid,
            "aux_body_points_3d_filled": vision.aux_body_points_3d_filled,
            "aux_hand_points_3d_filled": vision.aux_hand_points_3d_filled,
            "aux_support_state": vision.aux_support_state,
            "raw_body_count_2d": vision_body_kpts_2d.len(),
            "raw_body_count_3d": vision_body_kpts_3d.len(),
            "raw_body_points_2d_valid": raw_body_points_2d_valid,
            "raw_body_points_3d_valid": raw_body_points_3d_valid,
            "raw_hand_count_2d": vision_hand_kpts_2d.len(),
            "raw_hand_count_3d": vision_hand_kpts_3d.len(),
            "raw_hand_points_2d_valid": raw_hand_points_2d_valid,
            "raw_hand_points_3d_valid": raw_hand_points_3d_valid,
            "left_hand_points_2d_valid": left_hand_points_2d_valid,
            "left_hand_points_3d_valid": left_hand_points_3d_valid,
            "right_hand_points_2d_valid": right_hand_points_2d_valid,
            "right_hand_points_3d_valid": right_hand_points_3d_valid,
            "left_wrist_2d": hand_wrist_2d(&vision_hand_kpts_2d, true),
            "right_wrist_2d": hand_wrist_2d(&vision_hand_kpts_2d, false),
            "left_wrist": if expose_phone_hand_geometry { hand_wrist(&vision_hand, true) } else { None },
            "right_wrist": if expose_phone_hand_geometry { hand_wrist(&vision_hand, false) } else { None },
            "device_pose": if vision.fresh { vision.device_pose.clone() } else { None },
            "stereo_anchor_position_m": device_pose_anchor.as_ref().map(|(anchor, _)| *anchor),
            "stereo_anchor_source": device_pose_anchor.as_ref().map(|(_, source)| *source),
            "imu": if vision.fresh { vision.imu.clone() } else { None },
        },
        "stereo": {
            "available": !stereo_body.is_empty(),
            "fresh": stereo.fresh,
            "operator_track_id": stereo.operator_track_id,
            "body_layout": stereo.body_layout.as_str(),
            "body_space": if stereo.body_space.is_empty() { STEREO_PAIR_FRAME } else { stereo.body_space.as_str() },
            "left_shoulder": body_joint(&stereo_body, 5),
            "right_shoulder": body_joint(&stereo_body, 6),
            "left_wrist": body_joint(&stereo_body, 9),
            "right_wrist": body_joint(&stereo_body, 10),
            "persons": stereo.persons.iter().map(|person| json!({
                "operator_track_id": person.operator_track_id,
                "body_center": stereo_person_body_center(person),
                "left_shoulder": body_joint(&person.body_kpts_3d, 5),
                "right_shoulder": body_joint(&person.body_kpts_3d, 6),
                "left_wrist": body_joint(&person.body_kpts_3d, 9),
                "right_wrist": body_joint(&person.body_kpts_3d, 10),
                "stereo_confidence": person.stereo_confidence,
            })).collect::<Vec<_>>(),
        },
        "wifi": {
            "available": !wifi_body_debug.is_empty(),
            "fresh": wifi.fresh,
            "operator_track_id": wifi.operator_track_id,
            "body_layout": wifi.body_layout.as_str(),
            "body_space": wifi_debug_space,
            "left_shoulder": body_joint(&wifi_body_debug, 5),
            "right_shoulder": body_joint(&wifi_body_debug, 6),
            "left_wrist": body_joint(&wifi_body_debug, 9),
            "right_wrist": body_joint(&wifi_body_debug, 10),
        },
        "association": {
            "selected_operator_track_id": operator.estimate.association.selected_operator_track_id,
            "anchor_source": operator.estimate.association.anchor_source,
            "stereo_operator_track_id": operator.estimate.association.stereo_operator_track_id,
            "wifi_operator_track_id": operator.estimate.association.wifi_operator_track_id,
            "iphone_operator_track_id": operator.estimate.association.iphone_operator_track_id,
            "iphone_visible_hand_count": operator.estimate.association.iphone_visible_hand_count,
            "hand_match_count": operator.estimate.association.hand_match_count,
            "hand_match_score": operator.estimate.association.hand_match_score,
            "wifi_association_score": operator.estimate.association.wifi_association_score,
            "wifi_layout_score": operator.estimate.association.wifi_layout_score,
            "wifi_zone_score": operator.estimate.association.wifi_zone_score,
            "wifi_motion_energy": operator.estimate.association.wifi_motion_energy,
            "wifi_doppler_hz": operator.estimate.association.wifi_doppler_hz,
            "wifi_signal_quality": operator.estimate.association.wifi_signal_quality,
            "wifi_zone_summary_reliable": operator.estimate.association.wifi_zone_summary_reliable,
            "left_wrist_gap_m": operator.estimate.association.left_wrist_gap_m,
            "right_wrist_gap_m": operator.estimate.association.right_wrist_gap_m,
            "device_pose_candidate_track_id": device_pose_stereo_candidate
                .as_ref()
                .map(|candidate| candidate.track_id.as_str()),
            "device_pose_candidate_distance_m": device_pose_stereo_candidate
                .as_ref()
                .map(|candidate| candidate.distance_m),
            "device_pose_candidate_margin_m": device_pose_stereo_candidate
                .as_ref()
                .map(|candidate| candidate.margin_m),
            "device_pose_candidate_source": device_pose_anchor
                .as_ref()
                .map(|(_, source)| *source),
        },
        "calibration": {
            "iphone_stereo_available": iphone_calibration.is_some(),
            "iphone_stereo": iphone_calibration,
            "wifi_stereo_available": wifi_calibration.is_some(),
            "wifi_stereo": wifi_calibration,
        },
        "wifi_diagnostics": {
            "layout_node_count": wifi.diagnostics.layout_node_count,
            "layout_score": wifi.diagnostics.layout_score,
            "zone_score": wifi.diagnostics.zone_score,
            "zone_summary_reliable": wifi.diagnostics.zone_summary_reliable,
            "motion_energy": wifi.diagnostics.motion_energy,
            "doppler_hz": wifi.diagnostics.doppler_hz,
            "signal_quality": wifi.diagnostics.signal_quality,
            "vital_signal_quality": wifi.diagnostics.vital_signal_quality,
            "stream_fps": wifi.diagnostics.stream_fps,
        }
    }))
}

fn extract_header_text(headers: &HeaderMap, key: &str) -> String {
    headers
        .get(key)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_default()
        .to_string()
}

async fn get_association_teacher(State(state): State<AppState>) -> Json<serde_json::Value> {
    let edge_time_ns = state.gate.edge_time_ns();
    let wall_time_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let vision = state.vision.snapshot(state.config.vision_stale_ms);
    let stereo = state.stereo.snapshot(state.config.stereo_stale_ms);
    let wifi = state.wifi_pose.snapshot(state.config.wifi_pose_stale_ms);
    let operator = state.operator.snapshot(state.config.operator_hold_ms);
    let iphone_calibration = state.iphone_stereo_calibration.snapshot();
    let wifi_calibration = state.wifi_stereo_calibration.snapshot();

    let iphone_hand = canonicalize_hand_points_3d(
        &transform_points_3d(&vision.hand_kpts_3d, iphone_calibration.as_ref()),
        vision.hand_layout,
    );
    let stereo_body = canonicalize_body_points_3d(&stereo.body_kpts_3d, stereo.body_layout);
    let stereo_body_space = if stereo.body_space.is_empty() {
        STEREO_PAIR_FRAME
    } else {
        stereo.body_space.as_str()
    };
    let wifi_body_space = normalized_wifi_body_space(&wifi.body_space, wifi_calibration.is_some());
    let wifi_body_points = if wifi_body_space == CANONICAL_BODY_FRAME {
        wifi.body_kpts_3d.clone()
    } else {
        transform_points_3d(&wifi.body_kpts_3d, wifi_calibration.as_ref())
    };
    let wifi_body = stabilize_wifi_canonical_body_points(&canonicalize_body_points_3d(
        &wifi_body_points,
        wifi.body_layout,
    ));
    let wifi_body_debug = wifi_body.clone();
    let wifi_debug_space = wifi_body_space;
    let fused_body = operator.estimate.operator_state.body_kpts_3d.clone();
    let fused_hand = operator.estimate.operator_state.hand_kpts_3d.clone();

    let stereo_valid = valid_point_count_3d(&stereo_body);
    let fused_valid = valid_point_count_3d(&fused_body);
    let wifi_valid = valid_point_count_3d(&wifi_body_debug);
    let teacher = select_teacher_body(
        operator.estimate.source,
        &operator.estimate.association.anchor_source,
        &fused_body,
        fused_valid,
        &stereo_body,
        stereo_valid,
    );
    let fused_body_space = fused_body_space_from_breakdown(
        operator.estimate.fusion_breakdown.body_source,
        stereo_body_space,
        wifi_body_space,
    );
    let iphone_hand_space = normalized_iphone_hand_space(iphone_calibration.as_ref());
    let stereo_hand_space = stereo_body_space;
    let fused_hand_space =
        fused_hand_space_from_source(operator.estimate.source, stereo_hand_space);

    Json(json!({
        "timestamp": wall_time_s,
        "edge_time_ns": edge_time_ns,
        "teacher": {
            "available": teacher.is_some(),
            "source": teacher.as_ref().map(|(source, _, _)| *source),
            "body_layout": teacher.as_ref().map(|(_, layout, _)| *layout),
            "body_space": teacher.as_ref().map(|(source, _, _)| if *source == "stereo" { stereo_body_space } else { fused_body_space }),
            "body_kpts_3d": teacher.as_ref().map(|(_, _, points)| points.to_vec()),
            "valid_joint_count": teacher.as_ref().map(|(_, _, points)| valid_point_count_3d(points)).unwrap_or(0),
        },
        "fused": {
            "available": fused_valid > 0,
            "source": operator.estimate.source.as_str(),
            "body_layout": "coco_body_17",
            "body_space": fused_body_space,
            "valid_joint_count": fused_valid,
            "body_kpts_3d": fused_body,
            "hand_layout": "mediapipe_hand_21x2",
            "hand_space": fused_hand_space,
            "hand_kpts_3d": fused_hand,
        },
        "stereo": {
            "available": stereo_valid > 0,
            "fresh": stereo.fresh,
            "operator_track_id": stereo.operator_track_id,
            "body_layout": stereo.body_layout.as_str(),
            "body_space": stereo_body_space,
            "valid_joint_count": stereo_valid,
            "body_kpts_3d": stereo_body,
        },
        "wifi": {
            "available": wifi_valid > 0,
            "fresh": wifi.fresh,
            "operator_track_id": wifi.operator_track_id,
            "body_layout": wifi.body_layout.as_str(),
            "body_space": wifi_debug_space,
            "valid_joint_count": wifi_valid,
            "body_kpts_3d": wifi_body_debug,
        },
        "iphone": {
            "available": valid_point_count_3d(&iphone_hand) > 0,
            "fresh": vision.fresh,
            "operator_track_id": vision.operator_track_id,
            "hand_layout": vision.hand_layout.as_str(),
            "hand_space": iphone_hand_space,
            "hand_kpts_3d": iphone_hand,
            "device_pose": vision.device_pose,
            "imu": vision.imu,
        },
        "association": {
            "selected_operator_track_id": operator.estimate.association.selected_operator_track_id,
            "anchor_source": operator.estimate.association.anchor_source,
            "iphone_visible_hand_count": operator.estimate.association.iphone_visible_hand_count,
            "hand_match_count": operator.estimate.association.hand_match_count,
            "hand_match_score": operator.estimate.association.hand_match_score,
            "left_wrist_gap_m": operator.estimate.association.left_wrist_gap_m,
            "right_wrist_gap_m": operator.estimate.association.right_wrist_gap_m,
        }
    }))
}

fn select_teacher_body<'a>(
    fused_source: OperatorSource,
    anchor_source: &str,
    fused_body: &'a [[f32; 3]],
    fused_valid: usize,
    stereo_body: &'a [[f32; 3]],
    stereo_valid: usize,
) -> Option<(&'static str, &'static str, &'a [[f32; 3]])> {
    const MIN_TEACHER_BODY_JOINTS: usize = 8;

    let fused_uses_stereo = matches!(
        fused_source,
        OperatorSource::FusedMultiSource3d
            | OperatorSource::FusedStereoVision3d
            | OperatorSource::FusedStereoVision2dProjected
    ) || anchor_source.contains("stereo");

    if fused_valid >= MIN_TEACHER_BODY_JOINTS
        && fused_uses_stereo
        && fused_teacher_body_is_clean(fused_body, stereo_body, stereo_valid)
    {
        return Some(("fused", "coco_body_17", fused_body));
    }
    if stereo_valid >= MIN_TEACHER_BODY_JOINTS {
        return Some(("stereo", "coco_body_17", stereo_body));
    }
    if fused_valid >= MIN_TEACHER_BODY_JOINTS && teacher_body_is_reasonable(fused_body) {
        return Some(("fused", "coco_body_17", fused_body));
    }
    None
}

fn fused_teacher_body_is_clean(
    fused_body: &[[f32; 3]],
    stereo_body: &[[f32; 3]],
    stereo_valid: usize,
) -> bool {
    if !teacher_body_is_reasonable(fused_body) {
        return false;
    }
    if stereo_valid < 8 {
        return true;
    }
    teacher_body_matches_stereo(fused_body, stereo_body)
}

fn teacher_body_is_reasonable(points: &[[f32; 3]]) -> bool {
    const MAX_TEACHER_BODY_ABS_COORD_M: f32 = 8.0;
    const MAX_TEACHER_BODY_SPAN_M: f32 = 4.5;

    let valid = points
        .iter()
        .copied()
        .filter(valid_point)
        .collect::<Vec<_>>();
    if valid.len() < 8 {
        return false;
    }
    let max_abs = valid
        .iter()
        .flat_map(|point| point.iter())
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if max_abs > MAX_TEACHER_BODY_ABS_COORD_M {
        return false;
    }

    let mut max_span = 0.0_f32;
    for (index, point) in valid.iter().enumerate() {
        for other in &valid[(index + 1)..] {
            max_span = max_span.max(dist3(*point, *other));
        }
    }
    max_span <= MAX_TEACHER_BODY_SPAN_M
}

fn teacher_body_matches_stereo(fused_body: &[[f32; 3]], stereo_body: &[[f32; 3]]) -> bool {
    const MAX_TEACHER_STEREO_MEAN_DELTA_M: f32 = 0.75;
    const CORE_JOINTS: [usize; 6] = [5, 6, 9, 10, 11, 12];

    let deltas = CORE_JOINTS
        .iter()
        .filter_map(|index| {
            let fused = fused_body.get(*index).copied().filter(valid_point)?;
            let stereo = stereo_body.get(*index).copied().filter(valid_point)?;
            Some(dist3(fused, stereo))
        })
        .collect::<Vec<_>>();
    if deltas.len() < 3 {
        return false;
    }
    let mean_delta = deltas.iter().sum::<f32>() / deltas.len() as f32;
    mean_delta <= MAX_TEACHER_STEREO_MEAN_DELTA_M
}

fn normalized_iphone_hand_space(calibration: Option<&IphoneStereoExtrinsic>) -> &'static str {
    match calibration {
        Some(calibration) if calibration.target_frame.trim() == OPERATOR_FRAME => STEREO_PAIR_FRAME,
        Some(calibration) if calibration.target_frame.trim() == STEREO_PAIR_FRAME => {
            STEREO_PAIR_FRAME
        }
        _ => OPERATOR_FRAME,
    }
}

fn fused_hand_space_from_source(source: OperatorSource, stereo_hand_space: &str) -> &str {
    match source {
        OperatorSource::Stereo
        | OperatorSource::FusedStereoVision3d
        | OperatorSource::FusedStereoVision2dProjected => stereo_hand_space,
        _ => OPERATOR_FRAME,
    }
}

fn fused_body_space_from_breakdown<'a>(
    body_source: OperatorPartSource,
    stereo_body_space: &'a str,
    wifi_body_space: &'a str,
) -> &'a str {
    match body_source {
        OperatorPartSource::Stereo
        | OperatorPartSource::FusedStereoVision3d
        | OperatorPartSource::FusedStereoVision2dProjected => stereo_body_space,
        OperatorPartSource::WifiPose3d => wifi_body_space,
        OperatorPartSource::FusedMultiSource3d => {
            if stereo_body_space == wifi_body_space
                || (stereo_body_space != CANONICAL_BODY_FRAME
                    && wifi_body_space == CANONICAL_BODY_FRAME)
            {
                stereo_body_space
            } else {
                "mixed_frame"
            }
        }
        OperatorPartSource::Vision3d | OperatorPartSource::Vision2dProjected => OPERATOR_FRAME,
        OperatorPartSource::None => {
            if !stereo_body_space.is_empty() {
                stereo_body_space
            } else {
                wifi_body_space
            }
        }
    }
}

fn normalized_wifi_body_space<'a>(body_space: &'a str, has_wifi_calibration: bool) -> &'a str {
    if body_space.is_empty() || (body_space == OPERATOR_FRAME && !has_wifi_calibration) {
        CANONICAL_BODY_FRAME
    } else {
        body_space
    }
}

#[derive(Clone, Debug)]
struct DevicePoseStereoCandidate {
    track_id: String,
    distance_m: f32,
    margin_m: f32,
}

fn resolve_stereo_device_anchor(
    device_pose: Option<&VisionDevicePose>,
    calibration: Option<&IphoneStereoExtrinsic>,
) -> Option<([f32; 3], &'static str)> {
    let pose = device_pose?;
    let position = pose.position_m;
    if position_is_useful(position) && pose.target_space.trim() == STEREO_PAIR_FRAME {
        return Some((position, "iphone_device_pose_stereo_frame"));
    }
    let calibration = calibration?;
    if position_is_useful(position) {
        let anchor = calibration.apply_point(position);
        if !valid_point(&anchor) {
            return None;
        }
        Some((anchor, "iphone_device_pose_calibrated"))
    } else {
        let anchor = calibration.extrinsic_translation_m;
        if !valid_point(&anchor) {
            return None;
        }
        Some((anchor, "iphone_stereo_calibration_origin"))
    }
}

fn select_device_pose_stereo_candidate(
    persons: &[StereoTrackedPersonSnapshot],
    device_anchor: Option<[f32; 3]>,
) -> Option<DevicePoseStereoCandidate> {
    const MAX_DEVICE_POSE_TRACK_DISTANCE_M: f32 = 1.85;
    const MIN_DEVICE_POSE_TRACK_MARGIN_M: f32 = 0.2;

    let anchor = device_anchor?;
    let mut scored = persons
        .iter()
        .filter_map(|person| {
            let track_id = person.operator_track_id.as_ref()?;
            let center = stereo_person_body_center(person)?;
            Some((track_id.clone(), dist3(center, anchor)))
        })
        .collect::<Vec<_>>();
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let (track_id, best_distance) = scored.first()?.clone();
    if best_distance > MAX_DEVICE_POSE_TRACK_DISTANCE_M {
        return None;
    }
    let margin_m = scored
        .get(1)
        .map(|(_, distance)| distance - best_distance)
        .unwrap_or(MAX_DEVICE_POSE_TRACK_DISTANCE_M);
    if margin_m < MIN_DEVICE_POSE_TRACK_MARGIN_M {
        return None;
    }
    Some(DevicePoseStereoCandidate {
        track_id,
        distance_m: best_distance,
        margin_m,
    })
}

fn stereo_person_body_center(person: &StereoTrackedPersonSnapshot) -> Option<[f32; 3]> {
    const TORSO_INDICES: [usize; 4] = [5, 6, 11, 12];
    let valid = TORSO_INDICES
        .iter()
        .filter_map(|index| body_joint(&person.body_kpts_3d, *index))
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return None;
    }
    let count = valid.len() as f32;
    Some([
        valid.iter().map(|point| point[0]).sum::<f32>() / count,
        valid.iter().map(|point| point[1]).sum::<f32>() / count,
        valid.iter().map(|point| point[2]).sum::<f32>() / count,
    ])
}

fn position_is_useful(point: [f32; 3]) -> bool {
    valid_point(&point) && point.iter().any(|value| value.abs() > 1e-5)
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn body_joint(points: &[[f32; 3]], index: usize) -> Option<[f32; 3]> {
    points.get(index).copied().filter(valid_point)
}

fn hand_wrist(points: &[[f32; 3]], is_left: bool) -> Option<[f32; 3]> {
    let index = if is_left { 0 } else { 21 };
    points.get(index).copied().filter(valid_point)
}

fn hand_wrist_2d(points: &[[f32; 2]], is_left: bool) -> Option<[f32; 2]> {
    let index = if is_left { 0 } else { 21 };
    points.get(index).copied().filter(valid_point_2d)
}

fn hand_valid_point_count_2d(points: &[[f32; 2]], is_left: bool) -> usize {
    let range = if is_left { 0..21 } else { 21..42 };
    range
        .filter_map(|index| points.get(index))
        .filter(|point| valid_point_2d(point))
        .count()
}

fn hand_valid_point_count_3d(points: &[[f32; 3]], is_left: bool) -> usize {
    let range = if is_left { 0..21 } else { 21..42 };
    range
        .filter_map(|index| points.get(index))
        .filter(|point| valid_point(point))
        .count()
}

fn valid_point_count_2d(points: &[[f32; 2]]) -> usize {
    points.iter().filter(|point| valid_point_2d(point)).count()
}

fn valid_point_count_3d(points: &[[f32; 3]]) -> usize {
    points.iter().filter(|point| valid_point(point)).count()
}

fn valid_point(point: &[f32; 3]) -> bool {
    point.iter().all(|value| value.is_finite()) && point.iter().any(|value| value.abs() > 1e-6)
}

fn valid_point_2d(point: &[f32; 2]) -> bool {
    point.iter().all(|value| value.is_finite()) && point.iter().any(|value| value.abs() > 1e-6)
}

#[cfg(test)]
mod tests {
    use super::{
        phone_hand_geometry_is_hint_trustworthy, resolve_stereo_device_anchor,
        select_device_pose_stereo_candidate, select_teacher_body,
    };
    use crate::calibration::IphoneStereoExtrinsic;
    use crate::operator::OperatorSource;
    use crate::sensing::{StereoTrackedPersonSnapshot, VisionDevicePose, VisionSnapshot};

    fn stereo_body() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.4, -0.2, 2.0],
            [0.4, -0.2, 2.0],
            [-0.15, -0.6, 2.0],
            [0.15, -0.6, 2.0],
            [-0.2, -1.0, 2.0],
            [0.2, -1.0, 2.0],
            [-0.2, -1.3, 2.0],
            [0.2, -1.3, 2.0],
        ]
    }

    #[test]
    fn teacher_falls_back_to_stereo_when_fused_body_is_unreasonable() {
        let stereo = stereo_body();
        let mut fused = stereo.clone();
        fused[5] = [42000.0, -55000.0, 12000.0];
        fused[6] = [-38000.0, 61000.0, -9000.0];

        let teacher = select_teacher_body(
            OperatorSource::FusedMultiSource3d,
            "stereo+wifi_prior",
            &fused,
            17,
            &stereo,
            17,
        );

        assert_eq!(teacher.map(|(source, _, _)| source), Some("stereo"));
    }

    #[test]
    fn teacher_keeps_fused_when_it_stays_close_to_stereo() {
        let stereo = stereo_body();
        let mut fused = stereo.clone();
        fused[9] = [-0.36, -0.22, 2.03];
        fused[10] = [0.38, -0.18, 1.98];

        let teacher = select_teacher_body(
            OperatorSource::FusedMultiSource3d,
            "stereo+iphone_hand+wifi_prior",
            &fused,
            17,
            &stereo,
            17,
        );

        assert_eq!(teacher.map(|(source, _, _)| source), Some("fused"));
    }

    #[test]
    fn device_pose_anchor_uses_calibration_origin_when_pose_has_no_translation() {
        let pose = VisionDevicePose {
            position_m: [0.0, 0.0, 0.0],
            target_space: "device_motion_reference_frame".to_string(),
            ..VisionDevicePose::default()
        };
        let calibration = IphoneStereoExtrinsic {
            source_frame: "device_motion_reference_frame".to_string(),
            target_frame: "stereo_pair_frame".to_string(),
            extrinsic_version: "test".to_string(),
            extrinsic_scale: 1.0,
            extrinsic_translation_m: [0.3, -0.1, 1.2],
            extrinsic_rotation_quat: [0.0, 0.0, 0.0, 1.0],
            sample_count: 1,
            rms_error_m: 0.0,
            solved_edge_time_ns: 1,
        };

        let anchor =
            resolve_stereo_device_anchor(Some(&pose), Some(&calibration)).expect("device anchor");

        assert_eq!(anchor.0, [0.3, -0.1, 1.2]);
        assert_eq!(anchor.1, "iphone_stereo_calibration_origin");
    }

    #[test]
    fn device_pose_candidate_selects_nearest_stereo_track_with_margin() {
        let persons = vec![
            StereoTrackedPersonSnapshot {
                operator_track_id: Some("stereo-person-1".to_string()),
                body_kpts_3d: vec![
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.30, 1.0, 1.00],
                    [0.50, 1.0, 1.00],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.25, 0.6, 1.0],
                    [0.55, 0.6, 1.0],
                    [0.32, 0.3, 1.00],
                    [0.48, 0.3, 1.00],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                ..StereoTrackedPersonSnapshot::default()
            },
            StereoTrackedPersonSnapshot {
                operator_track_id: Some("stereo-person-2".to_string()),
                body_kpts_3d: vec![
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.20, 1.0, 1.30],
                    [1.40, 1.0, 1.30],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.15, 0.6, 1.3],
                    [1.45, 0.6, 1.3],
                    [1.22, 0.3, 1.30],
                    [1.38, 0.3, 1.30],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                ..StereoTrackedPersonSnapshot::default()
            },
        ];

        let candidate = select_device_pose_stereo_candidate(&persons, Some([0.42, 0.65, 1.05]))
            .expect("candidate");

        assert_eq!(candidate.track_id, "stereo-person-1");
        assert!(candidate.distance_m < 0.5);
        assert!(candidate.margin_m > 0.2);
    }

    #[test]
    fn depth_reprojected_phone_hands_are_not_exposed_as_geometry_hints() {
        let vision = VisionSnapshot {
            fresh: true,
            hand_3d_source: "depth_reprojected".to_string(),
            ..VisionSnapshot::default()
        };

        assert!(!phone_hand_geometry_is_hint_trustworthy(&vision));
    }
}
