#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from validate_training_thresholds import (
    DEFAULT_THRESHOLD_CONFIG,
    jsonl_rows,
    load_json,
    load_manifest,
    percentile,
    resolve_bundle_root,
    summarize_time_sync,
)


COCO_BODY_17 = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a factual SLAM + time-sync benchmark report for a CHEK EGO bundle."
    )
    parser.add_argument("--bundle", required=True, help="Extracted bundle directory or .tar.gz raw bundle.")
    parser.add_argument("--tier", choices=["basic", "stereo", "pro"], default="pro")
    parser.add_argument("--threshold-config", default=str(DEFAULT_THRESHOLD_CONFIG))
    parser.add_argument("--output", required=True, help="Path to write qa/slam_time_sync_benchmark.json.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-valid-body-joints", type=int, default=8)
    return parser.parse_args()


def finite_triplet(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        point = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in point):
        return None
    if all(abs(item) < 1e-9 for item in point):
        return None
    return point


def normalized_to_px(point: Any, width: int, height: int) -> tuple[float, float] | None:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if x < 0.0 or y < 0.0:
        return None
    if x <= 1.5 and y <= 1.5:
        return (x * float(width), y * float(height))
    return (x, y)


def count_valid_points(points: Any) -> int:
    if not isinstance(points, list):
        return 0
    return sum(1 for point in points if finite_triplet(point) is not None)


def stereo_reprojection_errors(row: dict[str, Any]) -> list[float]:
    calibration = row.get("calibration") if isinstance(row.get("calibration"), dict) else {}
    left_intrinsics = calibration.get("left_intrinsics") if isinstance(calibration.get("left_intrinsics"), dict) else {}
    right_intrinsics = calibration.get("right_intrinsics") if isinstance(calibration.get("right_intrinsics"), dict) else {}
    try:
        baseline_m = float(calibration.get("baseline_m"))
        left_fx = float(left_intrinsics.get("fx_px"))
        left_fy = float(left_intrinsics.get("fy_px"))
        left_cx = float(left_intrinsics.get("cx_px"))
        left_cy = float(left_intrinsics.get("cy_px"))
        right_fx = float(right_intrinsics.get("fx_px"))
        right_fy = float(right_intrinsics.get("fy_px"))
        right_cx = float(right_intrinsics.get("cx_px"))
        right_cy = float(right_intrinsics.get("cy_px"))
        left_w = int(left_intrinsics.get("reference_image_w"))
        left_h = int(left_intrinsics.get("reference_image_h"))
        right_w = int(right_intrinsics.get("reference_image_w"))
        right_h = int(right_intrinsics.get("reference_image_h"))
    except (TypeError, ValueError):
        return []
    if not all(math.isfinite(value) for value in [baseline_m, left_fx, left_fy, left_cx, left_cy, right_fx, right_fy, right_cx, right_cy]):
        return []

    body_3d = row.get("body_kpts_3d") if isinstance(row.get("body_kpts_3d"), list) else []
    left_2d = row.get("left_body_kpts_2d") if isinstance(row.get("left_body_kpts_2d"), list) else []
    right_2d = row.get("right_body_kpts_2d") if isinstance(row.get("right_body_kpts_2d"), list) else []
    errors: list[float] = []
    for index in range(min(COCO_BODY_17, len(body_3d), len(left_2d), len(right_2d))):
        point_3d = finite_triplet(body_3d[index])
        if point_3d is None:
            continue
        x, y, z = point_3d
        if z <= 0:
            continue
        left_observed = normalized_to_px(left_2d[index], left_w, left_h)
        right_observed = normalized_to_px(right_2d[index], right_w, right_h)
        left_projected = (left_fx * x / z + left_cx, left_cy - left_fy * y / z)
        right_projected = (right_fx * (x - baseline_m) / z + right_cx, right_cy - right_fy * y / z)
        if left_observed is not None:
            errors.append(math.dist(left_observed, left_projected))
        if right_observed is not None:
            errors.append(math.dist(right_observed, right_projected))
    return [value for value in errors if math.isfinite(value)]


def summarize_stereo(rows: list[dict[str, Any]], min_valid_body_joints: int) -> dict[str, Any]:
    valid_body_rows = 0
    confidence_values: list[float] = []
    continuity_gaps: list[float] = []
    reprojection_errors: list[float] = []
    source_times: list[int] = []
    for row in rows:
        body_count = count_valid_points(row.get("body_kpts_3d"))
        if body_count >= min_valid_body_joints:
            valid_body_rows += 1
        confidence = row.get("stereo_confidence")
        if isinstance(confidence, (int, float)):
            confidence_values.append(float(confidence))
        selection = row.get("selection") if isinstance(row.get("selection"), dict) else {}
        continuity = selection.get("continuity_gap_m")
        if isinstance(continuity, (int, float)) and math.isfinite(float(continuity)):
            continuity_gaps.append(float(continuity))
        source_time_ns = row.get("source_time_ns")
        if isinstance(source_time_ns, int):
            source_times.append(source_time_ns)
        reprojection_errors.extend(stereo_reprojection_errors(row))
    row_count = len(rows)
    duration_s = ((max(source_times) - min(source_times)) / 1_000_000_000.0) if len(source_times) >= 2 else None
    return {
        "row_count": row_count,
        "valid_body_rows": valid_body_rows,
        "body_tracking_coverage_percent": (valid_body_rows / row_count * 100.0) if row_count else None,
        "stereo_confidence_p50": percentile(confidence_values, 0.50),
        "stereo_confidence_p95": percentile(confidence_values, 0.95),
        "continuity_gap_m_p95": percentile(continuity_gaps, 0.95),
        "reprojection_error_px_p95": percentile(reprojection_errors, 0.95),
        "reprojection_error_sample_count": len(reprojection_errors),
        "duration_s": duration_s,
    }


def build_report(root: Path, threshold_config: dict[str, Any], tier: str, source_bundle: Path, min_valid_body_joints: int) -> dict[str, Any]:
    manifest = load_manifest(root)
    stereo_rows = jsonl_rows(root / "raw" / "stereo" / "pose3d.jsonl")
    phone_rows = jsonl_rows(root / "raw" / "iphone" / "wide" / "kpts_depth.jsonl")
    if not phone_rows:
        phone_rows = jsonl_rows(root / "raw" / "iphone" / "capture_pose.jsonl")
    fused_rows = jsonl_rows(root / "fused" / "human_demo_pose.jsonl")
    stereo = summarize_stereo(stereo_rows, min_valid_body_joints)
    time_sync = summarize_time_sync(root, threshold_config.get("time_sync") or {}, tier)
    blockers: list[str] = []
    metrics = {
        "trajectory_drift_percent": None,
        "reprojection_error_px_p95": stereo["reprojection_error_px_p95"],
        "pose_graph_residual_p95": None,
        "body_tracking_coverage_percent": stereo["body_tracking_coverage_percent"],
    }
    if metrics["trajectory_drift_percent"] is None:
        blockers.append("trajectory_drift_requires_ground_truth_or_loop_closure")
    if metrics["pose_graph_residual_p95"] is None:
        blockers.append("pose_graph_residual_requires_slam_pose_graph")
    if metrics["reprojection_error_px_p95"] is None:
        blockers.append("reprojection_error_requires_stereo_2d_3d_calibration")
    if metrics["body_tracking_coverage_percent"] is None:
        blockers.append("body_tracking_requires_stereo_pose_rows")

    status = "measured_with_blockers" if blockers else "measured"
    return {
        "schema_version": "1.0.0",
        "type": "slam_time_sync_benchmark",
        "status": status,
        "source_bundle": str(source_bundle),
        "session_id": manifest.get("session_id"),
        "task_id": manifest.get("task_id"),
        "capture_device_id": manifest.get("capture_device_id"),
        "tier": tier,
        "metrics": metrics,
        "blockers": blockers,
        "time_sync": {key: value for key, value in time_sync.items() if key != "checks"},
        "stereo": stereo,
        "coverage": {
            "phone_pose_rows": len(phone_rows),
            "stereo_pose_rows": len(stereo_rows),
            "fused_pose_rows": len(fused_rows),
            "min_valid_body_joints": min_valid_body_joints,
        },
        "methodology": {
            "body_tracking_coverage_percent": "stereo pose rows with at least min_valid_body_joints finite 3D body joints divided by stereo pose rows",
            "reprojection_error_px_p95": "rectified stereo pinhole reprojection using packet 3D points, normalized 2D keypoints, intrinsics, and baseline",
            "trajectory_drift_percent": "not emitted without ground truth, loop closure, or an equivalent reference trajectory",
            "pose_graph_residual_p95": "not emitted without a SLAM pose graph optimizer report",
        },
    }


def main() -> int:
    args = parse_args()
    source_bundle = Path(args.bundle).expanduser()
    threshold_config = load_json(Path(args.threshold_config).expanduser())
    with tempfile.TemporaryDirectory(prefix="chek-ego-benchmark-") as temp_dir:
        root = resolve_bundle_root(source_bundle, Path(temp_dir))
        report = build_report(
            root,
            threshold_config,
            args.tier,
            source_bundle,
            max(1, int(args.min_valid_body_joints)),
        )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print(f"wrote {output}")
        print(f"status={report['status']}")
        if report["blockers"]:
            print("blockers=" + ",".join(report["blockers"]))
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
