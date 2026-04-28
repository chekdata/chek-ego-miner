#!/usr/bin/env python3
"""Recompute edge session QA/manifest metadata from on-disk facts.

Use this as a fallback when a remote edge binary has stale in-memory counters
and the session artifacts on disk are already correct.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def summarize_csi_index(path: Path) -> dict[str, int]:
    rows_with_nodes = 0
    max_node_count = 0
    if not path.exists():
        return {"rows_with_nodes": 0, "max_node_count": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        node_count = int(value.get("node_count") or 0)
        if node_count > 0:
            rows_with_nodes += 1
        max_node_count = max(max_node_count, node_count)
    return {"rows_with_nodes": rows_with_nodes, "max_node_count": max_node_count}


def summarize_media_index_frames(path: Path) -> int:
    return count_jsonl_lines(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_pretty(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_line_counters(session_root: Path) -> dict[str, int]:
    raw_iphone_wide = session_root / "raw" / "iphone" / "wide"
    raw_csi = session_root / "raw" / "csi"
    raw_stereo = session_root / "raw" / "stereo"
    raw_wifi = session_root / "raw" / "wifi"
    raw_robot = session_root / "raw" / "robot"
    fused = session_root / "fused"
    teleop = session_root / "teleop"
    chunk_dir = session_root / "chunks"
    sync = session_root / "sync"
    labels = session_root / "labels"
    return {
        "labels": count_jsonl_lines(labels / "labels.jsonl"),
        "raw_iphone": count_jsonl_lines(raw_iphone_wide / "kpts_depth.jsonl"),
        "raw_iphone_pose_imu": count_jsonl_lines(raw_iphone_wide / "pose_imu.jsonl"),
        "raw_iphone_depth": count_jsonl_lines(raw_iphone_wide / "depth" / "index.jsonl"),
        "raw_iphone_media": count_jsonl_lines(raw_iphone_wide / "media_index.jsonl"),
        "raw_csi": count_jsonl_lines(raw_csi / "index.jsonl"),
        "raw_stereo": count_jsonl_lines(raw_stereo / "pose3d.jsonl"),
        "raw_wifi": count_jsonl_lines(raw_wifi / "pose3d.jsonl"),
        "raw_stereo_media": count_jsonl_lines(raw_stereo / "media_index.jsonl"),
        "fused_state": count_jsonl_lines(fused / "fusion_state.jsonl"),
        "human_demo_pose": count_jsonl_lines(fused / "human_demo_pose.jsonl"),
        "teleop": count_jsonl_lines(teleop / "teleop_frame.jsonl"),
        "raw_robot_state": count_jsonl_lines(raw_robot / "state.jsonl"),
        "chunks": count_jsonl_lines(chunk_dir / "chunk_state.jsonl"),
        "sync": count_jsonl_lines(sync / "time_sync_samples.jsonl"),
        "frame_correspondence": count_jsonl_lines(sync / "frame_correspondence.jsonl"),
    }


def build_checks(session_root: Path, line_counters: dict[str, int]) -> list[dict[str, Any]]:
    calibration_dir = session_root / "calibration"
    csi_packets_bytes = (session_root / "raw" / "csi" / "packets.bin").stat().st_size if (session_root / "raw" / "csi" / "packets.bin").exists() else 0
    csi_summary = summarize_csi_index(session_root / "raw" / "csi" / "index.jsonl")
    fisheye_frames = summarize_media_index_frames(
        session_root / "raw" / "iphone" / "fisheye" / "media_index.jsonl"
    )
    media_tracks_present = any(
        True for _ in (session_root / "raw").rglob("media_index.jsonl")
    )
    has_iphone_calibration = (calibration_dir / "iphone_capture.json").exists()
    wifi_or_csi_present = line_counters["raw_wifi"] > 0 or (
        line_counters["raw_csi"] > 0
        and csi_packets_bytes > 0
        and csi_summary["max_node_count"] > 0
    )
    fisheye_track_present = fisheye_frames >= 10
    def score(ok: bool) -> float:
        return 1.0 if ok else 0.0
    return [
        {
            "id": "capture_pose_present",
            "ok": line_counters["raw_iphone"] > 0,
            "score": score(line_counters["raw_iphone"] > 0),
            "detail": f"raw/iphone/wide/kpts_depth.jsonl 行数={line_counters['raw_iphone']}",
        },
        {
            "id": "pose_imu_present",
            "ok": line_counters["raw_iphone_pose_imu"] > 0,
            "score": score(line_counters["raw_iphone_pose_imu"] > 0),
            "detail": f"raw/iphone/wide/pose_imu.jsonl 行数={line_counters['raw_iphone_pose_imu']}",
        },
        {
            "id": "raw_depth_present",
            "ok": line_counters["raw_iphone_depth"] > 0,
            "score": score(line_counters["raw_iphone_depth"] > 0),
            "detail": f"raw/iphone/wide/depth/index.jsonl 行数={line_counters['raw_iphone_depth']}",
        },
        {
            "id": "iphone_calibration_present",
            "ok": has_iphone_calibration,
            "score": score(has_iphone_calibration),
            "detail": f"calibration/iphone_capture.json {'已生成' if has_iphone_calibration else '缺失'}",
        },
        {
            "id": "time_sync_present",
            "ok": line_counters["sync"] > 0,
            "score": score(line_counters["sync"] > 0),
            "detail": f"sync/time_sync_samples.jsonl 行数={line_counters['sync']}",
        },
        {
            "id": "human_demo_pose_present",
            "ok": line_counters["human_demo_pose"] > 0,
            "score": score(line_counters["human_demo_pose"] > 0),
            "detail": f"fused/human_demo_pose.jsonl 行数={line_counters['human_demo_pose']}",
        },
        {
            "id": "teleop_frame_present",
            "ok": line_counters["teleop"] > 0,
            "score": score(line_counters["teleop"] > 0),
            "detail": f"teleop/teleop_frame.jsonl 行数={line_counters['teleop']}",
        },
        {
            "id": "robot_state_present",
            "ok": line_counters["raw_robot_state"] > 0,
            "score": score(line_counters["raw_robot_state"] > 0),
            "detail": f"raw/robot/state.jsonl 行数={line_counters['raw_robot_state']}",
        },
        {
            "id": "stereo_pose_present",
            "ok": line_counters["raw_stereo"] > 0,
            "score": score(line_counters["raw_stereo"] > 0),
            "detail": f"raw/stereo/pose3d.jsonl 行数={line_counters['raw_stereo']}",
        },
        {
            "id": "wifi_or_csi_present",
            "ok": wifi_or_csi_present,
            "score": score(wifi_or_csi_present),
            "detail": (
                f"raw/wifi/pose3d 行数={line_counters['raw_wifi']}，"
                f"raw/csi/index 行数={line_counters['raw_csi']}，"
                f"raw/csi/packets.bin bytes={csi_packets_bytes}，"
                f"CSI 有效节点行数={csi_summary['rows_with_nodes']}，"
                f"最大 node_count={csi_summary['max_node_count']}"
            ),
        },
        {
            "id": "media_tracks_present",
            "ok": media_tracks_present,
            "score": score(media_tracks_present),
            "detail": "存在至少一条 media_index.jsonl" if media_tracks_present else "缺少 media_index.jsonl",
        },
        {
            "id": "fisheye_track_present",
            "ok": fisheye_track_present,
            "score": score(fisheye_track_present),
            "detail": f"raw/iphone/fisheye/media_index.jsonl 行数={fisheye_frames}",
        },
    ]


def recommended_action(missing: str) -> str:
    mapping = {
        "capture_pose_present": "补录手机主链 capture pose 与深度事实。",
        "iphone_calibration_present": "确保手机主链 calibration snapshot 已落盘。",
        "time_sync_present": "补采 time/sync 样本，避免多模态时间轴不可对齐。",
        "human_demo_pose_present": "检查 fused/human_demo_pose 是否稳定输出。",
        "teleop_frame_present": "检查 teleop frame 输出，确保机器人 target 已落盘。",
        "stereo_pose_present": "建议补齐双目 pose，提高 whole-body 几何质量。",
        "wifi_or_csi_present": "建议补齐 Wi-Fi pose/CSI，保留多源观测。",
        "fisheye_track_present": "建议开启超广角连续辅路；若机型只支持快照辅路，至少提高抽帧频率并确认正式 fisheye 轨已落盘。",
        "media_tracks_present": "建议补录至少一条可回放媒体轨。",
    }
    return mapping.get(missing, f"检查 `{missing}` 对应的采集链路。")


def local_quality_core_ids(manifest: dict[str, Any], upload_manifest: dict[str, Any]) -> set[str]:
    session_context = manifest.get("session_context") or upload_manifest.get("session_context") or {}
    runtime_profile = str(session_context.get("runtime_profile") or "teleop_fullstack").strip()
    if runtime_profile not in {
        "raw_capture_only",
        "capture_plus_facts",
        "capture_plus_vlm",
        "teleop_fullstack",
    }:
        runtime_profile = "teleop_fullstack"
    runtime_flags = session_context.get("runtime_flags") or {}
    phone_ingest_enabled = bool(runtime_flags.get("phone_ingest_enabled", True))
    fusion_enabled = bool(runtime_flags.get("fusion_enabled", True))
    stereo_enabled = bool(runtime_flags.get("stereo_enabled", runtime_profile == "teleop_fullstack"))
    wifi_enabled = bool(runtime_flags.get("wifi_enabled", runtime_profile == "teleop_fullstack"))
    control_enabled = bool(runtime_flags.get("control_enabled", runtime_profile == "teleop_fullstack"))

    core_ids: set[str] = set()
    if phone_ingest_enabled:
        core_ids.update({"capture_pose_present", "iphone_calibration_present"})
    if fusion_enabled or stereo_enabled or wifi_enabled or control_enabled:
        core_ids.add("time_sync_present")
    if runtime_profile == "teleop_fullstack" and control_enabled:
        core_ids.update({"human_demo_pose_present", "teleop_frame_present"})
    return core_ids


def recompute_session(session_root: Path, write: bool) -> dict[str, Any]:
    qa_path = session_root / "qa" / "local_quality_report.json"
    upload_manifest_path = session_root / "upload" / "upload_manifest.json"
    manifest_path = session_root / "manifest.json"

    qa = load_json(qa_path)
    upload_manifest = load_json(upload_manifest_path)
    manifest = load_json(manifest_path)

    line_counters = build_line_counters(session_root)
    checks = build_checks(session_root, line_counters)
    missing_artifacts = [check["id"] for check in checks if not check["ok"]]
    total = max(len(checks), 1)
    passed = sum(1 for check in checks if check["ok"])
    score_percent = round((passed / total) * 100.0, 2)
    core_ids = local_quality_core_ids(manifest, upload_manifest)
    core_missing = any(item in core_ids for item in missing_artifacts)
    optional_missing = any(item not in core_ids for item in missing_artifacts)
    status = "reject_local" if core_missing else "retry_recommended" if optional_missing else "pass"
    ready_for_upload = status != "reject_local"

    qa.update(
        {
            "generated_unix_ms": now_unix_ms(),
            "status": status,
            "ready_for_upload": ready_for_upload,
            "score_percent": score_percent,
            "checks": checks,
            "missing_artifacts": missing_artifacts,
            "recommended_actions": [recommended_action(item) for item in missing_artifacts],
        }
    )

    upload_manifest["generated_unix_ms"] = now_unix_ms()
    upload_manifest["ready_for_upload"] = ready_for_upload
    artifacts = upload_manifest.get("artifacts") or []
    upload_manifest["artifact_count"] = len(artifacts)
    upload_manifest["ready_artifact_count"] = sum(1 for item in artifacts if item.get("exists"))

    recorder_state = manifest.setdefault("recorder_state", {})
    recorder_state["has_iphone_calibration"] = (session_root / "calibration" / "iphone_capture.json").exists()
    recorder_state["has_stereo_calibration"] = (session_root / "calibration" / "stereo_pair.json").exists()
    recorder_state["has_wifi_calibration"] = (session_root / "calibration" / "wifi_pose.json").exists()
    recorder_state["line_counters"] = line_counters
    manifest["generated_unix_ms"] = now_unix_ms()

    if write:
        write_json_pretty(qa_path, qa)
        write_json_pretty(upload_manifest_path, upload_manifest)
        write_json_pretty(manifest_path, manifest)

    return {
        "session_root": str(session_root),
        "status": status,
        "score_percent": score_percent,
        "ready_for_upload": ready_for_upload,
        "missing_artifacts": missing_artifacts,
        "line_counters": line_counters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = recompute_session(args.session_root, write=args.write)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
