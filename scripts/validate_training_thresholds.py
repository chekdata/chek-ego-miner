#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD_CONFIG = REPO_ROOT / "configs" / "slam_time_sync_training_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a CHEK EGO capture bundle against the public SLAM + time-sync "
            "training-candidate contract."
        )
    )
    parser.add_argument("--bundle", required=True, help="Extracted bundle directory or .tar.gz raw bundle.")
    parser.add_argument("--tier", choices=["basic", "stereo", "pro"], default="pro")
    parser.add_argument("--threshold-config", default=str(DEFAULT_THRESHOLD_CONFIG))
    parser.add_argument(
        "--slam-benchmark-report",
        help="Optional qa/slam_time_sync_benchmark.json generated outside the bundle.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-path")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def safe_extract_tar(tar_path: Path, target_dir: Path) -> Path:
    target_root = target_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            member_path = (target_dir / member.name).resolve()
            try:
                member_path.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(f"unsafe tar member path: {member.name}")
            if member.isdev():
                raise ValueError(f"device tar members are not supported: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"link tar members are not supported: {member.name}")
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported tar member type: {member.name}")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unable to read tar member: {member.name}")
            with source, member_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            mode = member.mode & 0o777
            if mode:
                member_path.chmod(mode)
    return target_dir


def resolve_bundle_root(raw_bundle: Path, temp_root: Path | None = None) -> Path:
    if raw_bundle.is_dir():
        return raw_bundle.resolve()
    if raw_bundle.is_file() and raw_bundle.name.endswith((".tar.gz", ".tgz", ".tar")):
        if temp_root is None:
            raise ValueError("temp_root is required for archive bundles")
        return safe_extract_tar(raw_bundle.resolve(), temp_root)
    raise FileNotFoundError(f"bundle not found or unsupported: {raw_bundle}")


def first_existing(root: Path, candidates: list[str]) -> Path | None:
    for relpath in candidates:
        path = root / relpath
        if path.exists():
            return path
    return None


def line_count(root: Path, candidates: list[str]) -> int:
    path = first_existing(root, candidates)
    return len(jsonl_rows(path)) if path is not None else 0


def load_manifest(root: Path) -> dict[str, Any]:
    for relpath in ("bundle_manifest.json", "demo_capture_bundle.json", "manifest.json"):
        path = root / relpath
        if path.is_file():
            payload = load_json(path)
            payload["_manifest_relpath"] = relpath
            return payload
    return {"_manifest_relpath": ""}


def build_check(check_id: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": check_id,
        "ok": bool(ok),
        "detail": detail,
    }
    payload.update(extra)
    return payload


def summarize_time_sync(root: Path, threshold: dict[str, Any], tier: str) -> dict[str, Any]:
    rows = jsonl_rows(root / "sync" / "time_sync_samples.jsonl")
    accepted = [row for row in rows if row.get("accepted") is True]
    accepted_for_mapping = [row for row in rows if row.get("accepted_for_mapping") is True]
    ratio = (len(accepted_for_mapping) / len(rows)) if rows else 0.0
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_kind = str(row.get("source_kind") or "unknown")
        bucket = by_source.setdefault(source_kind, {"rows": [], "rtt_ms": [], "offset_ms": []})
        bucket["rows"].append(row)
        rtt_ns = row.get("rtt_ns")
        if isinstance(rtt_ns, (int, float)):
            bucket["rtt_ms"].append(float(rtt_ns) / 1_000_000.0)
        offset_ns = row.get("clock_offset_ns")
        if isinstance(offset_ns, (int, float)):
            bucket["offset_ms"].append(float(offset_ns) / 1_000_000.0)

    per_source_rtt = threshold.get("per_source_rtt_ok_ms") if isinstance(threshold.get("per_source_rtt_ok_ms"), dict) else {}
    default_rtt = float(threshold.get("max_rtt_ms") or 0.0)
    max_offset_span_ms = float(threshold.get("max_clock_offset_span_ms") or 0.0)
    source_summaries: dict[str, dict[str, Any]] = {}
    rtt_violations: list[str] = []
    offset_span_violations: list[str] = []
    for source_kind, bucket in sorted(by_source.items()):
        rtt_values = bucket["rtt_ms"]
        offset_values = bucket["offset_ms"]
        rtt_p95 = percentile(rtt_values, 0.95)
        offset_span = (max(offset_values) - min(offset_values)) if offset_values else None
        override_rtt = per_source_rtt.get(source_kind)
        rtt_budget = float(default_rtt if override_rtt is None else override_rtt)
        rtt_ok = rtt_p95 is not None and (rtt_budget <= 0 or rtt_p95 <= rtt_budget)
        offset_ok = offset_span is not None and (max_offset_span_ms <= 0 or offset_span <= max_offset_span_ms)
        if not rtt_ok:
            rtt_violations.append(f"{source_kind}:p95={rtt_p95} budget={rtt_budget}")
        if not offset_ok:
            offset_span_violations.append(f"{source_kind}:span={offset_span} budget={max_offset_span_ms}")
        source_summaries[source_kind] = {
            "sample_count": len(bucket["rows"]),
            "rtt_ms_p50": percentile(rtt_values, 0.50),
            "rtt_ms_p95": rtt_p95,
            "rtt_budget_ms": rtt_budget,
            "offset_span_ms": offset_span,
            "offset_span_budget_ms": max_offset_span_ms,
            "rtt_ok": rtt_ok,
            "offset_span_ok": offset_ok,
        }

    min_source_kinds = threshold.get("minimum_source_kinds") if isinstance(threshold.get("minimum_source_kinds"), dict) else {}
    min_sources = int(min_source_kinds.get(tier, 1) or 1)
    min_ratio = float(threshold.get("min_accepted_for_mapping_ratio") or 0.0)
    checks = [
        build_check(
            "time_sync_samples_present",
            len(rows) > 0,
            f"sync/time_sync_samples.jsonl rows={len(rows)}",
        ),
        build_check(
            "time_sync_mapping_acceptance_ratio",
            ratio >= min_ratio,
            f"accepted_for_mapping={len(accepted_for_mapping)}/{len(rows)} ratio={ratio:.3f} threshold={min_ratio:.3f}",
        ),
        build_check(
            "time_sync_source_kind_count",
            len(by_source) >= min_sources,
            f"source_kind_count={len(by_source)} threshold={min_sources}",
        ),
        build_check(
            "time_sync_rtt_p95_within_budget",
            not rtt_violations and bool(by_source),
            "; ".join(rtt_violations) or "all source p95 RTT values within budget",
        ),
        build_check(
            "time_sync_offset_span_within_budget",
            not offset_span_violations and bool(by_source),
            "; ".join(offset_span_violations) or "all source offset spans within budget",
        ),
    ]
    return {
        "sample_count": len(rows),
        "accepted_count": len(accepted),
        "accepted_for_mapping_count": len(accepted_for_mapping),
        "accepted_for_mapping_ratio": ratio,
        "by_source_kind": source_summaries,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def summarize_vlm(root: Path, threshold: dict[str, Any], tier: str) -> dict[str, Any]:
    events = jsonl_rows(root / "derived" / "vision" / "vlm_events.jsonl")
    segments = jsonl_rows(root / "derived" / "vision" / "vlm_segments.jsonl")
    latencies = [
        float(row["latency_ms"])
        for row in [*events, *segments]
        if isinstance(row.get("latency_ms"), (int, float))
    ]
    fallback_rows = [
        row
        for row in [*events, *segments]
        if "fallback" in str(row.get("inference_source") or "").lower()
        or "fallback" in str(row.get("model_id") or "").lower()
    ]
    required = tier in set(threshold.get("required_for_tiers") or [])
    min_events = int(threshold.get("min_event_count") or 0)
    min_segments = int(threshold.get("min_segment_count") or 0)
    max_p95_latency = float(threshold.get("max_p95_latency_ms") or 0.0)
    latency_p95 = percentile(latencies, 0.95)
    checks = [
        build_check(
            "vlm_events_present",
            (not required) or len(events) >= min_events,
            f"events={len(events)} threshold={min_events}",
        ),
        build_check(
            "vlm_segments_present",
            (not required) or len(segments) >= min_segments,
            f"segments={len(segments)} threshold={min_segments}",
        ),
        build_check(
            "vlm_latency_p95_within_budget",
            (not required)
            or (
                latency_p95 is not None
                and (max_p95_latency <= 0 or latency_p95 <= max_p95_latency)
            ),
            f"latency_p95_ms={latency_p95} threshold={max_p95_latency}",
        ),
        build_check(
            "vlm_no_fallback_inference",
            (not required) or not threshold.get("disallow_fallback_inference") or not fallback_rows,
            f"fallback_rows={len(fallback_rows)}",
        ),
    ]
    return {
        "event_count": len(events),
        "segment_count": len(segments),
        "latency_ms_p95": latency_p95,
        "first_event": events[0] if events else None,
        "first_segment": segments[0] if segments else None,
        "fallback_row_count": len(fallback_rows),
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def summarize_spatial(root: Path, threshold: dict[str, Any], tier: str) -> dict[str, Any]:
    min_pose_rows = int(threshold.get("min_pose_rows") or 1)
    min_media_rows = int(threshold.get("min_media_track_rows") or 1)
    phone_pose_rows = line_count(root, ["raw/iphone/wide/kpts_depth.jsonl", "raw/iphone/capture_pose.jsonl"])
    stereo_pose_rows = line_count(root, ["raw/stereo/pose3d.jsonl"])
    wifi_pose_rows = line_count(root, ["raw/wifi/pose3d.jsonl"])
    fisheye_media_rows = line_count(root, ["raw/iphone/fisheye/media_index.jsonl", "raw/iphone/aux/media_index.jsonl"])
    require_stereo = tier in set(threshold.get("require_stereo_pose_tiers") or [])
    require_wifi = tier in set(threshold.get("require_wifi_pose_tiers") or [])
    require_fisheye = tier in set(threshold.get("require_fisheye_track_tiers") or [])
    require_phone = bool(threshold.get("require_phone_pose"))
    checks = [
        build_check(
            "phone_pose_present",
            (not require_phone) or phone_pose_rows >= min_pose_rows,
            f"phone_pose_rows={phone_pose_rows} threshold={min_pose_rows}",
        ),
        build_check(
            "stereo_pose_present",
            (not require_stereo) or stereo_pose_rows >= min_pose_rows,
            f"stereo_pose_rows={stereo_pose_rows} threshold={min_pose_rows}",
        ),
        build_check(
            "wifi_pose_present",
            (not require_wifi) or wifi_pose_rows >= min_pose_rows,
            f"wifi_pose_rows={wifi_pose_rows} threshold={min_pose_rows}",
        ),
        build_check(
            "fisheye_track_present",
            (not require_fisheye) or fisheye_media_rows >= min_media_rows,
            f"fisheye_media_rows={fisheye_media_rows} threshold={min_media_rows}",
        ),
    ]
    return {
        "phone_pose_rows": phone_pose_rows,
        "stereo_pose_rows": stereo_pose_rows,
        "wifi_pose_rows": wifi_pose_rows,
        "fisheye_media_rows": fisheye_media_rows,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def metric_budget_check(metric: str, value: Any, budgets: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False, f"{metric}={value!r}"
    budget = budgets.get(metric)
    if not isinstance(budget, dict):
        return True, f"{metric}={float(value)!r}"
    min_value = budget.get("min")
    max_value = budget.get("max")
    checks: list[str] = [f"{metric}={float(value)!r}"]
    if isinstance(min_value, (int, float)):
        checks.append(f"min={float(min_value)!r}")
        if float(value) < float(min_value):
            return False, " ".join(checks)
    if isinstance(max_value, (int, float)):
        checks.append(f"max={float(max_value)!r}")
        if float(value) > float(max_value):
            return False, " ".join(checks)
    return True, " ".join(checks)


def summarize_slam_benchmark(
    root: Path,
    threshold: dict[str, Any],
    report_override: Path | None = None,
) -> dict[str, Any]:
    candidates = [
        "qa/slam_time_sync_benchmark.json",
        "qa/slam_benchmark_report.json",
        "derived/slam/benchmark.json",
    ]
    report_path = report_override if report_override is not None else first_existing(root, candidates)
    required = bool(threshold.get("require_slam_benchmark_report"))
    required_metrics = list(threshold.get("required_slam_metrics") or [])
    metric_budgets = threshold.get("slam_metric_budgets") if isinstance(threshold.get("slam_metric_budgets"), dict) else {}
    if report_path is None:
        checks = [
            build_check(
                "slam_benchmark_report_present",
                not required,
                "missing qa/slam_time_sync_benchmark.json or equivalent",
            )
        ]
        for metric in required_metrics:
            checks.append(build_check(f"slam_metric_{metric}", False, "benchmark report missing"))
        return {
            "report_relpath": "",
            "metrics": {},
            "checks": checks,
            "ok": all(item["ok"] for item in checks),
        }

    report = load_json(report_path)
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    checks = [
        build_check(
            "slam_benchmark_report_present",
            True,
            str(report_path.relative_to(root)) if report_path.is_relative_to(root) else str(report_path),
        )
    ]
    for metric in required_metrics:
        metric_ok, metric_detail = metric_budget_check(metric, metrics.get(metric), metric_budgets)
        checks.append(
            build_check(
                f"slam_metric_{metric}",
                metric_ok,
                metric_detail,
            )
        )
    return {
        "report_relpath": str(report_path.relative_to(root)) if report_path.is_relative_to(root) else str(report_path),
        "metrics": metrics,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def validate(
    root: Path,
    threshold_config: dict[str, Any],
    tier: str,
    source_bundle: Path,
    slam_benchmark_report: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(root)
    time_sync = summarize_time_sync(root, threshold_config.get("time_sync") or {}, tier)
    vlm = summarize_vlm(root, threshold_config.get("vlm") or {}, tier)
    spatial = summarize_spatial(root, threshold_config.get("spatial") or {}, tier)
    slam = summarize_slam_benchmark(root, threshold_config.get("training") or {}, slam_benchmark_report)
    all_checks = [
        *time_sync["checks"],
        *vlm["checks"],
        *spatial["checks"],
        *slam["checks"],
    ]
    failed_checks = [item for item in all_checks if not item["ok"]]
    threshold_status = str(threshold_config.get("status") or "candidate")
    require_frozen = bool((threshold_config.get("training") or {}).get("require_frozen_thresholds_for_training_ready"))
    signal_candidate_ready = time_sync["ok"] and vlm["ok"] and spatial["ok"]
    training_ready = signal_candidate_ready and slam["ok"] and (threshold_status == "frozen" or not require_frozen)
    blockers = [str(item["id"]) for item in failed_checks]
    if require_frozen and threshold_status != "frozen":
        blockers.append("threshold_contract_not_frozen")
    return {
        "schema_version": "1.0.0",
        "validator": "chek-ego-miner.validate_training_thresholds",
        "threshold_version": threshold_config.get("version"),
        "threshold_status": threshold_status,
        "tier": tier,
        "source_bundle": str(source_bundle),
        "bundle_root": str(root),
        "session_id": manifest.get("session_id"),
        "trip_id": manifest.get("trip_id"),
        "task_id": manifest.get("task_id"),
        "capture_device_id": manifest.get("capture_device_id"),
        "manifest_relpath": manifest.get("_manifest_relpath"),
        "signal_candidate_ready": signal_candidate_ready,
        "training_ready": training_ready,
        "blockers": blockers,
        "time_sync": {key: value for key, value in time_sync.items() if key != "checks"},
        "vlm": {key: value for key, value in vlm.items() if key != "checks"},
        "spatial": {key: value for key, value in spatial.items() if key != "checks"},
        "slam_benchmark": {key: value for key, value in slam.items() if key != "checks"},
        "checks": all_checks,
    }


def print_human(payload: dict[str, Any]) -> None:
    print("CHEK EGO Miner training threshold validation")
    print(f"- tier: {payload['tier']}")
    print(f"- session_id: {payload.get('session_id') or 'unknown'}")
    print(f"- signal_candidate_ready: {payload['signal_candidate_ready']}")
    print(f"- training_ready: {payload['training_ready']}")
    if payload["blockers"]:
        print("- blockers:")
        for item in payload["blockers"]:
            print(f"  - {item}")


def main() -> int:
    args = parse_args()
    source_bundle = Path(args.bundle).expanduser()
    threshold_config = load_json(Path(args.threshold_config).expanduser())
    with tempfile.TemporaryDirectory(prefix="chek-ego-threshold-") as temp_dir:
        root = resolve_bundle_root(source_bundle, Path(temp_dir))
        slam_benchmark_report = Path(args.slam_benchmark_report).expanduser().resolve() if args.slam_benchmark_report else None
        payload = validate(root, threshold_config, args.tier, source_bundle, slam_benchmark_report)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).expanduser().write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print_human(payload)
        print("- json:")
        print(encoded)
    return 0 if payload["training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
