#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_host_basics import build_report as build_host_report
from scripts.readiness_public import evaluate_readiness


SCRIPTS_DIR = REPO_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a public-safe CHEK EGO Miner tier E2E summary.",
    )
    parser.add_argument("--tier", choices=["lite", "stereo", "pro"], required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--report-path", help="Optional file path to write the JSON report.")
    parser.add_argument("--output-dir", default="./artifacts/public-e2e")
    parser.add_argument(
        "--capture-smoke",
        action="store_true",
        help="Open the selected camera and read one frame during readiness.",
    )
    parser.add_argument("--capture-timeout", type=float, default=8)
    parser.add_argument("--capture-device-index", type=int, default=0)
    parser.add_argument("--capture-device-name", default="")
    parser.add_argument("--capture-video-size", default="1280x720")
    parser.add_argument("--capture-framerate", default="30")
    parser.add_argument(
        "--run-basic-e2e",
        action="store_true",
        help="Run the synthetic basic local capture flow after readiness.",
    )
    parser.add_argument("--edge-base-url", default="")
    parser.add_argument("--edge-token", default="chek-ego-miner-local-token")
    parser.add_argument("--trip-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--interval-seconds", type=float, default=0.8)
    return parser.parse_args()


def _vlm_policy(tier: str, host_report: dict[str, Any]) -> dict[str, Any]:
    tools = host_report.get("tools") if isinstance(host_report.get("tools"), dict) else {}
    host = host_report.get("host") if isinstance(host_report.get("host"), dict) else {}
    if tier == "pro":
        return {
            "required_for_tier": True,
            "policy": "required_for_pro_but_validated_by_readiness_assets_and_sidecar_checks",
            "host_hint": {
                "system": host.get("system"),
                "machine": host.get("machine"),
                "node_present": bool(tools.get("node")),
            },
        }
    return {
        "required_for_tier": False,
        "policy": "optional_supported_for_capable_hosts",
        "host_hint": {
            "system": host.get("system"),
            "machine": host.get("machine"),
        },
    }


def _upload_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "disabled_by_default",
        "cloud_upload_attempted": False,
        "local_edge_base_url": args.edge_base_url.rstrip("/") if args.edge_base_url else "",
        "explicit_local_basic_e2e_requested": bool(args.run_basic_e2e),
        "note": (
            "public-e2e never uploads to cloud by itself. The synthetic basic flow only targets "
            "the explicit local edge base URL supplied by the operator."
        ),
    }


def _basic_e2e_blocker(args: argparse.Namespace) -> str:
    missing: list[str] = []
    if not args.edge_base_url:
        missing.append("--edge-base-url")
    if not args.trip_id:
        missing.append("--trip-id")
    if not args.session_id:
        missing.append("--session-id")
    if not missing:
        return ""
    return "run-basic-e2e requires " + ", ".join(missing)


def _redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value == "--edge-token":
            redacted[index + 1] = "<redacted>"
    return redacted


def run_basic_e2e(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    blocker = _basic_e2e_blocker(args)
    if blocker:
        return {
            "requested": True,
            "ok": False,
            "skipped": False,
            "reason": blocker,
            "returncode": 2,
        }

    report_path = output_dir / "basic_e2e_report.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "run_basic_e2e.py"),
        "--edge-base-url",
        args.edge_base_url,
        "--edge-token",
        args.edge_token,
        "--trip-id",
        args.trip_id,
        "--session-id",
        args.session_id,
        "--duration-seconds",
        str(args.duration_seconds),
        "--interval-seconds",
        str(args.interval_seconds),
        "--output-dir",
        str(output_dir / "basic-e2e"),
        "--report-path",
        str(report_path),
        "--json",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    payload: dict[str, Any] = {}
    if report_path.is_file():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    return {
        "requested": True,
        "ok": completed.returncode == 0 and bool(payload.get("ok")),
        "skipped": False,
        "reason": "ok" if completed.returncode == 0 and bool(payload.get("ok")) else "basic_e2e_failed",
        "returncode": completed.returncode,
        "command": _redact_command(command),
        "report_path": str(report_path),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "report": payload,
    }


def build_public_e2e_report(
    args: argparse.Namespace,
    *,
    host_report: dict[str, Any] | None = None,
    basic_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    host_report = host_report or build_host_report(
        capture_smoke=args.capture_smoke,
        capture_timeout=args.capture_timeout,
        capture_device_index=args.capture_device_index,
        capture_device_name=args.capture_device_name,
        capture_video_size=args.capture_video_size,
        capture_framerate=args.capture_framerate,
    )
    readiness = evaluate_readiness(
        args.tier,
        host_report,
        require_capture_smoke=args.capture_smoke,
    )
    if basic_result is None:
        basic_result = run_basic_e2e(args, output_dir) if args.run_basic_e2e else {
            "requested": False,
            "ok": None,
            "skipped": True,
            "reason": "not_requested",
        }

    local_capture_result = (
        "passed"
        if basic_result.get("requested") and basic_result.get("ok")
        else "failed"
        if basic_result.get("requested")
        else "skipped"
    )
    checks_ok = bool(readiness.get("ready")) and (
        bool(basic_result.get("ok")) if basic_result.get("requested") else True
    )
    payload = {
        "schema_version": "1.0.0",
        "type": "chek_ego_miner_public_e2e",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier": args.tier,
        "output_dir": str(output_dir),
        "ok": checks_ok,
        "host_os": (host_report.get("host") or {}).get("system"),
        "host_arch": (host_report.get("host") or {}).get("machine"),
        "hardware_tier": args.tier,
        "camera_readiness": {
            "device_count": len(host_report.get("video_devices") or []),
            "capture_smoke_requested": args.capture_smoke,
            "capture_smoke": readiness.get("capture_smoke"),
        },
        "vlm_policy": _vlm_policy(args.tier, host_report),
        "local_capture_result": local_capture_result,
        "upload": _upload_policy(args),
        "host_basics": host_report,
        "readiness": readiness,
        "basic_e2e": basic_result,
    }
    return payload


def print_human(payload: dict[str, Any]) -> None:
    print("CHEK EGO Miner public E2E")
    print(f"- tier: {payload['tier']}")
    print(f"- ok: {'yes' if payload['ok'] else 'no'}")
    print(f"- host: {payload.get('host_os')} {payload.get('host_arch')}")
    camera = payload.get("camera_readiness") or {}
    print(f"- camera devices: {camera.get('device_count')}")
    smoke = camera.get("capture_smoke")
    if isinstance(smoke, dict) and smoke.get("requested"):
        print(f"- capture smoke: {'ok' if smoke.get('ok') else 'failed'} ({smoke.get('reason')})")
    print(f"- local capture: {payload.get('local_capture_result')}")
    upload = payload.get("upload") or {}
    print(f"- upload: {upload.get('mode')} (cloud attempted: {upload.get('cloud_upload_attempted')})")
    readiness = payload.get("readiness") or {}
    if readiness.get("blockers"):
        print("- blockers:")
        for item in readiness.get("blockers") or []:
            print(f"  - {item.get('code')}: {item.get('message')}")


def main() -> int:
    args = parse_args()
    payload = build_public_e2e_report(args)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).expanduser().write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print_human(payload)
        print("- json:")
        print(encoded)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
