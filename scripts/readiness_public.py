#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_host_basics import build_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-safe readiness check for CHEK EGO Miner.")
    parser.add_argument("--tier", choices=["lite", "stereo", "pro"], required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--report-path", help="Optional file path to write the JSON report.")
    return parser.parse_args()


def evaluate_readiness(tier: str, report: dict[str, object]) -> dict[str, object]:
    tools = report["tools"]
    host = report["host"]
    video_devices = report["video_devices"]
    windows_tools = report["windows_tools"]
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    for tool_name in ("git", "curl"):
        if not tools.get(tool_name):
            blockers.append(
                {
                    "code": f"{tool_name}_missing",
                    "message": f"Missing required tool: {tool_name}",
                }
            )

    if tier in {"stereo", "pro"}:
        if host["system"] == "Windows":
            warnings.append(
                {
                    "code": "windows_camera_manual_check_required",
                    "message": "Windows camera visibility still needs a manual Device Manager or vendor-tool check.",
                }
            )
            if not windows_tools.get("bash"):
                warnings.append(
                    {
                        "code": "windows_bash_missing",
                        "message": "A usable bash is recommended for current compatibility wrappers.",
                    }
                )
        elif len(video_devices) < 2:
            blockers.append(
                {
                    "code": "stereo_device_not_detected",
                    "message": "Stereo or Pro tiers expect at least two visible video devices on this host.",
                }
            )

    if tier == "pro":
        vlm_assets = {
            "sidecar_script": REPO_ROOT / "scripts" / "edge_vlm_sidecar.py",
            "start_script": REPO_ROOT / "scripts" / "start_edge_vlm_sidecar.sh",
            "requirements": REPO_ROOT / "scripts" / "edge_vlm_requirements.txt",
            "fetcher": REPO_ROOT / "scripts" / "fetch_vlm_models.py",
            "manifest": REPO_ROOT / "model-candidates" / "manifests" / "model_inventory.json",
        }
        if host["system"] != "Linux":
            warnings.append(
                {
                    "code": "pro_prefers_linux",
                    "message": "The Pro tier is primarily oriented around Linux edge hosts.",
                }
            )
        if host["machine"] not in {"aarch64", "arm64"}:
            warnings.append(
                {
                    "code": "pro_non_arm64_host",
                    "message": "This host is not an ARM64 Linux edge machine; treat Pro readiness as partial.",
                }
            )
        if not tools.get("node"):
            warnings.append(
                {
                    "code": "node_missing",
                    "message": "Node.js is recommended for local UI-related workflows.",
                }
            )
        for asset_name, path in vlm_assets.items():
            if not path.is_file():
                blockers.append(
                    {
                        "code": f"pro_{asset_name}_missing",
                        "message": f"Missing required Pro VLM asset: {path.relative_to(REPO_ROOT)}",
                    }
                )

    ready = len(blockers) == 0
    return {
        "tier": tier,
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "host": report["host"],
        "tier_hints": report["tier_hints"],
    }


def print_human(payload: dict[str, object]) -> None:
    print(f"CHEK EGO Miner readiness: {payload['tier']}")
    print(f"- ready: {'yes' if payload['ready'] else 'no'}")
    if payload["blockers"]:
        print("- blockers:")
        for item in payload["blockers"]:
            print(f"  - {item['code']}: {item['message']}")
    if payload["warnings"]:
        print("- warnings:")
        for item in payload["warnings"]:
            print(f"  - {item['code']}: {item['message']}")
    print("- tier hints:")
    for hint in payload["tier_hints"]:
        print(f"  - {hint}")


def main() -> int:
    args = parse_args()
    report = build_report()
    payload = evaluate_readiness(args.tier, report)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
        return 0
    print_human(payload)
    print("- json:")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
