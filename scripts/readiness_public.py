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
        professional_assets = {
            "stereo_script": REPO_ROOT / "scripts" / "stereo_pose_producer.py",
            "stereo_autostart": REPO_ROOT / "scripts" / "edge_stereo_autostart.sh",
            "wifi_bridge_script": REPO_ROOT / "scripts" / "wifi_pose_bridge.py",
            "wifi_bridge_autostart": REPO_ROOT / "scripts" / "edge_wifi_bridge_autostart.sh",
            "wifi_sensing_autostart": REPO_ROOT / "scripts" / "edge_wifi_sensing_autostart.sh",
            "wifi_workspace": REPO_ROOT / "RuView" / "rust-port" / "wifi-densepose-rs" / "Cargo.toml",
            "professional_bootstrap": REPO_ROOT / "scripts" / "bootstrap_jetson_professional_runtime.sh",
        }
        host_bootstrap_assets = {
            "stereo_calibration": REPO_ROOT / "data" / "ruview" / "runtime" / "stereo_pair_calibration.json",
            "wifi_model": REPO_ROOT / "RuView" / "rust-port" / "wifi-densepose-rs" / "data" / "models" / "trained-supervised-live.rvf",
            "wifi_ui": REPO_ROOT / "RuView" / "ui",
            "edge_binary": REPO_ROOT / "edge-orchestrator" / "target" / "debug" / "edge-orchestrator",
            "leap_binary": REPO_ROOT / "ruview-leap-bridge" / "target" / "debug" / "ruview-leap-bridge",
            "unitree_binary": REPO_ROOT / "ruview-unitree-bridge" / "target" / "debug" / "ruview-unitree-bridge",
            "workstation_dist": REPO_ROOT / "RuView" / "ui-react" / "dist",
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
        for asset_name, path in professional_assets.items():
            if not path.is_file():
                blockers.append(
                    {
                        "code": f"pro_{asset_name}_missing",
                        "message": f"Missing required Pro asset: {path.relative_to(REPO_ROOT)}",
                    }
                )
        for asset_name, path in host_bootstrap_assets.items():
            exists = path.is_dir() if asset_name in {"wifi_ui", "workstation_dist"} else path.is_file()
            if not exists:
                blockers.append(
                    {
                        "code": f"pro_{asset_name}_missing",
                        "message": f"Missing required Pro runtime asset: {path.relative_to(REPO_ROOT)}",
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
