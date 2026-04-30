#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from scripts.camera_probe import build_camera_report
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from camera_probe import build_camera_report


def command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if output else executable


def detect_video_devices() -> list[str]:
    return [str(item) for item in list(build_camera_report().get("video_devices", []))]


def windows_runtime_tools() -> dict[str, bool]:
    return {
        "pwsh": shutil.which("pwsh") is not None,
        "powershell": shutil.which("powershell") is not None,
        "bash": shutil.which("bash") is not None,
        "schtasks": shutil.which("schtasks") is not None,
    }


def generic_tools() -> dict[str, str | None]:
    return {
        "python": sys.version.splitlines()[0],
        "git": command_version("git"),
        "bash": command_version("bash"),
        "curl": command_version("curl"),
        "node": command_version("node"),
        "npm": command_version("npm"),
    }


def classify_tier_hints(system_name: str, machine: str, video_devices: list[str], win_tools: dict[str, bool]) -> list[str]:
    hints: list[str] = []
    if system_name in {"Darwin", "Linux", "Windows"}:
        hints.append("lite_possible")
    if len(video_devices) >= 2:
        hints.append("stereo_candidate")
    if system_name == "Windows":
        if win_tools.get("bash"):
            hints.append("windows_bash_present")
        if win_tools.get("schtasks"):
            hints.append("windows_scheduler_present")
    if system_name == "Linux" and machine in {"aarch64", "arm64"}:
        hints.append("arm64_linux_detected")
    return hints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check basic host prerequisites for CHEK EGO Miner.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--report-path", help="Optional file path to write the JSON report.")
    return parser.parse_args()


def build_report() -> dict[str, object]:
    system_name = platform.system()
    machine = platform.machine()
    camera_report = build_camera_report()
    video_devices = [str(item) for item in list(camera_report.get("video_devices", []))]
    win_tools = windows_runtime_tools() if os.name == "nt" else {}
    return {
        "host": {
            "system": system_name,
            "release": platform.release(),
            "machine": machine,
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "tools": generic_tools(),
        "video_devices": video_devices,
        "video_device_details": camera_report.get("devices", []),
        "camera_probe": camera_report,
        "windows_tools": win_tools,
        "tier_hints": classify_tier_hints(system_name, machine, video_devices, win_tools),
    }


def print_human(report: dict[str, object]) -> None:
    host = report["host"]
    tools = report["tools"]
    video_devices = report["video_devices"]
    windows_tools = report["windows_tools"]
    print("CHEK EGO Miner host basics")
    print(f"- system: {host['system']} {host['release']} ({host['machine']})")
    print(f"- python: {tools['python']}")
    print(f"- git: {tools['git'] or 'missing'}")
    print(f"- bash: {tools['bash'] or 'missing'}")
    print(f"- curl: {tools['curl'] or 'missing'}")
    print(f"- node: {tools['node'] or 'missing'}")
    print(f"- npm: {tools['npm'] or 'missing'}")
    if os.name == "nt":
        print("- windows tools:")
        for key, value in windows_tools.items():
            print(f"  - {key}: {'present' if value else 'missing'}")
    else:
        print(f"- video devices: {len(video_devices)}")
        for device in list(report.get("video_device_details", []))[:8]:
            if isinstance(device, dict):
                print(f"  - [{device.get('backend')}:{device.get('index')}] {device.get('name')}")
            else:
                print(f"  - {device}")
    print("- tier hints:")
    for hint in report["tier_hints"]:
        print(f"  - {hint}")


def main() -> int:
    args = parse_args()
    report = build_report()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).write_text(payload + "\n", encoding="utf-8")

    if args.json:
        print(payload)
        return 0

    print_human(report)
    print("- json:")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
