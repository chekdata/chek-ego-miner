#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _command_path(command: str) -> str:
    return shutil.which(command) or ""


def _run(
    command: list[str],
    *,
    timeout: float,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": _as_text(exc.stdout),
            "stderr": _as_text(exc.stderr),
            "timeout": True,
        }
    except OSError as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "timeout": False,
        }


def _system_profiler_cameras() -> list[dict[str, Any]]:
    profiler = _command_path("system_profiler")
    if not profiler:
        return []
    result = _run([profiler, "SPCameraDataType", "-json"], timeout=8)
    if result["returncode"] != 0 or not str(result["stdout"]).strip():
        return []
    try:
        payload = json.loads(str(result["stdout"]))
    except json.JSONDecodeError:
        return []
    devices: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("SPCameraDataType") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("_name") or item.get("spcamera_model-id") or "").strip()
        if not name:
            continue
        devices.append(
            {
                "id": f"system_profiler:{index}",
                "index": index,
                "name": name,
                "backend": "system_profiler",
                "source": "system_profiler",
                "model_id": str(item.get("spcamera_model-id") or "").strip(),
                "unique_id": str(item.get("spcamera_unique-id") or "").strip(),
            }
        )
    return devices


def _ffmpeg_avfoundation_cameras() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ffmpeg = _command_path("ffmpeg")
    if not ffmpeg:
        return [], {"available": False, "reason": "ffmpeg not found"}
    result = _run(
        [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        timeout=8,
    )
    text = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
    devices: list[dict[str, Any]] = []
    in_video_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = re.search(r"\]\s+\[(\d+)\]\s+(.+)$", line)
        if not match:
            continue
        index = int(match.group(1))
        name = match.group(2).strip()
        if name.lower().startswith("capture screen"):
            continue
        devices.append(
            {
                "id": f"avfoundation:{index}",
                "index": index,
                "name": name,
                "backend": "avfoundation",
                "source": "ffmpeg",
            }
        )
    return devices, {
        "available": True,
        "returncode": result["returncode"],
        "timeout": result["timeout"],
        "stderr_tail": str(result.get("stderr") or "")[-2000:],
    }


def _linux_video_devices() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for index, device_path in enumerate(sorted(glob.glob("/dev/video*"))):
        devices.append(
            {
                "id": device_path,
                "index": index,
                "name": Path(device_path).name,
                "path": device_path,
                "backend": "v4l2",
                "source": "glob",
            }
        )
    return devices


def _windows_video_devices() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shell = _command_path("pwsh") or _command_path("powershell")
    if not shell:
        return [], {"available": False, "reason": "powershell not found"}
    result = _run(
        [
            shell,
            "-NoProfile",
            "-Command",
            (
                "$devices = Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.Service -eq 'usbvideo' -or $_.PNPClass -eq 'Camera' }; "
                "$devices | ForEach-Object { @{ Name = $_.Name; DeviceID = $_.DeviceID } } | "
                "ConvertTo-Json -Compress"
            ),
        ],
        timeout=8,
    )
    devices: list[dict[str, Any]] = []
    if result["returncode"] == 0 and str(result["stdout"]).strip():
        try:
            payload = json.loads(str(result["stdout"]))
        except json.JSONDecodeError:
            payload = []
        if isinstance(payload, dict):
            payload = [payload]
        for index, item in enumerate(payload if isinstance(payload, list) else []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name") or "").strip()
            if not name:
                continue
            devices.append(
                {
                    "id": str(item.get("DeviceID") or f"windows-camera:{index}").strip(),
                    "index": index,
                    "name": name,
                    "backend": "windows-camera",
                    "source": "powershell",
                }
            )
    return devices, {
        "available": True,
        "returncode": result["returncode"],
        "timeout": result["timeout"],
        "stderr_tail": str(result.get("stderr") or "")[-2000:],
    }


def _dedupe_devices(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for device in devices:
        name = str(device.get("name") or "").strip()
        path = str(device.get("path") or "").strip()
        key = (path or name).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(device)
    return ordered


def _capture_smoke_macos(
    *,
    device_index: int,
    timeout: float,
    video_size: str,
    framerate: str,
) -> dict[str, Any]:
    ffmpeg = _command_path("ffmpeg")
    if not ffmpeg:
        return {"requested": True, "ok": False, "reason": "ffmpeg_not_found"}
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-f",
        "avfoundation",
        "-pixel_format",
        "uyvy422",
        "-framerate",
        framerate,
        "-video_size",
        video_size,
        "-i",
        f"{device_index}:none",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    result = _run(command, timeout=timeout)
    ok = result["returncode"] == 0
    reason = "ok" if ok else ("timeout_opening_camera" if result["timeout"] else "ffmpeg_failed")
    return {
        "requested": True,
        "ok": ok,
        "reason": reason,
        "backend": "avfoundation",
        "device_index": device_index,
        "timeout_seconds": timeout,
        "command": command,
        "returncode": result["returncode"],
        "stdout_tail": str(result.get("stdout") or "")[-2000:],
        "stderr_tail": str(result.get("stderr") or "")[-4000:],
    }


def _capture_smoke_linux(
    *,
    device_path: str,
    timeout: float,
) -> dict[str, Any]:
    ffmpeg = _command_path("ffmpeg")
    if not ffmpeg:
        return {"requested": True, "ok": False, "reason": "ffmpeg_not_found"}
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-f",
        "v4l2",
        "-i",
        device_path,
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    result = _run(command, timeout=timeout)
    ok = result["returncode"] == 0
    return {
        "requested": True,
        "ok": ok,
        "reason": "ok" if ok else ("timeout_opening_camera" if result["timeout"] else "ffmpeg_failed"),
        "backend": "v4l2",
        "device_path": device_path,
        "timeout_seconds": timeout,
        "command": command,
        "returncode": result["returncode"],
        "stdout_tail": str(result.get("stdout") or "")[-2000:],
        "stderr_tail": str(result.get("stderr") or "")[-4000:],
    }


def build_camera_report(
    *,
    capture_smoke: bool = False,
    timeout: float = 8,
    device_index: int = 0,
    video_size: str = "1280x720",
    framerate: str = "30",
) -> dict[str, Any]:
    system_name = platform.system()
    tools = {
        "ffmpeg": _command_path("ffmpeg"),
        "system_profiler": _command_path("system_profiler"),
        "pwsh": _command_path("pwsh"),
        "powershell": _command_path("powershell"),
    }
    diagnostics: dict[str, Any] = {}

    if system_name == "Darwin":
        profiler_devices = _system_profiler_cameras()
        ffmpeg_devices, ffmpeg_probe = _ffmpeg_avfoundation_cameras()
        diagnostics["system_profiler_device_count"] = len(profiler_devices)
        diagnostics["ffmpeg_avfoundation"] = ffmpeg_probe
        devices = _dedupe_devices([*ffmpeg_devices, *profiler_devices])
        smoke = (
            _capture_smoke_macos(
                device_index=device_index,
                timeout=timeout,
                video_size=video_size,
                framerate=framerate,
            )
            if capture_smoke and 0 <= device_index < len(devices)
            else {"requested": capture_smoke, "ok": False, "reason": "device_not_found"}
            if capture_smoke
            else {"requested": False}
        )
    elif system_name == "Linux":
        devices = _linux_video_devices()
        smoke = (
            _capture_smoke_linux(device_path=str(devices[device_index].get("path")), timeout=timeout)
            if capture_smoke and 0 <= device_index < len(devices)
            else {"requested": capture_smoke, "ok": False, "reason": "device_not_found"}
            if capture_smoke
            else {"requested": False}
        )
    elif system_name == "Windows":
        devices, windows_probe = _windows_video_devices()
        diagnostics["windows_camera_probe"] = windows_probe
        smoke = {
            "requested": capture_smoke,
            "ok": False,
            "reason": "capture_smoke_not_implemented_for_windows",
        } if capture_smoke else {"requested": False}
    else:
        devices = []
        smoke = {"requested": capture_smoke, "ok": False, "reason": "unsupported_platform"} if capture_smoke else {"requested": False}

    video_devices = [
        f"{device.get('backend')}:{device.get('index')}:{device.get('name')}"
        for device in devices
    ]
    return {
        "host": {
            "system": system_name,
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "tools": tools,
        "device_count": len(devices),
        "video_devices": video_devices,
        "devices": devices,
        "capture_smoke": smoke,
        "diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe local camera devices for CHEK EGO Miner.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--report-path", help="Optional file path to write the JSON report.")
    parser.add_argument("--capture-smoke", action="store_true", help="Try to open the camera and read one frame.")
    parser.add_argument("--timeout", type=float, default=8, help="Capture-smoke timeout in seconds.")
    parser.add_argument("--device-index", type=int, default=0, help="Camera device index to open for capture smoke.")
    parser.add_argument("--video-size", default="1280x720", help="macOS AVFoundation capture size.")
    parser.add_argument("--framerate", default="30", help="macOS AVFoundation capture framerate.")
    return parser.parse_args()


def print_human(report: dict[str, Any]) -> None:
    host = report["host"]
    print("CHEK EGO Miner camera probe")
    print(f"- system: {host['system']} {host['release']} ({host['machine']})")
    print(f"- device count: {report['device_count']}")
    for device in report["devices"]:
        print(f"  - [{device.get('backend')}:{device.get('index')}] {device.get('name')}")
    smoke = report["capture_smoke"]
    if smoke.get("requested"):
        print(f"- capture smoke: {'ok' if smoke.get('ok') else 'failed'} ({smoke.get('reason')})")


def main() -> int:
    args = parse_args()
    report = build_camera_report(
        capture_smoke=args.capture_smoke,
        timeout=args.timeout,
        device_index=args.device_index,
        video_size=args.video_size,
        framerate=args.framerate,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print_human(report)
        print("- json:")
        print(payload)
    smoke = report.get("capture_smoke") or {}
    if smoke.get("requested") and not smoke.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
