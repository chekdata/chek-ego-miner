from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_readiness_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "readiness_public.py"
    spec = importlib.util.spec_from_file_location("readiness_public", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


READINESS = load_readiness_module()


def base_report(*, capture_smoke: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "tools": {"git": "git version 2.0", "curl": "curl 8.0"},
        "host": {
            "system": "Darwin",
            "release": "24.6.0",
            "machine": "arm64",
            "python_executable": "/usr/bin/python3",
        },
        "video_devices": [
            "avfoundation:0:FaceTime HD Camera",
            "avfoundation:1:USB Camera",
        ],
        "video_device_details": [],
        "windows_tools": {},
        "tier_hints": ["lite_possible", "stereo_candidate"],
        "camera_probe": {
            "capture_smoke": capture_smoke or {"requested": False},
        },
    }


def test_stereo_readiness_without_capture_smoke_preserves_device_count_gate() -> None:
    result = READINESS.evaluate_readiness("stereo", base_report())

    assert result["ready"] is True
    assert result["capture_smoke_required"] is False


def test_stereo_readiness_with_capture_smoke_requires_readable_frame() -> None:
    result = READINESS.evaluate_readiness(
        "stereo",
        base_report(capture_smoke={"requested": True, "ok": False, "reason": "timeout_opening_camera"}),
        require_capture_smoke=True,
    )

    assert result["ready"] is False
    assert any(blocker["code"] == "stereo_capture_smoke_failed" for blocker in result["blockers"])


def test_stereo_readiness_with_capture_smoke_passes_after_frame_read() -> None:
    result = READINESS.evaluate_readiness(
        "stereo",
        base_report(capture_smoke={"requested": True, "ok": True, "reason": "ok"}),
        require_capture_smoke=True,
    )

    assert result["ready"] is True
    assert result["capture_smoke"]["ok"] is True
