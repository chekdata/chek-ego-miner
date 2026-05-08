from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_public_e2e_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_public_e2e.py"
    spec = importlib.util.spec_from_file_location("run_public_e2e", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PUBLIC_E2E = load_public_e2e_module()


def args_for(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "tier": "lite",
        "output_dir": str(tmp_path / "public-e2e"),
        "capture_smoke": False,
        "capture_timeout": 8,
        "capture_device_index": 0,
        "capture_device_name": "",
        "capture_video_size": "1280x720",
        "capture_framerate": "30",
        "run_basic_e2e": False,
        "edge_base_url": "",
        "edge_token": "chek-ego-miner-local-token",
        "trip_id": "",
        "session_id": "",
        "duration_seconds": 8.0,
        "interval_seconds": 0.8,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def host_report() -> dict[str, object]:
    return {
        "tools": {"git": "git version 2.0", "curl": "curl 8.0", "node": "v20.0.0"},
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
            "capture_smoke": {"requested": False},
        },
    }


def test_default_public_e2e_is_local_and_skips_upload(tmp_path: Path) -> None:
    payload = PUBLIC_E2E.build_public_e2e_report(
        args_for(tmp_path),
        host_report=host_report(),
    )

    assert payload["ok"] is True
    assert payload["basic_e2e"]["requested"] is False
    assert payload["local_capture_result"] == "skipped"
    assert payload["upload"]["mode"] == "disabled_by_default"
    assert payload["upload"]["cloud_upload_attempted"] is False


def test_run_basic_e2e_requires_explicit_local_target(tmp_path: Path) -> None:
    payload = PUBLIC_E2E.build_public_e2e_report(
        args_for(tmp_path, run_basic_e2e=True),
        host_report=host_report(),
    )

    assert payload["ok"] is False
    assert payload["local_capture_result"] == "failed"
    assert payload["basic_e2e"]["returncode"] == 2
    assert "--edge-base-url" in payload["basic_e2e"]["reason"]
    assert "--trip-id" in payload["basic_e2e"]["reason"]
    assert "--session-id" in payload["basic_e2e"]["reason"]


def test_pro_vlm_policy_is_required_even_when_readiness_blocks(tmp_path: Path) -> None:
    payload = PUBLIC_E2E.build_public_e2e_report(
        args_for(tmp_path, tier="pro"),
        host_report=host_report(),
        basic_result={"requested": False, "ok": None, "skipped": True, "reason": "not_requested"},
    )

    assert payload["ok"] is False
    assert payload["vlm_policy"]["required_for_tier"] is True
    assert payload["vlm_policy"]["policy"].startswith("required_for_pro")
    assert payload["readiness"]["ready"] is False


def test_basic_e2e_command_redacts_edge_token() -> None:
    command = [
        "/usr/bin/python3",
        "scripts/run_basic_e2e.py",
        "--edge-token",
        "secret-token",
        "--trip-id",
        "trip-1",
    ]

    assert PUBLIC_E2E._redact_command(command) == [
        "/usr/bin/python3",
        "scripts/run_basic_e2e.py",
        "--edge-token",
        "<redacted>",
        "--trip-id",
        "trip-1",
    ]
