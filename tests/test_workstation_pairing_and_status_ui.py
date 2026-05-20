from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKSTATION = load_module(
    "workstation_server",
    REPO_ROOT / "RuView" / "ui-react" / "scripts" / "workstation_server.py",
)
PUBLIC_CLI = load_module(
    "chek_ego_miner_public_cli",
    REPO_ROOT / "cli" / "chek_ego_miner" / "main.py",
)


def pairing_config(
    device_registry_path: Path | None = None,
    upload_token_authority_db_path: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        bind_host="127.0.0.1",
        port=3010,
        public_host="192.168.1.20",
        edge_http_base="http://127.0.0.1:8080",
        edge_public_base="http://192.168.1.20:8080",
        edge_ws_base="ws://127.0.0.1:8765/stream/fusion",
        edge_ws_public_base="ws://192.168.1.20:8765/stream/fusion",
        status_ui_public_base="http://192.168.1.20:3010",
        profile_id="ego_wide_rgbd_multi_iphone_v1",
        pairing_ttl_sec=300,
        upload_token_ttl_sec=3600,
        device_registry_path=device_registry_path,
        upload_token_authority_db_path=upload_token_authority_db_path,
        operator_hint="ego-capture",
        scene_hint="kitchen-demo",
        pairing_challenges={},
        device_registry=WORKSTATION.load_device_registry(device_registry_path) if device_registry_path else {},
    )


def test_pairing_envelope_uses_lan_public_urls() -> None:
    config = pairing_config()

    envelope = WORKSTATION.build_pairing_envelope(config)

    assert envelope["type"] == "chek_ego_edge_pairing"
    assert envelope["profile_id"] == "ego_wide_rgbd_multi_iphone_v1"
    assert envelope["edge_base_url"] == "http://192.168.1.20:8080"
    assert envelope["edge_ws_url"] == "ws://192.168.1.20:8765/stream/fusion"
    assert envelope["status_ui_url"] == "http://192.168.1.20:3010/#/capture"
    assert len(envelope["pairing_code"]) == 6
    assert envelope["pairing_challenge"] in config.pairing_challenges


def test_pairing_exchange_registers_device_identity() -> None:
    config = pairing_config()
    envelope = WORKSTATION.build_pairing_envelope(config)

    status, response = WORKSTATION.exchange_pairing_challenge(
        config,
        {
            "pairing_challenge": envelope["pairing_challenge"],
            "pairing_code": envelope["pairing_code"],
            "device_id": "iphone-15-pro-a",
            "device_name": "Alice iPhone",
            "login_identity": "alice@example.com",
        },
    )

    assert status == 200
    assert response["ok"] is True
    assert response["device"]["device_id"] == "iphone-15-pro-a"
    assert response["device"]["login_identity"] == "alice@example.com"
    assert "upload_token_sha256" not in response["device"]
    assert response["scoped_upload_token"]
    assert (
        config.device_registry["iphone-15-pro-a"]["upload_token_sha256"]
        == WORKSTATION.hash_upload_token(response["scoped_upload_token"])
    )
    assert config.device_registry["iphone-15-pro-a"]["device_name"] == "Alice iPhone"


def test_pairing_exchange_mirrors_token_to_sqlite_authority(tmp_path: Path) -> None:
    registry_path = tmp_path / "devices.json"
    authority_path = tmp_path / "token_authority.sqlite3"
    config = pairing_config(registry_path, authority_path)
    envelope = WORKSTATION.build_pairing_envelope(config)

    status, response = WORKSTATION.exchange_pairing_challenge(
        config,
        {
            "pairing_challenge": envelope["pairing_challenge"],
            "pairing_code": envelope["pairing_code"],
            "device_id": "iphone-15-pro-a",
            "device_name": "Alice iPhone",
            "login_identity": "alice@example.com",
        },
    )

    assert status == 200
    assert response["scoped_upload_token"]
    with sqlite3.connect(authority_path) as conn:
        token_row = conn.execute(
            "SELECT device_id, profile_id, token_sha256, status FROM upload_tokens"
        ).fetchone()
        device_row = conn.execute(
            "SELECT device_name, login_identity FROM paired_devices WHERE device_id = ?",
            ("iphone-15-pro-a",),
        ).fetchone()
    assert token_row == (
        "iphone-15-pro-a",
        "ego_wide_rgbd_multi_iphone_v1",
        WORKSTATION.hash_upload_token(response["scoped_upload_token"]),
        "issued_by_workstation_pairing_endpoint",
    )
    assert device_row == ("Alice iPhone", "alice@example.com")


def test_device_registry_persists_pairing_and_status_updates(tmp_path: Path) -> None:
    registry_path = tmp_path / "devices.json"
    config = pairing_config(registry_path)
    envelope = WORKSTATION.build_pairing_envelope(config)

    status, response = WORKSTATION.exchange_pairing_challenge(
        config,
        {
            "pairing_challenge": envelope["pairing_challenge"],
            "pairing_code": envelope["pairing_code"],
            "device_id": "iphone-15-pro-a",
            "device_name": "Alice iPhone",
            "login_identity": "alice@example.com",
        },
    )

    assert status == 200
    assert registry_path.exists()
    assert response["device"]["login_identity"] == "alice@example.com"

    reloaded = pairing_config(registry_path)
    assert reloaded.device_registry["iphone-15-pro-a"]["device_name"] == "Alice iPhone"
    assert reloaded.device_registry["iphone-15-pro-a"]["upload_token_sha256"]

    status, response = WORKSTATION.update_device_status(
        reloaded,
        {
            "device_id": "iphone-15-pro-a",
            "session_id": "session-001",
            "upload_queue_depth": 2,
            "last_ack": {"session_id": "session-001", "edge_time_ns": 12345},
            "ingest_status": "accepted",
        },
    )

    assert status == 200
    assert response["device"]["session_id"] == "session-001"
    assert response["device"]["upload_queue_depth"] == 2
    assert "upload_token_sha256" not in response["device"]

    persisted = WORKSTATION.load_device_registry(registry_path)
    assert persisted["iphone-15-pro-a"]["session_id"] == "session-001"
    assert persisted["iphone-15-pro-a"]["last_ack"]["edge_time_ns"] == 12345
    assert persisted["iphone-15-pro-a"]["upload_token_sha256"]


def test_pairing_exchange_rejects_wrong_code() -> None:
    config = pairing_config()
    envelope = WORKSTATION.build_pairing_envelope(config)

    status, response = WORKSTATION.exchange_pairing_challenge(
        config,
        {
            "pairing_challenge": envelope["pairing_challenge"],
            "pairing_code": "000000",
            "device_id": "iphone-15-pro-a",
        },
    )

    assert status == 403
    assert response["error"] == "pairing_code_mismatch"
    assert config.device_registry == {}


def test_pairing_exchange_rejects_expired_challenge() -> None:
    config = pairing_config()
    envelope = WORKSTATION.build_pairing_envelope(config)
    config.pairing_challenges[envelope["pairing_challenge"]]["expires_unix_ms"] = 0

    status, response = WORKSTATION.exchange_pairing_challenge(
        config,
        {
            "pairing_challenge": envelope["pairing_challenge"],
            "pairing_code": envelope["pairing_code"],
            "device_id": "iphone-expired-a",
        },
    )

    assert status == 404
    assert response["error"] == "unknown_or_expired_pairing_challenge"
    assert envelope["pairing_challenge"] not in config.pairing_challenges


def test_device_registry_payload_supports_multiple_iphones_without_token_hashes() -> None:
    config = pairing_config()

    for device_id, login_identity in [
        ("iphone-15-pro-a", "alice@example.com"),
        ("iphone-15-pro-b", "bob@example.com"),
    ]:
        envelope = WORKSTATION.build_pairing_envelope(config)
        status, response = WORKSTATION.exchange_pairing_challenge(
            config,
            {
                "pairing_challenge": envelope["pairing_challenge"],
                "pairing_code": envelope["pairing_code"],
                "device_id": device_id,
                "login_identity": login_identity,
            },
        )
        assert status == 200
        assert response["scoped_upload_token"]

    payload = WORKSTATION.build_device_registry_payload(config)

    assert [device["device_id"] for device in payload["devices"]] == [
        "iphone-15-pro-a",
        "iphone-15-pro-b",
    ]
    assert [device["login_identity"] for device in payload["devices"]] == [
        "alice@example.com",
        "bob@example.com",
    ]
    assert all("upload_token_sha256" not in device for device in payload["devices"])


def test_capture_proxy_allowlist_includes_status_ui_dependencies() -> None:
    allowed_paths = {
        "/live-preview.json",
        "/storage/status",
        "/storage/sessions",
        "/time",
        "/time/sync",
        "/time/sync/current",
        "/control/state",
        "/control/disarm",
        "/control/keepalive",
        "/session/start",
        "/session/stop",
        "/common_task/upload_chunk",
        "/chunk/upload",
        "/chunk/cleaned",
        "/ingest/phone_vision_frame",
        "/ingest/capture_pose",
        "/network/uplink",
        "/stereo-watchdog.json",
        "/live-preview/file/iphone-wide-preview.jpg",
    }

    for path in allowed_paths:
        assert WORKSTATION.normalize_proxy_path(path) == path


def test_status_ui_url_defaults_to_capture_route_with_token() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "cli/chek_ego_miner/main.py",
            "status",
            "--edge-root",
            str(REPO_ROOT),
            "--edge-token",
            "local-token",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status_ui"]["url"] == "http://127.0.0.1:3010/?token=local-token#/capture"
    assert payload["operator_receipt"]["pairing_envelope_url"] == "http://127.0.0.1:3010/pairing/envelope"
    assert payload["operator_receipt"]["lan_ready_for_iphone"] is False


def test_install_defaults_to_open_status_ui_but_can_disable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "cli/chek_ego_miner/main.py",
            "install",
            "--edge-root",
            str(REPO_ROOT),
            "--no-open-ui",
            "--edge-token",
            "local-token",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status_ui"]["url"] == "http://127.0.0.1:3010/?token=local-token#/capture"
    assert payload["status_ui"]["open_requested"] is False
    assert payload["operator_receipt"]["quality_targets"]["minimum_usable_episode_seconds"] == 30


def test_public_cli_start_defaults_to_basic_direct_stack(monkeypatch) -> None:
    captured_args: list[str] = []

    def fake_run_runtime_cli(args: list[str]) -> int:
        captured_args.extend(args)
        return 0

    monkeypatch.setattr(PUBLIC_CLI, "run_runtime_cli", fake_run_runtime_cli)

    assert PUBLIC_CLI.dispatch_passthrough("start", ["--edge-root", str(REPO_ROOT), "--no-open-ui"]) == 0
    assert captured_args == [
        "service",
        "restart",
        "--direct",
        "--profile",
        "basic",
        "--edge-root",
        str(REPO_ROOT),
        "--no-open-ui",
    ]


def test_public_cli_start_preserves_explicit_profile(monkeypatch) -> None:
    captured_args: list[str] = []

    def fake_run_runtime_cli(args: list[str]) -> int:
        captured_args.extend(args)
        return 0

    monkeypatch.setattr(PUBLIC_CLI, "run_runtime_cli", fake_run_runtime_cli)

    assert PUBLIC_CLI.dispatch_passthrough("start", ["--profile", "enhanced", "--bind-host", "0.0.0.0"]) == 0
    assert captured_args == [
        "service",
        "restart",
        "--direct",
        "--profile",
        "enhanced",
        "--bind-host",
        "0.0.0.0",
    ]
