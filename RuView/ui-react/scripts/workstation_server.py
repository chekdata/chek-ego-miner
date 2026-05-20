#!/usr/bin/env python3

import argparse
import hashlib
import http.client
import ipaddress
import json
import mimetypes
import os
import posixpath
import secrets
import shutil
import sqlite3
import ssl
import subprocess
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
ALLOWED_UPSTREAM_RESPONSE_HEADERS = {
    "cache-control": "Cache-Control",
    "content-disposition": "Content-Disposition",
    "content-encoding": "Content-Encoding",
    "content-length": "Content-Length",
    "content-type": "Content-Type",
    "etag": "ETag",
    "last-modified": "Last-Modified",
}
KNOWN_ASSET_EXTENSIONS = {
    ".css",
    ".gif",
    ".html",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".map",
    ".png",
    ".svg",
    ".txt",
    ".wasm",
    ".webm",
    ".woff",
    ".woff2",
}
PAIRING_TYPE = "chek_ego_edge_pairing"
PAIRING_PROFILE_ID = "ego_wide_rgbd_multi_iphone_v1"


@dataclass(frozen=True)
class ProxyTarget:
    scheme: str
    host: str
    port: int
    base_path: str


def validate_proxy_base(raw_url: str) -> ProxyTarget:
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("proxy base must use http or https")
    if not parsed.hostname:
        raise ValueError("proxy base must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("proxy base cannot include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("proxy base cannot include query or fragment")
    if parsed.scheme == "http":
        host = parsed.hostname
        try:
            ip = ipaddress.ip_address(host)
            if not (ip.is_loopback or ip.is_private or ip.is_link_local):
                raise ValueError("http proxy base must target localhost or a private address")
        except ValueError:
            if host.lower() != "localhost":
                raise ValueError("http proxy base must target localhost or a private address")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_path = parsed.path or "/"
    if not base_path.startswith("/"):
        raise ValueError("proxy base path must start with /")
    base_path = posixpath.normpath(base_path)
    if not base_path.startswith("/"):
        raise ValueError("proxy base path is invalid")
    return ProxyTarget(
        scheme=parsed.scheme,
        host=parsed.hostname,
        port=port,
        base_path=base_path,
    )


def sanitize_header_value(value: str) -> str:
    return value.replace("\r", "").replace("\n", "")


def is_safe_proxy_token(value: str) -> bool:
    if not value:
        return True
    sanitized = (
        value.replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .replace("~", "")
    )
    return sanitized.isalnum()


def normalize_asset_request_key(raw_path: str) -> str | None:
    value = raw_path.strip("/")
    if not value:
        return ""
    normalized = posixpath.normpath("/" + value).lstrip("/")
    if not normalized:
        return ""
    for segment in normalized.split("/"):
        if not segment or segment in {".", ".."} or not is_safe_proxy_token(segment):
            return None
    return normalized


def build_dist_file_index(dist_dir: Path) -> dict[str, str]:
    dist_root = dist_dir.resolve()
    files: dict[str, str] = {}
    for path in dist_root.rglob("*"):
        if not path.is_file():
            continue
        files[path.relative_to(dist_root).as_posix()] = os.path.realpath(path)
    return files


def normalize_proxy_path(raw_suffix: str) -> str:
    normalized = posixpath.normpath("/" + raw_suffix.lstrip("/"))
    if normalized == "/control/state":
        return "/control/state"
    if normalized == "/control/disarm":
        return "/control/disarm"
    if normalized == "/live-preview.json":
        return "/live-preview.json"
    if normalized == "/time":
        return "/time"
    if normalized == "/time/sync":
        return "/time/sync"
    if normalized == "/time/sync/current":
        return "/time/sync/current"
    if normalized == "/storage/status":
        return "/storage/status"
    if normalized == "/storage/sessions":
        return "/storage/sessions"
    if normalized == "/stereo-watchdog.json":
        return "/stereo-watchdog.json"
    if normalized == "/network/uplink":
        return "/network/uplink"
    if normalized == "/session/start":
        return "/session/start"
    if normalized == "/session/stop":
        return "/session/stop"
    if normalized == "/common_task/upload_chunk":
        return "/common_task/upload_chunk"
    if normalized == "/chunk/upload":
        return "/chunk/upload"
    if normalized == "/chunk/cleaned":
        return "/chunk/cleaned"
    if normalized == "/control/keepalive":
        return "/control/keepalive"
    if normalized == "/ingest/phone_vision_frame":
        return "/ingest/phone_vision_frame"
    if normalized == "/ingest/capture_pose":
        return "/ingest/capture_pose"
    if normalized == "/api/replay/session":
        return "/api/replay/session"
    if normalized == "/api/v1/model/info":
        return "/api/v1/model/info"
    if normalized == "/api/v1/pose/current":
        return "/api/v1/pose/current"
    if normalized == "/api/v1/pose/zones/summary":
        return "/api/v1/pose/zones/summary"
    if normalized == "/api/v1/recording/start":
        return "/api/v1/recording/start"
    if normalized == "/api/v1/sensing/latest":
        return "/api/v1/sensing/latest"
    if normalized == "/api/v1/stream/status":
        return "/api/v1/stream/status"
    if normalized == "/api/v1/train/start":
        return "/api/v1/train/start"
    if normalized == "/api/v1/vital-signs":
        return "/api/v1/vital-signs"
    if normalized == "/association/hint":
        return "/association/hint"
    if normalized == "/health":
        return "/health"
    if normalized.startswith("/api/replay/frame/"):
        frame_id = normalized.removeprefix("/api/replay/frame/")
        if frame_id.isdigit():
            return f"/api/replay/frame/{frame_id}"
    if normalized == "/live-preview/file" or normalized.startswith("/live-preview/file/"):
        return normalized
    raise ValueError("invalid proxy path")


def build_proxy_request_target(base_path: str, raw_suffix: str) -> str:
    suffix = normalize_proxy_path(raw_suffix)
    if base_path == "/":
        request_target = suffix
    elif suffix == "/":
        request_target = base_path
    else:
        request_target = base_path.rstrip("/") + suffix
    return request_target


def build_https_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def systemd_properties(service_name: str) -> dict[str, str]:
    result = run_command(
        [
            "systemctl",
            "show",
            service_name,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
        ]
    )
    props: dict[str, str] = {}
    if result.returncode != 0:
        return props
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        props[key] = value
    return props


def preview_age_sec(preview_path: Path) -> float | None:
    try:
        stat_result = preview_path.stat()
    except FileNotFoundError:
        return None
    return max(0.0, time.time() - stat_result.st_mtime)


def load_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load_device_registry(path: Path) -> dict[str, dict]:
    payload = load_json_file(path)
    raw_devices = payload.get("devices", {})
    if isinstance(raw_devices, list):
        devices = {
            str(item.get("device_id") or "").strip(): item
            for item in raw_devices
            if isinstance(item, dict) and str(item.get("device_id") or "").strip()
        }
        return devices
    if isinstance(raw_devices, dict):
        return {
            str(device_id).strip(): item
            for device_id, item in raw_devices.items()
            if str(device_id).strip() and isinstance(item, dict)
        }
    return {}


def save_device_registry(config) -> None:
    registry_path = getattr(config, "device_registry_path", None)
    if not registry_path:
        return
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "profile_id": config.profile_id,
        "updated_unix_ms": _now_unix_ms(),
        "devices": sorted(
            config.device_registry.values(),
            key=lambda item: str(item.get("device_id") or ""),
        ),
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def hash_upload_token(upload_token: str) -> str:
    return hashlib.sha256(upload_token.encode("utf-8")).hexdigest()


def token_authority_db_path(config) -> Path | None:
    raw = str(getattr(config, "upload_token_authority_db_path", "") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def ensure_token_authority_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paired_devices (
                device_id TEXT PRIMARY KEY,
                device_name TEXT NOT NULL DEFAULT '',
                login_identity TEXT NOT NULL DEFAULT '',
                profile_id TEXT NOT NULL DEFAULT '',
                paired_unix_ms INTEGER NOT NULL,
                last_seen_unix_ms INTEGER NOT NULL,
                last_session_id TEXT,
                upload_queue_depth INTEGER,
                ingest_status TEXT,
                status TEXT NOT NULL DEFAULT 'paired'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_tokens (
                token_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                token_sha256 TEXT NOT NULL UNIQUE,
                issued_unix_ms INTEGER NOT NULL,
                expires_unix_ms INTEGER NOT NULL,
                revoked_unix_ms INTEGER,
                status TEXT NOT NULL,
                last_used_unix_ms INTEGER,
                last_used_session_id TEXT,
                FOREIGN KEY(device_id) REFERENCES paired_devices(device_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_upload_tokens_device ON upload_tokens(device_id, status, expires_unix_ms)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_authority_audit (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_unix_ms INTEGER NOT NULL,
                device_id TEXT,
                token_id TEXT,
                session_id TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )


def mirror_scoped_upload_token_to_authority(
    config,
    *,
    record: dict,
    upload_token: str,
    issued_unix_ms: int,
    expires_unix_ms: int,
) -> None:
    db_path = token_authority_db_path(config)
    if db_path is None:
        return
    ensure_token_authority_schema(db_path)
    token_id = secrets.token_urlsafe(18)
    device_id = str(record.get("device_id") or "").strip()
    profile_id = str(record.get("profile_id") or config.profile_id).strip()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO paired_devices (
                device_id, device_name, login_identity, profile_id,
                paired_unix_ms, last_seen_unix_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'paired')
            ON CONFLICT(device_id) DO UPDATE SET
                device_name=excluded.device_name,
                login_identity=excluded.login_identity,
                profile_id=excluded.profile_id,
                last_seen_unix_ms=excluded.last_seen_unix_ms,
                status='paired'
            """,
            (
                device_id,
                str(record.get("device_name") or device_id).strip(),
                str(record.get("login_identity") or "unknown").strip(),
                profile_id,
                int(record.get("paired_unix_ms") or issued_unix_ms),
                issued_unix_ms,
            ),
        )
        conn.execute(
            """
            INSERT INTO upload_tokens (
                token_id, device_id, profile_id, token_sha256,
                issued_unix_ms, expires_unix_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'issued_by_workstation_pairing_endpoint')
            """,
            (
                token_id,
                device_id,
                profile_id,
                hash_upload_token(upload_token),
                issued_unix_ms,
                expires_unix_ms,
            ),
        )
        conn.execute(
            """
            INSERT INTO token_authority_audit (
                event_id, event_type, event_unix_ms, device_id, token_id, payload_json
            ) VALUES (?, 'token_issued_by_workstation_proxy', ?, ?, ?, ?)
            """,
            (
                secrets.token_urlsafe(18),
                issued_unix_ms,
                device_id,
                token_id,
                json.dumps({"profile_id": profile_id}, sort_keys=True),
            ),
        )


def mirror_device_status_to_authority(config, record: dict, now_ms: int) -> None:
    db_path = token_authority_db_path(config)
    if db_path is None:
        return
    ensure_token_authority_schema(db_path)
    device_id = str(record.get("device_id") or "").strip()
    if not device_id:
        return
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO paired_devices (
                device_id, device_name, login_identity, profile_id,
                paired_unix_ms, last_seen_unix_ms, last_session_id,
                upload_queue_depth, ingest_status, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'paired')
            ON CONFLICT(device_id) DO UPDATE SET
                device_name=excluded.device_name,
                login_identity=excluded.login_identity,
                profile_id=excluded.profile_id,
                last_seen_unix_ms=excluded.last_seen_unix_ms,
                last_session_id=excluded.last_session_id,
                upload_queue_depth=excluded.upload_queue_depth,
                ingest_status=excluded.ingest_status,
                status='paired'
            """,
            (
                device_id,
                str(record.get("device_name") or device_id).strip(),
                str(record.get("login_identity") or "unknown").strip(),
                str(record.get("profile_id") or config.profile_id).strip(),
                int(record.get("paired_unix_ms") or now_ms),
                now_ms,
                record.get("session_id"),
                record.get("upload_queue_depth"),
                record.get("ingest_status"),
            ),
        )


def public_device_record(record: dict) -> dict:
    hidden_keys = {"upload_token_sha256", "last_pairing_code"}
    return {key: value for key, value in record.items() if key not in hidden_keys}


def _now_unix_ms() -> int:
    return int(time.time() * 1000)


def _public_base_from_host(*, scheme: str, host: str, port: int) -> str:
    normalized_host = host.strip() or "127.0.0.1"
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    return f"{scheme}://{normalized_host}:{port}"


def _status_ui_base(config) -> str:
    if config.status_ui_public_base:
        return config.status_ui_public_base.rstrip("/")
    public_host = config.public_host or config.bind_host
    return _public_base_from_host(scheme="http", host=public_host, port=config.port)


def _edge_public_base(config) -> str:
    return (config.edge_public_base or config.edge_http_base).rstrip("/")


def _edge_ws_public_base(config) -> str:
    return (config.edge_ws_public_base or config.edge_ws_base).rstrip("/")


def _cleanup_pairing_state(config) -> bool:
    changed = False
    now_ms = _now_unix_ms()
    expired = [
        challenge
        for challenge, item in config.pairing_challenges.items()
        if int(item.get("expires_unix_ms", 0)) <= now_ms
    ]
    for challenge in expired:
        config.pairing_challenges.pop(challenge, None)
        changed = True
    for device_id, item in list(config.device_registry.items()):
        token_expires_ms = int(item.get("token_expires_unix_ms", 0) or 0)
        if token_expires_ms and token_expires_ms <= now_ms and item.get("upload_token_status") != "expired":
            item["upload_token_status"] = "expired"
            changed = True
    return changed


def build_pairing_envelope(config) -> dict:
    if _cleanup_pairing_state(config):
        save_device_registry(config)
    created_ms = _now_unix_ms()
    ttl_sec = max(30, int(config.pairing_ttl_sec))
    expires_ms = created_ms + ttl_sec * 1000
    pairing_code = f"{secrets.randbelow(1_000_000):06d}"
    challenge = secrets.token_urlsafe(24)
    envelope = {
        "type": PAIRING_TYPE,
        "version": "1.0",
        "profile_id": config.profile_id,
        "edge_base_url": _edge_public_base(config),
        "edge_ws_url": _edge_ws_public_base(config),
        "status_ui_url": f"{_status_ui_base(config)}/#/capture",
        "pairing_code": pairing_code,
        "pairing_challenge": challenge,
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_ms / 1000)),
        "expires_unix_ms": expires_ms,
        "operator_hint": config.operator_hint,
        "scene_hint": config.scene_hint,
    }
    config.pairing_challenges[challenge] = {
        "pairing_code": pairing_code,
        "created_unix_ms": created_ms,
        "expires_unix_ms": expires_ms,
        "status": "issued",
    }
    return envelope


def _read_json_body(handler: BaseHTTPRequestHandler, max_bytes: int = 16 * 1024) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        return {}
    if content_length > max_bytes:
        raise ValueError("request body too large")
    raw = handler.rfile.read(content_length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid json body") from exc
    if not isinstance(payload, dict):
        raise ValueError("expected json object")
    return payload


def exchange_pairing_challenge(config, payload: dict) -> tuple[int, dict]:
    if _cleanup_pairing_state(config):
        save_device_registry(config)
    challenge = str(payload.get("pairing_challenge") or "").strip()
    pairing_code = str(payload.get("pairing_code") or "").strip()
    device_id = str(payload.get("device_id") or "").strip()
    login_identity = str(payload.get("login_identity") or payload.get("operator_id") or "").strip()
    device_name = str(payload.get("device_name") or "").strip()

    if not challenge or not pairing_code or not device_id:
        return 400, {
            "ok": False,
            "error": "missing_required_fields",
            "required": ["pairing_challenge", "pairing_code", "device_id"],
        }

    issued = config.pairing_challenges.get(challenge)
    if not issued:
        return 404, {"ok": False, "error": "unknown_or_expired_pairing_challenge"}
    if issued.get("pairing_code") != pairing_code:
        return 403, {"ok": False, "error": "pairing_code_mismatch"}

    now_ms = _now_unix_ms()
    token_ttl_sec = max(300, int(config.upload_token_ttl_sec))
    token_expires_ms = now_ms + token_ttl_sec * 1000
    upload_token = secrets.token_urlsafe(32)
    record = {
        "device_id": device_id,
        "device_name": device_name or device_id,
        "login_identity": login_identity or "unknown",
        "profile_id": config.profile_id,
        "paired_unix_ms": now_ms,
        "last_seen_unix_ms": now_ms,
        "last_pairing_code": pairing_code,
        "upload_token_sha256": hash_upload_token(upload_token),
        "upload_token_status": "issued_by_workstation_pairing_endpoint",
        "token_expires_unix_ms": token_expires_ms,
        "last_ack": None,
        "upload_queue_depth": None,
        "session_id": None,
    }
    config.device_registry[device_id] = record
    issued["status"] = "exchanged"
    issued["exchanged_unix_ms"] = now_ms
    issued["device_id"] = device_id
    mirror_scoped_upload_token_to_authority(
        config,
        record=record,
        upload_token=upload_token,
        issued_unix_ms=now_ms,
        expires_unix_ms=token_expires_ms,
    )
    save_device_registry(config)

    return 200, {
        "ok": True,
        "device": public_device_record(record),
        "scoped_upload_token": upload_token,
        "token_type": "chek_edge_pairing_bearer",
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(token_expires_ms / 1000)),
        "expires_unix_ms": token_expires_ms,
        "edge_base_url": _edge_public_base(config),
        "edge_ws_url": _edge_ws_public_base(config),
        "status_ui_url": f"{_status_ui_base(config)}/#/capture",
    }


def update_device_status(config, payload: dict) -> tuple[int, dict]:
    if _cleanup_pairing_state(config):
        save_device_registry(config)
    device_id = str(payload.get("device_id") or "").strip()
    if not device_id:
        return 400, {"ok": False, "error": "missing_required_fields", "required": ["device_id"]}

    now_ms = _now_unix_ms()
    record = dict(config.device_registry.get(device_id) or {})
    record.setdefault("device_id", device_id)
    record.setdefault("device_name", device_id)
    record.setdefault("login_identity", "unknown")
    record.setdefault("profile_id", config.profile_id)
    record.setdefault("paired_unix_ms", now_ms)

    for key in ("device_name", "login_identity", "profile_id", "session_id", "upload_token_status"):
        if key in payload and payload.get(key) is not None:
            record[key] = str(payload.get(key)).strip()

    if "upload_queue_depth" in payload:
        try:
            record["upload_queue_depth"] = max(0, int(payload.get("upload_queue_depth")))
        except (TypeError, ValueError):
            return 400, {"ok": False, "error": "invalid_upload_queue_depth"}

    if "last_ack" in payload:
        last_ack = payload.get("last_ack")
        if last_ack is not None and not isinstance(last_ack, (dict, str)):
            return 400, {"ok": False, "error": "invalid_last_ack"}
        record["last_ack"] = last_ack

    for key in ("last_error", "ingest_status"):
        if key in payload:
            value = payload.get(key)
            record[key] = None if value is None else str(value).strip()

    record["last_seen_unix_ms"] = now_ms
    record["status_source"] = "workstation_device_status_endpoint"
    config.device_registry[device_id] = record
    mirror_device_status_to_authority(config, record, now_ms)
    save_device_registry(config)
    return 200, {"ok": True, "device": public_device_record(record)}


def build_device_registry_payload(config) -> dict:
    if _cleanup_pairing_state(config):
        save_device_registry(config)
    return {
        "generated_unix_ms": _now_unix_ms(),
        "profile_id": config.profile_id,
        "devices": sorted(
            [public_device_record(record) for record in config.device_registry.values()],
            key=lambda item: str(item.get("device_id") or ""),
        ),
        "active_pairing_challenges": len(config.pairing_challenges),
    }


def build_stereo_watchdog_payload(config) -> dict:
    payload = load_json_file(config.stereo_watchdog_status_path)
    stereo_service = systemd_properties("chek-edge-stereo.service")
    watchdog_service = systemd_properties("chek-edge-stereo-watchdog.service")

    if "preview_age_sec" not in payload:
        payload["preview_age_sec"] = preview_age_sec(config.stereo_preview_path)
    payload["status_path"] = str(config.stereo_watchdog_status_path)
    payload["preview_path"] = str(config.stereo_preview_path)
    payload["stereo_service_state"] = stereo_service.get("ActiveState", "unknown")
    payload["stereo_service_substate"] = stereo_service.get("SubState", "unknown")
    payload["watchdog_service_state"] = watchdog_service.get("ActiveState", "unknown")
    payload["watchdog_service_substate"] = watchdog_service.get("SubState", "unknown")
    payload["watchdog_enabled"] = payload["watchdog_service_state"] in {"active", "activating"}

    if "status" not in payload:
        if not payload["watchdog_enabled"]:
            payload["status"] = "offline"
        elif payload.get("healthy"):
            payload["status"] = "healthy"
        else:
            payload["status"] = "unknown"

    if "healthy" not in payload:
        payload["healthy"] = (
            payload["watchdog_enabled"]
            and payload["stereo_service_state"] in {"active", "activating"}
            and not payload.get("reasons")
        )

    return payload


class WorkstationServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, config):
        super().__init__(server_address, handler_class)
        self.config = config


class WorkstationHandler(BaseHTTPRequestHandler):
    server_version = "CHEKWorkstationServer/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.handle_request(with_body=True)

    def do_HEAD(self):
        self.handle_request(with_body=False)

    def do_POST(self):
        self.handle_request(with_body=True)

    def do_PUT(self):
        self.handle_request(with_body=True)

    def do_PATCH(self):
        self.handle_request(with_body=True)

    def do_DELETE(self):
        self.handle_request(with_body=True)

    def do_OPTIONS(self):
        self.handle_request(with_body=False)

    def handle_request(self, with_body: bool):
        parsed = urlsplit(self.path)
        path = parsed.path or "/"

        if path == "/healthz":
            self.respond_json(200, {"status": "ok"})
            return

        if path in {"/pairing/envelope", "/pairing/envelope.json", "/api/v1/pairing/envelope"}:
            self.respond_json(200, build_pairing_envelope(self.server.config))
            return

        if path in {"/devices.json", "/api/v1/devices"}:
            self.respond_json(200, build_device_registry_payload(self.server.config))
            return

        if path in {"/pairing/exchange", "/api/v1/pairing/exchange"}:
            if self.command != "POST":
                self.send_error(405, "method not allowed")
                return
            try:
                payload = _read_json_body(self)
            except ValueError as error:
                self.respond_json(400, {"ok": False, "error": str(error)})
                return
            status, response = exchange_pairing_challenge(self.server.config, payload)
            self.respond_json(status, response)
            return

        if path in {"/devices/status", "/api/v1/devices/status"}:
            if self.command not in {"POST", "PATCH"}:
                self.send_error(405, "method not allowed")
                return
            try:
                payload = _read_json_body(self)
            except ValueError as error:
                self.respond_json(400, {"ok": False, "error": str(error)})
                return
            status, response = update_device_status(self.server.config, payload)
            self.respond_json(status, response)
            return

        if path == "/stereo-preview.jpg":
            self.send_known_file(self.server.config.stereo_preview_path, with_body, cache_control="no-store")
            return

        if path == "/stereo-watchdog.json":
            self.respond_json(200, build_stereo_watchdog_payload(self.server.config))
            return

        if path in {"/live-preview.json", "/storage/status", "/storage/sessions"}:
            self.proxy_request("", self.server.config.proxy_map["/edge"], with_body)
            return

        if path == "/live-preview/file" or path.startswith("/live-preview/file/"):
            self.proxy_request("", self.server.config.proxy_map["/edge"], with_body)
            return

        for prefix, target in self.server.config.proxy_map.items():
            if path == prefix or path.startswith(prefix + "/"):
                self.proxy_request(prefix, target, with_body)
                return

        self.serve_app(path, with_body)

    def serve_app(self, path: str, with_body: bool):
        request_key = normalize_asset_request_key(path)
        if request_key is None:
            self.send_error(400, "invalid path")
            return
        target_path = self.server.config.dist_index_path
        cache_control = None

        if request_key:
            mapped_path = self.server.config.dist_files.get(request_key)
            if mapped_path is not None:
                target_path = mapped_path
                if request_key == "observatory.html" or request_key.startswith("observatory/"):
                    cache_control = "no-store"
            elif os.path.splitext(request_key)[1] in KNOWN_ASSET_EXTENSIONS:
                self.send_error(404, "asset not found")
                return

        if not os.path.isfile(target_path):
            self.send_error(404, "file not found")
            return
        try:
            with open(target_path, "rb") as handle:
                payload = handle.read()
        except OSError:
            self.send_error(500, "failed to read file")
            return

        content_type = (mimetypes.guess_type(target_path)[0] or "application/octet-stream").replace(
            "\r", ""
        ).replace("\n", "")
        self.send_payload(content_type, payload, with_body, cache_control, os.path.splitext(target_path)[1])

    def send_known_file(self, target: Path, with_body: bool, cache_control: str | None = None):
        target_path = os.fspath(target)
        if not os.path.isfile(target_path):
            self.send_error(404, "file not found")
            return
        try:
            with open(target_path, "rb") as handle:
                payload = handle.read()
        except OSError:
            self.send_error(500, "failed to read file")
            return
        content_type = (mimetypes.guess_type(target_path)[0] or "application/octet-stream").replace(
            "\r", ""
        ).replace("\n", "")
        self.send_payload(content_type, payload, with_body, cache_control, os.path.splitext(target_path)[1])

    def send_payload(
        self,
        content_type: str,
        payload: bytes,
        with_body: bool,
        cache_control: str | None,
        suffix: str,
    ):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if cache_control is not None:
            self.send_header("Cache-Control", cache_control)
        elif suffix == ".html":
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        if with_body:
            self.wfile.write(payload)

    def proxy_request(self, prefix: str, target: ProxyTarget, with_body: bool):
        parsed = urlsplit(self.path)
        stripped = parsed.path[len(prefix):] or "/"
        if parsed.query:
            self.send_error(400, "proxy query is not supported")
            return
        try:
            request_target = build_proxy_request_target(target.base_path, stripped)
        except ValueError:
            self.send_error(400, "invalid proxy path")
            return
        request_target_token = (
            request_target.replace("/", "")
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
            .replace("?", "")
            .replace("&", "")
            .replace("=", "")
            .replace("~", "")
        )
        if request_target_token and not request_target_token.isalnum():
            self.send_error(400, "invalid proxy path")
            return

        body = None
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length > 0:
            body = self.rfile.read(content_length)

        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            header_key = key.lower()
            if header_key in HOP_BY_HOP_HEADERS or header_key == "host":
                continue
            headers[key] = sanitize_header_value(value)

        connection: http.client.HTTPConnection | None = None
        try:
            if target.scheme == "https":
                connection = http.client.HTTPSConnection(
                    target.host,
                    target.port,
                    timeout=120,
                    context=build_https_context(),
                )
            else:
                connection = http.client.HTTPConnection(target.host, target.port, timeout=120)
            connection.request(self.command, request_target, body=body, headers=headers)
            upstream = connection.getresponse()
            try:
                self.send_response(upstream.status)
                for key, value in upstream.headers.items():
                    header_key = key.lower()
                    if header_key in HOP_BY_HOP_HEADERS:
                        continue
                    canonical_name = ALLOWED_UPSTREAM_RESPONSE_HEADERS.get(header_key)
                    if canonical_name is None:
                        continue
                    self.send_header(canonical_name, sanitize_header_value(value))
                self.end_headers()
                if with_body and self.command != "HEAD":
                    shutil.copyfileobj(upstream, self.wfile)
            finally:
                upstream.close()
        except (http.client.HTTPException, OSError) as error:
            message = f"upstream unavailable: {error}"
            self.send_error(502, message)
        finally:
            if connection is not None:
                connection.close()

    def respond_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, location: str):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


def parse_args():
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Serve CHEK React workstation dist with same-origin HTTP proxying.")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3010)
    parser.add_argument("--public-host", default="")
    parser.add_argument("--dist-dir", default=str(default_root / "dist"))
    parser.add_argument("--edge-http-base", default="http://127.0.0.1:8080")
    parser.add_argument("--edge-public-base", default="")
    parser.add_argument("--edge-ws-base", default="ws://127.0.0.1:8765")
    parser.add_argument("--edge-ws-public-base", default="")
    parser.add_argument("--status-ui-public-base", default="")
    parser.add_argument("--profile-id", default=PAIRING_PROFILE_ID)
    parser.add_argument("--pairing-ttl-sec", type=int, default=300)
    parser.add_argument("--upload-token-ttl-sec", type=int, default=3600)
    parser.add_argument("--device-registry-path", default="")
    parser.add_argument("--upload-token-authority-db-path", default="")
    parser.add_argument("--operator-hint", default="ego-capture")
    parser.add_argument("--scene-hint", default="unset")
    parser.add_argument("--sensing-http-base", default="http://127.0.0.1:18080")
    parser.add_argument("--sim-control-http-base", default="http://127.0.0.1:3011")
    parser.add_argument("--replay-http-base", default="http://127.0.0.1:3020")
    parser.add_argument("--stereo-preview-path", default="/tmp/stereo-uvc-preview.jpg")
    parser.add_argument("--stereo-watchdog-status-path", default="/run/chek-edge/stereo-watchdog.json")
    return parser.parse_args()


def main():
    args = parse_args()
    dist_dir = Path(args.dist_dir).resolve()
    if not (dist_dir / "index.html").exists():
        raise SystemExit(f"React workstation dist not found: {dist_dir / 'index.html'}")
    device_registry_path = (
        Path(args.device_registry_path).expanduser()
        if args.device_registry_path
        else dist_dir.parent / ".workstation-device-registry.json"
    ).resolve()

    config = argparse.Namespace(
        dist_dir=dist_dir,
        dist_files=build_dist_file_index(dist_dir),
        dist_index_path=os.path.realpath(dist_dir / "index.html"),
        bind_host=args.bind,
        port=args.port,
        public_host=args.public_host,
        edge_http_base=args.edge_http_base,
        edge_public_base=args.edge_public_base,
        edge_ws_base=args.edge_ws_base,
        edge_ws_public_base=args.edge_ws_public_base,
        status_ui_public_base=args.status_ui_public_base,
        profile_id=args.profile_id,
        pairing_ttl_sec=args.pairing_ttl_sec,
        upload_token_ttl_sec=args.upload_token_ttl_sec,
        device_registry_path=device_registry_path,
        upload_token_authority_db_path=args.upload_token_authority_db_path,
        operator_hint=args.operator_hint,
        scene_hint=args.scene_hint,
        pairing_challenges={},
        device_registry=load_device_registry(device_registry_path),
        proxy_map={
            "/edge": validate_proxy_base(args.edge_http_base),
            "/sensing": validate_proxy_base(args.sensing_http_base),
            "/sim-control": validate_proxy_base(args.sim_control_http_base),
            "/replay": validate_proxy_base(args.replay_http_base),
        },
        stereo_preview_path=Path(args.stereo_preview_path).resolve(),
        stereo_watchdog_status_path=Path(args.stereo_watchdog_status_path).resolve(),
    )
    server = WorkstationServer((args.bind, args.port), WorkstationHandler, config)
    print(f"CHEK workstation server listening on http://{args.bind}:{args.port}")
    print(f"  dist   -> {dist_dir}")
    print(f"  edge   -> {args.edge_http_base}")
    print(f"  pairing -> {_status_ui_base(config)}/pairing/envelope")
    print(f"  devices -> {device_registry_path}")
    if token_authority_db_path(config) is not None:
        print(f"  token authority -> {token_authority_db_path(config)}")
    print(f"  sensing-> {args.sensing_http_base}")
    print(f"  simctl -> {args.sim_control_http_base}")
    print(f"  replay -> {args.replay_http_base}")
    print(f"  stereo -> {config.stereo_preview_path}")
    print(f"  watchdog -> {config.stereo_watchdog_status_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
