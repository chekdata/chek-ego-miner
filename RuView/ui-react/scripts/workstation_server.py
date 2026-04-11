#!/usr/bin/env python3

import argparse
import ipaddress
import json
import mimetypes
import posixpath
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

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


def normalize_relative_path(raw_path: str) -> Path:
    normalized = posixpath.normpath("/" + raw_path.lstrip("/")).lstrip("/")
    return Path(normalized)


def safe_join(root: Path, raw_path: str) -> Path:
    root = root.resolve()
    candidate = (root / normalize_relative_path(raw_path)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes root")
    return candidate


def validate_proxy_base(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("proxy base must use http or https")
    if not parsed.hostname:
        raise ValueError("proxy base must include a hostname")
    if parsed.scheme == "http":
        host = parsed.hostname
        try:
            ip = ipaddress.ip_address(host)
            if not (ip.is_loopback or ip.is_private or ip.is_link_local):
                raise ValueError("http proxy base must target localhost or a private address")
        except ValueError:
            if host.lower() != "localhost":
                raise ValueError("http proxy base must target localhost or a private address")
    return parsed.geturl().rstrip("/")


def sanitize_header_value(value: str) -> str:
    return value.replace("\r", "").replace("\n", "")


def ensure_allowed_file_target(target: Path, allowed_roots: list[Path]) -> Path:
    candidate = target.resolve()
    for root in allowed_roots:
        resolved_root = root.resolve()
        if resolved_root.is_dir():
            if candidate == resolved_root or resolved_root in candidate.parents:
                return candidate
        elif candidate == resolved_root:
            return candidate
    raise ValueError("file target is outside allowed roots")


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

        if path == "/stereo-preview.jpg":
            self.send_file(self.server.config.stereo_preview_path, with_body, cache_control="no-store")
            return

        if path == "/stereo-watchdog.json":
            self.respond_json(200, build_stereo_watchdog_payload(self.server.config))
            return

        for prefix, target in self.server.config.proxy_map.items():
            if path == prefix or path.startswith(prefix + "/"):
                self.proxy_request(prefix, target, with_body)
                return

        self.serve_app(path, with_body)

    def serve_app(self, path: str, with_body: bool):
        dist_dir = self.server.config.dist_dir
        relative = path.lstrip("/")
        target = None
        cache_control = None

        if relative:
            try:
                candidate = safe_join(dist_dir, relative)
            except ValueError:
                self.send_error(400, "invalid path")
                return
            if candidate.is_file():
                target = candidate
                if relative == "observatory.html" or relative.startswith("observatory/"):
                    cache_control = "no-store"
            elif candidate.suffix in KNOWN_ASSET_EXTENSIONS:
                self.send_error(404, "asset not found")
                return

        if target is None:
            target = dist_dir / "index.html"

        self.send_file(target, with_body, cache_control=cache_control)

    def send_file(self, target: Path, with_body: bool, cache_control: str | None = None):
        try:
            target = ensure_allowed_file_target(
                target,
                [
                    self.server.config.dist_dir,
                    self.server.config.stereo_preview_path,
                    self.server.config.stereo_watchdog_status_path,
                ],
            )
        except ValueError:
            self.send_error(400, "invalid file target")
            return
        if not target.exists() or not target.is_file():
            self.send_error(404, "file not found")
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if cache_control is not None:
            self.send_header("Cache-Control", cache_control)
        elif target.suffix == ".html":
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        if with_body:
            self.wfile.write(payload)

    def proxy_request(self, prefix: str, target_base: str, with_body: bool):
        parsed = urlsplit(self.path)
        stripped = parsed.path[len(prefix):] or "/"
        upstream_url = target_base.rstrip("/") + stripped
        if parsed.query:
            upstream_url = urlunsplit(urlsplit(upstream_url)._replace(query=parsed.query))

        body = None
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length > 0:
            body = self.rfile.read(content_length)

        request = Request(upstream_url, data=body, method=self.command)
        for key, value in self.headers.items():
            header_key = key.lower()
            if header_key in HOP_BY_HOP_HEADERS or header_key == "host":
                continue
            request.add_header(key, value)

        try:
            with urlopen(request, timeout=120) as upstream:
                self.send_response(upstream.status)
                for key, value in upstream.headers.items():
                    header_key = key.lower()
                    if header_key in HOP_BY_HOP_HEADERS:
                        continue
                    self.send_header(key, sanitize_header_value(value))
                self.end_headers()
                if with_body and self.command != "HEAD":
                    shutil.copyfileobj(upstream, self.wfile)
        except HTTPError as error:
            self.send_response(error.code)
            for key, value in error.headers.items():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, sanitize_header_value(value))
            payload = error.read()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if with_body and self.command != "HEAD" and payload:
                self.wfile.write(payload)
        except URLError as error:
            message = f"upstream unavailable: {error.reason}"
            self.send_error(502, message)

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
    parser.add_argument("--dist-dir", default=str(default_root / "dist"))
    parser.add_argument("--edge-http-base", default="http://127.0.0.1:8080")
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

    config = argparse.Namespace(
        dist_dir=dist_dir,
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
