#!/usr/bin/env python3

import argparse
import http.client
import ipaddress
import json
import mimetypes
import os
import posixpath
import shutil
import ssl
import subprocess
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

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


def sanitize_proxy_suffix(raw_suffix: str) -> str:
    normalized = posixpath.normpath("/" + raw_suffix.lstrip("/"))
    if not normalized.startswith("/"):
        raise ValueError("invalid proxy path")
    encoded_segments: list[str] = []
    for segment in normalized.split("/"):
        if not segment:
            continue
        if segment == ".." or not is_safe_proxy_token(segment):
            raise ValueError("invalid proxy path")
        encoded_segments.append(quote(segment, safe="-._~"))
    suffix = "/" + "/".join(encoded_segments)
    if raw_suffix.endswith("/") and suffix != "/" and not suffix.endswith("/"):
        suffix += "/"
    return suffix


def sanitize_proxy_query(raw_query: str) -> str:
    if not raw_query:
        return ""
    pairs = parse_qsl(raw_query, keep_blank_values=True)
    for key, value in pairs:
        if not is_safe_proxy_token(key) or not is_safe_proxy_token(value):
            raise ValueError("invalid proxy query")
    return urlencode(pairs, doseq=True, safe="-._~")


def build_proxy_request_target(base_path: str, raw_suffix: str, raw_query: str) -> str:
    suffix = sanitize_proxy_suffix(raw_suffix)
    if base_path == "/":
        request_target = suffix
    elif suffix == "/":
        request_target = base_path
    else:
        request_target = base_path.rstrip("/") + suffix
    query = sanitize_proxy_query(raw_query)
    if query:
        request_target = f"{request_target}?{query}"
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
            self.send_known_file(self.server.config.stereo_preview_path, with_body, cache_control="no-store")
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
        try:
            request_target = build_proxy_request_target(target.base_path, stripped, parsed.query)
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
        dist_files=build_dist_file_index(dist_dir),
        dist_index_path=os.path.realpath(dist_dir / "index.html"),
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
