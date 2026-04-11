#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import socket
import ssl
import statistics
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open multiple real WebSocket clients and record bounded fanout metrics.")
    parser.add_argument("--url", required=True, help="WebSocket endpoint URL")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--report-path")
    return parser.parse_args()


def recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed while receiving frame")
        chunks.extend(chunk)
    return bytes(chunks)


def send_ws_frame(sock: socket.socket, opcode: int, payload: bytes = b"") -> None:
    mask_key = os.urandom(4)
    first = 0x80 | (opcode & 0x0F)
    length = len(payload)
    header = bytearray([first])
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    masked = bytes(payload[i] ^ mask_key[i % 4] for i in range(length))
    sock.sendall(bytes(header) + mask_key + masked)


def recv_ws_frame(sock: socket.socket) -> tuple[int, bytes]:
    first, second = recv_exact(sock, 2)
    opcode = first & 0x0F
    masked = (second & 0x80) != 0
    length = second & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", recv_exact(sock, 2))
    elif length == 127:
        (length,) = struct.unpack("!Q", recv_exact(sock, 8))
    mask_key = recv_exact(sock, 4) if masked else b""
    payload = recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(payload[i] ^ mask_key[i % 4] for i in range(length))
    return opcode, payload


def open_websocket(url: str, timeout: float) -> socket.socket:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"ws", "wss"}:
        raise ValueError(f"unsupported websocket scheme: {parsed.scheme}")

    host = parsed.hostname or ""
    port = parsed.port or (443 if scheme == "wss" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    if scheme == "ws" and not is_loopback_or_private_host(host):
        raise ValueError("plain ws:// connections are only allowed for localhost or private addresses")

    raw_sock = socket.create_connection((host, port), timeout=timeout)
    if scheme == "wss":
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        sock: socket.socket = context.wrap_socket(raw_sock, server_hostname=host)
    else:
        sock = raw_sock
    sock.settimeout(timeout)

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(request)

    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("websocket handshake failed: empty response")
        response.extend(chunk)
        if len(response) > 65536:
            raise ConnectionError("websocket handshake failed: oversized headers")

    header_blob = bytes(response).split(b"\r\n\r\n", 1)[0].decode("utf-8", errors="replace")
    lines = header_blob.split("\r\n")
    if not lines or "101" not in lines[0]:
        raise ConnectionError(f"websocket handshake failed: {lines[0] if lines else header_blob}")

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key_name, value = line.split(":", 1)
        headers[key_name.strip().lower()] = value.strip()

    accept_expected = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
    ).decode("ascii")
    if headers.get("sec-websocket-accept") != accept_expected:
        raise ConnectionError("websocket handshake failed: invalid Sec-WebSocket-Accept")

    return sock


def is_loopback_or_private_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def client_worker(url: str, timeout: float, duration_seconds: float, results: list[dict[str, object]], index: int, lock: threading.Lock) -> None:
    started_at = time.perf_counter()
    message_count = 0
    first_message_ms: float | None = None
    first_payload_bytes: int | None = None
    errors: list[str] = []
    sock: socket.socket | None = None
    try:
        sock = open_websocket(url, timeout)
        deadline = time.perf_counter() + max(duration_seconds, 0.0)
        while time.perf_counter() < deadline:
            opcode, payload = recv_ws_frame(sock)
            if opcode == 0x1:
                message_count += 1
                if first_message_ms is None:
                    first_message_ms = (time.perf_counter() - started_at) * 1000
                    first_payload_bytes = len(payload)
            elif opcode == 0x8:
                break
            elif opcode == 0x9:
                send_ws_frame(sock, 0xA, payload)
            elif opcode == 0xA:
                continue
        try:
            send_ws_frame(sock, 0x8, b"")
        except OSError:
            pass
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    result = {
        "client_index": index,
        "message_count": message_count,
        "first_message_ms": first_message_ms,
        "first_payload_bytes": first_payload_bytes,
        "elapsed_ms": (time.perf_counter() - started_at) * 1000,
        "errors": errors,
        "ok": message_count > 0 and not errors,
    }
    with lock:
        results.append(result)


def main() -> int:
    args = parse_args()
    threads: list[threading.Thread] = []
    results: list[dict[str, object]] = []
    lock = threading.Lock()
    for index in range(args.clients):
        thread = threading.Thread(
            target=client_worker,
            args=(args.url, args.connect_timeout, args.duration_seconds, results, index, lock),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    first_message_latencies = [
        float(item["first_message_ms"])
        for item in results
        if item.get("first_message_ms") is not None
    ]
    message_counts = [int(item["message_count"]) for item in results]
    ok_clients = sum(1 for item in results if item["ok"])
    report = {
        "url": args.url,
        "clients": args.clients,
        "duration_seconds": args.duration_seconds,
        "connect_timeout": args.connect_timeout,
        "ok_clients": ok_clients,
        "error_clients": len(results) - ok_clients,
        "success_rate": ok_clients / len(results) if results else 0.0,
        "message_count": {
            "min": min(message_counts) if message_counts else 0,
            "mean": statistics.mean(message_counts) if message_counts else 0.0,
            "p50": percentile([float(value) for value in message_counts], 0.50),
            "p95": percentile([float(value) for value in message_counts], 0.95),
            "max": max(message_counts) if message_counts else 0,
        },
        "first_message_ms": {
            "p50": percentile(first_message_latencies, 0.50),
            "p95": percentile(first_message_latencies, 0.95),
            "max": max(first_message_latencies) if first_message_latencies else 0.0,
        },
        "client_results": sorted(results, key=lambda item: item["client_index"]),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report_path:
        Path(args.report_path).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
