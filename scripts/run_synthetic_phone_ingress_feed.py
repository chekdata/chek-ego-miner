#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRg"
    "yIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyM"
    "jL/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL"
    "/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fA"
    "kM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd"
    "4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19j"
    "Z2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQ"
    "oL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzU"
    "vAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0d"
    "XZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1"
    "dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDsKKKK/Oz3D//Z"
)
TINY_DEPTH_F32_B64 = base64.b64encode(
    b"".join(struct.pack("<f", value) for value in (0.45, 0.46, 0.47, 0.48))
).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously feed synthetic phone vision ingress + deadman keepalive into an edge host."
    )
    parser.add_argument("--edge-base-url", required=True)
    parser.add_argument("--edge-token", default="chek-ego-miner-local-token")
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--interval-seconds", type=float, default=0.8)
    parser.add_argument("--device-id", default="iphone-synthetic-feed-001")
    parser.add_argument("--operator-track-id", default="operator-synthetic")
    parser.add_argument("--deadman-device-id", default="benchmark-synthetic-feed")
    parser.add_argument("--camera-mode", default="synthetic_phone_depth")
    parser.add_argument(
        "--image-path",
        help="Optional local image path. When provided, the image is re-encoded as JPEG and sent as the primary frame.",
    )
    parser.add_argument("--report-path")
    return parser.parse_args()


def http_json(
    base_url: str,
    path: str,
    *,
    method: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    raw = None
    if payload is not None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}{path}",
        method=method,
        headers=headers,
        data=raw,
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def build_keepalive_packet(args: argparse.Namespace, seq: int) -> dict[str, Any]:
    return {
        "type": "control_keepalive_packet",
        "schema_version": "1.0.0",
        "trip_id": args.trip_id,
        "session_id": args.session_id,
        "device_id": args.deadman_device_id,
        "source_time_ns": int(time.time_ns()),
        "seq": seq,
        "deadman_pressed": True,
    }


def load_primary_image_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.image_path:
        return {
            "image_w": 2,
            "image_h": 2,
            "sensor_image_w": 2,
            "sensor_image_h": 2,
            "camera_has_depth": True,
            "camera_calibration": {
                "fxPx": 1.0,
                "fyPx": 1.0,
                "cxPx": 1.0,
                "cyPx": 1.0,
                "referenceImageW": 2,
                "referenceImageH": 2,
            },
            "primary_image_jpeg_b64": TINY_JPEG_B64,
            "depth_f32_b64": TINY_DEPTH_F32_B64,
            "depth_w": 2,
            "depth_h": 2,
        }

    image_path = Path(args.image_path).expanduser().resolve()
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        image_w, image_h = rgb_image.size
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=90)
        jpeg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "image_w": image_w,
        "image_h": image_h,
        "sensor_image_w": image_w,
        "sensor_image_h": image_h,
        "camera_has_depth": False,
        "camera_calibration": {
            "fxPx": float(max(image_w, 1)),
            "fyPx": float(max(image_h, 1)),
            "cxPx": float(image_w) / 2.0,
            "cyPx": float(image_h) / 2.0,
            "referenceImageW": image_w,
            "referenceImageH": image_h,
        },
        "primary_image_jpeg_b64": jpeg_b64,
        "depth_f32_b64": None,
        "depth_w": None,
        "depth_h": None,
    }


def build_phone_frame_packet(
    args: argparse.Namespace,
    frame_id: int,
    image_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "tripId": args.trip_id,
        "sessionId": args.session_id,
        "deviceId": args.device_id,
        "operatorTrackId": args.operator_track_id,
        "sourceTimeNs": int(time.time_ns()),
        "frameId": frame_id,
        "cameraMode": args.camera_mode,
        "imageW": image_payload["image_w"],
        "imageH": image_payload["image_h"],
        "sensorImageW": image_payload["sensor_image_w"],
        "sensorImageH": image_payload["sensor_image_h"],
        "normalizedWasRotatedRight": False,
        "cameraHasDepth": image_payload["camera_has_depth"],
        "cameraCalibration": image_payload["camera_calibration"],
        "devicePose": {
            "position_m": [0.1, 0.2, 0.3],
            "rotation_deg": [1.0, 2.0, 3.0],
            "target_space": "operator_frame",
            "source": "synthetic",
        },
        "deviceMotion": {
            "accel": [0.01, 0.02, 0.03],
            "gyro": [0.11, 0.12, 0.13],
        },
        "primaryImageJpegB64": image_payload["primary_image_jpeg_b64"],
        "depthF32B64": image_payload["depth_f32_b64"],
        "depthW": image_payload["depth_w"],
        "depthH": image_payload["depth_h"],
    }


def main() -> int:
    args = parse_args()
    image_payload = load_primary_image_payload(args)
    started_at = time.time()
    deadline = started_at + max(args.duration_seconds, 0.1)
    seq = 1
    frame_id = 1
    keepalive_ok = 0
    frame_ok = 0
    errors: list[str] = []
    last_keepalive: dict[str, Any] | None = None
    last_frame: dict[str, Any] | None = None

    while time.time() < deadline:
        try:
            last_keepalive = http_json(
                args.edge_base_url,
                "/control/keepalive",
                method="POST",
                token=args.edge_token,
                payload=build_keepalive_packet(args, seq),
            )
            keepalive_ok += 1
        except Exception as exc:
            errors.append(f"keepalive[{seq}]: {exc}")

        try:
            last_frame = http_json(
                args.edge_base_url,
                "/ingest/phone_vision_frame",
                method="POST",
                token=args.edge_token,
                payload=build_phone_frame_packet(args, frame_id, image_payload),
            )
            frame_ok += 1
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            errors.append(f"phone_vision_frame[{frame_id}]: HTTP {exc.code} {body[:400]}")
        except Exception as exc:
            errors.append(f"phone_vision_frame[{frame_id}]: {exc}")

        seq += 1
        frame_id += 1
        time.sleep(max(args.interval_seconds, 0.05))

    control_state: dict[str, Any] | None = None
    live_preview: dict[str, Any] | None = None
    try:
        control_state = http_json(
            args.edge_base_url,
            "/control/state",
            method="GET",
            token=args.edge_token,
        )
    except Exception as exc:
        errors.append(f"control/state: {exc}")
    try:
        live_preview = http_json(
            args.edge_base_url,
            "/live-preview.json",
            method="GET",
            token=args.edge_token,
        )
    except Exception as exc:
        errors.append(f"live-preview.json: {exc}")

    report = {
        "schema_version": "1.0.0",
        "generated_at_unix_ms": int(time.time() * 1000),
        "edge_base_url": args.edge_base_url.rstrip("/"),
        "trip_id": args.trip_id,
        "session_id": args.session_id,
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "keepalive_ok": keepalive_ok,
        "phone_frame_ok": frame_ok,
        "last_keepalive": last_keepalive,
        "last_phone_frame": last_frame,
        "control_state": control_state,
        "live_preview": {
            "runtime_profile": (live_preview or {}).get("runtime_profile"),
            "crowd_upload_enabled": (live_preview or {}).get("crowd_upload_enabled"),
            "session_resolution": (live_preview or {}).get("session_resolution"),
            "target_human_state": (live_preview or {}).get("target_human_state"),
            "phone_capture": (live_preview or {}).get("phone_capture"),
            "vlm_summary": (live_preview or {}).get("vlm_summary"),
        },
        "errors": errors,
        "ok": keepalive_ok > 0 and frame_ok > 0 and not errors,
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report_path:
        Path(args.report_path).write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
