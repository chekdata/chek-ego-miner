#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from rtmlib import RTMPose


BODY_ORDER = [
    0,   # nose
    2,   # left eye
    5,   # right eye
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
]

BODY_POINT_COUNT = 17
HAND_POINT_COUNT = 42
AUX_BODY_CACHE_HOLD_NS = 2_000_000_000
AUX_HAND_CACHE_HOLD_NS = 1_200_000_000
PRIMARY_BODY_CACHE_HOLD_NS = 900_000_000
PRIMARY_HAND_CACHE_HOLD_NS = 2_000_000_000
EGO_HAND_ONLY_BODY_CACHE_HOLD_NS = 4_000_000_000
HAND_ROI_CACHE_HOLD_NS = 4_000_000_000
DEPTH_SEARCH_MAX_RADIUS_PX = 6
HAND_DEPTH_SEARCH_MAX_RADIUS_PX = 16
HAND_BODY_CONF_MIN = 4
DEBUG_FRAME_DUMP_DIR = Path("/tmp/edge_phone_vision_debug")
HAND_SEED_EARLY_ACCEPT_CONFIDENCE = 0.35
HAND_SEED_EARLY_ACCEPT_VALID = 12
BUSY_INFER_CARRY_HOLD_NS = 12_000_000_000
BUSY_INFER_WAIT_FOR_RESULT_MS = 8_000
BUSY_INFER_WAIT_STEP_MS = 120
RTMPOSE_HAND_FALLBACK_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


DEBUG_FRAME_DUMP_ENABLED = _env_flag("EDGE_PHONE_VISION_DEBUG_DUMP")


def _zeros_2d(count: int) -> List[List[float]]:
    return [[0.0, 0.0] for _ in range(count)]


def _zeros_3d(count: int) -> List[List[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(count)]


def _count_valid_2d(points: List[List[float]]) -> int:
    return sum(
        1
        for point in points
        if _valid_2d(point)
    )


def _count_valid_3d(points: List[List[float]]) -> int:
    return sum(
        1
        for point in points
        if len(point) >= 3
        and np.isfinite(point[0])
        and np.isfinite(point[1])
        and np.isfinite(point[2])
        and point[2] > 0.0
    )


def _depth_sum(points: List[List[float]]) -> float:
    total = 0.0
    for point in points:
        if (
            len(point) >= 3
            and np.isfinite(point[0])
            and np.isfinite(point[1])
            and np.isfinite(point[2])
            and point[2] > 0.0
        ):
            total += float(point[2])
    return total


def _valid_3d(point: List[float]) -> bool:
    return (
        len(point) >= 3
        and np.isfinite(point[0])
        and np.isfinite(point[1])
        and np.isfinite(point[2])
        and point[2] > 0.0
    )


def _valid_2d(point: List[float]) -> bool:
    return (
        len(point) >= 2
        and np.isfinite(point[0])
        and np.isfinite(point[1])
        and (abs(point[0]) > 1e-6 or abs(point[1]) > 1e-6)
    )


def _jpeg_b64_to_rgb(data_b64: str) -> np.ndarray:
    raw = base64.b64decode(data_b64.encode("utf-8"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("jpeg decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _prepare_rtmlib_checkpoint(model_path: Path) -> str:
    # rtmlib treats onnx_model as a downloadable checkpoint identifier. Use a
    # local file:// URL so it can resolve through its normal checkpoint loader
    # without relying on external network access.
    if not model_path.exists():
        return os.environ.get("EDGE_PHONE_VISION_RTMPOSE_HAND_URL", RTMPOSE_HAND_FALLBACK_URL)
    cache_dir = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / model_path.name
    source_stat = model_path.stat()
    if (not cache_path.exists()) or cache_path.stat().st_size != source_stat.st_size:
        shutil.copy2(model_path, cache_path)
    return model_path.resolve().as_uri()


def _rotate_rgb(rgb: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "rot90":
        return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "rot180":
        return cv2.rotate(rgb, cv2.ROTATE_180)
    if rotation == "rot270":
        return cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rgb


def _restore_points_to_original(
    points: List[List[float]],
    rotation: str,
    orig_w: int,
    orig_h: int,
) -> List[List[float]]:
    if rotation == "orig":
        return [list(point) for point in points]
    restored: List[List[float]] = _zeros_2d(len(points))
    for index, point in enumerate(points):
        if not _valid_2d(point):
            continue
        x = float(point[0])
        y = float(point[1])
        if rotation == "rot90":
            restored[index] = [y, float(orig_h) - x]
        elif rotation == "rot180":
            restored[index] = [float(orig_w) - x, float(orig_h) - y]
        elif rotation == "rot270":
            restored[index] = [float(orig_w) - y, x]
        else:
            restored[index] = [x, y]
    return restored


def _bbox_from_hand_seed(
    wrist: List[float],
    elbow: List[float],
    shoulder_width: float,
    image_w: int,
    image_h: int,
) -> Optional[List[float]]:
    if not _valid_2d(wrist):
        return None
    wx, wy = float(wrist[0]), float(wrist[1])
    if _valid_2d(elbow):
        ex, ey = float(elbow[0]), float(elbow[1])
        dx = wx - ex
        dy = wy - ey
        forearm = max(float(np.hypot(dx, dy)), 1.0)
        center_x = wx + dx * 0.35
        center_y = wy + dy * 0.35
        half = max(forearm * 1.2, shoulder_width * 0.28, 72.0)
    else:
        center_x = wx
        center_y = wy
        half = max(shoulder_width * 0.35, 96.0)

    x0 = max(0.0, center_x - half)
    y0 = max(0.0, center_y - half)
    x1 = min(float(image_w - 1), center_x + half)
    y1 = min(float(image_h - 1), center_y + half)
    if x1 - x0 < 48.0 or y1 - y0 < 48.0:
        return None
    return [x0, y0, x1, y1]


def _bbox_from_points(
    points: List[List[float]],
    image_w: int,
    image_h: int,
) -> Optional[List[float]]:
    valid = [point for point in points if _valid_2d(point)]
    if len(valid) < HAND_BODY_CONF_MIN:
        return None
    xs = [float(point[0]) for point in valid]
    ys = [float(point[1]) for point in valid]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    width = max_x - min_x
    height = max_y - min_y
    half = max(width, height) * 0.75
    half = max(half, 72.0)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    x0 = max(0.0, center_x - half)
    y0 = max(0.0, center_y - half)
    x1 = min(float(image_w - 1), center_x + half)
    y1 = min(float(image_h - 1), center_y + half)
    if x1 - x0 < 48.0 or y1 - y0 < 48.0:
        return None
    return [x0, y0, x1, y1]


def _depth_b64_to_array(data_b64: str, width: int, height: int) -> np.ndarray:
    raw = base64.b64decode(data_b64.encode("utf-8"))
    depth = np.frombuffer(raw, dtype="<f4")
    expected = int(width) * int(height)
    if depth.size != expected:
        raise ValueError(f"depth shape mismatch: got={depth.size} expected={expected}")
    return depth.reshape((int(height), int(width)))


def _make_warmup_payload() -> Dict[str, Any]:
    warmup_rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(warmup_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode warmup frame")
    return {
        "schema_version": "1.0.0",
        "trip_id": "",
        "session_id": "",
        "device_id": "warmup",
        "operator_track_id": "primary_operator",
        "source_time_ns": 0,
        "frame_id": 0,
        "camera_mode": "warmup",
        "image_w": 96,
        "image_h": 96,
        "sensor_image_w": 96,
        "sensor_image_h": 96,
        "normalized_was_rotated_right": False,
        "camera_has_depth": False,
        "camera_calibration": None,
        "device_pose": None,
        "imu": None,
        "primary_image_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("utf-8"),
        "aux_image_jpeg_b64": None,
        "depth_f32_b64": None,
        "depth_w": None,
        "depth_h": None,
    }


def _dump_debug_frame(
    *,
    kind: str,
    rgb: np.ndarray,
    payload: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> None:
    if not DEBUG_FRAME_DUMP_ENABLED:
        return
    try:
        DEBUG_FRAME_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_path = DEBUG_FRAME_DUMP_DIR / f"latest-{kind}.jpg"
        meta_path = DEBUG_FRAME_DUMP_DIR / f"latest-{kind}.json"
        cv2.imwrite(str(image_path), bgr)
        meta = {
            "kind": kind,
            "device_id": payload.get("device_id"),
            "session_id": payload.get("session_id"),
            "trip_id": payload.get("trip_id"),
            "frame_id": payload.get("frame_id"),
            "camera_mode": payload.get("camera_mode"),
            "source_time_ns": payload.get("source_time_ns"),
            "diagnostics": diagnostics,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _diagnostic_reason(
    *,
    depth_input_present: bool,
    calibration_present: bool,
    primary_body_valid_2d: int,
    primary_hand_valid_2d: int,
    raw_body_valid_3d: int,
    raw_hand_valid_3d: int,
    body_filled_from_primary: int,
    hand_filled_from_primary: int,
    body_filled_from_aux: int,
    hand_filled_from_aux: int,
    body_filled_from_hand_context: int,
) -> str:
    if primary_body_valid_2d == 0 and primary_hand_valid_2d == 0:
        return "vision_2d_empty"
    if not depth_input_present:
        return "depth_unavailable"
    if not calibration_present:
        return "camera_calibration_missing"
    if raw_body_valid_3d > 0 or raw_hand_valid_3d > 0:
        if body_filled_from_primary > 0 or hand_filled_from_primary > 0:
            return "vision_3d_partial_primary_fused"
        if body_filled_from_aux > 0 or hand_filled_from_aux > 0:
            return "vision_3d_partial_aux_fused"
        return "vision_3d_ok"
    if body_filled_from_primary > 0 or hand_filled_from_primary > 0:
        return "depth_sparse_primary_hold"
    if body_filled_from_aux > 0 or hand_filled_from_aux > 0:
        return "depth_sparse_aux_hold"
    if body_filled_from_hand_context > 0:
        return "ego_hand_only_body_hold"
    if primary_body_valid_2d == 0 and primary_hand_valid_2d > 0:
        return "ego_hand_only_visible"
    return "depth_reproject_missing"


class PhoneVisionRuntime:
    def __init__(self, pose_model_path: Path, rtmlib_hand_model_path: Path):
        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self._pose = vision.PoseLandmarker.create_from_options(pose_options)
        self._rtmpose_hand = RTMPose(
            onnx_model=_prepare_rtmlib_checkpoint(rtmlib_hand_model_path),
            model_input_size=(256, 256),
            device="cpu",
        )
        self._last_body_kpts3d: List[List[float]] = _zeros_3d(BODY_POINT_COUNT)
        self._last_hand_kpts3d: List[List[float]] = _zeros_3d(HAND_POINT_COUNT)
        self._last_body_time_ns = 0
        self._last_hand_time_ns = 0
        self._last_left_hand_kpts2d: List[List[float]] = _zeros_2d(21)
        self._last_right_hand_kpts2d: List[List[float]] = _zeros_2d(21)
        self._last_left_hand_time_ns = 0
        self._last_right_hand_time_ns = 0

    def _detect_best_orientation(
        self,
        rgb: np.ndarray,
        image_w: int,
        image_h: int,
        source_time_ns: int,
    ) -> Tuple[List[List[float]], float, List[List[float]], float, str]:
        rotations = ["orig", "rot90", "rot180", "rot270"]
        best_rotation = "orig"
        best_body = _zeros_2d(BODY_POINT_COUNT)
        best_body_conf = 0.0
        best_hand = _zeros_2d(HAND_POINT_COUNT)
        best_hand_conf = 0.0
        best_score = -1.0

        for rotation in rotations:
            rotated = _rotate_rgb(rgb, rotation)
            rotated_h, rotated_w = rotated.shape[:2]
            body_points, body_conf = self._detect_body(rotated, rotated_w, rotated_h)
            body_points = _restore_points_to_original(body_points, rotation, image_w, image_h)
            hand_left, hand_right, hand_conf = self._detect_hands(
                rgb,
                image_w,
                image_h,
                body_points,
                source_time_ns,
            )
            hand_points = hand_left + hand_right
            body_valid = _count_valid_2d(body_points)
            hand_valid = _count_valid_2d(hand_points)
            score = float(body_valid * 4 + hand_valid * 2) + body_conf + hand_conf
            if score <= best_score:
                continue
            best_score = score
            best_rotation = rotation
            best_body = body_points
            best_body_conf = body_conf
            best_hand = hand_points
            best_hand_conf = hand_conf

            # Fast path: keep the cheap common case cheap.
            if rotation == "orig" and (body_valid > 0 or hand_valid > 0):
                break

        return best_body, best_body_conf, best_hand, best_hand_conf, best_rotation

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        total_start = time.perf_counter()
        rgb = _jpeg_b64_to_rgb(payload["primary_image_jpeg_b64"])
        decode_ms = (time.perf_counter() - total_start) * 1000.0
        image_h, image_w = rgb.shape[:2]
        source_time_ns = int(payload["source_time_ns"])
        payload_image_w = int(payload.get("image_w", image_w))
        payload_image_h = int(payload.get("image_h", image_h))
        primary_detect_start = time.perf_counter()
        body_kpts2d, body_conf, hand_kpts2d, hand_conf, primary_rotation = self._detect_best_orientation(
            rgb, image_w, image_h, source_time_ns
        )
        primary_detect_ms = (time.perf_counter() - primary_detect_start) * 1000.0
        primary_body_valid_2d = _count_valid_2d(body_kpts2d)
        primary_hand_valid_2d = _count_valid_2d(hand_kpts2d)

        aux_body_valid_2d = 0
        aux_hand_valid_2d = 0
        aux_support_state = "none"
        aux_rotation = "orig"
        aux_rgb = None
        aux_body_conf = 0.0
        aux_hand_conf = 0.0
        aux_detect_ms = 0.0
        aux_b64 = payload.get("aux_image_jpeg_b64")
        if isinstance(aux_b64, str) and aux_b64.strip():
            aux_detect_start = time.perf_counter()
            aux_rgb = _jpeg_b64_to_rgb(aux_b64)
            aux_image_h, aux_image_w = aux_rgb.shape[:2]
            aux_body_kpts2d, aux_body_conf, aux_hand_kpts2d, aux_hand_conf, aux_rotation = self._detect_best_orientation(
                aux_rgb,
                int(aux_image_w),
                int(aux_image_h),
                source_time_ns,
            )
            aux_body_valid_2d = _count_valid_2d(aux_body_kpts2d)
            aux_hand_valid_2d = _count_valid_2d(aux_hand_kpts2d)
            if aux_body_valid_2d > 0 or aux_hand_valid_2d > 0:
                aux_support_state = "aux_detected_reference"
            else:
                aux_support_state = "present_no_detection"

            if primary_body_valid_2d == 0 and aux_body_valid_2d > 0:
                body_conf = max(body_conf, aux_body_conf * 0.6)
                aux_support_state = "aux_body_detected"
            if primary_hand_valid_2d == 0 and aux_hand_valid_2d > 0:
                hand_conf = max(hand_conf, aux_hand_conf * 0.6)
                aux_support_state = "aux_hand_detected"
            aux_detect_ms = (time.perf_counter() - aux_detect_start) * 1000.0

        body_kpts3d = _zeros_3d(len(body_kpts2d))
        hand_kpts3d = _zeros_3d(len(hand_kpts2d))
        body_valid_3d = 0
        hand_valid_3d = 0
        mean_depth = 0.0
        valid_ratio = 0.0
        body_depth_sum = 0.0
        hand_depth_sum = 0.0

        calibration = payload.get("camera_calibration")
        depth_b64 = payload.get("depth_f32_b64")
        depth_w = payload.get("depth_w")
        depth_h = payload.get("depth_h")
        depth_input_present = bool(calibration and depth_b64 and depth_w and depth_h)
        reproject_start = time.perf_counter()
        if calibration and depth_b64 and depth_w and depth_h:
            depth_map = _depth_b64_to_array(depth_b64, int(depth_w), int(depth_h))
            body_kpts3d, body_valid_3d, body_depth_sum = self._reproject_points(
                points2d=body_kpts2d,
                payload=payload,
                calibration=calibration,
                depth_map=depth_map,
            )
            hand_kpts3d, hand_valid_3d, hand_depth_sum = self._reproject_points(
                points2d=hand_kpts2d,
                payload=payload,
                calibration=calibration,
                depth_map=depth_map,
                search_radius_px=HAND_DEPTH_SEARCH_MAX_RADIUS_PX,
            )
        reproject_ms = (time.perf_counter() - reproject_start) * 1000.0
        body_filled_from_aux = 0
        hand_filled_from_aux = 0
        body_filled_from_primary = 0
        hand_filled_from_primary = 0
        body_filled_from_hand_context = 0
        if aux_rgb is not None:
            body_filled_from_aux = self._fill_from_recent_cache(
                target_points=body_kpts3d,
                cached_points=self._last_body_kpts3d,
                current_time_ns=source_time_ns,
                cached_time_ns=self._last_body_time_ns,
                hold_ns=AUX_BODY_CACHE_HOLD_NS,
                aux_detected_valid=aux_body_valid_2d > 0,
            )
            hand_filled_from_aux = self._fill_from_recent_cache(
                target_points=hand_kpts3d,
                cached_points=self._last_hand_kpts3d,
                current_time_ns=source_time_ns,
                cached_time_ns=self._last_hand_time_ns,
                hold_ns=AUX_HAND_CACHE_HOLD_NS,
                aux_detected_valid=aux_hand_valid_2d > 0,
            )

        raw_body_valid_3d = body_valid_3d
        raw_hand_valid_3d = hand_valid_3d
        if body_valid_3d < primary_body_valid_2d and primary_body_valid_2d > 0:
            body_filled_from_primary = self._fill_from_recent_cache(
                target_points=body_kpts3d,
                cached_points=self._last_body_kpts3d,
                current_time_ns=source_time_ns,
                cached_time_ns=self._last_body_time_ns,
                hold_ns=PRIMARY_BODY_CACHE_HOLD_NS,
                aux_detected_valid=True,
            )
        if hand_valid_3d < primary_hand_valid_2d and primary_hand_valid_2d > 0:
            hand_filled_from_primary = self._fill_from_recent_cache(
                target_points=hand_kpts3d,
                cached_points=self._last_hand_kpts3d,
                current_time_ns=source_time_ns,
                cached_time_ns=self._last_hand_time_ns,
                hold_ns=PRIMARY_HAND_CACHE_HOLD_NS,
                aux_detected_valid=True,
            )
        if (
            body_valid_3d == 0
            and primary_body_valid_2d == 0
            and primary_hand_valid_2d > 0
        ):
            body_filled_from_hand_context = self._fill_from_recent_cache(
                target_points=body_kpts3d,
                cached_points=self._last_body_kpts3d,
                current_time_ns=source_time_ns,
                cached_time_ns=self._last_body_time_ns,
                hold_ns=EGO_HAND_ONLY_BODY_CACHE_HOLD_NS,
                aux_detected_valid=True,
            )

        body_valid_3d = _count_valid_3d(body_kpts3d)
        hand_valid_3d = _count_valid_3d(hand_kpts3d)
        body_depth_sum = _depth_sum(body_kpts3d)
        hand_depth_sum = _depth_sum(hand_kpts3d)
        total_points = len(body_kpts2d) + len(hand_kpts2d)
        total_valid = body_valid_3d + hand_valid_3d
        if total_valid > 0:
            mean_depth = float((body_depth_sum + hand_depth_sum) / total_valid)
            valid_ratio = float(total_valid / max(total_points, 1))

        if body_valid_3d > 0:
            self._last_body_kpts3d = [list(point) for point in body_kpts3d]
            self._last_body_time_ns = source_time_ns
        if hand_valid_3d > 0:
            if len(self._last_hand_kpts3d) != len(hand_kpts3d):
                self._last_hand_kpts3d = _zeros_3d(len(hand_kpts3d))
            merged_hand_kpts3d = [list(point) for point in self._last_hand_kpts3d]
            for index, point in enumerate(hand_kpts3d):
                if not _valid_3d(point):
                    continue
                merged_hand_kpts3d[index] = list(point)
            self._last_hand_kpts3d = merged_hand_kpts3d
            self._last_hand_time_ns = source_time_ns
        if len(hand_kpts2d) >= 42:
            left_hand_2d = [list(point) for point in hand_kpts2d[:21]]
            right_hand_2d = [list(point) for point in hand_kpts2d[21:42]]
            if _count_valid_2d(left_hand_2d) >= HAND_BODY_CONF_MIN:
                self._last_left_hand_kpts2d = left_hand_2d
                self._last_left_hand_time_ns = source_time_ns
            if _count_valid_2d(right_hand_2d) >= HAND_BODY_CONF_MIN:
                self._last_right_hand_kpts2d = right_hand_2d
                self._last_right_hand_time_ns = source_time_ns

        body_3d_source = "edge_depth_reprojected" if body_valid_3d > 0 else "none"
        hand_3d_source = "edge_depth_reprojected" if hand_valid_3d > 0 else "none"
        if body_filled_from_aux > 0:
            body_3d_source = "edge_depth_reprojected_aux_supported"
        elif body_filled_from_hand_context > 0:
            body_3d_source = "edge_depth_reprojected_hand_context_hold"
        elif body_filled_from_primary > 0 and raw_body_valid_3d > 0:
            body_3d_source = "edge_depth_reprojected_primary_fused"
        elif body_filled_from_primary > 0:
            body_3d_source = "edge_depth_reprojected_primary_hold"
        if hand_filled_from_aux > 0:
            hand_3d_source = "edge_depth_reprojected_aux_supported"
        elif hand_filled_from_primary > 0 and raw_hand_valid_3d > 0:
            hand_3d_source = "edge_depth_reprojected_primary_fused"
        elif hand_filled_from_primary > 0:
            hand_3d_source = "edge_depth_reprojected_primary_hold"
        if aux_rgb is not None and aux_support_state == "aux_detected_reference":
            if body_filled_from_aux > 0 or hand_filled_from_aux > 0:
                aux_support_state = "aux_reference_fused"

        diagnostic_reason = _diagnostic_reason(
            depth_input_present=depth_input_present,
            calibration_present=calibration is not None,
            primary_body_valid_2d=primary_body_valid_2d,
            primary_hand_valid_2d=primary_hand_valid_2d,
            raw_body_valid_3d=raw_body_valid_3d,
            raw_hand_valid_3d=raw_hand_valid_3d,
            body_filled_from_primary=body_filled_from_primary,
            hand_filled_from_primary=hand_filled_from_primary,
            body_filled_from_aux=body_filled_from_aux,
            hand_filled_from_aux=hand_filled_from_aux,
            body_filled_from_hand_context=body_filled_from_hand_context,
        )

        packet = {
            "type": "capture_pose_packet",
            "schema_version": payload.get("schema_version", "1.0.0"),
            "trip_id": payload.get("trip_id", ""),
            "session_id": payload.get("session_id", ""),
            "device_id": payload.get("device_id", "unknown"),
            "device_class": "B",
            "platform": "ios",
            "operator_track_id": payload.get("operator_track_id") or "primary_operator",
            "source_time_ns": source_time_ns,
            "frame_id": int(payload["frame_id"]),
            "camera": {
                "mode": payload.get("camera_mode", "teleop_phone_edge_authoritative"),
                "has_depth": bool(payload.get("camera_has_depth", False)),
                "image_w": image_w,
                "image_h": image_h,
                "calibration": calibration,
            },
            "body_layout": "coco_body_17" if _count_valid_2d(body_kpts2d) > 0 else None,
            "hand_layout": "mediapipe_hand_21",
            "body_kpts_2d": body_kpts2d,
            "hand_kpts_2d": hand_kpts2d,
            "body_kpts_3d": body_kpts3d if body_valid_3d > 0 else None,
            "hand_kpts_3d": hand_kpts3d if hand_valid_3d > 0 else None,
            "depth_summary": {
                "valid_ratio": valid_ratio,
                "z_mean_m": mean_depth,
            } if body_valid_3d > 0 or hand_valid_3d > 0 else None,
            "capture_profile": {
                "body_3d_source": body_3d_source,
                "hand_3d_source": hand_3d_source,
                "diagnostic_reason": diagnostic_reason,
                "decoded_image_w": image_w,
                "decoded_image_h": image_h,
                "payload_image_w": payload_image_w,
                "payload_image_h": payload_image_h,
                "primary_image_size_mismatch": (
                    payload_image_w != image_w or payload_image_h != image_h
                ),
                "primary_body_points_2d_valid": primary_body_valid_2d if primary_body_valid_2d > 0 else None,
                "primary_hand_points_2d_valid": primary_hand_valid_2d if primary_hand_valid_2d > 0 else None,
                "raw_body_points_3d_valid": raw_body_valid_3d if raw_body_valid_3d > 0 else None,
                "raw_hand_points_3d_valid": raw_hand_valid_3d if raw_hand_valid_3d > 0 else None,
                "body_points_3d_valid": body_valid_3d if body_valid_3d > 0 else None,
                "hand_points_3d_valid": hand_valid_3d if hand_valid_3d > 0 else None,
                "execution_mode": "edge_authoritative_phone_vision",
                "aux_snapshot_present": aux_rgb is not None,
                "depth_input_present": depth_input_present,
                "aux_body_points_2d_valid": aux_body_valid_2d,
                "aux_hand_points_2d_valid": aux_hand_valid_2d,
                "primary_body_points_3d_filled": body_filled_from_primary if body_filled_from_primary > 0 else None,
                "primary_hand_points_3d_filled": hand_filled_from_primary if hand_filled_from_primary > 0 else None,
                "aux_body_points_3d_filled": body_filled_from_aux if body_filled_from_aux > 0 else None,
                "aux_hand_points_3d_filled": hand_filled_from_aux if hand_filled_from_aux > 0 else None,
                "hand_context_body_points_3d_filled": body_filled_from_hand_context if body_filled_from_hand_context > 0 else None,
                "aux_support_state": aux_support_state,
                "primary_detection_rotation": primary_rotation,
                "aux_detection_rotation": aux_rotation,
            },
            "confidence": {
                "body": body_conf,
                "hand": hand_conf,
            },
        }
        if payload.get("device_pose") is not None:
            packet["device_pose"] = payload.get("device_pose")
        if payload.get("imu") is not None:
            packet["imu"] = payload.get("imu")
        diagnostics = {
            "body_points_2d_valid": primary_body_valid_2d,
            "hand_points_2d_valid": primary_hand_valid_2d,
            "body_points_3d_valid": body_valid_3d,
            "hand_points_3d_valid": hand_valid_3d,
            "aux_snapshot_present": aux_rgb is not None,
            "aux_body_points_2d_valid": aux_body_valid_2d,
            "aux_hand_points_2d_valid": aux_hand_valid_2d,
            "primary_body_points_3d_filled": body_filled_from_primary,
            "primary_hand_points_3d_filled": hand_filled_from_primary,
            "aux_support_state": aux_support_state,
            "aux_body_points_3d_filled": body_filled_from_aux,
            "aux_hand_points_3d_filled": hand_filled_from_aux,
            "hand_context_body_points_3d_filled": body_filled_from_hand_context,
            "diagnostic_reason": diagnostic_reason,
            "depth_input_present": depth_input_present,
            "primary_detection_rotation": primary_rotation,
            "aux_detection_rotation": aux_rotation,
            "decoded_image_w": image_w,
            "decoded_image_h": image_h,
            "payload_image_w": payload_image_w,
            "payload_image_h": payload_image_h,
            "primary_image_size_mismatch": payload_image_w != image_w or payload_image_h != image_h,
            "mode": "edge_authoritative_phone_vision",
        }
        dump_kind = "zero"
        if body_valid_3d > 0 or hand_valid_3d > 0 or primary_body_valid_2d > 0 or primary_hand_valid_2d > 0:
            dump_kind = "hit"
        debug_dump_start = time.perf_counter()
        _dump_debug_frame(
            kind=dump_kind,
            rgb=rgb,
            payload=payload,
            diagnostics=diagnostics,
        )
        debug_dump_ms = (time.perf_counter() - debug_dump_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        diagnostics["timing_ms"] = {
            "decode": round(decode_ms, 2),
            "primary_detect": round(primary_detect_ms, 2),
            "aux_detect": round(aux_detect_ms, 2),
            "reproject": round(reproject_ms, 2),
            "debug_dump": round(debug_dump_ms, 2),
            "total": round(total_ms, 2),
        }
        return {"ok": True, "capture_pose_packet": packet, "diagnostics": diagnostics}

    def _fill_from_recent_cache(
        self,
        *,
        target_points: List[List[float]],
        cached_points: List[List[float]],
        current_time_ns: int,
        cached_time_ns: int,
        hold_ns: int,
        aux_detected_valid: bool,
    ) -> int:
        if not aux_detected_valid or cached_time_ns <= 0:
            return 0
        if current_time_ns <= cached_time_ns:
            age_ns = 0
        else:
            age_ns = current_time_ns - cached_time_ns
        if age_ns > hold_ns:
            return 0
        filled = 0
        for index in range(min(len(target_points), len(cached_points))):
            if _valid_3d(target_points[index]):
                continue
            if not _valid_3d(cached_points[index]):
                continue
            target_points[index] = list(cached_points[index])
            filled += 1
        return filled

    def _detect_body(self, rgb: np.ndarray, image_w: int, image_h: int) -> Tuple[List[List[float]], float]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._pose.detect(mp_image)
        if not result.pose_landmarks:
            return _zeros_2d(17), 0.0
        landmarks = result.pose_landmarks[0]
        points: List[List[float]] = _zeros_2d(17)
        confidences: List[float] = []
        for idx, pose_index in enumerate(BODY_ORDER):
            lm = landmarks[pose_index]
            x = float(lm.x) * image_w
            y = float(lm.y) * image_h
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            points[idx] = [x, y]
            presence = float(max(0.0, min(1.0, getattr(lm, "presence", 0.0))))
            visibility = float(max(0.0, min(1.0, getattr(lm, "visibility", 0.0))))
            confidence = max(visibility, presence)
            if confidence <= 0.0:
                confidence = 1.0
            confidences.append(confidence)
        confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return points, confidence

    def _detect_hands(
        self,
        rgb: np.ndarray,
        image_w: int,
        image_h: int,
        body_points_2d: List[List[float]],
        source_time_ns: int,
    ) -> Tuple[List[List[float]], List[List[float]], float]:
        left = _zeros_2d(21)
        right = _zeros_2d(21)
        left_conf = 0.0
        right_conf = 0.0

        left_shoulder = body_points_2d[5] if len(body_points_2d) > 5 else [0.0, 0.0]
        right_shoulder = body_points_2d[6] if len(body_points_2d) > 6 else [0.0, 0.0]
        shoulder_width = 0.0
        if _valid_2d(left_shoulder) and _valid_2d(right_shoulder):
            shoulder_width = float(
                np.hypot(
                    float(left_shoulder[0]) - float(right_shoulder[0]),
                    float(left_shoulder[1]) - float(right_shoulder[1]),
                )
            )
        shoulder_width = max(shoulder_width, float(min(image_w, image_h)) * 0.18)

        seeds = [
            ("left", body_points_2d[9] if len(body_points_2d) > 9 else [0.0, 0.0], body_points_2d[7] if len(body_points_2d) > 7 else [0.0, 0.0]),
            ("right", body_points_2d[10] if len(body_points_2d) > 10 else [0.0, 0.0], body_points_2d[8] if len(body_points_2d) > 8 else [0.0, 0.0]),
        ]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for side, wrist, elbow in seeds:
            candidate_bboxes: List[Tuple[str, List[float]]] = []
            bbox = _bbox_from_hand_seed(
                wrist=wrist,
                elbow=elbow,
                shoulder_width=shoulder_width,
                image_w=image_w,
                image_h=image_h,
            )
            if bbox is not None:
                candidate_bboxes.append(("seed", bbox))
            cached_points = (
                self._last_left_hand_kpts2d if side == "left" else self._last_right_hand_kpts2d
            )
            cached_time_ns = (
                self._last_left_hand_time_ns if side == "left" else self._last_right_hand_time_ns
            )
            if (
                cached_time_ns > 0
                and source_time_ns >= cached_time_ns
                and source_time_ns - cached_time_ns <= HAND_ROI_CACHE_HOLD_NS
            ):
                cached_bbox = _bbox_from_points(
                    cached_points,
                    image_w=image_w,
                    image_h=image_h,
                )
                if cached_bbox is not None:
                    candidate_bboxes.append(("cache", cached_bbox))

            best_points = _zeros_2d(21)
            best_confidence = 0.0
            best_valid = 0
            seen_bbox_keys = set()
            for candidate_source, candidate_bbox in candidate_bboxes:
                bbox_key = tuple(round(value, 1) for value in candidate_bbox)
                if bbox_key in seen_bbox_keys:
                    continue
                seen_bbox_keys.add(bbox_key)
                try:
                    keypoints, scores = self._rtmpose_hand(bgr, bboxes=[candidate_bbox])
                except Exception:
                    continue
                keypoints = np.asarray(keypoints, dtype=np.float32)
                scores = np.asarray(scores, dtype=np.float32)
                if keypoints.ndim != 3 or keypoints.shape[0] == 0:
                    continue
                if scores.ndim != 2 or scores.shape[0] == 0:
                    continue
                points = _zeros_2d(21)
                confidence = float(np.mean(scores[0])) if scores.shape[1] > 0 else 0.0
                valid = 0
                for point_idx, point in enumerate(keypoints[0][:21]):
                    x = float(point[0])
                    y = float(point[1])
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    points[point_idx] = [x, y]
                    valid += 1
                if confidence < 0.20 or valid < HAND_BODY_CONF_MIN:
                    continue
                candidate_score = confidence + float(valid) * 0.02
                best_score = best_confidence + float(best_valid) * 0.02
                if candidate_score <= best_score:
                    continue
                best_points = points
                best_confidence = confidence
                best_valid = valid
                if (
                    candidate_source == "seed"
                    and valid >= HAND_SEED_EARLY_ACCEPT_VALID
                    and confidence >= HAND_SEED_EARLY_ACCEPT_CONFIDENCE
                ):
                    break

            if best_valid < HAND_BODY_CONF_MIN:
                continue
            if side == "left":
                left = best_points
                left_conf = best_confidence
            else:
                right = best_points
                right_conf = best_confidence
        valid_conf = [value for value in [left_conf, right_conf] if value > 0]
        hand_conf = float(sum(valid_conf) / len(valid_conf)) if valid_conf else 0.0
        return left, right, hand_conf

    def _reproject_points(
        self,
        points2d: List[List[float]],
        payload: Dict[str, Any],
        calibration: Dict[str, Any],
        depth_map: np.ndarray,
        *,
        search_radius_px: int = DEPTH_SEARCH_MAX_RADIUS_PX,
    ) -> Tuple[List[List[float]], int, float]:
        points3d = _zeros_3d(len(points2d))
        valid_count = 0
        depth_sum = 0.0
        for index, point in enumerate(points2d):
            if not _valid_2d(point):
                continue
            ref_point = _normalized_point_to_reference_pixel(
                point=point,
                normalized_image_w=int(payload["image_w"]),
                normalized_image_h=int(payload["image_h"]),
                sensor_image_w=int(payload.get("sensor_image_w", payload["image_w"])),
                sensor_image_h=int(payload.get("sensor_image_h", payload["image_h"])),
                normalized_was_rotated_right=bool(payload.get("normalized_was_rotated_right", False)),
                ref_w=float(calibration["reference_image_w"]),
                ref_h=float(calibration["reference_image_h"]),
            )
            if ref_point is None:
                continue
            depth = _sample_depth(
                ref_x=ref_point[0],
                ref_y=ref_point[1],
                ref_w=float(calibration["reference_image_w"]),
                ref_h=float(calibration["reference_image_h"]),
                depth_map=depth_map,
                search_radius_px=search_radius_px,
            )
            if depth is None or not np.isfinite(depth) or depth <= 0.05:
                continue
            fx = float(calibration["fx_px"])
            fy = float(calibration["fy_px"])
            cx = float(calibration["cx_px"])
            cy = float(calibration["cy_px"])
            x = ((ref_point[0] - cx) / max(fx, 1e-6)) * depth
            y = ((cy - ref_point[1]) / max(fy, 1e-6)) * depth
            points3d[index] = [float(x), float(y), float(depth)]
            valid_count += 1
            depth_sum += float(depth)
        return points3d, valid_count, depth_sum


def _normalized_point_to_reference_pixel(
    point: List[float],
    normalized_image_w: int,
    normalized_image_h: int,
    sensor_image_w: int,
    sensor_image_h: int,
    normalized_was_rotated_right: bool,
    ref_w: float,
    ref_h: float,
) -> Optional[Tuple[float, float]]:
    if not _valid_2d(point):
        return None
    x = float(point[0])
    y = float(point[1])
    if normalized_was_rotated_right:
        sensor_x = y * float(sensor_image_w) / max(float(normalized_image_h), 1.0)
        sensor_y = float(sensor_image_h) - (
            x * float(sensor_image_h) / max(float(normalized_image_w), 1.0)
        )
    else:
        sensor_x = x * float(sensor_image_w) / max(float(normalized_image_w), 1.0)
        sensor_y = y * float(sensor_image_h) / max(float(normalized_image_h), 1.0)
    ref_x = (sensor_x / max(float(sensor_image_w), 1.0)) * ref_w
    ref_y = (sensor_y / max(float(sensor_image_h), 1.0)) * ref_h
    return ref_x, ref_y


def _sample_depth(
    ref_x: float,
    ref_y: float,
    ref_w: float,
    ref_h: float,
    depth_map: np.ndarray,
    *,
    search_radius_px: int = DEPTH_SEARCH_MAX_RADIUS_PX,
) -> Optional[float]:
    depth_h, depth_w = depth_map.shape
    px = int((ref_x / max(ref_w, 1.0)) * depth_w)
    py = int((ref_y / max(ref_h, 1.0)) * depth_h)
    for radius in range(1, max(int(search_radius_px), 1) + 1):
        values: List[float] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sx = min(max(px + dx, 0), depth_w - 1)
                sy = min(max(py + dy, 0), depth_h - 1)
                value = float(depth_map[sy, sx])
                if np.isfinite(value) and value > 0:
                    values.append(value)
        if values:
            values.sort()
            return values[len(values) // 2]
    return None


class PhoneVisionHandler(BaseHTTPRequestHandler):
    runtime: PhoneVisionRuntime | None = None
    infer_lock = threading.Lock()
    cache_lock = threading.RLock()
    infer_condition = threading.Condition(cache_lock)
    last_response_key: Optional[Tuple[str, int, int]] = None
    last_response_payload: Optional[Dict[str, Any]] = None

    @classmethod
    def _request_key(cls, payload: Dict[str, Any]) -> Tuple[str, int, int]:
        return (
            str(payload.get("device_id") or ""),
            int(payload.get("frame_id") or 0),
            int(payload.get("source_time_ns") or 0),
        )

    @classmethod
    def _cached_response_for(cls, request_key: Tuple[str, int, int]) -> Optional[Dict[str, Any]]:
        with cls.cache_lock:
            if cls.last_response_key != request_key or cls.last_response_payload is None:
                return None
            return json.loads(json.dumps(cls.last_response_payload, ensure_ascii=False))

    @classmethod
    def _store_cached_response(
        cls,
        request_key: Tuple[str, int, int],
        payload: Dict[str, Any],
    ) -> None:
        with cls.cache_lock:
            cls.last_response_key = request_key
            cls.last_response_payload = json.loads(json.dumps(payload, ensure_ascii=False))
            cls.infer_condition.notify_all()

    @classmethod
    def _busy_carry_response_for(cls, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now_source_time_ns = int(payload.get("source_time_ns") or 0)
        with cls.cache_lock:
            if cls.last_response_payload is None:
                return None
            carried = json.loads(json.dumps(cls.last_response_payload, ensure_ascii=False))
        packet = carried.get("capture_pose_packet")
        if not isinstance(packet, dict):
            return None
        last_source_time_ns = int(packet.get("source_time_ns") or 0)
        last_frame_id = int(packet.get("frame_id") or 0)
        current_trip_id = str(payload.get("trip_id") or "")
        current_session_id = str(payload.get("session_id") or "")
        current_device_id = str(payload.get("device_id") or "")
        last_trip_id = str(packet.get("trip_id") or "")
        last_session_id = str(packet.get("session_id") or "")
        last_device_id = str(packet.get("device_id") or "")
        if now_source_time_ns <= 0 or last_source_time_ns <= 0:
            return None
        if current_session_id and last_session_id and current_session_id != last_session_id:
            return None
        if current_trip_id and last_trip_id and current_trip_id != last_trip_id:
            return None
        if current_device_id and last_device_id and current_device_id != last_device_id:
            return None
        out_of_order_request = now_source_time_ns < last_source_time_ns
        if abs(now_source_time_ns - last_source_time_ns) > BUSY_INFER_CARRY_HOLD_NS:
            return None

        packet["trip_id"] = payload.get("trip_id", packet.get("trip_id", ""))
        packet["session_id"] = payload.get("session_id", packet.get("session_id", ""))
        packet["device_id"] = payload.get("device_id", packet.get("device_id", "unknown"))
        packet["operator_track_id"] = payload.get("operator_track_id") or packet.get("operator_track_id") or "primary_operator"
        if not out_of_order_request:
            packet["source_time_ns"] = now_source_time_ns
            packet["frame_id"] = int(payload.get("frame_id") or packet.get("frame_id") or 0)
        camera = packet.get("camera")
        if isinstance(camera, dict):
            camera["mode"] = payload.get("camera_mode", camera.get("mode", "teleop_phone_edge_authoritative"))
            camera["has_depth"] = bool(payload.get("camera_has_depth", camera.get("has_depth", False)))
        if not out_of_order_request and payload.get("device_pose") is not None:
            packet["device_pose"] = payload.get("device_pose")
        if not out_of_order_request and payload.get("imu") is not None:
            packet["imu"] = payload.get("imu")

        diagnostics = carried.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            carried["diagnostics"] = diagnostics
        diagnostics["busy"] = True
        diagnostics["busy_infer_carried"] = True
        diagnostics["busy_infer_last_source_time_ns"] = last_source_time_ns
        diagnostics["busy_infer_last_frame_id"] = last_frame_id
        diagnostics["busy_infer_out_of_order"] = out_of_order_request
        diagnostics["busy_infer_request_source_time_ns"] = now_source_time_ns
        diagnostics["busy_infer_request_frame_id"] = int(payload.get("frame_id") or 0)
        diagnostics["carried_frame_id"] = packet.get("frame_id")
        diagnostics["mode"] = "edge_authoritative_phone_vision"
        carried["ok"] = True
        return carried

    @classmethod
    def _wait_for_busy_result(
        cls,
        request_key: Tuple[str, int, int],
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + (BUSY_INFER_WAIT_FOR_RESULT_MS / 1000.0)
        with cls.infer_condition:
            while time.monotonic() < deadline:
                cached = cls._cached_response_for(request_key)
                if cached is not None:
                    return cached
                carried = cls._busy_carry_response_for(payload)
                if carried is not None:
                    return carried
                if not cls.infer_lock.locked():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                cls.infer_condition.wait(timeout=min(remaining, BUSY_INFER_WAIT_STEP_MS / 1000.0))
            cached = cls._cached_response_for(request_key)
            if cached is not None:
                return cached
            carried = cls._busy_carry_response_for(payload)
            if carried is not None:
                return carried
        return None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"ok": True, "service": "edge_phone_vision_service"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": {"code": "not_found", "message": self.path}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": {"code": "not_found", "message": self.path}})
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be object")
            request_key = self._request_key(payload)
            cached = self._cached_response_for(request_key)
            if cached is not None:
                self._write_json(HTTPStatus.OK, cached)
                return
            assert self.runtime is not None
            if not self.infer_lock.acquire(blocking=False):
                waited = self._wait_for_busy_result(request_key, payload)
                if waited is not None:
                    self._write_json(HTTPStatus.OK, waited)
                    return
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "ok": False,
                        "error": {
                            "code": "busy_infer",
                            "message": "phone vision infer busy; retry latest frame shortly",
                        },
                        "diagnostics": {
                            "mode": "edge_authoritative_phone_vision",
                            "busy": True,
                            "frame_id": payload.get("frame_id"),
                            "device_id": payload.get("device_id"),
                        },
                    },
                )
                return
            try:
                cached = self._cached_response_for(request_key)
                if cached is not None:
                    self._write_json(HTTPStatus.OK, cached)
                    return
                response = self.runtime.infer(payload)
                self._store_cached_response(request_key, response)
            finally:
                self.infer_lock.release()
                with self.cache_lock:
                    self.infer_condition.notify_all()
            diagnostics = response.get("diagnostics", {}) if isinstance(response, dict) else {}
            print(
                json.dumps(
                    {
                        "ok": True,
                        "event": "infer",
                        "device_id": payload.get("device_id"),
                        "frame_id": payload.get("frame_id"),
                        "camera_mode": payload.get("camera_mode"),
                        "has_depth": bool(payload.get("camera_has_depth", False)),
                        "aux_snapshot_present": diagnostics.get("aux_snapshot_present", False),
                        "aux_body_points_2d_valid": diagnostics.get("aux_body_points_2d_valid", 0),
                        "aux_hand_points_2d_valid": diagnostics.get("aux_hand_points_2d_valid", 0),
                        "aux_support_state": diagnostics.get("aux_support_state", "none"),
                        "body_points_2d_valid": diagnostics.get("body_points_2d_valid"),
                        "hand_points_2d_valid": diagnostics.get("hand_points_2d_valid"),
                        "body_points_3d_valid": diagnostics.get("body_points_3d_valid"),
                        "hand_points_3d_valid": diagnostics.get("hand_points_3d_valid"),
                        "timing_ms": diagnostics.get("timing_ms", {}),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            self._write_json(HTTPStatus.OK, response)
        except Exception as error:  # noqa: BLE001
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "ok": False,
                    "error": {"code": "infer_failed", "message": str(error)},
                },
            )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _write_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            return False
        return True


def _parse_bind(default_base: str) -> Tuple[str, int]:
    parsed = urlparse(default_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3031
    return host, port


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--rtmpose-hand-model",
        default=str(Path(__file__).resolve().parent.parent / "model-candidates" / "rtmlib" / "rtmpose-m-hand.onnx"),
    )
    parser.add_argument(
        "--pose-model",
        default=str(Path(__file__).resolve().parent.parent / "model-candidates" / "mediapipe" / "pose_landmarker_heavy.task"),
    )
    args = parser.parse_args()

    base = os.environ.get("EDGE_PHONE_VISION_SERVICE_BASE", "http://127.0.0.1:3031")
    host, port = _parse_bind(base)
    if args.host:
        host = args.host
    if args.port:
        port = args.port

    runtime = PhoneVisionRuntime(
        pose_model_path=Path(args.pose_model),
        rtmlib_hand_model_path=Path(args.rtmpose_hand_model),
    )
    try:
        runtime.infer(_make_warmup_payload())
        print(json.dumps({"ok": True, "event": "warmup_ready"}, ensure_ascii=False), flush=True)
    except Exception as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "ok": False,
                    "event": "warmup_failed",
                    "error": str(error),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    PhoneVisionHandler.runtime = runtime
    server = ThreadingHTTPServer((host, port), PhoneVisionHandler)
    print(json.dumps({"ok": True, "host": host, "port": port, "service": "edge_phone_vision_service"}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
