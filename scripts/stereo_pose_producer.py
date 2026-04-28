#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import requests

try:
    from rtmlib import PoseTracker, Wholebody
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 rtmlib。先执行：python3 -m pip install --user rtmlib onnxruntime"
    ) from exc


COCO_BODY_17 = 17
TORSO_INDICES = (5, 6, 11, 12)
WRIST_INDICES = (9, 10)
IPHONE_HINT_PRIORITY_MAX_GAP_M = 0.85
STEREO_CONTINUITY_PRIORITY_MAX_GAP_M = 0.65
STEREO_BODY_CENTER_CONTINUITY_MAX_M = 0.9
STEREO_SELECTED_TRACK_HOLD_MAX_M = 1.6
STEREO_TRACK_REUSE_MAX_M = 1.25
STEREO_TRACK_CACHE_MAX = 12
STEREO_TRACK_RECENT_PREFERENCE_WINDOW = 8
STEREO_TRACK_RECENT_REUSE_MAX_M = 0.75
STEREO_DEPTH_OUTLIER_MIN_DELTA_M = 0.75
STEREO_DEPTH_OUTLIER_RATIO = 0.3
STEREO_TRACK_PERSISTENCE_WINDOW = 4
STEREO_TRACK_PERSISTENCE_DECAY = 0.88
STEREO_TRACK_PERSISTENCE_MIN_SEPARATION_M = 0.28
LOW_ROI_TRACK_STABILIZE_WINDOW = 5
LOW_ROI_TRACK_STABILIZE_MAX_GAP_M = 0.75
LOW_ROI_TRACK_STABILIZE_MIN_GAP_M = 0.06

BodyPoint3D = list[float] | None


@dataclass
class CameraCalibration:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    reference_image_w: int
    reference_image_h: int


@dataclass
class StereoCalibration:
    left: CameraCalibration
    right: CameraCalibration
    baseline_m: float
    sensor_frame: str
    operator_frame: str
    extrinsic_version: str


@dataclass
class PersonCandidate:
    keypoints: np.ndarray
    scores: np.ndarray
    bbox_height_px: float
    center_xy: tuple[float, float]
    mean_score: float
    source_tag: str = "full_frame"
    joint_score_threshold_override: float | None = None


@dataclass
class AssociationHint:
    selected_operator_track_id: str | None
    device_pose_candidate_track_id: str | None
    iphone_left_wrist: tuple[float, float, float] | None
    iphone_right_wrist: tuple[float, float, float] | None
    stereo_left_wrist: tuple[float, float, float] | None
    stereo_right_wrist: tuple[float, float, float] | None


@dataclass
class SelectedStereoCandidate:
    left_index: int
    right_index: int
    left_candidate: PersonCandidate
    right_candidate: PersonCandidate
    body_3d: list[BodyPoint3D]
    body_3d_valid: list[bool]
    triangulated_ratio: float
    stereo_confidence: float
    operator_track_id: str
    selection_reason: str
    hand_hint_gap_m: float | None
    continuity_gap_m: float | None


@dataclass
class StereoTrackMemory:
    next_track_index: int = 1
    current_track_id: str | None = None
    last_body_center: tuple[float, float, float] | None = None
    last_left_wrist: tuple[float, float, float] | None = None
    last_right_wrist: tuple[float, float, float] | None = None
    track_centers: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    track_prev_centers: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    track_center_velocity: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    track_body_signatures: dict[str, tuple[float, float]] = field(default_factory=dict)
    track_seen_seq: dict[str, int] = field(default_factory=dict)
    track_candidates: dict[str, SelectedStereoCandidate] = field(default_factory=dict)
    seen_seq: int = 0

    def _local_continuity_gap_m(self, body_3d: list[BodyPoint3D]) -> float | None:
        center = body_center_point(body_3d)
        if center is not None and self.last_body_center is not None:
            return math.dist(center, self.last_body_center)
        return candidate_hand_gap_m(body_3d, self.last_left_wrist, self.last_right_wrist)

    @staticmethod
    def _is_stereo_track_id(track_id: str | None) -> bool:
        return bool(track_id) and track_id.startswith("stereo-person-")

    def _nearest_cached_track_id(
        self,
        body_3d: list[BodyPoint3D],
        reserved_track_ids: set[str] | None = None,
    ) -> str | None:
        center = body_center_point(body_3d)
        current_signature = body_signature(body_3d)
        if center is None or not self.track_centers:
            return None
        recent_candidates: list[tuple[int, float, str]] = []
        best_track_id: str | None = None
        best_key: tuple[float, float, int] | None = None
        for track_id, cached_center in self.track_centers.items():
            if reserved_track_ids and track_id in reserved_track_ids:
                continue
            seen_seq = self.track_seen_seq.get(track_id, 0)
            age = max(self.seen_seq - seen_seq, 0)
            velocity = self.track_center_velocity.get(track_id)
            predicted_center = cached_center
            if velocity is not None and age > 0:
                prediction_horizon = min(age, 3)
                predicted_center = (
                    cached_center[0] + velocity[0] * prediction_horizon,
                    cached_center[1] + velocity[1] * prediction_horizon,
                    cached_center[2] + velocity[2] * prediction_horizon,
                )
            distance = math.dist(center, predicted_center)
            signature_delta = body_signature_distance(
                current_signature,
                self.track_body_signatures.get(track_id),
            )
            if (
                age <= STEREO_TRACK_RECENT_PREFERENCE_WINDOW
                and distance <= STEREO_TRACK_RECENT_REUSE_MAX_M
            ):
                recent_candidates.append((seen_seq, signature_delta * 0.35 + distance, track_id))
            if distance > STEREO_TRACK_REUSE_MAX_M:
                continue
            # Prefer the nearest cached track; break ties by most recently seen.
            key = (distance, signature_delta, -seen_seq)
            if best_key is None or key < best_key:
                best_key = key
                best_track_id = track_id
        if recent_candidates:
            recent_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            return recent_candidates[0][2]
        return best_track_id

    def _trim_track_cache(self) -> None:
        if len(self.track_centers) <= STEREO_TRACK_CACHE_MAX:
            return
        stale_track_ids = sorted(
            self.track_centers.keys(),
            key=lambda track_id: self.track_seen_seq.get(track_id, 0),
        )[: len(self.track_centers) - STEREO_TRACK_CACHE_MAX]
        for track_id in stale_track_ids:
            self.track_centers.pop(track_id, None)
            self.track_prev_centers.pop(track_id, None)
            self.track_center_velocity.pop(track_id, None)
            self.track_body_signatures.pop(track_id, None)
            self.track_seen_seq.pop(track_id, None)
            self.track_candidates.pop(track_id, None)

    def propose_track_id(
        self,
        body_3d: list[BodyPoint3D],
        hint: AssociationHint | None,
        hand_hint_gap_m: float | None,
        continuity_gap_m: float | None,
        reserved_track_ids: set[str] | None = None,
    ) -> str:
        local_continuity_gap_m = self._local_continuity_gap_m(body_3d)
        hint_track_id = hint.selected_operator_track_id if hint is not None else None
        device_pose_track_id = hint.device_pose_candidate_track_id if hint is not None else None
        if (
            hint is not None
            and self._is_stereo_track_id(device_pose_track_id)
            and not (reserved_track_ids and device_pose_track_id in reserved_track_ids)
            and (hand_hint_gap_m is None or hand_hint_gap_m > IPHONE_HINT_PRIORITY_MAX_GAP_M)
            and (
                continuity_gap_m is None
                or continuity_gap_m <= STEREO_CONTINUITY_PRIORITY_MAX_GAP_M
                or local_continuity_gap_m is None
                or local_continuity_gap_m <= STEREO_SELECTED_TRACK_HOLD_MAX_M
            )
        ):
            track_id = device_pose_track_id or ""
        elif (
            hint is not None
            and self._is_stereo_track_id(hint_track_id)
            and not (reserved_track_ids and hint_track_id in reserved_track_ids)
            and (
                (hand_hint_gap_m is not None and hand_hint_gap_m <= IPHONE_HINT_PRIORITY_MAX_GAP_M)
                or (
                    continuity_gap_m is not None
                    and continuity_gap_m <= STEREO_CONTINUITY_PRIORITY_MAX_GAP_M
                )
            )
        ):
            track_id = hint_track_id or ""
        elif (
            self.current_track_id
            and not (reserved_track_ids and self.current_track_id in reserved_track_ids)
            and hint_track_id == self.current_track_id
            and local_continuity_gap_m is not None
            and local_continuity_gap_m <= STEREO_SELECTED_TRACK_HOLD_MAX_M
        ):
            track_id = self.current_track_id
        elif (
            self.current_track_id
            and not (reserved_track_ids and self.current_track_id in reserved_track_ids)
            and (
                (
                    continuity_gap_m is not None
                    and continuity_gap_m <= STEREO_CONTINUITY_PRIORITY_MAX_GAP_M
                )
                or (
                    local_continuity_gap_m is not None
                    and local_continuity_gap_m <= STEREO_BODY_CENTER_CONTINUITY_MAX_M
                )
            )
        ):
            track_id = self.current_track_id
        else:
            cached_track_id = self._nearest_cached_track_id(body_3d, reserved_track_ids)
            if cached_track_id is not None:
                track_id = cached_track_id
            else:
                while True:
                    track_id = f"stereo-person-{self.next_track_index}"
                    self.next_track_index += 1
                    if not reserved_track_ids or track_id not in reserved_track_ids:
                        break
        return track_id

    def _record_track_observation(self, track_id: str, body_3d: list[BodyPoint3D]) -> None:
        if track_id.startswith("stereo-person-"):
            suffix = track_id.removeprefix("stereo-person-")
            if suffix.isdigit():
                self.next_track_index = max(self.next_track_index, int(suffix) + 1)
        body_center = body_center_point(body_3d)
        signature = body_signature(body_3d)
        if body_center is not None:
            self.seen_seq += 1
            previous_center = self.track_centers.get(track_id)
            if previous_center is not None:
                self.track_prev_centers[track_id] = previous_center
                self.track_center_velocity[track_id] = (
                    body_center[0] - previous_center[0],
                    body_center[1] - previous_center[1],
                    body_center[2] - previous_center[2],
                )
            self.track_centers[track_id] = body_center
            if signature is not None:
                self.track_body_signatures[track_id] = signature
            self.track_seen_seq[track_id] = self.seen_seq
            self._trim_track_cache()

    def _record_track_candidate(self, candidate: SelectedStereoCandidate) -> None:
        self.track_candidates[candidate.operator_track_id] = copy.deepcopy(candidate)

    def predicted_track_center(
        self,
        track_id: str,
    ) -> tuple[tuple[float, float, float] | None, int]:
        cached_center = self.track_centers.get(track_id)
        if cached_center is None:
            return None, 0
        seen_seq = self.track_seen_seq.get(track_id, 0)
        age = max(self.seen_seq - seen_seq, 0)
        velocity = self.track_center_velocity.get(track_id)
        predicted_center = cached_center
        if velocity is not None and age > 0:
            prediction_horizon = min(age, 3)
            predicted_center = (
                cached_center[0] + velocity[0] * prediction_horizon,
                cached_center[1] + velocity[1] * prediction_horizon,
                cached_center[2] + velocity[2] * prediction_horizon,
            )
        return predicted_center, age

    def inject_persisted_candidates(
        self,
        selected: list[SelectedStereoCandidate],
        max_persons: int,
    ) -> list[SelectedStereoCandidate]:
        if not selected or len(selected) >= max_persons or len(selected) != 1:
            return selected
        reserved_track_ids = {candidate.operator_track_id for candidate in selected}
        primary_center = body_center_point(selected[0].body_3d)
        supplemented = list(selected)
        recent_track_ids = sorted(
            (
                track_id
                for track_id in self.track_candidates.keys()
                if track_id not in reserved_track_ids
            ),
            key=lambda track_id: self.track_seen_seq.get(track_id, 0),
            reverse=True,
        )
        for track_id in recent_track_ids:
            age = max(self.seen_seq - self.track_seen_seq.get(track_id, 0), 0)
            if age <= 0 or age > STEREO_TRACK_PERSISTENCE_WINDOW:
                continue
            cached = self.track_candidates.get(track_id)
            if cached is None:
                continue
            cached_center = body_center_point(cached.body_3d)
            if (
                primary_center is not None
                and cached_center is not None
                and math.dist(primary_center, cached_center) < STEREO_TRACK_PERSISTENCE_MIN_SEPARATION_M
            ):
                continue
            persisted = copy.deepcopy(cached)
            persisted.stereo_confidence = max(
                0.0,
                float(cached.stereo_confidence) * (STEREO_TRACK_PERSISTENCE_DECAY ** age),
            )
            persisted.selection_reason = "track_persistence"
            persisted.hand_hint_gap_m = None
            persisted.continuity_gap_m = None
            supplemented.append(persisted)
            reserved_track_ids.add(track_id)
            break
        return supplemented

    def commit_track_candidates(self, selected: list[SelectedStereoCandidate]) -> None:
        if not selected:
            return
        primary = selected[0]
        self.current_track_id = primary.operator_track_id
        self.last_body_center = body_center_point(primary.body_3d)
        self.last_left_wrist = parse_hint_point(primary.body_3d[9] if len(primary.body_3d) > 9 else None)
        self.last_right_wrist = parse_hint_point(primary.body_3d[10] if len(primary.body_3d) > 10 else None)
        for candidate in selected:
            self._record_track_observation(candidate.operator_track_id, candidate.body_3d)
            self._record_track_candidate(candidate)


@dataclass
class RawChunkTrack:
    media_track: str
    file_name: str
    file_path: Path
    frame_size: tuple[int, int]
    writer: cv2.VideoWriter


@dataclass
class PendingRawChunk:
    chunk_index: int
    frame_source_time_ns: list[int]
    tracks: list[RawChunkTrack]


@dataclass
class StereoFrameSnapshot:
    seq: int
    source_time_ns: int
    left_frame: np.ndarray
    right_frame: np.ndarray


class StereoCaptureWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: StereoFrameSnapshot | None = None
        self._seq = 0
        self._stereo_cap: cv2.VideoCapture | None = None
        self._left_cap: cv2.VideoCapture | None = None
        self._right_cap: cv2.VideoCapture | None = None
        self._consecutive_capture_failures = 0
        self._capture_reopen_threshold = max(int(args.capture_reopen_threshold), 1)
        self._capture_reopen_sleep_s = max(float(args.capture_reopen_sleep_s), 0.05)
        self._frame_interval = 1.0 / max(args.fps, 0.1)

    def start(self) -> None:
        self._reopen_captures("initial open")
        self._thread = threading.Thread(
            target=self._run,
            name="stereo-capture-worker",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._release_captures()

    def get_current(self) -> StereoFrameSnapshot | None:
        with self._lock:
            return self._latest

    def wait_for_latest(
        self,
        last_seq: int | None,
        timeout_s: float,
    ) -> StereoFrameSnapshot | None:
        deadline = time.monotonic() + max(timeout_s, 0.05)
        with self._condition:
            while True:
                current = self._latest
                if current is not None and (last_seq is None or current.seq > last_seq):
                    return current
                if self._stop_event.is_set():
                    return current
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return current if current is not None and last_seq is None else None
                self._condition.wait(timeout=remaining)

    def _release_captures(self) -> None:
        if self._stereo_cap is not None:
            self._stereo_cap.release()
            self._stereo_cap = None
        if self._left_cap is not None:
            self._left_cap.release()
            self._left_cap = None
        if self._right_cap is not None:
            self._right_cap.release()
            self._right_cap = None

    def _reopen_captures(self, reason: str) -> None:
        print(f"[capture] reopening: {reason}", flush=True)
        self._release_captures()
        if self.args.input_mode == "uvc-sbs":
            self._stereo_cap = open_capture(
                self.args.stereo_device,
                self.args.width * 2,
                self.args.height,
                self.args.fps,
                "stereo-sbs",
            )
        else:
            self._left_cap = open_capture(
                self.args.left_device,
                self.args.width,
                self.args.height,
                self.args.fps,
                "left",
            )
            self._right_cap = open_capture(
                self.args.right_device,
                self.args.width,
                self.args.height,
                self.args.fps,
                "right",
            )

    def _publish_frame(self, left_frame: np.ndarray, right_frame: np.ndarray) -> None:
        snapshot = StereoFrameSnapshot(
            seq=self._seq,
            source_time_ns=time.monotonic_ns(),
            left_frame=left_frame,
            right_frame=right_frame,
        )
        self._seq += 1
        with self._condition:
            self._latest = snapshot
            self._condition.notify_all()

    def _capture_once(self) -> bool:
        if self.args.input_mode == "uvc-sbs":
            assert self._stereo_cap is not None
            ok_stereo, stereo_frame = self._stereo_cap.read()
            if not ok_stereo or stereo_frame is None:
                return False
            try:
                left_frame, right_frame = split_side_by_side(stereo_frame)
            except RuntimeError:
                return False
        else:
            assert self._left_cap is not None and self._right_cap is not None
            ok_left, left_frame = self._left_cap.read()
            ok_right, right_frame = self._right_cap.read()
            if not ok_left or left_frame is None or not ok_right or right_frame is None:
                return False

        left_frame = resize_for_inference(left_frame, self.args.width, self.args.height)
        right_frame = resize_for_inference(right_frame, self.args.width, self.args.height)
        self._publish_frame(left_frame, right_frame)
        return True

    def _run(self) -> None:
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            if self._capture_once():
                self._consecutive_capture_failures = 0
            else:
                self._consecutive_capture_failures += 1
                if self._consecutive_capture_failures >= self._capture_reopen_threshold:
                    try:
                        self._reopen_captures(
                            f"capture read failed {self._consecutive_capture_failures}x"
                        )
                        self._consecutive_capture_failures = 0
                        time.sleep(self._capture_reopen_sleep_s)
                    except RuntimeError as error:
                        print(f"[capture] reopen failed: {error}", file=sys.stderr)
                        time.sleep(self._capture_reopen_sleep_s)
                else:
                    time.sleep(0.05)
                continue
            sleep_until(next_deadline, self._frame_interval)
            next_deadline += self._frame_interval


class StereoDebugPreviewWriter:
    def __init__(self, args: argparse.Namespace, capture_worker: StereoCaptureWorker) -> None:
        self.args = args
        self.capture_worker = capture_worker
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_written_seq: int | None = None
        self._latest_left_candidate: PersonCandidate | None = None
        self._latest_right_candidate: PersonCandidate | None = None
        self._interval_s = 1.0 / max(args.debug_preview_fps, 0.1)
        self._enabled = bool(
            args.debug_preview or args.debug_left_frame or args.debug_right_frame
        )

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="stereo-debug-preview-writer",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def publish_candidates(
        self,
        left_candidate: PersonCandidate | None,
        right_candidate: PersonCandidate | None,
    ) -> None:
        with self._lock:
            self._latest_left_candidate = copy.deepcopy(left_candidate)
            self._latest_right_candidate = copy.deepcopy(right_candidate)

    def clear_candidates(self) -> None:
        self.publish_candidates(None, None)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self.capture_worker.get_current()
            if snapshot is not None and snapshot.seq != self._last_written_seq:
                with self._lock:
                    left_candidate = self._latest_left_candidate
                    right_candidate = self._latest_right_candidate
                try:
                    if self.args.debug_left_frame or self.args.debug_right_frame:
                        save_high_res_debug_frames(
                            self.args.debug_left_frame or "",
                            self.args.debug_right_frame or "",
                            snapshot.left_frame,
                            snapshot.right_frame,
                            self.args.debug_preview_jpeg_quality,
                        )
                    if self.args.debug_preview:
                        save_preview(
                            self.args.debug_preview,
                            snapshot.left_frame,
                            snapshot.right_frame,
                            left_candidate,
                            right_candidate,
                            self.args.debug_preview_max_width,
                            self.args.debug_preview_jpeg_quality,
                        )
                    self._last_written_seq = snapshot.seq
                except Exception as exc:
                    print(f"[preview] write failed: {exc}", file=sys.stderr)
            self._stop_event.wait(self._interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取双目视频流，生成 stereo_pose_packet 并上送到 edge"
    )
    parser.add_argument("--http-base", required=True, help="例如 http://127.0.0.1:8080")
    parser.add_argument("--edge-token", required=True)
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--device-id", default="stereo-v4l2-001")
    parser.add_argument("--left-device", default="/dev/video0")
    parser.add_argument("--right-device", default="/dev/video1")
    parser.add_argument("--stereo-device", default="/dev/video0")
    parser.add_argument(
        "--input-mode",
        choices=("dual-devices", "uvc-sbs"),
        default="dual-devices",
        help="dual-devices=左右目分别来自两个设备；uvc-sbs=单个 UVC side-by-side 设备",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--inference-width",
        type=int,
        default=0,
        help="双目人体检测推理宽度；0 表示跟随 capture 宽度",
    )
    parser.add_argument(
        "--inference-height",
        type=int,
        default=0,
        help="双目人体检测推理高度；0 表示跟随 capture 高度",
    )
    parser.add_argument("--model-mode", choices=("lightweight", "balanced", "performance"), default="lightweight")
    parser.add_argument("--body-score-threshold", type=float, default=0.35)
    parser.add_argument("--joint-score-threshold", type=float, default=0.25)
    parser.add_argument("--min-disparity-px", type=float, default=2.0)
    parser.add_argument(
        "--max-epipolar-y-delta-px",
        type=float,
        default=36.0,
        help="左右目同名关节允许的最大 y 像素差；超过则不生成该关节 3D。",
    )
    parser.add_argument(
        "--min-depth-m",
        type=float,
        default=0.35,
        help="双目三角化允许的最小人体关节深度，防止错误配对生成近场假 3D。",
    )
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument(
        "--min-triangulated-joints",
        type=int,
        default=6,
        help="单个人体候选至少需要多少个几何自洽的双目关节。",
    )
    parser.add_argument("--baseline-m", type=float, default=0.060)
    parser.add_argument("--fx-px", type=float, default=721.1)
    parser.add_argument("--fy-px", type=float, default=720.8)
    parser.add_argument("--cx-px", type=float, default=640.0)
    parser.add_argument("--cy-px", type=float, default=360.0)
    parser.add_argument(
        "--calibration-path",
        default="",
        help="可选：从 JSON 文件读取双目标定；存在时优先覆盖 fx/fy/cx/cy/baseline 等运行参数。",
    )
    parser.add_argument("--sensor-frame", default="stereo_pair_frame")
    parser.add_argument("--operator-frame", default="operator_frame")
    parser.add_argument("--extrinsic-version", default="stereo-v4l2-default")
    parser.add_argument("--operator-track-id", default="stereo-primary-001")
    parser.add_argument("--association-hint-url", default="", help="可选：读取 edge /association/hint，用 iPhone 手腕/上一帧双目结果辅助选人。")
    parser.add_argument("--ensure-session", action="store_true", help="启动前自动调用 /session/start 与 /time/sync")
    parser.add_argument(
        "--time-sync-interval-s",
        type=float,
        default=2.0,
        help="周期性重新上报 /time/sync 的间隔秒数；0 表示只在启动时同步一次。",
    )
    parser.add_argument(
        "--time-sync-sample-count",
        type=int,
        default=5,
        help="每次 /time/sync 前先对 /time 采样多少次，取 RTT 最小的样本。",
    )
    parser.add_argument(
        "--time-sync-timeout-s",
        type=float,
        default=2.0,
        help="单次 /time 探测和 /time/sync 上报的超时秒数。",
    )
    parser.add_argument("--upload-raw-media", action="store_true", help="同时把双目原始 left/right/preview 分轨切 chunk 上传到 edge")
    parser.add_argument("--raw-chunk-duration-s", type=float, default=2.0, help="原始视频 chunk 时长（秒）")
    parser.add_argument("--raw-upload-timeout-s", type=float, default=8.0, help="原始视频 chunk 上传超时（秒）")
    parser.add_argument(
        "--raw-upload-queue-size",
        type=int,
        default=4,
        help="原始视频 chunk 后台上传队列长度；超出后直接丢弃新生成的 chunk，避免阻塞主循环",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续运行")
    parser.add_argument("--debug-preview", help="可选：保存最近一帧双目拼接预览图片")
    parser.add_argument("--debug-left-frame", help="可选：保存最近一帧高分辨率左目图片")
    parser.add_argument("--debug-right-frame", help="可选：保存最近一帧高分辨率右目图片")
    parser.add_argument("--debug-preview-fps", type=float, default=5.0, help="调试预览图最高写盘频率")
    parser.add_argument("--debug-preview-max-width", type=int, default=1280, help="调试预览图最大宽度")
    parser.add_argument("--debug-preview-jpeg-quality", type=int, default=80, help="调试预览图 JPEG 质量")
    parser.add_argument(
        "--capture-reopen-threshold",
        type=int,
        default=1,
        help="连续多少次读帧失败后，主动重开 VideoCapture；1 表示首个 timeout 就重开。",
    )
    parser.add_argument(
        "--capture-reopen-sleep-s",
        type=float,
        default=0.5,
        help="主动重开 VideoCapture 后额外等待多久再继续读帧。",
    )
    parser.add_argument("--max-persons", type=int, default=4, help="每帧最多输出多少个双目人体目标")
    parser.add_argument(
        "--det-frequency",
        type=int,
        default=3,
        help="全图主 pass 每隔多少帧重新跑 detector；中间帧由 pose tracker 续跟。",
    )
    parser.add_argument(
        "--tracking-thr",
        type=float,
        default=0.3,
        help="全图主 pass 的 pose tracker IoU 阈值。",
    )
    parser.add_argument(
        "--parallel-inference-workers",
        type=int,
        default=2,
        help="左右目推理并行 worker 数；1 表示串行，2 表示左右目并行。",
    )
    parser.add_argument(
        "--low-roi-second-pass",
        action="store_true",
        help="对低位/遮挡区域再跑第二遍人体检测，用于补召回桌边坐姿或被前景挡住的人。",
    )
    parser.add_argument(
        "--low-roi-second-pass-interval",
        type=int,
        default=3,
        help="当主 pass 已经检测到 1 个人时，第二遍 ROI 每隔多少帧再跑一次；0/1 表示每帧都跑。",
    )
    parser.add_argument(
        "--low-roi-top-fraction",
        type=float,
        default=0.38,
        help="第二遍 ROI 的顶部起始位置，占整幅图高度比例。",
    )
    parser.add_argument(
        "--low-roi-height-fraction",
        type=float,
        default=0.62,
        help="第二遍 ROI 的高度，占整幅图高度比例。",
    )
    parser.add_argument(
        "--low-roi-left-fraction",
        type=float,
        default=0.0,
        help="第二遍 ROI 的左侧起始位置，占整幅图宽度比例。",
    )
    parser.add_argument(
        "--low-roi-width-fraction",
        type=float,
        default=1.0,
        help="第二遍 ROI 的宽度，占整幅图宽度比例。",
    )
    parser.add_argument(
        "--low-roi-inference-width",
        type=int,
        default=0,
        help="第二遍 ROI 检测推理宽度；0 表示复用主推理宽度。",
    )
    parser.add_argument(
        "--low-roi-inference-height",
        type=int,
        default=0,
        help="第二遍 ROI 检测推理高度；0 表示复用主推理高度。",
    )
    parser.add_argument(
        "--low-roi-body-score-threshold",
        type=float,
        default=0.0,
        help="第二遍 ROI 的 body score 阈值；0 表示复用主阈值。",
    )
    parser.add_argument(
        "--low-roi-joint-score-threshold",
        type=float,
        default=0.0,
        help="第二遍 ROI 的 joint score 阈值；0 表示复用主阈值。",
    )
    return parser.parse_args()


def _require_positive_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _require_finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _require_positive_int(value: object, label: str) -> int:
    if not isinstance(value, (int, float)) or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _load_runtime_calibration(path: Path, runtime_width: int, runtime_height: int) -> StereoCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stereo calibration file must contain a JSON object")

    def parse_camera(key: str) -> CameraCalibration:
        camera = payload.get(key)
        if not isinstance(camera, dict):
            raise ValueError(f"{key} missing from stereo calibration")
        reference_w = _require_positive_int(camera.get("reference_image_w"), f"{key}.reference_image_w")
        reference_h = _require_positive_int(camera.get("reference_image_h"), f"{key}.reference_image_h")
        scale_x = float(runtime_width) / float(reference_w)
        scale_y = float(runtime_height) / float(reference_h)
        return CameraCalibration(
            fx_px=_require_positive_number(camera.get("fx_px"), f"{key}.fx_px") * scale_x,
            fy_px=_require_positive_number(camera.get("fy_px"), f"{key}.fy_px") * scale_y,
            cx_px=_require_finite_number(camera.get("cx_px"), f"{key}.cx_px") * scale_x,
            cy_px=_require_finite_number(camera.get("cy_px"), f"{key}.cy_px") * scale_y,
            reference_image_w=runtime_width,
            reference_image_h=runtime_height,
        )

    calibration = StereoCalibration(
        left=parse_camera("left_intrinsics"),
        right=parse_camera("right_intrinsics"),
        baseline_m=_require_positive_number(payload.get("baseline_m"), "baseline_m"),
        sensor_frame=str(payload.get("sensor_frame") or "stereo_pair_frame"),
        operator_frame=str(payload.get("operator_frame") or "operator_frame"),
        extrinsic_version=str(payload.get("extrinsic_version") or f"runtime-file:{path.name}"),
    )
    print(
        f"[stereo] loaded runtime calibration from {path} "
        f"(runtime={runtime_width}x{runtime_height}, "
        f"baseline={calibration.baseline_m:.4f}m, version={calibration.extrinsic_version})"
    )
    return calibration


def build_calibration(args: argparse.Namespace) -> StereoCalibration:
    calibration_path = Path(args.calibration_path).expanduser() if args.calibration_path else None
    if calibration_path and calibration_path.exists():
        return _load_runtime_calibration(calibration_path, int(args.width), int(args.height))

    left = CameraCalibration(
        fx_px=args.fx_px,
        fy_px=args.fy_px,
        cx_px=args.cx_px,
        cy_px=args.cy_px,
        reference_image_w=args.width,
        reference_image_h=args.height,
    )
    right = CameraCalibration(
        fx_px=args.fx_px,
        fy_px=args.fy_px,
        cx_px=args.cx_px,
        cy_px=args.cy_px,
        reference_image_w=args.width,
        reference_image_h=args.height,
    )
    return StereoCalibration(
        left=left,
        right=right,
        baseline_m=args.baseline_m,
        sensor_frame=args.sensor_frame,
        operator_frame=args.operator_frame,
        extrinsic_version=args.extrinsic_version,
    )


def default_association_hint_url(http_base: str) -> str:
    return f"{http_base.rstrip('/')}/association/hint"


class StereoRawChunkUploader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.enabled = bool(args.upload_raw_media)
        self.chunk_duration_ns = max(int(args.raw_chunk_duration_s * 1_000_000_000), 250_000_000)
        self.frame_rate_hz = max(float(args.fps), 0.1)
        self.temp_root = (
            Path(tempfile.mkdtemp(prefix="chek-stereo-raw-")) if self.enabled else Path(".")
        )
        self.chunk_index = 0
        self.chunk_start_ns: int | None = None
        self.frame_source_time_ns: list[int] = []
        self.tracks: dict[str, RawChunkTrack] = {}
        self.upload_queue: queue.Queue[PendingRawChunk | None] | None = None
        self.upload_thread: threading.Thread | None = None
        if self.enabled:
            self.upload_queue = queue.Queue(maxsize=max(int(args.raw_upload_queue_size), 1))
            self.upload_thread = threading.Thread(
                target=self._upload_loop,
                name="stereo-raw-uploader",
                daemon=True,
            )
            self.upload_thread.start()

    def append(
        self,
        source_time_ns: int,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        preview_frame: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        if self.chunk_start_ns is None:
            self._open_chunk(left_frame, right_frame, preview_frame, source_time_ns)

        self.tracks["left"].writer.write(left_frame)
        self.tracks["right"].writer.write(right_frame)
        self.tracks["preview"].writer.write(preview_frame)
        self.frame_source_time_ns.append(int(source_time_ns))

        if source_time_ns - (self.chunk_start_ns or source_time_ns) >= self.chunk_duration_ns:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self.frame_source_time_ns or not self.tracks:
            self._reset_chunk_state()
            return

        current_tracks = list(self.tracks.values())
        frame_source_time_ns = list(self.frame_source_time_ns)
        chunk_index = self.chunk_index
        self._release_writers(current_tracks)

        try:
            pending = PendingRawChunk(
                chunk_index=chunk_index,
                frame_source_time_ns=frame_source_time_ns,
                tracks=current_tracks,
            )
            if self.upload_queue is not None:
                try:
                    self.upload_queue.put_nowait(pending)
                except queue.Full:
                    print(
                        f"[stereo-raw] upload queue full, dropping chunk={chunk_index}",
                        file=sys.stderr,
                    )
                    self._cleanup_tracks(current_tracks)
        finally:
            self.chunk_index += 1
            self._reset_chunk_state()

    def close(self) -> None:
        try:
            self.flush()
            if self.upload_queue is not None:
                self.upload_queue.put(None)
            if self.upload_thread is not None:
                self.upload_thread.join(timeout=max(float(self.args.raw_upload_timeout_s), 1.0) * 3.0)
        finally:
            if self.enabled:
                shutil.rmtree(self.temp_root, ignore_errors=True)

    def _open_chunk(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        preview_frame: np.ndarray,
        source_time_ns: int,
    ) -> None:
        chunk_dir = self.temp_root / f"{self.chunk_index:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.tracks = {
            "left": self._open_track(chunk_dir, "left", left_frame, fourcc),
            "right": self._open_track(chunk_dir, "right", right_frame, fourcc),
            "preview": self._open_track(chunk_dir, "preview", preview_frame, fourcc),
        }
        self.chunk_start_ns = int(source_time_ns)
        self.frame_source_time_ns = []

    def _open_track(
        self,
        chunk_dir: Path,
        media_track: str,
        frame: np.ndarray,
        fourcc: int,
    ) -> RawChunkTrack:
        height, width = frame.shape[:2]
        file_name = f"{media_track}__chunk_{self.chunk_index:06d}.mp4"
        file_path = chunk_dir / file_name
        writer = cv2.VideoWriter(str(file_path), fourcc, self.frame_rate_hz, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法打开原始视频 writer: {file_path}")
        return RawChunkTrack(
            media_track=media_track,
            file_name=file_name,
            file_path=file_path,
            frame_size=(width, height),
            writer=writer,
        )

    def _release_writers(self, tracks: Iterable[RawChunkTrack]) -> None:
        for track in tracks:
            track.writer.release()

    def _cleanup_tracks(self, tracks: Iterable[RawChunkTrack]) -> None:
        for track in tracks:
            try:
                track.file_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _upload_track(
        self,
        session: requests.Session,
        track: RawChunkTrack,
        chunk_index: int,
        frame_source_time_ns: list[int],
    ) -> None:
        with track.file_path.open("rb") as handle:
            files = {
                "file": (track.file_name, handle, "video/mp4"),
            }
            metadata = {
                "trip_id": self.args.trip_id,
                "session_id": self.args.session_id,
                "device_id": self.args.device_id,
                "media_scope": "stereo",
                "media_track": track.media_track,
                "source_kind": "stereo_pair",
                "clock_domain": "stereo_monotonic_ns",
                "chunk_index": chunk_index,
                "file_type": "video",
                "file_name": track.file_name,
                "source_time_ns": frame_source_time_ns[-1],
                "source_start_time_ns": frame_source_time_ns[0],
                "source_end_time_ns": frame_source_time_ns[-1],
                "frame_source_time_ns": frame_source_time_ns,
                "frame_count": len(frame_source_time_ns),
                "frame_rate_hz": self.frame_rate_hz,
            }
            response = session.post(
                f"{self.args.http_base.rstrip('/')}/common_task/upload_chunk",
                headers={"Authorization": f"Bearer {self.args.edge_token}"},
                data={"metadata": json.dumps(metadata, ensure_ascii=False)},
                files=files,
                timeout=self.args.raw_upload_timeout_s,
            )
            response.raise_for_status()

    def _upload_loop(self) -> None:
        if self.upload_queue is None:
            return
        upload_session = requests.Session()
        try:
            while True:
                pending = self.upload_queue.get()
                try:
                    if pending is None:
                        return
                    for track in pending.tracks:
                        try:
                            self._upload_track(
                                upload_session,
                                track,
                                pending.chunk_index,
                                pending.frame_source_time_ns,
                            )
                        except requests.RequestException as exc:
                            print(
                                f"[stereo-raw] upload failed track={track.media_track} chunk={pending.chunk_index}: {exc}",
                                file=sys.stderr,
                            )
                    self._cleanup_tracks(pending.tracks)
                finally:
                    self.upload_queue.task_done()
        finally:
            upload_session.close()

    def _reset_chunk_state(self) -> None:
        self.chunk_start_ns = None
        self.frame_source_time_ns = []
        self.tracks = {}


def decode_fourcc(value: float) -> str:
    raw = int(value)
    chars = [chr((raw >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    if all(32 <= ord(ch) <= 126 for ch in chars):
        return "".join(chars)
    return f"0x{raw:08x}"


def open_capture(
    path_or_index: str,
    width: int,
    height: int,
    fps: float,
    label: str,
) -> cv2.VideoCapture:
    if path_or_index.isdigit():
        cap = cv2.VideoCapture(int(path_or_index), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(path_or_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频设备：{path_or_index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC) or 0.0)
    print(
        f"[capture] {label}: requested={width}x{height}@{fps:g} MJPG "
        f"actual={actual_width}x{actual_height}@{actual_fps:.3f} {actual_fourcc}",
        flush=True,
    )
    return cap


def resize_for_inference(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def run_estimator(
    estimator: Wholebody,
    frame: np.ndarray,
    inference_width: int,
    inference_height: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    inference_frame = resize_for_inference(frame, inference_width, inference_height)
    keypoints, scores = estimator(inference_frame)
    if keypoints is None or len(keypoints) == 0:
        return keypoints, scores
    if frame.shape[1] == inference_width and frame.shape[0] == inference_height:
        return keypoints, scores

    scaled = np.asarray(keypoints, dtype=np.float32).copy()
    scale_x = frame.shape[1] / float(inference_width)
    scale_y = frame.shape[0] / float(inference_height)
    scaled[..., 0] *= scale_x
    scaled[..., 1] *= scale_y
    return scaled, scores


def run_dual_estimator_pass(
    executor: ThreadPoolExecutor | None,
    left_estimator: Wholebody,
    left_frame: np.ndarray,
    right_estimator: Wholebody,
    right_frame: np.ndarray,
    inference_width: int,
    inference_height: int,
) -> tuple[tuple[np.ndarray | None, np.ndarray | None], tuple[np.ndarray | None, np.ndarray | None]]:
    if executor is None:
        return (
            run_estimator(left_estimator, left_frame, inference_width, inference_height),
            run_estimator(right_estimator, right_frame, inference_width, inference_height),
        )
    left_future = executor.submit(
        run_estimator,
        left_estimator,
        left_frame,
        inference_width,
        inference_height,
    )
    right_future = executor.submit(
        run_estimator,
        right_estimator,
        right_frame,
        inference_width,
        inference_height,
    )
    return left_future.result(), right_future.result()


def split_side_by_side(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_width = frame.shape[1]
    half_width = frame_width // 2
    if half_width <= 0:
        raise RuntimeError("side-by-side 帧宽异常，无法拆分左右目")
    left = frame[:, :half_width]
    right = frame[:, half_width : half_width * 2]
    if left.size == 0 or right.size == 0:
        raise RuntimeError("side-by-side 帧拆分失败")
    return left, right


def extract_candidates(
    keypoints: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    source_tag: str = "full_frame",
    joint_score_threshold_override: float | None = None,
) -> list[PersonCandidate]:
    if keypoints is None or scores is None or len(keypoints) == 0:
        return []
    candidates: list[PersonCandidate] = []
    for person_kpts, person_scores in zip(keypoints, scores):
        body = np.asarray(person_kpts[:COCO_BODY_17], dtype=np.float32)
        body_scores = np.asarray(person_scores[:COCO_BODY_17], dtype=np.float32)
        valid = body_scores >= score_threshold
        if int(valid.sum()) < 6:
            continue
        valid_points = body[valid]
        y_min, y_max = float(valid_points[:, 1].min()), float(valid_points[:, 1].max())
        torso_points = body[[index for index in TORSO_INDICES if index < len(body)]]
        torso_scores = body_scores[[index for index in TORSO_INDICES if index < len(body_scores)]]
        torso_valid = torso_scores >= score_threshold
        anchor_points = torso_points[torso_valid] if torso_valid.any() else valid_points
        center_xy = (float(anchor_points[:, 0].mean()), float(anchor_points[:, 1].mean()))
        candidates.append(
            PersonCandidate(
                keypoints=body,
                scores=body_scores,
                bbox_height_px=max(y_max - y_min, 1.0),
                center_xy=center_xy,
                mean_score=float(body_scores[valid].mean()),
                source_tag=source_tag,
                joint_score_threshold_override=joint_score_threshold_override,
            )
        )
    candidates.sort(key=lambda candidate: candidate.mean_score, reverse=True)
    return candidates


def clamp_fraction(value: float, default: float) -> float:
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def compute_low_roi(
    frame: np.ndarray,
    top_fraction: float,
    height_fraction: float,
    left_fraction: float,
    width_fraction: float,
) -> tuple[int, int, int, int] | None:
    frame_h, frame_w = frame.shape[:2]
    top_fraction = clamp_fraction(top_fraction, 0.38)
    height_fraction = clamp_fraction(height_fraction, 0.62)
    left_fraction = clamp_fraction(left_fraction, 0.0)
    width_fraction = clamp_fraction(width_fraction, 1.0)
    y0 = int(round(frame_h * top_fraction))
    h = int(round(frame_h * height_fraction))
    x0 = int(round(frame_w * left_fraction))
    w = int(round(frame_w * width_fraction))
    if h <= 0 or w <= 0:
        return None
    y0 = max(0, min(y0, frame_h - 1))
    x0 = max(0, min(x0, frame_w - 1))
    y1 = max(y0 + 1, min(frame_h, y0 + h))
    x1 = max(x0 + 1, min(frame_w, x0 + w))
    if y1 - y0 < 32 or x1 - x0 < 32:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def build_low_roi_regions(
    frame: np.ndarray,
    top_fraction: float,
    height_fraction: float,
    left_fraction: float,
    width_fraction: float,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    base_roi = compute_low_roi(
        frame,
        top_fraction,
        height_fraction,
        left_fraction,
        width_fraction,
    )
    if base_roi is None:
        return []
    regions: list[tuple[str, tuple[int, int, int, int]]] = [("low_roi", base_roi)]
    x0, y0, roi_w, roi_h = base_roi
    if roi_w >= 320:
        half_w = max(roi_w // 2, 1)
        overlap = max(int(round(roi_w * 0.08)), 12)
        left_roi = (x0, y0, min(roi_w, half_w + overlap), roi_h)
        right_x0 = max(x0, x0 + roi_w - (half_w + overlap))
        right_roi = (right_x0, y0, x0 + roi_w - right_x0, roi_h)
        if left_roi[2] >= 96:
            regions.append(("low_roi_left", left_roi))
        if right_roi[2] >= 96:
            regions.append(("low_roi_right", right_roi))
    return regions


def run_estimator_on_roi(
    estimator: Wholebody,
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    inference_width: int,
    inference_height: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    x0, y0, roi_w, roi_h = roi
    roi_frame = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]
    if roi_frame.size == 0:
        return None, None
    keypoints, scores = run_estimator(
        estimator,
        roi_frame,
        inference_width,
        inference_height,
    )
    if keypoints is None or len(keypoints) == 0:
        return keypoints, scores
    remapped = np.asarray(keypoints, dtype=np.float32).copy()
    remapped[..., 0] += float(x0)
    remapped[..., 1] += float(y0)
    return remapped, scores


def candidate_duplicate_distance_px(
    left: PersonCandidate,
    right: PersonCandidate,
) -> float:
    dx = left.center_xy[0] - right.center_xy[0]
    dy = left.center_xy[1] - right.center_xy[1]
    return math.hypot(dx, dy)


def merge_candidate_passes(
    primary: list[PersonCandidate],
    secondary: list[PersonCandidate],
) -> list[PersonCandidate]:
    merged = list(primary)
    for candidate in secondary:
        duplicate_index: int | None = None
        duplicate_score: tuple[float, float] | None = None
        for idx, existing in enumerate(merged):
            center_gap = candidate_duplicate_distance_px(candidate, existing)
            height_ratio = candidate.bbox_height_px / max(existing.bbox_height_px, 1.0)
            height_ratio = max(height_ratio, 1.0 / max(height_ratio, 1e-6))
            allowed_gap = max(
                24.0,
                min(candidate.bbox_height_px, existing.bbox_height_px) * 0.32,
            )
            if center_gap <= allowed_gap and height_ratio <= 1.8:
                score = (existing.mean_score, -center_gap)
                duplicate_index = idx
                duplicate_score = score
                break
        if duplicate_index is None:
            merged.append(candidate)
            continue
        existing = merged[duplicate_index]
        candidate_from_low_roi = candidate.source_tag.startswith("low_roi")
        existing_from_low_roi = existing.source_tag.startswith("low_roi")
        prefer_candidate = (
            candidate.mean_score > existing.mean_score + 0.02
            or (
                candidate_from_low_roi
                and not existing_from_low_roi
                and candidate.mean_score >= existing.mean_score - 0.03
            )
        )
        if prefer_candidate:
            merged[duplicate_index] = candidate
    merged.sort(key=lambda item: item.mean_score, reverse=True)
    return merged


def parse_hint_point(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        point = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in point):
        return None
    if all(abs(v) < 1e-6 for v in point):
        return None
    return point


def fetch_association_hint(
    session: requests.Session,
    args: argparse.Namespace,
) -> AssociationHint | None:
    url = args.association_hint_url or default_association_hint_url(args.http_base)
    response = session.get(
        url,
        headers={"Authorization": f"Bearer {args.edge_token}"},
        timeout=2,
    )
    response.raise_for_status()
    payload = response.json()
    iphone = payload.get("iphone") or {}
    stereo = payload.get("stereo") or {}
    association = payload.get("association") or {}
    return AssociationHint(
        selected_operator_track_id=association.get("selected_operator_track_id"),
        device_pose_candidate_track_id=association.get("device_pose_candidate_track_id"),
        iphone_left_wrist=parse_hint_point(iphone.get("left_wrist")),
        iphone_right_wrist=parse_hint_point(iphone.get("right_wrist")),
        stereo_left_wrist=parse_hint_point(stereo.get("left_wrist")),
        stereo_right_wrist=parse_hint_point(stereo.get("right_wrist")),
    )


def base_pair_cost(left: PersonCandidate, right: PersonCandidate) -> float:
    return (
        abs(left.center_xy[1] - right.center_xy[1])
        + abs(left.bbox_height_px - right.bbox_height_px) * 0.25
        - (left.mean_score + right.mean_score) * 40.0
    )


def candidate_hand_gap_m(
    body_3d: list[BodyPoint3D],
    left_hint: tuple[float, float, float] | None,
    right_hint: tuple[float, float, float] | None,
) -> float | None:
    gaps: list[float] = []
    if left_hint is not None and len(body_3d) > 9:
        left_point = parse_hint_point(body_3d[9])
        if left_point is not None:
            gaps.append(math.dist(left_point, left_hint))
    if right_hint is not None and len(body_3d) > 10:
        right_point = parse_hint_point(body_3d[10])
        if right_point is not None:
            gaps.append(math.dist(right_point, right_hint))
    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def body_center_point(body_3d: list[BodyPoint3D]) -> tuple[float, float, float] | None:
    center_points: list[tuple[float, float, float]] = []
    for index in TORSO_INDICES:
        if index >= len(body_3d):
            continue
        point = parse_hint_point(body_3d[index])
        if point is None:
            continue
        center_points.append(point)
    if not center_points:
        return None
    xs, ys, zs = zip(*center_points)
    return (
        float(sum(xs) / len(xs)),
        float(sum(ys) / len(ys)),
        float(sum(zs) / len(zs)),
    )


def body_signature(body_3d: list[BodyPoint3D]) -> tuple[float, float] | None:
    points = [parse_hint_point(point) for point in body_3d]
    valid_points = [point for point in points if point is not None]
    if not valid_points:
        return None
    ys = [point[1] for point in valid_points]
    body_height = float(max(ys) - min(ys))
    left_shoulder = points[5] if len(points) > 5 else None
    right_shoulder = points[6] if len(points) > 6 else None
    shoulder_width = (
        math.dist(left_shoulder, right_shoulder)
        if left_shoulder is not None and right_shoulder is not None
        else max(0.0, body_height * 0.32)
    )
    return (shoulder_width, body_height)


def body_signature_distance(
    current: tuple[float, float] | None,
    cached: tuple[float, float] | None,
) -> float:
    if current is None or cached is None:
        return 0.0
    shoulder_delta = abs(current[0] - cached[0]) / max(0.08, cached[0])
    height_delta = abs(current[1] - cached[1]) / max(0.20, cached[1])
    return shoulder_delta * 0.6 + height_delta * 0.4


def stabilize_low_roi_candidate(
    candidate: SelectedStereoCandidate,
    track_memory: StereoTrackMemory,
) -> SelectedStereoCandidate:
    if not (
        candidate.left_candidate.source_tag.startswith("low_roi")
        or candidate.right_candidate.source_tag.startswith("low_roi")
    ):
        return candidate
    predicted_center, age = track_memory.predicted_track_center(candidate.operator_track_id)
    if predicted_center is None or age <= 0 or age > LOW_ROI_TRACK_STABILIZE_WINDOW:
        return candidate
    current_center = body_center_point(candidate.body_3d)
    if current_center is None:
        return candidate
    gap_m = math.dist(current_center, predicted_center)
    if gap_m <= LOW_ROI_TRACK_STABILIZE_MIN_GAP_M or gap_m > LOW_ROI_TRACK_STABILIZE_MAX_GAP_M:
        return candidate
    candidate.continuity_gap_m = (
        min(candidate.continuity_gap_m, gap_m) if candidate.continuity_gap_m is not None else gap_m
    )
    if candidate.selection_reason == "geometry_score":
        candidate.selection_reason = "geometry_score+low_roi_track_continuity"
    else:
        candidate.selection_reason = f"{candidate.selection_reason}+low_roi_track_continuity"
    return candidate


def is_depth_outlier(depth: float, reference_depth: float) -> bool:
    max_delta = max(
        STEREO_DEPTH_OUTLIER_MIN_DELTA_M,
        reference_depth * STEREO_DEPTH_OUTLIER_RATIO,
    )
    return abs(depth - reference_depth) > max_delta


def pair_priority_rank(
    hand_hint_gap_m: float | None,
    continuity_gap_m: float | None,
    cost: float,
) -> tuple[int, float, float, float]:
    if hand_hint_gap_m is not None and hand_hint_gap_m <= IPHONE_HINT_PRIORITY_MAX_GAP_M:
        return (0, hand_hint_gap_m, continuity_gap_m or float("inf"), cost)
    if continuity_gap_m is not None and continuity_gap_m <= STEREO_CONTINUITY_PRIORITY_MAX_GAP_M:
        return (1, continuity_gap_m, hand_hint_gap_m or float("inf"), cost)
    return (2, hand_hint_gap_m or float("inf"), continuity_gap_m or float("inf"), cost)


def enumerate_candidate_pairs(
    left_candidates: list[PersonCandidate],
    right_candidates: list[PersonCandidate],
    calibration: StereoCalibration,
    args: argparse.Namespace,
    hint: AssociationHint | None,
    track_memory: StereoTrackMemory,
) -> list[tuple[tuple[int, float, float, float], SelectedStereoCandidate]]:
    candidates: list[tuple[tuple[int, float, float, float], SelectedStereoCandidate]] = []
    ordered_left = sorted(
        enumerate(left_candidates),
        key=lambda item: item[1].center_xy[0],
    )
    for left_index, left in ordered_left:
        for right_index, right in enumerate(right_candidates):
            disparity = left.center_xy[0] - right.center_xy[0]
            if disparity <= 1.0:
                continue
            try:
                body_3d, body_3d_valid, triangulated_ratio = triangulate_body(
                    left,
                    right,
                    calibration,
                    args.joint_score_threshold,
                    args.min_disparity_px,
                    args.max_epipolar_y_delta_px,
                    args.min_depth_m,
                    args.max_depth_m,
                    args.min_triangulated_joints,
                )
            except RuntimeError:
                continue
            stereo_confidence = min(
                1.0,
                max(
                    0.0,
                    (left.mean_score + right.mean_score) * 0.5
                    * max(0.3, triangulated_ratio),
                ),
            )
            cost = base_pair_cost(left, right)
            hand_hint_gap_m = None
            continuity_gap_m = None
            selection_reason = "geometry_score"
            local_continuity_gap_m = track_memory._local_continuity_gap_m(body_3d)
            if hint is not None:
                hand_hint_gap_m = candidate_hand_gap_m(
                    body_3d,
                    hint.iphone_left_wrist,
                    hint.iphone_right_wrist,
                )
                if hand_hint_gap_m is not None:
                    cost += hand_hint_gap_m * 800.0
                    selection_reason = "iphone_hand_match"
                else:
                    continuity_gap_m = candidate_hand_gap_m(
                        body_3d,
                        hint.stereo_left_wrist,
                        hint.stereo_right_wrist,
                    )
                    if continuity_gap_m is not None:
                        cost += continuity_gap_m * 500.0
                        selection_reason = "stereo_continuity"
            if (
                hand_hint_gap_m is None
                and continuity_gap_m is None
                and local_continuity_gap_m is not None
            ):
                continuity_gap_m = local_continuity_gap_m
                cost += local_continuity_gap_m * 450.0
                selection_reason = "local_stereo_continuity"
            if hint is not None and hint.device_pose_candidate_track_id:
                nearest_cached_track_id = track_memory._nearest_cached_track_id(body_3d)
                if nearest_cached_track_id == hint.device_pose_candidate_track_id:
                    cost -= 700.0
                    if selection_reason == "geometry_score":
                        selection_reason = "device_pose_track_match"
            rank = pair_priority_rank(hand_hint_gap_m, continuity_gap_m, cost)
            candidates.append((
                rank,
                SelectedStereoCandidate(
                    left_index=left_index,
                    right_index=right_index,
                    left_candidate=left,
                    right_candidate=right,
                    body_3d=body_3d,
                    body_3d_valid=body_3d_valid,
                    triangulated_ratio=triangulated_ratio,
                    stereo_confidence=stereo_confidence,
                    operator_track_id="",
                    selection_reason=selection_reason,
                    hand_hint_gap_m=hand_hint_gap_m,
                    continuity_gap_m=continuity_gap_m,
                ),
            ))
    candidates.sort(
        key=lambda item: (
            item[0],
            -item[1].stereo_confidence,
            -item[1].triangulated_ratio,
        )
    )
    return candidates


def select_best_pairs(
    left_candidates: list[PersonCandidate],
    right_candidates: list[PersonCandidate],
    calibration: StereoCalibration,
    args: argparse.Namespace,
    hint: AssociationHint | None,
    track_memory: StereoTrackMemory,
) -> list[SelectedStereoCandidate]:
    ranked_candidates = enumerate_candidate_pairs(
        left_candidates,
        right_candidates,
        calibration,
        args,
        hint,
        track_memory,
    )
    selected: list[SelectedStereoCandidate] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    reserved_track_ids: set[str] = set()
    max_persons = max(int(getattr(args, "max_persons", 1) or 1), 1)

    for _, candidate in ranked_candidates:
        if candidate.left_index in used_left or candidate.right_index in used_right:
            continue
        track_id = track_memory.propose_track_id(
            candidate.body_3d,
            hint,
            candidate.hand_hint_gap_m,
            candidate.continuity_gap_m,
            reserved_track_ids,
        )
        candidate.operator_track_id = track_id
        reserved_track_ids.add(track_id)
        used_left.add(candidate.left_index)
        used_right.add(candidate.right_index)
        selected.append(candidate)
        if len(selected) >= max_persons:
            break

    selected = [stabilize_low_roi_candidate(candidate, track_memory) for candidate in selected]
    track_memory.commit_track_candidates(selected)
    return track_memory.inject_persisted_candidates(selected, max_persons)


def triangulate_body(
    left: PersonCandidate,
    right: PersonCandidate,
    calibration: StereoCalibration,
    joint_score_threshold: float,
    min_disparity_px: float,
    max_epipolar_y_delta_px: float,
    min_depth_m: float,
    max_depth_m: float,
    min_triangulated_joints: int,
) -> tuple[list[BodyPoint3D], list[bool], float]:
    body_3d: list[BodyPoint3D] = [None] * COCO_BODY_17
    valid_depths: list[float] = []
    low_roi_relaxed = left.source_tag.startswith("low_roi") or right.source_tag.startswith("low_roi")
    left_joint_threshold = (
        float(left.joint_score_threshold_override)
        if left.joint_score_threshold_override is not None
        else float(joint_score_threshold)
    )
    right_joint_threshold = (
        float(right.joint_score_threshold_override)
        if right.joint_score_threshold_override is not None
        else float(joint_score_threshold)
    )
    effective_min_disparity_px = (
        max(0.75, float(min_disparity_px) * 0.6)
        if low_roi_relaxed
        else float(min_disparity_px)
    )
    effective_max_epipolar_y_delta_px = max(0.0, float(max_epipolar_y_delta_px))
    if low_roi_relaxed:
        effective_max_epipolar_y_delta_px *= 1.5
    effective_min_depth_m = max(0.0, float(min_depth_m))

    for index in range(COCO_BODY_17):
        if left.scores[index] < left_joint_threshold or right.scores[index] < right_joint_threshold:
            continue
        epipolar_y_delta = abs(float(left.keypoints[index, 1]) - float(right.keypoints[index, 1]))
        if epipolar_y_delta > effective_max_epipolar_y_delta_px:
            continue
        disparity = float(left.keypoints[index, 0] - right.keypoints[index, 0])
        if disparity <= effective_min_disparity_px:
            continue
        depth = calibration.left.fx_px * calibration.baseline_m / disparity
        if not math.isfinite(depth) or depth < effective_min_depth_m or depth > max_depth_m:
            continue
        x = (float(left.keypoints[index, 0]) - calibration.left.cx_px) * depth / calibration.left.fx_px
        y = (calibration.left.cy_px - float(left.keypoints[index, 1])) * depth / calibration.left.fy_px
        body_3d[index] = [x, y, depth]
        valid_depths.append(depth)

    fallback_depth = torso_median_depth(body_3d) or (float(np.median(valid_depths)) if valid_depths else None)
    if fallback_depth is not None:
        for index, point in enumerate(body_3d):
            if point is None:
                continue
            if is_depth_outlier(float(point[2]), fallback_depth):
                body_3d[index] = None

    valid_joint_count = sum(point is not None for point in body_3d)
    if valid_joint_count < max(1, int(min_triangulated_joints)):
        raise RuntimeError("双目几何自洽关节不足，当前候选跳过")

    body_3d_valid = [point is not None for point in body_3d]
    triangulated_ratio = valid_joint_count / float(COCO_BODY_17)
    return body_3d, body_3d_valid, triangulated_ratio


def torso_median_depth(points: list[list[float] | None]) -> float | None:
    depths = [
        points[index][2]
        for index in TORSO_INDICES + WRIST_INDICES
        if index < len(points) and points[index] is not None
    ]
    if not depths:
        return None
    return float(np.median(np.asarray(depths, dtype=np.float32)))


def build_packet(
    args: argparse.Namespace,
    calibration: StereoCalibration,
    primary_candidate: SelectedStereoCandidate,
    selected_candidates: list[SelectedStereoCandidate],
    frame_seq: int,
    source_time_ns: int,
) -> dict:
    body_3d = primary_candidate.body_3d

    def serialize_body_points_3d(body_points: list[BodyPoint3D]) -> list[list[float]]:
        serialized: list[list[float]] = []
        for point in body_points[:COCO_BODY_17]:
            parsed = parse_hint_point(point)
            if parsed is None:
                serialized.append([0.0, 0.0, 0.0])
                continue
            serialized.append([float(parsed[0]), float(parsed[1]), float(parsed[2])])
        while len(serialized) < COCO_BODY_17:
            serialized.append([0.0, 0.0, 0.0])
        return serialized

    def missing_body_indices(valid_mask: list[bool]) -> list[int]:
        padded = list(valid_mask[:COCO_BODY_17])
        while len(padded) < COCO_BODY_17:
            padded.append(False)
        return [index for index, valid in enumerate(padded) if not valid]

    def normalize_body_points_2d(
        keypoints_2d: np.ndarray,
        width: int,
        height: int,
    ) -> list[list[float]]:
        normalized: list[list[float]] = []
        safe_width = max(1, width)
        safe_height = max(1, height)
        for point in np.asarray(keypoints_2d, dtype=np.float32)[:COCO_BODY_17]:
            if len(point) < 2:
                normalized.append([-1.0, -1.0])
                continue
            x, y = float(point[0]), float(point[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                normalized.append([-1.0, -1.0])
                continue
            normalized.append([
                max(0.0, min(1.0, x / float(safe_width))),
                max(0.0, min(1.0, y / float(safe_height))),
            ])
        while len(normalized) < COCO_BODY_17:
            normalized.append([-1.0, -1.0])
        return normalized

    def serialize_candidate(candidate: SelectedStereoCandidate) -> dict:
        return {
            "operator_track_id": candidate.operator_track_id,
            "body_kpts_3d": serialize_body_points_3d(candidate.body_3d),
            "body_kpts_3d_valid": candidate.body_3d_valid,
            "body_kpts_3d_missing_indices": missing_body_indices(candidate.body_3d_valid),
            "hand_kpts_3d": [],
            "left_body_kpts_2d": normalize_body_points_2d(
                candidate.left_candidate.keypoints,
                calibration.left.reference_image_w,
                calibration.left.reference_image_h,
            ),
            "right_body_kpts_2d": normalize_body_points_2d(
                candidate.right_candidate.keypoints,
                calibration.right.reference_image_w,
                calibration.right.reference_image_h,
            ),
            "stereo_confidence": round(float(candidate.stereo_confidence), 4),
            "selection": {
                "selection_reason": candidate.selection_reason,
                "source_tag_left": candidate.left_candidate.source_tag,
                "source_tag_right": candidate.right_candidate.source_tag,
                "hand_hint_gap_m": candidate.hand_hint_gap_m,
                "continuity_gap_m": candidate.continuity_gap_m,
            },
        }

    return {
        "type": "stereo_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": args.trip_id,
        "session_id": args.session_id,
        "device_id": args.device_id,
        "operator_track_id": primary_candidate.operator_track_id,
        "source_time_ns": int(source_time_ns),
        "left_frame_id": frame_seq,
        "right_frame_id": frame_seq,
        "body_layout": "coco_body_17",
        "body_kpts_3d": serialize_body_points_3d(body_3d),
        "body_kpts_3d_valid": primary_candidate.body_3d_valid,
        "body_kpts_3d_missing_indices": missing_body_indices(primary_candidate.body_3d_valid),
        "hand_kpts_3d": [],
        "left_body_kpts_2d": normalize_body_points_2d(
            primary_candidate.left_candidate.keypoints,
            calibration.left.reference_image_w,
            calibration.left.reference_image_h,
        ),
        "right_body_kpts_2d": normalize_body_points_2d(
            primary_candidate.right_candidate.keypoints,
            calibration.right.reference_image_w,
            calibration.right.reference_image_h,
        ),
        "stereo_confidence": round(float(primary_candidate.stereo_confidence), 4),
        "selection": {
            "selection_reason": primary_candidate.selection_reason,
            "source_tag_left": primary_candidate.left_candidate.source_tag,
            "source_tag_right": primary_candidate.right_candidate.source_tag,
            "hand_hint_gap_m": primary_candidate.hand_hint_gap_m,
            "continuity_gap_m": primary_candidate.continuity_gap_m,
        },
        "persons": [serialize_candidate(candidate) for candidate in selected_candidates],
        "calibration": {
            "sensor_frame": calibration.sensor_frame,
            "operator_frame": calibration.operator_frame,
            "extrinsic_version": calibration.extrinsic_version,
            "baseline_m": calibration.baseline_m,
            "capture_image_w": int(args.width),
            "capture_image_h": int(args.height),
            "inference_image_w": int(args.inference_width) if int(args.inference_width) > 0 else int(args.width),
            "inference_image_h": int(args.inference_height) if int(args.inference_height) > 0 else int(args.height),
            "model_mode": args.model_mode,
            "left_intrinsics": {
                "fx_px": calibration.left.fx_px,
                "fy_px": calibration.left.fy_px,
                "cx_px": calibration.left.cx_px,
                "cy_px": calibration.left.cy_px,
                "reference_image_w": calibration.left.reference_image_w,
                "reference_image_h": calibration.left.reference_image_h,
            },
            "right_intrinsics": {
                "fx_px": calibration.right.fx_px,
                "fy_px": calibration.right.fy_px,
                "cx_px": calibration.right.cx_px,
                "cy_px": calibration.right.cy_px,
                "reference_image_w": calibration.right.reference_image_w,
                "reference_image_h": calibration.right.reference_image_h,
            },
        },
    }


def save_preview(
    debug_preview: str,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    left_candidate: PersonCandidate | None,
    right_candidate: PersonCandidate | None,
    max_width: int,
    jpeg_quality: int,
) -> None:
    merged = np.hstack((left_frame, right_frame))
    width_offset = left_frame.shape[1]
    for candidate, offset in ((left_candidate, 0), (right_candidate, width_offset)):
        if candidate is None:
            continue
        for point, score in zip(candidate.keypoints, candidate.scores):
            if score < 0.25:
                continue
            cv2.circle(
                merged,
                (int(point[0]) + offset, int(point[1])),
                3,
                (0, 255, 0),
                -1,
            )
    if max_width > 0 and merged.shape[1] > max_width:
        scale = max_width / float(merged.shape[1])
        resized_h = max(1, int(round(merged.shape[0] * scale)))
        merged = cv2.resize(merged, (max_width, resized_h), interpolation=cv2.INTER_AREA)
    save_atomic_jpeg(debug_preview, merged, jpeg_quality)


def save_atomic_jpeg(target_path: str, image: np.ndarray, jpeg_quality: int) -> None:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(95, int(jpeg_quality)))],
    )
    if not ok:
        raise RuntimeError(f"failed to encode jpeg: {target_path}")
    tmp_path = target.with_suffix(f"{target.suffix}.tmp")
    tmp_path.write_bytes(encoded.tobytes())
    tmp_path.replace(target)


def save_high_res_debug_frames(
    debug_left_frame: str,
    debug_right_frame: str,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    jpeg_quality: int,
) -> None:
    if debug_left_frame:
        save_atomic_jpeg(debug_left_frame, left_frame, jpeg_quality)
    if debug_right_frame:
        save_atomic_jpeg(debug_right_frame, right_frame, jpeg_quality)


def send_packet(
    session: requests.Session,
    http_base: str,
    edge_token: str,
    payload: dict,
) -> None:
    debug_path = os.environ.get("CHEK_STEREO_PACKET_DEBUG_PATH", "").strip()
    if debug_path:
        summary = {
            "source_time_ns": payload.get("source_time_ns"),
            "left_len": len(payload.get("left_body_kpts_2d") or []),
            "right_len": len(payload.get("right_body_kpts_2d") or []),
            "left_null": payload.get("left_body_kpts_2d") is None,
            "right_null": payload.get("right_body_kpts_2d") is None,
        }
        with open(debug_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # `requests` posts to the edge axum server were dropping stereo 2D arrays into
    # `null` on ingest, while `urllib` preserved the same JSON byte-for-byte.
    request_body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        f"{http_base.rstrip('/')}/ingest/stereo_pose",
        data=request_body,
        headers={
            "Authorization": f"Bearer {edge_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status_code = getattr(response, "status", None) or response.getcode()
            if status_code >= 400:
                raise requests.HTTPError(
                    f"stereo pose ingest failed with status {status_code}"
                )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise requests.HTTPError(
            f"stereo pose ingest failed with status {exc.code}: {detail}"
        ) from exc


def measure_time_sync(
    session: requests.Session,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    headers = {"Authorization": f"Bearer {args.edge_token}"}
    best_sample: tuple[int, int] | None = None
    sample_count = max(int(args.time_sync_sample_count), 1)
    for _ in range(sample_count):
        t0 = time.monotonic_ns()
        edge_time_response = session.get(
            f"{args.http_base.rstrip('/')}/time",
            headers=headers,
            timeout=args.time_sync_timeout_s,
        )
        t1 = time.monotonic_ns()
        edge_time_response.raise_for_status()
        edge_time_ns = int((edge_time_response.json() or {}).get("edge_time_ns") or 0)
        rtt_ns = max(0, t1 - t0)
        clock_offset_ns = edge_time_ns - ((t0 + t1) // 2)
        if best_sample is None or rtt_ns < best_sample[1]:
            best_sample = (clock_offset_ns, rtt_ns)
    if best_sample is None:
        raise RuntimeError("failed to collect any /time sample")
    return best_sample[0], best_sample[1], sample_count


def post_time_sync(
    session: requests.Session,
    args: argparse.Namespace,
    clock_offset_ns: int,
    rtt_ns: int,
    sample_count: int,
) -> None:
    headers = {"Authorization": f"Bearer {args.edge_token}"}
    session.post(
        f"{args.http_base.rstrip('/')}/time/sync",
        headers=headers,
        json={
            "schema_version": "1.0.0",
            "trip_id": args.trip_id,
            "session_id": args.session_id,
            "device_id": args.device_id,
            "source_kind": "stereo_pair",
            "clock_domain": "stereo_monotonic_ns",
            "clock_offset_ns": int(clock_offset_ns),
            "rtt_ns": int(rtt_ns),
            "sample_count": int(sample_count),
        },
        timeout=args.time_sync_timeout_s,
    ).raise_for_status()


def ensure_session_ready(
    session: requests.Session,
    args: argparse.Namespace,
) -> tuple[int, int]:
    headers = {"Authorization": f"Bearer {args.edge_token}"}
    if args.ensure_session:
        session.post(
            f"{args.http_base.rstrip('/')}/session/start",
            headers=headers,
            json={
                "schema_version": "1.0.0",
                "trip_id": args.trip_id,
                "session_id": args.session_id,
                "device_id": args.device_id,
            },
            timeout=5,
        ).raise_for_status()
    clock_offset_ns, rtt_ns, sample_count = measure_time_sync(session, args)
    post_time_sync(session, args, clock_offset_ns, rtt_ns, sample_count)
    return clock_offset_ns, rtt_ns


def run_periodic_time_sync(
    stop_event: threading.Event,
    args: argparse.Namespace,
) -> None:
    interval_s = float(args.time_sync_interval_s)
    if interval_s <= 0:
        return
    client = requests.Session()
    try:
        while not stop_event.wait(interval_s):
            try:
                clock_offset_ns, rtt_ns, sample_count = measure_time_sync(client, args)
                post_time_sync(client, args, clock_offset_ns, rtt_ns, sample_count)
            except requests.RequestException as error:
                print(f"[time-sync] stereo refresh failed: {error}", file=sys.stderr)
            except Exception as error:  # pragma: no cover - runtime protection
                print(f"[time-sync] stereo refresh failed: {error}", file=sys.stderr)
    finally:
        client.close()


def should_stop(stop_flag: dict[str, bool]) -> bool:
    return stop_flag["stop"]


def install_signal_handlers(stop_flag: dict[str, bool]) -> None:
    def handle_signal(_signum: int, _frame: object) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main() -> int:
    args = parse_args()
    calibration = build_calibration(args)
    inference_width = int(args.inference_width) if int(args.inference_width) > 0 else int(args.width)
    inference_height = int(args.inference_height) if int(args.inference_height) > 0 else int(args.height)
    low_roi_inference_width = (
        int(args.low_roi_inference_width)
        if int(args.low_roi_inference_width) > 0
        else inference_width
    )
    low_roi_inference_height = (
        int(args.low_roi_inference_height)
        if int(args.low_roi_inference_height) > 0
        else inference_height
    )
    low_roi_body_score_threshold = (
        float(args.low_roi_body_score_threshold)
        if float(args.low_roi_body_score_threshold) > 0.0
        else float(args.body_score_threshold)
    )
    low_roi_joint_score_threshold = (
        float(args.low_roi_joint_score_threshold)
        if float(args.low_roi_joint_score_threshold) > 0.0
        else float(args.joint_score_threshold)
    )
    stop_flag = {"stop": False}
    install_signal_handlers(stop_flag)

    capture_worker = StereoCaptureWorker(args)
    capture_worker.start()
    tracker_solution = partial(
        Wholebody,
        det_input_size=(inference_width, inference_height),
    )
    left_estimator = PoseTracker(
        tracker_solution,
        det_frequency=max(args.det_frequency, 1),
        tracking=True,
        tracking_thr=args.tracking_thr,
        mode=args.model_mode,
        backend="onnxruntime",
        device="cpu",
    )
    right_estimator = PoseTracker(
        tracker_solution,
        det_frequency=max(args.det_frequency, 1),
        tracking=True,
        tracking_thr=args.tracking_thr,
        mode=args.model_mode,
        backend="onnxruntime",
        device="cpu",
    )
    left_roi_estimator = Wholebody(mode=args.model_mode, backend="onnxruntime", device="cpu")
    right_roi_estimator = Wholebody(mode=args.model_mode, backend="onnxruntime", device="cpu")
    client = requests.Session()
    clock_offset_ns, rtt_ns = ensure_session_ready(client, args)
    stop_event = threading.Event()
    sync_thread = None
    if float(args.time_sync_interval_s) > 0:
        sync_thread = threading.Thread(
            target=run_periodic_time_sync,
            args=(stop_event, args),
            name="stereo-time-sync",
            daemon=True,
        )
        sync_thread.start()
    raw_uploader = StereoRawChunkUploader(args)
    track_memory = StereoTrackMemory()
    parallel_workers = max(int(args.parallel_inference_workers), 1)
    inference_executor = (
        ThreadPoolExecutor(max_workers=parallel_workers)
        if parallel_workers > 1
        else None
    )

    print(
        f"[inference] capture={args.width}x{args.height} infer={inference_width}x{inference_height} mode={args.model_mode}",
        flush=True,
    )
    print(
        f"[inference] tracker det_frequency={max(args.det_frequency, 1)} tracking_thr={args.tracking_thr:.2f}",
        flush=True,
    )
    print(
        f"[inference] parallel_workers={parallel_workers}",
        flush=True,
    )
    print(
        json.dumps(
            {
                "time_sync": "ready",
                "clock_offset_ns": clock_offset_ns,
                "rtt_ns": rtt_ns,
                "sample_count": max(int(args.time_sync_sample_count), 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.low_roi_second_pass:
        print(
            f"[inference] low-roi enabled top={args.low_roi_top_fraction:.2f} "
            f"height={args.low_roi_height_fraction:.2f} left={args.low_roi_left_fraction:.2f} "
            f"width={args.low_roi_width_fraction:.2f} infer={low_roi_inference_width}x{low_roi_inference_height} "
            f"body_thr={low_roi_body_score_threshold:.2f} joint_thr={low_roi_joint_score_threshold:.2f} "
            f"interval={max(args.low_roi_second_pass_interval, 1)}",
            flush=True,
        )

    preview_writer = StereoDebugPreviewWriter(args, capture_worker)
    preview_writer.start()
    frame_seq = 0
    last_consumed_capture_seq: int | None = None

    try:
        while not should_stop(stop_flag):
            association_hint = None
            try:
                association_hint = fetch_association_hint(client, args)
            except requests.RequestException:
                association_hint = None
            snapshot = capture_worker.wait_for_latest(
                last_consumed_capture_seq,
                timeout_s=max(1.5 / max(args.fps, 0.1), 0.3),
            )
            if snapshot is None:
                time.sleep(0.05)
                continue
            last_consumed_capture_seq = snapshot.seq
            left_frame = snapshot.left_frame
            right_frame = snapshot.right_frame

            (left_keypoints, left_scores), (right_keypoints, right_scores) = run_dual_estimator_pass(
                inference_executor,
                left_estimator,
                left_frame,
                right_estimator,
                right_frame,
                inference_width,
                inference_height,
            )
            left_candidates = extract_candidates(
                left_keypoints, left_scores, args.body_score_threshold, "full_frame"
            )
            right_candidates = extract_candidates(
                right_keypoints, right_scores, args.body_score_threshold, "full_frame"
            )
            low_roi_interval = max(args.low_roi_second_pass_interval, 1)
            min_primary_candidates = min(len(left_candidates), len(right_candidates))
            need_low_roi_second_pass = (
                args.low_roi_second_pass
                and (
                    min_primary_candidates == 0
                    or (
                        min_primary_candidates == 1
                        and frame_seq % low_roi_interval == 0
                    )
                )
            )
            if need_low_roi_second_pass:
                low_roi_regions = build_low_roi_regions(
                    left_frame,
                    args.low_roi_top_fraction,
                    args.low_roi_height_fraction,
                    args.low_roi_left_fraction,
                    args.low_roi_width_fraction,
                )
                for roi_tag, low_roi in low_roi_regions:
                    if inference_executor is None:
                        left_roi_keypoints, left_roi_scores = run_estimator_on_roi(
                            left_roi_estimator,
                            left_frame,
                            low_roi,
                            low_roi_inference_width,
                            low_roi_inference_height,
                        )
                        right_roi_keypoints, right_roi_scores = run_estimator_on_roi(
                            right_roi_estimator,
                            right_frame,
                            low_roi,
                            low_roi_inference_width,
                            low_roi_inference_height,
                        )
                    else:
                        left_roi_future = inference_executor.submit(
                            run_estimator_on_roi,
                            left_roi_estimator,
                            left_frame,
                            low_roi,
                            low_roi_inference_width,
                            low_roi_inference_height,
                        )
                        right_roi_future = inference_executor.submit(
                            run_estimator_on_roi,
                            right_roi_estimator,
                            right_frame,
                            low_roi,
                            low_roi_inference_width,
                            low_roi_inference_height,
                        )
                        left_roi_keypoints, left_roi_scores = left_roi_future.result()
                        right_roi_keypoints, right_roi_scores = right_roi_future.result()
                    left_low_roi_candidates = extract_candidates(
                        left_roi_keypoints,
                        left_roi_scores,
                        low_roi_body_score_threshold,
                        roi_tag,
                        low_roi_joint_score_threshold,
                    )
                    right_low_roi_candidates = extract_candidates(
                        right_roi_keypoints,
                        right_roi_scores,
                        low_roi_body_score_threshold,
                        roi_tag,
                        low_roi_joint_score_threshold,
                    )
                    left_candidates = merge_candidate_passes(
                        left_candidates,
                        left_low_roi_candidates,
                    )
                    right_candidates = merge_candidate_passes(
                        right_candidates,
                        right_low_roi_candidates,
                    )
            pairs = select_best_pairs(
                left_candidates,
                right_candidates,
                calibration,
                args,
                association_hint,
                track_memory,
            )
            primary_pair = pairs[0] if pairs else None

            if primary_pair is None:
                preview_writer.clear_candidates()
                continue

            left_candidate = primary_pair.left_candidate
            right_candidate = primary_pair.right_candidate
            preview_writer.publish_candidates(left_candidate, right_candidate)

            source_time_ns = snapshot.source_time_ns
            preview_frame = np.hstack((left_frame, right_frame))
            raw_uploader.append(source_time_ns, left_frame, right_frame, preview_frame)
            payload = build_packet(
                args,
                calibration,
                primary_pair,
                pairs,
                frame_seq,
                source_time_ns,
            )
            send_packet(client, args.http_base, args.edge_token, payload)

            frame_seq += 1
            if args.max_frames and frame_seq >= args.max_frames:
                break
    finally:
        stop_event.set()
        if sync_thread is not None:
            sync_thread.join(timeout=1.0)
        preview_writer.close()
        capture_worker.close()
        raw_uploader.close()
        client.close()
        if inference_executor is not None:
            inference_executor.shutdown(wait=False, cancel_futures=True)
    return 0


def sleep_until(next_deadline: float, frame_interval: float) -> None:
    now = time.monotonic()
    wait_s = next_deadline - now
    if wait_s > 0:
        time.sleep(wait_s)
    else:
        time.sleep(min(frame_interval, 0.01))


if __name__ == "__main__":
    sys.exit(main())
