#!/usr/bin/env python3

import argparse
import copy
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

FORWARD_BIAS_M = {
    "nose": 0.01,
    "left_eye": 0.01,
    "right_eye": 0.01,
    "left_ear": 0.005,
    "right_ear": 0.005,
    "left_shoulder": 0.0,
    "right_shoulder": 0.0,
    "left_elbow": 0.02,
    "right_elbow": 0.02,
    "left_wrist": 0.035,
    "right_wrist": 0.035,
    "left_hip": -0.005,
    "right_hip": -0.005,
    "left_knee": -0.015,
    "right_knee": -0.015,
    "left_ankle": -0.025,
    "right_ankle": -0.025,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 Wi‑Fi DensePose 骨骼桥接到 edge-orchestrator 的 /ingest/wifi_pose。")
    parser.add_argument("--wifi-base-url", default="http://127.0.0.1:18080", help="Wi‑Fi sensing server HTTP 根地址。")
    parser.add_argument("--edge-base-url", default="http://127.0.0.1:8080", help="edge-orchestrator HTTP 根地址。")
    parser.add_argument("--edge-token", default="chek-ego-miner-local-token", help="edge 鉴权 token。")
    parser.add_argument("--trip-id", default="", help="trip_id。")
    parser.add_argument("--session-id", default="", help="session_id。")
    parser.add_argument("--device-id", default="wifi-pose-bridge-001", help="Wi‑Fi pose 设备 ID。")
    parser.add_argument("--operator-track-id", default="wifi-operator-main", help="写入 Edge 的 operator_track_id。")
    parser.add_argument("--association-hint-url", default="", help="可选：读取 edge /association/hint，用双目/手机手腕辅助选人。")
    parser.add_argument(
        "--source-label",
        default="wifi_densepose_body_backbone_bridge",
        help="source_label。默认按 body backbone 标记，供 teleop body 主链使用。",
    )
    parser.add_argument("--sensor-frame", default="wifi_pose_frame", help="Wi‑Fi 传感器坐标系名称。")
    parser.add_argument("--operator-frame", default="operator_frame", help="operator_frame 名称。")
    parser.add_argument("--extrinsic-version", default="wifi-ext-001", help="Wi‑Fi 外参版本。")
    parser.add_argument("--poll-ms", type=int, default=100, help="轮询周期，毫秒。")
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
    parser.add_argument("--person-id", type=int, default=-1, help="指定 person_id；默认取最高置信度。")
    parser.add_argument("--shoulder-width-m", type=float, default=0.16, help="归一化目标肩宽，米。")
    parser.add_argument("--base-depth-m", type=float, default=0.82, help="归一化后的基础前向深度，米。")
    parser.add_argument(
        "--layout-hold-ms",
        type=int,
        default=10000,
        help="layout 诊断保活窗口。节点在该时间内出现过，就继续计入 layout_node_count/layout_score。",
    )
    parser.add_argument(
        "--person-hold-ms",
        type=int,
        default=1200,
        help="Wi‑Fi 人体保活窗口。短时掉帧时继续沿用最近一次有效单人骨架，避免 teleop body 闪断。",
    )
    parser.add_argument("--once", action="store_true", help="只抓一帧并上送一次。")
    parser.add_argument("--verbose", action="store_true", help="输出调试信息。")
    return parser.parse_args()


def default_association_hint_url(edge_base_url: str) -> str:
    return urllib.parse.urljoin(edge_base_url.rstrip("/") + "/", "association/hint")


def http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def sync_time(args: argparse.Namespace) -> Tuple[int, int]:
    time_url = urllib.parse.urljoin(args.edge_base_url.rstrip("/") + "/", "time")
    sync_url = urllib.parse.urljoin(args.edge_base_url.rstrip("/") + "/", "time/sync")
    best_sample: Tuple[int, int] | None = None
    sample_count = max(int(args.time_sync_sample_count), 1)
    for _ in range(sample_count):
        t0 = time.monotonic_ns()
        time_resp = http_json("GET", time_url, token=args.edge_token)
        t1 = time.monotonic_ns()
        edge_time_ns = int(time_resp.get("edge_time_ns", 0))
        source_mid_ns = (t0 + t1) // 2
        clock_offset_ns = edge_time_ns - source_mid_ns
        rtt_ns = max(0, t1 - t0)
        if best_sample is None or rtt_ns < best_sample[1]:
            best_sample = (clock_offset_ns, rtt_ns)
    if best_sample is None:
        raise RuntimeError("failed to collect any /time sample")
    clock_offset_ns, rtt_ns = best_sample
    http_json(
        "POST",
        sync_url,
        {
            "schema_version": "1.0.0",
            "trip_id": args.trip_id,
            "session_id": args.session_id,
            "device_id": args.device_id,
            "source_kind": "wifi_pose",
            "clock_domain": "python_monotonic_ns",
            "clock_offset_ns": clock_offset_ns,
            "rtt_ns": rtt_ns,
            "sample_count": sample_count,
        },
        token=args.edge_token,
    )
    return clock_offset_ns, rtt_ns


def fetch_current_pose(args: argparse.Namespace) -> Dict[str, Any]:
    url = urllib.parse.urljoin(args.wifi_base_url.rstrip("/") + "/", "api/v1/pose/current")
    return http_json("GET", url)


def fetch_tracked_pose(args: argparse.Namespace) -> Dict[str, Any]:
    url = urllib.parse.urljoin(args.wifi_base_url.rstrip("/") + "/", "api/v1/pose/tracked")
    return http_json("GET", url)


def fetch_sensing_latest(args: argparse.Namespace) -> Dict[str, Any]:
    url = urllib.parse.urljoin(args.wifi_base_url.rstrip("/") + "/", "api/v1/sensing/latest")
    return http_json("GET", url)


def fetch_zone_summary(args: argparse.Namespace) -> Dict[str, Any]:
    url = urllib.parse.urljoin(args.wifi_base_url.rstrip("/") + "/", "api/v1/pose/zones/summary")
    return http_json("GET", url)


def fetch_stream_status(args: argparse.Namespace) -> Dict[str, Any]:
    url = urllib.parse.urljoin(args.wifi_base_url.rstrip("/") + "/", "api/v1/stream/status")
    return http_json("GET", url)


def fetch_vital_signs(args: argparse.Namespace) -> Dict[str, Any]:
    url = urllib.parse.urljoin(args.wifi_base_url.rstrip("/") + "/", "api/v1/vital-signs")
    return http_json("GET", url)


def fetch_association_hint(args: argparse.Namespace) -> Dict[str, Any]:
    url = args.association_hint_url or default_association_hint_url(args.edge_base_url)
    return http_json("GET", url, token=args.edge_token)


def pose_target_space(payload: Dict[str, Any]) -> str:
    tracked = payload.get("tracked_person")
    if isinstance(tracked, dict):
        target_space = tracked.get("target_space")
        if isinstance(target_space, str) and target_space:
            return target_space
    model_status = payload.get("model_status")
    if isinstance(model_status, dict):
        target_space = model_status.get("target_space")
        if isinstance(target_space, str) and target_space:
            return target_space
    return "wifi_pose_pixels"


def parse_hint_point(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        point = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in point):
        return None
    if all(abs(v) < 1e-6 for v in point):
        return None
    return point


def person_match_cost(normalized_body: List[List[float]], hint: Dict[str, Any]) -> Optional[float]:
    stereo = hint.get("stereo") or {}
    iphone = hint.get("iphone") or {}

    gaps: List[float] = []
    for body_index, hint_key in ((9, "left_wrist"), (10, "right_wrist")):
        if body_index >= len(normalized_body):
            continue
        point = normalized_body[body_index]
        if not point or all(abs(v) < 1e-6 for v in point):
            continue
        hint_point = parse_hint_point(stereo.get(hint_key))
        if hint_point is None:
            hint_point = parse_hint_point(iphone.get(hint_key))
        if hint_point is None:
            continue
        gaps.append(math.dist([float(v) for v in point], hint_point))

    for body_index, hint_key in ((5, "left_shoulder"), (6, "right_shoulder")):
        if body_index >= len(normalized_body):
            continue
        point = normalized_body[body_index]
        if not point or all(abs(v) < 1e-6 for v in point):
            continue
        hint_point = parse_hint_point(stereo.get(hint_key))
        if hint_point is None:
            continue
        gaps.append(math.dist([float(v) for v in point], hint_point) * 0.35)

    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def derive_layout_score(nodes: Any) -> Tuple[int, float]:
    if not isinstance(nodes, list):
        return 0, 0.0
    positions: List[List[float]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        position = node.get("position")
        if (
            isinstance(position, list)
            and len(position) == 3
            and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in position)
        ):
            positions.append([float(position[0]), float(position[1]), float(position[2])])
    node_count = len(positions)
    if node_count < 2:
        return node_count, 0.0

    unique_positions = {
        (round(position[0], 3), round(position[1], 3), round(position[2], 3))
        for position in positions
    }
    if len(unique_positions) < 2:
        return node_count, 0.0

    pairwise: List[float] = []
    for i in range(node_count):
        for j in range(i + 1, node_count):
            pairwise.append(math.dist(positions[i], positions[j]))
    if not pairwise:
        return node_count, 0.0
    mean_span = sum(pairwise) / len(pairwise)
    max_span = max(pairwise)

    xs = [position[0] for position in positions]
    zs = [position[2] for position in positions]
    centroid_x = sum(xs) / node_count
    centroid_z = sum(zs) / node_count
    width = max(xs) - min(xs)
    depth = max(zs) - min(zs)
    spread_area = width * depth

    quadrants = set()
    for position in positions:
        dx = position[0] - centroid_x
        dz = position[2] - centroid_z
        if abs(dx) < 0.12 and abs(dz) < 0.12:
            continue
        quadrants.add((dx >= 0.0, dz >= 0.0))

    node_score = min(1.0, len(unique_positions) / 4.0)
    coverage_score = min(1.0, len(quadrants) / 4.0)
    span_score = min(1.0, max_span / 3.0)
    area_score = min(1.0, spread_area / 4.0)
    balance_score = 0.0
    if width > 1e-6 and depth > 1e-6:
        balance_score = min(width, depth) / max(width, depth)

    score = (
        node_score * 0.38
        + coverage_score * 0.22
        + span_score * 0.22
        + area_score * 0.10
        + balance_score * 0.08
    )
    if node_count < 4:
        score *= 0.72 + 0.07 * node_count
    return node_count, round(min(1.0, score), 4)


class LayoutTracker:
    def __init__(self, hold_ms: int) -> None:
        self.hold_s = max(0.0, float(hold_ms) / 1000.0)
        self._recent_nodes: Dict[int, Tuple[float, List[float]]] = {}

    def update(self, nodes: Any) -> Tuple[int, float]:
        now = time.monotonic()
        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_id = node.get("node_id")
                position = node.get("position")
                if not isinstance(node_id, int):
                    continue
                if (
                    isinstance(position, list)
                    and len(position) == 3
                    and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in position)
                ):
                    self._recent_nodes[node_id] = (
                        now,
                        [float(position[0]), float(position[1]), float(position[2])],
                    )

        cutoff = now - self.hold_s
        self._recent_nodes = {
            node_id: (seen_at, position)
            for node_id, (seen_at, position) in self._recent_nodes.items()
            if seen_at >= cutoff
        }
        active_nodes = [
            {"node_id": node_id, "position": position}
            for node_id, (_seen_at, position) in sorted(self._recent_nodes.items())
        ]
        return derive_layout_score(active_nodes)


class PersonHoldTracker:
    def __init__(self, hold_ms: int) -> None:
        self.hold_s = max(0.0, float(hold_ms) / 1000.0)
        self._last_good: Optional[
            Tuple[
                float,
                Dict[str, Any],
                Tuple[List[List[float]], List[List[float]], float, str, str, str],
                int,
                Optional[str],
                Optional[str],
            ]
        ] = None

    def remember(
        self,
        person: Dict[str, Any],
        normalized: Tuple[List[List[float]], List[List[float]], float, str, str, str],
        total_persons: int,
        selection_reason: Optional[str],
        operator_track_id: Optional[str],
    ) -> None:
        self._last_good = (
            time.monotonic(),
            copy.deepcopy(person),
            copy.deepcopy(normalized),
            int(total_persons),
            selection_reason,
            operator_track_id,
        )

    def recall(
        self,
    ) -> Optional[
        Tuple[
            Dict[str, Any],
            Tuple[List[List[float]], List[List[float]], float, str, str, str],
            int,
            Optional[str],
            Optional[str],
        ]
    ]:
        if self._last_good is None:
            return None
        seen_at, person, normalized, total_persons, selection_reason, operator_track_id = self._last_good
        if time.monotonic() - seen_at > self.hold_s:
            self._last_good = None
            return None
        return (
            copy.deepcopy(person),
            copy.deepcopy(normalized),
            int(total_persons),
            selection_reason,
            operator_track_id,
        )


def association_track_candidates(hint: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(hint, dict):
        return []
    candidates: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in candidates:
            candidates.append(value)

    for key in (
        "iphone_operator_track_id",
        "selected_operator_track_id",
        "stereo_operator_track_id",
        "wifi_operator_track_id",
    ):
        add(hint.get(key))

    for nested_key in ("stereo", "wifi", "iphone"):
        nested = hint.get(nested_key)
        if isinstance(nested, dict):
            add(nested.get("operator_track_id"))

    return candidates


def continuity_cost_m(
    current_body: List[List[float]],
    previous_body: List[List[float]],
) -> Optional[float]:
    gaps: List[float] = []
    for index in (5, 6, 9, 10, 11, 12):
        if index >= len(current_body) or index >= len(previous_body):
            continue
        current = current_body[index]
        previous = previous_body[index]
        if not current or not previous:
            continue
        if all(abs(v) < 1e-6 for v in current) or all(abs(v) < 1e-6 for v in previous):
            continue
        gaps.append(math.dist([float(v) for v in current], [float(v) for v in previous]))
    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def derive_zone_score(zone_summary: Dict[str, Any]) -> Tuple[float, bool]:
    zones = zone_summary.get("zones")
    if not isinstance(zones, dict) or not zones:
        return 0.0, False
    counts: List[int] = []
    statuses: List[str] = []
    energies: List[float] = []
    peaks: List[float] = []
    hotspot_counts: List[int] = []
    rich_schema = False
    for value in zones.values():
        if not isinstance(value, dict):
            continue
        counts.append(int(value.get("person_count", 0)))
        statuses.append(str(value.get("status", "")))
        if "energy" in value or "peak" in value or "hotspot_count" in value:
            rich_schema = True
        try:
            energies.append(float(value.get("energy", 0.0) or 0.0))
        except (TypeError, ValueError):
            energies.append(0.0)
        try:
            peaks.append(float(value.get("peak", 0.0) or 0.0))
        except (TypeError, ValueError):
            peaks.append(0.0)
        hotspot_counts.append(int(value.get("hotspot_count", 0) or 0))
    if not counts:
        return 0.0, False
    default_placeholder = (
        len(counts) == 4
        and counts[0] in (0, 1)
        and sum(counts[1:]) == 0
        and statuses[:1] == ["monitored"]
        and all(status == "clear" for status in statuses[1:])
    )
    if default_placeholder:
        return 0.0, False

    summary_obj = zone_summary.get("summary") if isinstance(zone_summary.get("summary"), dict) else {}
    summary_hotspot_count = int(summary_obj.get("hotspot_count", sum(hotspot_counts)) or 0)

    if rich_schema:
        top_energy = max(energies) if energies else 0.0
        sorted_energies = sorted(energies, reverse=True)
        second_energy = sorted_energies[1] if len(sorted_energies) > 1 else 0.0
        top_peak = max(peaks) if peaks else 0.0
        sorted_peaks = sorted(peaks, reverse=True)
        second_peak = sorted_peaks[1] if len(sorted_peaks) > 1 else 0.0
        occupied_zones = sum(1 for count in counts if count > 0) or sum(1 for status in statuses if status == "occupied")
        candidate_zones = sum(1 for status in statuses if status in ("occupied", "candidate"))
        strong_zone_count = sum(
            1 for energy, peak in zip(energies, peaks)
            if energy >= 0.18 or peak >= 0.18
        )

        if top_energy <= 1e-6 and top_peak <= 1e-6 and summary_hotspot_count <= 0 and sum(counts) <= 0:
            return 0.0, True

        dominance = max(0.0, min(1.0, (top_energy - second_energy) / max(top_energy, 0.18)))
        peak_strength = max(0.0, min(1.0, top_peak / 0.65))
        focus_score = 1.0 if occupied_zones == 1 else max(0.0, 1.0 - occupied_zones * 0.35)
        if occupied_zones == 0:
            focus_score = 0.58 if candidate_zones == 1 else max(0.0, 0.4 - (candidate_zones - 1) * 0.15)
        hotspot_focus = 1.0 if summary_hotspot_count <= 1 else max(0.0, 1.0 - (summary_hotspot_count - 1) * 0.22)
        ambiguity_penalty = max(0.0, 1.0 - max(0, strong_zone_count - 1) * 0.16)
        score = (
            focus_score * 0.34
            + dominance * 0.22
            + peak_strength * 0.24
            + hotspot_focus * 0.12
            + ambiguity_penalty * 0.08
        )
        return round(max(0.0, min(1.0, score)), 4), True

    total = sum(max(0, count) for count in counts)
    if total <= 0:
        return 0.0, True
    return round(max(counts) / float(total), 4), True


def derive_wifi_diagnostics(
    sensing_latest: Dict[str, Any],
    zone_summary: Dict[str, Any],
    stream_status: Dict[str, Any],
    vital_signs: Dict[str, Any],
    layout_tracker: Optional[LayoutTracker],
) -> Dict[str, Any]:
    if layout_tracker is None:
        node_count, layout_score = derive_layout_score(sensing_latest.get("nodes"))
    else:
        node_count, layout_score = layout_tracker.update(sensing_latest.get("nodes"))
    zone_score, zone_summary_reliable = derive_zone_score(zone_summary)
    features = sensing_latest.get("features") if isinstance(sensing_latest.get("features"), dict) else {}
    diagnostics = {
        "layout_node_count": node_count,
        "layout_score": layout_score,
        "zone_score": zone_score,
        "zone_summary_reliable": zone_summary_reliable,
        "motion_energy": float(features.get("motion_band_power", 0.0) or 0.0),
        "doppler_hz": abs(float(features.get("dominant_freq_hz", 0.0) or 0.0)),
        "signal_quality": float(
            sensing_latest.get("signal_quality_score", sensing_latest.get("classification", {}).get("confidence", 0.0))
            or 0.0
        ),
        "stream_fps": float(stream_status.get("fps", 0.0) or 0.0),
        "vital_signal_quality": None,
    }
    vital_obj = vital_signs.get("vital_signs") if isinstance(vital_signs.get("vital_signs"), dict) else {}
    if "signal_quality" in vital_obj:
        try:
            diagnostics["vital_signal_quality"] = float(vital_obj.get("signal_quality"))
        except (TypeError, ValueError):
            diagnostics["vital_signal_quality"] = None
    return diagnostics


def select_person(
    payload: Dict[str, Any],
    person_id: int,
    args: argparse.Namespace,
    hint: Optional[Dict[str, Any]],
    held_reference: Optional[
        Tuple[
            Dict[str, Any],
            Tuple[List[List[float]], List[List[float]], float, str, str, str],
            int,
            Optional[str],
            Optional[str],
        ]
    ],
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Tuple[List[List[float]], List[List[float]], float, str, str, str]],
    Optional[str],
]:
    persons = payload.get("persons") or []
    if not isinstance(persons, list) or not persons:
        return None, None, None
    target_space = pose_target_space(payload)
    if person_id >= 0:
        for person in persons:
            if int(person.get("id", -1)) == person_id:
                normalized = normalize_body(person, args, target_space)
                return person, normalized, "fixed_person_id"
        return None, None, None

    held_person = held_reference[0] if held_reference is not None else None
    held_normalized = held_reference[1] if held_reference is not None else None
    held_person_id = int(held_person.get("id", -1)) if isinstance(held_person, dict) else -1
    best: Optional[
        Tuple[
            int,
            int,
            float,
            float,
            float,
            Dict[str, Any],
            Optional[Tuple[List[List[float]], List[List[float]], float, str, str, str]],
            Optional[str],
        ]
    ] = None
    for person in persons:
        normalized = normalize_body(person, args, target_space)
        person_conf = float(person.get("confidence", 0.0))
        match_reason = None
        cost = 9999.0
        source_rank = 2
        if hint is not None and normalized is not None:
            match_cost = person_match_cost(normalized[0], hint)
            if match_cost is not None:
                cost = match_cost
                match_reason = "stereo_or_iphone_hint"
                source_rank = 0
        if match_reason is None:
            cost = 1.0 - person_conf
            match_reason = "max_confidence"
            source_rank = 1

        same_person_rank = 1
        continuity_cost = 9999.0
        if held_normalized is not None and normalized is not None:
            continuity = continuity_cost_m(normalized[0], held_normalized[0])
            if continuity is not None:
                continuity_cost = continuity
        if int(person.get("id", -9999)) == held_person_id:
            same_person_rank = 0
        elif continuity_cost <= 0.18:
            same_person_rank = 0

        candidate = (
            source_rank,
            same_person_rank,
            cost,
            continuity_cost,
            -person_conf,
            person,
            normalized,
            match_reason,
        )
        if best is None or candidate[:5] < best[:5]:
            best = candidate
    if best is None:
        return None, None, None
    return best[5], best[6], best[7]


def ordered_keypoints(person: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    keypoints = person.get("keypoints")
    if not isinstance(keypoints, list) or not keypoints:
        return None
    by_name = {
        str(item.get("name")): item
        for item in keypoints
        if isinstance(item, dict) and item.get("name") is not None
    }
    if all(name in by_name for name in COCO17_NAMES):
        return [by_name[name] for name in COCO17_NAMES]
    if len(keypoints) >= len(COCO17_NAMES):
        return keypoints[: len(COCO17_NAMES)]
    return None


def tracked_person_contract(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tracked = payload.get("tracked_person")
    if not isinstance(tracked, dict):
        return None
    track_id = tracked.get("track_id")
    if not isinstance(track_id, str) or not track_id:
        return None
    gate = tracked.get("coherence_gate_decision")
    if isinstance(gate, str) and gate.lower() == "reject":
        return None
    lifecycle = tracked.get("lifecycle_state")
    if isinstance(lifecycle, str) and lifecycle.lower() == "terminated":
        return None
    return tracked


def tracked_contract_to_person(contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    keypoints = contract.get("raw_keypoints") or contract.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) < len(COCO17_NAMES):
        body = contract.get("raw_body_kpts_3d") or contract.get("body_kpts_3d")
        if not isinstance(body, list) or len(body) < len(COCO17_NAMES):
            return None
        keypoints = []
        confidence = float(contract.get("person_confidence", 0.0))
        for idx, name in enumerate(COCO17_NAMES):
            point = body[idx]
            if not isinstance(point, list) or len(point) < 3:
                return None
            keypoints.append(
                {
                    "name": name,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "confidence": confidence,
                }
            )
    return {
        "id": int(contract.get("source_person_id", 0)),
        "confidence": float(contract.get("person_confidence", 0.0)),
        "keypoints": keypoints,
        "bbox": contract.get("bbox", {}),
        "zone": contract.get("zone", "tracked"),
    }


def tracked_contract_to_normalized(contract: Dict[str, Any]) -> Optional[Tuple[List[List[float]], List[List[float]], float, str, str, str]]:
    canonical = contract.get("canonical_body_kpts_3d")
    raw = contract.get("raw_body_kpts_3d") or contract.get("body_kpts_3d")
    if not isinstance(canonical, list) or len(canonical) < len(COCO17_NAMES):
        return None
    if not isinstance(raw, list) or len(raw) < len(COCO17_NAMES):
        return None
    try:
        normalized = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in canonical[: len(COCO17_NAMES)]
        ]
        raw_points = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in raw[: len(COCO17_NAMES)]
        ]
    except (TypeError, ValueError, IndexError):
        return None
    confidence = float(contract.get("person_confidence", 0.0))
    raw_space = str(
        contract.get("raw_target_space")
        or contract.get("target_space")
        or pose_target_space({"tracked_person": contract})
    )
    notes = "wifi_pose_bridge forwarded tracked_person canonical body contract"
    return (
        normalized,
        raw_points,
        confidence,
        "canonical_body_frame",
        raw_space,
        notes,
    )


def point3_from_keypoint(keypoint: Dict[str, Any]) -> List[float]:
    return [
        float(keypoint.get("x", 0.0)),
        float(keypoint.get("y", 0.0)),
        float(keypoint.get("z", 0.0)),
    ]


def dist2(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return math.hypot(float(a.get("x", 0.0)) - float(b.get("x", 0.0)), float(a.get("y", 0.0)) - float(b.get("y", 0.0)))


def normalize_body(
    person: Dict[str, Any],
    args: argparse.Namespace,
    target_space: str,
) -> Optional[Tuple[List[List[float]], List[List[float]], float, str, str, str]]:
    points = ordered_keypoints(person)
    if points is None:
        return None

    raw = [point3_from_keypoint(point) for point in points]
    by_name = {name: point for name, point in zip(COCO17_NAMES, points)}
    left_shoulder = by_name.get("left_shoulder")
    right_shoulder = by_name.get("right_shoulder")
    left_hip = by_name.get("left_hip")
    right_hip = by_name.get("right_hip")
    if not left_shoulder or not right_shoulder or not left_hip or not right_hip:
        return None

    confidence_sum = 0.0
    confidence_count = 0
    for point in points:
        confidence = float(point.get("confidence", 0.0))
        if math.isfinite(confidence):
            confidence_sum += confidence
            confidence_count += 1
    mean_kpt_conf = confidence_sum / confidence_count if confidence_count > 0 else 0.0
    person_conf = float(person.get("confidence", 0.0))
    body_conf = max(0.0, min(1.0, mean_kpt_conf * person_conf))

    if target_space == "operator_frame":
        normalized = [[round(p[0], 6), round(p[1], 6), round(p[2], 6)] for p in raw]
        return (
            normalized,
            normalized,
            body_conf,
            "operator_frame",
            "operator_frame",
            "wifi_pose_bridge forwarded pose/current operator_frame model output",
        )

    hip_center_x = (float(left_hip.get("x", 0.0)) + float(right_hip.get("x", 0.0))) * 0.5
    hip_center_y = (float(left_hip.get("y", 0.0)) + float(right_hip.get("y", 0.0))) * 0.5
    torso_center_z = (
        float(left_shoulder.get("z", 0.0))
        + float(right_shoulder.get("z", 0.0))
        + float(left_hip.get("z", 0.0))
        + float(right_hip.get("z", 0.0))
    ) * 0.25

    shoulder_px = dist2(left_shoulder, right_shoulder)
    if not math.isfinite(shoulder_px) or shoulder_px <= 1e-6:
        return None
    meters_per_px = args.shoulder_width_m / shoulder_px

    normalized: List[List[float]] = []
    for name, point in zip(COCO17_NAMES, points):
        x_m = (float(point.get("x", 0.0)) - hip_center_x) * meters_per_px
        y_m = (hip_center_y - float(point.get("y", 0.0))) * meters_per_px
        z_delta = (float(point.get("z", 0.0)) - torso_center_z) * 0.12
        z_m = args.base_depth_m + z_delta + FORWARD_BIAS_M.get(name, 0.0)
        normalized.append([round(x_m, 6), round(y_m, 6), round(z_m, 6)])
    return (
        normalized,
        raw,
        body_conf,
        "canonical_body_frame",
        "wifi_pose_pixels",
        "wifi_pose_bridge normalized from pose/current pixels as teleop body backbone",
    )


def post_wifi_pose(
    args: argparse.Namespace,
    source_time_ns: int,
    normalized_body: List[List[float]],
    raw_body: List[List[float]],
    body_confidence: float,
    body_space: str,
    raw_body_space: str,
    calibration_notes: str,
    person: Dict[str, Any],
    total_persons: int,
    operator_track_id: str,
    selection_reason: Optional[str],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    url = urllib.parse.urljoin(args.edge_base_url.rstrip("/") + "/", "ingest/wifi_pose")
    payload = {
        "schema_version": "1.0.0",
        "trip_id": args.trip_id,
        "session_id": args.session_id,
        "device_id": args.device_id,
        "operator_track_id": operator_track_id,
        "source_time_ns": source_time_ns,
        "body_layout": "coco_body_17",
        "body_space": body_space,
        "body_kpts_3d": normalized_body,
        "body_confidence": body_confidence,
        "source_label": args.source_label,
        "selection_reason": selection_reason,
        "person_id": int(person.get("id", 0)),
        "total_persons": total_persons,
        "raw_body_layout": "coco_body_17",
        "raw_body_space": raw_body_space,
        "raw_body_kpts_3d": raw_body,
        "calibration": {
            "sensor_frame": args.sensor_frame,
            "operator_frame": args.operator_frame,
            "extrinsic_version": args.extrinsic_version,
            "notes": calibration_notes,
        },
        "diagnostics": diagnostics,
    }
    return http_json("POST", url, payload, token=args.edge_token)


def run_once(args: argparse.Namespace) -> int:
    return run_once_with_tracker(args, None, None)


def run_once_with_tracker(
    args: argparse.Namespace,
    layout_tracker: Optional[LayoutTracker],
    person_tracker: Optional[PersonHoldTracker],
) -> int:
    tracked_payload: Dict[str, Any] = {}
    try:
        tracked_payload = fetch_tracked_pose(args)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        tracked_payload = {}
    pose_payload = fetch_current_pose(args)
    sensing_latest = fetch_sensing_latest(args)
    zone_summary = fetch_zone_summary(args)
    stream_status = fetch_stream_status(args)
    vital_signs = fetch_vital_signs(args)
    diagnostics = derive_wifi_diagnostics(
        sensing_latest,
        zone_summary,
        stream_status,
        vital_signs,
        layout_tracker,
    )
    diagnostics["pose_target_space"] = pose_target_space(pose_payload)
    association_hint = None
    try:
        association_hint = fetch_association_hint(args)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        association_hint = None
    held_reference = person_tracker.recall() if person_tracker is not None else None
    tracked_contract = tracked_person_contract(tracked_payload)
    if tracked_contract is None:
        tracked_contract = tracked_person_contract(pose_payload)
    if tracked_contract is not None:
        tracked_person = tracked_contract_to_person(tracked_contract)
        normalized = tracked_contract_to_normalized(tracked_contract)
        tracked_target = str(
            tracked_contract.get("raw_target_space")
            or tracked_contract.get("target_space")
            or pose_target_space(pose_payload)
        )
        if tracked_person is not None:
            if normalized is None:
                normalized = normalize_body(tracked_person, args, tracked_target)
            if normalized is not None:
                person = tracked_person
                selection_reason = "tracked_person_contract"
            else:
                person, normalized, selection_reason = select_person(
                    pose_payload,
                    args.person_id,
                    args,
                    association_hint,
                    held_reference,
                )
                tracked_contract = None
        else:
            person, normalized, selection_reason = select_person(
                pose_payload,
                args.person_id,
                args,
                association_hint,
                held_reference,
            )
            tracked_contract = None
    else:
        person, normalized, selection_reason = select_person(
            pose_payload,
            args.person_id,
            args,
            association_hint,
            held_reference,
        )
    total_persons = int(pose_payload.get("total_persons", 1))
    diagnostics["person_hold_reused"] = False
    diagnostics["person_hold_ms"] = int(getattr(args, "person_hold_ms", 0) or 0)
    diagnostics["tracked_person_contract"] = (
        {
            "track_id": tracked_contract.get("track_id"),
            "lifecycle_state": tracked_contract.get("lifecycle_state"),
            "coherence_gate_decision": tracked_contract.get("coherence_gate_decision"),
            "target_space": tracked_contract.get("target_space"),
        }
        if tracked_contract is not None
        else None
    )
    diagnostics["tracked_pose_endpoint_used"] = tracked_person_contract(tracked_payload) is not None

    if person is None or normalized is None:
        if held_reference is not None:
            person, normalized, total_persons, previous_reason, held_track_id = held_reference
            selection_reason = "held_last_person" if previous_reason else "held_last_person"
            diagnostics["person_hold_reused"] = True
        else:
            held_track_id = None
    else:
        held_track_id = held_reference[4] if held_reference is not None else None
    if person is None:
        print("未获取到可用 person。", file=sys.stderr)
        return 2
    if normalized is None:
        print("Wi‑Fi 骨骼无法归一化为 COCO17 operator_frame。", file=sys.stderr)
        return 3
    normalized_body, raw_body, body_confidence, body_space, raw_body_space, calibration_notes = normalized
    operator_track_id = args.operator_track_id
    preferred_track_candidates = association_track_candidates(association_hint)
    selected_person_id = int(person.get("id", -1))
    held_person_id = int(held_reference[0].get("id", -2)) if held_reference is not None else -2
    if tracked_contract is not None and isinstance(tracked_contract.get("track_id"), str):
        operator_track_id = str(tracked_contract.get("track_id"))
    elif selection_reason == "stereo_or_iphone_hint" and preferred_track_candidates:
        operator_track_id = preferred_track_candidates[0]
    elif selected_person_id == held_person_id and held_track_id:
        operator_track_id = held_track_id
    elif preferred_track_candidates:
        operator_track_id = preferred_track_candidates[0]
    if operator_track_id == args.operator_track_id:
        operator_track_id = f"wifi-person-{int(person.get('id', 0))}"
    diagnostics["preferred_hint_track_id"] = preferred_track_candidates[0] if preferred_track_candidates else None
    diagnostics["held_operator_track_id"] = held_track_id
    diagnostics["selection_reason"] = selection_reason
    source_time_ns = time.monotonic_ns()
    if person_tracker is not None and not diagnostics["person_hold_reused"]:
        person_tracker.remember(person, normalized, total_persons, selection_reason, operator_track_id)
    response = post_wifi_pose(
        args,
        source_time_ns,
        normalized_body,
        raw_body,
        body_confidence,
        body_space,
        raw_body_space,
        calibration_notes,
        person,
        total_persons,
        operator_track_id,
        selection_reason,
        diagnostics,
    )
    if args.verbose:
        print(json.dumps(
            {
                "ok": response.get("ok", False),
                "person_id": int(person.get("id", 0)),
                "total_persons": int(pose_payload.get("total_persons", 1)),
                "operator_track_id": operator_track_id,
                "selection_reason": selection_reason,
                "body_confidence": round(body_confidence, 4),
                "left_wrist": normalized_body[9],
                "right_wrist": normalized_body[10],
                "wifi_diagnostics": diagnostics,
            },
            ensure_ascii=False,
        ))
    return 0 if response.get("ok") else 4


def main() -> int:
    args = parse_args()
    try:
        clock_offset_ns, rtt_ns = sync_time(args)
        if args.verbose:
            print(
                json.dumps(
                    {
                        "clock_offset_ns": clock_offset_ns,
                        "rtt_ns": rtt_ns,
                        "wifi_base_url": args.wifi_base_url,
                        "edge_base_url": args.edge_base_url,
                    },
                    ensure_ascii=False,
                )
            )
        if args.once:
            return run_once(args)

        layout_tracker = LayoutTracker(args.layout_hold_ms)
        person_tracker = PersonHoldTracker(args.person_hold_ms)
        next_time_sync_at = (
            time.monotonic() + float(args.time_sync_interval_s)
            if float(args.time_sync_interval_s) > 0
            else float("inf")
        )
        while True:
            started_at = time.monotonic()
            try:
                if time.monotonic() >= next_time_sync_at:
                    clock_offset_ns, rtt_ns = sync_time(args)
                    if args.verbose:
                        print(
                            json.dumps(
                                {
                                    "time_sync": "refresh",
                                    "clock_offset_ns": clock_offset_ns,
                                    "rtt_ns": rtt_ns,
                                },
                                ensure_ascii=False,
                            )
                        )
                exit_code = run_once_with_tracker(args, layout_tracker, person_tracker)
                if exit_code != 0 and args.verbose:
                    print(f"本轮上送失败，exit_code={exit_code}", file=sys.stderr)
            except urllib.error.HTTPError as error:
                print(f"HTTP 错误: {error}", file=sys.stderr)
            except urllib.error.URLError as error:
                print(f"网络错误: {error}", file=sys.stderr)
            except Exception as error:
                print(f"桥接异常: {error}", file=sys.stderr)
            finally:
                if float(args.time_sync_interval_s) > 0 and time.monotonic() >= next_time_sync_at:
                    next_time_sync_at = time.monotonic() + max(
                        float(args.time_sync_interval_s), 0.5
                    )

            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            sleep_ms = max(0.0, args.poll_ms - elapsed_ms)
            time.sleep(sleep_ms / 1000.0)
    except KeyboardInterrupt:
        return 0
    except urllib.error.HTTPError as error:
        print(f"HTTP 错误: {error}", file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print(f"网络错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
