#!/usr/bin/env python3
"""LEAP retarget 标定工具。

用途：
1. 从 Edge `/stream/teleop` 采集 `open_palm / closed_fist` 两组 16 维手部目标。
2. 基于采集结果求解左右手 `scale + offset` 标定。
3. 将结果写回 `config/leap.toml` 的 `left/right_joint_scale` 与 `left/right_joint_offset`。

说明：
- 该工具只校准 bridge 内部 LEAP 命令层，不改变 Edge 输出的手部事实。
- 自动求解主要覆盖手指屈曲链；`thumb_yaw` 默认保持 1:1，不做自动增益放大。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime as _dt
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import websockets
except ImportError:  # pragma: no cover - 运行时兜底
    websockets = None


LEAP_COMMAND_DIM = 16
DEFAULT_WS_URL = "ws://127.0.0.1:8765/stream/teleop?token=chek-ego-miner-local-token"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8090/health"
POSE_SEQUENCE = ("open_palm", "closed_fist")
POSE_HINTS = {
    "open_palm": "请摆出张开手掌、五指自然伸直的姿态。",
    "closed_fist": "请摆出稳定握拳姿态，保持 2 秒不要抖动。",
}
HAND_LAYOUTS = {"anatomical_target_16", "anatomical_joint_16", None}
FLEX_CLOSE_REFERENCE = [
    None,
    0.90,
    1.20,
    1.00,
    1.10,
    1.40,
    1.00,
    1.10,
    1.40,
    1.00,
    1.10,
    1.40,
    1.00,
    1.10,
    1.40,
    1.00,
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEAP retarget 标定采集与写回工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="从 /stream/teleop 采集标定样本")
    add_capture_arguments(capture)

    apply = subparsers.add_parser("apply", help="根据采样文件求解标定并写回 leap.toml")
    add_apply_arguments(apply)

    wizard = subparsers.add_parser("wizard", help="先采集再写回")
    add_capture_arguments(wizard, output_name="capture_output")
    add_apply_arguments(wizard)

    local_wizard = subparsers.add_parser("local-wizard", help="本机一键采集、写回并校验运行态")
    add_capture_arguments(local_wizard, output_name="capture_output")
    add_apply_arguments(local_wizard)
    local_wizard.add_argument(
        "--capture-file",
        help="可选：复用已有采样文件，传入后跳过实时采集",
    )
    local_wizard.add_argument(
        "--stack-script",
        default=str(repo_root() / "scripts/teleop_local_stack.sh"),
        help="本机联调栈脚本路径，用于重启 leap bridge",
    )
    local_wizard.add_argument(
        "--health-url",
        default=DEFAULT_HEALTH_URL,
        help="重启后校验运行态的 health 地址",
    )
    local_wizard.add_argument(
        "--health-timeout-seconds",
        type=float,
        default=20.0,
        help="等待运行态重新加载标定的超时时间",
    )
    local_wizard.add_argument(
        "--no-restart-stack",
        action="store_true",
        help="写回后不自动执行 teleop_local_stack.sh restart",
    )

    return parser.parse_args()


def add_capture_arguments(parser: argparse.ArgumentParser, output_name: str = "output") -> None:
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="teleop WebSocket 地址")
    parser.add_argument(
        f"--{output_name.replace('_', '-')}",
        dest=output_name,
        default="target/calibration/leap_retarget_capture.json",
        help="输出采样文件路径",
    )
    parser.add_argument("--samples", type=int, default=12, help="每个姿态平均采样帧数")
    parser.add_argument("--timeout-seconds", type=float, default=8.0, help="每个姿态采样超时")
    parser.add_argument(
        "--auto-advance-seconds",
        type=float,
        default=0.0,
        help="若大于 0，则每个姿态自动倒计时后开始采集，避免手工回车",
    )


def add_apply_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capture", required=False, help="采样文件路径")
    parser.add_argument(
        "--config",
        default="config/leap.toml",
        help="要写回的 leap.toml 路径",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right", "both"],
        default="both",
        help="仅写回某一侧，默认双手",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.5,
        help="自动求解 scale 的最小值",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=2.5,
        help="自动求解 scale 的最大值",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写回配置")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="写回配置时不生成备份文件",
    )
    parser.add_argument(
        "--report-json",
        help="可选：将求解结果写到报告 JSON",
    )


def main() -> int:
    args = parse_args()
    if args.command == "capture":
        return asyncio.run(capture_main(args))
    if args.command == "apply":
        if not args.capture:
            raise SystemExit("apply 子命令必须通过 --capture 指定采样文件")
        return apply_main(args)
    if args.command == "wizard":
        capture_args = copy.deepcopy(args)
        capture_args.output = args.capture_output
        capture_result = asyncio.run(capture_main(capture_args))
        if capture_result != 0:
            return capture_result
        apply_args = copy.deepcopy(args)
        apply_args.capture = args.capture_output
        return apply_main(apply_args)
    if args.command == "local-wizard":
        return local_wizard_main(args)
    return 1


async def capture_main(args: argparse.Namespace) -> int:
    if websockets is None:
        print("缺少 `websockets` 依赖，请先执行：python3 -m pip install websockets", file=sys.stderr)
        return 2

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"连接 teleop 流：{args.ws_url}")
    async with websockets.connect(args.ws_url, max_size=8 * 1024 * 1024) as ws:  # type: ignore[arg-type]
        poses: dict[str, dict[str, Any]] = {}
        for pose_name in POSE_SEQUENCE:
            print("")
            print(f"[{pose_name}] {POSE_HINTS[pose_name]}")
            await wait_for_pose_ready(args.auto_advance_seconds)
            sample = await capture_pose(ws, pose_name, args.samples, args.timeout_seconds)
            poses[pose_name] = sample
            print(
                f"已采集 {pose_name}: left={sample['left']['sample_count']} 帧, "
                f"right={sample['right']['sample_count']} 帧"
            )

    payload = {
        "schema_version": "1.0.0",
        "captured_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {
            "type": "edge_teleop_stream",
            "ws_url": args.ws_url,
        },
        "poses": poses,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"采样文件已写入：{output_path}")
    return 0


async def wait_for_pose_ready(auto_advance_seconds: float) -> None:
    if auto_advance_seconds > 0:
        seconds = int(math.ceil(auto_advance_seconds))
        for remain in range(seconds, 0, -1):
            print(f"  {remain}s 后自动开始采集...", flush=True)
            await asyncio.sleep(1.0)
        return
    input("姿态稳定后按回车开始采集...")


async def capture_pose(
    ws: Any,
    pose_name: str,
    sample_count: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    left_samples: list[list[float]] = []
    right_samples: list[list[float]] = []
    last_frame: dict[str, Any] | None = None
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    while len(left_samples) < sample_count or len(right_samples) < sample_count:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError(f"{pose_name} 采样超时，请确认 teleop 流正在输出 16 维手部目标")

        raw_message = await asyncio.wait_for(ws.recv(), timeout=remaining)
        frame = parse_ws_frame(raw_message)
        if frame is None:
            continue
        last_frame = frame

        left_vec = extract_hand_vector(frame, "left")
        right_vec = extract_hand_vector(frame, "right")
        if left_vec is not None and len(left_samples) < sample_count:
            left_samples.append(left_vec)
        if right_vec is not None and len(right_samples) < sample_count:
            right_samples.append(right_vec)

    assert last_frame is not None
    return {
        "name": pose_name,
        "edge_time_ns": last_frame.get("edge_time_ns"),
        "hand_joint_layout": last_frame.get("hand_joint_layout"),
        "hand_target_layout": last_frame.get("hand_target_layout"),
        "left": {
            "source": "left_hand_target" if last_frame.get("left_hand_target") else "left_hand_joints",
            "sample_count": len(left_samples),
            "mean_vector": average_vectors(left_samples),
        },
        "right": {
            "source": "right_hand_target" if last_frame.get("right_hand_target") else "right_hand_joints",
            "sample_count": len(right_samples),
            "mean_vector": average_vectors(right_samples),
        },
    }


def parse_ws_frame(raw_message: Any) -> dict[str, Any] | None:
    if isinstance(raw_message, bytes):
        try:
            raw_message = raw_message.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(raw_message, str):
        return None
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != "teleop_frame_v1":
        return None
    return payload


def extract_hand_vector(frame: dict[str, Any], side: str) -> list[float] | None:
    target_key = f"{side}_hand_target"
    joint_key = f"{side}_hand_joints"
    target_layout = frame.get("hand_target_layout")
    joint_layout = frame.get("hand_joint_layout")

    target = frame.get(target_key)
    if is_valid_hand_vector(target) and target_layout in HAND_LAYOUTS:
        return [float(value) for value in target]

    joints = frame.get(joint_key)
    if is_valid_hand_vector(joints) and joint_layout in HAND_LAYOUTS:
        return [float(value) for value in joints]
    return None


def is_valid_hand_vector(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != LEAP_COMMAND_DIM:
        return False
    return all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)


def average_vectors(samples: list[list[float]]) -> list[float]:
    out = [0.0] * LEAP_COMMAND_DIM
    for sample in samples:
        for idx, value in enumerate(sample):
            out[idx] += value
    return [value / len(samples) for value in out]


def apply_main(args: argparse.Namespace) -> int:
    capture_path = Path(args.capture).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    capture = json.loads(capture_path.read_text(encoding="utf-8"))

    report = build_report(
        capture=capture,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        requested_side=args.side,
        config_path=str(config_path),
    )

    print_report(report)

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"标定报告已写入：{report_path}")

    if args.dry_run:
        print("dry-run 模式：未写回 leap.toml")
        return 0

    write_report_to_toml(config_path, report, create_backup=not args.no_backup)
    print(f"标定结果已写回：{config_path}")
    return 0


def local_wizard_main(args: argparse.Namespace) -> int:
    capture_path = args.capture_file or args.capture_output
    if args.capture_file:
        print(f"复用现有采样文件：{Path(args.capture_file).expanduser().resolve()}")
    else:
        capture_args = copy.deepcopy(args)
        capture_args.output = args.capture_output
        capture_result = asyncio.run(capture_main(capture_args))
        if capture_result != 0:
            return capture_result

    apply_args = copy.deepcopy(args)
    apply_args.capture = capture_path
    apply_result = apply_main(apply_args)
    if apply_result != 0 or args.dry_run:
        return apply_result

    if args.no_restart_stack:
        print("已写回配置；由于指定了 --no-restart-stack，本次不校验运行态，请在重启 bridge 后手动检查 /health。")
        return 0

    run_stack_restart(Path(args.stack_script).expanduser().resolve())
    health = wait_for_bridge_health(
        health_url=args.health_url,
        timeout_seconds=args.health_timeout_seconds,
        requested_side=args.side,
    )
    print_health_summary(args.health_url, health)
    return 0


def run_stack_restart(stack_script: Path) -> None:
    if not stack_script.exists():
        raise FileNotFoundError(f"未找到本机联调栈脚本：{stack_script}")
    print(f"重启本机联调栈：{stack_script}")
    subprocess.run([str(stack_script), "restart"], check=True)


def wait_for_bridge_health(
    health_url: str,
    timeout_seconds: float,
    requested_side: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib_request.Request(health_url, headers={"Accept": "application/json"})
            with urllib_request.urlopen(request, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if is_health_ready(payload, requested_side):
                return payload
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1.0)
    if last_error is not None:
        raise RuntimeError(f"等待 bridge health 超时：{last_error}") from last_error
    raise RuntimeError("等待 bridge health 超时：标定状态未生效")


def is_health_ready(payload: dict[str, Any], requested_side: str) -> bool:
    side_names = ["left", "right"] if requested_side == "both" else [requested_side]
    for side in side_names:
        if not payload.get(f"{side}_retarget_calibrated"):
            return False
        if int(payload.get(f"{side}_retarget_joint_count", 0)) != LEAP_COMMAND_DIM:
            return False
    return True


def print_health_summary(health_url: str, health: dict[str, Any]) -> None:
    print("")
    print(f"运行态标定已生效：{health_url}")
    for side in ("left", "right"):
        if f"{side}_retarget_calibrated" not in health:
            continue
        print(
            f"- {side}: calibrated={health[f'{side}_retarget_calibrated']}, "
            f"joint_count={health.get(f'{side}_retarget_joint_count', 0)}, "
            f"scale_non_default={health.get(f'{side}_retarget_non_default_scale_count', 0)}, "
            f"offset_non_zero={health.get(f'{side}_retarget_non_zero_offset_count', 0)}"
        )


def build_report(
    capture: dict[str, Any],
    scale_min: float,
    scale_max: float,
    requested_side: str,
    config_path: str,
) -> dict[str, Any]:
    open_pose = capture["poses"]["open_palm"]
    fist_pose = capture["poses"]["closed_fist"]

    sides = ["left", "right"] if requested_side == "both" else [requested_side]
    side_reports: dict[str, Any] = {}
    for side in sides:
        open_target = open_pose[side]["mean_vector"]
        fist_target = fist_pose[side]["mean_vector"]
        open_cmd = anatomical_target_to_leap_command(side, open_target)
        fist_cmd = anatomical_target_to_leap_command(side, fist_target)
        scale, offset, warnings = solve_scale_offset(open_cmd, fist_cmd, scale_min, scale_max)
        side_reports[side] = {
            "joint_scale": scale,
            "joint_offset": offset,
            "open_command": open_cmd,
            "closed_command": fist_cmd,
            "predicted_open_after_calibration": apply_scale_offset(open_cmd, scale, offset),
            "predicted_closed_after_calibration": apply_scale_offset(fist_cmd, scale, offset),
            "warnings": warnings,
        }

    return {
        "schema_version": "1.0.0",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "capture_source": capture.get("source", {}),
        "config_path": config_path,
        "sides": side_reports,
    }


def anatomical_target_to_leap_command(side: str, target: list[float]) -> list[float]:
    if len(target) != LEAP_COMMAND_DIM:
        raise ValueError("手部目标维度必须为 16")
    thumb_abduction = clamp(target[0], 0.0, math.pi / 3.0)
    thumb_yaw = -thumb_abduction if side == "left" else thumb_abduction
    out = [
        thumb_yaw,
        clamp(target[1], 0.0, 1.2),
        clamp(target[2], 0.0, 1.3),
        clamp(target[3], 0.0, 1.35),
    ]
    out.extend(clamp(value, 0.0, math.pi / 2.0) for value in target[4:16])
    return out


def solve_scale_offset(
    open_cmd: list[float],
    closed_cmd: list[float],
    scale_min: float,
    scale_max: float,
) -> tuple[list[float], list[float], list[str]]:
    desired_open = [open_cmd[0]] + [0.0] * 15
    desired_closed = [closed_cmd[0]] + [float(value) for value in FLEX_CLOSE_REFERENCE[1:]]
    scale = [1.0] * LEAP_COMMAND_DIM
    offset = [0.0] * LEAP_COMMAND_DIM
    warnings: list[str] = []

    for idx in range(LEAP_COMMAND_DIM):
        if idx == 0:
            continue

        raw_open = open_cmd[idx]
        raw_closed = closed_cmd[idx]
        target_open = desired_open[idx]
        target_closed = desired_closed[idx]
        delta_raw = raw_closed - raw_open

        if delta_raw <= 1.0e-4:
            offset[idx] = target_open - raw_open
            warnings.append(f"joint[{idx}] 收到的 closed_fist 幅度不足，已退回仅做零位补偿")
            continue

        raw_scale = (target_closed - target_open) / delta_raw
        clamped_scale = clamp(raw_scale, scale_min, scale_max)
        if abs(clamped_scale - raw_scale) > 1.0e-6:
            warnings.append(
                f"joint[{idx}] scale 从 {raw_scale:.4f} 被限制到 {clamped_scale:.4f}"
            )
        scale[idx] = clamped_scale
        offset[idx] = target_open - raw_open * clamped_scale

    return scale, offset, warnings


def apply_scale_offset(values: list[float], scale: list[float], offset: list[float]) -> list[float]:
    return [values[idx] * scale[idx] + offset[idx] for idx in range(len(values))]


def write_report_to_toml(config_path: Path, report: dict[str, Any], create_backup: bool) -> None:
    original = config_path.read_text(encoding="utf-8")
    updated = original
    for side, side_report in report["sides"].items():
        updated = replace_toml_array(updated, f"{side}_joint_scale", side_report["joint_scale"])
        updated = replace_toml_array(updated, f"{side}_joint_offset", side_report["joint_offset"])

    if create_backup:
        backup_path = config_path.with_suffix(config_path.suffix + f".bak.{timestamp_slug()}")
        shutil.copy2(config_path, backup_path)
        print(f"已备份原配置：{backup_path}")
    config_path.write_text(updated, encoding="utf-8")


def replace_toml_array(content: str, key: str, values: list[float]) -> str:
    line = f"{key} = [{', '.join(format_float(value) for value in values)}]"
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=\s*\[.*?\]$")
    if pattern.search(content):
        return pattern.sub(line, content, count=1)
    suffix = "" if content.endswith("\n") else "\n"
    return f"{content}{suffix}{line}\n"


def print_report(report: dict[str, Any]) -> None:
    print("")
    print("标定结果摘要：")
    for side, side_report in report["sides"].items():
        print(f"- {side}:")
        print(f"  scale  = [{', '.join(format_float(v) for v in side_report['joint_scale'])}]")
        print(f"  offset = [{', '.join(format_float(v) for v in side_report['joint_offset'])}]")
        warnings = side_report["warnings"]
        if warnings:
            print("  warnings:")
            for warning in warnings:
                print(f"    - {warning}")


def format_float(value: float) -> str:
    text = f"{value:.6f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def timestamp_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
