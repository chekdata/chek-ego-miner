#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
edge-orchestrator 冒烟验证脚本（本机）：

- 启动 edge-orchestrator（随机挑选空闲端口）
- 通过 HTTP/WS 完成一次最小闭环验证：
  - session/start
  - keepalive（deadman）上行
  - bridge_state_packet 上行（unitree/leap ready）
  - time/sync
  - control/arm -> teleop_frame_v1.control_state 只有在 deadman 按住时才会变为 armed
  - 停止 keepalive -> 进入 fault，输出回到 disarmed

依赖：
- Rust 工具链（cargo）
- Python venv 安装 websockets：`pip install websockets`
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import websockets


class SharedState:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.teleop_frame: dict | None = None
        self.fusion_state: dict | None = None
        self.keepalive_enabled: bool = True
        self.deadman_pressed: bool = False

    async def set_deadman_pressed(self, pressed: bool) -> None:
        async with self._lock:
            self.deadman_pressed = pressed

    async def set_keepalive_enabled(self, enabled: bool) -> None:
        async with self._lock:
            self.keepalive_enabled = enabled

    async def snapshot_keepalive(self) -> tuple[bool, bool]:
        async with self._lock:
            return self.keepalive_enabled, self.deadman_pressed

    async def update_teleop(self, frame: dict) -> None:
        async with self._lock:
            self.teleop_frame = frame

    async def update_fusion(self, packet: dict) -> None:
        async with self._lock:
            self.fusion_state = packet


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
        temp_socket.bind(("127.0.0.1", 0))
        return int(temp_socket.getsockname()[1])


def http_get_json(url: str, edge_token: str, timeout_s: float = 2.0) -> dict:
    request = urllib.request.Request(
        url=url,
        method="GET",
        headers={"Authorization": f"Bearer {edge_token}"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def http_post_json(url: str, edge_token: str, payload: dict, timeout_s: float = 2.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {edge_token}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


async def wait_until(predicate, timeout_s: float, tick_s: float = 0.05) -> None:
    start = time.monotonic()
    while True:
        if predicate():
            return
        if time.monotonic() - start > timeout_s:
            raise TimeoutError("等待条件超时")
        await asyncio.sleep(tick_s)


async def run_ws_fusion(shared: SharedState, ws_url: str, trip_id: str, session_id: str) -> None:
    async with websockets.connect(ws_url, ping_interval=5, ping_timeout=5) as websocket:
        seq = 0

        async def sender() -> None:
            nonlocal seq
            while True:
                # 20Hz
                keepalive_enabled, pressed = await shared.snapshot_keepalive()
                if not keepalive_enabled:
                    await asyncio.sleep(0.05)
                    continue
                seq += 1
                packet = {
                    "type": "control_keepalive_packet",
                    "schema_version": "1.0.0",
                    "trip_id": trip_id,
                    "session_id": session_id,
                    "device_id": "client-verify-001",
                    "source_time_ns": time.time_ns(),
                    "seq": seq,
                    "deadman_pressed": pressed,
                }
                await websocket.send(json.dumps(packet))
                await asyncio.sleep(0.05)

        async def receiver() -> None:
            while True:
                message = await websocket.recv()
                if not isinstance(message, str):
                    continue
                try:
                    packet = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if packet.get("type") == "fusion_state_packet":
                    await shared.update_fusion(packet)

        await asyncio.gather(sender(), receiver())


async def run_ws_teleop(shared: SharedState, ws_url: str, trip_id: str, session_id: str) -> None:
    async with websockets.connect(ws_url, ping_interval=5, ping_timeout=5) as websocket:
        async def sender() -> None:
            while True:
                # 10Hz 上报 bridge ready，避免 stale（默认 500ms）导致 preflight/teleop 被判为不健康
                edge_time_ns = 0
                base_packet = {
                    "type": "bridge_state_packet",
                    "schema_version": "1.0.0",
                    "trip_id": trip_id,
                    "session_id": session_id,
                    "robot_type": "G1_29",
                    "end_effector_type": "LEAP_V2",
                    "edge_time_ns": edge_time_ns,
                    "is_ready": True,
                    "fault_code": "",
                    "fault_message": "",
                    "last_command_edge_time_ns": 0,
                }

                unitree_packet = dict(base_packet)
                unitree_packet["bridge_id"] = "unitree-bridge-01"
                await websocket.send(json.dumps(unitree_packet))

                leap_packet = dict(base_packet)
                leap_packet["bridge_id"] = "leap-bridge-01"
                await websocket.send(json.dumps(leap_packet))
                await asyncio.sleep(0.1)

        async def receiver() -> None:
            while True:
                message = await websocket.recv()
                if not isinstance(message, str):
                    continue
                try:
                    frame = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if frame.get("schema_version") == "teleop_frame_v1":
                    await shared.update_teleop(frame)

        await asyncio.gather(sender(), receiver())


async def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cargo_path = os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo_path):
        print("未找到 cargo，请先安装 Rust 工具链：~/.cargo/bin/cargo", file=sys.stderr)
        return 2

    http_port = pick_free_port()
    ws_port = pick_free_port()
    while ws_port == http_port:
        ws_port = pick_free_port()
    http_base = f"http://127.0.0.1:{http_port}"
    ws_base = f"ws://127.0.0.1:{ws_port}"
    edge_token = os.environ.get("EDGE_TOKEN", "edge-token-verify-001")

    trip_id = "trip-verify-001"
    session_id = "sess-verify-001"

    env = os.environ.copy()
    env["EDGE_HTTP_ADDR"] = f"127.0.0.1:{http_port}"
    env["EDGE_WS_ADDR"] = f"127.0.0.1:{ws_port}"
    env["EDGE_TOKEN"] = edge_token
    env["EXTRINSIC_VERSION"] = env.get("EXTRINSIC_VERSION", "ext-verify-0.1.0")
    env["RUST_LOG"] = env.get("RUST_LOG", "info")

    print(f"启动服务：HTTP={http_base} WS={ws_base}", flush=True)
    process = subprocess.Popen(
        [cargo_path, "run"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    async def pump_logs() -> None:
        if process.stdout is None:
            return
        while True:
            line = await asyncio.to_thread(process.stdout.readline)
            if not line:
                return
            sys.stdout.write(line)

    log_task = asyncio.create_task(pump_logs())

    try:
        # 等待 /health 就绪
        start = time.monotonic()
        while True:
            try:
                data = await asyncio.to_thread(http_get_json, f"{http_base}/health", edge_token)
                if data.get("status") == "ok":
                    break
            except Exception:
                pass

            if time.monotonic() - start > 30.0:
                raise TimeoutError("等待 /health 超时")

            await asyncio.sleep(0.2)
    except Exception:
        print("服务启动失败：/health 未就绪", file=sys.stderr)
        process.terminate()
        await log_task
        return 3

    shared = SharedState()
    fusion_task = asyncio.create_task(
        run_ws_fusion(
            shared,
            f"{ws_base}/stream/fusion?token={edge_token}",
            trip_id=trip_id,
            session_id=session_id,
        )
    )
    teleop_task = asyncio.create_task(
        run_ws_teleop(
            shared,
            f"{ws_base}/stream/teleop?token={edge_token}",
            trip_id=trip_id,
            session_id=session_id,
        )
    )

    try:
        # session/start
        print("调用 /session/start ...", flush=True)
        await asyncio.to_thread(
            http_post_json,
            f"{http_base}/session/start",
            edge_token,
            {
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "device_id": "client-verify-001",
            },
        )

        # 等待 teleop 帧出现（说明 teleop publisher 正常）
        await wait_until(lambda: shared.teleop_frame is not None, timeout_s=5.0)
        print("teleop 帧已收到", flush=True)

        # time/sync（使 preflight.time_sync_ok=true）
        print("调用 /time/sync ...", flush=True)
        await asyncio.to_thread(
            http_post_json,
            f"{http_base}/time/sync",
            edge_token,
            {
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "device_id": "client-verify-001",
                "clock_offset_ns": 0,
                "rtt_ns": 1_000_000,
                "sample_count": 8,
            },
        )

        # 等待 preflight + deadman link_ok 都准备好（避免时序竞态导致 arm 失败）
        async def preflight_ready() -> bool:
            try:
                state = await asyncio.to_thread(http_get_json, f"{http_base}/control/state", edge_token)
            except Exception:
                return False
            preflight = state.get("preflight") or {}
            deadman = state.get("deadman") or {}
            return bool(
                preflight.get("unitree_bridge_ready")
                and preflight.get("leap_bridge_ready")
                and preflight.get("time_sync_ok")
                and preflight.get("extrinsic_ok")
                and preflight.get("lan_control_ok")
                and deadman.get("link_ok")
            )

        print("等待 preflight 就绪（bridge/time_sync/extrinsic/lan_control/deadman.link_ok）...", flush=True)
        start = time.monotonic()
        while True:
            if await preflight_ready():
                break
            if time.monotonic() - start > 10.0:
                state = await asyncio.to_thread(http_get_json, f"{http_base}/control/state", edge_token)
                raise TimeoutError(f"preflight 未就绪：{state}")
            await asyncio.sleep(0.1)

        # 先保持 deadman_pressed=false，确保 teleop 不会进入 armed 输出
        await shared.set_deadman_pressed(False)

        # control/arm（deadman_link_ok 由 keepalive 保证；bridge_ready 由 teleop ws sender 保证）
        print("调用 /control/arm ...", flush=True)
        arm_response = await asyncio.to_thread(
            http_post_json,
            f"{http_base}/control/arm",
            edge_token,
            {
                "schema_version": "1.0.0",
                "trip_id": trip_id,
                "session_id": session_id,
                "robot_type": "G1_29",
                "end_effector_type": "LEAP_V2",
                "operator_id": "op-verify-001",
            },
        )
        if arm_response.get("state") != "armed":
            print(f"ARM 失败：{arm_response}", file=sys.stderr)
            return 4

        print("ARM 成功，验证 deadman 松开时 teleop_frame_v1.control_state=disarmed ...", flush=True)
        await wait_until(
            lambda: (shared.teleop_frame or {}).get("control_state") == "disarmed",
            timeout_s=3.0,
        )

        print("按住 deadman（deadman_pressed=true），验证 teleop_frame_v1.control_state=armed ...", flush=True)
        await shared.set_deadman_pressed(True)
        await wait_until(
            lambda: (shared.teleop_frame or {}).get("control_state") == "armed",
            timeout_s=3.0,
        )

        print("停止 keepalive，验证进入 fault 并停止输出 armed ...", flush=True)
        await shared.set_keepalive_enabled(False)
        await asyncio.sleep(0.3)
        state_after_timeout = await asyncio.to_thread(
            http_get_json, f"{http_base}/control/state", edge_token
        )
        if state_after_timeout.get("state") != "fault":
            print(f"deadman 超时后未进入 fault：{state_after_timeout}", file=sys.stderr)
            return 5

        await wait_until(
            lambda: (shared.teleop_frame or {}).get("control_state") == "disarmed",
            timeout_s=3.0,
        )

        print("验证通过", flush=True)
        return 0
    finally:
        fusion_task.cancel()
        teleop_task.cancel()
        try:
            await asyncio.gather(fusion_task, teleop_task, return_exceptions=True)
        except Exception:
            pass

        # 优雅停止服务
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=10.0)
            except TimeoutError:
                process.kill()

        log_task.cancel()
        try:
            await asyncio.gather(log_task, return_exceptions=True)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 130
    raise SystemExit(exit_code)
