# Quickstart：Edge Orchestrator Bootstrap

## 1) 前置条件

- Rust stable 工具链（建议用 `rustup` 安装）
- `teleop-protocol` 钉住文件（版本 + schema 校验和），默认读取 `protocol_pin.json`
- （推荐）Python venv + `websockets`：用于一键冒烟验证脚本

## 2) 本地启动

```bash
cd edge-orchestrator

# 现场/联调建议开启最小鉴权（PRD 2.2）：除 /health 外均需携带 token
export EDGE_TOKEN=edge-token-demo-001

# 必填：否则 preflight.extrinsic_ok=false，无法 ARM
export EXTRINSIC_VERSION=ext-demo-0.1.0

cargo run
```

默认监听：

- HTTP：`0.0.0.0:8080`
- WS：`0.0.0.0:8765`

## 3) 一键冒烟验证（推荐）

脚本会自动：

- 随机挑选空闲端口启动服务
- 完成 `session/start` + `time/sync` + `control/arm`
- 验证 Deadman：松开不输出、按住才输出、超时进入 `fault`
- 验证最小鉴权：HTTP `Authorization: Bearer <EDGE_TOKEN>` + WS `?token=<EDGE_TOKEN>`

```bash
cd edge-orchestrator
python3 -m venv .venv
. .venv/bin/activate
pip install websockets
python -u scripts/verify_smoke.py
```

## 4) 手动验证（可选）

1) `POST /session/start`

```bash
curl -sS -X POST http://127.0.0.1:8080/session/start \
  -H "Authorization: Bearer ${EDGE_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"schema_version":"1.0.0","trip_id":"trip-001","session_id":"sess-001","device_id":"client-001"}'
```

2) WS 连接（示例）：

- fusion：`ws://127.0.0.1:8765/stream/fusion?token=${EDGE_TOKEN}`（可选加 `&format=cbor`）
- teleop：`ws://127.0.0.1:8765/stream/teleop?token=${EDGE_TOKEN}`（可选加 `&format=cbor`）

## 5) Done Criteria

- 启动时协议钉住校验通过
- Deadman 行为通过（超时进入 `fault`，且 motion 立即停止）
- `/health` 可用；`/metrics`（若启用鉴权则需 token）能看到 bridge/deadman/time_sync 等 gauges
