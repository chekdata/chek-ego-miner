# Quickstart：ruview-leap-bridge（LEAP 双手桥接，MVP）

本 quickstart 以“本地 demo 模式”为默认路径：不依赖 `edge-orchestrator` 也能跑通闭环（含 `/health` 与 `/metrics`）。

## 1) 前置条件

- Rust stable 工具链（建议 1.85+）
- （可选）若要联调 edge：需要能访问 `ws://<edge_ip>:8765/stream/teleop`

## 2) 本地运行（demo 模式）

```bash
cd /Users/jasonhong/Desktop/CHEK-humanoid/ruview-leap-bridge
cargo run
```

默认读取 `config/leap.toml`。当 `edge_teleop_ws_url` 为空时，bridge 会自动生成 `teleop_frame_v1` demo 帧并尝试下发（Mock LEAP 客户端）。

健康与指标：

- `GET http://127.0.0.1:8090/health`
- `GET http://127.0.0.1:8090/metrics`

## 3) 联调 edge（可选）

编辑 `config/leap.toml`：

- `edge_teleop_ws_url = "ws://<edge_ip>:8765/stream/teleop"`（可选加 `?format=cbor` 使用二进制帧）
- （若 edge 启用了鉴权）`edge_token = "<token>"`

然后运行：

```bash
cd /Users/jasonhong/Desktop/CHEK-humanoid/ruview-leap-bridge
cargo run
```

联调行为：

- 下行：消费 `teleop_frame_v1`（仅处理 `end_effector_type="LEAP_V2"` 的帧；兼容 JSON Text 与 CBOR Binary）
- 上行：在同一 WS 连接里回传 `bridge_state_packet`（用于 edge 侧 `bridge_ready` 与排障）

## 4) 一键验证（必做）

```bash
cd /Users/jasonhong/Desktop/CHEK-humanoid/ruview-leap-bridge
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## 5) Done 标准（MVP）

- 双手配对：<=20ms 正常，超窗进入 hold/freeze 退化
- 门控：`control_state!="armed"` 或 keepalive 超时必须阻断输出
- 硬件：offline/overtemp/error 会使 `is_ready=false` 并通过 `bridge_state_packet` 回传
