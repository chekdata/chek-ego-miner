[English](./README.md) | [简体中文](./README.zh-CN.md)

# CHEK EGO Miner

众包解决机器人数据紧缺问题。

一台手机配一台电脑，就能开始新时代的数据“挖矿”：
采 EGO 数据，贡献数据，赚取 token。

## 先看这里

- 下载 iOS 应用：[TestFlight](https://testflight.apple.com/join/RrYdeDUv)
- 选择你的硬件方案：[硬件指南](./docs/hardware.md)
- 让 agent 一步一步带你做：
  - [Codex 指南](./docs/agent-guides/codex.md)
  - [Claude 指南](./docs/agent-guides/claude.md)
  - [OpenClaw 指南](./docs/agent-guides/openclaw.md)
- 检索和下载大家贡献的数据：
  - [EGO Dataset 数据门户](https://www-dev.chekkk.com/humanoid/ego-dataset)

## 这个项目是干什么的

`CHEK EGO Miner` 是一个面向公开用户的仓库，服务下面几类场景：

- 用手机和电脑开始采集第一视角 EGO 数据
- 升级到双目摄像头或边缘机，提升采集质量和吞吐
- 借助 Codex、Claude、OpenClaw 这类 agent 做安装与排障
- 把数据贡献到众包数据体系
- 检索和下载别人贡献的数据

## 系统视图

```mermaid
flowchart LR
  Phone["iPhone + CHEK App"] --> Host["电脑或边缘机"]
  Camera["自有摄像头或双目摄像头"] --> Host
  Agent["Codex / Claude / OpenClaw"] --> Host
  Host --> Upload["上传 EGO Session"]
  Upload --> Portal["数据门户"]
  Upload --> Rewards["Token 奖励"]
```

## 硬件三档

| 档位 | 方案 | 适合谁 |
| --- | --- | --- |
| `Lite` | 电脑 + 自己的摄像头 | 想最低门槛开跑的人 |
| `Stereo` | 电脑 + 外接双目摄像头 | 想要更好空间质量的人 |
| `Pro` | 边缘机 + 双目摄像头 | 想做专用采集与更高吞吐的人 |

此外还建议准备一个第一视角手机支架。购买思路、选型标准和搜索关键词见
[硬件指南](./docs/hardware.md)，其中也包含新增的淘宝和抖音购买链接示例。

## Agent 手把手安装

如果你不想自己啃长文档，可以直接从下面开始：

- [AGENTS.md](./AGENTS.md)
- 复制一个现成 prompt 给 agent：
  - [Lite 安装 Prompt](./prompts/install-lite.md)
  - [Stereo 安装 Prompt](./prompts/install-stereo.md)
  - [Pro 边缘机 Prompt](./prompts/install-pro-edge.md)
  - [摄像头排障 Prompt](./prompts/troubleshoot-camera.md)

推荐操作方式：

1. 先告诉 agent 你是 `Lite / Stereo / Pro` 哪一档。
2. 再告诉 agent 你的操作系统，以及你已经装好了什么。
3. 要求 agent 一次只给你一步，并等待你反馈结果。
4. 不要让 agent 跳过硬件检查、App 安装和相机验证。

## 第一个本地检查

在进入更长的安装流程前，可以先跑轻量自检：

```bash
python3 scripts/check_host_basics.py
```

如果你要检查这个 public 仓里是否混进了 internal-only 信息，可以跑：

```bash
./scripts/scan_public_safety.sh .
```

或者直接用 public CLI：

```bash
./cli/chek-ego-miner doctor
./cli/chek-ego-miner readiness --tier lite
```

如果你要在 Linux 或 macOS 宿主上按 public 仓重装并跑一条完整 `basic` 闭环，可以直接从下面开始：

```bash
./cli/chek-ego-miner install \
  --profile basic \
  --apply \
  --system-install \
  --enable-services

python3 -m pip install --user --break-system-packages -r scripts/edge_phone_vision_requirements.txt
./cli/chek-ego-miner fetch-phone-vision-models --json
./scripts/start_edge_phone_vision_service.sh

./cli/chek-ego-miner basic-e2e \
  --edge-base-url http://127.0.0.1:8080 \
  --edge-token chek-ego-miner-local-token \
  --trip-id trip-public-basic-e2e \
  --session-id sess-public-basic-e2e \
  --output-dir ./artifacts/basic-e2e \
  --json
```

如果 macOS 上的 Homebrew `python3` 因 PEP 668 拒绝 `pip install --user`，
可以把同样的依赖装到兼容解释器里，例如 `python3.10`；启动脚本检测到后会自动切过去。

这条 lane 目前已经在下面两类宿主上拿到真实 evidence：

- dedicated `Linux x86_64` 边缘机：
  - public 仓重装成功
  - `systemd-user` basic service 启动成功
  - synthetic 采集 -> 本地下载 -> public download 导出 成功
  - `public_download/demo_capture_bundle.json` 验证 `score_percent = 100.0`
- 本地 `macOS arm64` 开发机：
  - `install --apply --system-install --enable-services` 会自动把 runtime staging 到 `~/.chek-edge/runtime/macos/basic`
  - `launchd` basic service 能从该 staging 根目录成功启动
  - `./scripts/start_edge_phone_vision_service.sh` 会自动选择兼容的本地 Python 解释器
  - `basic-e2e` 产出的 `public_download/demo_capture_bundle.json` 验证 `score_percent = 100.0`
- `time_sync_samples` 在单手机 basic lane 上当前作为 advisory，不再阻塞通过

## 数据门户

当前可以通过下面的入口检索和下载大家贡献的数据：

- [https://www-dev.chekkk.com/humanoid/ego-dataset](https://www-dev.chekkk.com/humanoid/ego-dataset)

## 当前 public 范围

这个仓现在已经能承担公开可复现的 `basic` Linux 与 macOS lane，主要负责：

- onboarding
- 硬件盘点与购买建议
- agent 使用指南
- 对外安装文档与 prompts
- 数据贡献流程
- 数据检索入口
- public `basic` synthetic capture -> download -> validation 回归

## 项目理念

- 众包解决机器人数据瓶颈
- 降低 EGO 数据采集门槛
- 把 agent 辅助安装当成第一公民
- 让一次采集变成可复用的机器人数据资产

## 文档入口

- [硬件指南](./docs/hardware.md)
- [首发公告](./docs/launch-announcement.zh-CN.md)
- [公开路线图](./docs/roadmap.zh-CN.md)
- [v0.1.0 发布说明](./docs/releases/v0.1.0.md)
- [Quickstart](./docs/quickstart.md)
- [硬件与内部 profile 映射](./docs/profile-mapping.md)
- [诊断工具](./docs/diagnostics.md)
- [Token 奖励说明](./docs/token-rewards.md)
- [隐私、同意与数据许可](./docs/privacy-data-license.md)
- [常见问题](./docs/faq.md)
- [开源发布检查清单](./docs/open-source-release-checklist.md)
- [Codex 指南](./docs/agent-guides/codex.md)
- [Claude 指南](./docs/agent-guides/claude.md)
- [OpenClaw 指南](./docs/agent-guides/openclaw.md)
- [TODO](./TODO.md)

## 贡献方式

见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 安全问题

见 [SECURITY.md](./SECURITY.md)。

## 许可证

见 [LICENSE](./LICENSE)。
