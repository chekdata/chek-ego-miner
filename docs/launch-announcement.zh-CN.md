# 首发公告

## 一句话版本

`CHEK EGO Miner` 已正式公开：一台手机配一台电脑，就能开始采集 EGO 数据、贡献数据并赚取 token。

## 短版公告

我们把 `CHEK EGO Miner` 做成了一个真正面向公开用户的仓库，目标很直接：
降低第一视角机器人数据采集的硬件与安装门槛，让更多人能一起参与，缓解机器人数据紧缺问题。

现在你可以直接从 TestFlight 下载 iOS 应用，按预算选择 `Lite / Stereo / Pro`
硬件方案，再让 Codex、Claude 或 OpenClaw 一步一步带你完成安装、检查与排障。

## 中版公告

`CHEK EGO Miner` 背后的核心判断是：
一台手机加一台电脑，就应该足以开始贡献有价值的机器人 EGO 数据。

当前 public 仓已经提供：

- iOS 应用 TestFlight 入口
- `Lite / Stereo / Pro` 三档硬件指南
- Codex、Claude、OpenClaw 的 agent 安装指引
- public-safe 的安装、诊断与验证入口
- 数据门户入口，支持检索和下载大家贡献的数据

今天我们真实验证过的 public 基线，是 dedicated `Linux x86_64` 和本地
`macOS arm64` 上的 `basic` lane。也就是说，现在对外写出来的能力，是有真机
evidence 支撑的，不是内部 staging 里的理想状态。

## 长版公告

机器人数据供给，长期以来都太稀缺、太昂贵，也太依赖少数内部采集链路。
`CHEK EGO Miner` 想把这个问题拆开来做：

- 用众包方式缓解机器人数据瓶颈
- 让用户先用自己已经有的设备开始采集
- 把 agent 手把手带装机，当成第一公民能力

你可以从最小门槛开始：

1. 下载 App
2. 选择硬件方案
3. 让 agent 带你一步一步完成安装和检查
4. 开始采集并贡献 EGO 数据
5. 去数据门户检索和下载别人贡献的数据

## 今天已经可以对外承诺的能力

- TestFlight 下载入口可用
- 数据门户入口可用
- `Lite / Stereo / Pro` 硬件指南已发布
- Codex、Claude、OpenClaw 指南已发布
- public `basic` 重装与验证，已经在以下宿主上拿到 live evidence：
  - dedicated `Linux x86_64`
  - 本地 `macOS arm64`

## 现在还不对外承诺的能力

- Windows public live evidence 还没补完
- Jetson public live evidence 还没补完
- `Stereo / Pro` 真机硬件验收还没补完
- GitHub Dependabot 还在显示旧告警，等待对已修复锁文件自动刷新

## 公开入口

- 下载 App：<https://testflight.apple.com/join/RrYdeDUv>
- 选择硬件：[硬件指南](./hardware.md)
- 让 agent 带装：
  - [Codex 指南](./agent-guides/codex.md)
  - [Claude 指南](./agent-guides/claude.md)
  - [OpenClaw 指南](./agent-guides/openclaw.md)
- 检索和下载数据：<https://www-dev.chekkk.com/humanoid/ego-dataset>

## 对外短文案素材

### 社交媒体短文案

`CHEK EGO Miner` 已公开。一台手机配一台电脑，就能开始采集 EGO 数据、贡献数据、赚取 token。TestFlight、硬件指南、agent prompts 和数据门户已经全部就位。

### 稍长版介绍

我们把 `CHEK EGO Miner` 开源出来了。

核心理念很简单：不要再把机器人数据采集当成只能内部做的事情。一台手机、一台电脑，加上 agent 手把手带安装，就可以开始贡献第一视角 EGO 数据；同一个 public 仓里，也已经给出了 App 下载、硬件指南、安装 prompts 和数据门户入口。
