# Public Roadmap

## 基本原则

public 对外承诺必须比真实 evidence 更保守。先把真实宿主上验证通过的 lane 做稳，再一条一条往外扩。

## 现在已经具备的

- public 仓已经具备 `LICENSE`、贡献、安全、隐私、奖励等基础文档
- TestFlight App 入口可用
- 数据门户入口可用
- `Lite / Stereo / Pro` 硬件指南可用
- Codex、Claude、OpenClaw 的 prompts 和使用指南可用
- public `doctor`、`readiness`、`install`、`basic-e2e` 流程可用
- `basic` lane 已在以下宿主拿到 live evidence：
  - dedicated `Linux x86_64`
  - 本地 `macOS arm64`

## 下一步

- 等 GitHub Dependabot 对已修复锁文件完成刷新并自动清告警
- 补 Windows public live evidence
- 补 Jetson public live evidence
- 补 `Stereo` 真机验收
- 补 `Pro` 真机验收

## 再往后

- 把 public download 的验证范围从当前 `basic` bundle 子集继续扩展
- 补更多面向贡献者的数据打包和验证样例
- 让奖励反馈和贡献结果更透明
- 补更多首个贡献者 walkthrough 截图与示例产物

## 在验证完成前不对外承诺

- 所有宿主都已经实现完整的一条命令对齐
- `Stereo / Pro` 已完全自动化
- Windows 或 Jetson 已经完整支持
