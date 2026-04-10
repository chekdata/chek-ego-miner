# CHEK EGO Miner TODO

## Phase 1: Public repo foundation

- [x] 确定 public repo 名称为 `CHEK EGO Miner`
- [x] 创建 public repo 本地骨架
- [x] 落首版 `README.md` / `README.zh-CN.md`
- [x] 落首版 `TODO.md`
- [x] 落 `AGENTS.md`
- [x] 落 `docs/hardware.md`
- [x] 落 Codex / Claude / OpenClaw 指南
- [x] 落 public/private 边界文档
- [x] 落可复制 prompts

## Phase 2: Open-source hygiene

- [x] 确定开源许可证
- [x] 增加 `LICENSE`
- [x] 增加 `CONTRIBUTING.md`
- [x] 增加 `SECURITY.md`
- [x] 增加 `CODE_OF_CONDUCT.md`
- [x] 建立 public secret / internal-host 扫描基线
- [x] 建立 release checklist
- [x] 清掉 public 默认分支上的首批依赖安全告警

## Phase 3: Public content quality

- [x] 为 README 增加截图或架构图
- [x] 为三档硬件补充推荐 SKU 与实物示意
- [x] 补“如何赚 token”的公开说明
- [x] 补“数据许可 / 隐私 / 同意”的公开说明
- [x] 补常见问题与排障 FAQ

## Phase 4: Capability migration

- [x] 从 private repo 迁移首批可公开的 CLI / script / docs
- [x] 迁移首批 public-safe 诊断脚本：
  - [x] Charuco 校准板生成
  - [x] HTTP probe
  - [x] WebSocket probe
- [x] 增加 public 轻量宿主自检脚本 `check_host_basics.py`
- [x] 迁移对外可用的 install smoke 或 doctor/readiness 能力
- [x] 把硬件档位与内部 profile 映射写成稳定公共文档
- [x] 建立 public demo / quickstart lane

## Phase 5: Public release readiness

- [x] 清理 public repo 中所有 internal-only 信息
- [x] 复核外链可用性：
  - [x] TestFlight
  - [x] Dataset portal
- [x] 完成 first external-user walkthrough
- [x] 完成 first agent-guided walkthrough
- [x] 准备首个 public tag

## Phase 6: Public basic E2E hardening

- [x] 把 dedicated Linux x86_64 边缘机按 public 仓重装跑通
- [x] 补 public `basic-e2e` synthetic capture -> download -> validation lane
- [x] 补 public phone-vision sidecar 启动脚本、依赖说明与模型拉取
- [x] 让 public download 子集通过 `score_percent = 100.0`
- [x] 把单手机 `basic` lane 的 `time_sync_samples` 降级为 advisory

## Phase 7: Multi-host live evidence

- [x] 补 macOS `basic` install-driven `launchd` live evidence
- [x] 补 macOS `basic-e2e` synthetic capture -> download -> validation lane
- [x] 让 `start_edge_phone_vision_service.sh` 自动选择兼容 Python
- [ ] 补 Windows public live evidence（外部宿主需恢复在线并提供管理员权限）
- [ ] 补 Jetson public live evidence（外部共享宿主需允许重装与验收）
- [ ] 补 `Stereo / Pro` 真机 acceptance evidence（外部双目/边缘机硬件）
