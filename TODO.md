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
- [ ] 准备首个 public tag
