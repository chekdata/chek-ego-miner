# Open-Source Release Checklist

## Required before first public release

- [x] choose and add `LICENSE`
- [x] run `./scripts/scan_public_safety.sh .`
- [x] verify `README.md` and `README.zh-CN.md` still match the real public scope
- [x] verify TestFlight link still works
- [x] verify dataset portal link still works
- [x] verify no internal-only rollout docs were copied into the repo
- [x] verify no internal hostnames, IPs, or tokens remain
- [ ] verify GitHub Dependabot alerts have refreshed after the initial dependency patch
- [x] verify GitHub issues are enabled and templates are present
- [x] verify a GitHub release exists for the first public tag
- [x] verify `AGENTS.md`, hardware guide, and prompts are present
- [x] verify `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` are present
- [x] perform one external-user walkthrough
- [x] perform one agent-guided walkthrough

## Current release baseline

- source code license: `Apache-2.0`
- public links last checked: `2026-04-11`
- default-branch vulnerable lockfile entries patched locally and pushed: `2026-04-10`
- GitHub Dependabot rechecked: `2026-04-11`
- GitHub dependency graph SBOM already reports `quinn-proto 0.11.14` and `rustls-webpki 0.103.10`: `2026-04-11`
- GitHub issues enabled: `2026-04-11`
- GitHub automated security fixes enabled: `2026-04-11`
- GitHub release published: `v0.1.0`
- GitHub blocker issues:
  - `#1` stale Dependabot refresh
  - `#2` Windows public live evidence
  - `#3` Jetson public live evidence
  - `#4` `Stereo / Pro` true-hardware acceptance
- public host lanes with live evidence:
  - dedicated `Linux x86_64` basic lane
  - local `macOS arm64` basic lane
- current external blockers:
  - GitHub Dependabot still shows 6 open alerts while the patched default-branch lockfiles wait for refresh
  - Windows public live evidence
  - Jetson public live evidence
  - `Stereo / Pro` true-hardware acceptance

## Recommended before every release

- [ ] update TODO status
- [ ] update screenshots if onboarding changed
- [ ] review hardware guide for outdated advice
- [ ] review agent prompts for stale instructions
