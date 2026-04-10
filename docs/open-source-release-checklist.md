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
- [x] verify `AGENTS.md`, hardware guide, and prompts are present
- [x] verify `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` are present
- [x] perform one external-user walkthrough
- [x] perform one agent-guided walkthrough

## Current release baseline

- source code license: `Apache-2.0`
- public links last checked: `2026-04-10`
- default-branch vulnerable lockfile entries patched locally and pushed: `2026-04-10`
- public host lanes with live evidence:
  - dedicated `Linux x86_64` basic lane
  - local `macOS arm64` basic lane
- current external blockers:
  - GitHub Dependabot alerts still need to refresh against the patched default branch
  - Windows public live evidence
  - Jetson public live evidence
  - `Stereo / Pro` true-hardware acceptance

## Recommended before every release

- [ ] update TODO status
- [ ] update screenshots if onboarding changed
- [ ] review hardware guide for outdated advice
- [ ] review agent prompts for stale instructions
