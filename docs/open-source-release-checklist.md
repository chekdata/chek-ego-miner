# Open-Source Release Checklist

## Required before first public release

- [ ] choose and add `LICENSE`
- [ ] run `./scripts/scan_public_safety.sh .`
- [ ] verify `README.md` and `README.zh-CN.md` still match the real public scope
- [ ] verify TestFlight link still works
- [ ] verify dataset portal link still works
- [ ] verify no internal-only rollout docs were copied into the repo
- [ ] verify no internal hostnames, IPs, or tokens remain
- [ ] verify `AGENTS.md`, hardware guide, and prompts are present
- [ ] verify `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` are present
- [ ] perform one external-user walkthrough
- [ ] perform one agent-guided walkthrough

## Recommended before every release

- [ ] update TODO status
- [ ] update screenshots if onboarding changed
- [ ] review hardware guide for outdated advice
- [ ] review agent prompts for stale instructions
