# Contributing

## Scope

This is the public-facing repo for CHEK EGO Miner.

Use it for:

- onboarding docs
- hardware guidance
- agent guidance
- public setup and contribution flow
- public-safe scripts and manifests once migrated

Do not add:

- internal hostnames
- private IP addresses
- internal rollout notes
- secrets, tokens, or environment-specific credentials

## Pull request expectations

- keep the public promise narrower than verified evidence
- prefer simple language and reproducible instructions
- avoid public docs that depend on private infrastructure details
- if you add new setup steps, also add troubleshooting notes

## Before merging

Check at least:

```bash
./scripts/scan_public_safety.sh .
```

If the scan shows a real internal host or token, remove or sanitize it before merging.
