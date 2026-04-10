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

## Public contribution agreement

By opening a pull request here, you confirm that:

- you have the right to submit the code, docs, prompts, or assets in the PR
- your contribution does not intentionally include secrets or internal-only infrastructure details
- any screenshots, examples, or sample data are safe to publish
- any guidance that affects capture or upload flow stays consistent with the privacy and consent rules in [docs/privacy-data-license.md](./docs/privacy-data-license.md)

## Good first contributions

Helpful public-facing contributions include:

- improving setup instructions
- clarifying hardware buying guidance
- tightening troubleshooting steps
- improving agent prompts and walkthroughs
- fixing public-safe scripts and manifests
- improving dataset discovery and contribution documentation

## Pull request expectations

- document only what contributors can actually reproduce
- prefer simple language and reproducible instructions
- avoid public docs that depend on private infrastructure details
- if you add new setup steps, also add troubleshooting notes

## For data-related docs

If your change touches capture, upload, privacy, or reward guidance:

- keep consent language clear and non-ambiguous
- do not promise fixed payouts unless a published program page already does
- separate code-license language from dataset-license language
- prefer examples that show safe, reusable, public-friendly flows

## Before merging

Check at least:

```bash
./scripts/scan_public_safety.sh .
```

If the scan shows a real internal host or token, remove or sanitize it before merging.

Also review:

- [docs/privacy-data-license.md](./docs/privacy-data-license.md)
- [docs/token-rewards.md](./docs/token-rewards.md)
- [docs/diagnostics.md](./docs/diagnostics.md)
