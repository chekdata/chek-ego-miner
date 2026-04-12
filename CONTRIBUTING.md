# Contributing

## What belongs in this repo

Use this repo for:

- onboarding docs
- hardware guidance
- agent guidance
- setup and contribution flow
- scripts and manifests that are safe to publish

Do not add:

- internal hostnames or private IP addresses
- rollout notes or recovery commands that are not meant for public use
- secrets, tokens, or environment-specific credentials

## Before you open a pull request

By opening a pull request here, you confirm that:

- you have the right to submit the code, docs, prompts, or assets in the PR
- your contribution does not intentionally include secrets or private
  infrastructure details
- any screenshots, examples, or sample data are safe to publish
- any guidance that affects capture or upload flow stays consistent with the
  privacy and consent rules in
  [docs/privacy-data-license.md](./docs/privacy-data-license.md)

## Useful contributions

- improving setup instructions
- clarifying hardware buying guidance
- tightening troubleshooting steps
- improving agent prompts and walkthroughs
- fixing scripts and manifests that are safe to publish
- improving dataset discovery and contribution documentation

## Pull request checklist

- document only what another public user can reproduce
- prefer simple language and copyable instructions
- avoid docs that depend on private infrastructure details
- if you add new setup steps, also add troubleshooting notes

## If your change touches data policy

- keep consent language clear and non-ambiguous
- do not promise fixed payouts unless a published program page already does
- keep code-license language separate from dataset-license language
- prefer examples that show safe, reusable flows

## Safety check

Run at least:

```bash
./scripts/scan_public_safety.sh .
```

If the scan finds a real internal host or token, remove or sanitize it before
merging.

Also review:

- [docs/privacy-data-license.md](./docs/privacy-data-license.md)
- [docs/token-rewards.md](./docs/token-rewards.md)
- [docs/diagnostics.md](./docs/diagnostics.md)
