# Public / Private Split

## Why split

The private engineering repo contains internal execution evidence, internal host
details, and rollout context that should not be published as-is.

`CHEK EGO Miner` is the public-first home for contributors and operators.

## Public repo should contain

- onboarding docs
- hardware guidance
- agent guidance
- public-safe install and diagnostic docs
- migrated public-safe scripts and manifests
- contribution flow
- dataset portal entrypoints

## Private repo should contain

- real hostnames and IP addresses
- Tailscale node details
- internal execution reports
- internal rollout notes
- non-public configs
- environment-specific bring-up commands

## Maintenance model

- public repo is the mainline for reusable code and docs
- private repo is an overlay for non-public information
- do not maintain two long-lived copies of the same main codebase

## Sanitize before migrating

Remove or replace:

- internal URLs
- real hostnames
- real IPs
- debug tokens
- private auth headers
- private filesystem paths that only work on internal machines
