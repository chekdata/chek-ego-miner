# Codex Guide

## When to use Codex

Use Codex when you want a coding-focused agent to:

- guide installation step by step
- check host readiness
- explain shell commands
- help debug camera or runtime issues

## Good starting prompt

See [../../prompts/install-lite.md](../../prompts/install-lite.md),
[../../prompts/install-stereo.md](../../prompts/install-stereo.md), or
[../../prompts/install-pro-edge.md](../../prompts/install-pro-edge.md).

## Recommended instruction style

Tell Codex:

- your hardware tier
- your OS
- what you have already installed
- that you want one step at a time
- that you want each step explained before moving on

## Example

“Treat me as a beginner. I am using the `Lite` setup on macOS. Guide me one step
at a time to install everything needed for CHEK EGO Miner, verify my camera, and
confirm I can begin capturing EGO data. After each step, wait for my result
before continuing.”
