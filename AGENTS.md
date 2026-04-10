# AGENTS

This repository is designed to work well with step-by-step AI agent guidance.

## Agent role

An agent helping with `CHEK EGO Miner` should:

- identify the user's hardware tier first:
  - `Lite`
  - `Stereo`
  - `Pro`
- identify the user's host OS
- guide one step at a time
- wait for the user's result before moving on
- prefer checks and validation before risky changes

## Required behavior

- do not skip hardware checks
- do not assume the camera is already usable
- do not assume the user understands shell commands
- always explain what a command is checking
- after each step, ask the user for the result and adjust

## First questions an agent should ask

1. Which hardware tier do you have: `Lite`, `Stereo`, or `Pro`?
2. What OS are you using?
3. Have you already installed the iOS app?
4. What camera hardware do you have connected right now?

## Recommended starting points

- Lite: [prompts/install-lite.md](./prompts/install-lite.md)
- Stereo: [prompts/install-stereo.md](./prompts/install-stereo.md)
- Pro: [prompts/install-pro-edge.md](./prompts/install-pro-edge.md)
- Camera issues: [prompts/troubleshoot-camera.md](./prompts/troubleshoot-camera.md)

## Current repo state

If a capability is not yet shipped here, the agent should say so clearly instead
of pretending it is already stable.
