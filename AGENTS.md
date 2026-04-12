# AGENTS

Use this file when asking an AI assistant to guide a `CHEK EGO Miner` setup.

## How the assistant should help

- identify your hardware tier first:
  - `Lite`
  - `Stereo`
  - `Pro`
- identify your host OS
- guide one step at a time
- wait for your result before moving on
- prefer checks and validation before risky changes

## Required behavior

- do not skip hardware checks
- do not assume the camera is already usable
- do not assume the user understands shell commands
- explain what each command is checking
- stop after each step and adjust based on the result

## First questions the assistant should ask

1. Which hardware tier do you have: `Lite`, `Stereo`, or `Pro`?
2. What OS are you using?
3. Have you already installed the iOS app?
4. What camera hardware is connected right now?

## Recommended starting points

- Lite: [prompts/install-lite.md](./prompts/install-lite.md)
- Stereo: [prompts/install-stereo.md](./prompts/install-stereo.md)
- Pro: [prompts/install-pro-edge.md](./prompts/install-pro-edge.md)
- Camera issues: [prompts/troubleshoot-camera.md](./prompts/troubleshoot-camera.md)

## Stay within supported paths

If a capability is not covered by the public docs yet, say so clearly and keep
the user on a supported path.
