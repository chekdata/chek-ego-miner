# Launch Announcement

## One-line launch copy

CHEK EGO Miner is now public: one phone plus one computer is enough to start
capturing EGO data, contributing it, and earning tokens.

## Short announcement

We are opening CHEK EGO Miner to make first-person robot-data collection more
accessible. The goal is simple: reduce the hardware and setup friction so more
people can help solve robot data scarcity.

Start with the CHEK iOS app on TestFlight, pick the hardware tier that matches
your budget, let Codex, Claude, or OpenClaw guide you step by step, and begin
capturing EGO sessions from a public-first repo.

## Medium announcement

CHEK EGO Miner is a public repo built around a practical idea: one phone plus
one computer should be enough to start contributing useful robot data.

The repo already includes:

- a TestFlight entry for the iOS app
- hardware guidance for `Lite`, `Stereo`, and `Pro` setups
- agent-ready prompts for Codex, Claude, and OpenClaw
- a public-safe install and diagnostics surface
- a dataset portal entrypoint for browsing and downloading contributed data

Today, the verified public baseline is the `basic` lane on dedicated
`Linux x86_64` hosts and local `macOS arm64` hosts. That means the current
public promise is grounded in live evidence, not just internal staging claims.

## Long announcement

Robot-data supply is still too narrow, too expensive, and too dependent on
small internal pipelines. CHEK EGO Miner is our public attempt to widen that
funnel.

The project is built around three ideas:

- crowdsource the robot-data bottleneck instead of treating capture as a closed pipeline
- let users start small with hardware they already own
- treat agent-guided bring-up as a first-class installation path

You can start with an iPhone, a computer, and a camera you already have. If
you want better spatial signals, move up to a stereo camera. If you want a
dedicated station, move up to an edge machine.

The public repo is designed to help contributors move through the full loop:

1. install the app
2. choose hardware
3. let an agent guide install and validation
4. capture and contribute EGO data
5. browse and download useful contributed datasets

## Verified public claims today

- TestFlight app download is live
- the dataset portal link is live
- `Lite`, `Stereo`, and `Pro` hardware guidance is published
- Codex, Claude, and OpenClaw setup guides are published
- public `basic` reinstall + validation has live evidence on:
  - dedicated `Linux x86_64`
  - local `macOS arm64`

## What we are not claiming yet

- Windows public live evidence is not complete yet
- Jetson public live evidence is not complete yet
- `Stereo / Pro` true-hardware acceptance evidence is not complete yet
- GitHub Dependabot still shows stale open alerts while the patched lockfiles wait for refresh

## Call to action

- Download the app: <https://testflight.apple.com/join/RrYdeDUv>
- Choose hardware: [Hardware Guide](./hardware.md)
- Let an agent guide setup:
  - [Codex Guide](./agent-guides/codex.md)
  - [Claude Guide](./agent-guides/claude.md)
  - [OpenClaw Guide](./agent-guides/openclaw.md)
- Browse contributed data: <https://www-dev.chekkk.com/humanoid/ego-dataset>

## Social copy starters

### X / short post

CHEK EGO Miner is now public. One phone plus one computer is enough to start
capturing EGO data, contributing it, and earning tokens. TestFlight, hardware
guide, agent prompts, and dataset portal are all live.

### LinkedIn / longer post

We just opened CHEK EGO Miner.

The idea is simple: crowdsource robot-data collection instead of treating it as
an internal-only bottleneck. With one phone, one computer, and agent-guided
setup, contributors can start capturing EGO data, contribute sessions, and
browse public datasets from the same public-first repo.

The current verified public baseline covers the `basic` lane on dedicated
Linux x86_64 and local macOS arm64 hosts, with TestFlight, hardware guidance,
agent prompts, and dataset discovery already live.
