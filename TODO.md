# CHEK EGO Miner Public Roadmap

This file is the public-facing TODO list for contributor setup and validation.
It intentionally separates local checks from cloud contribution. Local probes do
not upload data unless a documented bind/upload flow is explicitly run.

## Current Public Lanes

| Lane | Status | Public gate |
| --- | --- | --- |
| `Lite` host setup | `supported` | `check_host_basics`, `readiness --tier lite`, `camera-probe --capture-smoke`, `basic-e2e` |
| `Stereo` setup | `in-progress` | stereo camera probe, calibration, readiness, capture validation |
| `Pro` Jetson setup | `in-progress` | Jetson bootstrap, service health, VLM sidecar, stereo/Wi-Fi readiness |
| Guided install | `supported` | AGENTS.md plus Codex / Claude / OpenClaw prompts |
| Local VLM | `supported for capable hosts` | model fetch, sidecar startup, non-empty inference output |
| Training thresholds | `candidate only` | `validate_training_thresholds.py` refuses final `training_ready` until thresholds are frozen |
| Cloud contribution | `guarded` | requires explicit public bind/upload path, consent and upload scope |

## Near-Term TODO

- Keep `Lite` as the reliable first public path:
  - host basics
  - camera smoke
  - local services
  - basic capture bundle
  - local validation
- Expand `Stereo` with a public calibration checklist, stereo capture smoke and failure recovery.
- Expand `Pro` with Jetson installer drift checks, service status evidence and VLM sidecar diagnostics.
- Add a single public E2E command that can report:
  - host OS
  - hardware tier
  - camera readiness
  - VLM policy
  - local capture result
  - whether upload is disabled, skipped or explicitly enabled
- Add contributor-facing evidence templates for:
  - first contributor
  - interrupted install resume
  - agent-guided install
  - returning user readiness check
  - camera troubleshooting
  - upload troubleshooting
- Keep README and `docs/public-validation-matrix.md` in sync when a lane changes.

## Not Yet Claimed

- Full public `Stereo` training readiness.
- Full public `Pro` training readiness.
- Automatic cloud upload from local diagnostic commands.
- Final `SLAM + time-sync` training-ready status before thresholds are frozen.
- A single same-session public proof from iPhone bind/upload through worker,
  public portal and downstream commercial surfaces.

## Evidence Rule

Do not mark a lane complete unless the public command, expected output and
failure behavior are documented. If the evidence came from the internal
runtime repo, link or name that evidence clearly and do not imply that every
public contributor can reproduce it without the same hardware and credentials.
