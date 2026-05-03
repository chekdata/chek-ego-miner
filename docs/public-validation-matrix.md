# Public Validation Matrix

This matrix tracks what a public contributor can validate from this repository
without relying on private internal knowledge.

## Status Legend

- `supported`: documented and runnable by a contributor
- `in-progress`: documented direction exists, but the public flow is not fully complete
- `guarded`: intentionally requires explicit consent, credentials or upload scope
- `not-claimed`: not supported as a public claim yet

## Lanes

| Lane | Contributor-facing status | Checks | Evidence expectation |
| --- | --- | --- | --- |
| Host basics | `supported` | `python3 scripts/check_host_basics.py --json` | OS, architecture, Python and camera visibility are reported |
| Camera smoke | `supported` | `./cli/chek-ego-miner camera-probe --capture-smoke` | At least one current-session camera frame can be opened and read |
| Lite readiness | `supported` | `./cli/chek-ego-miner readiness --tier lite --json` | Missing tools and supported host facts are explicit |
| Stereo readiness | `in-progress` | `./cli/chek-ego-miner readiness --tier stereo --json` | Stereo devices and calibration gaps are explicit |
| Pro readiness | `in-progress` | `./cli/chek-ego-miner readiness --tier pro --json` | Jetson/runtime services, VLM, stereo and Wi-Fi gaps are explicit |
| Basic local E2E | `supported` | `./cli/chek-ego-miner basic-e2e ... --json` | A local bundle and `public_download/demo_capture_bundle.json` are produced |
| VLM sidecar | `supported for capable hosts` | `fetch-vlm-models`, `vlm-start`, sidecar health/infer | Non-empty inference output, no hidden fallback |
| Training threshold | `candidate only` | `generate_slam_time_sync_benchmark.py`, `validate_training_thresholds.py` | `training_ready=true` only after frozen thresholds and required benchmark metrics |
| Cloud contribution | `guarded` | future bind/upload contributor flow | Must require explicit account, consent, device binding and upload scope |

## Human Scenario Checklist

| Scenario | Public status | Required evidence |
| --- | --- | --- |
| First contributor | `in-progress` | clean machine, tier choice, host check, camera smoke, first local bundle |
| Interrupted install resume | `in-progress` | rerun readiness, detect existing runtime, continue without reinstalling blindly |
| Agent-guided install | `supported` | assistant asks tier, OS, app state and camera hardware before commands |
| Returning user readiness | `in-progress` | compare current host/camera/VLM state with prior known-good setup |
| Camera troubleshooting | `supported` | OS-visible device plus capture smoke result |
| Upload troubleshooting | `guarded` | account/bind/upload scope checked before any upload claim |

## Current Cross-Repo Evidence

The internal runtime repo has fresh 2026-05-03 E2E evidence for physical iPhone
13 Pro plus MacBook, Windows and Ubuntu hosts, including host camera, VLM,
replay display and download. That evidence proves the shared runtime path can
work across those hosts. It does not by itself mean every public contributor
has completed the same cloud contribution path from this repo.

## Data Safety Rule

Local diagnostics and readiness checks stay local. A command should only upload
when its name, documentation and output make that explicit, and when the user
has configured the required account, consent, device binding and upload scope.
