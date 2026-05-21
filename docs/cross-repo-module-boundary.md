# CHEK EGO Cross-Repo Module Boundary

Date: 2026-05-21
Status: active boundary

This document defines which EGO edge modules are intentionally shared between
`chek-ego-miner` and `chek-edge-runtime`, which repo owns the durable runtime
implementation, and which parts must stay public-safe.

The target is shared semantics, not duplicated operations code.

## Boundary Rules

1. `chek-ego-miner` owns the public contributor promise: install, pair, capture,
   upload explicitly, and verify contribution evidence from public docs.
2. `chek-edge-runtime` owns the deployed edge-machine operations promise:
   service lifecycle, recovery, fleet and QA operations, post-processing, and
   private deployment topology.
3. Both repos must preserve the same pairing, device, session, media-scope,
   upload ACK, health, and profile semantics.
4. Internal hostnames, private operator URLs, raw tokens, token hashes, local DB
   paths, fleet runbooks, and factory recovery details must not be required for
   the public contributor path.
5. When a behavior touches both repos, update the shared contract first, then
   implementation and tests on each side.

## Module Boundary Table

| Module | Shared contract | `chek-ego-miner` public surface | `chek-edge-runtime` runtime surface | Must not drift |
| --- | --- | --- | --- | --- |
| `edge_core` | Device identity, session lifecycle names, profile IDs, runtime health meanings. | Public explanation of what an edge session is and how a contributor verifies it. | Runtime state machines, local service lifecycle, config loading, control-plane integration. | `session_id`, `trip_id`, `profile_id`, health/readiness meanings. |
| `edge_auth_binding` | QR pairing envelope, scoped upload token, token-to-device binding, owner resolution. | Public QR pairing, safe token behavior, stale-pairing recovery copy. | Token authority, SQLite/local registry, private operator/task context. | Scoped token validation, `device_id`, `login_identity`, `operator_id` fallback order. |
| `device_health` | Public device status fields and readiness semantics. | Contributor-facing status: registered, stale, reachable, ready, capturing, uploading, stopped. | Fleet telemetry, recovery actions, private diagnostics, service watchdogs. | `/devices.json` must not label history-only records as currently paired. |
| `edge_capture_usb` | Capture device identity, USB source capability flags, media index semantics. | Public USB/phone setup and basic validation. | Host adapters, USB recovery, factory bring-up, platform-specific services. | Capability flags and manifest identity fields. |
| `edge_capture_stereo` | Stereo profile IDs, calibration artifact names, media scope names. | Public hardware tier docs and validation evidence for Stereo / Pro. | Calibration, QA, post-processing, benchmark and field-maintenance workflows. | Artifact naming, media scope layout, profile IDs. |
| `edge_preview_packager` | Preview bundle metadata and public evidence fields. | Contributor evidence export and safe preview packaging. | Internal preview generation, workstation packaging, QA artifacts. | Evidence manifest fields and privacy boundary. |
| `edge_upload_agent` | Upload ACK, cleaned status, retry-visible states, cloud owner handoff fields. | Explicit upload path and contribution proof. | Upload worker, retry policy, cloud sync, control-plane headers. | ACK semantics, cleaned counts, owner resolution. |
| `edge_pro_extensions` | Pro profile IDs and public capability flags. | Public Pro hardware tier, limits, and validation matrix. | Jetson/VLM/SLAM/professional workers, fleet ops, private performance tuning. | Profile IDs and public capability truthfulness. |
| `jetson_services` | Public capability names and required artifact contract. | Public preflight guidance for Jetson/Pro where supported. | Service installation, GPU/CPU providers, worker orchestration, recovery. | Public capability names and required artifact paths. |
| `local_ui` | QR, readiness, device status, capture HUD, upload progress meanings. | Contributor RuView / WebView path and public-safe labels. | Workstation UI, operator console, debug and fleet views. | UI status semantics: do not show "paired" when only registered history exists. |

## Allowed Duplication

These may exist in both repos:

- Module names and public module summaries.
- JSON schemas, contract fixtures, and validation examples.
- Public-safe scripts for readiness, QR pairing, and basic EGO validation.
- Documentation that describes shared behavior from each repo's audience angle.

## Duplication To Avoid

These should have one canonical rule and cross-repo tests:

- Pairing URL generation and stale/private URL rejection.
- Scoped upload token validation.
- Device registry status mapping.
- Multi-phone owner resolution.
- Upload ACK and cleaned status semantics.
- Media scope layout for Android and iOS tracks.
- Health/readiness transitions shown in UI.

## Device Status Vocabulary

Use these meanings consistently in UI, `/devices.json`, and tests:

- `registered`: a device record exists.
- `paired`: a valid pairing/token binding exists.
- `reachable`: the current edge/workstation URL is reachable from the device.
- `ready`: capture dependencies are satisfied and Start can be enabled.
- `capturing`: a session is actively recording.
- `uploading`: chunks are still being sent or acknowledged.
- `stopped`: capture was stopped for the session.
- `stale`: the stored pairing is not usable anymore.
- `invalid`: pairing, URL, token, or device identity failed validation.

A history-only device record is `registered`, not `paired`. A stale loopback or
private URL must not enable capture on a phone.

## Multi-Phone Ownership

Multiple phones attached to the same edge host are sibling devices. The edge host
does not currently bind to one global main user in the local EGO pairing layer.

Effective session owner resolution is:

1. `session_context.operator_id`.
2. The `user_one_id` encoded in a scoped upload token.
3. Captured manifest identity for audit only, not silent ownership transfer.
4. `unresolved` when no explicit owner exists.

Do not infer owner from pair order, the latest upload, or the first phone that
scanned a QR code.

## Change Workflow

For any cross-repo change:

1. Update `docs/repo-business-contract.md` or this boundary document.
2. Add or update contract fixtures / schemas.
3. Update `chek-ego-miner` public validation if the contributor promise changes.
4. Update `chek-edge-runtime` runtime validation if deployment behavior changes.
5. Run the cross-repo module drift check before merging.

