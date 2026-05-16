# EGO Miner Easy Install, QR Pairing, and Status UI Plan

Date: 2026-05-12

## What Exists Today

`chek-ego-miner` already has multi-OS install scaffolding:

- `install/backends/macos.py`
- `install/backends/linux.py`
- `install/backends/windows.py`
- `install/backends/jetson.py`

The CLI already forwards to install / status / start / bind / service-install / public-e2e entry points, and the repo already has a workstation server under `RuView/ui-react/scripts/workstation_server.py` plus `scripts/teleop_local_stack.sh` for local launches.

2026-05-12 execution update:

- `RuView/ui-react/dist` now contains the capture-first RuView build from the source workstation UI.
- The workstation server now exposes short-lived pairing and local device-registry endpoints.
- The workstation server now forwards capture-page status dependencies, including `/live-preview.json`, `/storage/status`, `/storage/sessions`, and the matching `/edge/...` proxy paths.
- The capture UI now prefers `/devices.json` registry rows for the device table, so paired iPhone device IDs and login identities can be displayed before the full Edge registry lands.
- The workstation server can persist the device registry to disk and accept narrow per-device status updates at `/devices/status` for session, last ACK, queue depth, and ingest status.
- Edge chunk ingest can optionally PATCH workstation `/devices/status` on chunk ACK when `EDGE_WORKSTATION_DEVICE_STATUS_URL` is configured; the local stack wires this URL to the RuView workstation server by default.
- Edge chunk ingest can validate pairing-issued scoped upload tokens from the shared workstation device registry when `EDGE_UPLOAD_TOKEN_REGISTRY_PATH` points at that registry. The registry stores only `upload_token_sha256`, not the raw token.
- Scoped-token upload now writes session identity audit fields into the exported manifests: `capture_device_id`, `login_identity`, `device_name`, `pairing_profile_id`, and `upload_auth_kind`.
- Local integration tests cover two scoped-token devices uploading concurrently to one Edge process, including per-session identity isolation. This is not a substitute for the required two-real-iPhone LAN test.
- `chek-edge-runtime` now has a local browser smoke for the capture page that serves the production RuView bundle with mocked Edge / registry endpoints and verifies two iPhone registry rows are visible.
- `chek-ego-miner status --open-ui` can open the capture page on demand; `install` and `service restart` open it by default and keep `--no-open-ui` as the escape hatch.
- CLI `install`, `status`, and `service restart` responses now include an `operator_receipt` with the capture UI URL, pairing envelope URL, device registry URL, LAN readiness hint, and the core quality targets.
- `chek-ego-miner start` now gives operators a one-command local path: it forwards to `service restart --direct`, defaults to `--profile basic` for pure EGO capture unless a profile is explicitly supplied, and opens RuView `/capture` through the runtime restart flow.
- The local stack default route and printed viewer URL now point at `#/capture`.
- The iOS repo now has implementation notes and JSON contract fixtures under `/Users/jasonhong/Desktop/开发项目/chekapp-ios-dev-cloud/specs/004-ego-wide-depth-multi-iphone-qr/` for the QR envelope, pairing exchange, scoped upload metadata, and public device registry.
- The iOS repo now has the QR pairing client/store layer: decode QR JSON, validate envelope, call `POST /pairing/exchange`, persist the scoped upload session, prefer paired Edge LAN URLs, bind uploads to paired `device_id`, and use the scoped token for upload auth.
- The iOS Teleop Capture Gate now exposes a RuView QR scanner sheet plus paste-JSON fallback and invokes `EdgePairingClient.pair(...)`.
- 2026-05-15 / 2026-05-16 CST real-device update: Zexin iPhone completed a hotspot LAN pairing and short capture against `http://172.20.10.2:3010/edge`, proving the single-iPhone QR/scoped-token path, upload ACK status writeback, and Edge HTTP capture-pose ingest on a real device.
- 2026-05-16 CST two-device update: two real iPhones paired to one workstation/Edge host on the same LAN (`192.168.31.91`), both produced scoped-token upload ACKs, and both generated ready-for-upload EGO sessions with device/login/auth context in `session_context`.
- Production token authority is now implemented as an Edge-owned SQLite / in-process authority. See `docs/production-durable-token-authority.md`.

What is still missing for the new dataset direction:

- continuous fisheye/secondary track validation for profiles that require more than a single auxiliary snapshot;
- a longer multi-iPhone soak after whole-disk storage pressure is cleared;
- production GitOps / DEV merge and environment sync.

## Product Goal

Make the public install path feel like:

1. one installer or one command per OS;
2. launch services;
3. automatically open the local frontend status page;
4. show Edge HTTP health, QR code, LAN IP, storage, QA, and VLM status;
5. let multiple iPhones pair by scanning the QR code;
6. record each iPhone's login identity and stable device ID.

## Target Install Experience

Recommended installation shape:

```bash
./cli/chek-ego-miner install --profile basic --apply --system-install --enable-services
./cli/chek-ego-miner start --bind-host 0.0.0.0 --public-host <edge-lan-ip>
```

The second command should:

- open the browser automatically on macOS / Windows / Linux where supported;
- point to a local status page, ideally `http://127.0.0.1:3010` or a nearby standard port;
- show the current Edge base URL and LAN IP;
- render a QR code that iPhones can scan;
- show live connection cards for every paired iPhone;
- expose a per-device status timeline and last upload ack.

## QR Pairing Payload

The QR code should encode a short-lived pairing envelope, not a long-lived bearer token. The workstation server now serves this shape from:

- `GET /pairing/envelope`
- `GET /pairing/envelope.json`
- `GET /api/v1/pairing/envelope`

```json
{
  "type": "chek_ego_edge_pairing",
  "version": "1.0",
  "profile_id": "ego_wide_rgbd_multi_iphone_v1",
  "edge_base_url": "http://192.168.1.20:8080",
  "status_ui_url": "http://192.168.1.20:3010",
  "pairing_code": "123456",
  "pairing_challenge": "base64url-random",
  "expires_at": "2026-05-12T10:30:00Z",
  "operator_hint": "optional-human-label",
  "scene_hint": "optional-scene-label"
}
```

Required pairing behaviors:

- QR refreshes before expiry. Current status: implemented in RuView and workstation server.
- iOS scans the code and exchanges the pairing challenge for a scoped upload token. Current status: endpoint implemented at `POST /pairing/exchange`; the iOS client path has been validated on one-phone hotspot LAN and two-phone same-LAN runs.
- Edge records both the device ID and the login identity used during pairing. Current status: workstation-server registry persists the identity, stores only the scoped-token hash, and Edge upload ingest can validate the scoped token against `metadata.device_id`.
- Edge records pairing identity into session metadata. Current status: successful scoped-token uploads refresh `manifest.json`, `demo_capture_bundle.json`, and `upload/upload_manifest.json` with device/login/auth context.
- A single Edge host may hold multiple active iPhones at once. Current status: UI, registry contract, upload-token validation, a two-device local concurrent upload smoke, and a two-real-iPhone LAN capture all support multiple devices.

## Real Device Evidence

Latest verified two-iPhone LAN run:

- Date/time: 2026-05-16 09:12-09:15 CST.
- Devices: two real iPhones paired through the same RuView status UI envelope against workstation/Edge host `192.168.31.91`.
- Runtime URLs: `status_ui_url=http://192.168.31.91:3010/#/capture`, `edge_base_url=http://192.168.31.91:3010/edge`, `edge_ws_url=ws://192.168.31.91:8765/stream/fusion`.
- Bridge logs:
  - `/Users/jasonhong/Desktop/开发项目/chekapp-ios-dev-cloud/.temp/ios/device_bridge_logs/bridge-zexin-16pro-ego-two-phone-20260516-091517.jsonl`
  - `/Users/jasonhong/Desktop/开发项目/chekapp-ios-dev-cloud/.temp/ios/device_bridge_logs/bridge-zexin-13pro-ego-two-phone-20260516-091517.jsonl`
- Good log markers: `pairing_exchange_ok`, `launch_bootstrap_ok`, `auto_enter_allowed`, `session_start_ok`, `teleop_phone_back_wide_depth`, `primary_frame_packet`, `upload_ok`, and `chunk_ack stored`.
- Workstation registry status: `/devices.json` reported two distinct device IDs with `ingest_status=acked`, `upload_queue_depth=0`, and per-device `session_id` / `last_ack`.
- Device/session A: `device_id=BDFAAA1C-E772-4A2C-B344-6421F98560D7`, session `bd550c06-324e-421b-81a0-cd868ece2d1d`, `last_ack.chunk_index=55`.
- Device/session B: `device_id=EF2C2093-1969-499A-AE82-3255818B539E`, session `8b1481e7-3ca1-4da7-97d2-c79f9688dd78`, `last_ack.chunk_index=21`.
- Manifest identity: both `manifest.json`, `demo_capture_bundle.json`, and `upload/upload_manifest.json` carry the scoped pairing context under `session_context`, including `capture_device_id`, `login_identity`, `device_name`, `pairing_profile_id`, and `upload_auth_kind=scoped_upload_token`.
- QA status after metadata recompute: both sessions report `ready_for_upload=true`, `status=retry_recommended`, `score_percent=91.67`, and `missing_artifacts=["fisheye_track_present"]`.
- Core EGO artifacts present in both sessions: `capture_pose_present`, `pose_imu_present`, `raw_depth_present`, `iphone_calibration_present`, `time_sync_present`, `human_demo_pose_present`, `teleop_frame_present`, `media_tracks_present`, and `robot_state_present` treated as satisfied because `control_enabled=false`.
- Line counts after recompute:
  - `bd550c06-324e-421b-81a0-cd868ece2d1d`: `raw/iphone/wide/kpts_depth.jsonl=1442`, `pose_imu=722`, `depth/index=722`, `raw/iphone/depth/media_index.jsonl=56`, `sync/time_sync_samples.jsonl=46`, `chunks/chunk_state.jsonl=189`.
  - `8b1481e7-3ca1-4da7-97d2-c79f9688dd78`: `raw/iphone/wide/kpts_depth.jsonl=114`, `pose_imu=57`, `depth/index=57`, `raw/iphone/depth/media_index.jsonl=23`, `sync/time_sync_samples.jsonl=18`, `chunks/chunk_state.jsonl=86`.
- Storage caveat: `/storage/status` remained `critical` with rolling pool `over_hard_limit`; this run intentionally stayed short. Do not run long captures until storage is cleaned or moved.

Merged review media generated from that run:

- Session `bd550c06-324e-421b-81a0-cd868ece2d1d`
  - Main review video: `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/bd550c06-324e-421b-81a0-cd868ece2d1d/derived/media/iphone_main_merged.mp4`
  - Fisheye auxiliary review video: `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/bd550c06-324e-421b-81a0-cd868ece2d1d/derived/media/iphone_fisheye_merged.mp4`
  - Media manifest: `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/bd550c06-324e-421b-81a0-cd868ece2d1d/derived/media/media_manifest.json`
  - Materialization summary: 56 main chunks, 416 main frames, H.264 1280x720, duration 113.383333 seconds.
- Session `8b1481e7-3ca1-4da7-97d2-c79f9688dd78`
  - Main review video: `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/8b1481e7-3ca1-4da7-97d2-c79f9688dd78/derived/media/iphone_main_merged.mp4`
  - Fisheye auxiliary review video: `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/8b1481e7-3ca1-4da7-97d2-c79f9688dd78/derived/media/iphone_fisheye_merged.mp4`
  - Media manifest: `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/8b1481e7-3ca1-4da7-97d2-c79f9688dd78/derived/media/media_manifest.json`
  - Materialization summary: 22 main chunks, 163 main frames, H.264 1280x720, duration 43.868333 seconds.

Why the first session originally had no merged video:

- The phone uploaded per-chunk H.264/depth media correctly, and Edge persisted those chunk indexes as canonical raw artifacts.
- The previous recorder/upload manifest did not materialize a single review MP4 from `raw/iphone/wide/media_index.jsonl`, even though the dataset docs expected one.
- `scripts/materialize_session_media.py` now fills that gap and the recorder manifest records the derived paths under `derived_media_manifest`, `iphone_main_merged_video`, and `iphone_fisheye_merged_video`.

Retained local test data:

- `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/bd550c06-324e-421b-81a0-cd868ece2d1d`
- `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/8b1481e7-3ca1-4da7-97d2-c79f9688dd78`
- `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/session/48e661b3-f251-4b6a-81e9-8c23a80b2a5c`

Session directory map:

- `raw/`: original iPhone data, including wide RGB video chunks, depth indexes, pose/IMU, keypoints, and auxiliary fisheye media.
- `chunks/`: per-chunk receive/store/ack state used by uploader and retry logic.
- `sync/`: time sync samples and frame correspondence records.
- `calibration/`: iPhone capture calibration exported with the session.
- `fused/`: human-demo/fusion artifacts consumed by QA and downstream dataset tooling.
- `teleop/`: compatibility frame artifacts; still present even when `control_enabled=false` for pure EGO.
- `qa/`: local quality report and upload/readiness decisions.
- `upload/`: upload manifest, queue state, and exported session metadata.
- `derived/`: post-processed artifacts such as merged review videos.
- `labels/`, `preview/`, and `clips/`: labels, keyframes, previews, and optional clips used by review tooling.

Local storage cleanup performed:

- Old reject/probe/smoke sessions were removed.
- The three ACKed true-device sessions above were retained as protected evidence.
- Edge rolling pool is now empty/tracked; `/storage/status` remains `critical` because the whole disk free ratio is still under the Edge critical threshold, not because stale Edge sessions remain.
- Long captures and soak runs should wait until the machine has substantially more free disk space or the Edge data directory is moved to a larger volume.

Latest verified single-iPhone run:

- Date/time: 2026-05-15 16:29-16:30 CST.
- Device: Zexin, `device_id=EF2C2093-1969-499A-AE82-3255818B539E`, profile `ego_wide_rgbd_multi_iphone_v1`.
- Runtime URLs: `status_ui_url=http://172.20.10.2:3010/#/capture`, `edge_base_url=http://172.20.10.2:3010/edge`, `edge_ws_url=ws://172.20.10.2:8765/stream/fusion`.
- Session: `48e661b3-f251-4b6a-81e9-8c23a80b2a5c`.
- Bridge log: `/Users/jasonhong/Desktop/开发项目/chekapp-ios-dev-cloud/.temp/ios/device_bridge_logs/bridge-ego-20260516-003041.jsonl`.
- Good log markers: `pairing_exchange_ok`, `launch_bootstrap_ok`, `health_ok`, `scoped_ego_pairing_health_ok_skip_legacy_gate`, `auto_enter_allowed`, `session_start_ok`, `phone_vision_http infer_ok`, `diagnostic_reason=vision_3d_ok`, `capture_pose send_http_primary accepted=1`, `upload_ok`, `chunk_ack stored`, and `chunk_cleaned posted_ok`.
- Device registry status: `/devices.json` reported `session_id=48e661b3-f251-4b6a-81e9-8c23a80b2a5c`, `last_ack.chunk_index=27`, `upload_queue_depth=0`, and `ingest_status=acked`.
- Manifest identity: `capture_device_id`, `login_identity`, `device_name`, `pairing_profile_id`, and `upload_auth_kind=scoped_upload_token` are present.
- QA status after metadata recompute: `ready_for_upload=true`, `status=retry_recommended`, `missing_artifacts=["fisheye_track_present"]`.
- Core EGO artifacts present: `capture_pose_present`, `pose_imu_present`, `raw_depth_present`, `iphone_calibration_present`, `time_sync_present`, `human_demo_pose_present`, `teleop_frame_present`, `media_tracks_present`.
- Line counts: `raw/iphone/wide/kpts_depth.jsonl=342`, `raw/iphone/wide/pose_imu.jsonl=171`, `raw/iphone/wide/depth/index.jsonl=171`, `raw/iphone/depth/media_index.jsonl=28`, `sync/time_sync_samples.jsonl=23`, `chunks/chunk_state.jsonl=84`.
- Pure EGO QA rule: `robot_state_present` is now optional when `control_enabled=false`; the verified session reports `raw/robot/state.jsonl 行数=0（control disabled by runtime profile）`.
- Remaining recommendation: `raw/iphone/fisheye/media_index.jsonl` has one auxiliary snapshot, not a continuous fisheye track. Treat this as non-blocking for the scoped EGO upload path, but still unresolved if the dataset profile requires continuous auxiliary/fisheye media.
- Storage caveat: local storage was still `critical` / rolling pool `over_hard_limit`; avoid long captures until storage is cleaned or moved.

## Status UI

The default UI should emphasize the capture workflow, not teleop. Suggested panels:

- `Edge` health and LAN URL.
- `QR Pairing` card with refresh timer.
- `Devices` table with device name, stable ID, login identity, profile, last ack, session state, upload queue, and local network status.
- `Storage` card with free space and estimated time remaining.
- `Media QA` card with RGB FPS, bitrate, depth alignment, and reject reasons.
- `VLM` card with queue depth and semantic summary state.
- `Logs` card with last pairing, last upload, and last reject event.

## Install Backend Assessment

Current backend support is already broader than the old public docs imply:

- macOS: currently optimized for basic workstation-style bring-up and launchd user agents.
- Linux: standard host bring-up and systemd-user support.
- Windows: workstation / mini PC support with task-scheduler wrappers.
- Jetson: professional / legacy runtime compatibility.

That means the install story does not need a new OS matrix. What it does need is:

- a thinner, friendlier top-level command;
- automatic browser launch;
- clear UI-first guidance;
- an operator-friendly pairing screen;
- device registry state displayed in the frontend.

## Suggested Implementation Backlog

Current execution status:

- [x] Add a dedicated `capture` / `devices` route in the RuView local frontend.
- [x] Make the capture route the default local frontend entry.
- [x] Hide teleop-heavy workspaces from the default sidebar while keeping them reachable for debug with `?nav=all`.
- [x] Render a QR pairing payload on the frontend with code, expiry, copy action, and LAN URL warning.
- [x] Show Edge HTTP health, iPhone ingress, synthetic device row, storage, media QA, and VLM status from existing status sources.
- [x] Add `status --open-ui` launcher behavior in `chek-ego-miner`.
- [x] Add `install --open-ui` and `service restart --open-ui` optional browser launch.
- [x] Move QR payload generation from frontend-only draft to a local ego-miner workstation endpoint with server-side expiry and challenge tracking.
- [x] Add a local per-device registry model to the workstation server.
- [x] Add a scoped token exchange endpoint for the QR pairing challenge.
- [x] Make the local stack default profile capture-first and print `#/capture` as the viewer route.
- [x] Package the capture-first RuView build under `RuView/ui-react/dist` for this repo.
- [x] Allow the workstation server to proxy the capture page's live-preview and storage status dependencies.
- [x] Replace the synthetic-only RuView device table with `/devices.json` registry rows when available.
- [x] Add persistent workstation-server registry storage and a narrow `/devices/status` endpoint for per-device ACK / queue / session updates.
- [x] Add optional Edge chunk-ACK-to-workstation status updates via `EDGE_WORKSTATION_DEVICE_STATUS_URL`.
- [x] Make Edge orchestrator accept and validate the scoped upload token issued by pairing through `EDGE_UPLOAD_TOKEN_REGISTRY_PATH`.
- [x] Bind scoped-token uploads into exported session metadata with `device_id + login_identity + session_id` audit fields.
- [x] Add a local two-device concurrent scoped-token upload smoke test and fix temp upload filename collisions under concurrent multipart uploads.
- [x] Wire the visible iOS QR scanner / capture screen action to the new `EdgePairingClient.pair(...)` path.
- [x] Add iOS pairing client/store support for QR envelope validation, `POST /pairing/exchange`, paired Edge URL selection, scoped upload token auth, and paired `device_id` upload metadata.
- [x] Open the browser automatically by default after install and service restart where safe, with `--no-open-ui` available.
- [x] Add install/status operator receipt hints for macOS, Linux, Windows, and Jetson-friendly JSON output.
- [x] Add tests for QR serialization, pair expiry, wrong-code rejection, multiple iPhone registry payloads, status updates, and CLI operator receipts.
- [x] Add a browser-level capture page smoke that proves registry rows render as visible device cards.
- [x] Add a public one-command `chek-ego-miner start` path for local services + RuView capture page.
- [x] Add iOS-side implementation notes and JSON contract fixtures for the QR pairing / scoped upload integration.
- [x] Validate one real iPhone over hotspot LAN through QR/scoped pairing, auto-enter capture, HTTP capture-pose ingest, scoped upload, ACK writeback, and manifest identity fields.
- [x] Validate two real iPhones on one LAN through concurrent QR/scoped pairing, upload ACK writeback, distinct session IDs, and manifest identity fields.

## Concrete Rectification TODO

P0 implemented in this pass:

- RuView default entry is the capture device page.
- Teleop / Wi-Fi / control / engineering routes are hidden from normal sidebar use.
- Capture page no longer depends on `/control/state` for QA, so pure EGO usage is not blocked by teleop auth.
- Workstation server issues short-lived QR pairing envelopes.
- Workstation server can exchange a pairing challenge and record device ID + login identity in a local registry.
- Workstation server can persist the registry on disk and reload it after restart.
- Workstation server exposes `/devices/status` for status writers to update `session_id`, `upload_queue_depth`, `last_ack`, and `ingest_status`.
- Edge upload chunk ACK can update `/devices/status` when `EDGE_WORKSTATION_DEVICE_STATUS_URL` is set; local stack sets it to the workstation server.
- Edge upload ingest validates scoped pairing tokens from the registry when `EDGE_UPLOAD_TOKEN_REGISTRY_PATH` is set, and binds scoped uploads to `metadata.device_id`.
- Exported manifests include scoped upload identity context, so downstream dataset consumers can audit which paired iPhone/login produced a session.
- Concurrent local uploads now use collision-resistant temp names and have a two-device scoped-token regression test.
- Workstation server no longer rejects the capture page's core live-preview/storage paths at the proxy whitelist.
- RuView device table reads the local registry when available and falls back to live-preview when the registry endpoint is absent.
- `chek-ego-miner status --open-ui` opens the capture status page.
- CLI JSON output now includes `operator_receipt` so install/status screens can show the pairing URL, LAN readiness, and 30-second usable-episode quality target without asking operators to read logs.
- Local stack default runtime profile is capture-first and prints capture URLs.
- Edge now exposes `POST /ingest/capture_pose`, so iOS HTTP primary capture-pose ingest no longer depends on WebSocket fallback.
- The local phone-vision sidecar dependency set includes `rtmlib`, and the verified run produced `vision_3d_ok` with body/hand 3D points.
- QA now accepts depth from either the legacy wide depth index or the chunk media index and treats missing robot state as non-blocking when `control_enabled=false`.

P0 still required before real multi-iPhone collection:

- Decide whether `ego_wide_rgbd_multi_iphone_v1` requires continuous fisheye media. If yes, fix the iOS secondary/fisheye capture path and rerun a short capture; if no, relax/document the QA recommendation for this profile.
- Clear or move local storage before any long capture or multi-iPhone soak; Edge rolling sessions are clean, but `/storage/status` is still `critical` due to whole-disk pressure.

P1 recommended before pilot:

- Keep the iOS contract fixtures current as the app implementation lands, especially if upload metadata field names change during Swift integration.

P2 after pilot:

- Add VLM queue/status details per session and per device.
- Add a concurrency soak report for multi-iPhone upload and storage pressure.

## Current Risks

- If the UI still centers teleop or fusion, operators will treat the page as a control console instead of a capture console.
- If the install path still requires several manual shell commands, the “easy install” goal will not land.
- If QR payloads contain long-lived tokens, revocation and leak risk will be too high.
- If device identity is only human-readable labels, multiple phones on one Edge will become ambiguous fast.

## Useful Existing Entry Points

- `./cli/chek-ego-miner install`
- `./cli/chek-ego-miner status`
- `./cli/chek-ego-miner bind`
- `./cli/chek-ego-miner public-e2e --tier lite`
- `RuView/ui-react/scripts/workstation_server.py`
- `scripts/teleop_local_stack.sh`

## Open Questions

- Whether `status` should also open the browser by default. Current behavior keeps `status` non-invasive and explicit behind `--open-ui`, while `install` / `service restart` default to opening RuView.
- Whether the QR page should live on port `3010` or be served from the same Edge port for simpler pairing.
