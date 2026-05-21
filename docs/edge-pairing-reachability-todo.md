# Edge Pairing Reachability Productization ToDo

Date: 2026-05-21
Status: implemented

## Goal

Make phone pairing, capture readiness, and workstation device status describe the real edge-machine state instead of a stale or partially reachable local setup.

## ToDo

1. Workstation pairing contract
   - DONE Add explicit transport metadata to the pairing envelope.
   - DONE Keep `edge_base_url`, `edge_ws_url`, and `status_ui_url` compatible with existing clients.
   - DONE Add endpoint roles and warnings so clients can distinguish HTTP upload, optional fusion WS, and status UI reachability.
   - DONE Allow `/edge/control/profile` through the public workstation proxy.

2. Device registry status semantics
   - DONE Keep raw tokens, token hashes, and local paths private.
   - DONE Derive `pairing_state`, `online_state`, `last_ack_state`, and `lifecycle_state`.
   - DONE Do not show a token-only or history-only record as live/online.

3. Runtime capture page
   - DONE Replace iPhone-only wording with phone wording where Android is supported.
   - DONE Show `已签发/未接入/实时在线/已 ACK/令牌过期` based on derived status.
   - DONE Surface endpoint mismatch warnings from the pairing envelope.

4. Android client
   - DONE Store and preserve `transport_profile`.
   - DONE Allow loopback only for explicit USB-reverse/debug transport, not normal QR/LAN pairing.
   - DONE Surface unreachable QR/address errors in Chinese with concrete next action.

5. iOS client
   - DONE Treat pure EGO scoped-upload mode as HTTP data capture first.
   - DONE Do not pop a destructive "edge machine disconnected" dialog or disarm just because optional fusion WS is down.
   - DONE Keep the long-disconnect safety dialog for teleop/control-critical mode.

6. Validation and release path
   - DONE Run workstation Python tests/syntax checks.
   - DONE Run edge-runtime type/build/smoke checks.
   - DONE Run Android compile/unit tests and install on Xiaomi.
   - BLOCKED Run iOS compile/test if iOS code changed: local Debug simulator and generic iOS builds reach the link phase but fail because the local third-party `UMAPM` framework is missing from DerivedData/Pods search paths, not because of the pairing Swift changes.
   - DOING Push shared DEV lanes or create the required PR for protected public main, then state any remaining merge gap.

## Validation Evidence

- `chek-app`: `:app:compileDebugKotlin :app:testDebugUnitTest` passed; `:chek:installDebug` installed on Xiaomi `21121119SC`; unreachable Edge smoke showed the gate in checking/waiting state with `开始采集` disabled.
- `chek-ego-miner`: `python3 -m py_compile RuView/ui-react/scripts/workstation_server.py` passed; `python3 -m pytest tests/test_workstation_pairing_and_status_ui.py` passed with 13 tests.
- `chek-edge-runtime`: `python3 -m py_compile RuView/ui-react/scripts/workstation_server.py`, `npm run check`, `npm run build`, `node scripts/capture_page_registry_smoke.mjs`, and `python3 scripts/check_cross_repo_module_contract.py --public-repo <path-to-public-repo>` passed.

## Local Data Boundary

The untracked `data/` directory is runtime capture/session evidence and must not be committed with this contract change.
