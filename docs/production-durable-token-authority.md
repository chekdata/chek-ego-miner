# Production Durable Token Authority For EGO Edge Pairing

Date: 2026-05-16
Status: Implemented local SQLite / in-process authority

## Summary

Edge Orchestrator now owns the durable local token authority for EGO iPhone pairing. The previous workstation JSON registry remains as a local/pilot compatibility mirror, but production validation prefers the Edge-owned SQLite authority.

Implemented properties:

- short-lived QR pairing envelope is issued by Edge;
- pairing challenge/code state is persisted in SQLite;
- raw scoped upload token is returned once from `POST /pairing/exchange`;
- only SHA-256 token hashes are stored;
- upload validation runs in-process before `/common_task/upload_chunk` accepts data;
- ACK/session/queue status is written back through the authority;
- `/devices.json` exposes public device state without raw tokens, token hashes, or local database paths;
- workstation server can mirror transitional tokens/status into the same SQLite database when it is still serving the RuView status page.

## Runtime Configuration

Primary environment variables:

- `EDGE_PAIRING_AUTHORITY=sqlite`
- `EDGE_UPLOAD_TOKEN_AUTHORITY_DB_PATH=/path/to/token_authority.sqlite3`
- `EDGE_PAIRING_PROFILE_ID=ego_wide_rgbd_multi_iphone_v1`
- `EDGE_PAIRING_TTL_SEC=300`
- `EDGE_PAIRING_UPLOAD_TOKEN_TTL_SEC=3600`
- `EDGE_PAIRING_EDGE_BASE_URL=http://<edge-lan-ip>:8080`
- `EDGE_PAIRING_EDGE_WS_URL=ws://<edge-lan-ip>:8765/stream/fusion`
- `EDGE_PAIRING_STATUS_UI_URL=http://<edge-lan-ip>:3010/#/capture`
- `EDGE_PAIRING_OPERATOR_HINT=<operator label>`
- `EDGE_PAIRING_SCENE_HINT=<scene label>`

If `EDGE_UPLOAD_TOKEN_AUTHORITY_DB_PATH` is omitted and SQLite authority is enabled, the local stack defaults to:

```text
${EDGE_DATA_DIR}/pairing/token_authority.sqlite3
```

The local helper `scripts/teleop_local_stack.sh` now wires this path automatically and starts RuView with `--upload-token-authority-db-path` so transitional workstation pairing/status flows mirror into the Edge authority.

## API Contract

Public endpoints:

- `GET /pairing/envelope`
  - returns the QR payload for RuView and iOS;
  - stores challenge/code/expiry in SQLite.
- `POST /pairing/exchange`
  - validates challenge, code, profile, and expiry;
  - upserts the paired device identity;
  - creates one scoped upload token row;
  - returns the raw `scoped_upload_token` once.
- `GET /devices.json`
  - returns public device rows and `authority.kind="sqlite"`;
  - never returns raw tokens, token hashes, or the SQLite database path.

Protected endpoint:

- `POST /pairing/revoke`
  - requires the Edge bearer token;
  - revokes active tokens by device or token identifier.

Upload endpoint:

- `POST /common_task/upload_chunk`
  - extracts the bearer token;
  - hashes the bearer token;
  - validates hash, expiry, revocation state, device ID, and profile through the in-process authority;
  - falls back to the workstation JSON registry only for local/pilot compatibility;
  - writes `session_context.capture_device_id`, `login_identity`, `device_name`, `pairing_profile_id`, and `upload_auth_kind=scoped_upload_token` into session manifests;
  - records ACK/session/queue state back into the authority after chunk ACK.

## SQLite Schema

The current authority manages these durable tables:

- `paired_devices`
  - device identity, profile, login identity, last seen timestamp, last session ID, upload queue depth, ingest status, and last ACK JSON.
- `pairing_challenges`
  - QR challenge/code, profile, expiry, and consumed timestamp.
- `upload_tokens`
  - token ID, device ID, profile ID, SHA-256 token hash, issue/expiry/revoke timestamps, and status.
- `token_authority_audit`
  - append-only authority events for envelope, exchange, revoke, validation, and status writeback.

## Security Notes

- Raw scoped upload tokens are never written to registry JSON, `/devices.json`, or session manifests.
- `/devices.json` does not expose token hash material or the local SQLite path.
- Scoped uploads must provide `metadata.device_id` when scoped-token validation is configured.
- The global Edge token remains accepted for trusted local control flows; scoped EGO uploads continue to record `upload_auth_kind=scoped_upload_token`.
- Revocation is enforced on subsequent upload requests.
- JSON-registry validation is compatibility fallback only and should not be the production deployment mode.

## Verified Tests

Rust:

```bash
cargo test --manifest-path edge-orchestrator/Cargo.toml
cargo test --manifest-path edge-orchestrator/Cargo.toml sqlite_pairing_authority_exchanges_validates_and_revokes_tokens
```

Python:

```bash
python3 -m pytest tests/test_workstation_pairing_and_status_ui.py tests/test_recompute_edge_session_metadata.py
```

Covered behaviors:

- Edge-owned envelope issue and exchange;
- SQLite token validation through upload ingest;
- wrong-device upload rejection;
- session identity context recorded in manifests;
- in-process ACK/status writeback;
- public `/devices.json` redaction;
- revocation followed by upload rejection;
- workstation transitional exchange mirroring into SQLite.

## Live Local Verification

Latest local stack:

- Edge: `http://192.168.31.91:8080`
- RuView: `http://192.168.31.91:3010/?token=chek-ego-miner-local-token#/capture`
- Fusion WS: `ws://192.168.31.91:8765/stream/fusion`
- SQLite authority path: `/Users/jasonhong/Desktop/开发项目/chek-ego-miner/edge-orchestrator/target/codex-local/teleop-stack/edge-data/pairing/token_authority.sqlite3`

Validated probes:

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/pairing/envelope
curl -sS http://127.0.0.1:8080/devices.json
```

Expected public device authority shape:

```json
{
  "authority": {
    "kind": "sqlite"
  }
}
```

## Operational Readiness Gate

For production rollout, the authority side is no longer just a design document. The remaining release gates are operational:

- push and merge the local branches into the chosen DEV / production lanes;
- deploy the Edge build with `EDGE_PAIRING_AUTHORITY=sqlite`;
- run a post-deploy QR pair/exchange/upload/revoke smoke;
- keep whole-disk storage above the Edge critical threshold before long captures;
- decide whether continuous fisheye is mandatory for `ego_wide_rgbd_multi_iphone_v1`.
