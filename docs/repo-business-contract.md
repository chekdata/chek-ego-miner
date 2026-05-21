# CHEK EGO Miner / Edge Runtime Business Contract

Date: 2026-05-20
Status: active contract

This document defines how `chek-ego-miner` and `chek-edge-runtime` should line up at the business layer. They are not meant to be file-identical repositories. They are two product lanes that must share the same EGO data, pairing, session, and upload semantics.

For module-by-module ownership, allowed duplication, and drift checks, see
[Cross-Repo Module Boundary](./cross-repo-module-boundary.md).

## Repository Roles

| Area | `chek-ego-miner` | `chek-edge-runtime` |
| --- | --- | --- |
| Business role | Public-first contributor entry point. | Internal / self-hosted edge runtime operations lane. |
| Primary user | External contributor using a phone, PC, stereo kit, or Pro edge setup to capture and contribute EGO data. | Internal operator, hardware bring-up engineer, factory / field maintainer, QA operator, and edge fleet owner. |
| Product promise | A contributor can understand the hardware tier, install the local stack, pair phones, capture EGO sessions, upload explicitly, and find contribution evidence from public docs. | A deployed edge machine can install, start, observe, recover, preview, post-process, QA, and upload real device sessions across `basic`, `enhanced`, and `professional` profiles. |
| User-facing surface | Public README, hardware guide, agent guides, public validation matrix, QR pairing / capture-first RuView path, public-safe CLI. | `chek-edge` CLI, local RuView / workstation UI, system services, hardware profile manifests, post-process workers, fleet / QA / debug docs. |
| What should be shared | Pairing envelope contract, scoped upload token semantics, device registry fields, session manifest fields, media scope layout, upload ACK semantics, storage / health status meanings, profile IDs. | Same shared contracts. Runtime may add internal-only hardware and ops fields, but it must not redefine the shared EGO contract. |
| What should differ | Public wording, public installation guidance, safe examples, contributor support paths, public evidence rules. | Internal host runbooks, factory bring-up, deeper hardware recovery, private operator URLs, deployment topology, fleet evidence. |
| Current state | Public QR pairing, scoped-token upload identity, iOS/Android EGO capture evidence, and public boundary docs exist. Remaining public work is evidence expansion for Stereo / Pro, Windows / Jetson, and worker-to-reward-to-download continuity. | Runtime modularization and real-device post-processing are in progress. It has deeper edge operations, upload worker, post-process, QA, benchmark, and RuView integration paths, but the workspace still contains broad WIP that should be normalized before treating it as clean DEV source of truth. |

## Contract Source Of Truth

`chek-ego-miner` is the canonical public source for contributor-facing EGO
semantics. `chek-edge-runtime` is the canonical runtime source for edge-machine
operations. Neither repo may redefine the shared EGO contract in isolation.

Shared contract objects:

- `PairingEnvelope`
- `ScopedUploadToken`
- `DeviceRegistryEntry`
- `SessionManifest`
- `MediaScopeLayout`
- `UploadAck`
- `StorageHealth`
- `ProfileId`
- `OwnerResolution`
- `/devices.json` public status schema
- `PairingTransportProfile`
- `PairingEndpointContract`

When one of these objects changes, update this contract first, then update both
repo implementations and tests.

## Shared Business Contract

The two repos must stay aligned on these fields and behaviors:

- `profile_id` identifies the capture contract, such as `ego_wide_rgbd_multi_iphone_v1`.
- `device_id` identifies the physical phone / capture device and is required for scoped phone uploads.
- `login_identity` records the phone-side account or operator identity captured during QR pairing.
- `session_id` and `trip_id` identify the capture session and upload bundle.
- `operator_id` is the control-plane session owner when a cloud upload path has an explicit operator / task context.
- `task_id` / `task_ids` are consent and task-routing context; required by task-scoped cloud upload policies.
- `upload_auth_kind=scoped_upload_token` distinguishes public phone pairing uploads from trusted local edge-token traffic.
- `/devices.json` must expose public device status but must never expose raw upload tokens, token hashes, or local DB paths.
- Session manifests must carry enough identity to audit where a session came from: `capture_device_id`, `login_identity`, `device_name`, `pairing_profile_id`, and upload auth kind.
- `transport_profile` tells the client whether the pairing is LAN/direct, workstation-proxied HTTP, USB-reverse/debug, or unknown.
- `connectivity_contract` tells the client which endpoint is used for pairing HTTP, Edge HTTP upload/control, optional fusion WS, and status UI.
- `connectivity_warnings` are product-facing warnings. Clients and status pages must surface them as reachability concerns, not as successful pairing.

## Device Status Semantics

Public UI and `/devices.json` must distinguish current reachability from
history. A device record that only proves a past pairing is `registered`, not
currently `paired`.

Use these meanings consistently:

- `registered`: a device record exists.
- `paired`: a valid pairing/token binding exists.
- `reachable`: the current edge/workstation URL is reachable from the device.
- `ready`: capture dependencies are satisfied and Start can be enabled.
- `capturing`: a session is actively recording.
- `uploading`: chunks are still being sent or acknowledged.
- `stopped`: capture was stopped for the session.
- `stale`: the stored pairing is not usable anymore.
- `invalid`: pairing, URL, token, or device identity failed validation.

Stale loopback URLs, private-host URLs, expired tokens, or missing edge services
must not be shown as ready-to-capture pairing.

Derived `/devices.json` fields:

- `pairing_state`: `registered`, `paired`, or `expired`.
- `online_state`: `not_connected`, `active_session`, `acknowledged`, or `stale`.
- `last_ack_state`: `missing` or `acknowledged`.
- `lifecycle_state`: UI-friendly combined state such as `paired_pending_device_status`, `active_session`, `live_ack`, or `stale`.

A row with only `upload_token_status=issued_by_workstation_pairing_endpoint` and no
`session_id`, `upload_queue_depth`, or `last_ack` is a signed pairing/token record
only. It is not live and must not be displayed as current online pairing.

## Pairing Reachability Contract

The pairing envelope stays backward compatible with existing fields:

- `edge_base_url`: HTTP upload/control endpoint used by phone clients.
- `edge_ws_url`: optional fusion/control WebSocket endpoint.
- `status_ui_url`: workstation status page.

New clients should also read:

- `transport_profile`: `lan_direct`, `workstation_proxy`, `usb_reverse`, or `unknown`.
- `connectivity_contract.endpoints`: role-labeled endpoint URLs.
- `connectivity_contract.required_for_capture`: `edge_http` and `pairing_http` are required for pure EGO capture; `edge_ws` is optional unless teleop/control mode is enabled.
- `connectivity_warnings`: warnings such as loopback-only addresses or a WS URL that appears to be advertised from a loopback source.

Pure EGO capture is allowed to continue when HTTP upload is healthy and optional
fusion WS is unavailable. Teleop/control-critical modes may still disarm on long
WS/control disconnects.

## Multi-Phone Ownership Model

There is currently no single global "main user" bound to the whole edge machine by the EGO pairing layer.

The active model is per-device and per-session:

1. Each phone scans the QR pairing envelope and exchanges it with its own `device_id`, optional `device_name`, and `login_identity`.
2. The edge / workstation registry stores each paired phone as a separate row keyed by `device_id`.
3. Scoped phone uploads must include `metadata.device_id`; Edge validates the scoped token against that device and records the phone identity into the session manifest.
4. If the session later syncs to the cloud control plane, the effective session owner is `session_context.operator_id` first. If that is absent, the runtime can fall back to the `user_one_id` encoded in a `crowd-scope::<user_one_id>::<capture_device_id>::...` upload scope token.
5. Concurrent phones are siblings under the same edge host. One phone does not automatically become the parent, owner, or "main user" for other phones.

In product terms: the edge host may be operated by one contributor account, but the current technical source of truth is the session owner plus per-device phone identity, not a global host owner field.

If the product later needs a visible "edge owner" or "primary contributor" concept, add it as an explicit binding object in the control plane and mirror it into local status. Do not infer it from the first phone that paired or the most recent phone that uploaded.

## Decision Rule

When a behavior touches public contribution, keep `chek-ego-miner` readable and reproducible from public docs. When a behavior touches hardware recovery, private deployment, or fleet operations, keep it in `chek-edge-runtime` unless it becomes part of the public contributor promise.

When both repos need the same behavior, converge on a shared contract, generated artifact, or versioned package. Do not let the same business rule drift in two independent implementations.

Before merging a cross-repo behavior change, run the module boundary drift check
from either repo and keep the result with the validation evidence.
