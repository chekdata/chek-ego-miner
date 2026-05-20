# CHEK EGO Miner / Edge Runtime Business Contract

Date: 2026-05-20
Status: active contract

This document defines how `chek-ego-miner` and `chek-edge-runtime` should line up at the business layer. They are not meant to be file-identical repositories. They are two product lanes that must share the same EGO data, pairing, session, and upload semantics.

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
