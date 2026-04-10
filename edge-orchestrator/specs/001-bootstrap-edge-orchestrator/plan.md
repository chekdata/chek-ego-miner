# Implementation Plan: Edge Orchestrator Bootstrap

**Branch**: `001-bootstrap-edge-orchestrator` | **Date**: 2026-03-04 | **Spec**: `/specs/001-bootstrap-edge-orchestrator/spec.md`
**Input**: Feature specification from `/specs/001-bootstrap-edge-orchestrator/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Implement an initial edge orchestrator service in Rust that provides session/safety APIs,
dual WebSocket streams (`/stream/fusion`, `/stream/teleop`), deadman-based control gating,
chunk lifecycle tracking, and protocol-version pinning/validation.

## Technical Context

**Language/Version**: Rust stable (1.85+)  
**Primary Dependencies**: `axum`, `tokio`, `serde`, `tracing`, `metrics`  
**Storage**: Local files/JSONL for chunk state and logs (MVP), in-memory cache for live gate state  
**Testing**: `cargo test` (unit/integration/contract)  
**Target Platform**: Linux on Jetson Orin Nano and equivalent ARM64/x86 edge hosts  
**Project Type**: Web service + stream gateway  
**Performance Goals**: teleop dispatch p95 <= 30ms, keepalive timeout reaction <= 200ms  
**Constraints**: Deadman mandatory for motion, dual-stream isolation, explicit protocol version pinning  
**Scale/Scope**: Single-site edge node, 1-2 operator clients, continuous 8h sessions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Deadman safety gate defined (`armed + deadman.pressed` before motion output).
- [x] Keepalive timeout defined (`timeout -> fault`, immediate motion stop).
- [x] Dual-stream isolation defined (`/stream/teleop` independent from `/stream/fusion`).
- [x] Protocol source pinned to `teleop-protocol`.
- [x] `teleops-reference/*` excluded from release provenance.
- [x] Production runtime path is Rust.

## Project Structure

### Documentation (this feature)

```text
specs/001-bootstrap-edge-orchestrator/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── api/
│   ├── routes_session.rs
│   ├── routes_safety.rs
│   ├── routes_chunk.rs
│   └── routes_health.rs
├── ws/
│   ├── stream_fusion.rs
│   └── stream_teleop.rs
├── control/
│   ├── gate.rs
│   └── keepalive.rs
├── recorder/
│   └── chunk_state_machine.rs
├── protocol/
│   └── version_guard.rs
└── main.rs

tests/
├── contract/
│   ├── test_protocol_guard.rs
│   └── test_api_contract.rs
├── integration/
│   ├── test_deadman_timeout.rs
│   ├── test_disarm_gate.rs
│   └── test_stream_isolation.rs
└── unit/
    └── test_chunk_state_machine.rs
```

**Structure Decision**: Single Rust service layout with explicit module boundaries for
API, stream handling, gating, recorder state machine, and protocol guards.

## Complexity Tracking

No constitution violations identified for this bootstrap scope.
