# Implementation Plan: Unitree Bridge Bootstrap

**Branch**: `001-bootstrap-unitree-bridge` | **Date**: 2026-03-04 | **Spec**: `/specs/001-bootstrap-unitree-bridge/spec.md`
**Input**: Feature specification from `/specs/001-bootstrap-unitree-bridge/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a Rust bridge service that consumes `teleop_frame_v1`, enforces control/safety gates,
maps valid commands to Unitree DDS topics, blocks illegal endpoint mapping, and emits
diagnostic bridge state packets.

## Technical Context

**Language/Version**: Rust stable (1.85+)  
**Primary Dependencies**: `serde`, `tokio`, `tracing`, Unitree SDK2/DDS adapter crate  
**Storage**: In-memory runtime state + structured logs  
**Testing**: `cargo test` with fixture replay and mock DDS publisher  
**Target Platform**: Linux ARM64/x86 in LAN-CONTROL segment  
**Project Type**: Stream consumer + protocol bridge  
**Performance Goals**: command mapping+publish p95 <= 20ms  
**Constraints**: deadman gate required, endpoint type guard required, protocol pinning required  
**Scale/Scope**: Single robot body bridge instance per deployment

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Deadman gate enforced before command publish.
- [x] Timeout -> `fault` and stop-output path defined.
- [x] Control/fusion decoupling respected (bridge consumes teleop stream only).
- [x] Protocol source pinned to `teleop-protocol`.
- [x] Reference repositories excluded from runtime artifact source.
- [x] Rust runtime baseline satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/001-bootstrap-unitree-bridge/
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
├── bridge/
│   ├── parser.rs
│   ├── validator.rs
│   ├── mapper.rs
│   ├── gate.rs
│   └── publisher.rs
├── dds/
│   └── unitree_client.rs
├── state/
│   └── bridge_state.rs
└── main.rs

tests/
├── contract/
│   └── test_protocol_schema.rs
├── integration/
│   ├── test_gate_behavior.rs
│   ├── test_endpoint_type_guard.rs
│   └── test_dds_publish.rs
└── unit/
    └── test_mapping.rs
```

**Structure Decision**: Rust single-service bridge with parser/mapper/gate/publisher split
for deterministic control flow and explicit failure codes.

## Complexity Tracking

No constitution violations identified for this bootstrap scope.
