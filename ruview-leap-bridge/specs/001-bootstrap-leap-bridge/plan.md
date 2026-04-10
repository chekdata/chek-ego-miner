# Implementation Plan: LEAP Bridge Bootstrap

**Branch**: `001-bootstrap-leap-bridge` | **Date**: 2026-03-04 | **Spec**: `/specs/001-bootstrap-leap-bridge/spec.md`
**Input**: Feature specification from `/specs/001-bootstrap-leap-bridge/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a Rust LEAP bridge that retargets teleop hand inputs, enforces dual-hand pairing
window, applies safety/deadman gates, and reports hardware readiness/fault states.

## Technical Context

**Language/Version**: Rust stable (1.85+)  
**Primary Dependencies**: `tokio`, `serde`, `tracing`, LEAP SDK binding layer  
**Storage**: In-memory ring buffers for pairing + structured logs  
**Testing**: `cargo test` + integration replay tests  
**Target Platform**: Linux ARM64/x86 near LEAP controllers  
**Project Type**: Stream consumer + hardware command bridge  
**Performance Goals**: dual-hand command publish p95 <= 20ms  
**Constraints**: pairing window <=20ms, stale >=200ms degradation, deadman required  
**Scale/Scope**: Single operator, two LEAP hands per bridge instance

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Deadman/arm gate required before output.
- [x] Timeout -> fault/stop behavior specified.
- [x] Bridge consumes teleop stream only, preserving dual-stream separation.
- [x] Protocol version pinned to `teleop-protocol`.
- [x] Reference repos not used for runtime artifact source.
- [x] Rust runtime baseline met.

## Project Structure

### Documentation (this feature)

```text
specs/001-bootstrap-leap-bridge/
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
│   ├── retarget.rs
│   ├── pairing.rs
│   ├── gate.rs
│   └── publisher.rs
├── leap/
│   └── client.rs
├── state/
│   └── hardware_state.rs
└── main.rs

tests/
├── contract/
│   └── test_protocol_schema.rs
├── integration/
│   ├── test_dual_hand_pairing.rs
│   ├── test_keepalive_timeout.rs
│   └── test_hardware_dropout.rs
└── unit/
    └── test_joint_limits.rs
```

**Structure Decision**: Single Rust bridge with explicit pairing, retarget, gate, and
hardware-state modules for deterministic behavior.

## Complexity Tracking

No constitution violations identified for this bootstrap scope.
