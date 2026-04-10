# Research: Edge Orchestrator Bootstrap

## Decision 1: Runtime and Framework
- **Decision**: Use Rust + `axum` + `tokio`.
- **Rationale**: Matches production real-time baseline and existing project direction.
- **Alternatives considered**:
  - Python/FastAPI: faster iteration but not acceptable for real-time motion gate path.
  - Node.js: strong tooling but GC jitter risk in critical control path.

## Decision 2: Dual Stream Model
- **Decision**: Split `/stream/fusion` and `/stream/teleop` into separate channels and queues.
- **Rationale**: Prevent sensing bursts from starving teleop dispatch.
- **Alternatives considered**:
  - Single multiplexed stream: simpler but failure/backpressure coupling is unsafe.

## Decision 3: Deadman Gate
- **Decision**: Motion output requires `armed=true` and valid keepalive.
- **Rationale**: Enforces explicit operator intent and fail-safe timeout behavior.
- **Alternatives considered**:
  - Arm-only gate: unsafe under network stalls and client freeze.

## Decision 4: Protocol Pinning
- **Decision**: Runtime checks and reports pinned `teleop-protocol` version at startup.
- **Rationale**: Avoids drift between orchestrator and bridge behavior.
- **Alternatives considered**:
  - Best-effort compatibility: ambiguous failures and harder rollback.

## Decision 5: Chunk Lifecycle Baseline
- **Decision**: Implement `received->stored->acked->cleaned` as explicit state machine.
- **Rationale**: Required for recovery correctness and auditability.
- **Alternatives considered**:
  - Stateless ack handling: cannot diagnose replay inconsistencies.
