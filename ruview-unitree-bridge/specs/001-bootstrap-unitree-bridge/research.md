# Research: Unitree Bridge Bootstrap

## Decision 1: Safety Before Publish
- **Decision**: Apply gate (`armed + deadman + safety_state`) before DDS publish.
- **Rationale**: Ensures no command reaches robot in unsafe state.
- **Alternatives considered**:
  - Post-publish gate in downstream layer: too late for command suppression.

## Decision 2: Endpoint Type Guard
- **Decision**: Enforce `end_effector_type` allow/deny matrix in bridge.
- **Rationale**: Prevent wrong-hand topic routing and actuator mismatch.
- **Alternatives considered**:
  - Guard only in orchestrator: bridge remains vulnerable to malformed direct input.

## Decision 3: Mapping Path
- **Decision**: Prefer `arm_q_target` direct mapping; fallback to IK from wrist pose.
- **Rationale**: Deterministic and lower latency when q-target exists.
- **Alternatives considered**:
  - Always IK: more compute and failure surface.

## Decision 4: Fault Observability
- **Decision**: Emit `bridge_state_packet` with `is_ready`, `fault_code`, and timestamps.
- **Rationale**: Required for field diagnosis when "has frame but no motion."
- **Alternatives considered**:
  - Logs only: insufficient for runtime UI and health checks.

## Decision 5: Protocol Compatibility
- **Decision**: Fail startup on protocol incompatibility.
- **Rationale**: Avoid hidden runtime drift.
- **Alternatives considered**:
  - Soft warning: unsafe in production deployments.
