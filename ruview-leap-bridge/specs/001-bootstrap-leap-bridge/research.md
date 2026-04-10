# Research: LEAP Bridge Bootstrap

## Decision 1: Pairing Window Enforcement
- **Decision**: Enforce left/right pairing window <= 20ms.
- **Rationale**: Keeps two-hand output coherent and avoids asymmetric jitter.
- **Alternatives considered**:
  - Independent per-hand publish: easier but can destabilize teleop behavior.

## Decision 2: Stale-Hand Degradation
- **Decision**: `<200ms` hold previous hand target; `>=200ms` enter degrade mode.
- **Rationale**: Smooths transient loss and fails safe on sustained loss.
- **Alternatives considered**:
  - Immediate freeze on any delay: too sensitive to normal network variance.

## Decision 3: Joint Safety Limits
- **Decision**: Clamp retarget output per configured joint limits before publish.
- **Rationale**: Prevents invalid angle bursts and mechanical stress.
- **Alternatives considered**:
  - Assume upstream already valid: unsafe for runtime resilience.

## Decision 4: Hardware State Feedback
- **Decision**: Track online/temp/error state from LEAP devices in runtime state.
- **Rationale**: Enables safe suppression and fast recovery decisions.
- **Alternatives considered**:
  - Fire-and-forget publish only: no control over degraded hardware states.

## Decision 5: Runtime Baseline
- **Decision**: Rust service for publish loop and gating path.
- **Rationale**: Aligns with production non-GC control-path requirement.
- **Alternatives considered**:
  - Python control service: unsuitable for hard timeout and deterministic loop.
