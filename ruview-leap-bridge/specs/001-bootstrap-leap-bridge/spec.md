# Feature Specification: LEAP Bridge Bootstrap

**Feature Branch**: `001-bootstrap-leap-bridge`  
**Created**: 2026-03-04  
**Status**: Draft  
**Input**: User description: "Bootstrap LEAP bridge dual-hand retargeting with pairing window and deadman"

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Dual-Hand Retargeting with Safety Gate (Priority: P1)

As an operator, I can drive both LEAP hands from teleop input while motion remains blocked
unless arm + deadman conditions are satisfied.

**Why this priority**: It is the minimum safe path for dexterous hand teleoperation.

**Independent Test**: Feed paired left/right frames and verify retarget output only under
valid gate; verify timeout/disarm blocks command output.

**Acceptance Scenarios**:

1. **Given** valid left/right targets and `armed + deadman`, **When** processed, **Then** both hand commands are emitted.
2. **Given** keepalive timeout, **When** new targets arrive, **Then** no command is emitted and fault is reported.

---

### User Story 2 - Pairing Window and Degradation Policy (Priority: P2)

As a controls engineer, I can enforce a <=20ms pairing window for two hands with defined
short/long timeout degradation behavior.

**Why this priority**: Prevents asymmetric hand motion and unstable behavior.

**Independent Test**: Simulate delayed one-hand frames; verify short timeout holds last value,
long timeout degrades to `limit/freeze` policy.

**Acceptance Scenarios**:

1. **Given** hand delta <=20ms, **When** pairing runs, **Then** command pair is considered valid.
2. **Given** one hand stale >=200ms, **When** pairing runs, **Then** configured degraded behavior is applied.

---

### User Story 3 - Hardware Health and Recovery (Priority: P3)

As an operations engineer, I can observe LEAP state (online/temp/error) and recover from
dropout without unsafe output.

**Why this priority**: Field stability depends on controlled response to hardware faults.

**Independent Test**: Inject hand controller disconnect/overtemp states; verify bridge state
changes and output suppression rules.

**Acceptance Scenarios**:

1. **Given** LEAP offline, **When** command publish is attempted, **Then** bridge reports not-ready and does not emit command.
2. **Given** temperature threshold exceeded, **When** loop runs, **Then** output enters protective mode.

---

### Edge Cases

- Left/right hand target lengths differ from expected joint count.
- One hand stream stops while the other continues.
- LEAP reconnect occurs after prolonged dropout.
- Retargeted angles exceed configured per-joint limit.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Bridge MUST parse teleop hand targets and validate dimensions.
- **FR-002**: Bridge MUST retarget left/right hand targets into LEAP command vectors.
- **FR-003**: Bridge MUST enforce control gate (`armed + deadman`) before output.
- **FR-004**: Bridge MUST enforce two-hand pairing window (default <=20ms).
- **FR-005**: Bridge MUST apply degradation policy for one-hand stale data.
- **FR-006**: Bridge MUST enforce per-joint limit and clamp unsafe values.
- **FR-007**: Bridge MUST report LEAP readiness/fault/temperature state.
- **FR-008**: Bridge MUST pin and report active `teleop-protocol` version.

### Key Entities *(include if feature involves data)*

- **HandTargetFrame**: Per-hand input target vector with timestamp and confidence.
- **PairedHandFrame**: Left/right paired frame with pairing delta and validity flag.
- **LeapCommandFrame**: Final command vectors for LEAP left and right devices.
- **LeapHardwareState**: Online status, temperature, and error code for each hand.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Pairing-window validation passes for 100% of <=20ms synthetic test cases.
- **SC-002**: Keepalive timeout blocks output within 200ms in 100% of tests.
- **SC-003**: Over-limit joint targets are clamped in 100% of enforcement tests.
- **SC-004**: Hardware dropout transitions to not-ready state in <=100ms.
