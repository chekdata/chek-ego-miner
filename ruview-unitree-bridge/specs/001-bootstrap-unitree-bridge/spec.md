# Feature Specification: Unitree Bridge Bootstrap

**Feature Branch**: `001-bootstrap-unitree-bridge`  
**Created**: 2026-03-04  
**Status**: Draft  
**Input**: User description: "Bootstrap Unitree bridge from teleop_frame_v1 to DDS with safety gates"

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

### User Story 1 - Safe Teleop to DDS Mapping (Priority: P1)

As a controls engineer, I can feed `teleop_frame_v1` into bridge and get validated Unitree
DDS outputs only when safety gate conditions are satisfied.

**Why this priority**: It is the core production path for body teleoperation.

**Independent Test**: Replay fixture frames and verify correct DDS message fields while
enforcing deadman/disarm/estop gating.

**Acceptance Scenarios**:

1. **Given** valid teleop frame and `armed + deadman`, **When** frame is processed, **Then** DDS command is emitted.
2. **Given** `disarmed` or deadman timeout, **When** frame is processed, **Then** no motion command is emitted.

---

### User Story 2 - Endpoint Type Guard (Priority: P2)

As a safety reviewer, I can ensure `end_effector_type` is honored so LEAP mode does not
accidentally publish Dex topics and vice versa.

**Why this priority**: Prevents wrong actuator path and field incidents.

**Independent Test**: Parameterized tests for `LEAP_V2`, `DEX3`, `DEX1` verify topic allow/deny matrix.

**Acceptance Scenarios**:

1. **Given** `end_effector_type=LEAP_V2`, **When** mapping executes, **Then** Dex topics are blocked.
2. **Given** `end_effector_type=DEX3`, **When** mapping executes, **Then** Dex3 topics are allowed.

---

### User Story 3 - Bridge State and Fault Diagnostics (Priority: P3)

As an operator, I can receive `bridge_state_packet` readiness/fault details for fast
troubleshooting.

**Why this priority**: Required for field debugging when robot does not move.

**Independent Test**: Inject malformed frames and DDS errors; verify `is_ready=false`,
fault code, and recovery behavior.

**Acceptance Scenarios**:

1. **Given** schema mismatch, **When** frame arrives, **Then** bridge rejects with explicit fault code.
2. **Given** DDS transport outage, **When** publish is attempted, **Then** readiness flips false and fault is surfaced.

---

### Edge Cases

- Teleop frame has wrong joint vector length.
- Wrist pose exists but IK fails to converge.
- DDS transport reconnects after transient outage.
- Timestamp drifts backward across consecutive frames.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Bridge MUST parse and validate `teleop_frame_v1` against protocol schema.
- **FR-002**: Bridge MUST publish Unitree control commands to configured DDS topics.
- **FR-003**: Bridge MUST gate all motion output by `armed` and deadman validity.
- **FR-004**: Bridge MUST enforce `safety_state` precedence (`estop > freeze > limit > normal`).
- **FR-005**: Bridge MUST enforce `end_effector_type` topic whitelist and deny illegal mapping.
- **FR-006**: Bridge MUST expose readiness/fault state via `bridge_state_packet`.
- **FR-007**: Bridge MUST log reject reason codes for parser, gate, IK, and DDS errors.
- **FR-008**: Bridge MUST pin `teleop-protocol` version and fail startup on incompatibility.

### Key Entities *(include if feature involves data)*

- **TeleopFrameV1**: Input command frame with wrist pose, hand targets, safety, and endpoint type.
- **BridgeSafetyState**: Effective gate state computed from arm/disarm/deadman/safety_state.
- **DdsCommandPacket**: Unitree-ready command payload for `rt/arm_sdk` or `rt/lowcmd`.
- **BridgeStatePacket**: Readiness + fault_code + last_processed timestamp.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of invalid frames are rejected with structured reason code.
- **SC-002**: Motion command publish p95 latency <= 20ms under load
  (teleop input 50Hz, command payload <= 8KB, 1 active robot session).
- **SC-003**: 100% of endpoint type guard tests pass for allow/deny matrix.
- **SC-004**: Bridge state endpoint updates within 100ms on DDS fault transitions.
