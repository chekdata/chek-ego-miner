# Feature Specification: Edge Orchestrator Bootstrap

**Feature Branch**: `001-bootstrap-edge-orchestrator`  
**Created**: 2026-03-04  
**Status**: Draft  
**Input**: User description: "Bootstrap edge orchestrator control and fusion streams with deadman gating"

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

### User Story 1 - Safe Motion Gate (Priority: P1)

As an operator, I can arm/disarm control and use deadman keepalive so the system only
emits motion commands when explicitly enabled.

**Why this priority**: This is the primary safety boundary and must exist before any
teleoperation can be accepted for field use.

**Independent Test**: Start only API and control stream; verify motion output is blocked
until `armed=true` and valid keepalive is present, and blocked again on timeout/disarm.

**Acceptance Scenarios**:

1. **Given** `disarmed`, **When** teleop frames arrive, **Then** no motion packet is emitted.
2. **Given** `armed` with keepalive at 20Hz, **When** teleop frames arrive, **Then** motion is emitted.
3. **Given** `armed`, **When** keepalive exceeds timeout (200ms default), **Then** state becomes `fault` and motion stops.

---

### User Story 2 - Dual Stream Isolation (Priority: P2)

As a runtime engineer, I can run `/stream/fusion` and `/stream/teleop` independently so
fusion throughput spikes do not block control delivery.

**Why this priority**: Control stability under load is mandatory for teleoperation.

**Independent Test**: Saturate fusion stream traffic while replaying teleop input; confirm
teleop jitter and drop metrics stay within thresholds.

**Acceptance Scenarios**:

1. **Given** high fusion load, **When** teleop stream is active, **Then** control stream remains responsive.
2. **Given** teleop stream disruption, **When** fusion stream remains active, **Then** fusion data path still serves clients.

---

### User Story 3 - Chunk and Observability Baseline (Priority: P3)

As an operations engineer, I can inspect chunk lifecycle and bridge health metrics so
faults are diagnosable during long-running sessions.

**Why this priority**: Without state visibility, failures are hard to recover in field deployment.

**Independent Test**: Run a simulated session and verify `received->stored->acked->cleaned`
transitions, deadman metrics, and bridge readiness endpoints.

**Acceptance Scenarios**:

1. **Given** normal upload flow, **When** chunk is persisted and acked, **Then** lifecycle state is queryable.
2. **Given** bridge unready, **When** control is armed, **Then** orchestrator refuses motion and logs reason code.

---

### Edge Cases

- Keepalive packet arrives late but out-of-order with teleop frame.
- Bridge reports ready=false after control already armed.
- Duplicate `chunk/cleaned` callbacks are sent.
- Fusion stream clients disconnect while control stream remains active.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose session control APIs: `start/stop/pause/resume`.
- **FR-002**: System MUST expose safety control APIs: `arm/disarm/estop/release`.
- **FR-003**: System MUST enforce motion gate: `armed=true` and valid deadman keepalive.
- **FR-004**: System MUST enter `fault` and stop motion on keepalive timeout.
- **FR-005**: System MUST expose `/stream/fusion` and `/stream/teleop` as separate WS channels.
- **FR-006**: System MUST track chunk lifecycle state machine and provide queryable status.
- **FR-007**: System MUST ingest bridge readiness/fault state and block unsafe motion output.
- **FR-008**: System MUST emit structured metrics/logs for deadman timeout, gate rejection, and stream lag.
- **FR-009**: System MUST pin and report active `teleop-protocol` version at runtime.
- **FR-010**: System MUST reject startup if protocol version is outside allowed compatibility range.

### Key Entities *(include if feature involves data)*

- **ControlState**: Current control mode (`disarmed|armed|fault`) plus last keepalive timestamp.
- **KeepalivePacket**: Deadman heartbeat containing operator/session identifiers and press state.
- **ChunkState**: Lifecycle record keyed by `trip_id + session_id + chunk_index`.
- **BridgeState**: Readiness and fault code from unitree/leap bridge.
- **ProtocolVersionInfo**: Loaded protocol version and compatibility window.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of keepalive timeout tests force `fault` and stop motion within 200ms.
- **SC-002**: Under fusion stress (1 fusion client at 200 msgs/s, 256KB per msg, concurrent teleop at 50Hz),
  `/stream/teleop` p95 dispatch latency remains <= 30ms.
- **SC-003**: Chunk lifecycle transitions are complete and queryable for >= 99.9% of chunks in soak tests.
- **SC-004**: Runtime status endpoint always reports protocol version and bridge readiness state.
