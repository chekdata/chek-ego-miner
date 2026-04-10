---

description: "Task list template for feature implementation"
---

# Tasks: Unitree Bridge Bootstrap

**Input**: Design documents from `/specs/001-bootstrap-unitree-bridge/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md (if present), quickstart.md (if present)

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure


## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create bridge module structure (`src/bridge`, `src/dds`, `src/state`)
- [x] T002 Initialize dependencies and feature flags in `Cargo.toml`
- [x] T003 [P] Configure lint/format and CI basics

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [x] T004 Implement `teleop_frame_v1` parser and schema checks in `src/bridge/parser.rs`
- [x] T005 [P] Implement safety gate logic in `src/bridge/gate.rs`
- [x] T006 [P] Implement endpoint-type validator in `src/bridge/validator.rs`
- [x] T007 Implement Unitree DDS publisher abstraction in `src/dds/unitree_client.rs`（trait + Mock）
- [x] T008 Configure structured fault codes and logs（`src/reason.rs` + 指标/日志）
- [x] T009 Setup runtime config loading and protocol version pinning（`src/config.rs` + `src/protocol/version_guard.rs` + `protocol_pin.json`）
- [x] T030 [P] Implement `safety_state` precedence evaluator (`estop > freeze > limit > normal`) in `src/bridge/gate.rs`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Safe Teleop to DDS Mapping (Priority: P1) 🎯 MVP

**Goal**: Map valid teleop frames into Unitree DDS commands with mandatory safety gate.

**Independent Test**: Replay fixture frames and verify publish/no-publish behavior under gate conditions.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T010 [P] [US1] Contract test parser/schema in `tests/contract/test_protocol_schema.rs`
- [x] T011 [P] [US1] Integration test gate behavior in `tests/integration/test_gate_behavior.rs`
- [x] T031 [P] [US1] Integration test `safety_state` precedence in `tests/integration/test_gate_behavior.rs`（estop/freeze/limit）

### Implementation for User Story 1

- [x] T012 [US1] Implement mapping from teleop frame to arm command in `src/bridge/mapper.rs`
- [x] T013 [US1] Implement DDS publish path in `src/bridge/publisher.rs`
- [x] T014 [US1] Integrate gate checks before publish
- [x] T015 [US1] Add error handling and reject reason codes
- [x] T016 [US1] Add operation logging and metrics

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Endpoint Type Guard (Priority: P2)

**Goal**: Enforce endpoint allow/deny matrix by `end_effector_type`.

**Independent Test**: LEAP mode blocks Dex topics; Dex modes allow only matching topic families.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [x] T017 [P] [US2] Integration test endpoint guard matrix in `tests/integration/test_endpoint_type_guard.rs`

### Implementation for User Story 2

- [x] T018 [US2] Implement endpoint guard logic in `src/bridge/validator.rs`
- [x] T019 [US2] Integrate guard with mapper/publisher pipeline
- [x] T020 [US2] Add deny-path logging and metrics（`endpoint_guard_denied_count`）

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Bridge State and Fault Diagnostics (Priority: P3)

**Goal**: Expose reliable bridge readiness and fault diagnostics.

**Independent Test**: Simulated parser/DDS faults update bridge state and recovery correctly.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [x] T021 [P] [US3] Integration test DDS fault behavior in `tests/integration/test_dds_publish.rs`

### Implementation for User Story 3

- [x] T022 [US3] Implement `bridge_state_packet` model in `src/state/bridge_state.rs`
- [x] T023 [US3] Emit readiness/fault transitions in bridge runtime（DDS publish fail -> unready；恢复后 ready）
- [x] T024 [US3] Add state reporting endpoint or stream hook（HTTP `/health` + WS 上行 `bridge_state_packet`）

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T025 [P] Documentation updates in `README.md` and `specs/.../quickstart.md`
- [x] T026 Code cleanup and refactoring
- [x] T027 Performance tune publish path（MVP：100Hz interval + 轻量映射）
- [x] T028 Security hardening (input validation and unsafe mode controls)（NaN/Inf + 维度校验 + 单调性丢弃）
- [x] T029 Run quickstart validation（fmt/clippy/test 全绿）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for [endpoint] in tests/contract/test_[name].py"
Task: "Integration test for [user journey] in tests/integration/test_[name].py"

# Launch all models for User Story 1 together:
Task: "Create [Entity1] model in src/models/[entity1].py"
Task: "Create [Entity2] model in src/models/[entity2].py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## 任务优先级与里程碑（周计划，可排期）

### 优先级分层
- **P0（阻塞）**: T001-T009, T030（解析、gate、validator、DDS抽象、版本钉住、优先级引擎）
- **P1（MVP）**: T010-T016, T031（安全映射与发布主链路 + 优先级验证）
- **P2（增强）**: T017-T020（`end_effector_type` 防错下发）
- **P3（可观测）**: T021-T024（bridge 状态与故障诊断）
- **P4（收口）**: T025-T029（文档、性能、安全、回归）

### 周计划（建议 6 周）
- **W1 基础搭建**: 完成 T001-T003，桥接服务最小编译链路。
- **W2 阻塞项收敛**: 完成 T004-T009, T030，解析与gate优先级链路可运行。
- **W3 P1 MVP**: 完成 T010-T016, T031，交付可控 DDS 发布闭环。
- **W4 P2 防错下发**: 完成 T017-T020，完成 LEAP/Dex allow-deny 矩阵验证。
- **W5 P3 诊断能力**: 完成 T021-T024，bridge readiness/fault 可对外输出。
- **W6 收口发布**: 完成 T025-T029，执行 quickstart 与回归压测。

### 每周验收出口（Exit Criteria）
- **W1 Exit**: 模块边界固定，CI可重复通过。
- **W2 Exit**: 非安全条件下发布严格被抑制，日志有拒绝原因。
- **W3 Exit**: 合法帧在 armed + keepalive 条件下稳定发布。
- **W4 Exit**: `end_effector_type` 错配场景 100% 拒绝。
- **W5 Exit**: 故障注入后状态迁移和恢复行为可验证。
- **W6 Exit**: 联调流程一次通过，版本可冻结。

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
