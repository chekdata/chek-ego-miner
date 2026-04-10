---

description: "Task list template for feature implementation"
---

# Tasks: LEAP Bridge Bootstrap

**Input**: Design documents from `/specs/001-bootstrap-leap-bridge/`
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

- [x] T001 Create bridge and hardware modules (`src/bridge`, `src/leap`, `src/state`)
- [x] T002 Initialize Rust dependencies in `Cargo.toml`
- [x] T003 [P] Configure lint/format and CI basics

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [x] T004 Implement hand frame parser/validator in `src/bridge/parser.rs`
- [x] T005 [P] Implement retargeting core in `src/bridge/retarget.rs`
- [x] T006 [P] Implement pairing-window engine in `src/bridge/pairing.rs`
- [x] T007 Implement safety gate (`armed + deadman`) in `src/bridge/gate.rs`（MVP：armed + keepalive + estop 阻断；deadman 由 edge 折叠进 control_state，再做二次校验）
- [x] T008 Configure structured fault codes and logs（`src/reason.rs` + 指标/日志）
- [x] T009 Setup runtime config and protocol pinning（`src/config.rs` + `src/protocol/version_guard.rs` + `protocol_pin.json`）
- [x] T031 [P] Implement per-joint clamp enforcement in `src/bridge/retarget.rs`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Dual-Hand Retargeting with Safety Gate (Priority: P1) 🎯 MVP

**Goal**: Emit safe LEAP commands for both hands from valid paired teleop frames.

**Independent Test**: With valid gate and paired frames, both commands publish; with invalid gate, no publish.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T010 [P] [US1] Contract schema test in `tests/contract/test_protocol_schema.rs`
- [x] T011 [P] [US1] Integration keepalive timeout test in `tests/integration/test_keepalive_timeout.rs`

### Implementation for User Story 1

- [x] T012 [US1] Implement LEAP publish flow in `src/bridge/publisher.rs`（MVP：Mock 客户端；Edge WS 可选）
- [x] T013 [US1] Implement LEAP client wrapper in `src/leap/client.rs`（trait + Mock）
- [x] T014 [US1] Integrate gate + retarget + publish flow
- [x] T015 [US1] Add validation and reject reasons（解析/维度/单调性/keepalive 等）
- [x] T016 [US1] Add metrics/logging for publish and reject paths

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Pairing Window and Degradation Policy (Priority: P2)

**Goal**: Enforce <=20ms pairing and stale-hand degradation.

**Independent Test**: Delayed one-hand input triggers hold/limit/freeze policy correctly.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [x] T017 [P] [US2] Integration pairing-window test in `tests/integration/test_dual_hand_pairing.rs`

### Implementation for User Story 2

- [x] T018 [US2] Implement stale-hand policy in `src/bridge/pairing.rs`
- [x] T019 [US2] Add configuration for windows/thresholds in `config/leap.toml`
- [x] T020 [US2] Add degrade-state logs/metrics

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Hardware Health and Recovery (Priority: P3)

**Goal**: Observe hardware health and safely suppress output on device faults.

**Independent Test**: Simulated dropout/overtemp updates state and blocks output.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [x] T021 [P] [US3] Integration hardware dropout test in `tests/integration/test_hardware_dropout.rs`
- [x] T022 [P] [US3] Unit joint limit clamp test in `tests/unit/test_joint_limits.rs`

### Implementation for User Story 3

- [x] T023 [US3] Implement hardware state model in `src/state/hardware_state.rs`
- [x] T024 [US3] Implement recovery and readiness transitions（offline/overtemp/error -> not-ready；恢复后 ready）
- [x] T025 [US3] Expose bridge state packet generation（`src/bridge/publisher.rs` 上行 `bridge_state_packet`）

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T026 [P] Documentation updates in `README.md` and `specs/.../quickstart.md`
- [x] T027 Code cleanup and refactoring（main.rs handler/state 修正；模块拆分）
- [x] T028 Performance tune command loop（MVP：100Hz interval + 轻量解析/限幅；后续可替换 ringbuf/zero-copy）
- [x] T029 Security hardening (input sanity + protective defaults)（协议 pin + NaN/Inf + 单调性丢弃 + 维度校验 + 默认阻断）
- [x] T030 Run quickstart validation（fmt/clippy/test 全绿）

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
- **P0（阻塞）**: T001-T009, T031（解析、retarget、pairing、gate、版本钉住、关节限幅）
- **P1（MVP）**: T010-T016（双手输出主链路 + Deadman）
- **P2（增强）**: T017-T020（配对窗口与退化策略）
- **P3（可观测）**: T021-T025（硬件健康、恢复与状态上报）
- **P4（收口）**: T026-T030（文档、性能、安全、回归）

### 周计划（建议 6 周）
- **W1 基础搭建**: 完成 T001-T003，工程可编译、CI可运行。
- **W2 阻塞项收敛**: 完成 T004-T009, T031，双手输入解析、关节限幅与gate基础完成。
- **W3 P1 MVP**: 完成 T010-T016，交付“安全门控下双手可控输出”。
- **W4 P2 配对/退化**: 完成 T017-T020，<=20ms 配对约束与 stale-hand 策略落地。
- **W5 P3 硬件观测**: 完成 T021-T025，掉线/过温下安全抑制与恢复闭环。
- **W6 收口发布**: 完成 T026-T030，完成 quickstart 和现场联调回归。

### 每周验收出口（Exit Criteria）
- **W1 Exit**: 模块结构稳定，`cargo check` 和 CI 通过。
- **W2 Exit**: gate 与协议检查生效，基础日志/错误码可见。
- **W3 Exit**: 正常输入双手可发布，非安全状态严格不发布。
- **W4 Exit**: 配对超窗行为符合策略（hold/freeze）并可复现。
- **W5 Exit**: 硬件异常场景可检测、可抑制、可恢复。
- **W6 Exit**: quickstart 全步骤一次通过，可进入版本冻结。

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
