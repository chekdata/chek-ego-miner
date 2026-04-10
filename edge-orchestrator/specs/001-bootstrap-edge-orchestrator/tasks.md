---

description: "Task list template for feature implementation"
---

# Tasks: Edge Orchestrator Bootstrap

**Input**: Design documents from `/specs/001-bootstrap-edge-orchestrator/`
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

- [x] T001 Create module skeleton under `src/api`, `src/ws`, `src/control`, `src/recorder`, `src/protocol`
- [x] T002 Initialize Rust service dependencies in `Cargo.toml` (`axum`, `tokio`, `serde`, `tracing`, `metrics`)
- [x] T003 [P] Configure lint/format settings (`rustfmt.toml`, clippy settings)（已添加 `rustfmt.toml`/`clippy.toml`，并通过 `cargo fmt --check` + `cargo clippy -- -D warnings`）

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement base router and health endpoints in `src/api/routes_health.rs`
- [x] T005 [P] Implement control state model (`disarmed|armed|fault`) in `src/control/gate.rs`
- [x] T006 [P] Implement keepalive model/parser in `src/control/keepalive.rs`
- [x] T007 Implement chunk lifecycle entity in `src/recorder/chunk_state_machine.rs`（MVP：仅实现 `received/cleaned` 与查询，`stored/acked` 预留）
- [x] T008 Implement structured error/log conventions and reason codes（新增统一 reason 常量 `src/reason.rs`；ARM preflight 失败给出可诊断 reason；watchdog 日志结构化）
- [x] T009 Implement config loading for timeout/stream settings
- [x] T010 Add protocol pinning guard in `src/protocol/version_guard.rs`
- [x] T034 [P] Implement session APIs `start/stop/pause/resume` in `src/api/routes_session.rs`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Safe Motion Gate (Priority: P1) 🎯 MVP

**Goal**: Enforce deadman + arm gating for all motion output paths.

**Independent Test**: Teleop frames are ignored unless armed and keepalive valid; timeout/disarm stops motion.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**
- [x] T011 [P] [US1] Contract test for `POST /control/arm|disarm` in `tests/contract/test_api_contract.rs`
- [x] T012 [P] [US1] Integration test keepalive timeout in `tests/integration/test_deadman_timeout.rs`
- [x] T013 [P] [US1] Integration test disarm immediate stop in `tests/integration/test_disarm_gate.rs`
- [x] T035 [P] [US1] Contract test for `POST /safety/estop|release` in `tests/contract/test_api_contract.rs`

### Implementation for User Story 1

- [x] T014 [US1] Implement arm/disarm handlers in `src/api/routes_control.rs`
- [x] T015 [US1] Implement keepalive ingestion and timeout enforcement in `src/ws/stream_fusion.rs` + `src/control/tasks.rs` + `src/control/gate.rs`
- [x] T016 [US1] Enforce gate before teleop forward in `src/ws/tasks.rs`
- [x] T017 [US1] Emit gate rejection reason logs/metrics
- [x] T036 [US1] Implement estop/release handlers and precedence wiring in `src/api/routes_safety.rs` + `src/control/gate.rs`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Dual Stream Isolation (Priority: P2)

**Goal**: Keep `/stream/fusion` and `/stream/teleop` independent under load.

**Independent Test**: Fusion stress does not degrade teleop delivery p95 beyond threshold.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [x] T018 [P] [US2] Integration test stream isolation in `tests/integration/test_stream_isolation.rs`
- [x] T019 [P] [US2] Contract test for WS route availability in `tests/contract/test_api_contract.rs`

### Implementation for User Story 2

- [x] T020 [US2] Implement fusion stream server in `src/ws/stream_fusion.rs`
- [x] T021 [US2] Implement teleop stream server in `src/ws/stream_teleop.rs`
- [x] T022 [US2] Add independent queue/backpressure config per stream（独立 broadcast 通道与容量：`FUSION_BROADCAST_CAPACITY` / `TELEOP_BROADCAST_CAPACITY`）
- [x] T023 [US2] Add stream lag metrics per channel（`*_lagged_count` counters）

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Chunk and Observability Baseline (Priority: P3)

**Goal**: Track chunk lifecycle and bridge/protocol observability.

**Independent Test**: Session simulation exposes complete chunk states and readiness/fault diagnostics.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [x] T024 [P] [US3] Unit test chunk state transitions in `tests/unit/test_chunk_state_machine.rs`
- [x] T025 [P] [US3] Contract test protocol pinning guard in `tests/contract/test_protocol_guard.rs`

### Implementation for User Story 3

- [x] T026 [US3] Implement `POST /chunk/cleaned` and state transitions（MVP：cleaned 回执 + `/chunk/state` 查询）
- [x] T027 [US3] Implement bridge-state ingest and readiness block logic
- [x] T028 [US3] Expose health/metrics for protocol version and bridge readiness（`/health` 回传 protocol+bridge；Prometheus 增加 bridge/deadman/time_sync 等 gauges）

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T029 [P] Documentation updates in `README.md` and `specs/.../quickstart.md`（README/quickstart 已同步，包含鉴权与一键冒烟验证）
- [x] T030 Code cleanup and refactoring
- [x] T031 Performance tuning for teleop path under fusion load（周期任务改为 `tokio::time::interval`；增加 stream isolation 压力测试用例）
- [x] T032 Security hardening (auth token checks, unsafe route lock-down)（实现 PRD 2.2：`EDGE_TOKEN`，HTTP/WS 鉴权）
- [x] T033 Run quickstart validation end-to-end（`cargo test` + `scripts/verify_smoke.py` 已通过）

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
- **P0（阻塞）**: T001-T010, T034（基础骨架、Deadman、协议钉住、chunk状态机、session API）
- **P1（MVP）**: T011-T017, T035-T036（安全门控闭环 + estop/release）
- **P2（增强）**: T018-T023（双流隔离能力）
- **P3（可观测）**: T024-T028（chunk与bridge可观测）
- **P4（收口）**: T029-T033（文档、性能、安全、联调回归）

### 周计划（建议 6 周）
- **W1 基础搭建**: 完成 T001-T003，仓库可编译，CI绿灯。
- **W2 阻塞项收敛**: 完成 T004-T010, T034，Deadman + 协议版本守卫 + session API 具备运行能力。
- **W3 P1 MVP**: 完成 T011-T017, T035-T036，交付“未armed不动、超时即fault、estop优先”的最小闭环。
- **W4 P2 双流隔离**: 完成 T018-T023，完成融合压测下 teleop 延迟基线验证。
- **W5 P3 可观测**: 完成 T024-T028，具备 chunk 生命周期和 bridge 就绪/故障可见性。
- **W6 收口发布**: 完成 T029-T033，完成 quickstart 全链路回归并冻结版本。

### 每周验收出口（Exit Criteria）
- **W1 Exit**: `cargo check` 与 CI 通过，目录结构和依赖定型。
- **W2 Exit**: gate/keepalive/config/version_guard 全部可运行并可观测。
- **W3 Exit**: US1 测试通过，安全拒绝原因可追踪。
- **W4 Exit**: 双流压测报告达标（teleop p95 在阈值内）。
- **W5 Exit**: chunk 与 bridge 状态数据可对外查询/上报。
- **W6 Exit**: quickstart 演练一次通过，可进入联调发布窗口。

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
