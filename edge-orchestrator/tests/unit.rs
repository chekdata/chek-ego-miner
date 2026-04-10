// 合并运行 `tests/unit/*` 下的用例（Cargo 仅会把 `tests/*.rs` 视为集成测试入口）。

#[path = "unit/test_chunk_state_machine.rs"]
mod test_chunk_state_machine;
