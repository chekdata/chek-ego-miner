// 合并运行 `tests/contract/*` 下的用例（Cargo 仅会把 `tests/*.rs` 视为集成测试入口）。

#[path = "contract/test_api_contract.rs"]
mod test_api_contract;

#[path = "contract/test_protocol_guard.rs"]
mod test_protocol_guard;
