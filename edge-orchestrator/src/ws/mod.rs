pub mod stream_fusion;
pub mod stream_teleop;
pub mod tasks;
pub mod transport;
pub mod types;

use axum::Router;

use crate::AppState;

pub fn ws_router(state: AppState) -> Router {
    Router::new()
        .merge(stream_fusion::router(state.clone()))
        .merge(stream_teleop::router(state.clone()))
}
