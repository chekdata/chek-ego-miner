pub mod routes_association;
pub mod routes_calibration;
pub mod routes_chunk;
pub mod routes_control;
pub mod routes_esp32_ota;
pub mod routes_evolution;
pub mod routes_health;
pub mod routes_ingest;
pub mod routes_network;
pub mod routes_safety;
pub mod routes_session;
pub mod routes_storage_compat;
pub mod routes_time;
pub mod routes_upload_chunk;
pub mod routes_upload_queue;
pub mod routes_workstation;

use axum::middleware::from_fn_with_state;
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::AppState;

fn protected_routes(state: AppState) -> Router {
    Router::new()
        .merge(routes_calibration::router(state.clone()))
        .merge(routes_association::router(state.clone()))
        .merge(routes_time::router(state.clone()))
        .merge(routes_session::router(state.clone()))
        .merge(routes_ingest::router(state.clone()))
        .merge(routes_network::router(state.clone()))
        .merge(routes_evolution::router(state.clone()))
        .merge(routes_safety::router(state.clone()))
        .merge(routes_control::router(state.clone()))
        .merge(routes_esp32_ota::router(state.clone()))
        .merge(routes_chunk::router(state.clone()))
        .merge(routes_upload_chunk::router(state.clone()))
        .merge(routes_upload_queue::router(state.clone()))
        .merge(routes_storage_compat::protected_router(state.clone()))
}

pub fn http_router(state: AppState) -> Router {
    let protected_router = protected_routes(state.clone()).route_layer(from_fn_with_state(
        state.clone(),
        crate::auth::require_http_auth,
    ));
    let edge_alias_router = Router::new()
        .merge(routes_health::router(state.clone()))
        .merge(routes_storage_compat::public_router(state.clone()))
        .merge(routes_workstation::router(state.clone()))
        .merge(
            protected_routes(state.clone()).route_layer(from_fn_with_state(
                state.clone(),
                crate::auth::require_http_auth,
            )),
        );

    Router::new()
        .merge(routes_health::router(state.clone()))
        .merge(routes_storage_compat::public_router(state.clone()))
        .merge(routes_workstation::router(state.clone()))
        .merge(protected_router)
        .nest("/edge", edge_alias_router)
        .layer(TraceLayer::new_for_http())
}
