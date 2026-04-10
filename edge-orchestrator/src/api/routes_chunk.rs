use axum::extract::State;
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ChunkCleanedRequest {
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub chunk_index: u32,
    pub device_id: String,
    pub source_time_ns: u64,
}

#[derive(Deserialize)]
pub struct ChunkKeyQuery {
    pub trip_id: String,
    pub session_id: String,
    pub chunk_index: u32,
}

#[derive(Serialize)]
pub struct ChunkStateResponse {
    pub trip_id: String,
    pub session_id: String,
    pub chunk_index: u32,
    pub state: String,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/chunk/cleaned", post(chunk_cleaned))
        .route("/chunk/state", get(get_state))
        .with_state(state)
}

async fn chunk_cleaned(
    State(state): State<AppState>,
    Json(req): Json<ChunkCleanedRequest>,
) -> Json<serde_json::Value> {
    let from_state = state
        .chunk_sm
        .get_state(&req.trip_id, &req.session_id, req.chunk_index);
    state
        .chunk_sm
        .mark_cleaned(&req.trip_id, &req.session_id, req.chunk_index);
    metrics::counter!("chunk_cleaned_count").increment(1);

    state
        .recorder
        .record_chunk_state_event(
            &state.protocol,
            &state.config,
            &req.trip_id,
            &req.session_id,
            req.chunk_index,
            &from_state,
            "cleaned",
            state.gate.edge_time_ns(),
            "",
            "",
        )
        .await;
    Json(serde_json::json!({"ok": true}))
}

async fn get_state(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<ChunkKeyQuery>,
) -> Json<ChunkStateResponse> {
    let s = state
        .chunk_sm
        .get_state(&q.trip_id, &q.session_id, q.chunk_index);
    Json(ChunkStateResponse {
        trip_id: q.trip_id,
        session_id: q.session_id,
        chunk_index: q.chunk_index,
        state: s,
    })
}
