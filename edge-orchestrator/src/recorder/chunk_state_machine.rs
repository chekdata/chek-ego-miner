use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ChunkKey {
    trip_id: String,
    session_id: String,
    chunk_index: u32,
}

impl ChunkKey {
    fn new(trip_id: &str, session_id: &str, chunk_index: u32) -> Self {
        Self {
            trip_id: trip_id.to_string(),
            session_id: session_id.to_string(),
            chunk_index,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkState {
    Received,
    Stored,
    Acked,
    Cleaned,
}

impl ChunkState {
    fn as_str(&self) -> &'static str {
        match self {
            ChunkState::Received => "received",
            ChunkState::Stored => "stored",
            ChunkState::Acked => "acked",
            ChunkState::Cleaned => "cleaned",
        }
    }
}

#[derive(Clone, Debug)]
struct ChunkEntry {
    state: ChunkState,
    file_types: HashSet<String>,
    received_at: Option<Instant>,
    stored_at: Option<Instant>,
    acked_at: Option<Instant>,
    cleaned_at: Option<Instant>,
}

impl Default for ChunkEntry {
    fn default() -> Self {
        Self {
            state: ChunkState::Received,
            file_types: HashSet::new(),
            received_at: None,
            stored_at: None,
            acked_at: None,
            cleaned_at: None,
        }
    }
}

#[derive(Default)]
pub struct ChunkStateMachine {
    inner: RwLock<HashMap<ChunkKey, ChunkEntry>>,
}

impl ChunkStateMachine {
    fn is_complete(file_types: &HashSet<String>, required: &[String]) -> bool {
        if required.is_empty() {
            return true;
        }
        required.iter().all(|t| file_types.contains(t))
    }

    /// 记录一次 `upload_chunk` 到达并成功落盘。
    ///
    /// 返回：若该 chunk **首次**满足“完整条件”，则返回 true（调用方应下发一次 `chunk_ack(status=stored)`）。
    pub fn note_file_stored(
        &self,
        trip_id: &str,
        session_id: &str,
        chunk_index: u32,
        file_type: Option<&str>,
        required_file_types: &[String],
    ) -> bool {
        let now = Instant::now();
        let k = ChunkKey::new(trip_id, session_id, chunk_index);
        let mut inner = self.inner.write().expect("chunk state lock poisoned");
        let entry = inner.entry(k).or_default();

        if entry.received_at.is_none() {
            entry.received_at = Some(now);
        }

        if let Some(t) = file_type.filter(|s| !s.trim().is_empty()) {
            entry.file_types.insert(t.to_string());
        }

        // 仅允许单向前进：Received -> Stored -> Acked -> Cleaned
        if entry.state == ChunkState::Received
            && Self::is_complete(&entry.file_types, required_file_types)
        {
            entry.state = ChunkState::Stored;
            entry.stored_at = Some(now);
            return true;
        }
        false
    }

    pub fn mark_acked(&self, trip_id: &str, session_id: &str, chunk_index: u32) {
        let now = Instant::now();
        let k = ChunkKey::new(trip_id, session_id, chunk_index);
        let mut inner = self.inner.write().expect("chunk state lock poisoned");
        let entry = inner.entry(k).or_default();
        match entry.state {
            ChunkState::Received => {
                // 非预期：先 ack 再 stored。保持 received，但记录时间，便于排障。
                entry.acked_at = Some(now);
            }
            ChunkState::Stored => {
                entry.state = ChunkState::Acked;
                entry.acked_at = Some(now);
            }
            ChunkState::Acked | ChunkState::Cleaned => {}
        }
    }

    pub fn mark_cleaned(&self, trip_id: &str, session_id: &str, chunk_index: u32) {
        let now = Instant::now();
        let k = ChunkKey::new(trip_id, session_id, chunk_index);
        let mut inner = self.inner.write().expect("chunk state lock poisoned");
        let entry = inner.entry(k).or_default();
        // 幂等：无论当前状态，cleaned 都是终态。
        entry.state = ChunkState::Cleaned;
        entry.cleaned_at = Some(now);
    }

    pub fn get_state(&self, trip_id: &str, session_id: &str, chunk_index: u32) -> String {
        let k = ChunkKey::new(trip_id, session_id, chunk_index);
        let inner = self.inner.read().expect("chunk state lock poisoned");
        inner
            .get(&k)
            .map(|e| e.state.as_str())
            .unwrap_or(ChunkState::Received.as_str())
            .to_string()
    }

    pub fn session_stats(
        &self,
        trip_id: &str,
        session_id: &str,
        acked_to_cleaned_sla: Duration,
    ) -> ChunkSessionStats {
        let inner = self.inner.read().expect("chunk state lock poisoned");

        let mut acked_total: u64 = 0;
        let mut cleaned_within_sla: u64 = 0;

        let mut min_idx: Option<u32> = None;
        let mut max_idx: Option<u32> = None;
        let mut stored_indices: HashSet<u32> = HashSet::new();

        for (k, e) in inner.iter() {
            if k.trip_id != trip_id || k.session_id != session_id {
                continue;
            }

            match e.state {
                ChunkState::Stored | ChunkState::Acked | ChunkState::Cleaned => {
                    stored_indices.insert(k.chunk_index);
                    min_idx = Some(
                        min_idx
                            .map(|v| v.min(k.chunk_index))
                            .unwrap_or(k.chunk_index),
                    );
                    max_idx = Some(
                        max_idx
                            .map(|v| v.max(k.chunk_index))
                            .unwrap_or(k.chunk_index),
                    );
                }
                ChunkState::Received => {}
            }

            if matches!(e.state, ChunkState::Acked | ChunkState::Cleaned) || e.acked_at.is_some() {
                acked_total += 1;
                if let (Some(acked_at), Some(cleaned_at)) = (e.acked_at, e.cleaned_at) {
                    if cleaned_at >= acked_at
                        && cleaned_at.duration_since(acked_at) <= acked_to_cleaned_sla
                    {
                        cleaned_within_sla += 1;
                    }
                }
            }
        }

        let stored_unique = stored_indices.len() as u64;
        let expected = match (min_idx, max_idx) {
            (Some(min), Some(max)) if max >= min => (max - min + 1) as u64,
            _ => 0,
        };

        ChunkSessionStats {
            acked_total,
            cleaned_within_sla,
            stored_unique,
            stored_expected: expected,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkSessionStats {
    pub acked_total: u64,
    pub cleaned_within_sla: u64,
    pub stored_unique: u64,
    pub stored_expected: u64,
}
