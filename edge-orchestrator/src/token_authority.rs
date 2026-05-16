use std::path::{Path, PathBuf};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ring::digest;
use ring::rand::{SecureRandom, SystemRandom};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::auth::UploadAuthContext;

#[derive(Clone, Debug)]
pub struct SqliteTokenAuthority {
    db_path: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairingEnvelope {
    #[serde(rename = "type")]
    pub ty: &'static str,
    pub version: &'static str,
    pub profile_id: String,
    pub edge_base_url: String,
    pub edge_ws_url: String,
    pub status_ui_url: String,
    pub pairing_code: String,
    pub pairing_challenge: String,
    pub expires_unix_ms: i64,
    pub operator_hint: String,
    pub scene_hint: String,
}

#[derive(Debug, Deserialize)]
pub struct PairingExchangeRequest {
    pub pairing_challenge: String,
    pub pairing_code: String,
    pub device_id: String,
    #[serde(default)]
    pub device_name: String,
    #[serde(default, alias = "operator_id")]
    pub login_identity: String,
}

#[derive(Debug, Deserialize)]
pub struct PairingRevokeRequest {
    #[serde(default)]
    pub device_id: String,
    #[serde(default)]
    pub token_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct PublicDeviceRecord {
    pub device_id: String,
    pub device_name: String,
    pub login_identity: String,
    pub profile_id: String,
    pub paired_unix_ms: i64,
    pub last_seen_unix_ms: i64,
    pub token_expires_unix_ms: i64,
    pub upload_token_status: String,
    pub session_id: Option<String>,
    pub upload_queue_depth: Option<i64>,
    pub ingest_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_ack: Option<serde_json::Value>,
}

#[derive(Clone, Debug)]
pub struct PairingPublicUrls {
    pub edge_base_url: String,
    pub edge_ws_url: String,
    pub status_ui_url: String,
}

#[derive(Clone, Debug)]
pub struct DeviceAckStatusUpdate {
    pub device_id: String,
    pub session_id: String,
    pub chunk_index: u32,
    pub edge_time_ns: u64,
    pub stored_size_bytes: u64,
    pub upload_queue_depth: i64,
    pub ingest_status: String,
}

impl SqliteTokenAuthority {
    pub async fn open(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        let authority = Self {
            db_path: db_path.into(),
        };
        authority.migrate().await?;
        Ok(authority)
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub async fn issue_pairing_envelope(
        &self,
        profile_id: String,
        urls: PairingPublicUrls,
        ttl_sec: u64,
        operator_hint: String,
        scene_hint: String,
    ) -> Result<PairingEnvelope, String> {
        let db_path = self.db_path.clone();
        run_db(db_path, move |conn| {
            let now = now_unix_ms();
            let expires = now.saturating_add((ttl_sec.max(30) as i64).saturating_mul(1000));
            let pairing_code = format!("{:06}", random_u32()? % 1_000_000);
            let pairing_challenge = random_token(24)?;
            conn.execute(
                "INSERT INTO pairing_challenges (
                    challenge, pairing_code_sha256, profile_id, created_unix_ms,
                    expires_unix_ms, status, operator_hint, scene_hint
                ) VALUES (?1, ?2, ?3, ?4, ?5, 'issued', ?6, ?7)",
                params![
                    pairing_challenge,
                    sha256_hex(&pairing_code),
                    profile_id,
                    now,
                    expires,
                    operator_hint,
                    scene_hint
                ],
            )
            .map_err(|error| format!("写入 pairing challenge 失败: {error}"))?;
            Ok(PairingEnvelope {
                ty: "chek_ego_edge_pairing",
                version: "1.0",
                profile_id,
                edge_base_url: urls.edge_base_url,
                edge_ws_url: urls.edge_ws_url,
                status_ui_url: urls.status_ui_url,
                pairing_code,
                pairing_challenge,
                expires_unix_ms: expires,
                operator_hint,
                scene_hint,
            })
        })
        .await
    }

    pub async fn exchange_pairing_challenge(
        &self,
        req: PairingExchangeRequest,
        token_ttl_sec: u64,
    ) -> Result<(PublicDeviceRecord, String, i64), String> {
        let db_path = self.db_path.clone();
        run_db(db_path, move |conn| {
            let challenge = non_empty(req.pairing_challenge, "pairing_challenge")?;
            let pairing_code = non_empty(req.pairing_code, "pairing_code")?;
            let device_id = non_empty(req.device_id, "device_id")?;
            let device_name = normalize(req.device_name).unwrap_or_else(|| device_id.clone());
            let login_identity =
                normalize(req.login_identity).unwrap_or_else(|| "unknown".to_string());
            let now = now_unix_ms();
            let issued = conn
                .query_row(
                    "SELECT pairing_code_sha256, profile_id, expires_unix_ms, status
                     FROM pairing_challenges WHERE challenge = ?1",
                    params![challenge],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, i64>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    },
                )
                .optional()
                .map_err(|error| format!("读取 pairing challenge 失败: {error}"))?
                .ok_or_else(|| "unknown_or_expired_pairing_challenge".to_string())?;
            let (expected_code_hash, profile_id, challenge_expires, challenge_status) = issued;
            if challenge_status != "issued" {
                return Err("pairing_challenge_already_used".to_string());
            }
            if challenge_expires <= now {
                conn.execute(
                    "UPDATE pairing_challenges SET status = 'expired' WHERE challenge = ?1",
                    params![challenge],
                )
                .ok();
                return Err("unknown_or_expired_pairing_challenge".to_string());
            }
            if !expected_code_hash.eq_ignore_ascii_case(&sha256_hex(&pairing_code)) {
                return Err("pairing_code_mismatch".to_string());
            }

            let upload_token = random_token(32)?;
            let token_hash = sha256_hex(&upload_token);
            let token_id = random_token(18)?;
            let token_expires =
                now.saturating_add((token_ttl_sec.max(300) as i64).saturating_mul(1000));
            conn.execute(
                "INSERT INTO paired_devices (
                    device_id, device_name, login_identity, profile_id,
                    paired_unix_ms, last_seen_unix_ms, status
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?5, 'paired')
                ON CONFLICT(device_id) DO UPDATE SET
                    device_name = excluded.device_name,
                    login_identity = excluded.login_identity,
                    profile_id = excluded.profile_id,
                    last_seen_unix_ms = excluded.last_seen_unix_ms,
                    status = 'paired'",
                params![device_id, device_name, login_identity, profile_id, now],
            )
            .map_err(|error| format!("写入 paired device 失败: {error}"))?;
            conn.execute(
                "INSERT INTO upload_tokens (
                    token_id, device_id, profile_id, token_sha256,
                    issued_unix_ms, expires_unix_ms, status
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active')",
                params![
                    token_id,
                    device_id,
                    profile_id,
                    token_hash,
                    now,
                    token_expires
                ],
            )
            .map_err(|error| format!("写入 upload token 失败: {error}"))?;
            conn.execute(
                "UPDATE pairing_challenges
                 SET status = 'exchanged', exchanged_unix_ms = ?2, device_id = ?3
                 WHERE challenge = ?1",
                params![challenge, now, device_id],
            )
            .map_err(|error| format!("更新 pairing challenge 失败: {error}"))?;
            insert_audit(
                conn,
                "token_issued",
                Some(&device_id),
                Some(&token_id),
                None,
                &format!(r#"{{"profile_id":"{}"}}"#, escape_json_string(&profile_id)),
            )?;
            Ok((
                PublicDeviceRecord {
                    device_id,
                    device_name,
                    login_identity,
                    profile_id,
                    paired_unix_ms: now,
                    last_seen_unix_ms: now,
                    token_expires_unix_ms: token_expires,
                    upload_token_status: "active".to_string(),
                    session_id: None,
                    upload_queue_depth: None,
                    ingest_status: None,
                    last_ack: None,
                },
                upload_token,
                token_expires,
            ))
        })
        .await
    }

    pub async fn record_device_ack_status(
        &self,
        update: DeviceAckStatusUpdate,
    ) -> Result<(), String> {
        let db_path = self.db_path.clone();
        run_db(db_path, move |conn| {
            let device_id = non_empty(update.device_id, "device_id")?;
            let session_id = non_empty(update.session_id, "session_id")?;
            let now = now_unix_ms();
            let last_ack = serde_json::json!({
                "session_id": session_id,
                "chunk_index": update.chunk_index,
                "edge_time_ns": update.edge_time_ns,
                "stored_size_bytes": update.stored_size_bytes,
            });
            let last_ack_json =
                serde_json::to_string(&last_ack).map_err(|error| error.to_string())?;
            conn.execute(
                "UPDATE upload_tokens
                 SET last_used_session_id = ?2, last_used_unix_ms = ?3
                 WHERE device_id = ?1
                   AND status IN ('active', 'issued_by_workstation_pairing_endpoint')",
                params![device_id, session_id, now],
            )
            .ok();
            let changed = conn
                .execute(
                    "UPDATE paired_devices
                     SET last_seen_unix_ms = ?2,
                         last_session_id = ?3,
                         upload_queue_depth = ?4,
                         ingest_status = ?5,
                         last_ack_json = ?6
                     WHERE device_id = ?1",
                    params![
                        device_id,
                        now,
                        session_id,
                        update.upload_queue_depth,
                        update.ingest_status,
                        last_ack_json
                    ],
                )
                .map_err(|error| format!("更新设备 ACK 状态失败: {error}"))?;
            if changed == 0 {
                return Err("paired_device_not_found".to_string());
            }
            insert_audit(
                conn,
                "device_ack_status",
                Some(&device_id),
                None,
                Some(&session_id),
                &serde_json::json!({
                    "chunk_index": update.chunk_index,
                    "edge_time_ns": update.edge_time_ns,
                    "stored_size_bytes": update.stored_size_bytes,
                    "upload_queue_depth": update.upload_queue_depth,
                    "ingest_status": update.ingest_status,
                })
                .to_string(),
            )?;
            Ok(())
        })
        .await
    }

    pub async fn validate_upload_token(
        &self,
        token: &str,
        device_id: Option<&str>,
    ) -> Result<UploadAuthContext, String> {
        let db_path = self.db_path.clone();
        let token_hash = sha256_hex(token);
        let expected_device_id = device_id.and_then(|value| normalize(value.to_string()));
        run_db(db_path, move |conn| {
            let row = conn
                .query_row(
                    "SELECT
                        t.token_id, t.device_id, t.profile_id, t.expires_unix_ms,
                        t.revoked_unix_ms, t.status,
                        COALESCE(d.device_name, ''),
                        COALESCE(d.login_identity, '')
                     FROM upload_tokens t
                     LEFT JOIN paired_devices d ON d.device_id = t.device_id
                     WHERE t.token_sha256 = ?1",
                    params![token_hash],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, i64>(3)?,
                            row.get::<_, Option<i64>>(4)?,
                            row.get::<_, String>(5)?,
                            row.get::<_, String>(6)?,
                            row.get::<_, String>(7)?,
                        ))
                    },
                )
                .optional()
                .map_err(|error| format!("读取 SQLite token authority 失败: {error}"))?
                .ok_or_else(|| "scoped_upload_token 不匹配".to_string())?;
            let (
                token_id,
                stored_device_id,
                profile_id,
                expires_unix_ms,
                revoked_unix_ms,
                status,
                device_name,
                login_identity,
            ) = row;
            if let Some(expected) = expected_device_id.as_deref() {
                if expected != stored_device_id {
                    insert_audit(conn, "token_device_mismatch", Some(&stored_device_id), Some(&token_id), None, "{}").ok();
                    return Err("scoped_upload_token 与 device_id 不匹配".to_string());
                }
            }
            if revoked_unix_ms.unwrap_or(0) > 0 || status.eq_ignore_ascii_case("revoked") {
                insert_audit(conn, "token_revoked_rejected", Some(&stored_device_id), Some(&token_id), None, "{}").ok();
                return Err("scoped_upload_token 已撤销".to_string());
            }
            let now = now_unix_ms();
            if expires_unix_ms <= now {
                conn.execute(
                    "UPDATE upload_tokens SET status = 'expired' WHERE token_id = ?1 AND status != 'revoked'",
                    params![token_id],
                )
                .ok();
                insert_audit(conn, "token_expired_rejected", Some(&stored_device_id), Some(&token_id), None, "{}").ok();
                return Err("scoped_upload_token 已过期".to_string());
            }
            if !(status.eq_ignore_ascii_case("active")
                || status.eq_ignore_ascii_case("issued_by_workstation_pairing_endpoint"))
            {
                insert_audit(conn, "token_inactive_rejected", Some(&stored_device_id), Some(&token_id), None, "{}").ok();
                return Err("scoped_upload_token 状态不可用".to_string());
            }
            conn.execute(
                "UPDATE upload_tokens SET last_used_unix_ms = ?2 WHERE token_id = ?1",
                params![token_id, now],
            )
            .ok();
            conn.execute(
                "UPDATE paired_devices SET last_seen_unix_ms = ?2 WHERE device_id = ?1",
                params![stored_device_id, now],
            )
            .ok();
            insert_audit(conn, "token_validated", Some(&stored_device_id), Some(&token_id), None, "{}").ok();
            Ok(UploadAuthContext {
                auth_kind: "scoped_upload_token".to_string(),
                device_id: Some(stored_device_id),
                device_name: normalize(device_name),
                login_identity: normalize(login_identity),
                profile_id: normalize(profile_id),
            })
        })
        .await
    }

    pub async fn revoke(&self, req: PairingRevokeRequest) -> Result<u64, String> {
        let db_path = self.db_path.clone();
        run_db(db_path, move |conn| {
            let now = now_unix_ms();
            let changed = if let Some(token_id) = normalize(req.token_id) {
                conn.execute(
                    "UPDATE upload_tokens
                     SET status = 'revoked', revoked_unix_ms = ?2
                     WHERE token_id = ?1 AND status != 'revoked'",
                    params![token_id, now],
                )
            } else if let Some(device_id) = normalize(req.device_id) {
                conn.execute(
                    "UPDATE upload_tokens
                     SET status = 'revoked', revoked_unix_ms = ?2
                     WHERE device_id = ?1 AND status != 'revoked'",
                    params![device_id, now],
                )
            } else {
                return Err("device_id 或 token_id 至少需要一个".to_string());
            }
            .map_err(|error| format!("撤销 token 失败: {error}"))?;
            Ok(changed as u64)
        })
        .await
    }

    pub async fn public_devices(
        &self,
        profile_id: Option<String>,
    ) -> Result<Vec<PublicDeviceRecord>, String> {
        let db_path = self.db_path.clone();
        run_db(db_path, move |conn| {
            let mut sql = String::from(
                "SELECT
                    d.device_id, d.device_name, d.login_identity, d.profile_id,
                    d.paired_unix_ms, d.last_seen_unix_ms,
                    COALESCE(MAX(t.expires_unix_ms), 0),
                    COALESCE(MAX(t.status), ''),
                    d.last_session_id, d.upload_queue_depth, d.ingest_status, d.last_ack_json
                 FROM paired_devices d
                 LEFT JOIN upload_tokens t ON t.device_id = d.device_id",
            );
            let mut devices = Vec::new();
            if let Some(profile_id) = profile_id.filter(|value| !value.trim().is_empty()) {
                sql.push_str(" WHERE d.profile_id = ?1");
                sql.push_str(
                    " GROUP BY d.device_id ORDER BY d.last_seen_unix_ms DESC, d.device_id ASC",
                );
                let mut stmt = conn
                    .prepare(&sql)
                    .map_err(|error| format!("准备 devices 查询失败: {error}"))?;
                let rows = stmt
                    .query_map(params![profile_id], public_device_from_row)
                    .map_err(|error| format!("查询 devices 失败: {error}"))?;
                for row in rows {
                    devices.push(row.map_err(|error| format!("读取 devices 失败: {error}"))?);
                }
            } else {
                sql.push_str(
                    " GROUP BY d.device_id ORDER BY d.last_seen_unix_ms DESC, d.device_id ASC",
                );
                let mut stmt = conn
                    .prepare(&sql)
                    .map_err(|error| format!("准备 devices 查询失败: {error}"))?;
                let rows = stmt
                    .query_map([], public_device_from_row)
                    .map_err(|error| format!("查询 devices 失败: {error}"))?;
                for row in rows {
                    devices.push(row.map_err(|error| format!("读取 devices 失败: {error}"))?);
                }
            }
            Ok(devices)
        })
        .await
    }

    async fn migrate(&self) -> Result<(), String> {
        let db_path = self.db_path.clone();
        run_db(db_path, |conn| {
            conn.execute_batch(
                "
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;
                PRAGMA busy_timeout = 5000;
                CREATE TABLE IF NOT EXISTS paired_devices (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL DEFAULT '',
                    login_identity TEXT NOT NULL DEFAULT '',
                    profile_id TEXT NOT NULL DEFAULT '',
                    paired_unix_ms INTEGER NOT NULL,
                    last_seen_unix_ms INTEGER NOT NULL,
                    last_session_id TEXT,
                    upload_queue_depth INTEGER,
                    ingest_status TEXT,
                    last_ack_json TEXT,
                    status TEXT NOT NULL DEFAULT 'paired'
                );
                CREATE TABLE IF NOT EXISTS pairing_challenges (
                    challenge TEXT PRIMARY KEY,
                    pairing_code_sha256 TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    created_unix_ms INTEGER NOT NULL,
                    expires_unix_ms INTEGER NOT NULL,
                    exchanged_unix_ms INTEGER,
                    device_id TEXT,
                    status TEXT NOT NULL,
                    operator_hint TEXT NOT NULL DEFAULT '',
                    scene_hint TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_pairing_challenges_status
                    ON pairing_challenges(status, expires_unix_ms);
                CREATE TABLE IF NOT EXISTS upload_tokens (
                    token_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    token_sha256 TEXT NOT NULL UNIQUE,
                    issued_unix_ms INTEGER NOT NULL,
                    expires_unix_ms INTEGER NOT NULL,
                    revoked_unix_ms INTEGER,
                    status TEXT NOT NULL,
                    last_used_unix_ms INTEGER,
                    last_used_session_id TEXT,
                    FOREIGN KEY(device_id) REFERENCES paired_devices(device_id)
                );
                CREATE INDEX IF NOT EXISTS idx_upload_tokens_device
                    ON upload_tokens(device_id, status, expires_unix_ms);
                CREATE INDEX IF NOT EXISTS idx_upload_tokens_expires
                    ON upload_tokens(expires_unix_ms);
                CREATE TABLE IF NOT EXISTS token_authority_audit (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    event_unix_ms INTEGER NOT NULL,
                    device_id TEXT,
                    token_id TEXT,
                    session_id TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_token_authority_audit_device
                    ON token_authority_audit(device_id, event_unix_ms);
                ",
            )
            .map_err(|error| format!("初始化 SQLite token authority 失败: {error}"))?;
            if !table_has_column(conn, "paired_devices", "last_ack_json")? {
                conn.execute(
                    "ALTER TABLE paired_devices ADD COLUMN last_ack_json TEXT",
                    [],
                )
                .map_err(|error| format!("升级 paired_devices.last_ack_json 失败: {error}"))?;
            }
            Ok(())
        })
        .await
    }
}

fn public_device_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<PublicDeviceRecord> {
    Ok(PublicDeviceRecord {
        device_id: row.get(0)?,
        device_name: row.get(1)?,
        login_identity: row.get(2)?,
        profile_id: row.get(3)?,
        paired_unix_ms: row.get(4)?,
        last_seen_unix_ms: row.get(5)?,
        token_expires_unix_ms: row.get(6)?,
        upload_token_status: row.get(7)?,
        session_id: row.get(8)?,
        upload_queue_depth: row.get(9)?,
        ingest_status: row.get(10)?,
        last_ack: row
            .get::<_, Option<String>>(11)?
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok()),
    })
}

async fn run_db<T, F>(db_path: PathBuf, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce(&Connection) -> Result<T, String> + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "创建 SQLite authority 目录失败: {} ({error})",
                    parent.display()
                )
            })?;
        }
        let conn = Connection::open(&db_path).map_err(|error| {
            format!(
                "打开 SQLite token authority 失败: {} ({error})",
                db_path.display()
            )
        })?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|error| format!("设置 SQLite busy timeout 失败: {error}"))?;
        f(&conn)
    })
    .await
    .map_err(|error| format!("SQLite token authority worker 失败: {error}"))?
}

fn insert_audit(
    conn: &Connection,
    event_type: &str,
    device_id: Option<&str>,
    token_id: Option<&str>,
    session_id: Option<&str>,
    payload_json: &str,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO token_authority_audit (
            event_id, event_type, event_unix_ms, device_id, token_id, session_id, payload_json
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            random_token(18)?,
            event_type,
            now_unix_ms(),
            device_id,
            token_id,
            session_id,
            payload_json
        ],
    )
    .map_err(|error| format!("写入 token authority audit 失败: {error}"))?;
    Ok(())
}

fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, String> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|error| format!("读取 SQLite schema 失败: {error}"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("查询 SQLite schema 失败: {error}"))?;
    for row in rows {
        let name = row.map_err(|error| format!("读取 SQLite schema column 失败: {error}"))?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn random_token(byte_len: usize) -> Result<String, String> {
    let rng = SystemRandom::new();
    let mut bytes = vec![0_u8; byte_len];
    rng.fill(&mut bytes)
        .map_err(|_| "生成随机 token 失败".to_string())?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

fn random_u32() -> Result<u32, String> {
    let rng = SystemRandom::new();
    let mut bytes = [0_u8; 4];
    rng.fill(&mut bytes)
        .map_err(|_| "生成随机 pairing code 失败".to_string())?;
    Ok(u32::from_be_bytes(bytes))
}

fn sha256_hex(value: &str) -> String {
    let digest = digest::digest(&digest::SHA256, value.as_bytes());
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{:02x}", *byte))
        .collect()
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn normalize(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn non_empty(value: String, field: &str) -> Result<String, String> {
    normalize(value).ok_or_else(|| format!("{field} 不能为空"))
}

fn escape_json_string(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
