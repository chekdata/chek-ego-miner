use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ControlKeepalivePacket {
    #[serde(rename = "type")]
    pub ty: String,
    pub schema_version: String,
    pub trip_id: String,
    pub session_id: String,
    pub device_id: String,
    pub source_time_ns: u64,
    pub seq: u64,
    pub deadman_pressed: bool,
}
