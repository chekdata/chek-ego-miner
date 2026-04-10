use std::io::Write;

use axum::extract::ws::Message;
use base64::Engine;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::ser::Serializer;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransportMode {
    #[default]
    Full,
    Delta,
}

impl TransportMode {
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some(raw) if raw.eq_ignore_ascii_case("delta") => Self::Delta,
            _ => Self::Full,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Delta => "delta",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompressionMode {
    #[default]
    None,
    Gzip,
}

impl CompressionMode {
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            Some(raw) if raw.eq_ignore_ascii_case("gzip") => Self::Gzip,
            _ => Self::None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Gzip => "gzip",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TransportOptions {
    pub use_cbor: bool,
    pub mode: TransportMode,
    pub compression: CompressionMode,
}

impl TransportOptions {
    pub fn uses_envelope(self) -> bool {
        self.mode != TransportMode::Full || self.compression != CompressionMode::None
    }
}

pub struct TransportEncoder {
    options: TransportOptions,
    stream: &'static str,
    payload_schema_version: &'static str,
    allow_delta: bool,
    last_value: Option<serde_json::Value>,
    last_sequence: Option<u64>,
}

impl TransportEncoder {
    pub fn new(
        options: TransportOptions,
        stream: &'static str,
        payload_schema_version: &'static str,
        allow_delta: bool,
    ) -> Self {
        Self {
            options,
            stream,
            payload_schema_version,
            allow_delta,
            last_value: None,
            last_sequence: None,
        }
    }

    pub fn encode_packet<T: Serialize>(
        &mut self,
        sequence: u64,
        packet: &T,
    ) -> Result<Message, String> {
        if !self.options.uses_envelope() {
            return encode_raw_packet(self.options.use_cbor, packet);
        }

        let current_value =
            serde_json::to_value(packet).map_err(|e| format!("packet to_value 失败: {e}"))?;
        let mut encoding = TransportMode::Full;
        let mut base_sequence = None;
        let payload_value = if self.options.mode == TransportMode::Delta
            && self.allow_delta
            && self.last_value.is_some()
        {
            encoding = TransportMode::Delta;
            base_sequence = self.last_sequence;
            build_merge_patch(self.last_value.as_ref().unwrap(), &current_value)
        } else {
            current_value.clone()
        };

        let packet = if self.options.compression == CompressionMode::None {
            StreamTransportPacket {
                ty: "stream_transport_packet",
                schema_version: "1.0.0",
                stream: self.stream,
                payload_schema_version: self.payload_schema_version,
                sequence,
                encoding: encoding.as_str(),
                compression: self.options.compression.as_str(),
                payload_format: "json_merge_patch",
                base_sequence,
                payload_json: Some(payload_value),
                payload_bytes: None,
            }
        } else {
            let payload_bytes = gzip_json_value(&payload_value)?;
            StreamTransportPacket {
                ty: "stream_transport_packet",
                schema_version: "1.0.0",
                stream: self.stream,
                payload_schema_version: self.payload_schema_version,
                sequence,
                encoding: encoding.as_str(),
                compression: self.options.compression.as_str(),
                payload_format: "json_merge_patch+gzip",
                base_sequence,
                payload_json: None,
                payload_bytes: Some(PayloadBytes(payload_bytes)),
            }
        };

        self.last_value = Some(current_value);
        self.last_sequence = Some(sequence);

        if self.options.use_cbor {
            serde_cbor::to_vec(&packet)
                .map(Message::Binary)
                .map_err(|e| format!("stream_transport_packet 序列化失败（CBOR）: {e}"))
        } else {
            serde_json::to_string(&packet)
                .map(Message::Text)
                .map_err(|e| format!("stream_transport_packet 序列化失败（JSON）: {e}"))
        }
    }
}

pub fn encode_raw_packet<T: Serialize>(use_cbor: bool, packet: &T) -> Result<Message, String> {
    if use_cbor {
        serde_cbor::to_vec(packet)
            .map(Message::Binary)
            .map_err(|e| format!("packet 序列化失败（CBOR）: {e}"))
    } else {
        serde_json::to_string(packet)
            .map(Message::Text)
            .map_err(|e| format!("packet 序列化失败（JSON）: {e}"))
    }
}

#[derive(Clone, Debug)]
pub struct PayloadBytes(pub Vec<u8>);

impl Serialize for PayloadBytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            let encoded = base64::engine::general_purpose::STANDARD.encode(&self.0);
            serializer.serialize_str(&encoded)
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct StreamTransportPacket {
    #[serde(rename = "type")]
    ty: &'static str,
    schema_version: &'static str,
    stream: &'static str,
    payload_schema_version: &'static str,
    sequence: u64,
    encoding: &'static str,
    compression: &'static str,
    payload_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_sequence: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload_json: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload_bytes: Option<PayloadBytes>,
}

fn gzip_json_value(value: &serde_json::Value) -> Result<Vec<u8>, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| format!("payload JSON 序列化失败: {e}"))?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&bytes)
        .map_err(|e| format!("payload gzip 压缩失败: {e}"))?;
    encoder
        .finish()
        .map_err(|e| format!("payload gzip 收尾失败: {e}"))
}

fn build_merge_patch(
    previous: &serde_json::Value,
    current: &serde_json::Value,
) -> serde_json::Value {
    if previous == current {
        return serde_json::json!({});
    }

    match (previous, current) {
        (serde_json::Value::Object(previous), serde_json::Value::Object(current)) => {
            let mut patch = serde_json::Map::new();
            for (key, value) in current {
                match previous.get(key) {
                    Some(previous_value) => {
                        let nested = build_merge_patch(previous_value, value);
                        if !nested.is_null()
                            && !matches!(&nested, serde_json::Value::Object(map) if map.is_empty())
                        {
                            patch.insert(key.clone(), nested);
                        }
                    }
                    None => {
                        patch.insert(key.clone(), value.clone());
                    }
                }
            }
            for key in previous.keys() {
                if !current.contains_key(key) {
                    patch.insert(key.clone(), serde_json::Value::Null);
                }
            }
            serde_json::Value::Object(patch)
        }
        _ => current.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::build_merge_patch;

    #[test]
    fn merge_patch_keeps_only_changed_fields() {
        let previous = serde_json::json!({
            "quality": { "vision_conf": 0.9, "csi_conf": 0.8 },
            "fusion_seq": 1,
        });
        let current = serde_json::json!({
            "quality": { "vision_conf": 0.95, "csi_conf": 0.8 },
            "fusion_seq": 2,
        });
        let patch = build_merge_patch(&previous, &current);
        assert_eq!(
            patch,
            serde_json::json!({
                "quality": { "vision_conf": 0.95 },
                "fusion_seq": 2,
            })
        );
    }
}
