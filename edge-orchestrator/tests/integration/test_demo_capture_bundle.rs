use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::Path as AxumPath;
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use base64::Engine;
use futures_util::SinkExt;
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, RgbImage};
use reqwest::multipart;
use tokio_tungstenite::tungstenite::Message;

use crate::support;

#[tokio::test]
async fn recorder_should_emit_demo_bundle_calibration_and_time_sync_artifacts() -> anyhow::Result<()>
{
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-demo-bundle-001";
    let session_id = "sess-demo-bundle-001";
    let device_id = "iphone-demo-001";

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "source_kind": "iphone_capture",
            "clock_domain": "ios_uptime_ns",
            "clock_offset_ns": 0,
            "rtt_ns": 1_500_000,
            "sample_count": 8
        }))
        .send()
        .await?
        .error_for_status()?;

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "stereo-demo-001",
            "source_kind": "stereo_pair",
            "clock_domain": "stereo_uptime_ns",
            "clock_offset_ns": 0,
            "rtt_ns": 1_200_000,
            "sample_count": 6
        }))
        .send()
        .await?
        .error_for_status()?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;
    ws.send(Message::Text(
        capture_pose_packet(trip_id, session_id, device_id, 7, 123_456_789).to_string(),
    ))
    .await?;

    client
        .post(format!("{}/ingest/stereo_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&stereo_pose_packet(
            trip_id,
            session_id,
            223_456_789,
            100,
            101,
        ))
        .send()
        .await?
        .error_for_status()?;

    for attempt in 0..3u64 {
        ws.send(Message::Text(
            capture_pose_packet(
                trip_id,
                session_id,
                device_id,
                8 + attempt,
                123_456_989 + attempt * 100,
            )
            .to_string(),
        ))
        .await?;
        client
            .post(format!("{}/ingest/stereo_pose", server.http_base))
            .bearer_auth(&server.edge_token)
            .json(&stereo_pose_packet(
                trip_id,
                session_id,
                223_456_989 + attempt * 100,
                102 + attempt * 2,
                103 + attempt * 2,
            ))
            .send()
            .await?
            .error_for_status()?;
        tokio::time::sleep(Duration::from_millis(80)).await;
    }

    client
        .post(format!("{}/ingest/wifi_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&wifi_pose_packet(trip_id, session_id, 323_456_789))
        .send()
        .await?
        .error_for_status()?;

    let csi_sync_edge_time_ns = client
        .get(format!("{}/time", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?
        .get("edge_time_ns")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing edge_time_ns"))?;
    let csi_source_time_base_ns = 900_000_000u64;
    let csi_clock_offset_ns = i64::try_from(csi_sync_edge_time_ns).unwrap_or_default()
        - i64::try_from(csi_source_time_base_ns).unwrap_or_default();
    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "device_id": "csi-node-07",
            "source_kind": "wifi_csi_node",
            "clock_domain": "esp32_boot_ns",
            "clock_offset_ns": csi_clock_offset_ns,
            "rtt_ns": 900_000,
            "sample_count": 1
        }))
        .send()
        .await?
        .error_for_status()?;

    let csi_socket = std::net::UdpSocket::bind(("127.0.0.1", 0))?;
    csi_socket.connect(&server.csi_addr)?;
    for sequence in 1..=3u32 {
        csi_socket.send(&build_csi_packet(
            7,
            sequence,
            -44,
            -92,
            900_000_000 + u64::from(sequence) * 1_000_000,
        ))?;
    }

    let upload_meta = serde_json::json!({
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": device_id,
        "media_scope": "iphone",
        "media_track": "main",
        "source_kind": "iphone_capture",
        "clock_domain": "ios_uptime_ns",
        "chunk_index": 0,
        "file_type": "csv",
        "file_name": "chunk_0.csv",
        "source_start_time_ns": 120_000_000u64,
        "source_end_time_ns": 240_000_000u64,
        "frame_source_time_ns": [120_000_000u64, 180_000_000u64, 240_000_000u64],
        "frame_count": 3,
        "frame_rate_hz": 15.0,
        "frame_correspondence": [
            {
                "frame_index": 1,
                "matches": [
                    {
                        "device_id": "stereo-demo-001",
                        "media_scope": "stereo",
                        "media_track": "preview",
                        "source_kind": "stereo_pair",
                        "frame_index": 1,
                        "frame_source_time_ns": 380_000_000u64
                    }
                ]
            }
        ]
    });
    let form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(b"csv-demo".to_vec()).file_name("chunk_0.csv"),
        )
        .text("metadata", upload_meta.to_string());
    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&server.edge_token)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    let stereo_upload_meta = serde_json::json!({
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "stereo-demo-001",
        "media_scope": "stereo",
        "media_track": "preview",
        "source_kind": "stereo_pair",
        "clock_domain": "stereo_uptime_ns",
        "chunk_index": 2,
        "file_type": "video",
        "file_name": "stereo_chunk_2.mp4",
        "source_start_time_ns": 320_000_000u64,
        "source_end_time_ns": 440_000_000u64,
        "frame_source_time_ns": [320_000_000u64, 380_000_000u64, 440_000_000u64],
        "frame_count": 3,
        "frame_rate_hz": 15.0
    });
    let stereo_form = multipart::Form::new()
        .part(
            "file",
            multipart::Part::bytes(b"fake-mp4-demo".to_vec()).file_name("stereo_chunk_2.mp4"),
        )
        .text("metadata", stereo_upload_meta.to_string());
    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&server.edge_token)
        .multipart(stereo_form)
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server.data_dir.join("session").join(session_id);
    wait_for_file(base_dir.join("demo_capture_bundle.json")).await?;
    wait_for_file(base_dir.join("sync").join("time_sync_samples.jsonl")).await?;
    wait_for_path_exists(base_dir.join("sync").join("frame_correspondence.jsonl")).await?;
    wait_for_file(base_dir.join("calibration").join("iphone_capture.json")).await?;
    wait_for_file(base_dir.join("calibration").join("stereo_pair.json")).await?;
    wait_for_file(base_dir.join("calibration").join("wifi_pose.json")).await?;
    wait_for_file(
        base_dir
            .join("raw")
            .join("csi")
            .join("chunks")
            .join("index.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("raw")
            .join("iphone")
            .join("wide")
            .join("media_index.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("raw")
            .join("stereo")
            .join("media_index.jsonl"),
    )
    .await?;
    wait_for_file(base_dir.join("raw").join("wifi").join("pose3d.jsonl")).await?;
    wait_for_file(base_dir.join("fused").join("human_demo_pose.jsonl")).await?;
    wait_for_file_contains(
        base_dir.join("fused").join("human_demo_pose.jsonl"),
        &format!("\"session_id\":\"{session_id}\""),
    )
    .await?;
    wait_for_file(base_dir.join("teleop").join("teleop_frame.jsonl")).await?;
    wait_for_file_contains(
        base_dir.join("teleop").join("teleop_frame.jsonl"),
        &format!("\"session_id\":\"{session_id}\""),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("offline")
            .join("offline_manifest.json"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("offline")
            .join("iphone_pose_v2.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("offline")
            .join("stereo_pose_v2.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("offline")
            .join("wifi_pose_v2.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("offline")
            .join("fusion_state_v2.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("offline")
            .join("human_demo_pose_v2.jsonl"),
    )
    .await?;
    wait_for_file(base_dir.join("qa").join("local_quality_report.json")).await?;
    wait_for_file(base_dir.join("qa").join("upload_policy.json")).await?;
    wait_for_file(base_dir.join("upload").join("upload_manifest.json")).await?;
    wait_for_file(base_dir.join("upload").join("upload_queue.json")).await?;
    wait_for_quality_ready(base_dir.join("qa").join("local_quality_report.json")).await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_manifest.json"),
        "\"ready_for_upload\": true",
    )
    .await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_queue.json"),
        "\"ready_for_upload\": true",
    )
    .await?;
    wait_for_file_contains(
        base_dir.join("demo_capture_bundle.json"),
        "\"stereo.preview\"",
    )
    .await?;
    wait_for_file_contains(
        base_dir.join("sync").join("time_sync_samples.jsonl"),
        "\"device_id\":\"csi-node-07\"",
    )
    .await?;

    let bundle: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(base_dir.join("demo_capture_bundle.json")).await?)?;
    assert_eq!(
        bundle.get("type").and_then(|v| v.as_str()),
        Some("demo_capture_bundle")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("manifest"))
            .and_then(|v| v.as_str()),
        Some("manifest.json")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("time_sync_samples"))
            .and_then(|v| v.as_str()),
        Some("sync/time_sync_samples.jsonl")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("local_quality_report"))
            .and_then(|v| v.as_str()),
        Some("qa/local_quality_report.json")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("upload_policy"))
            .and_then(|v| v.as_str()),
        Some("qa/upload_policy.json")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("upload_manifest"))
            .and_then(|v| v.as_str()),
        Some("upload/upload_manifest.json")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("upload_queue"))
            .and_then(|v| v.as_str()),
        Some("upload/upload_queue.json")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("upload_receipts"))
            .and_then(|v| v.as_str()),
        Some("upload/upload_receipts.jsonl")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("human_demo_pose"))
            .and_then(|v| v.as_str()),
        Some("fused/human_demo_pose.jsonl")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("frame_correspondence"))
            .and_then(|v| v.as_str()),
        Some("sync/frame_correspondence.jsonl")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("csi_chunk_index"))
            .and_then(|v| v.as_str()),
        Some("raw/csi/chunks/index.jsonl")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("offline_manifest"))
            .and_then(|v| v.as_str()),
        Some("derived/offline/offline_manifest.json")
    );
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|v| v.get("wifi_pose"))
            .and_then(|v| v.as_str()),
        Some("raw/wifi/pose3d.jsonl")
    );
    let media_tracks = bundle
        .get("media_tracks")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(media_tracks.iter().any(|track| {
        track.get("id").and_then(|v| v.as_str()) == Some("iphone.main")
            && track.get("media_index").and_then(|v| v.as_str())
                == Some("raw/iphone/wide/media_index.jsonl")
    }));
    assert!(media_tracks.iter().any(|track| {
        track.get("id").and_then(|v| v.as_str()) == Some("stereo.preview")
            && track.get("media_index").and_then(|v| v.as_str())
                == Some("raw/stereo/preview/media_index.jsonl")
    }));
    let chunk_dirs = bundle
        .get("chunk_dirs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(chunk_dirs
        .iter()
        .any(|value| value.as_str() == Some("raw/iphone/wide/chunks")));
    assert!(chunk_dirs
        .iter()
        .any(|value| value.as_str() == Some("raw/stereo/preview/chunks")));
    assert!(chunk_dirs
        .iter()
        .any(|value| value.as_str() == Some("raw/csi/chunks")));
    let calibration_paths = bundle
        .get("calibration_snapshot_paths")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(calibration_paths
        .iter()
        .any(|v| v.as_str() == Some("calibration/edge_frames.json")));
    assert!(calibration_paths
        .iter()
        .any(|v| v.as_str() == Some("calibration/iphone_capture.json")));
    assert!(calibration_paths
        .iter()
        .any(|v| v.as_str() == Some("calibration/stereo_pair.json")));
    assert!(calibration_paths
        .iter()
        .any(|v| v.as_str() == Some("calibration/wifi_pose.json")));

    let time_sync =
        tokio::fs::read_to_string(base_dir.join("sync").join("time_sync_samples.jsonl")).await?;
    assert!(time_sync.contains("\"type\":\"time_sync_sample\""));
    assert!(time_sync.contains("\"source_kind\":\"iphone_capture\""));
    assert!(time_sync.contains("\"device_id\":\"csi-node-07\""));
    assert!(time_sync.contains("\"used_active_session\":true"));

    let frame_correspondence =
        tokio::fs::read_to_string(base_dir.join("sync").join("frame_correspondence.jsonl")).await?;
    assert!(frame_correspondence.contains("\"type\":\"frame_correspondence\""));
    assert!(frame_correspondence.contains("\"media_track\":\"main\""));
    assert!(frame_correspondence.contains("\"media_track\":\"preview\""));
    assert!(frame_correspondence.contains("\"frame_source_time_ns\":180000000"));
    assert!(frame_correspondence.contains("\"frame_edge_time_ns\":180000000"));
    assert!(frame_correspondence.contains("\"frame_source_time_ns\":380000000"));
    assert!(frame_correspondence.contains("\"frame_edge_time_ns\":380000000"));

    let iphone_calibration: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(base_dir.join("calibration").join("iphone_capture.json")).await?,
    )?;
    assert_eq!(
        iphone_calibration
            .get("sensor_kind")
            .and_then(|v| v.as_str()),
        Some("iphone_capture")
    );
    assert_eq!(
        iphone_calibration
            .get("intrinsics")
            .and_then(|v| v.get("reference_image_w"))
            .and_then(|v| v.as_u64()),
        Some(1280)
    );

    let stereo_calibration: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(base_dir.join("calibration").join("stereo_pair.json")).await?,
    )?;
    assert_eq!(
        stereo_calibration
            .get("sensor_kind")
            .and_then(|v| v.as_str()),
        Some("stereo_pair")
    );
    assert_eq!(
        stereo_calibration
            .get("left_intrinsics")
            .and_then(|v| v.get("reference_image_h"))
            .and_then(|v| v.as_u64()),
        Some(720)
    );

    let wifi_calibration: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(base_dir.join("calibration").join("wifi_pose.json")).await?,
    )?;
    assert_eq!(
        wifi_calibration.get("sensor_kind").and_then(|v| v.as_str()),
        Some("wifi_pose")
    );
    assert_eq!(
        wifi_calibration
            .get("operator_frame")
            .and_then(|v| v.as_str()),
        Some("operator_frame")
    );

    let media_index = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("iphone")
            .join("wide")
            .join("media_index.jsonl"),
    )
    .await?;
    assert!(media_index.contains("\"type\":\"media_chunk_index\""));
    assert!(media_index.contains("raw/iphone/wide/chunks/000000/csv__chunk_0.csv"));
    assert!(media_index.contains("\"media_scope\":\"iphone\""));
    assert!(media_index.contains("\"media_track\":\"main\""));
    assert!(media_index.contains("\"source_kind\":\"iphone_capture\""));
    assert!(media_index.contains("\"clock_domain\":\"ios_uptime_ns\""));
    assert!(media_index.contains("\"media_alignment_kind\":\"frame_level_indexed\""));
    assert!(media_index.contains("\"time_sync_status\":\"frame_indexed\""));
    assert!(media_index.contains("\"frame_source_time_ns\":[120000000,180000000,240000000]"));
    assert!(media_index.contains("\"frame_edge_time_ns\":[120000000,180000000,240000000]"));

    let manifest: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(base_dir.join("manifest.json")).await?)?;
    assert_eq!(
        manifest.get("type").and_then(|v| v.as_str()),
        Some("session_manifest")
    );
    assert_eq!(
        manifest.get("schema_version").and_then(|v| v.as_str()),
        Some("2.0.0")
    );
    assert!(manifest
        .get("artifacts")
        .and_then(|v| v.as_array())
        .is_some_and(|items| !items.is_empty()));
    assert_eq!(
        manifest
            .get("recorder_state")
            .and_then(|v| v.get("has_iphone_calibration"))
            .and_then(|v| v.as_bool()),
        Some(true)
    );

    let local_quality: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(base_dir.join("qa").join("local_quality_report.json")).await?,
    )?;
    assert_eq!(
        local_quality.get("type").and_then(|v| v.as_str()),
        Some("local_quality_report")
    );
    assert!(matches!(
        local_quality.get("status").and_then(|v| v.as_str()),
        Some("pass") | Some("retry_recommended")
    ));
    assert!(local_quality
        .get("checks")
        .and_then(|v| v.as_array())
        .is_some_and(|items| !items.is_empty()));

    let upload_policy: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(base_dir.join("qa").join("upload_policy.json")).await?,
    )?;
    assert_eq!(
        upload_policy.get("type").and_then(|v| v.as_str()),
        Some("upload_policy")
    );
    assert_eq!(
        upload_policy
            .get("upload_policy")
            .and_then(|v| v.get("mode"))
            .and_then(|v| v.as_str()),
        Some("edge_crowd_upload_v1")
    );

    let upload_manifest: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(base_dir.join("upload").join("upload_manifest.json")).await?,
    )?;
    assert_eq!(
        upload_manifest.get("type").and_then(|v| v.as_str()),
        Some("upload_manifest")
    );
    assert_eq!(
        upload_manifest
            .get("ready_for_upload")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    assert!(upload_manifest
        .get("artifacts")
        .and_then(|v| v.as_array())
        .is_some_and(|items| !items.is_empty()));

    let initial_upload_queue: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(base_dir.join("upload").join("upload_queue.json")).await?,
    )?;
    assert_eq!(
        initial_upload_queue.get("type").and_then(|v| v.as_str()),
        Some("upload_queue")
    );
    assert_eq!(
        initial_upload_queue
            .get("ready_for_upload")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    assert!(initial_upload_queue
        .get("entries")
        .and_then(|v| v.as_array())
        .is_some_and(|items| !items.is_empty()));

    client
        .post(format!("{}/upload/receipt", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "asset_id": "iphone_media_index",
            "status": "uploading",
            "receipt_source": "integration_test",
            "remote_upload_id": "upload-job-1"
        }))
        .send()
        .await?
        .error_for_status()?;

    client
        .post(format!("{}/upload/receipt", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "trip_id": trip_id,
            "session_id": session_id,
            "asset_id": "iphone_media_index",
            "status": "acked",
            "receipt_source": "integration_test",
            "remote_upload_id": "upload-job-1",
            "remote_object_key": "crowd-data/sessions/sess-demo-bundle-001/raw/iphone/wide/media_index.jsonl"
        }))
        .send()
        .await?
        .error_for_status()?;

    wait_for_file(base_dir.join("upload").join("upload_receipts.jsonl")).await?;

    let queue_via_api = client
        .get(format!(
            "{}/upload/queue?session_id={session_id}",
            server.http_base
        ))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;
    let iphone_media_queue_entry = queue_via_api
        .get("entries")
        .and_then(|v| v.as_array())
        .and_then(|entries| {
            entries.iter().find(|entry| {
                entry.get("asset_id").and_then(|v| v.as_str()) == Some("iphone_media_index")
            })
        })
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing iphone_media_index upload queue entry"))?;
    assert_eq!(
        iphone_media_queue_entry
            .get("status")
            .and_then(|v| v.as_str()),
        Some("acked")
    );
    assert_eq!(
        iphone_media_queue_entry
            .get("remote_upload_id")
            .and_then(|v| v.as_str()),
        Some("upload-job-1")
    );
    assert_eq!(
        iphone_media_queue_entry
            .get("remote_object_key")
            .and_then(|v| v.as_str()),
        Some("crowd-data/sessions/sess-demo-bundle-001/raw/iphone/wide/media_index.jsonl")
    );

    let upload_receipts =
        tokio::fs::read_to_string(base_dir.join("upload").join("upload_receipts.jsonl")).await?;
    assert!(upload_receipts.contains("\"type\":\"upload_receipt\""));
    assert!(upload_receipts.contains("\"asset_id\":\"iphone_media_index\""));
    assert!(upload_receipts.contains("\"status\":\"uploading\""));
    assert!(upload_receipts.contains("\"status\":\"acked\""));

    let aggregate_media_index = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("iphone")
            .join("wide")
            .join("media_index.jsonl"),
    )
    .await?;
    assert!(aggregate_media_index.contains("raw/iphone/wide/chunks/000000/csv__chunk_0.csv"));

    let stereo_media_index = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("stereo")
            .join("preview")
            .join("media_index.jsonl"),
    )
    .await?;
    assert!(stereo_media_index.contains("\"type\":\"stereo_media_chunk_index\""));
    assert!(
        stereo_media_index.contains("raw/stereo/preview/chunks/000002/video__stereo_chunk_2.mp4")
    );
    assert!(stereo_media_index.contains("\"media_scope\":\"stereo\""));
    assert!(stereo_media_index.contains("\"media_track\":\"preview\""));
    assert!(stereo_media_index.contains("\"source_kind\":\"stereo_pair\""));
    assert!(stereo_media_index.contains("\"clock_domain\":\"stereo_uptime_ns\""));
    assert!(stereo_media_index.contains("\"media_alignment_kind\":\"frame_level_indexed\""));
    assert!(stereo_media_index.contains("\"time_sync_status\":\"frame_indexed\""));
    assert!(stereo_media_index.contains("\"frame_edge_time_ns\":[320000000,380000000,440000000]"));

    let aggregate_stereo_media_index = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("stereo")
            .join("media_index.jsonl"),
    )
    .await?;
    assert!(aggregate_stereo_media_index
        .contains("raw/stereo/preview/chunks/000002/video__stereo_chunk_2.mp4"));

    let stereo_pose =
        tokio::fs::read_to_string(base_dir.join("raw").join("stereo").join("pose3d.jsonl")).await?;
    assert!(stereo_pose.contains("\"body_layout\":\"pico_body_24\""));
    assert!(stereo_pose.contains("\"hand_layout\":\"pico_hand_26\""));

    let wifi_pose =
        tokio::fs::read_to_string(base_dir.join("raw").join("wifi").join("pose3d.jsonl")).await?;
    assert!(wifi_pose.contains("\"type\":\"wifi_pose_packet\""));
    assert!(wifi_pose.contains("\"body_layout\":\"pico_body_24\""));
    assert!(wifi_pose.contains("\"source_label\":\"wifi_densepose\""));

    let csi_chunk_index = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("csi")
            .join("chunks")
            .join("index.jsonl"),
    )
    .await?;
    assert!(csi_chunk_index.contains("\"type\":\"csi_chunk_index\""));
    assert!(csi_chunk_index.contains("\"packet_count\":3"));
    assert!(csi_chunk_index.contains("raw/csi/chunks/000000/csi__chunk_000000.bin"));
    assert!(csi_chunk_index.contains("\"node_ids\":[7]"));
    assert!(csi_chunk_index.contains("\"sequence_start\":1"));
    assert!(csi_chunk_index.contains("\"sequence_end\":3"));
    assert!(csi_chunk_index.contains("\"clock_domain\":\"esp32_boot_ns\""));
    assert!(csi_chunk_index.contains("\"time_sync_status\":\"mapped_to_edge_time\""));
    assert!(csi_chunk_index.contains("\"segment_start_source_time_ns\":901000000"));
    assert!(csi_chunk_index.contains("\"segment_end_source_time_ns\":903000000"));

    let csi_index =
        tokio::fs::read_to_string(base_dir.join("raw").join("csi").join("index.jsonl")).await?;
    assert!(csi_index.contains("\"device_id\":\"csi-node-07\""));
    assert!(csi_index.contains("\"clock_domain\":\"esp32_boot_ns\""));
    assert!(csi_index.contains("\"time_sync_status\":\"mapped_to_edge_time\""));

    let human_demo_pose =
        tokio::fs::read_to_string(base_dir.join("fused").join("human_demo_pose.jsonl")).await?;
    let representative_demo_pose: serde_json::Value = serde_json::from_str(
        human_demo_pose
            .lines()
            .filter(|line| !line.trim().is_empty())
            .next()
            .ok_or_else(|| anyhow::anyhow!("human_demo_pose.jsonl 为空"))?,
    )?;
    assert_eq!(
        representative_demo_pose
            .get("canonical_pose")
            .and_then(|v| v.get("body_layout"))
            .and_then(|v| v.as_str()),
        Some("coco_body_17")
    );
    assert!(representative_demo_pose
        .get("canonical_pose")
        .and_then(|v| v.get("left_hand_joints"))
        .and_then(|v| v.as_array())
        .map(Vec::len)
        .is_none_or(|len| len == 16));
    assert!(representative_demo_pose
        .get("fusion_debug")
        .and_then(|v| v.get("body_source"))
        .and_then(|v| v.as_str())
        .is_some());
    assert!(representative_demo_pose
        .get("fusion_debug")
        .and_then(|v| v.get("stereo_body_joint_count"))
        .and_then(|v| v.as_u64())
        .is_some());

    let offline_manifest: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(
            base_dir
                .join("derived")
                .join("offline")
                .join("offline_manifest.json"),
        )
        .await?,
    )?;
    assert_eq!(
        offline_manifest.get("type").and_then(|v| v.as_str()),
        Some("offline_manifest")
    );
    assert_eq!(
        offline_manifest
            .get("pipeline")
            .and_then(|v| v.get("generation_mode"))
            .and_then(|v| v.as_str()),
        Some("bootstrap_live_mirror")
    );
    let offline_artifacts = offline_manifest
        .get("artifacts")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(offline_artifacts.iter().any(|artifact| {
        artifact.get("id").and_then(|v| v.as_str()) == Some("iphone_pose_v2")
            && artifact.get("relpath").and_then(|v| v.as_str())
                == Some("derived/offline/iphone_pose_v2.jsonl")
    }));
    assert!(offline_artifacts.iter().any(|artifact| {
        artifact.get("id").and_then(|v| v.as_str()) == Some("human_demo_pose_v2")
            && artifact.get("relpath").and_then(|v| v.as_str())
                == Some("derived/offline/human_demo_pose_v2.jsonl")
    }));

    let offline_iphone_pose = tokio::fs::read_to_string(
        base_dir
            .join("derived")
            .join("offline")
            .join("iphone_pose_v2.jsonl"),
    )
    .await?;
    assert!(offline_iphone_pose.contains("\"type\":\"capture_pose_packet\""));

    let offline_stereo_pose = tokio::fs::read_to_string(
        base_dir
            .join("derived")
            .join("offline")
            .join("stereo_pose_v2.jsonl"),
    )
    .await?;
    assert!(offline_stereo_pose.contains("\"type\":\"stereo_pose_packet\""));

    let offline_wifi_pose = tokio::fs::read_to_string(
        base_dir
            .join("derived")
            .join("offline")
            .join("wifi_pose_v2.jsonl"),
    )
    .await?;
    assert!(offline_wifi_pose.contains("\"type\":\"wifi_pose_packet\""));

    let offline_fusion_state = tokio::fs::read_to_string(
        base_dir
            .join("derived")
            .join("offline")
            .join("fusion_state_v2.jsonl"),
    )
    .await?;
    assert!(offline_fusion_state.contains("\"type\":\"fusion_state_packet\""));

    let offline_human_demo_pose = tokio::fs::read_to_string(
        base_dir
            .join("derived")
            .join("offline")
            .join("human_demo_pose_v2.jsonl"),
    )
    .await?;
    assert!(offline_human_demo_pose.contains("\"type\":\"human_demo_pose_packet\""));

    Ok(())
}

#[tokio::test]
async fn recorder_should_persist_pose_imu_depth_and_robot_state_artifacts() -> anyhow::Result<()> {
    let server = support::TestServer::spawn_with_env(vec![(
        "EDGE_PHONE_VISION_PROCESSING_ENABLED".to_string(),
        "0".to_string(),
    )])
    .await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-raw-artifacts-001";
    let session_id = "sess-raw-artifacts-001";
    let device_id = "iphone-raw-001";

    client
        .post(format!("{}/ingest/phone_vision_frame", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&phone_vision_frame_packet(
            trip_id,
            session_id,
            device_id,
            33,
            555_000_123,
        ))
        .send()
        .await?
        .error_for_status()?;

    let teleop_url = format!(
        "{}/stream/teleop?token={}",
        server.ws_base, server.edge_token
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(teleop_url).await?;
    ws.send(Message::Text(
        bridge_state_packet(trip_id, session_id).to_string(),
    ))
    .await?;

    let base_dir = server.data_dir.join("session").join(session_id);
    let depth_relpath = format!(
        "raw/iphone/wide/depth/frame_{:010}__{:020}.f32le",
        33, 555_000_123u64
    );
    wait_for_file(
        base_dir
            .join("raw")
            .join("iphone")
            .join("wide")
            .join("pose_imu.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("raw")
            .join("iphone")
            .join("wide")
            .join("depth")
            .join("index.jsonl"),
    )
    .await?;
    wait_for_file(base_dir.join(&depth_relpath)).await?;
    wait_for_file(base_dir.join("raw").join("robot").join("state.jsonl")).await?;
    wait_for_file(base_dir.join("calibration").join("iphone_capture.json")).await?;
    wait_for_file(base_dir.join("manifest.json")).await?;

    let pose_imu = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("iphone")
            .join("wide")
            .join("pose_imu.jsonl"),
    )
    .await?;
    assert!(pose_imu.contains("\"type\":\"phone_vision_input\""));
    assert!(pose_imu.contains("\"device_pose\""));
    assert!(pose_imu.contains("\"imu\""));

    let depth_index = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("iphone")
            .join("wide")
            .join("depth")
            .join("index.jsonl"),
    )
    .await?;
    assert!(depth_index.contains("\"type\":\"iphone_depth_frame\""));
    assert!(depth_index.contains(&depth_relpath));

    let raw_depth = tokio::fs::read(base_dir.join(&depth_relpath)).await?;
    assert_eq!(raw_depth.len(), 4 * 4);

    let iphone_calibration =
        tokio::fs::read_to_string(base_dir.join("calibration").join("iphone_capture.json")).await?;
    assert!(iphone_calibration.contains("\"sensor_kind\": \"iphone_capture\""));
    assert!(iphone_calibration.contains("\"intrinsics\""));

    let robot_state =
        tokio::fs::read_to_string(base_dir.join("raw").join("robot").join("state.jsonl")).await?;
    assert!(robot_state.contains("\"type\":\"robot_state_packet\""));
    assert!(robot_state.contains("\"control_state\":\"armed\""));
    let robot_state_event: serde_json::Value = serde_json::from_str(
        robot_state
            .lines()
            .filter(|line| !line.trim().is_empty())
            .last()
            .ok_or_else(|| anyhow::anyhow!("robot state 为空"))?,
    )?;
    let arm_q_commanded = robot_state_event
        .get("arm_q_commanded")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(arm_q_commanded.len(), 4);
    let approx_eq = |lhs: f64, rhs: f64| (lhs - rhs).abs() < 1e-4;
    assert!(approx_eq(
        arm_q_commanded
            .first()
            .and_then(|value| value.as_f64())
            .unwrap_or_default(),
        0.1
    ));
    assert!(approx_eq(
        arm_q_commanded
            .get(3)
            .and_then(|value| value.as_f64())
            .unwrap_or_default(),
        0.4
    ));

    let manifest = serde_json::from_slice::<serde_json::Value>(
        &tokio::fs::read(base_dir.join("manifest.json")).await?,
    )?;
    let artifacts = manifest
        .get("artifacts")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(artifacts
        .iter()
        .any(|item| item.get("id").and_then(|value| value.as_str()) == Some("pose_imu")));
    assert!(artifacts
        .iter()
        .any(|item| item.get("id").and_then(|value| value.as_str()) == Some("iphone_depth_index")));
    assert!(artifacts
        .iter()
        .any(|item| item.get("id").and_then(|value| value.as_str()) == Some("robot_state")));

    Ok(())
}

#[tokio::test]
async fn upload_manifest_should_filter_raw_artifacts_for_metadata_only_policy() -> anyhow::Result<()>
{
    let server = support::TestServer::spawn_with_env(vec![
        (
            "EDGE_PHONE_VISION_PROCESSING_ENABLED".to_string(),
            "0".to_string(),
        ),
        (
            "EDGE_RUNTIME_PROFILE".to_string(),
            "capture_plus_vlm".to_string(),
        ),
        (
            "EDGE_CROWD_UPLOAD_POLICY_MODE".to_string(),
            "metadata_only".to_string(),
        ),
    ])
    .await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-metadata-only-001";
    let session_id = "sess-metadata-only-001";
    let device_id = "iphone-metadata-only-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
        }))
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server.data_dir.join("session").join(session_id);
    wait_for_file(base_dir.join("upload").join("upload_manifest.json")).await?;

    let upload_manifest = serde_json::from_slice::<serde_json::Value>(
        &tokio::fs::read(base_dir.join("upload").join("upload_manifest.json")).await?,
    )?;
    assert_eq!(
        upload_manifest
            .get("upload_policy")
            .and_then(|value| value.get("artifact_policy_mode"))
            .and_then(|value| value.as_str()),
        Some("metadata_only")
    );

    let artifacts = upload_manifest
        .get("artifacts")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let artifact_ids = artifacts
        .iter()
        .filter_map(|item| item.get("id").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(artifact_ids.contains(&"vlm_events"));
    assert!(!artifact_ids.contains(&"iphone_depth_frames"));
    assert!(!artifact_ids.contains(&"preview_manifest"));

    let session_context = upload_manifest
        .get("session_context")
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        session_context
            .get("runtime_profile")
            .and_then(|value| value.as_str()),
        Some("capture_plus_vlm")
    );
    assert_eq!(
        session_context
            .get("raw_residency")
            .and_then(|value| value.as_str()),
        Some("edge_only")
    );

    Ok(())
}

#[tokio::test]
async fn capture_plus_vlm_should_emit_preview_keyframes_segments_and_gif_clips(
) -> anyhow::Result<()> {
    let server = support::TestServer::spawn_with_env(vec![
        (
            "EDGE_PHONE_VISION_PROCESSING_ENABLED".to_string(),
            "0".to_string(),
        ),
        (
            "EDGE_RUNTIME_PROFILE".to_string(),
            "capture_plus_vlm".to_string(),
        ),
        (
            "EDGE_CROWD_UPLOAD_POLICY_MODE".to_string(),
            "metadata_plus_preview".to_string(),
        ),
        ("EDGE_VLM_KEYFRAME_INTERVAL_MS".to_string(), "5".to_string()),
        ("EDGE_VLM_SEGMENT_WINDOW_MS".to_string(), "5".to_string()),
    ])
    .await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-preview-vlm-001";
    let session_id = "sess-preview-vlm-001";
    let device_id = "iphone-preview-vlm-001";

    client
        .post(format!("{}/ingest/phone_vision_frame", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&phone_vision_frame_packet(
            trip_id,
            session_id,
            device_id,
            41,
            777_000_001,
        ))
        .send()
        .await?
        .error_for_status()?;
    tokio::time::sleep(Duration::from_millis(15)).await;
    client
        .post(format!("{}/ingest/phone_vision_frame", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&phone_vision_frame_packet(
            trip_id,
            session_id,
            device_id,
            42,
            777_100_001,
        ))
        .send()
        .await?
        .error_for_status()?;

    client
        .post(format!("{}/session/stop", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
        }))
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server.data_dir.join("session").join(session_id);
    wait_for_file(base_dir.join("preview").join("preview_manifest.json")).await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("vision")
            .join("vlm_events.jsonl"),
    )
    .await?;
    wait_for_file(
        base_dir
            .join("derived")
            .join("vision")
            .join("vlm_segments.jsonl"),
    )
    .await?;
    wait_for_path_exists(base_dir.join("preview").join("keyframes")).await?;
    wait_for_path_exists(base_dir.join("preview").join("clips")).await?;

    let preview_manifest = serde_json::from_slice::<serde_json::Value>(
        &tokio::fs::read(base_dir.join("preview").join("preview_manifest.json")).await?,
    )?;
    assert_eq!(
        preview_manifest
            .get("status")
            .and_then(|value| value.as_str()),
        Some("ready")
    );
    assert_eq!(
        preview_manifest
            .get("vlm_status")
            .and_then(|value| value.as_str()),
        Some("ready")
    );
    let keyframes = preview_manifest
        .get("keyframes")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(!keyframes.is_empty());
    let clips = preview_manifest
        .get("clips")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(!clips.is_empty());

    let keyframe_relpath = keyframes[0]
        .get("relpath")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing keyframe relpath"))?;
    wait_for_file(base_dir.join(keyframe_relpath)).await?;

    let clip_relpath = clips[0]
        .get("relpath")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing clip relpath"))?;
    wait_for_file(base_dir.join(clip_relpath)).await?;

    let vlm_events = tokio::fs::read_to_string(
        base_dir
            .join("derived")
            .join("vision")
            .join("vlm_events.jsonl"),
    )
    .await?;
    assert!(vlm_events.contains("\"type\":\"vlm_event\""));
    assert!(vlm_events.contains("\"caption\""));
    assert!(vlm_events.contains("\"action_guess\""));

    let vlm_segments = tokio::fs::read_to_string(
        base_dir
            .join("derived")
            .join("vision")
            .join("vlm_segments.jsonl"),
    )
    .await?;
    assert!(vlm_segments.contains("\"type\":\"vlm_segment\""));
    assert!(vlm_segments.contains("\"keyframe_ids\""));

    wait_for_path_exists(base_dir.join("derived").join("vision").join("embeddings")).await?;

    Ok(())
}

#[tokio::test]
async fn upload_chunk_should_persist_fisheye_media_and_calibration() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-fisheye-001";
    let session_id = "sess-fisheye-001";
    let device_id = "iphone-fisheye-001";

    let upload_meta = serde_json::json!({
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": device_id,
        "media_scope": "iphone",
        "media_track": "fisheye",
        "source_kind": "iphone_capture",
        "clock_domain": "ios_uptime_ns",
        "chunk_index": 4,
        "file_type": "video",
        "file_name": "iphone_fisheye_chunk_4.mp4",
        "source_start_time_ns": 420_000_000u64,
        "source_end_time_ns": 560_000_000u64,
        "frame_source_time_ns": [420_000_000u64, 490_000_000u64, 560_000_000u64],
        "frame_count": 3,
        "frame_rate_hz": 15.0,
        "camera_calibration": {
            "intrinsics": [402.0, 0.0, 320.0, 0.0, 401.0, 240.0, 0.0, 0.0, 1.0],
            "distortion": [0.01, -0.03, 0.002, 0.0],
            "image_size": [1280, 720]
        }
    });

    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&server.edge_token)
        .multipart(
            multipart::Form::new()
                .part(
                    "file",
                    multipart::Part::bytes(b"fake-fisheye-mp4".to_vec())
                        .file_name("iphone_fisheye_chunk_4.mp4"),
                )
                .text("metadata", upload_meta.to_string()),
        )
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server.data_dir.join("session").join(session_id);
    wait_for_file(
        base_dir
            .join("raw")
            .join("iphone")
            .join("fisheye")
            .join("media_index.jsonl"),
    )
    .await?;
    wait_for_file(base_dir.join("calibration").join("iphone_fisheye.json")).await?;
    wait_for_file(base_dir.join("demo_capture_bundle.json")).await?;

    let fisheye_index = tokio::fs::read_to_string(
        base_dir
            .join("raw")
            .join("iphone")
            .join("fisheye")
            .join("media_index.jsonl"),
    )
    .await?;
    assert!(fisheye_index.contains("\"media_track\":\"fisheye\""));
    assert!(fisheye_index.contains("iphone_fisheye_chunk_4.mp4"));

    let fisheye_calibration =
        tokio::fs::read_to_string(base_dir.join("calibration").join("iphone_fisheye.json")).await?;
    assert!(fisheye_calibration.contains("\"sensor\": \"iphone_fisheye\""));
    assert!(fisheye_calibration.contains("\"camera_calibration\""));

    let bundle = serde_json::from_str::<serde_json::Value>(
        &tokio::fs::read_to_string(base_dir.join("demo_capture_bundle.json")).await?,
    )?;
    assert_eq!(
        bundle
            .get("artifacts")
            .and_then(|value| value.get("iphone_fisheye_media_index"))
            .and_then(|value| value.as_str()),
        Some("raw/iphone/fisheye/media_index.jsonl")
    );
    assert!(bundle
        .get("calibration_snapshot_paths")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .any(|item| item.as_str() == Some("calibration/iphone_fisheye.json"))
        })
        .unwrap_or(false));

    Ok(())
}

#[derive(Debug, Clone)]
struct UploadCall {
    asset_id: String,
    session_id: String,
    relpath: String,
    body_len: usize,
}

#[derive(Debug, Clone)]
struct ControlPlaneCall {
    path: String,
    body: serde_json::Value,
}

#[derive(Debug, Clone)]
struct TransportUploadCall {
    asset_id: String,
    session_id: String,
    relpath: String,
    scope_token: String,
    body_len: usize,
}

#[tokio::test]
async fn crowd_upload_worker_should_ack_session_assets_via_remote_endpoint() -> anyhow::Result<()> {
    let upload_calls = Arc::new(tokio::sync::Mutex::new(Vec::<UploadCall>::new()));
    let expected_upload_token = "crowd-upload-test-token";
    let upload_app = Router::new().route(
        "/edge/upload/artifact",
        post({
            let upload_calls = upload_calls.clone();
            move |headers: HeaderMap, body: Bytes| {
                let upload_calls = upload_calls.clone();
                async move {
                    let auth = headers
                        .get(AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default();
                    if auth != format!("Bearer {expected_upload_token}") {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({
                                "ok": false,
                                "error": { "message": "invalid bearer token" }
                            })),
                        );
                    }

                    let asset_id = headers
                        .get("x-chek-asset-id")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    let session_id = headers
                        .get("x-chek-session-id")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    let relpath = headers
                        .get("x-chek-relpath")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    upload_calls.lock().await.push(UploadCall {
                        asset_id: asset_id.clone(),
                        session_id: session_id.clone(),
                        relpath: relpath.clone(),
                        body_len: body.len(),
                    });

                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "ok": true,
                            "remote_object_key": format!("crowd-data/sessions/{session_id}/{relpath}"),
                            "remote_upload_id": format!("upload-job-{asset_id}"),
                        })),
                    )
                }
            }
        }),
    );
    let upload_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let upload_addr = upload_listener.local_addr()?;
    let upload_server = tokio::spawn(async move {
        let _ = axum::serve(upload_listener, upload_app).await;
    });

    let server = support::TestServer::spawn_with_env(vec![
        ("EDGE_CROWD_UPLOAD_ENABLED".to_string(), "true".to_string()),
        ("EDGE_CROWD_UPLOAD_POLL_MS".to_string(), "100".to_string()),
        (
            "EDGE_CROWD_UPLOAD_ARTIFACT_URL".to_string(),
            format!(
                "http://127.0.0.1:{}/edge/upload/artifact",
                upload_addr.port()
            ),
        ),
        (
            "EDGE_CROWD_UPLOAD_TOKEN".to_string(),
            expected_upload_token.to_string(),
        ),
    ])
    .await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-demo-upload-worker-001";
    let session_id = "sess-demo-upload-worker-001";
    let device_id = "iphone-demo-upload-worker-001";

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "source_kind": "iphone_capture",
            "clock_domain": "ios_uptime_ns",
            "clock_offset_ns": 0,
            "rtt_ns": 1_500_000,
            "sample_count": 8
        }))
        .send()
        .await?
        .error_for_status()?;

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "stereo-demo-upload-worker-001",
            "source_kind": "stereo_pair",
            "clock_domain": "stereo_uptime_ns",
            "clock_offset_ns": 0,
            "rtt_ns": 1_200_000,
            "sample_count": 6
        }))
        .send()
        .await?
        .error_for_status()?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;
    ws.send(Message::Text(
        capture_pose_packet(trip_id, session_id, device_id, 7, 123_456_789).to_string(),
    ))
    .await?;

    client
        .post(format!("{}/ingest/stereo_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&stereo_pose_packet(
            trip_id,
            session_id,
            223_456_789,
            100,
            101,
        ))
        .send()
        .await?
        .error_for_status()?;

    for attempt in 0..3u64 {
        ws.send(Message::Text(
            capture_pose_packet(
                trip_id,
                session_id,
                device_id,
                8 + attempt,
                123_456_989 + attempt * 100,
            )
            .to_string(),
        ))
        .await?;
        client
            .post(format!("{}/ingest/stereo_pose", server.http_base))
            .bearer_auth(&server.edge_token)
            .json(&stereo_pose_packet(
                trip_id,
                session_id,
                223_456_989 + attempt * 100,
                102 + attempt * 2,
                103 + attempt * 2,
            ))
            .send()
            .await?
            .error_for_status()?;
        tokio::time::sleep(Duration::from_millis(80)).await;
    }

    client
        .post(format!("{}/ingest/wifi_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&wifi_pose_packet(trip_id, session_id, 323_456_789))
        .send()
        .await?
        .error_for_status()?;

    let csi_sync_edge_time_ns = client
        .get(format!("{}/time", server.http_base))
        .bearer_auth(&server.edge_token)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?
        .get("edge_time_ns")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing edge_time_ns"))?;
    let csi_source_time_base_ns = 910_000_000u64;
    let csi_clock_offset_ns = i64::try_from(csi_sync_edge_time_ns).unwrap_or_default()
        - i64::try_from(csi_source_time_base_ns).unwrap_or_default();
    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "device_id": "csi-node-upload-worker-001",
            "source_kind": "wifi_csi_node",
            "clock_domain": "esp32_boot_ns",
            "clock_offset_ns": csi_clock_offset_ns,
            "rtt_ns": 900_000,
            "sample_count": 1
        }))
        .send()
        .await?
        .error_for_status()?;

    let csi_socket = std::net::UdpSocket::bind(("127.0.0.1", 0))?;
    csi_socket.connect(&server.csi_addr)?;
    for sequence in 1..=3u32 {
        csi_socket.send(&build_csi_packet(
            7,
            sequence,
            -44,
            -92,
            910_000_000 + u64::from(sequence) * 1_000_000,
        ))?;
    }

    let upload_meta = serde_json::json!({
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": device_id,
        "media_scope": "iphone",
        "media_track": "main",
        "source_kind": "iphone_capture",
        "clock_domain": "ios_uptime_ns",
        "chunk_index": 0,
        "file_type": "csv",
        "file_name": "chunk_0.csv",
        "source_start_time_ns": 120_000_000u64,
        "source_end_time_ns": 240_000_000u64,
        "frame_source_time_ns": [120_000_000u64, 180_000_000u64, 240_000_000u64],
        "frame_count": 3,
        "frame_rate_hz": 15.0,
        "frame_correspondence": [
            {
                "frame_index": 1,
                "matches": [
                    {
                        "device_id": "stereo-demo-upload-worker-001",
                        "media_scope": "stereo",
                        "media_track": "preview",
                        "source_kind": "stereo_pair",
                        "frame_index": 1,
                        "frame_source_time_ns": 380_000_000u64
                    }
                ]
            }
        ]
    });
    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&server.edge_token)
        .multipart(
            multipart::Form::new()
                .part(
                    "file",
                    multipart::Part::bytes(b"csv-demo-upload-worker".to_vec())
                        .file_name("chunk_0.csv"),
                )
                .text("metadata", upload_meta.to_string()),
        )
        .send()
        .await?
        .error_for_status()?;

    let stereo_upload_meta = serde_json::json!({
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "stereo-demo-upload-worker-001",
        "media_scope": "stereo",
        "media_track": "preview",
        "source_kind": "stereo_pair",
        "clock_domain": "stereo_uptime_ns",
        "chunk_index": 2,
        "file_type": "video",
        "file_name": "stereo_chunk_2.mp4",
        "source_start_time_ns": 320_000_000u64,
        "source_end_time_ns": 440_000_000u64,
        "frame_source_time_ns": [320_000_000u64, 380_000_000u64, 440_000_000u64],
        "frame_count": 3,
        "frame_rate_hz": 15.0
    });
    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&server.edge_token)
        .multipart(
            multipart::Form::new()
                .part(
                    "file",
                    multipart::Part::bytes(b"fake-mp4-upload-worker".to_vec())
                        .file_name("stereo_chunk_2.mp4"),
                )
                .text("metadata", stereo_upload_meta.to_string()),
        )
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server.data_dir.join("session").join(session_id);
    wait_for_file(base_dir.join("fused").join("human_demo_pose.jsonl")).await?;
    wait_for_file_contains(
        base_dir.join("fused").join("human_demo_pose.jsonl"),
        &format!("\"session_id\":\"{session_id}\""),
    )
    .await?;
    wait_for_file(base_dir.join("teleop").join("teleop_frame.jsonl")).await?;
    wait_for_file_contains(
        base_dir.join("teleop").join("teleop_frame.jsonl"),
        &format!("\"session_id\":\"{session_id}\""),
    )
    .await?;
    wait_for_file(base_dir.join("qa").join("local_quality_report.json")).await?;
    wait_for_file(base_dir.join("upload").join("upload_manifest.json")).await?;
    wait_for_file(base_dir.join("upload").join("upload_queue.json")).await?;
    wait_for_quality_ready(base_dir.join("qa").join("local_quality_report.json")).await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_manifest.json"),
        "\"ready_for_upload\": true",
    )
    .await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_queue.json"),
        "\"ready_for_upload\": true",
    )
    .await?;
    wait_for_file(base_dir.join("upload").join("upload_receipts.jsonl")).await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_receipts.jsonl"),
        "\"receipt_source\":\"edge_uploader_worker\"",
    )
    .await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_receipts.jsonl"),
        "\"status\":\"acked\"",
    )
    .await?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut iphone_media_status = None;
    while tokio::time::Instant::now() < deadline {
        let queue = client
            .get(format!(
                "{}/upload/queue?session_id={session_id}",
                server.http_base
            ))
            .bearer_auth(&server.edge_token)
            .send()
            .await?
            .error_for_status()?
            .json::<serde_json::Value>()
            .await?;
        iphone_media_status = queue
            .get("entries")
            .and_then(|value| value.as_array())
            .and_then(|entries| {
                entries.iter().find(|entry| {
                    entry.get("asset_id").and_then(|value| value.as_str())
                        == Some("iphone_media_index")
                })
            })
            .and_then(|entry| entry.get("status").and_then(|value| value.as_str()))
            .map(str::to_string);
        if iphone_media_status.as_deref() == Some("acked") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(iphone_media_status.as_deref(), Some("acked"));

    let calls = upload_calls.lock().await.clone();
    assert!(!calls.is_empty());
    assert!(calls.iter().any(|call| {
        call.session_id == session_id
            && !call.asset_id.is_empty()
            && !call.relpath.is_empty()
            && call.body_len > 0
    }));

    upload_server.abort();
    Ok(())
}

#[tokio::test]
async fn crowd_upload_worker_should_register_session_artifacts_and_receipts_via_control_plane(
) -> anyhow::Result<()> {
    let control_calls = Arc::new(tokio::sync::Mutex::new(Vec::<ControlPlaneCall>::new()));
    let transport_calls = Arc::new(tokio::sync::Mutex::new(Vec::<TransportUploadCall>::new()));
    let expected_upload_token = "crowd-control-test-token";
    let expected_scope_token = "crowd-upload-scope-token";

    let transport_app = Router::new().route(
        "/artifacts",
        post({
            let transport_calls = transport_calls.clone();
            move |headers: HeaderMap, body: Bytes| {
                let transport_calls = transport_calls.clone();
                async move {
                    let auth = headers
                        .get(AUTHORIZATION)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default();
                    if auth != format!("Bearer {expected_upload_token}") {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({
                                "ok": false,
                                "error": { "message": "invalid bearer token" }
                            })),
                        );
                    }
                    let scope_token = headers
                        .get("x-chek-upload-scope-token")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    if scope_token != expected_scope_token {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(serde_json::json!({
                                "ok": false,
                                "error": { "message": "invalid scope token" }
                            })),
                        );
                    }
                    let asset_id = headers
                        .get("x-chek-asset-id")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    let session_id = headers
                        .get("x-chek-session-id")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    let relpath = headers
                        .get("x-chek-relpath")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string();
                    transport_calls.lock().await.push(TransportUploadCall {
                        asset_id: asset_id.clone(),
                        session_id: session_id.clone(),
                        relpath: relpath.clone(),
                        scope_token,
                        body_len: body.len(),
                    });
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "ok": true,
                            "remote_object_key": format!("crowd-data/sessions/{session_id}/{relpath}"),
                            "remote_upload_id": format!("transport-job-{asset_id}"),
                        })),
                    )
                }
            }
        }),
    );
    let transport_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let transport_addr = transport_listener.local_addr()?;
    let transport_server = tokio::spawn(async move {
        let _ = axum::serve(transport_listener, transport_app).await;
    });
    let transport_url = format!("http://127.0.0.1:{}/artifacts", transport_addr.port());

    let control_app = Router::new()
        .route(
            "/v1/edge/sessions/upsert",
            post({
                let control_calls = control_calls.clone();
                move |headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                    let control_calls = control_calls.clone();
                    async move {
                        let auth = headers
                            .get(AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or_default();
                        if auth != format!("Bearer {expected_upload_token}") {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(serde_json::json!({
                                    "ok": false,
                                    "error": { "message": "invalid bearer token" }
                                })),
                            );
                        }
                        control_calls.lock().await.push(ControlPlaneCall {
                            path: "/v1/edge/sessions/upsert".to_string(),
                            body: body.clone(),
                        });
                        (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "session": { "session_id": body.get("session_id").cloned().unwrap_or_default() }
                            })),
                        )
                    }
                }
            }),
        )
        .route(
            "/v1/edge/sessions/:session_id/artifacts",
            post({
                let control_calls = control_calls.clone();
                let transport_url = transport_url.clone();
                move |AxumPath(session_id): AxumPath<String>, headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                    let control_calls = control_calls.clone();
                    let transport_url = transport_url.clone();
                    async move {
                        let auth = headers
                            .get(AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or_default();
                        if auth != format!("Bearer {expected_upload_token}") {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(serde_json::json!({
                                    "ok": false,
                                    "error": { "message": "invalid bearer token" }
                                })),
                            );
                        }
                        control_calls.lock().await.push(ControlPlaneCall {
                            path: format!("/v1/edge/sessions/{session_id}/artifacts"),
                            body: body.clone(),
                        });
                        let relpath = body
                            .get("relpath")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default();
                        let storage_key = format!("crowd-data/sessions/{session_id}/{relpath}");
                        (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "artifact": {
                                    "asset_id": body.get("asset_id").cloned().unwrap_or_default(),
                                    "relpath": relpath,
                                    "status": "declared"
                                },
                                "upload_target": {
                                    "storage_key": storage_key,
                                    "transport": {
                                        "mode": "binary_post",
                                        "method": "POST",
                                        "url": transport_url,
                                        "auth_strategy": "bearer",
                                        "supports_multipart": false,
                                        "scope_token_required": true,
                                        "scope_token_header": "X-Chek-Upload-Scope-Token",
                                        "storage_key_header": "X-Chek-Storage-Key",
                                        "headers": {
                                            "X-Chek-Session-Id": session_id,
                                            "X-Chek-Relpath": relpath,
                                            "X-Chek-Storage-Key": storage_key
                                        }
                                    }
                                }
                            })),
                        )
                    }
                }
            }),
        )
        .route(
            "/v1/edge/sessions/:session_id/receipts",
            post({
                let control_calls = control_calls.clone();
                move |AxumPath(session_id): AxumPath<String>, headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                    let control_calls = control_calls.clone();
                    async move {
                        let auth = headers
                            .get(AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or_default();
                        if auth != format!("Bearer {expected_upload_token}") {
                            return (
                                StatusCode::UNAUTHORIZED,
                                Json(serde_json::json!({
                                    "ok": false,
                                    "error": { "message": "invalid bearer token" }
                                })),
                            );
                        }
                        control_calls.lock().await.push(ControlPlaneCall {
                            path: format!("/v1/edge/sessions/{session_id}/receipts"),
                            body: body.clone(),
                        });
                        (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "session": { "session_id": session_id },
                                "session_upload_summary": {
                                    "required_artifacts_uploaded": 1
                                }
                            })),
                        )
                    }
                }
            }),
        );
    let control_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let control_addr = control_listener.local_addr()?;
    let control_server = tokio::spawn(async move {
        let _ = axum::serve(control_listener, control_app).await;
    });

    let server = support::TestServer::spawn_with_env(vec![
        ("EDGE_CROWD_UPLOAD_ENABLED".to_string(), "true".to_string()),
        ("EDGE_CROWD_UPLOAD_POLL_MS".to_string(), "100".to_string()),
        (
            "EDGE_CROWD_UPLOAD_CONTROL_BASE_URL".to_string(),
            format!("http://127.0.0.1:{}", control_addr.port()),
        ),
        (
            "EDGE_CROWD_UPLOAD_TOKEN".to_string(),
            expected_upload_token.to_string(),
        ),
        (
            "EDGE_CROWD_UPLOAD_SCOPE_TOKEN".to_string(),
            expected_scope_token.to_string(),
        ),
    ])
    .await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-demo-control-plane-001";
    let session_id = "sess-demo-control-plane-001";
    let device_id = "iphone-demo-control-plane-001";

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "source_kind": "iphone_capture",
            "clock_domain": "ios_uptime_ns",
            "clock_offset_ns": 0,
            "rtt_ns": 1_500_000,
            "sample_count": 8
        }))
        .send()
        .await?
        .error_for_status()?;

    client
        .post(format!("{}/time/sync", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": "stereo-demo-control-plane-001",
            "source_kind": "stereo_pair",
            "clock_domain": "stereo_uptime_ns",
            "clock_offset_ns": 0,
            "rtt_ns": 1_200_000,
            "sample_count": 6
        }))
        .send()
        .await?
        .error_for_status()?;

    let fusion_url = format!(
        "{}/stream/fusion?token={}",
        server.ws_base, server.edge_token
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(fusion_url).await?;
    ws.send(Message::Text(
        capture_pose_packet(trip_id, session_id, device_id, 7, 123_456_789).to_string(),
    ))
    .await?;

    client
        .post(format!("{}/ingest/stereo_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&stereo_pose_packet(
            trip_id,
            session_id,
            223_456_789,
            100,
            101,
        ))
        .send()
        .await?
        .error_for_status()?;

    for attempt in 0..3u64 {
        ws.send(Message::Text(
            capture_pose_packet(
                trip_id,
                session_id,
                device_id,
                8 + attempt,
                123_456_989 + attempt * 100,
            )
            .to_string(),
        ))
        .await?;
        client
            .post(format!("{}/ingest/stereo_pose", server.http_base))
            .bearer_auth(&server.edge_token)
            .json(&stereo_pose_packet(
                trip_id,
                session_id,
                223_456_989 + attempt * 100,
                102 + attempt * 2,
                103 + attempt * 2,
            ))
            .send()
            .await?
            .error_for_status()?;
        tokio::time::sleep(Duration::from_millis(80)).await;
    }

    client
        .post(format!("{}/ingest/wifi_pose", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&wifi_pose_packet(trip_id, session_id, 323_456_789))
        .send()
        .await?
        .error_for_status()?;

    let upload_meta = serde_json::json!({
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": device_id,
        "media_scope": "iphone",
        "media_track": "main",
        "source_kind": "iphone_capture",
        "clock_domain": "ios_uptime_ns",
        "chunk_index": 0,
        "file_type": "csv",
        "file_name": "chunk_0.csv",
        "source_start_time_ns": 120_000_000u64,
        "source_end_time_ns": 240_000_000u64,
        "frame_source_time_ns": [120_000_000u64, 180_000_000u64, 240_000_000u64],
        "frame_count": 3,
        "frame_rate_hz": 15.0
    });
    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&server.edge_token)
        .multipart(
            multipart::Form::new()
                .part(
                    "file",
                    multipart::Part::bytes(b"csv-demo-control-plane".to_vec())
                        .file_name("chunk_0.csv"),
                )
                .text("metadata", upload_meta.to_string()),
        )
        .send()
        .await?
        .error_for_status()?;

    let stereo_upload_meta = serde_json::json!({
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "stereo-demo-control-plane-001",
        "media_scope": "stereo",
        "media_track": "preview",
        "source_kind": "stereo_pair",
        "clock_domain": "stereo_uptime_ns",
        "chunk_index": 2,
        "file_type": "video",
        "file_name": "stereo_chunk_2.mp4",
        "source_start_time_ns": 320_000_000u64,
        "source_end_time_ns": 440_000_000u64,
        "frame_source_time_ns": [320_000_000u64, 380_000_000u64, 440_000_000u64],
        "frame_count": 3,
        "frame_rate_hz": 15.0
    });
    client
        .post(format!("{}/common_task/upload_chunk", server.http_base))
        .bearer_auth(&server.edge_token)
        .multipart(
            multipart::Form::new()
                .part(
                    "file",
                    multipart::Part::bytes(b"fake-mp4-control-plane".to_vec())
                        .file_name("stereo_chunk_2.mp4"),
                )
                .text("metadata", stereo_upload_meta.to_string()),
        )
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server.data_dir.join("session").join(session_id);
    wait_for_file(base_dir.join("qa").join("local_quality_report.json")).await?;
    wait_for_file(base_dir.join("upload").join("upload_manifest.json")).await?;
    wait_for_file(base_dir.join("upload").join("upload_queue.json")).await?;
    wait_for_quality_ready(base_dir.join("qa").join("local_quality_report.json")).await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_manifest.json"),
        "\"ready_for_upload\": true",
    )
    .await?;
    wait_for_file_contains(
        base_dir.join("upload").join("upload_receipts.jsonl"),
        "\"status\":\"acked\"",
    )
    .await?;

    let control_calls = control_calls.lock().await.clone();
    assert!(control_calls
        .iter()
        .any(|call| call.path == "/v1/edge/sessions/upsert"));
    assert!(control_calls
        .iter()
        .any(|call| call.path.ends_with("/artifacts")));
    assert!(control_calls
        .iter()
        .any(|call| call.path.ends_with("/receipts")));
    assert!(control_calls.iter().any(|call| {
        call.path == "/v1/edge/sessions/upsert"
            && call
                .body
                .pointer("/metadata/upload_manifest/artifacts")
                .and_then(|value| value.as_array())
                .map(|artifacts| !artifacts.is_empty())
                .unwrap_or(false)
    }));
    assert!(control_calls.iter().any(|call| {
        call.path.ends_with("/receipts")
            && call
                .body
                .get("storage_key")
                .and_then(|value| value.as_str())
                .map(|value| !value.is_empty())
                .unwrap_or(false)
    }));

    let transport_calls = transport_calls.lock().await.clone();
    assert!(!transport_calls.is_empty());
    assert!(transport_calls.iter().any(|call| {
        call.session_id == session_id
            && call.scope_token == expected_scope_token
            && !call.asset_id.is_empty()
            && !call.relpath.is_empty()
            && call.body_len > 0
    }));

    control_server.abort();
    transport_server.abort();
    Ok(())
}

#[tokio::test]
async fn session_start_context_should_flow_into_manifest_and_demo_bundle() -> anyhow::Result<()> {
    let server = support::TestServer::spawn().await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-demo-context-001";
    let session_id = "sess-demo-context-001";
    let device_id = "iphone-demo-context-001";

    client
        .post(format!("{}/session/start", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&serde_json::json!({
            "schema_version": "1.0.0",
            "trip_id": trip_id,
            "session_id": session_id,
            "device_id": device_id,
            "operator_id": "operator-teleop-001",
            "task_id": "123456789012345",
            "task_ids": ["123456789012345", "543210987654321"]
        }))
        .send()
        .await?
        .error_for_status()?;

    let base_dir = server.data_dir.join("session").join(session_id);
    wait_for_file(base_dir.join("manifest.json")).await?;
    wait_for_file(base_dir.join("demo_capture_bundle.json")).await?;
    wait_for_file_contains(
        base_dir.join("manifest.json"),
        &format!("\"capture_device_id\": \"{device_id}\""),
    )
    .await?;
    wait_for_file_contains(
        base_dir.join("demo_capture_bundle.json"),
        "\"task_id\": \"123456789012345\"",
    )
    .await?;

    let manifest: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(base_dir.join("manifest.json")).await?)?;
    assert_eq!(
        manifest
            .get("session_context")
            .and_then(|value| value.get("capture_device_id"))
            .and_then(|value| value.as_str()),
        Some(device_id)
    );
    assert_eq!(
        manifest
            .get("session_context")
            .and_then(|value| value.get("operator_id"))
            .and_then(|value| value.as_str()),
        Some("operator-teleop-001")
    );
    assert_eq!(
        manifest
            .get("session_context")
            .and_then(|value| value.get("task_id"))
            .and_then(|value| value.as_str()),
        Some("123456789012345")
    );
    assert_eq!(
        manifest
            .get("session_context")
            .and_then(|value| value.get("task_ids"))
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            }),
        Some(vec![
            "123456789012345".to_string(),
            "543210987654321".to_string(),
        ])
    );

    let bundle: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(base_dir.join("demo_capture_bundle.json")).await?)?;
    assert_eq!(
        bundle
            .get("session_context")
            .and_then(|value| value.get("task_id"))
            .and_then(|value| value.as_str()),
        Some("123456789012345")
    );
    Ok(())
}

#[tokio::test]
async fn phone_vision_ingest_should_respect_runtime_disable_flag() -> anyhow::Result<()> {
    let server = support::TestServer::spawn_with_env([("EDGE_PHONE_INGEST_ENABLED", "0")]).await?;
    let client = reqwest::Client::new();

    let trip_id = "trip-phone-disabled-001";
    let session_id = "sess-phone-disabled-001";
    let device_id = "iphone-phone-disabled-001";

    let response = client
        .post(format!("{}/ingest/phone_vision_frame", server.http_base))
        .bearer_auth(&server.edge_token)
        .json(&phone_vision_frame_packet(
            trip_id,
            session_id,
            device_id,
            1,
            123_456_789,
        ))
        .send()
        .await?;
    response.error_for_status_ref()?;
    let payload = response.json::<serde_json::Value>().await?;
    assert_eq!(
        payload.get("ok").and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        payload
            .get("error")
            .and_then(|value| value.get("code"))
            .and_then(|value| value.as_str()),
        Some("phone_ingest_disabled")
    );

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        tokio::fs::metadata(server.data_dir.join("session").join(session_id))
            .await
            .is_err(),
        "disabled phone ingest should not create a session directory"
    );
    Ok(())
}

fn capture_pose_packet(
    trip_id: &str,
    session_id: &str,
    device_id: &str,
    frame_id: u64,
    source_time_ns: u64,
) -> serde_json::Value {
    serde_json::json!({
        "type": "capture_pose_packet",
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": device_id,
        "device_class": "B",
        "platform": "ios",
        "source_time_ns": source_time_ns,
        "frame_id": frame_id,
        "camera": {
            "mode": "phone_back_depth",
            "has_depth": true,
            "image_w": 1280,
            "image_h": 720,
            "calibration": {
                "fx_px": 930.4,
                "fy_px": 931.0,
                "cx_px": 640.0,
                "cy_px": 360.0,
                "reference_image_w": 1280,
                "reference_image_h": 720
            }
        },
        "body_layout": "coco_body_17",
        "hand_layout": "mediapipe_hand_21",
        "body_kpts_3d": [[0.0, 0.2, 0.8], [0.0, 0.1, 0.8]],
        "hand_kpts_3d": build_dual_mediapipe_hands(),
        "depth_summary": {
            "valid_ratio": 0.82,
            "z_mean_m": 0.74
        },
        "capture_profile": {
            "body_3d_source": "depth_reprojected",
            "hand_3d_source": "depth_reprojected",
            "body_points_3d_valid": 15,
            "hand_points_3d_valid": 32
        },
        "confidence": {
            "body": 0.96,
            "hand": 0.93
        }
    })
}

fn phone_vision_frame_packet(
    trip_id: &str,
    session_id: &str,
    device_id: &str,
    frame_id: u64,
    source_time_ns: u64,
) -> serde_json::Value {
    let depth_samples = [
        0.45f32.to_le_bytes(),
        0.46f32.to_le_bytes(),
        0.47f32.to_le_bytes(),
        0.48f32.to_le_bytes(),
    ]
    .concat();
    let depth_b64 = base64::engine::general_purpose::STANDARD.encode(depth_samples);
    let jpeg_b64 = base64::engine::general_purpose::STANDARD.encode(tiny_jpeg_bytes());
    serde_json::json!({
        "schema_version": "1.0.0",
        "tripId": trip_id,
        "sessionId": session_id,
        "deviceId": device_id,
        "operatorTrackId": "operator-demo-raw",
        "sourceTimeNs": source_time_ns,
        "frameId": frame_id,
        "cameraMode": "phone_back_depth",
        "imageW": 1280,
        "imageH": 720,
        "sensorImageW": 1920,
        "sensorImageH": 1440,
        "normalizedWasRotatedRight": false,
        "cameraHasDepth": true,
        "cameraCalibration": {
            "fxPx": 930.4,
            "fyPx": 931.0,
            "cxPx": 640.0,
            "cyPx": 360.0,
            "referenceImageW": 1280,
            "referenceImageH": 720
        },
        "devicePose": {
            "position_m": [0.1, 0.2, 0.3],
            "rotation_deg": [1.0, 2.0, 3.0],
            "target_space": "operator_frame",
            "source": "arkit"
        },
        "deviceMotion": {
            "accel": [0.01, 0.02, 0.03],
            "gyro": [0.11, 0.12, 0.13]
        },
        "primaryImageJpegB64": jpeg_b64,
        "depthF32B64": depth_b64,
        "depthW": 2,
        "depthH": 2
    })
}

fn tiny_jpeg_bytes() -> Vec<u8> {
    let image = RgbImage::from_pixel(2, 2, image::Rgb([220, 180, 40]));
    let mut bytes = Vec::new();
    let mut encoder = JpegEncoder::new(&mut bytes);
    encoder
        .encode(
            image.as_raw(),
            image.width(),
            image.height(),
            ColorType::Rgb8.into(),
        )
        .expect("encode tiny jpeg");
    bytes
}

fn bridge_state_packet(trip_id: &str, session_id: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "bridge_state_packet",
        "schema_version": "1.0.0",
        "bridge_id": "bridge-demo-001",
        "trip_id": trip_id,
        "session_id": session_id,
        "robot_type": "unitree_g1",
        "end_effector_type": "dex3",
        "edge_time_ns": 666_000_123u64,
        "is_ready": true,
        "fault_code": "",
        "fault_message": "",
        "last_command_edge_time_ns": 665_999_000u64,
        "control_state": "armed",
        "safety_state": "normal",
        "body_control_enabled": true,
        "hand_control_enabled": true,
        "arm_q_commanded": [0.1, 0.2, 0.3, 0.4],
        "arm_tau_commanded": [0.0, 0.0, 0.0, 0.0],
        "left_hand_q_commanded": [0.5, 0.6],
        "right_hand_q_commanded": [0.7, 0.8]
    })
}

fn stereo_pose_packet(
    trip_id: &str,
    session_id: &str,
    source_time_ns: u64,
    left_frame_id: u64,
    right_frame_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "stereo-demo-001",
        "source_time_ns": source_time_ns,
        "left_frame_id": left_frame_id,
        "right_frame_id": right_frame_id,
        "body_layout": "pico_body_24",
        "hand_layout": "pico_hand_26",
        "body_kpts_3d": [[0.0, 0.2, 0.8], [0.0, 0.1, 0.8]],
        "hand_kpts_3d": build_dual_pico_hands(),
        "calibration": {
            "sensor_frame": "stereo_front_frame",
            "operator_frame": "operator_frame",
            "extrinsic_version": "stereo-ext-v1",
            "left_intrinsics": {
                "fx_px": 720.0,
                "fy_px": 721.0,
                "cx_px": 640.0,
                "cy_px": 360.0,
                "reference_image_w": 1280,
                "reference_image_h": 720
            },
            "right_intrinsics": {
                "fx_px": 719.0,
                "fy_px": 720.0,
                "cx_px": 640.0,
                "cy_px": 360.0,
                "reference_image_w": 1280,
                "reference_image_h": 720
            }
        },
        "stereo_confidence": 0.95
    })
}

fn wifi_pose_packet(trip_id: &str, session_id: &str, source_time_ns: u64) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "1.0.0",
        "trip_id": trip_id,
        "session_id": session_id,
        "device_id": "wifi-demo-001",
        "operator_track_id": "wifi-operator-main",
        "source_time_ns": source_time_ns,
        "body_layout": "pico_body_24",
        "body_space": "operator_frame",
        "body_kpts_3d": [[0.0, 0.2, 0.8], [0.0, 0.1, 0.8], [0.02, 0.18, 0.82], [0.05, 0.14, 0.85]],
        "body_confidence": 0.72,
        "source_label": "wifi_densepose",
        "person_id": 0,
        "total_persons": 1,
        "raw_body_layout": "pico_body_24",
        "raw_body_space": "wifi_pose_frame",
        "raw_body_kpts_3d": [[0.01, 0.19, 0.81], [0.01, 0.09, 0.81], [0.03, 0.17, 0.83], [0.06, 0.13, 0.86]],
        "calibration": {
            "sensor_frame": "wifi_pose_frame",
            "operator_frame": "operator_frame",
            "extrinsic_version": "wifi-ext-v1",
            "extrinsic_translation_m": [0.12, 0.0, 0.05],
            "extrinsic_rotation_quat": [0.0, 0.0, 0.0, 1.0],
            "notes": "wifi pose array calibration"
        }
    })
}

fn build_dual_mediapipe_hands() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    out.extend(build_mediapipe_hand_points([-0.132, 0.304, 0.853], true));
    out.extend(build_mediapipe_hand_points([0.132, 0.209, 0.853], false));
    out
}

fn build_dual_pico_hands() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    out.extend(build_pico_hand_points([-0.132, 0.304, 0.853], true));
    out.extend(build_pico_hand_points([0.132, 0.209, 0.853], false));
    out
}

fn build_mediapipe_hand_points(wrist: [f32; 3], left: bool) -> Vec<[f32; 3]> {
    let thumb_sign = if left { -1.0 } else { 1.0 };
    let finger_sign = if left { -1.0 } else { 1.0 };
    vec![
        wrist,
        offset(wrist, 0.010 * thumb_sign, 0.020, 0.0),
        offset(wrist, 0.022 * thumb_sign, 0.040, 0.0),
        offset(wrist, 0.034 * thumb_sign, 0.061, 0.0),
        offset(wrist, 0.046 * thumb_sign, 0.080, 0.0),
        offset(wrist, 0.018 * thumb_sign, 0.040, 0.0),
        offset(wrist, 0.020 * thumb_sign, 0.058, 0.0),
        offset(wrist, 0.024 * thumb_sign, 0.081, 0.0),
        offset(wrist, 0.028 * thumb_sign, 0.102, 0.0),
        offset(wrist, 0.004 * thumb_sign, 0.043, 0.0),
        offset(wrist, 0.005 * thumb_sign, 0.064, 0.0),
        offset(wrist, 0.005 * thumb_sign, 0.088, 0.0),
        offset(wrist, 0.005 * thumb_sign, 0.112, 0.0),
        offset(wrist, -0.012 * thumb_sign, 0.038, 0.0),
        offset(wrist, -0.014 * thumb_sign, 0.054, 0.0),
        offset(wrist, -0.018 * thumb_sign, 0.078, 0.0),
        offset(wrist, -0.022 * thumb_sign, 0.101, 0.0),
        offset(wrist, -0.028 * finger_sign, 0.028, 0.0),
        offset(wrist, -0.033 * finger_sign, 0.039, 0.0),
        offset(wrist, -0.037 * finger_sign, 0.063, 0.0),
        offset(wrist, -0.041 * finger_sign, 0.087, 0.0),
    ]
}

fn build_pico_hand_points(wrist: [f32; 3], left: bool) -> Vec<[f32; 3]> {
    let toward_thumb = if left { -1.0 } else { 1.0 };
    let toward_pinky = if left { -1.0 } else { 1.0 };
    vec![
        offset(wrist, 0.0, 0.030, 0.0),
        wrist,
        offset(wrist, 0.010 * toward_thumb, 0.020, 0.0),
        offset(wrist, 0.022 * toward_thumb, 0.040, 0.0),
        offset(wrist, 0.034 * toward_thumb, 0.061, 0.0),
        offset(wrist, 0.046 * toward_thumb, 0.080, 0.0),
        offset(wrist, 0.018 * toward_thumb, 0.040, 0.0),
        offset(wrist, 0.020 * toward_thumb, 0.058, 0.0),
        offset(wrist, 0.024 * toward_thumb, 0.081, 0.0),
        offset(wrist, 0.028 * toward_thumb, 0.102, 0.0),
        offset(wrist, 0.031 * toward_thumb, 0.122, 0.0),
        offset(wrist, 0.004 * toward_thumb, 0.043, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.064, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.088, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.112, 0.0),
        offset(wrist, 0.005 * toward_thumb, 0.133, 0.0),
        offset(wrist, -0.012 * toward_thumb, 0.038, 0.0),
        offset(wrist, -0.014 * toward_thumb, 0.054, 0.0),
        offset(wrist, -0.018 * toward_thumb, 0.078, 0.0),
        offset(wrist, -0.022 * toward_thumb, 0.101, 0.0),
        offset(wrist, -0.026 * toward_thumb, 0.121, 0.0),
        offset(wrist, -0.028 * toward_pinky, 0.028, 0.0),
        offset(wrist, -0.033 * toward_pinky, 0.039, 0.0),
        offset(wrist, -0.037 * toward_pinky, 0.063, 0.0),
        offset(wrist, -0.041 * toward_pinky, 0.087, 0.0),
        offset(wrist, -0.045 * toward_pinky, 0.109, 0.0),
    ]
}

fn offset(base: [f32; 3], dx: f32, dy: f32, dz: f32) -> [f32; 3] {
    [base[0] + dx, base[1] + dy, base[2] + dz]
}

fn build_csi_packet(
    node_id: u8,
    sequence: u32,
    rssi: i8,
    noise_floor: i8,
    source_time_ns: u64,
) -> Vec<u8> {
    let mut packet = vec![0u8; 32];
    packet[0..4].copy_from_slice(&0xC5110005u32.to_le_bytes());
    packet[4] = node_id;
    packet[5] = 3;
    packet[6..8].copy_from_slice(&30u16.to_le_bytes());
    packet[8..12].copy_from_slice(&5_785u32.to_le_bytes());
    packet[12..16].copy_from_slice(&sequence.to_le_bytes());
    packet[16] = rssi as u8;
    packet[17] = noise_floor as u8;
    packet[20..28].copy_from_slice(&source_time_ns.to_le_bytes());
    packet[28..32].copy_from_slice(&[1, 2, 3, 4]);
    packet
}

async fn wait_for_file(path: PathBuf) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待文件超时: {}", path.display());
        }
        if let Ok(meta) = tokio::fs::metadata(&path).await {
            if meta.len() > 0 {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_path_exists(path: PathBuf) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("等待路径超时: {}", path.display());
        }
        if tokio::fs::try_exists(&path).await? {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_file_contains(path: PathBuf, needle: &str) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut last_content: Option<String> = None;
    loop {
        if tokio::time::Instant::now() > deadline {
            let detail = last_content
                .as_deref()
                .map(|content| {
                    let json_summary = serde_json::from_str::<serde_json::Value>(content)
                        .ok()
                        .map(|value| {
                            let ready = value
                                .get("ready_for_upload")
                                .and_then(|v| v.as_bool())
                                .map(|v| format!("ready_for_upload={v}"))
                                .unwrap_or_default();
                            let status = value
                                .get("status")
                                .and_then(|v| v.as_str())
                                .map(|v| format!("status={v}"))
                                .unwrap_or_default();
                            let missing = value
                                .get("missing_artifacts")
                                .and_then(|v| v.as_array())
                                .map(|items| {
                                    let joined = items
                                        .iter()
                                        .filter_map(|item| item.as_str())
                                        .collect::<Vec<_>>()
                                        .join(",");
                                    format!("missing_artifacts=[{joined}]")
                                })
                                .unwrap_or_default();
                            [ready, status, missing]
                                .into_iter()
                                .filter(|item| !item.is_empty())
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                        .filter(|summary| !summary.is_empty())
                        .map(|summary| format!("{summary}; "));
                    let compact = content.replace('\n', " ");
                    let preview = compact.chars().take(400).collect::<String>();
                    format!(
                        "; {}current_content={preview}",
                        json_summary.unwrap_or_default()
                    )
                })
                .unwrap_or_else(|| "; current_content=<unreadable or missing>".to_string());
            anyhow::bail!(
                "等待文件内容超时: {} contains {}{}",
                path.display(),
                needle,
                detail
            );
        }
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            last_content = Some(content.clone());
            if content.contains(needle) {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_quality_ready(path: PathBuf) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut last_content: Option<String> = None;
    loop {
        if tokio::time::Instant::now() > deadline {
            let detail = last_content
                .as_deref()
                .map(|content| {
                    let json_summary = serde_json::from_str::<serde_json::Value>(content)
                        .ok()
                        .map(|value| {
                            let status = value
                                .get("status")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let missing = value
                                .get("missing_artifacts")
                                .and_then(|v| v.as_array())
                                .map(|items| {
                                    items
                                        .iter()
                                        .filter_map(|item| item.as_str())
                                        .collect::<Vec<_>>()
                                        .join(",")
                                })
                                .unwrap_or_default();
                            format!("status={status}; missing_artifacts=[{missing}]")
                        })
                        .unwrap_or_else(|| "status=<unparseable>".to_string());
                    let compact = content.replace('\n', " ");
                    let preview = compact.chars().take(400).collect::<String>();
                    format!("{json_summary}; current_content={preview}")
                })
                .unwrap_or_else(|| "current_content=<unreadable or missing>".to_string());
            anyhow::bail!(
                "等待 local quality ready 超时: {} ({detail})",
                path.display()
            );
        }
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            last_content = Some(content.clone());
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) {
                if matches!(
                    value.get("status").and_then(|v| v.as_str()),
                    Some("pass") | Some("retry_recommended")
                ) {
                    return Ok(());
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
