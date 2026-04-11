use clap::Parser;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use wifi_densepose_sensing_server::rvf_container::{RvfBuilder, RvfReader};
use wifi_densepose_sensing_server::rvf_pipeline::{
    decode_sona_profile_deltas, SEG_AGGREGATE_WEIGHTS,
};

const DEFAULT_SONA_ALPHA: f32 = 8.0;
const SEG_META: u8 = 0x07;

#[derive(Debug, Parser)]
#[command(
    name = "inject-sona-profile",
    about = "Inject an exact geometry-conditioned SONA profile into an RVF"
)]
struct Args {
    /// Base RVF that will receive the new profile.
    #[arg(long)]
    base_rvf: PathBuf,

    /// Candidate RVF whose weights define the overlay delta.
    #[arg(long)]
    candidate_rvf: PathBuf,

    /// Output RVF path.
    #[arg(long)]
    output_rvf: PathBuf,

    /// Scene/profile name to store inside the SONA aggregate-weight segment.
    #[arg(long)]
    profile: String,

    /// Scaling alpha used for the rank-1 exact delta encoding.
    #[arg(long, default_value_t = DEFAULT_SONA_ALPHA)]
    alpha: f32,

    /// Replace an existing profile with the same name if present.
    #[arg(long, default_value_t = false)]
    replace_existing: bool,
}

fn model_id(reader: &RvfReader, fallback_path: &Path) -> String {
    reader
        .manifest()
        .and_then(|manifest| {
            manifest
                .get("model_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            fallback_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| fallback_path.display().to_string())
}

fn exact_rank1_sona(delta: &[f32], alpha: f32) -> Result<(Vec<f32>, Vec<f32>), String> {
    if !alpha.is_finite() || alpha == 0.0 {
        return Err("alpha must be a finite non-zero value".to_string());
    }
    let lora_a = delta.iter().map(|value| *value / alpha).collect::<Vec<_>>();
    let lora_b = vec![1.0f32];
    Ok((lora_a, lora_b))
}

fn build_sona_payload(
    profile: &str,
    base_model_id: &str,
    candidate_model_id: &str,
    delta: &[f32],
    alpha: f32,
) -> Result<Vec<u8>, String> {
    let (lora_a, lora_b) = exact_rank1_sona(delta, alpha)?;
    let max_abs_delta = delta.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let mean_abs_delta = if delta.is_empty() {
        0.0
    } else {
        delta.iter().map(|value| value.abs() as f64).sum::<f64>() as f32 / delta.len() as f32
    };
    serde_json::to_vec(&json!({
        "env": profile,
        "rank": 1,
        "alpha": alpha,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "source": "candidate_delta_import",
        "base_model_id": base_model_id,
        "candidate_model_id": candidate_model_id,
        "delta_stats": {
            "param_count": delta.len(),
            "max_abs_delta": max_abs_delta,
            "mean_abs_delta": mean_abs_delta,
        }
    }))
    .map_err(|err| format!("failed to serialize SONA payload: {err}"))
}

fn ensure_base_model_metadata(
    mut metadata: Value,
    base_model_id: &str,
    default_sona_profile: &str,
) -> Value {
    if let Some(training) = metadata
        .as_object_mut()
        .map(|root| root.entry("training").or_insert_with(|| json!({})))
        .and_then(Value::as_object_mut)
    {
        training.insert(
            "base_model_id".to_string(),
            Value::String(base_model_id.to_string()),
        );
        training.insert(
            "default_sona_profile".to_string(),
            Value::String(default_sona_profile.to_string()),
        );
    }

    if let Some(model_config) = metadata
        .as_object_mut()
        .map(|root| root.entry("model_config").or_insert_with(|| json!({})))
        .and_then(Value::as_object_mut)
    {
        model_config.insert(
            "base_model_id".to_string(),
            Value::String(base_model_id.to_string()),
        );
        model_config.insert(
            "default_sona_profile".to_string(),
            Value::String(default_sona_profile.to_string()),
        );
    }

    metadata
}

fn write_profiled_rvf(args: &Args) -> Result<Value, String> {
    let base_reader = RvfReader::from_file(&args.base_rvf)?;
    let candidate_reader = RvfReader::from_file(&args.candidate_rvf)?;

    let base_weights = base_reader
        .weights()
        .ok_or_else(|| format!("{} has no weight segment", args.base_rvf.display()))?;
    let candidate_weights = candidate_reader
        .weights()
        .ok_or_else(|| format!("{} has no weight segment", args.candidate_rvf.display()))?;

    if base_weights.len() != candidate_weights.len() {
        return Err(format!(
            "weight length mismatch: base={} candidate={}",
            base_weights.len(),
            candidate_weights.len()
        ));
    }

    let delta = candidate_weights
        .iter()
        .zip(base_weights.iter())
        .map(|(candidate, base)| candidate - base)
        .collect::<Vec<_>>();
    let base_model_id = model_id(&base_reader, &args.base_rvf);
    let candidate_model_id = model_id(&candidate_reader, &args.candidate_rvf);
    let payload = build_sona_payload(
        &args.profile,
        &base_model_id,
        &candidate_model_id,
        &delta,
        args.alpha,
    )?;

    let mut builder = RvfBuilder::new();
    let mut replaced_existing = false;
    for (header, segment_payload) in base_reader.segments() {
        if header.seg_type == SEG_META {
            if let Ok(metadata) = serde_json::from_slice::<Value>(segment_payload) {
                builder.add_metadata(&ensure_base_model_metadata(
                    metadata,
                    &base_model_id,
                    &args.profile,
                ));
                continue;
            }
        }
        if header.seg_type == SEG_AGGREGATE_WEIGHTS {
            let existing_env = serde_json::from_slice::<Value>(segment_payload)
                .ok()
                .and_then(|value| value.get("env").and_then(Value::as_str).map(str::to_string));
            if existing_env.as_deref() == Some(args.profile.as_str()) {
                if !args.replace_existing {
                    return Err(format!(
                        "profile '{}' already exists in {}; use --replace-existing to overwrite",
                        args.profile,
                        args.base_rvf.display()
                    ));
                }
                replaced_existing = true;
                continue;
            }
        }
        builder.add_raw_segment(header.seg_type, segment_payload);
    }
    builder.add_raw_segment(SEG_AGGREGATE_WEIGHTS, &payload);
    builder
        .write_to_file(&args.output_rvf)
        .map_err(|err| format!("failed to write {}: {err}", args.output_rvf.display()))?;

    let output_reader = RvfReader::from_file(&args.output_rvf)?;
    let output_profiles = decode_sona_profile_deltas(&output_reader);
    let output_delta = output_profiles
        .get(&args.profile)
        .ok_or_else(|| format!("profile '{}' missing after write", args.profile))?;
    if output_delta.len() != delta.len() {
        return Err(format!(
            "decoded profile length mismatch: expected {} got {}",
            delta.len(),
            output_delta.len()
        ));
    }

    let max_abs_error = output_delta
        .iter()
        .zip(delta.iter())
        .map(|(decoded, original)| (decoded - original).abs())
        .fold(0.0f32, f32::max);

    Ok(json!({
        "status": "ok",
        "output_rvf": args.output_rvf,
        "profile": args.profile,
        "base_model_id": base_model_id,
        "candidate_model_id": candidate_model_id,
        "replaced_existing": replaced_existing,
        "base_segment_count": base_reader.segment_count(),
        "output_segment_count": output_reader.segment_count(),
        "param_count": delta.len(),
        "max_abs_error": max_abs_error,
        "available_profiles": output_profiles.keys().cloned().collect::<Vec<_>>(),
    }))
}

fn main() {
    let args = Args::parse();
    match write_profiled_rvf(&args) {
        Ok(summary) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&summary)
                    .unwrap_or_else(|_| "{\"status\":\"ok\"}".to_string())
            );
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn exact_rank1_sona_round_trips_delta() {
        let delta = vec![0.5f32, -1.25, 3.0, -0.125];
        let dir = tempdir().unwrap();
        let path = dir.path().join("profiled.rvf");

        let mut builder = RvfBuilder::new();
        builder.add_manifest("base", "0.1.0", "base");
        builder.add_weights(&[0.0, 0.0, 0.0, 0.0]);
        let payload = build_sona_payload("scene-geometry", "base", "candidate", &delta, 8.0)
            .expect("payload should build");
        builder.add_raw_segment(SEG_AGGREGATE_WEIGHTS, &payload);
        builder.write_to_file(&path).unwrap();

        let reader = RvfReader::from_file(&path).unwrap();
        let decoded = decode_sona_profile_deltas(&reader);
        let recovered = decoded.get("scene-geometry").unwrap();
        assert_eq!(recovered.len(), delta.len());
        for (lhs, rhs) in recovered.iter().zip(delta.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }
}
