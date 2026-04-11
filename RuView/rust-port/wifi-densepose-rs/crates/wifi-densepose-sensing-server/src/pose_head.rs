use serde::{Deserialize, Serialize};

fn default_residual_hidden_dim() -> usize {
    64
}

fn default_residual_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PoseHeadConfig {
    Linear {
        n_features: usize,
        n_targets: usize,
    },
    ResidualMlp {
        n_features: usize,
        n_targets: usize,
        #[serde(default = "default_residual_hidden_dim")]
        hidden_dim: usize,
        #[serde(default = "default_residual_scale")]
        residual_scale: f64,
    },
}

impl PoseHeadConfig {
    pub fn linear(n_features: usize, n_targets: usize) -> Self {
        Self::Linear {
            n_features,
            n_targets,
        }
    }

    pub fn residual_mlp(
        n_features: usize,
        n_targets: usize,
        hidden_dim: usize,
        residual_scale: f64,
    ) -> Self {
        Self::ResidualMlp {
            n_features,
            n_targets,
            hidden_dim: hidden_dim.max(1),
            residual_scale,
        }
    }

    pub fn n_features(&self) -> usize {
        match self {
            Self::Linear { n_features, .. } | Self::ResidualMlp { n_features, .. } => *n_features,
        }
    }

    pub fn n_targets(&self) -> usize {
        match self {
            Self::Linear { n_targets, .. } | Self::ResidualMlp { n_targets, .. } => *n_targets,
        }
    }

    pub fn hidden_dim(&self) -> Option<usize> {
        match self {
            Self::ResidualMlp { hidden_dim, .. } => Some(*hidden_dim),
            Self::Linear { .. } => None,
        }
    }

    pub fn residual_scale(&self) -> f64 {
        match self {
            Self::ResidualMlp { residual_scale, .. } => *residual_scale,
            Self::Linear { .. } => 0.0,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Linear { .. } => "linear",
            Self::ResidualMlp { .. } => "residual_mlp",
        }
    }

    pub fn expected_params(&self) -> usize {
        match self {
            Self::Linear {
                n_features,
                n_targets,
            } => n_targets * n_features + n_targets,
            Self::ResidualMlp {
                n_features,
                n_targets,
                hidden_dim,
                ..
            } => {
                n_targets * n_features
                    + n_targets
                    + hidden_dim * n_features
                    + hidden_dim
                    + n_targets * hidden_dim
                    + n_targets
            }
        }
    }

    pub fn description(&self) -> String {
        match self {
            Self::Linear {
                n_features,
                n_targets,
            } => format!("linear, {n_features} features, {n_targets} targets"),
            Self::ResidualMlp {
                n_features,
                n_targets,
                hidden_dim,
                residual_scale,
            } => format!(
                "residual_mlp, {n_features} features, {n_targets} targets, hidden_dim={hidden_dim}, residual_scale={residual_scale:.3}"
            ),
        }
    }

    pub fn from_metadata(
        metadata: Option<&serde_json::Value>,
        fallback_n_features: usize,
        fallback_n_targets: usize,
    ) -> Self {
        metadata
            .and_then(|value| value.get("model_config").cloned())
            .and_then(|value| serde_json::from_value::<PoseHeadConfig>(value).ok())
            .unwrap_or_else(|| Self::linear(fallback_n_features, fallback_n_targets))
    }
}

fn relu(value: f64) -> f64 {
    value.max(0.0)
}

pub fn forward_with_f64_params(
    config: &PoseHeadConfig,
    params: &[f64],
    features: &[f64],
) -> Option<Vec<f64>> {
    if params.len() < config.expected_params() || features.len() < config.n_features() {
        return None;
    }

    match config {
        PoseHeadConfig::Linear {
            n_features,
            n_targets,
        } => {
            let weights_end = n_targets * n_features;
            let mut outputs = vec![0.0; *n_targets];
            for target_idx in 0..*n_targets {
                let row_start = target_idx * n_features;
                let mut sum = params[weights_end + target_idx];
                for feature_idx in 0..*n_features {
                    sum += params[row_start + feature_idx] * features[feature_idx];
                }
                outputs[target_idx] = sum;
            }
            Some(outputs)
        }
        PoseHeadConfig::ResidualMlp {
            n_features,
            n_targets,
            hidden_dim,
            residual_scale,
        } => {
            let linear_w_end = n_targets * n_features;
            let linear_b_end = linear_w_end + n_targets;
            let hidden_w_end = linear_b_end + hidden_dim * n_features;
            let hidden_b_end = hidden_w_end + hidden_dim;
            let out_w_end = hidden_b_end + n_targets * hidden_dim;

            let mut hidden = vec![0.0; *hidden_dim];
            for hidden_idx in 0..*hidden_dim {
                let row_start = linear_b_end + hidden_idx * n_features;
                let mut sum = params[hidden_w_end + hidden_idx];
                for feature_idx in 0..*n_features {
                    sum += params[row_start + feature_idx] * features[feature_idx];
                }
                hidden[hidden_idx] = relu(sum);
            }

            let mut outputs = vec![0.0; *n_targets];
            for target_idx in 0..*n_targets {
                let linear_row_start = target_idx * n_features;
                let residual_row_start = hidden_b_end + target_idx * hidden_dim;

                let mut linear = params[linear_w_end + target_idx];
                for feature_idx in 0..*n_features {
                    linear += params[linear_row_start + feature_idx] * features[feature_idx];
                }

                let mut residual = params[out_w_end + target_idx];
                for hidden_idx in 0..*hidden_dim {
                    residual += params[residual_row_start + hidden_idx] * hidden[hidden_idx];
                }

                outputs[target_idx] = linear + residual_scale * residual;
            }

            Some(outputs)
        }
    }
}

pub fn forward_with_f32_params(
    config: &PoseHeadConfig,
    params: &[f32],
    features: &[f64],
) -> Option<Vec<f64>> {
    if params.len() < config.expected_params() || features.len() < config.n_features() {
        return None;
    }
    let params_f64: Vec<f64> = params.iter().map(|value| *value as f64).collect();
    forward_with_f64_params(config, &params_f64, features)
}

#[cfg(test)]
mod tests {
    use super::{forward_with_f32_params, forward_with_f64_params, PoseHeadConfig};

    #[test]
    fn linear_head_round_trip_forward() {
        let cfg = PoseHeadConfig::linear(2, 2);
        let params = vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5];
        let outputs = forward_with_f64_params(&cfg, &params, &[3.0, 7.0]).unwrap();
        assert_eq!(outputs, vec![3.5, 6.5]);
    }

    #[test]
    fn residual_mlp_zero_residual_matches_linear_branch() {
        let cfg = PoseHeadConfig::residual_mlp(2, 2, 3, 1.0);
        let params = vec![
            1.0, 0.0, 0.0, 1.0, // linear w
            0.5, -0.5, // linear b
            1.0, 0.0, 0.0, 1.0, -1.0, 1.0, // hidden w
            0.0, 0.0, 0.0, // hidden b
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // out w
            0.0, 0.0, // out b
        ];
        let outputs = forward_with_f64_params(&cfg, &params, &[3.0, 7.0]).unwrap();
        assert_eq!(outputs, vec![3.5, 6.5]);
    }

    #[test]
    fn residual_mlp_f32_forward_applies_residual_branch() {
        let cfg = PoseHeadConfig::residual_mlp(2, 1, 2, 0.5);
        let params = vec![
            1.0, 0.0, // linear w
            0.0, // linear b
            1.0, 0.0, 0.0, 1.0, // hidden w
            0.0, 0.0, // hidden b
            1.0, 1.0, // out w
            0.0, // out b
        ];
        let outputs = forward_with_f32_params(&cfg, &params, &[2.0, 4.0]).unwrap();
        assert!((outputs[0] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn parse_from_metadata_falls_back_to_linear() {
        let cfg = PoseHeadConfig::from_metadata(None, 10, 51);
        assert_eq!(cfg, PoseHeadConfig::linear(10, 51));
    }
}
