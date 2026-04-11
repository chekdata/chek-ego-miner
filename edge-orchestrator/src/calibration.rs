use std::fs;
use std::path::{Component, PathBuf};
use std::sync::RwLock;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use tracing::warn;

const IPHONE_SIMILARITY_MIN_SCALE: f32 = 0.75;
const IPHONE_SIMILARITY_MAX_SCALE: f32 = 1.25;
const IPHONE_SIMILARITY_MAX_RMS_FACTOR: f32 = 0.92;

#[derive(Clone, Copy, Debug)]
pub struct PointPair {
    pub source: [f32; 3],
    pub target: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IphoneStereoExtrinsic {
    pub source_frame: String,
    pub target_frame: String,
    pub extrinsic_version: String,
    #[serde(default = "unit_scale")]
    pub extrinsic_scale: f32,
    pub extrinsic_translation_m: [f32; 3],
    pub extrinsic_rotation_quat: [f32; 4],
    pub sample_count: usize,
    pub rms_error_m: f32,
    pub solved_edge_time_ns: u64,
}

impl IphoneStereoExtrinsic {
    pub fn apply_point(&self, point: [f32; 3]) -> [f32; 3] {
        if !valid_point(point) {
            return point;
        }
        let scale = if self.extrinsic_scale.is_finite() && self.extrinsic_scale > 0.0 {
            self.extrinsic_scale
        } else {
            1.0
        };
        let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            self.extrinsic_rotation_quat[3],
            self.extrinsic_rotation_quat[0],
            self.extrinsic_rotation_quat[1],
            self.extrinsic_rotation_quat[2],
        ));
        let scaled = Vector3::new(point[0], point[1], point[2]) * scale;
        let translated = rotation.transform_vector(&scaled)
            + Vector3::new(
                self.extrinsic_translation_m[0],
                self.extrinsic_translation_m[1],
                self.extrinsic_translation_m[2],
            );
        [translated.x, translated.y, translated.z]
    }
}

fn unit_scale() -> f32 {
    1.0
}

pub struct IphoneStereoCalibrationStore {
    path: PathBuf,
    inner: RwLock<Option<IphoneStereoExtrinsic>>,
}

pub type WifiStereoExtrinsic = IphoneStereoExtrinsic;

pub struct WifiStereoCalibrationStore {
    path: PathBuf,
    inner: RwLock<Option<WifiStereoExtrinsic>>,
}

impl WifiStereoCalibrationStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let safe_path = if path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            warn!(
                path=%path.display(),
                "invalid wifi-stereo calibration path; using inert path"
            );
            PathBuf::from("runtime/invalid_wifi_stereo_extrinsic.json")
        } else {
            path.clone()
        };
        let loaded = fs::read_to_string(&safe_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<WifiStereoExtrinsic>(&raw).ok());
        if loaded.is_none() && safe_path.exists() {
            warn!(path=%safe_path.display(), "failed to load wifi-stereo extrinsic; continuing without calibration");
        }
        Self {
            path: safe_path,
            inner: RwLock::new(loaded),
        }
    }

    pub fn snapshot(&self) -> Option<WifiStereoExtrinsic> {
        self.inner
            .read()
            .expect("wifi-stereo calibration lock poisoned")
            .clone()
    }

    pub fn save(&self, calibration: WifiStereoExtrinsic) -> anyhow::Result<()> {
        if self
            .path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(anyhow::anyhow!(
                "wifi_stereo_extrinsic_path 不能包含 . 或 .. 路径段"
            ));
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let raw = serde_json::to_vec_pretty(&calibration)?;
        fs::write(&self.path, raw)?;
        *self
            .inner
            .write()
            .expect("wifi-stereo calibration lock poisoned") = Some(calibration);
        Ok(())
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl IphoneStereoCalibrationStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let safe_path = if path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            warn!(
                path=%path.display(),
                "invalid iphone-stereo calibration path; using inert path"
            );
            PathBuf::from("runtime/invalid_iphone_stereo_extrinsic.json")
        } else {
            path.clone()
        };
        let loaded = fs::read_to_string(&safe_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<IphoneStereoExtrinsic>(&raw).ok());
        if loaded.is_none() && safe_path.exists() {
            warn!(path=%safe_path.display(), "failed to load iphone-stereo extrinsic; continuing without calibration");
        }
        Self {
            path: safe_path,
            inner: RwLock::new(loaded),
        }
    }

    pub fn snapshot(&self) -> Option<IphoneStereoExtrinsic> {
        self.inner
            .read()
            .expect("iphone-stereo calibration lock poisoned")
            .clone()
    }

    pub fn save(&self, calibration: IphoneStereoExtrinsic) -> anyhow::Result<()> {
        if self
            .path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            return Err(anyhow::anyhow!(
                "iphone_stereo_extrinsic_path 不能包含 . 或 .. 路径段"
            ));
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let raw = serde_json::to_vec_pretty(&calibration)?;
        fs::write(&self.path, raw)?;
        *self
            .inner
            .write()
            .expect("iphone-stereo calibration lock poisoned") = Some(calibration);
        Ok(())
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

pub fn transform_points_3d(
    points: &[[f32; 3]],
    calibration: Option<&IphoneStereoExtrinsic>,
) -> Vec<[f32; 3]> {
    match calibration {
        Some(calibration) => points
            .iter()
            .copied()
            .map(|point| calibration.apply_point(point))
            .collect(),
        None => points.to_vec(),
    }
}

pub fn solve_rigid_transform(
    pairs: &[PointPair],
    solved_edge_time_ns: u64,
) -> anyhow::Result<IphoneStereoExtrinsic> {
    anyhow::ensure!(
        pairs.len() >= 6,
        "至少需要 6 个 iPhone↔双目配对点，当前只有 {} 个",
        pairs.len()
    );

    let (rotation, translation, filtered_pairs) = solve_trimmed_kabsch(pairs)?;
    let rotation3 = Rotation3::from_matrix_unchecked(rotation);
    let quat = UnitQuaternion::from_rotation_matrix(&rotation3);
    let rms_error_m = compute_rms_error(&filtered_pairs, &rotation, &translation);

    Ok(IphoneStereoExtrinsic {
        source_frame: "iphone_capture_frame".to_string(),
        target_frame: "stereo_pair_frame".to_string(),
        extrinsic_version: format!("iphone-stereo-live-{}", solved_edge_time_ns),
        extrinsic_scale: 1.0,
        extrinsic_translation_m: [translation.x, translation.y, translation.z],
        extrinsic_rotation_quat: [quat.i, quat.j, quat.k, quat.w],
        sample_count: filtered_pairs.len(),
        rms_error_m,
        solved_edge_time_ns,
    })
}

pub fn solve_iphone_stereo_transform(
    pairs: &[PointPair],
    solved_edge_time_ns: u64,
) -> anyhow::Result<IphoneStereoExtrinsic> {
    let rigid = solve_rigid_transform(pairs, solved_edge_time_ns)?;
    let similarity = solve_iphone_similarity_transform(pairs, solved_edge_time_ns).ok();

    match similarity {
        Some(candidate)
            if candidate.extrinsic_scale.is_finite()
                && (IPHONE_SIMILARITY_MIN_SCALE..=IPHONE_SIMILARITY_MAX_SCALE)
                    .contains(&candidate.extrinsic_scale)
                && candidate.rms_error_m
                    <= rigid.rms_error_m * IPHONE_SIMILARITY_MAX_RMS_FACTOR =>
        {
            Ok(candidate)
        }
        _ => Ok(rigid),
    }
}

fn solve_iphone_similarity_transform(
    pairs: &[PointPair],
    solved_edge_time_ns: u64,
) -> anyhow::Result<IphoneStereoExtrinsic> {
    anyhow::ensure!(
        pairs.len() >= 6,
        "至少需要 6 个 iPhone↔双目配对点，当前只有 {} 个",
        pairs.len()
    );

    let (scale, rotation, translation, filtered_pairs) = solve_trimmed_umeyama(pairs)?;
    let rotation3 = Rotation3::from_matrix_unchecked(rotation);
    let quat = UnitQuaternion::from_rotation_matrix(&rotation3);
    let rms_error_m = compute_rms_error_with_scale(&filtered_pairs, scale, &rotation, &translation);

    Ok(IphoneStereoExtrinsic {
        source_frame: "iphone_capture_frame".to_string(),
        target_frame: "stereo_pair_frame".to_string(),
        extrinsic_version: format!("iphone-stereo-live-{}", solved_edge_time_ns),
        extrinsic_scale: scale,
        extrinsic_translation_m: [translation.x, translation.y, translation.z],
        extrinsic_rotation_quat: [quat.i, quat.j, quat.k, quat.w],
        sample_count: filtered_pairs.len(),
        rms_error_m,
        solved_edge_time_ns,
    })
}

pub fn solve_similarity_transform(
    pairs: &[PointPair],
    solved_edge_time_ns: u64,
) -> anyhow::Result<WifiStereoExtrinsic> {
    anyhow::ensure!(
        pairs.len() >= 6,
        "至少需要 6 个 Wi‑Fi↔双目配对点，当前只有 {} 个",
        pairs.len()
    );

    let (scale, rotation, translation, filtered_pairs) = solve_trimmed_umeyama(pairs)?;
    let rotation3 = Rotation3::from_matrix_unchecked(rotation);
    let quat = UnitQuaternion::from_rotation_matrix(&rotation3);
    let rms_error_m = compute_rms_error_with_scale(&filtered_pairs, scale, &rotation, &translation);

    Ok(WifiStereoExtrinsic {
        source_frame: "wifi_pose_frame".to_string(),
        target_frame: "operator_frame".to_string(),
        extrinsic_version: format!("wifi-stereo-live-{}", solved_edge_time_ns),
        extrinsic_scale: scale,
        extrinsic_translation_m: [translation.x, translation.y, translation.z],
        extrinsic_rotation_quat: [quat.i, quat.j, quat.k, quat.w],
        sample_count: filtered_pairs.len(),
        rms_error_m,
        solved_edge_time_ns,
    })
}

fn solve_trimmed_kabsch(
    pairs: &[PointPair],
) -> anyhow::Result<(Matrix3<f32>, Vector3<f32>, Vec<PointPair>)> {
    let (mut rotation, mut translation) = solve_kabsch(pairs)?;
    let mut filtered = pairs.to_vec();

    for _ in 0..2 {
        let residuals = filtered
            .iter()
            .map(|pair| residual_m(pair, &rotation, &translation))
            .collect::<Vec<_>>();
        if residuals.len() < 6 {
            break;
        }

        let mut sorted = residuals.clone();
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let threshold = (median * 1.8).clamp(0.08, 0.25);
        let next_filtered = filtered
            .iter()
            .copied()
            .zip(residuals.iter().copied())
            .filter_map(|(pair, residual)| (residual <= threshold).then_some(pair))
            .collect::<Vec<_>>();
        if next_filtered.len() < 6 || next_filtered.len() == filtered.len() {
            break;
        }
        filtered = next_filtered;
        (rotation, translation) = solve_kabsch(&filtered)?;
    }

    Ok((rotation, translation, filtered))
}

fn solve_trimmed_umeyama(
    pairs: &[PointPair],
) -> anyhow::Result<(f32, Matrix3<f32>, Vector3<f32>, Vec<PointPair>)> {
    let (mut scale, mut rotation, mut translation) = solve_umeyama(pairs)?;
    let mut filtered = pairs.to_vec();

    for _ in 0..2 {
        let residuals = filtered
            .iter()
            .map(|pair| residual_m_with_scale(pair, scale, &rotation, &translation))
            .collect::<Vec<_>>();
        if residuals.len() < 6 {
            break;
        }

        let mut sorted = residuals.clone();
        sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let threshold = (median * 1.8).clamp(0.08, 0.35);
        let next_filtered = filtered
            .iter()
            .copied()
            .zip(residuals.iter().copied())
            .filter_map(|(pair, residual)| (residual <= threshold).then_some(pair))
            .collect::<Vec<_>>();
        if next_filtered.len() < 6 || next_filtered.len() == filtered.len() {
            break;
        }
        filtered = next_filtered;
        (scale, rotation, translation) = solve_umeyama(&filtered)?;
    }

    Ok((scale, rotation, translation, filtered))
}

fn solve_kabsch(pairs: &[PointPair]) -> anyhow::Result<(Matrix3<f32>, Vector3<f32>)> {
    anyhow::ensure!(
        pairs.len() >= 6,
        "至少需要 6 个 iPhone↔双目配对点，当前只有 {} 个",
        pairs.len()
    );

    let source_points: Vec<Vector3<f32>> = pairs
        .iter()
        .map(|pair| Vector3::new(pair.source[0], pair.source[1], pair.source[2]))
        .collect();
    let target_points: Vec<Vector3<f32>> = pairs
        .iter()
        .map(|pair| Vector3::new(pair.target[0], pair.target[1], pair.target[2]))
        .collect();

    let src_centroid = centroid(&source_points);
    let dst_centroid = centroid(&target_points);

    let mut covariance = Matrix3::<f32>::zeros();
    for (src, dst) in source_points.iter().zip(target_points.iter()) {
        let src_delta = src - src_centroid;
        let dst_delta = dst - dst_centroid;
        covariance += src_delta * dst_delta.transpose();
    }

    let svd = covariance.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("无法求解 iPhone↔双目标定：SVD(U) 失败"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("无法求解 iPhone↔双目标定：SVD(Vt) 失败"))?;
    anyhow::ensure!(
        svd.singular_values.len() >= 2 && svd.singular_values[1] > 1e-6,
        "标定样本退化，手腕运动不足以求解稳定外参"
    );

    let mut rotation = v_t.transpose() * u.transpose();
    if rotation.determinant() < 0.0 {
        let mut reflection = Matrix3::<f32>::identity();
        reflection[(2, 2)] = -1.0;
        rotation = v_t.transpose() * reflection * u.transpose();
    }
    let translation = dst_centroid - rotation * src_centroid;
    Ok((rotation, translation))
}

fn solve_umeyama(pairs: &[PointPair]) -> anyhow::Result<(f32, Matrix3<f32>, Vector3<f32>)> {
    anyhow::ensure!(
        pairs.len() >= 6,
        "至少需要 6 个 Wi‑Fi↔双目配对点，当前只有 {} 个",
        pairs.len()
    );

    let source_points: Vec<Vector3<f32>> = pairs
        .iter()
        .map(|pair| Vector3::new(pair.source[0], pair.source[1], pair.source[2]))
        .collect();
    let target_points: Vec<Vector3<f32>> = pairs
        .iter()
        .map(|pair| Vector3::new(pair.target[0], pair.target[1], pair.target[2]))
        .collect();

    let src_centroid = centroid(&source_points);
    let dst_centroid = centroid(&target_points);

    let mut covariance = Matrix3::<f32>::zeros();
    let mut src_var = 0.0f32;
    let n = source_points.len() as f32;
    for (src, dst) in source_points.iter().zip(target_points.iter()) {
        let src_delta = src - src_centroid;
        let dst_delta = dst - dst_centroid;
        covariance += dst_delta * src_delta.transpose();
        src_var += src_delta.norm_squared();
    }
    covariance /= n;
    src_var /= n;
    anyhow::ensure!(src_var > 1e-6, "Wi‑Fi↔双目标定样本退化，源点变化不足");

    let svd = covariance.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| anyhow::anyhow!("无法求解 Wi‑Fi↔双目标定：SVD(U) 失败"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow::anyhow!("无法求解 Wi‑Fi↔双目标定：SVD(Vt) 失败"))?;
    anyhow::ensure!(
        svd.singular_values.len() >= 2 && svd.singular_values[1] > 1e-6,
        "Wi‑Fi↔双目标定样本退化，人体姿态变化不足以求解稳定外参"
    );

    let mut correction = Matrix3::<f32>::identity();
    if (u.determinant() * v_t.determinant()) < 0.0 {
        correction[(2, 2)] = -1.0;
    }

    let rotation = u * correction * v_t;
    let mut trace = 0.0f32;
    for index in 0..3 {
        trace += svd.singular_values[index] * correction[(index, index)];
    }
    let scale = (trace / src_var).clamp(0.2, 4.0);
    let translation = dst_centroid - scale * rotation * src_centroid;
    Ok((scale, rotation, translation))
}

fn compute_rms_error(
    pairs: &[PointPair],
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
) -> f32 {
    let mut err_sum = 0.0f32;
    for pair in pairs {
        let src = Vector3::new(pair.source[0], pair.source[1], pair.source[2]);
        let dst = Vector3::new(pair.target[0], pair.target[1], pair.target[2]);
        let aligned = rotation * src + translation;
        err_sum += (aligned - dst).norm_squared();
    }
    (err_sum / pairs.len() as f32).sqrt()
}

fn compute_rms_error_with_scale(
    pairs: &[PointPair],
    scale: f32,
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
) -> f32 {
    let mut err_sum = 0.0f32;
    for pair in pairs {
        let src = Vector3::new(pair.source[0], pair.source[1], pair.source[2]);
        let dst = Vector3::new(pair.target[0], pair.target[1], pair.target[2]);
        let aligned = scale * (rotation * src) + translation;
        err_sum += (aligned - dst).norm_squared();
    }
    (err_sum / pairs.len() as f32).sqrt()
}

fn residual_m(pair: &PointPair, rotation: &Matrix3<f32>, translation: &Vector3<f32>) -> f32 {
    let src = Vector3::new(pair.source[0], pair.source[1], pair.source[2]);
    let dst = Vector3::new(pair.target[0], pair.target[1], pair.target[2]);
    let aligned = rotation * src + translation;
    (aligned - dst).norm()
}

fn residual_m_with_scale(
    pair: &PointPair,
    scale: f32,
    rotation: &Matrix3<f32>,
    translation: &Vector3<f32>,
) -> f32 {
    let src = Vector3::new(pair.source[0], pair.source[1], pair.source[2]);
    let dst = Vector3::new(pair.target[0], pair.target[1], pair.target[2]);
    let aligned = scale * (rotation * src) + translation;
    (aligned - dst).norm()
}

fn centroid(points: &[Vector3<f32>]) -> Vector3<f32> {
    let mut sum = Vector3::<f32>::zeros();
    for point in points {
        sum += point;
    }
    sum / points.len() as f32
}

fn valid_point(point: [f32; 3]) -> bool {
    point.iter().all(|value| value.is_finite()) && point.iter().any(|value| value.abs() > 1e-6)
}

#[cfg(test)]
mod tests {
    use super::{
        solve_iphone_stereo_transform, solve_rigid_transform, solve_similarity_transform, PointPair,
    };

    #[test]
    fn solve_rigid_transform_should_recover_translation() {
        let source = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.2, 0.3],
            [-0.1, 0.2, 0.1],
        ];
        let translation = [0.3, -0.2, 0.5];
        let pairs = source
            .iter()
            .copied()
            .map(|point| PointPair {
                source: point,
                target: [
                    point[0] + translation[0],
                    point[1] + translation[1],
                    point[2] + translation[2],
                ],
            })
            .collect::<Vec<_>>();

        let solved = solve_rigid_transform(&pairs, 123).expect("solve translation-only transform");

        assert!((solved.extrinsic_translation_m[0] - translation[0]).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[1] - translation[1]).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[2] - translation[2]).abs() < 1e-4);
        assert!(solved.rms_error_m < 1e-4);
    }

    #[test]
    fn solve_similarity_transform_should_recover_scale() {
        let source = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.2, 0.3],
            [-0.1, 0.2, 0.1],
        ];
        let scale = 2.2f32;
        let translation = [0.3, -0.2, 0.5];
        let pairs = source
            .iter()
            .copied()
            .map(|point| PointPair {
                source: point,
                target: [
                    point[0] * scale + translation[0],
                    point[1] * scale + translation[1],
                    point[2] * scale + translation[2],
                ],
            })
            .collect::<Vec<_>>();

        let solved = solve_similarity_transform(&pairs, 123).expect("solve similarity transform");

        assert!((solved.extrinsic_scale - scale).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[0] - translation[0]).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[1] - translation[1]).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[2] - translation[2]).abs() < 1e-4);
        assert!(solved.rms_error_m < 1e-4);
    }

    #[test]
    fn solve_iphone_stereo_transform_should_prefer_similarity_for_mild_depth_scale_bias() {
        let source = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.2, 0.3],
            [-0.1, 0.2, 0.1],
        ];
        let scale = 1.12f32;
        let translation = [0.18, -0.05, 0.32];
        let pairs = source
            .iter()
            .copied()
            .map(|point| PointPair {
                source: point,
                target: [
                    point[0] * scale + translation[0],
                    point[1] * scale + translation[1],
                    point[2] * scale + translation[2],
                ],
            })
            .collect::<Vec<_>>();

        let solved =
            solve_iphone_stereo_transform(&pairs, 456).expect("solve iphone-stereo transform");

        assert!((solved.extrinsic_scale - scale).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[0] - translation[0]).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[1] - translation[1]).abs() < 1e-4);
        assert!((solved.extrinsic_translation_m[2] - translation[2]).abs() < 1e-4);
        assert!(solved.rms_error_m < 1e-4);
    }
}
