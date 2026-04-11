//! WiFi-DensePose Sensing Server library.
//!
//! This crate provides:
//! - Vital sign detection from WiFi CSI amplitude data
//! - RVF (RuVector Format) binary container for model weights

pub mod dataset;
pub mod embedding;
pub mod graph_transformer;
pub mod pose_head;
pub mod rvf_container;
pub mod rvf_pipeline;
pub mod sona;
pub mod sparse_inference;
pub mod trainer;
pub mod vital_signs;
