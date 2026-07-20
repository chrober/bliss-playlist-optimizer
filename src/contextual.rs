// SPDX-License-Identifier: GPL-3.0-only

use std::fmt;

use bliss_mixer_core::scoring::{adaptive_distance, mean_feature_vector, select_adaptive_matrix};
use bliss_mixer_core::FeatureVector;
use ndarray::Array2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextualError {
    EmptySeeds,
    InvalidLearnedPercent(u16),
    MatrixUnavailable,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySeeds => formatter.write_str("adaptive scoring requires at least one seed"),
            Self::InvalidLearnedPercent(value) => write!(
                formatter,
                "learned matrix percentage {value} is outside 0..=100"
            ),
            Self::MatrixUnavailable => {
                formatter.write_str("no adaptive matrix was selected for the seed context")
            }
        }
    }
}

impl std::error::Error for ContextualError {}

pub fn adaptive_distance_from_seeds(
    seeds: &[FeatureVector],
    candidate: &FeatureVector,
    learned_matrix: &Array2<f32>,
    learned_percent: u16,
) -> Result<f64, ContextualError> {
    if seeds.is_empty() {
        return Err(ContextualError::EmptySeeds);
    }
    let selection = select_adaptive_matrix(seeds, Some(learned_matrix), learned_percent)
        .map_err(|_| ContextualError::InvalidLearnedPercent(learned_percent))?;
    let matrix = selection
        .matrix
        .as_ref()
        .ok_or(ContextualError::MatrixUnavailable)?;
    let mean = mean_feature_vector(seeds).ok_or(ContextualError::EmptySeeds)?;
    Ok(f64::from(adaptive_distance(&mean, candidate, matrix)))
}
