// SPDX-License-Identifier: GPL-3.0-only

use std::fmt;

use ndarray::Array2;
use rayon::prelude::*;

use crate::contextual::{adaptive_distance_from_seeds, ContextualError};
use crate::route::RouteTrack;

pub const DEFAULT_MAX_LEG_PERCENTILE: f64 = 0.70;
pub const DEFAULT_MAX_DETOUR_PERCENTILE: f64 = 1.30;

#[derive(Clone, Debug)]
pub struct BridgeConfig {
    pub seed_limit: usize,
    pub learned_percent: u16,
    pub artist_window: usize,
    pub album_window: usize,
    pub max_leg_percentile: f64,
    pub max_detour_percentile: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrozenReference {
    distances: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BridgeGap {
    pub position: usize,
    pub direct_distance: f64,
    pub direct_percentile: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BridgeCandidateEvaluation {
    pub candidate: usize,
    pub left_distance: f64,
    pub right_distance: f64,
    pub left_percentile: f64,
    pub right_percentile: f64,
    pub max_percentile: f64,
    pub detour_percentile: f64,
    pub repeat_safe: bool,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BridgeError {
    InvalidGap,
    InvalidTrackIndex(usize),
    EmptyReference,
    Scoring(ContextualError),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGap => formatter.write_str("bridge position is not an internal gap"),
            Self::InvalidTrackIndex(index) => write!(formatter, "invalid track index {index}"),
            Self::EmptyReference => {
                formatter.write_str("frozen bridge reference distribution is empty")
            }
            Self::Scoring(error) => write!(formatter, "adaptive bridge scoring failed: {error}"),
        }
    }
}

fn validate_indices(indices: &[usize], track_count: usize) -> Result<(), BridgeError> {
    if let Some(index) = indices.iter().find(|index| **index >= track_count) {
        return Err(BridgeError::InvalidTrackIndex(*index));
    }
    Ok(())
}

impl std::error::Error for BridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scoring(error) => Some(error),
            _ => None,
        }
    }
}

impl FrozenReference {
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    pub fn percentile(&self, value: f64) -> Result<f64, BridgeError> {
        if self.distances.is_empty() {
            return Err(BridgeError::EmptyReference);
        }
        let below = self
            .distances
            .partition_point(|distance| distance.total_cmp(&value).is_lt());
        Ok(below as f64 / self.distances.len().saturating_sub(1).max(1) as f64)
    }
}

fn contextual_distance(
    prefix: &[usize],
    candidate: usize,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
) -> Result<f64, BridgeError> {
    validate_indices(prefix, tracks.len())?;
    validate_indices(&[candidate], tracks.len())?;
    let seed_start = prefix.len().saturating_sub(config.seed_limit);
    let seeds = prefix[seed_start..]
        .iter()
        .map(|index| tracks[*index].features)
        .collect::<Vec<_>>();
    adaptive_distance_from_seeds(
        &seeds,
        &tracks[candidate].features,
        learned_matrix,
        config.learned_percent,
    )
    .map_err(BridgeError::Scoring)
}

pub fn build_frozen_reference(
    route: &[usize],
    original_candidates: &[usize],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
) -> Result<FrozenReference, BridgeError> {
    validate_indices(route, tracks.len())?;
    validate_indices(original_candidates, tracks.len())?;
    if route.len() < 2 || config.seed_limit == 0 {
        return Err(BridgeError::EmptyReference);
    }
    let chunks = (1..route.len())
        .into_par_iter()
        .map(|position| {
            let seed_start = position.saturating_sub(config.seed_limit);
            let seed_indexes = &route[seed_start..position];
            original_candidates
                .iter()
                .copied()
                .filter(|candidate| !seed_indexes.contains(candidate))
                .map(|candidate| {
                    contextual_distance(
                        &route[..position],
                        candidate,
                        tracks,
                        learned_matrix,
                        config,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Vec<_>>();
    let mut distances = Vec::new();
    for chunk in chunks {
        distances.extend(chunk?);
    }
    if distances.is_empty() {
        return Err(BridgeError::EmptyReference);
    }
    distances.sort_by(f64::total_cmp);
    Ok(FrozenReference { distances })
}

pub fn evaluate_gap(
    route: &[usize],
    position: usize,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<BridgeGap, BridgeError> {
    validate_indices(route, tracks.len())?;
    if position == 0 || position >= route.len() {
        return Err(BridgeError::InvalidGap);
    }
    let direct_distance = contextual_distance(
        &route[..position],
        route[position],
        tracks,
        learned_matrix,
        config,
    )?;
    Ok(BridgeGap {
        position,
        direct_distance,
        direct_percentile: reference.percentile(direct_distance)?,
    })
}

fn repeat_safe(route: &[usize], tracks: &[RouteTrack], config: &BridgeConfig) -> bool {
    for right in 0..route.len() {
        for left in 0..right {
            let distance = right - left;
            let left_track = &tracks[route[left]];
            let right_track = &tracks[route[right]];
            if config.artist_window > 0
                && distance <= config.artist_window
                && !right_track.artist_key.is_empty()
                && left_track.artist_key == right_track.artist_key
            {
                return false;
            }
            if config.album_window > 0
                && distance <= config.album_window
                && !right_track.album_key.is_empty()
                && left_track.album_key == right_track.album_key
            {
                return false;
            }
        }
    }
    true
}

pub fn evaluate_candidate(
    route: &[usize],
    position: usize,
    candidate: usize,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<BridgeCandidateEvaluation, BridgeError> {
    validate_indices(route, tracks.len())?;
    validate_indices(&[candidate], tracks.len())?;
    if position == 0 || position >= route.len() {
        return Err(BridgeError::InvalidGap);
    }
    let mut tentative = route.to_vec();
    tentative.insert(position, candidate);
    let left_distance = contextual_distance(
        &tentative[..position],
        candidate,
        tracks,
        learned_matrix,
        config,
    )?;
    let right_distance = contextual_distance(
        &tentative[..=position],
        tentative[position + 1],
        tracks,
        learned_matrix,
        config,
    )?;
    let left_percentile = reference.percentile(left_distance)?;
    let right_percentile = reference.percentile(right_distance)?;
    let max_percentile = left_percentile.max(right_percentile);
    let detour_percentile = left_percentile + right_percentile;
    let repeat_safe = !route.contains(&candidate) && repeat_safe(&tentative, tracks, config);
    let accepted = repeat_safe
        && max_percentile <= config.max_leg_percentile
        && detour_percentile <= config.max_detour_percentile;
    Ok(BridgeCandidateEvaluation {
        candidate,
        left_distance,
        right_distance,
        left_percentile,
        right_percentile,
        max_percentile,
        detour_percentile,
        repeat_safe,
        accepted,
    })
}

pub fn rank_candidates(
    route: &[usize],
    position: usize,
    candidates: &[usize],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<Vec<BridgeCandidateEvaluation>, BridgeError> {
    let attempts = candidates
        .par_iter()
        .map(|candidate| {
            evaluate_candidate(
                route,
                position,
                *candidate,
                tracks,
                learned_matrix,
                config,
                reference,
            )
        })
        .collect::<Vec<_>>();
    let mut evaluations = attempts.into_iter().collect::<Result<Vec<_>, _>>()?;
    evaluations.sort_by(|left, right| {
        right
            .accepted
            .cmp(&left.accepted)
            .then_with(|| left.max_percentile.total_cmp(&right.max_percentile))
            .then_with(|| left.detour_percentile.total_cmp(&right.detour_percentile))
            .then_with(|| left.candidate.cmp(&right.candidate))
    });
    Ok(evaluations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(value: f32, artist: &str, album: &str) -> RouteTrack {
        RouteTrack {
            features: std::array::from_fn(|index| value + index as f32 / 100.0),
            artist_key: artist.to_owned(),
            album_key: album.to_owned(),
        }
    }

    fn config() -> BridgeConfig {
        BridgeConfig {
            seed_limit: 2,
            learned_percent: 20,
            artist_window: 1,
            album_window: 1,
            max_leg_percentile: DEFAULT_MAX_LEG_PERCENTILE,
            max_detour_percentile: DEFAULT_MAX_DETOUR_PERCENTILE,
        }
    }

    fn tracks() -> Vec<RouteTrack> {
        vec![
            track(0.0, "a", "album-a"),
            track(1.0, "b", "album-b"),
            track(2.0, "c", "album-c"),
            track(1.2, "a", "album-d"),
            track(4.0, "e", "album-e"),
            track(9.0, "f", "album-f"),
        ]
    }

    #[test]
    fn frozen_reference_and_two_sided_scoring_match_the_bridge_contract() {
        let tracks = tracks();
        let route = [0, 2, 4];
        let matrix = Array2::eye(23);
        let reference =
            build_frozen_reference(&route, &route, &tracks, &matrix, &config()).unwrap();
        assert_eq!(reference.len(), 3);

        let direct = evaluate_gap(&route, 1, &tracks, &matrix, &config(), &reference).unwrap();
        let bridge =
            evaluate_candidate(&route, 1, 1, &tracks, &matrix, &config(), &reference).unwrap();
        assert!(bridge.accepted);
        assert_ne!(bridge.right_distance, direct.direct_distance);

        let repeated =
            evaluate_candidate(&route, 1, 3, &tracks, &matrix, &config(), &reference).unwrap();
        assert!(!repeated.repeat_safe);
        assert!(!repeated.accepted);

        let existing =
            evaluate_candidate(&route, 1, 0, &tracks, &matrix, &config(), &reference).unwrap();
        assert!(!existing.repeat_safe);
    }

    #[test]
    fn candidate_ranking_is_identical_with_one_and_four_workers() {
        let tracks = tracks();
        let route = [0, 2, 4];
        let matrix = Array2::eye(23);
        let reference =
            build_frozen_reference(&route, &route, &tracks, &matrix, &config()).unwrap();
        let candidates = [5, 3, 1, 0];
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                rank_candidates(
                    &route,
                    1,
                    &candidates,
                    &tracks,
                    &matrix,
                    &config(),
                    &reference,
                )
            })
            .unwrap();
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                rank_candidates(
                    &route,
                    1,
                    &candidates,
                    &tracks,
                    &matrix,
                    &config(),
                    &reference,
                )
            })
            .unwrap();
        assert_eq!(one, four);
        assert_eq!(one[0].candidate, 1);
        assert!(one[0].accepted);
    }

    #[test]
    fn invalid_public_indexes_fail_without_panicking() {
        let tracks = tracks();
        assert_eq!(
            evaluate_gap(
                &[0, 99],
                1,
                &tracks,
                &Array2::eye(23),
                &config(),
                &FrozenReference {
                    distances: vec![1.0],
                },
            ),
            Err(BridgeError::InvalidTrackIndex(99))
        );
    }
}
