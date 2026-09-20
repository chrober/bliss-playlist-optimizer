// SPDX-License-Identifier: GPL-3.0-only

use std::fmt;

use bliss_mixer_core::FeatureVector;
use ndarray::Array2;
use rayon::prelude::*;

use crate::contextual::{
    adaptive_distance_from_seeds, adaptive_distance_with_matrix, prepare_adaptive_context,
    ContextualError, PreparedAdaptiveContext,
};
use crate::route::RouteTrack;

pub const DEFAULT_MAX_LEG_PERCENTILE: f64 = 0.70;
pub const DEFAULT_MAX_DETOUR_PERCENTILE: f64 = 1.30;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GapContextMode {
    #[default]
    Rolling,
    Frozen,
}

#[derive(Clone, Debug)]
pub struct BridgeConfig {
    pub seed_limit: usize,
    pub learned_percent: u16,
    pub artist_window: usize,
    pub album_window: usize,
    pub max_leg_percentile: f64,
    pub max_detour_percentile: f64,
    pub gap_context_mode: GapContextMode,
}

pub struct ShortlistScoringContext<'a> {
    pub tracks: &'a [RouteTrack],
    pub learned_matrix: &'a Array2<f32>,
    pub config: &'a BridgeConfig,
    pub reference: &'a FrozenReference,
}

#[derive(Clone, Copy)]
pub(crate) struct CandidateScoringContext<'a> {
    pub learned_matrix: &'a Array2<f32>,
    pub config: &'a BridgeConfig,
    pub reference: &'a FrozenReference,
    pub frozen_matrix: Option<&'a Array2<f32>>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointSlot {
    Opening,
    Closing,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EndpointCandidateEvaluation {
    pub candidate: usize,
    pub distance: f64,
    pub percentile: f64,
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
    pub fn from_distances(mut distances: Vec<f64>) -> Result<Self, BridgeError> {
        if distances.is_empty() {
            return Err(BridgeError::EmptyReference);
        }
        distances.sort_by(f64::total_cmp);
        Ok(Self { distances })
    }

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

fn recent_context(
    prefix: &[usize],
    tracks: &[RouteTrack],
    config: &BridgeConfig,
) -> Result<Vec<FeatureVector>, BridgeError> {
    validate_indices(prefix, tracks.len())?;
    let seed_start = prefix.len().saturating_sub(config.seed_limit);
    Ok(prefix[seed_start..]
        .iter()
        .map(|index| tracks[*index].features)
        .collect())
}

pub(crate) fn contextual_distance(
    prefix: &[usize],
    candidate: usize,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
) -> Result<f64, BridgeError> {
    validate_indices(&[candidate], tracks.len())?;
    adaptive_distance_from_seeds(
        &recent_context(prefix, tracks, config)?,
        &tracks[candidate].features,
        learned_matrix,
        config.learned_percent,
    )
    .map_err(BridgeError::Scoring)
}

pub(crate) fn contextual_distance_with_matrix(
    prefix: &[usize],
    candidate: usize,
    tracks: &[RouteTrack],
    matrix: &Array2<f32>,
    config: &BridgeConfig,
) -> Result<f64, BridgeError> {
    validate_indices(&[candidate], tracks.len())?;
    adaptive_distance_with_matrix(
        &recent_context(prefix, tracks, config)?,
        &tracks[candidate].features,
        matrix,
    )
    .map_err(BridgeError::Scoring)
}

fn prepared_context(
    prefix: &[usize],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
) -> Result<PreparedAdaptiveContext, BridgeError> {
    prepare_adaptive_context(
        &recent_context(prefix, tracks, config)?,
        learned_matrix,
        config.learned_percent,
    )
    .map_err(BridgeError::Scoring)
}

pub fn gap_context_matrix(
    route: &[usize],
    position: usize,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
) -> Result<Option<Array2<f32>>, BridgeError> {
    if config.gap_context_mode == GapContextMode::Rolling {
        return Ok(None);
    }
    if position == 0 || position >= route.len() {
        return Err(BridgeError::InvalidGap);
    }
    Ok(Some(
        prepared_context(&route[..position], tracks, learned_matrix, config)?
            .matrix()
            .clone(),
    ))
}

pub fn shortlist_candidates(
    route: &[usize],
    position: usize,
    candidates: &[usize],
    limit: usize,
    context: ShortlistScoringContext<'_>,
) -> Result<Vec<usize>, BridgeError> {
    validate_indices(route, context.tracks.len())?;
    validate_indices(candidates, context.tracks.len())?;
    if position == 0 || position >= route.len() {
        return Err(BridgeError::InvalidGap);
    }
    if limit == 0 || candidates.is_empty() {
        return Ok(Vec::new());
    }
    if candidates.len() <= limit {
        return Ok(candidates.to_vec());
    }

    let mut ranked = rank_candidates(
        route,
        position,
        candidates,
        context.tracks,
        context.learned_matrix,
        context.config,
        context.reference,
    )?;
    ranked.truncate(limit);
    Ok(ranked
        .into_iter()
        .map(|evaluation| evaluation.candidate)
        .collect())
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
    FrozenReference::from_distances(distances)
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

pub(crate) fn repeat_windows_safe_at(
    route: &[usize],
    tracks: &[RouteTrack],
    inserted_position: usize,
    artist_window: usize,
    album_window: usize,
) -> bool {
    let inserted = &tracks[route[inserted_position]];
    for (position, track_index) in route.iter().enumerate() {
        if position == inserted_position {
            continue;
        }
        let distance = position.abs_diff(inserted_position);
        let other = &tracks[*track_index];
        if artist_window > 0
            && distance <= artist_window
            && !inserted.artist_key.is_empty()
            && inserted.artist_key == other.artist_key
        {
            return false;
        }
        if album_window > 0
            && distance <= album_window
            && !inserted.album_key.is_empty()
            && inserted.album_key == other.album_key
        {
            return false;
        }
    }
    true
}

pub(crate) fn repeat_safe_at(
    route: &[usize],
    tracks: &[RouteTrack],
    config: &BridgeConfig,
    inserted_position: usize,
) -> bool {
    repeat_windows_safe_at(
        route,
        tracks,
        inserted_position,
        config.artist_window,
        config.album_window,
    )
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
    evaluate_candidate_in_context(
        route,
        position,
        candidate,
        tracks,
        CandidateScoringContext {
            learned_matrix,
            config,
            reference,
            frozen_matrix: None,
        },
    )
}

pub(crate) fn evaluate_candidate_in_context(
    route: &[usize],
    position: usize,
    candidate: usize,
    tracks: &[RouteTrack],
    scoring: CandidateScoringContext<'_>,
) -> Result<BridgeCandidateEvaluation, BridgeError> {
    let CandidateScoringContext {
        learned_matrix,
        config,
        reference,
        frozen_matrix,
    } = scoring;
    validate_indices(route, tracks.len())?;
    validate_indices(&[candidate], tracks.len())?;
    if position == 0 || position >= route.len() {
        return Err(BridgeError::InvalidGap);
    }
    let mut tentative = route.to_vec();
    tentative.insert(position, candidate);
    let distance = |prefix: &[usize], candidate| match frozen_matrix {
        Some(matrix) => contextual_distance_with_matrix(prefix, candidate, tracks, matrix, config),
        None => contextual_distance(prefix, candidate, tracks, learned_matrix, config),
    };
    let left_distance = distance(&tentative[..position], candidate)?;
    let right_distance = distance(&tentative[..=position], tentative[position + 1])?;
    let left_percentile = reference.percentile(left_distance)?;
    let right_percentile = reference.percentile(right_distance)?;
    let max_percentile = left_percentile.max(right_percentile);
    let detour_percentile = left_percentile + right_percentile;
    let repeat_safe =
        !route.contains(&candidate) && repeat_safe_at(&tentative, tracks, config, position);
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
    rank_candidates_in_context(
        route,
        position,
        candidates,
        tracks,
        CandidateScoringContext {
            learned_matrix,
            config,
            reference,
            frozen_matrix: None,
        },
    )
}

pub(crate) fn rank_candidates_in_context(
    route: &[usize],
    position: usize,
    candidates: &[usize],
    tracks: &[RouteTrack],
    scoring: CandidateScoringContext<'_>,
) -> Result<Vec<BridgeCandidateEvaluation>, BridgeError> {
    let CandidateScoringContext {
        learned_matrix,
        config,
        reference,
        frozen_matrix,
    } = scoring;
    validate_indices(route, tracks.len())?;
    validate_indices(candidates, tracks.len())?;
    if position == 0 || position >= route.len() {
        return Err(BridgeError::InvalidGap);
    }
    let left_context = frozen_matrix
        .is_none()
        .then(|| prepared_context(&route[..position], tracks, learned_matrix, config))
        .transpose()?;
    let attempts = candidates
        .par_iter()
        .map(|candidate| {
            let mut tentative = route.to_vec();
            tentative.insert(position, *candidate);
            let left_distance = match frozen_matrix {
                Some(matrix) => contextual_distance_with_matrix(
                    &tentative[..position],
                    *candidate,
                    tracks,
                    matrix,
                    config,
                )?,
                None => left_context
                    .as_ref()
                    .expect("rolling scoring prepares a left context")
                    .distance_to(&tracks[*candidate].features),
            };
            let right_distance = match frozen_matrix {
                Some(matrix) => contextual_distance_with_matrix(
                    &tentative[..=position],
                    tentative[position + 1],
                    tracks,
                    matrix,
                    config,
                )?,
                None => contextual_distance(
                    &tentative[..=position],
                    tentative[position + 1],
                    tracks,
                    learned_matrix,
                    config,
                )?,
            };
            let left_percentile = reference.percentile(left_distance)?;
            let right_percentile = reference.percentile(right_distance)?;
            let max_percentile = left_percentile.max(right_percentile);
            let detour_percentile = left_percentile + right_percentile;
            let repeat_safe =
                !route.contains(candidate) && repeat_safe_at(&tentative, tracks, config, position);
            let accepted = repeat_safe
                && max_percentile <= config.max_leg_percentile
                && detour_percentile <= config.max_detour_percentile;
            Ok(BridgeCandidateEvaluation {
                candidate: *candidate,
                left_distance,
                right_distance,
                left_percentile,
                right_percentile,
                max_percentile,
                detour_percentile,
                repeat_safe,
                accepted,
            })
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

pub fn evaluate_endpoint_candidate(
    route: &[usize],
    slot: EndpointSlot,
    candidate: usize,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<EndpointCandidateEvaluation, BridgeError> {
    validate_indices(route, tracks.len())?;
    validate_indices(&[candidate], tracks.len())?;
    if route.is_empty() {
        return Err(BridgeError::InvalidGap);
    }
    let mut tentative = route.to_vec();
    let distance = match slot {
        EndpointSlot::Opening => {
            tentative.insert(0, candidate);
            contextual_distance(
                &tentative[..1],
                tentative[1],
                tracks,
                learned_matrix,
                config,
            )?
        }
        EndpointSlot::Closing => {
            let distance =
                contextual_distance(&tentative, candidate, tracks, learned_matrix, config)?;
            tentative.push(candidate);
            distance
        }
    };
    let percentile = reference.percentile(distance)?;
    let inserted_position = match slot {
        EndpointSlot::Opening => 0,
        EndpointSlot::Closing => tentative.len().saturating_sub(1),
    };
    let repeat_safe = !route.contains(&candidate)
        && repeat_safe_at(&tentative, tracks, config, inserted_position);
    Ok(EndpointCandidateEvaluation {
        candidate,
        distance,
        percentile,
        repeat_safe,
        accepted: repeat_safe && percentile <= config.max_leg_percentile,
    })
}

pub fn rank_endpoint_candidates(
    route: &[usize],
    slot: EndpointSlot,
    candidates: &[usize],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<Vec<EndpointCandidateEvaluation>, BridgeError> {
    let attempts = candidates
        .par_iter()
        .map(|candidate| {
            evaluate_endpoint_candidate(
                route,
                slot,
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
            .then_with(|| left.percentile.total_cmp(&right.percentile))
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
            gap_context_mode: GapContextMode::Rolling,
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
    fn frozen_gap_context_reuses_one_matrix_while_the_rolling_mode_reselects_it() {
        let mut tracks = tracks();
        tracks[0].features = std::array::from_fn(|index| if index % 2 == 0 { 0.2 } else { 2.0 });
        tracks[2].features = std::array::from_fn(|index| if index % 3 == 0 { 3.0 } else { 0.4 });
        tracks[1].features = std::array::from_fn(|index| if index % 5 == 0 { 1.5 } else { 4.0 });
        tracks[4].features = std::array::from_fn(|index| index as f32 / 7.0);
        let route = [0, 2, 4];
        let matrix = Array2::eye(23);
        let reference = FrozenReference::from_distances(vec![0.0, 100.0]).unwrap();
        let rolling_config = BridgeConfig {
            seed_limit: 3,
            learned_percent: 20,
            artist_window: 0,
            album_window: 0,
            max_leg_percentile: 1.0,
            max_detour_percentile: 2.0,
            gap_context_mode: GapContextMode::Rolling,
        };
        let frozen_config = BridgeConfig {
            gap_context_mode: GapContextMode::Frozen,
            ..rolling_config.clone()
        };
        let frozen_matrix = gap_context_matrix(&route, 2, &tracks, &matrix, &frozen_config)
            .unwrap()
            .unwrap();

        let rolling = evaluate_candidate_in_context(
            &route,
            2,
            1,
            &tracks,
            CandidateScoringContext {
                learned_matrix: &matrix,
                config: &rolling_config,
                reference: &reference,
                frozen_matrix: None,
            },
        )
        .unwrap();
        let frozen = evaluate_candidate_in_context(
            &route,
            2,
            1,
            &tracks,
            CandidateScoringContext {
                learned_matrix: &matrix,
                config: &frozen_config,
                reference: &reference,
                frozen_matrix: Some(&frozen_matrix),
            },
        )
        .unwrap();

        assert!((rolling.left_distance - frozen.left_distance).abs() < 1e-6);
        assert!((rolling.right_distance - frozen.right_distance).abs() > 1e-6);
        assert!(
            gap_context_matrix(&route, 2, &tracks, &matrix, &rolling_config)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn destination_candidate_cannot_use_explicit_destination_repeat_exemption() {
        let tracks = vec![
            track(0.0, "same-artist", "source-album"),
            track(2.0, "same-artist", "destination-album"),
            track(1.0, "other-artist", "bridge-album"),
            track(1.1, "same-artist", "other-album"),
            track(1.2, "album-conflict-artist", "destination-album"),
        ];
        let route = [0, 1];
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 1,
            learned_percent: 20,
            artist_window: 5,
            album_window: 5,
            max_leg_percentile: 1.0,
            max_detour_percentile: 2.0,
            gap_context_mode: GapContextMode::Rolling,
        };
        let reference =
            build_frozen_reference(&route, &[0, 1, 2, 3, 4], &tracks, &matrix, &config).unwrap();

        let permitted =
            evaluate_candidate(&route, 1, 2, &tracks, &matrix, &config, &reference).unwrap();
        assert!(permitted.repeat_safe);

        let conflicts_with_destination =
            evaluate_candidate(&route, 1, 3, &tracks, &matrix, &config, &reference).unwrap();
        assert!(!conflicts_with_destination.repeat_safe);
        assert!(!conflicts_with_destination.accepted);
        let conflicts_with_destination_album =
            evaluate_candidate(&route, 1, 4, &tracks, &matrix, &config, &reference).unwrap();
        assert!(!conflicts_with_destination_album.repeat_safe);
        assert!(!conflicts_with_destination_album.accepted);
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
    fn acoustic_shortlist_is_worker_deterministic_and_retains_the_strict_winner() {
        let tracks = tracks();
        let route = [0, 2, 4];
        let matrix = Array2::eye(23);
        let reference =
            build_frozen_reference(&route, &route, &tracks, &matrix, &config()).unwrap();
        let candidates = [5, 3, 1];
        let exhaustive = rank_candidates(
            &route,
            1,
            &candidates,
            &tracks,
            &matrix,
            &config(),
            &reference,
        )
        .unwrap();
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                shortlist_candidates(
                    &route,
                    1,
                    &candidates,
                    2,
                    ShortlistScoringContext {
                        tracks: &tracks,
                        learned_matrix: &matrix,
                        config: &config(),
                        reference: &reference,
                    },
                )
            })
            .unwrap();
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                shortlist_candidates(
                    &route,
                    1,
                    &candidates,
                    2,
                    ShortlistScoringContext {
                        tracks: &tracks,
                        learned_matrix: &matrix,
                        config: &config(),
                        reference: &reference,
                    },
                )
            })
            .unwrap();
        assert_eq!(one, four);
        assert!(one.contains(&exhaustive[0].candidate));
    }

    #[test]
    fn acoustic_shortlist_has_high_strict_winner_recall_on_contextual_corpus() {
        let tracks = (0..768)
            .map(|track_index| RouteTrack {
                features: std::array::from_fn(|feature_index| {
                    let x = track_index as f32 * 0.017 + feature_index as f32 * 0.113;
                    x.sin() + (x * 0.37).cos() * 0.4 + track_index as f32 / 4096.0
                }),
                artist_key: format!("artist-{track_index}"),
                album_key: format!("album-{track_index}"),
            })
            .collect::<Vec<_>>();
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 3,
            learned_percent: 20,
            artist_window: 0,
            album_window: 0,
            max_leg_percentile: 1.0,
            max_detour_percentile: 2.0,
            gap_context_mode: GapContextMode::Rolling,
        };
        let candidates = (64..tracks.len()).collect::<Vec<_>>();
        let mut retained = 0usize;
        let trials = 16usize;
        for trial in 0..trials {
            let base = trial * 3;
            let route = [base, base + 7, base + 19, base + 31];
            let position = 1 + trial % 3;
            let reference =
                build_frozen_reference(&route, &route, &tracks, &matrix, &config).unwrap();
            let strict = rank_candidates(
                &route,
                position,
                &candidates,
                &tracks,
                &matrix,
                &config,
                &reference,
            )
            .unwrap();
            let shortlist = shortlist_candidates(
                &route,
                position,
                &candidates,
                128,
                ShortlistScoringContext {
                    tracks: &tracks,
                    learned_matrix: &matrix,
                    config: &config,
                    reference: &reference,
                },
            )
            .unwrap();
            retained += usize::from(shortlist.contains(&strict[0].candidate));
        }
        assert!(
            retained >= 15,
            "strict winner retained in {retained} of {trials} trials"
        );
    }

    #[test]
    fn endpoint_scoring_is_one_sided_repeat_safe_and_worker_deterministic() {
        let tracks = tracks();
        let route = [0, 2, 4];
        let matrix = Array2::eye(23);
        let reference =
            build_frozen_reference(&route, &route, &tracks, &matrix, &config()).unwrap();
        let candidates = [5, 3, 1, 0];

        let opening = evaluate_endpoint_candidate(
            &route,
            EndpointSlot::Opening,
            1,
            &tracks,
            &matrix,
            &config(),
            &reference,
        )
        .unwrap();
        let closing = evaluate_endpoint_candidate(
            &route,
            EndpointSlot::Closing,
            1,
            &tracks,
            &matrix,
            &config(),
            &reference,
        )
        .unwrap();
        assert!(opening.accepted);
        assert!(closing.accepted);
        assert_ne!(opening.distance, closing.distance);

        let existing = evaluate_endpoint_candidate(
            &route,
            EndpointSlot::Opening,
            0,
            &tracks,
            &matrix,
            &config(),
            &reference,
        )
        .unwrap();
        assert!(!existing.repeat_safe);
        assert!(!existing.accepted);

        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                rank_endpoint_candidates(
                    &route,
                    EndpointSlot::Opening,
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
                rank_endpoint_candidates(
                    &route,
                    EndpointSlot::Opening,
                    &candidates,
                    &tracks,
                    &matrix,
                    &config(),
                    &reference,
                )
            })
            .unwrap();
        assert_eq!(one, four);
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
