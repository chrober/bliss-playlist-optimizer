// SPDX-License-Identifier: GPL-3.0-only

//! Shared bounded path search between two immutable track anchors.
//!
//! This module deliberately does not know whether a path belongs to a live
//! destination action, one playlist gap, or one leg of a multi-gap plan.  The
//! caller owns those outer-planner decisions and supplies frozen context,
//! membership exclusions, candidate evidence, repeat policy, and one adjacent
//! distance function.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use rayon::prelude::*;

use crate::bridge::repeat_windows_safe_at;
use crate::route::RouteTrack;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchoredPathCandidate {
    pub track: usize,
    pub semantic_support: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnchoredPathSearchConfig {
    pub max_intermediates: usize,
    pub candidate_limit: usize,
    pub beam_width: usize,
    /// Number of complete alternatives retained for each intermediate count.
    /// Existing destination callers request one; multi-gap planners may ask
    /// for more so they can resolve cross-gap repeat and membership conflicts.
    pub alternatives_per_count: usize,
    pub variation_percent: u8,
    pub generation_seed: u64,
    pub artist_window: usize,
    pub album_window: usize,
    pub track_window: usize,
}

#[derive(Clone, Copy)]
pub struct AnchoredPathRequest<'a> {
    /// Route context ending at `left_anchor`. It is returned in every option.
    pub route_prefix: &'a [usize],
    /// Listening history used only for repeat checks, never returned as route
    /// members. Existing duplicates in this immutable history are tolerated.
    pub immutable_history: &'a [usize],
    /// Tracks already owned by the outer plan and therefore unavailable as
    /// generated intermediates.
    pub unavailable_tracks: &'a [usize],
    pub left_anchor: usize,
    pub right_anchor: usize,
    pub candidates: &'a [AnchoredPathCandidate],
    pub tracks: &'a [RouteTrack],
    pub config: AnchoredPathSearchConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnchoredPathSearchStats {
    pub evaluated_states: usize,
    pub retained_states: usize,
    pub structural_upper_bound: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnchoredPathOption {
    pub intermediates: Vec<usize>,
    pub route: Vec<usize>,
    pub transition_sum: f64,
    pub worst_transition: f64,
    pub stats: AnchoredPathSearchStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnchoredPathError {
    InvalidConfig(&'static str),
    InvalidRoute(&'static str),
    InvalidTrackIndex(usize),
}

impl fmt::Display for AnchoredPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) | Self::InvalidRoute(message) => {
                formatter.write_str(message)
            }
            Self::InvalidTrackIndex(track) => {
                write!(
                    formatter,
                    "anchored path references unknown track index {track}"
                )
            }
        }
    }
}

impl std::error::Error for AnchoredPathError {}

#[derive(Clone, Debug)]
struct PathState {
    intermediates: Vec<usize>,
    transition_sum: f64,
    worst_transition: f64,
    lower_sum: f64,
    lower_worst: f64,
    semantic_support: f64,
}

fn partial_path_order(left: &PathState, right: &PathState) -> std::cmp::Ordering {
    left.lower_worst
        .total_cmp(&right.lower_worst)
        .then_with(|| left.lower_sum.total_cmp(&right.lower_sum))
        .then_with(|| right.semantic_support.total_cmp(&left.semantic_support))
        .then_with(|| left.intermediates.cmp(&right.intermediates))
}

fn complete_path_order(left: &PathState, right: &PathState) -> std::cmp::Ordering {
    left.worst_transition
        .total_cmp(&right.worst_transition)
        .then_with(|| left.transition_sum.total_cmp(&right.transition_sum))
        .then_with(|| right.semantic_support.total_cmp(&left.semantic_support))
        .then_with(|| left.intermediates.cmp(&right.intermediates))
}

fn variation_key(seed: u64, route: &[usize], position: usize, candidate: usize) -> u64 {
    let mut value = seed
        ^ (candidate as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (position as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    for track in route {
        value ^= (*track as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = value.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn varied_pool_length(accepted: usize, variation_percent: u8) -> usize {
    if accepted == 0 || variation_percent == 0 {
        return accepted.min(1);
    }
    1 + (accepted.min(32).saturating_sub(1) * usize::from(variation_percent) / 100)
}

fn generated_positions_repeat_safe(
    route: &[usize],
    first_generated: usize,
    generated_count: usize,
    tracks: &[RouteTrack],
    config: AnchoredPathSearchConfig,
) -> bool {
    (first_generated..first_generated.saturating_add(generated_count)).all(|position| {
        repeat_windows_safe_at(
            route,
            tracks,
            position,
            config.artist_window,
            config.album_window,
        ) && (config.track_window == 0
            || route.iter().enumerate().all(|(other_position, track)| {
                other_position == position
                    || other_position.abs_diff(position) > config.track_window
                    || *track != route[position]
            }))
    })
}

fn validate_track_indices(
    indices: impl IntoIterator<Item = usize>,
    track_count: usize,
) -> Result<(), AnchoredPathError> {
    for track in indices {
        if track >= track_count {
            return Err(AnchoredPathError::InvalidTrackIndex(track));
        }
    }
    Ok(())
}

fn validate_request(request: &AnchoredPathRequest<'_>) -> Result<(), AnchoredPathError> {
    if request.route_prefix.last().copied() != Some(request.left_anchor) {
        return Err(AnchoredPathError::InvalidRoute(
            "anchored path route prefix must end at its left anchor",
        ));
    }
    if request.left_anchor == request.right_anchor {
        return Err(AnchoredPathError::InvalidRoute(
            "anchored path requires distinct left and right anchors",
        ));
    }
    if request.config.candidate_limit == 0 || request.config.beam_width == 0 {
        return Err(AnchoredPathError::InvalidConfig(
            "anchored path candidate and beam limits must be at least one",
        ));
    }
    if request.config.alternatives_per_count == 0 {
        return Err(AnchoredPathError::InvalidConfig(
            "anchored path alternatives per count must be at least one",
        ));
    }
    if request.config.variation_percent > 100 {
        return Err(AnchoredPathError::InvalidConfig(
            "anchored path variation percent must not exceed 100",
        ));
    }
    validate_track_indices(request.route_prefix.iter().copied(), request.tracks.len())?;
    validate_track_indices(
        request.immutable_history.iter().copied(),
        request.tracks.len(),
    )?;
    validate_track_indices(
        request.unavailable_tracks.iter().copied(),
        request.tracks.len(),
    )?;
    validate_track_indices(
        [request.left_anchor, request.right_anchor],
        request.tracks.len(),
    )?;
    validate_track_indices(
        request.candidates.iter().map(|candidate| candidate.track),
        request.tracks.len(),
    )
}

/// Finds complete paths from the left anchor to the right anchor for every
/// feasible intermediate count from zero through the configured maximum.
///
/// The search is pure: it never mutates outer-planner state. Existing callers
/// retain one option per count; future multi-gap planners can request several
/// alternatives and choose a globally compatible combination.
pub fn search_anchored_paths<F>(
    request: AnchoredPathRequest<'_>,
    adjacent_distance: F,
) -> Result<Vec<AnchoredPathOption>, AnchoredPathError>
where
    F: Fn(usize, usize) -> f64 + Sync + Copy,
{
    validate_request(&request)?;

    let unavailable = request
        .unavailable_tracks
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut candidates_by_track = BTreeMap::<usize, f64>::new();
    for candidate in request.candidates {
        if !unavailable.contains(&candidate.track) {
            candidates_by_track.insert(candidate.track, candidate.semantic_support);
        }
    }
    let candidates = candidates_by_track.keys().copied().collect::<Vec<_>>();
    let structural_upper_bound = request.config.max_intermediates.min(candidates.len());
    let direct = adjacent_distance(request.left_anchor, request.right_anchor);
    let mut direct_route = request.route_prefix.to_vec();
    direct_route.push(request.right_anchor);
    let mut options = vec![AnchoredPathOption {
        intermediates: Vec::new(),
        route: direct_route,
        transition_sum: direct,
        worst_transition: direct,
        stats: AnchoredPathSearchStats {
            evaluated_states: 1,
            retained_states: 1,
            structural_upper_bound,
        },
    }];

    let mut repeat_prefix = request.immutable_history.to_vec();
    repeat_prefix.extend(request.route_prefix.iter().copied());
    let first_generated = repeat_prefix.len();

    for requested in 1..=request.config.max_intermediates {
        let mut evaluated_states = 1usize;
        let mut retained_states = 1usize;
        let mut frontier = vec![PathState {
            intermediates: Vec::with_capacity(requested),
            transition_sum: 0.0,
            worst_transition: 0.0,
            lower_sum: direct,
            lower_worst: direct / (requested + 1) as f64,
            semantic_support: 0.0,
        }];

        for layer in 0..requested {
            let mut next = frontier
                .par_iter()
                .flat_map_iter(|state| {
                    let left = state
                        .intermediates
                        .last()
                        .copied()
                        .unwrap_or(request.left_anchor);
                    let remaining_edges = requested - layer;
                    let mut variants = candidates
                        .iter()
                        .copied()
                        .filter(|candidate| !state.intermediates.contains(candidate))
                        .filter_map(|candidate| {
                            let distance_to_destination = requested - layer;
                            let candidate_track = &request.tracks[candidate];
                            let destination_track = &request.tracks[request.right_anchor];
                            let conflicts_with_destination = (request.config.artist_window > 0
                                && distance_to_destination <= request.config.artist_window
                                && !candidate_track.artist_key.is_empty()
                                && candidate_track.artist_key == destination_track.artist_key)
                                || (request.config.album_window > 0
                                    && distance_to_destination <= request.config.album_window
                                    && !candidate_track.album_key.is_empty()
                                    && candidate_track.album_key == destination_track.album_key);
                            if conflicts_with_destination {
                                return None;
                            }

                            let mut partial = repeat_prefix.clone();
                            partial.extend(state.intermediates.iter().copied());
                            partial.push(candidate);
                            if !generated_positions_repeat_safe(
                                &partial,
                                partial.len() - 1,
                                1,
                                request.tracks,
                                request.config,
                            ) {
                                return None;
                            }

                            let edge = adjacent_distance(left, candidate);
                            let remaining_distance =
                                adjacent_distance(candidate, request.right_anchor);
                            let transition_sum = state.transition_sum + edge;
                            let worst_transition = state.worst_transition.max(edge);
                            let mut intermediates = state.intermediates.clone();
                            intermediates.push(candidate);
                            Some(PathState {
                                intermediates,
                                transition_sum,
                                worst_transition,
                                lower_sum: transition_sum + remaining_distance,
                                lower_worst: worst_transition
                                    .max(remaining_distance / remaining_edges as f64),
                                semantic_support: state.semantic_support
                                    + candidates_by_track.get(&candidate).copied().unwrap_or(0.0),
                            })
                        })
                        .collect::<Vec<_>>();
                    variants.sort_by(partial_path_order);
                    variants.truncate(request.config.candidate_limit);
                    variants
                })
                .collect::<Vec<_>>();
            evaluated_states = evaluated_states.saturating_add(next.len());
            next.sort_by(partial_path_order);
            next.dedup_by(|left, right| left.intermediates == right.intermediates);
            next.truncate(request.config.beam_width);
            retained_states = retained_states.saturating_add(next.len());
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }

        let mut complete = frontier
            .into_iter()
            .filter_map(|mut state| {
                let mut route = repeat_prefix.clone();
                route.extend(state.intermediates.iter().copied());
                route.push(request.right_anchor);
                if !generated_positions_repeat_safe(
                    &route,
                    first_generated,
                    requested,
                    request.tracks,
                    request.config,
                ) {
                    return None;
                }
                let final_edge =
                    adjacent_distance(*state.intermediates.last()?, request.right_anchor);
                state.transition_sum += final_edge;
                state.worst_transition = state.worst_transition.max(final_edge);
                state.lower_sum = state.transition_sum;
                state.lower_worst = state.worst_transition;
                Some(state)
            })
            .collect::<Vec<_>>();
        complete.sort_by(complete_path_order);
        let Some(best) = complete.first() else {
            continue;
        };
        let worst_limit = best.worst_transition + (best.worst_transition * 0.02).max(1.0e-9);
        let sum_limit = best.transition_sum + (best.transition_sum * 0.05).max(1.0e-9);
        let quality_band = complete
            .into_iter()
            .take_while(|candidate| candidate.worst_transition <= worst_limit)
            .filter(|candidate| candidate.transition_sum <= sum_limit)
            .collect::<Vec<_>>();
        let selection_pool =
            varied_pool_length(quality_band.len(), request.config.variation_percent)
                .max(request.config.alternatives_per_count)
                .min(quality_band.len());
        let mut selected = quality_band
            .into_iter()
            .take(selection_pool)
            .collect::<Vec<_>>();
        if request.config.variation_percent > 0 {
            selected.sort_by(|left, right| {
                variation_key(
                    request.config.generation_seed,
                    &left.intermediates,
                    requested,
                    request.right_anchor,
                )
                .cmp(&variation_key(
                    request.config.generation_seed,
                    &right.intermediates,
                    requested,
                    request.right_anchor,
                ))
                .then_with(|| complete_path_order(left, right))
            });
        }
        selected.truncate(request.config.alternatives_per_count);

        for state in selected {
            let mut route = request.route_prefix.to_vec();
            route.extend(state.intermediates.iter().copied());
            route.push(request.right_anchor);
            options.push(AnchoredPathOption {
                intermediates: state.intermediates,
                route,
                transition_sum: state.transition_sum,
                worst_transition: state.worst_transition,
                stats: AnchoredPathSearchStats {
                    evaluated_states,
                    retained_states,
                    structural_upper_bound,
                },
            });
        }
    }

    Ok(options)
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

    fn config(max_intermediates: usize) -> AnchoredPathSearchConfig {
        AnchoredPathSearchConfig {
            max_intermediates,
            candidate_limit: 16,
            beam_width: 64,
            alternatives_per_count: 1,
            variation_percent: 0,
            generation_seed: 42,
            artist_window: 1,
            album_window: 1,
            track_window: 8,
        }
    }

    #[test]
    fn returns_direct_and_bounded_intermediate_options() {
        let tracks = vec![
            track(0.0, "a", "a"),
            track(1.0, "b", "b"),
            track(2.0, "c", "c"),
            track(3.0, "d", "d"),
        ];
        let candidates = [
            AnchoredPathCandidate {
                track: 1,
                semantic_support: 0.0,
            },
            AnchoredPathCandidate {
                track: 2,
                semantic_support: 0.0,
            },
        ];
        let options = search_anchored_paths(
            AnchoredPathRequest {
                route_prefix: &[0],
                immutable_history: &[],
                unavailable_tracks: &[0, 3],
                left_anchor: 0,
                right_anchor: 3,
                candidates: &candidates,
                tracks: &tracks,
                config: config(2),
            },
            |left, right| left.abs_diff(right) as f64,
        )
        .unwrap();

        assert_eq!(options.len(), 3);
        assert_eq!(options[0].route, vec![0, 3]);
        assert_eq!(options[1].intermediates.len(), 1);
        assert_eq!(options[2].route, vec![0, 1, 2, 3]);
    }

    #[test]
    fn immutable_history_duplicates_are_tolerated_but_generated_repeats_are_not() {
        let tracks = vec![
            track(0.0, "a", "a"),
            track(1.0, "b", "b"),
            track(2.0, "c", "c"),
        ];
        let candidates = [AnchoredPathCandidate {
            track: 1,
            semantic_support: 0.0,
        }];
        let options = search_anchored_paths(
            AnchoredPathRequest {
                route_prefix: &[0],
                immutable_history: &[0, 0],
                unavailable_tracks: &[0, 2],
                left_anchor: 0,
                right_anchor: 2,
                candidates: &candidates,
                tracks: &tracks,
                config: AnchoredPathSearchConfig {
                    artist_window: 0,
                    album_window: 0,
                    ..config(1)
                },
            },
            |left, right| left.abs_diff(right) as f64,
        )
        .unwrap();
        assert_eq!(options[1].route, vec![0, 1, 2]);

        let repeated_generated_track = search_anchored_paths(
            AnchoredPathRequest {
                route_prefix: &[0],
                immutable_history: &[1, 1],
                unavailable_tracks: &[0, 2],
                left_anchor: 0,
                right_anchor: 2,
                candidates: &candidates,
                tracks: &tracks,
                config: AnchoredPathSearchConfig {
                    artist_window: 0,
                    album_window: 0,
                    ..config(1)
                },
            },
            |left, right| left.abs_diff(right) as f64,
        )
        .unwrap();
        assert_eq!(repeated_generated_track.len(), 1);
    }

    #[test]
    fn can_retain_multiple_alternatives_for_an_outer_multi_gap_planner() {
        let tracks = vec![
            track(0.0, "a", "a"),
            track(1.0, "b", "b"),
            track(1.0, "c", "c"),
            track(2.0, "d", "d"),
        ];
        let candidates = [
            AnchoredPathCandidate {
                track: 1,
                semantic_support: 0.0,
            },
            AnchoredPathCandidate {
                track: 2,
                semantic_support: 0.0,
            },
        ];
        let options = search_anchored_paths(
            AnchoredPathRequest {
                route_prefix: &[0],
                immutable_history: &[],
                unavailable_tracks: &[0, 3],
                left_anchor: 0,
                right_anchor: 3,
                candidates: &candidates,
                tracks: &tracks,
                config: AnchoredPathSearchConfig {
                    alternatives_per_count: 2,
                    ..config(1)
                },
            },
            |left, right| (tracks[left].features[0] - tracks[right].features[0]).abs() as f64,
        )
        .unwrap();

        let one_intermediate = options
            .iter()
            .filter(|option| option.intermediates.len() == 1)
            .collect::<Vec<_>>();
        assert_eq!(one_intermediate.len(), 2);
        assert_ne!(
            one_intermediate[0].intermediates,
            one_intermediate[1].intermediates
        );
    }
}
