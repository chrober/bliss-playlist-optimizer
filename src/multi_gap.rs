// SPDX-License-Identifier: GPL-3.0-only

//! Global planning across several immutable source-track gaps.

use std::collections::HashSet;
use std::fmt;

use rayon::prelude::*;

use crate::anchored_path::{
    search_anchored_paths, AnchoredPathCandidate, AnchoredPathError, AnchoredPathRequest,
    AnchoredPathScorer, AnchoredPathSearchConfig,
};
use crate::bridge::repeat_windows_safe_at;
use crate::route::RouteTrack;

pub const MAX_TRACKS_PER_GAP: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct FixedGap<'a> {
    pub left: usize,
    pub right: usize,
    pub candidates: &'a [AnchoredPathCandidate],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedGapPlanConfig {
    pub tracks_per_gap: usize,
    pub candidate_limit: usize,
    pub beam_width: usize,
    pub alternatives_per_gap: usize,
    pub variation_percent: u8,
    pub generation_seed: u64,
    pub artist_window: usize,
    pub album_window: usize,
    pub track_window: usize,
}

#[derive(Clone, Copy)]
pub struct FixedGapPlanRequest<'a> {
    pub source_route: &'a [usize],
    pub immutable_history: &'a [usize],
    pub gaps: &'a [FixedGap<'a>],
    pub tracks: &'a [RouteTrack],
    pub config: FixedGapPlanConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatWindowRepairConfig {
    pub max_added_tracks: usize,
    pub candidate_limit: usize,
    pub beam_width: usize,
    pub alternatives_per_gap: usize,
    pub variation_percent: u8,
    pub generation_seed: u64,
    pub artist_window: usize,
    pub album_window: usize,
    pub track_window: usize,
}

#[derive(Clone, Copy)]
pub struct RepeatWindowRepairRequest<'a> {
    pub source_route: &'a [usize],
    pub immutable_history: &'a [usize],
    pub gaps: &'a [FixedGap<'a>],
    pub tracks: &'a [RouteTrack],
    pub config: RepeatWindowRepairConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedGapSelection {
    pub left: usize,
    pub right: usize,
    pub intermediates: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FixedGapPlanStats {
    pub evaluated_states: usize,
    pub retained_states: usize,
    pub pruned_states: usize,
    pub completed_gaps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedGapProgressPhase {
    Searching,
    Completed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedGapProgressEvent {
    pub phase: FixedGapProgressPhase,
    pub gap: usize,
    pub total_gaps: usize,
    pub completed_gaps: usize,
    pub evaluated_states: usize,
    pub retained_states: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixedGapPlan {
    pub final_route: Vec<usize>,
    pub gaps: Vec<FixedGapSelection>,
    pub transition_sum: f64,
    pub worst_transition: f64,
    pub guidance_adjustment: f64,
    pub stats: FixedGapPlanStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedGapPlanError {
    InvalidConfig(&'static str),
    InvalidSource(&'static str),
    InvalidTrackIndex(usize),
    NoCompleteRoute {
        blocking_gap: usize,
        evaluated_states: usize,
        retained_states: usize,
    },
    FinalRouteInvalid(&'static str),
    Path(AnchoredPathError),
}

impl fmt::Display for FixedGapPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message)
            | Self::InvalidSource(message)
            | Self::FinalRouteInvalid(message) => formatter.write_str(message),
            Self::InvalidTrackIndex(index) => {
                write!(formatter, "multi-gap plan references unknown track index {index}")
            }
            Self::NoCompleteRoute {
                blocking_gap,
                evaluated_states,
                retained_states,
            } => write!(
                formatter,
                "no complete route remains at gap {blocking_gap} after evaluating {evaluated_states} states and retaining {retained_states}"
            ),
            Self::Path(error) => write!(formatter, "anchored gap search failed: {error}"),
        }
    }
}

impl std::error::Error for FixedGapPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Path(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct GlobalState {
    route: Vec<usize>,
    gaps: Vec<FixedGapSelection>,
    added_track_count: usize,
    transition_sum: f64,
    worst_transition: f64,
    guidance_adjustment: f64,
    stats: FixedGapPlanStats,
}

fn state_order(left: &GlobalState, right: &GlobalState) -> std::cmp::Ordering {
    left.worst_transition
        .total_cmp(&right.worst_transition)
        .then_with(|| left.transition_sum.total_cmp(&right.transition_sum))
        .then_with(|| {
            right
                .guidance_adjustment
                .total_cmp(&left.guidance_adjustment)
        })
        .then_with(|| left.route.cmp(&right.route))
}

fn variation_key(seed: u64, route: &[usize]) -> u64 {
    route.iter().fold(seed, |value, track| {
        let mixed = value ^ (*track as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb)
    })
}

fn validate_request(request: &FixedGapPlanRequest<'_>) -> Result<(), FixedGapPlanError> {
    if request.source_route.len() < 2 {
        return Err(FixedGapPlanError::InvalidSource(
            "fixed-gap planning requires at least two source tracks",
        ));
    }
    if request.gaps.len() != request.source_route.len() - 1 {
        return Err(FixedGapPlanError::InvalidSource(
            "fixed-gap definitions must match every adjacent source pair",
        ));
    }
    if !(1..=MAX_TRACKS_PER_GAP).contains(&request.config.tracks_per_gap) {
        return Err(FixedGapPlanError::InvalidConfig(
            "tracks per gap must be between one and eight",
        ));
    }
    if request.config.candidate_limit == 0
        || request.config.beam_width == 0
        || request.config.alternatives_per_gap == 0
    {
        return Err(FixedGapPlanError::InvalidConfig(
            "candidate, beam, and alternative limits must be at least one",
        ));
    }
    if request.config.variation_percent > 100 {
        return Err(FixedGapPlanError::InvalidConfig(
            "variation percent must not exceed one hundred",
        ));
    }
    for (position, gap) in request.gaps.iter().enumerate() {
        if [gap.left, gap.right]
            != [
                request.source_route[position],
                request.source_route[position + 1],
            ]
        {
            return Err(FixedGapPlanError::InvalidSource(
                "fixed-gap anchors must follow the source route",
            ));
        }
    }
    for index in request
        .source_route
        .iter()
        .copied()
        .chain(request.immutable_history.iter().copied())
        .chain(
            request
                .gaps
                .iter()
                .flat_map(|gap| gap.candidates.iter().map(|candidate| candidate.track)),
        )
    {
        if index >= request.tracks.len() {
            return Err(FixedGapPlanError::InvalidTrackIndex(index));
        }
    }
    Ok(())
}

fn validate_repair_request(
    request: &RepeatWindowRepairRequest<'_>,
) -> Result<(), FixedGapPlanError> {
    if request.source_route.len() < 2 {
        return Err(FixedGapPlanError::InvalidSource(
            "repeat-window repair requires at least two source tracks",
        ));
    }
    if request.gaps.len() != request.source_route.len() - 1 {
        return Err(FixedGapPlanError::InvalidSource(
            "repair gap definitions must match every adjacent source pair",
        ));
    }
    if request.config.max_added_tracks == 0 {
        return Err(FixedGapPlanError::InvalidConfig(
            "repeat-window repair requires a positive addition budget",
        ));
    }
    if request.config.candidate_limit == 0
        || request.config.beam_width == 0
        || request.config.alternatives_per_gap == 0
    {
        return Err(FixedGapPlanError::InvalidConfig(
            "candidate, beam, and alternative limits must be at least one",
        ));
    }
    if request.config.variation_percent > 100 {
        return Err(FixedGapPlanError::InvalidConfig(
            "variation percent must not exceed one hundred",
        ));
    }
    for (position, gap) in request.gaps.iter().enumerate() {
        if [gap.left, gap.right]
            != [
                request.source_route[position],
                request.source_route[position + 1],
            ]
        {
            return Err(FixedGapPlanError::InvalidSource(
                "repair gap anchors must follow the source route",
            ));
        }
    }
    for index in request
        .source_route
        .iter()
        .copied()
        .chain(request.immutable_history.iter().copied())
        .chain(
            request
                .gaps
                .iter()
                .flat_map(|gap| gap.candidates.iter().map(|candidate| candidate.track)),
        )
    {
        if index >= request.tracks.len() {
            return Err(FixedGapPlanError::InvalidTrackIndex(index));
        }
    }
    Ok(())
}

fn final_route_valid(request: &FixedGapPlanRequest<'_>, state: &GlobalState) -> bool {
    let expected_len = request.source_route.len()
        + request.config.tracks_per_gap * request.source_route.len().saturating_sub(1);
    if state.route.len() != expected_len
        || state.gaps.len() != request.gaps.len()
        || state
            .gaps
            .iter()
            .any(|gap| gap.intermediates.len() != request.config.tracks_per_gap)
    {
        return false;
    }
    let source_set = request.source_route.iter().copied().collect::<HashSet<_>>();
    let routed_sources = state
        .route
        .iter()
        .copied()
        .filter(|track| source_set.contains(track))
        .collect::<Vec<_>>();
    if routed_sources != request.source_route {
        return false;
    }
    let generated = state
        .gaps
        .iter()
        .flat_map(|gap| gap.intermediates.iter().copied())
        .collect::<Vec<_>>();
    if generated.iter().any(|track| source_set.contains(track))
        || generated.iter().copied().collect::<HashSet<_>>().len() != generated.len()
    {
        return false;
    }

    let mut repeat_route = request.immutable_history.to_vec();
    repeat_route.extend(state.route.iter().copied());
    let first_mutable = request.immutable_history.len();
    (first_mutable..repeat_route.len()).all(|position| {
        repeat_windows_safe_at(
            &repeat_route,
            request.tracks,
            position,
            request.config.artist_window,
            request.config.album_window,
        ) && (request.config.track_window == 0
            || repeat_route
                .iter()
                .enumerate()
                .all(|(other_position, track)| {
                    other_position == position
                        || other_position.abs_diff(position) > request.config.track_window
                        || *track != repeat_route[position]
                }))
    })
}

fn repair_final_route_valid(request: &RepeatWindowRepairRequest<'_>, state: &GlobalState) -> bool {
    if state.added_track_count > request.config.max_added_tracks
        || state.route.len() != request.source_route.len() + state.added_track_count
        || state.gaps.len() != request.gaps.len()
    {
        return false;
    }
    let source_set = request.source_route.iter().copied().collect::<HashSet<_>>();
    let routed_sources = state
        .route
        .iter()
        .copied()
        .filter(|track| source_set.contains(track))
        .collect::<Vec<_>>();
    if routed_sources != request.source_route {
        return false;
    }
    let generated = state
        .gaps
        .iter()
        .flat_map(|gap| gap.intermediates.iter().copied())
        .collect::<Vec<_>>();
    if generated.len() != state.added_track_count
        || generated.iter().any(|track| source_set.contains(track))
        || generated.iter().copied().collect::<HashSet<_>>().len() != generated.len()
    {
        return false;
    }

    let mut repeat_route = request.immutable_history.to_vec();
    repeat_route.extend(state.route.iter().copied());
    let first_mutable = request.immutable_history.len();
    (first_mutable..repeat_route.len()).all(|position| {
        repeat_windows_safe_at(
            &repeat_route,
            request.tracks,
            position,
            request.config.artist_window,
            request.config.album_window,
        ) && (request.config.track_window == 0
            || repeat_route
                .iter()
                .enumerate()
                .all(|(other_position, track)| {
                    other_position == position
                        || other_position.abs_diff(position) > request.config.track_window
                        || *track != repeat_route[position]
                }))
    })
}

pub fn plan_fixed_gap_count<S>(
    request: FixedGapPlanRequest<'_>,
    scorer: S,
) -> Result<FixedGapPlan, FixedGapPlanError>
where
    S: AnchoredPathScorer,
{
    plan_fixed_gap_count_with_progress(request, scorer, |_| {})
}

pub fn plan_fixed_gap_count_with_progress<S, F>(
    request: FixedGapPlanRequest<'_>,
    scorer: S,
    mut progress: F,
) -> Result<FixedGapPlan, FixedGapPlanError>
where
    S: AnchoredPathScorer,
    F: FnMut(FixedGapProgressEvent),
{
    validate_request(&request)?;
    let source_unavailable = request.source_route.to_vec();
    let mut states = vec![GlobalState {
        route: vec![request.source_route[0]],
        gaps: Vec::with_capacity(request.gaps.len()),
        added_track_count: 0,
        transition_sum: 0.0,
        worst_transition: 0.0,
        guidance_adjustment: 0.0,
        stats: FixedGapPlanStats {
            evaluated_states: 1,
            retained_states: 1,
            ..FixedGapPlanStats::default()
        },
    }];

    for (gap_index, gap) in request.gaps.iter().enumerate() {
        progress(FixedGapProgressEvent {
            phase: FixedGapProgressPhase::Searching,
            gap: gap_index + 1,
            total_gaps: request.gaps.len(),
            completed_gaps: gap_index,
            evaluated_states: states.len(),
            retained_states: states.len(),
        });
        let batches = states
            .par_iter()
            .map(|state| {
                let mut unavailable = source_unavailable.clone();
                unavailable.extend(state.route.iter().copied());
                let options = search_anchored_paths(
                    AnchoredPathRequest {
                        route_prefix: &state.route,
                        immutable_history: request.immutable_history,
                        unavailable_tracks: &unavailable,
                        left_anchor: gap.left,
                        right_anchor: gap.right,
                        candidates: gap.candidates,
                        tracks: request.tracks,
                        config: AnchoredPathSearchConfig {
                            max_intermediates: request.config.tracks_per_gap,
                            candidate_limit: request.config.candidate_limit,
                            beam_width: request.config.beam_width,
                            alternatives_per_count: request.config.alternatives_per_gap,
                            variation_percent: 0,
                            generation_seed: request.config.generation_seed,
                            artist_window: request.config.artist_window,
                            album_window: request.config.album_window,
                            track_window: request.config.track_window,
                        },
                    },
                    &scorer,
                )
                .map_err(FixedGapPlanError::Path)?;
                Ok(options
                    .into_iter()
                    .filter(|option| option.intermediates.len() == request.config.tracks_per_gap)
                    .map(|option| GlobalState {
                        added_track_count: state.added_track_count + option.intermediates.len(),
                        route: option.route,
                        gaps: state
                            .gaps
                            .iter()
                            .cloned()
                            .chain(std::iter::once(FixedGapSelection {
                                left: gap.left,
                                right: gap.right,
                                intermediates: option.intermediates,
                            }))
                            .collect(),
                        transition_sum: state.transition_sum + option.transition_sum,
                        worst_transition: state.worst_transition.max(option.worst_transition),
                        guidance_adjustment: state.guidance_adjustment + option.guidance_adjustment,
                        stats: FixedGapPlanStats {
                            evaluated_states: state
                                .stats
                                .evaluated_states
                                .saturating_add(option.stats.evaluated_states),
                            retained_states: state.stats.retained_states,
                            pruned_states: state.stats.pruned_states,
                            completed_gaps: gap_index + 1,
                        },
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Vec<Result<Vec<_>, FixedGapPlanError>>>();

        let mut expanded = Vec::new();
        for batch in batches {
            expanded.extend(batch?);
        }
        let evaluated = expanded.len();
        expanded.sort_by(state_order);
        expanded.dedup_by(|left, right| left.route == right.route);
        let before_prune = expanded.len();
        expanded.truncate(request.config.beam_width);
        let retained = expanded.len();
        let pruned = before_prune.saturating_sub(retained);
        for state in &mut expanded {
            state.stats.evaluated_states = state.stats.evaluated_states.saturating_add(evaluated);
            state.stats.retained_states = state.stats.retained_states.saturating_add(retained);
            state.stats.pruned_states = state.stats.pruned_states.saturating_add(pruned);
        }
        if expanded.is_empty() {
            return Err(FixedGapPlanError::NoCompleteRoute {
                blocking_gap: gap_index,
                evaluated_states: evaluated,
                retained_states: 0,
            });
        }
        states = expanded;
        progress(FixedGapProgressEvent {
            phase: FixedGapProgressPhase::Completed,
            gap: gap_index + 1,
            total_gaps: request.gaps.len(),
            completed_gaps: gap_index + 1,
            evaluated_states: evaluated,
            retained_states: retained,
        });
    }

    states.retain(|state| final_route_valid(&request, state));
    if states.is_empty() {
        return Err(FixedGapPlanError::FinalRouteInvalid(
            "no retained route passed the final global proof",
        ));
    }
    states.sort_by(state_order);
    let best = &states[0];
    let worst_limit = best.worst_transition + (best.worst_transition * 0.02).max(1.0e-9);
    let sum_limit = best.transition_sum + (best.transition_sum * 0.05).max(1.0e-9);
    let quality_count = states
        .iter()
        .take_while(|state| state.worst_transition <= worst_limit)
        .filter(|state| state.transition_sum <= sum_limit)
        .count()
        .max(1);
    let selection_pool = if request.config.variation_percent == 0 {
        1
    } else {
        1 + (quality_count.saturating_sub(1) * usize::from(request.config.variation_percent) / 100)
    };
    let mut selected = states.into_iter().take(selection_pool).collect::<Vec<_>>();
    if request.config.variation_percent > 0 {
        selected.sort_by_key(|state| variation_key(request.config.generation_seed, &state.route));
    }
    let state = selected.remove(0);
    Ok(FixedGapPlan {
        final_route: state.route,
        gaps: state.gaps,
        transition_sum: state.transition_sum,
        worst_transition: state.worst_transition,
        guidance_adjustment: state.guidance_adjustment,
        stats: state.stats,
    })
}

pub fn plan_repeat_window_repair<S>(
    request: RepeatWindowRepairRequest<'_>,
    scorer: S,
) -> Result<FixedGapPlan, FixedGapPlanError>
where
    S: AnchoredPathScorer,
{
    plan_repeat_window_repair_with_progress(request, scorer, |_| {})
}

pub fn plan_repeat_window_repair_with_progress<S, F>(
    request: RepeatWindowRepairRequest<'_>,
    scorer: S,
    mut progress: F,
) -> Result<FixedGapPlan, FixedGapPlanError>
where
    S: AnchoredPathScorer,
    F: FnMut(FixedGapProgressEvent),
{
    validate_repair_request(&request)?;
    let source_unavailable = request.source_route.to_vec();
    let mut states = vec![GlobalState {
        route: vec![request.source_route[0]],
        gaps: Vec::with_capacity(request.gaps.len()),
        added_track_count: 0,
        transition_sum: 0.0,
        worst_transition: 0.0,
        guidance_adjustment: 0.0,
        stats: FixedGapPlanStats {
            evaluated_states: 1,
            retained_states: 1,
            ..FixedGapPlanStats::default()
        },
    }];

    for (gap_index, gap) in request.gaps.iter().enumerate() {
        progress(FixedGapProgressEvent {
            phase: FixedGapProgressPhase::Searching,
            gap: gap_index + 1,
            total_gaps: request.gaps.len(),
            completed_gaps: gap_index,
            evaluated_states: states.len(),
            retained_states: states.len(),
        });
        let batches = states
            .par_iter()
            .map(|state| {
                let remaining = request
                    .config
                    .max_added_tracks
                    .saturating_sub(state.added_track_count);
                let mut unavailable = source_unavailable.clone();
                unavailable.extend(state.route.iter().copied());
                let options = search_anchored_paths(
                    AnchoredPathRequest {
                        route_prefix: &state.route,
                        immutable_history: request.immutable_history,
                        unavailable_tracks: &unavailable,
                        left_anchor: gap.left,
                        right_anchor: gap.right,
                        candidates: gap.candidates,
                        tracks: request.tracks,
                        config: AnchoredPathSearchConfig {
                            max_intermediates: remaining,
                            candidate_limit: request.config.candidate_limit,
                            beam_width: request.config.beam_width,
                            alternatives_per_count: request.config.alternatives_per_gap,
                            variation_percent: 0,
                            generation_seed: request.config.generation_seed,
                            artist_window: request.config.artist_window,
                            album_window: request.config.album_window,
                            track_window: request.config.track_window,
                        },
                    },
                    &scorer,
                )
                .map_err(FixedGapPlanError::Path)?;
                Ok(options
                    .into_iter()
                    .filter(|option| option.intermediates.len() <= remaining)
                    .map(|option| {
                        let added_track_count =
                            state.added_track_count + option.intermediates.len();
                        GlobalState {
                            route: option.route,
                            gaps: state
                                .gaps
                                .iter()
                                .cloned()
                                .chain(std::iter::once(FixedGapSelection {
                                    left: gap.left,
                                    right: gap.right,
                                    intermediates: option.intermediates,
                                }))
                                .collect(),
                            added_track_count,
                            transition_sum: state.transition_sum + option.transition_sum,
                            worst_transition: state.worst_transition.max(option.worst_transition),
                            guidance_adjustment: state.guidance_adjustment
                                + option.guidance_adjustment,
                            stats: FixedGapPlanStats {
                                evaluated_states: state
                                    .stats
                                    .evaluated_states
                                    .saturating_add(option.stats.evaluated_states),
                                retained_states: state.stats.retained_states,
                                pruned_states: state.stats.pruned_states,
                                completed_gaps: gap_index + 1,
                            },
                        }
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Vec<Result<Vec<_>, FixedGapPlanError>>>();

        let mut expanded = Vec::new();
        for batch in batches {
            expanded.extend(batch?);
        }
        let evaluated = expanded.len();
        expanded.sort_by(|left, right| {
            left.added_track_count
                .cmp(&right.added_track_count)
                .then_with(|| state_order(left, right))
        });
        expanded.dedup_by(|left, right| left.route == right.route);
        let before_prune = expanded.len();
        let mut per_count = vec![Vec::new(); request.config.max_added_tracks + 1];
        for state in expanded {
            if per_count[state.added_track_count].len() < request.config.beam_width {
                per_count[state.added_track_count].push(state);
            }
        }
        let mut retained_states = per_count.into_iter().flatten().collect::<Vec<_>>();
        let retained = retained_states.len();
        let pruned = before_prune.saturating_sub(retained);
        for state in &mut retained_states {
            state.stats.evaluated_states = state.stats.evaluated_states.saturating_add(evaluated);
            state.stats.retained_states = state.stats.retained_states.saturating_add(retained);
            state.stats.pruned_states = state.stats.pruned_states.saturating_add(pruned);
        }
        if retained_states.is_empty() {
            return Err(FixedGapPlanError::NoCompleteRoute {
                blocking_gap: gap_index,
                evaluated_states: evaluated,
                retained_states: 0,
            });
        }
        states = retained_states;
        progress(FixedGapProgressEvent {
            phase: FixedGapProgressPhase::Completed,
            gap: gap_index + 1,
            total_gaps: request.gaps.len(),
            completed_gaps: gap_index + 1,
            evaluated_states: evaluated,
            retained_states: retained,
        });
    }

    states.retain(|state| repair_final_route_valid(&request, state));
    if states.is_empty() {
        return Err(FixedGapPlanError::FinalRouteInvalid(
            "no retained repeat-window repair passed the final global proof",
        ));
    }
    states.sort_by(|left, right| {
        left.added_track_count
            .cmp(&right.added_track_count)
            .then_with(|| state_order(left, right))
    });
    let minimum_count = states[0].added_track_count;
    states.retain(|state| state.added_track_count == minimum_count);
    let best = &states[0];
    let worst_limit = best.worst_transition + (best.worst_transition * 0.02).max(1.0e-9);
    let sum_limit = best.transition_sum + (best.transition_sum * 0.05).max(1.0e-9);
    let quality_count = states
        .iter()
        .take_while(|state| state.worst_transition <= worst_limit)
        .filter(|state| state.transition_sum <= sum_limit)
        .count()
        .max(1);
    let selection_pool = if request.config.variation_percent == 0 {
        1
    } else {
        1 + (quality_count.saturating_sub(1) * usize::from(request.config.variation_percent) / 100)
    };
    let mut selected = states.into_iter().take(selection_pool).collect::<Vec<_>>();
    if request.config.variation_percent > 0 {
        selected.sort_by_key(|state| variation_key(request.config.generation_seed, &state.route));
    }
    let state = selected.remove(0);
    Ok(FixedGapPlan {
        final_route: state.route,
        gaps: state.gaps,
        transition_sum: state.transition_sum,
        worst_transition: state.worst_transition,
        guidance_adjustment: state.guidance_adjustment,
        stats: state.stats,
    })
}
