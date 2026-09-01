// SPDX-License-Identifier: GPL-3.0-only

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;

use crate::anchored_path::{
    search_anchored_paths, AnchoredPathCandidate, AnchoredPathError, AnchoredPathRequest,
    AnchoredPathSearchConfig,
};
use crate::bridge::{
    gap_context_matrix, rank_candidates_in_context, rank_endpoint_candidates,
    BridgeCandidateEvaluation, BridgeConfig, BridgeError, CandidateScoringContext,
    EndpointCandidateEvaluation, EndpointSlot, FrozenReference,
};
use crate::route::{self, RouteTrack};
use crate::semantic::{CandidateSemantics, GapEvidence, SemanticPool};

pub const MAX_EXACT_TRACKS_PER_GAP: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct AutomaticGap {
    pub original_position: usize,
    pub left: usize,
    pub right: usize,
    pub direct_distance: f64,
    pub direct_percentile: f64,
    pub semantics: GapEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutomaticSelectionConfig {
    pub max_added_tracks: usize,
    pub trigger_percentile: f64,
    pub track_guidance_percent: u8,
    pub artist_guidance_percent: u8,
    pub variation_percent: u8,
    pub generation_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactSelectionConfig {
    pub requested_added_tracks: usize,
    pub candidate_limit: usize,
    pub beam_width: usize,
    pub max_tracks_per_gap: usize,
    pub track_guidance_percent: u8,
    pub artist_guidance_percent: u8,
    pub variation_percent: u8,
    pub generation_seed: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GuidanceConfig {
    track_percent: u8,
    artist_percent: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VariationConfig {
    percent: u8,
    seed: u64,
    minimum_pool: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvolvingAcceptance {
    FullBridge,
    ReachableFromLeft,
}

impl EvolvingAcceptance {
    fn accepts(self, evaluation: &BridgeCandidateEvaluation, config: &BridgeConfig) -> bool {
        match self {
            Self::FullBridge => evaluation.accepted,
            Self::ReachableFromLeft => {
                evaluation.repeat_safe && evaluation.left_percentile <= config.max_leg_percentile
            }
        }
    }
}

impl AutomaticSelectionConfig {
    fn guidance(self) -> GuidanceConfig {
        GuidanceConfig {
            track_percent: self.track_guidance_percent,
            artist_percent: self.artist_guidance_percent,
        }
    }

    fn variation(self) -> VariationConfig {
        VariationConfig {
            percent: self.variation_percent,
            seed: self.generation_seed,
            minimum_pool: 2,
        }
    }
}

impl ExactSelectionConfig {
    fn guidance(self) -> GuidanceConfig {
        GuidanceConfig {
            track_percent: self.track_guidance_percent,
            artist_percent: self.artist_guidance_percent,
        }
    }

    fn variation(self) -> VariationConfig {
        VariationConfig {
            percent: self.variation_percent,
            seed: self.generation_seed,
            minimum_pool: self.candidate_limit.saturating_add(1),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactEndpointSlot {
    pub anchor: usize,
    pub semantics: GapEvidence,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExactEndpointSlots {
    pub opening: Option<ExactEndpointSlot>,
    pub closing: Option<ExactEndpointSlot>,
}

#[derive(Clone, Copy)]
pub struct ExactScoringContext<'a> {
    pub tracks: &'a [RouteTrack],
    pub learned_matrix: &'a Array2<f32>,
    pub config: &'a BridgeConfig,
    pub reference: &'a FrozenReference,
}

#[derive(Clone, Copy)]
struct GapRankingContext<'a> {
    scoring: ExactScoringContext<'a>,
    frozen_matrix: Option<&'a Array2<f32>>,
}

#[derive(Clone, Copy)]
pub struct DestinationRepeatContext<'a> {
    pub history_route: &'a [usize],
    pub track_window: usize,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionReason {
    Selected,
    BelowThreshold,
    BudgetExhausted,
    NoEligibleCandidate,
    RepeatConflict,
    AcousticRejected,
    NoImprovement,
    NotSelected,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedBridge {
    pub semantics: CandidateSemantics,
    pub evaluation: BridgeCandidateEvaluation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GapDecision {
    pub original_position: usize,
    pub route_position: usize,
    pub left: usize,
    pub right: usize,
    pub direct_distance: f64,
    pub direct_percentile: f64,
    pub semantic_pool: SemanticPool,
    pub reason: DecisionReason,
    pub selected: Option<SelectedBridge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutomaticSelection {
    pub final_route: Vec<usize>,
    pub decisions: Vec<GapDecision>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactSearchStats {
    pub max_tracks_per_gap: usize,
    pub evaluated_states: usize,
    pub retained_states: usize,
    pub maximum_additions_found: usize,
    pub structural_upper_bound: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactSelection {
    pub requested_added_tracks: usize,
    pub final_route: Option<Vec<usize>>,
    pub decisions: Vec<GapDecision>,
    pub endpoint_decisions: Vec<EndpointDecision>,
    pub stats: ExactSearchStats,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DestinationRouteOption {
    pub added_track_count: usize,
    pub selection: ExactSelection,
    pub adjacent_transition_sum: f64,
    pub adjacent_worst_transition: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedEndpoint {
    pub semantics: CandidateSemantics,
    pub evaluation: EndpointCandidateEvaluation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EndpointDecision {
    pub slot: EndpointSlot,
    pub anchor: usize,
    pub semantic_pool: SemanticPool,
    pub reason: DecisionReason,
    pub selected: Option<SelectedEndpoint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewError {
    InvalidOriginalGap(usize),
    InvalidExactConfig(&'static str),
    FinalRouteInvalid(&'static str),
    Scoring(BridgeError),
    RouteScoring(route::RouteError),
}

impl fmt::Display for PreviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOriginalGap(position) => {
                write!(
                    formatter,
                    "original gap {position} is absent from the evolving route"
                )
            }
            Self::InvalidExactConfig(message) => formatter.write_str(message),
            Self::FinalRouteInvalid(message) => {
                write!(formatter, "exact-count final route is invalid: {message}")
            }
            Self::Scoring(error) => write!(formatter, "automatic bridge scoring failed: {error}"),
            Self::RouteScoring(error) => {
                write!(formatter, "exact-count route scoring failed: {error}")
            }
        }
    }
}

impl std::error::Error for PreviewError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scoring(error) => Some(error),
            Self::RouteScoring(error) => Some(error),
            Self::InvalidOriginalGap(_)
            | Self::InvalidExactConfig(_)
            | Self::FinalRouteInvalid(_) => None,
        }
    }
}

fn route_position(route: &[usize], gap: &AutomaticGap) -> Option<usize> {
    route
        .windows(2)
        .position(|anchors| anchors == [gap.left, gap.right])
        .map(|position| position + 1)
}

fn gap_right_position(route: &[usize], gap: &AutomaticGap) -> Option<usize> {
    let left = route.iter().position(|track| *track == gap.left)?;
    let right = route.iter().position(|track| *track == gap.right)?;
    (left < right).then_some(right)
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

fn varied_pool_length(accepted: usize, variation: VariationConfig) -> usize {
    if accepted == 0 || variation.percent == 0 {
        return accepted.min(1);
    }
    let ceiling = accepted.min(32);
    let floor = variation.minimum_pool.clamp(1, ceiling);
    floor + (ceiling.saturating_sub(floor) * usize::from(variation.percent) / 100)
}

fn rank_for_evolving_route(
    route: &[usize],
    position: usize,
    semantics: &[CandidateSemantics],
    context: GapRankingContext<'_>,
    guidance: GuidanceConfig,
    variation: VariationConfig,
    acceptance: EvolvingAcceptance,
) -> Result<Vec<BridgeCandidateEvaluation>, PreviewError> {
    let GapRankingContext {
        scoring:
            ExactScoringContext {
                tracks,
                learned_matrix,
                config,
                reference,
            },
        frozen_matrix,
    } = context;
    let semantics_by_candidate = semantics
        .iter()
        .map(|candidate| (candidate.candidate, candidate))
        .collect::<HashMap<_, _>>();
    let candidates = semantics
        .iter()
        .map(|candidate| candidate.candidate)
        .collect::<Vec<_>>();
    let mut evaluations = rank_candidates_in_context(
        route,
        position,
        &candidates,
        tracks,
        CandidateScoringContext {
            learned_matrix,
            config,
            reference,
            frozen_matrix,
        },
    )
    .map_err(PreviewError::Scoring)?;
    evaluations.sort_by(|left, right| {
        acceptance
            .accepts(right, config)
            .cmp(&acceptance.accepts(left, config))
            .then_with(|| {
                semantics_by_candidate[&left.candidate]
                    .adjusted_percentile(
                        left.max_percentile,
                        guidance.track_percent,
                        guidance.artist_percent,
                    )
                    .total_cmp(
                        &semantics_by_candidate[&right.candidate].adjusted_percentile(
                            right.max_percentile,
                            guidance.track_percent,
                            guidance.artist_percent,
                        ),
                    )
            })
            .then_with(|| {
                semantics_by_candidate[&left.candidate]
                    .adjusted_percentile(
                        left.detour_percentile,
                        guidance.track_percent,
                        guidance.artist_percent,
                    )
                    .total_cmp(
                        &semantics_by_candidate[&right.candidate].adjusted_percentile(
                            right.detour_percentile,
                            guidance.track_percent,
                            guidance.artist_percent,
                        ),
                    )
            })
            .then_with(|| left.max_percentile.total_cmp(&right.max_percentile))
            .then_with(|| left.detour_percentile.total_cmp(&right.detour_percentile))
            .then_with(|| left.candidate.cmp(&right.candidate))
    });
    let accepted = evaluations
        .iter()
        .take_while(|item| acceptance.accepts(item, config))
        .count();
    let pool = varied_pool_length(accepted, variation);
    if pool > 1 {
        evaluations[..pool]
            .sort_by_key(|item| variation_key(variation.seed, route, position, item.candidate));
    }
    Ok(evaluations)
}

fn local_objective(distance_sum: f64, worst_distance: f64) -> f64 {
    distance_sum + 2.0 * worst_distance
}

fn gap_search_objective(
    route: &[usize],
    original_route: &[usize],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
) -> Result<f64, PreviewError> {
    if config.gap_context_mode == crate::bridge::GapContextMode::Rolling {
        return route::evaluate_adaptive_sequence(
            route,
            tracks,
            learned_matrix,
            config.seed_limit,
            config.learned_percent,
        )
        .map(|metrics| metrics.objective)
        .map_err(PreviewError::RouteScoring);
    }

    let mut gap_matrices = Vec::with_capacity(original_route.len().saturating_sub(1));
    for anchors in original_route.windows(2) {
        let left_position = route.iter().position(|track| *track == anchors[0]).ok_or(
            PreviewError::FinalRouteInvalid("source gap left anchor is absent"),
        )?;
        let right_position = route.iter().position(|track| *track == anchors[1]).ok_or(
            PreviewError::FinalRouteInvalid("source gap right anchor is absent"),
        )?;
        let matrix = gap_context_matrix(route, left_position + 1, tracks, learned_matrix, config)
            .map_err(PreviewError::Scoring)?
            .expect("frozen gap mode always prepares a matrix");
        gap_matrices.push((left_position, right_position, matrix));
    }

    let mut distance_sum = 0.0_f64;
    let mut worst_distance = 0.0_f64;
    for position in 1..route.len() {
        let frozen_matrix = gap_matrices
            .iter()
            .find(|(left, right, _)| *left < position && position <= *right)
            .map(|(_, _, matrix)| matrix);
        let distance = match frozen_matrix {
            Some(matrix) => crate::bridge::contextual_distance_with_matrix(
                &route[..position],
                route[position],
                tracks,
                matrix,
                config,
            ),
            None => crate::bridge::contextual_distance(
                &route[..position],
                route[position],
                tracks,
                learned_matrix,
                config,
            ),
        }
        .map_err(PreviewError::Scoring)?;
        distance_sum += distance;
        worst_distance = worst_distance.max(distance);
    }
    Ok(local_objective(distance_sum, worst_distance))
}

pub fn select_automatic_bridges(
    original_route: &[usize],
    gaps: &[AutomaticGap],
    selection_config: &AutomaticSelectionConfig,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<AutomaticSelection, PreviewError> {
    let mut ordered_gaps = gaps.to_vec();
    ordered_gaps.sort_by_key(|gap| gap.original_position);
    let mut final_route = original_route.to_vec();
    let mut decisions = Vec::with_capacity(ordered_gaps.len());
    let mut added = 0usize;

    for gap in ordered_gaps {
        let position = route_position(&final_route, &gap)
            .ok_or(PreviewError::InvalidOriginalGap(gap.original_position))?;
        let mut reason = if gap.direct_percentile <= selection_config.trigger_percentile {
            DecisionReason::BelowThreshold
        } else if added >= selection_config.max_added_tracks {
            DecisionReason::BudgetExhausted
        } else if gap.semantics.candidates.is_empty() {
            DecisionReason::NoEligibleCandidate
        } else {
            DecisionReason::NoImprovement
        };
        let mut selected = None;

        if gap.direct_percentile > selection_config.trigger_percentile
            && added < selection_config.max_added_tracks
            && !gap.semantics.candidates.is_empty()
        {
            let frozen_matrix =
                gap_context_matrix(&final_route, position, tracks, learned_matrix, config)
                    .map_err(PreviewError::Scoring)?;
            let evaluations = rank_for_evolving_route(
                &final_route,
                position,
                &gap.semantics.candidates,
                GapRankingContext {
                    scoring: ExactScoringContext {
                        tracks,
                        learned_matrix,
                        config,
                        reference,
                    },
                    frozen_matrix: frozen_matrix.as_ref(),
                },
                selection_config.guidance(),
                selection_config.variation(),
                EvolvingAcceptance::FullBridge,
            )?;
            if let Some(evaluation) = evaluations.iter().find(|candidate| {
                let inserted = local_objective(
                    candidate.left_distance + candidate.right_distance,
                    candidate.left_distance.max(candidate.right_distance),
                );
                let direct = local_objective(gap.direct_distance, gap.direct_distance);
                candidate.accepted && inserted < direct
            }) {
                let semantics = gap
                    .semantics
                    .candidates
                    .iter()
                    .find(|candidate| candidate.candidate == evaluation.candidate)
                    .expect("every evaluation has frozen candidate semantics")
                    .clone();
                final_route.insert(position, evaluation.candidate);
                selected = Some(SelectedBridge {
                    semantics,
                    evaluation: evaluation.clone(),
                });
                added += 1;
                reason = DecisionReason::Selected;
            } else if evaluations.iter().all(|candidate| !candidate.repeat_safe) {
                reason = DecisionReason::RepeatConflict;
            } else if evaluations.iter().all(|candidate| !candidate.accepted) {
                reason = DecisionReason::AcousticRejected;
            }
        }

        decisions.push(GapDecision {
            original_position: gap.original_position,
            route_position: position,
            left: gap.left,
            right: gap.right,
            direct_distance: gap.direct_distance,
            direct_percentile: gap.direct_percentile,
            semantic_pool: gap.semantics.pool,
            reason,
            selected,
        });
    }

    Ok(AutomaticSelection {
        final_route,
        decisions,
    })
}

#[derive(Clone, Debug)]
struct ExactState {
    route: Vec<usize>,
    decisions: Vec<GapDecision>,
    objective: f64,
}

fn exact_state_precedes(left: &ExactState, right: &ExactState) -> bool {
    left.objective.total_cmp(&right.objective).is_lt()
        || (left.objective == right.objective && left.route < right.route)
}

fn exact_decision(
    gap: &AutomaticGap,
    route_position: usize,
    reason: DecisionReason,
    selected: Option<SelectedBridge>,
) -> GapDecision {
    GapDecision {
        original_position: gap.original_position,
        route_position,
        left: gap.left,
        right: gap.right,
        direct_distance: gap.direct_distance,
        direct_percentile: gap.direct_percentile,
        semantic_pool: gap.semantics.pool,
        reason,
        selected,
    }
}

#[derive(Clone, Debug)]
struct MultiExactState {
    route: Vec<usize>,
    gap_selections: Vec<Vec<usize>>,
    objective: f64,
}

fn multi_exact_state_precedes(left: &MultiExactState, right: &MultiExactState) -> bool {
    left.objective.total_cmp(&right.objective).is_lt()
        || (left.objective == right.objective && left.route < right.route)
}

fn sort_and_prune_multi_states(states: &mut Vec<MultiExactState>, beam_width: usize) {
    states.sort_by(|left, right| {
        if multi_exact_state_precedes(left, right) {
            std::cmp::Ordering::Less
        } else if multi_exact_state_precedes(right, left) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    states.dedup_by(|left, right| left.route == right.route);
    states.truncate(beam_width);
}

fn final_exact_decisions(
    final_route: &[usize],
    gaps: &[AutomaticGap],
    gap_selections: &[Vec<usize>],
    scoring: ExactScoringContext<'_>,
    selection_config: &ExactSelectionConfig,
) -> Result<Vec<GapDecision>, PreviewError> {
    let ExactScoringContext {
        tracks,
        learned_matrix,
        config,
        reference,
    } = scoring;
    if gaps.len() != gap_selections.len() {
        return Err(PreviewError::FinalRouteInvalid(
            "gap selection count does not match the original gap count",
        ));
    }
    let mut decisions = Vec::new();
    for (gap, selected_candidates) in gaps.iter().zip(gap_selections) {
        if selected_candidates.is_empty() {
            let position = gap_right_position(final_route, gap)
                .ok_or(PreviewError::InvalidOriginalGap(gap.original_position))?;
            decisions.push(exact_decision(
                gap,
                position,
                DecisionReason::NotSelected,
                None,
            ));
            continue;
        }

        for candidate in selected_candidates {
            let position = final_route
                .iter()
                .position(|track| track == candidate)
                .ok_or(PreviewError::FinalRouteInvalid(
                    "selected bridge is absent from the final route",
                ))?;
            let semantics = gap
                .semantics
                .candidates
                .iter()
                .find(|item| item.candidate == *candidate)
                .ok_or(PreviewError::FinalRouteInvalid(
                    "selected bridge has no frozen semantic evidence",
                ))?
                .clone();
            let mut route_without_candidate = final_route.to_vec();
            route_without_candidate.remove(position);
            let left_position = route_without_candidate
                .iter()
                .position(|track| *track == gap.left)
                .ok_or(PreviewError::InvalidOriginalGap(gap.original_position))?;
            let frozen_matrix = gap_context_matrix(
                &route_without_candidate,
                left_position + 1,
                tracks,
                learned_matrix,
                config,
            )
            .map_err(PreviewError::Scoring)?;
            let evaluation = rank_for_evolving_route(
                &route_without_candidate,
                position,
                std::slice::from_ref(&semantics),
                GapRankingContext {
                    scoring: ExactScoringContext {
                        tracks,
                        learned_matrix,
                        config,
                        reference,
                    },
                    frozen_matrix: frozen_matrix.as_ref(),
                },
                selection_config.guidance(),
                selection_config.variation(),
                EvolvingAcceptance::FullBridge,
            )?
            .into_iter()
            .next()
            .ok_or(PreviewError::FinalRouteInvalid(
                "selected bridge has no final contextual evaluation",
            ))?;
            if !evaluation.accepted {
                return Err(PreviewError::FinalRouteInvalid(
                    "selected bridge fails final contextual validation",
                ));
            }
            decisions.push(exact_decision(
                gap,
                position,
                DecisionReason::Selected,
                Some(SelectedBridge {
                    semantics,
                    evaluation,
                }),
            ));
        }
    }
    Ok(decisions)
}

fn select_exact_count_multi_gap_bridges(
    original_route: &[usize],
    gaps: &[AutomaticGap],
    selection_config: &ExactSelectionConfig,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<ExactSelection, PreviewError> {
    let mut ordered_gaps = gaps.to_vec();
    ordered_gaps.sort_by_key(|gap| gap.original_position);
    let unique_candidates = ordered_gaps
        .iter()
        .flat_map(|gap| gap.semantics.candidates.iter())
        .map(|candidate| candidate.candidate)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let structural_upper_bound = ordered_gaps
        .len()
        .saturating_mul(selection_config.max_tracks_per_gap)
        .min(unique_candidates);
    let initial_objective = gap_search_objective(
        original_route,
        original_route,
        tracks,
        learned_matrix,
        config,
    )?;
    let mut states = vec![MultiExactState {
        route: original_route.to_vec(),
        gap_selections: Vec::with_capacity(ordered_gaps.len()),
        objective: initial_objective,
    }];
    let mut evaluated_states = 1usize;
    let mut retained_states = 1usize;

    for gap in &ordered_gaps {
        let batches = states
            .par_iter()
            .map(|state| {
                let left_position = state
                    .route
                    .iter()
                    .position(|track| *track == gap.left)
                    .ok_or(PreviewError::InvalidOriginalGap(gap.original_position))?;
                let frozen_matrix = gap_context_matrix(
                    &state.route,
                    left_position + 1,
                    tracks,
                    learned_matrix,
                    config,
                )
                .map_err(PreviewError::Scoring)?;
                let already_added = state.route.len() - original_route.len();
                let depth_limit = selection_config
                    .requested_added_tracks
                    .saturating_sub(already_added)
                    .min(selection_config.max_tracks_per_gap);
                let mut completed = Vec::new();
                let mut frontier = vec![(state.clone(), Vec::<usize>::new(), true)];

                for depth in 0..=depth_limit {
                    for (variant, selected, right_connected) in &frontier {
                        if *right_connected {
                            let mut finalized = variant.clone();
                            finalized.gap_selections.push(selected.clone());
                            completed.push(finalized);
                        }
                    }
                    if depth == depth_limit || frontier.is_empty() {
                        break;
                    }

                    let mut next = Vec::new();
                    for (variant, selected, _) in frontier {
                        let position = gap_right_position(&variant.route, gap)
                            .ok_or(PreviewError::InvalidOriginalGap(gap.original_position))?;
                        let evaluations = rank_for_evolving_route(
                            &variant.route,
                            position,
                            &gap.semantics.candidates,
                            GapRankingContext {
                                scoring: ExactScoringContext {
                                    tracks,
                                    learned_matrix,
                                    config,
                                    reference,
                                },
                                frozen_matrix: frozen_matrix.as_ref(),
                            },
                            selection_config.guidance(),
                            selection_config.variation(),
                            EvolvingAcceptance::ReachableFromLeft,
                        )?;
                        for evaluation in evaluations
                            .into_iter()
                            .filter(|candidate| {
                                EvolvingAcceptance::ReachableFromLeft.accepts(candidate, config)
                            })
                            .take(selection_config.candidate_limit)
                        {
                            let mut inserted = variant.clone();
                            inserted.route.insert(position, evaluation.candidate);
                            inserted.objective = gap_search_objective(
                                &inserted.route,
                                original_route,
                                tracks,
                                learned_matrix,
                                config,
                            )?;
                            let mut inserted_selection = selected.clone();
                            inserted_selection.push(evaluation.candidate);
                            next.push((inserted, inserted_selection, evaluation.accepted));
                        }
                    }
                    next.sort_by(|(left, left_selection, _), (right, right_selection, _)| {
                        if multi_exact_state_precedes(left, right) {
                            std::cmp::Ordering::Less
                        } else if multi_exact_state_precedes(right, left) {
                            std::cmp::Ordering::Greater
                        } else {
                            left_selection.cmp(right_selection)
                        }
                    });
                    next.dedup_by(|(left, _, _), (right, _, _)| left.route == right.route);
                    next.truncate(selection_config.beam_width);
                    frontier = next;
                }
                let evaluated = completed.len();
                Ok((completed, evaluated))
            })
            .collect::<Vec<Result<_, PreviewError>>>();

        let mut buckets = BTreeMap::<usize, Vec<MultiExactState>>::new();
        for batch in batches {
            let (expanded, evaluated) = batch?;
            evaluated_states += evaluated;
            for state in expanded {
                let added = state.route.len() - original_route.len();
                buckets.entry(added).or_default().push(state);
            }
        }
        states.clear();
        for bucket in buckets.values_mut() {
            sort_and_prune_multi_states(bucket, selection_config.beam_width);
            retained_states += bucket.len();
            states.append(bucket);
        }
    }

    let maximum_additions_found = states
        .iter()
        .map(|state| state.route.len() - original_route.len())
        .max()
        .unwrap_or(0);
    let selected = states
        .into_iter()
        .filter(|state| {
            state.route.len() - original_route.len() == selection_config.requested_added_tracks
        })
        .min_by(|left, right| {
            if multi_exact_state_precedes(left, right) {
                std::cmp::Ordering::Less
            } else if multi_exact_state_precedes(right, left) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    let (final_route, decisions) = if let Some(state) = selected {
        let decisions = final_exact_decisions(
            &state.route,
            &ordered_gaps,
            &state.gap_selections,
            ExactScoringContext {
                tracks,
                learned_matrix,
                config,
                reference,
            },
            selection_config,
        )?;
        (Some(state.route), decisions)
    } else {
        (None, Vec::new())
    };

    Ok(ExactSelection {
        requested_added_tracks: selection_config.requested_added_tracks,
        final_route,
        decisions,
        endpoint_decisions: Vec::new(),
        stats: ExactSearchStats {
            max_tracks_per_gap: selection_config.max_tracks_per_gap,
            evaluated_states,
            retained_states,
            maximum_additions_found,
            structural_upper_bound,
        },
    })
}

fn anchored_path_error(error: AnchoredPathError) -> PreviewError {
    match error {
        AnchoredPathError::InvalidConfig(message) => PreviewError::InvalidExactConfig(message),
        AnchoredPathError::InvalidRoute(message) => PreviewError::FinalRouteInvalid(message),
        AnchoredPathError::InvalidTrackIndex(_) => {
            PreviewError::FinalRouteInvalid("anchored path contains an unknown track index")
        }
    }
}

/// Searches the final source transition as a bounded, fixed-matrix path.
/// Acoustic distances are supplied by one caller-owned index, so all bridge
/// counts share preprocessing and the quick action remains latency-bounded.
pub fn select_destination_bridge_routes<F>(
    original_route: &[usize],
    gap: &AutomaticGap,
    max_added_tracks: usize,
    selection_config: &ExactSelectionConfig,
    repeat: DestinationRepeatContext<'_>,
    scoring: ExactScoringContext<'_>,
    adjacent_distance: F,
) -> Result<Vec<DestinationRouteOption>, PreviewError>
where
    F: Fn(usize, usize) -> f64 + Sync + Copy,
{
    if original_route.len() < 2
        || original_route[original_route.len() - 2] != gap.left
        || original_route.last().copied() != Some(gap.right)
    {
        return Err(PreviewError::FinalRouteInvalid(
            "destination search requires the final source transition as its only gap",
        ));
    }
    if max_added_tracks > MAX_EXACT_TRACKS_PER_GAP {
        return Err(PreviewError::InvalidExactConfig(
            "destination bridge budget exceeds the supported limit",
        ));
    }
    if selection_config.candidate_limit == 0 || selection_config.beam_width == 0 {
        return Err(PreviewError::InvalidExactConfig(
            "destination candidate and beam limits must be at least one",
        ));
    }

    let prefix = &original_route[..original_route.len() - 1];
    let candidates = gap
        .semantics
        .candidates
        .iter()
        .map(|candidate| AnchoredPathCandidate {
            track: candidate.candidate,
            semantic_support: candidate.guidance_score(
                selection_config.track_guidance_percent,
                selection_config.artist_guidance_percent,
            ),
        })
        .collect::<Vec<_>>();
    let anchored_options = search_anchored_paths(
        AnchoredPathRequest {
            route_prefix: prefix,
            immutable_history: repeat.history_route,
            unavailable_tracks: original_route,
            left_anchor: gap.left,
            right_anchor: gap.right,
            candidates: &candidates,
            tracks: scoring.tracks,
            config: AnchoredPathSearchConfig {
                max_intermediates: max_added_tracks,
                candidate_limit: selection_config.candidate_limit,
                beam_width: selection_config.beam_width,
                alternatives_per_count: 1,
                variation_percent: selection_config.variation_percent,
                generation_seed: selection_config.generation_seed,
                artist_window: scoring.config.artist_window,
                album_window: scoring.config.album_window,
                track_window: repeat.track_window,
            },
        },
        adjacent_distance,
    )
    .map_err(anchored_path_error)?;

    let mut options = Vec::with_capacity(anchored_options.len());
    for anchored in anchored_options {
        let requested = anchored.intermediates.len();
        let decisions = final_exact_decisions(
            &anchored.route,
            std::slice::from_ref(gap),
            std::slice::from_ref(&anchored.intermediates),
            scoring,
            selection_config,
        )?;
        options.push(DestinationRouteOption {
            added_track_count: requested,
            selection: ExactSelection {
                requested_added_tracks: requested,
                final_route: Some(anchored.route),
                decisions,
                endpoint_decisions: Vec::new(),
                stats: ExactSearchStats {
                    max_tracks_per_gap: max_added_tracks.max(1),
                    evaluated_states: anchored.stats.evaluated_states,
                    retained_states: anchored.stats.retained_states,
                    maximum_additions_found: requested,
                    structural_upper_bound: anchored.stats.structural_upper_bound,
                },
            },
            adjacent_transition_sum: anchored.transition_sum,
            adjacent_worst_transition: anchored.worst_transition,
        });
    }
    Ok(options)
}

/// Searches an excursion through one mandatory waypoint and back to a locked
/// rejoin anchor. The bridge budget is shared across both adjacent gaps.
/// Each outward option is carried into the return search so uniqueness and
/// repeat windows constrain the complete audible route.
pub fn select_destination_waypoint_routes<F>(
    original_route: &[usize],
    gaps: &[AutomaticGap],
    max_added_tracks: usize,
    selection_config: &ExactSelectionConfig,
    repeat: DestinationRepeatContext<'_>,
    scoring: ExactScoringContext<'_>,
    adjacent_distance: F,
) -> Result<Vec<DestinationRouteOption>, PreviewError>
where
    F: Fn(usize, usize) -> f64 + Sync + Copy,
{
    if original_route.len() < 3 || gaps.len() != 2 {
        return Err(PreviewError::FinalRouteInvalid(
            "waypoint destination search requires two locked adjacent gaps",
        ));
    }
    let start_position = original_route.len() - 3;
    let start = original_route[start_position];
    let waypoint = original_route[start_position + 1];
    let rejoin = original_route[start_position + 2];
    if gaps[0].left != start
        || gaps[0].right != waypoint
        || gaps[1].left != waypoint
        || gaps[1].right != rejoin
    {
        return Err(PreviewError::FinalRouteInvalid(
            "waypoint destination gaps do not match start, waypoint, and rejoin anchors",
        ));
    }
    if max_added_tracks > MAX_EXACT_TRACKS_PER_GAP {
        return Err(PreviewError::InvalidExactConfig(
            "waypoint bridge budget exceeds the supported total limit",
        ));
    }

    let outward_source = &original_route[..original_route.len() - 1];
    let outward_options = select_destination_bridge_routes(
        outward_source,
        &gaps[0],
        max_added_tracks,
        selection_config,
        repeat,
        scoring,
        adjacent_distance,
    )?;
    let option_groups = outward_options
        .into_par_iter()
        .map(
            |outward| -> Result<Vec<DestinationRouteOption>, PreviewError> {
                let outward_route = outward
                    .selection
                    .final_route
                    .as_ref()
                    .expect("destination options always contain a route");
                let outward_count = outward.added_track_count;
                let mut return_source = outward_route.clone();
                return_source.push(rejoin);
                let remaining = max_added_tracks.saturating_sub(outward_count);
                let return_options = select_destination_bridge_routes(
                    &return_source,
                    &gaps[1],
                    remaining,
                    selection_config,
                    repeat,
                    scoring,
                    adjacent_distance,
                )?;

                let mut combined = Vec::with_capacity(return_options.len());
                for returning in return_options {
                    let total = outward_count + returning.added_track_count;
                    let final_route = returning
                        .selection
                        .final_route
                        .as_ref()
                        .expect("destination options always contain a route")
                        .clone();
                    let waypoint_position = final_route
                        .iter()
                        .position(|track| *track == waypoint)
                        .expect("the mandatory waypoint remains in the route");
                    let rejoin_position = final_route
                        .iter()
                        .position(|track| *track == rejoin)
                        .expect("the rejoin anchor remains in the route");
                    let route_start = final_route
                        .iter()
                        .position(|track| *track == start)
                        .expect("the start anchor remains in the route");
                    let outward_bridges = final_route[route_start + 1..waypoint_position].to_vec();
                    let return_bridges =
                        final_route[waypoint_position + 1..rejoin_position].to_vec();
                    let decisions = final_exact_decisions(
                        &final_route,
                        gaps,
                        &[outward_bridges, return_bridges],
                        scoring,
                        selection_config,
                    )?;
                    let (transition_sum, worst_transition) = final_route
                        [route_start..=rejoin_position]
                        .windows(2)
                        .map(|edge| adjacent_distance(edge[0], edge[1]))
                        .fold((0.0_f64, 0.0_f64), |(sum, worst), distance| {
                            (sum + distance, worst.max(distance))
                        });
                    let option = DestinationRouteOption {
                        added_track_count: total,
                        selection: ExactSelection {
                            requested_added_tracks: total,
                            final_route: Some(final_route),
                            decisions,
                            endpoint_decisions: Vec::new(),
                            stats: ExactSearchStats {
                                max_tracks_per_gap: max_added_tracks.max(1),
                                evaluated_states: outward
                                    .selection
                                    .stats
                                    .evaluated_states
                                    .saturating_add(returning.selection.stats.evaluated_states),
                                retained_states: outward
                                    .selection
                                    .stats
                                    .retained_states
                                    .saturating_add(returning.selection.stats.retained_states),
                                maximum_additions_found: total,
                                structural_upper_bound: max_added_tracks,
                            },
                        },
                        adjacent_transition_sum: transition_sum,
                        adjacent_worst_transition: worst_transition,
                    };
                    combined.push(option);
                }
                Ok(combined)
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let mut best_by_count = BTreeMap::<usize, DestinationRouteOption>::new();
    for option in option_groups.into_iter().flatten() {
        let total = option.added_track_count;
        let replace = best_by_count.get(&total).is_none_or(|current| {
            option
                .adjacent_worst_transition
                .total_cmp(&current.adjacent_worst_transition)
                .then_with(|| {
                    option
                        .adjacent_transition_sum
                        .total_cmp(&current.adjacent_transition_sum)
                })
                .then_with(|| {
                    option
                        .selection
                        .final_route
                        .as_ref()
                        .cmp(&current.selection.final_route.as_ref())
                })
                .is_lt()
        });
        if replace {
            best_by_count.insert(total, option);
        }
    }

    Ok(best_by_count.into_values().collect())
}
pub fn select_exact_count_bridges(
    original_route: &[usize],
    gaps: &[AutomaticGap],
    selection_config: &ExactSelectionConfig,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<ExactSelection, PreviewError> {
    if selection_config.candidate_limit == 0 {
        return Err(PreviewError::InvalidExactConfig(
            "exact-count candidate limit must be at least one",
        ));
    }
    if selection_config.beam_width == 0 {
        return Err(PreviewError::InvalidExactConfig(
            "exact-count beam width must be at least one",
        ));
    }
    if selection_config.max_tracks_per_gap == 0 {
        return Err(PreviewError::InvalidExactConfig(
            "exact-count max tracks per gap must be at least one",
        ));
    }
    if selection_config.max_tracks_per_gap > MAX_EXACT_TRACKS_PER_GAP {
        return Err(PreviewError::InvalidExactConfig(
            "exact-count max tracks per gap exceeds the supported limit",
        ));
    }
    if selection_config.max_tracks_per_gap == 1 {
        select_exact_count_single_gap_bridges(
            original_route,
            gaps,
            selection_config,
            tracks,
            learned_matrix,
            config,
            reference,
        )
    } else {
        select_exact_count_multi_gap_bridges(
            original_route,
            gaps,
            selection_config,
            tracks,
            learned_matrix,
            config,
            reference,
        )
    }
}

fn rank_endpoint_for_route(
    route: &[usize],
    slot: EndpointSlot,
    endpoint: &ExactEndpointSlot,
    candidate_limit: usize,
    guidance: GuidanceConfig,
    variation: VariationConfig,
    scoring: ExactScoringContext<'_>,
) -> Result<Vec<SelectedEndpoint>, PreviewError> {
    let ExactScoringContext {
        tracks,
        learned_matrix,
        config,
        reference,
    } = scoring;
    let semantics_by_candidate = endpoint
        .semantics
        .candidates
        .iter()
        .map(|candidate| (candidate.candidate, candidate))
        .collect::<HashMap<_, _>>();
    let candidates = endpoint
        .semantics
        .candidates
        .iter()
        .map(|candidate| candidate.candidate)
        .collect::<Vec<_>>();
    let mut evaluations = rank_endpoint_candidates(
        route,
        slot,
        &candidates,
        tracks,
        learned_matrix,
        config,
        reference,
    )
    .map_err(PreviewError::Scoring)?;
    evaluations.sort_by(|left, right| {
        right
            .accepted
            .cmp(&left.accepted)
            .then_with(|| {
                semantics_by_candidate[&left.candidate]
                    .adjusted_percentile(
                        left.percentile,
                        guidance.track_percent,
                        guidance.artist_percent,
                    )
                    .total_cmp(
                        &semantics_by_candidate[&right.candidate].adjusted_percentile(
                            right.percentile,
                            guidance.track_percent,
                            guidance.artist_percent,
                        ),
                    )
            })
            .then_with(|| left.percentile.total_cmp(&right.percentile))
            .then_with(|| left.candidate.cmp(&right.candidate))
    });
    let accepted = evaluations.iter().take_while(|item| item.accepted).count();
    let pool = varied_pool_length(accepted, variation);
    if pool > 1 {
        let position = match slot {
            EndpointSlot::Opening => 0,
            EndpointSlot::Closing => route.len(),
        };
        evaluations[..pool]
            .sort_by_key(|item| variation_key(variation.seed, route, position, item.candidate));
    }
    Ok(evaluations
        .into_iter()
        .filter(|evaluation| evaluation.accepted)
        .take(candidate_limit)
        .map(|evaluation| SelectedEndpoint {
            semantics: semantics_by_candidate[&evaluation.candidate].clone(),
            evaluation,
        })
        .collect())
}

#[derive(Clone, Debug)]
struct EndpointExactVariant {
    route: Vec<usize>,
    decisions: Vec<GapDecision>,
    endpoint_decisions: Vec<EndpointDecision>,
    objective: f64,
}

fn endpoint_variant_precedes(left: &EndpointExactVariant, right: &EndpointExactVariant) -> bool {
    left.objective.total_cmp(&right.objective).is_lt()
        || (left.objective == right.objective && left.route < right.route)
}

pub fn select_exact_count_bridges_with_endpoints(
    original_route: &[usize],
    gaps: &[AutomaticGap],
    selection_config: &ExactSelectionConfig,
    endpoints: &ExactEndpointSlots,
    scoring: ExactScoringContext<'_>,
) -> Result<ExactSelection, PreviewError> {
    let ExactScoringContext {
        tracks,
        learned_matrix,
        config,
        reference,
    } = scoring;
    if endpoints.opening.is_none() && endpoints.closing.is_none() {
        return select_exact_count_bridges(
            original_route,
            gaps,
            selection_config,
            tracks,
            learned_matrix,
            config,
            reference,
        );
    }

    let opening_choices = if endpoints.opening.is_some() {
        [false, true].as_slice()
    } else {
        [false].as_slice()
    };
    let closing_choices = if endpoints.closing.is_some() {
        [false, true].as_slice()
    } else {
        [false].as_slice()
    };
    let mut variants = Vec::new();
    let mut ordered_gaps = gaps.to_vec();
    ordered_gaps.sort_by_key(|gap| gap.original_position);
    let mut evaluated_states = 0usize;
    let mut retained_states = 0usize;
    let mut maximum_additions_found = 0usize;

    for use_opening in opening_choices {
        for use_closing in closing_choices {
            let endpoint_count = usize::from(*use_opening) + usize::from(*use_closing);
            if endpoint_count > selection_config.requested_added_tracks {
                continue;
            }
            let internal_requested = selection_config.requested_added_tracks - endpoint_count;
            let internal_config = ExactSelectionConfig {
                requested_added_tracks: internal_requested,
                ..*selection_config
            };
            let internal = select_exact_count_bridges(
                original_route,
                gaps,
                &internal_config,
                tracks,
                learned_matrix,
                config,
                reference,
            )?;
            evaluated_states += internal.stats.evaluated_states;
            retained_states += internal.stats.retained_states;
            maximum_additions_found =
                maximum_additions_found.max(internal.stats.maximum_additions_found);
            let Some(internal_route) = internal.final_route else {
                continue;
            };

            let opening_options = if *use_opening {
                rank_endpoint_for_route(
                    &internal_route,
                    EndpointSlot::Opening,
                    endpoints.opening.as_ref().expect("opening slot is enabled"),
                    selection_config.candidate_limit,
                    selection_config.guidance(),
                    selection_config.variation(),
                    scoring,
                )?
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>()
            } else {
                vec![None]
            };
            for opening in opening_options {
                let mut opened_route = internal_route.clone();
                let mut endpoint_decisions = Vec::new();
                if let Some(selected) = &opening {
                    opened_route.insert(0, selected.evaluation.candidate);
                }
                if let Some(endpoint) = &endpoints.opening {
                    endpoint_decisions.push(EndpointDecision {
                        slot: EndpointSlot::Opening,
                        anchor: endpoint.anchor,
                        semantic_pool: endpoint.semantics.pool,
                        reason: if opening.is_some() {
                            DecisionReason::Selected
                        } else {
                            DecisionReason::NotSelected
                        },
                        selected: opening.clone(),
                    });
                }

                let closing_options = if *use_closing {
                    rank_endpoint_for_route(
                        &opened_route,
                        EndpointSlot::Closing,
                        endpoints.closing.as_ref().expect("closing slot is enabled"),
                        selection_config.candidate_limit,
                        selection_config.guidance(),
                        selection_config.variation(),
                        scoring,
                    )?
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>()
                } else {
                    vec![None]
                };
                for closing in closing_options {
                    let mut route = opened_route.clone();
                    let mut candidate_decisions = endpoint_decisions.clone();
                    if let Some(selected) = &closing {
                        route.push(selected.evaluation.candidate);
                    }
                    if let Some(endpoint) = &endpoints.closing {
                        candidate_decisions.push(EndpointDecision {
                            slot: EndpointSlot::Closing,
                            anchor: endpoint.anchor,
                            semantic_pool: endpoint.semantics.pool,
                            reason: if closing.is_some() {
                                DecisionReason::Selected
                            } else {
                                DecisionReason::NotSelected
                            },
                            selected: closing.clone(),
                        });
                    }
                    let objective = gap_search_objective(
                        &route,
                        original_route,
                        tracks,
                        learned_matrix,
                        config,
                    )?;
                    let gap_selections = ordered_gaps
                        .iter()
                        .map(|gap| {
                            internal
                                .decisions
                                .iter()
                                .filter(|decision| {
                                    decision.original_position == gap.original_position
                                })
                                .filter_map(|decision| {
                                    decision
                                        .selected
                                        .as_ref()
                                        .map(|selected| selected.evaluation.candidate)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let decisions = match final_exact_decisions(
                        &route,
                        &ordered_gaps,
                        &gap_selections,
                        scoring,
                        selection_config,
                    ) {
                        Ok(decisions) => decisions,
                        Err(PreviewError::FinalRouteInvalid(_)) => continue,
                        Err(error) => return Err(error),
                    };
                    evaluated_states += 1;
                    retained_states += 1;
                    maximum_additions_found =
                        maximum_additions_found.max(route.len() - original_route.len());
                    variants.push(EndpointExactVariant {
                        route,
                        decisions,
                        endpoint_decisions: candidate_decisions,
                        objective,
                    });
                }
            }
        }
    }

    let unique_candidates = gaps
        .iter()
        .flat_map(|gap| gap.semantics.candidates.iter())
        .chain(
            endpoints
                .opening
                .iter()
                .flat_map(|endpoint| endpoint.semantics.candidates.iter()),
        )
        .chain(
            endpoints
                .closing
                .iter()
                .flat_map(|endpoint| endpoint.semantics.candidates.iter()),
        )
        .map(|candidate| candidate.candidate)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let endpoint_capacity =
        usize::from(endpoints.opening.is_some()) + usize::from(endpoints.closing.is_some());
    let structural_upper_bound = gaps
        .len()
        .saturating_mul(selection_config.max_tracks_per_gap)
        .saturating_add(endpoint_capacity)
        .min(unique_candidates);
    let selected = variants.into_iter().min_by(|left, right| {
        if endpoint_variant_precedes(left, right) {
            std::cmp::Ordering::Less
        } else if endpoint_variant_precedes(right, left) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let (final_route, decisions, endpoint_decisions) = selected
        .map(|state| (Some(state.route), state.decisions, state.endpoint_decisions))
        .unwrap_or_else(|| (None, Vec::new(), Vec::new()));

    Ok(ExactSelection {
        requested_added_tracks: selection_config.requested_added_tracks,
        final_route,
        decisions,
        endpoint_decisions,
        stats: ExactSearchStats {
            max_tracks_per_gap: selection_config.max_tracks_per_gap,
            evaluated_states: evaluated_states.max(1),
            retained_states: retained_states.max(1),
            maximum_additions_found,
            structural_upper_bound,
        },
    })
}

fn select_exact_count_single_gap_bridges(
    original_route: &[usize],
    gaps: &[AutomaticGap],
    selection_config: &ExactSelectionConfig,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<ExactSelection, PreviewError> {
    if selection_config.candidate_limit == 0 {
        return Err(PreviewError::InvalidExactConfig(
            "exact-count candidate limit must be at least one",
        ));
    }
    if selection_config.beam_width == 0 {
        return Err(PreviewError::InvalidExactConfig(
            "exact-count beam width must be at least one",
        ));
    }

    let mut ordered_gaps = gaps.to_vec();
    ordered_gaps.sort_by_key(|gap| gap.original_position);
    let unique_candidates = ordered_gaps
        .iter()
        .flat_map(|gap| gap.semantics.candidates.iter())
        .map(|candidate| candidate.candidate)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let structural_upper_bound = ordered_gaps.len().min(unique_candidates);
    let prune_unreachable_states =
        selection_config.requested_added_tracks <= structural_upper_bound;
    let initial_objective = gap_search_objective(
        original_route,
        original_route,
        tracks,
        learned_matrix,
        config,
    )?;
    let mut states = vec![ExactState {
        route: original_route.to_vec(),
        decisions: Vec::with_capacity(ordered_gaps.len()),
        objective: initial_objective,
    }];
    let mut evaluated_states = 1usize;
    let mut retained_states = 1usize;
    let mut maximum_additions_seen = 0usize;

    let gap_count = ordered_gaps.len();
    for (gap_index, gap) in ordered_gaps.iter().enumerate() {
        let remaining_after_gap = gap_count.saturating_sub(gap_index + 1);
        let batches = states
            .par_iter()
            .map(|state| {
                let position = route_position(&state.route, gap)
                    .ok_or(PreviewError::InvalidOriginalGap(gap.original_position))?;
                let mut expanded = Vec::new();

                let added = state.route.len() - original_route.len();
                let mut local_max_added = added;
                if !prune_unreachable_states
                    || added + remaining_after_gap >= selection_config.requested_added_tracks
                {
                    let mut skipped = state.clone();
                    skipped.decisions.push(exact_decision(
                        gap,
                        position,
                        DecisionReason::NotSelected,
                        None,
                    ));
                    expanded.push(skipped);
                }
                if added < selection_config.requested_added_tracks
                    && !gap.semantics.candidates.is_empty()
                {
                    let frozen_matrix =
                        gap_context_matrix(&state.route, position, tracks, learned_matrix, config)
                            .map_err(PreviewError::Scoring)?;
                    let evaluations = rank_for_evolving_route(
                        &state.route,
                        position,
                        &gap.semantics.candidates,
                        GapRankingContext {
                            scoring: ExactScoringContext {
                                tracks,
                                learned_matrix,
                                config,
                                reference,
                            },
                            frozen_matrix: frozen_matrix.as_ref(),
                        },
                        selection_config.guidance(),
                        selection_config.variation(),
                        EvolvingAcceptance::FullBridge,
                    )?;
                    for evaluation in evaluations
                        .into_iter()
                        .filter(|candidate| candidate.accepted)
                        .take(selection_config.candidate_limit)
                    {
                        let semantics = gap
                            .semantics
                            .candidates
                            .iter()
                            .find(|candidate| candidate.candidate == evaluation.candidate)
                            .expect("every evaluation has frozen candidate semantics")
                            .clone();
                        let mut inserted = state.clone();
                        inserted.route.insert(position, evaluation.candidate);
                        local_max_added =
                            local_max_added.max(inserted.route.len() - original_route.len());
                        inserted.objective = gap_search_objective(
                            &inserted.route,
                            original_route,
                            tracks,
                            learned_matrix,
                            config,
                        )?;
                        inserted.decisions.push(exact_decision(
                            gap,
                            position,
                            DecisionReason::Selected,
                            Some(SelectedBridge {
                                semantics,
                                evaluation,
                            }),
                        ));
                        expanded.push(inserted);
                    }
                }
                Ok((expanded, local_max_added))
            })
            .collect::<Vec<Result<_, PreviewError>>>();

        let mut buckets = BTreeMap::<usize, Vec<ExactState>>::new();
        for batch in batches {
            let (expanded, local_max_added) = batch?;
            maximum_additions_seen = maximum_additions_seen.max(local_max_added);
            for state in expanded {
                evaluated_states += 1;
                let added = state.route.len() - original_route.len();
                buckets.entry(added).or_default().push(state);
            }
        }
        states.clear();
        for bucket in buckets.values_mut() {
            bucket.sort_by(|left, right| {
                if exact_state_precedes(left, right) {
                    std::cmp::Ordering::Less
                } else if exact_state_precedes(right, left) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            bucket.dedup_by(|left, right| left.route == right.route);
            bucket.truncate(selection_config.beam_width);
            retained_states += bucket.len();
            states.append(bucket);
        }
    }

    let maximum_additions_found = states
        .iter()
        .map(|state| state.route.len() - original_route.len())
        .max()
        .unwrap_or(0)
        .max(maximum_additions_seen);
    let selected = states
        .into_iter()
        .filter(|state| {
            state.route.len() - original_route.len() == selection_config.requested_added_tracks
        })
        .min_by(|left, right| {
            if exact_state_precedes(left, right) {
                std::cmp::Ordering::Less
            } else if exact_state_precedes(right, left) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    let (final_route, decisions) = selected
        .map(|state| (Some(state.route), state.decisions))
        .unwrap_or_else(|| (None, Vec::new()));

    Ok(ExactSelection {
        requested_added_tracks: selection_config.requested_added_tracks,
        final_route,
        decisions,
        endpoint_decisions: Vec::new(),
        stats: ExactSearchStats {
            max_tracks_per_gap: 1,
            evaluated_states,
            retained_states,
            maximum_additions_found,
            structural_upper_bound,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::build_frozen_reference;
    use crate::semantic::SemanticTier;

    fn track(value: f32, artist: &str) -> RouteTrack {
        RouteTrack {
            features: std::array::from_fn(|index| value + index as f32 / 100.0),
            artist_key: artist.to_owned(),
            album_key: format!("album-{artist}"),
        }
    }

    fn semantics(candidate: usize) -> CandidateSemantics {
        CandidateSemantics {
            candidate,
            tier: SemanticTier::BlissOnly,
            evidence: Vec::new(),
        }
    }

    fn gap(position: usize, left: usize, right: usize, candidate: usize) -> AutomaticGap {
        AutomaticGap {
            original_position: position,
            left,
            right,
            direct_distance: 10.0,
            direct_percentile: 1.0,
            semantics: GapEvidence {
                pool: SemanticPool::BlissOnly,
                candidates: vec![semantics(candidate)],
            },
        }
    }

    #[test]
    fn bounded_variation_is_reproducible_and_seeded() {
        let strict = VariationConfig::default();
        let low = VariationConfig {
            percent: 1,
            seed: 101,
            minimum_pool: 9,
        };
        let full = VariationConfig {
            percent: 100,
            seed: 101,
            minimum_pool: 9,
        };
        assert_eq!(varied_pool_length(0, full), 0);
        assert_eq!(varied_pool_length(20, strict), 1);
        assert_eq!(varied_pool_length(20, low), 9);
        assert_eq!(varied_pool_length(20, full), 20);
        assert_eq!(varied_pool_length(100, full), 32);

        let route = [3, 8, 13];
        let ordered = |seed| {
            let mut candidates = (1..=16).collect::<Vec<_>>();
            candidates.sort_by_key(|candidate| variation_key(seed, &route, 2, *candidate));
            candidates
        };
        assert_eq!(ordered(101), ordered(101));
        assert_ne!(ordered(101), ordered(202));
    }

    #[test]
    fn selection_is_left_to_right_budgeted_and_worker_deterministic() {
        let tracks = vec![
            track(0.0, "a"),
            track(1.0, "bridge-a"),
            track(2.0, "b"),
            track(3.0, "bridge-b"),
            track(4.0, "c"),
        ];
        let route = [0, 2, 4];
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 2,
            learned_percent: 20,
            artist_window: 1,
            album_window: 1,
            max_leg_percentile: 0.70,
            max_detour_percentile: 1.30,
            gap_context_mode: crate::bridge::GapContextMode::Rolling,
        };
        let reference = build_frozen_reference(&route, &route, &tracks, &matrix, &config).unwrap();
        let gaps = [gap(1, 0, 2, 1), gap(2, 2, 4, 3)];
        let selection_config = AutomaticSelectionConfig {
            max_added_tracks: 1,
            trigger_percentile: 0.70,
            track_guidance_percent: 0,
            artist_guidance_percent: 0,
            variation_percent: 0,
            generation_seed: 20_260_721,
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                select_automatic_bridges(
                    &route,
                    &gaps,
                    &selection_config,
                    &tracks,
                    &matrix,
                    &config,
                    &reference,
                )
            })
            .unwrap();
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                select_automatic_bridges(
                    &route,
                    &gaps,
                    &selection_config,
                    &tracks,
                    &matrix,
                    &config,
                    &reference,
                )
            })
            .unwrap();
        assert_eq!(one, four);
        assert_eq!(one.final_route, vec![0, 1, 2, 4]);
        assert_eq!(one.decisions[0].reason, DecisionReason::Selected);
        assert_eq!(one.decisions[1].reason, DecisionReason::BudgetExhausted);
        assert!(
            one.decisions[0]
                .selected
                .as_ref()
                .unwrap()
                .evaluation
                .detour_percentile
                < 1.0
        );
    }

    #[test]
    fn below_threshold_gap_is_a_visible_no_op() {
        let tracks = vec![track(0.0, "a"), track(1.0, "b"), track(2.0, "c")];
        let route = [0, 2];
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 1,
            learned_percent: 20,
            artist_window: 1,
            album_window: 1,
            max_leg_percentile: 0.70,
            max_detour_percentile: 1.30,
            gap_context_mode: crate::bridge::GapContextMode::Rolling,
        };
        let reference = build_frozen_reference(&route, &route, &tracks, &matrix, &config).unwrap();
        let mut smooth = gap(1, 0, 2, 1);
        smooth.direct_percentile = 0.70;
        let selection_config = AutomaticSelectionConfig {
            max_added_tracks: 1,
            trigger_percentile: 0.70,
            track_guidance_percent: 0,
            artist_guidance_percent: 0,
            variation_percent: 0,
            generation_seed: 20_260_721,
        };
        let selected = select_automatic_bridges(
            &route,
            &[smooth],
            &selection_config,
            &tracks,
            &matrix,
            &config,
            &reference,
        )
        .unwrap();
        assert_eq!(selected.final_route, route);
        assert_eq!(selected.decisions[0].reason, DecisionReason::BelowThreshold);
    }

    #[test]
    fn exact_count_search_is_worker_deterministic_and_not_partial() {
        let tracks = vec![
            track(0.0, "a"),
            track(1.0, "bridge-a"),
            track(2.0, "b"),
            track(3.0, "bridge-b"),
            track(4.0, "c"),
        ];
        let route = [0, 2, 4];
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 2,
            learned_percent: 20,
            artist_window: 1,
            album_window: 1,
            max_leg_percentile: 0.70,
            max_detour_percentile: 1.30,
            gap_context_mode: crate::bridge::GapContextMode::Rolling,
        };
        let reference = build_frozen_reference(&route, &route, &tracks, &matrix, &config).unwrap();
        let gaps = [gap(1, 0, 2, 1), gap(2, 2, 4, 3)];
        let exact = ExactSelectionConfig {
            requested_added_tracks: 2,
            candidate_limit: 2,
            beam_width: 16,
            max_tracks_per_gap: 1,
            track_guidance_percent: 0,
            artist_guidance_percent: 0,
            variation_percent: 0,
            generation_seed: 20_260_721,
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                select_exact_count_bridges(
                    &route, &gaps, &exact, &tracks, &matrix, &config, &reference,
                )
            })
            .unwrap();
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                select_exact_count_bridges(
                    &route, &gaps, &exact, &tracks, &matrix, &config, &reference,
                )
            })
            .unwrap();
        assert_eq!(one, four);
        assert_eq!(one.final_route, Some(vec![0, 1, 2, 3, 4]));
        assert_eq!(
            one.decisions
                .iter()
                .map(|decision| decision.reason)
                .collect::<Vec<_>>(),
            vec![DecisionReason::Selected, DecisionReason::Selected]
        );

        let impossible = select_exact_count_bridges(
            &route,
            &gaps,
            &ExactSelectionConfig {
                requested_added_tracks: 3,
                ..exact
            },
            &tracks,
            &matrix,
            &config,
            &reference,
        )
        .unwrap();
        assert_eq!(impossible.final_route, None);
        assert!(impossible.decisions.is_empty());
        assert_eq!(impossible.stats.maximum_additions_found, 2);
        assert_eq!(impossible.stats.structural_upper_bound, 2);
    }

    #[test]
    fn exact_count_can_route_multiple_bridges_inside_one_preserved_gap() {
        let tracks = vec![
            track(0.0, "anchor-a"),
            track(0.7, "bridge-a"),
            track(1.3, "bridge-b"),
            track(2.0, "anchor-b"),
        ];
        let route = [0, 3];
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 2,
            learned_percent: 20,
            artist_window: 1,
            album_window: 1,
            max_leg_percentile: 1.0,
            max_detour_percentile: 2.0,
            gap_context_mode: crate::bridge::GapContextMode::Rolling,
        };
        let reference =
            build_frozen_reference(&route, &[0, 1, 2, 3], &tracks, &matrix, &config).unwrap();
        let gaps = [AutomaticGap {
            original_position: 1,
            left: 0,
            right: 3,
            direct_distance: 10.0,
            direct_percentile: 1.0,
            semantics: GapEvidence {
                pool: SemanticPool::BlissOnly,
                candidates: vec![semantics(1), semantics(2)],
            },
        }];
        let selection_config = ExactSelectionConfig {
            requested_added_tracks: 2,
            candidate_limit: 2,
            beam_width: 16,
            max_tracks_per_gap: 2,
            track_guidance_percent: 0,
            artist_guidance_percent: 0,
            variation_percent: 0,
            generation_seed: 20_260_721,
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                select_exact_count_bridges(
                    &route,
                    &gaps,
                    &selection_config,
                    &tracks,
                    &matrix,
                    &config,
                    &reference,
                )
            })
            .unwrap();
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                select_exact_count_bridges(
                    &route,
                    &gaps,
                    &selection_config,
                    &tracks,
                    &matrix,
                    &config,
                    &reference,
                )
            })
            .unwrap();

        assert_eq!(one, four);
        let final_route = one.final_route.unwrap();
        assert_eq!(final_route.first(), Some(&0));
        assert_eq!(final_route.last(), Some(&3));
        assert_eq!(final_route.len(), 4);
        assert!(final_route.contains(&1));
        assert!(final_route.contains(&2));
        assert_eq!(one.decisions.len(), 2);
        assert!(one
            .decisions
            .iter()
            .all(|decision| decision.reason == DecisionReason::Selected));
        assert_eq!(one.stats.max_tracks_per_gap, 2);
        assert_eq!(one.stats.structural_upper_bound, 2);

        let single_per_gap = select_exact_count_bridges(
            &route,
            &gaps,
            &ExactSelectionConfig {
                max_tracks_per_gap: 1,
                ..selection_config
            },
            &tracks,
            &matrix,
            &config,
            &reference,
        )
        .unwrap();
        assert!(single_per_gap.final_route.is_none());
        assert_eq!(single_per_gap.stats.structural_upper_bound, 1);
    }

    #[test]
    fn exact_count_can_build_a_path_when_no_single_bridge_reaches_both_anchors() {
        let tracks = vec![
            track(0.0, "anchor-a"),
            track(1.0, "bridge-a"),
            track(2.0, "bridge-b"),
            track(3.0, "anchor-b"),
        ];
        let route = [0, 3];
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 1,
            learned_percent: 20,
            artist_window: 1,
            album_window: 1,
            max_leg_percentile: 0.25,
            max_detour_percentile: 0.50,
            gap_context_mode: crate::bridge::GapContextMode::Rolling,
        };
        let reference =
            build_frozen_reference(&route, &[0, 1, 2, 3], &tracks, &matrix, &config).unwrap();
        let gaps = [AutomaticGap {
            original_position: 1,
            left: 0,
            right: 3,
            direct_distance: 10.0,
            direct_percentile: 1.0,
            semantics: GapEvidence {
                pool: SemanticPool::BlissOnly,
                candidates: vec![semantics(1), semantics(2)],
            },
        }];
        let exact = ExactSelectionConfig {
            requested_added_tracks: 1,
            candidate_limit: 2,
            beam_width: 16,
            max_tracks_per_gap: 2,
            track_guidance_percent: 0,
            artist_guidance_percent: 0,
            variation_percent: 0,
            generation_seed: 20_260_811,
        };

        let one_bridge = select_exact_count_bridges(
            &route, &gaps, &exact, &tracks, &matrix, &config, &reference,
        )
        .unwrap();
        assert_eq!(one_bridge.final_route, None);

        let two_bridges = select_exact_count_bridges(
            &route,
            &gaps,
            &ExactSelectionConfig {
                requested_added_tracks: 2,
                ..exact
            },
            &tracks,
            &matrix,
            &config,
            &reference,
        )
        .unwrap();
        assert_eq!(two_bridges.final_route, Some(vec![0, 1, 2, 3]));
        assert_eq!(two_bridges.stats.maximum_additions_found, 2);
    }

    #[test]
    fn explicit_endpoint_slots_make_an_otherwise_impossible_count_deterministic() {
        let tracks = vec![
            track(1.0, "anchor-a"),
            track(0.0, "opening"),
            track(3.0, "closing"),
            track(2.0, "anchor-b"),
        ];
        let route = [0, 3];
        let matrix = Array2::eye(23);
        let config = BridgeConfig {
            seed_limit: 2,
            learned_percent: 20,
            artist_window: 1,
            album_window: 1,
            max_leg_percentile: 1.0,
            max_detour_percentile: 2.0,
            gap_context_mode: crate::bridge::GapContextMode::Rolling,
        };
        let reference =
            build_frozen_reference(&route, &[0, 1, 2, 3], &tracks, &matrix, &config).unwrap();
        let exact = ExactSelectionConfig {
            requested_added_tracks: 2,
            candidate_limit: 2,
            beam_width: 16,
            max_tracks_per_gap: 1,
            track_guidance_percent: 0,
            artist_guidance_percent: 0,
            variation_percent: 0,
            generation_seed: 20_260_721,
        };
        let without_endpoints =
            select_exact_count_bridges(&route, &[], &exact, &tracks, &matrix, &config, &reference)
                .unwrap();
        assert!(without_endpoints.final_route.is_none());
        assert_eq!(without_endpoints.stats.structural_upper_bound, 0);

        let endpoints = ExactEndpointSlots {
            opening: Some(ExactEndpointSlot {
                anchor: 0,
                semantics: GapEvidence {
                    pool: SemanticPool::BlissOnly,
                    candidates: vec![semantics(1)],
                },
            }),
            closing: Some(ExactEndpointSlot {
                anchor: 3,
                semantics: GapEvidence {
                    pool: SemanticPool::BlissOnly,
                    candidates: vec![semantics(2)],
                },
            }),
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                select_exact_count_bridges_with_endpoints(
                    &route,
                    &[],
                    &exact,
                    &endpoints,
                    ExactScoringContext {
                        tracks: &tracks,
                        learned_matrix: &matrix,
                        config: &config,
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
                select_exact_count_bridges_with_endpoints(
                    &route,
                    &[],
                    &exact,
                    &endpoints,
                    ExactScoringContext {
                        tracks: &tracks,
                        learned_matrix: &matrix,
                        config: &config,
                        reference: &reference,
                    },
                )
            })
            .unwrap();

        assert_eq!(one, four);
        assert_eq!(one.final_route, Some(vec![1, 0, 3, 2]));
        assert_eq!(one.endpoint_decisions.len(), 2);
        assert!(one
            .endpoint_decisions
            .iter()
            .all(|decision| decision.reason == DecisionReason::Selected));
        assert_eq!(one.stats.structural_upper_bound, 2);
    }
}
