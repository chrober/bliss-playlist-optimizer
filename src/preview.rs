// SPDX-License-Identifier: GPL-3.0-only

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;

use crate::bridge::{
    rank_candidates, BridgeCandidateEvaluation, BridgeConfig, BridgeError, FrozenReference,
};
use crate::route::{self, RouteTrack};
use crate::semantic::{CandidateSemantics, GapEvidence, SemanticPool};

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactSelectionConfig {
    pub requested_added_tracks: usize,
    pub candidate_limit: usize,
    pub beam_width: usize,
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
    pub stats: ExactSearchStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewError {
    InvalidOriginalGap(usize),
    InvalidExactConfig(&'static str),
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
            Self::InvalidOriginalGap(_) | Self::InvalidExactConfig(_) => None,
        }
    }
}

fn route_position(route: &[usize], gap: &AutomaticGap) -> Option<usize> {
    route
        .windows(2)
        .position(|anchors| anchors == [gap.left, gap.right])
        .map(|position| position + 1)
}

fn rank_for_evolving_route(
    route: &[usize],
    position: usize,
    semantics: &[CandidateSemantics],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &BridgeConfig,
    reference: &FrozenReference,
) -> Result<Vec<BridgeCandidateEvaluation>, PreviewError> {
    let semantics_by_candidate = semantics
        .iter()
        .map(|candidate| (candidate.candidate, candidate))
        .collect::<HashMap<_, _>>();
    let candidates = semantics
        .iter()
        .map(|candidate| candidate.candidate)
        .collect::<Vec<_>>();
    let mut evaluations = rank_candidates(
        route,
        position,
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
                    .compare_priority(semantics_by_candidate[&right.candidate])
            })
            .then_with(|| left.max_percentile.total_cmp(&right.max_percentile))
            .then_with(|| left.detour_percentile.total_cmp(&right.detour_percentile))
            .then_with(|| left.candidate.cmp(&right.candidate))
    });
    Ok(evaluations)
}

fn local_objective(distance_sum: f64, worst_distance: f64) -> f64 {
    distance_sum + 2.0 * worst_distance
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
            let evaluations = rank_for_evolving_route(
                &final_route,
                position,
                &gap.semantics.candidates,
                tracks,
                learned_matrix,
                config,
                reference,
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

    let mut ordered_gaps = gaps.to_vec();
    ordered_gaps.sort_by_key(|gap| gap.original_position);
    let unique_candidates = ordered_gaps
        .iter()
        .flat_map(|gap| gap.semantics.candidates.iter())
        .map(|candidate| candidate.candidate)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let structural_upper_bound = ordered_gaps.len().min(unique_candidates);
    let initial_metrics = route::evaluate_adaptive_sequence(
        original_route,
        tracks,
        learned_matrix,
        config.seed_limit,
        config.learned_percent,
    )
    .map_err(PreviewError::RouteScoring)?;
    let mut states = vec![ExactState {
        route: original_route.to_vec(),
        decisions: Vec::with_capacity(ordered_gaps.len()),
        objective: initial_metrics.objective,
    }];
    let mut evaluated_states = 1usize;
    let mut retained_states = 1usize;

    for gap in &ordered_gaps {
        let batches = states
            .par_iter()
            .map(|state| {
                let position = route_position(&state.route, gap)
                    .ok_or(PreviewError::InvalidOriginalGap(gap.original_position))?;
                let mut expanded = Vec::new();
                let mut skipped = state.clone();
                skipped.decisions.push(exact_decision(
                    gap,
                    position,
                    DecisionReason::NotSelected,
                    None,
                ));
                expanded.push(skipped);

                let added = state.route.len() - original_route.len();
                if added < selection_config.requested_added_tracks
                    && !gap.semantics.candidates.is_empty()
                {
                    let evaluations = rank_for_evolving_route(
                        &state.route,
                        position,
                        &gap.semantics.candidates,
                        tracks,
                        learned_matrix,
                        config,
                        reference,
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
                        inserted.objective = route::evaluate_adaptive_sequence(
                            &inserted.route,
                            tracks,
                            learned_matrix,
                            config.seed_limit,
                            config.learned_percent,
                        )
                        .map_err(PreviewError::RouteScoring)?
                        .objective;
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
                Ok(expanded)
            })
            .collect::<Vec<Result<Vec<_>, PreviewError>>>();

        let mut buckets = BTreeMap::<usize, Vec<ExactState>>::new();
        for batch in batches {
            for state in batch? {
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
        .unwrap_or(0);
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
        stats: ExactSearchStats {
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
        };
        let reference = build_frozen_reference(&route, &route, &tracks, &matrix, &config).unwrap();
        let gaps = [gap(1, 0, 2, 1), gap(2, 2, 4, 3)];
        let selection_config = AutomaticSelectionConfig {
            max_added_tracks: 1,
            trigger_percentile: 0.70,
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
        };
        let reference = build_frozen_reference(&route, &route, &tracks, &matrix, &config).unwrap();
        let mut smooth = gap(1, 0, 2, 1);
        smooth.direct_percentile = 0.70;
        let selection_config = AutomaticSelectionConfig {
            max_added_tracks: 1,
            trigger_percentile: 0.70,
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
        };
        let reference = build_frozen_reference(&route, &route, &tracks, &matrix, &config).unwrap();
        let gaps = [gap(1, 0, 2, 1), gap(2, 2, 4, 3)];
        let exact = ExactSelectionConfig {
            requested_added_tracks: 2,
            candidate_limit: 2,
            beam_width: 16,
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
}
