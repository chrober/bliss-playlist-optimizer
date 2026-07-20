// SPDX-License-Identifier: GPL-3.0-only

use std::collections::HashMap;
use std::fmt;

use bliss_mixer_core::FeatureVector;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::contextual::adaptive_distance_from_seeds;

const WORST_LEG_WEIGHT: f64 = 2.0;
const ARC_PLACEMENT_WEIGHT: f64 = 0.12;
const ARC_PRIMARY_TOLERANCE: f64 = 1.08;
const ARC_ERROR_IMPROVEMENT: f64 = 0.90;
const GREEDY_WIDTH: usize = 4;
const EPSILON: f64 = 1e-12;

type ScoreCache = HashMap<(Vec<usize>, usize), f64>;

#[derive(Clone, Debug)]
pub struct RouteTrack {
    pub features: FeatureVector,
    pub artist_key: String,
    pub album_key: String,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub seed_limit: usize,
    pub learned_percent: u16,
    pub deterministic_seed: u64,
    pub restart_count: usize,
    pub artist_window: usize,
    pub album_window: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteMetrics {
    pub transition_sum: f64,
    pub worst_transition: f64,
    pub objective: f64,
    pub arc_error: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatViolation {
    pub kind: &'static str,
    pub positions: [usize; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateRoute {
    pub strategy: &'static str,
    pub route: Vec<usize>,
    pub metrics: RouteMetrics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteSearchResult {
    pub primary: CandidateRoute,
    pub arc: CandidateRoute,
    pub selected: CandidateRoute,
    pub violations: Vec<RepeatViolation>,
    pub search_tasks: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteError {
    TooFewTracks,
    InvalidSeedLimit,
    InvalidLearnedPercent(u16),
    Scoring(String),
    Infeasible,
}

impl fmt::Display for RouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewTracks => formatter.write_str("route search requires at least two tracks"),
            Self::InvalidSeedLimit => {
                formatter.write_str("adaptive seed limit must be at least one")
            }
            Self::InvalidLearnedPercent(value) => {
                write!(
                    formatter,
                    "learned matrix percentage {value} is outside 0..=100"
                )
            }
            Self::Scoring(message) => write!(formatter, "adaptive route scoring failed: {message}"),
            Self::Infeasible => {
                formatter.write_str("no route satisfies the configured repeat windows")
            }
        }
    }
}

impl std::error::Error for RouteError {}

pub fn optimize_adaptive_route(
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &SearchConfig,
) -> Result<RouteSearchResult, RouteError> {
    if tracks.len() < 2 {
        return Err(RouteError::TooFewTracks);
    }
    if config.seed_limit == 0 {
        return Err(RouteError::InvalidSeedLimit);
    }
    if config.learned_percent > 100 {
        return Err(RouteError::InvalidLearnedPercent(config.learned_percent));
    }

    let intensities = intensity_values(tracks);
    let targets = arc_targets(tracks.len());
    let arc_context = Some((intensities.as_slice(), targets.as_slice()));
    let mut primary = search_candidate(tracks, learned_matrix, config, None, "adaptive")?;
    let mut primary_cache = ScoreCache::new();
    primary.metrics = route_metrics(
        &primary.route,
        tracks,
        learned_matrix,
        config,
        arc_context,
        &mut primary_cache,
    )?;
    let arc = search_candidate(tracks, learned_matrix, config, arc_context, "adaptive-arc")?;
    let selected = if arc.metrics.objective <= primary.metrics.objective * ARC_PRIMARY_TOLERANCE
        && arc.metrics.arc_error <= primary.metrics.arc_error * ARC_ERROR_IMPROVEMENT
    {
        arc.clone()
    } else {
        primary.clone()
    };
    let violations = repeat_violations(&selected.route, tracks, config);
    if !violations.is_empty() {
        return Err(RouteError::Infeasible);
    }

    Ok(RouteSearchResult {
        primary,
        arc,
        selected,
        violations,
        search_tasks: config.restart_count * 2 + 5,
    })
}

pub fn evaluate_adaptive_sequence(
    route: &[usize],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    seed_limit: usize,
    learned_percent: u16,
) -> Result<RouteMetrics, RouteError> {
    if route.len() < 2 {
        return Err(RouteError::TooFewTracks);
    }
    if seed_limit == 0 {
        return Err(RouteError::InvalidSeedLimit);
    }
    if learned_percent > 100 {
        return Err(RouteError::InvalidLearnedPercent(learned_percent));
    }
    if let Some(index) = route.iter().find(|index| **index >= tracks.len()) {
        return Err(RouteError::Scoring(format!(
            "invalid track index {index} for {} tracks",
            tracks.len()
        )));
    }
    let config = SearchConfig {
        seed_limit,
        learned_percent,
        deterministic_seed: 0,
        restart_count: 0,
        artist_window: 0,
        album_window: 0,
    };
    route_metrics(
        route,
        tracks,
        learned_matrix,
        &config,
        None,
        &mut ScoreCache::new(),
    )
}

fn search_candidate(
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &SearchConfig,
    arc_context: Option<(&[f64], &[f64])>,
    strategy: &'static str,
) -> Result<CandidateRoute, RouteError> {
    let fixed_starts = if arc_context.is_some() { 3 } else { 2 };
    let task_count = fixed_starts + config.restart_count;
    let attempts: Vec<Result<CandidateRoute, RouteError>> = (0..task_count)
        .into_par_iter()
        .map(|task| {
            let mut score_cache = ScoreCache::new();
            let start = match task {
                0 => (0..tracks.len()).collect(),
                1 => (0..tracks.len()).rev().collect(),
                2 if arc_context.is_some() => intensity_order(arc_context.unwrap().0),
                _ => greedy_route(
                    tracks,
                    learned_matrix,
                    config,
                    derived_seed(config.deterministic_seed, task - fixed_starts),
                    &mut score_cache,
                )?,
            };
            let route = improve_route(
                start,
                tracks,
                learned_matrix,
                config,
                arc_context,
                &mut score_cache,
            )?;
            if !repeat_violations(&route, tracks, config).is_empty() {
                return Err(RouteError::Infeasible);
            }
            let metrics = route_metrics(
                &route,
                tracks,
                learned_matrix,
                config,
                arc_context,
                &mut score_cache,
            )?;
            Ok(CandidateRoute {
                strategy,
                route,
                metrics,
            })
        })
        .collect();

    let mut best: Option<CandidateRoute> = None;
    let mut first_error = None;
    for attempt in attempts {
        match attempt {
            Ok(candidate) => {
                if best.as_ref().is_none_or(|current| {
                    candidate_precedes(&candidate, current, arc_context.is_some())
                }) {
                    best = Some(candidate);
                }
            }
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    best.ok_or_else(|| first_error.unwrap_or(RouteError::Infeasible))
}

fn candidate_precedes(left: &CandidateRoute, right: &CandidateRoute, arc_aware: bool) -> bool {
    let left_score = search_score(&left.metrics, arc_aware);
    let right_score = search_score(&right.metrics, arc_aware);
    left_score.total_cmp(&right_score).is_lt()
        || (left_score == right_score && left.route < right.route)
}

fn greedy_route(
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &SearchConfig,
    seed: u64,
    score_cache: &mut ScoreCache,
) -> Result<Vec<usize>, RouteError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut remaining: Vec<usize> = (0..tracks.len()).collect();
    let first_position = rng.gen_range(0..remaining.len());
    let mut route = vec![remaining.remove(first_position)];

    while !remaining.is_empty() {
        let mut feasible: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|candidate| is_feasible_append(&route, *candidate, tracks, config))
            .collect();
        if feasible.is_empty() {
            feasible.clone_from(&remaining);
        }
        let mut ranked: Vec<(f64, usize)> = feasible
            .into_iter()
            .map(|candidate| {
                Ok((
                    transition_distance(
                        &route,
                        candidate,
                        tracks,
                        learned_matrix,
                        config,
                        score_cache,
                    )?,
                    candidate,
                ))
            })
            .collect::<Result<_, RouteError>>()?;
        ranked.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let width = ranked.len().min(GREEDY_WIDTH);
        let pick = ((rng.gen::<f64>().powi(2) * width as f64) as usize).min(width - 1);
        let chosen = ranked[pick].1;
        route.push(chosen);
        remaining.retain(|candidate| *candidate != chosen);
    }
    Ok(route)
}

fn improve_route(
    mut route: Vec<usize>,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &SearchConfig,
    arc_context: Option<(&[f64], &[f64])>,
    score_cache: &mut ScoreCache,
) -> Result<Vec<usize>, RouteError> {
    loop {
        let current = route_metrics(
            &route,
            tracks,
            learned_matrix,
            config,
            arc_context,
            score_cache,
        )?;
        let current_score = search_score(&current, arc_context.is_some());
        let mut best_route = route.clone();
        let mut best_score = current_score;

        for start in 0..route.len().saturating_sub(1) {
            for end in start + 1..route.len() {
                let mut candidate = route.clone();
                candidate[start..=end].reverse();
                consider_neighbor(
                    candidate,
                    &mut best_route,
                    &mut best_score,
                    tracks,
                    learned_matrix,
                    config,
                    arc_context,
                    score_cache,
                )?;
            }
        }
        for source in 0..route.len() {
            let mut shortened = route.clone();
            let moved = shortened.remove(source);
            for destination in 0..=shortened.len() {
                let mut candidate = shortened.clone();
                candidate.insert(destination, moved);
                if candidate != route {
                    consider_neighbor(
                        candidate,
                        &mut best_route,
                        &mut best_score,
                        tracks,
                        learned_matrix,
                        config,
                        arc_context,
                        score_cache,
                    )?;
                }
            }
        }

        if best_score + EPSILON < current_score {
            route = best_route;
        } else {
            return Ok(route);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn consider_neighbor(
    candidate: Vec<usize>,
    best_route: &mut Vec<usize>,
    best_score: &mut f64,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &SearchConfig,
    arc_context: Option<(&[f64], &[f64])>,
    score_cache: &mut ScoreCache,
) -> Result<(), RouteError> {
    if !repeat_violations(&candidate, tracks, config).is_empty() {
        return Ok(());
    }
    let metrics = route_metrics(
        &candidate,
        tracks,
        learned_matrix,
        config,
        arc_context,
        score_cache,
    )?;
    let score = search_score(&metrics, arc_context.is_some());
    if score + EPSILON < *best_score
        || ((score - *best_score).abs() <= EPSILON && candidate.as_slice() < best_route.as_slice())
    {
        *best_score = score;
        *best_route = candidate;
    }
    Ok(())
}

fn route_metrics(
    route: &[usize],
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &SearchConfig,
    arc_context: Option<(&[f64], &[f64])>,
    score_cache: &mut ScoreCache,
) -> Result<RouteMetrics, RouteError> {
    let mut transition_sum = 0.0;
    let mut worst_transition = 0.0_f64;
    for position in 1..route.len() {
        let distance = transition_distance(
            &route[..position],
            route[position],
            tracks,
            learned_matrix,
            config,
            score_cache,
        )?;
        transition_sum += distance;
        worst_transition = worst_transition.max(distance);
    }
    let arc_error = arc_context.map_or(0.0, |(intensities, targets)| {
        route
            .iter()
            .zip(targets)
            .map(|(track, target)| (intensities[*track] - target).abs())
            .sum()
    });
    Ok(RouteMetrics {
        transition_sum,
        worst_transition,
        objective: transition_sum + WORST_LEG_WEIGHT * worst_transition,
        arc_error,
    })
}

fn transition_distance(
    prefix: &[usize],
    candidate: usize,
    tracks: &[RouteTrack],
    learned_matrix: &Array2<f32>,
    config: &SearchConfig,
    score_cache: &mut ScoreCache,
) -> Result<f64, RouteError> {
    let start = prefix.len().saturating_sub(config.seed_limit);
    let cache_key = (prefix[start..].to_vec(), candidate);
    if let Some(distance) = score_cache.get(&cache_key) {
        return Ok(*distance);
    }
    let seeds: Vec<FeatureVector> = prefix[start..]
        .iter()
        .map(|index| tracks[*index].features)
        .collect();
    let distance = adaptive_distance_from_seeds(
        &seeds,
        &tracks[candidate].features,
        learned_matrix,
        config.learned_percent,
    )
    .map_err(|error| RouteError::Scoring(error.to_string()))?;
    score_cache.insert(cache_key, distance);
    Ok(distance)
}

fn search_score(metrics: &RouteMetrics, arc_aware: bool) -> f64 {
    metrics.objective
        + if arc_aware {
            ARC_PLACEMENT_WEIGHT * metrics.arc_error
        } else {
            0.0
        }
}

fn is_feasible_append(
    route: &[usize],
    candidate: usize,
    tracks: &[RouteTrack],
    config: &SearchConfig,
) -> bool {
    let item = &tracks[candidate];
    let artist_ok = config.artist_window == 0
        || !route.iter().rev().take(config.artist_window).any(|index| {
            !item.artist_key.is_empty() && tracks[*index].artist_key == item.artist_key
        });
    let album_ok =
        config.album_window == 0
            || !route.iter().rev().take(config.album_window).any(|index| {
                !item.album_key.is_empty() && tracks[*index].album_key == item.album_key
            });
    artist_ok && album_ok
}

pub fn repeat_violations(
    route: &[usize],
    tracks: &[RouteTrack],
    config: &SearchConfig,
) -> Vec<RepeatViolation> {
    let mut violations = Vec::new();
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
                violations.push(RepeatViolation {
                    kind: "artist",
                    positions: [left, right],
                });
            }
            if config.album_window > 0
                && distance <= config.album_window
                && !right_track.album_key.is_empty()
                && left_track.album_key == right_track.album_key
            {
                violations.push(RepeatViolation {
                    kind: "album",
                    positions: [left, right],
                });
            }
        }
    }
    violations
}

fn intensity_values(tracks: &[RouteTrack]) -> Vec<f64> {
    const INDEXES: [usize; 5] = [0, 1, 2, 4, 8];
    let mut sums = vec![0.0; tracks.len()];
    let denominator = tracks.len().saturating_sub(1).max(1) as f64;
    for feature in INDEXES {
        let mut ordered: Vec<usize> = (0..tracks.len()).collect();
        ordered.sort_by(|left, right| {
            tracks[*left].features[feature]
                .total_cmp(&tracks[*right].features[feature])
                .then_with(|| left.cmp(right))
        });
        for (rank, track) in ordered.into_iter().enumerate() {
            sums[track] += rank as f64 / denominator;
        }
    }
    sums.into_iter()
        .map(|sum| sum / INDEXES.len() as f64)
        .collect()
}

fn intensity_order(intensities: &[f64]) -> Vec<usize> {
    let mut route: Vec<usize> = (0..intensities.len()).collect();
    route.sort_by(|left, right| {
        intensities[*left]
            .total_cmp(&intensities[*right])
            .then_with(|| left.cmp(right))
    });
    route
}

fn arc_targets(size: usize) -> Vec<f64> {
    let peak = (((size.saturating_sub(1)) as f64 * 0.70).round() as usize).max(1);
    (0..size)
        .map(|position| {
            if position <= peak {
                0.25 + 0.60 * position as f64 / peak as f64
            } else {
                0.85 - 0.50 * (position - peak) as f64 / size.saturating_sub(1 + peak).max(1) as f64
            }
        })
        .collect()
}

fn derived_seed(base: u64, restart: usize) -> u64 {
    let mut value = base.wrapping_add((restart as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(value: f32, artist: &str) -> RouteTrack {
        RouteTrack {
            features: std::array::from_fn(|index| value * (index + 1) as f32 / 10.0),
            artist_key: artist.to_owned(),
            album_key: format!("album-{value}"),
        }
    }

    fn config() -> SearchConfig {
        SearchConfig {
            seed_limit: 3,
            learned_percent: 20,
            deterministic_seed: 2026,
            restart_count: 8,
            artist_window: 1,
            album_window: 1,
        }
    }

    #[test]
    fn parallel_restarts_are_deterministic_and_preserve_membership() {
        let tracks = vec![
            track(0.0, "a"),
            track(3.0, "b"),
            track(1.0, "c"),
            track(4.0, "d"),
            track(2.0, "e"),
        ];
        let one_worker = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| optimize_adaptive_route(&tracks, &Array2::eye(23), &config()))
            .unwrap();
        let four_workers = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| optimize_adaptive_route(&tracks, &Array2::eye(23), &config()))
            .unwrap();
        assert_eq!(one_worker, four_workers);
        let first = one_worker;
        let mut members = first.selected.route.clone();
        members.sort_unstable();
        assert_eq!(members, (0..tracks.len()).collect::<Vec<_>>());
        assert!(first.violations.is_empty());
    }

    #[test]
    fn reports_infeasible_repeat_windows() {
        let tracks = vec![track(0.0, "same"), track(1.0, "same"), track(2.0, "same")];
        assert_eq!(
            optimize_adaptive_route(&tracks, &Array2::eye(23), &config()),
            Err(RouteError::Infeasible)
        );
    }
}
