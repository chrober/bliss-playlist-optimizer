use std::collections::BTreeMap;

use bliss_playlist_optimizer::anchored_path::{AnchoredPathCandidate, FixedAdjacentScorer};
use bliss_playlist_optimizer::multi_gap::{
    minimum_repeat_spacing_additions, plan_fixed_gap_count, plan_fixed_gap_count_with_progress,
    plan_repeat_window_repair, FixedGap, FixedGapPlanConfig, FixedGapPlanError,
    FixedGapPlanRequest, FixedGapProgressPhase, RepeatWindowRepairConfig,
    RepeatWindowRepairRequest,
};
use bliss_playlist_optimizer::route::RouteTrack;

fn track(value: f32, artist: &str, album: &str) -> RouteTrack {
    RouteTrack {
        features: std::array::from_fn(|index| value + index as f32 / 100.0),
        artist_key: artist.to_owned(),
        artist_repeat_keys: Vec::new(),
        album_key: album.to_owned(),
    }
}

fn candidate(track: usize) -> AnchoredPathCandidate {
    AnchoredPathCandidate {
        track,
        guidance_adjustment: 0.0,
    }
}

fn config(tracks_per_gap: usize) -> FixedGapPlanConfig {
    FixedGapPlanConfig {
        tracks_per_gap,
        candidate_limit: 16,
        beam_width: 64,
        alternatives_per_gap: 8,
        variation_percent: 0,
        generation_seed: 42,
        artist_window: 0,
        album_window: 0,
        track_window: 100,
    }
}

fn repair_config(max_added_tracks: usize) -> RepeatWindowRepairConfig {
    RepeatWindowRepairConfig {
        max_added_tracks,
        candidate_limit: 16,
        beam_width: 64,
        alternatives_per_gap: 8,
        variation_percent: 0,
        generation_seed: 42,
        artist_window: 2,
        album_window: 0,
        track_window: 100,
    }
}

#[test]
fn repeat_spacing_lower_bound_accounts_for_overlapping_source_constraints() {
    let tracks = vec![
        track(0.0, "same", "album-a"),
        track(1.0, "other", "album-b"),
        track(2.0, "same", "album-c"),
        track(3.0, "same", "album-d"),
    ];

    assert_eq!(
        minimum_repeat_spacing_additions(&[0, 1, 2, 3], &tracks, 2, 0, 0).unwrap(),
        3
    );
}

#[test]
fn repair_uses_zero_tracks_in_safe_gaps_and_two_where_required() {
    let tracks = vec![
        track(0.0, "repeated", "source-a"),
        track(1.0, "bridge-one", "bridge-one"),
        track(2.0, "bridge-two", "bridge-two"),
        track(3.0, "repeated", "source-b"),
        track(4.0, "unused", "unused"),
        track(5.0, "unused-two", "unused-two"),
        track(6.0, "final", "source-c"),
    ];
    let gaps = [
        FixedGap {
            left: 0,
            right: 3,
            candidates: &[candidate(1), candidate(2)],
        },
        FixedGap {
            left: 3,
            right: 6,
            candidates: &[candidate(4), candidate(5)],
        },
    ];

    let plan = plan_repeat_window_repair(
        RepeatWindowRepairRequest {
            source_route: &[0, 3, 6],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: repair_config(3),
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
    )
    .unwrap();

    assert_eq!(plan.final_route, vec![0, 1, 2, 3, 6]);
    assert_eq!(plan.gaps[0].intermediates, vec![1, 2]);
    assert!(plan.gaps[1].intermediates.is_empty());
}

#[test]
fn repair_minimizes_total_additions_before_acoustic_score() {
    let tracks = vec![
        track(0.0, "repeated", "source-a"),
        track(1.0, "bridge-one", "bridge-one"),
        track(2.0, "bridge-two", "bridge-two"),
        track(4.0, "repeated", "source-b"),
    ];
    let gaps = [FixedGap {
        left: 0,
        right: 3,
        candidates: &[candidate(1), candidate(2)],
    }];
    let distances = BTreeMap::from([
        ((0, 1), 10.0),
        ((1, 3), 10.0),
        ((0, 2), 10.0),
        ((2, 3), 10.0),
        ((1, 2), 0.1),
    ]);
    let mut repair = repair_config(2);
    repair.artist_window = 1;

    let plan = plan_repeat_window_repair(
        RepeatWindowRepairRequest {
            source_route: &[0, 3],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: repair,
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| {
            distances
                .get(&(left, right))
                .copied()
                .or_else(|| distances.get(&(right, left)).copied())
                .unwrap_or(100.0)
        }),
    )
    .unwrap();

    assert_eq!(plan.gaps[0].intermediates.len(), 1);
}

#[test]
fn repair_keeps_separate_total_count_beams_for_later_source_conflicts() {
    let tracks = vec![
        track(0.0, "repeated", "source-a"),
        track(1.0, "spacing", "spacing"),
        track(2.0, "unused", "unused"),
        track(3.0, "middle", "source-b"),
        track(4.0, "unused-two", "unused-two"),
        track(5.0, "unused-three", "unused-three"),
        track(6.0, "repeated", "source-c"),
    ];
    let gaps = [
        FixedGap {
            left: 0,
            right: 3,
            candidates: &[candidate(1)],
        },
        FixedGap {
            left: 3,
            right: 6,
            candidates: &[],
        },
    ];
    let mut repair = repair_config(1);
    repair.beam_width = 1;

    let plan = plan_repeat_window_repair(
        RepeatWindowRepairRequest {
            source_route: &[0, 3, 6],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: repair,
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
    )
    .unwrap();

    assert_eq!(plan.final_route, vec![0, 1, 3, 6]);
}

#[test]
fn repair_respects_global_budget_without_returning_a_partial_route() {
    let tracks = vec![
        track(0.0, "repeated", "source-a"),
        track(1.0, "bridge-one", "bridge-one"),
        track(2.0, "bridge-two", "bridge-two"),
        track(3.0, "repeated", "source-b"),
    ];
    let gaps = [FixedGap {
        left: 0,
        right: 3,
        candidates: &[candidate(1), candidate(2)],
    }];

    let error = plan_repeat_window_repair(
        RepeatWindowRepairRequest {
            source_route: &[0, 3],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: repair_config(1),
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        FixedGapPlanError::NoCompleteRoute { .. } | FixedGapPlanError::FinalRouteInvalid(_)
    ));
}

#[test]
fn repair_is_deterministic_across_one_and_four_rayon_workers() {
    let tracks = vec![
        track(0.0, "repeated", "source-a"),
        track(1.0, "bridge-one", "bridge-one"),
        track(2.0, "bridge-two", "bridge-two"),
        track(3.0, "repeated", "source-b"),
        track(4.0, "final", "source-c"),
    ];
    let gaps = [
        FixedGap {
            left: 0,
            right: 3,
            candidates: &[candidate(1), candidate(2)],
        },
        FixedGap {
            left: 3,
            right: 4,
            candidates: &[candidate(1), candidate(2)],
        },
    ];
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                plan_repeat_window_repair(
                    RepeatWindowRepairRequest {
                        source_route: &[0, 3, 4],
                        immutable_history: &[],
                        gaps: &gaps,
                        tracks: &tracks,
                        config: repair_config(3),
                    },
                    FixedAdjacentScorer::new(|left: usize, right: usize| {
                        left.abs_diff(right) as f64
                    }),
                )
                .unwrap()
            })
    };

    assert_eq!(run(1), run(4));
}

#[test]
fn repair_variation_never_selects_a_larger_valid_total() {
    let tracks = vec![
        track(0.0, "repeated", "source-a"),
        track(1.0, "bridge-one", "bridge-one"),
        track(2.0, "bridge-two", "bridge-two"),
        track(3.0, "repeated", "source-b"),
    ];
    let gaps = [FixedGap {
        left: 0,
        right: 3,
        candidates: &[candidate(1), candidate(2)],
    }];
    let mut repair = repair_config(2);
    repair.artist_window = 1;
    repair.variation_percent = 100;

    let plan = plan_repeat_window_repair(
        RepeatWindowRepairRequest {
            source_route: &[0, 3],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: repair,
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
    )
    .unwrap();

    assert_eq!(plan.gaps[0].intermediates.len(), 1);
}

#[test]
fn fills_every_internal_gap_with_the_exact_requested_count() {
    let tracks = (0..9)
        .map(|index| {
            track(
                index as f32,
                &format!("artist-{index}"),
                &format!("album-{index}"),
            )
        })
        .collect::<Vec<_>>();
    let gaps = [
        FixedGap {
            left: 0,
            right: 3,
            candidates: &[candidate(1), candidate(2), candidate(7), candidate(8)],
        },
        FixedGap {
            left: 3,
            right: 6,
            candidates: &[candidate(4), candidate(5), candidate(7), candidate(8)],
        },
    ];

    let plan = plan_fixed_gap_count(
        FixedGapPlanRequest {
            source_route: &[0, 3, 6],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: config(2),
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
    )
    .unwrap();

    assert_eq!(plan.final_route.len(), 7);
    assert_eq!(plan.gaps.len(), 2);
    assert!(plan.gaps.iter().all(|gap| gap.intermediates.len() == 2));
    assert_eq!(plan.final_route.first(), Some(&0));
    assert_eq!(plan.final_route.last(), Some(&6));
    let sources = plan
        .final_route
        .iter()
        .copied()
        .filter(|track| [0, 3, 6].contains(track))
        .collect::<Vec<_>>();
    assert_eq!(sources, vec![0, 3, 6]);
}

#[test]
fn reports_start_and_completion_for_every_gap() {
    let tracks = (0..7)
        .map(|index| {
            track(
                index as f32,
                &format!("artist-{index}"),
                &format!("album-{index}"),
            )
        })
        .collect::<Vec<_>>();
    let gaps = [
        FixedGap {
            left: 0,
            right: 3,
            candidates: &[candidate(1), candidate(2)],
        },
        FixedGap {
            left: 3,
            right: 6,
            candidates: &[candidate(4), candidate(5)],
        },
    ];
    let mut events = Vec::new();

    plan_fixed_gap_count_with_progress(
        FixedGapPlanRequest {
            source_route: &[0, 3, 6],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: config(1),
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
        |event| events.push(event),
    )
    .unwrap();

    assert_eq!(events.len(), 4);
    assert_eq!(
        events
            .iter()
            .map(|event| (
                event.phase,
                event.gap,
                event.total_gaps,
                event.completed_gaps
            ))
            .collect::<Vec<_>>(),
        vec![
            (FixedGapProgressPhase::Searching, 1, 2, 0),
            (FixedGapProgressPhase::Completed, 1, 2, 1),
            (FixedGapProgressPhase::Searching, 2, 2, 1),
            (FixedGapProgressPhase::Completed, 2, 2, 2),
        ]
    );
    assert!(events
        .iter()
        .filter(|event| event.phase == FixedGapProgressPhase::Completed)
        .all(|event| event.evaluated_states > 0 && event.retained_states > 0));
}

#[test]
fn retains_a_worse_early_alternative_when_the_local_winner_blocks_a_later_gap() {
    let tracks = vec![
        track(0.0, "source-a", "source-a"),
        track(1.0, "shared", "candidate-one"),
        track(2.0, "other", "candidate-two"),
        track(3.0, "source-b", "source-b"),
        track(4.0, "shared", "candidate-four"),
        track(6.0, "source-c", "source-c"),
    ];
    let gaps = [
        FixedGap {
            left: 0,
            right: 3,
            candidates: &[candidate(1), candidate(2)],
        },
        FixedGap {
            left: 3,
            right: 5,
            candidates: &[candidate(4)],
        },
    ];
    let distances = BTreeMap::from([
        ((0, 1), 1.0),
        ((1, 3), 1.0),
        ((0, 2), 2.0),
        ((2, 3), 2.0),
        ((3, 4), 1.0),
        ((4, 5), 1.0),
    ]);
    let mut repeat_config = config(1);
    repeat_config.artist_window = 2;

    let plan = plan_fixed_gap_count(
        FixedGapPlanRequest {
            source_route: &[0, 3, 5],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: repeat_config,
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| {
            distances
                .get(&(left, right))
                .copied()
                .or_else(|| distances.get(&(right, left)).copied())
                .unwrap_or(100.0)
        }),
    )
    .unwrap();

    assert_eq!(plan.final_route, vec![0, 2, 3, 4, 5]);
}

#[test]
fn source_anchors_are_unavailable_as_generated_tracks() {
    let tracks = (0..5)
        .map(|index| {
            track(
                index as f32,
                &format!("artist-{index}"),
                &format!("album-{index}"),
            )
        })
        .collect::<Vec<_>>();
    let gaps = [
        FixedGap {
            left: 0,
            right: 2,
            candidates: &[candidate(2), candidate(1)],
        },
        FixedGap {
            left: 2,
            right: 4,
            candidates: &[candidate(4), candidate(3)],
        },
    ];

    let plan = plan_fixed_gap_count(
        FixedGapPlanRequest {
            source_route: &[0, 2, 4],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: config(1),
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
    )
    .unwrap();

    assert_eq!(plan.final_route, vec![0, 1, 2, 3, 4]);
}

#[test]
fn guidance_breaks_a_global_acoustic_tie_without_admitting_other_tracks() {
    let tracks = vec![
        track(0.0, "a", "a"),
        track(1.0, "b", "b"),
        track(1.0, "c", "c"),
        track(2.0, "d", "d"),
    ];
    let gaps = [FixedGap {
        left: 0,
        right: 3,
        candidates: &[
            AnchoredPathCandidate {
                track: 1,
                guidance_adjustment: 0.0,
            },
            AnchoredPathCandidate {
                track: 2,
                guidance_adjustment: 0.5,
            },
        ],
    }];

    let plan = plan_fixed_gap_count(
        FixedGapPlanRequest {
            source_route: &[0, 3],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: config(1),
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| {
            (tracks[left].features[0] - tracks[right].features[0]).abs() as f64
        }),
    )
    .unwrap();

    assert_eq!(plan.final_route, vec![0, 2, 3]);
}

#[test]
fn infeasible_later_gap_returns_no_partial_plan() {
    let tracks = vec![
        track(0.0, "a", "a"),
        track(1.0, "bridge", "bridge"),
        track(2.0, "b", "b"),
        track(3.0, "bridge", "other"),
        track(4.0, "c", "c"),
    ];
    let gaps = [
        FixedGap {
            left: 0,
            right: 2,
            candidates: &[candidate(1)],
        },
        FixedGap {
            left: 2,
            right: 4,
            candidates: &[candidate(3)],
        },
    ];
    let mut repeat_config = config(1);
    repeat_config.artist_window = 3;

    let error = plan_fixed_gap_count(
        FixedGapPlanRequest {
            source_route: &[0, 2, 4],
            immutable_history: &[],
            gaps: &gaps,
            tracks: &tracks,
            config: repeat_config,
        },
        FixedAdjacentScorer::new(|left: usize, right: usize| left.abs_diff(right) as f64),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        FixedGapPlanError::NoCompleteRoute {
            blocking_gap: 1,
            ..
        }
    ));
}

#[test]
fn parallel_worker_counts_produce_the_same_plan() {
    let tracks = (0..9)
        .map(|index| {
            track(
                index as f32,
                &format!("artist-{index}"),
                &format!("album-{index}"),
            )
        })
        .collect::<Vec<_>>();
    let gaps = [
        FixedGap {
            left: 0,
            right: 4,
            candidates: &[candidate(1), candidate(2), candidate(3), candidate(7)],
        },
        FixedGap {
            left: 4,
            right: 8,
            candidates: &[candidate(5), candidate(6), candidate(7), candidate(2)],
        },
    ];
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                plan_fixed_gap_count(
                    FixedGapPlanRequest {
                        source_route: &[0, 4, 8],
                        immutable_history: &[],
                        gaps: &gaps,
                        tracks: &tracks,
                        config: config(2),
                    },
                    FixedAdjacentScorer::new(|left: usize, right: usize| {
                        left.abs_diff(right) as f64
                    }),
                )
                .unwrap()
            })
    };

    assert_eq!(run(1), run(4));
}
