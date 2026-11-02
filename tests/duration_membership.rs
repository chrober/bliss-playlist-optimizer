use bliss_playlist_optimizer::duration_membership::{
    find_memberships, find_memberships_with_filter, Candidate,
};

#[test]
fn finds_an_alternative_when_the_most_relevant_candidate_overshoots() {
    let candidates = [
        Candidate {
            id: 1,
            duration_ms: 110_000,
        },
        Candidate {
            id: 2,
            duration_ms: 60_000,
        },
        Candidate {
            id: 3,
            duration_ms: 40_000,
        },
    ];
    let result = find_memberships(&candidates, 100_000, 100_000, 3, 20_000, 4);
    assert!(result.memberships.contains(&vec![2, 3]));
}

#[test]
fn returns_an_empty_membership_when_no_addition_is_needed() {
    let result = find_memberships(&[], 0, 60_000, 10, 100, 4);
    assert_eq!(result.memberships, vec![Vec::<usize>::new()]);
}

#[test]
fn reports_a_search_limit_separately_from_exhausting_all_possibilities() {
    let candidates = [
        Candidate {
            id: 1,
            duration_ms: 50,
        },
        Candidate {
            id: 2,
            duration_ms: 70,
        },
    ];
    let result = find_memberships(&candidates, 100, 100, 2, 1, 4);
    assert!(result.search_limited);
    assert!(result.memberships.is_empty());
}

#[test]
fn never_selects_invalid_or_repeated_track_ids() {
    let candidates = [
        Candidate {
            id: 1,
            duration_ms: 50,
        },
        Candidate {
            id: 1,
            duration_ms: 50,
        },
        Candidate {
            id: 2,
            duration_ms: 0,
        },
    ];
    let result = find_memberships(&candidates, 100, 100, 3, 1000, 4);
    assert!(result.memberships.is_empty());
}

#[test]
fn considers_longer_memberships_inside_the_range_for_repeat_safe_routing() {
    let candidates = [
        Candidate {
            id: 1,
            duration_ms: 50,
        },
        Candidate {
            id: 2,
            duration_ms: 40,
        },
    ];
    let result = find_memberships(&candidates, 50, 100, 2, 1000, 10);
    assert!(result.memberships.contains(&vec![1]));
    assert!(result.memberships.contains(&vec![1, 2]));
}

#[test]
fn handles_large_ranked_candidate_inventories_under_a_bounded_search() {
    let candidates = (0..200_000)
        .map(|id| Candidate {
            id,
            duration_ms: 180_000,
        })
        .collect::<Vec<_>>();
    let result = find_memberships(&candidates, 540_000, 540_000, 10, 100, 1);
    assert_eq!(result.memberships, vec![vec![0, 1, 2]]);
}

#[test]
fn rejected_memberships_do_not_exhaust_the_result_budget() {
    let candidates = [
        Candidate {
            id: 1,
            duration_ms: 50,
        },
        Candidate {
            id: 2,
            duration_ms: 50,
        },
    ];
    let result = find_memberships_with_filter(&candidates, 50, 50, 2, 1000, 1, |membership| {
        !membership.contains(&1)
    });
    assert_eq!(result.memberships, vec![vec![2]]);
}
