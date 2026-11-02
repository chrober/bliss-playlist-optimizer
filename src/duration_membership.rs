// SPDX-License-Identifier: GPL-3.0-only
//! Bounded enumeration of duration-feasible candidate memberships.
//!
//! Candidate order is supplied by the caller's Bliss-first ranking. A result
//! still needs the caller's repeat-window and route validation.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub id: usize,
    pub duration_ms: u64,
}

#[derive(Debug)]
pub struct MembershipSearch {
    pub memberships: Vec<Vec<usize>>,
    /// The node budget ended before the search space was exhausted.
    pub search_limited: bool,
}

/// Enumerate combinations in caller-provided relevance order. Exhaustion of a
/// node budget is not evidence that an acceptable combination does not exist.
pub fn find_memberships(
    candidates: &[Candidate],
    minimum_ms: u64,
    maximum_ms: u64,
    maximum_tracks: usize,
    node_budget: usize,
    maximum_results: usize,
) -> MembershipSearch {
    find_memberships_with_filter(
        candidates,
        minimum_ms,
        maximum_ms,
        maximum_tracks,
        node_budget,
        maximum_results,
        |_| true,
    )
}

/// Apply necessary host constraints before a membership consumes the result budget.
pub fn find_memberships_with_filter<F: Fn(&[usize]) -> bool>(
    candidates: &[Candidate],
    minimum_ms: u64,
    maximum_ms: u64,
    maximum_tracks: usize,
    node_budget: usize,
    maximum_results: usize,
    accept: F,
) -> MembershipSearch {
    let mut result = MembershipSearch {
        memberships: Vec::new(),
        search_limited: false,
    };
    if minimum_ms > maximum_ms || maximum_results == 0 {
        return result;
    }
    if minimum_ms == 0 {
        if accept(&[]) {
            result.memberships.push(Vec::new());
        }
        return result;
    }
    let mut seen = HashSet::new();
    let candidates: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.duration_ms > 0 && seen.insert(candidate.id))
        .collect();
    let mut remaining = vec![0u64; candidates.len() + 1];
    for index in (0..candidates.len()).rev() {
        remaining[index] = remaining[index + 1].saturating_add(candidates[index].duration_ms);
    }
    let mut stack = vec![(0usize, 0u64, Vec::<usize>::new())];
    let mut seen_memberships = HashSet::new();
    let mut nodes = 0usize;
    while let Some((index, duration, selected)) = stack.pop() {
        if nodes >= node_budget {
            result.search_limited = true;
            break;
        }
        nodes += 1;
        if duration >= minimum_ms && seen_memberships.insert(selected.clone()) && accept(&selected)
        {
            result.memberships.push(selected.clone());
            if result.memberships.len() >= maximum_results {
                result.search_limited = !stack.is_empty() || index < candidates.len();
                break;
            }
        }
        if index >= candidates.len()
            || selected.len() >= maximum_tracks
            || duration.saturating_add(remaining[index]) < minimum_ms
        {
            continue;
        }
        // Push exclusion first so inclusion is explored first in relevance order.
        stack.push((index + 1, duration, selected.clone()));
        if let Some(next) = duration.checked_add(candidates[index].duration_ms) {
            if next <= maximum_ms {
                let mut included = selected;
                included.push(candidates[index].id);
                stack.push((index + 1, next, included));
            }
        }
    }
    result
}
