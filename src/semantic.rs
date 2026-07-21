// SPDX-License-Identifier: GPL-3.0-only

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct EvidenceBundle {
    pub schema_version: u8,
    pub frozen_at: String,
    pub providers: Vec<ProviderState>,
    pub edges: Vec<EvidenceEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticError {
    UnsupportedSchemaVersion(u8),
    DuplicateProvider(String),
    UndeclaredProvider(String),
}

impl fmt::Display for SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported semantic evidence version {version}")
            }
            Self::DuplicateProvider(provider) => {
                write!(formatter, "duplicate semantic provider state '{provider}'")
            }
            Self::UndeclaredProvider(provider) => {
                write!(
                    formatter,
                    "semantic edge references undeclared provider '{provider}'"
                )
            }
        }
    }
}

impl std::error::Error for SemanticError {}

impl EvidenceBundle {
    pub fn validate(&self) -> Result<(), SemanticError> {
        if self.schema_version != 1 {
            return Err(SemanticError::UnsupportedSchemaVersion(self.schema_version));
        }
        let mut providers = HashSet::with_capacity(self.providers.len());
        for provider in &self.providers {
            if !providers.insert(provider.provider.as_str()) {
                return Err(SemanticError::DuplicateProvider(provider.provider.clone()));
            }
        }
        if let Some(edge) = self
            .edges
            .iter()
            .find(|edge| !providers.contains(edge.provider.as_str()))
        {
            return Err(SemanticError::UndeclaredProvider(edge.provider.clone()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProviderState {
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_or_algorithm: Option<String>,
    pub state: ProviderStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub error_codes: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ProviderStatus {
    Disabled,
    Fresh,
    Cached,
    Stale,
    Partial,
    Unavailable,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct EvidenceEdge {
    pub provider: String,
    pub dataset_or_algorithm: Option<String>,
    pub source: Entity,
    pub candidate: Entity,
    pub scope: EvidenceScope,
    pub raw_rank: Option<u64>,
    pub raw_score: Option<f64>,
    pub identity_confidence: f64,
    pub observed_at: Option<String>,
    pub cache_state: Option<CacheState>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Entity {
    pub kind: EntityKind,
    pub id: String,
    pub mbid: Option<String>,
    pub name: Option<String>,
    pub title: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    Recording,
    Artist,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceScope {
    EndpointLocal,
    CollectionFallback,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheState {
    Fresh,
    Cached,
    Stale,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackIdentity {
    pub recording_id: String,
    pub recording_mbid: Option<String>,
    pub artist_ids: Vec<String>,
    pub artist_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateIdentity {
    pub candidate: usize,
    pub track: TrackIdentity,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SemanticTier {
    RecordingBoth,
    RecordingOne,
    ArtistLocal,
    ArtistCollection,
    BlissOnly,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticPool {
    EndpointLocal,
    CollectionFallback,
    BlissOnly,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SourceEndpoint {
    Left,
    Right,
    Collection,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MatchedEvidence {
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_or_algorithm: Option<String>,
    pub source_endpoint: SourceEndpoint,
    pub source_id: String,
    pub kind: EntityKind,
    pub scope: EvidenceScope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_rank: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_score: Option<f64>,
    pub identity_confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_state: Option<CacheState>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSemantics {
    pub candidate: usize,
    pub tier: SemanticTier,
    pub evidence: Vec<MatchedEvidence>,
}

impl CandidateSemantics {
    pub fn compare_priority(&self, other: &Self) -> Ordering {
        self.tier
            .cmp(&other.tier)
            .then_with(|| {
                other
                    .max_identity_confidence()
                    .total_cmp(&self.max_identity_confidence())
            })
            .then_with(|| compare_optional_rank(self.best_raw_rank(), other.best_raw_rank()))
    }

    fn max_identity_confidence(&self) -> f64 {
        self.evidence
            .iter()
            .map(|evidence| evidence.identity_confidence)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0)
    }

    fn best_raw_rank(&self) -> Option<u64> {
        self.evidence
            .iter()
            .filter_map(|evidence| evidence.raw_rank)
            .min()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GapEvidence {
    pub pool: SemanticPool,
    pub candidates: Vec<CandidateSemantics>,
}

fn compare_optional_rank(left: Option<u64>, right: Option<u64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

pub fn normalize_identity(value: &str) -> String {
    value
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn canonical_artist_id(name: &str) -> String {
    format!("artist:{}", normalize_identity(name))
}

fn recording_matches(entity: &Entity, track: &TrackIdentity) -> bool {
    entity.kind == EntityKind::Recording
        && (entity.id == track.recording_id
            || entity
                .mbid
                .as_ref()
                .zip(track.recording_mbid.as_ref())
                .is_some_and(|(left, right)| left.eq_ignore_ascii_case(right)))
}

fn artist_matches(entity: &Entity, track: &TrackIdentity) -> bool {
    if entity.kind != EntityKind::Artist {
        return false;
    }
    let id_match = track
        .artist_ids
        .iter()
        .any(|identity| entity.id.eq_ignore_ascii_case(identity));
    let mbid_match = entity.mbid.as_ref().is_some_and(|mbid| {
        track
            .artist_ids
            .iter()
            .any(|identity| mbid.eq_ignore_ascii_case(identity))
    });
    let name_match = entity
        .name
        .as_deref()
        .is_some_and(|name| normalize_identity(name) == track.artist_name);
    id_match || mbid_match || name_match
}

fn source_matches(edge: &EvidenceEdge, track: &TrackIdentity) -> bool {
    match edge.source.kind {
        EntityKind::Recording => recording_matches(&edge.source, track),
        EntityKind::Artist => artist_matches(&edge.source, track),
    }
}

fn candidate_matches(edge: &EvidenceEdge, track: &TrackIdentity) -> bool {
    if edge.source.kind != edge.candidate.kind {
        return false;
    }
    match edge.candidate.kind {
        EntityKind::Recording => recording_matches(&edge.candidate, track),
        EntityKind::Artist => artist_matches(&edge.candidate, track),
    }
}

fn matched_evidence(edge: &EvidenceEdge, source_endpoint: SourceEndpoint) -> MatchedEvidence {
    MatchedEvidence {
        provider: edge.provider.clone(),
        dataset_or_algorithm: edge.dataset_or_algorithm.clone(),
        source_endpoint,
        source_id: edge.source.id.clone(),
        kind: edge.source.kind,
        scope: edge.scope,
        raw_rank: edge.raw_rank,
        raw_score: edge.raw_score,
        identity_confidence: edge.identity_confidence,
        observed_at: edge.observed_at.clone(),
        cache_state: edge.cache_state,
    }
}

fn sort_evidence(evidence: &mut [MatchedEvidence]) {
    evidence.sort_by(|left, right| {
        left.source_endpoint
            .cmp(&right.source_endpoint)
            .then_with(|| left.kind.cmp(&right.kind))
            .then_with(|| left.provider.cmp(&right.provider))
            .then_with(|| compare_optional_rank(left.raw_rank, right.raw_rank))
            .then_with(|| left.source_id.cmp(&right.source_id))
    });
}

fn local_candidate(
    bundle: &EvidenceBundle,
    left: &TrackIdentity,
    right: &TrackIdentity,
    candidate: &CandidateIdentity,
) -> Option<CandidateSemantics> {
    let mut evidence = Vec::new();
    let mut recording_left = false;
    let mut recording_right = false;
    let mut artist_local = false;
    for edge in &bundle.edges {
        if edge.scope != EvidenceScope::EndpointLocal || !candidate_matches(edge, &candidate.track)
        {
            continue;
        }
        if source_matches(edge, left) {
            recording_left |= edge.source.kind == EntityKind::Recording;
            artist_local |= edge.source.kind == EntityKind::Artist;
            evidence.push(matched_evidence(edge, SourceEndpoint::Left));
        }
        if source_matches(edge, right) {
            recording_right |= edge.source.kind == EntityKind::Recording;
            artist_local |= edge.source.kind == EntityKind::Artist;
            evidence.push(matched_evidence(edge, SourceEndpoint::Right));
        }
    }
    let tier = if recording_left && recording_right {
        SemanticTier::RecordingBoth
    } else if recording_left || recording_right {
        SemanticTier::RecordingOne
    } else if artist_local {
        SemanticTier::ArtistLocal
    } else {
        return None;
    };
    sort_evidence(&mut evidence);
    Some(CandidateSemantics {
        candidate: candidate.candidate,
        tier,
        evidence,
    })
}

fn collection_candidate(
    bundle: &EvidenceBundle,
    collection_sources: &[TrackIdentity],
    candidate: &CandidateIdentity,
) -> Option<CandidateSemantics> {
    let mut evidence = bundle
        .edges
        .iter()
        .filter(|edge| {
            edge.scope == EvidenceScope::CollectionFallback
                && edge.source.kind == EntityKind::Artist
                && edge.candidate.kind == EntityKind::Artist
                && collection_sources
                    .iter()
                    .any(|source| artist_matches(&edge.source, source))
                && artist_matches(&edge.candidate, &candidate.track)
        })
        .map(|edge| matched_evidence(edge, SourceEndpoint::Collection))
        .collect::<Vec<_>>();
    if evidence.is_empty() {
        return None;
    }
    sort_evidence(&mut evidence);
    Some(CandidateSemantics {
        candidate: candidate.candidate,
        tier: SemanticTier::ArtistCollection,
        evidence,
    })
}

fn endpoint_candidate(
    bundle: &EvidenceBundle,
    anchor: &TrackIdentity,
    source_endpoint: SourceEndpoint,
    candidate: &CandidateIdentity,
) -> Option<CandidateSemantics> {
    let mut evidence = Vec::new();
    let mut recording = false;
    let mut artist = false;
    for edge in &bundle.edges {
        if edge.scope != EvidenceScope::EndpointLocal
            || !candidate_matches(edge, &candidate.track)
            || !source_matches(edge, anchor)
        {
            continue;
        }
        recording |= edge.source.kind == EntityKind::Recording;
        artist |= edge.source.kind == EntityKind::Artist;
        evidence.push(matched_evidence(edge, source_endpoint));
    }
    let tier = if recording {
        SemanticTier::RecordingOne
    } else if artist {
        SemanticTier::ArtistLocal
    } else {
        return None;
    };
    sort_evidence(&mut evidence);
    Some(CandidateSemantics {
        candidate: candidate.candidate,
        tier,
        evidence,
    })
}

pub fn select_endpoint_candidates(
    bundle: &EvidenceBundle,
    anchor: &TrackIdentity,
    source_endpoint: SourceEndpoint,
    collection_sources: &[TrackIdentity],
    candidates: &[CandidateIdentity],
) -> GapEvidence {
    let local = candidates
        .par_iter()
        .filter_map(|candidate| endpoint_candidate(bundle, anchor, source_endpoint, candidate))
        .collect::<Vec<_>>();
    if !local.is_empty() {
        return GapEvidence {
            pool: SemanticPool::EndpointLocal,
            candidates: local,
        };
    }

    let collection = candidates
        .par_iter()
        .filter_map(|candidate| collection_candidate(bundle, collection_sources, candidate))
        .collect::<Vec<_>>();
    if !collection.is_empty() {
        return GapEvidence {
            pool: SemanticPool::CollectionFallback,
            candidates: collection,
        };
    }

    GapEvidence {
        pool: SemanticPool::BlissOnly,
        candidates: candidates
            .par_iter()
            .map(|candidate| CandidateSemantics {
                candidate: candidate.candidate,
                tier: SemanticTier::BlissOnly,
                evidence: Vec::new(),
            })
            .collect(),
    }
}

pub fn select_gap_candidates(
    bundle: &EvidenceBundle,
    left: &TrackIdentity,
    right: &TrackIdentity,
    collection_sources: &[TrackIdentity],
    candidates: &[CandidateIdentity],
) -> GapEvidence {
    let local = candidates
        .par_iter()
        .filter_map(|candidate| local_candidate(bundle, left, right, candidate))
        .collect::<Vec<_>>();
    if !local.is_empty() {
        return GapEvidence {
            pool: SemanticPool::EndpointLocal,
            candidates: local,
        };
    }

    let collection = candidates
        .par_iter()
        .filter_map(|candidate| collection_candidate(bundle, collection_sources, candidate))
        .collect::<Vec<_>>();
    if !collection.is_empty() {
        return GapEvidence {
            pool: SemanticPool::CollectionFallback,
            candidates: collection,
        };
    }

    GapEvidence {
        pool: SemanticPool::BlissOnly,
        candidates: candidates
            .par_iter()
            .map(|candidate| CandidateSemantics {
                candidate: candidate.candidate,
                tier: SemanticTier::BlissOnly,
                evidence: Vec::new(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(recording_id: &str, artist: &str) -> TrackIdentity {
        TrackIdentity {
            recording_id: recording_id.to_owned(),
            recording_mbid: None,
            artist_ids: vec![canonical_artist_id(artist)],
            artist_name: normalize_identity(artist),
        }
    }

    fn entity(kind: EntityKind, id: &str) -> Entity {
        Entity {
            kind,
            id: id.to_owned(),
            mbid: None,
            name: None,
            title: None,
        }
    }

    fn edge(
        source_kind: EntityKind,
        source_id: &str,
        candidate_kind: EntityKind,
        candidate_id: &str,
        scope: EvidenceScope,
        rank: u64,
    ) -> EvidenceEdge {
        EvidenceEdge {
            provider: "fixture".to_owned(),
            dataset_or_algorithm: Some("fixture-v1".to_owned()),
            source: entity(source_kind, source_id),
            candidate: entity(candidate_kind, candidate_id),
            scope,
            raw_rank: Some(rank),
            raw_score: None,
            identity_confidence: 1.0,
            observed_at: Some("2026-07-20T00:00:00Z".to_owned()),
            cache_state: Some(CacheState::Cached),
        }
    }

    #[test]
    fn endpoint_recordings_precede_artist_and_suppress_collection_fallback() {
        let left = track("left", "Artist Left");
        let right = track("right", "Artist Right");
        let recording = CandidateIdentity {
            candidate: 10,
            track: track("candidate-recording", "Artist Recording"),
        };
        let artist = CandidateIdentity {
            candidate: 11,
            track: track("candidate-artist", "Artist Local"),
        };
        let collection = CandidateIdentity {
            candidate: 12,
            track: track("candidate-collection", "Artist Collection"),
        };
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-07-20T00:00:00Z".to_owned(),
            providers: Vec::new(),
            edges: vec![
                edge(
                    EntityKind::Recording,
                    "left",
                    EntityKind::Recording,
                    "candidate-recording",
                    EvidenceScope::EndpointLocal,
                    2,
                ),
                edge(
                    EntityKind::Recording,
                    "right",
                    EntityKind::Recording,
                    "candidate-recording",
                    EvidenceScope::EndpointLocal,
                    1,
                ),
                edge(
                    EntityKind::Artist,
                    &canonical_artist_id("Artist Left"),
                    EntityKind::Artist,
                    &canonical_artist_id("Artist Local"),
                    EvidenceScope::EndpointLocal,
                    1,
                ),
                edge(
                    EntityKind::Artist,
                    &canonical_artist_id("Artist Right"),
                    EntityKind::Artist,
                    &canonical_artist_id("Artist Collection"),
                    EvidenceScope::CollectionFallback,
                    1,
                ),
            ],
        };
        let collection_sources = vec![left.clone(), right.clone()];
        let candidates = vec![recording, artist, collection];
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                select_gap_candidates(&bundle, &left, &right, &collection_sources, &candidates)
            });
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                select_gap_candidates(&bundle, &left, &right, &collection_sources, &candidates)
            });
        assert_eq!(one, four);
        let selected = one;
        assert_eq!(selected.pool, SemanticPool::EndpointLocal);
        assert_eq!(selected.candidates.len(), 2);
        assert_eq!(selected.candidates[0].tier, SemanticTier::RecordingBoth);
        assert_eq!(selected.candidates[1].tier, SemanticTier::ArtistLocal);
        assert_eq!(
            selected.candidates[0].compare_priority(&selected.candidates[1]),
            Ordering::Less
        );
    }

    #[test]
    fn one_anchor_endpoint_never_fabricates_two_sided_recording_support() {
        let anchor = track("anchor", "Artist Anchor");
        let other_source = track("other", "Artist Other");
        let candidate = CandidateIdentity {
            candidate: 10,
            track: track("candidate", "Artist Candidate"),
        };
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-07-20T00:00:00Z".to_owned(),
            providers: Vec::new(),
            edges: vec![
                edge(
                    EntityKind::Recording,
                    "anchor",
                    EntityKind::Recording,
                    "candidate",
                    EvidenceScope::EndpointLocal,
                    1,
                ),
                edge(
                    EntityKind::Recording,
                    "other",
                    EntityKind::Recording,
                    "candidate",
                    EvidenceScope::EndpointLocal,
                    2,
                ),
            ],
        };
        let candidates = [candidate];
        let collection_sources = [anchor.clone(), other_source];
        let opening = select_endpoint_candidates(
            &bundle,
            &anchor,
            SourceEndpoint::Right,
            &collection_sources,
            &candidates,
        );
        assert_eq!(opening.pool, SemanticPool::EndpointLocal);
        assert_eq!(opening.candidates[0].tier, SemanticTier::RecordingOne);
        assert_eq!(opening.candidates[0].evidence.len(), 1);
        assert_eq!(
            opening.candidates[0].evidence[0].source_endpoint,
            SourceEndpoint::Right
        );

        let empty = EvidenceBundle {
            edges: Vec::new(),
            ..bundle
        };
        let fallback = select_endpoint_candidates(
            &empty,
            &anchor,
            SourceEndpoint::Left,
            &collection_sources,
            &candidates,
        );
        assert_eq!(fallback.pool, SemanticPool::BlissOnly);
        assert_eq!(fallback.candidates[0].tier, SemanticTier::BlissOnly);
    }

    #[test]
    fn failed_provider_without_edges_falls_back_to_bliss() {
        let left = track("left", "Artist Left");
        let right = track("right", "Artist Right");
        let candidate = CandidateIdentity {
            candidate: 10,
            track: track("candidate", "Artist Candidate"),
        };
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-07-20T00:00:00Z".to_owned(),
            providers: vec![ProviderState {
                provider: "offline-provider".to_owned(),
                dataset_or_algorithm: None,
                state: ProviderStatus::Failed,
                request_count: Some(1),
                failure_count: Some(1),
                error_codes: vec!["timeout".to_owned()],
            }],
            edges: Vec::new(),
        };
        let selected = select_gap_candidates(
            &bundle,
            &left,
            &right,
            &[left.clone(), right.clone()],
            &[candidate],
        );
        assert_eq!(selected.pool, SemanticPool::BlissOnly);
        assert_eq!(selected.candidates[0].tier, SemanticTier::BlissOnly);
    }

    #[test]
    fn collection_evidence_is_used_only_after_the_local_pool_is_empty() {
        let left = track("left", "Artist Left");
        let right = track("right", "Artist Right");
        let candidate = CandidateIdentity {
            candidate: 12,
            track: track("candidate", "Artist Collection"),
        };
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-07-20T00:00:00Z".to_owned(),
            providers: Vec::new(),
            edges: vec![edge(
                EntityKind::Artist,
                &canonical_artist_id("Artist Left"),
                EntityKind::Artist,
                &canonical_artist_id("Artist Collection"),
                EvidenceScope::CollectionFallback,
                1,
            )],
        };
        let selected = select_gap_candidates(
            &bundle,
            &left,
            &right,
            &[left.clone(), right.clone()],
            &[candidate],
        );
        assert_eq!(selected.pool, SemanticPool::CollectionFallback);
        assert_eq!(selected.candidates[0].tier, SemanticTier::ArtistCollection);
    }

    #[test]
    fn every_edge_requires_one_declared_provider_state() {
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-07-20T00:00:00Z".to_owned(),
            providers: Vec::new(),
            edges: vec![edge(
                EntityKind::Recording,
                "source",
                EntityKind::Recording,
                "candidate",
                EvidenceScope::EndpointLocal,
                1,
            )],
        };
        assert_eq!(
            bundle.validate(),
            Err(SemanticError::UndeclaredProvider("fixture".to_owned()))
        );
    }
}
