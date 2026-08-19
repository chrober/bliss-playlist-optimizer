// SPDX-License-Identifier: GPL-3.0-only

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
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
    pub title_name: String,
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

    fn evidence_strength(evidence: &MatchedEvidence) -> f64 {
        evidence
            .raw_score
            .map(|score| score.clamp(0.0, 1.0))
            .or_else(|| {
                evidence
                    .raw_rank
                    .map(|rank| 1.0 / (1.0 + (rank.saturating_sub(1) as f64 / 10.0)))
            })
            .unwrap_or(0.5)
            * evidence.identity_confidence.clamp(0.0, 1.0)
    }

    pub fn track_support(&self) -> f64 {
        let left = self
            .evidence
            .iter()
            .filter(|evidence| {
                evidence.provider.eq_ignore_ascii_case("last.fm")
                    && evidence.kind == EntityKind::Recording
                    && evidence.source_endpoint == SourceEndpoint::Left
            })
            .map(Self::evidence_strength)
            .max_by(f64::total_cmp);
        let right = self
            .evidence
            .iter()
            .filter(|evidence| {
                evidence.provider.eq_ignore_ascii_case("last.fm")
                    && evidence.kind == EntityKind::Recording
                    && evidence.source_endpoint == SourceEndpoint::Right
            })
            .map(Self::evidence_strength)
            .max_by(f64::total_cmp);
        match (left, right) {
            (Some(left), Some(right)) => (left + right) / 2.0,
            (Some(value), None) | (None, Some(value)) => value * 0.65,
            (None, None) => 0.0,
        }
    }

    pub fn artist_support(&self) -> f64 {
        self.evidence
            .iter()
            .filter(|evidence| {
                evidence.provider.eq_ignore_ascii_case("last.fm")
                    && evidence.kind == EntityKind::Artist
            })
            .map(|evidence| {
                let scope_factor = if evidence.scope == EvidenceScope::CollectionFallback {
                    0.5
                } else {
                    1.0
                };
                Self::evidence_strength(evidence) * scope_factor
            })
            .max_by(f64::total_cmp)
            .unwrap_or(0.0)
    }

    pub fn guidance_score(&self, track_percent: u8, artist_percent: u8) -> f64 {
        ((f64::from(track_percent) / 100.0) * self.track_support()
            + (f64::from(artist_percent) / 100.0) * self.artist_support())
        .min(1.0)
    }

    pub fn seed_guidance_score(&self, track_percent: u8, artist_percent: u8) -> f64 {
        let recording_support = self
            .evidence
            .iter()
            .filter(|evidence| {
                evidence.provider.eq_ignore_ascii_case("last.fm")
                    && evidence.kind == EntityKind::Recording
            })
            .map(Self::evidence_strength)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        ((f64::from(track_percent) / 100.0) * recording_support
            + (f64::from(artist_percent) / 100.0) * self.artist_support())
        .min(1.0)
    }

    pub fn adjusted_percentile(
        &self,
        acoustic_percentile: f64,
        track_percent: u8,
        artist_percent: u8,
    ) -> f64 {
        const MAX_GUIDANCE_ADJUSTMENT: f64 = 0.10;
        acoustic_percentile
            - MAX_GUIDANCE_ADJUSTMENT * self.guidance_score(track_percent, artist_percent)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GapEvidence {
    pub pool: SemanticPool,
    pub candidates: Vec<CandidateSemantics>,
}

#[derive(Clone, Debug)]
pub struct CandidateLookup {
    recording: HashMap<String, Vec<usize>>,
    artist: HashMap<String, Vec<usize>>,
}

#[derive(Default)]
struct CandidateAccumulator {
    evidence: Vec<MatchedEvidence>,
    recording_left: bool,
    recording_right: bool,
    artist_local: bool,
}

impl CandidateLookup {
    pub fn new(candidates: &[CandidateIdentity]) -> Self {
        let mut lookup = Self {
            recording: HashMap::new(),
            artist: HashMap::new(),
        };
        for candidate in candidates {
            for key in recording_keys_for_track(&candidate.track) {
                lookup
                    .recording
                    .entry(key)
                    .or_default()
                    .push(candidate.candidate);
            }
            for key in artist_keys_for_track(&candidate.track) {
                lookup
                    .artist
                    .entry(key)
                    .or_default()
                    .push(candidate.candidate);
            }
        }
        lookup
    }

    pub fn from_library_candidates<'a, I>(bundle: &EvidenceBundle, candidates: I) -> Self
    where
        I: IntoIterator<Item = (usize, u64, &'a str, &'a str)>,
    {
        let mut required_recording_rows = HashSet::new();
        let mut required_recording_pairs = HashMap::<String, HashMap<String, String>>::new();
        let mut required_artist_keys = HashMap::<String, Vec<String>>::new();

        for edge in &bundle.edges {
            match edge.candidate.kind {
                EntityKind::Recording => {
                    for key in recording_keys_for_entity(&edge.candidate) {
                        if let Some(row_id) = key
                            .strip_prefix("bliss-row-")
                            .and_then(|value| value.parse::<u64>().ok())
                        {
                            required_recording_rows.insert(row_id);
                        }
                        if let Some(pair) = key.strip_prefix("title_artist:") {
                            if let Some((title, artist)) = pair.split_once('\0') {
                                required_recording_pairs
                                    .entry(artist.to_owned())
                                    .or_default()
                                    .insert(title.to_owned(), key);
                            }
                        }
                    }
                }
                EntityKind::Artist => {
                    for key in artist_keys_for_entity(&edge.candidate) {
                        let normalized_name = key
                            .strip_prefix("artist:")
                            .or_else(|| key.strip_prefix("artist_name:"));
                        if let Some(normalized_name) = normalized_name {
                            required_artist_keys
                                .entry(normalized_name.to_owned())
                                .or_default()
                                .push(key);
                        }
                    }
                }
            }
        }
        for keys in required_artist_keys.values_mut() {
            keys.sort();
            keys.dedup();
        }

        let mut lookup = Self {
            recording: HashMap::new(),
            artist: HashMap::new(),
        };
        for (candidate, row_id, title_key, artist_key) in candidates {
            if required_recording_rows.contains(&row_id) {
                lookup
                    .recording
                    .entry(format!("bliss-row-{row_id}"))
                    .or_default()
                    .push(candidate);
            }
            let recording_key = if required_recording_pairs.is_empty() {
                None
            } else {
                required_recording_pairs
                    .get(artist_key)
                    .and_then(|titles| titles.get(title_key))
                    .cloned()
                    .or_else(|| {
                        (identity_key_needs_normalization(artist_key)
                            || identity_key_needs_normalization(title_key))
                        .then(|| {
                            let normalized_artist = normalize_identity(artist_key);
                            let normalized_title = normalize_identity(title_key);
                            required_recording_pairs
                                .get(&normalized_artist)
                                .and_then(|titles| titles.get(&normalized_title))
                                .cloned()
                        })
                        .flatten()
                    })
            };
            if let Some(recording_key) = recording_key {
                lookup
                    .recording
                    .entry(recording_key)
                    .or_default()
                    .push(candidate);
            }
            if !required_artist_keys.is_empty() {
                if let Some(keys) = required_artist_keys.get(artist_key) {
                    for key in keys {
                        lookup
                            .artist
                            .entry(key.clone())
                            .or_default()
                            .push(candidate);
                    }
                } else if identity_key_needs_normalization(artist_key) {
                    let normalized_artist = normalize_identity(artist_key);
                    if let Some(keys) = required_artist_keys.get(&normalized_artist) {
                        for key in keys {
                            lookup
                                .artist
                                .entry(key.clone())
                                .or_default()
                                .push(candidate);
                        }
                    }
                }
            }
        }
        lookup
    }

    fn candidates_for_entity(&self, entity: &Entity) -> Vec<usize> {
        let keys = match entity.kind {
            EntityKind::Recording => recording_keys_for_entity(entity),
            EntityKind::Artist => artist_keys_for_entity(entity),
        };
        let map = match entity.kind {
            EntityKind::Recording => &self.recording,
            EntityKind::Artist => &self.artist,
        };
        let mut seen = HashSet::new();
        let mut matches = Vec::new();
        for key in keys {
            if let Some(candidates) = map.get(&key) {
                for candidate in candidates {
                    if seen.insert(*candidate) {
                        matches.push(*candidate);
                    }
                }
            }
        }
        matches
    }
}

fn compare_optional_rank(left: Option<u64>, right: Option<u64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn recording_title_artist_key(title: &str, artist: &str) -> String {
    format!(
        "title_artist:{}\u{0}{}",
        normalize_identity(title),
        normalize_identity(artist)
    )
}

fn artist_name_key(name: &str) -> String {
    format!("artist_name:{}", normalize_identity(name))
}

fn recording_keys_for_track(track: &TrackIdentity) -> Vec<String> {
    let mut keys = vec![track.recording_id.clone()];
    if let Some(mbid) = &track.recording_mbid {
        keys.push(mbid.to_ascii_lowercase());
    }
    keys.push(recording_title_artist_key(
        &track.title_name,
        &track.artist_name,
    ));
    keys.sort();
    keys.dedup();
    keys
}

fn artist_keys_for_track(track: &TrackIdentity) -> Vec<String> {
    let mut keys = track.artist_ids.clone();
    keys.push(artist_name_key(&track.artist_name));
    keys.sort();
    keys.dedup();
    keys
}

fn recording_keys_for_entity(entity: &Entity) -> Vec<String> {
    let mut keys = Vec::new();
    keys.push(entity.id.clone());
    if let Some(mbid) = &entity.mbid {
        keys.push(mbid.to_ascii_lowercase());
    }
    if let (Some(title), Some(artist)) = (entity.title.as_deref(), entity.name.as_deref()) {
        keys.push(recording_title_artist_key(title, artist));
    }
    keys.sort();
    keys.dedup();
    keys
}

fn artist_keys_for_entity(entity: &Entity) -> Vec<String> {
    let mut keys = Vec::new();
    keys.push(entity.id.clone());
    if let Some(mbid) = &entity.mbid {
        keys.push(mbid.to_ascii_lowercase());
    }
    if let Some(name) = entity.name.as_deref() {
        keys.push(artist_name_key(name));
    }
    keys.sort();
    keys.dedup();
    keys
}

fn identity_key_needs_normalization(value: &str) -> bool {
    if value.trim() != value {
        return true;
    }
    let mut previous_space = false;
    for character in value.chars() {
        if character.is_whitespace() {
            if character != ' ' || previous_space {
                return true;
            }
            previous_space = true;
        } else {
            if character.is_uppercase() {
                return true;
            }
            previous_space = false;
        }
    }
    false
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
                .is_some_and(|(left, right)| left.eq_ignore_ascii_case(right))
            || entity.title.as_deref().is_some_and(|title| {
                normalize_identity(title) == track.title_name
                    && entity
                        .name
                        .as_deref()
                        .is_some_and(|artist| normalize_identity(artist) == track.artist_name)
            }))
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
        let local = local
            .into_iter()
            .map(|candidate| (candidate.candidate, candidate))
            .collect::<std::collections::HashMap<_, _>>();
        return GapEvidence {
            pool: SemanticPool::EndpointLocal,
            candidates: candidates
                .par_iter()
                .map(|candidate| {
                    local
                        .get(&candidate.candidate)
                        .cloned()
                        .unwrap_or_else(|| CandidateSemantics {
                            candidate: candidate.candidate,
                            tier: SemanticTier::BlissOnly,
                            evidence: Vec::new(),
                        })
                })
                .collect(),
        };
    }

    let collection = candidates
        .par_iter()
        .filter_map(|candidate| collection_candidate(bundle, collection_sources, candidate))
        .collect::<Vec<_>>();
    if !collection.is_empty() {
        let collection = collection
            .into_iter()
            .map(|candidate| (candidate.candidate, candidate))
            .collect::<std::collections::HashMap<_, _>>();
        return GapEvidence {
            pool: SemanticPool::CollectionFallback,
            candidates: candidates
                .par_iter()
                .map(|candidate| {
                    collection
                        .get(&candidate.candidate)
                        .cloned()
                        .unwrap_or_else(|| CandidateSemantics {
                            candidate: candidate.candidate,
                            tier: SemanticTier::BlissOnly,
                            evidence: Vec::new(),
                        })
                })
                .collect(),
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

pub fn select_collection_candidates(
    bundle: &EvidenceBundle,
    collection_sources: &[TrackIdentity],
    candidates: &[CandidateIdentity],
) -> Vec<CandidateSemantics> {
    candidates
        .par_iter()
        .filter_map(|candidate| collection_candidate(bundle, collection_sources, candidate))
        .collect()
}

pub fn select_seed_candidates(
    bundle: &EvidenceBundle,
    collection_sources: &[TrackIdentity],
    candidates: &[CandidateIdentity],
) -> Vec<CandidateSemantics> {
    candidates
        .par_iter()
        .filter_map(|candidate| {
            let mut evidence = bundle
                .edges
                .iter()
                .filter(|edge| {
                    candidate_matches(edge, &candidate.track)
                        && collection_sources
                            .iter()
                            .any(|source| source_matches(edge, source))
                })
                .map(|edge| {
                    matched_evidence(
                        edge,
                        if edge.scope == EvidenceScope::CollectionFallback {
                            SourceEndpoint::Collection
                        } else {
                            SourceEndpoint::Left
                        },
                    )
                })
                .collect::<Vec<_>>();
            if evidence.is_empty() {
                return None;
            }
            sort_evidence(&mut evidence);
            let tier = if evidence
                .iter()
                .any(|evidence| evidence.kind == EntityKind::Recording)
            {
                SemanticTier::RecordingOne
            } else if evidence
                .iter()
                .any(|evidence| evidence.scope == EvidenceScope::EndpointLocal)
            {
                SemanticTier::ArtistLocal
            } else {
                SemanticTier::ArtistCollection
            };
            Some(CandidateSemantics {
                candidate: candidate.candidate,
                tier,
                evidence,
            })
        })
        .collect()
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
        let local = local
            .into_iter()
            .map(|candidate| (candidate.candidate, candidate))
            .collect::<std::collections::HashMap<_, _>>();
        return GapEvidence {
            pool: SemanticPool::EndpointLocal,
            candidates: candidates
                .par_iter()
                .map(|candidate| {
                    local
                        .get(&candidate.candidate)
                        .cloned()
                        .unwrap_or_else(|| CandidateSemantics {
                            candidate: candidate.candidate,
                            tier: SemanticTier::BlissOnly,
                            evidence: Vec::new(),
                        })
                })
                .collect(),
        };
    }

    let collection = candidates
        .par_iter()
        .filter_map(|candidate| collection_candidate(bundle, collection_sources, candidate))
        .collect::<Vec<_>>();
    if !collection.is_empty() {
        let collection = collection
            .into_iter()
            .map(|candidate| (candidate.candidate, candidate))
            .collect::<std::collections::HashMap<_, _>>();
        return GapEvidence {
            pool: SemanticPool::CollectionFallback,
            candidates: candidates
                .par_iter()
                .map(|candidate| {
                    collection
                        .get(&candidate.candidate)
                        .cloned()
                        .unwrap_or_else(|| CandidateSemantics {
                            candidate: candidate.candidate,
                            tier: SemanticTier::BlissOnly,
                            evidence: Vec::new(),
                        })
                })
                .collect(),
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

fn semantics_from_accumulator(
    candidate: usize,
    mut accumulator: CandidateAccumulator,
) -> CandidateSemantics {
    let tier = if accumulator.recording_left && accumulator.recording_right {
        SemanticTier::RecordingBoth
    } else if accumulator.recording_left || accumulator.recording_right {
        SemanticTier::RecordingOne
    } else if accumulator.artist_local {
        SemanticTier::ArtistLocal
    } else {
        SemanticTier::ArtistCollection
    };
    sort_evidence(&mut accumulator.evidence);
    CandidateSemantics {
        candidate,
        tier,
        evidence: accumulator.evidence,
    }
}

pub fn select_gap_candidate_matches(
    bundle: &EvidenceBundle,
    left: &TrackIdentity,
    right: &TrackIdentity,
    collection_sources: &[TrackIdentity],
    lookup: &CandidateLookup,
) -> GapEvidence {
    let mut local = HashMap::<usize, CandidateAccumulator>::new();
    for edge in &bundle.edges {
        if edge.scope != EvidenceScope::EndpointLocal || edge.source.kind != edge.candidate.kind {
            continue;
        }
        let left_match = source_matches(edge, left);
        let right_match = source_matches(edge, right);
        if !left_match && !right_match {
            continue;
        }
        for candidate in lookup.candidates_for_entity(&edge.candidate) {
            let accumulator = local.entry(candidate).or_default();
            if left_match {
                accumulator.recording_left |= edge.source.kind == EntityKind::Recording;
                accumulator.artist_local |= edge.source.kind == EntityKind::Artist;
                accumulator
                    .evidence
                    .push(matched_evidence(edge, SourceEndpoint::Left));
            }
            if right_match {
                accumulator.recording_right |= edge.source.kind == EntityKind::Recording;
                accumulator.artist_local |= edge.source.kind == EntityKind::Artist;
                accumulator
                    .evidence
                    .push(matched_evidence(edge, SourceEndpoint::Right));
            }
        }
    }
    if !local.is_empty() {
        let mut candidates = local
            .into_iter()
            .map(|(candidate, accumulator)| semantics_from_accumulator(candidate, accumulator))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|candidate| candidate.candidate);
        return GapEvidence {
            pool: SemanticPool::EndpointLocal,
            candidates,
        };
    }

    let mut collection = HashMap::<usize, CandidateAccumulator>::new();
    for edge in &bundle.edges {
        if edge.scope != EvidenceScope::CollectionFallback
            || edge.source.kind != EntityKind::Artist
            || edge.candidate.kind != EntityKind::Artist
            || !collection_sources
                .iter()
                .any(|source| artist_matches(&edge.source, source))
        {
            continue;
        }
        for candidate in lookup.candidates_for_entity(&edge.candidate) {
            collection
                .entry(candidate)
                .or_default()
                .evidence
                .push(matched_evidence(edge, SourceEndpoint::Collection));
        }
    }
    if !collection.is_empty() {
        let mut candidates = collection
            .into_iter()
            .map(|(candidate, accumulator)| semantics_from_accumulator(candidate, accumulator))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|candidate| candidate.candidate);
        return GapEvidence {
            pool: SemanticPool::CollectionFallback,
            candidates,
        };
    }

    GapEvidence {
        pool: SemanticPool::BlissOnly,
        candidates: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(recording_id: &str, artist: &str) -> TrackIdentity {
        TrackIdentity {
            recording_id: recording_id.to_owned(),
            recording_mbid: None,
            title_name: normalize_identity(recording_id),
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
    fn indexed_gap_matches_legacy_semantics_without_bliss_placeholders() {
        let left = track("left", "Artist Left");
        let right = track("right", "Artist Right");
        let candidates = vec![
            CandidateIdentity {
                candidate: 10,
                track: track("Polly", "Nirvana"),
            },
            CandidateIdentity {
                candidate: 11,
                track: track("Unrelated", "Other Artist"),
            },
        ];
        let mut recording_edge = edge(
            EntityKind::Recording,
            "left",
            EntityKind::Recording,
            "recording:nirvana|polly",
            EvidenceScope::EndpointLocal,
            1,
        );
        recording_edge.candidate.name = Some("Nirvana".to_owned());
        recording_edge.candidate.title = Some("Polly".to_owned());
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-07-20T00:00:00Z".to_owned(),
            providers: Vec::new(),
            edges: vec![recording_edge],
        };
        let legacy = select_gap_candidates(
            &bundle,
            &left,
            &right,
            &[left.clone(), right.clone()],
            &candidates,
        );
        let indexed = select_gap_candidate_matches(
            &bundle,
            &left,
            &right,
            &[left.clone(), right.clone()],
            &CandidateLookup::new(&candidates),
        );
        let scoped_lookup = CandidateLookup::from_library_candidates(
            &bundle,
            candidates.iter().enumerate().map(|(row_id, candidate)| {
                (
                    candidate.candidate,
                    row_id as u64,
                    candidate.track.title_name.as_str(),
                    candidate.track.artist_name.as_str(),
                )
            }),
        );
        let scoped = select_gap_candidate_matches(
            &bundle,
            &left,
            &right,
            &[left.clone(), right.clone()],
            &scoped_lookup,
        );
        let legacy_supported = legacy
            .candidates
            .into_iter()
            .filter(|candidate| candidate.tier != SemanticTier::BlissOnly)
            .collect::<Vec<_>>();
        assert_eq!(indexed.pool, SemanticPool::EndpointLocal);
        assert_eq!(indexed.candidates, legacy_supported);
        assert_eq!(scoped, indexed);
    }

    #[test]
    fn scoped_lookup_keeps_two_hundred_thousand_candidates_memory_bounded() {
        const MATCHING_ROW: usize = 123_456;
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-07-20T00:00:00Z".to_owned(),
            providers: Vec::new(),
            edges: vec![edge(
                EntityKind::Recording,
                "left",
                EntityKind::Recording,
                "bliss-row-123456",
                EvidenceScope::EndpointLocal,
                1,
            )],
        };

        let lookup = CandidateLookup::from_library_candidates(
            &bundle,
            (0..200_000).map(|candidate| {
                (
                    candidate,
                    candidate as u64,
                    "unrelated title",
                    "unrelated artist",
                )
            }),
        );

        assert_eq!(
            lookup.recording.get("bliss-row-123456"),
            Some(&vec![MATCHING_ROW])
        );
        assert_eq!(lookup.recording.values().map(Vec::len).sum::<usize>(), 1);
        assert!(lookup.artist.is_empty());
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
        assert_eq!(selected.candidates.len(), 3);
        assert_eq!(selected.candidates[0].tier, SemanticTier::RecordingBoth);
        assert_eq!(selected.candidates[1].tier, SemanticTier::ArtistLocal);
        assert_eq!(selected.candidates[2].tier, SemanticTier::BlissOnly);
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
    fn lastfm_recording_metadata_matches_and_guidance_is_bounded() {
        let left = track("source", "Source Artist");
        let right = track("right", "Right Artist");
        let mut local_track = track("bliss-row-10", "Similar Artist");
        local_track.title_name = normalize_identity("Similar Song");
        let candidate = CandidateIdentity {
            candidate: 10,
            track: local_track,
        };
        let bliss_only = CandidateIdentity {
            candidate: 11,
            track: track("bliss-row-11", "Other Artist"),
        };
        let mut recording_edge = edge(
            EntityKind::Recording,
            "source",
            EntityKind::Recording,
            "lastfm-result",
            EvidenceScope::EndpointLocal,
            1,
        );
        recording_edge.provider = "last.fm".to_owned();
        recording_edge.dataset_or_algorithm = Some("track.getSimilar".to_owned());
        recording_edge.candidate.name = Some("Similar Artist".to_owned());
        recording_edge.candidate.title = Some("Similar Song".to_owned());
        recording_edge.raw_score = Some(0.9);
        let bundle = EvidenceBundle {
            schema_version: 1,
            frozen_at: "2026-08-04T00:00:00Z".to_owned(),
            providers: Vec::new(),
            edges: vec![recording_edge],
        };
        let selected = select_gap_candidates(
            &bundle,
            &left,
            &right,
            &[left.clone(), right.clone()],
            &[candidate, bliss_only],
        );
        assert_eq!(selected.candidates.len(), 2);
        let guided = selected
            .candidates
            .iter()
            .find(|candidate| candidate.candidate == 10)
            .unwrap();
        assert_eq!(guided.tier, SemanticTier::RecordingOne);
        assert!(guided.track_support() > 0.0);
        assert_eq!(guided.artist_support(), 0.0);
        assert_eq!(guided.adjusted_percentile(0.5, 0, 100), 0.5);
        let fully_guided = guided.adjusted_percentile(0.5, 100, 100);
        assert!(fully_guided < 0.5);
        assert!(fully_guided >= 0.4);
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
