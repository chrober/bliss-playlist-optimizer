// SPDX-License-Identifier: GPL-3.0-only

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bincode::Options;
use bliss_mixer_core::database::{BlissDatabase, SUPPORTED_SCHEMA_IDENTITY};
use bliss_mixer_core::scoring::score_adaptive_sequence;
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use bliss_playlist_optimizer::{bridge, preview, route, semantic};

const PROGRAM: &str = "bliss-playlist-optimizer";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const REQUEST_SCHEMA: &str = include_str!("../schemas/optimizer-request-v1.schema.json");
const SEMANTIC_SCHEMA: &str = include_str!("../schemas/semantic-evidence-v1.schema.json");
const LOCAL_CANDIDATE_INVENTORY_SCHEMA: &str =
    include_str!("../schemas/lms-local-candidate-inventory-v1.schema.json");
const DEFAULT_RETAINED_CANDIDATES: usize = 5;
const EXACT_COUNT_BEAM_WIDTH: usize = 64;
const SEMANTIC_SHORTLIST_RESERVE: usize = 32;
const LIBRARY_CACHE_VERSION: u8 = 1;
const MAX_LIBRARY_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const LIBRARY_CACHE_MAGIC: &[u8] = b"bliss-playlist-optimizer-library-cache-v1\n";

#[derive(Debug, Deserialize)]
struct Request {
    job_id: String,
    artifacts: Artifacts,
    source_tracks: Vec<SourceTrack>,
    scoring: Scoring,
    route: RouteSettings,
    repeat_windows: RepeatWindows,
    extension: ExtensionSettings,
    semantic_evidence: Artifact,
}

#[derive(Debug, Deserialize)]
struct Artifacts {
    database: Artifact,
    learned_matrix: Option<Artifact>,
    local_candidate_inventory: Option<Artifact>,
}

#[derive(Debug, Deserialize)]
struct Artifact {
    path: String,
    sha256: Option<String>,
    schema_identity: Option<String>,
    cache_identity: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LocalCandidateInventory {
    schema_identity: String,
    database_cache_identity: String,
    allowed_row_ids: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct SourceTrack {
    id: String,
    database_file: Option<String>,
    title: Option<String>,
    artist: Option<String>,
    album: Option<String>,
    recording_mbid: Option<String>,
    #[serde(default)]
    artist_mbids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Scoring {
    algorithm: String,
    adaptive: Option<AdaptiveSettings>,
}

#[derive(Debug, Deserialize)]
struct AdaptiveSettings {
    seed_limit: usize,
    learned_percent: u16,
}

#[derive(Debug, Deserialize)]
struct RouteSettings {
    ordering_policy: String,
    objective: String,
    start_track_id: Option<String>,
    destination_track_id: Option<String>,
    search: SearchSettings,
}

#[derive(Debug, Deserialize)]
struct SearchSettings {
    deterministic_seed: u64,
    restart_count: usize,
    time_budget_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RepeatWindows {
    artist: usize,
    album: usize,
    track: usize,
}

#[derive(Debug, Deserialize)]
struct ExtensionSettings {
    mode: String,
    additional_track_count: Option<usize>,
    target_track_count: Option<usize>,
    allow_opening_track: Option<bool>,
    allow_closing_track: Option<bool>,
    candidate_limit: Option<usize>,
    max_tracks_per_gap: Option<usize>,
    max_added_tracks: Option<usize>,
    trigger_percentile: Option<f64>,
    shortlist_limit: Option<usize>,
}
#[derive(Debug, Serialize)]
struct ValidationSummary {
    schema_version: u8,
    program: &'static str,
    version: &'static str,
    job_id: String,
    valid: bool,
    request_sha256: String,
    database_schema: &'static str,
    database_sha256: String,
    learned_matrix_sha256: Option<String>,
    local_candidate_inventory_sha256: Option<String>,
    local_candidate_track_count: Option<usize>,
    semantic_evidence_sha256: String,
    source_track_count: usize,
}

#[derive(Debug, Serialize)]
struct ScoringArtifact {
    schema_version: u8,
    artifact_kind: &'static str,
    program: &'static str,
    version: &'static str,
    core_api: &'static str,
    job_id: String,
    request_sha256: String,
    database_sha256: String,
    learned_matrix_sha256: String,
    semantic_evidence_sha256: String,
    algorithm_requested: String,
    learned_percent: u16,
    seed_limit: usize,
    parallel_execution: &'static str,
    source_track_ids: Vec<String>,
    legs: Vec<ContextualLeg>,
    transition_sum: f64,
    worst_transition: f64,
    objective: f64,
}

#[derive(Debug, Serialize)]
struct ContextualLeg {
    position: usize,
    seed_start: usize,
    seed_track_ids: Vec<String>,
    candidate_track_id: String,
    algorithm: String,
    distance: f64,
}

#[derive(Debug, Serialize)]
struct RouteArtifact {
    schema_version: u8,
    artifact_kind: &'static str,
    program: &'static str,
    version: &'static str,
    core_api: &'static str,
    job_id: String,
    request_sha256: String,
    database_sha256: String,
    learned_matrix_sha256: String,
    semantic_evidence_sha256: String,
    algorithm_requested: String,
    learned_percent: u16,
    seed_limit: usize,
    deterministic_seed: u64,
    restart_count: usize,
    parallel_execution: &'static str,
    search_tasks: usize,
    selected_strategy: &'static str,
    selected_track_ids: Vec<String>,
    primary: RouteCandidateArtifact,
    arc: RouteCandidateArtifact,
    repeat_validation: RepeatValidationArtifact,
    #[serde(skip_serializing_if = "Option::is_none")]
    performance: Option<PerformanceArtifact>,
}

#[derive(Debug, Serialize)]
struct RouteCandidateArtifact {
    strategy: &'static str,
    track_ids: Vec<String>,
    transition_sum: f64,
    worst_transition: f64,
    objective: f64,
    arc_error: f64,
}

#[derive(Debug, Serialize)]
struct RepeatValidationArtifact {
    valid: bool,
    track_window_satisfied_by_unique_membership: bool,
    violations: Vec<RepeatViolationArtifact>,
}

#[derive(Debug, Serialize)]
struct RepeatViolationArtifact {
    kind: &'static str,
    positions: [usize; 2],
}

#[derive(Debug, Serialize)]
struct BridgeAnalysisArtifact {
    schema_version: u8,
    artifact_kind: &'static str,
    program: &'static str,
    version: &'static str,
    core_api: &'static str,
    job_id: String,
    request_sha256: String,
    database_sha256: String,
    learned_matrix_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_candidate_inventory_sha256: Option<String>,
    semantic_evidence_sha256: String,
    algorithm_requested: String,
    ordering_policy: String,
    learned_percent: u16,
    seed_limit: usize,
    deterministic_seed: u64,
    restart_count: usize,
    parallel_execution: &'static str,
    selected_strategy: &'static str,
    source_track_ids: Vec<String>,
    selected_track_ids: Vec<String>,
    selected_route_objective: f64,
    usable_library_track_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_candidate_track_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    non_local_candidate_excluded_count: Option<usize>,
    eligible_candidate_count: usize,
    frozen_reference_count: usize,
    trigger_percentile: Option<f64>,
    max_leg_percentile: f64,
    max_detour_percentile: f64,
    retained_candidate_limit: usize,
    semantic_mode: String,
    provider_states: Vec<semantic::ProviderState>,
    gaps: Vec<BridgeGapArtifact>,
    selection_preview: SelectionPreviewArtifact,
    #[serde(skip_serializing_if = "Option::is_none")]
    performance: Option<PerformanceArtifact>,
}

#[derive(Debug, Serialize)]
struct PerformanceArtifact {
    total_ms: u64,
    database_cache: &'static str,
    stages: Vec<StageTimingArtifact>,
}

#[derive(Debug, Serialize)]
struct StageTimingArtifact {
    stage: &'static str,
    elapsed_ms: u64,
}

#[derive(Debug, Serialize)]
struct BridgeGapArtifact {
    position: usize,
    left_track_id: String,
    right_track_id: String,
    direct_distance: f64,
    direct_percentile: f64,
    triggering: Option<bool>,
    semantic_pool: semantic::SemanticPool,
    semantic_candidate_count: usize,
    semantic_excluded_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    shortlisted_candidate_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic_shortlist_excluded_count: Option<usize>,
    evaluated_candidate_count: usize,
    accepted_candidate_count: usize,
    repeat_rejected_count: usize,
    acoustic_rejected_count: usize,
    accepted_candidates: Vec<BridgeCandidateArtifact>,
}

#[derive(Debug, Serialize)]
struct BridgeCandidateArtifact {
    candidate_id: String,
    semantic_tier: semantic::SemanticTier,
    semantic_evidence: Vec<semantic::MatchedEvidence>,
    left_distance: f64,
    right_distance: f64,
    left_percentile: f64,
    right_percentile: f64,
    max_percentile: f64,
    detour_percentile: f64,
}

#[derive(Debug, Serialize)]
struct AutomaticSelectionArtifact {
    mode: &'static str,
    processing_order: &'static str,
    max_added_tracks: usize,
    added_track_count: usize,
    original_subsequence_preserved: bool,
    unique_membership: bool,
    final_sequence: Vec<PreviewSequenceEntryArtifact>,
    decisions: Vec<PreviewDecisionArtifact>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum SelectionPreviewArtifact {
    Automatic(AutomaticSelectionArtifact),
    Exact(ExactSelectionArtifact),
    SeedGrowth(SeedGrowthSelectionArtifact),
}

#[derive(Debug, Serialize)]
struct SeedGrowthSelectionArtifact {
    mode: &'static str,
    processing_order: &'static str,
    target_track_count: usize,
    requested_added_tracks: usize,
    feasible: bool,
    added_track_count: usize,
    original_subsequence_preserved: bool,
    unique_membership: bool,
    relevance_reference_track_count: usize,
    final_sequence: Vec<PreviewSequenceEntryArtifact>,
    selected_additions: Vec<SeedGrowthAdditionArtifact>,
}

#[derive(Debug, Serialize)]
struct SeedGrowthAdditionArtifact {
    candidate_id: String,
    relevance_distance: f64,
}

#[derive(Debug, Serialize)]
struct ExactSelectionArtifact {
    mode: &'static str,
    processing_order: &'static str,
    requested_added_tracks: usize,
    feasible: bool,
    added_track_count: usize,
    original_subsequence_preserved: Option<bool>,
    unique_membership: Option<bool>,
    final_sequence: Option<Vec<PreviewSequenceEntryArtifact>>,
    decisions: Vec<ExactPreviewDecisionArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint_policy: Option<EndpointPolicyArtifact>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    endpoint_decisions: Vec<EndpointDecisionArtifact>,
    search: ExactSearchArtifact,
    infeasibility: Option<ExactInfeasibilityArtifact>,
}

#[derive(Debug, Serialize)]
struct EndpointPolicyArtifact {
    opening_enabled: bool,
    closing_enabled: bool,
    maximum_opening_tracks: usize,
    maximum_closing_tracks: usize,
}

#[derive(Debug, Serialize)]
struct EndpointDecisionArtifact {
    slot: &'static str,
    anchor_track_id: String,
    semantic_pool: semantic::SemanticPool,
    reason: preview::DecisionReason,
    selected_track: Option<EndpointCandidateArtifact>,
}

#[derive(Debug, Serialize)]
struct EndpointCandidateArtifact {
    candidate_id: String,
    semantic_tier: semantic::SemanticTier,
    semantic_evidence: Vec<semantic::MatchedEvidence>,
    distance: f64,
    percentile: f64,
}

#[derive(Debug, Serialize)]
struct ExactSearchArtifact {
    beam_width: usize,
    candidate_limit: usize,
    max_tracks_per_gap: usize,
    evaluated_states: usize,
    retained_states: usize,
    maximum_additions_found: usize,
    structural_upper_bound: usize,
}

#[derive(Debug, Serialize)]
struct ExactInfeasibilityArtifact {
    code: &'static str,
    requested_added_tracks: usize,
    maximum_additions_found: usize,
    structural_upper_bound: usize,
}

#[derive(Debug, Serialize)]
struct ExactPreviewDecisionArtifact {
    original_position: usize,
    route_position: usize,
    left_track_id: String,
    right_track_id: String,
    direct_distance: f64,
    direct_percentile: f64,
    semantic_pool: semantic::SemanticPool,
    reason: preview::DecisionReason,
    selected_bridge: Option<BridgeCandidateArtifact>,
}

#[derive(Debug, Serialize)]
struct PreviewSequenceEntryArtifact {
    position: usize,
    kind: &'static str,
    track_id: String,
}

#[derive(Debug, Serialize)]
struct PreviewDecisionArtifact {
    original_position: usize,
    route_position: usize,
    left_track_id: String,
    right_track_id: String,
    direct_distance: f64,
    direct_percentile: f64,
    triggering: bool,
    semantic_pool: semantic::SemanticPool,
    reason: preview::DecisionReason,
    selected_bridge: Option<BridgeCandidateArtifact>,
}

#[derive(Clone, Deserialize, Serialize)]
struct LibraryTrack {
    row_id: u64,
    file: String,
    artist_key: String,
    title_key: String,
    route_track: route::RouteTrack,
}

#[derive(Deserialize, Serialize)]
struct LibraryCache {
    format_version: u8,
    database_path: String,
    database_identity: String,
    database_sha256: String,
    library: Vec<LibraryTrack>,
}

struct RuntimeOptions {
    timings: bool,
    cache_dir: Option<PathBuf>,
}

impl RuntimeOptions {
    fn disabled() -> Self {
        Self {
            timings: false,
            cache_dir: None,
        }
    }
}

#[derive(Default)]
struct StageTimings {
    stages: Vec<StageTimingArtifact>,
}

impl StageTimings {
    fn record(&mut self, stage: &'static str, elapsed: Duration) {
        self.stages.push(StageTimingArtifact {
            stage,
            elapsed_ms: elapsed.as_millis() as u64,
        });
    }

    fn finish(
        self,
        enabled: bool,
        started: Instant,
        database_cache: &'static str,
    ) -> Option<PerformanceArtifact> {
        enabled.then(|| PerformanceArtifact {
            total_ms: started.elapsed().as_millis() as u64,
            database_cache,
            stages: self.stages,
        })
    }
}

struct ValidatedRequest {
    summary: ValidationSummary,
    request: Request,
    learned_matrix: Option<Array2<f32>>,
    semantic_bundle: semantic::EvidenceBundle,
    library: Option<Vec<LibraryTrack>>,
    local_candidate_rows: Option<HashSet<u64>>,
    database_cache: &'static str,
}
#[derive(Debug, Serialize)]
struct CommandFailure {
    schema_version: u8,
    valid: bool,
    code: &'static str,
    message: String,
}

impl CommandFailure {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            valid: false,
            code,
            message: message.into(),
        }
    }
}

fn usage() -> &'static str {
    "Usage:\n  bliss-playlist-optimizer version [--json]\n  bliss-playlist-optimizer validate --request <request.json>\n  bliss-playlist-optimizer score --request <request.json>\n  bliss-playlist-optimizer route --request <request.json> [--timings] [--cache-dir <directory>]\n  bliss-playlist-optimizer bridge --request <request.json> [--timings] [--cache-dir <directory>]"
}

fn parse_request_command(args: &[String]) -> Option<(&str, &Path, RuntimeOptions)> {
    if args.len() < 3 || args[1] != "--request" {
        return None;
    }
    let command = args[0].as_str();
    if !matches!(command, "validate" | "score" | "route" | "bridge") {
        return None;
    }
    let mut options = RuntimeOptions::disabled();
    let mut index = 3;
    while index < args.len() {
        match args[index].as_str() {
            "--timings" if !options.timings => {
                options.timings = true;
                index += 1;
            }
            "--cache-dir" if options.cache_dir.is_none() && index + 1 < args.len() => {
                options.cache_dir = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            _ => return None,
        }
    }
    Some((command, Path::new(&args[2]), options))
}

fn default_parallel_workers(available: usize) -> usize {
    available.saturating_sub(1).max(1)
}

fn configure_parallelism() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(default_parallel_workers(available))
        .build_global()
        .expect("Rayon pool must be configured before scoring starts");
}

fn hash_artifact(artifact: &Artifact, kind: &'static str) -> Result<String, CommandFailure> {
    let file = File::open(&artifact.path).map_err(|error| {
        CommandFailure::new(
            "ARTIFACT_UNREADABLE",
            format!("cannot read {kind} artifact '{}': {error}", artifact.path),
        )
    })?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|error| {
            CommandFailure::new(
                "ARTIFACT_UNREADABLE",
                format!("cannot read {kind} artifact '{}': {error}", artifact.path),
            )
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let actual = format!("{:x}", digest.finalize());
    verify_artifact_hash(artifact, kind, &actual)?;
    Ok(actual)
}

fn verify_artifact_hash(
    artifact: &Artifact,
    kind: &'static str,
    actual: &str,
) -> Result<(), CommandFailure> {
    if let Some(expected) = &artifact.sha256 {
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(CommandFailure::new(
                "ARTIFACT_HASH_MISMATCH",
                format!(
                    "{kind} artifact '{}' does not match its declared SHA-256",
                    artifact.path
                ),
            ));
        }
    }
    Ok(())
}

fn library_cache_path(cache_dir: &Path, database_path: &str) -> PathBuf {
    let key = format!("{:x}", Sha256::digest(database_path.as_bytes()));
    cache_dir.join(format!("library-{key}.bin"))
}

fn load_library_cache(cache_dir: &Path, artifact: &Artifact) -> Option<LibraryCache> {
    let identity = artifact.cache_identity.as_deref()?;
    let path = library_cache_path(cache_dir, &artifact.path);
    let metadata = fs::metadata(&path).ok()?;
    if metadata.len() > MAX_LIBRARY_CACHE_BYTES {
        return None;
    }
    let bytes = fs::read(path).ok()?;
    let encoded = bytes.strip_prefix(LIBRARY_CACHE_MAGIC)?;
    if encoded.len() < 65 || encoded[64] != b'\n' {
        return None;
    }
    let declared_hash = std::str::from_utf8(&encoded[..64]).ok()?;
    let payload = &encoded[65..];
    let actual_hash = format!("{:x}", Sha256::digest(payload));
    if !actual_hash.eq_ignore_ascii_case(declared_hash) {
        return None;
    }
    let cache: LibraryCache = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_LIBRARY_CACHE_BYTES)
        .deserialize(payload)
        .ok()?;
    if cache.format_version != LIBRARY_CACHE_VERSION
        || cache.database_path != artifact.path
        || cache.database_identity != identity
        || verify_artifact_hash(artifact, "database", &cache.database_sha256).is_err()
    {
        return None;
    }
    Some(cache)
}

fn store_library_cache(cache_dir: &Path, artifact: &Artifact, cache: &LibraryCache) {
    if artifact.cache_identity.is_none() || fs::create_dir_all(cache_dir).is_err() {
        return;
    }
    let Ok(payload) = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(cache)
    else {
        return;
    };
    if payload.len() as u64 > MAX_LIBRARY_CACHE_BYTES {
        return;
    }
    let payload_hash = format!("{:x}", Sha256::digest(&payload));
    let mut bytes = Vec::with_capacity(LIBRARY_CACHE_MAGIC.len() + 65 + payload.len());
    bytes.extend_from_slice(LIBRARY_CACHE_MAGIC);
    bytes.extend_from_slice(payload_hash.as_bytes());
    bytes.push(b'\n');
    bytes.extend_from_slice(&payload);
    let destination = library_cache_path(cache_dir, &artifact.path);
    let temporary = destination.with_extension(format!("tmp-{}", std::process::id()));
    if fs::write(&temporary, bytes).is_err() {
        return;
    }
    if fs::rename(&temporary, &destination).is_err() {
        let _ = fs::remove_file(&destination);
        let _ = fs::rename(&temporary, &destination);
    }
}

fn read_artifact(
    artifact: &Artifact,
    kind: &'static str,
) -> Result<(Vec<u8>, String), CommandFailure> {
    let bytes = fs::read(&artifact.path).map_err(|error| {
        CommandFailure::new(
            "ARTIFACT_UNREADABLE",
            format!("cannot read {kind} artifact '{}': {error}", artifact.path),
        )
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if let Some(expected) = &artifact.sha256 {
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(CommandFailure::new(
                "ARTIFACT_HASH_MISMATCH",
                format!(
                    "{kind} artifact '{}' does not match its declared SHA-256",
                    artifact.path
                ),
            ));
        }
    }
    Ok((bytes, actual))
}

fn load_local_candidate_inventory(
    artifact: &Artifact,
    database_artifact: &Artifact,
    library: &[LibraryTrack],
) -> Result<(HashSet<u64>, String), CommandFailure> {
    if artifact.schema_identity.as_deref() != Some("lms-local-candidate-inventory-v1") {
        return Err(CommandFailure::new(
            "CANDIDATE_INVENTORY_SCHEMA_MISMATCH",
            "artifacts.local_candidate_inventory must declare lms-local-candidate-inventory-v1",
        ));
    }
    let database_identity = database_artifact.cache_identity.as_deref().ok_or_else(|| {
        CommandFailure::new(
            "CANDIDATE_INVENTORY_DATABASE_IDENTITY_REQUIRED",
            "the database cache identity is required when a local candidate inventory is supplied",
        )
    })?;
    let (bytes, hash) = read_artifact(artifact, "local candidate inventory")?;
    let value = parse_json(&bytes, "local candidate inventory")?;
    validate_json(
        &value,
        LOCAL_CANDIDATE_INVENTORY_SCHEMA,
        "local candidate inventory",
    )?;
    let inventory: LocalCandidateInventory = serde_json::from_value(value).map_err(|error| {
        CommandFailure::new(
            "CANDIDATE_INVENTORY_INVALID",
            format!("failed to decode local candidate inventory: {error}"),
        )
    })?;
    if inventory.schema_identity != "lms-local-candidate-inventory-v1" {
        return Err(CommandFailure::new(
            "CANDIDATE_INVENTORY_SCHEMA_MISMATCH",
            "the local candidate inventory payload has an unsupported schema identity",
        ));
    }
    if inventory.database_cache_identity != database_identity {
        return Err(CommandFailure::new(
            "CANDIDATE_INVENTORY_DATABASE_MISMATCH",
            "the local candidate inventory was generated for a different bliss.db identity",
        ));
    }
    let rows = inventory
        .allowed_row_ids
        .into_iter()
        .collect::<HashSet<_>>();
    let library_rows = library
        .iter()
        .map(|track| track.row_id)
        .collect::<HashSet<_>>();
    if let Some(unknown) = rows.iter().find(|row_id| !library_rows.contains(row_id)) {
        return Err(CommandFailure::new(
            "CANDIDATE_INVENTORY_UNKNOWN_ROW",
            format!("local candidate inventory contains unknown or unusable Bliss row {unknown}"),
        ));
    }
    Ok((rows, hash))
}

fn validate_json(
    value: &Value,
    schema_source: &str,
    kind: &'static str,
) -> Result<(), CommandFailure> {
    let schema: Value =
        serde_json::from_str(schema_source).expect("embedded schema must be valid JSON");
    let validator = jsonschema::validator_for(&schema).expect("embedded schema must compile");
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|error| error.to_string())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(CommandFailure::new(
            "INVALID_REQUEST",
            format!("{kind} schema validation failed: {}", errors.join("; ")),
        ))
    }
}

fn parse_json(bytes: &[u8], kind: &'static str) -> Result<Value, CommandFailure> {
    serde_json::from_slice(bytes).map_err(|error| {
        CommandFailure::new("INVALID_JSON", format!("invalid {kind} JSON: {error}"))
    })
}

fn decode_request(path: &Path) -> Result<Request, CommandFailure> {
    let request_bytes = fs::read(path).map_err(|error| {
        CommandFailure::new(
            "REQUEST_UNREADABLE",
            format!("cannot read request '{}': {error}", path.display()),
        )
    })?;
    let request_value = parse_json(&request_bytes, "request")?;
    validate_json(&request_value, REQUEST_SCHEMA, "request")?;
    serde_json::from_value(request_value).map_err(|error| {
        CommandFailure::new("INVALID_REQUEST", format!("cannot decode request: {error}"))
    })
}

fn decode_request_once(path: &Path) -> Result<(Request, String), CommandFailure> {
    let request_bytes = fs::read(path).map_err(|error| {
        CommandFailure::new(
            "REQUEST_UNREADABLE",
            format!("cannot read request '{}': {error}", path.display()),
        )
    })?;
    let request_sha256 = format!("{:x}", Sha256::digest(&request_bytes));
    let request_value = parse_json(&request_bytes, "request")?;
    validate_json(&request_value, REQUEST_SCHEMA, "request")?;
    let request = serde_json::from_value(request_value).map_err(|error| {
        CommandFailure::new("INVALID_REQUEST", format!("cannot decode request: {error}"))
    })?;
    Ok((request, request_sha256))
}

fn prepare_runtime_request(
    path: &Path,
    options: &RuntimeOptions,
    timings: &mut StageTimings,
) -> Result<ValidatedRequest, CommandFailure> {
    let started = Instant::now();
    let (request, request_sha256) = decode_request_once(path)?;
    timings.record("request_decode", started.elapsed());

    if let Some(identity) = &request.artifacts.database.schema_identity {
        if identity != "TracksV2" && identity != SUPPORTED_SCHEMA_IDENTITY {
            return Err(CommandFailure::new(
                "DATABASE_SCHEMA_MISMATCH",
                format!("unsupported database schema identity '{identity}'"),
            ));
        }
    }

    let started = Instant::now();
    let cached = options
        .cache_dir
        .as_deref()
        .and_then(|cache_dir| load_library_cache(cache_dir, &request.artifacts.database));
    timings.record("database_cache_read", started.elapsed());

    let (database_sha256, library, database_cache) = if let Some(cache) = cached {
        (cache.database_sha256, Some(cache.library), "hit")
    } else {
        let started = Instant::now();
        let database_sha256 = hash_artifact(&request.artifacts.database, "database")?;
        timings.record("database_hash", started.elapsed());

        let started = Instant::now();
        let database = BlissDatabase::open_read_only(&request.artifacts.database.path)
            .map_err(|error| CommandFailure::new("DATABASE_INVALID", error.to_string()))?;
        database
            .quick_check()
            .map_err(|error| CommandFailure::new("DATABASE_INTEGRITY_FAILED", error.to_string()))?;
        timings.record("database_open_and_integrity", started.elapsed());

        let started = Instant::now();
        let library = load_usable_library(&database)?;
        timings.record("library_decode", started.elapsed());

        if let (Some(cache_dir), Some(identity)) = (
            options.cache_dir.as_deref(),
            request.artifacts.database.cache_identity.as_deref(),
        ) {
            let started = Instant::now();
            store_library_cache(
                cache_dir,
                &request.artifacts.database,
                &LibraryCache {
                    format_version: LIBRARY_CACHE_VERSION,
                    database_path: request.artifacts.database.path.clone(),
                    database_identity: identity.to_owned(),
                    database_sha256: database_sha256.clone(),
                    library: library.clone(),
                },
            );
            timings.record("database_cache_write", started.elapsed());
        }
        let cache_state =
            if options.cache_dir.is_some() && request.artifacts.database.cache_identity.is_some() {
                "miss"
            } else {
                "disabled"
            };
        (database_sha256, Some(library), cache_state)
    };

    let started = Instant::now();
    let (local_candidate_rows, local_candidate_inventory_sha256) =
        if let Some(inventory) = &request.artifacts.local_candidate_inventory {
            let (rows, hash) = load_local_candidate_inventory(
                inventory,
                &request.artifacts.database,
                library
                    .as_deref()
                    .expect("runtime preparation always loads the library"),
            )?;
            (Some(rows), Some(hash))
        } else {
            (None, None)
        };
    timings.record("local_candidate_inventory_load", started.elapsed());

    let started = Instant::now();
    let (learned_matrix, learned_matrix_sha256) =
        if let Some(matrix) = &request.artifacts.learned_matrix {
            let (bytes, hash) = read_artifact(matrix, "learned matrix")?;
            let text = std::str::from_utf8(&bytes).map_err(|error| {
                CommandFailure::new("MATRIX_INVALID", format!("matrix is not UTF-8: {error}"))
            })?;
            let parsed = bliss_mixer_core::matrix::parse_learned_matrix(text)
                .map_err(|error| CommandFailure::new("MATRIX_INVALID", error.to_string()))?;
            (Some(parsed), Some(hash))
        } else {
            if matches!(
                request.scoring.algorithm.as_str(),
                "learned_matrix" | "adaptive"
            ) {
                return Err(CommandFailure::new(
                    "MATRIX_REQUIRED",
                    format!(
                        "{} scoring requires artifacts.learned_matrix",
                        request.scoring.algorithm
                    ),
                ));
            }
            (None, None)
        };
    timings.record("learned_matrix_load", started.elapsed());

    if let Some(identity) = &request.semantic_evidence.schema_identity {
        if identity != "semantic-evidence-v1" {
            return Err(CommandFailure::new(
                "SEMANTIC_SCHEMA_MISMATCH",
                format!("unsupported semantic evidence schema identity '{identity}'"),
            ));
        }
    }
    let started = Instant::now();
    let (semantic_bytes, semantic_evidence_sha256) =
        read_artifact(&request.semantic_evidence, "semantic evidence")?;
    let semantic_value = parse_json(&semantic_bytes, "semantic evidence")?;
    validate_json(&semantic_value, SEMANTIC_SCHEMA, "semantic evidence")?;
    let semantic_bundle: semantic::EvidenceBundle = serde_json::from_value(semantic_value)
        .map_err(|error| {
            CommandFailure::new(
                "SEMANTIC_EVIDENCE_INVALID",
                format!("failed to decode semantic evidence: {error}"),
            )
        })?;
    semantic_bundle.validate().map_err(|error| {
        CommandFailure::new(
            "SEMANTIC_EVIDENCE_INVALID",
            format!("invalid semantic evidence: {error}"),
        )
    })?;
    timings.record("semantic_evidence_load", started.elapsed());

    let started = Instant::now();
    let library = library.expect("runtime preparation always loads the library");
    let file_to_index = library
        .iter()
        .enumerate()
        .map(|(index, track)| (track.file.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut source_ids = HashSet::new();
    let mut database_files = HashSet::new();
    for track in &request.source_tracks {
        if !source_ids.insert(track.id.as_str()) {
            return Err(CommandFailure::new(
                "DUPLICATE_SOURCE_TRACK",
                format!("duplicate source track id '{}'", track.id),
            ));
        }
        let database_file = track.database_file.as_deref().ok_or_else(|| {
            CommandFailure::new(
                "TRACK_IDENTITY_INCOMPLETE",
                format!("source track '{}' has no database_file identity", track.id),
            )
        })?;
        if !database_files.insert(database_file) {
            return Err(CommandFailure::new(
                "DUPLICATE_SOURCE_TRACK",
                format!("duplicate Bliss file identity '{database_file}'"),
            ));
        }
        let Some(library_index) = file_to_index.get(database_file).copied() else {
            return Err(CommandFailure::new(
                "TRACK_NOT_ANALYZED",
                format!(
                    "source track '{}' is absent or ignored in the Bliss database",
                    track.id
                ),
            ));
        };
        if local_candidate_rows
            .as_ref()
            .is_some_and(|rows| !rows.contains(&library[library_index].row_id))
        {
            return Err(CommandFailure::new(
                "SOURCE_NOT_IN_LOCAL_CANDIDATE_INVENTORY",
                format!(
                    "source track '{}' is not present in the frozen LMS-local inventory",
                    track.id
                ),
            ));
        }
    }
    timings.record("source_resolution", started.elapsed());

    let summary = ValidationSummary {
        schema_version: 1,
        program: PROGRAM,
        version: VERSION,
        job_id: request.job_id.clone(),
        valid: true,
        request_sha256,
        database_schema: SUPPORTED_SCHEMA_IDENTITY,
        database_sha256,
        learned_matrix_sha256,
        local_candidate_inventory_sha256,
        local_candidate_track_count: local_candidate_rows.as_ref().map(HashSet::len),
        semantic_evidence_sha256,
        source_track_count: request.source_tracks.len(),
    };
    Ok(ValidatedRequest {
        summary,
        request,
        learned_matrix,
        semantic_bundle,
        library: Some(library),
        local_candidate_rows,
        database_cache,
    })
}

fn validate_request(path: &Path) -> Result<ValidationSummary, CommandFailure> {
    let request_bytes = fs::read(path).map_err(|error| {
        CommandFailure::new(
            "REQUEST_UNREADABLE",
            format!("cannot read request '{}': {error}", path.display()),
        )
    })?;
    let request_sha256 = format!("{:x}", Sha256::digest(&request_bytes));
    let request = decode_request(path)?;

    if let Some(identity) = &request.artifacts.database.schema_identity {
        if identity != "TracksV2" && identity != SUPPORTED_SCHEMA_IDENTITY {
            return Err(CommandFailure::new(
                "DATABASE_SCHEMA_MISMATCH",
                format!("unsupported database schema identity '{identity}'"),
            ));
        }
    }
    let (_, database_sha256) = read_artifact(&request.artifacts.database, "database")?;
    let database = BlissDatabase::open_read_only(&request.artifacts.database.path)
        .map_err(|error| CommandFailure::new("DATABASE_INVALID", error.to_string()))?;
    database
        .quick_check()
        .map_err(|error| CommandFailure::new("DATABASE_INTEGRITY_FAILED", error.to_string()))?;

    let (local_candidate_rows, local_candidate_inventory_sha256) =
        if let Some(inventory) = &request.artifacts.local_candidate_inventory {
            let library = load_usable_library(&database)?;
            let (rows, hash) =
                load_local_candidate_inventory(inventory, &request.artifacts.database, &library)?;
            (Some(rows), Some(hash))
        } else {
            (None, None)
        };

    let learned_matrix_sha256 = if let Some(matrix) = &request.artifacts.learned_matrix {
        let (_, hash) = read_artifact(matrix, "learned matrix")?;
        bliss_mixer_core::matrix::load_learned_matrix(&matrix.path)
            .map_err(|error| CommandFailure::new("MATRIX_INVALID", error.to_string()))?;
        Some(hash)
    } else {
        if matches!(
            request.scoring.algorithm.as_str(),
            "learned_matrix" | "adaptive"
        ) {
            return Err(CommandFailure::new(
                "MATRIX_REQUIRED",
                format!(
                    "{} scoring requires artifacts.learned_matrix",
                    request.scoring.algorithm
                ),
            ));
        }
        None
    };

    if let Some(identity) = &request.semantic_evidence.schema_identity {
        if identity != "semantic-evidence-v1" {
            return Err(CommandFailure::new(
                "SEMANTIC_SCHEMA_MISMATCH",
                format!("unsupported semantic evidence schema identity '{identity}'"),
            ));
        }
    }
    let (semantic_bytes, semantic_evidence_sha256) =
        read_artifact(&request.semantic_evidence, "semantic evidence")?;
    let semantic_value = parse_json(&semantic_bytes, "semantic evidence")?;
    validate_json(&semantic_value, SEMANTIC_SCHEMA, "semantic evidence")?;

    let mut source_ids = HashSet::new();
    let mut database_files = HashSet::new();
    for track in &request.source_tracks {
        if !source_ids.insert(track.id.as_str()) {
            return Err(CommandFailure::new(
                "DUPLICATE_SOURCE_TRACK",
                format!("duplicate source track id '{}'", track.id),
            ));
        }
        let database_file = track.database_file.as_deref().ok_or_else(|| {
            CommandFailure::new(
                "TRACK_IDENTITY_INCOMPLETE",
                format!("source track '{}' has no database_file identity", track.id),
            )
        })?;
        if !database_files.insert(database_file) {
            return Err(CommandFailure::new(
                "DUPLICATE_SOURCE_TRACK",
                format!("duplicate Bliss file identity '{database_file}'"),
            ));
        }
        let row_id = database
            .usable_row_id_for_file(database_file)
            .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?;
        let Some(row_id) = row_id else {
            return Err(CommandFailure::new(
                "TRACK_NOT_ANALYZED",
                format!(
                    "source track '{}' is absent or ignored in the Bliss database",
                    track.id
                ),
            ));
        };
        if local_candidate_rows
            .as_ref()
            .is_some_and(|rows| !rows.contains(&row_id))
        {
            return Err(CommandFailure::new(
                "SOURCE_NOT_IN_LOCAL_CANDIDATE_INVENTORY",
                format!(
                    "source track '{}' is not present in the frozen LMS-local inventory",
                    track.id
                ),
            ));
        }
    }

    Ok(ValidationSummary {
        schema_version: 1,
        program: PROGRAM,
        version: VERSION,
        job_id: request.job_id,
        valid: true,
        request_sha256,
        database_schema: SUPPORTED_SCHEMA_IDENTITY,
        database_sha256,
        learned_matrix_sha256,
        local_candidate_inventory_sha256,
        local_candidate_track_count: local_candidate_rows.as_ref().map(HashSet::len),
        semantic_evidence_sha256,
        source_track_count: request.source_tracks.len(),
    })
}

fn score_request(path: &Path) -> Result<ScoringArtifact, CommandFailure> {
    let validation = validate_request(path)?;
    let request = decode_request(path)?;
    if request.scoring.algorithm != "adaptive" {
        return Err(CommandFailure::new(
            "SCORING_ALGORITHM_UNSUPPORTED",
            format!(
                "the score command currently supports adaptive scoring, not '{}'",
                request.scoring.algorithm
            ),
        ));
    }
    let settings = request.scoring.adaptive.as_ref().ok_or_else(|| {
        CommandFailure::new(
            "ADAPTIVE_SETTINGS_REQUIRED",
            "adaptive scoring requires scoring.adaptive settings",
        )
    })?;
    let matrix_artifact = request.artifacts.learned_matrix.as_ref().ok_or_else(|| {
        CommandFailure::new(
            "MATRIX_REQUIRED",
            "adaptive scoring requires artifacts.learned_matrix",
        )
    })?;
    let learned_matrix = bliss_mixer_core::matrix::load_learned_matrix(&matrix_artifact.path)
        .map_err(|error| CommandFailure::new("MATRIX_INVALID", error.to_string()))?;
    let database = BlissDatabase::open_read_only(&request.artifacts.database.path)
        .map_err(|error| CommandFailure::new("DATABASE_INVALID", error.to_string()))?;

    let mut features = Vec::with_capacity(request.source_tracks.len());
    for track in &request.source_tracks {
        let database_file = track.database_file.as_deref().ok_or_else(|| {
            CommandFailure::new(
                "TRACK_IDENTITY_INCOMPLETE",
                format!("source track '{}' has no database_file identity", track.id),
            )
        })?;
        let row_id = database
            .usable_row_id_for_file(database_file)
            .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?
            .ok_or_else(|| {
                CommandFailure::new(
                    "TRACK_NOT_ANALYZED",
                    format!(
                        "source track '{}' is absent or ignored in the Bliss database",
                        track.id
                    ),
                )
            })?;
        let metrics = database
            .raw_metrics(row_id)
            .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?
            .ok_or_else(|| {
                CommandFailure::new(
                    "TRACK_METRICS_MISSING",
                    format!("source track '{}' has no Bliss feature vector", track.id),
                )
            })?;
        features.push(metrics);
    }

    let scored = score_adaptive_sequence(
        &features,
        Some(&learned_matrix),
        settings.learned_percent,
        settings.seed_limit,
    )
    .map_err(|error| CommandFailure::new("ADAPTIVE_SCORING_FAILED", error.to_string()))?;

    let legs: Vec<_> = scored
        .into_iter()
        .map(|leg| ContextualLeg {
            position: leg.position,
            seed_start: leg.seed_start,
            seed_track_ids: request.source_tracks[leg.seed_start..leg.position]
                .iter()
                .map(|track| track.id.clone())
                .collect(),
            candidate_track_id: request.source_tracks[leg.position].id.clone(),
            algorithm: leg.algorithm.to_string(),
            distance: f64::from(leg.distance),
        })
        .collect();
    if legs.iter().any(|leg| !leg.distance.is_finite()) {
        return Err(CommandFailure::new(
            "NON_FINITE_SCORE",
            "adaptive scoring produced a non-finite transition",
        ));
    }
    let transition_sum: f64 = legs.iter().map(|leg| leg.distance).sum();
    let worst_transition = legs.iter().map(|leg| leg.distance).fold(0.0_f64, f64::max);

    Ok(ScoringArtifact {
        schema_version: 1,
        artifact_kind: "contextual-adaptive-scoring-v1",
        program: PROGRAM,
        version: VERSION,
        core_api: "0.1",
        job_id: request.job_id,
        request_sha256: validation.request_sha256,
        database_sha256: validation.database_sha256,
        learned_matrix_sha256: validation
            .learned_matrix_sha256
            .expect("adaptive validation requires a learned matrix"),
        semantic_evidence_sha256: validation.semantic_evidence_sha256,
        algorithm_requested: request.scoring.algorithm,
        learned_percent: settings.learned_percent,
        seed_limit: settings.seed_limit,
        parallel_execution: "rayon-indexed",
        source_track_ids: request
            .source_tracks
            .iter()
            .map(|track| track.id.clone())
            .collect(),
        legs,
        transition_sum,
        worst_transition,
        objective: transition_sum + 2.0 * worst_transition,
    })
}

fn load_usable_library(database: &BlissDatabase) -> Result<Vec<LibraryTrack>, CommandFailure> {
    let tracks = database
        .all_usable_tracks()
        .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?;
    let mut library = Vec::with_capacity(tracks.len());
    for track in tracks {
        let row_id = track.row_id;
        let features = track.features;
        let metadata = track.metadata;
        let artist = metadata.artist.unwrap_or_default();
        let album = metadata.album.unwrap_or_default();
        let title = metadata.title.unwrap_or_default();
        library.push(LibraryTrack {
            row_id,
            file: metadata.file,
            artist_key: repeat_key(&artist),
            title_key: repeat_key(&title),
            route_track: route::RouteTrack {
                features,
                artist_key: repeat_key(&artist),
                album_key: repeat_key(&album),
            },
        });
    }
    Ok(library)
}

fn bridge_candidate_id(row_id: u64) -> String {
    format!("bliss-row-{row_id}")
}

fn source_semantic_identity(
    source: &SourceTrack,
    library_track: &LibraryTrack,
) -> semantic::TrackIdentity {
    let artist_name = source
        .artist
        .as_deref()
        .map(semantic::normalize_identity)
        .unwrap_or_else(|| library_track.artist_key.clone());
    let mut artist_ids = source.artist_mbids.clone();
    artist_ids.push(semantic::canonical_artist_id(&artist_name));
    artist_ids.sort();
    artist_ids.dedup();
    semantic::TrackIdentity {
        recording_id: source.id.clone(),
        recording_mbid: source.recording_mbid.clone(),
        artist_ids,
        artist_name,
    }
}

fn candidate_semantic_identity(
    library_index: usize,
    library_track: &LibraryTrack,
) -> semantic::CandidateIdentity {
    semantic::CandidateIdentity {
        candidate: library_index,
        track: semantic::TrackIdentity {
            recording_id: bridge_candidate_id(library_track.row_id),
            recording_mbid: None,
            artist_ids: vec![semantic::canonical_artist_id(&library_track.artist_key)],
            artist_name: library_track.artist_key.clone(),
        },
    }
}

fn bridge_candidate_artifact(
    evaluation: &bridge::BridgeCandidateEvaluation,
    semantics: &semantic::CandidateSemantics,
    library: &[LibraryTrack],
) -> BridgeCandidateArtifact {
    BridgeCandidateArtifact {
        candidate_id: bridge_candidate_id(library[evaluation.candidate].row_id),
        semantic_tier: semantics.tier,
        semantic_evidence: semantics.evidence.clone(),
        left_distance: evaluation.left_distance,
        right_distance: evaluation.right_distance,
        left_percentile: evaluation.left_percentile,
        right_percentile: evaluation.right_percentile,
        max_percentile: evaluation.max_percentile,
        detour_percentile: evaluation.detour_percentile,
    }
}

fn endpoint_candidate_artifact(
    selected: &preview::SelectedEndpoint,
    library: &[LibraryTrack],
) -> EndpointCandidateArtifact {
    EndpointCandidateArtifact {
        candidate_id: bridge_candidate_id(library[selected.evaluation.candidate].row_id),
        semantic_tier: selected.semantics.tier,
        semantic_evidence: selected.semantics.evidence.clone(),
        distance: selected.evaluation.distance,
        percentile: selected.evaluation.percentile,
    }
}

fn select_seed_growth(
    target_track_count: usize,
    source_library_indices: &[usize],
    selected_library_route: &[usize],
    eligible_candidates: &[usize],
    tracks: &[route::RouteTrack],
    learned_matrix: &Array2<f32>,
    route_config: &route::SearchConfig,
) -> Result<(Vec<usize>, Vec<(usize, f64)>, &'static str, f64), CommandFailure> {
    if target_track_count <= source_library_indices.len() {
        return Err(CommandFailure::new(
            "SEED_GROWTH_TARGET_INVALID",
            format!(
                "seed-growth target {target_track_count} must exceed the {} source tracks",
                source_library_indices.len()
            ),
        ));
    }
    let requested = target_track_count - source_library_indices.len();
    if requested > eligible_candidates.len() {
        return Err(CommandFailure::new(
            "SEED_GROWTH_INFEASIBLE",
            format!(
                "seed growth needs {requested} additions but only {} local analyzed candidates are eligible",
                eligible_candidates.len()
            ),
        ));
    }

    // The complete, immutable source set defines relevance. Newly selected
    // tracks never enter this context, which prevents iterative taste drift.
    let seed_features = source_library_indices
        .iter()
        .map(|index| tracks[*index].features)
        .collect::<Vec<_>>();
    let relevance = bliss_playlist_optimizer::contextual::prepare_adaptive_context(
        &seed_features,
        learned_matrix,
        route_config.learned_percent,
    )
    .map_err(|error| CommandFailure::new("SEED_GROWTH_SCORING_FAILED", error.to_string()))?;
    let mut ranked = eligible_candidates
        .par_iter()
        .map(|candidate| {
            (
                *candidate,
                relevance.distance_to(&tracks[*candidate].features),
            )
        })
        .collect::<Vec<_>>();
    ranked.par_sort_unstable_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    // Membership selection follows relevance order while applying the
    // necessary per-key capacity implied by each repeat window. Routing then
    // optimizes the complete fixed membership, allowing added tracks to make a
    // repeated-artist or repeated-album seed set feasible.
    let artist_capacity = if route_config.artist_window == 0 {
        usize::MAX
    } else {
        target_track_count.div_ceil(route_config.artist_window + 1)
    };
    let album_capacity = if route_config.album_window == 0 {
        usize::MAX
    } else {
        target_track_count.div_ceil(route_config.album_window + 1)
    };
    let mut artist_counts = HashMap::<&str, usize>::new();
    let mut album_counts = HashMap::<&str, usize>::new();
    for index in source_library_indices {
        *artist_counts.entry(&tracks[*index].artist_key).or_default() += 1;
        *album_counts.entry(&tracks[*index].album_key).or_default() += 1;
    }
    if artist_counts.values().any(|count| *count > artist_capacity)
        || album_counts.values().any(|count| *count > album_capacity)
    {
        return Err(CommandFailure::new(
            "SEED_GROWTH_INFEASIBLE",
            "the source membership exceeds the requested target's repeat-window capacity",
        ));
    }

    let mut membership = selected_library_route.to_vec();
    let mut additions = Vec::with_capacity(requested);
    for (candidate, distance) in ranked {
        let artist = tracks[candidate].artist_key.as_str();
        let album = tracks[candidate].album_key.as_str();
        if artist_counts.get(artist).copied().unwrap_or(0) >= artist_capacity
            || album_counts.get(album).copied().unwrap_or(0) >= album_capacity
        {
            continue;
        }
        membership.push(candidate);
        additions.push((candidate, distance));
        *artist_counts.entry(artist).or_default() += 1;
        *album_counts.entry(album).or_default() += 1;
        if additions.len() == requested {
            break;
        }
    }
    if additions.len() != requested {
        return Err(CommandFailure::new(
            "SEED_GROWTH_INFEASIBLE",
            format!(
                "repeat-safe membership selection found {} of {requested} required additions",
                additions.len()
            ),
        ));
    }
    let route_tracks = membership
        .iter()
        .map(|index| tracks[*index].clone())
        .collect::<Vec<_>>();
    let result = route::optimize_adaptive_route(&route_tracks, learned_matrix, route_config)
        .map_err(|error| CommandFailure::new("SEED_GROWTH_ROUTE_FAILED", error.to_string()))?;
    let selected_strategy = result.selected.strategy;
    let selected_objective = result.selected.metrics.objective;
    let final_route = result
        .selected
        .route
        .iter()
        .map(|index| membership[*index])
        .collect::<Vec<_>>();
    Ok((
        final_route,
        additions,
        selected_strategy,
        selected_objective,
    ))
}

#[cfg(test)]
fn optimize_route_request(path: &Path) -> Result<RouteArtifact, CommandFailure> {
    optimize_route_request_with_options(path, &RuntimeOptions::disabled())
}

fn optimize_route_request_with_options(
    path: &Path,
    options: &RuntimeOptions,
) -> Result<RouteArtifact, CommandFailure> {
    let overall_started = Instant::now();
    let mut timings = StageTimings::default();
    let validated = prepare_runtime_request(path, options, &mut timings)?;
    let ValidatedRequest {
        summary: validation,
        request,
        learned_matrix,
        semantic_bundle: _,
        library,
        local_candidate_rows: _,
        database_cache,
    } = validated;
    if request.scoring.algorithm != "adaptive" {
        return Err(CommandFailure::new(
            "SCORING_ALGORITHM_UNSUPPORTED",
            format!(
                "the route command currently supports adaptive scoring, not '{}'",
                request.scoring.algorithm
            ),
        ));
    }
    if request.route.ordering_policy != "optimize_order" {
        return Err(CommandFailure::new(
            "ROUTE_POLICY_UNSUPPORTED",
            format!(
                "the route command currently supports optimize_order, not '{}'",
                request.route.ordering_policy
            ),
        ));
    }
    if request.route.objective != "bottleneck_then_sum" {
        return Err(CommandFailure::new(
            "ROUTE_OBJECTIVE_UNSUPPORTED",
            format!(
                "the route command currently supports bottleneck_then_sum, not '{}'",
                request.route.objective
            ),
        ));
    }
    if request.route.start_track_id.is_some() || request.route.destination_track_id.is_some() {
        return Err(CommandFailure::new(
            "ROUTE_LOCK_UNSUPPORTED",
            "start and destination locks are not implemented in this route-search slice",
        ));
    }
    if request.route.search.time_budget_ms.is_some() {
        return Err(CommandFailure::new(
            "TIME_BUDGET_UNSUPPORTED",
            "time-budget termination is not deterministic and is not implemented yet",
        ));
    }
    if request.extension.mode != "none" {
        return Err(CommandFailure::new(
            "EXTENSION_MODE_UNSUPPORTED",
            "route search must complete before bridge extension is enabled",
        ));
    }

    let adaptive = request.scoring.adaptive.as_ref().ok_or_else(|| {
        CommandFailure::new(
            "ADAPTIVE_SETTINGS_REQUIRED",
            "adaptive scoring requires scoring.adaptive settings",
        )
    })?;
    let seed_limit = adaptive.seed_limit;
    let learned_percent = adaptive.learned_percent;
    let deterministic_seed = request.route.search.deterministic_seed;
    let restart_count = request.route.search.restart_count;
    let learned_matrix =
        learned_matrix.expect("adaptive runtime validation requires a learned matrix");
    let library = library.expect("runtime validation always provides a decoded library");
    let file_to_track = library
        .iter()
        .map(|track| (track.file.as_str(), track))
        .collect::<HashMap<_, _>>();
    let started = Instant::now();
    let mut tracks = Vec::with_capacity(request.source_tracks.len());
    for source in &request.source_tracks {
        let database_file = source.database_file.as_deref().ok_or_else(|| {
            CommandFailure::new(
                "TRACK_IDENTITY_INCOMPLETE",
                format!("source track '{}' has no database_file identity", source.id),
            )
        })?;
        let library_track = file_to_track.get(database_file).ok_or_else(|| {
            CommandFailure::new(
                "TRACK_NOT_ANALYZED",
                format!(
                    "source track '{}' is absent or ignored in the Bliss database",
                    source.id
                ),
            )
        })?;
        let artist = source
            .artist
            .clone()
            .unwrap_or_else(|| library_track.artist_key.clone());
        let album = source
            .album
            .clone()
            .unwrap_or_else(|| library_track.route_track.album_key.clone());
        tracks.push(route::RouteTrack {
            features: library_track.route_track.features,
            artist_key: repeat_key(&artist),
            album_key: repeat_key(&album),
        });
    }
    timings.record("source_track_materialization", started.elapsed());

    let config = route::SearchConfig {
        seed_limit,
        learned_percent,
        deterministic_seed,
        restart_count,
        artist_window: request.repeat_windows.artist,
        album_window: request.repeat_windows.album,
    };
    let started = Instant::now();
    let result = route::optimize_adaptive_route(&tracks, &learned_matrix, &config)
        .map_err(|error| CommandFailure::new("ROUTE_SEARCH_FAILED", error.to_string()))?;
    timings.record("route_search", started.elapsed());
    let selected_track_ids = route_track_ids(&result.selected.route, &request.source_tracks);
    let track_window_satisfied_by_unique_membership = request.repeat_windows.track == 0
        || selected_track_ids.iter().collect::<HashSet<_>>().len() == selected_track_ids.len();
    let primary = route_candidate_artifact(&result.primary, &request.source_tracks);
    let arc = route_candidate_artifact(&result.arc, &request.source_tracks);
    let violations: Vec<_> = result
        .violations
        .into_iter()
        .map(|violation| RepeatViolationArtifact {
            kind: violation.kind,
            positions: violation.positions,
        })
        .collect();

    let mut artifact = RouteArtifact {
        schema_version: 1,
        artifact_kind: "adaptive-route-v1",
        program: PROGRAM,
        version: VERSION,
        core_api: "0.1",
        job_id: request.job_id,
        request_sha256: validation.request_sha256,
        database_sha256: validation.database_sha256,
        learned_matrix_sha256: validation
            .learned_matrix_sha256
            .expect("adaptive validation requires a learned matrix"),
        semantic_evidence_sha256: validation.semantic_evidence_sha256,
        algorithm_requested: request.scoring.algorithm,
        learned_percent,
        seed_limit,
        deterministic_seed,
        restart_count,
        parallel_execution: "rayon-restarts-indexed",
        search_tasks: result.search_tasks,
        selected_strategy: result.selected.strategy,
        selected_track_ids,
        primary,
        arc,
        repeat_validation: RepeatValidationArtifact {
            valid: violations.is_empty(),
            track_window_satisfied_by_unique_membership,
            violations,
        },
        performance: None,
    };
    artifact.performance = timings.finish(options.timings, overall_started, database_cache);
    Ok(artifact)
}

fn analyze_bridge_validated(
    validation: ValidationSummary,
    request: Request,
    semantic_bundle: semantic::EvidenceBundle,
    learned_matrix: Array2<f32>,
    library: Vec<LibraryTrack>,
    local_candidate_rows: Option<HashSet<u64>>,
    timings: &mut StageTimings,
) -> Result<BridgeAnalysisArtifact, CommandFailure> {
    let adaptive = request.scoring.adaptive.as_ref().ok_or_else(|| {
        CommandFailure::new(
            "ADAPTIVE_SETTINGS_REQUIRED",
            "adaptive scoring requires scoring.adaptive settings",
        )
    })?;
    let seed_limit = adaptive.seed_limit;
    let learned_percent = adaptive.learned_percent;
    let deterministic_seed = request.route.search.deterministic_seed;
    let restart_count = request.route.search.restart_count;
    let retained_candidate_limit = request
        .extension
        .candidate_limit
        .unwrap_or(DEFAULT_RETAINED_CANDIDATES);
    let shortlist_limit = request.extension.shortlist_limit.unwrap_or(usize::MAX);
    let (max_added_tracks, trigger_percentile, requested_exact_count) =
        match request.extension.mode.as_str() {
            "automatic" => (
                Some(request.extension.max_added_tracks.ok_or_else(|| {
                    CommandFailure::new(
                        "AUTOMATIC_BRIDGE_BUDGET_REQUIRED",
                        "automatic extension requires extension.max_added_tracks",
                    )
                })?),
                Some(request.extension.trigger_percentile.ok_or_else(|| {
                    CommandFailure::new(
                        "AUTOMATIC_TRIGGER_REQUIRED",
                        "automatic extension requires extension.trigger_percentile",
                    )
                })?),
                None,
            ),
            "exact_count" => (
                None,
                None,
                Some(request.extension.additional_track_count.ok_or_else(|| {
                    CommandFailure::new(
                        "EXACT_COUNT_REQUIRED",
                        "exact_count extension requires extension.additional_track_count",
                    )
                })?),
            ),
            "seed_growth" => (None, None, None),
            _ => unreachable!("bridge mode is checked before analysis"),
        };
    let started = Instant::now();
    let mut file_to_index = HashMap::with_capacity(library.len());
    for (index, track) in library.iter().enumerate() {
        if file_to_index.insert(track.file.clone(), index).is_some() {
            return Err(CommandFailure::new(
                "DATABASE_INVALID",
                format!("duplicate usable Bliss file identity '{}'", track.file),
            ));
        }
    }

    let mut source_files = HashSet::new();
    let mut source_identities = HashSet::new();
    let mut source_library_indices = Vec::with_capacity(request.source_tracks.len());
    let mut source_route_tracks = Vec::with_capacity(request.source_tracks.len());
    let mut source_semantic_identities = Vec::with_capacity(request.source_tracks.len());
    for source in &request.source_tracks {
        let database_file = source.database_file.as_deref().ok_or_else(|| {
            CommandFailure::new(
                "TRACK_IDENTITY_INCOMPLETE",
                format!("source track '{}' has no database_file identity", source.id),
            )
        })?;
        let library_index = file_to_index.get(database_file).copied().ok_or_else(|| {
            CommandFailure::new(
                "TRACK_NOT_ANALYZED",
                format!(
                    "source track '{}' is absent or ignored in the Bliss database",
                    source.id
                ),
            )
        })?;
        let library_track = &library[library_index];
        let artist_key = source
            .artist
            .as_deref()
            .map(repeat_key)
            .unwrap_or_else(|| library_track.artist_key.clone());
        let album_key = source
            .album
            .as_deref()
            .map(repeat_key)
            .unwrap_or_else(|| library_track.route_track.album_key.clone());
        let title_key = source
            .title
            .as_deref()
            .map(repeat_key)
            .unwrap_or_else(|| library_track.title_key.clone());
        source_files.insert(library_track.file.clone());
        source_identities.insert((artist_key.clone(), title_key));
        source_library_indices.push(library_index);
        source_semantic_identities.push(source_semantic_identity(source, library_track));
        source_route_tracks.push(route::RouteTrack {
            features: library_track.route_track.features,
            artist_key,
            album_key,
        });
    }
    timings.record("source_track_materialization", started.elapsed());

    let route_config = route::SearchConfig {
        seed_limit,
        learned_percent,
        deterministic_seed,
        restart_count,
        artist_window: request.repeat_windows.artist,
        album_window: request.repeat_windows.album,
    };
    let base_route_config = if request.extension.mode == "seed_growth" {
        route::SearchConfig {
            artist_window: 0,
            album_window: 0,
            ..route_config.clone()
        }
    } else {
        route_config.clone()
    };
    let started = Instant::now();
    let (
        selected_local_route,
        mut selected_strategy,
        mut selected_route_objective,
        parallel_execution,
    ) = match request.route.ordering_policy.as_str() {
        "optimize_order" => {
            let result = route::optimize_adaptive_route(
                &source_route_tracks,
                &learned_matrix,
                &base_route_config,
            )
            .map_err(|error| CommandFailure::new("ROUTE_SEARCH_FAILED", error.to_string()))?;
            (
                result.selected.route,
                result.selected.strategy,
                result.selected.metrics.objective,
                "rayon-route-restarts-and-candidates-indexed",
            )
        }
        "preserve_order" => {
            let preserved = (0..source_route_tracks.len()).collect::<Vec<_>>();
            let violations =
                route::repeat_violations(&preserved, &source_route_tracks, &base_route_config);
            if !violations.is_empty() {
                return Err(CommandFailure::new(
                    "PRESERVED_ANCHOR_REPEAT_CONFLICT",
                    format!(
                        "the immutable source order has {} repeat-window violation(s)",
                        violations.len()
                    ),
                ));
            }
            let metrics = route::evaluate_adaptive_sequence(
                &preserved,
                &source_route_tracks,
                &learned_matrix,
                seed_limit,
                learned_percent,
            )
            .map_err(|error| {
                CommandFailure::new("PRESERVED_ROUTE_SCORING_FAILED", error.to_string())
            })?;
            (
                preserved,
                "preserve-order",
                metrics.objective,
                "rayon-candidates-indexed",
            )
        }
        _ => unreachable!("bridge route policy is checked before analysis"),
    };
    timings.record("route_search", started.elapsed());
    let selected_library_route = selected_local_route
        .iter()
        .map(|index| source_library_indices[*index])
        .collect::<Vec<_>>();
    let selected_track_ids = route_track_ids(&selected_local_route, &request.source_tracks);

    let started = Instant::now();
    let eligible_candidates = library
        .iter()
        .enumerate()
        .filter(|(_, track)| {
            local_candidate_rows
                .as_ref()
                .is_none_or(|rows| rows.contains(&track.row_id))
                && !source_files.contains(&track.file)
                && !source_identities.contains(&(track.artist_key.clone(), track.title_key.clone()))
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let semantic_candidates = eligible_candidates
        .iter()
        .map(|index| candidate_semantic_identity(*index, &library[*index]))
        .collect::<Vec<_>>();
    let bridge_tracks = library
        .iter()
        .map(|track| track.route_track.clone())
        .collect::<Vec<_>>();
    let bridge_config = bridge::BridgeConfig {
        seed_limit,
        learned_percent,
        artist_window: request.repeat_windows.artist,
        album_window: request.repeat_windows.album,
        max_leg_percentile: bridge::DEFAULT_MAX_LEG_PERCENTILE,
        max_detour_percentile: bridge::DEFAULT_MAX_DETOUR_PERCENTILE,
    };
    timings.record("candidate_preparation", started.elapsed());
    let started = Instant::now();
    let reference = bridge::build_frozen_reference(
        &selected_library_route,
        &selected_library_route,
        &bridge_tracks,
        &learned_matrix,
        &bridge_config,
    )
    .map_err(|error| CommandFailure::new("BRIDGE_SCORING_FAILED", error.to_string()))?;
    timings.record("frozen_reference", started.elapsed());

    let mut shortlist_elapsed = Duration::ZERO;
    let mut strict_scoring_elapsed = Duration::ZERO;
    let mut gaps = Vec::with_capacity(selected_library_route.len() - 1);
    let mut preview_gaps = Vec::with_capacity(selected_library_route.len() - 1);
    let mut semantic_assisted = false;
    let first_gap = if request.extension.mode == "seed_growth" {
        selected_library_route.len()
    } else {
        1
    };
    for position in first_gap..selected_library_route.len() {
        let gap = bridge::evaluate_gap(
            &selected_library_route,
            position,
            &bridge_tracks,
            &learned_matrix,
            &bridge_config,
            &reference,
        )
        .map_err(|error| CommandFailure::new("BRIDGE_SCORING_FAILED", error.to_string()))?;
        let left_source_index = selected_local_route[position - 1];
        let right_source_index = selected_local_route[position];
        let mut gap_semantics = semantic::select_gap_candidates(
            &semantic_bundle,
            &source_semantic_identities[left_source_index],
            &source_semantic_identities[right_source_index],
            &source_semantic_identities,
            &semantic_candidates,
        );
        semantic_assisted |= gap_semantics.pool != semantic::SemanticPool::BlissOnly;
        let semantic_candidate_count = gap_semantics.candidates.len();
        let shortlist_started = Instant::now();
        if gap_semantics.candidates.len() > shortlist_limit {
            let mut reserved = gap_semantics
                .candidates
                .iter()
                .filter(|candidate| candidate.tier != semantic::SemanticTier::BlissOnly)
                .collect::<Vec<_>>();
            reserved.sort_by(|left, right| {
                left.compare_priority(right)
                    .then_with(|| left.candidate.cmp(&right.candidate))
            });
            reserved.truncate(SEMANTIC_SHORTLIST_RESERVE.min(shortlist_limit));
            let mut selected = reserved
                .iter()
                .map(|candidate| candidate.candidate)
                .collect::<HashSet<_>>();
            let remaining = gap_semantics
                .candidates
                .iter()
                .map(|candidate| candidate.candidate)
                .filter(|candidate| !selected.contains(candidate))
                .collect::<Vec<_>>();
            let acoustic = bridge::shortlist_candidates(
                &selected_library_route,
                position,
                &remaining,
                shortlist_limit.saturating_sub(selected.len()),
                bridge::ShortlistScoringContext {
                    tracks: &bridge_tracks,
                    learned_matrix: &learned_matrix,
                    config: &bridge_config,
                    reference: &reference,
                },
            )
            .map_err(|error| CommandFailure::new("BRIDGE_SHORTLIST_FAILED", error.to_string()))?;
            selected.extend(acoustic);
            gap_semantics
                .candidates
                .retain(|candidate| selected.contains(&candidate.candidate));
        }
        shortlist_elapsed += shortlist_started.elapsed();
        let shortlisted_candidate_count = gap_semantics.candidates.len();
        preview_gaps.push(preview::AutomaticGap {
            original_position: position,
            left: selected_library_route[position - 1],
            right: selected_library_route[position],
            direct_distance: gap.direct_distance,
            direct_percentile: gap.direct_percentile,
            semantics: gap_semantics.clone(),
        });
        let semantics_by_candidate = gap_semantics
            .candidates
            .iter()
            .map(|candidate| (candidate.candidate, candidate))
            .collect::<HashMap<_, _>>();
        let gap_candidate_indices = gap_semantics
            .candidates
            .iter()
            .map(|candidate| candidate.candidate)
            .collect::<Vec<_>>();
        let scoring_started = Instant::now();
        let mut evaluations = bridge::rank_candidates(
            &selected_library_route,
            position,
            &gap_candidate_indices,
            &bridge_tracks,
            &learned_matrix,
            &bridge_config,
            &reference,
        )
        .map_err(|error| CommandFailure::new("BRIDGE_SCORING_FAILED", error.to_string()))?;
        strict_scoring_elapsed += scoring_started.elapsed();
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
        let accepted_candidate_count = evaluations
            .iter()
            .filter(|candidate| candidate.accepted)
            .count();
        let repeat_rejected_count = evaluations
            .iter()
            .filter(|candidate| !candidate.repeat_safe)
            .count();
        let acoustic_rejected_count =
            evaluations.len() - accepted_candidate_count - repeat_rejected_count;
        let accepted_candidates = evaluations
            .iter()
            .filter(|candidate| candidate.accepted)
            .take(retained_candidate_limit)
            .map(|candidate| {
                bridge_candidate_artifact(
                    candidate,
                    semantics_by_candidate[&candidate.candidate],
                    &library,
                )
            })
            .collect();
        gaps.push(BridgeGapArtifact {
            position,
            left_track_id: selected_track_ids[position - 1].clone(),
            right_track_id: selected_track_ids[position].clone(),
            direct_distance: gap.direct_distance,
            direct_percentile: gap.direct_percentile,
            triggering: trigger_percentile.map(|threshold| gap.direct_percentile > threshold),
            semantic_pool: gap_semantics.pool,
            semantic_candidate_count,
            semantic_excluded_count: eligible_candidates.len() - semantic_candidate_count,
            shortlisted_candidate_count: (shortlisted_candidate_count < semantic_candidate_count)
                .then_some(shortlisted_candidate_count),
            acoustic_shortlist_excluded_count: (shortlisted_candidate_count
                < semantic_candidate_count)
                .then_some(semantic_candidate_count - shortlisted_candidate_count),
            evaluated_candidate_count: evaluations.len(),
            accepted_candidate_count,
            repeat_rejected_count,
            acoustic_rejected_count,
            accepted_candidates,
        });
    }
    timings.record("gap_candidate_shortlisting", shortlist_elapsed);
    timings.record("gap_candidate_scoring", strict_scoring_elapsed);

    let original_ids_by_library = selected_local_route
        .iter()
        .zip(selected_library_route.iter())
        .map(|(source_index, library_index)| {
            (
                *library_index,
                request.source_tracks[*source_index].id.clone(),
            )
        })
        .collect::<HashMap<_, _>>();
    let sequence_artifact = |route: &[usize]| {
        route
            .iter()
            .enumerate()
            .map(|(position, library_index)| {
                if let Some(track_id) = original_ids_by_library.get(library_index) {
                    PreviewSequenceEntryArtifact {
                        position,
                        kind: "original",
                        track_id: track_id.clone(),
                    }
                } else {
                    PreviewSequenceEntryArtifact {
                        position,
                        kind: "bridge",
                        track_id: bridge_candidate_id(library[*library_index].row_id),
                    }
                }
            })
            .collect::<Vec<_>>()
    };
    let started = Instant::now();
    let selection_preview = match request.extension.mode.as_str() {
        "automatic" => {
            let max_added_tracks =
                max_added_tracks.expect("automatic request has a validated bridge budget");
            let trigger_percentile =
                trigger_percentile.expect("automatic request has a validated trigger");
            let selection = preview::select_automatic_bridges(
                &selected_library_route,
                &preview_gaps,
                &preview::AutomaticSelectionConfig {
                    max_added_tracks,
                    trigger_percentile,
                },
                &bridge_tracks,
                &learned_matrix,
                &bridge_config,
                &reference,
            )
            .map_err(|error| CommandFailure::new("BRIDGE_PREVIEW_FAILED", error.to_string()))?;
            let preview_decisions = selection
                .decisions
                .iter()
                .map(|decision| PreviewDecisionArtifact {
                    original_position: decision.original_position,
                    route_position: decision.route_position,
                    left_track_id: original_ids_by_library[&decision.left].clone(),
                    right_track_id: original_ids_by_library[&decision.right].clone(),
                    direct_distance: decision.direct_distance,
                    direct_percentile: decision.direct_percentile,
                    triggering: decision.direct_percentile > trigger_percentile,
                    semantic_pool: decision.semantic_pool,
                    reason: decision.reason,
                    selected_bridge: decision.selected.as_ref().map(|selected| {
                        bridge_candidate_artifact(
                            &selected.evaluation,
                            &selected.semantics,
                            &library,
                        )
                    }),
                })
                .collect::<Vec<_>>();
            let added_track_count = selection.final_route.len() - selected_library_route.len();
            let unique_membership = selection.final_route.iter().collect::<HashSet<_>>().len()
                == selection.final_route.len();
            SelectionPreviewArtifact::Automatic(AutomaticSelectionArtifact {
                mode: "automatic",
                processing_order: "left-to-right-original-gaps",
                max_added_tracks,
                added_track_count,
                original_subsequence_preserved: selection
                    .final_route
                    .iter()
                    .filter(|index| original_ids_by_library.contains_key(index))
                    .eq(selected_library_route.iter()),
                unique_membership,
                final_sequence: sequence_artifact(&selection.final_route),
                decisions: preview_decisions,
            })
        }
        "exact_count" => {
            let requested_added_tracks =
                requested_exact_count.expect("exact-count request has a validated count");
            let max_tracks_per_gap = request.extension.max_tracks_per_gap.unwrap_or(1);
            let opening_enabled = request.extension.allow_opening_track.unwrap_or(false);
            let closing_enabled = request.extension.allow_closing_track.unwrap_or(false);
            let endpoint_slots = preview::ExactEndpointSlots {
                opening: opening_enabled.then(|| {
                    let local_index = selected_local_route[0];
                    preview::ExactEndpointSlot {
                        anchor: selected_library_route[0],
                        semantics: semantic::select_endpoint_candidates(
                            &semantic_bundle,
                            &source_semantic_identities[local_index],
                            semantic::SourceEndpoint::Right,
                            &source_semantic_identities,
                            &semantic_candidates,
                        ),
                    }
                }),
                closing: closing_enabled.then(|| {
                    let local_index = *selected_local_route
                        .last()
                        .expect("validated routes have at least two tracks");
                    preview::ExactEndpointSlot {
                        anchor: *selected_library_route
                            .last()
                            .expect("validated routes have at least two tracks"),
                        semantics: semantic::select_endpoint_candidates(
                            &semantic_bundle,
                            &source_semantic_identities[local_index],
                            semantic::SourceEndpoint::Left,
                            &source_semantic_identities,
                            &semantic_candidates,
                        ),
                    }
                }),
            };
            semantic_assisted |= endpoint_slots
                .opening
                .iter()
                .chain(endpoint_slots.closing.iter())
                .any(|endpoint| endpoint.semantics.pool != semantic::SemanticPool::BlissOnly);
            let selection = preview::select_exact_count_bridges_with_endpoints(
                &selected_library_route,
                &preview_gaps,
                &preview::ExactSelectionConfig {
                    requested_added_tracks,
                    candidate_limit: retained_candidate_limit,
                    beam_width: EXACT_COUNT_BEAM_WIDTH,
                    max_tracks_per_gap,
                },
                &endpoint_slots,
                preview::ExactScoringContext {
                    tracks: &bridge_tracks,
                    learned_matrix: &learned_matrix,
                    config: &bridge_config,
                    reference: &reference,
                },
            )
            .map_err(|error| CommandFailure::new("BRIDGE_PREVIEW_FAILED", error.to_string()))?;
            let feasible = selection.final_route.is_some();
            let decisions = selection
                .decisions
                .iter()
                .map(|decision| ExactPreviewDecisionArtifact {
                    original_position: decision.original_position,
                    route_position: decision.route_position,
                    left_track_id: original_ids_by_library[&decision.left].clone(),
                    right_track_id: original_ids_by_library[&decision.right].clone(),
                    direct_distance: decision.direct_distance,
                    direct_percentile: decision.direct_percentile,
                    semantic_pool: decision.semantic_pool,
                    reason: decision.reason,
                    selected_bridge: decision.selected.as_ref().map(|selected| {
                        bridge_candidate_artifact(
                            &selected.evaluation,
                            &selected.semantics,
                            &library,
                        )
                    }),
                })
                .collect::<Vec<_>>();
            let endpoint_decisions = selection
                .endpoint_decisions
                .iter()
                .map(|decision| EndpointDecisionArtifact {
                    slot: match decision.slot {
                        bridge::EndpointSlot::Opening => "opening",
                        bridge::EndpointSlot::Closing => "closing",
                    },
                    anchor_track_id: original_ids_by_library[&decision.anchor].clone(),
                    semantic_pool: decision.semantic_pool,
                    reason: decision.reason,
                    selected_track: decision
                        .selected
                        .as_ref()
                        .map(|selected| endpoint_candidate_artifact(selected, &library)),
                })
                .collect::<Vec<_>>();
            let final_sequence = selection.final_route.as_deref().map(&sequence_artifact);
            let added_track_count = selection
                .final_route
                .as_ref()
                .map_or(0, |route| route.len() - selected_library_route.len());
            let original_subsequence_preserved = selection.final_route.as_ref().map(|route| {
                route
                    .iter()
                    .filter(|index| original_ids_by_library.contains_key(index))
                    .eq(selected_library_route.iter())
            });
            let unique_membership = selection
                .final_route
                .as_ref()
                .map(|route| route.iter().collect::<HashSet<_>>().len() == route.len());
            let infeasibility = (!feasible).then_some(ExactInfeasibilityArtifact {
                code: if requested_added_tracks > selection.stats.structural_upper_bound {
                    "EXACT_COUNT_INFEASIBLE"
                } else {
                    "EXACT_COUNT_NOT_FOUND_WITHIN_SEARCH_BOUNDS"
                },
                requested_added_tracks,
                maximum_additions_found: selection.stats.maximum_additions_found,
                structural_upper_bound: selection.stats.structural_upper_bound,
            });
            SelectionPreviewArtifact::Exact(ExactSelectionArtifact {
                mode: "exact_count",
                processing_order: if endpoint_slots.opening.is_some()
                    || endpoint_slots.closing.is_some()
                {
                    "bounded-endpoints-and-original-gaps-beam-search"
                } else {
                    "left-to-right-original-gaps-beam-search"
                },
                requested_added_tracks,
                feasible,
                added_track_count,
                original_subsequence_preserved,
                unique_membership,
                final_sequence,
                decisions,
                endpoint_policy: (request.extension.allow_opening_track.is_some()
                    || request.extension.allow_closing_track.is_some())
                .then_some(EndpointPolicyArtifact {
                    opening_enabled,
                    closing_enabled,
                    maximum_opening_tracks: usize::from(opening_enabled),
                    maximum_closing_tracks: usize::from(closing_enabled),
                }),
                endpoint_decisions,
                search: ExactSearchArtifact {
                    beam_width: EXACT_COUNT_BEAM_WIDTH,
                    candidate_limit: retained_candidate_limit,
                    max_tracks_per_gap: selection.stats.max_tracks_per_gap,
                    evaluated_states: selection.stats.evaluated_states,
                    retained_states: selection.stats.retained_states,
                    maximum_additions_found: selection.stats.maximum_additions_found,
                    structural_upper_bound: selection.stats.structural_upper_bound,
                },
                infeasibility,
            })
        }
        "seed_growth" => {
            let target_track_count = request.extension.target_track_count.ok_or_else(|| {
                CommandFailure::new(
                    "SEED_GROWTH_TARGET_REQUIRED",
                    "seed_growth extension requires extension.target_track_count",
                )
            })?;
            let (final_route, additions, growth_strategy, growth_objective) = select_seed_growth(
                target_track_count,
                &source_library_indices,
                &selected_library_route,
                &eligible_candidates,
                &bridge_tracks,
                &learned_matrix,
                &route_config,
            )?;
            selected_strategy = growth_strategy;
            selected_route_objective = growth_objective;
            let requested_added_tracks = target_track_count - source_library_indices.len();
            let original_subsequence_preserved = final_route
                .iter()
                .filter(|index| original_ids_by_library.contains_key(index))
                .eq(selected_library_route.iter());
            let unique_membership =
                final_route.iter().collect::<HashSet<_>>().len() == final_route.len();
            SelectionPreviewArtifact::SeedGrowth(SeedGrowthSelectionArtifact {
                mode: "seed_growth",
                processing_order: "full-seed-relevance-then-complete-membership-route",
                target_track_count,
                requested_added_tracks,
                feasible: true,
                added_track_count: additions.len(),
                original_subsequence_preserved,
                unique_membership,
                relevance_reference_track_count: source_library_indices.len(),
                final_sequence: sequence_artifact(&final_route),
                selected_additions: additions
                    .into_iter()
                    .map(
                        |(candidate, relevance_distance)| SeedGrowthAdditionArtifact {
                            candidate_id: bridge_candidate_id(library[candidate].row_id),
                            relevance_distance,
                        },
                    )
                    .collect(),
            })
        }
        _ => unreachable!("bridge mode is checked before analysis"),
    };
    timings.record("bridge_selection", started.elapsed());

    Ok(BridgeAnalysisArtifact {
        schema_version: 1,
        artifact_kind: "contextual-bridge-analysis-v1",
        program: PROGRAM,
        version: VERSION,
        core_api: "0.1",
        job_id: request.job_id,
        request_sha256: validation.request_sha256,
        database_sha256: validation.database_sha256,
        learned_matrix_sha256: validation
            .learned_matrix_sha256
            .expect("adaptive validation requires a learned matrix"),
        local_candidate_inventory_sha256: validation.local_candidate_inventory_sha256,
        semantic_evidence_sha256: validation.semantic_evidence_sha256,
        algorithm_requested: request.scoring.algorithm,
        ordering_policy: request.route.ordering_policy,
        learned_percent,
        seed_limit,
        deterministic_seed,
        restart_count,
        parallel_execution,
        selected_strategy,
        source_track_ids: request
            .source_tracks
            .iter()
            .map(|track| track.id.clone())
            .collect(),
        selected_track_ids,
        selected_route_objective,
        usable_library_track_count: library.len(),
        local_candidate_track_count: validation.local_candidate_track_count,
        non_local_candidate_excluded_count: validation
            .local_candidate_track_count
            .map(|count| library.len().saturating_sub(count)),
        eligible_candidate_count: eligible_candidates.len(),
        frozen_reference_count: reference.len(),
        trigger_percentile,
        max_leg_percentile: bridge::DEFAULT_MAX_LEG_PERCENTILE,
        max_detour_percentile: bridge::DEFAULT_MAX_DETOUR_PERCENTILE,
        retained_candidate_limit,
        semantic_mode: if semantic_assisted {
            "semantic-assisted".to_owned()
        } else if semantic_bundle.edges.is_empty() && semantic_bundle.providers.is_empty() {
            "bliss-only-empty-graph".to_owned()
        } else {
            "bliss-only-no-usable-edges".to_owned()
        },
        provider_states: semantic_bundle.providers,
        gaps,
        selection_preview,
        performance: None,
    })
}

#[cfg(test)]
fn analyze_bridge_request(path: &Path) -> Result<BridgeAnalysisArtifact, CommandFailure> {
    analyze_bridge_request_with_options(path, &RuntimeOptions::disabled())
}

fn analyze_bridge_request_with_options(
    path: &Path,
    options: &RuntimeOptions,
) -> Result<BridgeAnalysisArtifact, CommandFailure> {
    let overall_started = Instant::now();
    let mut timings = StageTimings::default();
    let validated = prepare_runtime_request(path, options, &mut timings)?;
    let ValidatedRequest {
        summary: validation,
        request,
        learned_matrix,
        semantic_bundle,
        library,
        local_candidate_rows,
        database_cache,
    } = validated;
    if request.scoring.algorithm != "adaptive" {
        return Err(CommandFailure::new(
            "SCORING_ALGORITHM_UNSUPPORTED",
            format!(
                "the bridge command currently supports adaptive scoring, not '{}'",
                request.scoring.algorithm
            ),
        ));
    }
    if !matches!(
        request.route.ordering_policy.as_str(),
        "optimize_order" | "preserve_order"
    ) {
        return Err(CommandFailure::new(
            "ROUTE_POLICY_UNSUPPORTED",
            format!(
                "the bridge command currently supports optimize_order or preserve_order, not '{}'",
                request.route.ordering_policy
            ),
        ));
    }
    if request.route.objective != "bottleneck_then_sum" {
        return Err(CommandFailure::new(
            "ROUTE_OBJECTIVE_UNSUPPORTED",
            format!(
                "the bridge command currently supports bottleneck_then_sum, not '{}'",
                request.route.objective
            ),
        ));
    }
    if request.route.start_track_id.is_some() || request.route.destination_track_id.is_some() {
        return Err(CommandFailure::new(
            "ROUTE_LOCK_UNSUPPORTED",
            "start and destination locks are not implemented in bridge analysis",
        ));
    }
    if request.route.search.time_budget_ms.is_some() {
        return Err(CommandFailure::new(
            "TIME_BUDGET_UNSUPPORTED",
            "time-budget termination is not deterministic and is not implemented yet",
        ));
    }
    if !matches!(
        request.extension.mode.as_str(),
        "automatic" | "exact_count" | "seed_growth"
    ) {
        return Err(CommandFailure::new(
            "EXTENSION_MODE_UNSUPPORTED",
            format!(
                "the bridge command currently analyzes automatic, exact_count, or seed_growth extension, not '{}'",
                request.extension.mode
            ),
        ));
    }
    if request.extension.max_tracks_per_gap.is_some() && request.extension.mode != "exact_count" {
        return Err(CommandFailure::new(
            "MAX_TRACKS_PER_GAP_UNSUPPORTED",
            "extension.max_tracks_per_gap is supported only for exact_count requests",
        ));
    }
    if (request.extension.allow_opening_track.is_some()
        || request.extension.allow_closing_track.is_some())
        && request.extension.mode != "exact_count"
    {
        return Err(CommandFailure::new(
            "ENDPOINT_SLOTS_UNSUPPORTED",
            "endpoint slots are supported only for exact_count requests",
        ));
    }
    if request.extension.max_tracks_per_gap.unwrap_or(1) > 1
        && request.route.ordering_policy != "preserve_order"
    {
        return Err(CommandFailure::new(
            "MULTI_TRACK_GAPS_REQUIRE_PRESERVE_ORDER",
            "more than one bridge per gap currently requires preserve_order",
        ));
    }
    let mut artifact = analyze_bridge_validated(
        validation,
        request,
        semantic_bundle,
        learned_matrix.expect("adaptive runtime validation requires a learned matrix"),
        library.expect("runtime validation always provides a decoded library"),
        local_candidate_rows,
        &mut timings,
    )?;
    artifact.performance = timings.finish(options.timings, overall_started, database_cache);
    Ok(artifact)
}

fn repeat_key(value: &str) -> String {
    value.trim().to_lowercase()
}

fn route_track_ids(route: &[usize], tracks: &[SourceTrack]) -> Vec<String> {
    route
        .iter()
        .map(|index| tracks[*index].id.clone())
        .collect()
}

fn route_candidate_artifact(
    candidate: &route::CandidateRoute,
    tracks: &[SourceTrack],
) -> RouteCandidateArtifact {
    RouteCandidateArtifact {
        strategy: candidate.strategy,
        track_ids: route_track_ids(&candidate.route, tracks),
        transition_sum: candidate.metrics.transition_sum,
        worst_transition: candidate.metrics.worst_transition,
        objective: candidate.metrics.objective,
        arc_error: candidate.metrics.arc_error,
    }
}
fn print_result<T: Serialize>(result: Result<T, CommandFailure>) {
    match result {
        Ok(output) => println!(
            "{}",
            serde_json::to_string(&output).expect("command output serializes")
        ),
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::to_string(&error).expect("command error serializes")
            );
            std::process::exit(1);
        }
    }
}

fn main() {
    configure_parallelism();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some((command, path, options)) = parse_request_command(&args) {
        match command {
            "validate" if !options.timings && options.cache_dir.is_none() => {
                print_result(validate_request(path));
            }
            "score" if !options.timings && options.cache_dir.is_none() => {
                print_result(score_request(path));
            }
            "route" => print_result(optimize_route_request_with_options(path, &options)),
            "bridge" => print_result(analyze_bridge_request_with_options(path, &options)),
            _ => {
                eprintln!("{}", usage());
                std::process::exit(2);
            }
        }
        return;
    }
    match args.as_slice() {
        [command] if command == "version" => println!("{PROGRAM} {VERSION}"),
        [command, format] if command == "version" && format == "--json" => {
            println!(
                "{{\"schema_version\":1,\"program\":\"{PROGRAM}\",\"version\":\"{VERSION}\",\"core_api\":\"0.1\"}}"
            );
        }
        _ => {
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_mentions_the_supported_commands() {
        assert!(usage().contains("version"));
        assert!(usage().contains("validate"));
        assert!(usage().contains("score"));
        assert!(usage().contains("route"));
        assert!(usage().contains("bridge"));
        assert!(usage().contains("--timings"));
        assert!(usage().contains("--cache-dir"));
        assert_eq!(default_parallel_workers(1), 1);
        assert_eq!(default_parallel_workers(2), 1);
        assert_eq!(default_parallel_workers(4), 3);
    }

    #[test]
    fn seed_growth_reaches_exact_target_without_relevance_drift() {
        let tracks = (0..32)
            .map(|index| route::RouteTrack {
                features: std::array::from_fn(|feature| {
                    index as f32 / 100.0 + feature as f32 / 1000.0
                }),
                artist_key: format!("artist-{index}"),
                album_key: format!("album-{index}"),
            })
            .collect::<Vec<_>>();
        let config = route::SearchConfig {
            seed_limit: 3,
            learned_percent: 20,
            deterministic_seed: 20260721,
            restart_count: 0,
            artist_window: 5,
            album_window: 10,
        };
        let candidates = (2..tracks.len()).collect::<Vec<_>>();
        let (grown, additions, _, _) = select_seed_growth(
            25,
            &[0, 1],
            &[0, 1],
            &candidates,
            &tracks,
            &Array2::eye(23),
            &config,
        )
        .unwrap();
        assert_eq!(grown.len(), 25);
        assert_eq!(additions.len(), 23);
        assert_eq!(grown.iter().collect::<HashSet<_>>().len(), 25);
        assert_eq!(
            grown
                .iter()
                .filter(|index| **index == 0 || **index == 1)
                .copied()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(route::repeat_violations(&grown, &tracks, &config).is_empty());
    }

    #[test]
    fn decoded_library_cache_is_identity_bound_and_round_trips() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();
        let mut request = decode_request(Path::new(
            "fixtures/synthetic/automatic-bridge-request.json",
        ))
        .unwrap();
        request.artifacts.database.cache_identity = Some("fixture-identity-v1".to_owned());
        let database = BlissDatabase::open_read_only(&request.artifacts.database.path).unwrap();
        let library = load_usable_library(&database).unwrap();
        let database_sha256 = hash_artifact(&request.artifacts.database, "database").unwrap();
        let cache_dir = std::env::temp_dir().join(format!(
            "bliss-playlist-optimizer-cache-test-{}",
            std::process::id()
        ));
        let cache = LibraryCache {
            format_version: LIBRARY_CACHE_VERSION,
            database_path: request.artifacts.database.path.clone(),
            database_identity: request.artifacts.database.cache_identity.clone().unwrap(),
            database_sha256: database_sha256.clone(),
            library: library.clone(),
        };
        store_library_cache(&cache_dir, &request.artifacts.database, &cache);
        let loaded = load_library_cache(&cache_dir, &request.artifacts.database).unwrap();
        assert_eq!(loaded.database_sha256, database_sha256);
        assert_eq!(loaded.library.len(), library.len());
        assert_eq!(loaded.library[0].file, library[0].file);

        let cache_path = library_cache_path(&cache_dir, &request.artifacts.database.path);
        let mut corrupted = fs::read(&cache_path).unwrap();
        let last = corrupted.last_mut().unwrap();
        *last ^= 0x01;
        fs::write(&cache_path, corrupted).unwrap();
        assert!(load_library_cache(&cache_dir, &request.artifacts.database).is_none());

        store_library_cache(&cache_dir, &request.artifacts.database, &cache);
        request.artifacts.database.cache_identity = Some("changed".to_owned());
        assert!(load_library_cache(&cache_dir, &request.artifacts.database).is_none());
        let _ = fs::remove_dir_all(cache_dir);
        std::env::set_current_dir(original).unwrap();
    }

    #[test]
    fn local_candidate_inventory_is_hash_and_database_bound() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();
        let temporary_root = std::env::temp_dir().join(format!(
            "bliss-playlist-optimizer-inventory-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&temporary_root).unwrap();
        let mut request = decode_request(Path::new(
            "fixtures/synthetic/automatic-bridge-request.json",
        ))
        .unwrap();
        request.artifacts.database.cache_identity = Some("inventory-fixture-v1".to_owned());
        let database = BlissDatabase::open_read_only(&request.artifacts.database.path).unwrap();
        let library = load_usable_library(&database).unwrap();
        let allowed = vec![library[0].row_id, library[1].row_id];
        let inventory_path = temporary_root.join("inventory.json");
        let inventory = serde_json::json!({
            "schema_version": 1,
            "schema_identity": "lms-local-candidate-inventory-v1",
            "generated_at": 1,
            "database_cache_identity": "inventory-fixture-v1",
            "lms_scan_time": 1,
            "lms_local_track_count": 2,
            "usable_bliss_row_count": library.len(),
            "allowed_row_ids": allowed,
        });
        let bytes = serde_json::to_vec(&inventory).unwrap();
        fs::write(&inventory_path, &bytes).unwrap();
        let artifact = Artifact {
            path: inventory_path.to_string_lossy().into_owned(),
            sha256: Some(format!("{:x}", Sha256::digest(&bytes))),
            schema_identity: Some("lms-local-candidate-inventory-v1".to_owned()),
            cache_identity: None,
        };

        let (rows, _) =
            load_local_candidate_inventory(&artifact, &request.artifacts.database, &library)
                .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.contains(&library[0].row_id));

        request.artifacts.database.cache_identity = Some("changed".to_owned());
        let failure =
            load_local_candidate_inventory(&artifact, &request.artifacts.database, &library)
                .unwrap_err();
        assert_eq!(failure.code, "CANDIDATE_INVENTORY_DATABASE_MISMATCH");

        let _ = fs::remove_dir_all(temporary_root);
        std::env::set_current_dir(original).unwrap();
    }

    #[test]
    fn bridge_search_excludes_every_row_outside_the_local_inventory() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();
        let temporary_root = std::env::temp_dir().join(format!(
            "bliss-playlist-optimizer-inventory-filter-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&temporary_root).unwrap();
        let mut request: Value = serde_json::from_slice(
            &fs::read("fixtures/synthetic/preserve-automatic-request.json").unwrap(),
        )
        .unwrap();
        request["artifacts"]["database"]["cache_identity"] =
            Value::String("inventory-filter-fixture-v1".to_owned());
        let database_path = request["artifacts"]["database"]["path"].as_str().unwrap();
        let database = BlissDatabase::open_read_only(database_path).unwrap();
        let library = load_usable_library(&database).unwrap();
        let allowed = request["source_tracks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|track| {
                database
                    .usable_row_id_for_file(track["database_file"].as_str().unwrap())
                    .unwrap()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let allowed_count = allowed.len();
        let inventory_path = temporary_root.join("inventory.json");
        let inventory = serde_json::json!({
            "schema_version": 1,
            "schema_identity": "lms-local-candidate-inventory-v1",
            "generated_at": 1,
            "database_cache_identity": "inventory-filter-fixture-v1",
            "lms_scan_time": 1,
            "lms_local_track_count": allowed_count,
            "usable_bliss_row_count": library.len(),
            "allowed_row_ids": allowed,
        });
        let inventory_bytes = serde_json::to_vec(&inventory).unwrap();
        fs::write(&inventory_path, &inventory_bytes).unwrap();
        request["artifacts"]["local_candidate_inventory"] = serde_json::json!({
            "path": inventory_path.to_string_lossy(),
            "sha256": format!("{:x}", Sha256::digest(&inventory_bytes)),
            "schema_identity": "lms-local-candidate-inventory-v1",
        });
        let request_path = temporary_root.join("request.json");
        fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();

        let artifact = analyze_bridge_request(&request_path).unwrap();
        assert_eq!(artifact.local_candidate_track_count, Some(allowed_count));
        assert_eq!(artifact.eligible_candidate_count, 0);
        assert_eq!(
            artifact.non_local_candidate_excluded_count,
            Some(library.len() - allowed_count)
        );
        match artifact.selection_preview {
            SelectionPreviewArtifact::Automatic(preview) => {
                assert_eq!(preview.added_track_count, 0)
            }
            SelectionPreviewArtifact::Exact(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected automatic preview")
            }
        }

        let _ = fs::remove_dir_all(temporary_root);
        std::env::set_current_dir(original).unwrap();
    }

    #[test]
    fn timed_bridge_request_reports_cold_miss_then_deterministic_warm_hit() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();
        let temporary_root = std::env::temp_dir().join(format!(
            "bliss-playlist-optimizer-runtime-test-{}",
            std::process::id()
        ));
        let cache_dir = temporary_root.join("cache");
        fs::create_dir_all(&temporary_root).unwrap();
        let mut request: Value = serde_json::from_slice(
            &fs::read("fixtures/synthetic/automatic-bridge-request.json").unwrap(),
        )
        .unwrap();
        request["artifacts"]["database"]["cache_identity"] =
            Value::String("fixture-runtime-v1".to_owned());
        let request_path = temporary_root.join("request.json");
        fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
        let options = RuntimeOptions {
            timings: true,
            cache_dir: Some(cache_dir),
        };

        let mut cold = analyze_bridge_request_with_options(&request_path, &options).unwrap();
        let mut warm = analyze_bridge_request_with_options(&request_path, &options).unwrap();
        assert_eq!(cold.performance.as_ref().unwrap().database_cache, "miss");
        assert_eq!(warm.performance.as_ref().unwrap().database_cache, "hit");
        assert!(cold
            .performance
            .as_ref()
            .unwrap()
            .stages
            .iter()
            .any(|stage| stage.stage == "library_decode"));
        assert!(!warm
            .performance
            .as_ref()
            .unwrap()
            .stages
            .iter()
            .any(|stage| stage.stage == "library_decode"));
        cold.performance = None;
        warm.performance = None;
        assert_eq!(
            serde_json::to_string(&cold).unwrap(),
            serde_json::to_string(&warm).unwrap()
        );

        let _ = fs::remove_dir_all(temporary_root);
        std::env::set_current_dir(original).unwrap();
    }

    #[test]
    fn shortlisted_bridge_preview_matches_exhaustive_fixture_selection() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();
        let temporary_root = std::env::temp_dir().join(format!(
            "bliss-playlist-optimizer-shortlist-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&temporary_root).unwrap();
        let source = Path::new("fixtures/synthetic/automatic-bridge-request.json");
        let exhaustive = analyze_bridge_request(source).unwrap();
        let mut request: Value = serde_json::from_slice(&fs::read(source).unwrap()).unwrap();
        request["extension"]["shortlist_limit"] = Value::from(5);
        let request_path = temporary_root.join("request.json");
        fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
        let shortlisted = analyze_bridge_request(&request_path).unwrap();

        let exhaustive_json = serde_json::to_value(&exhaustive).unwrap();
        let shortlisted_json = serde_json::to_value(&shortlisted).unwrap();
        assert_eq!(
            exhaustive_json["selection_preview"]["final_sequence"],
            shortlisted_json["selection_preview"]["final_sequence"]
        );
        assert_eq!(
            exhaustive_json["selection_preview"]["added_track_count"],
            shortlisted_json["selection_preview"]["added_track_count"]
        );
        assert_eq!(
            exhaustive.selected_route_objective,
            shortlisted.selected_route_objective
        );
        assert!(shortlisted.gaps.iter().all(|gap| {
            gap.shortlisted_candidate_count == Some(5)
                && gap.acoustic_shortlist_excluded_count == Some(1)
                && gap.evaluated_candidate_count == 5
        }));

        let _ = fs::remove_dir_all(temporary_root);
        std::env::set_current_dir(original).unwrap();
    }

    #[test]
    fn published_requests_validate_and_match_the_python_scoring_oracle() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();

        let validation = validate_request(Path::new("examples/reorder-only-request.json"));
        let artifact = score_request(Path::new(
            "fixtures/synthetic/adaptive-scoring-request.json",
        ));
        let (
            route_artifact,
            bridge_artifact,
            semantic_bridge_artifact,
            preview_artifact,
            exact_artifact,
            infeasible_exact_artifact,
            preserve_automatic_artifact,
            preserve_exact_artifact,
            preserve_multi_track_artifact,
        ) = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                (
                    optimize_route_request(Path::new(
                        "fixtures/synthetic/adaptive-scoring-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/automatic-bridge-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/semantic-bridge-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/automatic-preview-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/exact-count-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/exact-count-infeasible-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/preserve-automatic-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/preserve-exact-count-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/preserve-multi-track-gap-request.json",
                    )),
                )
            });
        let (
            exact_one_worker,
            infeasible_exact_one_worker,
            preserve_automatic_one_worker,
            preserve_exact_one_worker,
            preserve_multi_track_one_worker,
        ) = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                (
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/exact-count-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/exact-count-infeasible-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/preserve-automatic-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/preserve-exact-count-request.json",
                    )),
                    analyze_bridge_request(Path::new(
                        "fixtures/synthetic/preserve-multi-track-gap-request.json",
                    )),
                )
            });

        let conflict_path = Path::new("fixtures/synthetic/preserve-automatic-request.json");
        let mut conflict_timings = StageTimings::default();
        let conflict = prepare_runtime_request(
            conflict_path,
            &RuntimeOptions::disabled(),
            &mut conflict_timings,
        )
        .unwrap();
        let mut conflict_request = conflict.request;
        let first_artist = conflict_request.source_tracks[0].artist.clone();
        conflict_request.source_tracks[1].artist = first_artist;
        let preserve_repeat_conflict = analyze_bridge_validated(
            conflict.summary,
            conflict_request,
            conflict.semantic_bundle,
            conflict.learned_matrix.unwrap(),
            conflict.library.unwrap(),
            conflict.local_candidate_rows,
            &mut conflict_timings,
        )
        .unwrap_err();

        std::env::set_current_dir(original).unwrap();
        let summary = validation.unwrap();
        assert!(summary.valid);
        assert_eq!(summary.source_track_count, 2);
        assert_eq!(summary.database_schema, SUPPORTED_SCHEMA_IDENTITY);

        let artifact = artifact.unwrap();
        assert_eq!(artifact.source_track_ids.len(), 12);
        assert_eq!(artifact.legs.len(), 11);
        assert_eq!(artifact.parallel_execution, "rayon-indexed");
        let native_expected =
            include_str!("../fixtures/synthetic/expected-native-scoring-v1.json").trim();
        assert_eq!(serde_json::to_string(&artifact).unwrap(), native_expected);
        let expected: Value = serde_json::from_str(include_str!(
            "../fixtures/synthetic/expected-python-oracle-v1.json"
        ))
        .unwrap();
        let source = &expected["source_order_scoring"];
        for (actual, key) in [
            (artifact.objective, "objective"),
            (artifact.transition_sum, "transition_sum"),
            (artifact.worst_transition, "worst_transition"),
        ] {
            let expected = source[key].as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 1e-5,
                "{key}: native={actual}, python={expected}"
            );
        }
        let route_artifact = route_artifact.unwrap();
        let route_expected =
            include_str!("../fixtures/synthetic/expected-native-route-v1.json").trim();
        assert_eq!(
            serde_json::to_string(&route_artifact).unwrap(),
            route_expected
        );
        assert_eq!(route_artifact.selected_strategy, "adaptive-arc");
        assert_eq!(
            route_artifact.selected_track_ids,
            (1..=12)
                .map(|index| format!("track-{index:02}"))
                .collect::<Vec<_>>()
        );
        let python_route = &expected;
        for (actual, key) in [
            (route_artifact.arc.objective, "objective"),
            (route_artifact.arc.transition_sum, "transition_sum"),
            (route_artifact.arc.worst_transition, "worst_transition"),
        ] {
            let expected = python_route[key].as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 1e-5,
                "route {key}: native={actual}, python={expected}"
            );
        }

        let bridge_artifact = bridge_artifact.unwrap();
        let bridge_expected =
            include_str!("../fixtures/synthetic/expected-native-bridge-analysis-v1.json").trim();
        assert_eq!(
            serde_json::to_string(&bridge_artifact).unwrap(),
            bridge_expected
        );
        assert_eq!(bridge_artifact.usable_library_track_count, 18);
        assert_eq!(bridge_artifact.eligible_candidate_count, 6);
        assert_eq!(bridge_artifact.frozen_reference_count, 102);
        assert_eq!(bridge_artifact.gaps.len(), 11);
        assert!(bridge_artifact
            .gaps
            .iter()
            .all(|gap| gap.triggering == Some(false)));
        assert!(bridge_artifact
            .gaps
            .iter()
            .flat_map(|gap| &gap.accepted_candidates)
            .all(|candidate| candidate.candidate_id.starts_with("bliss-row-")));
        assert_eq!(bridge_artifact.semantic_mode, "bliss-only-empty-graph");
        assert!(bridge_artifact.provider_states.is_empty());
        assert!(bridge_artifact
            .gaps
            .iter()
            .all(|gap| gap.semantic_pool == semantic::SemanticPool::BlissOnly));

        let semantic_bridge_artifact = semantic_bridge_artifact.unwrap();
        let semantic_bridge_expected =
            include_str!("../fixtures/synthetic/expected-native-semantic-bridge-analysis-v1.json")
                .trim();
        assert_eq!(
            serde_json::to_string(&semantic_bridge_artifact).unwrap(),
            semantic_bridge_expected
        );
        assert_eq!(semantic_bridge_artifact.semantic_mode, "semantic-assisted");
        assert!(semantic_bridge_artifact
            .provider_states
            .iter()
            .any(|provider| provider.state == semantic::ProviderStatus::Failed));
        assert_eq!(
            semantic_bridge_artifact.gaps[8]
                .accepted_candidates
                .iter()
                .map(|candidate| candidate.semantic_tier)
                .collect::<Vec<_>>(),
            vec![
                semantic::SemanticTier::RecordingBoth,
                semantic::SemanticTier::ArtistLocal,
            ]
        );
        assert_eq!(
            semantic_bridge_artifact.gaps[9].accepted_candidates[0].semantic_tier,
            semantic::SemanticTier::RecordingOne
        );
        assert_eq!(
            semantic_bridge_artifact.gaps[10].semantic_pool,
            semantic::SemanticPool::CollectionFallback
        );
        assert_eq!(
            semantic_bridge_artifact.gaps[10].accepted_candidates[0].semantic_tier,
            semantic::SemanticTier::ArtistCollection
        );

        let preview_artifact = preview_artifact.unwrap();
        let preview_expected =
            include_str!("../fixtures/synthetic/expected-native-automatic-preview-v1.json").trim();
        assert_eq!(
            serde_json::to_string(&preview_artifact).unwrap(),
            preview_expected
        );
        let automatic = match &preview_artifact.selection_preview {
            SelectionPreviewArtifact::Automatic(automatic) => automatic,
            SelectionPreviewArtifact::Exact(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected automatic preview")
            }
        };
        assert_eq!(automatic.max_added_tracks, 1);
        assert_eq!(automatic.added_track_count, 1);
        assert!(automatic.original_subsequence_preserved);
        assert!(automatic.unique_membership);
        assert_eq!(
            automatic
                .final_sequence
                .iter()
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "track-01",
                "track-02",
                "bliss-row-3",
                "track-11",
                "track-12",
            ]
        );
        assert_eq!(
            automatic.decisions[1].reason,
            preview::DecisionReason::Selected
        );
        assert!(automatic.decisions[1].selected_bridge.is_some());

        let exact_artifact = exact_artifact.unwrap();
        let exact_expected =
            include_str!("../fixtures/synthetic/expected-native-exact-count-v1.json").trim();
        assert_eq!(
            serde_json::to_string(&exact_artifact).unwrap(),
            exact_expected
        );
        assert_eq!(
            serde_json::to_string(&exact_artifact).unwrap(),
            serde_json::to_string(&exact_one_worker.unwrap()).unwrap()
        );
        let exact = match &exact_artifact.selection_preview {
            SelectionPreviewArtifact::Exact(exact) => exact,
            SelectionPreviewArtifact::Automatic(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected exact-count preview")
            }
        };
        assert!(exact.feasible);
        assert_eq!(exact.requested_added_tracks, 2);
        assert_eq!(exact.added_track_count, 2);
        assert_eq!(
            exact
                .final_sequence
                .as_ref()
                .unwrap()
                .iter()
                .filter(|entry| entry.kind == "bridge")
                .count(),
            2
        );
        assert!(exact.infeasibility.is_none());

        let infeasible_exact_artifact = infeasible_exact_artifact.unwrap();
        let infeasible_expected =
            include_str!("../fixtures/synthetic/expected-native-exact-count-infeasible-v1.json")
                .trim();
        assert_eq!(
            serde_json::to_string(&infeasible_exact_artifact).unwrap(),
            infeasible_expected
        );
        assert_eq!(
            serde_json::to_string(&infeasible_exact_artifact).unwrap(),
            serde_json::to_string(&infeasible_exact_one_worker.unwrap()).unwrap()
        );
        let infeasible = match &infeasible_exact_artifact.selection_preview {
            SelectionPreviewArtifact::Exact(exact) => exact,
            SelectionPreviewArtifact::Automatic(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected exact-count preview")
            }
        };
        assert!(!infeasible.feasible);
        assert_eq!(infeasible.added_track_count, 0);
        assert!(infeasible.final_sequence.is_none());
        assert!(infeasible.decisions.is_empty());
        assert_eq!(
            infeasible
                .infeasibility
                .as_ref()
                .unwrap()
                .maximum_additions_found,
            3
        );
        assert_eq!(
            infeasible
                .infeasibility
                .as_ref()
                .unwrap()
                .structural_upper_bound,
            6
        );

        let preserve_automatic_artifact = preserve_automatic_artifact.unwrap();
        let preserve_automatic_expected =
            include_str!("../fixtures/synthetic/expected-native-preserve-automatic-v1.json").trim();
        assert_eq!(
            serde_json::to_string(&preserve_automatic_artifact).unwrap(),
            preserve_automatic_expected
        );
        assert_eq!(
            serde_json::to_string(&preserve_automatic_artifact).unwrap(),
            serde_json::to_string(&preserve_automatic_one_worker.unwrap()).unwrap()
        );
        assert_eq!(
            preserve_automatic_artifact.ordering_policy,
            "preserve_order"
        );
        assert_eq!(
            preserve_automatic_artifact.selected_strategy,
            "preserve-order"
        );
        assert_eq!(
            preserve_automatic_artifact.source_track_ids,
            preserve_automatic_artifact.selected_track_ids
        );
        let preserve_automatic = match &preserve_automatic_artifact.selection_preview {
            SelectionPreviewArtifact::Automatic(automatic) => automatic,
            SelectionPreviewArtifact::Exact(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected automatic preview")
            }
        };
        assert_eq!(preserve_automatic.added_track_count, 1);
        assert_eq!(
            preserve_automatic
                .final_sequence
                .iter()
                .filter(|entry| entry.kind == "original")
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            preserve_automatic_artifact
                .source_track_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            preserve_automatic
                .final_sequence
                .iter()
                .filter(|entry| entry.kind == "bridge")
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            vec!["bliss-row-5"]
        );

        let preserve_exact_artifact = preserve_exact_artifact.unwrap();
        let preserve_exact_expected =
            include_str!("../fixtures/synthetic/expected-native-preserve-exact-count-v1.json")
                .trim();
        assert_eq!(
            serde_json::to_string(&preserve_exact_artifact).unwrap(),
            preserve_exact_expected
        );
        assert_eq!(
            serde_json::to_string(&preserve_exact_artifact).unwrap(),
            serde_json::to_string(&preserve_exact_one_worker.unwrap()).unwrap()
        );
        assert_eq!(preserve_exact_artifact.ordering_policy, "preserve_order");
        assert_eq!(preserve_exact_artifact.selected_strategy, "preserve-order");
        assert_eq!(
            preserve_exact_artifact.source_track_ids,
            preserve_exact_artifact.selected_track_ids
        );
        let preserve_exact = match &preserve_exact_artifact.selection_preview {
            SelectionPreviewArtifact::Exact(exact) => exact,
            SelectionPreviewArtifact::Automatic(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected exact-count preview")
            }
        };
        assert!(preserve_exact.feasible);
        assert_eq!(preserve_exact.added_track_count, 2);
        let preserve_exact_sequence = preserve_exact.final_sequence.as_ref().unwrap();
        assert_eq!(
            preserve_exact_sequence
                .iter()
                .filter(|entry| entry.kind == "original")
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            preserve_exact_artifact
                .source_track_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            preserve_exact_sequence
                .iter()
                .filter(|entry| entry.kind == "bridge")
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            vec!["bliss-row-5", "bliss-row-8"]
        );
        assert_eq!(
            preserve_repeat_conflict.code,
            "PRESERVED_ANCHOR_REPEAT_CONFLICT"
        );

        let preserve_multi_track_artifact = preserve_multi_track_artifact.unwrap();
        let preserve_multi_track_expected =
            include_str!("../fixtures/synthetic/expected-native-preserve-multi-track-gap-v1.json")
                .trim();
        assert_eq!(
            serde_json::to_string(&preserve_multi_track_artifact).unwrap(),
            preserve_multi_track_expected
        );
        assert_eq!(
            serde_json::to_string(&preserve_multi_track_artifact).unwrap(),
            serde_json::to_string(&preserve_multi_track_one_worker.unwrap()).unwrap()
        );
        assert_eq!(
            preserve_multi_track_artifact.source_track_ids,
            preserve_multi_track_artifact.selected_track_ids
        );
        let preserve_multi_track = match &preserve_multi_track_artifact.selection_preview {
            SelectionPreviewArtifact::Exact(exact) => exact,
            SelectionPreviewArtifact::Automatic(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected exact-count preview")
            }
        };
        assert!(preserve_multi_track.feasible);
        assert_eq!(preserve_multi_track.requested_added_tracks, 4);
        assert_eq!(preserve_multi_track.added_track_count, 4);
        assert_eq!(preserve_multi_track.search.max_tracks_per_gap, 2);
        assert!(
            preserve_multi_track.requested_added_tracks
                > preserve_multi_track_artifact.source_track_ids.len() - 1
        );
        let preserve_multi_track_sequence = preserve_multi_track.final_sequence.as_ref().unwrap();
        assert_eq!(
            preserve_multi_track_sequence
                .iter()
                .filter(|entry| entry.kind == "original")
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            preserve_multi_track_artifact
                .source_track_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            preserve_multi_track_sequence
                .iter()
                .filter(|entry| entry.kind == "bridge")
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            vec!["bliss-row-5", "bliss-row-6", "bliss-row-8", "bliss-row-7"]
        );
        assert_eq!(
            preserve_multi_track
                .decisions
                .iter()
                .filter(|decision| decision.reason == preview::DecisionReason::Selected)
                .count(),
            4
        );
    }

    #[test]
    fn endpoint_slot_fixture_is_exact_and_worker_deterministic() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();
        let path = Path::new("fixtures/synthetic/preserve-endpoint-slots-request.json");
        let four = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| analyze_bridge_request(path));
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| analyze_bridge_request(path));
        std::env::set_current_dir(original).unwrap();

        let artifact = four.unwrap();
        let expected =
            include_str!("../fixtures/synthetic/expected-native-preserve-endpoint-slots-v1.json")
                .trim();
        assert_eq!(serde_json::to_string(&artifact).unwrap(), expected);
        assert_eq!(
            serde_json::to_string(&artifact).unwrap(),
            serde_json::to_string(&one.unwrap()).unwrap()
        );

        let exact = match &artifact.selection_preview {
            SelectionPreviewArtifact::Exact(exact) => exact,
            SelectionPreviewArtifact::Automatic(_) | SelectionPreviewArtifact::SeedGrowth(_) => {
                panic!("expected exact-count preview")
            }
        };
        assert!(exact.feasible);
        assert_eq!(exact.requested_added_tracks, 4);
        assert_eq!(exact.added_track_count, 4);
        assert_eq!(exact.search.max_tracks_per_gap, 1);
        assert_eq!(exact.search.structural_upper_bound, 5);
        assert!(exact.requested_added_tracks > artifact.source_track_ids.len() - 1);
        let policy = exact.endpoint_policy.as_ref().unwrap();
        assert!(policy.opening_enabled);
        assert!(policy.closing_enabled);
        assert_eq!(policy.maximum_opening_tracks, 1);
        assert_eq!(policy.maximum_closing_tracks, 1);
        assert_eq!(exact.endpoint_decisions.len(), 2);
        assert_eq!(exact.endpoint_decisions[0].slot, "opening");
        assert_eq!(exact.endpoint_decisions[1].slot, "closing");
        assert!(exact
            .endpoint_decisions
            .iter()
            .all(|decision| decision.reason == preview::DecisionReason::Selected));

        let sequence = exact.final_sequence.as_ref().unwrap();
        assert_eq!(
            sequence
                .iter()
                .filter(|entry| entry.kind == "original")
                .map(|entry| entry.track_id.as_str())
                .collect::<Vec<_>>(),
            artifact
                .source_track_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(sequence.first().unwrap().kind, "bridge");
        assert_eq!(sequence.last().unwrap().kind, "bridge");
        assert_eq!(
            exact
                .decisions
                .iter()
                .filter(|decision| decision.reason == preview::DecisionReason::Selected)
                .map(|decision| decision.route_position)
                .collect::<Vec<_>>(),
            vec![3, 5]
        );
    }
}
