// SPDX-License-Identifier: GPL-3.0-only

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use bliss_mixer_core::database::{BlissDatabase, SUPPORTED_SCHEMA_IDENTITY};
use bliss_mixer_core::scoring::score_adaptive_sequence;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use bliss_playlist_optimizer::{bridge, route};

const PROGRAM: &str = "bliss-playlist-optimizer";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const REQUEST_SCHEMA: &str = include_str!("../schemas/optimizer-request-v1.schema.json");
const SEMANTIC_SCHEMA: &str = include_str!("../schemas/semantic-evidence-v1.schema.json");
const BRIDGE_TRIGGER_PERCENTILE: f64 = 0.70;
const DEFAULT_RETAINED_CANDIDATES: usize = 5;

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
}

#[derive(Debug, Deserialize)]
struct Artifact {
    path: String,
    sha256: Option<String>,
    schema_identity: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SourceTrack {
    id: String,
    database_file: Option<String>,
    title: Option<String>,
    artist: Option<String>,
    album: Option<String>,
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
    candidate_limit: Option<usize>,
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
    semantic_evidence_sha256: String,
    algorithm_requested: String,
    learned_percent: u16,
    seed_limit: usize,
    deterministic_seed: u64,
    restart_count: usize,
    parallel_execution: &'static str,
    selected_strategy: &'static str,
    selected_track_ids: Vec<String>,
    selected_route_objective: f64,
    usable_library_track_count: usize,
    eligible_candidate_count: usize,
    frozen_reference_count: usize,
    trigger_percentile: f64,
    max_leg_percentile: f64,
    max_detour_percentile: f64,
    retained_candidate_limit: usize,
    semantic_mode: &'static str,
    gaps: Vec<BridgeGapArtifact>,
}

#[derive(Debug, Serialize)]
struct BridgeGapArtifact {
    position: usize,
    left_track_id: String,
    right_track_id: String,
    direct_distance: f64,
    direct_percentile: f64,
    triggering: bool,
    evaluated_candidate_count: usize,
    accepted_candidate_count: usize,
    repeat_rejected_count: usize,
    acoustic_rejected_count: usize,
    accepted_candidates: Vec<BridgeCandidateArtifact>,
}

#[derive(Debug, Serialize)]
struct BridgeCandidateArtifact {
    candidate_id: String,
    left_distance: f64,
    right_distance: f64,
    left_percentile: f64,
    right_percentile: f64,
    max_percentile: f64,
    detour_percentile: f64,
}

struct LibraryTrack {
    row_id: u64,
    file: String,
    artist_key: String,
    title_key: String,
    route_track: route::RouteTrack,
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
    "Usage:\n  bliss-playlist-optimizer version [--json]\n  bliss-playlist-optimizer validate --request <request.json>\n  bliss-playlist-optimizer score --request <request.json>\n  bliss-playlist-optimizer route --request <request.json>\n  bliss-playlist-optimizer bridge --request <request.json>"
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
        if row_id.is_none() {
            return Err(CommandFailure::new(
                "TRACK_NOT_ANALYZED",
                format!(
                    "source track '{}' is absent or ignored in the Bliss database",
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
    let metrics = database
        .all_raw_metrics()
        .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?;
    let mut library = Vec::with_capacity(metrics.len());
    for (row_id, features) in metrics {
        let metadata = database
            .metadata(row_id)
            .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?
            .ok_or_else(|| {
                CommandFailure::new(
                    "TRACK_METADATA_MISSING",
                    format!("usable Bliss row {row_id} has no metadata"),
                )
            })?;
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

fn require_empty_semantic_graph(artifact: &Artifact) -> Result<(), CommandFailure> {
    let bytes = fs::read(&artifact.path).map_err(|error| {
        CommandFailure::new(
            "SEMANTIC_EVIDENCE_UNREADABLE",
            format!("failed to read semantic evidence: {error}"),
        )
    })?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        CommandFailure::new(
            "SEMANTIC_EVIDENCE_INVALID",
            format!("failed to decode semantic evidence: {error}"),
        )
    })?;
    if value
        .get("edges")
        .and_then(Value::as_array)
        .is_some_and(|edges| !edges.is_empty())
    {
        return Err(CommandFailure::new(
            "SEMANTIC_EVIDENCE_UNSUPPORTED",
            "this bridge-analysis slice supports Bliss-only ranking; non-empty semantic edges are not ignored",
        ));
    }
    Ok(())
}

fn bridge_candidate_id(row_id: u64) -> String {
    format!("bliss-row-{row_id}")
}

fn optimize_route_request(path: &Path) -> Result<RouteArtifact, CommandFailure> {
    let validation = validate_request(path)?;
    let request = decode_request(path)?;
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

    let mut tracks = Vec::with_capacity(request.source_tracks.len());
    for source in &request.source_tracks {
        let database_file = source.database_file.as_deref().ok_or_else(|| {
            CommandFailure::new(
                "TRACK_IDENTITY_INCOMPLETE",
                format!("source track '{}' has no database_file identity", source.id),
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
                        source.id
                    ),
                )
            })?;
        let features = database
            .raw_metrics(row_id)
            .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?
            .ok_or_else(|| {
                CommandFailure::new(
                    "TRACK_METRICS_MISSING",
                    format!("source track '{}' has no Bliss feature vector", source.id),
                )
            })?;
        let metadata = database
            .metadata(row_id)
            .map_err(|error| CommandFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?
            .ok_or_else(|| {
                CommandFailure::new(
                    "TRACK_METADATA_MISSING",
                    format!("source track '{}' has no Bliss metadata", source.id),
                )
            })?;
        let artist = source
            .artist
            .clone()
            .or(metadata.artist)
            .unwrap_or_default();
        let album = source.album.clone().or(metadata.album).unwrap_or_default();
        tracks.push(route::RouteTrack {
            features,
            artist_key: repeat_key(&artist),
            album_key: repeat_key(&album),
        });
    }

    let config = route::SearchConfig {
        seed_limit,
        learned_percent,
        deterministic_seed,
        restart_count,
        artist_window: request.repeat_windows.artist,
        album_window: request.repeat_windows.album,
    };
    let result = route::optimize_adaptive_route(&tracks, &learned_matrix, &config)
        .map_err(|error| CommandFailure::new("ROUTE_SEARCH_FAILED", error.to_string()))?;
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

    Ok(RouteArtifact {
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
    })
}

fn analyze_bridge_validated(
    validation: ValidationSummary,
    request: Request,
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
    let library = load_usable_library(&database)?;
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
        source_route_tracks.push(route::RouteTrack {
            features: library_track.route_track.features,
            artist_key,
            album_key,
        });
    }

    let route_config = route::SearchConfig {
        seed_limit,
        learned_percent,
        deterministic_seed,
        restart_count,
        artist_window: request.repeat_windows.artist,
        album_window: request.repeat_windows.album,
    };
    let route_result =
        route::optimize_adaptive_route(&source_route_tracks, &learned_matrix, &route_config)
            .map_err(|error| CommandFailure::new("ROUTE_SEARCH_FAILED", error.to_string()))?;
    let selected_local_route = route_result.selected.route.clone();
    let selected_library_route = selected_local_route
        .iter()
        .map(|index| source_library_indices[*index])
        .collect::<Vec<_>>();
    let selected_track_ids = route_track_ids(&selected_local_route, &request.source_tracks);

    let eligible_candidates = library
        .iter()
        .enumerate()
        .filter(|(_, track)| {
            !source_files.contains(&track.file)
                && !source_identities.contains(&(track.artist_key.clone(), track.title_key.clone()))
        })
        .map(|(index, _)| index)
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
    let reference = bridge::build_frozen_reference(
        &selected_library_route,
        &selected_library_route,
        &bridge_tracks,
        &learned_matrix,
        &bridge_config,
    )
    .map_err(|error| CommandFailure::new("BRIDGE_SCORING_FAILED", error.to_string()))?;

    let mut gaps = Vec::with_capacity(selected_library_route.len() - 1);
    for position in 1..selected_library_route.len() {
        let gap = bridge::evaluate_gap(
            &selected_library_route,
            position,
            &bridge_tracks,
            &learned_matrix,
            &bridge_config,
            &reference,
        )
        .map_err(|error| CommandFailure::new("BRIDGE_SCORING_FAILED", error.to_string()))?;
        let evaluations = bridge::rank_candidates(
            &selected_library_route,
            position,
            &eligible_candidates,
            &bridge_tracks,
            &learned_matrix,
            &bridge_config,
            &reference,
        )
        .map_err(|error| CommandFailure::new("BRIDGE_SCORING_FAILED", error.to_string()))?;
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
            .map(|candidate| BridgeCandidateArtifact {
                candidate_id: bridge_candidate_id(library[candidate.candidate].row_id),
                left_distance: candidate.left_distance,
                right_distance: candidate.right_distance,
                left_percentile: candidate.left_percentile,
                right_percentile: candidate.right_percentile,
                max_percentile: candidate.max_percentile,
                detour_percentile: candidate.detour_percentile,
            })
            .collect();
        gaps.push(BridgeGapArtifact {
            position,
            left_track_id: selected_track_ids[position - 1].clone(),
            right_track_id: selected_track_ids[position].clone(),
            direct_distance: gap.direct_distance,
            direct_percentile: gap.direct_percentile,
            triggering: gap.direct_percentile > BRIDGE_TRIGGER_PERCENTILE,
            evaluated_candidate_count: evaluations.len(),
            accepted_candidate_count,
            repeat_rejected_count,
            acoustic_rejected_count,
            accepted_candidates,
        });
    }

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
        semantic_evidence_sha256: validation.semantic_evidence_sha256,
        algorithm_requested: request.scoring.algorithm,
        learned_percent,
        seed_limit,
        deterministic_seed,
        restart_count,
        parallel_execution: "rayon-route-restarts-and-candidates-indexed",
        selected_strategy: route_result.selected.strategy,
        selected_track_ids,
        selected_route_objective: route_result.selected.metrics.objective,
        usable_library_track_count: library.len(),
        eligible_candidate_count: eligible_candidates.len(),
        frozen_reference_count: reference.len(),
        trigger_percentile: BRIDGE_TRIGGER_PERCENTILE,
        max_leg_percentile: bridge::DEFAULT_MAX_LEG_PERCENTILE,
        max_detour_percentile: bridge::DEFAULT_MAX_DETOUR_PERCENTILE,
        retained_candidate_limit,
        semantic_mode: "bliss-only-empty-graph",
        gaps,
    })
}

fn analyze_bridge_request(path: &Path) -> Result<BridgeAnalysisArtifact, CommandFailure> {
    let validation = validate_request(path)?;
    let request = decode_request(path)?;
    if request.scoring.algorithm != "adaptive" {
        return Err(CommandFailure::new(
            "SCORING_ALGORITHM_UNSUPPORTED",
            format!(
                "the bridge command currently supports adaptive scoring, not '{}'",
                request.scoring.algorithm
            ),
        ));
    }
    if request.route.ordering_policy != "optimize_order" {
        return Err(CommandFailure::new(
            "ROUTE_POLICY_UNSUPPORTED",
            format!(
                "the bridge command currently supports optimize_order, not '{}'",
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
    if request.extension.mode != "automatic" {
        return Err(CommandFailure::new(
            "EXTENSION_MODE_UNSUPPORTED",
            format!(
                "the bridge command currently analyzes automatic extension, not '{}'",
                request.extension.mode
            ),
        ));
    }
    require_empty_semantic_graph(&request.semantic_evidence)?;
    analyze_bridge_validated(validation, request)
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
    match args.as_slice() {
        [command] if command == "version" => println!("{PROGRAM} {VERSION}"),
        [command, format] if command == "version" && format == "--json" => {
            println!(
                "{{\"schema_version\":1,\"program\":\"{PROGRAM}\",\"version\":\"{VERSION}\",\"core_api\":\"0.1\"}}"
            );
        }
        [command, request_option, path]
            if command == "validate" && request_option == "--request" =>
        {
            print_result(validate_request(Path::new(path)));
        }
        [command, request_option, path] if command == "score" && request_option == "--request" => {
            print_result(score_request(Path::new(path)));
        }
        [command, request_option, path] if command == "route" && request_option == "--request" => {
            print_result(optimize_route_request(Path::new(path)));
        }
        [command, request_option, path] if command == "bridge" && request_option == "--request" => {
            print_result(analyze_bridge_request(Path::new(path)));
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
        assert_eq!(default_parallel_workers(1), 1);
        assert_eq!(default_parallel_workers(2), 1);
        assert_eq!(default_parallel_workers(4), 3);
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
        let (route_artifact, bridge_artifact) = rayon::ThreadPoolBuilder::new()
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
                )
            });

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
        assert!(bridge_artifact.gaps.iter().all(|gap| !gap.triggering));
        assert!(bridge_artifact
            .gaps
            .iter()
            .flat_map(|gap| &gap.accepted_candidates)
            .all(|candidate| candidate.candidate_id.starts_with("bliss-row-")));
    }
}
