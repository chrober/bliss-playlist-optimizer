// SPDX-License-Identifier: GPL-3.0-only

use std::fs;
use std::path::Path;

use bliss_mixer_core::database::{BlissDatabase, SUPPORTED_SCHEMA_IDENTITY};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const PROGRAM: &str = "bliss-playlist-optimizer";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const REQUEST_SCHEMA: &str = include_str!("../schemas/optimizer-request-v1.schema.json");
const SEMANTIC_SCHEMA: &str = include_str!("../schemas/semantic-evidence-v1.schema.json");

#[derive(Debug, Deserialize)]
struct Request {
    job_id: String,
    artifacts: Artifacts,
    source_tracks: Vec<SourceTrack>,
    scoring: Scoring,
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
}

#[derive(Debug, Deserialize)]
struct Scoring {
    algorithm: String,
}

#[derive(Debug, Serialize)]
struct ValidationSummary {
    schema_version: u8,
    program: &'static str,
    version: &'static str,
    job_id: String,
    valid: bool,
    database_schema: &'static str,
    database_sha256: String,
    learned_matrix_sha256: Option<String>,
    semantic_evidence_sha256: String,
    source_track_count: usize,
}

#[derive(Debug, Serialize)]
struct ValidationFailure {
    schema_version: u8,
    valid: bool,
    code: &'static str,
    message: String,
}

impl ValidationFailure {
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
    "Usage:\n  bliss-playlist-optimizer version [--json]\n  bliss-playlist-optimizer validate --request <request.json>"
}

fn read_artifact(
    artifact: &Artifact,
    kind: &'static str,
) -> Result<(Vec<u8>, String), ValidationFailure> {
    let bytes = fs::read(&artifact.path).map_err(|error| {
        ValidationFailure::new(
            "ARTIFACT_UNREADABLE",
            format!("cannot read {kind} artifact '{}': {error}", artifact.path),
        )
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if let Some(expected) = &artifact.sha256 {
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(ValidationFailure::new(
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
) -> Result<(), ValidationFailure> {
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
        Err(ValidationFailure::new(
            "INVALID_REQUEST",
            format!("{kind} schema validation failed: {}", errors.join("; ")),
        ))
    }
}

fn parse_json(bytes: &[u8], kind: &'static str) -> Result<Value, ValidationFailure> {
    serde_json::from_slice(bytes).map_err(|error| {
        ValidationFailure::new("INVALID_JSON", format!("invalid {kind} JSON: {error}"))
    })
}

fn validate_request(path: &Path) -> Result<ValidationSummary, ValidationFailure> {
    let request_bytes = fs::read(path).map_err(|error| {
        ValidationFailure::new(
            "REQUEST_UNREADABLE",
            format!("cannot read request '{}': {error}", path.display()),
        )
    })?;
    let request_value = parse_json(&request_bytes, "request")?;
    validate_json(&request_value, REQUEST_SCHEMA, "request")?;
    let request: Request = serde_json::from_value(request_value).map_err(|error| {
        ValidationFailure::new("INVALID_REQUEST", format!("cannot decode request: {error}"))
    })?;

    if let Some(identity) = &request.artifacts.database.schema_identity {
        if identity != "TracksV2" && identity != SUPPORTED_SCHEMA_IDENTITY {
            return Err(ValidationFailure::new(
                "DATABASE_SCHEMA_MISMATCH",
                format!("unsupported database schema identity '{identity}'"),
            ));
        }
    }
    let (_, database_sha256) = read_artifact(&request.artifacts.database, "database")?;
    let database = BlissDatabase::open_read_only(&request.artifacts.database.path)
        .map_err(|error| ValidationFailure::new("DATABASE_INVALID", error.to_string()))?;
    database
        .quick_check()
        .map_err(|error| ValidationFailure::new("DATABASE_INTEGRITY_FAILED", error.to_string()))?;

    let learned_matrix_sha256 = if let Some(matrix) = &request.artifacts.learned_matrix {
        let (_, hash) = read_artifact(matrix, "learned matrix")?;
        bliss_mixer_core::matrix::load_learned_matrix(&matrix.path)
            .map_err(|error| ValidationFailure::new("MATRIX_INVALID", error.to_string()))?;
        Some(hash)
    } else {
        if request.scoring.algorithm == "learned_matrix" {
            return Err(ValidationFailure::new(
                "MATRIX_REQUIRED",
                "learned_matrix scoring requires artifacts.learned_matrix",
            ));
        }
        None
    };

    if let Some(identity) = &request.semantic_evidence.schema_identity {
        if identity != "semantic-evidence-v1" {
            return Err(ValidationFailure::new(
                "SEMANTIC_SCHEMA_MISMATCH",
                format!("unsupported semantic evidence schema identity '{identity}'"),
            ));
        }
    }
    let (semantic_bytes, semantic_evidence_sha256) =
        read_artifact(&request.semantic_evidence, "semantic evidence")?;
    let semantic_value = parse_json(&semantic_bytes, "semantic evidence")?;
    validate_json(&semantic_value, SEMANTIC_SCHEMA, "semantic evidence")?;

    for track in &request.source_tracks {
        let database_file = track.database_file.as_deref().ok_or_else(|| {
            ValidationFailure::new(
                "TRACK_IDENTITY_INCOMPLETE",
                format!("source track '{}' has no database_file identity", track.id),
            )
        })?;
        let row_id = database
            .usable_row_id_for_file(database_file)
            .map_err(|error| ValidationFailure::new("DATABASE_QUERY_FAILED", error.to_string()))?;
        if row_id.is_none() {
            return Err(ValidationFailure::new(
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
        database_schema: SUPPORTED_SCHEMA_IDENTITY,
        database_sha256,
        learned_matrix_sha256,
        semantic_evidence_sha256,
        source_track_count: request.source_tracks.len(),
    })
}

fn main() {
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
            match validate_request(Path::new(path)) {
                Ok(summary) => println!(
                    "{}",
                    serde_json::to_string(&summary).expect("summary serializes")
                ),
                Err(error) => {
                    eprintln!(
                        "{}",
                        serde_json::to_string(&error).expect("error serializes")
                    );
                    std::process::exit(1);
                }
            }
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
    }

    #[test]
    fn published_request_validates_against_real_artifacts() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(repository).unwrap();
        let result = validate_request(Path::new("examples/reorder-only-request.json"));
        std::env::set_current_dir(original).unwrap();
        let summary = result.unwrap();
        assert!(summary.valid);
        assert_eq!(summary.source_track_count, 2);
        assert_eq!(summary.database_schema, SUPPORTED_SCHEMA_IDENTITY);
    }
}
