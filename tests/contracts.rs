use serde_json::Value;
use std::path::{Path, PathBuf};

fn read_json(path: &Path) -> Value {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&contents)
        .unwrap_or_else(|error| panic!("invalid JSON in {}: {error}", path.display()))
}

fn repository_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn assert_valid(schema_path: &str, example_path: &str) {
    let schema = read_json(&repository_path(schema_path));
    let example = read_json(&repository_path(example_path));
    let validator = jsonschema::validator_for(&schema)
        .unwrap_or_else(|error| panic!("invalid schema {schema_path}: {error}"));
    let errors: Vec<String> = validator
        .iter_errors(&example)
        .map(|error| error.to_string())
        .collect();
    assert!(
        errors.is_empty(),
        "{example_path} failed {schema_path}:\n{}",
        errors.join("\n")
    );
}

#[test]
fn published_examples_satisfy_their_v1_contracts() {
    for (schema, example) in [
        (
            "schemas/optimizer-request-v1.schema.json",
            "examples/reorder-only-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/adaptive-scoring-request.json",
        ),
        (
            "schemas/semantic-evidence-v1.schema.json",
            "examples/semantic-evidence-empty.json",
        ),
        (
            "schemas/progress-event-v1.schema.json",
            "examples/progress-event.json",
        ),
        (
            "schemas/optimizer-result-v1.schema.json",
            "examples/success-result.json",
        ),
        (
            "schemas/scoring-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-scoring-v1.json",
        ),
        (
            "schemas/route-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-route-v1.json",
        ),
    ] {
        assert_valid(schema, example);
    }
}
