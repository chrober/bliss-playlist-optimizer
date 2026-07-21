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

fn assert_no_empty_object_keys(value: &Value, path: &str) {
    match value {
        Value::Object(object) => {
            assert!(!object.contains_key(""), "empty JSON key at {path}");
            for (key, child) in object {
                assert_no_empty_object_keys(child, &format!("{path}/{key}"));
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                assert_no_empty_object_keys(child, &format!("{path}/{index}"));
            }
        }
        _ => {}
    }
}

fn assert_valid(schema_path: &str, example_path: &str) {
    let schema = read_json(&repository_path(schema_path));
    assert!(
        schema.get("$schema").is_some(),
        "{schema_path} has no $schema keyword"
    );
    assert!(
        schema.get("$id").is_some(),
        "{schema_path} has no $id keyword"
    );
    assert_no_empty_object_keys(&schema, schema_path);
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
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/automatic-bridge-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/semantic-bridge-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/automatic-preview-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/exact-count-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/exact-count-infeasible-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/preserve-automatic-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/preserve-exact-count-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/preserve-multi-track-gap-request.json",
        ),
        (
            "schemas/optimizer-request-v1.schema.json",
            "fixtures/synthetic/preserve-endpoint-slots-request.json",
        ),
        (
            "schemas/semantic-evidence-v1.schema.json",
            "examples/semantic-evidence-empty.json",
        ),
        (
            "schemas/semantic-evidence-v1.schema.json",
            "fixtures/synthetic/semantic-evidence-mixed.json",
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
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-bridge-analysis-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-semantic-bridge-analysis-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-automatic-preview-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-exact-count-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-exact-count-infeasible-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-preserve-automatic-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-preserve-exact-count-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-preserve-multi-track-gap-v1.json",
        ),
        (
            "schemas/bridge-analysis-artifact-v1.schema.json",
            "fixtures/synthetic/expected-native-preserve-endpoint-slots-v1.json",
        ),
    ] {
        assert_valid(schema, example);
    }
}

#[test]
fn semantic_contract_rejects_cross_kind_and_recording_collection_edges() {
    let schema = read_json(&repository_path("schemas/semantic-evidence-v1.schema.json"));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let example = read_json(&repository_path(
        "fixtures/synthetic/semantic-evidence-mixed.json",
    ));

    let mut cross_kind = example.clone();
    cross_kind["edges"][0]["candidate"]["kind"] = Value::String("artist".to_owned());
    assert!(!validator.is_valid(&cross_kind));

    let mut recording_collection = example;
    recording_collection["edges"][0]["scope"] = Value::String("collection_fallback".to_owned());
    assert!(!validator.is_valid(&recording_collection));
}

#[test]
fn automatic_preview_contract_binds_reasons_to_selected_bridge_payloads() {
    let schema = read_json(&repository_path(
        "schemas/bridge-analysis-artifact-v1.schema.json",
    ));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let example = read_json(&repository_path(
        "fixtures/synthetic/expected-native-automatic-preview-v1.json",
    ));

    let mut selected_without_bridge = example.clone();
    selected_without_bridge["selection_preview"]["decisions"][1]["selected_bridge"] = Value::Null;
    assert!(!validator.is_valid(&selected_without_bridge));

    let mut skipped_with_bridge = example;
    let bridge =
        skipped_with_bridge["selection_preview"]["decisions"][1]["selected_bridge"].clone();
    skipped_with_bridge["selection_preview"]["decisions"][0]["selected_bridge"] = bridge;
    assert!(!validator.is_valid(&skipped_with_bridge));
}

#[test]
fn automatic_request_requires_a_budget_and_trigger() {
    let schema = read_json(&repository_path("schemas/optimizer-request-v1.schema.json"));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let example = read_json(&repository_path(
        "fixtures/synthetic/automatic-preview-request.json",
    ));

    let mut without_budget = example.clone();
    without_budget["extension"]
        .as_object_mut()
        .unwrap()
        .remove("max_added_tracks");
    assert!(!validator.is_valid(&without_budget));

    let mut without_trigger = example;
    without_trigger["extension"]
        .as_object_mut()
        .unwrap()
        .remove("trigger_percentile");
    assert!(!validator.is_valid(&without_trigger));
}

#[test]
fn exact_count_request_requires_the_requested_addition_count() {
    let schema = read_json(&repository_path("schemas/optimizer-request-v1.schema.json"));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let mut example = read_json(&repository_path(
        "fixtures/synthetic/exact-count-request.json",
    ));
    example["extension"]
        .as_object_mut()
        .unwrap()
        .remove("additional_track_count");
    assert!(!validator.is_valid(&example));
}

#[test]
fn multi_track_gap_bound_requires_exact_count_and_preserve_order() {
    let schema = read_json(&repository_path("schemas/optimizer-request-v1.schema.json"));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let example = read_json(&repository_path(
        "fixtures/synthetic/preserve-multi-track-gap-request.json",
    ));
    assert!(validator.is_valid(&example));

    let mut zero_bound = example.clone();
    zero_bound["extension"]["max_tracks_per_gap"] = Value::from(0);
    assert!(!validator.is_valid(&zero_bound));

    let mut excessive_bound = example.clone();
    excessive_bound["extension"]["max_tracks_per_gap"] = Value::from(9);
    assert!(!validator.is_valid(&excessive_bound));

    let mut automatic = example.clone();
    automatic["extension"]["mode"] = Value::String("automatic".to_owned());
    assert!(!validator.is_valid(&automatic));

    let mut optimized = example;
    optimized["route"]["ordering_policy"] = Value::String("optimize_order".to_owned());
    assert!(!validator.is_valid(&optimized));
}

#[test]
fn endpoint_slots_require_exact_count_mode() {
    let schema = read_json(&repository_path("schemas/optimizer-request-v1.schema.json"));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let example = read_json(&repository_path(
        "fixtures/synthetic/preserve-endpoint-slots-request.json",
    ));
    assert!(validator.is_valid(&example));

    let mut automatic = example;
    automatic["extension"]["mode"] = Value::String("automatic".to_owned());
    automatic["extension"]["max_added_tracks"] = Value::from(4);
    automatic["extension"]["trigger_percentile"] = Value::from(0.7);
    automatic["extension"]
        .as_object_mut()
        .unwrap()
        .remove("additional_track_count");
    assert!(!validator.is_valid(&automatic));
}

#[test]
fn endpoint_decision_contract_binds_reason_to_selected_track() {
    let schema = read_json(&repository_path(
        "schemas/bridge-analysis-artifact-v1.schema.json",
    ));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let example = read_json(&repository_path(
        "fixtures/synthetic/expected-native-preserve-endpoint-slots-v1.json",
    ));

    let mut selected_without_track = example.clone();
    selected_without_track["selection_preview"]["endpoint_decisions"][0]["selected_track"] =
        Value::Null;
    assert!(!validator.is_valid(&selected_without_track));

    let mut skipped_with_track = example;
    skipped_with_track["selection_preview"]["endpoint_decisions"][0]["reason"] =
        Value::String("not_selected".to_owned());
    assert!(!validator.is_valid(&skipped_with_track));
}

#[test]
fn exact_count_contract_never_represents_infeasibility_as_a_partial_route() {
    let schema = read_json(&repository_path(
        "schemas/bridge-analysis-artifact-v1.schema.json",
    ));
    let validator = jsonschema::validator_for(&schema).unwrap();
    let feasible = read_json(&repository_path(
        "fixtures/synthetic/expected-native-exact-count-v1.json",
    ));
    let infeasible = read_json(&repository_path(
        "fixtures/synthetic/expected-native-exact-count-infeasible-v1.json",
    ));

    let requested = feasible["selection_preview"]["requested_added_tracks"]
        .as_u64()
        .unwrap();
    let bridge_count = feasible["selection_preview"]["final_sequence"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["kind"] == "bridge")
        .count() as u64;
    assert_eq!(bridge_count, requested);
    assert_eq!(
        feasible["selection_preview"]["added_track_count"]
            .as_u64()
            .unwrap(),
        requested
    );

    assert_eq!(infeasible["selection_preview"]["added_track_count"], 0);
    assert!(infeasible["selection_preview"]["final_sequence"].is_null());
    assert!(infeasible["selection_preview"]["decisions"]
        .as_array()
        .unwrap()
        .is_empty());

    let mut infeasible_with_partial = infeasible.clone();
    infeasible_with_partial["selection_preview"]["final_sequence"] =
        feasible["selection_preview"]["final_sequence"].clone();
    assert!(!validator.is_valid(&infeasible_with_partial));

    let mut feasible_without_route = feasible;
    feasible_without_route["selection_preview"]["final_sequence"] = Value::Null;
    assert!(!validator.is_valid(&feasible_without_route));
}

#[test]
fn preserve_order_artifacts_keep_source_ids_as_the_original_subsequence() {
    for path in [
        "fixtures/synthetic/expected-native-preserve-automatic-v1.json",
        "fixtures/synthetic/expected-native-preserve-exact-count-v1.json",
        "fixtures/synthetic/expected-native-preserve-multi-track-gap-v1.json",
        "fixtures/synthetic/expected-native-preserve-endpoint-slots-v1.json",
    ] {
        let artifact = read_json(&repository_path(path));
        assert_eq!(artifact["ordering_policy"], "preserve_order");
        assert_eq!(artifact["selected_strategy"], "preserve-order");
        assert_eq!(artifact["source_track_ids"], artifact["selected_track_ids"]);

        let original_ids = artifact["selection_preview"]["final_sequence"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|entry| entry["kind"] == "original")
            .map(|entry| entry["track_id"].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            original_ids,
            artifact["source_track_ids"].as_array().unwrap().clone()
        );
    }
}

#[test]
fn multi_track_gap_artifact_exceeds_the_internal_gap_count_without_partial_output() {
    let artifact = read_json(&repository_path(
        "fixtures/synthetic/expected-native-preserve-multi-track-gap-v1.json",
    ));
    let preview = &artifact["selection_preview"];
    let source_count = artifact["source_track_ids"].as_array().unwrap().len();
    let requested = preview["requested_added_tracks"].as_u64().unwrap() as usize;
    assert!(requested > source_count - 1);
    assert_eq!(
        preview["added_track_count"].as_u64().unwrap() as usize,
        requested
    );
    assert_eq!(preview["search"]["max_tracks_per_gap"], 2);

    let mut selected_per_gap = std::collections::HashMap::<u64, usize>::new();
    for decision in preview["decisions"].as_array().unwrap() {
        if decision["reason"] == "selected" {
            *selected_per_gap
                .entry(decision["original_position"].as_u64().unwrap())
                .or_default() += 1;
        }
    }
    assert!(selected_per_gap.values().any(|count| *count > 1));
}

#[test]
fn endpoint_slot_artifact_exceeds_internal_capacity_only_by_explicit_opt_in() {
    let artifact = read_json(&repository_path(
        "fixtures/synthetic/expected-native-preserve-endpoint-slots-v1.json",
    ));
    let preview = &artifact["selection_preview"];
    let source_count = artifact["source_track_ids"].as_array().unwrap().len();
    let requested = preview["requested_added_tracks"].as_u64().unwrap() as usize;
    let per_gap = preview["search"]["max_tracks_per_gap"].as_u64().unwrap() as usize;
    let internal_capacity = (source_count - 1) * per_gap;

    assert!(requested > internal_capacity);
    assert_eq!(
        preview["added_track_count"].as_u64().unwrap() as usize,
        requested
    );
    assert_eq!(preview["endpoint_policy"]["maximum_opening_tracks"], 1);
    assert_eq!(preview["endpoint_policy"]["maximum_closing_tracks"], 1);
    assert_eq!(preview["endpoint_decisions"].as_array().unwrap().len(), 2);
    assert!(preview["endpoint_decisions"]
        .as_array()
        .unwrap()
        .iter()
        .all(|decision| decision["reason"] == "selected"));
    let sequence = preview["final_sequence"].as_array().unwrap();
    assert_eq!(sequence.first().unwrap()["kind"], "bridge");
    assert_eq!(sequence.last().unwrap()["kind"], "bridge");
}
