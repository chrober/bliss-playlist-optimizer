# Synthetic parity fixture

This fixture contains no real library metadata, filesystem paths, or analysis
results. Regenerate it with `python generate_fixture.py`.

`manifest.json` records artifact hashes and a complete Python-oracle command.
The playlist uses Lyrion's three-line extended-M3U entry form: `#EXTURL`,
`#EXTINF`, and the absolute path.

Generated oracle output is ignored; the source fixture and reviewed expected
snapshots are the versioned parity inputs. `generate_fixture.py` also writes
`adaptive-scoring-request.json`.

From the repository root, run:

```text
cargo run -- score --request fixtures/synthetic/adaptive-scoring-request.json
cargo run -- route --request fixtures/synthetic/adaptive-scoring-request.json
cargo run -- bridge --request fixtures/synthetic/automatic-bridge-request.json
```

The score result must exactly match `expected-native-scoring-v1.json`; its
aggregate metrics must remain within `1e-5` of `source_order_scoring` in the
Python snapshot.

The route result must exactly match `expected-native-route-v1.json`. That
snapshot was generated with one Rayon worker, while the Rust parity test uses
four; exact equality therefore covers worker-count determinism. The selected
order and aggregate metrics must also agree with the Python oracle within
`1e-5`.

After running the command recorded in `manifest.json`, execute
`python verify_python_oracle.py`. The verifier compares only portable,
privacy-safe values and allows no repeat-window violations.

The bridge result must exactly match
expected-native-bridge-analysis-v1.json and satisfy its versioned schema. The
fixture proves 18 usable library rows, six eligible non-curated candidates, 102
frozen contextual reference scores, 11 internal gaps, opaque candidate IDs,
and no automatic trigger for the already-smooth selected route. Manual runs
with one and four Rayon workers must be byte-identical.

Regeneration writes both the adaptive scoring request and the automatic bridge
request; their hashes are recorded in manifest.json.
