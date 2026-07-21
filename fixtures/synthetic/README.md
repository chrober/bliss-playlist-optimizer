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
cargo run -- bridge --request fixtures/synthetic/semantic-bridge-request.json
cargo run -- bridge --request fixtures/synthetic/automatic-preview-request.json
cargo run -- bridge --request fixtures/synthetic/exact-count-request.json
cargo run -- bridge --request fixtures/synthetic/exact-count-infeasible-request.json
cargo run -- bridge --request fixtures/synthetic/preserve-automatic-request.json
cargo run -- bridge --request fixtures/synthetic/preserve-exact-count-request.json
cargo run -- bridge --request fixtures/synthetic/preserve-multi-track-gap-request.json
cargo run -- bridge --request fixtures/synthetic/preserve-endpoint-slots-request.json
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

The semantic bridge result must exactly match
expected-native-semantic-bridge-analysis-v1.json. Its frozen graph records a
failed LastMix/Last.fm provider alongside partial cached ListenBrainz evidence.
It covers recording support from both endpoints, recording support from one,
endpoint-local artist support, collection-artist fallback only where the local
pool is empty, and deterministic Bliss acoustic/repeat gates beneath every
semantic tier. No provider call is made by the native optimizer.

The automatic preview result must exactly match
expected-native-automatic-preview-v1.json. Its four original anchors create a
middle gap at percentile 0.40 under a declared 0.30 trigger. With a one-track
budget, the preview inserts opaque candidate `bliss-row-3` between track 02 and
track 11, preserves the original subsequence, proves unique membership, and
reports below-threshold no-ops for the other gaps. It never writes a playlist.

The feasible exact-count result must exactly match
expected-native-exact-count-v1.json. It requests two additions and returns two
unique opaque bridges while preserving all four originals as an ordered
subsequence. The bounded search retains separate beams per addition count and
the one-worker and four-worker artifacts must be byte-identical.

The infeasible exact-count result must exactly match
expected-native-exact-count-infeasible-v1.json. It requests seven additions
from a six-candidate library under the full acoustic and repeat gates; the
search finds a maximum of three. The artifact therefore exposes no final
sequence or partial decisions. Its structural upper bound is six, so the
seven-track request is proven impossible and reports
`EXACT_COUNT_INFEASIBLE`.

The preserve-order automatic and exact-count results must exactly match
`expected-native-preserve-automatic-v1.json` and
`expected-native-preserve-exact-count-v1.json`. Their deliberately unsorted
four-track source order is retained as immutable anchors. The automatic result
adds `bliss-row-5`; the exact-count result adds `bliss-row-5` and
`bliss-row-8`. Both artifacts prove that `source_track_ids`,
`selected_track_ids`, and the final sequence filtered to original entries are
identical, and their one-worker and four-worker serializations are byte-equal.

The multi-track preserved-gap result must exactly match
`expected-native-preserve-multi-track-gap-v1.json`. It requests four
additions for four source anchors, so the requested count exceeds the three
internal gaps. With `max_tracks_per_gap = 2`, it returns two bridges between
track 11 and track 02 and two between track 02 and track 12. Every selected
bridge is reported with diagnostics recomputed against its final neighbors.
The one-worker and four-worker artifacts must be byte-identical.

The endpoint-slot result must exactly match
`expected-native-preserve-endpoint-slots-v1.json`. It requests four
additions while its four source anchors and one-track-per-gap policy provide
only three internal slots. Explicit opening and closing flags add one bounded
slot each, and the selected route uses both. The fixture proves one-sided
endpoint diagnostics, an unchanged original subsequence, final-route positions
for internal bridges, a structural upper bound of five, and byte equality
between one and four Rayon workers. Without the explicit endpoint flags this
count is structurally impossible under the same internal-gap policy.

Regeneration writes the adaptive scoring request, all bridge requests, and the
mixed semantic evidence bundle; their hashes are recorded in `manifest.json`.
