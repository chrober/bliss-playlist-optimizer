# bliss-playlist-optimizer

`bliss-playlist-optimizer` is the native orchestration layer for deterministic,
auditable playlist optimization. It consumes the shared `bliss-mixer-core`,
Lyrion metadata, repeat-window settings, and optional semantic evidence without
requiring Python on the server.

The current read-only contract slice exposes:

```text
cargo run -- validate --request examples/reorder-only-request.json
cargo run -- score --request fixtures/synthetic/adaptive-scoring-request.json
cargo run -- route --request fixtures/synthetic/adaptive-scoring-request.json
cargo run -- bridge --request fixtures/synthetic/automatic-bridge-request.json
```

`validate` checks both JSON schemas, declared artifact hashes, SQLite integrity
and `TracksV2` compatibility, the optional learned matrix, semantic evidence,
and exact usable Bliss identities for every unique source track. Relative
artifact paths are resolved against the process working directory; production
callers should pass absolute paths.

`score` emits a versioned contextual scoring artifact for the request's existing
order. Its adaptive behavior comes from the same shared core as the learned-
matrix-enabled `bliss-mixer` fork: one seed uses the learned matrix, while two or
more seeds dynamically blend the learned matrix with seed variance according to
`learned_percent`. The result is a sequence of contextual transition legs, not a
static pairwise matrix.

`route` performs fixed-set sequencing without writing a playlist. Every source
track appears exactly once. Artist and album look-back windows are hard
constraints; track repetition is impossible by unique membership. The primary
objective is the transition sum plus twice the worst transition. Deterministic
fixed starts and seeded greedy restarts are improved with reversal and relocation
moves. A separately searched energy-arc candidate is selected only when its
primary objective remains within 8% and its arc error improves by at least 10%.
The JSON artifact records both candidates, the selected strategy, hashes,
settings, and repeat validation.

Adaptive transition scores are cached privately within each restart. Independent
restarts run through indexed Rayon iteration and are reduced with stable
tie-breaking, so results are byte-identical across worker counts. By default the
executable leaves one logical CPU for Lyrion; set `RAYON_NUM_THREADS` to override
that policy. SQLite access and validation remain sequential.

The bridge command is a read-only analysis slice for automatic extension. It
enumerates usable TracksV2 rows in stable row-id order, excludes curated and
duplicate recording identities, optimizes the original route, builds the frozen
cross-context Adaptive reference distribution, and rescores both sides of each
candidate insertion with the bridge present in the outgoing context. It emits
opaque row IDs bound to the database hash, aggregate rejection counts, and a
bounded list of accepted candidates per gap; it exposes no library paths.
Independent candidates are ranked deterministically with Rayon.

This slice accepts an empty semantic graph and identifies its result as
Bliss-only. It rejects non-empty semantic edges rather than silently ignoring
them. Semantic evidence tiers, automatic insertion decisions, exact-count
policies, and playlist writing remain future slices.

Success is written as one JSON object to stdout. Validation or search failures
are written as one JSON object to stderr and exit with status 1; invalid CLI
usage exits with status 2. Playlist extension and playlist-file writing remain
future slices. The schemas in `schemas/` are the versioned compatibility
boundary.

The Python one-shot implementation remains the behavioral oracle until every
planned native mode has dedicated parity coverage.

## Development

Rust is pinned by `rust-toolchain.toml`. Open the repository in a Dev Container
for a self-contained Linux environment with Rust, SQLite tools, and Python, or
use any local rustup installation; both paths run the same toolchain version.

```text
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Licensed under GPL-3.0-only. See `LICENSE`.
