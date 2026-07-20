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
cargo run -- bridge --request fixtures/synthetic/semantic-bridge-request.json
cargo run -- bridge --request fixtures/synthetic/automatic-preview-request.json
cargo run -- bridge --request fixtures/synthetic/exact-count-request.json
cargo run -- bridge --request fixtures/synthetic/exact-count-infeasible-request.json
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

The bridge command consumes a frozen provider-neutral evidence graph. Recording
support for both or one endpoint precedes endpoint-local artist support. When
any usable endpoint-local evidence exists, collection and Bliss-only candidates
are excluded for that gap; collection-artist fallback is considered only when
the local pool is empty, followed by Bliss-only operation when no usable edge
remains. Provider states and every matched assertion retain provenance, rank or
score, identity confidence, observation time, and cache state. Raw scores from
different providers are reported but never compared. Disabled, unavailable,
partial, or failed providers are non-fatal and may coexist with cached evidence.
Within one evidence tier, candidates are ordered by identity confidence, then
the best provider-local ordinal rank when present, then acoustic worst-leg and
detour percentiles and stable row identity. Semantic candidate resolution and
acoustic candidate evaluation both use deterministic parallel iteration.

The same artifact now includes a read-only automatic selection preview. The
request declares both the severe-gap percentile and maximum added-track budget.
Original gaps are processed left-to-right so every Adaptive score includes all
earlier proposed bridges and later proposals cannot alter earlier contexts. A
bridge is selected only above the threshold, after all semantic, membership,
repeat, and acoustic gates pass, and when its two contextual legs improve the
local "sum plus twice the worst leg" objective over the direct transition. The
preview reports the proposed final sequence and a selected, below-threshold,
budget, eligibility, repeat, acoustic, or no-improvement reason for every gap.

Exact-count requests use a deterministic bounded beam search over the original
internal gaps. Search states are kept separately by addition count so a
lower-count route cannot crowd the requested count out of the beam. Every
tentative insertion is contextually rescored, unique, repeat-safe, and inside
the same acoustic gates; completed states are ordered by the full
bottleneck-then-sum route objective and stable route identity. Independent
state and candidate evaluations use indexed Rayon iteration and reduce
deterministically.

A feasible exact preview contains exactly the requested number of bridges. An
infeasible preview contains no final sequence and no partial decisions; it
reports both the maximum count found and the structural upper bound. Only a
request above that upper bound is labeled `EXACT_COUNT_INFEASIBLE`; failure
inside the bound is honestly labeled
`EXACT_COUNT_NOT_FOUND_WITHIN_SEARCH_BOUNDS`.
The current exact-count slice permits at most one bridge in each original
internal gap. Endpoint slots and multi-track routes inside one preserved-anchor
gap remain future extensions.

This remains analysis-only. Applying a preview and playlist writing are future
slices.

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
