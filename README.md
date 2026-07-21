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
cargo run -- bridge --request fixtures/synthetic/preserve-automatic-request.json
cargo run -- bridge --request fixtures/synthetic/preserve-exact-count-request.json
cargo run -- bridge --request fixtures/synthetic/preserve-multi-track-gap-request.json
cargo run -- bridge --request fixtures/synthetic/preserve-endpoint-slots-request.json
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
enumerates usable TracksV2 rows in stable row-id order and excludes curated and
duplicate recording identities. Depending on the declared ordering policy, it
either optimizes the original route or keeps the source order as immutable
anchors. It then builds the frozen cross-context Adaptive reference distribution
and rescores both sides of each candidate insertion with the bridge present in
the outgoing context. It emits opaque row IDs bound to the database hash,
aggregate rejection counts, and a bounded list of accepted candidates per gap;
it exposes no library paths. Independent candidates are ranked deterministically
with Rayon.

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
Exact-count requests default to one bridge in each original internal gap.
Preserve-order requests may opt into a larger, explicit
`extension.max_tracks_per_gap` bound from 1 through 8. The search appends
candidates before the right anchor, so candidate order forms a small route
inside the gap. It retains separate global beams per total addition count and a
bounded local frontier per gap depth. The structural upper bound is the smaller
of the unique frozen candidate count and
`internal gaps * max_tracks_per_gap`.

Every tentative append passes the existing frozen semantic pool, membership,
repeat, and two-sided acoustic gates and causes the complete route objective to
be recomputed. Once a route is selected, each inserted bridge is removed and
reinserted virtually so its published two-leg diagnostics reflect its final
neighbors and Adaptive context. All tracks in a chained gap currently come from
the semantic pool frozen for the original anchor endpoints.

Exact-count requests may independently opt into
`extension.allow_opening_track` and `extension.allow_closing_track`. Each
enabled endpoint has hard capacity one; endpoint tracks are never added unless
the corresponding flag is explicitly true. An opening candidate has no
invented incoming transition: it is scored only into the first source anchor,
using the candidate as the one-track Adaptive context. A closing candidate is
scored only from the complete preceding route into the candidate. Both must
pass unique-membership, complete-route repeat, and max-leg percentile gates.
The structural upper bound becomes the smaller of the unique candidate count
and `internal gaps * max_tracks_per_gap + enabled endpoint slots`.

Endpoint semantics are likewise one-sided. A recording edge from the real
anchor yields `recording_one`, never fabricated `recording_both` support;
endpoint-local artist evidence follows, then collection fallback, then
Bliss-only operation. Opening evidence records the source anchor as the right
endpoint and closing evidence records it as the left endpoint.

Endpoint exact search is a deterministic bounded staged search, not a claim of
joint global optimality. It enumerates the allowed opening/closing-use
combinations, obtains the best bounded internal-gap route for the remaining
count, enumerates retained endpoint candidates, and selects by the recomputed
complete-route objective and stable route identity. Published internal bridge
diagnostics are reconstructed against that complete route, including any
opening shift. The artifact separately records the endpoint policy, each
one-sided decision, and its evidence and percentile.

With `route.ordering_policy = preserve_order`, both automatic and exact-count
extension keep every source track in precisely its input position relative to
the other source tracks. The artifact records the source IDs separately from the
selected route IDs and tests their equality with the final original-track
subsequence. Because source tracks are immutable in this mode, an input order
that already violates an artist or album look-back window fails with
`PRESERVED_ANCHOR_REPEAT_CONFLICT`; this slice does not misrepresent a
bounded gap search as capable of repairing several interacting anchor
conflicts. Automatic mode remains limited to one bridge per gap.

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
