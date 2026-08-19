# bliss-playlist-optimizer

**bliss-playlist-optimizer** is the network-free Rust engine behind the Lyrion
plugin [Better Call Bliss](https://github.com/chrober/lms-better-call-bliss).
It turns a frozen playlist request, Bliss feature database, repeat rules, the
current learned similarity matrix, and optional semantic evidence into an
auditable proposed route. It can reorder fixed membership, analyze and select
bridge tracks, preserve source anchors, extend a fixed source set to an exact
target, or build a destination-locked route from a live queue tail.

The native engine is needed because scoring tens of thousands of analyzed
tracks and searching many contextual routes is computational work that does not
belong in the Lyrion plugin's Perl process. It shares Bliss database and
Adaptive similarity behavior with `bliss-mixer` through
`bliss-mixer-core`, uses deterministic parallel Rust search where useful,
and requires neither Python nor network access on the server.

This program deliberately does not call Last.fm, modify `bliss.db`, write
audio metadata, or create Lyrion playlists. Better Call Bliss owns provider
access, LMS identities, user interaction, Preview, and playlist persistence;
this repository owns the versioned native request/result contracts, validation,
scoring, selection, routing, and diagnostic artifacts. The user-facing playlist
modes and options are described in the plugin's
[strategy guide](https://github.com/chrober/lms-better-call-bliss/blob/main/ALGORITHMS.md).

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

Production callers can request structured native stage timings, an
identity-bound decoded-library cache, and a live status sidecar:

```text
bliss-playlist-optimizer bridge --request request.json --timings --cache-dir cache --progress progress.json
```

A caller that has already produced and validated the request through a trusted,
version-matched integration can explicitly add `--trusted-request` to `route`
or `bridge`. This skips repeated runtime JSON-Schema compilation and the
O(library-size) inventory-to-database cross-check. It does **not** skip typed
JSON decoding, declared artifact hashes, database/cache identity binding,
source-track resolution, semantic-evidence validation, membership enforcement,
or repeat constraints. The flag is intentionally unavailable to `validate` and
must never be used for arbitrary user-supplied request files. Callers should
feature-detect it through `version --json` rather than assume it exists.

`--progress` atomically replaces a small JSON file while the command is running.
It follows the same human-readable status-message idea as `bliss-analyser` and
`bliss-learner`, but deliberately uses a local sidecar instead of LMS JSON-RPC
push notifications.

That difference is intentional. `bliss-analyser` and `bliss-learner` are
long-running maintenance tools launched specifically by the LMS plugin; they can
be told the LMS JSON-RPC port and periodically push `msg:` updates back to
`lms-blissmixer`. `bliss-playlist-optimizer` is a stricter request/response
engine: stdout is the machine-readable final artifact, stderr is reserved for a
machine-readable failure, and the binary should stay useful outside Lyrion
without knowing anything about LMS host names, ports, authentication, players,
or plugin command names. If it pushed directly to LMS, the native engine would
become coupled to one plugin deployment model and every offline, authenticated,
or renamed-controller scenario would need extra native error handling.

The sidecar keeps the same UX value with cleaner boundaries. Better Call Bliss
owns LMS integration and polls the job-local file while the process is alive;
other callers can do the same or ignore it entirely. The file contains `stage`,
`msg`, elapsed seconds, and optional `current`/`total`/`percent` fields. Progress
writes are best-effort and never fail the optimization, so status reporting
cannot corrupt stdout, mask the real optimizer result, or turn a successful
playlist preview into a failed one.

The request's database artifact must include `cache_identity` for cache reuse.
Lyrion supplies its `device:inode:size:mtime` identity and independently rejects
a result if that identity changes while the job is running. A cold job streams
the database SHA-256, runs `quick_check`, reads every usable track with one bulk
query, and atomically replaces the versioned cache. A warm job reuses the hash,
integrity result, and decoded library only when the path and identity match.
Cache corruption, inconsistent metadata/feature vector counts, or an identity
change is a safe miss.

Cache format v2 stores compact metadata separately from the route features and
artist/album repeat keys. Destination jobs borrow those decoded route tracks
instead of cloning the complete library for every optimizer process. Optional
semantic lookup is evidence-scoped: it scans candidates once but retains index
entries only for recording/artist keys actually named by the evidence bundle.
Its retained memory is therefore proportional to evidence matches rather than
the full library. A 200,000-candidate regression test protects this property.
The remaining cold and warm setup passes are intentionally linear in library
size; the benchmark command below should be used on representative hardware.

Run a repeatable cold-then-warm benchmark with the server's Perl runtime:

```text
perl scripts/benchmark-request.pl --binary ./bliss-playlist-optimizer --command bridge --request request.json --iterations 3
```

Each JSON line reports external wall time, native total time, cache state, and
the individual native stages. The temporary benchmark cache is removed when the
script exits unless `--cache-dir` is supplied.

`validate` checks both JSON schemas, declared artifact hashes, SQLite integrity
and `TracksV2` compatibility, the learned matrix when supplied, semantic
evidence, and exact usable Bliss identities for every unique source track.
Relative artifact paths are resolved against the process working directory;
production callers should pass absolute paths.

`score` emits a versioned contextual scoring artifact for the request's existing
order. Adaptive behavior comes from the same shared core as the learned-matrix-
enabled `bliss-mixer` fork: one-track contexts use the learned matrix when it is
supplied, while two or more seed tracks dynamically blend the learned matrix
with seed variance according to `learned_percent`. If no learned matrix is
supplied, multi-track Adaptive contexts use variance alone and one-track
contexts fall back to `scoring.feature_weights`. Explicit `static` scoring also
uses `scoring.feature_weights` for every context by converting the 23 feature
weights into a fixed diagonal matrix. The result is a sequence of contextual
transition legs, not a static pairwise matrix.

`route` performs fixed-set sequencing without writing a playlist. Every source
track appears exactly once. Artist and album look-back windows are hard
constraints; track repetition is impossible by unique membership. The primary
objective is the transition sum plus twice the worst transition. Deterministic
fixed starts and seeded greedy restarts are improved with reversal and relocation
moves. A separately searched energy-arc candidate is selected only when its
primary objective remains within 8% and its arc error improves by at least 10%.
The JSON artifact records both candidates, the selected strategy, hashes,
settings, and repeat validation.

Requests may include the strategy-neutral `selection` block with
`variation_percent`, `generation_seed`, `lastfm_track_guidance_percent`, and
`lastfm_artist_guidance_percent`.
Variation zero preserves strict deterministic route, bridge, and fixed-source
extension choices. Higher values seed route search, reorder a bounded pool of
acoustically qualified bridge candidates, and let fixed-source extension perform
reproducible weighted sampling inside a bounded top acoustic pool.
The same seed and inputs reproduce membership across worker counts. Selection
is downstream of scoring rather than nested under Adaptive, so Static and
Forest can reuse it when those strategies are connected. The two Last.fm values
independently scale recording and artist evidence after local-inventory,
acoustic, uniqueness, and repeat-capacity qualification. Zero ignores that
evidence type. Bridge ranking caps the combined adjustment at ten percentile
points. Deterministic fixed-source extension caps semantic movement at 20% of its bounded
Bliss relevance pool; varied fixed-source extension uses a bounded evidence multiplier. These
are guidance strengths, not quotas, and even 100 cannot rescue an acoustically
rejected candidate. The deprecated `lastfm_artist_probability` spelling remains
an input alias for artist guidance. Omitting the block retains deterministic
zero-guidance defaults.

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
the outgoing context. A two-track source supplies only one self-referential
observation, which would assign its sole transition percentile zero regardless
of absolute distance. When fewer than two source observations exist, the
optimizer instead freezes a deterministic reference population from the current
local candidate inventory plus the source anchors. It emits opaque row IDs bound
to the database hash, aggregate rejection counts, and a bounded list of accepted
candidates per gap; it exposes no library paths. Independent candidates are
ranked deterministically with Rayon.

Large libraries may set `extension.shortlist_limit` to bound the candidates
that enter strict contextual bridge scoring and exact-count search. The
deterministic shortlist reuses the strict dynamic two-leg Adaptive ranker for
the original gap, including accepted status, worst-leg percentile, and detour
percentile. It only narrows the pool and never replaces final rescoring for
evolving search states or any
semantic, repeat, membership, and acoustic gate. Up to 32 candidates carrying
endpoint-local semantic evidence are reserved before the remaining shortlist is
filled acoustically. Per-gap diagnostics report the shortlisted and excluded
counts when narrowing occurred. Omitting the field preserves exhaustive
evaluation; the LMS plugin currently uses a conservative limit of 256.

The bridge command consumes a frozen provider-neutral evidence graph. Recording
support for both or one endpoint precedes endpoint-local artist support. When
any usable endpoint-local evidence exists, collection-artist evidence is not
used for that gap; it is considered only when the local evidence set is empty.
Candidates without a matching edge always remain in the Bliss pool. Provider
states and every matched assertion retain provenance, rank or score, identity
confidence, observation time, and cache state. Recording entities match by
optimizer identity, shared MBID, or normalized artist and title. Disabled,
unavailable, partial, or failed providers are non-fatal and may coexist with
cached evidence. Final acceptance is always acoustic and repeat-safe; configured
Last.fm guidance only applies the bounded post-qualification rank adjustment.
Semantic candidate resolution and acoustic candidate evaluation both use
deterministic parallel iteration.

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

Destination-route requests use `extension.mode=destination_route` together with
`route.ordering_policy=queue_destination`, `route.start_track_id`, and
`route.destination_track_id`. The final two source tracks must be the locked
start and destination; any earlier source tracks are read-only acoustic and
repeat context. Only the final gap is extended. `destination_mode=exact`
requires exactly `additional_track_count` intermediates and remains all-or-
nothing. Automatic accepts optional `min_added_tracks` and required
`max_added_tracks` bounds from zero through eight; the minimum must not exceed
the maximum. A minimum of zero permits the direct destination. Exact counts are
also bounded from zero through eight.

Destination routes use a dedicated fixed-matrix layered path search rather than  
the generic contextual gap-insertion search. It builds complete paths for the  
permitted intermediate counts and ranks them by worst adjacent Bliss distance,  
then adjacent-distance sum, semantic support, and deterministic identity. A  
lower bound based on the remaining endpoint distance keeps the beam focused  
without repeatedly rescoring the full library. Automatic returns the shortest  
permitted path whose measured adjacent percentiles meet `trigger_percentile`;  
if none qualifies, it returns the lowest-bottleneck repeat-safe best effort  
within the configured minimum and maximum. Exact uses  
the same path objective for precisely `additional_track_count` intermediates.  

`extension.search_effort` controls bounded search breadth independently from  
the quality target and bridge budget: `fast` uses a 128-track shortlist, six  
expansions per state, and beam width 32; `balanced` uses 256, eight, and 64; and  
`thorough` uses 512, sixteen, and 192. Older schema-v1 requests without this  
field retain `balanced` behavior. The distance index transforms every library
feature vector once and reuses O(23) pair lookups, so comparing more bridge  
depths does not repeat O(23^2) matrix work. Destination setup also reuses that
index for the two-track reference population. Before the expensive contextual
rerank, a deterministic prefilter retains tracks near the left endpoint, right
endpoint, and acoustic midpoint. Fast, Balanced, and Thorough cap that coarse
pool at 65,536, 131,072, and 262,144 candidates respectively. Libraries below
the selected cap retain the prior full contextual shortlist; larger libraries
bound the expensive work while preserving endpoint and path coverage.

Variation is applied only after complete routes have been ranked. It may choose  
reproducibly inside a narrow band of the deterministic winner (within 2% of its  
adjacent bottleneck and 5% of its adjacent sum), but it cannot alter graph  
reachability or relax repeat constraints. Candidate discovery still uses one  
bounded shortlist derived from the original destination gap; depth-specific  
full-library candidate expansion remains a possible later quality enhancement.  

Every feasible destination result publishes `selection_preview.route_quality`.  
It covers only the requested path from the captured queue tail through generated  
intermediates to the destination, not unrelated earlier context edges. Each  
actual neighboring edge uses `fixed-matrix-adjacent-distance` and a matching  
`source-relative-local-library-percentile`; the artifact also identifies the  
effective learned-matrix or Static-weight role and SHA-256, adjacent-distance  
sum, raw bottleneck, and worst adjacent percentile. Adaptive rolling-context  
values remain secondary candidate diagnostics rather than adjacent quality.  
Generated tracks remain unique and are checked against every artist and album
inside the configured repeat windows, including the explicit destination. The
destination itself remains immutable user intent: a conflict already present
solely between the captured queue context and destination does not make bridge
insertion impossible. Variation and the frozen provider-neutral evidence graph
cannot bypass generated-track membership or repeat constraints.

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

This remains analysis-only by design. Better Call Bliss applies accepted
previews and writes Lyrion playlists from the returned opaque track identities.

Success is written as one JSON object to stdout. Validation or search failures
are written as one JSON object to stderr and exit with status 1; invalid CLI
usage exits with status 2. The schemas in `schemas/` are the versioned
compatibility boundary.

The Python one-shot implementation remains the behavioral oracle until every
planned native mode has dedicated parity coverage.

## Release artifacts

The optimizer repository owns native binary builds and their test gate. The
`.github/workflows/release.yml` workflow runs formatting, Clippy, and tests,
then builds release binaries for the platform folders consumed by
[Better Call Bliss](https://github.com/chrober/lms-better-call-bliss):

- `bliss-playlist-optimizer-aarch64-linux`
- `bliss-playlist-optimizer-armhf-linux`
- `bliss-playlist-optimizer-x86_64-linux`
- `bliss-playlist-optimizer-mac`
- `bliss-playlist-optimizer-windows.exe`

Each asset is published with a `.sha256` file. Normal plugin publishing should
consume a pinned optimizer GitHub release instead of rebuilding optimizer source
inside the LMS plugin workflow. Workflow artifacts are still produced for dry
runs and development inspection.

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
