# bliss-playlist-optimizer

`bliss-playlist-optimizer` is the native orchestration layer for deterministic,
auditable playlist optimization. It will consume the shared
`bliss-mixer-core`, Lyrion metadata, repeat-window settings, and optional
semantic evidence without requiring Python on the server.

The first executable contract slice exposes `version --json` and a read-only
request validator:

```text
cargo run -- validate --request examples/reorder-only-request.json
```

`validate` checks both JSON schemas, declared artifact hashes, SQLite
integrity and `TracksV2` compatibility, the optional learned matrix, semantic
evidence, and exact usable Bliss identities for every source track. Relative
artifact paths are resolved against the process working directory; production
callers should pass absolute paths. Success is written as one JSON object to
stdout. Validation failures are written as one JSON object to stderr and exit
with status 1; invalid CLI usage exits with status 2.

The schemas in `schemas/` remain the versioned compatibility boundary. Route
optimization and playlist writing are intentionally not implemented yet.

The Python one-shot implementation remains the behavioral oracle until native
parity is measured against synthetic fixtures.

## Development

Rust is pinned by rust-toolchain.toml. Open the repository in a Dev Container
for a self-contained Linux environment with Rust, SQLite tools, and Python, or
use any local rustup installation; both paths run the same toolchain version.

```text
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Licensed under GPL-3.0-only. See `LICENSE`.

