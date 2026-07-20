# bliss-playlist-optimizer

`bliss-playlist-optimizer` is the native orchestration layer for deterministic,
auditable playlist optimization. It will consume the shared
`bliss-mixer-core`, Lyrion metadata, repeat-window settings, and optional
semantic evidence without requiring Python on the server.

This bootstrap checkpoint deliberately exposes only a machine-readable
`version --json` command. The schemas in `schemas/` are the compatibility
boundary to be implemented next; they are not yet a promise of a working
optimizer.

The Python one-shot implementation remains the behavioral oracle until native
parity is measured against synthetic fixtures.

## Development

```text
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Licensed under GPL-3.0-only. See `LICENSE`.

