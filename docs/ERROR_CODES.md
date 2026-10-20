# Stable error and warning codes

Version 1 reserves the following initial codes. Codes are machine-readable;
human-readable messages may improve without a schema-version change.

| Code | Meaning |
| --- | --- |
| `INVALID_REQUEST` | The request does not satisfy the supported v1 contract. |
| `UNSUPPORTED_SCHEMA` | An input contract or database schema is unsupported. |
| `ARTIFACT_HASH_MISMATCH` | A frozen artifact differs from its declared digest. |
| `PLAY_COUNTS_REQUIRED` | A non-zero play-count influence was requested without a frozen play-count snapshot. |
| `PLAY_COUNTS_SCHEMA_MISMATCH` | The play-count artifact does not declare the supported `lms-play-counts-v1` contract. |
| `PLAY_COUNTS_DATABASE_MISMATCH` | The play-count snapshot was captured for a different guarded Bliss database identity. |
| `TRACK_NOT_ANALYZED` | A requested track cannot be resolved to an analyzed database row. |
| `MATRIX_REQUIRED` | The requested optimizer mode currently requires `artifacts.learned_matrix`; Adaptive routing and bridge scoring still need the matrix for one-track context fallback. |
| `INFEASIBLE_REPEAT_WINDOWS` | No route satisfies the captured repeat constraints. |
| `INFEASIBLE_EXTENSION_COUNT` | The exact requested bridge count cannot be achieved safely. |
| `FIXED_GAP_COUNT_INVALID` | Fixed-gap mode is missing a per-gap count or the count is outside the supported `1..8` range. |
| `FIXED_GAP_COUNT_TOO_LARGE` | The derived exact fixed-gap result would overflow or exceed the 500-track final-size limit. |
| `FIXED_GAP_INSUFFICIENT_CANDIDATES` | The frozen eligible inventory does not contain enough unique additions for all requested source gaps. |
| `FIXED_GAP_SOURCE_REPEAT_CONFLICT` | Even after inserting the configured count into every gap, immutable source artists or albums would still violate a captured repeat window. |
| `FIXED_GAP_ROUTE_NOT_FOUND` | The bounded global planner could not retain a complete exact route through every source gap, or the completed route failed its independent proof. |
| `BRIDGE_SHORTLIST_FAILED` | Deterministic bridge-candidate shortlisting could not be completed. |
| `SEMANTIC_EVIDENCE_REDUCED` | Optional semantic evidence is partial, stale, or unavailable. |
| `CANCELLED` | The caller requested cancellation before atomic completion. |
| `INTERNAL_ERROR` | An unexpected failure prevented a safe result. |

The implementation may add codes, but must never silently reinterpret an
existing code within schema version 1.
