# Stable error and warning codes

Version 1 reserves the following initial codes. Codes are machine-readable;
human-readable messages may improve without a schema-version change.

| Code | Meaning |
| --- | --- |
| `INVALID_REQUEST` | The request does not satisfy the supported v1 contract. |
| `UNSUPPORTED_SCHEMA` | An input contract or database schema is unsupported. |
| `ARTIFACT_HASH_MISMATCH` | A frozen artifact differs from its declared digest. |
| `TRACK_NOT_ANALYZED` | A requested track cannot be resolved to an analyzed database row. |
| `MATRIX_REQUIRED` | The captured algorithm requires a learned matrix that is absent or invalid. |
| `INFEASIBLE_REPEAT_WINDOWS` | No route satisfies the captured repeat constraints. |
| `INFEASIBLE_EXTENSION_COUNT` | The exact requested bridge count cannot be achieved safely. |
| `SEMANTIC_EVIDENCE_REDUCED` | Optional semantic evidence is partial, stale, or unavailable. |
| `CANCELLED` | The caller requested cancellation before atomic completion. |
| `INTERNAL_ERROR` | An unexpected failure prevented a safe result. |

The implementation may add codes, but must never silently reinterpret an
existing code within schema version 1.

