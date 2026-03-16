# ADR-0002: Storage and Data Paths

## Status
Accepted

## Context
Bay-Max will handle model weights, face encodings, video frames, and databases. These artifacts vary in size and sensitivity. The repository must remain lightweight and must not contain private data or large files.

## Decision
- Use environment variables for all data, cache, and model paths: `BAYMAX_DATA_DIR`, `BAYMAX_CACHE_DIR`, `BAYMAX_MODEL_DIR`, `BAYMAX_DB_URL`.
- Store only source code, docs, configs, lightweight test fixtures, and reports in the repository.
- Never commit model weights, private user data, video recordings, checkpoints, or cache directories.
- Use SQLite for metadata storage (portable, zero-config).
- Define a vector store adapter interface for future embedding search.

## Consequences
- **Positive**: Repository stays small and safe to clone.
- **Positive**: Different environments (local, Sol cluster, CI) can configure paths independently.
- **Positive**: No risk of accidentally committing sensitive data.
- **Negative**: Developers must set up environment variables and create directories before running.
- **Mitigation**: `scripts/setup_sol.sh` and `.env.example` provide guidance.
