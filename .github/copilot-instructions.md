# Bay-Max: Copilot Instructions

## Project Overview
Bay-Max is a memory-first empathetic companion agent. It recognizes enrolled users from vision input, tracks sessions, stores meaningful interaction memory, and produces supportive personalized responses.

## Architecture
- **Style**: Modular monolith (not microservices)
- **Stack**: Python 3.11+, FastAPI, Pydantic v2, SQLite, Gradio
- **Source**: `src/baymax/` contains all library code
- **Apps**: `apps/api/` (FastAPI), `apps/demo/` (Gradio)

## Key Conventions
- Use Pydantic v2 models for all data schemas.
- Use environment variables for all data/model/cache paths (see `.env.example`).
- Never commit model weights, private data, video recordings, or cache directories.
- Prefer clear interfaces over premature optimization.
- Use safe, non-clinical language in all docs and comments.
- Separate observations from inferences in memory models.

## Development Commands
```bash
make install    # Install dependencies
make test       # Run tests
make lint       # Run linter
make run-api    # Start FastAPI server
make run-demo   # Start Gradio demo
```

## Testing
- Tests live in `tests/`.
- Use pytest with pytest-asyncio for async tests.
- Every module should have corresponding tests.

## Environment Variables
- `BAYMAX_DATA_DIR` - Path to data directory
- `BAYMAX_CACHE_DIR` - Path to cache directory
- `BAYMAX_MODEL_DIR` - Path to model weights
- `BAYMAX_DB_URL` - Database connection URL (default: sqlite)
