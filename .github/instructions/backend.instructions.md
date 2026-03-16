# Backend Instructions

## Code Organization
- All backend source code lives under `src/baymax/`.
- FastAPI app entry point: `apps/api/main.py`.
- Use Pydantic v2 `BaseModel` for request/response schemas.
- Use `pydantic-settings` for configuration (`src/baymax/config/settings.py`).

## Module Boundaries
Each module (`capture`, `perception`, `state`, `memory`, `planner`, `dialogue`, `orchestrator`) exposes its interface via an `interfaces.py` file. Depend on interfaces, not concrete implementations.

## Database
- SQLite for metadata storage (via `aiosqlite`).
- Vector store adapter interface for future embedding search.
- Do not leak database implementation details into API layer.

## Error Handling
- Raise `HTTPException` at the API layer only.
- Core modules should raise domain-specific exceptions.
- Use safe, supportive language in user-facing error messages.

## Environment
- All data paths come from environment variables.
- Never hard-code file paths.
- See `.env.example` for required variables.
