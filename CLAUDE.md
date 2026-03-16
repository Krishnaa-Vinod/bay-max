# CLAUDE.md - Instructions for Claude Code

## Project
Bay-Max is a memory-first empathetic companion agent MVP. It uses vision input for user recognition, tracks interaction sessions, stores meaningful observations and memories, and generates supportive personalized responses.

## Repository Layout
```
src/baymax/           # Core library (modular monolith)
  config/settings.py  # Pydantic settings from env vars
  core/enums.py       # Shared enums
  schemas/            # Pydantic data models (user, session, perception, memory, response)
  capture/            # Video/webcam ingestion interfaces
  perception/         # Face detection/recognition interfaces (stubs in iteration-001)
  state/              # Interaction state management
  memory/             # Memory storage, retrieval, salience, consolidation
  planner/            # Response strategy planning
  dialogue/           # Response generation (rule-based in iteration-001)
  orchestrator/       # End-to-end flow coordination
apps/api/main.py      # FastAPI application
apps/demo/gradio_app.py # Gradio demo
tests/                # pytest test suite
docs/                 # Documentation, ADRs, project state
reports/              # Iteration reports
scripts/              # Setup and run scripts
```

## Build & Test
```bash
pip install -e ".[dev]"   # Install with dev dependencies
make test                 # Run all tests
make lint                 # Run ruff linter
make run-api              # Start FastAPI on port 8000
make run-demo             # Start Gradio demo
```

## Key Constraints
- Python 3.11+ required.
- All data/model/cache paths must use environment variables (see `.env.example`).
- Never commit model weights, private data, videos, or caches.
- Use Pydantic v2 for all schemas.
- Use safe, non-clinical language.
- Iteration-001 uses rule-based dialogue (no external API keys needed).
- Perception modules are interface stubs in this iteration.

## Architecture Decisions
- Modular monolith, not microservices (see ADR-0001).
- SQLite for metadata, vector store adapter for embeddings (see ADR-0002).
- Vision-first MVP scope (see ADR-0003).

## Project Memory
- `docs/project_state.json` tracks iteration status (machine-readable).
- `docs/PROJECT_MEMORY.md` provides continuity context for agents.
- `reports/` contains per-iteration reports.

## Conventions
- Interfaces defined in `interfaces.py` per module.
- Separate observations from inferences in memory.
- Separate episodic memories from semantic facts.
- Support correction and deletion pathways in schemas.
