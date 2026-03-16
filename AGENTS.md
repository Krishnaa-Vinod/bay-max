# AGENTS.md - Instructions for AI Development Agents

## Project Context
Bay-Max is a vision-first empathetic companion MVP. This file provides context and workflow guidance for any AI agent working on this codebase.

## Supervisor
- **Krishnaa Vinod** is the project supervisor and final approver.

## Current Iteration
- **Iteration 001**: Bootstrap and initial architecture.
- See `docs/project_state.json` for machine-readable status.
- See `reports/` for iteration reports.

## What Exists
- Modular monolith under `src/baymax/` with defined module boundaries.
- FastAPI backend at `apps/api/main.py`.
- Gradio demo at `apps/demo/gradio_app.py`.
- SQLite-backed metadata storage.
- Rule-based dialogue provider (no external API keys).
- Perception interfaces are stubs awaiting implementation.

## Workflow
1. Read `docs/PROJECT_MEMORY.md` for context on previous work.
2. Read `docs/project_state.json` for current iteration state.
3. Read `docs/BACKLOG.md` for pending work items.
4. Implement changes following the modular monolith pattern.
5. Run `make test` and `make lint` before committing.
6. Update `docs/PROJECT_MEMORY.md` and `docs/project_state.json` after significant work.
7. Write an iteration report in `reports/` when completing a milestone.

## Engineering Principles
- Build a modular monolith, not microservices.
- Prefer clear interfaces over premature optimization.
- Make repo memory explicit and persistent in files.
- Never claim a feature is complete unless code, tests, and file evidence exist.
- Keep large artifacts out of git.
- Use environment variables for all data/model/cache locations.
- Use safe, non-clinical language in docs and code.

## Honesty Rules
- Do not state that a feature works unless you ran it or tested it.
- Do not invent benchmark numbers.
- Do not fake push status or commit hashes.
- Clearly separate completed work, partial work, and planned work.

## Non-Goals for Iteration 001
- Robotics, clinical diagnosis, wearables
- Doctor report ingestion
- Audio runtime features
- Fine-tuning or training large models
- Production deployment
