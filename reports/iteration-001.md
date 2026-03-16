# Iteration 001 Report: Bootstrap

## Branch
`bootstrap/iteration-001`

## Summary
Successfully bootstrapped the Bay-Max repository with full modular monolith architecture. All 8 tasks (T001-T008) completed. The codebase installs cleanly, all 32 tests pass, lint passes with zero errors, and the FastAPI server starts with all endpoints functional.

## Tasks Completed

### T001: Bootstrap Repository
- Cloned empty repository, created `bootstrap/iteration-001` branch
- Created full directory structure with all required files
- Set up `pyproject.toml` with hatchling build system
- Created `.gitignore` for Python, models, data, caches, databases, media
- Verified: `pip install -e ".[dev]"` succeeds, `import baymax` works

### T002: Create Project Instructions for AI Developers
- Created `.github/copilot-instructions.md` with repo-wide instructions
- Created path-specific instructions: `backend.instructions.md`, `docs.instructions.md`, `tests.instructions.md`
- Created `.github/pull_request_template.md`
- Created `CLAUDE.md` with full project context, build/test/lint commands
- Created `AGENTS.md` with workflow, principles, honesty rules

### T003: Implement Core Schemas and Settings
- Implemented 7 enum types in `core/enums.py`
- Implemented settings management via `pydantic-settings` in `config/settings.py`
- Implemented 10+ Pydantic v2 models across `schemas/user.py`, `session.py`, `perception.py`, `memory.py`, `response.py`
- Verified all schemas import and instantiate correctly

### T004: Implement Memory and Orchestration Skeleton
- Capture interfaces with `FrameSource` ABC and `StubFrameSource`
- Perception interfaces with `FaceDetector`, `FaceRecognizer`, `EngagementEstimator`, `EmotionEstimator` ABCs and stubs
- State management with `InteractionState` model and `StateManager`
- Memory module: `MetadataStore` ABC, `SQLiteMetadataStore`, `VectorStore` ABC, `StubVectorStore`, salience scoring, retrieval, consolidation stubs
- Planner: `ResponsePlanner` ABC, `SupportivePlanner` (rule-based)
- Dialogue: `DialogueProvider` ABC, `RuleBasedDialogue` (template-based)
- Orchestrator: `Orchestrator` service coordinating end-to-end flow
- Verified: user creation, session creation, memory query, response generation all work

### T005: Implement API and Demo Shell
- FastAPI app with 8 endpoints (all tested via curl)
- Gradio demo shell with Setup, Interact, and Memory tabs
- End-to-end happy path: create user -> create session -> query memories -> get response

### T006: Write Architecture and Project Memory Docs
- `docs/PROJECT_BRIEF.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `BACKLOG.md`
- `docs/PROJECT_MEMORY.md`, `CHANGELOG.md`
- `docs/project_state.json` (valid JSON, machine-readable)
- ADR-0001 (modular monolith), ADR-0002 (storage paths), ADR-0003 (vision-first MVP)

### T007: Add Developer Workflow and Tests
- Makefile with install, lint, test, run-api, run-demo, clean targets
- 32 pytest tests across 3 test files (all passing)
- CI workflow at `.github/workflows/ci.yml`
- Scripts: `setup_sol.sh`, `run_api.sh`, `run_demo.sh`
- Lint: zero errors with ruff

### T008: Commit, Push, and Report
- Ran lint and tests (32 passed, 0 failed)
- Committed with conventional commit messages
- Pushed branch to origin
- Reports written

## Test Results
- **Command**: `pytest tests/ -v`
- **Result**: 32 passed, 0 failed
- **Warnings**: 17 (all `datetime.utcnow()` deprecation from Pydantic)

## Lint Results
- **Command**: `ruff check src/ apps/ tests/`
- **Result**: All checks passed

## Commands Run
- `git clone https://github.com/Krishnaa-Vinod/bay-max.git .`
- `git checkout -b bootstrap/iteration-001`
- `python -m venv .venv && pip install -e ".[dev]"`
- `python -c "import baymax; ..."` (schema verification)
- `uvicorn apps.api.main:app ...` (API server test)
- `curl` against all 8 API endpoints
- `ruff check src/ apps/ tests/` then `--fix --unsafe-fixes`
- `pytest tests/ -v`

## Known Issues
1. SQLite store uses synchronous `sqlite3` wrapped in async methods (should migrate to aiosqlite)
2. Perception stubs return hardcoded values
3. Vector store stub uses naive dot product similarity
4. `datetime.utcnow()` deprecation warnings from Pydantic (cosmetic)

## Deviations from Prompt
- None significant. All required files, endpoints, schemas, and tests implemented as specified.

## Risks
- The sync SQLite wrapper in async context could cause blocking in production loads
- No real perception means the system cannot actually recognize users yet
- Rule-based dialogue is limited in variety

## Next Recommended Tasks
1. Implement real face detection using MediaPipe or RetinaFace
2. Add vector store with FAISS or LanceDB for semantic memory search
3. Replace rule-based dialogue with LLM-based generation
4. Migrate SQLite store to fully async aiosqlite
5. Add memory consolidation logic (episodic -> semantic)

## Open Questions
- Which face detection model to use? (MediaPipe vs RetinaFace vs other)
- Which vector store? (FAISS vs LanceDB vs ChromaDB)
- Local LLM vs API LLM for dialogue generation?
- Should we add a text input modality alongside vision?
