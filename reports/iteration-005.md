# Iteration 005 Report — Grounded Local Dialogue

## Branch
`feature/iteration-005-grounded-local-dialogue`

## Latest Commit SHA
`c978b2f`
## Summary

Iteration 005 extends Bay-Max from a rule-based response system to a memory-aware, grounded local dialogue system. The key changes are:

1. A **dialogue provider abstraction** with three backends: rule-based (default fallback), Ollama (local server), and HuggingFace Transformers (in-process model).
2. A **grounded prompt builder** that packages current state, retrieved memories, recent conversation turns, and explicit safety constraints into a structured prompt using a plan-then-verbalize pattern.
3. **Safety gating** on both user input (diagnosis-style pattern matching) and LLM output (unsafe assertion detection).
4. **Graceful fallback** from LLM backends to rule-based on initialisation or generation failure.
5. New API endpoint `GET /v1/dialogue/backends` and extended `/v1/respond` response with `backend`, `model_name`, `fallback_used`, and `safety_flags`.
6. Smoke test script (`scripts/dialogue_smoke.py`) and comprehensive automated tests (36 new, 194 total).
7. All required docs updated and two new docs created.

## Tasks

### Completed

- **T501**: Iteration-004 drift audit and docs/install parity fixes
  - README install table updated with dialogue extras
  - ROADMAP updated: iteration-004 marked Completed, iteration-005 items listed
  - CHANGELOG updated with missing iteration-004 entry and new iteration-005 entry

- **T502**: Dialogue backend abstraction and configuration
  - `DialogueProvider` interface extended with `prompt_context`, `backend_name`, `model_name`, `is_available()`
  - 13 new env vars with documented defaults
  - `create_dialogue_provider()` factory with fallback logic

- **T503**: Real free local-model backend
  - `OllamaDialogueProvider`: HTTP to Ollama `/api/chat`, configurable model
  - `TransformersDialogueProvider`: in-process HuggingFace chat model, lazy-loaded
  - Factory handles init failures, falls back to rule_based if enabled

- **T504**: Grounded prompt packaging and structured response parsing
  - `build_prompt_context()` assembles `GroundedPromptContext` from orchestrator data
  - `build_messages()` formats as OpenAI-style message list
  - Strategy-specific instructions per `ResponseStrategy` value
  - `SupportiveResponse` validated by Pydantic v2 schema

- **T505**: Safety and fallback behavior
  - `check_safety()`: input pattern matching (20+ diagnosis/medical-certainty phrases)
  - `check_output_safety()`: LLM output validation against unsafe assertion patterns
  - Orchestrator catches generation exceptions and falls back to rule_based
  - `safety_flags` and `fallback_used` propagated to API response

- **T506**: Memory-grounded dialogue wired into API and demo
  - `GET /v1/dialogue/backends` endpoint added
  - `/v1/respond` returns `backend`, `model_name`, `fallback_used`, `safety_flags`
  - Gradio demo updated: Backends tab added, Interact tab shows debug fields
  - Recent turns now fetched from SQLite and included in prompt context

- **T507**: Local smoke testing and evaluation flow
  - `scripts/dialogue_smoke.py` — 5 test cases
  - Outputs to `./artifacts/smoke_test_005.json`
  - `make smoke-test` target added to Makefile

- **T508**: Tests, docs, report, commit, push
  - 36 new automated tests in `tests/test_iteration_005.py`
  - All 194 tests passing, ruff lint clean
  - All required docs updated
  - 2 new docs created: `DIALOGUE_ARCHITECTURE.md`, `SAFETY_GUARDRAILS.md`
  - This report

## Tasks Partial

None — all tasks completed.

## Tasks Not Done

None.

## Files Created

- `src/baymax/dialogue/safety.py`
- `src/baymax/dialogue/prompt_builder.py`
- `src/baymax/dialogue/ollama_provider.py`
- `src/baymax/dialogue/transformers_provider.py`
- `src/baymax/dialogue/factory.py`
- `tests/test_iteration_005.py`
- `scripts/dialogue_smoke.py`
- `docs/DIALOGUE_ARCHITECTURE.md`
- `docs/SAFETY_GUARDRAILS.md`
- `reports/iteration-005.md` (this file)
- `reports/iteration-005.json`

## Files Modified

- `src/baymax/config/settings.py` — 13 new env var fields
- `src/baymax/dialogue/interfaces.py` — extended interface
- `src/baymax/dialogue/rule_based.py` — backend_name + model_name + updated signature
- `src/baymax/schemas/response.py` — 6 new schemas + 4 new fields on SupportiveResponse
- `src/baymax/orchestrator/service.py` — grounded respond(), get_dialogue_backends(), fallback
- `apps/api/main.py` — GET /v1/dialogue/backends, version 0.4.0
- `apps/demo/gradio_app.py` — Backends tab, debug fields in Interact tab
- `pyproject.toml` — version 0.4.0, dialogue extras
- `Makefile` — install-dialogue, smoke-test targets
- `.env.example` — new dialogue env vars
- `docs/README.md` → `README.md` — install table, dialogue section
- `docs/ROADMAP.md` — iteration-004 and -005 status
- `docs/ARCHITECTURE.md` — full module map update, new endpoint, dialogue flow
- `docs/CHANGELOG.md` — iteration-004 and -005 entries
- `docs/project_state.json` — iteration 005, 17 endpoints, updated issues/next tasks
- `docs/MODEL_STACK.md` — dialogue backends section
- `docs/LOCAL_TESTING.md` — dialogue testing commands, smoke test
- `docs/PROJECT_MEMORY.md` — full update for iteration-005

## Automated Tests

| Suite | Tests | Result |
|-------|-------|--------|
| `test_health.py` | 4 | Passed |
| `test_iteration_002.py` | 33 | Passed |
| `test_iteration_003.py` | 44 | Passed |
| `test_iteration_004.py` | 49 | Passed |
| `test_iteration_005.py` | 36 | Passed |
| `test_memory_models.py` | 21 | Passed |
| `test_project_state_schema.py` | 7 | Passed |
| **Total** | **194** | **Passed** |

Lint: `ruff check` — 0 errors.

## Manual Smoke Tests

| Case | Result | Notes |
|------|--------|-------|
| Rule-based baseline | **PASSED** | Response in 102ms, no LLM dependency |
| Grounded two-session recall | **PASSED** | 1 memory_ref returned after consolidation (SQL retrieval, not vector); vector search uses StubVectorStore in this env |
| Graceful no-memory response | **PASSED** | No fabricated memory |
| Safety refusal test | **PASSED** | `safety_flags=['diagnosis_request:diagnose me']` |
| Backend failure fallback | **PASSED** | rule_based fallback triggered for bad Ollama config |
| Ollama (real model) | **NOT RUN** | Ollama server not available in this environment |
| Transformers (real model) | **NOT RUN** | Models not downloaded in this environment |

## API Endpoints Verified

| Endpoint | Status | Notes |
|----------|--------|-------|
| GET /v1/dialogue/backends | Passed | Returns available_backends, active_backend, fallback_enabled |
| POST /v1/respond | Passed | Returns backend, model_name, fallback_used, safety_flags, memory_refs |
| Safety redirect | Passed | Diagnosis context → non-empty safety_flags + redirect message |

## Configuration Defaults Used

| Setting | Default | Notes |
|---------|---------|-------|
| `BAYMAX_DIALOGUE_BACKEND` | `rule_based` | No LLM backend configured |
| `BAYMAX_HF_CHAT_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Not downloaded in this env |
| `BAYMAX_OLLAMA_MODEL` | `""` | Not configured |
| `BAYMAX_DIALOGUE_TOP_K_MEMORIES` | `5` | Used in smoke test |
| `BAYMAX_DIALOGUE_TEMPERATURE` | `0.4` | Applied to LLM backends |

## Model Download Locations

No models were downloaded during this iteration. When configured:
- Ollama models: managed by Ollama daemon in its own storage
- HuggingFace models: cached to `BAYMAX_MODEL_DIR` (default `/scratch/$USER/bay-max/models`)

## Artifact Output Locations

- `./artifacts/smoke_test_005.json` — smoke test results (gitignored)

## Known Issues

1. SQLite store still uses synchronous `sqlite3` — not yet migrated to aiosqlite
2. Emotion estimator remains stubbed
3. Ollama and Transformers backends not verified with real model inference in this environment
4. LanceDB real backend not exercised in automated tests (StubVectorStore used)
5. Semantic fact extraction uses keyword heuristics — missed in both smoke test sessions above (0 semantic facts created from consolidation)
6. MediaPipe Pose uses legacy `mp.solutions.pose` API
7. Gradio demo uses sync-to-async ThreadPoolExecutor workaround

## Deviations from Prompt

1. `test_project_state_schema.py` already tests `>=8 endpoints`; now the project has 17 endpoints — test still passes.
2. Transformers provider is implemented but not exercised with a real model due to environment constraints — reported honestly as `not_run`.
3. `dialogue` extras in pyproject.toml include `accelerate` for device_map="auto" support.

## Risks

- LLM dialogue quality (tone, groundedness) is unverified without real model execution
- Ollama HTTP probing in `is_available()` adds a small latency bump on first check
- Safety pattern matching may miss paraphrased unsafe requests

## Open Questions

- Should the `check_safety()` patterns use fuzzy/semantic matching in a future iteration?
- Should LanceDB be tested with a real embedded backend in iteration-006?
- Is the Qwen/Qwen2.5-1.5B-Instruct model approved for use on Sol GPU?

## Next Recommended Tasks

1. Run `scripts/dialogue_smoke.py` with a real Ollama model to validate grounded recall
2. Migrate `SQLiteMetadataStore` to use `aiosqlite` throughout
3. Implement emotion estimation from face regions
4. Add LLM-based semantic fact extraction during consolidation
5. Verify LanceDB + SentenceTransformerEmbedder real round-trip
6. Add salience decay over time for episodic memories

## Doc Parity Check

| Document | Matches Code | Notes |
|----------|-------------|-------|
| README.md | Yes | Install table updated, dialogue section added |
| ROADMAP.md | Yes | Iterations 1-5 status accurate |
| ARCHITECTURE.md | Yes | 17 endpoints listed, dialogue flow documented |
| CHANGELOG.md | Yes | Entries for iterations 001-005 |
| MODEL_STACK.md | Yes | Dialogue backends section added |
| LOCAL_TESTING.md | Yes | Ollama + Transformers + smoke test commands |
| PROJECT_MEMORY.md | Yes | Updated for iteration-005 |
| project_state.json | Yes | iteration=005, 17 endpoints, known issues updated |
| DIALOGUE_ARCHITECTURE.md | Yes (new) | Matches implementation |
| SAFETY_GUARDRAILS.md | Yes (new) | Matches safety.py |
