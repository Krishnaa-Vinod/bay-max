# Changelog

## [0.4.0] - 2026-03-16 (Iteration 005)

### Added
- Dialogue provider interface extended with `prompt_context` parameter and `backend_name`/`model_name`/`is_available()` methods
- `OllamaDialogueProvider`: connects to a locally running Ollama server via HTTP `/api/chat`
- `TransformersDialogueProvider`: loads a HuggingFace chat model directly in-process (recommended: Qwen/Qwen2.5-1.5B-Instruct)
- `create_dialogue_provider()` factory with graceful fallback to `RuleBasedDialogue` on init failure
- `GroundedPromptBuilder` (`dialogue/prompt_builder.py`): packages state, memories, recent turns, safety rules into a structured prompt using plan-then-verbalize pattern
- Safety gating (`dialogue/safety.py`): `check_safety()` detects diagnosis-style requests on input; `check_output_safety()` validates LLM outputs
- `GET /v1/dialogue/backends` endpoint returning available backends and active configuration
- `SupportiveResponse` extended with `backend`, `model_name`, `fallback_used`, `safety_flags` fields
- New schemas: `DialogueBackendInfo`, `GroundedPromptContext`, `SafetyDecision`, `DialogueDebugTrace`, `GroundedResponse`, `DialogueSmokeTestResult`
- New settings: `BAYMAX_DIALOGUE_BACKEND`, `BAYMAX_ENABLE_DIALOGUE_DEBUG`, `BAYMAX_ENABLE_SAFE_HEALTH_MODE`, `BAYMAX_DIALOGUE_MAX_HISTORY_TURNS`, `BAYMAX_DIALOGUE_TOP_K_MEMORIES`, `BAYMAX_DIALOGUE_TEMPERATURE`, `BAYMAX_DIALOGUE_MAX_NEW_TOKENS`, `BAYMAX_OLLAMA_BASE_URL`, `BAYMAX_OLLAMA_MODEL`, `BAYMAX_HF_CHAT_MODEL`, `BAYMAX_HF_DTYPE`, `BAYMAX_ENABLE_RULE_BASED_FALLBACK`
- Smoke test script (`scripts/dialogue_smoke.py`) with 5 test cases
- Gradio demo: new Backends tab + debug fields in Interact tab response
- `install-dialogue` Makefile target and `dialogue` extras in pyproject.toml
- New docs: `DIALOGUE_ARCHITECTURE.md`, `SAFETY_GUARDRAILS.md`
- 36 new tests (194 total)

### Changed
- `pyproject.toml` version bumped to 0.4.0; API version bumped to 0.4.0
- `orchestrator/service.py`: `respond()` now builds grounded prompt context, applies safety check, routes to configured backend, handles fallback
- All required docs updated: README, ROADMAP, ARCHITECTURE, MODEL_STACK, LOCAL_TESTING, PROJECT_MEMORY, CHANGELOG

## [0.3.0] - 2026-03-16 (Iteration 004)

### Added
- Text embedding adapter (`TextEmbedder` ABC, `SentenceTransformerEmbedder`, `StubTextEmbedder`)
- LanceDB vector store for semantic memory search (`LanceDBVectorStore` + `StubVectorStore`)
- Typed conversation turns (`ChatTurn`, `TurnRole` enum, `store_chat_turn`, `get_chat_turns`)
- Session consolidation pipeline: turns + observations → `SessionSummary` → episodic memories → candidate semantic facts
- Temporal smoothing (majority-vote, configurable window) for engagement and posture
- Memory-aware responses: `memory_refs` and `state_summary` added to `SupportiveResponse`
- `MemorySummaryResponse`, `MemoryHit`, `ConsolidationResult`, `MemoryCorrectionRequest/Result`, `MemoryEmbeddingRecord`, `SessionSummary` schemas
- Memory correction: confirm, reject, or update semantic facts
- New API endpoints: `/v1/sessions/{id}/turns`, `/v1/sessions/{id}/consolidate`, `/v1/memory/summary/{user_id}`, `/v1/memory/search`, `/v1/memory/correct`
- 7-tab Gradio demo (Chat, Consolidate, Memory tabs added)
- 49 new tests (158 total)

### Changed
- API version bumped to 0.3.0
- `BaymaxSettings` extended with vector/embedding/memory/smoothing settings

## [0.2.1] - 2026-03-16 (Iteration 003)

### Added
- Pose estimation via MediaPipe Pose (33 landmarks, backend-agnostic interface)
- Body-state heuristics: posture (upright/slouched/reclined), lean (forward/neutral/backward), motion (low/medium/high)
- Engagement scoring (weighted 0.0–1.0 from face, pose, posture, lean, motion signals)
- Body-state observation writing on meaningful transitions
- Annotated debug frame output
- Local test CLI script (`scripts/local_test.py`)
- New Pydantic schemas: PoseLandmark2D, PoseResult, EngagementResult, AnnotatedArtifactRef, BodyStateObservation
- New enums: PostureLabel, LeanLabel, MotionLevel
- 44 new tests (109 total)

## [0.2.0] - 2026-03-16 (Iteration 002)

### Added
- Real face detection using MTCNN (facenet-pytorch)
- Real face embeddings using InceptionResnetV1 pretrained on VGGFace2 (512-d vectors)
- Cosine similarity face recognition against enrolled embeddings (threshold 0.75)
- Simple IoU + cosine face tracker for frame-to-frame continuity
- Face enrollment and frame analysis endpoints
- 33 new tests (65 total)

## [0.1.0] - 2026-03-16 (Iteration 001)

### Added
- Initial repository structure, Python project setup
- Core enums, Pydantic v2 schemas, settings, SQLite store
- Stub perception, rule-based dialogue, orchestrator
- FastAPI application with 8 endpoints, Gradio demo shell
- Architecture documentation and 3 ADRs
- CI workflow, Makefile
