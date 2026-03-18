# Changelog

## [0.10.0] - 2026-03-18 (Iteration 010b)

### Added
- New `apps/ui/` frontend application: Vite + React + TypeScript + Tailwind CSS diagnostic UI
- Three-column diagnostic layout: "What Bay-Max Sees", "What Bay-Max is Doing", "What Bay-Max Remembers"
- WebSocket endpoint `/ws/live` for real-time telemetry streaming (snapshots + pipeline events)
- HTTP endpoints: `/v1/live/frame/latest`, `/v1/live/ui-state`, `/v1/live/text-input`, `/v1/live/mic/toggle`, `/v1/live/memory/recent`
- Animated Bay-Max face SVG component with state-driven blink/eyes/mouth animations
- Color-coded activity feed with 100-event ring buffer and auto-scroll
- Live camera feed from backend-rendered annotated frames
- Affect visualization with valence/arousal mini-plot
- Memory panel showing retrieved memories with similarity scores
- Semantic facts panel and memory stats
- Session info panel with state/duration/turn count
- Text input and microphone toggle controls
- CORS middleware for localhost development ports
- New Makefile targets: `ui-install`, `run-ui`, `ui-build`, `ui-test`, `run-full`
- Frontend tests using Vitest + Testing Library
- New docs: `docs/COMPANION_UI_ARCHITECTURE.md`, `docs/WEBSOCKET_PROTOCOL.md`

### Changed
- API version bumped from 0.8.1 to 0.10.0
- `apps/api/main.py` now includes live WebSocket and HTTP routers
- `LiveRuntime` now updates annotated frame cache for UI endpoint
- Makefile updated with UI workflow commands

## [0.7.0] - 2026-03-17 (Iteration 008)

### Added
- New `src/baymax/audio/` package: `schemas.py` (audio Pydantic schemas), `microphone.py` (MicrophoneCapture + NullMicrophone), `vad.py` (SileroVAD + NullVAD), `transcriber.py` (ASRProvider ABC + FasterWhisperProvider + NullASRProvider), `echo_suppression.py` (EchoSuppressor), `speech_input_service.py` (SpeechInputService orchestrator)
- Full speech input pipeline: Microphone -> Silero VAD -> faster-whisper ASR -> echo suppression -> dialogue
- Speaking lock: mic audio discarded while TTS is active + configurable post-speech cooldown (1500ms default)
- Push-to-talk mode as alternative to continuous VAD
- Spoken user turns stored with `ChatTurn.source='speech'` and routed through existing memory/dialogue pipeline
- `docs/COMPANION_PERSONA.md`: Baymax-inspired companion persona style guide
- Persona style injected into system prompt via `prompt_builder.py`
- `GET /v1/audio/status` endpoint returning speech input/output runtime status
- `GET /v1/stt/backends` endpoint returning STT backend configuration
- `LiveRuntimeStatus` extended with speech input fields (speech_input_enabled, listening, vad_active, stt_backend, speaking_lock_active, last_heard_text, transcription_latency_ms, mic_mode)
- New Makefile targets: `install-speech`, `live-webcam-speech`, `verify-008`
- 18 new environment variables under `BAYMAX_*` for speech input (see `.env.example`)
- Audio Pydantic schemas: `AudioChunkInfo`, `VADDecision`, `SpeechSegment`, `TranscriptionResult`, `EchoDecision`, `SpeechInputStatus`, `AudioBackendInfo`, `SpeechArtifactManifest`
- `speech` optional dependency group in pyproject.toml (faster-whisper, sounddevice, soundfile)
- 60 new tests (356 total)

### Changed
- API version bumped from 0.6.0 to 0.7.0
- `pyproject.toml` version bumped to 0.7.0
- `LiveRuntime` rewritten with speech input integration (parallel async tasks for mic loop + transcription consumer)
- `Orchestrator.add_turn()` accepts `source` parameter

## [0.6.0] - 2026-03-17 (Iteration 007)

### Added
- New `src/baymax/tts/` package: `provider.py` (TTSProvider interface + factory + NullTTSProvider), `kokoro_provider.py` (KokoroTTSProvider), `piper_provider.py` (PiperTTSProvider), `speech_service.py` (SpeechService with queued playback), `schemas.py` (TTS Pydantic schemas)
- LiveRuntime TTS integration: proactive responses are spoken aloud when TTS is enabled
- `GET /v1/tts/backends` endpoint returning available TTS backends and active configuration
- `LiveRuntimeStatus` extended with speech fields (tts_backend, tts_available, speech_queue_depth, total_utterances)
- Verification script `scripts/verify_007.py` for webcam + TTS end-to-end validation
- New Makefile targets: `live-webcam-tts`, `verify-007`
- 12 new environment variables under `BAYMAX_TTS_*` (see `.env.example`)
- TTS Pydantic schemas: `TTSBackendInfo`, `TTSRequest`, `TTSResult`, `SpeechQueueStatus`
- New docs: `docs/TTS_ARCHITECTURE.md`, `docs/LIVE_VERIFICATION_GUIDE.md`
- 40 new tests (296 total)

### Changed
- API version bumped from 0.5.0 to 0.6.0
- `live_artifact_dir` default changed to `live_run_007`
- `pyproject.toml` version bumped to 0.6.0

## [0.5.0] - 2026-03-16 (Iteration 006)

### Added
- New `src/baymax/live/` package: `frame_source`, `session_supervisor`, `event_engine`, `proactive_scheduler`, `overlay`, `artifact_logger`, `runtime`, `schemas`
- `OpenCVFrameSource`: webcam and video-file frame reader with FPS throttling (BGR→RGB)
- `FolderFrameSource`: sorted-image-folder replay source for headless testing
- `SessionSupervisor`: state machine managing session start, pause, resume, end based on presence/absence thresholds
- `EventEngine`: companion event detection (person_arrived, person_departed, recognized/unknown arrival, recognition gained/lost, posture/engagement change with stability gating, quiet_companionship_due)
- `ProactiveScheduler`: dual-layer cooldown system (global any-response interval + per-event-type interval) with suppression reasons
- `LiveRuntime`: async orchestrator loop wiring frame source → perception → session lifecycle → events → proactive responses → artifact logging
- `render_overlay()`: headless-safe HUD renderer using NumPy draws (session state, identity, posture, engagement, last event, last response, debug info)
- `ArtifactLogger`: JSONL event log, JSONL response log, session timeline JSON, manifest JSON, summary Markdown
- `scripts/live_companion.py`: CLI runner with `--replay`, `--max-frames`, `--max-run-sec`, `--no-overlay`, `--debug`, `--artifact-dir` flags
- `GET /v1/live/status`: returns `LiveRuntimeStatus` when live runtime is active
- `POST /v1/live/control`: accepts `start`/`stop` actions; start must be done via CLI runner
- 16 new environment variables under `BAYMAX_LIVE_*` (see `.env.example`)
- New Pydantic v2 schemas: `LiveRuntimeStatus`, `LiveFrameResult`, `SessionLifecycleEvent`, `CompanionEvent`, `ProactiveDecision`, `CooldownState`, `LiveArtifactRef`, `LiveRunSummary`
- New docs: `docs/LIVE_MODE_ARCHITECTURE.md`, `docs/EVENT_ENGINE.md`
- Makefile targets: `live-webcam`, `live-replay`
- 62 new tests (256 total)

### Changed
- API version bumped from 0.4.0 to 0.5.0
- `apps/api/main.py`: added live runtime integration hooks and live endpoints
- All required docs updated: README, ROADMAP, ARCHITECTURE, LOCAL_TESTING, PROJECT_MEMORY, CHANGELOG

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
