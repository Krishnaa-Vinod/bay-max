# Bay-Max Roadmap

## Iteration 001 - Bootstrap (Completed)
- [x] Repository structure and Python project setup
- [x] Core schemas (user, session, perception, memory, response)
- [x] Settings management from environment variables
- [x] SQLite metadata store
- [x] Memory interfaces and retrieval
- [x] Rule-based dialogue provider
- [x] Orchestrator service
- [x] FastAPI with all required endpoints
- [x] Gradio demo shell
- [x] Documentation and ADRs
- [x] CI workflow

## Iteration 002 - Face Recognition (Completed)
- [x] OpenCV / NumPy image processing
- [x] Face detection with MTCNN model (facenet-pytorch)
- [x] Face embedding with InceptionResnetV1 (VGGFace2, 512-d)
- [x] Cosine similarity face recognition with configurable threshold
- [x] IoU + cosine face tracker for frame-to-frame continuity
- [x] Face enrollment and recognition pipeline
- [x] Comprehensive test suite (65 tests)

## Iteration 003 - Pose Estimation & Engagement (Completed)
- [x] MediaPipe Pose backend integration (33 landmarks)
- [x] Body-state heuristics (posture, lean, motion)
- [x] Engagement scoring from multiple signals (weighted 0.0–1.0)
- [x] Body-state observation events
- [x] Annotated debug output visualization
- [x] Local testing CLI (`scripts/local_test.py`)
- [x] 44 new tests (109 total)

## Iteration 004 - Memory Depth (Completed)
- [x] Typed conversation turns (ChatTurn, TurnRole)
- [x] Session consolidation pipeline (turns + observations → episodic → semantic)
- [x] Text embedding adapter (SentenceTransformer all-MiniLM-L6-v2, 384-d)
- [x] LanceDB vector store for semantic memory search
- [x] Temporal smoothing for engagement and posture (majority-vote)
- [x] Memory-aware responses (memory_refs + state_summary in SupportiveResponse)
- [x] Memory summary, semantic query, and correction endpoints
- [x] 49 new tests (158 total)

## Iteration 005 - Grounded Local Dialogue (Completed)
- [x] Dialogue provider interface extended for grounded backends
- [x] Rule-based fallback preserved and functional
- [x] Ollama-compatible local dialogue backend (configurable model)
- [x] HuggingFace Transformers local dialogue backend
- [x] Provider factory with graceful fallback on init failure
- [x] Grounded prompt builder (plan-then-verbalize pattern)
- [x] Safety gating: diagnosis-style and medical certainty requests redirected
- [x] Fallback from LLM backend to rule-based on generate() failure
- [x] GET /v1/dialogue/backends endpoint
- [x] /v1/respond returns backend, model_name, fallback_used, safety_flags
- [x] New schemas: DialogueBackendInfo, GroundedPromptContext, SafetyDecision, etc.
- [x] Gradio demo: Backends tab + debug fields in Interact tab
- [x] Smoke test script (`scripts/dialogue_smoke.py`)
- [x] All docs updated, 36 new tests (194 total)

## Iteration 006 - Live Webcam Continuity (Completed)
- [x] Continuous frame source abstraction (webcam + video replay + folder replay)
- [x] Automatic session lifecycle: start, pause, resume, end based on presence/absence
- [x] Event engine: arrival, departure, recognition gained/lost, posture/engagement change, quiet companionship
- [x] Proactive response scheduler with global and per-event-type cooldowns
- [x] Memory-grounded proactive responses via existing planner+dialogue stack
- [x] Live overlay renderer (headless-safe, NumPy-based)
- [x] ArtifactLogger: JSONL event log, response log, session timeline, manifest, summary
- [x] CLI runner `scripts/live_companion.py` for webcam and replay modes
- [x] `GET /v1/live/status` and `POST /v1/live/control` API endpoints
- [x] New docs: LIVE_MODE_ARCHITECTURE.md, EVENT_ENGINE.md
- [x] 62 new tests (256 total)

## Future Iterations
- Async SQLite migration (aiosqlite throughout)
- Emotion recognition from facial expressions
- Multi-session user journey tracking
- Salience-decay over time for episodic memory
- Conversation history and context window management
- Audio integration (Whisper STT, TTS)
- MediaPipe Tasks API migration for pose estimation
- LLM-based semantic fact extraction during consolidation
- Gradio live inspection tab for live-mode state
- Production deployment considerations
