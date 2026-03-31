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

## Iteration 007 - Webcam + TTS (Completed)
- [x] TTS provider abstraction (TTSProvider interface + factory)
- [x] Kokoro TTS backend (KokoroTTSProvider, Kokoro-82M, Apache-2.0)
- [x] Piper TTS backend (PiperTTSProvider, placeholder)
- [x] NullTTSProvider (silent fallback)
- [x] SpeechService with queued playback
- [x] LiveRuntime TTS integration (proactive responses spoken aloud)
- [x] GET /v1/tts/backends endpoint
- [x] LiveRuntimeStatus extended with speech fields
- [x] Verification script (scripts/verify_007.py)
- [x] New docs: TTS_ARCHITECTURE.md, LIVE_VERIFICATION_GUIDE.md
- [x] 40 new tests (296 total)

## Iteration 008 - Bidirectional Speech (Completed)
- [x] Speech input pipeline: Microphone -> Silero VAD -> faster-whisper ASR -> echo suppression
- [x] MicrophoneCapture (sounddevice) and NullMicrophone (headless stub)
- [x] Silero VAD with 4-state machine (IDLE/SPEECH_STARTED/SPEECH_ONGOING/SILENCE_AFTER_SPEECH)
- [x] FasterWhisperProvider (CTranslate2, base.en model) and NullASRProvider
- [x] EchoSuppressor (text-similarity based, SequenceMatcher)
- [x] SpeechInputService orchestrator with speaking lock and post-speech cooldown
- [x] Push-to-talk fallback mode alongside continuous VAD
- [x] Spoken turns integrated into session/memory flow (ChatTurn.source='speech')
- [x] Baymax-inspired companion persona style guide (docs/COMPANION_PERSONA.md)
- [x] Persona style injected into system prompt (prompt_builder.py)
- [x] GET /v1/audio/status and GET /v1/stt/backends endpoints
- [x] LiveRuntimeStatus extended with speech input fields
- [x] 60 new tests (356 total)

## Iteration 009 - Affect-Aware Companion (Completed)
- [x] Facial affect analysis module (perception/emotion.py, emotion_smoother.py)
- [x] MediaPipe Face Landmarker affect backend with blendshape-to-valence/arousal mapping
- [x] Null affect backend for graceful fallback
- [x] AffectSmoother with temporal smoothing and stability detection
- [x] Affect integration into LiveRuntime and state manager
- [x] Overlay display of valence/arousal/confidence in debug HUD
- [x] GET /v1/emotion/backends endpoint
- [x] 30 new tests (386 total)

## Iteration 010b - Companion UI (Completed)
- [x] Diagnostic web UI with Vite + React + TypeScript + Tailwind CSS
- [x] Three-column layout: camera feed, activity, memory panels
- [x] WebSocket telemetry endpoint (/ws/live) for real-time state streaming
- [x] HTTP endpoints for frame, UI state, text input, mic toggle, memory
- [x] Animated Bay-Max face SVG with pipeline state-driven animations
- [x] Color-coded activity feed with ring buffer and auto-scroll
- [x] Affect visualization with valence/arousal mini-plot
- [x] Memory panel with similarity scores
- [x] Text input and microphone controls
- [x] Backend-owned annotated frame preview
- [x] CORS middleware for local development
- [x] Developer workflow: ui-install, run-ui, run-full Makefile targets
- [x] New docs: COMPANION_UI_ARCHITECTURE.md, WEBSOCKET_PROTOCOL.md

## Iteration 013 - Friendly Voice Assistant Core (Current)
- [x] Single authoritative `AssistantVoiceState` enum for voice state machine
- [x] `AssistantStateMachine` class with explicit transitions
- [x] Activation modes: push_to_talk, continuous_vad, wake_phrase_gate
- [x] `ActivationGate` abstraction for future wake-word backend swapping
- [x] Wake phrase detection (ASR-based provisional implementation)
- [x] Follow-up window after speaking for natural conversation flow
- [x] Configuration settings for activation mode, wake phrases, follow-up window
- [x] State machine transition tests (43 tests)
- [x] New docs: VOICE_ASSISTANT_ARCHITECTURE.md
- [x] Updated README, ROADMAP, BACKLOG, PROJECT_BRIEF, project_state.json

## Future Iterations
- Dedicated wake-word engine integration (Porcupine, OpenWakeWord)
- Async SQLite migration (aiosqlite throughout)
- Multi-session user journey tracking
- Salience-decay over time for episodic memory
- Conversation history and context window management
- Speaker diarization for multi-person support
- MediaPipe Tasks API migration for pose estimation
- LLM-based semantic fact extraction during consolidation
- Proactive emotional check-ins (re-enable after voice core stable)
- Internal specialist agent delegation (behind single assistant boundary)
- Production deployment considerations
