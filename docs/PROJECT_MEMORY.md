# Bay-Max Project Memory

This file provides continuity context for AI agents working on the Bay-Max project. Update this file after completing significant work.

## Last Updated
2026-03-18 (Iteration 010b)

## Current State
- **Iteration**: 010b (Companion UI)
- **Branch**: feature/iteration-010b-companion-ui
- **Status**: Complete

## What Exists
- Full modular monolith structure under `src/baymax/`
- 10 modules: capture(stub), perception, state, memory, planner, dialogue, orchestrator, config, live, **tts**, **audio**
- Real face detection via facenet-pytorch MTCNN
- Real face embeddings via InceptionResnetV1 (vggface2, 512-d vectors)
- Cosine similarity face recognition against enrolled embeddings
- Simple IoU + cosine face tracker for frame-to-frame continuity
- Pose estimation via MediaPipe Pose (33 landmarks, legacy mp.solutions.pose API)
- Body-state heuristics: posture, lean, motion classification from landmarks
- Engagement scoring (weighted 0.0-1.0 from face, pose, posture, lean, motion signals)
- Temporal smoothing for engagement and posture (majority-vote over configurable window)
- Body-state observations written to memory on meaningful transitions
- Annotated debug frame output (face boxes, pose skeleton, engagement labels)
- Text embedding adapter (SentenceTransformerEmbedder, all-MiniLM-L6-v2, 384-d)
- LanceDB vector store for semantic memory search
- Typed conversation turns (ChatTurn, TurnRole) for pre-audio dialogue
- Session consolidation pipeline: turns + observations → episodic → semantic → vector search
- Memory-aware responses: top-k semantic search populates memory_refs in SupportiveResponse
- Memory correction: confirm, reject, or update semantic facts
- Plan-then-verbalize dialogue pattern with safety gating
- Ollama and Transformers local dialogue backends with rule-based fallback
- **[NEW] Live runtime module** (`src/baymax/live/`):
  - `OpenCVFrameSource` / `FolderFrameSource` for webcam and replay
  - `SessionSupervisor`: presence-based session lifecycle (start/pause/resume/end)
  - `EventEngine`: companion event detection with stability gating and jitter prevention
  - `ProactiveScheduler`: dual-layer cooldown (global + per-event-type)
  - `LiveRuntime`: async orchestrator loop
  - `render_overlay()`: headless-safe HUD renderer
  - `ArtifactLogger`: JSONL event/response logs, timeline, manifest, summary
- **[NEW] TTS module** (`src/baymax/tts/`):
  - Kokoro TTS default backend (KokoroTTSProvider, Kokoro-82M, Apache-2.0)
  - SpeechService with queued playback
  - LiveRuntime TTS integration (proactive responses spoken aloud)
  - PiperTTSProvider (placeholder), NullTTSProvider (silent fallback)
- **[NEW] Audio/Speech Input module** (`src/baymax/audio/`):
  - MicrophoneCapture (sounddevice) + NullMicrophone (headless stub)
  - SileroVAD: 4-state machine (IDLE/SPEECH_STARTED/SPEECH_ONGOING/SILENCE_AFTER_SPEECH)
  - FasterWhisperProvider: CTranslate2-based ASR (base.en default) + NullASRProvider
  - EchoSuppressor: text-similarity (SequenceMatcher) against recent TTS output
  - SpeechInputService: orchestrates mic -> VAD -> ASR -> echo check -> output queue
  - Speaking lock + post-speech cooldown to prevent echo loops
  - Push-to-talk fallback alongside continuous VAD
  - Spoken turns stored as ChatTurn with source='speech'
- **[NEW] Companion Persona** (`docs/COMPANION_PERSONA.md`):
  - Baymax-inspired calm, literal, gentle, nonjudgmental tone
  - Injected into system prompt via `_PERSONA_STYLE` in prompt_builder.py
- **[NEW] Companion UI** (`apps/ui/`):
  - Vite + React + TypeScript + Tailwind CSS diagnostic web UI
  - Three-column layout: What Bay-Max Sees / What Bay-Max is Doing / What Bay-Max Remembers
  - WebSocket telemetry client (`/ws/live`) for real-time runtime state and events
  - Backend-owned annotated frame preview from `/v1/live/frame/latest`
  - Animated Bay-Max face SVG with state-driven blink/eyes/mouth
  - Color-coded activity feed (100-event ring buffer)
  - Affect visualization with valence/arousal mini-plot
  - Memory panel with similarity scores and semantic facts
  - Text input and microphone toggle controls
  - TypeScript types and API client for all telemetry messages
- FastAPI backend with 29 endpoints (version 0.10.0)
- New WebSocket and HTTP endpoints in `apps/api/live_ws.py` and `apps/api/live_http.py`
- CLI runner `scripts/live_companion.py` for webcam, replay, TTS, and speech input modes
- 386 tests all passing, ruff lint clean

## What Is Stubbed
- Frame source / webcam capture via `capture` module (still returns empty; live mode uses `live/frame_source.py` directly)
- Emotion estimator (returns hardcoded values)
- Live webcam not exercised on this headless machine — replay/folder mode verified via automated tests
- Real microphone/speaker speech input not exercised on Sol (headless HPC) — NullMicrophone/NullVAD stubs used in tests

## Key Decisions Made
- Modular monolith over microservices (ADR-0001)
- SQLite for metadata, vector store adapter for embeddings (ADR-0002)
- Vision-first MVP scope, rule-based dialogue for iteration-001 (ADR-0003)
- facenet-pytorch for face detection/embeddings (Iteration 002)
- Cosine similarity for face matching, threshold 0.75 default (Iteration 002)
- Simple IoU+cosine tracker instead of ByteTrack dependency (Iteration 002)
- Observation writing only on meaningful events to avoid noise (Iterations 002-003)
- MediaPipe Pose for body pose estimation with stub fallback (Iteration 003)
- Backend-agnostic PoseEstimator interface for future RTMPose swap (Iteration 003)
- Simple weighted scoring for engagement (transparent, documented thresholds) (Iteration 003)
- sentence-transformers/all-MiniLM-L6-v2 for text memory embeddings (384-d) (Iteration 004)
- LanceDB for local vector retrieval (no server dependency) (Iteration 004)
- Majority-vote temporal smoothing over configurable window (Iteration 004)
- Simple heuristic semantic fact extraction (preference signals in user turns) (Iteration 004)
- Separate episodic from semantic with confidence and status tracking (Iteration 004)
- Plan-then-verbalize dialogue pattern: planner picks strategy, LLM verbalizes within constraints (Iteration 005)
- Ollama as primary local-LLM backend (easiest local testing, no extra Python deps) (Iteration 005)
- TransformersDialogueProvider as secondary backend, lazy model loading (Iteration 005)
- Safety gating on both user input and LLM output; rule-based fallback on backend failure (Iteration 005)
- All backends configurable via env vars; rule_based is default (Iteration 005)
- **Live mode uses `live/frame_source.py` directly rather than the stub `capture` module** (Iteration 006)
- Separate preview FPS from analysis cadence to avoid running heavy inference every frame (Iteration 006)
- Event-driven proactive responses, not per-frame chatting — cooldowns prevent spam (Iteration 006)
- Stability delay in EventEngine prevents jitter-triggered events for posture/engagement (Iteration 006)
- Headless-safe design throughout: cv2.imshow wrapped in try/except, NumPy overlay, folder replay (Iteration 006)
- Kokoro-82M as default TTS backend (open-weight, Apache-2.0, lightweight) (Iteration 007)
- faster-whisper as default ASR backend (CTranslate2, better local performance than openai-whisper) (Iteration 008)
- Silero VAD for speech activity detection (MIT, lightweight LSTM) (Iteration 008)
- Text-based echo suppression over acoustic echo cancellation (simpler, no DSP dependency) (Iteration 008)
- Speaking lock + cooldown instead of barge-in (simpler, avoids partial utterance handling) (Iteration 008)
- Push-to-talk as debugging fallback, not primary mode (Iteration 008)
- Persona style injected into system prompt, not into template strings (Iteration 008)

## Model Stack
- **Face Detection**: MTCNN (facenet-pytorch) - multi-task cascaded CNN
- **Face Embedding**: InceptionResnetV1 pretrained on vggface2 (512-d)
- **Face Matching**: Cosine similarity, configurable threshold (default 0.75)
- **Tracking**: Simple greedy IoU (0.4 weight) + cosine (0.6 weight) tracker
- **Pose Estimation**: MediaPipe Pose (33 landmarks, legacy mp.solutions.pose API)
- **Engagement**: Rule-based weighted scoring (see docs/ENGAGEMENT_HEURISTICS.md)
- **Text Embedding**: sentence-transformers/all-MiniLM-L6-v2 (384-d)
- **Vector Search**: LanceDB (local, serverless)
- **Dialogue**: Configurable — rule_based (default), Ollama (local server), or Transformers (local model)
- **Dialogue Safety**: Input check + output check for diagnosis-style requests
- **TTS**: Kokoro-82M (Apache-2.0, 24kHz, af_heart voice default)
- **VAD**: Silero VAD (MIT, LSTM-based, 16kHz, threshold 0.65)
- **ASR**: faster-whisper (CTranslate2, base.en, lazy-loaded)
- **Echo Suppression**: Text similarity (SequenceMatcher, threshold 0.85)
- **Companion Persona**: Baymax-inspired calm/literal/gentle tone (injected via prompt_builder.py)

## Known Issues
- SQLite store uses synchronous sqlite3 despite being wrapped in async
- Emotion estimator remains stubbed
- Gradio demo uses sync-to-async workaround
- MediaPipe Pose uses legacy API (not the newer Tasks API)
- LanceDB vector store requires separate install (pip install baymax[vector])
- Semantic fact extraction uses simple heuristics, not LLM-based
- Local LLM backends not verified with real models on this machine (environment lacks Ollama/models)
- Live webcam mode implemented but not exercised on this headless machine (Sol); replay mode verified
- PortAudio may not be installed on headless machines (required for TTS audio playback)
- Piper TTS backend is a placeholder (not yet fully implemented)
- No real speech was heard or transcribed on Sol — only automated/mock tests ran
- Silero VAD threshold (0.65) may need tuning for different microphone hardware
- Post-speech cooldown (1500ms) may need tuning for different room acoustics

## For Next Agent
1. Read this file and `docs/project_state.json` first
2. The live runtime is in `src/baymax/live/` -- start with `runtime.py` for the orchestration loop
3. Replay mode uses `FolderFrameSource` or `OpenCVFrameSource(str)` -- webcam uses `OpenCVFrameSource(int)`
4. Live mode is launched via `scripts/live_companion.py` -- see `make live-webcam` and `make live-replay`
5. Session lifecycle logic is in `live/session_supervisor.py`; event detection in `live/event_engine.py`
6. Proactive response decisions are in `live/proactive_scheduler.py`
7. Artifacts are written to `BAYMAX_LIVE_ARTIFACT_DIR` (default: `./artifacts/live_run_007`)
8. API endpoints `/v1/live/status` and `/v1/live/control` are in `apps/api/main.py`
9. All live settings have `BAYMAX_LIVE_*` env var prefixes (see `.env.example`)
10. The default backend is always `rule_based`; LLM backends require explicit configuration
11. TTS module is in `src/baymax/tts/` -- `provider.py` has the interface and factory, `kokoro_provider.py` has the Kokoro backend
12. SpeechService (`tts/speech_service.py`) manages queued playback; integrated into LiveRuntime
13. `GET /v1/tts/backends` returns available TTS backends and active configuration
14. TTS is enabled by default with Kokoro; set `BAYMAX_TTS_BACKEND=null` to disable
15. Install TTS dependencies with `pip install -e ".[tts]"`
16. Audio/speech input module is in `src/baymax/audio/` -- `speech_input_service.py` is the orchestrator
17. SpeechInputService manages: mic -> VAD -> ASR -> echo suppression -> output queue
18. Speaking lock is managed by SpeechInputService; LiveRuntime sets it during TTS playback
19. `GET /v1/audio/status` returns speech input/output runtime status
20. `GET /v1/stt/backends` returns STT backend configuration
21. Install speech input dependencies with `pip install -e ".[speech]"`
22. Companion persona style is in `docs/COMPANION_PERSONA.md` and injected via `dialogue/prompt_builder.py`
23. ChatTurn.source field distinguishes typed vs spoken turns ('typed' default, 'speech' for mic input)
24. Affect analysis module is in `src/baymax/perception/emotion.py` and `emotion_smoother.py`
25. Affect is integrated into LiveRuntime via `_run_affect_analysis()` method
26. Companion UI is in `apps/ui/` -- React + TypeScript + Tailwind CSS
27. WebSocket telemetry is in `apps/api/live_ws.py`; HTTP endpoints in `apps/api/live_http.py`
28. UI connects via WebSocket to `/ws/live` for real-time snapshots and events
29. UI fetches annotated frames from `/v1/live/frame/latest`
30. Run full stack with: Terminal 1: `make run-api`, Terminal 2: `make live-webcam`, Terminal 3: `make run-ui`
31. Frontend uses TypeScript types in `apps/ui/src/types/live.ts` for all WebSocket messages
32. See `docs/WEBSOCKET_PROTOCOL.md` for full telemetry message schema
33. See `docs/COMPANION_UI_ARCHITECTURE.md` for UI component architecture
