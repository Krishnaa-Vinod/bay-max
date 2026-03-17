# Bay-Max Project Memory

This file provides continuity context for AI agents working on the Bay-Max project. Update this file after completing significant work.

## Last Updated
2026-03-16 (Iteration 006)

## Current State
- **Iteration**: 006 (Live Webcam Continuity)
- **Branch**: feature/iteration-006-live-webcam-continuity
- **Status**: Complete

## What Exists
- Full modular monolith structure under `src/baymax/`
- 9 modules: capture(stub), perception, state, memory, planner, dialogue, orchestrator, config, **live**
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
- FastAPI backend with 19 endpoints (version 0.5.0)
- CLI runner `scripts/live_companion.py` for webcam and folder/video replay
- 256 tests all passing, ruff lint clean

## What Is Stubbed
- Frame source / webcam capture via `capture` module (still returns empty; live mode uses `live/frame_source.py` directly)
- Emotion estimator (returns hardcoded values)
- Live webcam not exercised on this headless machine — replay/folder mode verified via automated tests

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

## Known Issues
- SQLite store uses synchronous sqlite3 despite being wrapped in async
- Emotion estimator remains stubbed
- Gradio demo uses sync-to-async workaround
- MediaPipe Pose uses legacy API (not the newer Tasks API)
- LanceDB vector store requires separate install (pip install baymax[vector])
- Semantic fact extraction uses simple heuristics, not LLM-based
- Local LLM backends not verified with real models on this machine (environment lacks Ollama/models)
- Live webcam mode implemented but not exercised on this headless machine (Sol); replay mode verified

## For Next Agent
1. Read this file and `docs/project_state.json` first
2. The live runtime is in `src/baymax/live/` — start with `runtime.py` for the orchestration loop
3. Replay mode uses `FolderFrameSource` or `OpenCVFrameSource(str)` — webcam uses `OpenCVFrameSource(int)`
4. Live mode is launched via `scripts/live_companion.py` — see `make live-webcam` and `make live-replay`
5. Session lifecycle logic is in `live/session_supervisor.py`; event detection in `live/event_engine.py`
6. Proactive response decisions are in `live/proactive_scheduler.py`
7. Artifacts are written to `BAYMAX_LIVE_ARTIFACT_DIR` (default: `./artifacts/live_run_006`)
8. API endpoints `/v1/live/status` and `/v1/live/control` are in `apps/api/main.py`
9. All live settings have `BAYMAX_LIVE_*` env var prefixes (see `.env.example`)
10. The default backend is always `rule_based`; LLM backends require explicit configuration
