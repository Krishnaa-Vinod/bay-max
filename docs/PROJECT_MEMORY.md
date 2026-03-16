# Bay-Max Project Memory

This file provides continuity context for AI agents working on the Bay-Max project. Update this file after completing significant work.

## Last Updated
2026-03-16 (Iteration 005)

## Current State
- **Iteration**: 005 (Grounded Local Dialogue)
- **Branch**: feature/iteration-005-grounded-local-dialogue
- **Status**: Complete

## What Exists
- Full modular monolith structure under `src/baymax/`
- 8 modules: capture, perception, state, memory, planner, dialogue, orchestrator, config
- Real face detection via facenet-pytorch MTCNN
- Real face embeddings via InceptionResnetV1 (vggface2, 512-d vectors)
- Cosine similarity face recognition against enrolled embeddings
- Simple IoU + cosine face tracker for frame-to-frame continuity
- Pose estimation via MediaPipe Pose (33 landmarks, backend-agnostic interface)
- Body-state heuristics: posture, lean, motion classification from landmarks
- Engagement scoring (weighted 0.0-1.0 from face, pose, posture, lean, motion signals)
- Temporal smoothing for engagement and posture (majority-vote over configurable window)
- Body-state observations written to memory on meaningful transitions
- Annotated debug frame output (face boxes, pose skeleton, engagement labels)
- Local test CLI (`scripts/local_test.py`) for headless image analysis and frame replay
- **Text embedding adapter** with `TextEmbedder` ABC, `SentenceTransformerEmbedder` (all-MiniLM-L6-v2, 384-d), and `StubTextEmbedder`
- **LanceDB vector store** for semantic memory search (with `StubVectorStore` fallback)
- **Typed conversation turns** (`ChatTurn` with `TurnRole` user/system) for pre-audio dialogue
- **Session consolidation pipeline**: turns + observations → session summary → episodic memories → candidate semantic facts
- **Memory-aware responses**: top-k semantic search populates `memory_refs` and `state_summary` in `SupportiveResponse`
- **Memory inspection**: `MemorySummaryResponse` with counts, session summaries, recent episodic, confirmed facts
- **Memory correction**: confirm, reject, or update semantic facts via `CorrectionAction`
- SQLite tables: users, face_enrollments, face_embeddings, sessions, episodic_memories, semantic_facts, recognition_observations, body_state_observations, chat_turns, session_summaries
- Pydantic v2 schemas: full set including ChatTurn, SessionSummary, ConsolidationResult, MemoryHit, MemoryCorrectionRequest/Result, MemorySummaryResponse
- FastAPI backend with 17 endpoints (version 0.4.0)
- Gradio demo with 8 tabs (Setup, Enroll Face, Frame Analysis, Chat, Interact, Consolidate, Memory, Backends)
- 194 tests all passing, ruff lint clean

## What Is Stubbed
- Frame source / webcam capture (returns empty)
- Emotion estimator (returns hardcoded values)
- Dialogue LLM backends not verified with real models in this environment

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

## For Next Agent
1. Read this file and `docs/project_state.json` first
2. Check `docs/BACKLOG.md` for pending work
3. Dialogue backend is configurable via `BAYMAX_DIALOGUE_BACKEND` (rule_based/ollama/transformers)
4. The highest-impact next steps are: run a real local-LLM manual smoke test, migrate to aiosqlite, add emotion estimation, LLM-based fact extraction
5. All dialogue modules are in `src/baymax/dialogue/` with clean interfaces
6. The `scripts/dialogue_smoke.py` script covers all 5 manual test cases
7. All modules use interfaces; implement concrete classes without changing the interface contracts
8. Memory pipeline is fully functional: turns → consolidation → episodic/semantic → vector search → memory-aware responses
9. Install dialogue dependencies with `pip install -e ".[dialogue]"` for Transformers support
10. The default backend is always `rule_based`; LLM backends require explicit configuration
