# Bay-Max Project Memory

This file provides continuity context for AI agents working on the Bay-Max project. Update this file after completing significant work.

## Last Updated
2026-03-16 (Iteration 002)

## Current State
- **Iteration**: 002 (Vision & Identity)
- **Branch**: feature/iteration-002-vision-identity
- **Status**: Complete

## What Exists
- Full modular monolith structure under `src/baymax/`
- 8 modules: capture, perception, state, memory, planner, dialogue, orchestrator, config
- Real face detection via facenet-pytorch MTCNN
- Real face embeddings via InceptionResnetV1 (vggface2, 512-d vectors)
- Cosine similarity face recognition against enrolled embeddings
- Simple IoU + cosine face tracker for frame-to-frame continuity
- Face enrollment endpoint (upload image, detect 1 face, store embedding)
- Frame analysis endpoint (detect, embed, recognize, track, write observations)
- Recognition-aware InteractionState with identity_confidence, face_count, track IDs
- Meaningful-event-only observation writing (first_recognition, user_switch, known_to_unknown)
- SQLite tables: users, face_enrollments, face_embeddings, sessions, episodic_memories, semantic_facts, recognition_observations
- Pydantic v2 schemas: FaceBoundingBox, DetectedFace, RecognizedFace, UnknownFace, FaceEmbeddingRecord, FrameAnalysisResult, RecognitionObservation
- FastAPI backend with 10 endpoints
- Gradio demo with 5 tabs (Setup, Enroll Face, Frame Analysis, Interact, Memory)
- 65 tests all passing, ruff lint clean

## What Is Stubbed
- Frame source / webcam capture (returns empty)
- LanceDB vector store adapter (interface exists, stub in-memory implementation)
- Memory consolidation (returns empty)
- Engagement and emotion estimators (return hardcoded values)
- Dialogue is rule-based (no LLM)

## Key Decisions Made
- Modular monolith over microservices (ADR-0001)
- SQLite for metadata, vector store adapter for embeddings (ADR-0002)
- Vision-first MVP scope, rule-based dialogue for iteration-001 (ADR-0003)
- facenet-pytorch for face detection/embeddings (Iteration 002)
- Cosine similarity for face matching, threshold 0.75 default (Iteration 002)
- Simple IoU+cosine tracker instead of ByteTrack dependency (Iteration 002)
- Observation writing only on meaningful events to avoid noise (Iteration 002)

## Model Stack
- **Face Detection**: MTCNN (facenet-pytorch) - multi-task cascaded CNN
- **Face Embedding**: InceptionResnetV1 pretrained on vggface2 (512-d)
- **Face Matching**: Cosine similarity, configurable threshold (default 0.75)
- **Tracking**: Simple greedy IoU (0.4 weight) + cosine (0.6 weight) tracker
- **Dialogue**: Rule-based templates (no external API)

## Known Issues
- SQLite store uses synchronous sqlite3 despite being wrapped in async
- LanceDB vector store adapter not yet integrated
- Engagement/emotion estimators remain stubbed
- Gradio demo uses sync-to-async workaround

## For Next Agent
1. Read this file and `docs/project_state.json` first
2. Check `docs/BACKLOG.md` for pending work
3. The highest-impact next steps are: wire LanceDB, replace dialogue with Claude, add engagement/emotion estimation
4. All modules use interfaces; implement concrete classes without changing the interface contracts
5. The perception module now has real implementations alongside stubs - use FacenetDetector and CosineRecognizer for production, StubFaceDetector and StubFaceRecognizer for testing
