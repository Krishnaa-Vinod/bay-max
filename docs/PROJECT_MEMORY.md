# Bay-Max Project Memory

This file provides continuity context for AI agents working on the Bay-Max project. Update this file after completing significant work.

## Last Updated
2026-03-16 (Iteration 003)

## Current State
- **Iteration**: 003 (Pose Estimation & Engagement Heuristics)
- **Branch**: feature/iteration-003-pose-engagement-local-eval
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
- Body-state observations written to memory on meaningful transitions
- Annotated debug frame output (face boxes, pose skeleton, engagement labels)
- Local test CLI (`scripts/local_test.py`) for headless image analysis and frame replay
- Face enrollment endpoint (upload image, detect 1 face, store embedding)
- Frame analysis endpoint (detect, embed, recognize, track, pose, engage, write observations)
- Recognition-aware InteractionState with identity_confidence, face_count, track IDs
- Body-state InteractionState with pose_visible, posture, lean, motion, engagement_score
- SQLite tables: users, face_enrollments, face_embeddings, sessions, episodic_memories, semantic_facts, recognition_observations
- Pydantic v2 schemas: full set including PoseLandmark2D, PoseResult, EngagementResult, AnnotatedArtifactRef, BodyStateObservation
- FastAPI backend with 10 endpoints
- Gradio demo with 5 tabs (Setup, Enroll Face, Frame Analysis with annotated overlay, Interact, Memory)
- 109 tests all passing, ruff lint clean

## What Is Stubbed
- Frame source / webcam capture (returns empty)
- LanceDB vector store adapter (interface exists, stub in-memory implementation)
- Memory consolidation (returns empty)
- Emotion estimator (returns hardcoded values)
- Dialogue is rule-based (no LLM)

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

## Model Stack
- **Face Detection**: MTCNN (facenet-pytorch) - multi-task cascaded CNN
- **Face Embedding**: InceptionResnetV1 pretrained on vggface2 (512-d)
- **Face Matching**: Cosine similarity, configurable threshold (default 0.75)
- **Tracking**: Simple greedy IoU (0.4 weight) + cosine (0.6 weight) tracker
- **Pose Estimation**: MediaPipe Pose (33 landmarks, legacy mp.solutions.pose API)
- **Engagement**: Rule-based weighted scoring (see docs/ENGAGEMENT_HEURISTICS.md)
- **Dialogue**: Rule-based templates (no external API)

## Known Issues
- SQLite store uses synchronous sqlite3 despite being wrapped in async
- LanceDB vector store adapter not yet integrated
- Emotion estimator remains stubbed
- Gradio demo uses sync-to-async workaround
- MediaPipe Pose uses legacy API (not the newer Tasks API)

## For Next Agent
1. Read this file and `docs/project_state.json` first
2. Check `docs/BACKLOG.md` for pending work
3. The highest-impact next steps are: wire LanceDB, replace dialogue with Claude, add emotion estimation
4. All modules use interfaces; implement concrete classes without changing the interface contracts
5. The perception module now has real implementations alongside stubs - use FacenetDetector and CosineRecognizer for face, MediaPipePoseEstimator for pose
6. Engagement is now real (not stubbed) - uses transparent heuristics documented in `docs/ENGAGEMENT_HEURISTICS.md`
7. Test with `scripts/local_test.py` for quick validation of changes
