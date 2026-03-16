# Iteration 002 Report: Vision & Identity

## Branch
`feature/iteration-002-vision-identity`

## Summary
Replaced all stub perception components with a real local face recognition pipeline using facenet-pytorch. The system can now detect faces (MTCNN), compute 512-d embeddings (InceptionResnetV1-vggface2), recognize enrolled users via cosine similarity, track faces across frames, and write meaningful observations to memory. 65 tests pass, ruff lint clean, 10 API endpoints functional.

## Tasks Completed

### T201: Audit Iteration 001 Contracts
- Read all existing interfaces, schemas, and orchestrator code
- Identified 12 gaps between iteration 001 stubs and iteration 002 requirements
- Confirmed interface-first approach: new concrete classes implement existing ABCs

### T202: Real Face Detection and Embedding
- Created `FacenetDetector` using MTCNN for detection and InceptionResnetV1 for embedding
- Produces `DetectedFace` objects with bounding boxes, confidence scores, and 512-d embeddings
- Lazy model loading with graceful fallback to `StubFaceDetector` on failure
- Updated `FaceDetector` and `FaceRecognizer` ABCs with properly typed signatures

### T203: Real Face Enrollment
- `enroll_face(user_id, image)` detects exactly 1 face, extracts embedding, stores in SQLite
- Validates single-face constraint (rejects 0 or >1 faces)
- Returns `FaceEnrollment` with model name, face count, status

### T204: Frame Analysis with Recognition
- `analyze_frame(session_id, frame)` runs full pipeline: detect -> track -> recognize -> update state
- Matches detected embeddings against all enrolled embeddings
- Updates `InteractionState` with identity, confidence, face count, track IDs
- Returns `FrameAnalysisResult` with counts and observation details

### T205: Simple IoU + Cosine Tracker
- Created `SimpleTracker` with greedy assignment (0.4 IoU weight + 0.6 cosine weight)
- `TrackedFace` dataclass with track_id, bbox, embedding, user_id, frames_seen
- Tracks expire after 5 seconds of inactivity
- No external dependencies (no ByteTrack)

### T206: Recognition Observations
- Writes observations only on meaningful events: `first_recognition`, `user_switch`, `known_to_unknown`
- Persisted to `recognition_observations` table in SQLite
- Each observation includes session_id, user_id, event_type, content, confidence, evidence_refs

### T207: Gradio Demo Upgrade
- Added "Enroll Face" tab: user_id input + image upload -> enrollment result
- Added "Frame Analysis" tab: session_id input + image upload -> analysis result JSON
- 5 tabs total: Setup, Enroll Face, Frame Analysis, Interact, Memory

### T208: Tests, Lint, and Docs
- 33 new tests in `tests/test_iteration_002.py`
- 65 total tests passing (32 existing + 33 new)
- Ruff lint clean (0 errors)
- Created `docs/MODEL_STACK.md`
- Updated `docs/PROJECT_MEMORY.md` and `docs/project_state.json`

## New Schemas
- `FaceBoundingBox` (x1, y1, x2, y2, computed width/height)
- `DetectedFace` (bbox, detection_confidence, embedding, backend)
- `RecognizedFace` (user_id, display_name, bbox, match_confidence)
- `UnknownFace` (bbox, detection_confidence, best_match_score)
- `FaceEmbeddingRecord` (user_id, embedding, model_name, created_at)
- `FrameAnalysisResult` (session_id, faces_detected, recognized/unknown lists, observations_written)
- `RecognitionObservation` (session_id, user_id, event_type, content, confidence, evidence_refs)

## New SQLite Tables
- `face_embeddings` (id, user_id, embedding_json, model_name, created_at)
- `recognition_observations` (id, session_id, user_id, event_type, content, confidence, evidence_refs_json, created_at)

## API Changes
- `GET /v1/users` — list all registered users (new)
- `POST /v1/users/{user_id}/enroll/face` — now accepts image upload via `UploadFile` (was stub)
- `POST /v1/sessions/{session_id}/frames` — frame analysis with recognition (new)
- App version bumped to 0.2.0

## Test Results
- **Command**: `pytest tests/ -v`
- **Result**: 65 passed, 0 failed
- **Test breakdown**:
  - `test_iteration_002.py`: 33 tests (schemas, tracker, recognizer, state, SQLite, stubs)
  - `test_health.py`: 5 tests
  - `test_memory_models.py`: 21 tests
  - `test_project_state_schema.py`: 6 tests

## What Was Actually Tested
- Unit tests for all new Pydantic schemas (creation, computed fields, defaults)
- Unit tests for `_compute_iou` and `_cosine_similarity` helper functions
- Unit tests for `SimpleTracker` (create, maintain, expire, set_user_id, clear)
- Unit tests for `CosineRecognizer` (empty enrolled, match/no-match, best-of-multiple)
- Unit tests for `StateManager.update_recognition` (state transitions, defaults)
- Integration tests for `SQLiteMetadataStore` (list_users, face embeddings CRUD, observations, enrollments)
- Stub behavior verification (StubFaceDetector returns empty, StubFaceRecognizer returns None)

## What Was NOT Tested
- End-to-end pipeline with real MTCNN model loading and actual images (would require GPU or large CPU inference)
- Frame analysis through the API endpoint with real image data
- Gradio demo end-to-end
- Multi-face tracking across many frames
- Performance/latency of inference pipeline

## Lint Results
- **Command**: `ruff check src/ apps/ tests/`
- **Result**: All checks passed

## Known Issues
1. SQLite store uses synchronous `sqlite3` despite async method signatures
2. LanceDB vector store adapter not yet integrated (interface exists, stub in use)
3. Engagement and emotion estimators remain stubbed (return hardcoded values)
4. Gradio demo uses sync-to-async workaround
5. No end-to-end test with real model inference (unit tests use synthetic embeddings)

## Deviations from Prompt
- None significant. All 8 task areas implemented as specified.

## Risks
- Real MTCNN/InceptionResnetV1 loading was not exercised in CI-friendly tests
- Cosine similarity threshold of 0.75 may need tuning with real-world images
- Sync SQLite in async context remains a blocking concern under load
- Tracker's greedy assignment may struggle with >5 simultaneous faces

## Next Recommended Tasks
1. Wire LanceDB vector store for memory text embeddings
2. Replace rule-based dialogue with Claude LLM integration
3. Implement engagement/emotion estimation from face regions
4. Migrate to fully async aiosqlite
5. Add webcam live-stream support via capture module
6. Add end-to-end integration tests with real model inference
