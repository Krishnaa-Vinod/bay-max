# Changelog

## [0.2.0] - 2026-03-16 (Iteration 002)

### Added
- Real face detection using MTCNN (facenet-pytorch)
- Real face embeddings using InceptionResnetV1 pretrained on VGGFace2 (512-d vectors)
- Cosine similarity face recognition against enrolled embeddings (threshold 0.75)
- Simple IoU + cosine face tracker for frame-to-frame continuity
- Face enrollment endpoint accepting image uploads (`POST /v1/users/{user_id}/enroll/face`)
- Frame analysis endpoint with full recognition pipeline (`POST /v1/sessions/{session_id}/frames`)
- List users endpoint (`GET /v1/users`)
- Recognition-aware InteractionState (identity_confidence, face_count, track_ids)
- Meaningful-event-only observation writing (first_recognition, user_switch, known_to_unknown)
- New Pydantic schemas: FaceBoundingBox, DetectedFace, RecognizedFace, UnknownFace, FaceEmbeddingRecord, FrameAnalysisResult, RecognitionObservation
- New SQLite tables: face_embeddings, recognition_observations
- Gradio demo tabs for face enrollment and frame analysis
- 33 new tests (65 total)
- Model stack documentation (docs/MODEL_STACK.md)

### Changed
- Perception interfaces now use typed signatures (DetectedFace, FaceEmbeddingRecord)
- Device setting defaults to `cuda_if_available_else_cpu` (auto-detect)
- API version bumped to 0.2.0
- Orchestrator lazy-loads perception models with fallback to stubs

## [0.1.0] - 2026-03-16 (Iteration 001)

### Added
- Initial repository structure and Python project setup
- Core enums: SessionStatus, EngagementLevel, EmotionLabel, MemoryType, MemoryStatus, ResponseStrategy, PerceptionEventType
- Configuration settings from environment variables (BAYMAX_DATA_DIR, BAYMAX_CACHE_DIR, BAYMAX_MODEL_DIR, BAYMAX_DB_URL)
- Pydantic v2 schemas: UserProfile, FaceEnrollment, Session, PerceptionEvent, Observation, EpisodicMemory, SemanticFact, InterventionMemory, RetrievalLog, MemoryQuery, MemoryQueryResult, ProjectState, SupportiveResponse
- Capture module with FrameSource interface and StubFrameSource
- Perception module with interfaces for FaceDetector, FaceRecognizer, EngagementEstimator, EmotionEstimator (all with stub implementations)
- State management with InteractionState model and StateManager
- Memory module with MetadataStore interface, SQLiteMetadataStore, VectorStore interface, StubVectorStore, salience scoring, retrieval, and consolidation stubs
- Planner module with ResponsePlanner interface and SupportivePlanner
- Dialogue module with DialogueProvider interface and RuleBasedDialogue
- Orchestrator service coordinating end-to-end flow
- FastAPI application with 8 endpoints
- Gradio demo shell with Setup, Interact, and Memory tabs
- AI developer instructions (CLAUDE.md, AGENTS.md, .github/copilot-instructions.md)
- Architecture documentation and 3 ADRs
- CI workflow for lint and test
- Makefile with install, lint, test, run-api, run-demo targets
