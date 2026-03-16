# Changelog

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
