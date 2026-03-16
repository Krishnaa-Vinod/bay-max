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
- [x] OpenCV webcam capture integration
- [x] Face detection with MTCNN model
- [x] Face embedding with ArcFace model
- [x] Face enrollment and recognition pipeline
- [x] User registration and authentication flow
- [x] Perception events and memory integration
- [x] Local testing tools and Gradio demo
- [x] Comprehensive test suite

## Iteration 003 - Pose Estimation & Engagement (Completed)
- [x] MediaPipe Pose backend integration
- [x] Body-state heuristics (posture, lean, motion)
- [x] Engagement scoring from multiple signals
- [x] Body-state observation events
- [x] Annotated debug output visualization
- [x] Local testing tools and manual evaluation
- [x] Comprehensive test coverage
- [x] Documentation and heuristics specification

## Iteration 004 - Memory Depth (Planned)
- [ ] Vector store integration (FAISS or LanceDB)
- [ ] Semantic search over memories
- [ ] Memory consolidation (episodic -> semantic)
- [ ] Salience-based memory decay and pruning
- [ ] Retrieval-augmented response generation

## Iteration 005 - LLM Dialogue (Planned)
- [ ] LLM-based dialogue provider (local model or API)
- [ ] Prompt engineering for supportive responses
- [ ] Memory-augmented prompts
- [ ] Response safety filtering

## Future Iterations
- Emotion recognition from facial expressions
- Multi-session user journey tracking
- Proactive check-ins
- Conversation history and context window management
- Audio integration
- Production deployment considerations
