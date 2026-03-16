# Bay-Max Roadmap

## Iteration 001 - Bootstrap (Current)
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

## Iteration 002 - Vision Pipeline (Planned)
- [ ] OpenCV webcam capture integration
- [ ] Face detection with a lightweight model
- [ ] Face embedding and enrollment
- [ ] Basic face recognition for enrolled users
- [ ] Engagement estimation from face tracking
- [ ] Perception events flowing through the pipeline

## Iteration 003 - Memory Depth (Planned)
- [ ] Vector store integration (FAISS or LanceDB)
- [ ] Semantic search over memories
- [ ] Memory consolidation (episodic -> semantic)
- [ ] Salience-based memory decay and pruning
- [ ] Retrieval-augmented response generation

## Iteration 004 - LLM Dialogue (Planned)
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
