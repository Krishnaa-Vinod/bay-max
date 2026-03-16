# Bay-Max Backlog

## High Priority
- [ ] Implement real face detection using a lightweight model (e.g., MediaPipe, RetinaFace)
- [ ] Implement face embedding storage and retrieval for enrollment
- [ ] Add vector store implementation (FAISS or LanceDB) for semantic memory search
- [ ] Replace rule-based dialogue with LLM-based generation
- [ ] Add proper async SQLite operations using aiosqlite

## Medium Priority
- [ ] Add user update and deactivation endpoints
- [ ] Add session end endpoint
- [ ] Implement memory correction and deletion API
- [ ] Add engagement estimation from head pose / gaze
- [ ] Add emotion estimation from facial expressions
- [ ] Implement memory consolidation logic
- [ ] Add salience decay over time
- [ ] Add more comprehensive test coverage

## Low Priority
- [ ] Add OpenAPI schema documentation
- [ ] Add rate limiting to API endpoints
- [ ] Add structured logging
- [ ] Add metrics/observability
- [ ] Add database migration tooling
- [ ] Performance benchmarking for memory retrieval
- [ ] Multi-user concurrent session handling

## Technical Debt
- [ ] The SQLite store uses synchronous sqlite3; migrate to aiosqlite fully
- [ ] Perception stubs return hardcoded values; replace with real implementations
- [ ] Vector store stub uses naive dot product; replace with proper ANN search
- [ ] Gradio demo uses run_async workaround; improve async handling
