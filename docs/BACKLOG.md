# Bay-Max Backlog

## High Priority
- [ ] Replace rule-based dialogue with LLM-based generation (Claude)
- [ ] Add proper async SQLite operations using aiosqlite
- [ ] Implement emotion estimation from facial expressions
- [ ] Add LLM-based semantic fact extraction during consolidation

## Medium Priority
- [ ] Add user update and deactivation endpoints
- [ ] Add session end endpoint
- [ ] Add salience decay over time for episodic memories
- [ ] Add engagement estimation from head pose / gaze
- [ ] Add multi-person pose tracking support
- [ ] Migrate to MediaPipe Tasks API for pose estimation
- [ ] Add webcam live-stream support via capture module

## Low Priority
- [ ] Add OpenAPI schema documentation
- [ ] Add rate limiting to API endpoints
- [ ] Add structured logging
- [ ] Add metrics/observability
- [ ] Add database migration tooling
- [ ] Performance benchmarking for memory retrieval
- [ ] Multi-user concurrent session handling
- [ ] Camera calibration and pose normalization

## Completed (Iteration 004)
- [x] Add vector store implementation (LanceDB) for semantic memory search
- [x] Implement memory consolidation logic
- [x] Implement memory correction and deletion API
- [x] Add temporal smoothing for engagement transitions
- [x] Add typed conversation turns for pre-audio dialogue
- [x] Make responses memory-aware with recall

## Completed (Iteration 003)
- [x] Add engagement estimation from body pose
- [x] Implement pose estimation via MediaPipe

## Completed (Iteration 002)
- [x] Implement real face detection using facenet-pytorch MTCNN
- [x] Implement face embedding storage and retrieval for enrollment

## Technical Debt
- [ ] The SQLite store uses synchronous sqlite3; migrate to aiosqlite fully
- [ ] Gradio demo uses run_async workaround; improve async handling
- [ ] Semantic fact extraction uses simple keyword heuristics; needs LLM
- [ ] LanceDB vector store requires separate pip extra install
