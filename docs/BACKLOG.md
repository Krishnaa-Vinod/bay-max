# Bay-Max Backlog

## High Priority (Voice Assistant Core)
- [ ] Integrate AssistantStateMachine into LiveRuntime
- [ ] Normalize WebSocket/UI telemetry to use AssistantVoiceState vocabulary
- [ ] Add dedicated wake-word engine backend (Porcupine, OpenWakeWord)
- [ ] Add proper async SQLite operations using aiosqlite
- [ ] Stronger evaluation framework for dialogue quality

## Medium Priority
- [ ] Add LLM-based semantic fact extraction during consolidation
- [ ] Add salience decay over time for episodic memories
- [ ] Multi-session user journey tracking
- [ ] Speaker diarization for multi-person support
- [ ] Migrate to MediaPipe Tasks API for pose estimation
- [ ] Add user update and deactivation endpoints
- [ ] Add session end endpoint

## Low Priority
- [ ] Add OpenAPI schema documentation
- [ ] Add rate limiting to API endpoints
- [ ] Add structured logging
- [ ] Add metrics/observability
- [ ] Add database migration tooling
- [ ] Performance benchmarking for memory retrieval
- [ ] Multi-user concurrent session handling
- [ ] Camera calibration and pose normalization
- [ ] Complete Piper TTS backend implementation

## Completed (Iteration 013)
- [x] AssistantVoiceState enum for canonical voice states
- [x] AssistantStateMachine with explicit transitions
- [x] Activation modes: push_to_talk, continuous_vad, wake_phrase_gate
- [x] ActivationGate interface and implementations
- [x] Wake phrase detection (ASR-based provisional)
- [x] Follow-up window configuration
- [x] State machine tests (43 tests)
- [x] VOICE_ASSISTANT_ARCHITECTURE.md documentation

## Completed (Iteration 012)
- [x] Answer-first routing for direct questions
- [x] Empathy-first routing for emotional shares
- [x] Memory relevance gating (not forced)
- [x] Low-quality transcript clarification

## Completed (Iteration 011)
- [x] Voice-first runtime contract
- [x] Web tool broker (search, fetch, summarize)
- [x] Barge-in interrupt support
- [x] Sticky recognition across weak frames

## Technical Debt
- [ ] The SQLite store uses synchronous sqlite3; migrate to aiosqlite fully
- [ ] Gradio demo uses run_async workaround; improve async handling
- [ ] Semantic fact extraction uses simple keyword heuristics; needs LLM
- [ ] Wake phrase detection is ASR-gated; needs dedicated wake-word engine
- [ ] LanceDB vector store requires separate pip extra install
