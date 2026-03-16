# Iteration 004 Report: Memory Recall & Consolidation

## Branch
`feature/iteration-004-memory-recall`

## Summary
Implemented the full memory recall and consolidation pipeline: text embedding via sentence-transformers, LanceDB vector retrieval, typed conversation turns, session consolidation into episodic memories and semantic facts, temporal smoothing for engagement/posture, memory-aware response generation, memory inspection and correction, and expanded Gradio demo with 7 tabs. 158 tests pass, ruff lint clean, all prior functionality preserved.

## Tasks Completed

### T401: Create Working Branch, Audit Install/Docs, Fix pyproject Extras
- Created branch `feature/iteration-004-memory-recall` from iteration-003
- Fixed `pyproject.toml` extras: added `vector` group with `lancedb`, `sentence-transformers`, `pyarrow`; updated `all` extras to include `vector`
- Added `install-vector` target to Makefile
- Added new environment variables to `.env.example`

### T402: Implement Text Embedding Adapter + LanceDB Vector Store
- Created `TextEmbedder` ABC with `embed()` and `embed_batch()` in `src/baymax/memory/embedding.py`
- Implemented `SentenceTransformerEmbedder` (all-MiniLM-L6-v2, 384-d vectors)
- Implemented `StubTextEmbedder` for testing (deterministic hash-based vectors)
- Rewrote `VectorStore` ABC with `filter_user_id` parameter support
- Implemented `LanceDBVectorStore` using PyArrow schema (384-d float32 vectors)
- Added `StubVectorStore` with in-memory cosine similarity search
- Added settings: `text_embedding_model`, `text_embedding_dim`, `memory_top_k`, `memory_similarity_threshold`

### T403: Add Typed Conversation Turns and Session Text Memory
- Added `TurnRole` enum (user/system) to `core/enums.py`
- Created `ChatTurn` Pydantic model with id, session_id, user_id, role, text, timestamp
- Added `chat_turns` SQLite table with `store_chat_turn()` and `get_chat_turns()` methods
- Added `add_turn()` and `get_turns()` to orchestrator
- Added API endpoints: `POST/GET /v1/sessions/{session_id}/turns`

### T404: Implement Session Consolidation and Memory Promotion
- Created full consolidation pipeline in `src/baymax/memory/consolidate.py`
- `consolidate_session()`: fetches turns + observations, builds session summary, extracts episodic memories, extracts candidate semantic facts, stores everything
- `consolidate_episodic_to_semantic()`: promotes recurring episodic patterns to semantic facts
- `merge_duplicate_facts()`: deduplicates by exact content match, keeping highest confidence
- Semantic fact extraction uses heuristic preference signals ("i like", "i prefer", etc.)
- Added `SessionSummary`, `ConsolidationResult` schemas
- Added `session_summaries` SQLite table
- Added API endpoint: `POST /v1/sessions/{session_id}/consolidate`

### T405: Add Temporal Smoothing and Memory-Write Suppression
- Implemented majority-vote temporal smoothing for engagement levels over configurable window (default 5)
- Implemented posture smoothing with same mechanism
- Added `session_smoothing_window` setting
- Smoothing buffers maintained per-session in orchestrator

### T406: Make Responses Memory-Aware
- Updated `respond()` to perform semantic search before generating response
- Populates `memory_refs` (list of matching memory contents) and `state_summary` (session state snapshot)
- Lazy-loads text embedder and vector store on first use
- Added `_semantic_search()` helper for top-k similarity retrieval
- Added `memory_refs` and `state_summary` fields to `SupportiveResponse`

### T407: Add Memory Inspection and Correction Support
- Added `FactStatus` (candidate/confirmed/rejected) and `CorrectionAction` (confirm/reject/update) enums
- Created `MemoryCorrectionRequest`, `MemoryCorrectionResult`, `MemorySummaryResponse` schemas
- Added `get_semantic_fact()`, `update_semantic_fact()`, `count_episodic_memories()`, `count_semantic_facts()` to store interface
- Implemented `correct_memory()` and `get_memory_summary()` in orchestrator
- Added `query_memory_semantic()` for direct semantic search endpoint
- Added API endpoints: `GET /v1/memory/summary/{user_id}`, `POST /v1/memory/search`, `POST /v1/memory/correct`

### T408: Upgrade Gradio Demo and Manual Test Flow
- Added **Chat** tab: send typed turns (user/system), view turn history
- Added **Consolidate** tab: trigger session consolidation, view results
- Expanded **Memory** tab: memory summary view, semantic fact correction UI (confirm/reject/update)
- Total: 7 tabs (Setup, Enroll Face, Frame Analysis, Chat, Interact, Consolidate, Memory)

### T409: Tests, Docs, Report, Commit, and Push
- Created `tests/test_iteration_004.py` with 49 test cases across 11 test classes
- All 158 tests pass (109 existing + 49 new), ruff lint clean
- Updated `docs/project_state.json`, `docs/PROJECT_MEMORY.md`, `docs/BACKLOG.md`
- Created iteration reports

## New Schemas
- `ChatTurn` (id, session_id, user_id, role, text, timestamp)
- `SessionSummary` (session_id, user_id, summary_text, turn_count, observation_count, episodic_ids, semantic_ids)
- `MemoryEmbeddingRecord` (id, user_id, memory_type, source_id, content, embedding)
- `MemoryHit` (id, content, memory_type, similarity, user_id)
- `ConsolidationResult` (session_id, user_id, summary, episodic/semantic counts, errors)
- `MemoryCorrectionRequest` (fact_id, action, updated_content)
- `MemoryCorrectionResult` (fact_id, action, previous/new content, new_status)
- `MemorySummaryResponse` (episodic/semantic counts, session_summaries, recent_episodic, confirmed_facts)

## New Enums
- `TurnRole` (user, system)
- `FactStatus` (candidate, confirmed, rejected)
- `CorrectionAction` (confirm, reject, update)

## New SQLite Tables
- `chat_turns` (id, session_id, user_id, role, text, created_at)
- `session_summaries` (id, session_id, user_id, summary_text, turn_count, observation_count, episodic_ids_json, semantic_ids_json, created_at)

## API Changes (6 new endpoints)
- `POST /v1/sessions/{session_id}/turns` — Add a typed conversation turn
- `GET /v1/sessions/{session_id}/turns` — Get all turns for a session
- `POST /v1/sessions/{session_id}/consolidate` — Consolidate session into memories
- `GET /v1/memory/summary/{user_id}` — Get memory summary for a user
- `POST /v1/memory/search` — Semantic similarity search across memories
- `POST /v1/memory/correct` — Correct a semantic fact (confirm/reject/update)
- App version: 0.3.0

## Test Results
- **Command**: `pytest tests/ -v`
- **Result**: 158 passed, 0 failed
- **Test breakdown**:
  - `test_iteration_004.py`: 49 tests (settings, embeddings, vector store, turns, consolidation, smoothing, memory-aware responses, correction, schemas, API endpoints)
  - `test_iteration_003.py`: 44 tests (pose, engagement, body state)
  - `test_iteration_002.py`: 33 tests (face recognition pipeline)
  - Other existing tests: 32 tests
- **Lint**: `ruff check` — All checks passed

## What Was Actually Tested
- New settings defaults and types
- StubTextEmbedder: dimension, normalization, batch embedding
- StubVectorStore: add, search, delete, user filtering
- ChatTurn storage and retrieval ordering in SQLite
- Session consolidation with turns, observations, and empty sessions
- Semantic fact extraction from preference signals
- Session summary storage
- Episodic-to-semantic promotion thresholds
- Duplicate fact merging by content
- Temporal smoothing majority-vote logic
- SupportiveResponse memory_refs and state_summary fields
- Semantic fact CRUD (get, update, count)
- Memory correction and summary response schemas
- All 6 new API endpoints via httpx AsyncClient

## What Was NOT Tested
- End-to-end with real SentenceTransformerEmbedder model loading
- LanceDB vector store with actual LanceDB backend
- Full pipeline: turns → consolidation → semantic search → memory-aware response in single flow
- Performance/latency of embedding generation and vector search
- Concurrent session consolidation
- Real MediaPipe/facenet model integration with memory pipeline

## Lint Results
- **Command**: `ruff check src/ apps/ tests/`
- **Result**: All checks passed
- **Fixes applied**: Unused imports, import sorting

## Model Stack Updates

### Text Embedding
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Interface**: `TextEmbedder` ABC with `SentenceTransformerEmbedder` and `StubTextEmbedder`

### Vector Search
- **Backend**: LanceDB (local, serverless)
- **Schema**: PyArrow table with 384-d float32 vector column
- **Search**: Cosine similarity, configurable top_k and threshold
- **Fallback**: `StubVectorStore` with in-memory cosine similarity

## Known Issues
1. SQLite store uses synchronous sqlite3 despite async signatures
2. Emotion estimator remains stubbed
3. Gradio demo uses sync-to-async workaround
4. MediaPipe uses legacy API (may be deprecated)
5. LanceDB requires separate install (`pip install baymax[vector]`)
6. Semantic fact extraction uses simple keyword heuristics, not LLM-based
7. No salience decay for episodic memories

## Deviations from Prompt
- None significant. All 9 task areas implemented as specified.

## Risks
- SentenceTransformerEmbedder model loading not exercised in CI tests
- LanceDB integration not tested with real backend in automated tests
- Semantic fact extraction heuristics may miss complex preference patterns
- Temporal smoothing window size is global, not per-session configurable
- Memory correction has no audit trail beyond the fact status field

## Next Recommended Tasks
1. Replace rule-based dialogue with Claude LLM-based generation
2. Implement emotion estimation from facial expressions
3. Add LLM-based semantic fact extraction during consolidation
4. Migrate to fully async aiosqlite
5. Add salience decay over time for episodic memories
6. Add webcam live-stream support via capture module
7. Migrate to MediaPipe Tasks API for pose estimation
