# Bay-Max Project Memory

This file provides continuity context for AI agents working on the Bay-Max project. Update this file after completing significant work.

## Last Updated
2026-03-16 (Iteration 001)

## Current State
- **Iteration**: 001 (Bootstrap)
- **Branch**: bootstrap/iteration-001
- **Status**: Complete

## What Exists
- Full modular monolith structure under `src/baymax/`
- 7 modules: capture, perception, state, memory, planner, dialogue, orchestrator
- Pydantic v2 schemas for all data models
- SQLite-backed metadata store with CRUD for users, sessions, and memories
- Vector store adapter interface with in-memory stub
- Salience scoring utility
- Rule-based dialogue provider with template responses
- FastAPI backend with 8 endpoints (all tested)
- Gradio demo shell with 3 tabs
- Settings from environment variables

## What Is Stubbed
- Face detection, recognition, engagement, emotion estimation (return hardcoded values)
- Memory consolidation (returns empty)
- Vector store (in-memory naive implementation)
- Gradio demo uses sync workaround for async

## Key Decisions Made
- Modular monolith over microservices (ADR-0001)
- SQLite for metadata, vector store adapter for embeddings (ADR-0002)
- Vision-first MVP scope, rule-based dialogue for iteration-001 (ADR-0003)

## Known Issues
- SQLite store uses synchronous sqlite3 despite being wrapped in async
- Perception stubs return hardcoded values
- No real face enrollment processing

## For Next Agent
1. Read this file and `docs/project_state.json` first
2. Check `docs/BACKLOG.md` for pending work
3. The highest-impact next steps are implementing real face detection and vector store
4. All modules use interfaces; implement concrete classes without changing the interface contracts
