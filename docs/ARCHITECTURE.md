# Bay-Max Architecture

## Overview
Bay-Max follows a **modular monolith** architecture. All modules live in a single Python package (`src/baymax/`) with clear interface boundaries between them.

## Runtime Stack
- Python 3.11+
- FastAPI (API layer)
- Pydantic v2 (data validation)
- SQLite via `sqlite3` (metadata storage)
- Gradio (demo UI)
- Stub interfaces for: PyTorch, OpenCV, FAISS/LanceDB (future iterations)

## Module Map

```
src/baymax/
├── config/          Settings from environment variables
├── core/            Enums and shared types
├── schemas/         Pydantic data models
│   ├── user.py      UserProfile, FaceEnrollment
│   ├── session.py   Session
│   ├── perception.py PerceptionEvent, Observation
│   ├── memory.py    EpisodicMemory, SemanticFact, InterventionMemory, etc.
│   └── response.py  SupportiveResponse
├── capture/         Frame source interfaces (stub)
├── perception/      Face detection/recognition interfaces (stub)
├── state/           Interaction state management
├── memory/          Memory storage, retrieval, salience, consolidation
│   ├── interfaces.py   MetadataStore ABC
│   ├── store_sqlite.py SQLite implementation
│   ├── vector_store.py Vector store adapter (stub)
│   ├── salience.py     Salience scoring
│   ├── retrieve.py     Memory retrieval
│   └── consolidate.py  Memory consolidation (stub)
├── planner/         Response strategy selection
├── dialogue/        Response generation (rule-based)
└── orchestrator/    End-to-end flow coordination
```

## Data Flow

1. **Capture**: Frame source provides video frames (stub in iteration-001)
2. **Perception**: Detect faces, recognize users, estimate engagement/emotion (stub)
3. **State**: Update interaction state for the session
4. **Memory**: Store observations, retrieve relevant memories
5. **Planner**: Choose response strategy based on state and memories
6. **Dialogue**: Generate supportive response from strategy
7. **Orchestrator**: Coordinates steps 1-6

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /healthz | Health check |
| POST | /v1/users | Create user |
| GET | /v1/users/{id} | Get user |
| POST | /v1/users/{id}/enroll/face | Face enrollment placeholder |
| POST | /v1/sessions | Create session |
| GET | /v1/sessions/{id}/state | Get session state |
| POST | /v1/memory/query | Query memories |
| POST | /v1/respond | Generate response |

## Storage

- **SQLite**: Users, sessions, episodic memories, semantic facts (metadata store)
- **Vector Store**: Embedding-based similarity search (stub adapter in iteration-001)
- **File System**: Model weights, data, caches stored outside repo via env vars

## Key Design Decisions

See `docs/adr/` for Architecture Decision Records:
- ADR-0001: Modular monolith architecture
- ADR-0002: Storage and data paths
- ADR-0003: Vision-first MVP
