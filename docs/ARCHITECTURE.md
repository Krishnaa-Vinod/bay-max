# Bay-Max Architecture

## Overview
Bay-Max follows a **modular monolith** architecture. All modules live in a single Python package (`src/baymax/`) with clear interface boundaries between them.

## Runtime Stack
- Python 3.11+
- FastAPI (API layer)
- Pydantic v2 (data validation)
- SQLite via `sqlite3` (metadata storage, synchronous)
- LanceDB (optional vector store for semantic search)
- Gradio (demo UI)
- PyTorch + torchvision (face recognition models)
- MTCNN (face detection), InceptionResnetV1 (face embeddings)
- MediaPipe (pose estimation)
- sentence-transformers (text memory embeddings)
- Ollama or HuggingFace Transformers (local LLM dialogue, optional)
- Kokoro (TTS, speech synthesis)
- faster-whisper (ASR, speech-to-text)
- Silero VAD (voice activity detection)
- sounddevice (audio capture and playback)
- soundfile (WAV I/O)

## Module Map

```
src/baymax/
├── config/          Settings from environment variables (BaymaxSettings)
├── core/            Enums and shared types
├── schemas/         Pydantic data models
│   ├── user.py      UserProfile, FaceEnrollment
│   ├── session.py   Session
│   ├── perception.py PerceptionEvent, Observation, PoseResult, EngagementResult
│   ├── memory.py    EpisodicMemory, SemanticFact, ChatTurn, ConsolidationResult, etc.
│   └── response.py  SupportiveResponse, DialogueBackendInfo, GroundedPromptContext,
│                    SafetyDecision, GroundedResponse, DialogueDebugTrace
├── capture/         Frame source interfaces (stub)
├── perception/      Computer vision and engagement estimation
│   ├── interfaces.py    FaceDetector, FaceRecognizer, PoseEstimator ABCs + stubs
│   ├── facenet_adapter.py  MTCNN/InceptionResnetV1 face detection+embeddings
│   ├── pose_interface.py   Pose estimation interface
│   ├── mediapipe_pose.py   MediaPipe Pose implementation
│   ├── heuristics.py       Body-state and engagement heuristics + MotionTracker
│   ├── tracker.py          IoU+cosine face tracker
│   └── annotations.py      Debug visualization utilities
├── state/           Interaction state management (InteractionState, StateManager)
├── memory/          Memory storage, retrieval, salience, consolidation, embeddings
│   ├── interfaces.py   MetadataStore ABC
│   ├── store_sqlite.py SQLite implementation (sync despite async signatures)
│   ├── vector_store.py LanceDBVectorStore + StubVectorStore adapters
│   ├── embedding.py    SentenceTransformerEmbedder + StubTextEmbedder
│   ├── salience.py     Salience scoring
│   ├── retrieve.py     Memory retrieval wrapper
│   └── consolidate.py  Session consolidation pipeline
├── planner/         Response strategy selection (SupportivePlanner)
├── dialogue/        Response generation
│   ├── interfaces.py           DialogueProvider ABC
│   ├── rule_based.py           RuleBasedDialogue (template-based fallback)
│   ├── ollama_provider.py      OllamaDialogueProvider (httpx to local Ollama)
│   ├── transformers_provider.py TransformersDialogueProvider (local HF model)
│   ├── prompt_builder.py       GroundedPromptContext builder (plan-then-verbalize)
│   ├── safety.py               Safety gating (check_safety, check_output_safety)
│   └── factory.py              create_dialogue_provider() factory with fallback
├── tts/             Text-to-speech backends and speech service
│   ├── provider.py             TTSProvider interface + factory + NullTTSProvider
│   ├── kokoro_provider.py      KokoroTTSProvider (Kokoro-82M)
│   ├── piper_provider.py       PiperTTSProvider (placeholder)
│   ├── speech_service.py       SpeechService with queued playback
│   └── schemas.py              TTS Pydantic schemas
├── audio/           Speech input pipeline (microphone, VAD, ASR, echo suppression)
│   ├── schemas.py              Audio Pydantic schemas (AudioChunkInfo, VADDecision, etc.)
│   ├── microphone.py           MicrophoneCapture + NullMicrophone
│   ├── vad.py                  SileroVAD + NullVAD (voice activity detection)
│   ├── transcriber.py          ASRProvider ABC + FasterWhisperProvider + NullASRProvider
│   ├── echo_suppression.py     EchoSuppressor (text-similarity based)
│   └── speech_input_service.py SpeechInputService (full pipeline orchestrator)
└── orchestrator/    End-to-end flow coordination (Orchestrator)
```

## Data Flow

1. **Capture**: Frame source provides video frames (stub — webcam not yet implemented)
2. **Perception**: Face detection → recognition → pose estimation → engagement scoring
3. **State**: Update interaction state for the session
4. **Memory**: Store observations, retrieve relevant memories (SQL + vector search)
5. **Planner**: Choose response strategy based on state and memories
6. **Dialogue**:
   - Safety check: detect diagnosis-style or unsafe medical requests
   - Build GroundedPromptContext (state, memories, recent turns, safety rules)
   - Route to configured backend (rule_based / ollama / transformers)
   - Fallback to rule_based if backend fails
7. **Orchestrator**: Coordinates steps 1-6
7.5. **TTS**: SupportiveResponse text -> speech synthesis -> WAV -> optional playback
8. **Speech Input** (bidirectional loop):
   - Microphone capture -> Silero VAD (speech segmentation) -> faster-whisper ASR -> echo suppression
   - Speaking lock: mic audio discarded while TTS is active + post-speech cooldown
   - Valid transcriptions stored as ChatTurn with source='speech' -> orchestrator.respond() -> TTS

## Dialogue Architecture (Iteration 005)

Uses a **plan-then-verbalize** pattern:

```
SupportivePlanner → ResponseStrategy
                          ↓
              GroundedPromptBuilder
          (state + memories + turns + safety)
                          ↓
              DialogueProvider.generate()
          (rule_based | ollama | transformers)
                          ↓
              SafetyCheck (output validation)
                          ↓
              SupportiveResponse
          (text + backend + model_name + memory_refs
           + fallback_used + safety_flags)
```

See `docs/DIALOGUE_ARCHITECTURE.md` for full details.

## API Endpoints (v0.7.0)

| Method | Path | Purpose |
|--------|------|---------|
| GET | /healthz | Health check |
| POST | /v1/users | Create user |
| GET | /v1/users | List users |
| GET | /v1/users/{id} | Get user |
| POST | /v1/users/{id}/enroll/face | Face enrollment |
| POST | /v1/sessions | Create session |
| GET | /v1/sessions/{id}/state | Get session state |
| POST | /v1/sessions/{id}/frames | Submit frame for analysis |
| POST | /v1/memory/query | Query memories |
| POST | /v1/respond | Generate grounded supportive response |
| POST | /v1/sessions/{id}/turns | Add chat turn |
| GET | /v1/sessions/{id}/turns | Get chat turns |
| POST | /v1/sessions/{id}/consolidate | Consolidate session into memories |
| GET | /v1/memory/summary/{user_id} | Memory summary |
| POST | /v1/memory/search | Semantic memory search |
| POST | /v1/memory/correct | Correct a semantic fact |
| GET | /v1/dialogue/backends | Get dialogue backend configuration |
| GET | /v1/tts/backends | Get TTS backend configuration |
| GET | /v1/audio/status | Speech input/output runtime status |
| GET | /v1/stt/backends | STT backend configuration |

## Storage

- **SQLite**: Users, sessions, episodic memories, semantic facts, chat turns, session summaries
- **Vector Store**: LanceDB for embedding-based similarity search (StubVectorStore fallback)
- **File System**: Model weights, data, caches stored outside repo via env vars

## Key Design Decisions

See `docs/adr/` for Architecture Decision Records:
- ADR-0001: Modular monolith architecture
- ADR-0002: Storage and data paths
- ADR-0003: Vision-first MVP

Additional iteration decisions:
- Iteration 002: cosine similarity face matching, IoU+cosine tracker
- Iteration 003: MediaPipe Pose, heuristic engagement scoring
- Iteration 004: sentence-transformers embeddings, LanceDB, majority-vote smoothing
- Iteration 005: plan-then-verbalize dialogue, Ollama + Transformers backends, safety gating
- Iteration 006: live webcam continuity, event engine, proactive scheduler, artifact logging
- Iteration 007: TTS spoken output, webcam+TTS verification
- Iteration 008: Bidirectional speech (mic+VAD+ASR+echo suppression), Baymax-inspired persona
