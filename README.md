# Bay-Max

A friendly voice-first assistant with memory and empathetic understanding. Bay-Max listens to your questions, responds naturally in voice, remembers meaningful interactions, and recognizes enrolled users to personalize the experience.

## Quick Start: Voice Assistant Mode

```bash
# Create a virtual environment (Python 3.11+ required)
python3.11 -m venv .venv
source .venv/bin/activate

# Install recommended local full stack (dev + speech + tts + vector + dialogue)
pip install -e ".[all]"

# Set up environment
cp .env.example .env
# Edit .env with your paths

# Run tests
make test

# Start the voice assistant (backend + UI)
make run-local-full

# Open http://localhost:3000 in your browser
```

The assistant runs in **continuous VAD mode** by default — just speak and Bay-Max will respond.

## Core Experience (Iteration 013)

Bay-Max operates as a **friendly voice assistant first**:

1. **User speaks** → activated via VAD, wake phrase, or button
2. **Bay-Max listens** → captures speech until end-of-turn
3. **Bay-Max processes** → transcribe, understand, retrieve context
4. **Bay-Max responds** → clear, helpful answer delivered via TTS
5. **Follow-up window** → continue naturally without re-activating

Memory and perception enhance responses but don't dominate the experience.

## Activation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `continuous_vad` (default) | Always listening with VAD | Hands-free conversational use |
| `wake_phrase_gate` | Requires "Hey Baymax" | Shared environments |
| `push_to_talk` | Explicit button activation | Debugging, noisy environments |

Configure via `BAYMAX_ACTIVATION_MODE`. See [docs/VOICE_ASSISTANT_ARCHITECTURE.md](docs/VOICE_ASSISTANT_ARCHITECTURE.md) for details.

## Install Options

| Command                            | What it installs                  | Use when                               |
| ---------------------------------- | --------------------------------- | -------------------------------------- |
| `pip install -e ".[dev]"`          | Core + dev tools                  | Running tests only                     |
| `pip install -e ".[dev,vector]"`   | + LanceDB + sentence-transformers | Memory semantic search                 |
| `pip install -e ".[dev,dialogue]"` | + Transformers + PyTorch          | Local HuggingFace LLM dialogue         |
| `pip install -e ".[tts]"`          | Kokoro TTS + audio                | Spoken output                          |
| `pip install -e ".[speech]"`       | faster-whisper + audio            | Speech input                           |
| `pip install -e ".[all]"`          | Everything                        | Recommended full local companion stack |

For **Ollama-based dialogue** (easiest local LLM mode): no extra Python package required.
Install Ollama separately: https://ollama.ai, then `ollama pull qwen2.5:1.5b`.

## Dialogue Backends (Iteration 005)

Bay-Max supports three dialogue backends, selectable via `BAYMAX_DIALOGUE_BACKEND`:

| Backend                  | Env value      | Requirement                          |
| ------------------------ | -------------- | ------------------------------------ |
| Rule-based (default)     | `rule_based`   | None                                 |
| Ollama                   | `ollama`       | Ollama server running + model pulled |
| HuggingFace Transformers | `transformers` | `pip install -e ".[dialogue]"`       |

See `docs/LOCAL_TESTING.md` for full local-LLM testing instructions.

## TTS / Spoken Output (Iteration 007)

Bay-Max supports text-to-speech output via the `BAYMAX_TTS_BACKEND` setting:

| Backend             | Env value | Requirement               |
| ------------------- | --------- | ------------------------- |
| Kokoro (default)    | `kokoro`  | `pip install -e ".[tts]"` |
| Null (silent)       | `null`    | None                      |
| Piper (placeholder) | `piper`   | `piper-tts` + ONNX model  |

See `docs/TTS_ARCHITECTURE.md` for full TTS design details.

## Speech Input (Iteration 008)

Bay-Max supports bidirectional speech via `BAYMAX_STT_BACKEND`:

| Backend                  | Env value        | Requirement                  |
| ------------------------ | ---------------- | ---------------------------- |
| faster-whisper (default) | `faster_whisper` | `pip install -e ".[speech]"` |
| Null (no transcription)  | `null`           | None                         |

Features:

- Silero VAD for speech segmentation (configurable threshold and timing)
- Echo suppression to prevent responding to own TTS output
- Speaking lock with post-speech cooldown
- Push-to-talk mode for debugging
- Baymax-inspired companion persona style

See `docs/COMPANION_PERSONA.md` for the persona style guide.

## Project Structure

```
src/baymax/          # Core library
  config/            # Settings and configuration
  core/              # Enums and shared types
  schemas/           # Pydantic data models
  capture/           # Video/webcam ingestion interfaces
  perception/        # Face detection/recognition interfaces
  state/             # Interaction state management
  memory/            # Memory storage, retrieval, consolidation
  planner/           # Response strategy planning
  dialogue/          # Response generation (rule-based + LLM backends)
  tts/               # Text-to-speech (Kokoro, Piper, Null backends)
  audio/             # Speech input (microphone, VAD, ASR, echo suppression)
  orchestrator/      # End-to-end flow coordination
apps/api/            # FastAPI application + WebSocket telemetry
apps/ui/             # Companion diagnostic UI (Vite + React + TypeScript)
apps/demo/           # Gradio demo application
tests/               # Test suite
docs/                # Documentation and ADRs
scripts/             # Developer utility scripts
reports/             # Iteration reports
```

## Companion UI (Iteration 010b)

Bay-Max includes a diagnostic web UI for real-time runtime visualization:

```bash
# Install UI dependencies
make ui-install

# Recommended local full experience path: one backend process (API + live runtime)
make run-local-backend-full

# In another terminal, start the UI
make run-ui

# Open http://localhost:3000
```

One-command option (backend + UI together):

```bash
make run-local-full
```

Local backend modes:

```bash
# basic: text only (no speech input, no TTS)
make run-local-backend-basic

# voice: TTS only
make run-local-backend-voice

# full: TTS + speech input, with dialogue preference
# Ollama if available (stronger Qwen preferred: 7b -> 3b -> 1.5b -> 0.5b),
# else Transformers if explicitly configured, else rule_based fallback
make run-local-backend-full
```

In full mode, if dialogue falls back to rule-based, the UI shows:
LLM unavailable - using rule-based fallback

## Voice Assistant Core (Iteration 013)

Iteration 013 re-centers Bay-Max around a friendly voice-assistant core:

**State Machine**: One canonical state machine governs voice interaction:
```
idle → armed → listening → thinking → speaking → cooldown → (follow-up or idle)
```

**Activation Modes**:
- `continuous_vad` (default): Always listening with VAD
- `wake_phrase_gate`: "Hey Baymax" activation (provisional ASR-based)
- `push_to_talk`: Explicit button activation

**Follow-Up Window**: After speaking, Bay-Max stays ready for follow-up questions for a few seconds (configurable via `BAYMAX_FOLLOW_UP_WINDOW_SEC`).

**Barge-In**: Users can interrupt Bay-Max while speaking.

**Answer-First**: Direct questions get direct answers. Empathy-first only for emotional shares.

See [docs/VOICE_ASSISTANT_ARCHITECTURE.md](docs/VOICE_ASSISTANT_ARCHITECTURE.md) for the full architecture.

Advanced/debug only (split-process):

```bash
make run-api
make live-webcam
make run-ui
```

The UI shows three columns:

- **What Bay-Max Sees**: Live camera feed, identity, posture, engagement, affect
- **What Bay-Max is Doing**: Animated face, activity feed, last response, input controls
- **What Bay-Max Remembers**: Retrieved memories, semantic facts, memory stats

See [docs/COMPANION_UI_ARCHITECTURE.md](docs/COMPANION_UI_ARCHITECTURE.md) for UI architecture.
See [docs/WEBSOCKET_PROTOCOL.md](docs/WEBSOCKET_PROTOCOL.md) for WebSocket telemetry protocol.

## Previous Iterations

<details>
<summary>Voice-First Web Assistant (Iteration 011)</summary>

Iteration 011 introduced voice-first runtime with explicit mode transparency:

- `realtime_voice` (primary, if configured)
- `local_chained_voice` (local STT -> dialogue -> TTS fallback)
- `text_only` (debug/emergency fallback)

</details>

<details>
<summary>Answer-First Speech Focus (Iteration 012)</summary>

- Direct user questions and follow-ups are answer-first
- Emotional shares route to empathy-first responses
- Memory mentions are relevance-gated, not forced
- Low-quality transcripts trigger clarification instead of guessing

</details>

API endpoints:

- `POST /v1/realtime/session` (mode provisioning; no key leakage)
- `POST /v1/live/voice/interrupt` (barge-in interrupt)

Environment knobs:

- `BAYMAX_VOICE_MODE`
- `BAYMAX_ENABLE_REALTIME_VOICE`
- `BAYMAX_ENABLE_WEB_TOOLS`
- `BAYMAX_RECOGNITION_REFRESH_INTERVAL_SEC`

See [docs/LOCAL_TESTING.md](docs/LOCAL_TESTING.md) for the smoke workflow.

## Development

See [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) for AI developer instructions.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture.
See [docs/DIALOGUE_ARCHITECTURE.md](docs/DIALOGUE_ARCHITECTURE.md) for dialogue backend design.
See [docs/TTS_ARCHITECTURE.md](docs/TTS_ARCHITECTURE.md) for TTS spoken output design.
See [docs/LOCAL_TESTING.md](docs/LOCAL_TESTING.md) for local testing instructions.
See [docs/LIVE_VERIFICATION_GUIDE.md](docs/LIVE_VERIFICATION_GUIDE.md) for live webcam + TTS verification.
