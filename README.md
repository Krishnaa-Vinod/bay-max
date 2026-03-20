# Bay-Max

A memory-first empathetic companion agent that recognizes enrolled users from vision input, tracks sessions, stores meaningful interaction memory, and produces supportive personalized responses.

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install recommended local full stack (dev + speech + tts + vector + dialogue)
pip install -e ".[all]"

# Set up environment
cp .env.example .env
# Edit .env with your paths

# Run tests
make test

# Start the API server
make run-api

# Start the Gradio demo
make run-demo
```

## Install Options

| Command | What it installs | Use when |
|---------|-----------------|----------|
| `pip install -e ".[dev]"` | Core + dev tools | Running tests only |
| `pip install -e ".[dev,vector]"` | + LanceDB + sentence-transformers | Memory semantic search |
| `pip install -e ".[dev,dialogue]"` | + Transformers + PyTorch | Local HuggingFace LLM dialogue |
| `pip install -e ".[tts]"` | Kokoro TTS + audio | Spoken output |
| `pip install -e ".[speech]"` | faster-whisper + audio | Speech input |
| `pip install -e ".[all]"` | Everything | Recommended full local companion stack |

For **Ollama-based dialogue** (easiest local LLM mode): no extra Python package required.
Install Ollama separately: https://ollama.ai, then `ollama pull qwen2.5:1.5b`.

## Dialogue Backends (Iteration 005)

Bay-Max supports three dialogue backends, selectable via `BAYMAX_DIALOGUE_BACKEND`:

| Backend | Env value | Requirement |
|---------|-----------|-------------|
| Rule-based (default) | `rule_based` | None |
| Ollama | `ollama` | Ollama server running + model pulled |
| HuggingFace Transformers | `transformers` | `pip install -e ".[dialogue]"` |

See `docs/LOCAL_TESTING.md` for full local-LLM testing instructions.

## TTS / Spoken Output (Iteration 007)

Bay-Max supports text-to-speech output via the `BAYMAX_TTS_BACKEND` setting:

| Backend | Env value | Requirement |
|---------|-----------|-------------|
| Kokoro (default) | `kokoro` | `pip install -e ".[tts]"` |
| Null (silent) | `null` | None |
| Piper (placeholder) | `piper` | `piper-tts` + ONNX model |

See `docs/TTS_ARCHITECTURE.md` for full TTS design details.

## Speech Input (Iteration 008)

Bay-Max supports bidirectional speech via `BAYMAX_STT_BACKEND`:

| Backend | Env value | Requirement |
|---------|-----------|-------------|
| faster-whisper (default) | `faster_whisper` | `pip install -e ".[speech]"` |
| Null (no transcription) | `null` | None |

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
# Ollama if available, else Transformers if explicitly configured, else rule_based fallback
make run-local-backend-full
```

In full mode, if dialogue falls back to rule-based, the UI shows:
LLM unavailable - using rule-based fallback

Advanced/debug only (split-process, easier to misconfigure):

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

## Development

See [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) for AI developer instructions.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture.
See [docs/DIALOGUE_ARCHITECTURE.md](docs/DIALOGUE_ARCHITECTURE.md) for dialogue backend design.
See [docs/TTS_ARCHITECTURE.md](docs/TTS_ARCHITECTURE.md) for TTS spoken output design.
See [docs/LOCAL_TESTING.md](docs/LOCAL_TESTING.md) for local testing instructions.
See [docs/LIVE_VERIFICATION_GUIDE.md](docs/LIVE_VERIFICATION_GUIDE.md) for live webcam + TTS verification.
