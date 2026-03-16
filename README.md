# Bay-Max

A memory-first empathetic companion agent that recognizes enrolled users from vision input, tracks sessions, stores meaningful interaction memory, and produces supportive personalized responses.

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

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
  dialogue/          # Response generation
  orchestrator/      # End-to-end flow coordination
apps/api/            # FastAPI application
apps/demo/           # Gradio demo application
tests/               # Test suite
docs/                # Documentation and ADRs
scripts/             # Developer utility scripts
reports/             # Iteration reports
```

## Development

See [CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) for AI developer instructions.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture.
