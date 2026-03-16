#!/usr/bin/env bash
# run_api.sh - Start the Bay-Max FastAPI server
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
