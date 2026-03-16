#!/usr/bin/env bash
# run_demo.sh - Start the Bay-Max Gradio demo
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true
python apps/demo/gradio_app.py
