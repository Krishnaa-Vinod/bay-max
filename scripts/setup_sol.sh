#!/usr/bin/env bash
# setup_sol.sh - Set up Bay-Max environment on Sol cluster
# Usage: source scripts/setup_sol.sh

set -euo pipefail

echo "Setting up Bay-Max environment on Sol..."

# Create required directories
export BAYMAX_DATA_DIR="/scratch/$USER/bay-max/data"
export BAYMAX_CACHE_DIR="/scratch/$USER/bay-max/cache"
export BAYMAX_MODEL_DIR="/scratch/$USER/bay-max/models"
export BAYMAX_DB_URL="sqlite:///./baymax.db"

mkdir -p "$BAYMAX_DATA_DIR"
mkdir -p "$BAYMAX_CACHE_DIR"
mkdir -p "$BAYMAX_MODEL_DIR"

echo "Created directories:"
echo "  BAYMAX_DATA_DIR=$BAYMAX_DATA_DIR"
echo "  BAYMAX_CACHE_DIR=$BAYMAX_CACHE_DIR"
echo "  BAYMAX_MODEL_DIR=$BAYMAX_MODEL_DIR"
echo "  BAYMAX_DB_URL=$BAYMAX_DB_URL"

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -e ".[dev]" -q

echo ""
echo "Bay-Max environment ready. Run 'make test' to verify."
