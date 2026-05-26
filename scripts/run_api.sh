#!/usr/bin/env bash
# Launch the FastAPI server
set -euo pipefail

# Activate the project virtualenv if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"
MODE="${MODE:-dev}"  # dev or prod

if [ "$MODE" = "dev" ]; then
    echo "→ Starting FinSight API (dev mode) on $HOST:$PORT"
    uvicorn src.api.main:app --reload --host "$HOST" --port "$PORT"
else
    echo "→ Starting FinSight API (prod mode) on $HOST:$PORT with $WORKERS workers"
    uvicorn src.api.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
fi
