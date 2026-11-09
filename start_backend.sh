#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Resolve or bootstrap Python virtual environment
PYTHON_EXEC=""
if [ -f ".venv/bin/python3" ]; then
    PYTHON_EXEC=".venv/bin/python3"
elif [ -f "backend/.venv/bin/python3" ]; then
    PYTHON_EXEC="backend/.venv/bin/python3"
elif [ -f "../.venv/bin/python3" ]; then
    PYTHON_EXEC="../.venv/bin/python3"
else
    # Auto-create .venv if missing
    if command -v python3 >/dev/null 2>&1; then
        echo "[backend] No virtual environment found. Bootstrapping dedicated Python .venv..."
        python3 -m venv --system-site-packages .venv || true
        if [ -f ".venv/bin/python3" ]; then
            PYTHON_EXEC=".venv/bin/python3"
            if command -v nvidia-smi >/dev/null 2>&1; then
                echo "[backend] NVIDIA GPU detected! Ensuring GPU-accelerated requirements into .venv..."
                "$PYTHON_EXEC" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
                "$PYTHON_EXEC" -m pip install -r backend/requirements-gpu.txt --quiet 2>/dev/null || true
            else
                echo "[backend] Ensuring backend requirements into .venv..."
                "$PYTHON_EXEC" -m pip install -r backend/requirements.txt --quiet 2>/dev/null || true
            fi
            echo "[backend] Python environment successfully initialized."
        else
            PYTHON_EXEC="python3"
        fi
    else
        PYTHON_EXEC="python3"
    fi
fi

# If --setup-only is passed, exit after environment is ready
if [ "$1" = "--setup-only" ]; then
    echo "[backend] Environment setup complete with Python: $PYTHON_EXEC"
    exit 0
fi

# Check if port 8000 is already in use
PIDS=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    # Test if it's already responding healthy
    HEALTH=$(curl -s --connect-timeout 1 http://127.0.0.1:8000/api/health 2>/dev/null || true)
    if echo "$HEALTH" | grep -q "online"; then
        echo "[backend] Python Vision Engine is already running and healthy on port 8000."
        trap 'exit 0' SIGTERM SIGINT
        while true; do
            sleep 3600 &
            wait $!
        done
    else
        echo "[backend] Port 8000 is occupied by stale process ($PIDS). Releasing port..."
        for p in $PIDS; do
            kill -9 "$p" 2>/dev/null || true
        done
        sleep 1
    fi
fi

echo "[backend] Starting Reef Vision Studio FastAPI Engine on port 8000 using $PYTHON_EXEC..."
exec "$PYTHON_EXEC" backend/run_server.py --port 8000 --reload "$@"
