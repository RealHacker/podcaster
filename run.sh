#!/usr/bin/env bash
#
# Podcaster launcher for Mac/Linux.
#
# Loads env.sh (MiniMax API key), then starts the local web UI.
#   ./run.sh                  -> http://127.0.0.1:8760
#   ./run.sh --port 9000
#   ./run.sh --address 0.0.0.0
#   ./run.sh --reload         -> auto-reload while developing
#

set -e

PORT=8760
ADDRESS="127.0.0.1"
RELOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --address|-a)
            ADDRESS="$2"
            shift 2
            ;;
        --reload|-r)
            RELOAD="--reload"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--address ADDRESS] [--reload]"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")"

# --- API keys -------------------------------------------------------------
if [[ -f "env.sh" ]]; then
    # shellcheck source=/dev/null
    source env.sh
    echo "Loaded env.sh"
fi

if [[ -z "$MINIMAX_APIKEY" && -z "$MINIMAX_API_KEY" ]]; then
    echo "Warning: MINIMAX_APIKEY is not set. Add it to env.sh or set it in the Settings tab." >&2
fi

# --- dependencies ---------------------------------------------------------
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "Error: python was not found on PATH." >&2
    exit 1
fi

declare -A packages=(
    ["fastapi"]="fastapi"
    ["uvicorn"]="uvicorn"
    ["httpx"]="httpx"
    ["pypdf"]="pypdf"
    ["beautifulsoup4"]="bs4"
    ["pdfplumber"]="pdfplumber"
    ["lxml"]="lxml"
    ["python-multipart"]="multipart"
)

missing=()
for pip_name in "${!packages[@]}"; do
    import_name="${packages[$pip_name]}"
    if ! "$PYTHON" -c "import $import_name" 2>/dev/null; then
        missing+=("$pip_name")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Installing missing packages: ${missing[*]}"
    "$PYTHON" -m pip install --quiet --disable-pip-version-check "${missing[@]}"
fi

# --- ffmpeg ---------------------------------------------------------------
if ! command -v ffmpeg &>/dev/null; then
    echo "Warning: ffmpeg is not on PATH — audio export still works, video export will not." >&2
fi

# --- check port -----------------------------------------------------------
if command -v lsof &>/dev/null; then
    if lsof -i ":$PORT" -sTCP:LISTEN &>/dev/null; then
        pid=$(lsof -t -i ":$PORT" -sTCP:LISTEN 2>/dev/null | head -1)
        echo "Warning: Port $PORT is already in use (PID ${pid:-unknown}). Stop it, or run ./run.sh --port $((PORT + 1))." >&2
        exit 1
    fi
elif command -v ss &>/dev/null; then
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo "Warning: Port $PORT is already in use. Stop it, or run ./run.sh --port $((PORT + 1))." >&2
        exit 1
    fi
fi

# --- run ------------------------------------------------------------------
URL="http://${ADDRESS}:${PORT}"

echo ""
echo "  Podcaster running at  $URL"
echo "  Press Ctrl+C to stop."
echo ""

# shellcheck disable=SC2086
exec "$PYTHON" -m uvicorn app:app --host "$ADDRESS" --port "$PORT" $RELOAD
