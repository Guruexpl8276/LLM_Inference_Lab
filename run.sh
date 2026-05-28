#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  LLM Inference Lab — run.sh  (Linux / macOS / WSL)
#
#  - Uses ./.venv  unless $VENV_PATH is set
#  - Reads .env if present (does not overwrite existing env vars)
#  - Starts a local Ollama if installed and not running
#  - Launches uvicorn on $API_PORT (default 8000)
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ── Load .env (no clobber) ─────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    while IFS= read -r line; do
        case "$line" in
            ''|\#*) ;;
            *)
                key="${line%%=*}"
                if [ -z "${!key:-}" ]; then
                    eval "export $line"
                fi
                ;;
        esac
    done < "$PROJECT_DIR/.env"
    set +a
fi

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
PORT="${API_PORT:-8000}"
HOST="${API_HOST:-127.0.0.1}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# ── Create venv if missing ─────────────────────────────────────
if [ ! -x "$VENV_PATH/bin/python" ]; then
    echo "[setup] Creating venv at $VENV_PATH ..."
    PY="$(command -v python3 || command -v python || true)"
    if [ -z "$PY" ]; then
        echo "[X] Python 3 not found. Install Python and retry." >&2
        exit 1
    fi
    "$PY" -m venv "$VENV_PATH"
    echo "[setup] Installing dependencies ..."
    "$VENV_PATH/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
fi

VENV_PY="$VENV_PATH/bin/python"

# ── Banner ─────────────────────────────────────────────────────
cat <<EOF

═══════════════════════════════════════════
  LLM Inference Lab
═══════════════════════════════════════════
  Python   : $VENV_PY
  Port     : $PORT
${OLLAMA_MODELS:+  Models   : $OLLAMA_MODELS
}
  Dashboard  -> http://localhost:$PORT/dashboard
  API Docs   -> http://localhost:$PORT/docs
  Queue      -> http://localhost:$PORT/queue/status

  Press Ctrl+C to stop.

EOF

# ── Check / start Ollama ───────────────────────────────────────
if curl -fsS --max-time 2 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    echo "[OK] Ollama is running at $OLLAMA_HOST"
elif command -v ollama >/dev/null 2>&1; then
    echo "[!] Ollama not running — starting it now..."
    (nohup ollama serve >/tmp/ollama.log 2>&1 &)
    sleep 4
    echo "[OK] Ollama started."
else
    echo "[!] Ollama not installed. Install from: https://ollama.com/download"
    echo "    Server will start, but /chat and /generate will fail."
fi

# ── Launch FastAPI ─────────────────────────────────────────────
exec "$VENV_PY" -m uvicorn backend.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload
