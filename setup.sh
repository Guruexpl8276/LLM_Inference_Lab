#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  LLM Inference Lab — setup.sh  (Linux / macOS / WSL)
#
#  First-time installer:
#    1. Detects GPU + VRAM via nvidia-smi
#    2. Recommends a model preset for your card
#    3. Creates ./.venv and installs dependencies
#    4. Verifies Ollama; offers to pull recommended models
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo
echo "═══════════════════════════════════════════"
echo "  LLM Inference Lab — Setup"
echo "═══════════════════════════════════════════"

# ── [1/4] GPU detection ────────────────────────────────────────
echo
echo "[1/4] Detecting GPU ..."
vram_mb=0
gpu_name="Unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    line=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    gpu_name=$(echo "$line" | awk -F',' '{print $1}' | xargs)
    vram_mb=$(echo "$line" | awk -F',' '{print $2}' | xargs)
    echo "  GPU       : $gpu_name"
    echo "  Total VRAM: ${vram_mb} MB ($(awk "BEGIN {printf \"%.1f\", $vram_mb/1024}") GB)"
else
    echo "  nvidia-smi not found. CPU-only mode — inference will be slow."
fi

if   [ "$vram_mb" -le 0 ];        then tier="cpu";   models="tinyllama gemma3:1b"
elif [ "$vram_mb" -lt 4500 ];     then tier="4GB";   models="llama3.2:3b phi3:mini tinyllama"
elif [ "$vram_mb" -lt 7000 ];     then tier="6GB";   models="llama3.2:3b phi3:mini qwen2.5:3b gemma2:2b"
elif [ "$vram_mb" -lt 10000 ];    then tier="8GB";   models="llama3.1:8b qwen2.5:7b mistral:7b"
elif [ "$vram_mb" -lt 14000 ];    then tier="12GB";  models="llama3.1:8b qwen2.5:14b mistral-nemo"
elif [ "$vram_mb" -lt 22000 ];    then tier="16GB";  models="qwen2.5:14b llama3.1:8b mistral-nemo"
else                                   tier="24GB+"; models="llama3.1:8b qwen2.5:32b llama3.3:70b-q4"
fi

echo
echo "  Recommended preset: $tier"
echo "  Suggested models  : $models"

# ── [2/4] Python venv ──────────────────────────────────────────
echo
echo "[2/4] Setting up Python venv (./.venv) ..."
VENV_PATH="$PROJECT_DIR/.venv"
if [ ! -x "$VENV_PATH/bin/python" ]; then
    PY="$(command -v python3 || command -v python || true)"
    if [ -z "$PY" ]; then
        echo "  ERROR: Python 3.10+ not found on PATH. Install from https://python.org" >&2
        exit 1
    fi
    "$PY" -m venv "$VENV_PATH"
fi
"$VENV_PATH/bin/pip" install --upgrade pip >/dev/null
"$VENV_PATH/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
echo "  [OK] Dependencies installed."

# ── [3/4] Ollama ───────────────────────────────────────────────
echo
echo "[3/4] Checking Ollama ..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "  Ollama is NOT installed."
    echo "  Linux/macOS: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Or download from: https://ollama.com/download"
else
    echo "  Found: $(ollama --version)"
fi

# ── [4/4] Optional pulls ───────────────────────────────────────
echo
read -r -p "[4/4] Pull recommended models [$models]? (y/N) " ans
if [[ "$ans" =~ ^[Yy]$ ]] && command -v ollama >/dev/null 2>&1; then
    for m in $models; do
        echo "  Pulling $m ..."
        ollama pull "$m"
    done
fi

if [ ! -f "$PROJECT_DIR/.env" ] && [ -f "$PROJECT_DIR/.env.example" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "  Created .env from .env.example"
fi

echo
echo "═══════════════════════════════════════════"
echo "  Setup complete."
echo "═══════════════════════════════════════════"
echo
echo "  Start the server : ./run.sh"
echo "  Then open        : http://localhost:8000/dashboard"
echo
