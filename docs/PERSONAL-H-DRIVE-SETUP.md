# Running entirely off a non-system drive (H:, D:, external SSD)

If your `C:` drive is small and you want every byte — Python, venv, Ollama
binary, model blobs, pip cache, HuggingFace cache — to live on another
drive, here's the recipe. This is the setup the author uses; the file
exists in `docs/` for anyone who needs a worked example.

## Goal

| Thing | Default location | Off-system location |
|---|---|---|
| Python install | `C:\Users\...` | `H:\Python` (custom installer path) |
| Project venv | `./venv` | `H:\venvs\llm-gpu-lab` |
| Ollama binary | `%LOCALAPPDATA%\Programs\Ollama` | `H:\Ollama` |
| Ollama model blobs | `%USERPROFILE%\.ollama\models` | `H:\ollama_models` |
| pip cache | `~/.cache/pip` | `H:\pip_cache` |
| HF cache | `~/.cache/huggingface` | `H:\hf_cache` |
| Torch cache | `~/.cache/torch` | `H:\torch_cache` |
| Gradio temp | `%TEMP%` | `H:\gradio_temp` |
| Inference DB | `./inference_lab.db` | `H:\inference_lab.db` |

## Steps

### 1. Install Python to a custom path

Run the Python installer with **"Customize installation"** → set install
location to e.g. `H:\Python`. Tick *"Add Python to environment variables"*.

### 2. Install Ollama to a custom path

Ollama's Windows installer is Inno-Setup based and accepts a `/DIR` flag:

```powershell
# Download once
Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
    -OutFile "H:\Downloads\OllamaSetup.exe"

# Silent install to H:\Ollama
Start-Process "H:\Downloads\OllamaSetup.exe" `
    -ArgumentList '/DIR=H:\Ollama','/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART' `
    -Wait
```

Restart your terminal afterwards so `H:\Ollama` lands on PATH.

### 3. Tell the project where everything lives

Copy `.env.example` to `.env` and uncomment / set:

```dotenv
OLLAMA_MODELS=H:\ollama_models
INFERENCE_DB_PATH=H:\inference_lab.db
VENV_PATH=H:\venvs\llm-gpu-lab
```

### 4. Optional: redirect all cache dirs at shell level

The launcher scripts will pick up these env vars from `.env`. If you want
them set globally (e.g. so `pip` outside this project also uses H:), add to
your PowerShell profile:

```powershell
$env:OLLAMA_MODELS = "H:\ollama_models"
$env:HF_HOME       = "H:\hf_cache"
$env:PIP_CACHE_DIR = "H:\pip_cache"
$env:TORCH_HOME    = "H:\torch_cache"
$env:GRADIO_TEMP_DIR = "H:\gradio_temp"
```

### 5. Run

```powershell
# Creates H:\venvs\llm-gpu-lab if it doesn't exist
.\run.ps1 -VenvPath "H:\venvs\llm-gpu-lab"
```

Or just `.\run.ps1` if you set `VENV_PATH` in `.env`.

### 6. Pulling models lands them on H:

Since `OLLAMA_MODELS=H:\ollama_models` is set, every `ollama pull` writes to H:

```powershell
ollama pull llama3.2:3b      # ~2 GB → H:\ollama_models
ollama pull qwen2.5:3b       # ~2 GB → H:\ollama_models
ollama pull phi3:mini        # ~2.2 GB → H:\ollama_models
```

## C: drive footprint of this setup

With everything above, the *only* thing on C: from this project is:

- Windows registry entries (Ollama installer)
- Start Menu shortcuts (a few KB)
- A small `%USERPROFILE%\.ollama` directory holding Ollama's config (no models)

If you want to verify, after install run:

```powershell
"Ollama on C:"; Get-ChildItem "$env:USERPROFILE\.ollama" -Recurse | Measure-Object Length -Sum | Select-Object @{n='MB';e={[math]::Round($_.Sum/1MB,2)}}
"Project on H:";  Get-ChildItem H:\ollama_models -Recurse | Measure-Object Length -Sum | Select-Object @{n='MB';e={[math]::Round($_.Sum/1MB,2)}}
```
