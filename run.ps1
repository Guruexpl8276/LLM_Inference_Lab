# ═══════════════════════════════════════════════════════════════
#  LLM Inference Lab — run.ps1  (Windows / PowerShell)
#
#  - Uses .\.venv  unless $env:VENV_PATH or  -VenvPath  is set
#  - Reads .env if present (no clobber — existing $env:* wins)
#  - Auto-starts a local Ollama if installed and not running
#  - Launches uvicorn on $env:API_PORT (default 8000)
#
#  No drive-letter assumptions. Works from any directory.
# ═══════════════════════════════════════════════════════════════

param(
    [string]$VenvPath = "",
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectDir

# ── Load .env if present (does not overwrite already-set vars) ──
if (Test-Path "$ProjectDir\.env") {
    Get-Content "$ProjectDir\.env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $k, $v = $line -split "=", 2
            $k = $k.Trim(); $v = $v.Trim('"').Trim("'")
            if (-not (Test-Path "Env:$k")) { Set-Item -Path "Env:$k" -Value $v }
        }
    }
}

# ── Resolve venv + port ─────────────────────────────────────────
if (-not $VenvPath)   { $VenvPath = if ($env:VENV_PATH) { $env:VENV_PATH } else { "$ProjectDir\.venv" } }
if ($Port -eq 0)      { $Port     = if ($env:API_PORT)  { [int]$env:API_PORT } else { 8000 } }
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

# ── Create venv if missing ──────────────────────────────────────
if (-not (Test-Path $VenvPython)) {
    Write-Host "[setup] Creating venv at $VenvPath ..." -ForegroundColor Yellow
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pyExe) { $pyExe = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $pyExe) {
        Write-Host "[X] Python not found on PATH. Install from https://python.org and retry." -ForegroundColor Red
        exit 1
    }
    & $pyExe -m venv $VenvPath
    Write-Host "[setup] Installing dependencies ..." -ForegroundColor Yellow
    & "$VenvPath\Scripts\pip.exe" install -r "$ProjectDir\requirements.txt"
}

# ── Banner ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  LLM Inference Lab" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Python   : $VenvPython" -ForegroundColor Gray
Write-Host "  Port     : $Port" -ForegroundColor Gray
if ($env:OLLAMA_MODELS) {
    Write-Host "  Models   : $env:OLLAMA_MODELS" -ForegroundColor Gray
}
Write-Host ""
Write-Host "  Dashboard  -> http://localhost:$Port/dashboard" -ForegroundColor Green
Write-Host "  API Docs   -> http://localhost:$Port/docs" -ForegroundColor Green
Write-Host "  Queue      -> http://localhost:$Port/queue/status" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# ── Check / start Ollama ────────────────────────────────────────
$ollamaHost = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "http://localhost:11434" }
try {
    Invoke-RestMethod -Uri "$ollamaHost/api/tags" -TimeoutSec 2 -ErrorAction Stop | Out-Null
    Write-Host "[OK] Ollama is running at $ollamaHost" -ForegroundColor Green
} catch {
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        Write-Host "[!] Ollama is not installed (no ollama.exe on PATH)." -ForegroundColor Yellow
        Write-Host "    Install from: https://ollama.com/download" -ForegroundColor Yellow
        Write-Host "    The server will still start, but /chat and /generate will fail." -ForegroundColor DarkGray
    } else {
        Write-Host "[!] Ollama not running -- starting it now..." -ForegroundColor Yellow
        Start-Process $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 4
        Write-Host "[OK] Ollama started." -ForegroundColor Green
    }
}

# ── Launch FastAPI ──────────────────────────────────────────────
& $VenvPython -m uvicorn backend.main:app `
    --host ($(if ($env:API_HOST) { $env:API_HOST } else { "127.0.0.1" })) `
    --port $Port `
    --reload
