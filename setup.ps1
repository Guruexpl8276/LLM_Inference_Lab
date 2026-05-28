# ═══════════════════════════════════════════════════════════════
#  LLM Inference Lab — setup.ps1  (Windows / PowerShell)
#
#  First-time installer:
#    1. Detects GPU + VRAM via nvidia-smi
#    2. Recommends a model preset for your card
#    3. Creates a local .venv and installs dependencies
#    4. Verifies Ollama is installed; offers to pull recommended models
# ═══════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectDir

Write-Host ""
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  LLM Inference Lab — Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan

# ── [1/4] Detect GPU + VRAM ────────────────────────────────────
Write-Host ""
Write-Host "[1/4] Detecting GPU ..." -ForegroundColor Green
$gpuName  = "Unknown"
$vramMB   = 0
try {
    $smi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -eq 0 -and $smi) {
        $parts = ($smi -split ",") | ForEach-Object { $_.Trim() }
        $gpuName = $parts[0]
        $vramMB  = [int]$parts[1]
        Write-Host ("  GPU       : {0}" -f $gpuName) -ForegroundColor White
        Write-Host ("  Total VRAM: {0} MB ({1:N1} GB)" -f $vramMB, ($vramMB / 1024.0)) -ForegroundColor White
    } else {
        Write-Host "  nvidia-smi failed. CPU-only mode — inference will be very slow." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  nvidia-smi not found. Install NVIDIA drivers, or proceed CPU-only." -ForegroundColor Yellow
}

# Recommend a preset based on VRAM tier
$recommended = @()
if     ($vramMB -le 0)        { $tier = "cpu"   ; $recommended = @("tinyllama","gemma3:1b") }
elseif ($vramMB -lt 4500)     { $tier = "4GB"   ; $recommended = @("llama3.2:3b","phi3:mini","tinyllama") }
elseif ($vramMB -lt 7000)     { $tier = "6GB"   ; $recommended = @("llama3.2:3b","phi3:mini","qwen2.5:3b","gemma2:2b") }
elseif ($vramMB -lt 10000)    { $tier = "8GB"   ; $recommended = @("llama3.1:8b","qwen2.5:7b","mistral:7b") }
elseif ($vramMB -lt 14000)    { $tier = "12GB"  ; $recommended = @("llama3.1:8b","qwen2.5:14b","mistral-nemo") }
elseif ($vramMB -lt 22000)    { $tier = "16GB"  ; $recommended = @("qwen2.5:14b","llama3.1:8b","mistral-nemo") }
else                          { $tier = "24GB+" ; $recommended = @("llama3.1:8b","qwen2.5:32b","llama3.3:70b-q4") }

Write-Host ""
Write-Host ("  Recommended preset: {0}" -f $tier) -ForegroundColor Cyan
Write-Host ("  Suggested models  : {0}" -f ($recommended -join ", ")) -ForegroundColor Cyan

# ── [2/4] Python venv ──────────────────────────────────────────
Write-Host ""
Write-Host "[2/4] Setting up Python venv (.\.venv) ..." -ForegroundColor Green
$VenvPath = "$ProjectDir\.venv"
$VenvPython = "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pyExe) { $pyExe = (Get-Command py -ErrorAction SilentlyContinue).Source }
    if (-not $pyExe) {
        Write-Host "  ERROR: Python not found on PATH. Install Python 3.10+ from https://python.org" -ForegroundColor Red
        exit 1
    }
    & $pyExe -m venv $VenvPath
}
Write-Host "  Installing dependencies (this can take a minute)..." -ForegroundColor Yellow
& "$VenvPath\Scripts\pip.exe" install --upgrade pip | Out-Null
& "$VenvPath\Scripts\pip.exe" install -r "$ProjectDir\requirements.txt"
Write-Host "  [OK] Dependencies installed." -ForegroundColor Green

# ── [3/4] Ollama ───────────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Checking Ollama ..." -ForegroundColor Green
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Host "  Ollama is NOT installed." -ForegroundColor Yellow
    Write-Host "  Download and install from: https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "  Then re-run this script to pull models." -ForegroundColor Yellow
} else {
    $ver = & ollama --version 2>&1
    Write-Host ("  Found: {0}" -f $ver) -ForegroundColor White
}

# ── [4/4] Optional model pull ──────────────────────────────────
Write-Host ""
Write-Host "[4/4] Pull recommended models? (will use Ollama's default location)" -ForegroundColor Green
if ($env:OLLAMA_MODELS) {
    Write-Host ("       Models will be stored at: {0}" -f $env:OLLAMA_MODELS) -ForegroundColor Gray
}
$ans = Read-Host "  Pull [$($recommended -join ', ')]? (y/N)"
if ($ans -match '^[Yy]$' -and $ollamaCmd) {
    foreach ($m in $recommended) {
        Write-Host ("  Pulling {0} ..." -f $m) -ForegroundColor Yellow
        & ollama pull $m
    }
}

# Create .env from example if missing
if (-not (Test-Path "$ProjectDir\.env") -and (Test-Path "$ProjectDir\.env.example")) {
    Copy-Item "$ProjectDir\.env.example" "$ProjectDir\.env"
    Write-Host "  Created .env from .env.example" -ForegroundColor Gray
}

Write-Host ""
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Start the server:  .\run.ps1" -ForegroundColor Yellow
Write-Host "  Then open       :  http://localhost:8000/dashboard" -ForegroundColor Yellow
Write-Host ""
