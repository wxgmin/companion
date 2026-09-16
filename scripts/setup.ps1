<#
    Fresh-machine setup for the companion.

    Installs only what is missing and is safe to re-run. GPU dependencies are
    deliberately installed LAST and from the CUDA index, because a plain
    `pip install torch` on Windows resolves to a CPU-only wheel, which then
    fails at runtime with a confusing "cuda not available".

    Usage:
        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
        powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipModels
#>

[CmdletBinding()]
param(
    [switch]$SkipModels,
    [string]$ModelHome = "$env:USERPROFILE\LocalLLMs"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VoiceHome = Join-Path $ModelHome "Voice Models"

function Write-Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Cyan
}
function Write-Ok($msg)   { Write-Host "      OK   $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "      WARN $msg" -ForegroundColor Yellow }
function Write-Err2($msg)  { Write-Host "      FAIL $msg" -ForegroundColor Red }
function Have($exe) { return [bool](Get-Command $exe -ErrorAction SilentlyContinue) }

Write-Host "=" * 62
Write-Host "  companion setup"
Write-Host "=" * 62
Write-Host "  project : $Root"
Write-Host "  models  : $ModelHome"

$total = if ($SkipModels) { 5 } else { 7 }
$n = 0

# ---------------------------------------------------------------- python
$n++
Write-Step $n $total "python"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Err2 "python not found - install Python 3.11 or 3.12 and re-run"
    exit 1
}
$ver = (& python --version) 2>&1
Write-Ok "$ver at $($py.Source)"

# ----------------------------------------------------------------- ollama
$n++
Write-Step $n $total "ollama"
if (Have "ollama") {
    Write-Ok "found: $((Get-Command ollama).Source)"
} else {
    Write-Err2 "ollama not found - install from https://ollama.com/download"
    exit 1
}

# --------------------------------------------------------------- ffmpeg
$n++
Write-Step $n $total "ffmpeg"
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ff) {
    Write-Ok "found: $($ff.Source)"
    Write-Warn2 "ffmpeg must be a SHARED build - static builds break torchcodec"
} else {
    Write-Warn2 "ffmpeg not on PATH - needed for audio decoding"
    Write-Host "           winget install --id BtbN.FFmpeg.GPL.Shared"
}

# ------------------------------------------------------- orchestrator venv
$n++
Write-Step $n $total "orchestrator environment (CPU-only, no torch)"
$orcVenv = Join-Path $Root "orchestrator\.venv"
if (-not (Test-Path $orcVenv)) {
    & python -m venv $orcVenv
    Write-Ok "created $orcVenv"
} else {
    Write-Ok "already exists"
}
$orcPy = Join-Path $orcVenv "Scripts\python.exe"
& $orcPy -m pip install --quiet --upgrade pip
& $orcPy -m pip install --quiet `
    fastapi "uvicorn[standard]" faster-whisper sounddevice soundfile numpy `
    httpx websockets webrtcvad-wheels python-multipart
Write-Ok "dependencies installed"

& $orcPy -c "import faster_whisper, sounddevice, fastapi; print('      OK   imports verified')"
if ($LASTEXITCODE -ne 0) { Write-Err2 "import check failed"; exit 1 }

if (-not $SkipModels) {
    # ------------------------------------------------------------- models
    $n++
    Write-Step $n $total "language model (~16 GB)"
    $model = "orcarouter/Qwen3.8-27B-Uncensored:iq4_xs"
    $have = (& ollama list 2>&1 | Select-String -SimpleMatch $model)
    if ($have) {
        Write-Ok "$model present"
    } else {
        Write-Host "      pulling $model ..."
        & ollama pull $model
        if ($LASTEXITCODE -ne 0) { Write-Err2 "pull failed"; exit 1 }
        Write-Ok "pulled"
    }

    # ------------------------------------------------------------ chatterbox
    $n++
    Write-Step $n $total "voice model (chatterbox)"
    $cbDir = Join-Path $Root "chatterbox"
    if (-not (Test-Path (Join-Path $cbDir ".venv"))) {
        Write-Host "      creating chatterbox environment ..."
        New-Item -ItemType Directory -Force -Path $cbDir | Out-Null
        & python -m venv (Join-Path $cbDir ".venv")
    }
    $cbPy = Join-Path $cbDir ".venv\Scripts\python.exe"
    & $cbPy -m pip install --quiet --upgrade pip
    # order matters: torch is installed from the CUDA index LAST so the CPU
    # wheel pulled in as a transitive dependency gets replaced.
    & $cbPy -m pip install --quiet chatterbox-tts
    & $cbPy -m pip install --quiet --force-reinstall `
        --index-url https://download.pytorch.org/whl/cu126 torch torchaudio
    Write-Ok "chatterbox installed with CUDA torch"

    & $cbPy -c "import torch; assert torch.cuda.is_available(), 'CPU-only torch'; print('      OK   cuda', torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { Write-Err2 "torch is CPU-only - re-run this step"; exit 1 }
}

# ---------------------------------------------------------------- task agent
$n++
Write-Step $n $total "task agent (hermes)"
$hermesExe = Get-Command hermes -ErrorAction SilentlyContinue
if (-not $hermesExe) {
    Write-Warn2 "hermes not on PATH - she can talk but cannot do tasks"
    Write-Host "           install from https://github.com/NousResearch/hermes"
} else {
    Write-Ok "found: $($hermesExe.Source)"

    # Install the command-code skill so Hermes can hand coding work to cmdc.
    $hermesHome = if ($env:HERMES_HOME) { $env:HERMES_HOME } else { Join-Path $env:LOCALAPPDATA "hermes" }
    $skillSrc = Join-Path $Root "integrations\hermes\command-code\SKILL.md"
    $skillDst = Join-Path $hermesHome "skills\software-development\command-code"
    if ((Test-Path $skillSrc) -and (Test-Path $hermesHome)) {
        New-Item -ItemType Directory -Force -Path $skillDst | Out-Null
        Copy-Item $skillSrc (Join-Path $skillDst "SKILL.md") -Force
        Write-Ok "installed command-code skill"
    } else {
        Write-Warn2 "could not install the skill - Hermes home not found at $hermesHome"
    }

    if (Get-Command cmdc -ErrorAction SilentlyContinue) {
        Write-Ok "cmdc found - Hermes can delegate coding work"
    } else {
        Write-Warn2 "cmdc not on PATH - coding delegation from Hermes will not work"
        Write-Host "           npm install -g command-code"
    }
}

# ---------------------------------------------------------------- voice
Write-Host ""
Write-Host "=" * 62
Write-Host "  setup complete" -ForegroundColor Green
Write-Host "=" * 62
Write-Host ""
Write-Host "  NEXT: add her voice and chat history, then start."
Write-Host ""
Write-Host "  1. Put a 3-10 second clean clip of her voice at:"
Write-Host "       data\voice\reference.wav"
Write-Host "     (a plain voice note is fine; music and other speakers are not)"
Write-Host ""
Write-Host "  2. Put exported chats under data\persona\ and build the timeline:"
Write-Host "       powershell -File scripts\build_persona.ps1 -PersonaNames `"her name`""
Write-Host ""
Write-Host "  3. Check the stack, then start everything:"
Write-Host "       orchestrator\.venv\Scripts\python.exe orchestrator\verify_stack.py"
Write-Host "       companion.cmd --voice"
Write-Host ""
