# Start the whole ai-companion stack, in the right order.
#
#   pwsh -File scripts\start_all.ps1
#
# Architecture (three local services):
#
#   [ Open-LLM-VTuber :12393 ]  mic -> VAD -> ASR -> LLM -> TTS -> speaker, + Live2D
#        |          |
#        |          +--> [ Chatterbox :8092 ]   her cloned voice (OpenAI-compatible)
#        |
#        +--------------> [ Ollama :11434 ]     local LLM
#
# GPT-SoVITS (:9880) is still installed and can be swapped back in (see the
# commented block in Open-LLM-VTuber/conf.yaml), but Chatterbox is the current
# voice because it measured best on clone fidelity.
#
# VRAM BUDGET on a 24 GB card — this is the thing that bites:
#   Ollama Qwen3.8-27B  ~15 GB   (releases after 300 s idle; do NOT set keep_alive -1)
#   Chatterbox           ~4 GB
#   Open-LLM-VTuber      ~0.5 GB (Whisper CUDA + Silero)
#   Running all three at once is ~20 GB. Running a big TTS model INSTEAD of
#   Ollama needs Ollama unloaded first:  ollama stop <model>

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\paths.ps1"

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Port([int]$Port, [int]$TimeoutSec, [string]$What) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) { return $true }
        Start-Sleep -Milliseconds 750
    }
    Write-Warning "$What did not open port $Port within ${TimeoutSec}s."
    return $false
}

function Show-Vram {
    $line = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) -join ''
    Write-Host "  VRAM: $line"
}

# --- 1. Ollama (local LLM) -------------------------------------------------
if (Test-Port 11434) {
    Write-Host '[1/3] Ollama already listening on 11434.'
}
else {
    Write-Host '[1/3] Starting Ollama...'
    Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Minimized
    if (-not (Wait-Port 11434 30 'Ollama')) { Write-Warning 'Continuing anyway.' }
}

# --- 2. Chatterbox TTS -----------------------------------------------------
if (Test-Port 8092) {
    Write-Host '[2/3] Chatterbox already listening on 8092.'
}
else {
    Write-Host '[2/3] Starting Chatterbox (encodes the voice prompt once, ~30 s)...'
    Start-Process -FilePath 'pwsh' -ArgumentList '-NoExit', '-File', "$PSScriptRoot\run_chatterbox_server.ps1"
    if (-not (Wait-Port 8092 180 'Chatterbox')) { Write-Warning 'Continuing anyway.' }
}

# --- 3. Open-LLM-VTuber ---------------------------------------------------
if (Test-Port 12393) {
    Write-Host '[3/3] Open-LLM-VTuber already listening on 12393.'
}
else {
    Write-Host '[3/3] Starting Open-LLM-VTuber...'
    Start-Process -FilePath 'pwsh' `
        -ArgumentList '-NoExit', '-Command', "Set-Location '$OlvRepo'; `$env:PYTHONUTF8='1'; & '$OlvPython' -u run_server.py"
    if (-not (Wait-Port 12393 120 'Open-LLM-VTuber')) { Write-Warning 'Continuing anyway.' }
}

Write-Host ''
Write-Host 'Stack status:'
foreach ($p in @(@(11434, 'Ollama (LLM)'), @(8092, 'Chatterbox (voice)'), @(12393, 'Open-LLM-VTuber'))) {
    $state = if (Test-Port $p[0]) { 'UP  ' } else { 'DOWN' }
    Write-Host "  $state  $($p[0])  $($p[1])"
}
Show-Vram
Write-Host ''
Write-Host 'Open the UI at http://localhost:12393'
