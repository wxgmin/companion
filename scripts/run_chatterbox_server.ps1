# Serve Chatterbox as an OpenAI-compatible TTS endpoint on :8092.
#
#   pwsh -File scripts\run_chatterbox_server.ps1 [-Exaggeration 0.8]
#
# Open-LLM-VTuber drives this through its `openai_tts` adapter, so the voice is
# switched by config rather than code. The server encodes the reference voice
# ONCE at startup (~26 s) and then reuses it for every line.

param(
    [double]$Exaggeration = 0.8,
    [int]$Port = 8092
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

. "$PSScriptRoot\paths.ps1"

if (-not (Test-Path $ChatterboxPython)) {
    throw "Chatterbox venv not found at $ChatterboxPython"
}
if (-not (Test-Path $VoiceRef)) {
    throw "Voice reference not found at $VoiceRef"
}

Write-Host "[tts] Chatterbox on 127.0.0.1:$Port"
Write-Host "      reference   : $VoiceRef"
Write-Host "      exaggeration: $Exaggeration  (0.6 neutral, 0.8 for stronger emotion)"
Write-Host "      note: 0.8 is a working default; the voice is tuned later."

Set-Location $ChatterboxRepo
& $ChatterboxPython -u "$PSScriptRoot\chatterbox_server.py" `
    --ref $VoiceRef `
    --exaggeration $Exaggeration `
    --port $Port
