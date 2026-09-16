# Shared path definitions for the ai-companion scripts.
#
# Dot-source this at the top of a script:
#     . "$PSScriptRoot\paths.ps1"
#
# Scripts used to hardcode model paths, so reorganising the store broke five of
# them at once. Define once, here.

$Project    = 'C:\Users\Waiz\ai-companion'
$ModelStore = 'C:\Users\Waiz\LocalLLMs\Voice Models'

$GptSovitsWeights = Join-Path $ModelStore 'TTS - Voice Cloning\GPT-SoVITS\pretrained_models'
$Uvr5Weights      = Join-Path $ModelStore 'TTS - Voice Cloning\GPT-SoVITS\uvr5_weights'
$Qwen3TtsModels   = Join-Path $ModelStore 'TTS - Voice Cloning\Qwen3-TTS'
$WhisperModel     = Join-Path $ModelStore 'ASR - Speech to Text\faster-whisper-large-v3-turbo'

# Keep HuggingFace downloads inside the store rather than scattering caches
# across project folders and the user profile.
$env:HF_HOME = Join-Path $ModelStore 'cache\huggingface'
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

# Repos
$GptSovitsRepo = Join-Path $Project 'GPT-SoVITS'
$OlvRepo       = Join-Path $Project 'Open-LLM-VTuber'
$WeCloneRepo   = Join-Path $Project 'WeClone'
$Qwen3TtsRepo  = Join-Path $Project 'qwen3-tts'
$ChatterboxRepo = Join-Path $Project 'chatterbox'

# Voice reference (her cloned voice)
$VoiceRef = Join-Path $Project 'data\voice\reference.wav'

# Venv interpreters
$GptSovitsPython = Join-Path $GptSovitsRepo '.venv\Scripts\python.exe'
$OlvPython       = Join-Path $OlvRepo '.venv\Scripts\python.exe'
$WeClonePython   = Join-Path $WeCloneRepo '.venv\Scripts\python.exe'
$Qwen3TtsPython  = Join-Path $Qwen3TtsRepo '.venv\Scripts\python.exe'
$ChatterboxPython = Join-Path $ChatterboxRepo '.venv\Scripts\python.exe'

# Put the SHARED FFmpeg build on PATH: torchcodec needs the shared libraries,
# not the static build winget installs by default.
$ffBin = Get-ChildItem 'C:\Users\Waiz\AppData\Local\Microsoft\WinGet\Packages' -Directory -Filter 'BtbN.FFmpeg*Shared*' -ErrorAction SilentlyContinue |
    ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
    ForEach-Object { Join-Path $_.FullName 'bin' } |
    Where-Object { Test-Path (Join-Path $_ 'ffmpeg.exe') } | Select-Object -First 1
if ($ffBin) { $env:Path = "$ffBin;$env:Path" }
