# Serve MOSS-TTS via the vLLM-Omni container (GPU passthrough through WSL2).
#
#   pwsh -File scripts\run_moss_tts_server.ps1 [-Model <hf-id>]
#
# Why the container instead of a native env: MOSS-TTS pins transformers 5.0.0 and
# torch 2.9.1+cu128, and vLLM has no Windows wheels at all. The official
# vllm/vllm-omni image already contains a working stack, and it exposes the
# OpenAI-compatible /v1/audio/speech endpoint that Open-LLM-VTuber can drive
# directly via its openai_tts adapter.
#
# VRAM reality on a 24 GB RTX 3090 (from vLLM-Omni's own recipe):
#   MOSS-TTS 8B                 ~18 GB talker + ~8 GB codec = ~26 GB  -> won't fit
#   MOSS-TTS-Realtime 1.7B      ~6 GB  talker + ~8 GB codec = ~14 GB  -> fits
#   MOSS-TTS-Local-Transformer-v1.5 4B  ~11 GB total                  -> fits, best SIM
#
# The default here is the Local-Transformer-v1.5 because it has the highest
# speaker-similarity score of the family AND fits. Pass -Model to try another.

param(
    [string]$Model = 'OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5',
    [int]$Port = 8091,
    [string]$Name = 'moss-tts',
    # vLLM's DEFAULT KV-cache allocation is wildly oversized for a TTS workload.
    # Left alone it reserves tens of GiB and the engine dies with:
    #     ValueError: No available memory for the cache blocks
    # which looks like "the model is too big" but is actually configuration.
    # (Confirmed by the AI-Girls Lab DGX Spark write-up: Higgs' ~49 GiB default
    # cache dropped to ~10-12 GiB total once these were bounded.)
    [int]$MaxModelLen = 8192,
    [double]$GpuMemUtil = 0.85
)

$ErrorActionPreference = 'Stop'

. "$PSScriptRoot\paths.ps1"

# Where the container writes HuggingFace downloads. Keeping this inside the
# model store means the weights persist between container runs and stay in the
# labelled location rather than inside a Docker volume.
$HfVolume = Join-Path $ModelStore 'TTS - Voice Cloning\MOSS-TTS\hf-cache'
$VoiceVolume = Join-Path $Project 'data\voice'
New-Item -ItemType Directory -Force -Path $HfVolume | Out-Null

function Assert-Docker {
    docker info --format '{{.ServerVersion}}' *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Docker engine not responding. Starting Docker Desktop...'
        Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 3
            docker info --format '{{.ServerVersion}}' *> $null
            if ($LASTEXITCODE -eq 0) { break }
        }
        if ($LASTEXITCODE -ne 0) { throw 'Docker did not come up' }
    }
}
Assert-Docker

# clear any previous container with this name
$existing = docker ps -a --filter "name=^/$Name$" --format '{{.ID}}' 2>$null
if ($existing) {
    Write-Host "removing previous container '$Name'"
    docker rm -f $Name | Out-Null
}

Write-Host ''
Write-Host "model     : $Model"
Write-Host "port      : $Port"
Write-Host "hf cache  : $HfVolume"
Write-Host "voices    : $VoiceVolume  (mounted read-only at /voices)"
Write-Host ''
Write-Host 'starting container (first run downloads the weights, this is slow)...'
Write-Host ''

# NOTE: use the :latest tag, NOT v0.18.0. The v0.18.0 image predates MOSS-TTS
# support entirely -- it has no moss_tts*.yaml deploy configs and no
# MossTTSDelay/MossTTSLocal registry entries, so vLLM falls through to the
# diffusion loader and dies with:
#     ValueError: Model class MossTTSDelayModel not found in diffusion model registry
$Image = 'vllm/vllm-omni:v0.28.0'

# --allowed-local-media-path lets ref_audio be a file:// URI inside the container
docker run -d --name $Name `
    --gpus all `
    --shm-size 16g `
    -p "${Port}:8091" `
    -v "${HfVolume}:/root/.cache/huggingface" `
    -v "${VoiceVolume}:/voices:ro" `
    -e HF_HOME=/root/.cache/huggingface `
    --entrypoint vllm `
    $Image `
    serve $Model `
    --omni `
    --port 8091 `
    --max-model-len $MaxModelLen `
    --gpu-memory-utilization $GpuMemUtil `
    --allowed-local-media-path /voices `
    --trust-remote-code

Write-Host "container started: $Name"
Write-Host ''
Write-Host 'follow the logs with:'
Write-Host "  docker logs -f $Name"
Write-Host ''
Write-Host 'ready when this answers:'
Write-Host "  curl http://localhost:$Port/v1/audio/voices"
