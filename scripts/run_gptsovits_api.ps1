# Launch the GPT-SoVITS TTS API server (api_v2) on port 9880.
#
# Why this script exists instead of just calling python directly:
#
# torchaudio 2.9+ decodes audio through torchcodec, which loads FFmpeg's shared
# libraries at runtime. winget's default FFmpeg (Gyan) is a STATIC 'full_build'
# -- it ships ffmpeg.exe but no avcodec DLLs -- so loading fails with:
#
#     Could not load libtorchcodec ... libtorchcodec_core9.dll (or one of its dependencies)
#
# Upstream GPT-SoVITS avoids this by installing FFmpeg from conda, which is a
# shared build. We instead install BtbN's shared FFmpeg and put its bin dir on
# PATH here, before the interpreter starts.

$ErrorActionPreference = 'Stop'

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Shared paths + HF_HOME + the shared-FFmpeg PATH fix (torchcodec needs it).
. "$PSScriptRoot\paths.ps1"

$RepoRoot = $GptSovitsRepo
$Python = $GptSovitsPython

if (-not (Test-Path $Python)) {
    throw "GPT-SoVITS venv not found at $Python"
}

# Find the shared FFmpeg bin dir (contains ffmpeg.exe AND the av* DLLs).
$ffBin = $null
$pkgRoot = 'C:\Users\Waiz\AppData\Local\Microsoft\WinGet\Packages'
if (Test-Path $pkgRoot) {
    $ffBin = Get-ChildItem $pkgRoot -Directory -Filter 'BtbN.FFmpeg*Shared*' -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
        ForEach-Object { Join-Path $_.FullName 'bin' } |
        Where-Object { Test-Path (Join-Path $_ 'ffmpeg.exe') } |
        Select-Object -First 1
}

if ($ffBin) {
    $env:Path = "$ffBin;$env:Path"
    $dlls = (Get-ChildItem $ffBin -Filter 'av*.dll' -ErrorAction SilentlyContinue).Count
    Write-Host "[ffmpeg] shared build on PATH: $ffBin  ($dlls av* DLLs)"
}
else {
    Write-Warning "Shared FFmpeg not found - torchcodec will likely fail to decode reference audio."
    Write-Warning "Install it with: winget install BtbN.FFmpeg.GPL.Shared.9.0"
}

Set-Location $RepoRoot
Write-Host "[tts] starting api_v2 on 127.0.0.1:9880 (Ctrl+C to stop)"
& $Python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
