# A/B the GPT-SoVITS model versions against multiple reference clips.
#
#   pwsh -File scripts\ab_voice_models.ps1
#
# A full server restart per model version is deliberate. Initialising different
# weights via the /set_sovits_weights endpoint does NOT clear `prompt_cache`
# (TTS.py init_vits_weights), so the reference embeddings would still come from
# the previous model and the comparison would be meaningless.
#
# Produces  {model}__{ref}.wav  for every combination in data\voice\_work\ab\

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Root    = 'C:\Users\Waiz\ai-companion'
$Repo    = Join-Path $Root 'GPT-SoVITS'
$Python  = Join-Path $Repo '.venv\Scripts\python.exe'
$TtsTest = Join-Path $Root 'scripts\test_tts.py'
$Cand    = Join-Path $Root 'data\voice\_work\candidates'
$Out     = Join-Path $Root 'data\voice\_work\ab'

New-Item -ItemType Directory -Force -Path $Out | Out-Null

# --- what to test ----------------------------------------------------------
$Models = @('v2', 'v2Pro', 'v2ProPlus', 'v4')

$Refs = @(
    @{ Id = 'D';  File = "$Cand\norm_D_dereverb_7.02s.wav";
       Prompt = 'We were alone and the waiter thought we would be bored without the music.' },
    @{ Id = 'A';  File = "$Cand\norm_A_dereverb_6.75s.wav";
       Prompt = 'She is vibing. She was literally vibing and then when I asked her, she said no.' }
)

$Line = 'Okay so I was just thinking about what you said earlier and honestly I dont think you were wrong about any of it.'

# --- helpers ---------------------------------------------------------------
function Stop-Tts {
    $c = Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue
    if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }
    for ($i = 0; $i -lt 20; $i++) {
        if (-not (Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 400
    }
}

function Start-Tts([string]$Version) {
    $cfg = Join-Path $Repo "GPT_SoVITS\configs\tts_infer_$Version.yaml"
    if (-not (Test-Path $cfg)) { throw "missing config: $cfg" }

    # shared FFmpeg must be on PATH for torchcodec
    $ffBin = Get-ChildItem 'C:\Users\Waiz\AppData\Local\Microsoft\WinGet\Packages' -Directory -Filter 'BtbN.FFmpeg*Shared*' -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
        ForEach-Object { Join-Path $_.FullName 'bin' } |
        Where-Object { Test-Path (Join-Path $_ 'ffmpeg.exe') } | Select-Object -First 1
    if ($ffBin) { $env:Path = "$ffBin;$env:Path" }

    $env:HF_HOME = 'C:\Users\Waiz\LocalLLMs\Voice Models\cache\huggingface'

    Start-Process -FilePath 'pwsh' -WindowStyle Minimized -ArgumentList @(
        '-NoProfile', '-Command',
        "Set-Location '$Repo'; `$env:PYTHONUTF8='1'; & '$Python' api_v2.py -a 127.0.0.1 -p 9880 -c '$cfg' *> '$Out\server_$Version.log'"
    )

    for ($i = 0; $i -lt 90; $i++) {
        if (Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue) {
            Start-Sleep -Seconds 3   # let the pipeline settle before first request
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "server for $Version did not come up"
}

# --- run -------------------------------------------------------------------
foreach ($m in $Models) {
    Write-Host "`n================ $m ================" -ForegroundColor Cyan
    Stop-Tts
    try { Start-Tts $m } catch { Write-Host "  SKIP $m : $_" -ForegroundColor Red; continue }

    foreach ($r in $Refs) {
        $outFile = Join-Path $Out "$m`__$($r.Id).wav"
        Write-Host "  generating $m / ref $($r.Id)"
        & $Python $TtsTest --ref $r.File --prompt-text $r.Prompt --text $Line --out $outFile 2>&1 |
            Select-String -Pattern 'OK in|HTTP|audio:|Error' | ForEach-Object { "      $_" }
    }
}

Stop-Tts
Write-Host "`n=== produced ===" -ForegroundColor Green
Get-ChildItem $Out -Filter '*.wav' | Select-Object Name, @{n='KB';e={[math]::Round($_.Length/1KB,0)}} | Format-Table -AutoSize
