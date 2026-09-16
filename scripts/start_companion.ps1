# Start the whole companion in one step, ending with AIRI as the front-end.
#
#   pwsh -File scripts\start_companion.ps1 [-SkipAiriConfig] [-NoAiri]
#
# Order matters and is the reason this is a script rather than a list of
# commands to paste:
#
#   1. Ollama first - the 27B takes the longest to become usable, and AIRI
#      hitting an unloaded model looks like a hang.
#   2. Voice server next - it encodes her voice prompt once at startup, and
#      that ~10-40s has to finish before AIRI asks for speech.
#   3. AIRI last, so it finds everything already answering.
#
# Each step is idempotent: anything already listening is left alone, so running
# this twice does not spawn duplicates.

param(
    [switch]$SkipAiriConfig,   # do not rewrite AIRI's settings
    [switch]$NoAiri            # start services only (no UI)
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'

$Root = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\paths.ps1"

function Test-Port([int]$Port) {
    [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Port([int]$Port, [int]$Seconds, [string]$What) {
    for ($i = 0; $i -lt $Seconds; $i++) {
        if (Test-Port $Port) { Write-Host "  $What ready on :$Port"; return $true }
        Start-Sleep -Seconds 1
    }
    Write-Warning "$What did not come up on :$Port within ${Seconds}s"
    return $false
}

Write-Host ''
Write-Host '  companion' -ForegroundColor Magenta
Write-Host '  ---------'

# --- 1. Ollama ------------------------------------------------------------
Write-Host ''
Write-Host '[1/3] Ollama (local LLM)' -ForegroundColor Cyan
if (Test-Port 11434) {
    Write-Host '  already running'
} else {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollama) {
        Start-Process -FilePath $ollama.Source -ArgumentList 'serve' -WindowStyle Hidden
    } else {
        Write-Warning '  ollama not on PATH - start it manually or install it'
    }
    Wait-Port 11434 60 'ollama' | Out-Null
}

# --- 2. Voice server (speech + hearing on one port) -----------------------
Write-Host ''
Write-Host '[2/3] Voice server (her cloned voice + hearing)' -ForegroundColor Cyan
if (Test-Port 8092) {
    Write-Host '  already running'
} else {
    if (-not (Test-Path $ChatterboxPython)) { throw "voice venv missing: $ChatterboxPython" }
    if (-not (Test-Path $VoiceRef)) { throw "voice reference missing: $VoiceRef" }

    Start-Process -FilePath $ChatterboxPython `
        -ArgumentList '-u', "$PSScriptRoot\chatterbox_server.py", '--ref', $VoiceRef, '--exaggeration', '0.8' `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Root 'data\chatterbox.log') `
        -RedirectStandardError (Join-Path $Root 'data\chatterbox.err')
    # Encodes the voice prompt before it can answer, so this is a real wait.
    Wait-Port 8092 180 'voice server' | Out-Null
}

# --- 3. AIRI --------------------------------------------------------------
Write-Host ''
Write-Host '[3/3] AIRI (3D avatar)' -ForegroundColor Cyan

$airiExe = "$env:LOCALAPPDATA\Programs\AIRI\airi.exe"
$airiCfg = Join-Path $Root 'tools\airi-config'

# Write AIRI's settings only when they have never been written. The store is
# Chromium Local Storage; AIRI rewrites it on exit, so doing this every launch
# would fight the app for no reason.
$cfgMarker = Join-Path $airiCfg '.configured'
if ($SkipAiriConfig) {
    Write-Host '  settings: skipped (-SkipAiriConfig)'
} elseif (Test-Path $cfgMarker) {
    Write-Host '  settings: already configured'
} else {
    $airiRunning = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'airi|tamagotchi' })
    if ($airiRunning.Count -gt 0) {
        Write-Host '  settings: AIRI is open, leaving its settings alone'
    } elseif (-not (Test-Path "$airiCfg\node_modules\classic-level")) {
        Write-Host '  settings: installing the leveldb reader (once)...'
        Push-Location $airiCfg
        npm install --silent --no-audit --no-fund 2>&1 | Out-Null
        Pop-Location
    }
    if (Test-Path "$airiCfg\node_modules\classic-level") {
        Push-Location $airiCfg
        node configure_airi.mjs 2>&1 | ForEach-Object { "    $_" }
        Pop-Location
        New-Item -ItemType File -Force -Path $cfgMarker | Out-Null
    } else {
        Write-Warning '  could not install the leveldb reader; configure AIRI from its UI'
    }
}

if ($NoAiri) {
    Write-Host '  not launching (-NoAiri)'
} elseif (Test-Path $airiExe) {
    $running = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'airi|tamagotchi' })
    if ($running.Count -gt 0) {
        Write-Host '  already running'
    } else {
        Start-Process -FilePath $airiExe
        Write-Host '  launching...'
    }
} else {
    Write-Warning "  AIRI not found at $airiExe"
}

Write-Host ''
Write-Host '  services' -ForegroundColor Magenta
foreach ($p in @(
    @{ n = 'ollama      :11434'; v = 11434 },
    @{ n = 'voice       :8092';  v = 8092 }
)) {
    $up = Test-Port $p.v
    $label = if ($up) { 'up' } else { 'DOWN' }
    $colour = if ($up) { 'Green' } else { 'Red' }
    Write-Host ("    {0}  {1}" -f $p.n, $label) -ForegroundColor $colour
}
Write-Host ''
Write-Host '  Talk to her in the AIRI window.' -ForegroundColor Magenta
Write-Host ''
