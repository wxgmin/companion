# Reorganise downloaded models into a clearly labelled store, and re-point the
# junctions that make them visible to the projects.
#
#   pwsh -File scripts\reorganize_model_store.ps1
#
# Layout produced:
#   C:\Users\Waiz\LocalLLMs\Voice Models\
#     TTS - Voice Cloning\
#       GPT-SoVITS\pretrained_models\      (v2 + v2Pro/v2ProPlus/v4 weights)
#       GPT-SoVITS\uvr5_weights\           (vocal/music separation)
#       Qwen3-TTS\                         (0.6B/1.7B Base + Tokenizer)
#     ASR - Speech to Text\
#       faster-whisper-large-v3-turbo\
#     cache\huggingface\                   (HF_HOME, so downloads land here)

$ErrorActionPreference = 'Stop'

$store = 'C:\Users\Waiz\LocalLLMs'
$old = Join-Path $store 'ai-companion'
$new = Join-Path $store 'Voice Models'

$tts = Join-Path $new 'TTS - Voice Cloning'
$gsv = Join-Path $tts 'GPT-SoVITS'
$qwen = Join-Path $tts 'Qwen3-TTS'
$asr = Join-Path $new 'ASR - Speech to Text'
$cache = Join-Path $new 'cache\huggingface'

# stop the TTS server so nothing holds file handles while moving
$conn = Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Write-Host 'stopped GPT-SoVITS server'
}

foreach ($d in @(
        (Join-Path $gsv 'pretrained_models'),
        (Join-Path $gsv 'uvr5_weights'),
        $qwen,
        $asr,
        $cache
    )) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host "`n=== moving models ==="
$moves = @(
    @{ From = (Join-Path $old 'gpt-sovits-pretrained'); To = (Join-Path $gsv 'pretrained_models') },
    @{ From = (Join-Path $old 'uvr5');                  To = (Join-Path $gsv 'uvr5_weights') },
    @{ From = (Join-Path $old 'whisper');               To = (Join-Path $asr 'faster-whisper-large-v3-turbo') }
)
foreach ($m in $moves) {
    if (Test-Path $m.From) {
        Get-ChildItem $m.From -Force | ForEach-Object { Move-Item $_.FullName -Destination $m.To -Force }
        Remove-Item $m.From -Recurse -Force -ErrorAction SilentlyContinue
        $s = (Get-ChildItem $m.To -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        Write-Host ("  {0,9:N0} MB -> {1}" -f ($s / 1MB), ($m.To.Replace($new, '')))
    }
}
if (Test-Path (Join-Path $old 'hf-cache')) {
    Get-ChildItem (Join-Path $old 'hf-cache') -Force -ErrorAction SilentlyContinue |
        ForEach-Object { Move-Item $_.FullName -Destination $cache -Force -ErrorAction SilentlyContinue }
}
if (Test-Path $old) { Remove-Item $old -Recurse -Force -ErrorAction SilentlyContinue; Write-Host '  removed old ai-companion folder' }

Write-Host "`n=== re-pointing junctions ==="
$links = @(
    @{ Link = 'C:\Users\Waiz\ai-companion\GPT-SoVITS\GPT_SoVITS\pretrained_models'; Target = (Join-Path $gsv 'pretrained_models') },
    @{ Link = 'C:\Users\Waiz\ai-companion\GPT-SoVITS\tools\uvr5\uvr5_weights';        Target = (Join-Path $gsv 'uvr5_weights') },
    @{ Link = 'C:\Users\Waiz\ai-companion\Open-LLM-VTuber\models\whisper';           Target = (Join-Path $asr 'faster-whisper-large-v3-turbo') }
)
foreach ($l in $links) {
    if (Test-Path $l.Link) { Remove-Item $l.Link -Recurse -Force -ErrorAction SilentlyContinue }
    $parent = Split-Path $l.Link -Parent
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    New-Item -ItemType Junction -Path $l.Link -Target $l.Target | Out-Null
    $n = (Get-ChildItem $l.Link -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "  OK ($n files)  $(Split-Path $l.Link -Leaf)"
}

Write-Host "`n=== final store ==="
Get-ChildItem $new -Recurse -Directory -Depth 2 | ForEach-Object {
    $s = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($s -gt 0) {
        Write-Host ("  {0,9:N0} MB  {1}" -f ($s / 1MB), ($_.FullName.Replace($new, '')))
    }
}
