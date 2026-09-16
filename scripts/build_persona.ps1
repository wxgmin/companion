<#
    Build the persona training data from raw chat exports.

    Generic version: point it at a folder of exports and tell it who is who.

        pwsh -File scripts\build_persona.ps1 `
            -Source "C:\Users\me\Downloads\Messages" `
            -PersonaNames "her nickname","her other handle" `
            -MyNames "my name","my other handle"

    Directory layout it expects under -Source (subfolders can be nested any way):

        *.txt   WhatsApp "Export chat" output
        *.html  Instagram "Download your information" message exports

    Which name belongs to whom is the single most important input here. The
    persona is the training target; your own messages are context and are never
    trained on. Getting these backwards produces a model that talks like you,
    so the last step verifies the direction mechanically rather than trusting it.

    Steps:
      1. parse WhatsApp .txt             -> data\corpus\whatsapp-*.csv
      2. parse Instagram HTML            -> data\corpus\instagram-*.csv
      3. merge into ONE chronological timeline -> WeClone\dataset\csv
      4. WeClone make-dataset            -> QA pairs
      5. convert to OpenAI JSONL, validate, verify role direction
#>

[CmdletBinding()]
param(
    # Where the raw exports live. Defaults to the project's own data folder.
    [string]$Source,

    # Names that identify HER. Every message matching these becomes the
    # assistant turn that gets trained on.
    [string[]]$PersonaNames = @(),

    # Names that identify YOU. Context only, never trained on.
    [string[]]$MyNames = @('Waiz', 'Waiz Ul Haque'),

    # Where generated files go.
    [string]$Corpus,

    # Skip steps 4-5 and stop after building the merged timeline.
    [switch]$TimelineOnly
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$Root    = Split-Path -Parent $PSScriptRoot
if (-not $Source) { $Source = Join-Path $Root 'data\persona' }
if (-not $Corpus) { $Corpus = Join-Path $Root 'data\corpus' }

$Scripts = Join-Path $Root 'scripts'
$WeClone = Join-Path $Root 'WeClone'
$CsvDir  = Join-Path $WeClone 'dataset\csv'

if (-not (Test-Path $Source)) {
    throw "Source folder not found: $Source"
}
if ($PersonaNames.Count -eq 0) {
    throw "Pass -PersonaNames with the name(s) that identify her in the exports."
}

# WeClone runs the data pipeline; it needs its own environment.
$Wpy = Join-Path $WeClone '.venv\Scripts\python.exe'
if (-not (Test-Path $Wpy)) {
    throw "WeClone environment missing at $Wpy - see docs\GOTCHAS.md on the Windows patch."
}

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Note($msg)      { Write-Host "      $msg" -ForegroundColor DarkGray }

New-Item -ItemType Directory -Force -Path $Corpus | Out-Null
New-Item -ItemType Directory -Force -Path $CsvDir | Out-Null

Write-Host "source  : $Source"
Write-Host "persona : $($PersonaNames -join ', ')"
Write-Host "me      : $($MyNames -join ', ')"

# --- discover ---------------------------------------------------------------
$whatsapp = Get-ChildItem $Source -Recurse -Filter '*.txt' -File |
    Where-Object { $_.Name -like 'WhatsApp Chat with*' -or $_.Name -eq '_chat.txt' }
$instagram = Get-ChildItem $Source -Recurse -Filter '*.html' -File |
    Where-Object { $_.FullName -match 'message' }

Note "found $($whatsapp.Count) whatsapp file(s), $($instagram.Count) instagram file(s)"
if ($whatsapp.Count -eq 0 -and $instagram.Count -eq 0) {
    throw "No exports found. See data\persona\README.md for what to put there."
}

$csvs = @()

# --- 1. WhatsApp ------------------------------------------------------------
# The WhatsApp parser takes only --me: a two-party export means everyone who is
# not you is her, so there is no --persona flag to pass.
if ($whatsapp.Count -gt 0) {
    Step '1/5' 'Parsing WhatsApp'
    $i = 0
    foreach ($wa in $whatsapp) {
        $i++
        $out = Join-Path $Corpus "whatsapp-$i.csv"
        & $Wpy "$Scripts\parse_whatsapp.py" $wa.FullName `
            --out $out --room "whatsapp-$i" --me $MyNames[0]
        $csvs += $out
    }
} else {
    Step '1/5' 'WhatsApp - nothing to parse'
}

# --- 2. Instagram -----------------------------------------------------------
if ($instagram.Count -gt 0) {
    Step '2/5' 'Parsing Instagram'
    # Instagram exports one HTML per conversation; group by parent folder so
    # each account becomes its own source rather than one merged blob.
    # Both --me and --persona are repeatable here.
    $groups = $instagram | Group-Object { $_.Directory.Name }
    $i = 0
    foreach ($g in $groups) {
        $i++
        $out = Join-Path $Corpus "instagram-$i.csv"
        $args = @("$Scripts\parse_instagram.py", '--out', $out, "--room", "instagram-$i")
        foreach ($m in $MyNames) { $args += @('--me', $m) }
        foreach ($p in $PersonaNames) { $args += @('--persona', $p) }
        $args += $g.Group.FullName
        & $Wpy @args
        $csvs += $out
    }
} else {
    Step '2/5' 'Instagram - nothing to parse'
}

# --- 3. global timeline -----------------------------------------------------
Step '3/5' 'Merging into one global chronological timeline'
Note 'WeClone pairs across the whole concatenated list, so it reads exactly ONE csv.'
Note 'Ordering is (timestamp, source, id) so re-runs are deterministic.'

Get-ChildItem $CsvDir -Directory -EA SilentlyContinue | Remove-Item -Recurse -Force
Get-ChildItem $CsvDir -File -EA SilentlyContinue | Remove-Item -Force

$timeline = Join-Path $CsvDir 'global-timeline\global-timeline.csv'
& $Wpy "$Scripts\merge_timeline.py" --out $timeline @csvs

if ($TimelineOnly) {
    Write-Host "`nTimeline built: $timeline" -ForegroundColor Green
    exit 0
}

# --- 4. QA pairs ------------------------------------------------------------
Step '4/5' 'Running WeClone make-dataset'
Push-Location $WeClone
try {
    & $Wpy -u -m weclone.cli make-dataset
    if ($LASTEXITCODE -ne 0) { throw "make-dataset failed ($LASTEXITCODE)" }
}
finally { Pop-Location }

# --- 5. OpenAI JSONL + verification -----------------------------------------
Step '5/5' 'Converting to JSONL and verifying direction'
$Sft   = Join-Path $WeClone 'dataset\res_csv\sft\sft-my.json'
$Jsonl = Join-Path $Corpus 'openai_train.jsonl'

& $Wpy "$Scripts\to_openai_jsonl.py" --in $Sft --out $Jsonl
Write-Host ''
& $Wpy "$Scripts\validate_openai_jsonl.py" $Jsonl --show 0

Write-Host ''
$verifyArgs = @("$Scripts\verify_role_inversion.py", '--csv', $timeline, '--jsonl', $Jsonl)
foreach ($p in $PersonaNames) { $verifyArgs += @('--her', $p) }
foreach ($m in $MyNames)     { $verifyArgs += @('--him', $m) }
& $Wpy @verifyArgs

Write-Host "`nDataset built: $Jsonl" -ForegroundColor Green
Write-Host "Timeline     : $timeline" -ForegroundColor Green
