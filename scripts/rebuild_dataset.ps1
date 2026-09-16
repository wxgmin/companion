# Rebuild the fine-tuning dataset from the raw chat exports, end to end.
#
#   pwsh -File scripts\rebuild_dataset.ps1
#
# Steps:
#   1. parse WhatsApp .txt            -> data\corpus\whatsapp-chat.csv
#   2. parse Instagram HTML (x2 accts) -> data\corpus\instagram-acc*.csv
#   3. merge into ONE global chronological timeline -> WeClone\dataset\csv
#   4. WeClone make-dataset            -> conversations with QA pairs
#   5. convert to OpenAI JSONL + validate + verify role direction
#
# Re-run this whenever fresh exports are dropped in.

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# --- paths -----------------------------------------------------------------
$Root    = 'C:\Users\Waiz\ai-companion'
$Src     = 'C:\Users\Waiz\Downloads\kano\Messages'
$Scripts = Join-Path $Root 'scripts'
$Corpus  = Join-Path $Root 'data\corpus'
$WeClone = Join-Path $Root 'WeClone'
$CsvDir  = Join-Path $WeClone 'dataset\csv'

$Wpy = Join-Path $WeClone '.venv\Scripts\python.exe'      # data pipeline
$Opy = Join-Path $Root 'Open-LLM-VTuber\.venv\Scripts\python.exe'  # neither used here

# --- identity --------------------------------------------------------------
# The persona (her) is is_sender=1 / the training target.
# "Waiz" is is_sender=0 / context only and is NEVER trained on.
$Him  = @('Waiz', 'Waiz Ul Haque')
$Her  = @('Zayy-Nabb (ง''̀-''́)ง 🍟', 'notyourfavZE', 'Zainab Afridi ~ ZE')

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }

New-Item -ItemType Directory -Force -Path $Corpus | Out-Null
New-Item -ItemType Directory -Force -Path $CsvDir | Out-Null

# --- 1. WhatsApp -----------------------------------------------------------
Step '1/5' 'Parsing WhatsApp export'
$wa = Get-ChildItem $Src -Filter 'WhatsApp Chat with*.txt' -File | Select-Object -First 1
if (-not $wa) { throw "no WhatsApp .txt found in $Src" }
& $Wpy "$Scripts\parse_whatsapp.py" $wa.FullName `
    --out "$Corpus\whatsapp-chat.csv" --room 'whatsapp' --me 'Waiz'

# --- 2. Instagram ----------------------------------------------------------
Step '2/5' 'Parsing Instagram exports'
& $Wpy "$Scripts\parse_instagram.py" `
    --out "$Corpus\instagram-acc1.csv" --room 'instagram-acc1' `
    --me 'Waiz Ul Haque' --persona 'notyourfavZE' `
    "$Src\Insta messsages Acc 1\message_1.html" `
    "$Src\Insta messsages Acc 1\message_2.html"

& $Wpy "$Scripts\parse_instagram.py" `
    --out "$Corpus\instagram-acc2.csv" --room 'instagram-acc2' `
    --me 'Waiz Ul Haque' --persona 'Zainab Afridi ~ ZE' `
    "$Src\Insta messsages Acc 2\message_1.html"

# --- 3. global timeline ----------------------------------------------------
Step '3/5' 'Merging into one global chronological timeline'
# WeClone pairs across the whole concatenated list, so it reads exactly ONE csv.
# Per-conversation separation is handled by room_name + the match_qa patch.
Get-ChildItem $CsvDir -Directory | Remove-Item -Recurse -Force -EA SilentlyContinue
Get-ChildItem $CsvDir -File | Remove-Item -Force -EA SilentlyContinue

& $Wpy "$Scripts\merge_timeline.py" `
    --out "$CsvDir\global-timeline\global-timeline.csv" `
    "$Corpus\whatsapp-chat.csv" "$Corpus\instagram-acc1.csv" "$Corpus\instagram-acc2.csv"

# --- 4. QA pairs -----------------------------------------------------------
Step '4/5' 'Running WeClone make-dataset'
Push-Location $WeClone
try {
    & $Wpy -u -m weclone.cli make-dataset
    if ($LASTEXITCODE -ne 0) { throw "make-dataset failed ($LASTEXITCODE)" }
}
finally { Pop-Location }

# --- 5. OpenAI JSONL -------------------------------------------------------
Step '5/5' 'Converting to OpenAI JSONL'
$Sft    = Join-Path $WeClone 'dataset\res_csv\sft\sft-my.json'
$Jsonl  = Join-Path $Corpus 'openai_train.jsonl'
& $Wpy "$Scripts\to_openai_jsonl.py" --in $Sft --out $Jsonl

Write-Host ''
& $Wpy "$Scripts\validate_openai_jsonl.py" $Jsonl --show 0

Write-Host ''
& $Wpy "$Scripts\verify_role_inversion.py" --csv "$CsvDir\global-timeline\global-timeline.csv" `
    --jsonl $Jsonl `
    --her $Her[0] --her $Her[1] --her $Her[2] `
    --him 'Waiz' --him 'Waiz Ul Haque'

Write-Host "`nDataset rebuilt: $Jsonl" -ForegroundColor Green
