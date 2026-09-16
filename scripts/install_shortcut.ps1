# Put a one-click "Her" shortcut on the Desktop.
#
#   pwsh -File scripts\install_shortcut.ps1 [-Name "Her"]
#
# The shortcut runs the launcher windowed-but-minimised rather than hidden: if
# something needs a password, an install, or a port is taken, the reason has to
# be visible. A hidden window would make a silent failure look like a no-op.
#
# -WindowStyle Minimized keeps that log available without covering the desktop,
# and the launcher itself exits once AIRI is up.

param(
    [string]$Name = 'Her'
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot 'start_companion.ps1'
$Desktop = [Environment]::GetFolderPath('Desktop')
$Link = Join-Path $Desktop "$Name.lnk"

if (-not (Test-Path $Launcher)) { throw "launcher not found: $Launcher" }

$pwsh = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = (Get-Command powershell).Source }

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($Link)
$sc.TargetPath = $pwsh
$sc.Arguments = "-NoProfile -WindowStyle Minimized -File `"$Launcher`""
$sc.WorkingDirectory = $Root
$sc.Description = 'Start the companion and open AIRI'
$sc.IconLocation = "$env:LOCALAPPDATA\Programs\AIRI\airi.exe,0"
$sc.Save()

Write-Host ''
Write-Host "  shortcut created" -ForegroundColor Green
Write-Host "    $Link"
Write-Host ''
Write-Host "    runs : $pwsh -File `"$Launcher`""
Write-Host "    icon : AIRI"
Write-Host ''
Write-Host '  Double-click it to start Ollama, her voice, and AIRI together.' -ForegroundColor Magenta
Write-Host ''
