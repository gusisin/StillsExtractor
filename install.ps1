<#
  360 Cubemap Stills Extractor — lightweight installer (Windows)

  Usage:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    .\install.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-ErrorLine($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot
Write-Info "Project root: $ScriptRoot"

$venvPath = Join-Path $ScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Info "Creating virtual environment..."
    python -m venv $venvPath
} else {
    Write-Info "Virtual environment already exists."
}

$python = Join-Path $venvPath "Scripts\python.exe"
$pip = Join-Path $venvPath "Scripts\pip.exe"
Write-Info "Installing Python dependencies..."
& $python -m pip install --upgrade pip
& $pip install -r (Join-Path $ScriptRoot "requirements.txt")

foreach ($tool in @("ffmpeg", "ffprobe")) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        Write-Info "Found $tool on PATH."
    } else {
        Write-Warn "$tool not found on PATH. Install ffmpeg and add it to PATH before extracting."
    }
}

if (Get-Command magick -ErrorAction SilentlyContinue) {
    Write-Info "Found ImageMagick (magick) — optional metadata strip will use it."
} else {
    Write-Warn "ImageMagick not found — metadata strip falls back to Pillow (slower)."
}

Write-Info "Done. Activate and run:"
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "  python MP4_360_stills.py" -ForegroundColor Green
