<#
  Podcaster launcher.

  Loads env.ps1 (MiniMax API key), then starts the local web UI.
    .\run.ps1                 -> http://127.0.0.1:8760
    .\run.ps1 -Port 9000
    .\run.ps1 -Address 0.0.0.0
    .\run.ps1 -Reload         -> auto-reload while developing

  Note: the listen address is called -Address, not -Host: $Host is a
  read-only PowerShell automatic variable and cannot be used as a parameter.
#>
param(
  [int]$Port = 8760,
  [string]$Address = "127.0.0.1",
  [switch]$Reload
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# --- API keys -------------------------------------------------------------
$envFile = Join-Path $here 'env.ps1'
if (Test-Path $envFile) {
    . $envFile
    Write-Host "Loaded env.ps1" -ForegroundColor DarkGray
}
if (-not $env:MINIMAX_APIKEY) {
    Write-Warning "MINIMAX_APIKEY is not set. Add it to env.ps1 or set it in the Settings tab."
}

# --- dependencies ---------------------------------------------------------
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { throw "python was not found on PATH." }

# Pip package name -> import name; the two are not always the same.
$packages = @(
    @{ Pip = 'fastapi';          Import = 'fastapi' },
    @{ Pip = 'uvicorn';          Import = 'uvicorn' },
    @{ Pip = 'httpx';            Import = 'httpx' },
    @{ Pip = 'pypdf';            Import = 'pypdf' },
    @{ Pip = 'beautifulsoup4';   Import = 'bs4' },
    @{ Pip = 'pdfplumber';       Import = 'pdfplumber' },
    @{ Pip = 'lxml';             Import = 'lxml' },
    @{ Pip = 'python-multipart'; Import = 'multipart' }
)

$missing = @()
foreach ($pkg in $packages) {
    # A missing module prints a traceback to stderr, which PowerShell turns
    # into an error record. That is expected here, so the error stream is
    # muted for the check and only the exit code is looked at.
    $ErrorActionPreference = 'Continue'
    python -c "import $($pkg.Import)" 2>$null
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { $missing += $pkg.Pip }
}
if ($missing.Count -gt 0) {
    Write-Host "Installing missing packages: $($missing -join ', ')" -ForegroundColor Yellow
    $ErrorActionPreference = 'Continue'
    python -m pip install --quiet --disable-pip-version-check @missing
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "pip install failed for: $($missing -join ', ')" }
}

# --- ffmpeg ---------------------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg is not on PATH — audio export still works, video export will not."
}

# --- run ------------------------------------------------------------------
$url = "http://${Address}:${Port}"

# A stale server on the same port gives a confusing traceback, so say it plainly.
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Warning "Port $Port is already in use (PID $($busy[0].OwningProcess)). Stop it, or run .\run.ps1 -Port $((8760 + 1))."
    exit 1
}

Write-Host ""
Write-Host "  Podcaster running at  $url" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# $args is an automatic variable too, so the argument list gets its own name.
$uvicornArgs = @('-m', 'uvicorn', 'app:app', '--host', $Address, '--port', "$Port")
if ($Reload) { $uvicornArgs += '--reload' }
python @uvicornArgs
