# setup.ps1 - bootstrap the private-rag-assistant dev environment (Windows PowerShell)
# Usage: ./scripts/setup.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path "$root\.venv")) {
    python -m venv "$root\.venv"
}

& "$root\.venv\Scripts\pip.exe" install --upgrade pip
& "$root\.venv\Scripts\pip.exe" install -r "$root\requirements.txt"

# Optional: editable install exposes src/ as an importable package and pulls
# in the [dev] extras (pytest, etc). Not every project's pyproject.toml is
# configured for this yet, so failure here is non-fatal - the launch.json
# configs set PYTHONPATH=src directly and don't depend on it.
& "$root\.venv\Scripts\pip.exe" install -e "$root[dev]" 2>$null
if (-not $?) {
    Write-Host "(editable install skipped - requirements.txt install above already succeeded)"
}

if ((Test-Path "$root\.env.example") -and -not (Test-Path "$root\.env")) {
    Copy-Item "$root\.env.example" "$root\.env"
}

Write-Host "Setup complete. Activate with: $root\.venv\Scripts\Activate.ps1"
