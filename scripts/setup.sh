#!/usr/bin/env bash
# setup.sh - bootstrap the private-rag-assistant dev environment (bash / Unix)
# Usage: ./scripts/setup.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$root/.venv" ]; then
    python3 -m venv "$root/.venv"
fi

"$root/.venv/bin/pip" install --upgrade pip
"$root/.venv/bin/pip" install -r "$root/requirements.txt"

# Optional: editable install exposes src/ as an importable package and pulls
# in the [dev] extras (pytest, etc). Not every project's pyproject.toml is
# configured for this yet, so failure here is non-fatal - the launch.json
# configs set PYTHONPATH=src directly and don't depend on it.
"$root/.venv/bin/pip" install -e "$root[dev]" || echo "(editable install skipped - requirements.txt install above already succeeded)"

if [ -f "$root/.env.example" ] && [ ! -f "$root/.env" ]; then
    cp "$root/.env.example" "$root/.env"
fi

echo "Setup complete. Activate with: source $root/.venv/bin/activate"
