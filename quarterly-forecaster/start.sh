#!/usr/bin/env bash
# Linux/macOS equivalent of start.bat: start monitor + webhook + dashboard in the foreground.
#   ./start.sh            (Ctrl+C stops everything)
#   ./start.sh --open     also open the dashboard in the browser
#   ./start.sh --stop     stop a running launcher
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/pip install -q --no-deps -e .
fi
[ -f .env ] || { cp .env.example .env; echo "Created .env: add your keys."; }
grep -qE '^APP_PASSWORD_HASH=.+' .env || .venv/bin/python -m forecaster.auth --write-env
exec .venv/bin/python -m forecaster.launcher "$@"
