#!/usr/bin/env bash
# Demo mode on Linux/macOS: simulated live market. Ctrl+C stops it.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./start.sh --check
exec .venv/bin/python -m forecaster.launcher --demo "$@"
