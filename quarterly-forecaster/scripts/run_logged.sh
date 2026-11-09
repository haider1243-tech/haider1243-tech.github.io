#!/usr/bin/env bash
# run_logged.sh NAME CMD...: run CMD, copying its output to logs/NAME.log (and stdout for
# `docker compose logs`). Rotates the file at 10 MB. exec keeps CMD as the main process, so a crash
# still stops the container and Docker restarts it.
set -euo pipefail
name="$1"; shift
mkdir -p logs
f="logs/${name}.log"
if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 10485760 ]; then mv -f "$f" "$f.1"; fi
exec > >(tee -a "$f") 2>&1
exec "$@"
