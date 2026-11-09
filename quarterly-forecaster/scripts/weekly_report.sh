#!/usr/bin/env bash
# Sunday weekly report: refresh data, write reports/weekly_YYYY-MM-DD.{md,pdf}, send the
# Telegram summary. Schedule with cron (Sunday 07:23):
#   crontab -e
#   23 7 * * 0 /full/path/to/quarterly-forecaster/scripts/weekly_report.sh
# Logs: data/logs/report_YYYYMMDD.log. Windows: Task Scheduler, weekly on Sunday, action
#   .venv\Scripts\python.exe -m forecaster.weekly_report   (Start in: the project folder)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
LOG="data/logs/report_$(date +%Y%m%d).log"
LOCK="data/.weekly_report.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date '+%F %T') report already running; exiting" >>"$LOG"; exit 0
fi
trap 'rmdir "$LOCK"' EXIT
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
{
  echo "== $(date '+%F %T') weekly report start"
  "$PY" -m forecaster.weekly_report "$@"
  status=$?
  echo "== $(date '+%F %T') finished, exit $status"
} >>"$LOG" 2>&1
exit "${status:-1}"
