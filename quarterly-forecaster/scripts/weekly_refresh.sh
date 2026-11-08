#!/usr/bin/env bash
# Weekly refresh of the dashboard snapshots: market (macro, events, news),
# forecasts for core + watchlist tickers, and any backtest older than
# BACKTEST_STALE_DAYS (default 28, so backtests re-run about monthly).
#
# Schedule with cron (Saturday 06:17, after Friday's close):
#   crontab -e
#   17 6 * * 6 /full/path/to/quarterly-forecaster/scripts/weekly_refresh.sh
# Logs: data/logs/weekly_YYYYMMDD.log
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/logs
LOG="data/logs/weekly_$(date +%Y%m%d).log"
LOCK="data/.weekly_refresh.lock"
if ! mkdir "$LOCK" 2>/dev/null; then          # portable lock (no flock on macOS)
  echo "$(date '+%F %T') another refresh is running; exiting" >>"$LOG"
  exit 0
fi
trap 'rmdir "$LOCK"' EXIT

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

{
  echo "== $(date '+%F %T') weekly refresh start"
  "$PY" -m forecaster.pipeline --backtest-stale-days "${BACKTEST_STALE_DAYS:-28}" "$@"
  status=$?
  echo "== $(date '+%F %T') finished, exit $status"
} >>"$LOG" 2>&1
exit "${status:-1}"
