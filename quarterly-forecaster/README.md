# quarterly-forecaster

4-quarter-ahead probabilistic forecasts (P10 / P50 / P90) for SPY, QQQ, ^GSPC,
^IXIC, ^VIX and any tickers in `watchlist.yaml`, with a Streamlit dashboard.
See [CLAUDE.md](CLAUDE.md) for goals, design rules and every command.

> **Backtest verdict:** on S&P 500 2005–2022 (price + VIX inputs only) the
> ensemble does **not** reliably beat a naive historical-average baseline.
> See [results/backtest_sp500_2005_2022.md](results/backtest_sp500_2005_2022.md).
> The dashboard shows this on every page.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # add FRED_API_KEY / FINNHUB_API_KEY
pytest
```

Optional, for news sentiment (FinBERT, ~1–2 GB):
`pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-ml.txt`

## Dashboard

```bash
python -m forecaster.pipeline          # build snapshots (forecasts, macro, events, news)
streamlit run app.py
```

The app only reads snapshots in `data/snapshots/`; the **Refresh data** button in
the sidebar rebuilds them (selected ticker or all; optionally the backtest too).
Pages: Forecast (fan chart, regime, SHAP drivers), Macro & events (CPI, PPI, Fed
funds, next-90-day high-impact events, risk density), Sentiment & news (gauges,
daily index with shocks, top headlines). Backtest accuracy is shown above each page;
until a ticker has its own backtest, the committed S&P 500 reference result is shown
and labelled as such.

## Weekly auto-refresh

`scripts/weekly_refresh.sh` refreshes everything, re-runs any backtest older than
28 days (`BACKTEST_STALE_DAYS`), and logs to `data/logs/`. It uses a lock so runs
never overlap.

**macOS / Linux (cron)**, Saturdays 06:17, after Friday's close:

```bash
crontab -e
17 6 * * 6 /full/path/to/quarterly-forecaster/scripts/weekly_refresh.sh
```

**Windows (Task Scheduler)**: create a weekly task whose action is
`C:\path\to\quarterly-forecaster\.venv\Scripts\python.exe -m forecaster.pipeline --backtest-stale-days 28`
with "Start in" set to the project folder.

The computer must be on at that time; cron doesn't catch up on missed runs.
