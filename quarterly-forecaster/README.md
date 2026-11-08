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

## TradingView webhook → Telegram

`src/forecaster/webhook_server.py` receives TradingView alerts, runs risk checks
(regime, high-impact events in the next 24 h, news-sentiment shocks, VIX), and
returns **TAKE / CAUTION / SKIP** with reasons. It sends the verdict to Telegram and
logs every alert plus its actual +1h / +4h / +1d outcome to `data/alerts.sqlite`.
These are risk filters, not a trading signal.

1. In `.env` set `WEBHOOK_SECRET` (a long random string), `TELEGRAM_BOT_TOKEN` (from
   @BotFather) and `TELEGRAM_CHAT_ID` (send your bot a message, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates`).
2. Run the server and keep it running:
   `python -m forecaster.webhook_server --port 8000`
3. Expose it over HTTPS. TradingView only posts to ports 80/443, e.g.
   `cloudflared tunnel --url http://localhost:8000`, or a reverse proxy with TLS.
   Behind a proxy, add `--trust-proxy` if you use `WEBHOOK_ALLOWED_IPS`.
4. TradingView alert → Notifications → Webhook URL `https://<host>/tv-alert`, with this message:
   ```json
   {"ticker":"{{ticker}}","price":"{{close}}","strategy":"vwap_gold","side":"long","time":"{{timenow}}","secret":"<WEBHOOK_SECRET>"}
   ```
5. Keep the dashboard snapshots fresh (`python -m forecaster.pipeline`, or the weekly job).
   The checks read them. VIX comes from the OHLCV cache, so also refresh it daily
   (`python -m forecaster.market_data`), or the VIX check will report stale data.

The server fills in outcomes every 5 minutes while it runs. If it isn't always on,
schedule `python -m forecaster.webhook_server --resolve-outcomes` hourly; yfinance
keeps 5-minute bars for about 60 days. See how verdicts performed with
`python -m forecaster.webhook_server --stats`.

## VWAP gold strategy (Pine v6 + Python port)

`pine/vwap_gold.pine` is a TradingView strategy for XAUUSD:
- session VWAP (resets 18:00 New York) with ±1 and ±2 standard-deviation bands
- entries on VWAP reclaim or rejection, with volume confirmation
- DXY and US10Y trend filter
- entry blackout around listed events
- ATR stop, R-multiple target, 1% risk sizing and an end-of-day flatten
- `alert()` JSON in the webhook server's format

`src/forecaster/vwap_backtest.py` is a bar-for-bar port, so you can check it against
the Strategy Tester.

**Events.** `request.economic()` gives data values, not a release calendar. It can't
block entries *before* CPI, and FOMC meetings that hold rates don't show up in the
rate series. So the blackout uses an explicit list: FOMC 2021–2027 is built in. Add
CPI/PPI release times from FRED and paste the output into the strategy's *Events* box:

```bash
python -m forecaster.vwap_backtest --pine-events --fred-events > events.txt
```

**Comparing with TradingView**
1. Add the strategy to an XAUUSD chart (same symbol and timeframe you'll test, e.g. OANDA:XAUUSD 15m).
   Set its *Events* input. Note `mintick` and `point value` from the table at the top right.
2. Export chart data (chart menu → *Export chart data*, ISO or UNIX time). The strategy plots
   `DXY up`, `US10Y up` and `Econ release` in the data window so the export carries the exact filter values.
   If those columns are missing from your export, change those plots' `display` to `display.all`.
3. Strategy Tester → *List of trades* → export.
4. Run the port with the same settings, e.g.
   ```bash
   python -m forecaster.vwap_backtest XAUUSD_15.csv --events-file events.txt \
       --params '{"mintick": 0.001, "slippage_ticks": 0}' --compare tv_trades.csv --tv-tz America/New_York
   ```
   It prints Strategy-Tester-style stats, the share of TradingView trades it reproduces,
   and the trades it doesn't.

Five years of 15-minute bars is ~120k bars. TradingView only loads as much history as your
plan allows (Deep Backtesting covers more), and the export only contains loaded bars. Spot
XAUUSD volume is broker tick volume, so VWAP depends on the data vendor; compare like with like.

**Alerts.** Create one alert on the strategy with *alert() function calls only* and your webhook
URL. Put the secret in the strategy's *Webhook secret* input; it's visible to anyone you share
the script with, so keep the script private. Only entries send alerts.

## Order execution (OANDA v20, practice by default)

`src/forecaster/execution.py` turns **TAKE** verdicts into OANDA market orders, with the
stop-loss and take-profit attached on fill. CAUTION and SKIP are logged and never sent.
Alerts need a `stop` (the Pine strategy now sends `stop` and `target`); without one there's no order.

- **Practice only by default.** Live needs `OANDA_ENVIRONMENT=live` *and* `LIVE_TRADING=true`
  written in `.env`, read from the file itself, so a shell export doesn't count. Asking for live
  without the flag stops the server at startup.
- **Opt-in even on practice:** `EXECUTION_ENABLED=true`.
- **Hard limits:**
  - risk per trade, with a hard ceiling of 2%
  - max position units, counting what's already held
  - max open trades
  - max daily loss (NAV change since 17:00 New York)
  - alert age, and how far the alert price may be from OANDA's live price
  - duplicate alerts are never re-sent
- **Kill switch:** `python -m forecaster.execution --kill [--flatten]` creates `data/KILL_SWITCH`
  and blocks all orders until `--resume --yes`. It also trips by itself after 3 consecutive API
  errors, or if a position is found above `MAX_POSITION_UNITS`.
- `python -m forecaster.execution --status` shows the environment, limits, kill switch, account
  and recent order decisions. Every decision, including blocked ones and why, is in the
  `orders` table of `data/alerts.sqlite`, and is sent to Telegram.

Start the webhook server as before. With `EXECUTION_ENABLED=true` it routes TAKE verdicts here.
Run on practice for a good while and check `--status`, the `orders` table and
`webhook_server --stats` before considering anything else.
