# quarterly-forecaster

4-quarter-ahead probabilistic forecasts (P10 / P50 / P90) for SPY, QQQ, ^GSPC,
^IXIC, ^VIX and any tickers in `watchlist.yaml`, with a Streamlit dashboard.
See [CLAUDE.md](CLAUDE.md) for goals, design rules and every command.

> **Backtest verdict:** on S&P 500 2005–2022 (price + VIX inputs only) the
> ensemble does **not** reliably beat a naive historical-average baseline.
> See [results/backtest_sp500_2005_2022.md](results/backtest_sp500_2005_2022.md).
> The dashboard shows this on every page.

## Use it as a phone app (no code)

**Windows, one time:** install [Python 3.11+](https://www.python.org/downloads/) (tick *Add python.exe
to PATH*), then double-click **`setup.bat`**. It installs everything, asks for a dashboard password,
creates `.env` and `portfolio.yaml` for you to fill in, puts a **Quarterly Forecaster** shortcut on
the desktop, and sets auto-start (when you sign in, or at PC boot). If Tailscale is installed it also
shares the dashboard privately to your phone.

**Every day:** double-click the desktop shortcut (or `start.bat`). It starts the monitor, the
TradingView webhook and the dashboard together in the background (plus the Cloudflare tunnel if
`cloudflared/config.yml` exists) and opens the app. Crashed services restart by themselves. `stop.bat`
stops everything. Logs are in `logs/`. On Linux/macOS use `./start.sh` instead.

**On your phone:** with Tailscale connected, open `https://<pc-name>.<tailnet>.ts.net`, log in, then
*Share → Add to Home Screen* (iPhone Safari) or *⋮ → Install app / Add to Home screen* (Android Chrome).
It opens full-screen like an app. The link only works on your own Tailscale devices, and the app
still asks for the password; nothing is public.

Bottom tabs:
- **This Week**: market regime, VIX, a 3-line plain-English outlook, and SPY / QQQ / gold
  4-quarter P10–P50–P90 fan charts (CAD).
- **Live**: real-time tiles (SPY, QQQ, NVDA, AAPL, MSFT, your watchlist, gold, USD/CAD and a DXY proxy)
  updating every second with % change on the day and a millisecond "last tick" time; a tile turns red
  when its feed has been silent more than 10 s while its market is open. Below: gold 1-minute candles
  with session VWAP and ±1σ/±2σ bands. A second view shows free TradingView charts (S&P 500, Nasdaq,
  VIX, gold). Needs `FINNHUB_API_KEY` (stocks) and `OANDA_API_TOKEN` + `OANDA_ACCOUNT_ID` (practice
  account is fine; prices only, no orders). The feeds run only while the app runs (start.bat starts
  them). Check them with `python -m forecaster.live_feed --check 30`: it prints ticks per symbol.
  Finnhub's free websocket streams US stocks; the `.TO` watchlist ETFs may never tick on it.
- **Calendar**: the next 14 days of CPI, PPI, FOMC, jobs, GDP and earnings, colour- and label-coded
  by impact, with live countdowns.
- **Gold VWAP**: gold vs its session VWAP, the latest TradingView alerts with TAKE / CAUTION / SKIP
  and reasons, and the journal's win rate per verdict and horizon.
- **Portfolio**: holdings in CAD, P&L, allocation, the portfolio forecast range, and the
  contribution planner.
- **Reports**: every Sunday report, newest first, readable in the app (PDF download too).
- **Status**: where the project stands and what to do next.
  - **Build progress**: each module imported and its tests run (✅ built / ⚠️ tests failing / ❌ missing).
  - **Data health**: last refresh of prices, FRED macro, news and calendar, flagged red if older than 24 h
    or if a key is missing from `.env` (keys are shown as set/missing, never their values).
  - **Model health**: latest backtest coverage, hit rate, beats-baseline and run date.
  - **Phase tracker**: 1 Build → 2 Validate → 3 Journal → 4 Demo → 5 Live, each gate checked automatically
    (tests pass · backtest beats baseline · 30+ journaled alerts · 2+ months of practice P&L), current phase highlighted.
  - **Next step**: one line.

  The same summary is written to `STATUS.md` (gitignored; `data/STATUS.md` under Docker) whenever the
  app starts and on the Sunday job, and goes to Telegram with the weekly report.
  `python -m forecaster.project_status` runs it by hand (`--quick` skips the tests, `--send` sends it).
  Nothing in it ever turns on live trading: reaching Phase 5 only tells you the gates passed.

Every page shows **Last updated** times, and a **red banner** when a data source's last refresh
failed or its data is older than 24 h (8 days for the weekly forecasts).

Automatic refresh (the monitor does it; nothing to click): prices and news every 15 minutes, macro and
events at 06:15 and 18:15 New York time, and a full forecast rebuild plus the weekly report every
**Sunday at 6 pm Mountain time**.

## Currency: everything in CAD

Money values are reported in CAD. Canadian tickers (`XIU.TO`, `XEQT.TO`, `VFV.TO`, `ZAG.TO`
in `watchlist.yaml`) are already in CAD. USD-priced tickers (SPY, QQQ, ^GSPC, ^IXIC, gold,
sector ETFs) are converted with the daily USD/CAD rate (Yahoo `CAD=X`, back-filled from FRED
`DEXCAUS` for older history) *before* forecasting, so their ranges include currency risk, as a
Canadian holder would experience it. ^VIX, ^TNX and the dollar index aren't money and stay in
points or percent. Trading prices for orders (e.g. XAU_USD on OANDA) stay in their quote
currency, with the CAD equivalent shown next to them.

## Run it 24/7, privately

`docker compose up -d` runs four services: the dashboard (password-protected, loopback only, reached
from your phone via Tailscale), the TradingView webhook (only `/tv-alert`, through a Cloudflare
Tunnel), the 24/7 monitor (Telegram alerts and `/status`), and cloudflared. Step-by-step setup,
including Tailscale and Cloudflare Tunnel, is in **[DEPLOY.md](DEPLOY.md)**.

The monitor checks every 15 minutes, around the clock (gold trades nearly 24 h). Each alert is sent
once, with a short reason:
- warnings 24 h and 1 h before CPI, PPI, jobs and FOMC
- VIX spikes of 20% or more
- gold beyond 2 standard deviations from its session VWAP
- regime changes
- news-sentiment shocks

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
streamlit run app.py                   # or: python -m forecaster.launcher --open
```

The app only reads snapshots in `data/snapshots/` (plus `data/status.json` for the health banner);
it never runs models on page load. Layout and tabs: see *Use it as a phone app* above.

## Portfolio

```bash
cp portfolio.example.yaml portfolio.yaml   # gitignored: your holdings are never committed
# edit: ticker, shares, avg_cost, account (TFSA / FHSA / RRSP / cash), optional cash balances
python -m forecaster.pipeline              # or press Refresh data in the app
```

The **Portfolio** page shows:
- market value and unrealised P&L in CAD
- allocation by asset class, sector (ETFs looked through where sector weights are known), account and region
- volatility, beta vs the S&P/TSX 60 and the S&P 500 in CAD, and each holding's share of portfolio risk
- a 4-quarter P10/P50/P90 range for the whole portfolio, from 10,000 simulated paths
- a warning for any holding above 25% (`concentration_limit`)

The paths resample blocks of real *joint* daily CAD returns, so correlations between holdings are
kept. The method hasn't been backtested, and its median inherits the history window's drift.

For US-listed holdings, enter your CAD cost (`cost_currency: CAD`), or a USD cost plus `cost_fx`
(the USD/CAD rate when you bought). Otherwise the P&L leaves out currency gains or losses since
purchase, and the holding is labelled to say so.

## Contribution planner

The **Contribution planner** page takes a monthly CAD amount and a target allocation. Defaults come
from the `planner:` section of `portfolio.yaml`, and both are editable on the page. It suggests how
to split the contribution to move toward target:
- **Buy-only.** It never sells, and never suggests leverage, margin or options. Leveraged, inverse,
  option, futures and index symbols are refused as targets.
- **Exact totals.** The buys always add up to your contribution, or to the cent less with whole
  shares, where the leftover is shown.
- **Biggest gaps first.** It minimises the total distance to target, so the largest gaps fill
  first. It also estimates how many months until every weight is within 1 point of target.

It flags any of these, and suggests splitting the buy into 3 weekly parts on days without
high-impact events:
- elevated drawdown risk: next-quarter P(>10% drawdown) of 35% or more
- the high-volatility regime at 50% or more
- VIX at 25 or higher
- a high event-risk week: 6 or more high-impact event points in the next 7 days

These are rough checks: spreading a buy smooths the entry price but doesn't raise expected returns.

## Sunday weekly report

`python -m forecaster.weekly_report` refreshes data, then writes `reports/weekly_YYYY-MM-DD.md` and
`.pdf` and sends a short summary to your Telegram bot. The report covers:
- last week's moves: SPY, QQQ and gold in CAD, VIX in points, plus USD/CAD
- macro data released last week
- next week's high-impact events, with dates and typical release times
- the 4-quarter forecasts vs last week
- sentiment shifts and shocks
- your portfolio changes

Each run saves `reports/state/<date>.json`, which the next run compares against, so the first
report has nothing to compare with. `reports/` is gitignored because it contains your portfolio.

Schedule it for Sunday (cron): `23 7 * * 0 /full/path/to/quarterly-forecaster/scripts/weekly_report.sh`.
Useful flags: `--no-refresh`, `--no-send`, `--date YYYY-MM-DD`.

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

Gold alerts (XAUUSD, GOLD, GC1!, MGC1!) are also checked against the **live** OANDA gold price and its
session VWAP at the moment the alert arrives: a long whose price is already back below VWAP (or a
short back above it) is SKIP; hovering on the wrong side or beyond the 2σ band is CAUTION; a spot alert
price more than 0.3% from live is CAUTION. If the live feed is down or older than 10 s the check says
"can't verify" (CAUTION, never SKIP).

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
