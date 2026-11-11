# quarterly-forecaster

Probabilistic forecasts of index/ETF prices **4 quarters ahead**.

## Goals

1. **Coverage** — forecast `SPY`, `QQQ`, `^GSPC`, `^IXIC`, `^VIX` always, plus
   every ticker listed in `watchlist.yaml` (Yahoo Finance symbols).
2. **Probabilistic output only** — every forecast is a range: 10th / 50th / 90th
   percentile per quarter. Never emit or display a single-point prediction.
   The `Forecast` / `QuarterRange` dataclasses in `src/forecaster/model.py`
   enforce this (4 quarters, `p10 <= p50 <= p90`); every model must return them.
3. **Testable and cached** — every module is unit-testable without the network,
   and every external API call is cached to stay within free-tier rate limits
   (`forecaster.cache.Cache` for API responses, `market_data.ParquetStore` for
   daily OHLCV).

## Data sources (free first)

| Source | Use | Key | Module |
|---|---|---|---|
| yfinance | Daily OHLCV for the market universe (parquet cache) | none | `market_data.py` |
| yfinance | Daily/quarterly closes for the baseline model | none | `sources/prices.py` |
| FRED | Macro series + scheduled release dates | `FRED_API_KEY` | `sources/fred.py` → `macro.py`, `calendar.py` |
| Finnhub (free tier, ~60 req/min) | Quotes, fundamentals, earnings calendar | `FINNHUB_API_KEY` | `sources/finnhub.py` → `calendar.py` |
| Hardcoded | FOMC decision dates 2026–2027 | none | `calendar.py` |
| Finnhub `company-news` + `news?category=general` | Headlines per ticker + market | `FINNHUB_API_KEY` | `news.py` |
| FinBERT (`ProsusAI/finbert`, local) | Headline sentiment | none (`requirements-ml.txt`) | `news.py` |

Only add a paid source if a free one genuinely can't provide the data.

## Layout

```
src/forecaster/
  config.py        # .env + watchlist.yaml -> Settings; core tickers, horizon, percentiles
  cache.py         # file cache with TTL in /data (gitignored)
  market_data.py   # OHLCV universe, parquet cache, weekly/quarterly resampling, metrics
  macro.py         # FRED indicators: YoY, 3-month trend, surprise vs prior trend
  calendar.py      # CPI/PPI/jobs/GDP/FOMC/earnings events, impact tags, risk density per quarter
  news.py          # headlines -> FinBERT sentiment -> daily/weekly index + shock flags
  forecast.py      # ensemble: HMM regimes + regime/GARCH Monte Carlo + LightGBM quantiles, SHAP
  backtest.py      # walk-forward backtest vs naive baseline, calibration + stress periods
  pipeline.py      # runs models, writes JSON snapshots the dashboard reads
  dashboard.py     # Plotly figure builders (palette-validated, light/dark)
  webhook_server.py # FastAPI: TradingView alert -> risk checks -> TAKE/CAUTION/SKIP -> Telegram + SQLite
  vwap_backtest.py # bar-for-bar Python port of pine/vwap_gold.pine (TV broker-emulator rules)
  execution.py     # OANDA v20 orders for TAKE verdicts only; practice default; hard limits + kill switch
  portfolio.py     # portfolio.yaml -> CAD value, P&L, allocation, beta/vol, joint-bootstrap forecast
  weekly_report.py # Sunday report: Markdown + PDF in reports/ (gitignored), Telegram summary
  planner.py       # buy-only contribution split toward targets + timing flags; never leverage/options
  monitor.py       # 24/7 APScheduler checks -> Telegram (deduped) + /status bot; writes prices.json
  status.py        # data health (data/status.json): last success/failure per source -> red banner
  appdata.py       # pure helpers for the app tabs (outlook lines, calendar, gold journal, reports)
  project_status.py # Status tab + STATUS.md: module imports + pytest, data/key health, backtests, phase gates
  live_feed.py     # Finnhub trades websocket + OANDA pricing stream -> ring buffer, 1s/1m bars in data/live.sqlite
  intraday.py      # 1m history: OANDA (gold, with spread) / Alpaca (stocks) -> monthly parquet; session resampling
  strategies.py    # day-trading strategy library on causal Features (signal on close, fixed stop/target)
  lab_engine.py    # fast simulator: next-open fills, spread/slippage/commission, time exit; random entries
  lab_stats.py     # stats, day-clustered bootstrap, MC, p vs random, walk-forward, verdict, overfit warnings
  lab.py           # `python -m forecaster.lab download|run|show` -> data/lab/results.json + OOS trades
  shadow.py        # paper-trades every lab setup on live bars (same engine); decay vs backtest; no orders
  simfeed.py       # demo mode: local Finnhub/OANDA-protocol simulator + demo data seeding
  clock.py         # app-wide "now" (shifted in demo mode so the simulated market is open)
  launcher.py      # supervises monitor + webhook + dashboard (+ cloudflared) for start.bat/start.sh
  auth.py          # dashboard password (PBKDF2, ':'-separated hash) + login lockouts
  model.py         # Forecast contract + block-bootstrap baseline model
  cli.py           # `python -m forecaster`
  sources/         # one adapter per data provider
app.py             # mobile-first Streamlit app, 8 bottom tabs (reads data/snapshots/*.json only)
static/            # PWA manifest + icons (served at /app/static/; regenerate: scripts/make_icons.py)
start.bat, stop.bat, setup.bat, scripts/windows/   # Windows one-click launcher, shortcuts, auto-start
start.sh           # same launcher on Linux/macOS
pine/vwap_gold.pine # TradingView v6 strategy; keep in sync with vwap_backtest.py
scripts/weekly_refresh.sh  # cron job: pipeline + stale backtests, logs in data/logs/
scripts/weekly_report.sh   # Sunday cron job: weekly_report (refresh, md+pdf, Telegram)
results/           # committed backtest reports + reference summary (backtest_GSPC.json)
Dockerfile, docker-compose.yml, DEPLOY.md   # private 24/7 deployment (see Deployment rules)
cloudflared/config.example.yml  # tunnel ingress: ONLY ^/tv-alert$, everything else 404
tests/             # pytest, no network (test_app.py drives app.py with AppTest)
data/              # API response cache — contents never committed
```

## Rules for working in this repo

- **Secrets**: API keys live in `.env` only (template: `.env.example`). Never
  commit `.env`, never hardcode keys, never print them. Send keys in headers
  rather than URLs where the API allows it (Finnhub does).
- **Caching**: new source functions take a `Cache` and use
  `cache.get_or_fetch(Cache.make_key(...), fetch)`. No uncached network calls.
  Daily OHLCV goes through `ParquetStore` (`data/ohlcv/*.parquet`): it only
  downloads when the newest bar is older than the last completed business day,
  fetches incrementally, and throttles rechecks (holidays) via `_meta.json`.
- **Market universe**: watchlist + core + `SPY, QQQ, ^VIX, ^TNX, DX-Y.NYB, GC=F`
  + sector ETFs `XLK, XLF, XLE, XLV, XLY` + `CAD=X` (`market_data.market_universe`).
  The watchlist holds the Canadian ETFs `XIU.TO, XEQT.TO, VFV.TO, ZAG.TO`.
- **Currency: every money value is reported in CAD.** `ticker_currency()`:
  `.TO/.V/.NE/.CN` -> CAD (no conversion); `^VIX, ^TNX, DX-Y.NYB, CAD=X` ->
  not money (points / yields, never converted); everything else -> USD,
  converted with daily USD/CAD (Yahoo `CAD=X`, CAD per USD, back-filled from
  FRED `DEXCAUS` before 2003). Rates align point-in-time (last rate on or before
  each bar, never a later one); bars before the first rate are dropped.
  `validate_usd_cad` refuses rate series outside 1.0-2.0 (inverted = silent
  1.9x error). Forecasts are made on CAD prices, so P10-P90 includes FX risk;
  never convert a USD forecast at spot. Exceptions, on purpose: order prices
  and the VWAP backtest's primary stats stay in the instrument's quote currency
  (XAU_USD is USD; needed to match TradingView and OANDA), with CAD shown
  alongside. Overrides: `currencies:` map in watchlist.yaml.
- **Metric conventions** (`market_data.py`): log returns for vol/beta/corr;
  simple returns for quarterly returns; realized vol annualized with 252 days;
  drawdowns are negative fractions; RSI is Wilder's 14-day (SMA-seeded); always
  align prices on common dates *before* taking returns (calendars differ for
  futures/FX/indices).
- **Macro conventions** (`macro.py`): index series (CPI, core CPI, PPI, GDP,
  sentiment) in % changes; rate series (Fed funds, unemployment, 10Y-2Y) in
  percentage points. Daily series → monthly mean. "Surprise" = latest change
  minus mean of the prior 3 changes (no free consensus data); `surprise_z`
  scales by the std of past surprises, excluding the current one.
- **Calendar** (`calendar.py`): impact high = FOMC, CPI, jobs; medium = PPI,
  GDP, mega-cap earnings; low = other earnings. Points 3/2/1, FOMC meetings
  with projections (SEP) +1. Density = points per business day × 63, so the
  partial current quarter is comparable. The `missing` column flags quarters
  where a scheduled category has no dates yet: treat the score as a lower bound.
  **Update `FOMC_DECISION_DATES` each year** when the Fed publishes the next
  schedule (federalreserve.gov/monetarypolicy/fomccalendars.htm). Run as
  `python -m forecaster.calendar`, never by file path (shadows stdlib `calendar`).
- **News** (`news.py`): headlines, tags and scores live in `data/news.sqlite`.
  Scores are keyed by sha256(model + headline text), so a headline is scored
  once ever, even when it shows up under another id or ticker. Macro topics are
  regex-tagged on market news only (the free tier has no keyword search).
  Sentiment = P(pos) − P(neg). Shock = |z| > 2 against the prior 30 news days,
  with at least 3 headlines; expect about 5% false alarms on quiet noise. The free
  market-news endpoint returns only the latest ~100 items, so market history
  builds up only by running regularly. Tests use a fake scorer; never load the
  real model in tests.
- **No look-ahead (forecast.py) — non-negotiable.** All inputs go through
  `PointInTimeData.asof(T)`. Macro data uses *first-release* values and their
  publication dates (`FredClient.get_first_release`, ALFRED output_type=4),
  never `get_series` (revised values). Regime features come from HMMs refit on
  expanding windows at completed quarter-ends and are filtered forward-only;
  smoothed posteriors are only used inside a fit window. LightGBM training rows
  are purged to targets with date <= T. Any new feature must come with a test
  that changing post-T data leaves the forecast unchanged
  (`test_no_lookahead_future_data_changes_nothing`).
- **Ensemble conventions**: targets are the closes at the ends of the 4 quarters
  *after* T's quarter. Blend = weighted average of log-return quantiles
  (default 50/50 MC/LightGBM; MC only if < `min_train_rows`). Drawdown
  probability comes from the MC paths (start price counts as a peak). HMM
  fits run single-threaded (threadpoolctl) for bit-for-bit reproducibility.
- **Backtest** (`backtest.py`): origins at every quarter-end from 2004-12-31;
  the ensemble is refit from scratch at each one. Baseline = empirical
  10/50/90th percentiles of all past n-day log returns (same horizon). A target
  is scored only once its quarter is complete in the data. Model-vs-baseline
  differences get a circular block-bootstrap CI (blocks of 4 quarters, because
  horizons overlap). Report the verdict as the numbers say, including "does not
  beat the baseline". Don't tune on the backtest and then quote the same
  backtest as out-of-sample.
- **Dashboard** (`app.py`): never runs models on page load; it reads snapshots
  from `pipeline.py` (cached by file mtime). Mobile-first: dark theme, bottom tab
  bar (`st.radio` key `tab` in `st.container(key="bottomnav")`, synced with `?tab=`),
  test at 390 px wide (no horizontal page scroll; wide tables scroll in their own box).
  Every tab shows the red health banner (`status.problems`) and "Last updated".
  The honest backtest note (`model_note`) stays visible. Never show a P50 without
  its band. Charts follow the dataviz palette. Accent colour is blue, so red keeps
  its meaning (problems, SKIP, high impact), always with a text label.
- **Data health** (`status.py`): every refresher records success/failure per source
  (prices + news in the monitor, market/portfolio/forecasts in `pipeline.refresh_all`).
  Limits: 24 h, forecasts 8 days. News/portfolio are only expected when configured.
- **Schedule** (`monitor.py`): checks + prices/news every 15 min around the clock;
  macro/events refresh 06:15 + 18:15 NY; forecast rebuild + weekly report Sunday
  18:00 `America/Edmonton` (MT).
- **Project status** (`project_status.py`): the build check (import every module + run
  pytest in a subprocess with a scrubbed env and `FORECASTER_STATUS_CHILD=1`) runs on
  launcher/app start (skipped inside pytest, while one runs, or < 10 min after the last),
  on the Sunday weekly report, and on "Re-check now"; cached in data/project_status.json.
  Everything else is recomputed per view. Phases are strictly ordered; gates: all tests
  pass; primary backtest (own SPY/^GSPC run, else reference) `beats_baseline`; 30+ alerts
  with a resolved outcome; practice fills + daily NAV spanning 60+ days. Keys are only
  ever reported set/missing; tests assert no value reaches STATUS.md, JSON or Telegram.
  It never changes LIVE_TRADING. Add new modules to `MODULES` (and `TEST_FILES` if the
  test file isn't `test_<module>.py`). `conftest` hides the real .env and portfolio.yaml
  from every test, so results match between machines.
- **Live feed** (`live_feed.py`): one process (launcher service `live`, compose service
  `live`), so it runs only while the app runs. Finnhub's token must go in the websocket URL:
  redact URLs/exceptions with `redact()` before logging or storing; tests assert the key never
  reaches logs or the DB. OANDA token in the Authorization header only; prices are bid/ask mids;
  OANDA has no volume, so VWAP uses tick counts. Reconnect backoff 1->60 s with jitter (5 min
  after 401/403), reset after 60 s connected. Heartbeats: OANDA 15 s; Finnhub 15 s without a
  trade during US hours. Gaps (> 10 s silence while open) go to `gaps` + logs. Tiles go red
  when the *feed* is silent > 10 s while its market is open (thin symbols don't trade that often).
  Readers (`tiles`, `gold_vwap`, `vwap_frame`) are pure; the app and webhook only read
  live.sqlite. Tests use a local websockets server and an httpx streaming MockTransport; never
  hit Finnhub/OANDA. `FORECASTER_LIVE_FINNHUB_URL` / `FORECASTER_LIVE_OANDA_URL` point it at
  local simulators.
- **Webhook live gold check** (`check_live_gold`): judged at receipt time. Wrong side of VWAP
  by > 0.25σ -> skip (the setup's defining condition failed); slightly wrong side or past 2σ ->
  caution; spot alert price > 0.3% off live -> caution (futures skip that, basis); missing or
  > 10 s old -> unknown. Check data (price, vwap, sd, z) is journaled with the alert.
- **Strategy Lab** (`strategies.py`, `lab_engine.py`, `lab_stats.py`, `lab.py`), non-negotiable:
  signals are decided on a bar's close from causal features only and fill at the next open
  (`test_signals_never_use_future_bars` runs every strategy on truncated data; add new strategies
  to LIBRARY and that test covers them). Costs always on. Judge only out-of-sample (walk-forward)
  trades. Random-entry benchmarks and the bootstrap must respect day clustering (`trade_days`,
  `days=`); without it p-values were too optimistic. `test_luck_test_is_calibrated_on_random_walks`
  guards calibration. Never tune on OOS results. The verdict rules are the user's: EDGE = p < 0.05,
  100+ OOS trades, expectancy > 0 after costs. Report Bonferroni alongside. Gold sessions run
  18:00->17:00 for indicators, but trades are limited to 08:00-16:55 ET (`window_mask`). A day
  counts as complete if any bar of it reaches the time-exit point (gold's 15m bars end at 16:45/17:00).
  Stock history: Alpaca (free paper account, SIP -> IEX fallback); Finnhub candles are paid now.
- **Shadow trading** (`shadow.py`): the same Features/signals/`simulate` as the lab, on cached
  history + live 1m bars, with `allow_open=True`. Only signals after `started_at` count; closed
  trades are immutable (upsert only while open). It must never import or call
  `forecaster.execution` (a test greps for it); the Signal Board has no order buttons.
- **Demo mode** (`launcher --demo`, demo.bat): FORECASTER_CLOCK_OFFSET_S shifts `clock.now()` so
  market time starts at 09:44 ET on launch; FORECASTER_CACHE_DIR=data/demo (wiped and re-seeded each
  start, lab results kept); feeds point at `simfeed` on loopback with dummy keys; only simfeed, live,
  shadow and dashboard run (no monitor, webhook or tunnel). Anything that compares "now" with market
  sessions must use `forecaster.clock`, not `pd.Timestamp.now()`, or it breaks in demo mode. The app
  shows a demo banner whenever `clock.is_demo()`.
- **Launcher** (`launcher.py`): binds everything to 127.0.0.1, restarts crashed services
  with backoff, one instance (data/launcher.pid). The webhook gets `--trust-proxy` only
  when cloudflared runs. Windows scripts must keep CRLF line endings (`.gitattributes`).
- **Webhook** (`webhook_server.py`): fails closed (no `WEBHOOK_SECRET` -> 503),
  compares the secret in constant time, and never stores or logs it. It parses
  JSON from the raw body (TradingView sends text/plain). The request path reads
  local files only (snapshots, cached VIX); Telegram is sent in a background
  task, so TradingView gets an answer in ms. Verdict: any skip-level check, or
  >= 3 cautions -> SKIP; any caution or missing data -> CAUTION; else TAKE.
  Missing data never escalates to SKIP. Staleness is judged on the data's as-of
  date, not the snapshot's generation time. Event times are typical release times
  (CPI/PPI/jobs/GDP 08:30 ET, FOMC 14:00 ET). Outcomes are measured on the
  price-source symbol (e.g. GC=F for XAUUSD) against its own close at alert
  time, never the TradingView price (spot vs futures basis). The Telegram token
  is in the API URL, so error messages must not include the request URL.
- **VWAP strategy parity** (`pine/vwap_gold.pine` <-> `vwap_backtest.py`): any
  logic change goes in both files. The Pine default event list is generated from
  `FOMC_DECISIONS_NY`; a test checks they match. Execution follows TradingView's
  broker emulator: fill at the next open, exits can trigger on the fill bar, path
  open->high->low->close when the high is nearer the open (else low first), gaps
  fill at the open, and slippage applies to market and stop fills only.
  `request.economic()` can't block *before* a release; pre-event blocking
  needs the explicit event list. pandas 3 infers datetime resolution: always
  `.as_unit("ns")` before `.asi8` (a unit mismatch once blocked 99% of bars).
- **Execution safety** (`execution.py`), never weaken without the user asking:
  live requires `OANDA_ENVIRONMENT=live` AND `LIVE_TRADING=true` read from the
  `.env` *file* (`dotenv_values`), never from `os.environ`. Live without the flag
  raises at config load (fail closed). Orders only for verdict TAKE, only with
  `EXECUTION_ENABLED=true`, only with a stop. Hard ceilings are in code
  (`HARD_MAX_RISK_PCT`, `HARD_MAX_POSITION_UNITS`). Any API error or unexpected
  response blocks the order; 3 in a row trip the kill switch. Tests use
  `httpx.MockTransport`; never hit OANDA in tests. The token goes in the
  Authorization header only and is never logged or stored.
- **Portfolio** (`portfolio.py`): `portfolio.yaml` is personal and gitignored.
  Never commit it or print holdings in commits/PRs; `portfolio.example.yaml` is the
  template. P&L is CAD; a USD cost without `cost_fx` is converted at today's rate
  and labelled (it excludes FX since purchase). Risk uses current weights on daily
  CAD returns aligned by inner join (TSX/NYSE holidays differ); cash has zero vol.
  The forecast is a joint block bootstrap (rows of all holdings together, so
  correlation is kept; a test fails if assets are resampled independently).
  Buy-and-hold, not backtested. Say so wherever it's shown. Concentration flag:
  a holding (summed across accounts) strictly above `concentration_limit` (25%).
- **Weekly report** (`weekly_report.py`): one `Report` model rendered to both
  Markdown and PDF (reportlab), so they can't drift. Comparisons use the previous
  `reports/state/*.json`. `reports/` holds portfolio data and is gitignored. The
  PDF uses built-in fonts (cp1252): pass all text through `pdf_safe()` (no ≈, ⚠ or
  arrows; they render as boxes). Moves: Friday close vs the previous week's last
  close, in CAD (VIX in points). The Telegram summary is HTML-escaped and short.
- **Contribution planner** (`planner.py`), non-negotiable: buy-only (never
  sell), never leverage, margin, options, futures or inverse products
  (`check_instrument` refuses them; keep `LEVERAGED_OR_INVERSE` and the name
  patterns current, but don't match plain "short"/"ultra", which appear in
  ordinary bond funds). Buys are >= 0 and sum exactly to the contribution
  (water-fill least squares). Risk flags only ever suggest splitting a buy across
  weeks, never skipping or hedging, and say the drawdown probabilities run high.
- **Deployment privacy** (non-negotiable, `test_deploy.py` enforces it): the
  dashboard is published only as `127.0.0.1:8501` (Tailscale `serve`, never
  `funnel`). No other service publishes a port. The tunnel ingress is exactly
  `^/tv-alert$` -> webhook, then `http_status:404`. The webhook trusts
  `CF-Connecting-IP` only with `--trust-proxy` (it's reachable only via cloudflared)
  and rate-limits before checking the secret. The dashboard fails closed without
  `APP_PASSWORD_HASH`. Hashes use ':' separators, because compose expands `$`
  in env files. Telegram commands come in by long-polling, answered only in
  `TELEGRAM_CHAT_ID`. httpx INFO logs contain the bot-token URL: always use
  `monitor.setup_logging` (which silences them) in services. `init: true` on app
  services (PID 1 otherwise ignores SIGTERM). Monitor alerts are claimed by key in
  `data/monitor.sqlite` before sending (once ever) and unclaimed if Telegram fails.
  App tests log in via `conftest.authed_app`.
- **Injectable I/O**: sources accept a `session` / `downloader` argument so
  tests pass fakes. `load_settings` accepts an `env` dict. Keep it this way.
- **Tests**: every new module gets a test file. Tests must not touch the network
  or the real `/data` folder (use the `cache` fixture / `tmp_path`).
  Run: `pytest`.
- **Models**: the bootstrap model is a baseline. New models (regime-aware,
  macro-conditioned, VIX mean-reversion, etc.) should be benchmarked against it
  on calibration — do the realized prices land inside the 10–90 band ~80% of
  the time? — not just point accuracy.
- Use fixed seeds for reproducibility in tests.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-ml.txt  # only for news.py
cp .env.example .env      # then add your keys
pytest
python -m forecaster      # forecast core + watchlist
python -m forecaster AAPL MSFT --clear-cache
python -m forecaster.market_data   # refresh OHLCV cache, print metrics snapshot + correlations
python -m forecaster.macro         # macro indicator snapshot (needs FRED_API_KEY)
python -m forecaster.calendar      # upcoming events + risk density per quarter
python -m forecaster.news          # collect + score headlines, print latest index and shocks
python -m forecaster.forecast SPY  # ensemble forecast (add --asof YYYY-MM-DD for a backtest point)
python -m forecaster.backtest SPY --out results/spy   # walk-forward 2005-now (~5 min per ticker)
python -m forecaster.pipeline      # refresh dashboard snapshots (add --backtest-stale-days 28)
streamlit run app.py               # dashboard
scripts/weekly_refresh.sh          # what cron runs weekly
python -m forecaster.weekly_report --no-send   # build reports/weekly_<date>.md + .pdf
python -m forecaster.webhook_server --port 8000   # TradingView webhook (expose via HTTPS)
python -m forecaster.webhook_server --stats       # outcome stats by verdict
python -m forecaster.vwap_backtest --pine-events --fred-events   # event list for the Pine input
python -m forecaster.vwap_backtest XAUUSD_15.csv --compare tv_trades.csv   # port vs Strategy Tester
python -m forecaster.execution --status          # OANDA env, limits, kill switch, recent orders
python -m forecaster.execution --kill --flatten  # stop all orders and close positions
python -m forecaster.auth                        # make APP_PASSWORD_HASH for .env
python -m forecaster.monitor --once              # run the monitor checks once
docker compose up -d                             # 24/7 private deployment (DEPLOY.md)
python -m forecaster.launcher --open             # all services without Docker (start.bat / start.sh)
python -m forecaster.launcher --status|--stop|--check
python -m forecaster.auth --write-env            # set the dashboard password in .env
python -m forecaster.live_feed --check 30         # connect 30 s, print ticks per symbol (confirms feeds)
python -m forecaster.lab download --years 3        # 1m history (OANDA gold, Alpaca stocks)
python -m forecaster.lab run [--synthetic]         # walk-forward lab -> data/lab/results.json
python -m forecaster.shadow --once                 # one shadow-trading pass (the launcher runs it continuously)
python -m forecaster.launcher --demo [--open]      # simulated live market (demo.bat / demo.sh)
python -m forecaster.project_status [--quick] [--send]   # STATUS.md: build, data, model, phase, next step
```

(Run from this folder with `PYTHONPATH=src`, or `pip install -e .`.)

## Known limitations / next steps

- `^VIX` is mean-reverting; bootstrapping its log returns ignores that. Give it
  a dedicated model.
- **Backtest verdict (results/backtest_sp500_2005_2022.md, price+VIX only):
  the ensemble does NOT reliably beat the naive baseline.** Pinball skill is
  +4%, but the 90% CI includes 0. The median is worse than the historical
  average. LightGBM alone is overconfident (60% coverage). MC alone is too
  bearish after selloffs. The drawdown probabilities say 90% when the realised
  rate is about 40%. Fixes, in order: (1) shrink regime drifts and the P50
  toward the historical drift; (2) conformalise or regularise the LightGBM
  quantiles; (3) recalibrate drawdown probabilities on past origins only;
  (4) learn blend weights in a *nested* walk-forward (past origins only); (5)
  rerun with macro/rates features (needs FRED), which matters most for 2022.
- LightGBM has few *independent* samples (targets overlap by up to 4
  quarters); on synthetic data it gives pure-noise features non-zero SHAP.
  Needs stronger regularisation / feature pruning, validated by the backtest.
- News sentiment only exists from when collection started; it's NaN for most
  training rows until history accumulates.
- `GDP` is nominal; consider `GDPC1` (real GDP) for growth signals.
