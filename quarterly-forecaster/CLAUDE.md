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
  model.py         # Forecast contract + block-bootstrap baseline model
  cli.py           # `python -m forecaster`
  sources/         # one adapter per data provider
app.py             # Streamlit dashboard (reads data/snapshots/*.json only)
pine/vwap_gold.pine # TradingView v6 strategy; keep in sync with vwap_backtest.py
scripts/weekly_refresh.sh  # cron job: pipeline + stale backtests, logs in data/logs/
results/           # committed backtest reports + reference summary (backtest_GSPC.json)
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
  + sector ETFs `XLK, XLF, XLE, XLV, XLY` (`market_data.market_universe`).
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
  from `pipeline.py` (cached by file mtime). Backtest accuracy and its verdict
  sit above every page; with no per-ticker backtest, the committed S&P 500
  reference is shown and labelled as such. Never show a P50 without its band.
  Charts follow the dataviz palette (validated light + dark). Accent colour is
  blue (`.streamlit/config.toml`), so red keeps its meaning.
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
python -m forecaster.webhook_server --port 8000   # TradingView webhook (expose via HTTPS)
python -m forecaster.webhook_server --stats       # outcome stats by verdict
python -m forecaster.vwap_backtest --pine-events --fred-events   # event list for the Pine input
python -m forecaster.vwap_backtest XAUUSD_15.csv --compare tv_trades.csv   # port vs Strategy Tester
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
