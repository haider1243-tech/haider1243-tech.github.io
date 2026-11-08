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

Only add a paid source if a free one genuinely can't provide the data.

## Layout

```
src/forecaster/
  config.py        # .env + watchlist.yaml -> Settings; core tickers, horizon, percentiles
  cache.py         # file cache with TTL in /data (gitignored)
  market_data.py   # OHLCV universe, parquet cache, weekly/quarterly resampling, metrics
  macro.py         # FRED indicators: YoY, 3-month trend, surprise vs prior trend
  calendar.py      # CPI/PPI/jobs/GDP/FOMC/earnings events, impact tags, risk density per quarter
  model.py         # Forecast contract + block-bootstrap baseline model
  cli.py           # `python -m forecaster`
  sources/         # one adapter per data provider
tests/             # pytest, no network
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
cp .env.example .env      # then add your keys
pytest
python -m forecaster      # forecast core + watchlist
python -m forecaster AAPL MSFT --clear-cache
python -m forecaster.market_data   # refresh OHLCV cache, print metrics snapshot + correlations
python -m forecaster.macro         # macro indicator snapshot (needs FRED_API_KEY)
python -m forecaster.calendar      # upcoming events + risk density per quarter
```

(Run from this folder with `PYTHONPATH=src`, or `pip install -e .`.)

## Known limitations / next steps

- `^VIX` is mean-reverting; bootstrapping its log returns ignores that. Give it
  a dedicated model.
- Macro indicators, calendar risk density and market metrics are computed but
  not yet used as model inputs.
- `GDP` is nominal; consider `GDPC1` (real GDP) for growth signals.
- Add a backtest module that scores historical forecast calibration.
