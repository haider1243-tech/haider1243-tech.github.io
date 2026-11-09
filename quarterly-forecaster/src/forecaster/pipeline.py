"""Refresh pipeline: run the models and write JSON snapshots the app reads.

The ensemble takes ~10 s per ticker and a backtest several minutes, so the
dashboard never computes on page load. It reads the snapshots below, which
this module (the app's Refresh button and the weekly job) writes:

    data/snapshots/market.json            macro panel, events, risk density, news
    data/snapshots/ticker_<T>.json        forecast, regime, SHAP drivers, history
    data/snapshots/backtest_<T>.json      walk-forward accuracy summary
    results/backtest_GSPC.json            committed reference backtest (fallback)

Builders (`*_snapshot`, `backtest_summary`) are pure and tested; `refresh_*`
functions do the I/O.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from forecaster.config import PROJECT_ROOT, Settings

SNAPSHOT_VERSION = 1
PANEL_SERIES = ("CPIAUCSL", "CPILFESL", "PPIACO", "FEDFUNDS")
REFERENCE_BACKTEST = PROJECT_ROOT / "results" / "backtest_GSPC.json"

Warn = Callable[[str], None]


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def safe_name(ticker: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in ticker.upper()).strip("_")


def snapshot_dir(settings: Settings) -> Path:
    d = Path(settings.cache_dir) / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return _jsonable(x.reset_index().to_dict(orient="records"))
    if isinstance(x, (pd.Timestamp, datetime)):
        return None if pd.isna(x) else pd.Timestamp(x).isoformat()
    if isinstance(x, pd.Period):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if x is pd.NaT:
        return None
    return x


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1, allow_nan=False))
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Pure builders
# --------------------------------------------------------------------------- #

def ticker_snapshot(result, prices: pd.Series, top_n: int = 10, history_years: int = 2,
                    currency: str | None = "CAD", converted_from: str | None = None) -> dict:
    """Forecast snapshot from an EnsembleResult (forecast.ensemble_forecast).

    `currency` is the unit of every price in the snapshot ("CAD", or None for
    index points / yields such as ^VIX); `converted_from` is "USD" when the
    prices were converted with daily USD/CAD before forecasting.
    """
    fc = result.forecast
    t = result.table
    hist = prices.dropna().loc[result.as_of - pd.DateOffset(years=history_years): result.as_of]
    drivers = []
    if not result.drivers.empty:
        d = result.drivers.copy()
        d["abs"] = d["shap"].abs()
        d = d.sort_values("abs", ascending=False).groupby(["quarter", "quantile"]).head(top_n)
        drivers = d[["quarter", "quantile", "feature", "value", "shap", "base"]].to_dict("records")
    return {
        "version": SNAPSHOT_VERSION,
        "kind": "ticker",
        "ticker": fc.ticker,
        "currency": currency,
        "converted_from": converted_from,
        "generated_at": _now(),
        "as_of": result.as_of,
        "last_close": fc.last_close,
        "model": fc.model,
        "weights": result.weights,
        "n_train_rows": result.n_train_rows,
        "forecast": [
            {"quarter": q.quarter, "target_date": t.loc[q.quarter, "target_date"],
             "p10": q.p10, "p50": q.p50, "p90": q.p90,
             "prob_drawdown": t.loc[q.quarter, "prob_drawdown"]}
            for q in fc.quarters
        ],
        "prob_drawdown": result.prob_drawdown,
        "regime": {"current": result.regime["current"],
                   "drift_annualized": result.regime["drift_annualized"]},
        "garch": {"persistence": result.garch.persistence, "long_run_vol": result.garch.long_run_vol},
        "drivers": drivers,
        "history": [{"date": d, "close": v} for d, v in hist.resample("W-FRI").last().dropna().items()],
    }


def market_snapshot(
    today: pd.Timestamp,
    macro_histories: dict[str, pd.DataFrame] | None = None,
    macro_table: pd.DataFrame | None = None,
    events: list | None = None,
    density: pd.DataFrame | None = None,
    news_frame: pd.DataFrame | None = None,
    news_daily: pd.DataFrame | None = None,
    warnings: list[str] | None = None,
    macro_years: int = 5,
    headline_days: int = 14,
) -> dict:
    """Macro panel, events, risk density and news, all optional."""
    today = pd.Timestamp(today).normalize()
    macro_hist = {}
    for sid in PANEL_SERIES:
        h = (macro_histories or {}).get(sid)
        if h is not None and not h.empty:
            h = h.loc[today - pd.DateOffset(years=macro_years):, ["value", "yoy", "trend_3m", "surprise_z"]]
            macro_hist[sid] = [{"date": d, **r} for d, r in h.to_dict("index").items()]

    ev = [{"date": e.date, "category": e.category, "name": e.name, "impact": e.impact,
           "points": e.points, "ticker": e.ticker, "source": e.source, "detail": e.detail}
          for e in (events or [])]

    sentiment, shocks, headlines = {}, [], []
    if news_daily is not None and not news_daily.empty:
        for tag, g in news_daily.groupby(level="tag"):
            g = g.droplevel("tag")
            recent = g.loc[today - pd.Timedelta(days=120):]
            sentiment[tag] = [{"date": d, "sentiment": r["sentiment"], "headlines": r["headlines"],
                               "z": r["z"], "shock": bool(r["shock"])} for d, r in recent.iterrows()]
            for d, r in recent[recent["shock"]].iterrows():
                shocks.append({"tag": tag, "date": d, "sentiment": r["sentiment"], "z": r["z"],
                               "headlines": r["headlines"]})
    if news_frame is not None and not news_frame.empty:
        f = news_frame[news_frame["date"] >= today - pd.Timedelta(days=headline_days)]
        f = f.sort_values("ts", ascending=False).drop_duplicates(["id", "tag"]).head(400)
        headlines = f[["date", "ts", "tag", "headline", "source", "url", "sentiment"]].to_dict("records")

    return {
        "version": SNAPSHOT_VERSION,
        "kind": "market",
        "generated_at": _now(),
        "today": today,
        "macro_table": [] if macro_table is None or macro_table.empty else
            macro_table.reset_index().to_dict("records"),
        "macro_history": macro_hist,
        "events": ev,
        "risk_density": [] if density is None or density.empty else
            density.reset_index().to_dict("records"),
        "sentiment": sentiment,
        "shocks": sorted(shocks, key=lambda s: s["date"], reverse=True),
        "headlines": headlines,
        "warnings": warnings or [],
    }


def backtest_summary(report, result, ticker: str, note: str = "") -> dict:
    """Compact accuracy summary from backtest.evaluate() / run_backtest()."""
    scored = result.scored
    stress = {}
    if not report.stress.empty:
        for w in report.stress.index.get_level_values(0).unique():
            s = report.stress.loc[w]
            stress[w] = {"n": s.loc["model", "n"],
                         "model_coverage": s.loc["model", "coverage"],
                         "model_pinball": s.loc["model", "pinball"],
                         "baseline_pinball": s.loc["baseline", "pinball"]}
    return {
        "version": SNAPSHOT_VERSION,
        "kind": "backtest",
        "ticker": ticker,
        "generated_at": _now(),
        "note": note,
        "first_origin": scored["origin"].min() if len(scored) else None,
        "first_target": scored["target"].min() if len(scored) else None,
        "last_target": scored["target"].max() if len(scored) else None,
        "n_origins": report.n_origins,
        "n_forecasts": int(len(scored)),
        "model": report.overall.loc["model"].to_dict(),
        "baseline": report.overall.loc["baseline"].to_dict(),
        "skill": report.skill,
        "diff_ci": list(report.diff_ci),
        "beats_baseline": bool(report.diff_ci[1] < 0),
        "worse_than_baseline": bool(report.diff_ci[0] > 0),
        "verdict": report.verdict,
        "stress": stress,
    }


# --------------------------------------------------------------------------- #
# I/O: refresh steps
# --------------------------------------------------------------------------- #

def load_ticker(settings: Settings, ticker: str) -> dict | None:
    return read_json(snapshot_dir(settings) / f"ticker_{safe_name(ticker)}.json")


def load_portfolio_snapshot(settings: Settings) -> dict | None:
    return read_json(snapshot_dir(settings) / "portfolio.json")


def load_market(settings: Settings) -> dict | None:
    return read_json(snapshot_dir(settings) / "market.json")


def load_backtest(settings: Settings, ticker: str) -> tuple[dict | None, bool]:
    """(summary, is_reference). Falls back to the committed S&P 500 reference."""
    own = read_json(snapshot_dir(settings) / f"backtest_{safe_name(ticker)}.json")
    if own is not None:
        return own, False
    return read_json(REFERENCE_BACKTEST), True


def refresh_ticker(settings: Settings, ticker: str, warn: Warn = print) -> dict:  # pragma: no cover
    from forecaster.forecast import ensemble_forecast, load_data

    data = load_data(ticker, settings, on_warn=warn)
    snap = ticker_snapshot(ensemble_forecast(data), data.prices,
                           currency=data.currency, converted_from=data.converted_from)
    write_json(snapshot_dir(settings) / f"ticker_{safe_name(ticker)}.json", snap)
    return snap


def backtest_is_stale(settings: Settings, ticker: str, max_age_days: float) -> bool:
    s = read_json(snapshot_dir(settings) / f"backtest_{safe_name(ticker)}.json")
    if s is None:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(s["generated_at"])
    return age.total_seconds() > max_age_days * 86400


def refresh_backtest(settings: Settings, ticker: str, warn: Warn = print,
                     progress=None) -> dict:  # pragma: no cover
    from forecaster.backtest import BacktestConfig, evaluate, run_backtest
    from forecaster.forecast import load_data

    data = load_data(ticker, settings, on_warn=warn)
    res = run_backtest(data, BacktestConfig(), progress=progress)
    summary = backtest_summary(evaluate(res), res, ticker)
    d = snapshot_dir(settings)
    res.rows.to_csv(d / f"backtest_{safe_name(ticker)}_rows.csv", index=False)
    write_json(d / f"backtest_{safe_name(ticker)}.json", summary)
    return summary


def refresh_market(settings: Settings, warn: Warn = print, score_news: bool = True) -> dict:  # pragma: no cover
    from forecaster.cache import Cache
    from forecaster.calendar import build_calendar, risk_density
    from forecaster.config import load_watchlist
    from forecaster.macro import fetch_indicators, macro_snapshot
    from forecaster.news import FinBertScorer, NewsStore, collect_news, daily_index, score_pending
    from forecaster.sources.finnhub import FinnhubClient
    from forecaster.sources.fred import FredClient

    warnings: list[str] = []

    def w(msg: str) -> None:
        warnings.append(msg)
        warn(msg)

    today = pd.Timestamp.today().normalize()
    cache = Cache(settings.cache_dir, ttl_seconds=settings.cache_ttl_hours * 3600)
    fred = FredClient(settings.fred_api_key, cache) if settings.fred_api_key else None
    fh_cache = Cache(settings.cache_dir / "news_http", ttl_seconds=3600)
    fh = FinnhubClient(settings.finnhub_api_key, fh_cache) if settings.finnhub_api_key else None

    hist, table = {}, None
    if fred is None:
        w("FRED_API_KEY not set: macro panel and macro release dates unavailable")
    else:
        hist = fetch_indicators(fred, on_error=lambda s, e: w(f"FRED {s}: {e}"))
        table = macro_snapshot(hist)

    if fh is None:
        w("FINNHUB_API_KEY not set: earnings dates and news unavailable")
    watch = load_watchlist(PROJECT_ROOT / "watchlist.yaml")
    events = build_calendar(today, fred, fh, watch, on_error=lambda s, e: w(f"calendar {s}: {e}"))
    density = risk_density(events, today)
    from forecaster import status as st

    fred_errs = [m for m in warnings if m.startswith("FRED ")]
    if fred is None:
        st.record(settings.cache_dir, "macro", False, "FRED_API_KEY missing")
    else:
        st.record(settings.cache_dir, "macro", bool(hist) and not fred_errs,
                  "; ".join(fred_errs)[:300] or ("no macro series returned" if not hist else None))
    cal_errs = [m for m in warnings if m.startswith("calendar ")]
    st.record(settings.cache_dir, "calendar", bool(events) and not cal_errs,
              "; ".join(cal_errs)[:300] or ("no events" if not events else None))

    frame = daily = None
    store_path = settings.cache_dir / "news.sqlite"
    if fh is not None:
        store = NewsStore(store_path)
        collect_news(fh, store, settings.tickers, today - pd.Timedelta(days=30), today,
                     on_error=lambda s, e: w(f"news {s}: {e}"))
        if score_news:
            try:
                score_pending(store, FinBertScorer())
            except Exception as exc:
                w(f"FinBERT scoring skipped ({exc}); install requirements-ml.txt")
    if store_path.exists():
        frame = NewsStore(store_path).scored_frame()
        daily = daily_index(frame)

    snap = market_snapshot(today, hist, table, events, density, frame, daily, warnings)
    write_json(snapshot_dir(settings) / "market.json", snap)
    return snap


# Shown on the app's "This Week" card: always forecast these, plus core + watchlist.
APP_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "GC=F")


def default_forecast_tickers(settings: Settings) -> list[str]:
    return list(dict.fromkeys([*settings.tickers, *APP_TICKERS]))


def refresh_all(settings: Settings, tickers: list[str] | None = None, warn: Warn = print,
                market: bool = True, score_news: bool = True,
                backtest_stale_days: float | None = None, forecasts: bool = True) -> dict[str, str]:  # pragma: no cover
    """Refresh market + portfolio + (optionally) every ticker's forecast.

    Per-step failures are reported, not raised, and recorded in data/status.json
    (the app's red banner reads it).
    """
    from forecaster import status as st

    status: dict[str, str] = {}
    if market:
        try:
            refresh_market(settings, warn, score_news)
            status["market"] = "ok"
            st.record(settings.cache_dir, "market", True)
        except Exception as exc:
            status["market"] = f"failed: {exc}"
            st.record(settings.cache_dir, "market", False, str(exc))
            warn(f"market: {exc}")
        from forecaster.portfolio import PORTFOLIO_FILE, refresh_portfolio

        if PORTFOLIO_FILE.exists():
            try:
                refresh_portfolio(settings, warn=warn)
                status["portfolio"] = "ok"
                st.record(settings.cache_dir, "portfolio", True)
            except Exception as exc:
                status["portfolio"] = f"failed: {exc}"
                st.record(settings.cache_dir, "portfolio", False, str(exc))
                warn(f"portfolio: {exc}")
    if not forecasts:
        return status
    failed = []
    for t in tickers or default_forecast_tickers(settings):
        try:
            refresh_ticker(settings, t, warn)
            status[t] = "ok"
        except Exception as exc:
            status[t] = f"failed: {exc}"
            failed.append(t)
            warn(f"{t}: {exc}")
            continue
        if backtest_stale_days is not None and backtest_is_stale(settings, t, backtest_stale_days):
            try:
                refresh_backtest(settings, t, warn)
                status[f"{t} backtest"] = "ok"
            except Exception as exc:
                status[f"{t} backtest"] = f"failed: {exc}"
                warn(f"{t} backtest: {exc}")
    st.record(settings.cache_dir, "forecasts", not failed,
              None if not failed else f"forecast failed for {', '.join(failed)}")
    return status


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import sys

    from forecaster.config import load_settings

    ap = argparse.ArgumentParser(description="Refresh dashboard snapshots")
    ap.add_argument("tickers", nargs="*", help="default: core + watchlist")
    ap.add_argument("--no-market", action="store_true", help="skip macro/events/news")
    ap.add_argument("--no-news-scoring", action="store_true", help="collect news but skip FinBERT")
    ap.add_argument("--backtest-stale-days", type=float, default=None,
                    help="also re-run each ticker's backtest if older than N days")
    args = ap.parse_args(argv)

    settings = load_settings()
    status = refresh_all(settings, args.tickers or None,
                         warn=lambda m: print(f"[warn] {m}", file=sys.stderr),
                         market=not args.no_market, score_news=not args.no_news_scoring,
                         backtest_stale_days=args.backtest_stale_days)
    for k, v in status.items():
        print(f"{k:>16}: {v}")
    return 0 if all(v == "ok" for v in status.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
