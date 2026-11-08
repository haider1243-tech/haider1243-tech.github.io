"""Synthetic snapshots for pipeline/app tests (clearly fake data, no network)."""
import numpy as np
import pandas as pd

from forecaster.calendar import Event, fomc_events, risk_density
from forecaster.forecast import ensemble_forecast
from forecaster.macro import INDICATORS, compute_indicator
from forecaster.news import daily_index
from forecaster.pipeline import market_snapshot, ticker_snapshot, write_json
from test_forecast import FAST, simulate_world

TODAY = pd.Timestamp.today().normalize()


def make_ticker_snapshot(asof="2021-06-15"):
    world, _ = simulate_world()
    res = ensemble_forecast(world, asof=pd.Timestamp(asof), config=FAST)
    return ticker_snapshot(res, world.prices), res


def make_market_snapshot(today=TODAY):
    months = pd.date_range(today - pd.DateOffset(years=7), today, freq="MS")
    rng = np.random.default_rng(0)
    hist = {}
    for spec in INDICATORS:
        if spec.series_id not in ("CPIAUCSL", "CPILFESL", "PPIACO", "FEDFUNDS"):
            continue
        base = 3.0 + np.cumsum(rng.normal(0, 0.1, len(months))) if spec.kind == "rate" else \
            100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.002, len(months)))
        hist[spec.series_id] = compute_indicator(pd.Series(base, index=months), spec)
    events = fomc_events(today, today + pd.Timedelta(days=400)) + [
        Event(today + pd.Timedelta(days=10), "CPI", "CPI release", "high", "FRED"),
        Event(today + pd.Timedelta(days=12), "PPI", "PPI release", "medium", "FRED"),
        Event(today + pd.Timedelta(days=20), "Earnings", "AAPL earnings", "medium", "Finnhub", ticker="AAPL"),
    ]
    density = risk_density(events, today)
    days = pd.date_range(today - pd.Timedelta(days=60), today - pd.Timedelta(days=1))
    rows, k = [], 0
    for d in days:
        for tag in ("MARKET", "AAPL"):
            for _ in range(4):
                s = float(np.clip(rng.normal(-0.6 if d == days[-5] else 0.05, 0.15), -1, 1))
                rows.append({"id": str(k), "ts": int(d.timestamp()) + k, "headline": f"Fake headline {k}",
                             "source": "Test", "url": f"https://example.com/{k}", "tag": tag, "kind": "x",
                             "date": d, "positive": max(s, 0), "negative": max(-s, 0), "neutral": 0.0,
                             "sentiment": s})
                k += 1
    frame = pd.DataFrame(rows)
    return market_snapshot(today, hist, None, events, density, frame, daily_index(frame), ["test warning"])


def write_fixture_snapshots(snap_dir, ticker="SYN"):
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts, _ = make_ticker_snapshot()
    ts["ticker"] = ticker
    write_json(snap_dir / f"ticker_{ticker}.json", ts)
    write_json(snap_dir / "market.json", make_market_snapshot())
    return ts
