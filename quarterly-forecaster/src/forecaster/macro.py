"""Macro indicators from FRED: YoY change, 3-month trend, surprise vs prior trend.

Units
-----
Index-like series (CPI, core CPI, PPI, GDP, sentiment) use percent changes:
    change     period-over-period % change
    yoy        % change vs one year earlier
    trend_3m   % change over 3 months, annualized where `annualize_trend`
Rate-like series (Fed funds, unemployment, 10Y-2Y) use percentage-point
differences for all three (no annualizing).

Surprise
--------
Free sources have no consensus forecasts, so "surprise" is measured against
the series' own recent trend:
    prior_trend = mean of the previous 3 periods' `change`
    surprise    = change - prior_trend
    surprise_z  = surprise / std of past surprises (rolling, excludes current)
A positive CPI surprise_z means inflation came in hotter than its recent run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from forecaster.sources.fred import FredClient


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    name: str
    kind: Literal["index", "rate"]
    freq: Literal["D", "M", "Q"]
    annualize_trend: bool = False


INDICATORS: tuple[SeriesSpec, ...] = (
    SeriesSpec("CPIAUCSL", "CPI", "index", "M", annualize_trend=True),
    SeriesSpec("CPILFESL", "Core CPI", "index", "M", annualize_trend=True),
    SeriesSpec("PPIACO", "PPI (all commodities)", "index", "M", annualize_trend=True),
    SeriesSpec("FEDFUNDS", "Fed funds rate", "rate", "M"),
    SeriesSpec("UNRATE", "Unemployment rate", "rate", "M"),
    SeriesSpec("T10Y2Y", "10Y-2Y Treasury spread", "rate", "D"),
    SeriesSpec("GDP", "GDP (nominal, SAAR)", "index", "Q", annualize_trend=True),
    SeriesSpec("UMCSENT", "UMich consumer sentiment", "index", "M"),
)

PERIODS_PER_YEAR = {"M": 12, "Q": 4}
PRIOR_TREND_WINDOW = 3
SURPRISE_STD_WINDOW = 36
SURPRISE_STD_MIN_OBS = 12


def to_regular(series: pd.Series, freq: str) -> tuple[pd.Series, str]:
    """Put a series on a regular month/quarter grid. Daily -> monthly average."""
    s = series.dropna().sort_index()
    if freq == "D":
        return s.resample("MS").mean().dropna(), "M"
    rule = "MS" if freq == "M" else "QS"
    return s.resample(rule).last().dropna(), freq


def compute_indicator(series: pd.Series, spec: SeriesSpec) -> pd.DataFrame:
    """Full history with change, yoy, trend_3m, prior_trend, surprise, surprise_z."""
    s, freq = to_regular(series, spec.freq)
    ppy = PERIODS_PER_YEAR[freq]
    n3 = ppy // 4  # periods in 3 months: 3 monthly, 1 quarterly

    if spec.kind == "index":
        change = s.pct_change(fill_method=None) * 100
        yoy = (s / s.shift(ppy) - 1) * 100
        ratio = s / s.shift(n3)
        trend = ((ratio ** 4 - 1) if spec.annualize_trend else (ratio - 1)) * 100
    else:
        change = s.diff()
        yoy = s - s.shift(ppy)
        trend = s - s.shift(n3)

    prior = change.shift(1).rolling(PRIOR_TREND_WINDOW).mean()
    surprise = change - prior
    sd = surprise.shift(1).rolling(SURPRISE_STD_WINDOW, min_periods=SURPRISE_STD_MIN_OBS).std()
    z = surprise / sd.replace(0, np.nan)

    return pd.DataFrame(
        {
            "value": s,
            "change": change,
            "yoy": yoy,
            "trend_3m": trend,
            "prior_trend": prior,
            "surprise": surprise,
            "surprise_z": z,
        }
    )


def latest_row(df: pd.DataFrame, spec: SeriesSpec) -> dict:
    last = df.dropna(subset=["value"]).iloc[-1]
    return {
        "series_id": spec.series_id,
        "name": spec.name,
        "units": "pct" if spec.kind == "index" else "pp",
        "date": last.name,
        **{k: float(last[k]) for k in df.columns},
    }


def fetch_indicators(
    fred: FredClient,
    specs: tuple[SeriesSpec, ...] = INDICATORS,
    on_error=None,
) -> dict[str, pd.DataFrame]:
    """Download (cached) and compute every indicator. Returns {series_id: history}."""
    out: dict[str, pd.DataFrame] = {}
    for spec in specs:
        try:
            out[spec.series_id] = compute_indicator(fred.get_series(spec.series_id), spec)
        except Exception as exc:
            if on_error is None:
                raise
            on_error(spec.series_id, exc)
    return out


def macro_snapshot(histories: dict[str, pd.DataFrame], specs=INDICATORS) -> pd.DataFrame:
    """Latest value of each indicator with its yoy / trend / surprise metrics."""
    rows = [latest_row(histories[s.series_id], s) for s in specs if s.series_id in histories]
    return pd.DataFrame(rows).set_index("series_id") if rows else pd.DataFrame()


def main() -> int:  # pragma: no cover - network wrapper
    import sys

    from forecaster.cache import Cache
    from forecaster.config import load_settings

    settings = load_settings()
    if not settings.fred_api_key:
        print("FRED_API_KEY is not set: copy .env.example to .env and add a key", file=sys.stderr)
        return 1
    cache = Cache(settings.cache_dir, ttl_seconds=settings.cache_ttl_hours * 3600)
    fred = FredClient(settings.fred_api_key, cache)
    hist = fetch_indicators(fred, on_error=lambda s, e: print(f"[warn] {s}: {e}", file=sys.stderr))
    if not hist:
        return 1
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(macro_snapshot(hist).round(2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
