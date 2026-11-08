"""Probabilistic forecasting. Outputs percentile ranges — never a single point.

Baseline model: block bootstrap of historical quarterly log returns. Each
simulated path samples contiguous blocks of real quarters (preserving some
autocorrelation / volatility clustering), compounds them forward, and the
percentiles of the simulated price distribution form the forecast.

This is a baseline to beat, not a final model. Any new model must return a
`Forecast` so the percentile contract is enforced in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecaster.config import HORIZON_QUARTERS, PERCENTILES


@dataclass(frozen=True)
class QuarterRange:
    quarter: str  # e.g. "2027Q1"
    p10: float
    p50: float
    p90: float

    def __post_init__(self) -> None:
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError(f"Percentiles out of order for {self.quarter}: {self}")


@dataclass(frozen=True)
class Forecast:
    ticker: str
    as_of: pd.Timestamp
    last_close: float
    model: str
    quarters: tuple[QuarterRange, ...]

    def __post_init__(self) -> None:
        if len(self.quarters) != HORIZON_QUARTERS:
            raise ValueError(f"Forecast must cover {HORIZON_QUARTERS} quarters")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": self.ticker, "quarter": q.quarter, "p10": q.p10, "p50": q.p50, "p90": q.p90}
                for q in self.quarters
            ]
        )


def future_quarter_labels(as_of: pd.Timestamp, n: int = HORIZON_QUARTERS) -> list[str]:
    start = pd.Period(as_of, freq="Q") + 1
    return [str(start + i) for i in range(n)]


def block_bootstrap_forecast(
    ticker: str,
    quarterly_closes: pd.Series,
    n_paths: int = 10_000,
    block_size: int = 2,
    min_history: int = 20,
    seed: int | None = 42,
) -> Forecast:
    """Forecast 4 quarters ahead from quarterly closes via block bootstrap."""
    closes = quarterly_closes.dropna()
    log_ret = np.log(closes / closes.shift(1)).dropna().to_numpy()
    if len(log_ret) < min_history:
        raise ValueError(
            f"{ticker}: need >= {min_history} quarters of history, got {len(log_ret)}"
        )

    rng = np.random.default_rng(seed)
    h = HORIZON_QUARTERS
    n_blocks = int(np.ceil(h / block_size))
    starts = rng.integers(0, len(log_ret) - block_size + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_size)).reshape(n_paths, -1)[:, :h]
    paths = closes.iloc[-1] * np.exp(np.cumsum(log_ret[idx], axis=1))

    pct = np.percentile(paths, PERCENTILES, axis=0)  # shape (3, h)
    labels = future_quarter_labels(closes.index[-1])
    quarters = tuple(
        QuarterRange(labels[i], float(pct[0, i]), float(pct[1, i]), float(pct[2, i]))
        for i in range(h)
    )
    return Forecast(
        ticker=ticker,
        as_of=pd.Timestamp(closes.index[-1]),
        last_close=float(closes.iloc[-1]),
        model=f"block_bootstrap(block={block_size}, paths={n_paths})",
        quarters=quarters,
    )
