import pandas as pd
import pytest

from forecaster.model import (
    Forecast,
    QuarterRange,
    block_bootstrap_forecast,
    future_quarter_labels,
)
from forecaster.sources.prices import to_quarterly


@pytest.fixture
def quarterly(daily_prices):
    return to_quarterly(daily_prices["Close"])


def test_forecast_has_four_ordered_quarters(quarterly):
    f = block_bootstrap_forecast("TEST", quarterly, n_paths=2000)
    assert len(f.quarters) == 4
    for q in f.quarters:
        assert q.p10 < q.p50 < q.p90


def test_uncertainty_widens_with_horizon(quarterly):
    f = block_bootstrap_forecast("TEST", quarterly, n_paths=5000)
    widths = [q.p90 - q.p10 for q in f.quarters]
    assert widths == sorted(widths)


def test_deterministic_with_seed(quarterly):
    a = block_bootstrap_forecast("T", quarterly, n_paths=1000, seed=1)
    b = block_bootstrap_forecast("T", quarterly, n_paths=1000, seed=1)
    assert a == b


def test_quarter_labels_start_next_quarter():
    assert future_quarter_labels(pd.Timestamp("2025-09-30")) == [
        "2025Q4", "2026Q1", "2026Q2", "2026Q3",
    ]


def test_rejects_short_history(quarterly):
    with pytest.raises(ValueError):
        block_bootstrap_forecast("T", quarterly.iloc[:5])


def test_quarter_range_rejects_point_or_misordered():
    with pytest.raises(ValueError):
        QuarterRange("2026Q1", p10=110, p50=100, p90=120)


def test_forecast_requires_four_quarters():
    q = QuarterRange("2026Q1", 90, 100, 110)
    with pytest.raises(ValueError):
        Forecast("T", pd.Timestamp("2025-12-31"), 100.0, "m", (q,))


def test_to_frame_columns(quarterly):
    df = block_bootstrap_forecast("T", quarterly, n_paths=500).to_frame()
    assert list(df.columns) == ["ticker", "quarter", "p10", "p50", "p90"]
