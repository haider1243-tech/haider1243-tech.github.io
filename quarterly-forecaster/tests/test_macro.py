import numpy as np
import pandas as pd
import pytest

from conftest import RoutingSession
from forecaster.macro import (
    INDICATORS,
    SeriesSpec,
    compute_indicator,
    fetch_indicators,
    macro_snapshot,
)
from forecaster.sources.fred import FredClient

CPI = SeriesSpec("CPIAUCSL", "CPI", "index", "M", annualize_trend=True)
UNRATE = SeriesSpec("UNRATE", "Unemployment", "rate", "M")
GDP = SeriesSpec("GDP", "GDP", "index", "Q", annualize_trend=True)
SENT = SeriesSpec("UMCSENT", "Sentiment", "index", "M")
SPREAD = SeriesSpec("T10Y2Y", "Spread", "rate", "D")


def monthly(values, start="2015-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype=float)


def steady_cpi(n=60, m=0.0025):
    return monthly(100 * (1 + m) ** np.arange(n))


def test_required_series_present():
    assert {s.series_id for s in INDICATORS} == {
        "CPIAUCSL", "CPILFESL", "PPIACO", "FEDFUNDS", "UNRATE", "T10Y2Y", "GDP", "UMCSENT"
    }


def test_index_series_yoy_trend_and_zero_surprise():
    df = compute_indicator(steady_cpi(), CPI)
    last = df.iloc[-1]
    assert last["change"] == pytest.approx(0.25)
    assert last["yoy"] == pytest.approx((1.0025 ** 12 - 1) * 100)
    assert last["trend_3m"] == pytest.approx((1.0025 ** 12 - 1) * 100)  # 3m annualized
    assert last["prior_trend"] == pytest.approx(0.25)
    assert last["surprise"] == pytest.approx(0, abs=1e-9)


def test_hot_print_is_positive_surprise():
    rng = np.random.default_rng(0)
    m = 0.0025 + rng.normal(0, 0.0005, 60)
    m[-1] = 0.008  # hot month
    s = monthly(100 * np.cumprod(1 + m))
    last = compute_indicator(s, CPI).iloc[-1]
    assert last["surprise"] > 0.4
    assert last["surprise_z"] > 3


def test_surprise_z_excludes_current_obs():
    rng = np.random.default_rng(1)
    s = monthly(100 * np.cumprod(1 + 0.002 + rng.normal(0, 0.001, 80)))
    df = compute_indicator(s, CPI)
    i = -1
    expected_sd = df["surprise"].iloc[-37:-1].std()
    assert df["surprise_z"].iloc[i] == pytest.approx(df["surprise"].iloc[i] / expected_sd)


def test_surprise_z_nan_until_enough_history():
    df = compute_indicator(steady_cpi(10), CPI)
    assert df["surprise_z"].isna().all()


def test_rate_series_uses_point_differences():
    s = monthly(np.linspace(3.5, 4.7, 25))  # +0.05pp / month
    last = compute_indicator(s, UNRATE).iloc[-1]
    assert last["change"] == pytest.approx(0.05)
    assert last["yoy"] == pytest.approx(0.6)
    assert last["trend_3m"] == pytest.approx(0.15)
    assert last["surprise"] == pytest.approx(0, abs=1e-9)


def test_quarterly_gdp_trend_is_qoq_annualized():
    idx = pd.date_range("2015-01-01", periods=20, freq="QS")
    s = pd.Series(20000 * 1.01 ** np.arange(20), index=idx)
    last = compute_indicator(s, GDP).iloc[-1]
    assert last["trend_3m"] == pytest.approx((1.01 ** 4 - 1) * 100)
    assert last["yoy"] == pytest.approx((1.01 ** 4 - 1) * 100)
    assert last["change"] == pytest.approx(1.0)


def test_non_annualized_trend():
    s = monthly(100 * 1.01 ** np.arange(24))
    assert compute_indicator(s, SENT).iloc[-1]["trend_3m"] == pytest.approx((1.01 ** 3 - 1) * 100)


def test_daily_series_resampled_to_monthly_mean():
    idx = pd.bdate_range("2024-01-01", "2025-12-31")
    s = pd.Series(np.where(idx.month % 2 == 0, 1.0, -1.0), index=idx)
    df = compute_indicator(s, SPREAD)
    assert len(df) == 24
    assert set(df["value"].round(9)) == {1.0, -1.0}
    assert df["yoy"].dropna().eq(0).all()


def test_missing_values_dropped():
    s = steady_cpi(30)
    s.iloc[10] = np.nan
    df = compute_indicator(s, CPI)
    assert not df["value"].isna().any()


# --------------------------------------------------------------------------- #
# FRED integration (mocked HTTP)
# --------------------------------------------------------------------------- #

def fred_obs_handler(series_map):
    def handler(url, params):
        assert url.endswith("series/observations")
        s = series_map.get(params["series_id"])
        if s is None:
            return 400, {"error_message": "Bad series"}
        return {"observations": [{"date": d.strftime("%Y-%m-%d"), "value": str(v)} for d, v in s.items()]}
    return handler


def test_fetch_indicators_cached_and_snapshot(cache):
    series = {spec.series_id: steady_cpi() for spec in INDICATORS}
    session = RoutingSession(fred_obs_handler(series))
    fred = FredClient("k", cache, session=session)

    hist = fetch_indicators(fred)
    fetch_indicators(fred)
    assert len(session.calls) == len(INDICATORS)  # second pass fully cached

    snap = macro_snapshot(hist)
    assert list(snap.index) == [s.series_id for s in INDICATORS]
    assert {"value", "yoy", "trend_3m", "surprise", "surprise_z", "units", "date"} <= set(snap.columns)
    assert snap.loc["UNRATE", "units"] == "pp"
    assert snap.loc["CPIAUCSL", "units"] == "pct"


def test_fetch_indicators_on_error_skips(cache):
    session = RoutingSession(fred_obs_handler({"CPIAUCSL": steady_cpi()}))
    fred = FredClient("k", cache, session=session)
    errors = []
    hist = fetch_indicators(fred, specs=(CPI, UNRATE), on_error=lambda s, e: errors.append(s))
    assert list(hist) == ["CPIAUCSL"]
    assert errors == ["UNRATE"]


def test_fred_error_does_not_leak_api_key(cache):
    session = RoutingSession(lambda url, params: (400, {}))
    fred = FredClient("SUPERSECRET", cache, session=session)
    with pytest.raises(RuntimeError) as exc:
        fred.get_series("NOPE")
    assert "SUPERSECRET" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_fred_api_key_not_in_cache_filenames(cache):
    session = RoutingSession(fred_obs_handler({"CPIAUCSL": steady_cpi()}))
    FredClient("SUPERSECRET", cache, session=session).get_series("CPIAUCSL")
    assert all("SUPERSECRET" not in p.name for p in cache.directory.iterdir())
