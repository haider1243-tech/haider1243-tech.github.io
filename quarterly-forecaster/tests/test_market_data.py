from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from forecaster.market_data import (
    MACRO_TICKERS,
    SECTOR_ETFS,
    ParquetStore,
    beta,
    compute_metrics,
    correlation_matrix,
    fetch_universe,
    last_completed_business_day,
    market_universe,
    max_drawdown,
    moving_average_trend,
    normalize_ohlcv,
    quarterly_beta,
    quarterly_max_drawdown,
    quarterly_returns,
    realized_volatility,
    rsi,
    to_quarterly,
    to_weekly,
)


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #

def make_ohlcv(close: np.ndarray, start="2020-01-01", tz=None) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(close), tz=tz)
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.full(len(close), 1_000.0),
            "Dividends": 0.0,  # extra yfinance column, should be dropped
        },
        index=idx,
    )


def random_walk(n=800, seed=0, drift=0.0003, vol=0.01, start=100.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


class FakeDownloader:
    """Serves slices of a fixed 'full history' and records each request."""

    def __init__(self, histories: dict[str, pd.DataFrame]):
        self.histories = histories
        self.calls: list[tuple[str, pd.Timestamp | None]] = []

    def __call__(self, ticker, start):
        self.calls.append((ticker, start))
        df = self.histories.get(ticker, pd.DataFrame())
        if start is not None and not df.empty:
            df = df[df.index.tz_localize(None) >= start] if df.index.tz else df[df.index >= start]
        return df


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self):
        return self.t


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #

def test_universe_includes_required_tickers_deduped():
    u = market_universe(("SPY", "AAPL", "XLK"))
    for t in (*MACRO_TICKERS, *SECTOR_ETFS, "AAPL"):
        assert t in u
    assert len(u) == len(set(u))
    assert u[:2] == ("SPY", "AAPL")
    assert {"^TNX", "DX-Y.NYB", "GC=F"} <= set(u)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

def test_normalize_drops_extra_columns_and_tz():
    df = make_ohlcv(random_walk(10), tz="America/New_York")
    out = normalize_ohlcv(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing


def test_normalize_rejects_missing_columns():
    with pytest.raises(ValueError):
        normalize_ohlcv(pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-01-02"])))


def test_normalize_empty():
    assert normalize_ohlcv(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# Parquet store: stale-only refresh
# --------------------------------------------------------------------------- #

@pytest.fixture
def history():
    # 2020-01-01 .. ~2022-12 business days
    return make_ohlcv(random_walk(780), start="2020-01-01")


def test_last_completed_business_day():
    assert last_completed_business_day(datetime(2024, 6, 10, 9)) == pd.Timestamp("2024-06-07")  # Mon -> Fri
    assert last_completed_business_day(datetime(2024, 6, 12, 9)) == pd.Timestamp("2024-06-11")


def test_store_first_fetch_writes_parquet(tmp_path, history):
    last = history.index[-1]
    dl = FakeDownloader({"SPY": history})
    store = ParquetStore(tmp_path, downloader=dl, now=Clock(last + timedelta(days=1, hours=9)))
    df = store.get("SPY")
    assert dl.calls == [("SPY", None)]
    assert store.path("SPY").exists()
    assert store.path("SPY").suffix == ".parquet"
    assert len(df) == len(history)


def test_store_fresh_data_not_redownloaded(tmp_path, history):
    last = history.index[-1]
    dl = FakeDownloader({"SPY": history})
    store = ParquetStore(tmp_path, downloader=dl, now=Clock(last + timedelta(days=1, hours=9)))
    store.get("SPY")
    store.get("SPY")
    # New store instance (fresh process) reads parquet without downloading.
    ParquetStore(tmp_path, downloader=dl, now=Clock(last + timedelta(days=1, hours=9))).get("SPY")
    assert len(dl.calls) == 1


def test_store_stale_does_incremental_refresh(tmp_path, history):
    split = history.index[600]
    clock = Clock(split.to_pydatetime() + timedelta(days=1, hours=9))
    dl = FakeDownloader({"SPY": history[history.index <= split]})
    store = ParquetStore(tmp_path, downloader=dl, now=clock, overlap_days=5)
    store.get("SPY")

    # Time passes; source now has full history.
    dl.histories["SPY"] = history
    clock.t = history.index[-1].to_pydatetime() + timedelta(days=1, hours=9)
    df = store.get("SPY")

    assert dl.calls[-1] == ("SPY", split - pd.Timedelta(days=5))
    assert len(df) == len(history)
    assert not df.index.duplicated().any()
    pd.testing.assert_series_equal(
        df["Close"], normalize_ohlcv(history)["Close"], check_freq=False
    )


def test_store_recheck_throttle_on_holidays(tmp_path, history):
    last = history.index[-1]
    clock = Clock(last.to_pydatetime() + timedelta(days=3, hours=9))  # >1 bday later
    dl = FakeDownloader({"SPY": history})
    store = ParquetStore(tmp_path, downloader=dl, now=clock, recheck_after=timedelta(hours=12))
    store.get("SPY")          # full download
    store.get("SPY")          # stale by date, but checked just now -> no call
    assert len(dl.calls) == 1
    clock.t += timedelta(hours=13)
    store.get("SPY")          # recheck window passed -> incremental call
    assert len(dl.calls) == 2


def test_store_empty_refresh_keeps_cached(tmp_path, history):
    clock = Clock(history.index[-1].to_pydatetime() + timedelta(days=1, hours=9))
    dl = FakeDownloader({"SPY": history})
    store = ParquetStore(tmp_path, downloader=dl, now=clock)
    store.get("SPY")
    dl.histories["SPY"] = pd.DataFrame()  # source fails
    clock.t += timedelta(days=5)
    df = store.get("SPY")
    assert len(df) == len(history)


def test_store_empty_first_fetch_raises_and_writes_nothing(tmp_path):
    store = ParquetStore(tmp_path, downloader=FakeDownloader({}), now=Clock(datetime(2024, 1, 10)))
    with pytest.raises(ValueError):
        store.get("NOPE")
    assert not store.path("NOPE").exists()


def test_store_force_full_refresh(tmp_path, history):
    clock = Clock(history.index[-1].to_pydatetime() + timedelta(days=1, hours=9))
    dl = FakeDownloader({"SPY": history})
    store = ParquetStore(tmp_path, downloader=dl, now=clock)
    store.get("SPY")
    store.get("SPY", force=True)
    assert dl.calls == [("SPY", None), ("SPY", None)]


def test_safe_filenames(tmp_path):
    store = ParquetStore(tmp_path, downloader=FakeDownloader({}))
    for t in ("^VIX", "DX-Y.NYB", "GC=F"):
        name = store.path(t).name
        assert "^" not in name and "=" not in name and "/" not in name


def test_fetch_universe_skips_failures(tmp_path, history):
    clock = Clock(history.index[-1].to_pydatetime() + timedelta(days=1, hours=9))
    store = ParquetStore(tmp_path, downloader=FakeDownloader({"SPY": history}), now=clock)
    errors = []
    out = fetch_universe(["SPY", "BAD"], store, on_error=lambda t, e: errors.append(t))
    assert list(out) == ["SPY"]
    assert errors == ["BAD"]


def test_fetch_universe_raises_without_handler(tmp_path):
    store = ParquetStore(tmp_path, downloader=FakeDownloader({}), now=Clock(datetime(2024, 1, 10)))
    with pytest.raises(ValueError):
        fetch_universe(["BAD"], store)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #

def test_weekly_resample_ohlc_rules():
    df = normalize_ohlcv(make_ohlcv(np.arange(1, 11), start="2024-01-01"))  # Mon 1 .. Fri 12
    w = to_weekly(df)
    assert len(w) == 2
    wk1 = w.iloc[0]
    assert wk1["Open"] == pytest.approx(1 * 0.99)
    assert wk1["Close"] == 5
    assert wk1["High"] == pytest.approx(5 * 1.01)
    assert wk1["Low"] == pytest.approx(1 * 0.98)
    assert wk1["Volume"] == 5_000
    assert w.index[0] == pd.Timestamp("2024-01-05")  # Friday label


def test_quarterly_resample():
    df = normalize_ohlcv(make_ohlcv(random_walk(260), start="2024-01-01"))
    q = to_quarterly(df)
    assert all(d.is_quarter_end for d in q.index)
    assert q["Close"].iloc[0] == df.loc[:"2024-03-31", "Close"].iloc[-1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_quarterly_returns_known_values():
    idx = pd.to_datetime(["2024-03-29", "2024-06-28", "2024-09-30"])
    s = pd.Series([100.0, 110.0, 99.0], index=idx)
    r = quarterly_returns(s)
    assert r.tolist() == pytest.approx([0.10, -0.10])


def test_realized_vol_zero_for_constant_growth():
    idx = pd.bdate_range("2024-01-01", "2024-06-28")
    s = pd.Series(100 * np.exp(0.001 * np.arange(len(idx))), index=idx)
    assert np.allclose(realized_volatility(s).to_numpy(), 0, atol=1e-12)


def test_realized_vol_matches_input_vol():
    idx = pd.bdate_range("2000-01-01", periods=5000)
    s = pd.Series(random_walk(5000, vol=0.01, drift=0), index=idx)
    rv = realized_volatility(s, rule="YE")
    assert rv.mean() == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_max_drawdown_known():
    s = pd.Series([100, 120, 90, 130, 104], dtype=float)
    assert max_drawdown(s) == pytest.approx(-0.25)
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0.0


def test_quarterly_max_drawdown_resets_each_quarter():
    idx = pd.to_datetime(["2024-01-02", "2024-02-01", "2024-04-01", "2024-05-01"])
    s = pd.Series([100, 50, 60, 66], index=idx, dtype=float)
    dd = quarterly_max_drawdown(s)
    assert dd.tolist() == pytest.approx([-0.5, 0.0])


def test_ma_trend_labels():
    idx = pd.bdate_range("2020-01-01", periods=400)
    up = moving_average_trend(pd.Series(np.linspace(100, 200, 400), index=idx))
    down = moving_average_trend(pd.Series(np.linspace(200, 100, 400), index=idx))
    assert up["trend"].iloc[-1] == "up" and bool(up["golden_cross"].iloc[-1])
    assert down["trend"].iloc[-1] == "down"
    assert up["trend"].iloc[:199].isna().all()
    assert up["ma50"].iloc[-1] == pytest.approx(np.linspace(100, 200, 400)[-50:].mean())


def test_rsi_bounds_and_extremes():
    idx = pd.bdate_range("2024-01-01", periods=60)
    assert rsi(pd.Series(np.arange(60, dtype=float), index=idx)).iloc[-1] == 100
    assert rsi(pd.Series(np.arange(60, 0, -1, dtype=float), index=idx)).iloc[-1] == 0
    assert rsi(pd.Series(np.full(60, 5.0), index=idx)).iloc[-1] == 50
    r = rsi(pd.Series(random_walk(500), index=pd.bdate_range("2020-01-01", periods=500))).dropna()
    assert r.between(0, 100).all()
    assert rsi(pd.Series(np.arange(60, dtype=float), index=idx)).iloc[:13].isna().all()


def test_rsi_wilder_reference_value():
    # Classic Wilder example (StockCharts "RSI" ChartSchool table).
    closes = [44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245,
              45.8433, 46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028,
              46.0328, 46.4116, 46.2222, 45.6439]
    s = pd.Series(closes, index=pd.bdate_range("2024-01-01", periods=len(closes)))
    r = rsi(s)
    assert r.iloc[14] == pytest.approx(70.53, abs=0.05)
    assert r.iloc[15] == pytest.approx(66.32, abs=0.05)
    assert r.iloc[16] == pytest.approx(66.55, abs=0.05)


def test_beta_exact_multiple():
    idx = pd.bdate_range("2020-01-01", periods=500)
    bench = pd.Series(random_walk(500, seed=1), index=idx)
    asset = bench ** 2  # log returns exactly 2x
    assert beta(asset, bench) == pytest.approx(2.0)
    assert beta(bench, bench) == pytest.approx(1.0)
    qb = quarterly_beta(asset, bench)
    assert len(qb) > 0
    assert np.allclose(qb.to_numpy(), 2.0)


def test_beta_aligns_mismatched_calendars():
    idx = pd.bdate_range("2020-01-01", periods=300)
    bench = pd.Series(random_walk(300, seed=2), index=idx)
    asset = (bench ** 1.5).drop(idx[::7])  # missing days
    assert beta(asset, bench) == pytest.approx(1.5, rel=0.05)


def test_correlation_matrix():
    idx = pd.bdate_range("2020-01-01", periods=600)
    a = pd.Series(random_walk(600, seed=3), index=idx)
    b = pd.Series(random_walk(600, seed=4), index=idx)
    closes = {"A": a, "A2": a ** 2, "B": b, "INV": 1 / a}
    c = correlation_matrix(closes)
    assert np.allclose(np.diag(c), 1.0)
    assert c.loc["A", "A2"] == pytest.approx(1.0)
    assert c.loc["A", "INV"] == pytest.approx(-1.0)
    assert abs(c.loc["A", "B"]) < 0.2
    assert (c.to_numpy() == c.to_numpy().T).all()
    assert correlation_matrix(closes, rule="W-FRI", lookback=52).shape == (4, 4)


# --------------------------------------------------------------------------- #
# End to end with mocked downloads
# --------------------------------------------------------------------------- #

def test_compute_metrics_end_to_end(tmp_path):
    tickers = market_universe(("AAPL",))
    histories = {
        t: make_ohlcv(random_walk(800, seed=i), start="2021-01-01", tz="America/New_York")
        for i, t in enumerate(tickers)
    }
    last = pd.bdate_range("2021-01-01", periods=800)[-1]
    store = ParquetStore(tmp_path, downloader=FakeDownloader(histories),
                         now=Clock(last.to_pydatetime() + timedelta(days=1, hours=9)))
    frames = fetch_universe(tickers, store)
    m = compute_metrics(frames)

    assert set(m.snapshot.index) == set(tickers)
    assert m.snapshot.loc["SPY", "beta_vs_SPY"] == pytest.approx(1.0)
    assert m.snapshot["rsi14"].between(0, 100).all()
    assert (m.snapshot["max_drawdown"] <= 0).all()
    assert set(m.snapshot["trend"]) <= {"up", "down", "mixed"}

    assert m.quarterly.index.names == ["ticker", "quarter_end"]
    assert {"return", "realized_vol", "max_drawdown", "beta"} <= set(m.quarterly.columns)
    spy_q = m.quarterly.loc["SPY"]
    assert np.allclose(spy_q["beta"].dropna(), 1.0)

    assert m.correlation.shape == (len(tickers), len(tickers))
    assert len(list(tmp_path.glob("*.parquet"))) == len(tickers)


def test_compute_metrics_without_benchmark():
    df = normalize_ohlcv(make_ohlcv(random_walk(300)))
    m = compute_metrics({"AAPL": df})
    assert np.isnan(m.snapshot.loc["AAPL", "beta_vs_SPY"])
