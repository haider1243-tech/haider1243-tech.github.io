"""USD -> CAD conversion: classification, rate alignment, and CAD labels everywhere."""
import numpy as np
import pandas as pd
import pytest

from forecaster.config import load_currency_overrides, load_settings
from forecaster.dashboard import PALETTES, fan_chart, forecast_table, money
from forecaster.market_data import (
    CANADIAN_ETFS, USD_CAD_SYMBOL, ParquetStore, close_in_cad, compute_metrics, fetch_universe,
    market_universe, rate_on_dates, ticker_currency, to_cad, usd_cad_rates, validate_usd_cad,
)
from forecaster.vwap_backtest import cad_view
from forecaster.webhook_server import Alert, format_message, price_text


def rates_series(values=(1.30, 1.35, 1.40), start="2024-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def ohlcv(n=3, start="2024-01-02", price=100.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1, "Close": price, "Volume": 1000.0},
                        index=idx)


# --------------------------------------------------------------------------- #
# Classification and watchlist
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("t,cur", [
    ("XIU.TO", "CAD"), ("XEQT.TO", "CAD"), ("VFV.TO", "CAD"), ("ZAG.TO", "CAD"), ("ABC.V", "CAD"),
    ("SPY", "USD"), ("GC=F", "USD"), ("^GSPC", "USD"), ("XAUUSD", "USD"),
    ("^VIX", None), ("^TNX", None), ("DX-Y.NYB", None), ("CAD=X", None),
])
def test_ticker_currency(t, cur):
    assert ticker_currency(t) == cur


def test_overrides():
    assert ticker_currency("BTC-CAD", {"BTC-CAD": "CAD"}) == "CAD"
    assert ticker_currency("^GSPTSE", {"^GSPTSE": "none"}) is None


def test_watchlist_has_canadian_etfs_and_universe_has_fx():
    s = load_settings()
    assert set(CANADIAN_ETFS) <= set(s.tickers)
    u = market_universe(s.tickers)
    assert USD_CAD_SYMBOL in u and set(CANADIAN_ETFS) <= set(u)
    assert load_currency_overrides() == {}


def test_currency_overrides_file(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text("tickers: []\ncurrencies:\n  btc-usd: USD\n  ^GSPTSE: null\n")
    assert load_currency_overrides(p) == {"BTC-USD": "USD", "^GSPTSE": "none"}


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #

def test_rejects_inverted_or_broken_rates():
    with pytest.raises(ValueError, match="inverted"):
        validate_usd_cad(pd.Series([0.73, 0.74], index=pd.bdate_range("2024-01-02", periods=2)))
    with pytest.raises(ValueError):
        validate_usd_cad(pd.Series([], dtype=float))


def test_rate_on_dates_never_uses_a_later_rate():
    r = rates_series((1.30, 1.40), start="2024-01-02")            # Tue, Wed
    d = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-06"])
    v = rate_on_dates(r, d)
    assert np.isnan(v.iloc[0])                                      # before the first rate: unknown
    assert list(v.iloc[1:]) == [1.30, 1.40, 1.40]                   # holiday/weekend: previous rate


def test_rate_on_dates_handles_tz_aware_bars():
    r = rates_series((1.30, 1.40))
    d = pd.DatetimeIndex(["2024-01-02 15:00", "2024-01-03 15:00"], tz="UTC")
    assert list(rate_on_dates(r, d)) == [1.30, 1.40]


def test_usd_cad_rates_backfills_from_fred(tmp_path):
    yahoo = ohlcv(3, "2024-01-02").assign(Close=[1.35, 1.36, 1.37])
    store = ParquetStore(tmp_path, downloader=lambda t, s: yahoo, now=lambda: pd.Timestamp("2024-01-05 10:00"))

    class FakeFred:
        def get_series(self, sid):
            assert sid == "DEXCAUS"
            return pd.Series([1.20, 1.21, 1.99], index=pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02"]))

    r = usd_cad_rates(store, FakeFred())
    assert r.index.min() == pd.Timestamp("2023-12-28")
    assert r.loc["2024-01-02"] == 1.35                              # Yahoo wins where both exist
    warned = []

    class BrokenFred:
        def get_series(self, sid):
            raise RuntimeError("no key")

    r2 = usd_cad_rates(store, BrokenFred(), on_warn=warned.append)
    assert r2.index.min() == pd.Timestamp("2024-01-02") and warned


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def test_to_cad_converts_usd_prices_not_volume():
    out = to_cad(ohlcv(3), "SPY", rates_series())
    assert list(out["Close"]) == pytest.approx([130, 135, 140])
    assert list(out["High"]) == pytest.approx([101 * 1.30, 101 * 1.35, 101 * 1.40])
    assert (out["Volume"] == 1000).all()
    assert out.attrs == {"currency": "CAD", "converted_from": "USD"}


def test_to_cad_leaves_cad_and_index_series_alone():
    df = ohlcv(3)
    for t, cur in (("XIU.TO", "CAD"), ("^VIX", None), ("^TNX", None)):
        out = to_cad(df, t, rates_series())
        pd.testing.assert_frame_equal(out, df)
        assert out.attrs["currency"] == cur and out.attrs["converted_from"] is None


def test_to_cad_drops_bars_before_first_rate_and_requires_rates():
    out = to_cad(ohlcv(4, "2023-12-28"), "SPY", rates_series(start="2024-01-02"))
    assert out.index.min() == pd.Timestamp("2024-01-02")
    with pytest.raises(ValueError, match="USD/CAD"):
        to_cad(ohlcv(3), "SPY", None)


def test_conversion_is_point_in_time():
    """Changing future FX rates must not change converted past prices."""
    r = rates_series((1.30, 1.35, 1.40, 1.45))
    df = ohlcv(4)
    base = to_cad(df, "SPY", r)
    r2 = r.copy()
    r2.iloc[-1] = 1.99
    pd.testing.assert_frame_equal(to_cad(df, "SPY", r2).iloc[:3], base.iloc[:3])


def test_fetch_universe_and_metrics_in_cad(tmp_path):
    frames = {"SPY": ohlcv(300, "2023-01-02"), "XIU.TO": ohlcv(300, "2023-01-02", 30.0),
              "^VIX": ohlcv(300, "2023-01-02", 15.0), USD_CAD_SYMBOL: ohlcv(300, "2023-01-02", 1.35)}
    rng = np.random.default_rng(0)
    for k in frames:
        frames[k]["Close"] = frames[k]["Close"] * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    store = ParquetStore(tmp_path, downloader=lambda t, s: frames[t], now=lambda: pd.Timestamp("2024-03-01"))
    rates = frames[USD_CAD_SYMBOL]["Close"]
    out = fetch_universe(list(frames), store, rates=rates)
    spy_raw = store.get("SPY")["Close"]
    assert out["SPY"]["Close"].iloc[-1] == pytest.approx(spy_raw.iloc[-1] * rates.iloc[-1])
    assert out["XIU.TO"]["Close"].iloc[-1] == pytest.approx(store.get("XIU.TO")["Close"].iloc[-1])
    assert out["^VIX"]["Close"].iloc[-1] == pytest.approx(store.get("^VIX")["Close"].iloc[-1])
    snap = compute_metrics(out).snapshot
    assert snap.loc["SPY", "currency"] == "CAD" and snap.loc["XIU.TO", "currency"] == "CAD"
    assert snap.loc["^VIX", "currency"] == "index/yield"
    s = close_in_cad(store, "SPY", rates)
    assert s.attrs["converted_from"] == "USD" and s.iloc[-1] == pytest.approx(out["SPY"]["Close"].iloc[-1])


# --------------------------------------------------------------------------- #
# Reports show CAD
# --------------------------------------------------------------------------- #

def _snap(currency="CAD", converted="USD"):
    return {"currency": currency, "converted_from": converted, "as_of": "2026-11-06", "last_close": 700.0,
            "history": [{"date": "2026-10-30", "close": 690.0}, {"date": "2026-11-06", "close": 700.0}],
            "forecast": [{"quarter": f"2027Q{i}", "target_date": f"2027-{3 * i:02d}-28", "p10": 600 + i,
                          "p50": 700 + i, "p90": 800 + i, "prob_drawdown": 0.2} for i in range(1, 5)]}


def test_fan_chart_and_table_label_cad():
    fig = fan_chart(_snap(), PALETTES["light"])
    assert fig.layout.yaxis.title.text == "Price (CAD)"
    assert any(a.text.startswith("P50 C$") for a in fig.layout.annotations)
    assert "C$" in fig.data[0].hovertemplate
    assert {"P10 (CAD)", "P50 (CAD)", "P90 (CAD)"} <= set(forecast_table(_snap()).columns)
    assert money(1234.5, _snap()) == "C$1,234.50"


def test_index_series_are_not_labelled_as_money():
    s = _snap(currency=None, converted=None)
    assert fan_chart(s, PALETTES["light"]).layout.yaxis.title.text == "Price (index points)"
    assert "P50 (pts)" in forecast_table(s).columns
    assert money(15.2, s) == "15.20"


def test_telegram_price_shows_cad_equivalent():
    assert price_text("XAUUSD", 2650.4, 1.38) == "2,650.40 USD (≈ C$3,657.55)"
    assert price_text("XAUUSD", 2650.4, None) == "2,650.40 USD (no USD/CAD rate cached)"
    assert price_text("XIU.TO", 38.2, 1.38) == "C$38.20"
    assert price_text("^VIX", 18.0, 1.38) == "18.00 pts"
    a = Alert.model_validate({"ticker": "XAUUSD", "price": "2650.40", "strategy": "s", "side": "long", "secret": "x"})
    msg = format_message(a, pd.Timestamp("2026-11-09 14:00", tz="UTC"), "TAKE", [], [], usd_cad=1.38)
    assert "C$3,657.55" in msg


def test_vwap_cad_view_converts_each_trade_at_exit_rate():
    trades = pd.DataFrame({"pnl": [100.0, -50.0],
                           "exit_time": pd.to_datetime(["2024-01-02 15:00", "2024-01-04 15:00"], utc=True)})
    eq = pd.Series([10_000.0, 10_100.0, 10_050.0],
                   index=pd.to_datetime(["2024-01-02 15:00", "2024-01-03 15:00", "2024-01-04 15:00"], utc=True))
    c = cad_view(trades, eq, rates_series((1.30, 1.35, 1.40)), 10_000)
    assert list(c["trades"]["pnl_cad"]) == pytest.approx([130.0, -70.0])
    assert c["net_profit_cad"] == pytest.approx(60.0)
    assert c["start_equity_cad"] == pytest.approx(13_000) and c["end_equity_cad"] == pytest.approx(14_070)
    assert c["unconverted_trades"] == 0
