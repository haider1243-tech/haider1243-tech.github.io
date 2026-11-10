import httpx
import numpy as np
import pandas as pd
import pytest

from forecaster import intraday as I


class FakeSession:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def get(self, url, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.handler(url, params or {}, headers or {})

    def close(self):
        pass


def resp(status, js=None):
    return httpx.Response(status, json=js or {}, request=httpx.Request("GET", "https://x.test"))


def test_oanda_m1_paginates_mid_and_spread():
    start = pd.Timestamp("2024-03-04 00:00", tz="UTC")

    def handler(url, params, headers):
        t0 = pd.Timestamp(params["from"])
        cs = []
        for k in range(3):
            t = t0 + pd.Timedelta(minutes=k)
            if t >= start + pd.Timedelta(minutes=7):
                break
            cs.append({"time": t.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"), "complete": True, "volume": 12,
                       "mid": {"o": "2000", "h": "2001", "l": "1999", "c": "2000.5"},
                       "bid": {"o": "1999.8", "h": "2000.8", "l": "1998.8", "c": "2000.3"},
                       "ask": {"o": "2000.2", "h": "2001.2", "l": "1999.2", "c": "2000.7"}})
        return resp(200, {"candles": cs})

    s = FakeSession(handler)
    df = I.oanda_m1("XAU_USD", start, start + pd.Timedelta(minutes=10), "tok", session=s, count=3)
    assert len(df) == 7 and df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert df["Close"].iloc[0] == 2000.5 and df["Spread"].iloc[0] == pytest.approx(0.4) and df["Volume"].iloc[0] == 12
    assert s.calls[0][1]["price"] == "MBA" and s.calls[0][2]["Authorization"] == "Bearer tok"
    assert "tok" not in s.calls[0][0]
    with pytest.raises(I.DataUnavailable, match="OANDA_API_TOKEN"):
        I.oanda_m1("XAU_USD", start, start + pd.Timedelta(minutes=5), "bad", session=FakeSession(lambda *a: resp(401)))


def test_alpaca_pages_and_falls_back_to_iex():
    start = pd.Timestamp("2024-03-04 14:30", tz="UTC")

    def handler(url, params, headers):
        if params["feed"] == "sip":
            return resp(403, {"message": "subscription does not permit querying recent SIP data"})
        page = params.get("page_token")
        t = start + pd.Timedelta(minutes=0 if page is None else 2)
        bars = [{"t": (t + pd.Timedelta(minutes=k)).isoformat(), "o": 500, "h": 501, "l": 499, "c": 500.5, "v": 1000}
                for k in range(2)]
        return resp(200, {"bars": bars, "next_page_token": None if page else "p2"})

    s = FakeSession(handler)
    df = I.alpaca_m1("SPY", start, start + pd.Timedelta(hours=1), "kid", "sec", session=s)
    assert len(df) == 4 and [c[1]["feed"] for c in s.calls] == ["sip", "iex", "iex"]
    assert s.calls[0][2] == {"APCA-API-KEY-ID": "kid", "APCA-API-SECRET-KEY": "sec"}
    assert s.calls[0][1]["adjustment"] == "split" and s.calls[0][1]["timeframe"] == "1Min"


def test_finnhub_free_tier_403_is_explained():
    with pytest.raises(I.DataUnavailable, match="paid plan"):
        I.finnhub_m1("SPY", pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC"), "k",
                     session=FakeSession(lambda *a: resp(403)))


def test_store_caches_past_months_and_refreshes_current(tmp_path):
    now = pd.Timestamp("2024-03-15 12:00", tz="UTC")
    store = I.IntradayStore(tmp_path, clock=lambda: now)
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        idx = pd.date_range(a, min(b, a + pd.Timedelta(minutes=3)), freq="1min", inclusive="left")
        return pd.DataFrame({c: 1.0 for c in I.COLS}, index=idx)

    start, end = pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-03-15", tz="UTC")
    assert store.fill("oanda", "XAU_USD", start, end, fetch) == 3
    assert calls[0][0] == start and calls[1][0] == pd.Timestamp("2024-02-01", tz="UTC")
    assert store.fill("oanda", "XAU_USD", start, end, fetch) == 0          # all cached (March is fresh)
    assert not store.needs("oanda", "XAU_USD", pd.Period("2024-01", "M"))
    assert len(store.read("oanda", "XAU_USD", pd.Timestamp("2024-02-01", tz="UTC"))) == 6


def test_session_bars_and_anchored_resample():
    spec = I.INSTRUMENTS["SPY"]
    m1 = I.synthetic_m1("SPY", "2024-03-04", "2024-03-05")
    pre = m1.iloc[:5].copy()
    pre.index = pre.index - pd.Timedelta(hours=2)                         # 07:30 pre-market bars
    df = I.session_bars(pd.concat([pre, m1]).sort_index(), spec)
    assert len(df) == len(m1) == 2 * 390
    r5 = I.resample(df, 5, spec)
    assert len(r5) == 2 * 78 and r5.index.tz_convert(I.NY)[0].strftime("%H:%M") == "09:30"
    first = df.iloc[:5]
    assert r5["Open"].iloc[0] == first["Open"].iloc[0] and r5["High"].iloc[0] == first["High"].max()
    assert r5["Volume"].iloc[0] == first["Volume"].sum() and r5["Close"].iloc[0] == first["Close"].iloc[-1]
    h1 = I.resample(df, 60, spec)
    assert h1.index.tz_convert(I.NY)[:2].strftime("%H:%M").tolist() == ["09:30", "10:30"]
    assert len(I.daily_bars(df, spec)) == 2


def test_gold_session_runs_overnight_and_trades_in_the_window():
    spec = I.INSTRUMENTS["XAU_USD"]
    g = I.session_bars(I.synthetic_m1("XAU_USD", "2024-03-04", "2024-03-08"), spec)
    t = g.index.tz_convert(I.NY)
    assert not ((t.hour == 17)).any()                                       # daily break removed
    day = I.trading_day(g.index, spec.vwap_anchor)
    assert day[0] == pd.Timestamp("2024-03-04") and t[0].strftime("%a %H:%M") == "Sun 18:00"
    w = I.window_mask(g.index, spec)
    assert t[w].hour.min() == 8 and (t[w].hour * 60 + t[w].minute).max() == 16 * 60 + 54
    assert "Spread" in g and (g["Spread"] > 0).all()
