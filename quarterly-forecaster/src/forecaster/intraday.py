"""Historical intraday bars for the Strategy Lab: download, cache, resample.

    python -m forecaster.lab download --years 3      # (see lab.py) fills the cache below

Sources (free first)
  XAU_USD, FX   OANDA v20 practice REST /v3/instruments/{i}/candles, price=MBA (mid + bid + ask),
                M1 in pages of 5000 -> mid OHLC, tick volume and the real bid/ask spread per bar.
  SPY, QQQ      Alpaca market data v2 /stocks/{s}/bars, 1Min, split-adjusted. A free paper-only
                Alpaca account (email sign-up, open to Canadians) gives keys; historical SIP bars
                older than 15 minutes are allowed on the free plan, otherwise we fall back to the
                IEX feed. Regular hours only are kept (09:30-16:00 New York).
  (fallback)    Finnhub /stock/candle: on the free plan it now answers 403 (premium); we try it
                and say so instead of failing silently.
  daily         yfinance via market_data.ParquetStore (VIX bands for the lab's slices).

Cache: data/intraday/<source>/<symbol>/M1/<YYYY-MM>.parquet. Complete past months are never
re-downloaded; the current month is refreshed when older than an hour. Only 1-minute bars are
downloaded; 5m/15m/1h/daily are resampled locally, anchored to the session (09:30 for stocks),
so a 5m bar never mixes two sessions.

Bars: UTC DatetimeIndex = bar start; columns Open High Low Close Volume (+ Spread in price units
for OANDA: ask close - bid close). Stock bars have no spread column; the cost model supplies one.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("intraday")
NY = "America/New_York"
COLS = ["Open", "High", "Low", "Close", "Volume"]
TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
OANDA_REST = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}
ALPACA_DATA = "https://data.alpaca.markets"
FINNHUB_REST = "https://finnhub.io/api/v1"


class DataUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Instruments and sessions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    kind: str                 # "stock" | "fx"
    source: str               # "alpaca" | "oanda"
    session_start: str        # day-trading window (New York time)
    session_end: str
    vwap_anchor: str          # VWAP / trading-day reset (New York time)
    tick: float


INSTRUMENTS: dict[str, InstrumentSpec] = {
    "SPY": InstrumentSpec("SPY", "stock", "alpaca", "09:30", "16:00", "09:30", 0.01),
    "QQQ": InstrumentSpec("QQQ", "stock", "alpaca", "09:30", "16:00", "09:30", 0.01),
    # gold: day-trade the London/New York overlap through the COMEX session, flat before the
    # 17:00 daily break; VWAP resets 18:00 like the live tab, the Pine strategy and the monitor
    "XAU_USD": InstrumentSpec("XAU_USD", "fx", "oanda", "08:00", "16:55", "18:00", 0.01),
    "USD_CAD": InstrumentSpec("USD_CAD", "fx", "oanda", "08:00", "16:55", "18:00", 0.00001),
    **{s: InstrumentSpec(s, "stock", "alpaca", "09:30", "16:00", "09:30", 0.01) for s in ("NVDA", "AAPL", "MSFT", "TSLA")},
}


def spec_for(symbol: str) -> InstrumentSpec | None:
    """Known instrument, or a US stock spec for a plain ticker (Alpaca has US stocks only;
    TSX symbols like XIU.TO and indices have no intraday source, so None)."""
    symbol = symbol.upper()
    if symbol in INSTRUMENTS:
        return INSTRUMENTS[symbol]
    if symbol.isalpha() and len(symbol) <= 5:
        return InstrumentSpec(symbol, "stock", "alpaca", "09:30", "16:00", "09:30", 0.01)
    return None


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def trading_day(index: pd.DatetimeIndex, anchor: str) -> pd.Index:
    """Trading-day label per bar: the New York date of the session the bar belongs to. With an
    evening anchor (18:00) the session that starts Sunday evening is labelled Monday."""
    t = index.tz_convert(NY)
    a = _hm(anchor)
    if a >= 12 * 60:
        t = t + pd.Timedelta(minutes=24 * 60 - a)
    return t.normalize().tz_localize(None)


def session_bars(df: pd.DataFrame, spec: InstrumentSpec) -> pd.DataFrame:
    """Bars that belong to a trading session. Stocks: regular hours 09:30-16:00. FX/gold: the
    whole 23-hour session (indicators like the 18:00 VWAP need the overnight part), minus the
    17:00-18:00 break and the weekend. Trading itself is limited to the day-trading window by
    the engine (`window_mask`)."""
    if df.empty:
        return df
    t = df.index.tz_convert(NY)
    m = t.hour * 60 + t.minute
    if spec.kind == "stock":
        keep = (m >= _hm(spec.session_start)) & (m < _hm(spec.session_end)) & (t.dayofweek < 5)
    else:
        dow = t.dayofweek
        keep = ~((m >= 17 * 60) & (m < 18 * 60)) & ~(dow == 5) & ~((dow == 4) & (m >= 17 * 60)) \
            & ~((dow == 6) & (m < 18 * 60))
    return df[keep]


regular_hours = session_bars


def window_mask(index: pd.DatetimeIndex, spec: InstrumentSpec, minutes: int = 1) -> np.ndarray:
    """True for bars that lie entirely inside the day-trading window."""
    t = index.tz_convert(NY)
    m = t.hour * 60 + t.minute
    return np.asarray((m >= _hm(spec.session_start)) & (m + minutes <= _hm(spec.session_end)) & (t.dayofweek < 5))


def resample(df: pd.DataFrame, minutes: int, spec: InstrumentSpec) -> pd.DataFrame:
    """1m -> N-minute bars, anchored at the window start in New York time (09:30 for stocks),
    never spanning two sessions. Spread is averaged; volume summed."""
    if minutes == 1 or df.empty:
        return df
    t = df.index.tz_convert(NY)
    start = _hm(spec.vwap_anchor)
    mins = t.hour * 60 + t.minute
    bucket = (mins - start) // minutes * minutes + start
    key = t.normalize() + pd.to_timedelta(bucket, unit="m")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if "Spread" in df:
        agg["Spread"] = "mean"
    out = df.groupby(key.tz_convert("UTC")).agg(agg)
    out.index.name = None
    return out


def daily_bars(df: pd.DataFrame, spec: InstrumentSpec) -> pd.DataFrame:
    if df.empty:
        return df
    day = trading_day(df.index, spec.vwap_anchor)
    out = df.groupby(day).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    out.index.name = None
    return out


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

class IntradayStore:
    def __init__(self, root: Path, clock: Callable[[], pd.Timestamp] | None = None) -> None:
        self.root = Path(root)
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))

    def _path(self, source: str, symbol: str, month: pd.Period) -> Path:
        return self.root / source / symbol / "M1" / f"{month}.parquet"

    def months(self, source: str, symbol: str) -> list[pd.Period]:
        d = self.root / source / symbol / "M1"
        return sorted(pd.Period(p.stem, "M") for p in d.glob("*.parquet")) if d.exists() else []

    def needs(self, source: str, symbol: str, month: pd.Period) -> bool:
        p = self._path(source, symbol, month)
        if not p.exists():
            return True
        now = self.clock()
        if month < pd.Period(now.tz_convert("UTC").tz_localize(None), "M"):
            return False                                         # complete past month
        return time.time() - p.stat().st_mtime > 3600

    def write(self, source: str, symbol: str, month: pd.Period, df: pd.DataFrame) -> None:
        p = self._path(source, symbol, month)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)

    def read(self, source: str, symbol: str, start: pd.Timestamp | None = None,
             end: pd.Timestamp | None = None) -> pd.DataFrame:
        parts = []
        for m in self.months(source, symbol):
            if start is not None and m.end_time.tz_localize("UTC") < start:
                continue
            if end is not None and m.start_time.tz_localize("UTC") > end:
                continue
            parts.append(pd.read_parquet(self._path(source, symbol, m)))
        if not parts:
            return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC"))
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index < end]
        return df

    def fill(self, source: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
             fetch_month: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
             on_progress: Callable[[str], None] = lambda m: None) -> int:
        """Download every month in [start, end) that isn't cached yet. Returns months fetched."""
        n = 0
        for month in pd.period_range(start.tz_convert("UTC").tz_localize(None), end.tz_convert("UTC").tz_localize(None), freq="M"):
            if not self.needs(source, symbol, month):
                continue
            a = max(month.start_time.tz_localize("UTC"), start)
            b = min((month + 1).start_time.tz_localize("UTC"), end)
            df = fetch_month(a, b)
            self.write(source, symbol, month, df)
            on_progress(f"{symbol} {month}: {len(df):,} bars")
            n += 1
        return n


def load_bars(store: IntradayStore, symbol: str, timeframe: str, start: pd.Timestamp | None = None,
              end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Cached 1m bars -> the instrument's trading window -> timeframe."""
    spec = spec_for(symbol)
    df = session_bars(store.read(spec.source, symbol, start, end), spec)
    if timeframe == "1d":
        return daily_bars(df, spec)
    return resample(df, TIMEFRAMES[timeframe], spec)


# --------------------------------------------------------------------------- #
# Downloaders (network; tests pass fake sessions)
# --------------------------------------------------------------------------- #

def _iso(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def oanda_m1(instrument: str, start: pd.Timestamp, end: pd.Timestamp, token: str,
             host: str = OANDA_REST["practice"], session=None, count: int = 5000) -> pd.DataFrame:
    """M1 mid OHLC + tick volume + spread (ask close - bid close) for [start, end)."""
    import httpx

    own = session is None
    session = session or httpx.Client(timeout=30)
    rows, cur = [], start
    try:
        while cur < end:
            r = session.get(f"{host}/v3/instruments/{instrument}/candles",
                            params={"price": "MBA", "granularity": "M1", "from": _iso(cur), "count": count},
                            headers={"Authorization": f"Bearer {token}"})
            if r.status_code in (401, 403):
                raise DataUnavailable(f"OANDA {r.status_code}: check OANDA_API_TOKEN")
            r.raise_for_status()
            candles = r.json().get("candles", [])
            if not candles:
                break
            last = None
            for c in candles:
                t = pd.Timestamp(c["time"]).tz_convert("UTC") if pd.Timestamp(c["time"]).tzinfo else \
                    pd.Timestamp(c["time"], tz="UTC")
                last = t
                if t >= end or not c.get("complete", True):
                    continue
                m, b, a = c["mid"], c["bid"], c["ask"]
                rows.append((t, float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"]), float(c["volume"]),
                             float(a["c"]) - float(b["c"])))
            if last is None or last <= cur:
                break
            cur = last + pd.Timedelta(minutes=1)
    finally:
        if own:
            session.close()
    return _frame(rows, spread=True)


def alpaca_m1(symbol: str, start: pd.Timestamp, end: pd.Timestamp, key_id: str, secret: str,
              session=None, feed: str = "sip") -> pd.DataFrame:
    """1-minute bars (split-adjusted). Free plan: SIP history except the last 15 minutes; falls
    back to the IEX feed if SIP is refused."""
    import httpx

    own = session is None
    session = session or httpx.Client(timeout=30)
    end = min(end, pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16))
    rows, token = [], None
    try:
        while True:
            params = {"timeframe": "1Min", "start": _iso(start), "end": _iso(end), "limit": 10000,
                      "adjustment": "split", "feed": feed}
            if token:
                params["page_token"] = token
            r = session.get(f"{ALPACA_DATA}/v2/stocks/{symbol}/bars", params=params,
                            headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret})
            if r.status_code == 403 and feed == "sip":
                log.info("Alpaca refused SIP for %s; using the IEX feed", symbol)
                return alpaca_m1(symbol, start, end, key_id, secret, session=session, feed="iex")
            if r.status_code in (401, 403):
                raise DataUnavailable(f"Alpaca {r.status_code}: check ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY")
            r.raise_for_status()
            js = r.json()
            for b in js.get("bars") or []:
                rows.append((pd.Timestamp(b["t"]), float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]),
                             float(b["v"])))
            token = js.get("next_page_token")
            if not token:
                break
    finally:
        if own:
            session.close()
    return _frame(rows)


def finnhub_m1(symbol: str, start: pd.Timestamp, end: pd.Timestamp, key: str, session=None) -> pd.DataFrame:
    import httpx

    own = session is None
    session = session or httpx.Client(timeout=30)
    try:
        r = session.get(f"{FINNHUB_REST}/stock/candle", params={"symbol": symbol, "resolution": "1",
                        "from": int(start.timestamp()), "to": int(end.timestamp())},
                        headers={"X-Finnhub-Token": key})
        if r.status_code in (401, 403):
            raise DataUnavailable("Finnhub candles need a paid plan (free tier answers 403); use Alpaca")
        r.raise_for_status()
        js = r.json()
        if js.get("s") != "ok":
            return _frame([])
        rows = [(pd.Timestamp(t, unit="s", tz="UTC"), o, h, lo, c, v)
                for t, o, h, lo, c, v in zip(js["t"], js["o"], js["h"], js["l"], js["c"], js["v"])]
        return _frame(rows)
    finally:
        if own:
            session.close()


def _frame(rows: list[tuple], spread: bool = False) -> pd.DataFrame:
    cols = COLS + (["Spread"] if spread else [])
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    df = pd.DataFrame(rows, columns=["t", *cols]).set_index("t")
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    df.index.name = None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype(float)


# --------------------------------------------------------------------------- #
# Synthetic bars (tests + offline demo). A random walk: no strategy should find an edge here.
# --------------------------------------------------------------------------- #

def synthetic_m1(symbol: str, start: str, end: str, seed: int = 0, price: float | None = None,
                 drift_bps_per_day: float = 0.0, edge: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> pd.DataFrame:
    """Random-walk 1-minute bars over the instrument's trading window with a U-shaped intraday
    volatility/volume profile, overnight gaps and (for fx) a spread column. `edge` can plant a
    pattern for tests."""
    spec = spec_for(symbol)
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    if spec.kind == "fx":
        s0, n_min = _hm(spec.vwap_anchor) - 24 * 60, 23 * 60          # 18:00 the evening before -> 17:00
    else:
        s0, n_min = _hm(spec.session_start), _hm(spec.session_end) - _hm(spec.session_start)
    price = price or {"SPY": 450.0, "QQQ": 380.0, "XAU_USD": 1900.0, "USD_CAD": 1.36}.get(symbol, 200.0)
    daily_vol = {"SPY": 0.011, "QQQ": 0.014, "XAU_USD": 0.009, "USD_CAD": 0.004, "NVDA": 0.028,
                 "TSLA": 0.035, "AAPL": 0.016, "MSFT": 0.015}.get(symbol, 0.02)
    u = np.linspace(-1, 1, n_min)
    shape = 0.6 + 1.2 * u ** 2                                # U-shaped intraday activity
    shape /= np.sqrt((shape ** 2).mean())
    frames = []
    p = price
    for d in days:
        regime = rng.choice([0.7, 1.0, 1.6], p=[0.3, 0.5, 0.2])
        p *= np.exp(rng.normal(0, daily_vol * 0.35 * regime))          # overnight gap
        sig = daily_vol * regime / np.sqrt(n_min) * shape
        rets = rng.standard_t(5, n_min) * sig * np.sqrt(3 / 5) + drift_bps_per_day / 1e4 / n_min
        close = p * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[p], close[:-1]])
        wig = np.abs(rng.normal(0, sig * 0.6, (2, n_min))) * close
        high = np.maximum(open_, close) + wig[0]
        low = np.minimum(open_, close) - wig[1]
        vol = np.round(rng.gamma(2.0, 1.0, n_min) * shape ** 2 * (5000 if spec.kind == "stock" else 40)) + 1
        idx = (pd.Timestamp(d).tz_localize(NY) + pd.to_timedelta(s0 + np.arange(n_min), unit="m")).tz_convert("UTC")
        f = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
        if spec.kind == "fx":
            base = 0.30 if symbol == "XAU_USD" else close.mean() * 0.00012
            f["Spread"] = np.abs(rng.normal(base, base * 0.2, n_min)) + base * 0.4
        frames.append(f)
        p = close[-1]
    df = pd.concat(frames)
    df.index = df.index.as_unit("ns")
    return edge(df) if edge else df


def ny_window_days(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.tz_convert(NY).normalize().tz_localize(None))


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Period]:
    return pd.period_range(start.tz_localize(None), end.tz_localize(None), freq="M")
