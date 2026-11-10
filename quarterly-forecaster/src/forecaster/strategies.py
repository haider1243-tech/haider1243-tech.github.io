"""Day-trading strategy library for the Strategy Lab (SPY, QQQ, XAU_USD).

Every strategy has fixed, testable rules:
  * signals are decided on a bar's CLOSE using only that bar and earlier ones (all features are
    causal; tests check signals don't change when later bars are cut off),
  * the entry is a market order at the NEXT bar's open (lab_engine fills it with costs),
  * stop and target are prices fixed at the signal,
  * time exit: flat by the end of the day-trading window (engine), no entries in its first or
    last minute, at most `max_trades` entries per day.

Output of `signals()`: DataFrame indexed by bar position `i` with columns side (+1 long / -1
short), stop, target. `grid` lists the values the walk-forward optimiser may tune; keep grids
small (overfitting), the lab warns when a strategy tunes many parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecaster.intraday import INSTRUMENTS, NY, InstrumentSpec, _hm, trading_day, window_mask


# --------------------------------------------------------------------------- #
# Features (computed once per bar set; all causal)
# --------------------------------------------------------------------------- #

def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn.notna(), 50.0)


class Features:
    """Indicators shared by all strategies for one instrument + timeframe."""

    def __init__(self, bars: pd.DataFrame, spec: InstrumentSpec, minutes: int) -> None:
        self.bars, self.spec, self.minutes = bars, spec, minutes
        self._cache: dict = {}
        idx = bars.index
        self.n = len(bars)
        self.o, self.h, self.l, self.c = (bars[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        self.v = bars["Volume"].to_numpy(float)
        self.day = np.asarray(trading_day(idx, spec.vwap_anchor))
        t = idx.tz_convert(NY)
        self.mins = np.asarray(t.hour * 60 + t.minute)                # bar start, New York minutes
        self.in_window = window_mask(idx, spec, minutes)
        self.new_day = np.r_[True, self.day[1:] != self.day[:-1]]

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def close(self) -> pd.Series:
        return self.get("close", lambda: pd.Series(self.c, index=range(self.n)))

    # ---- session VWAP (+ volume-weighted sd), reset at the instrument's anchor ----
    def vwap(self) -> tuple[np.ndarray, np.ndarray]:
        def calc():
            src = (self.h + self.l + self.c) / 3
            v = np.maximum(self.v, 0) + 1e-12
            g = pd.Series(self.day)
            sv = pd.Series(v).groupby(g).cumsum().to_numpy()
            spv = pd.Series(src * v).groupby(g).cumsum().to_numpy()
            sp2v = pd.Series(src * src * v).groupby(g).cumsum().to_numpy()
            vw = spv / sv
            sd = np.sqrt(np.maximum(sp2v / sv - vw ** 2, 0.0))
            return vw, sd
        return self.get("vwap", calc)

    def atr(self, n: int = 14) -> np.ndarray:
        def calc():
            pc = np.r_[np.nan, self.c[:-1]]
            tr = np.nanmax(np.vstack([self.h - self.l, np.abs(self.h - pc), np.abs(self.l - pc)]), axis=0)
            return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
        return self.get(("atr", n), calc)

    def ema(self, n: int) -> np.ndarray:
        return self.get(("ema", n), lambda: _ema(self.close, n).to_numpy())

    def rsi(self, n: int) -> np.ndarray:
        return self.get(("rsi", n), lambda: _rsi(self.close, n).to_numpy())

    def bollinger(self, n: int, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def calc():
            m = self.close.rolling(n).mean()
            s = self.close.rolling(n).std(ddof=0)
            return m.to_numpy(), (m + k * s).to_numpy(), (m - k * s).to_numpy()
        return self.get(("bb", n, k), calc)

    def vol_avg(self, n: int = 20) -> np.ndarray:
        return self.get(("vavg", n), lambda: pd.Series(self.v).rolling(n, min_periods=5).mean().shift(1).to_numpy())

    def window_start_idx(self) -> np.ndarray:
        """Per bar: True on the first bar of the day-trading window of its day."""
        def calc():
            w = self.in_window
            prev_w = np.r_[False, w[:-1]] & ~self.new_day
            return w & ~prev_w
        return self.get("wstart", calc)

    def opening_range(self, minutes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """High/low of the first `minutes` of the window; `ready` True on bars after it completed."""
        def calc():
            start = _hm(self.spec.session_start)
            rel = self.mins - start
            in_rng = self.in_window & (rel >= 0) & (rel + self.minutes <= minutes)
            g = pd.Series(self.day)
            hi = pd.Series(np.where(in_rng, self.h, -np.inf)).groupby(g).cummax().to_numpy()
            lo = pd.Series(np.where(in_rng, self.l, np.inf)).groupby(g).cummin().to_numpy()
            ready = self.in_window & (rel >= minutes) & np.isfinite(hi) & np.isfinite(lo)
            return hi, lo, ready
        return self.get(("orb", minutes), calc)

    def prev_day(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Previous trading day's high, low, close (window bars only), mapped onto each bar."""
        def calc():
            w = self.in_window
            df = pd.DataFrame({"day": self.day, "h": np.where(w, self.h, np.nan), "l": np.where(w, self.l, np.nan),
                               "c": np.where(w, self.c, np.nan)})
            d = df.groupby("day").agg(h=("h", "max"), l=("l", "min"), c=("c", lambda x: x.dropna().iloc[-1]
                                                                      if x.notna().any() else np.nan))
            d = d.shift(1)
            m = d.reindex(self.day)
            return m["h"].to_numpy(), m["l"].to_numpy(), m["c"].to_numpy()
        return self.get("prevday", calc)

    def day_open(self) -> np.ndarray:
        """Open of the first window bar of the day, known from that bar onward (NaN before)."""
        def calc():
            ws = self.window_start_idx()
            s = pd.Series(np.where(ws, self.o, np.nan)).groupby(pd.Series(self.day)).ffill()
            return s.to_numpy()
        return self.get("dayopen", calc)


# --------------------------------------------------------------------------- #
# Strategy base
# --------------------------------------------------------------------------- #

@dataclass
class Strategy:
    key: str
    name: str
    description: str
    params: dict
    grid: dict = field(default_factory=dict)
    min_minutes: int = 1                         # smallest timeframe it makes sense on
    max_minutes: int = 60

    def with_params(self, **kw) -> Strategy:
        return Strategy(self.key, self.name, self.description, {**self.params, **kw}, self.grid,
                        self.min_minutes, self.max_minutes)

    def signals(self, f: Features) -> pd.DataFrame:
        return SIGNALS[self.key](f, self.params)

    def tuned(self) -> list[str]:
        return [k for k, v in self.grid.items() if len(v) > 1]


def _out(f: Features, long: np.ndarray, short: np.ndarray, stop_l, tgt_l, stop_s, tgt_s, sides: str) -> pd.DataFrame:
    long = np.asarray(long, bool) & f.in_window & (sides in ("both", "long"))
    short = np.asarray(short, bool) & f.in_window & (sides in ("both", "short"))
    short &= ~long
    i = np.flatnonzero(long | short)
    side = np.where(long[i], 1, -1)
    stop = np.where(side > 0, np.asarray(stop_l, float)[i] if np.ndim(stop_l) else stop_l,
                    np.asarray(stop_s, float)[i] if np.ndim(stop_s) else stop_s)
    tgt = np.where(side > 0, np.asarray(tgt_l, float)[i] if np.ndim(tgt_l) else tgt_l,
                   np.asarray(tgt_s, float)[i] if np.ndim(tgt_s) else tgt_s)
    ok = np.isfinite(stop) & np.isfinite(tgt) & (np.where(side > 0, stop < f.c[i], stop > f.c[i]))
    return pd.DataFrame({"side": side[ok], "stop": stop[ok], "target": tgt[ok]}, index=pd.Index(i[ok], name="i"))


def _cross_up(a: np.ndarray, b) -> np.ndarray:
    b = np.broadcast_to(b, a.shape)
    return np.r_[False, (a[1:] > b[1:]) & (a[:-1] <= b[:-1])]


def _cross_dn(a: np.ndarray, b) -> np.ndarray:
    b = np.broadcast_to(b, a.shape)
    return np.r_[False, (a[1:] < b[1:]) & (a[:-1] >= b[:-1])]


def _same_day_prev(f: Features, x: np.ndarray) -> np.ndarray:
    """x shifted one bar, NaN across a day boundary."""
    p = np.r_[np.nan, x[:-1]].astype(float)
    p[f.new_day] = np.nan
    return p


def _rr_targets(entry_ref: np.ndarray, stop_l: np.ndarray, stop_s: np.ndarray, rr: float):
    return entry_ref + rr * (entry_ref - stop_l), entry_ref - rr * (stop_s - entry_ref)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def vwap_reversion(f: Features, p: dict) -> pd.DataFrame:
    """Price stretched >= k sd from session VWAP, then a bar closes back toward it. Target: VWAP
    at the signal. Stop: atr_mult x ATR beyond the signal bar's extreme."""
    vw, sd = f.vwap()
    atr = f.atr()
    pc = _same_day_prev(f, f.c)
    k = p["k"]
    long = (pc < vw - k * sd) & (f.c > pc) & (f.c < vw) & (sd > 0)
    short = (pc > vw + k * sd) & (f.c < pc) & (f.c > vw) & (sd > 0)
    return _out(f, long, short, f.l - p["atr_mult"] * atr, vw, f.h + p["atr_mult"] * atr, vw, p["sides"])


def vwap_breakout(f: Features, p: dict) -> pd.DataFrame:
    """Close crosses beyond VWAP + k sd with volume > vol_mult x its 20-bar average. Stop: VWAP.
    Target: rr x risk."""
    vw, sd = f.vwap()
    va = f.vol_avg()
    hv = f.v > p["vol_mult"] * va
    up, dn = vw + p["k"] * sd, vw - p["k"] * sd
    long = _cross_up(f.c, up) & hv & (sd > 0)
    short = _cross_dn(f.c, dn) & hv & (sd > 0)
    tl, ts = _rr_targets(f.c, vw, vw, p["rr"])
    return _out(f, long, short, vw, tl, vw, ts, p["sides"])


def opening_range(f: Features, p: dict) -> pd.DataFrame:
    """First close beyond the opening range (first N minutes). Stop: the other side of the range
    (or its middle). Target: rr x risk. One trade a day."""
    hi, lo, ready = f.opening_range(p["minutes"])
    with np.errstate(invalid="ignore"):                    # range is +-inf before it forms
        mid = (hi + lo) / 2
    stop_l = mid if p["stop"] == "mid" else lo
    stop_s = mid if p["stop"] == "mid" else hi
    pc = _same_day_prev(f, f.c)
    long = ready & (f.c > hi) & ~(pc > hi)
    short = ready & (f.c < lo) & ~(pc < lo)
    tl, ts = _rr_targets(f.c, stop_l, stop_s, p["rr"])
    return _out(f, long, short, stop_l, tl, stop_s, ts, p["sides"])


def ema_cross(f: Features, p: dict) -> pd.DataFrame:
    """EMA fast crosses EMA slow. Stop: atr_mult x ATR. Target: rr x risk."""
    a, b = f.ema(p["fast"]), f.ema(p["slow"])
    atr = f.atr()
    long, short = _cross_up(a, b), _cross_dn(a, b)
    sl, ss = f.c - p["atr_mult"] * atr, f.c + p["atr_mult"] * atr
    tl, ts = _rr_targets(f.c, sl, ss, p["rr"])
    return _out(f, long, short, sl, tl, ss, ts, p["sides"])


def rsi2_reversion(f: Features, p: dict) -> pd.DataFrame:
    """RSI(2) below `low` in an uptrend (close above EMA trend) -> long; above 100-low in a
    downtrend -> short. Stop/target: ATR multiples."""
    r = f.rsi(2)
    tr = f.ema(p["trend"])
    atr = f.atr()
    long = (r < p["low"]) & (f.c > tr)
    short = (r > 100 - p["low"]) & (f.c < tr)
    return _out(f, long, short, f.c - p["stop_atr"] * atr, f.c + p["target_atr"] * atr,
                f.c + p["stop_atr"] * atr, f.c - p["target_atr"] * atr, p["sides"])


def bollinger_squeeze(f: Features, p: dict) -> pd.DataFrame:
    """Band width in its lowest `pct` over `lookback` bars (squeeze), then a close outside the
    band. Stop: middle band. Target: rr x risk."""
    m, up, lo = f.bollinger(p["n"], 2.0)
    width = pd.Series((up - lo) / m)
    thr = width.rolling(p["lookback"], min_periods=p["lookback"] // 2).quantile(p["pct"]).shift(1).to_numpy()
    squeezed = np.r_[False, (width.to_numpy()[:-1] <= thr[1:])]           # previous bar was in a squeeze
    long = squeezed & (f.c > up)
    short = squeezed & (f.c < lo)
    tl, ts = _rr_targets(f.c, m, m, p["rr"])
    return _out(f, long, short, m, tl, m, ts, p["sides"])


def gap(f: Features, p: dict) -> pd.DataFrame:
    """At the close of the first window bar: gap = day open / previous close - 1.
    go:   |gap| >= min_gap and the first bar closes in the gap's direction -> trade with the gap,
          stop beyond the first bar, target rr x risk.
    fade: |gap| >= min_gap and the first bar closes against it -> trade toward the previous
          close (target), stop atr_mult x ATR beyond the first bar."""
    _, _, pclose = f.prev_day()
    dopen = f.day_open()
    first = f.window_start_idx()
    g = dopen / pclose - 1
    big = np.abs(g) >= p["min_gap"]
    up_bar, dn_bar = f.c > f.o, f.c < f.o
    atr = f.atr()
    if p["mode"] == "go":
        long = first & big & (g > 0) & up_bar
        short = first & big & (g < 0) & dn_bar
        sl, ss = f.l, f.h
        tl, ts = _rr_targets(f.c, sl, ss, p["rr"])
    else:
        long = first & big & (g < 0) & up_bar
        short = first & big & (g > 0) & dn_bar
        sl, ss = f.l - p["atr_mult"] * atr, f.h + p["atr_mult"] * atr
        tl = ts = pclose
    return _out(f, long, short, sl, tl, ss, ts, p["sides"])


def prev_day_break(f: Features, p: dict) -> pd.DataFrame:
    """First close above the previous day's high (long) / below its low (short). Stop:
    atr_mult x ATR. Target: rr x risk."""
    ph, pl, _ = f.prev_day()
    pc = _same_day_prev(f, f.c)
    long = (f.c > ph) & ~(pc > ph)
    short = (f.c < pl) & ~(pc < pl)
    atr = f.atr()
    sl, ss = f.c - p["atr_mult"] * atr, f.c + p["atr_mult"] * atr
    tl, ts = _rr_targets(f.c, sl, ss, p["rr"])
    return _out(f, long, short, sl, tl, ss, ts, p["sides"])


SIGNALS = {"vwap_reversion": vwap_reversion, "vwap_breakout": vwap_breakout, "orb5": opening_range, "orb15": opening_range, "orb30": opening_range,
           "ema_cross": ema_cross, "rsi2": rsi2_reversion, "bb_squeeze": bollinger_squeeze,
           "gap_go": gap, "gap_fade": gap, "pdhl_break": prev_day_break}

LIBRARY: dict[str, Strategy] = {s.key: s for s in [
    Strategy("vwap_reversion", "VWAP mean reversion", vwap_reversion.__doc__,
             {"k": 2.0, "atr_mult": 0.5, "sides": "both", "max_trades": 2},
             {"k": [1.5, 2.0, 2.5], "atr_mult": [0.5, 1.0]}),
    Strategy("vwap_breakout", "VWAP breakout", vwap_breakout.__doc__,
             {"k": 1.0, "vol_mult": 1.5, "rr": 2.0, "sides": "both", "max_trades": 2},
             {"k": [0.5, 1.0, 1.5], "rr": [1.5, 2.0]}),
    Strategy("orb5", "Opening range breakout 5m", opening_range.__doc__,
             {"minutes": 5, "stop": "range", "rr": 2.0, "sides": "both", "max_trades": 1},
             {"stop": ["range", "mid"], "rr": [1.0, 2.0]}, min_minutes=1, max_minutes=5),
    Strategy("orb15", "Opening range breakout 15m", opening_range.__doc__,
             {"minutes": 15, "stop": "range", "rr": 2.0, "sides": "both", "max_trades": 1},
             {"stop": ["range", "mid"], "rr": [1.0, 2.0]}, min_minutes=1, max_minutes=15),
    Strategy("orb30", "Opening range breakout 30m", opening_range.__doc__,
             {"minutes": 30, "stop": "range", "rr": 2.0, "sides": "both", "max_trades": 1},
             {"stop": ["range", "mid"], "rr": [1.0, 2.0]}, min_minutes=1, max_minutes=30),
    Strategy("ema_cross", "EMA 9/21 crossover", ema_cross.__doc__,
             {"fast": 9, "slow": 21, "atr_mult": 1.5, "rr": 2.0, "sides": "both", "max_trades": 3},
             {"atr_mult": [1.0, 1.5, 2.0], "rr": [1.5, 2.0]}),
    Strategy("rsi2", "RSI(2) reversion", rsi2_reversion.__doc__,
             {"low": 10, "trend": 50, "stop_atr": 1.5, "target_atr": 1.0, "sides": "both", "max_trades": 3},
             {"low": [5, 10], "stop_atr": [1.0, 1.5], "target_atr": [0.75, 1.0]}),
    Strategy("bb_squeeze", "Bollinger squeeze breakout", bollinger_squeeze.__doc__,
             {"n": 20, "lookback": 100, "pct": 0.2, "rr": 2.0, "sides": "both", "max_trades": 2},
             {"pct": [0.1, 0.2], "rr": [1.5, 2.0]}),
    Strategy("gap_go", "Gap and go", gap.__doc__,
             {"mode": "go", "min_gap": 0.003, "rr": 2.0, "atr_mult": 0.5, "sides": "both", "max_trades": 1},
             {"min_gap": [0.002, 0.004, 0.006], "rr": [1.0, 2.0]}),
    Strategy("gap_fade", "Gap fade", gap.__doc__,
             {"mode": "fade", "min_gap": 0.003, "rr": 2.0, "atr_mult": 0.5, "sides": "both", "max_trades": 1},
             {"min_gap": [0.002, 0.004, 0.006], "atr_mult": [0.5, 1.0]}),
    Strategy("pdhl_break", "Previous-day high/low break", prev_day_break.__doc__,
             {"atr_mult": 1.5, "rr": 2.0, "sides": "both", "max_trades": 2},
             {"atr_mult": [1.0, 1.5, 2.0], "rr": [1.5, 2.0]}),
]}


def applicable(s: Strategy, minutes: int) -> bool:
    return s.min_minutes <= minutes <= s.max_minutes


def features_for(bars: pd.DataFrame, symbol: str, minutes: int) -> Features:
    return Features(bars, INSTRUMENTS[symbol], minutes)
