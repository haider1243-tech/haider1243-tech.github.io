"""Python port of pine/vwap_gold.pine, for checking against TradingView's Strategy Tester.

The logic here mirrors the Pine script bar for bar, including TradingView's
broker-emulator conventions. Anything that differs is listed under "Known
differences".

Session VWAP
    Resets at 18:00 New York (CME/OTC gold day). Session key = New York date of
    (bar open + 6 h), the same formula as the Pine script. VWAP and the standard
    deviation use hlc3 weighted by volume: sd = sqrt(sum(v*p^2)/sum(v) - vwap^2).
Signals, evaluated on bar close
    long reclaim     close[1] < vwap[1] and close > vwap
    long rejection   close[1] > vwap[1] and low <= vwap and close > vwap   (test from above, holds)
    short loss       close[1] > vwap[1] and close < vwap
    short rejection  close[1] < vwap[1] and high >= vwap and close < vwap
    plus: not beyond the 2-sigma band (no chasing), volume > SMA(volume, n) * mult,
    DXY/US10Y filter, bar open inside the entry window (New York), not in the
    flatten window, flat, and not inside an event blackout at the bar's close time.
Orders (TradingView defaults: process_orders_on_close = false)
    Market entry fills at the next bar's open, +/- slippage. Stop = signal close
    -/+ ATR*mult; TP = signal close +/- stop distance * R. Both are set from the
    signal bar, as strategy.exit does here. Quantity = equity * risk% /
    (stop distance * point value), capped by max leverage, floored to the quantity
    step. Exits can trigger on the fill bar.
Intrabar path (broker emulator without Bar Magnifier)
    If the high is closer to the open than the low: open -> high -> low -> close,
    otherwise open -> low -> high -> close. A gap through the stop or target
    fills at the open. Slippage applies to market and stop fills, not limits.
End of day
    Positions close at the close of the bar whose close time (NY) is in
    [flatten time, 18:00), like strategy.close_all(immediately = true).

Known differences from TradingView
    * Data: use TradingView's own export (chart -> Export chart data) of the same
      symbol and timeframe. Spot XAUUSD "volume" is broker tick volume, so VWAP
      changes from one data vendor to another.
    * ta.ema seeding: TradingView documents ema as seeded by the first value and
      rma (ATR) by an SMA; both are mirrored. Seeding differences fade within a few
      hundred bars.
    * request.economic() release timing can't be reproduced offline. Enable it in
      Pine and export its "Econ release" plot if you want that block here too.
    * The CSV layout of TradingView's "List of trades" export isn't verified here;
      `compare_with_tradingview` matches columns by keyword.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NY = "America/New_York"
SESSION_SHIFT = pd.Timedelta(hours=6)  # 18:00 NY + 6 h = next calendar day

# FOMC decision times (statement at 14:00 New York). Canonical list: the Pine
# script's default event input must match it (tests/test_vwap_backtest.py checks).
FOMC_DECISIONS_NY: tuple[str, ...] = (
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
)
FOMC_TIME_NY = "14:00"
RELEASE_TIME_NY = "08:30"  # CPI / PPI


@dataclass(frozen=True)
class VwapParams:
    # signals
    use_reclaim: bool = True
    use_rejection: bool = True
    allow_long: bool = True
    allow_short: bool = True
    band1: float = 1.0
    band2: float = 2.0
    use_volume: bool = True
    vol_len: int = 20
    vol_mult: float = 1.2
    entry_window: tuple[str, str] = ("03:00", "16:00")   # New York, bar open time
    flatten_at: str = "16:55"                           # New York, bar close time
    # risk
    risk_pct: float = 1.0
    atr_len: int = 14
    atr_mult: float = 1.5
    tp_r: float = 2.0
    qty_step: float = 1.0
    max_leverage: float = 20.0
    # macro filter
    use_macro: bool = True
    macro_ema_len: int = 50
    macro_strict: bool = True
    # event blackout
    use_events: bool = True
    block_before_min: int = 30
    block_after_min: int = 60
    use_econ_release: bool = False
    # costs / account (match the Pine strategy() properties)
    initial_capital: float = 10_000.0
    point_value: float = 1.0          # syminfo.pointvalue (OANDA:XAUUSD = 1)
    mintick: float = 0.01             # syminfo.mintick; check the Pine info table
    slippage_ticks: int = 0
    commission_per_order: float = 0.0


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

COLUMN_ALIASES = {
    "dxy_up": ("dxy up",), "us10y_up": ("us10y up",), "econ_release": ("econ release",),
}


def load_tradingview_csv(path: str | Path) -> pd.DataFrame:
    """TradingView 'Export chart data' CSV -> bars with a UTC open-time index.

    `time` may be UNIX seconds or ISO-8601 with an offset (both export options).
    Plot columns named "DXY up", "US10Y up" and "Econ release" are picked up if present.
    """
    raw = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}
    if "time" not in cols:
        raise ValueError(f"{path}: no 'time' column (got {list(raw.columns)})")
    t = raw[cols["time"]]
    idx = (pd.to_datetime(t.astype("int64"), unit="s", utc=True) if pd.api.types.is_numeric_dtype(t)
           else pd.to_datetime(t, utc=True))
    out = pd.DataFrame(index=pd.DatetimeIndex(idx, name="time"))
    for c in ("open", "high", "low", "close"):
        if c not in cols:
            raise ValueError(f"{path}: missing '{c}' column")
        out[c] = pd.to_numeric(raw[cols[c]], errors="coerce").to_numpy()
    out["volume"] = pd.to_numeric(raw[cols["volume"]], errors="coerce").to_numpy() if "volume" in cols else 0.0
    for name, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in cols:
                out[name] = pd.to_numeric(raw[cols[a]], errors="coerce").to_numpy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["open", "high", "low", "close"])


def infer_timeframe(index: pd.DatetimeIndex) -> pd.Timedelta:
    d = pd.Series(index).diff().dropna()
    if d.empty:
        raise ValueError("need at least two bars to infer the timeframe")
    return d.mode().iloc[0]


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #

def parse_events(lines: Iterable[str]) -> pd.DatetimeIndex:
    """'YYYY-MM-DD HH:MM [tag]' in New York time -> sorted UTC timestamps.

    Same tolerant format as the Pine input: commas or newlines, extra spaces ok.
    """
    out = []
    for raw in lines:
        for tok in str(raw).replace("\r", "").replace("\n", ",").split(","):
            t = tok.replace(" ", "").replace("\t", "")
            if len(t) < 15 or t[4] != "-" or t[7] != "-" or t[12] != ":":
                continue
            try:
                ts = pd.Timestamp(year=int(t[0:4]), month=int(t[5:7]), day=int(t[8:10]),
                                  hour=int(t[10:12]), minute=int(t[13:15]))
            except ValueError:
                continue
            out.append(ts.tz_localize(NY, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC"))
    return pd.DatetimeIndex(sorted(set(out)), tz="UTC")


def default_event_lines() -> list[str]:
    return [f"{d} {FOMC_TIME_NY} FOMC" for d in FOMC_DECISIONS_NY]


def fred_release_lines(fred, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:  # pragma: no cover
    """CPI and PPI release times (08:30 NY) from FRED's historical release dates."""
    from forecaster.calendar import MACRO_RELEASES, fred_release_events

    rel = tuple(r for r in MACRO_RELEASES if r.category in ("CPI", "PPI"))
    ev = fred_release_events(fred, start, end, rel, limit=1000)
    return [f"{e.date:%Y-%m-%d} {RELEASE_TIME_NY} {e.category}" for e in ev]


def pine_events_text(lines: Iterable[str]) -> str:
    """Event list formatted for the Pine 'Events' text box (one per line)."""
    evs = parse_events(lines)
    return "\n".join(f"{t.tz_convert(NY):%Y-%m-%d %H:%M}" for t in evs)


# --------------------------------------------------------------------------- #
# Indicators (TradingView-equivalent)
# --------------------------------------------------------------------------- #

def tv_ema(x: pd.Series, n: int) -> pd.Series:
    """ta.ema: alpha 2/(n+1), seeded with the first value."""
    return x.ewm(alpha=2 / (n + 1), adjust=False).mean()


def tv_rma(x: pd.Series, n: int) -> pd.Series:
    """ta.rma: seeded with SMA(n), then alpha = 1/n."""
    a = x.to_numpy(float)
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        out[n - 1] = a[:n].mean()
        for i in range(n, len(a)):
            out[i] = out[i - 1] + (a[i] - out[i - 1]) / n
    return pd.Series(out, index=x.index)


def tv_atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tv_rma(tr, n)


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def macro_flags(bars: pd.DataFrame, other: pd.Series | None, ema_len: int) -> pd.Series:
    """1.0 if `other` close > EMA, 0.0 if not, NaN before data; aligned like
    request.security(..., timeframe.period, lookahead_off): last value at or before each bar."""
    if other is None or other.dropna().empty:
        return pd.Series(np.nan, index=bars.index)
    o = other.dropna().sort_index()
    flag = (o > tv_ema(o, ema_len)).astype(float)
    pos = o.index.searchsorted(bars.index, side="right") - 1
    vals = np.where(pos >= 0, flag.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=bars.index)


def compute_indicators(bars: pd.DataFrame, p: VwapParams, events: pd.DatetimeIndex | None = None,
                       dxy: pd.Series | None = None, us10y: pd.Series | None = None) -> pd.DataFrame:
    df = bars.copy()
    tf = infer_timeframe(df.index)
    df["t_close"] = df.index + tf
    ny_open = df.index.tz_convert(NY)
    ny_close = pd.DatetimeIndex(df["t_close"]).tz_convert(NY)
    df["session"] = (df.index + SESSION_SHIFT).tz_convert(NY).strftime("%Y-%m-%d")

    src = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"].fillna(0.0)
    g = df["session"]
    sv = v.groupby(g).cumsum()
    spv = (src * v).groupby(g).cumsum()
    sp2v = (src * src * v).groupby(g).cumsum()
    vw = np.where(sv > 0, spv / sv.where(sv > 0), src)
    df["vwap"] = vw
    var = np.where(sv > 0, sp2v / sv.where(sv > 0) - df["vwap"] ** 2, 0.0)
    df["sd"] = np.sqrt(np.maximum(var, 0.0))
    for k, name in ((p.band1, "1"), (p.band2, "2")):
        df[f"up{name}"] = df["vwap"] + k * df["sd"]
        df[f"dn{name}"] = df["vwap"] - k * df["sd"]

    df["atr"] = tv_atr(df, p.atr_len)
    df["vol_sma"] = v.rolling(p.vol_len).mean()
    df["vol_ok"] = (not p.use_volume) | (df["vol_sma"].notna() & (v > df["vol_sma"] * p.vol_mult))

    om = ny_open.hour * 60 + ny_open.minute
    cm = ny_close.hour * 60 + ny_close.minute
    w0, w1 = (_minutes(x) for x in p.entry_window)
    df["in_window"] = (om >= w0) & (om < w1) if w0 <= w1 else (om >= w0) | (om < w1)
    df["flat_window"] = (cm >= _minutes(p.flatten_at)) & (cm < 18 * 60)

    # event blackout at the decision time (bar close)
    blocked = np.zeros(len(df), dtype=bool)
    if p.use_events and events is not None and len(events):
        # pin units: pandas 3 infers datetime resolution (us vs ns), and asi8 follows it
        e = pd.DatetimeIndex(events).as_unit("ns").asi8
        tc = pd.DatetimeIndex(df["t_close"]).as_unit("ns").asi8
        pre, post = p.block_before_min * 60_000_000_000, p.block_after_min * 60_000_000_000
        i = np.searchsorted(e, tc - post, side="left")
        ok = i < len(e)
        blocked[ok] = e[i[ok]] - pre <= tc[ok]
    if p.use_econ_release and "econ_release" in df:
        hit = df["econ_release"].fillna(0) > 0
        last = pd.Series(np.where(hit, df.index.as_unit("ns").asi8, np.nan), index=df.index).ffill()
        age = pd.DatetimeIndex(df["t_close"]).as_unit("ns").asi8 - last.to_numpy()
        blocked |= np.nan_to_num(age, nan=np.inf) <= p.block_after_min * 60_000_000_000
    df["blocked"] = blocked

    # macro: exported plot flags take priority (exact TradingView values)
    d_up = df["dxy_up"] if "dxy_up" in df else macro_flags(df, dxy, p.macro_ema_len)
    y_up = df["us10y_up"] if "us10y_up" in df else macro_flags(df, us10y, p.macro_ema_len)
    known = d_up.notna() & y_up.notna()
    against_long = (d_up == 1).astype(int) + (y_up == 1).astype(int)
    against_short = (d_up == 0).astype(int) + (y_up == 0).astype(int)
    lim = 0 if p.macro_strict else 1
    df["macro_long_ok"] = (not p.use_macro) | (known & (against_long <= lim))
    df["macro_short_ok"] = (not p.use_macro) | (known & (against_short <= lim))

    c, c1, vw1 = df["close"], df["close"].shift(1), df["vwap"].shift(1)
    long_rc = p.use_reclaim & (c1 < vw1) & (c > df["vwap"])
    long_rj = p.use_rejection & (c1 > vw1) & (df["low"] <= df["vwap"]) & (c > df["vwap"])
    short_rc = p.use_reclaim & (c1 > vw1) & (c < df["vwap"])
    short_rj = p.use_rejection & (c1 < vw1) & (df["high"] >= df["vwap"]) & (c < df["vwap"])
    base = df["vol_ok"] & df["in_window"] & ~df["flat_window"] & ~df["blocked"] & df["atr"].notna()
    df["long_sig"] = p.allow_long & (long_rc | long_rj) & (c < df["up2"]) & df["macro_long_ok"] & base
    df["short_sig"] = p.allow_short & (short_rc | short_rj) & (c > df["dn2"]) & df["macro_short_ok"] & base
    df["setup"] = np.where(long_rc, "reclaim", np.where(long_rj, "rejection",
                           np.where(short_rc, "reclaim", np.where(short_rj, "rejection", ""))))
    return df


# --------------------------------------------------------------------------- #
# Execution (TradingView broker emulator, no Bar Magnifier)
# --------------------------------------------------------------------------- #

@dataclass
class Trade:
    side: str
    setup: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    stop: float
    target: float
    risk_amount: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    commission: float = 0.0
    bars: int = 0

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk_amount if self.risk_amount else float("nan")


@dataclass
class BacktestRun:
    trades: pd.DataFrame
    equity: pd.Series                # bar-close equity incl. open P&L
    stats: dict
    open_trade: dict | None
    params: VwapParams
    bars: pd.DataFrame = field(repr=False)


def _exit_check(side: str, o: float, h: float, l: float, stop: float, target: float, check_open: bool):
    """(price, reason, is_stop) if stop/target hit on this bar, following TV's path rule."""
    if check_open:
        if (side == "long" and o <= stop) or (side == "short" and o >= stop):
            return o, "stop (gap)", True
        if (side == "long" and o >= target) or (side == "short" and o <= target):
            return o, "target (gap)", False
    high_first = (h - o) < (o - l) or ((h - o) == (o - l))  # high closer to open -> high first
    legs = ("high", "low") if high_first else ("low", "high")
    for leg in legs:
        if leg == "high":
            if side == "long" and h >= target:
                return target, "target", False
            if side == "short" and h >= stop:
                return stop, "stop", True
        else:
            if side == "long" and l <= stop:
                return stop, "stop", True
            if side == "short" and l <= target:
                return target, "target", False
    return None


def run_backtest(bars: pd.DataFrame, params: VwapParams = VwapParams(),
                 events: pd.DatetimeIndex | None = None,
                 dxy: pd.Series | None = None, us10y: pd.Series | None = None) -> BacktestRun:
    if events is None:
        events = parse_events(default_event_lines())
    df = compute_indicators(bars, params, events, dxy, us10y)
    tdf, equity, open_trade = execute(df, params)
    return BacktestRun(tdf, equity, summarize(tdf, equity, params.initial_capital), open_trade, params, df)


def execute(df: pd.DataFrame, p: VwapParams) -> tuple[pd.DataFrame, pd.Series, dict | None]:
    """Run the order logic over a frame with open/high/low/close, t_close, atr,
    long_sig, short_sig, flat_window and setup columns (from compute_indicators)."""
    O, H, L, C = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr, long_sig, short_sig = df["atr"].to_numpy(float), df["long_sig"].to_numpy(), df["short_sig"].to_numpy()
    flat_win, setup, idx = df["flat_window"].to_numpy(), df["setup"].to_numpy(), df.index
    tclose = pd.DatetimeIndex(df["t_close"])
    slip = p.slippage_ticks * p.mintick
    pv, com = p.point_value, p.commission_per_order

    realized = 0.0
    trades: list[Trade] = []
    pos: Trade | None = None
    pending: dict | None = None
    eq = np.empty(len(df))

    def close_pos(i: int, price: float, reason: str, t: pd.Timestamp) -> None:
        nonlocal pos, realized
        sign = 1 if pos.side == "long" else -1
        pos.exit_time, pos.exit_price, pos.exit_reason = t, price, reason
        pos.commission += com
        pos.pnl = sign * (price - pos.entry_price) * pos.qty * pv - pos.commission
        realized += pos.pnl
        trades.append(pos)
        pos = None

    for i in range(len(df)):
        just_filled = False
        if pending is not None:
            side = pending["side"]
            fill = O[i] + slip if side == "long" else O[i] - slip
            pos = Trade(side, pending["setup"], pending["signal_time"], idx[i], fill, pending["qty"],
                        pending["stop"], pending["target"], pending["risk"], commission=com)
            pending, just_filled = None, True

        if pos is not None:
            pos.bars += 1
            hit = _exit_check(pos.side, O[i], H[i], L[i], pos.stop, pos.target, check_open=True)
            if hit is not None:
                price, reason, is_stop = hit
                if is_stop:
                    price = price - slip if pos.side == "long" else price + slip
                close_pos(i, price, reason, idx[i])
            elif flat_win[i]:
                px = C[i] - slip if pos.side == "long" else C[i] + slip
                close_pos(i, px, "end of day", tclose[i])

        if pos is None and pending is None and (long_sig[i] or short_sig[i]):
            side = "long" if long_sig[i] else "short"
            dist = atr[i] * p.atr_mult
            equity = p.initial_capital + realized
            if dist > 0 and equity > 0:
                qty = equity * p.risk_pct / 100 / (dist * pv)
                qty = min(qty, equity * p.max_leverage / (C[i] * pv))
                qty = math.floor(qty / p.qty_step) * p.qty_step
                if qty > 0:
                    sgn = 1 if side == "long" else -1
                    pending = {"side": side, "setup": setup[i], "signal_time": tclose[i], "qty": qty,
                               "stop": C[i] - sgn * dist, "target": C[i] + sgn * dist * p.tp_r,
                               "risk": qty * dist * pv}

        open_pnl = 0.0
        if pos is not None:
            sign = 1 if pos.side == "long" else -1
            open_pnl = sign * (C[i] - pos.entry_price) * pos.qty * pv - pos.commission
        eq[i] = p.initial_capital + realized + open_pnl

    tdf = pd.DataFrame([{**asdict(t), "r_multiple": t.r_multiple} for t in trades])
    equity = pd.Series(eq, index=tclose, name="equity")
    open_trade = None
    if pos is not None:
        open_trade = {**asdict(pos), "unrealized": float(eq[-1] - p.initial_capital - realized)}
    return tdf, equity, open_trade


# --------------------------------------------------------------------------- #
# Stats (Strategy Tester style)
# --------------------------------------------------------------------------- #

def _max_dd(curve: np.ndarray) -> tuple[float, float]:
    if len(curve) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(curve)
    dd = peak - curve
    i = int(dd.argmax())
    return float(dd[i]), float(dd[i] / peak[i]) if peak[i] else 0.0


def summarize(trades: pd.DataFrame, equity: pd.Series, initial: float) -> dict:
    if trades.empty:
        return {"total_closed_trades": 0, "net_profit": 0.0, "net_profit_pct": 0.0}
    pnl = trades["pnl"].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    closed_curve = initial + np.concatenate([[0.0], np.cumsum(pnl)])
    dd_abs, dd_pct = _max_dd(closed_curve)
    bdd_abs, bdd_pct = _max_dd(equity.to_numpy())
    streak = best = 0
    for x in pnl:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)

    def side_stats(s):
        t = trades[trades["side"] == s]
        return {"trades": int(len(t)), "net_profit": float(t["pnl"].sum()),
                "percent_profitable": float((t["pnl"] > 0).mean()) if len(t) else float("nan")}

    years = trades.groupby(pd.DatetimeIndex(trades["exit_time"]).year)["pnl"].agg(["count", "sum"])
    return {
        "net_profit": float(pnl.sum()),
        "net_profit_pct": float(pnl.sum() / initial),
        "gross_profit": float(wins.sum()),
        "gross_loss": float(losses.sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "total_closed_trades": int(len(pnl)),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "percent_profitable": float(len(wins) / len(pnl)),
        "avg_trade": float(pnl.mean()),
        "avg_winning_trade": float(wins.mean()) if len(wins) else 0.0,
        "avg_losing_trade": float(losses.mean()) if len(losses) else 0.0,
        "ratio_avg_win_avg_loss": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else float("nan"),
        "largest_winning_trade": float(wins.max()) if len(wins) else 0.0,
        "largest_losing_trade": float(losses.min()) if len(losses) else 0.0,
        "avg_r_multiple": float(trades["r_multiple"].mean()),
        "max_drawdown_closed": dd_abs, "max_drawdown_closed_pct": dd_pct,
        "max_drawdown_bar_close": bdd_abs, "max_drawdown_bar_close_pct": bdd_pct,
        "max_consecutive_losses": int(best),
        "commission_paid": float(trades["commission"].sum()),
        "avg_bars_in_trade": float(trades["bars"].mean()),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        "long": side_stats("long"), "short": side_stats("short"),
        "by_year": {int(y): {"trades": int(r["count"]), "net_profit": float(r["sum"])} for y, r in years.iterrows()},
    }


def cad_view(trades: pd.DataFrame, equity: pd.Series, usd_cad: pd.Series, initial_usd: float) -> dict:
    """P&L in CAD: each trade converted at the USD/CAD rate of its exit date,
    the equity curve at each bar's date. The USD stats stay the reference for
    comparing with TradingView; this is the CAD report."""
    from forecaster.market_data import rate_on_dates

    if trades.empty:
        return {"net_profit_cad": 0.0, "trades": trades.assign(pnl_cad=[])}
    t = trades.copy()
    fx_exit = rate_on_dates(usd_cad, pd.DatetimeIndex(t["exit_time"]))
    t["usd_cad"] = fx_exit.to_numpy()
    t["pnl_cad"] = t["pnl"] * t["usd_cad"]
    eq_cad = equity * rate_on_dates(usd_cad, equity.index).to_numpy()
    start = float(eq_cad.dropna().iloc[0]) if eq_cad.notna().any() else float("nan")
    dd_abs, dd_pct = _max_dd(eq_cad.dropna().to_numpy())
    wins, losses = t.loc[t["pnl_cad"] > 0, "pnl_cad"], t.loc[t["pnl_cad"] <= 0, "pnl_cad"]
    return {
        "net_profit_cad": float(t["pnl_cad"].sum()),
        "start_equity_cad": start,
        "end_equity_cad": float(eq_cad.dropna().iloc[-1]) if eq_cad.notna().any() else float("nan"),
        "gross_profit_cad": float(wins.sum()), "gross_loss_cad": float(losses.sum()),
        "max_drawdown_cad": dd_abs, "max_drawdown_cad_pct": dd_pct,
        "unconverted_trades": int(t["usd_cad"].isna().sum()),
        "trades": t,
    }


# --------------------------------------------------------------------------- #
# Comparison with TradingView's "List of trades" export
# --------------------------------------------------------------------------- #

def load_tv_trades(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    """Parse a Strategy Tester 'List of trades' CSV into one row per trade.

    Columns are found by keyword because the export layout isn't verified here:
    a type column containing 'Entry'/'Exit' and 'long'/'short', a date column, a
    price column, and optionally a profit column.
    """
    raw = pd.read_csv(path)
    low = {c.lower(): c for c in raw.columns}
    pick = lambda *keys: next((low[c] for c in low if any(k in c for k in keys)), None)  # noqa: E731
    tcol, dcol, pcol = pick("type"), pick("date", "time"), pick("price")
    num = pick("trade #", "trade")
    prof = next((low[c] for c in low if c.startswith("profit") and "%" not in c and "cum" not in c), None)
    if not (tcol and dcol and pcol):
        raise ValueError(f"can't find type/date/price columns in {list(raw.columns)}")
    r = raw.copy()
    r["_t"] = pd.to_datetime(r[dcol]).dt.tz_localize(tz) if pd.to_datetime(r[dcol]).dt.tz is None \
        else pd.to_datetime(r[dcol])
    r["_t"] = r["_t"].dt.tz_convert("UTC")
    kind = r[tcol].astype(str).str.lower()
    ent, ext = r[kind.str.contains("entry")], r[kind.str.contains("exit")]
    key = num or None
    if key:
        m = ent.merge(ext, on=key, suffixes=("_e", "_x"))
        side = m[f"{tcol}_e"].astype(str).str.lower().str.contains("long").map({True: "long", False: "short"})
        out = pd.DataFrame({"side": side, "entry_time": m["_t_e"], "entry_price": m[f"{pcol}_e"].astype(float),
                            "exit_time": m["_t_x"], "exit_price": m[f"{pcol}_x"].astype(float)})
        if prof:
            out["pnl"] = pd.to_numeric(m[f"{prof}_x"], errors="coerce")
        return out.sort_values("entry_time").reset_index(drop=True)
    side = ent[tcol].astype(str).str.lower().str.contains("long").map({True: "long", False: "short"})
    return pd.DataFrame({"side": side.values, "entry_time": ent["_t"].values,
                         "entry_price": ent[pcol].astype(float).values}).sort_values("entry_time")


def compare_with_tradingview(run: BacktestRun, tv: pd.DataFrame, tolerance: pd.Timedelta | None = None) -> dict:
    """Match trades by side and entry time (within one bar). Reports what lines up."""
    tol = tolerance or infer_timeframe(run.bars.index)
    py = run.trades[["side", "entry_time", "entry_price", "exit_time", "exit_price", "pnl"]].copy()
    py["entry_time"] = pd.to_datetime(py["entry_time"], utc=True)
    tv = tv.copy()
    tv["entry_time"] = pd.to_datetime(tv["entry_time"], utc=True)
    used, matches = set(), []
    for i, t in tv.iterrows():
        cand = py[(py["side"] == t["side"]) & ((py["entry_time"] - t["entry_time"]).abs() <= tol)]
        cand = cand[~cand.index.isin(used)]
        if len(cand):
            j = (cand["entry_time"] - t["entry_time"]).abs().idxmin()
            used.add(j)
            matches.append((i, j))
    m = pd.DataFrame([{"tv": i, "py": j,
                       "entry_diff": py.loc[j, "entry_price"] - tv.loc[i, "entry_price"],
                       "pnl_diff": (py.loc[j, "pnl"] - tv.loc[i, "pnl"]) if "pnl" in tv else np.nan}
                      for i, j in matches])
    return {
        "tv_trades": int(len(tv)), "py_trades": int(len(py)), "matched": int(len(m)),
        "match_rate_tv": float(len(m) / len(tv)) if len(tv) else float("nan"),
        "only_in_tv": tv.drop(index=[i for i, _ in matches]).reset_index(drop=True),
        "only_in_py": py.drop(index=list(used)).reset_index(drop=True),
        "mean_abs_entry_diff": float(m["entry_diff"].abs().mean()) if len(m) else float("nan"),
        "mean_abs_pnl_diff": float(m["pnl_diff"].abs().mean()) if len(m) and m["pnl_diff"].notna().any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def format_stats(s: dict) -> str:
    if not s.get("total_closed_trades"):
        return "No closed trades."
    keys = ["net_profit", "net_profit_pct", "total_closed_trades", "percent_profitable", "profit_factor",
            "avg_trade", "ratio_avg_win_avg_loss", "avg_r_multiple", "largest_winning_trade",
            "largest_losing_trade", "max_drawdown_closed", "max_drawdown_closed_pct",
            "max_drawdown_bar_close_pct", "max_consecutive_losses", "commission_paid", "avg_bars_in_trade"]
    pct = {"net_profit_pct", "percent_profitable", "max_drawdown_closed_pct", "max_drawdown_bar_close_pct"}
    lines = [f"{k:>28}: {s[k]:.2%}" if k in pct else f"{k:>28}: {s[k]:,.2f}" for k in keys]
    lines.append(f"{'exit reasons':>28}: {s['exit_reasons']}")
    lines.append(f"{'long / short':>28}: {s['long']} / {s['short']}")
    lines += [f"{y:>28}: {v['trades']} trades, net {v['net_profit']:,.2f}" for y, v in s["by_year"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Backtest the VWAP gold strategy (port of pine/vwap_gold.pine)")
    ap.add_argument("csv", nargs="?", help="TradingView 'Export chart data' CSV (same symbol/timeframe as the tester)")
    ap.add_argument("--events-file", help="extra events, 'YYYY-MM-DD HH:MM' New York time, one per line")
    ap.add_argument("--fred-events", action="store_true", help="add CPI/PPI release times from FRED (needs FRED_API_KEY)")
    ap.add_argument("--pine-events", action="store_true", help="print the event list for the Pine input and exit")
    ap.add_argument("--dxy-csv", help="DXY bars CSV if the export has no 'DXY up' column")
    ap.add_argument("--us10y-csv", help="US10Y bars CSV if the export has no 'US10Y up' column")
    ap.add_argument("--compare", help="TradingView 'List of trades' CSV to compare against")
    ap.add_argument("--tv-tz", default="UTC", help="timezone of dates in the List of trades export")
    ap.add_argument("--params", help="JSON of VwapParams overrides, e.g. '{\"slippage_ticks\": 20}'")
    ap.add_argument("--usdcad-csv", help="USD/CAD daily CSV (time + close); default: cached CAD=X")
    ap.add_argument("--out", help="write trades CSV + stats JSON with this prefix")
    args = ap.parse_args(argv)

    lines = default_event_lines()
    if args.events_file:
        lines += Path(args.events_file).read_text().splitlines()
    if args.fred_events:
        from forecaster.cache import Cache
        from forecaster.config import load_settings
        from forecaster.sources.fred import FredClient

        s = load_settings()
        if not s.fred_api_key:
            print("FRED_API_KEY not set", file=sys.stderr)
            return 1
        lines += fred_release_lines(FredClient(s.fred_api_key, Cache(s.cache_dir)),
                                    pd.Timestamp("2015-01-01"), pd.Timestamp.today() + pd.DateOffset(years=1))
    if args.pine_events:
        print(pine_events_text(lines))
        return 0
    if not args.csv:
        ap.error("csv is required unless --pine-events")

    params = VwapParams(**json.loads(args.params)) if args.params else VwapParams()
    bars = load_tradingview_csv(args.csv)
    other = lambda f: load_tradingview_csv(f)["close"] if f else None  # noqa: E731
    run = run_backtest(bars, params, parse_events(lines), other(args.dxy_csv), other(args.us10y_csv))
    print(f"{len(bars):,} bars {bars.index[0]:%Y-%m-%d} -> {bars.index[-1]:%Y-%m-%d}, "
          f"timeframe {infer_timeframe(bars.index)}")
    if params.use_macro and "dxy_up" not in bars and not args.dxy_csv:
        print("[warn] macro filter on but no DXY data: every entry is blocked. Export the 'DXY up'/'US10Y up' "
              "plots, pass --dxy-csv/--us10y-csv, or set use_macro false.", file=sys.stderr)
    print("Results in the instrument's quote currency (USD for XAUUSD), matching TradingView's Strategy Tester:")
    print(format_stats(run.stats))
    fx = None
    if args.usdcad_csv:
        fx = load_tradingview_csv(args.usdcad_csv)["close"]
        fx.index = fx.index.tz_convert(None).normalize()
    else:
        from forecaster.config import load_settings
        from forecaster.market_data import USD_CAD_SYMBOL, ParquetStore

        cached = ParquetStore(load_settings().cache_dir / "ohlcv").read(USD_CAD_SYMBOL)
        fx = None if cached is None else cached["Close"]
    if fx is not None and len(run.trades):
        c = cad_view(run.trades, run.equity, fx, params.initial_capital)
        print(f"\nIn CAD (each trade at its exit-date USD/CAD): net {c['net_profit_cad']:,.2f} CAD, "
              f"equity C${c['start_equity_cad']:,.2f} -> C${c['end_equity_cad']:,.2f}, "
              f"max drawdown C${c['max_drawdown_cad']:,.2f} ({c['max_drawdown_cad_pct']:.1%})"
              + (f"; {c['unconverted_trades']} trades before the first rate" if c["unconverted_trades"] else ""))
        if args.out:
            c["trades"].to_csv(f"{args.out}_trades_cad.csv", index=False)
    else:
        print("\n[warn] no USD/CAD rates (run python -m forecaster.market_data or pass --usdcad-csv): "
              "CAD report skipped", file=sys.stderr)
    if run.open_trade:
        print(f"open trade (not in stats): {run.open_trade['side']} since {run.open_trade['entry_time']}")
    if args.compare:
        c = compare_with_tradingview(run, load_tv_trades(args.compare, args.tv_tz))
        print(f"\nTradingView comparison: {c['matched']}/{c['tv_trades']} TV trades matched "
              f"({c['match_rate_tv']:.0%}); Python has {c['py_trades']}; mean |entry diff| "
              f"{c['mean_abs_entry_diff']:.3f}; mean |pnl diff| {c['mean_abs_pnl_diff']:.2f}")
    if args.out:
        run.trades.to_csv(f"{args.out}_trades.csv", index=False)
        Path(f"{args.out}_stats.json").write_text(json.dumps(run.stats, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
