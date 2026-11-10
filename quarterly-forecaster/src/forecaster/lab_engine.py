"""Backtest engine for the Strategy Lab (fast NumPy event simulation).

Why not backtesting.py / vectorbt directly: walk-forward grids plus 1,000 random-entry strategies
per setup mean tens of thousands of runs; backtesting.py loops in Python per bar. This engine
simulates only from each entry to its exit with array slices. tests/test_lab_engine.py checks it
trade-for-trade against backtesting.py on the same rules.

Rules (one position per strategy at a time; long and short):
  * a signal on bar i's close fills at bar i+1's OPEN (no look-ahead), only if bar i+1 is in the
    same trading day, inside the day-trading window, not in its first minute, and there is room
    for at least one bar before the time exit; at most `max_trades` entries per day.
  * prices are treated as mids: buys pay mid + half spread, sells get mid - half spread. Spread
    comes from the bar's Spread column (OANDA bid/ask) or the cost model.
  * slippage applies to market orders (entry, time exit) and stops; limit targets don't slip.
  * stop: long exits when bid <= stop (low - half spread); a gap through the stop fills at the
    open (worse). target: long exits when bid >= target, at the target (or a better gap open).
    If a bar touches both, the STOP is assumed first (conservative).
  * time exit: at the close of the last bar that ends at least `last_minute_guard` minutes before
    the window end ("flat by the close", no trades in the last minute).
  * commission per unit, both sides. Returns are per unit, relative to the entry mid.
  * data ending before the exit (live shadow trading) leaves the trade open (exit NaN).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from forecaster.intraday import NY, InstrumentSpec, _hm


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.0        # per unit, per side (price units)
    spread: float = 0.0            # full bid/ask spread in price units, if the bars have no Spread column
    slippage: float = 0.0          # price units per market/stop fill
    spread_bps: float = 0.0        # alternatively a spread in bps of price (added to `spread`)
    slippage_bps: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Defaults, documented in README: IBKR-style fixed US commission US$0.005/share; SPY/QQQ spread a
# cent; gold on OANDA: no commission, the real bid/ask spread from the candles + a few cents slip.
DEFAULT_COSTS = {
    "SPY": CostModel(commission=0.005, spread=0.01, slippage=0.01),
    "QQQ": CostModel(commission=0.005, spread=0.01, slippage=0.01),
    "XAU_USD": CostModel(commission=0.0, spread=0.35, slippage=0.05),
}

TRADE_COLS = ["signal_i", "entry_i", "exit_i", "signal_time", "entry_time", "exit_time", "day", "side",
              "entry_ref", "entry", "stop", "target", "exit_ref", "exit", "reason", "bars_held",
              "gross_ret", "cost_ret", "net_ret", "r_mult"]


@dataclass
class Prepared:
    """Arrays derived once per bar set."""
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    hs: np.ndarray                  # half spread per bar
    slip: np.ndarray
    day: np.ndarray                 # trading-day label per bar
    entry_ok: np.ndarray            # bar may be an ENTRY (fill) bar
    last_exit: np.ndarray           # per bar: index of the day's time-exit bar (-1 if none)
    t: pd.DatetimeIndex
    complete_day: np.ndarray        # per bar: its day's window is fully in the data


def prepare(bars: pd.DataFrame, spec: InstrumentSpec, minutes: int, costs: CostModel,
            last_minute_guard: int = 1, first_minute_guard: int = 1) -> Prepared:
    from forecaster.intraday import trading_day

    o, h, lo, c = (bars[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    n = len(bars)
    spread = bars["Spread"].to_numpy(float) if "Spread" in bars else np.full(n, costs.spread)
    spread = np.where(np.isfinite(spread), spread, costs.spread) + c * costs.spread_bps / 1e4
    hs = spread / 2
    slip = np.full(n, costs.slippage) + c * costs.slippage_bps / 1e4
    day = np.asarray(trading_day(bars.index, spec.vwap_anchor))
    t = bars.index.tz_convert(NY)
    start = np.asarray(t.hour * 60 + t.minute)
    end = start + minutes
    w0, w1 = _hm(spec.session_start), _hm(spec.session_end)
    in_window = (start >= w0) & (end <= w1)
    exit_ok = in_window & (end <= w1 - last_minute_guard)
    entry_ok = in_window & (start >= w0 + first_minute_guard) & (end <= w1 - last_minute_guard)
    # last allowed exit bar per day
    last_exit = np.full(n, -1)
    if n:
        idx = np.arange(n)
        cand = pd.Series(np.where(exit_ok, idx, -1)).groupby(day).transform("max").to_numpy()
        last_exit = cand
    # a day is complete if the data reaches the time-exit point: some bar of that day (window or
    # not, e.g. gold's 16:45-17:00 bar) ends at/after it. Live data that stops mid-day isn't.
    complete = np.zeros(n, bool)
    if n:
        ends = pd.Series(end).groupby(day).transform("max").to_numpy()
        complete = ends >= w1 - last_minute_guard
    return Prepared(o, h, lo, c, hs, slip, day, entry_ok, last_exit, bars.index, complete)


def simulate(pp: Prepared, signals: pd.DataFrame, costs: CostModel, max_trades: int = 99,
             allow_open: bool = False) -> pd.DataFrame:
    """Signals (index = bar i, columns side, stop, target) -> trades DataFrame (TRADE_COLS).
    Days whose window isn't complete in the data are skipped, unless `allow_open` (live shadow
    trading), where a trade with no exit yet stays open."""
    rows = []
    n = len(pp.o)
    busy_until = -1
    per_day: dict = {}
    comm = 2 * costs.commission
    for i, side, stop, target in zip(signals.index.to_numpy(), signals["side"].to_numpy(),
                                     signals["stop"].to_numpy(float), signals["target"].to_numpy(float)):
        j = i + 1
        if j >= n or j <= busy_until or pp.day[j] != pp.day[i] or not pp.entry_ok[j]:
            continue
        complete = bool(pp.complete_day[j])
        if not complete and not allow_open:
            continue
        if per_day.get(pp.day[j], 0) >= max_trades:
            continue
        ref = pp.o[j]
        entry = ref + side * (pp.hs[j] + pp.slip[j])
        if not ((side > 0 and stop < entry < target) or (side < 0 and target < entry < stop)):
            continue                                          # gapped past the stop/target: no trade
        per_day[pp.day[j]] = per_day.get(pp.day[j], 0) + 1
        last = int(pp.last_exit[j])
        end = last if complete else n - 1
        k = slice(j, end + 1)
        hs, o, hh, ll = pp.hs[k], pp.o[k], pp.h[k], pp.lo[k]
        if side > 0:
            stop_hit, tgt_hit = ll - hs <= stop, hh - hs >= target
        else:
            stop_hit, tgt_hit = hh + hs >= stop, ll + hs <= target
        hit = stop_hit | tgt_hit
        if hit.any():
            m = int(np.argmax(hit))
            b = j + m
            open_px = o[m] - side * hs[m] if m > 0 else None      # bid (long) / ask (short) at this bar's open
            if stop_hit[m]:                                        # stop first if both (conservative)
                reason = "stop"
                px = stop if open_px is None else (min(stop, open_px) if side > 0 else max(stop, open_px))
                exit_ = px - side * pp.slip[b]
                exit_ref = px + side * hs[m]
            else:
                reason = "target"
                exit_ = target if open_px is None else (max(target, open_px) if side > 0 else min(target, open_px))
                exit_ref = exit_ + side * hs[m]
        elif complete:
            b, reason = last, "time"
            exit_ref = pp.c[b]
            exit_ = exit_ref - side * (pp.hs[b] + pp.slip[b])
        else:                                                     # live: no exit yet
            b, reason, exit_ref, exit_ = n - 1, "open", pp.c[n - 1], np.nan
        busy_until = b
        gross = side * (exit_ref - ref) / ref
        net = (side * (exit_ - entry) - comm) / ref if np.isfinite(exit_) else np.nan
        risk = abs(entry - stop)
        rows.append((i, j, b, pp.t[i], pp.t[j], pp.t[b], pp.day[j], int(side), ref, entry, stop, target,
                     exit_ref, exit_, reason, b - j + 1, gross,
                     gross - net if np.isfinite(net) else np.nan, net,
                     (net * ref / risk) if risk > 0 and np.isfinite(net) else np.nan))
    return pd.DataFrame(rows, columns=TRADE_COLS)


def run(bars: pd.DataFrame, spec: InstrumentSpec, minutes: int, signals: pd.DataFrame, costs: CostModel,
        max_trades: int = 99, allow_open: bool = False) -> pd.DataFrame:
    return simulate(prepare(bars, spec, minutes, costs), signals, costs, max_trades, allow_open)


# --------------------------------------------------------------------------- #
# Random-entry benchmark (same holding times, same costs, same windows)
# --------------------------------------------------------------------------- #

def random_entry_means(pp: Prepared, hold_bars: np.ndarray, sides: np.ndarray, costs: CostModel,
                       n_strategies: int = 1000, seed: int = 0, allowed_days: np.ndarray | None = None,
                       trade_days: np.ndarray | None = None) -> np.ndarray:
    """Mean net return of `n_strategies` random strategies with the same number of trades, holding
    times, sides and costs (exit at the holding time's close or the time exit, whichever is first).

    With `trade_days` (the real trades' days) each random trade is placed on the SAME day as a real
    trade, at a random allowed bar: this keeps the real strategy's clustering (several trades on one
    volatile day are correlated). Without it, entries are uniform over `allowed_days`, which treats
    trades as independent and makes p-values too optimistic for clustered strategies."""
    rng = np.random.default_rng(seed)
    n_tr = len(hold_bars)
    if n_tr == 0:
        return np.array([])
    cand = np.flatnonzero(pp.entry_ok & (pp.last_exit >= np.arange(len(pp.o))))
    if allowed_days is not None:
        cand = cand[np.isin(pp.day[cand], allowed_days)]
    if len(cand) == 0:
        return np.array([])
    pick = rng.random((n_strategies, n_tr))
    if trade_days is not None:
        cdays = pp.day[cand]                                     # cand is sorted, so days are grouped
        uniq, first, count = np.unique(cdays, return_index=True, return_counts=True)
        pos = np.searchsorted(uniq, np.asarray(trade_days, dtype=cdays.dtype))
        ok = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == np.asarray(trade_days, dtype=cdays.dtype))
        if not ok.all():                                         # a day without allowed bars: fall back
            return random_entry_means(pp, hold_bars, sides, costs, n_strategies, seed, allowed_days)
        j = cand[first[pos][None, :] + (pick * count[pos][None, :]).astype(int)]
        hold = np.broadcast_to(hold_bars, (n_strategies, n_tr))
    else:
        j = cand[(pick * len(cand)).astype(int)]
        hold = hold_bars[rng.integers(0, n_tr, (n_strategies, n_tr))]
    side = sides[rng.integers(0, n_tr, (n_strategies, n_tr))]
    b = np.minimum(j + np.maximum(hold, 1) - 1, pp.last_exit[j])
    ref = pp.o[j]
    entry = ref + side * (pp.hs[j] + pp.slip[j])
    exit_ = pp.c[b] - side * (pp.hs[b] + pp.slip[b])
    net = (side * (exit_ - entry) - 2 * costs.commission) / ref
    return net.mean(axis=1)
