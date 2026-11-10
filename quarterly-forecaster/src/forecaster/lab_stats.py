"""Strategy Lab statistics: performance, edge-vs-luck tests, slices, walk-forward, verdicts.

All returns are per-trade net returns (after spread, slippage and commission) relative to the
entry price, i.e. what one unit earned. Expectancy = mean net return per trade.

Edge vs luck, per setup:
  * bootstrap 95% CI of expectancy (trades resampled with replacement),
  * Monte Carlo of trade ORDER (permutations) -> distribution of max drawdown,
  * 1,000 random-entry strategies with the same number of trades, holding times, sides, windows
    and costs (lab_engine.random_entry_means) -> one-sided p-value of the real expectancy,
  * walk-forward: tune on a training window, trade the next unseen window, roll forward; only the
    out-of-sample (OOS) trades are judged.
Verdict: EDGE = p < 0.05 AND >= 100 OOS trades AND OOS expectancy > 0 after costs;
WEAK = OOS expectancy > 0 but one of the other two fails (or p < 0.20); otherwise NO EDGE / LUCK.
Many setups are tested at once, so a few will pass p < 0.05 by chance: `bonferroni_ok` says
whether the p-value also survives a correction for the number of setups tested.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NY = "America/New_York"
EDGE_P = 0.05
EDGE_MIN_TRADES = 100
WEAK_P = 0.20


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #

def max_drawdown(rets: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative (summed, fixed size) return, <= 0."""
    if len(rets) == 0:
        return 0.0
    eq = np.cumsum(rets)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    return float(np.min(np.r_[0.0, eq - peak]))


def longest_losing_streak(rets: np.ndarray) -> int:
    best = cur = 0
    for r in rets:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def summary(trades: pd.DataFrame, days: int | None = None) -> dict:
    """Trade count, win rate, avg win/loss, expectancy, profit factor, Sharpe (daily, annualised
    with 252 days; days without trades count as 0), max drawdown, longest losing streak."""
    t = trades.dropna(subset=["net_ret"]) if len(trades) else trades
    n = len(t)
    if n == 0:
        return {"trades": 0, "win_rate": None, "avg_win": None, "avg_loss": None, "expectancy": None,
                "profit_factor": None, "sharpe": None, "max_dd": 0.0, "losing_streak": 0, "avg_r": None,
                "total": 0.0, "avg_bars": None}
    r = t["net_ret"].to_numpy(float)
    wins, losses = r[r > 0], r[r <= 0]
    daily = t.groupby("day")["net_ret"].sum()
    n_days = max(days or 0, len(daily))
    d = np.r_[daily.to_numpy(float), np.zeros(n_days - len(daily))]
    sharpe = float(d.mean() / d.std(ddof=1) * np.sqrt(252)) if len(d) > 1 and d.std(ddof=1) > 0 else None
    gl = -losses.sum()
    return {
        "trades": n,
        "win_rate": float(len(wins) / n),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "expectancy": float(r.mean()),
        "profit_factor": float(wins.sum() / gl) if gl > 0 else None,
        "sharpe": sharpe,
        "max_dd": max_drawdown(r),
        "losing_streak": longest_losing_streak(r),
        "avg_r": float(np.nanmean(t["r_mult"])) if "r_mult" in t and t["r_mult"].notna().any() else None,
        "total": float(r.sum()),
        "avg_bars": float(t["bars_held"].mean()) if "bars_held" in t else None,
    }


# --------------------------------------------------------------------------- #
# Edge vs luck
# --------------------------------------------------------------------------- #

def bootstrap_ci(rets: np.ndarray, n: int = 5000, level: float = 0.95, seed: int = 0,
                 days: np.ndarray | None = None) -> tuple[float, float] | None:
    """Bootstrap CI of the mean trade return. With `days`, whole days are resampled (trades on the
    same day are correlated; resampling single trades would make the interval too narrow)."""
    rets = np.asarray(rets, float)
    if len(rets) < 2:
        return None
    rng = np.random.default_rng(seed)
    if days is None:
        means = rets[rng.integers(0, len(rets), (n, len(rets)))].mean(axis=1)
    else:
        _, inv = np.unique(np.asarray(days), return_inverse=True)
        sums, cnts = np.bincount(inv, weights=rets), np.bincount(inv)
        if len(sums) < 2:
            return None
        idx = rng.integers(0, len(sums), (n, len(sums)))
        means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def order_monte_carlo(rets: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Shuffle trade order: the mean can't change, the path can. Max drawdown percentiles."""
    rets = np.asarray(rets, float)
    if len(rets) < 2:
        return {"dd_median": None, "dd_p95": None}
    rng = np.random.default_rng(seed)
    dds = np.array([max_drawdown(rng.permutation(rets)) for _ in range(n)])
    return {"dd_median": float(np.median(dds)), "dd_p95": float(np.quantile(dds, 0.05))}


def p_value(real: float, random_means: np.ndarray) -> float | None:
    """One-sided: share of random strategies doing at least as well (+1 smoothing)."""
    if len(random_means) == 0 or real is None:
        return None
    return float((1 + np.sum(random_means >= real)) / (1 + len(random_means)))


def verdict(oos: dict, p: float | None) -> tuple[str, list[str]]:
    why = []
    n, e = oos.get("trades", 0), oos.get("expectancy")
    if e is None or e <= 0:
        why.append("out-of-sample expectancy after costs is not positive")
    if n < EDGE_MIN_TRADES:
        why.append(f"only {n} out-of-sample trades (need {EDGE_MIN_TRADES}+)")
    if p is None or p >= EDGE_P:
        why.append("p-value vs random entries is n/a" if p is None else
                   f"p = {p:.3f} vs random entries (need < {EDGE_P})")
    if not why:
        return "EDGE", ["p < 0.05 vs 1,000 random-entry strategies, 100+ OOS trades, positive expectancy after costs"]
    if e is not None and e > 0 and (n >= EDGE_MIN_TRADES or (p is not None and p < WEAK_P)):
        return "WEAK", why
    return "NO EDGE", why


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #

def vix_band(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "unknown"
    return "<15" if v < 15 else "15-20" if v < 20 else "20-30" if v < 30 else "30+"


def add_slice_columns(trades: pd.DataFrame, vix_prev_close: pd.Series | None = None,
                      events: dict[str, set] | None = None) -> pd.DataFrame:
    """Entry-time buckets. VIX band uses the PREVIOUS day's close (known before the open)."""
    t = trades.copy()
    if t.empty:
        return t
    et = pd.DatetimeIndex(t["entry_time"]).tz_convert(NY)
    t["year"] = et.year
    t["month"] = et.strftime("%Y-%m")
    t["week"] = et.strftime("%G-W%V")
    t["date"] = et.strftime("%Y-%m-%d")
    t["weekday"] = et.day_name().str[:3]
    t["time_of_day"] = [f"{h:02d}:{0 if m < 30 else 30:02d}" for h, m in zip(et.hour, et.minute)]
    day = pd.DatetimeIndex(pd.to_datetime(t["day"]))
    if vix_prev_close is not None and len(vix_prev_close):
        v = vix_prev_close.dropna().sort_index()
        vi = pd.DatetimeIndex(v.index)
        v.index = (vi.tz_localize(None) if vi.tz is not None else vi).normalize()
        pos = v.index.searchsorted(day, side="left") - 1          # last close strictly before the day
        vals = np.where(pos >= 0, v.to_numpy()[np.clip(pos, 0, None)], np.nan)
        t["vix_band"] = [vix_band(x) for x in vals]
    else:
        t["vix_band"] = "unknown"
    if events:
        lab = []
        for d in day.strftime("%Y-%m-%d"):
            hit = [k for k in ("FOMC", "CPI", "PPI") if d in events.get(k, set())]
            lab.append("+".join(hit) if hit else "normal")
        t["event_day"] = lab
    else:
        t["event_day"] = "unknown"
    return t


SLICES = ("year", "month", "week", "date", "weekday", "time_of_day", "vix_band", "event_day")


def slice_stats(trades: pd.DataFrame, by: str) -> list[dict]:
    if trades.empty or by not in trades:
        return []
    out = []
    for k, g in trades.groupby(by, sort=True):
        r = g["net_ret"].dropna().to_numpy(float)
        if len(r) == 0:
            continue
        out.append({"key": str(k), "trades": len(r), "win_rate": float((r > 0).mean()),
                    "expectancy": float(r.mean()), "total": float(r.sum())})
    return out


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #

@dataclass
class Fold:
    train: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]
    best: dict = field(default_factory=dict)
    is_stats: dict = field(default_factory=dict)
    oos_stats: dict = field(default_factory=dict)


def make_folds(days: pd.DatetimeIndex, train_months: int = 12, test_months: int = 3) -> list[Fold]:
    """Rolling windows on trading days: train [a, a+train), test [a+train, a+train+test), step = test."""
    if len(days) == 0:
        return []
    first, last = days.min().normalize(), days.max().normalize()
    folds = []
    a = first
    while True:
        b = a + pd.DateOffset(months=train_months)
        c = b + pd.DateOffset(months=test_months)
        if b > last:
            break
        folds.append(Fold((a, b), (b, min(c, last + pd.Timedelta(days=1)))))
        if c > last:
            break
        a = a + pd.DateOffset(months=test_months)
    return folds


def grid_combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]


def _in(trades: pd.DataFrame, span: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    d = pd.to_datetime(trades["day"])
    return trades[(d >= span[0]) & (d < span[1])]


def walk_forward(run_combo: Callable[[dict], pd.DataFrame], grid: dict, days: pd.DatetimeIndex,
                 train_months: int = 12, test_months: int = 3, min_is_trades: int = 30) -> dict:
    """`run_combo(params) -> trades over the whole history` (signals are causal, so computing them
    once on all data equals computing them per window). For each fold pick the combo with the best
    in-sample expectancy (with >= min_is_trades trades), then keep only its trades in the next,
    unseen test window. Returns OOS trades + per-fold choices."""
    combos = grid_combos(grid)
    runs = {i: run_combo(c) for i, c in enumerate(combos)}
    folds = make_folds(days, train_months, test_months)
    oos_parts, is_parts = [], []
    for f in folds:
        best_i, best_e = None, -np.inf
        for i, tr in runs.items():
            t = _in(tr, f.train)
            if len(t) < min_is_trades:
                continue
            e = t["net_ret"].mean()
            if e > best_e:
                best_i, best_e = i, e
        if best_i is None:                       # nothing traded enough in-sample: skip the fold
            f.best = {}
            continue
        f.best = combos[best_i]
        is_t, oos_t = _in(runs[best_i], f.train), _in(runs[best_i], f.test)
        f.is_stats, f.oos_stats = summary(is_t), summary(oos_t)
        oos_parts.append(oos_t.assign(fold=len(oos_parts)))
        is_parts.append(is_t)
    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else runs[0].iloc[0:0]
    is_all = pd.concat(is_parts, ignore_index=True) if is_parts else runs[0].iloc[0:0]
    latest = next((f.best for f in reversed(folds) if f.best), combos[0])
    return {"oos": oos, "is": is_all, "folds": folds, "combos": len(combos), "latest_params": latest,
            "runs": runs, "combo_list": combos}


def overfit_warnings(tuned: list[str], combos: int, folds: list[Fold], is_stats: dict, oos_stats: dict) -> list[str]:
    w = []
    if len(tuned) >= 3:
        w.append(f"{len(tuned)} parameters tuned ({', '.join(tuned)}): high risk of curve-fitting")
    if combos >= 20:
        w.append(f"{combos} parameter combinations tried per window: the in-sample winner is partly luck")
    used = [tuple(sorted(f.best.items())) for f in folds if f.best]
    if len(used) >= 3 and len(set(used)) / len(used) > 0.5:
        w.append(f"best parameters changed in {len(set(used))} of {len(used)} windows: unstable")
    ie, oe = is_stats.get("expectancy"), oos_stats.get("expectancy")
    if ie is not None and oe is not None and ie > 0:
        if oe <= 0:
            w.append("positive in-sample, not out-of-sample: the in-sample result was fitted noise")
        elif oe < 0.5 * ie:
            w.append(f"out-of-sample expectancy is {oe / ie:.0%} of in-sample: heavy decay")
    return w
