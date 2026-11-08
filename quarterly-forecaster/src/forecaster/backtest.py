"""Walk-forward backtest of the ensemble against a naive historical baseline.

Protocol
--------
* Origins T: calendar quarter-ends from `start` (default 2004-12-31, so the
  first forecasts are for 2005) up to the last date with data.
* At every origin the full ensemble is re-estimated from scratch on data
  available at T (`ensemble_forecast(data, asof=T)`: HMM, GARCH, LightGBM
  all refit each quarter). The expanding-window regime-feature table is built
  once and sliced to T; that is provably identical to rebuilding it
  (test_regime_table_has_no_lookahead) and saves most of the runtime.
* Each origin forecasts the ends of the 4 following quarters. A forecast is
  scored only once its target quarter is complete in the data.
* Baseline ("naive historical average" / climatology): the 10/50/90th
  percentiles of *all* past n-trading-day log returns available at T, where
  n matches the forecast horizon. Its drawdown probability is the historical
  frequency of a >threshold drawdown within n days. Same information set,
  zero modelling.

Metrics (log-return space)
--------------------------
coverage        share of outcomes inside [P10, P90]; ~80% if calibrated
below / above   share under P10 / over P90; ~10% each
hit_rate        sign(P50) == sign(outcome); note "always up" is a strong
                benchmark in equities and is what the baseline does
pinball         mean quantile loss over 0.1/0.5/0.9 (proper score, lower = better)
width           mean P90 - P10
brier_dd        Brier score of P(>10% drawdown by target)
skill           1 - pinball_model / pinball_baseline (> 0 means model better)

Overlapping horizons make rows dependent, so the model-minus-baseline loss
difference gets a circular block-bootstrap CI over origins (blocks of 4
quarters, which covers the maximum overlap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from forecaster.forecast import (
    QUANTILES,
    EnsembleConfig,
    PointInTimeData,
    business_days_between,
    ensemble_forecast,
    regime_feature_table,
    regime_observations,
    target_dates,
)

STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    # Rows whose *target date* falls in the window: forecasts made going into
    # and through each crisis.
    "2008 GFC": ("2008-01-01", "2009-06-30"),
    "2020 COVID": ("2020-01-01", "2020-12-31"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
}


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2004-12-31"
    end: str | None = None
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    drawdown_threshold: float = 0.10
    n_boot: int = 2000
    block: int = 4
    seed: int = 0


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def pinball(y: np.ndarray, f: np.ndarray, q: float) -> np.ndarray:
    d = np.asarray(y) - np.asarray(f)
    return np.maximum(q * d, (q - 1) * d)


def origins(prices: pd.Series, start: str, end: str | None = None) -> list[pd.Timestamp]:
    last = prices.index.max() if end is None else min(pd.Timestamp(end), prices.index.max())
    qs = pd.period_range(pd.Timestamp(start), last, freq="Q")
    out = [q.end_time.normalize() for q in qs]
    return [t for t in out if pd.Timestamp(start) <= t <= last]


def realized_close(prices: pd.Series, D: pd.Timestamp) -> float:
    """Close at target date D, or NaN if D's quarter isn't complete in the data."""
    last_bday = pd.bdate_range(end=D, periods=1)[0]
    if prices.index.max() < last_bday:
        return float("nan")
    p = prices.loc[:D]
    return float(p.iloc[-1]) if len(p) else float("nan")


def realized_drawdown(prices: pd.Series, T: pd.Timestamp, D: pd.Timestamp, p_T: float) -> float:
    """Worst drawdown from T through D, with the close at T counting as a peak."""
    path = np.concatenate([[p_T], prices.loc[(prices.index > T) & (prices.index <= D)].to_numpy()])
    return float(np.min(path / np.maximum.accumulate(path) - 1))


def baseline_quantiles(hist: pd.Series, n_days: int) -> np.ndarray:
    """Empirical 10/50/90 percentiles of all past n-day log returns."""
    lp = np.log(hist.dropna().to_numpy())
    r = lp[n_days:] - lp[:-n_days]
    if len(r) < 20:
        return np.full(3, np.nan)
    return np.quantile(r, QUANTILES)


def baseline_drawdown_prob(hist: pd.Series, n_days: int, threshold: float, stride: int = 5) -> float:
    """Historical frequency of a > threshold drawdown within n trading days."""
    lp = np.log(hist.dropna().to_numpy())
    if len(lp) <= n_days + 1:
        return float("nan")
    w = sliding_window_view(lp, n_days + 1)[::stride]
    dd = (w - np.maximum.accumulate(w, axis=1)).min(axis=1)
    return float(np.mean(dd < math.log(1 - threshold)))


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

@dataclass
class BacktestResult:
    rows: pd.DataFrame                    # one row per (origin, horizon), scored or not
    skipped: list[tuple[pd.Timestamp, str]]
    config: BacktestConfig

    @property
    def scored(self) -> pd.DataFrame:
        return self.rows.dropna(subset=["y"])


def run_backtest(
    data: PointInTimeData,
    config: BacktestConfig = BacktestConfig(),
    forecaster: Callable = ensemble_forecast,
    progress: Callable[[pd.Timestamp, int, int], None] | None = None,
) -> BacktestResult:
    cfg, ec = config, config.ensemble
    prices = data.prices.dropna().sort_index()

    obs = regime_observations(prices, data.vix)
    regimes = regime_feature_table(obs, prices.index.max(), ec.hmm_refit_every, ec.hmm_min_obs,
                                   ec.hmm_n_init, ec.hmm_n_iter, ec.seed)

    rows, skipped = [], []
    ts = origins(prices, cfg.start, cfg.end)
    for i, T in enumerate(ts):
        if progress:
            progress(T, i, len(ts))
        try:
            res = forecaster(data, asof=T, config=ec, regimes=regimes)
        except Exception as exc:
            skipped.append((T, f"{type(exc).__name__}: {exc}"))
            continue
        hist = prices.loc[:T]
        p_T = float(hist.iloc[-1])
        for h, D in enumerate(target_dates(T), start=1):
            n = business_days_between(T, D)
            lab = str(pd.Period(D, freq="Q"))
            m = res.table.loc[lab]
            b = baseline_quantiles(hist, n)
            pd_close = realized_close(prices, D)
            y = math.log(pd_close / p_T) if np.isfinite(pd_close) else np.nan
            dd = realized_drawdown(prices, T, D, p_T) if np.isfinite(pd_close) else np.nan
            rows.append({
                "origin": T, "horizon": h, "target": D, "n_days": n, "close_T": p_T,
                "y": y, "realized_dd": dd,
                "dd_event": float(dd < -cfg.drawdown_threshold) if np.isfinite(dd) else np.nan,
                "m_q10": m["blend_q10"], "m_q50": m["blend_q50"], "m_q90": m["blend_q90"],
                "m_dd": m["prob_drawdown"],
                "b_q10": b[0], "b_q50": b[1], "b_q90": b[2],
                "b_dd": baseline_drawdown_prob(hist, n, cfg.drawdown_threshold),
                "w_lgb": res.weights["lgb"], "n_train": res.n_train_rows,
                **{f"regime_{k}": v for k, v in res.regime["current"].items()},
            })
    return BacktestResult(pd.DataFrame(rows), skipped, cfg)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score(rows: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Calibration / accuracy metrics for one model ('m' or 'b')."""
    r = rows.dropna(subset=["y", f"{prefix}_q50"])
    if r.empty:
        return {k: np.nan for k in ("n", "coverage", "below_p10", "above_p90", "hit_rate",
                                    "pinball", "pinball_q10", "pinball_q50", "pinball_q90",
                                    "width", "brier_dd")}
    y = r["y"].to_numpy()
    q = {lvl: r[f"{prefix}_q{int(lvl * 100)}"].to_numpy() for lvl in QUANTILES}
    pl = {lvl: pinball(y, q[lvl], lvl).mean() for lvl in QUANTILES}
    dd = r.dropna(subset=["dd_event", f"{prefix}_dd"])
    return {
        "n": float(len(r)),
        "coverage": float(np.mean((y >= q[0.1]) & (y <= q[0.9]))),
        "below_p10": float(np.mean(y < q[0.1])),
        "above_p90": float(np.mean(y > q[0.9])),
        "hit_rate": float(np.mean(np.sign(q[0.5]) == np.sign(y))),
        "pinball": float(np.mean(list(pl.values()))),
        "pinball_q10": float(pl[0.1]), "pinball_q50": float(pl[0.5]), "pinball_q90": float(pl[0.9]),
        "width": float(np.mean(q[0.9] - q[0.1])),
        "brier_dd": float(np.mean((dd[f"{prefix}_dd"] - dd["dd_event"]) ** 2)) if len(dd) else np.nan,
    }


def per_origin_loss_diff(rows: pd.DataFrame) -> pd.Series:
    """Mean (model - baseline) pinball loss per origin, averaged over horizons."""
    r = rows.dropna(subset=["y", "m_q50", "b_q50"]).copy()
    lm = sum(pinball(r["y"], r[f"m_q{int(q * 100)}"], q) for q in QUANTILES) / 3
    lb = sum(pinball(r["y"], r[f"b_q{int(q * 100)}"], q) for q in QUANTILES) / 3
    r["d"] = lm - lb
    return r.groupby("origin")["d"].mean()


def block_bootstrap_ci(x: np.ndarray, n_boot: int = 2000, block: int = 4, alpha: float = 0.10,
                       seed: int = 0) -> tuple[float, float]:
    """Circular block bootstrap CI for the mean of a dependent series."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    b = min(block, n)
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(b)) % n
    means = x[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


@dataclass
class Report:
    overall: pd.DataFrame        # rows: model/baseline, cols: metrics
    by_horizon: pd.DataFrame     # MultiIndex (horizon, model)
    stress: pd.DataFrame         # MultiIndex (window, model) + crash details
    skill: float
    diff_ci: tuple[float, float] # 90% CI of mean (model - baseline) pinball per origin
    verdict: list[str]
    n_origins: int
    skipped: int

    def to_markdown(self) -> str:
        fmt = lambda df: "```\n" + df.round(4).to_string() + "\n```"  # noqa: E731
        lines = ["# Walk-forward backtest", "",
                 f"Origins scored: {self.n_origins} (skipped: {self.skipped}).  ",
                 f"Pinball skill vs baseline: {self.skill:+.1%}  ",
                 f"90% CI of mean loss difference (model - baseline, per origin): "
                 f"[{self.diff_ci[0]:+.5f}, {self.diff_ci[1]:+.5f}]", "",
                 "## Verdict", "", *[f"- {v}" for v in self.verdict], "",
                 "## Overall", "", fmt(self.overall), "",
                 "## By horizon (quarters ahead)", "", fmt(self.by_horizon), "",
                 "## Stress periods (rows whose target falls in the window)", "", fmt(self.stress), ""]
        return "\n".join(lines)


def _two(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"model": score(rows, "m"), "baseline": score(rows, "b")}).T


def _verdict(overall: pd.DataFrame, skill: float, ci: tuple[float, float],
             stress: pd.DataFrame) -> list[str]:
    m, b = overall.loc["model"], overall.loc["baseline"]
    v = []
    if not np.isfinite(skill):
        return ["Not enough scored forecasts to judge."]
    if ci[1] < 0:
        v.append(f"The ensemble BEATS the naive baseline on pinball loss ({skill:+.1%}), and the "
                 "90% block-bootstrap CI of the loss difference excludes zero.")
    elif ci[0] > 0:
        v.append(f"The ensemble is WORSE than the naive baseline on pinball loss ({skill:+.1%}); "
                 "the 90% CI excludes zero. It does not earn its complexity.")
    else:
        v.append(f"The ensemble does NOT reliably beat the naive baseline: skill {skill:+.1%}, but the "
                 f"90% CI of the loss difference [{ci[0]:+.5f}, {ci[1]:+.5f}] includes zero.")
    cov = m["coverage"]
    if cov < 0.70:
        v.append(f"Model bands are too narrow (overconfident): {cov:.0%} inside P10-P90 vs 80% target "
                 f"({m['below_p10']:.0%} below P10, {m['above_p90']:.0%} above P90).")
    elif cov > 0.90:
        v.append(f"Model bands are too wide (underconfident): {cov:.0%} inside P10-P90 vs 80% target.")
    else:
        v.append(f"Model coverage {cov:.0%} is within 70-90% of the 80% target "
                 f"(baseline {b['coverage']:.0%}).")
    if m["below_p10"] > 1.5 * 0.10:
        v.append(f"Downside tail is under-predicted: {m['below_p10']:.0%} of outcomes fell below P10.")
    up_rate = b["hit_rate"]
    v.append(f"Directional hit rate {m['hit_rate']:.0%} vs baseline {up_rate:.0%} "
             "(the baseline almost always says 'up', so it scores the share of up periods).")
    if np.isfinite(m["brier_dd"]) and np.isfinite(b["brier_dd"]):
        better = "better" if m["brier_dd"] < b["brier_dd"] else "worse"
        v.append(f"Drawdown-probability Brier {m['brier_dd']:.3f} vs baseline {b['brier_dd']:.3f} ({better}).")
    for w in stress.index.get_level_values(0).unique():
        s = stress.loc[w]
        if "model" in s.index and s.loc["model", "n"] > 0:
            v.append(f"{w}: model coverage {s.loc['model', 'coverage']:.0%}, below P10 "
                     f"{s.loc['model', 'below_p10']:.0%}; pinball {s.loc['model', 'pinball']:.4f} vs "
                     f"baseline {s.loc['baseline', 'pinball']:.4f}.")
        else:
            v.append(f"{w}: no scored forecasts (outside the data).")
    return v


def evaluate(result: BacktestResult, windows: dict[str, tuple[str, str]] = STRESS_WINDOWS) -> Report:
    rows = result.scored
    cfg = result.config
    overall = _two(rows)
    by_h = pd.concat({h: _two(g) for h, g in rows.groupby("horizon")}, names=["horizon", "model"])

    st = {}
    for name, (a, b) in windows.items():
        g = rows[(rows["target"] >= a) & (rows["target"] <= b)]
        t = _two(g)
        t["worst_outcome"] = g["y"].min() if len(g) else np.nan
        t["worst_drawdown"] = g["realized_dd"].min() if len(g) else np.nan
        st[name] = t
    stress = (pd.concat(st, names=["window", "model"]) if st else
              pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["window", "model"])))

    skill = 1 - overall.loc["model", "pinball"] / overall.loc["baseline", "pinball"] \
        if len(rows) else float("nan")
    d = per_origin_loss_diff(rows)
    ci = block_bootstrap_ci(d.to_numpy(), cfg.n_boot, cfg.block, seed=cfg.seed)
    return Report(overall, by_h, stress, float(skill), ci, _verdict(overall, skill, ci, stress),
                  int(rows["origin"].nunique()) if len(rows) else 0, len(result.skipped))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import sys
    import time

    from forecaster.config import load_settings
    from forecaster.forecast import load_data

    ap = argparse.ArgumentParser(description="Walk-forward backtest of the ensemble")
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--start", default="2004-12-31")
    ap.add_argument("--end", default=None)
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--out", default=None, help="write rows CSV + report markdown with this prefix")
    args = ap.parse_args(argv)

    settings = load_settings()
    data = load_data(args.ticker, settings, on_warn=lambda m: print(f"[warn] {m}", file=sys.stderr))
    cfg = BacktestConfig(start=args.start, end=args.end, ensemble=EnsembleConfig(n_paths=args.paths))
    t0 = time.time()
    res = run_backtest(data, cfg, progress=lambda T, i, n: print(
        f"[{i + 1}/{n}] {T.date()}  {time.time() - t0:,.0f}s", file=sys.stderr))
    rep = evaluate(res)
    md = rep.to_markdown()
    print(md)
    if args.out:
        res.rows.to_csv(f"{args.out}_rows.csv", index=False)
        with open(f"{args.out}_report.md", "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
