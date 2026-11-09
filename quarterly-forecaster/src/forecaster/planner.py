"""Contribution planner: split a monthly CAD contribution toward a target allocation.

Buy-only rebalancing. The contribution is split so the portfolio after buying is as
close as possible to target (least squares, no negative buys): water-filling on each
holding's gap to target. Nothing is ever sold, borrowed or leveraged, and suggested
buys always sum exactly to the contribution. Leveraged, inverse and option
instruments are refused as targets.

Risk flags suggest *splitting the buy across weeks*; they never suggest skipping,
timing with derivatives, or borrowing. Flags (all thresholds in PlannerConfig):
  * elevated drawdown risk: next-quarter P(>10% drawdown) for the portfolio or a
    target holding at or above `drawdown_flag`, or the high-volatility regime at
    or above `high_vol_flag`. The backtest found these probabilities overstated at
    the high end, so treat a flag as "consider", not "wait".
  * high VIX: ^VIX snapshot close at or above `vix_flag`.
  * high event-risk week: high-impact event points in the next 7 days at or above
    `event_points_flag` (CPI 3, jobs 3, FOMC 3, +1 with projections).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Known leveraged / inverse ETFs (US + Canada). Not exhaustive; name patterns catch the rest.
LEVERAGED_OR_INVERSE: frozenset[str] = frozenset({
    "TQQQ", "SQQQ", "QLD", "QID", "SPXL", "SPXS", "SPXU", "UPRO", "SSO", "SDS", "SH", "PSQ", "SOXL", "SOXS",
    "TNA", "TZA", "UDOW", "SDOW", "LABU", "LABD", "FAS", "FAZ", "TMF", "TMV", "UVXY", "SVXY", "VXX", "VIXY",
    "NUGT", "DUST", "JNUG", "JDST", "BOIL", "KOLD", "UCO", "SCO", "TSLL", "NVDL",
    "HXU.TO", "HXD.TO", "HQU.TO", "HQD.TO", "HSU.TO", "HSD.TO", "HND.TO", "HNU.TO", "HOU.TO", "HOD.TO",
    "HGU.TO", "HGD.TO", "HUV.TO", "HIU.TO", "HFU.TO", "HFD.TO", "HED.TO", "HEU.TO", "HBU.TO", "HBD.TO",
    "HZU.TO", "HZD.TO", "QQU.TO", "QQD.TO", "SPXU.TO",
})
# Unambiguous leverage/inverse markers only: plain "short"/"ultra" also appear in ordinary
# bond funds ("Short Term Bond", "Ultra-Short Income"), which must stay allowed.
_LEVERAGE_WORDS = re.compile(
    r"\b\d(\.\d+)?x\b|\bleveraged\b|\binverse\b|\bdaily\s+(bull|bear)\b|\b(bull|bear)\s+\d|"
    r"\bproshares\s+ultra|\bultrapro\b|\bbetapro\b|\bdirexion\s+daily\b", re.I)
_OPTION_SYMBOL = re.compile(r"^[A-Z.]{1,6}\d{6}[CP]\d{8}$")           # OCC: SPY241220C00500000
_FUTURES_OR_FX = re.compile(r"(=F|=X)$|^\^")


class PlannerError(ValueError):
    pass


@dataclass(frozen=True)
class PlannerConfig:
    drawdown_flag: float = 0.35
    high_vol_flag: float = 0.50
    vix_flag: float = 25.0
    event_points_flag: int = 6
    event_window_days: int = 7
    tranches: int = 3            # suggested number of weekly buys when flagged
    max_drift_ok: float = 0.01   # "on target" when every weight is within 1 point


def check_instrument(ticker: str, name: str | None = None) -> None:
    """Refuse leveraged, inverse, option, futures, FX and index targets."""
    t = ticker.strip().upper()
    if _OPTION_SYMBOL.match(t.replace(" ", "")):
        raise PlannerError(f"{t} looks like an option contract: the planner never suggests options")
    if _FUTURES_OR_FX.search(t):
        raise PlannerError(f"{t} is a future, FX rate or index, not something to buy with a contribution")
    if t in LEVERAGED_OR_INVERSE:
        raise PlannerError(f"{t} is a leveraged or inverse ETF: the planner never suggests leverage")
    if name and _LEVERAGE_WORDS.search(name):
        raise PlannerError(f"{t} ({name}) looks leveraged or inverse: the planner never suggests leverage")


def normalise_targets(targets: dict[str, float]) -> dict[str, float]:
    """Accept weights as fractions or percents; must be >= 0 and sum to 100% (+-0.5%)."""
    if not targets:
        raise PlannerError("enter at least one target")
    t = {k.strip().upper(): float(v) for k, v in targets.items() if k and str(k).strip()}
    if any(v < 0 or not math.isfinite(v) for v in t.values()):
        raise PlannerError("target weights must be >= 0")
    total = sum(t.values())
    if 99.5 <= total <= 100.5:                      # given in percent
        t = {k: v / 100 for k, v in t.items()}
        total = sum(t.values())
    if not (0.995 <= total <= 1.005):
        raise PlannerError(f"target weights must add up to 100% (they add up to {total:.1%})")
    return {k: v / total for k, v in t.items() if v > 0}


def waterfill(gaps: np.ndarray, amount: float) -> np.ndarray:
    """x_i = max(gap_i - lam, 0) with sum(x) = amount: least-squares buy-only split."""
    if amount <= 0:
        return np.zeros_like(gaps)
    lo, hi = float(gaps.min()) - amount - 1.0, float(gaps.max())
    for _ in range(200):
        lam = (lo + hi) / 2
        s = np.maximum(gaps - lam, 0).sum()
        if s > amount:
            lo = lam
        else:
            hi = lam
    x = np.maximum(gaps - hi, 0)
    return x * (amount / x.sum()) if x.sum() > 0 else x


@dataclass
class Plan:
    rows: pd.DataFrame               # per ticker
    amount: float
    leftover: float                  # cash not deployed (whole-share rounding)
    outside_targets: list[str]       # holdings with no target (never sold, just reported)
    max_drift_before: float
    max_drift_after: float
    months_to_target: int | None
    notes: list[str] = field(default_factory=list)


def plan_contribution(current: dict[str, float], targets: dict[str, float], amount: float,
                      prices: dict[str, float] | None = None, whole_shares: bool = False,
                      names: dict[str, str] | None = None, cfg: PlannerConfig = PlannerConfig()) -> Plan:
    """current: CAD value per ticker (all accounts). targets: ticker -> weight."""
    if not (amount > 0 and math.isfinite(amount)):
        raise PlannerError("monthly amount must be a positive number of CAD")
    tg = normalise_targets(targets)
    for t in tg:
        check_instrument(t, (names or {}).get(t))
    prices = prices or {}
    tickers = list(tg)
    V = np.array([max(float(current.get(t, 0.0)), 0.0) for t in tickers])
    w = np.array([tg[t] for t in tickers])
    total_after = V.sum() + amount
    gaps = w * total_after - V
    x = waterfill(gaps, amount)

    leftover = 0.0
    shares = [None] * len(tickers)
    if whole_shares:
        px = np.array([prices.get(t, np.nan) for t in tickers], dtype=float)
        if np.isnan(px).any():
            missing = [t for t, p in zip(tickers, px) if not np.isfinite(p)]
            raise PlannerError(f"whole shares need a price for: {', '.join(missing)}")
        n = np.floor(x / px + 1e-9)
        spent = n * px
        left = amount - spent.sum()
        # greedily add a share to whatever is furthest below target, while affordable
        while True:
            after = V + n * px
            deficit = w * total_after - after
            order = [i for i in np.argsort(-deficit) if px[i] <= left + 1e-9]
            if not order:
                break
            i = order[0]
            n[i] += 1
            left -= px[i]
        x, leftover = n * px, float(left)
        shares = list(n.astype(int))
    elif prices:
        shares = [x[i] / prices[t] if prices.get(t) else None for i, t in enumerate(tickers)]

    after = V + x
    invested_after = after.sum()
    wb = V / V.sum() if V.sum() > 0 else np.zeros_like(V)
    wa = after / invested_after if invested_after > 0 else np.zeros_like(after)
    rows = pd.DataFrame({
        "ticker": tickers, "current_value": V, "current_weight": wb, "target_weight": w,
        "buy": x, "shares": shares, "price": [prices.get(t) for t in tickers],
        "value_after": after, "weight_after": wa, "drift_before": wb - w, "drift_after": wa - w,
    })
    outside = sorted(t for t, v in current.items() if t not in tg and v > 0)
    notes = []
    if outside:
        notes.append(f"Held but not in your targets: {', '.join(outside)}. The planner never sells them; "
                     "they're left out of the weights above.")
    return Plan(rows, float(amount), leftover, outside, float(np.abs(wb - w).max()),
                float(np.abs(wa - w).max()), months_to_target(V, w, amount, cfg.max_drift_ok), notes)


def months_to_target(V: np.ndarray, w: np.ndarray, amount: float, tol: float, cap: int = 120) -> int | None:
    """Months of contributions (prices unchanged) until every weight is within `tol` of target."""
    v = V.astype(float).copy()
    for m in range(0, cap + 1):
        if v.sum() > 0 and np.abs(v / v.sum() - w).max() <= tol:
            return m
        v = v + waterfill(w * (v.sum() + amount) - v, amount)
    return None


# --------------------------------------------------------------------------- #
# Risk flags and tranche schedule
# --------------------------------------------------------------------------- #

def risk_flags(targets: list[str], ticker_snaps: dict[str, dict], portfolio: dict | None,
               market: dict | None, today: pd.Timestamp, cfg: PlannerConfig = PlannerConfig()) -> list[dict]:
    flags = []
    if portfolio and portfolio.get("forecast"):
        p = float(portfolio["forecast"][0].get("prob_drawdown") or 0)
        if p >= cfg.drawdown_flag:
            flags.append({"kind": "drawdown", "message":
                          f"Portfolio: {p:.0%} chance of a >10% drawdown by {portfolio['forecast'][0]['quarter']} "
                          f"(flag at {cfg.drawdown_flag:.0%})."})
    for t in targets:
        s = ticker_snaps.get(t)
        if not s or not s.get("forecast"):
            continue
        p = float(s["forecast"][0].get("prob_drawdown") or 0)
        if p >= cfg.drawdown_flag:
            flags.append({"kind": "drawdown", "message":
                          f"{t}: {p:.0%} chance of a >10% drawdown by {s['forecast'][0]['quarter']}."})
        hv = float((s.get("regime") or {}).get("current", {}).get("high_vol") or 0)
        if hv >= cfg.high_vol_flag:
            flags.append({"kind": "regime", "message": f"{t}: high-volatility regime probability {hv:.0%}."})
    vix = ticker_snaps.get("^VIX")
    if vix and vix.get("last_close") is not None and float(vix["last_close"]) >= cfg.vix_flag:
        flags.append({"kind": "vix", "message": f"VIX {float(vix['last_close']):.1f} (flag at {cfg.vix_flag:.0f})."})
    evs = upcoming_events(market, today, cfg.event_window_days)
    pts = sum(int(e.get("points") or 0) for e in evs)
    if pts >= cfg.event_points_flag:
        names = ", ".join(f"{e['name']} {pd.Timestamp(e['date']):%a %b %d}" for e in evs)
        flags.append({"kind": "events", "message":
                      f"High event-risk week: {names} ({pts} risk points; flag at {cfg.event_points_flag})."})
    return flags


def upcoming_events(market: dict | None, today: pd.Timestamp, days: int) -> list[dict]:
    t0 = pd.Timestamp(today).normalize()
    return sorted([e for e in (market or {}).get("events", [])
                   if e.get("impact") == "high" and t0 <= pd.Timestamp(e["date"]).normalize() <= t0 + pd.Timedelta(days=days)],
                  key=lambda e: e["date"])


def tranche_schedule(amount: float, today: pd.Timestamp, market: dict | None, n: int = 3) -> list[dict]:
    """Split `amount` into n equal weekly buys on business days with no high-impact event."""
    busy = {pd.Timestamp(e["date"]).normalize() for e in (market or {}).get("events", []) if e.get("impact") == "high"}
    out, d = [], pd.Timestamp(today).normalize() + pd.offsets.BDay(1)
    cents = round(amount * 100)
    per = [cents // n + (1 if i < cents % n else 0) for i in range(n)]
    for i in range(n):
        start = d + pd.Timedelta(weeks=i)
        day = next((start + pd.offsets.BDay(k) for k in range(5)
                    if (start + pd.offsets.BDay(k)).normalize() not in busy), start)  # all busy: keep Monday
        out.append({"date": day.normalize(), "amount": per[i] / 100})
    return out
