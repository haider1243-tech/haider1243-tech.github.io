"""Portfolio: holdings from portfolio.yaml -> CAD valuation, P&L, allocation, risk, forecast.

portfolio.yaml is personal data and gitignored; copy portfolio.example.yaml to start.

Valuation
    Market value in CAD: shares x latest close, with USD prices converted at the
    latest USD/CAD (market_data.to_cad). P&L in CAD:
      * cost_currency CAD (your adjusted cost base): full CAD P&L.
      * cost_currency USD with cost_fx (USD/CAD at purchase): full CAD P&L.
      * cost_currency USD without cost_fx: cost converted at today's rate, so the
        P&L covers the price move only, not FX gains or losses since purchase.
        The holding is labelled accordingly.
Allocation
    By asset class, region, account and sector. ETF sectors are looked through
    (e.g. XEQT -> its sector weights) when the weights are known (yfinance fund
    data, a built-in map, or a `sectors:` override); the rest is "Unclassified".
Risk (current weights, daily CAD returns, prices aligned on common dates first)
    Volatility = annualised st. dev. of the portfolio's daily return. Beta vs
    the S&P/TSX 60 (XIU.TO) and the S&P 500 in CAD (SPY converted). Per holding:
    volatility, beta, share of portfolio variance. Cash has zero volatility.
Forecast
    10,000 buy-and-hold paths to the next 4 quarter-ends, built by resampling
    blocks of 20 consecutive *joint* daily returns of all holdings from their
    common history. Correlations, fat tails and volatility clustering come from
    real history rather than an assumed model. Outputs P10/P50/P90 of portfolio
    value, P(loss) and P(>10% drawdown). Caveats: the drift is whatever the
    history window delivered (a strong decade gives an optimistic P50), and this
    method has not been backtested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from forecaster.config import PROJECT_ROOT
from forecaster.market_data import BASE_CURRENCY, rate_on_dates, ticker_currency, to_cad

ACCOUNT_TYPES = ("TFSA", "FHSA", "RRSP", "cash")       # cash = non-registered
CONCENTRATION_LIMIT = 0.25
BENCHMARKS = {"XIU.TO": "S&P/TSX 60 (XIU)", "SPY": "S&P 500 (SPY, in CAD)"}
PORTFOLIO_FILE = PROJECT_ROOT / "portfolio.yaml"
TRADING_DAYS = 252

# Built-in metadata; overridable per holding in portfolio.yaml.
KNOWN_ASSETS: dict[str, dict[str, Any]] = {
    "XIU.TO": {"name": "iShares S&P/TSX 60", "asset_class": "Equity", "region": "Canada", "diversified": True},
    "XEQT.TO": {"name": "iShares Core Equity ETF Portfolio", "asset_class": "Equity", "region": "Global",
                "diversified": True},
    "VFV.TO": {"name": "Vanguard S&P 500 Index ETF", "asset_class": "Equity", "region": "United States",
               "diversified": True},
    "ZAG.TO": {"name": "BMO Aggregate Bond Index ETF", "asset_class": "Fixed income", "region": "Canada",
               "diversified": True, "sectors": {"Fixed income": 1.0}},
    "SPY": {"name": "SPDR S&P 500", "asset_class": "Equity", "region": "United States", "diversified": True},
    "QQQ": {"name": "Invesco QQQ", "asset_class": "Equity", "region": "United States", "diversified": True},
    "XLK": {"name": "Technology Select Sector", "asset_class": "Equity", "region": "United States",
            "sectors": {"Technology": 1.0}},
    "XLF": {"name": "Financial Select Sector", "asset_class": "Equity", "region": "United States",
            "sectors": {"Financials": 1.0}},
    "XLE": {"name": "Energy Select Sector", "asset_class": "Equity", "region": "United States",
            "sectors": {"Energy": 1.0}},
    "XLV": {"name": "Health Care Select Sector", "asset_class": "Equity", "region": "United States",
            "sectors": {"Health Care": 1.0}},
    "XLY": {"name": "Consumer Discretionary Select Sector", "asset_class": "Equity", "region": "United States",
            "sectors": {"Consumer Cyclical": 1.0}},
    "GLD": {"name": "SPDR Gold Shares", "asset_class": "Commodity", "region": "Global", "sectors": {"Gold": 1.0}},
    "GC=F": {"name": "Gold futures", "asset_class": "Commodity", "region": "Global", "sectors": {"Gold": 1.0}},
}
SECTOR_NAMES = {
    "realestate": "Real Estate", "real_estate": "Real Estate", "consumer_cyclical": "Consumer Cyclical",
    "basic_materials": "Basic Materials", "consumer_defensive": "Consumer Defensive", "technology": "Technology",
    "communication_services": "Communication Services", "financial_services": "Financials",
    "utilities": "Utilities", "industrials": "Industrials", "energy": "Energy", "healthcare": "Health Care",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Holding:
    ticker: str
    shares: float
    avg_cost: float
    account: str
    cost_currency: str
    cost_fx: float | None = None
    asset_class: str | None = None
    region: str | None = None
    sectors: dict[str, float] | None = None
    name: str | None = None


@dataclass(frozen=True)
class CashBalance:
    account: str
    amount: float
    currency: str = "CAD"


@dataclass(frozen=True)
class Portfolio:
    holdings: tuple[Holding, ...]
    cash: tuple[CashBalance, ...] = ()
    concentration_limit: float = CONCENTRATION_LIMIT
    planner: dict = field(default_factory=dict)   # {"monthly_amount": float, "targets": {ticker: weight}}

    @property
    def tickers(self) -> list[str]:
        return list(dict.fromkeys(h.ticker for h in self.holdings))


class PortfolioError(ValueError):
    pass


def _account(v: Any, where: str) -> str:
    s = str(v or "").strip()
    match = {a.lower(): a for a in ACCOUNT_TYPES}.get(s.lower())
    if match is None:
        raise PortfolioError(f"{where}: account must be one of {', '.join(ACCOUNT_TYPES)} (got {v!r})")
    return match


def parse_portfolio(data: dict) -> Portfolio:
    if not isinstance(data, dict) or not isinstance(data.get("holdings"), list):
        raise PortfolioError("portfolio.yaml needs a 'holdings:' list")
    holdings = []
    for i, h in enumerate(data["holdings"], start=1):
        where = f"holding #{i}"
        if not isinstance(h, dict):
            raise PortfolioError(f"{where}: expected a mapping")
        t = str(h.get("ticker") or "").strip().upper()
        if not t:
            raise PortfolioError(f"{where}: missing ticker")
        where = f"holding #{i} ({t})"
        cur = ticker_currency(t)
        if cur is None:
            raise PortfolioError(f"{where}: {t} is an index or yield, not something you can hold")
        try:
            shares, cost = float(h["shares"]), float(h["avg_cost"])
        except (KeyError, TypeError, ValueError):
            raise PortfolioError(f"{where}: needs numeric 'shares' and 'avg_cost'") from None
        if not shares > 0 or not cost >= 0 or not (math.isfinite(shares) and math.isfinite(cost)):
            raise PortfolioError(f"{where}: shares must be > 0 and avg_cost >= 0")
        cost_cur = str(h.get("cost_currency") or cur).upper()
        if cost_cur not in ("CAD", "USD"):
            raise PortfolioError(f"{where}: cost_currency must be CAD or USD")
        fx = h.get("cost_fx")
        fx = None if fx is None else float(fx)
        if fx is not None and not (1.0 <= fx <= 2.0):
            raise PortfolioError(f"{where}: cost_fx is USD/CAD (CAD per USD, ~1.2-1.5), got {fx}")
        sectors = h.get("sectors")
        if sectors is not None:
            if not isinstance(sectors, dict) or sum(float(v) for v in sectors.values()) > 1.0001:
                raise PortfolioError(f"{where}: sectors must map names to weights summing to <= 1")
            sectors = {str(k): float(v) for k, v in sectors.items()}
        holdings.append(Holding(t, shares, cost, _account(h.get("account"), where), cost_cur, fx,
                                h.get("asset_class"), h.get("region"), sectors, h.get("name")))
    cash = []
    for i, c in enumerate(data.get("cash") or [], start=1):
        where = f"cash #{i}"
        try:
            amt = float(c["amount"])
        except (KeyError, TypeError, ValueError):
            raise PortfolioError(f"{where}: needs a numeric 'amount'") from None
        ccy = str(c.get("currency") or "CAD").upper()
        if ccy not in ("CAD", "USD"):
            raise PortfolioError(f"{where}: currency must be CAD or USD")
        cash.append(CashBalance(_account(c.get("account"), where), amt, ccy))
    limit = float(data.get("concentration_limit", CONCENTRATION_LIMIT))
    if not (0 < limit <= 1):
        raise PortfolioError("concentration_limit must be between 0 and 1")
    if not holdings and not cash:
        raise PortfolioError("portfolio is empty")
    planner = data.get("planner") or {}
    if not isinstance(planner, dict) or not isinstance(planner.get("targets", {}), dict):
        raise PortfolioError("planner: needs 'monthly_amount' and a 'targets' mapping")
    planner = {"monthly_amount": float(planner.get("monthly_amount") or 0) or None,
               "targets": {str(k).upper(): float(v) for k, v in (planner.get("targets") or {}).items()}}
    return Portfolio(tuple(holdings), tuple(cash), limit, planner)


def load_portfolio(path: Path = PORTFOLIO_FILE) -> Portfolio:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: copy portfolio.example.yaml to portfolio.yaml")
    return parse_portfolio(yaml.safe_load(Path(path).read_text()) or {})


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def classify(h: Holding, fetched: dict | None = None) -> dict:
    """asset_class / region / sectors / name / diversified, from overrides >
    built-in map > fetched metadata (yfinance) > 'Unclassified'."""
    known = KNOWN_ASSETS.get(h.ticker, {})
    f = fetched or {}
    sectors = h.sectors or known.get("sectors") or f.get("sectors") or {}
    asset_class = h.asset_class or known.get("asset_class") or f.get("asset_class") or "Unclassified"
    if not sectors and asset_class == "Fixed income":
        sectors = {"Fixed income": 1.0}
    return {
        "name": h.name or known.get("name") or f.get("name") or h.ticker,
        "asset_class": asset_class,
        "region": h.region or known.get("region") or f.get("region") or "Unclassified",
        "sectors": sectors,
        "diversified": bool(known.get("diversified") or f.get("diversified")),
    }


def yf_metadata(ticker: str) -> dict:  # pragma: no cover - network
    """Best-effort metadata from yfinance: name, asset class, fund sector weights."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    info = t.info or {}
    out: dict[str, Any] = {"name": info.get("longName") or info.get("shortName")}
    qt = (info.get("quoteType") or "").upper()
    text = " ".join(str(info.get(k) or "") for k in ("category", "longName", "longBusinessSummary")).lower()
    if any(w in text for w in ("bond", "fixed income", "treasury", "aggregate")):
        out["asset_class"] = "Fixed income"
    elif any(w in text for w in ("gold", "commodit", "silver")):
        out["asset_class"] = "Commodity"
    elif qt in ("ETF", "EQUITY", "MUTUALFUND"):
        out["asset_class"] = "Equity"
    if qt == "EQUITY" and info.get("sector"):
        out["sectors"] = {info["sector"]: 1.0}
    if qt in ("ETF", "MUTUALFUND"):
        out["diversified"] = True
        try:
            w = t.funds_data.sector_weightings or {}
            out["sectors"] = {SECTOR_NAMES.get(k, k.replace("_", " ").title()): float(v) for k, v in w.items() if v}
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Valuation, allocation, flags
# --------------------------------------------------------------------------- #

def valuation(pf: Portfolio, native_last: dict[str, float], usd_cad: float) -> pd.DataFrame:
    """One row per holding (and cash line). Values in CAD."""
    rows = []
    for h in pf.holdings:
        cur = ticker_currency(h.ticker)
        px = float(native_last[h.ticker])
        fx_now = usd_cad if cur == "USD" else 1.0
        mv = h.shares * px * fx_now
        note = ""
        if h.cost_currency == "CAD":
            cost = h.shares * h.avg_cost
        elif h.cost_fx is not None:
            cost = h.shares * h.avg_cost * h.cost_fx
        else:
            cost = h.shares * h.avg_cost * usd_cad
            note = "USD cost converted at today's rate: excludes FX gain/loss since purchase (add cost_fx)"
        rows.append({"ticker": h.ticker, "account": h.account, "shares": h.shares, "currency": cur,
                     "price": px, "price_cad": px * fx_now, "avg_cost": h.avg_cost,
                     "cost_currency": h.cost_currency, "market_value": mv, "cost": cost,
                     "pnl": mv - cost, "pnl_pct": (mv / cost - 1) if cost else float("nan"), "note": note})
    for c in pf.cash:
        v = c.amount * (usd_cad if c.currency == "USD" else 1.0)
        rows.append({"ticker": "CASH", "account": c.account, "shares": float("nan"), "currency": c.currency,
                     "price": float("nan"), "price_cad": float("nan"), "avg_cost": float("nan"),
                     "cost_currency": c.currency, "market_value": v, "cost": v, "pnl": 0.0, "pnl_pct": 0.0,
                     "note": f"{c.currency} cash" + (" converted at today's rate" if c.currency == "USD" else "")})
    df = pd.DataFrame(rows)
    total = df["market_value"].sum()
    df["weight"] = df["market_value"] / total if total > 0 else 0.0
    return df


def allocation(val: pd.DataFrame, meta: dict[str, dict]) -> dict[str, pd.Series]:
    """Weights by asset class, region, account and (look-through) sector. Each sums to 1."""
    w = val.groupby("ticker")["weight"].sum()
    ac, rg, sec = {}, {}, {}
    for t, wt in w.items():
        if t == "CASH":
            for d in (ac, rg, sec):
                d["Cash"] = d.get("Cash", 0) + wt
            continue
        m = meta[t]
        ac[m["asset_class"]] = ac.get(m["asset_class"], 0) + wt
        rg[m["region"]] = rg.get(m["region"], 0) + wt
        covered = 0.0
        for s, sw in (m.get("sectors") or {}).items():
            sec[s] = sec.get(s, 0) + wt * sw
            covered += sw
        if covered < 0.999:
            sec["Unclassified"] = sec.get("Unclassified", 0) + wt * (1 - covered)
    by_account = val.groupby("account")["weight"].sum()
    srt = lambda d: pd.Series(d).sort_values(ascending=False)  # noqa: E731
    return {"asset_class": srt(ac), "region": srt(rg), "sector": srt(sec),
            "account": by_account.sort_values(ascending=False)}


def concentration_flags(val: pd.DataFrame, meta: dict[str, dict], limit: float) -> list[dict]:
    """Holdings (summed across accounts) above `limit` of total portfolio value."""
    w = val[val["ticker"] != "CASH"].groupby("ticker")["weight"].sum()
    out = []
    for t, wt in w.sort_values(ascending=False).items():
        if wt > limit:
            m = meta.get(t, {})
            note = (f"{t} is itself a diversified fund, so this is less risky than {wt:.0%} in one company."
                    if m.get("diversified") else "")
            out.append({"ticker": t, "weight": float(wt), "limit": limit,
                        "message": f"{t} is {wt:.0%} of the portfolio (limit {limit:.0%}). {note}".strip()})
    return out


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #

def aligned_prices(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """CAD closes on dates where every series has a value (TSX/NYSE holidays differ)."""
    df = pd.concat({t: s for t, s in closes.items()}, axis=1, join="inner").dropna()
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def _clean(x: float, nd: int = 6) -> float:
    """Round and drop negative zero (a -0.0000001 beta otherwise prints as "-0.00")."""
    return float(round(float(x), nd)) + 0.0


def risk_metrics(closes: dict[str, pd.Series], weights: pd.Series, bench: dict[str, pd.Series],
                 lookback: int = 3 * TRADING_DAYS) -> dict:
    """Portfolio volatility and beta, per-holding risk. `weights` include CASH (zero vol)."""
    tickers = [t for t in weights.index if t != "CASH"]
    px = aligned_prices({**{t: closes[t] for t in tickers}, **{f"__b_{k}": v for k, v in bench.items()}})
    rets = px.pct_change().dropna().tail(lookback)
    if len(rets) < 60:
        raise ValueError(f"only {len(rets)} common daily returns; need 60+ for risk metrics")
    R = rets[tickers]
    w = weights[tickers].to_numpy()
    port = R.to_numpy() @ w                    # cash contributes 0 return
    cov = R.cov().to_numpy() * TRADING_DAYS
    var_p = float(w @ cov @ w)
    vol_p = math.sqrt(max(var_p, 0.0))
    contrib = w * (cov @ w) / var_p if var_p > 0 else np.zeros_like(w)
    out = {"n_days": int(len(rets)), "start": rets.index[0], "end": rets.index[-1],
           "volatility": vol_p, "beta": {}, "holdings": []}
    for k in bench:
        b = rets[f"__b_{k}"].to_numpy()
        vb = b.var(ddof=1)
        out["beta"][k] = _clean(np.cov(port, b, ddof=1)[0, 1] / vb) if vb > 0 else float("nan")
    for i, t in enumerate(tickers):
        r = R[t].to_numpy()
        row = {"ticker": t, "weight": float(w[i]), "volatility": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)),
               "risk_share": float(contrib[i])}
        for k in bench:
            b = rets[f"__b_{k}"].to_numpy()
            row[f"beta_{k}"] = _clean(np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1))
        out["holdings"].append(row)
    out["correlation"] = R.corr()
    return out


# --------------------------------------------------------------------------- #
# Forecast: joint block bootstrap
# --------------------------------------------------------------------------- #

def simulate_portfolio(closes: dict[str, pd.Series], shares: dict[str, float], cash_cad: float,
                       asof: pd.Timestamp, n_paths: int = 10_000, block: int = 20,
                       history_years: float = 10.0, seed: int = 42,
                       drawdown_threshold: float = 0.10) -> dict:
    """P10/P50/P90 portfolio value (CAD) at the next 4 quarter-ends from correlated paths."""
    from forecaster.forecast import business_days_between, target_dates

    tickers = [t for t in shares if shares[t] > 0]
    labels = [str(pd.Period(d, freq="Q")) for d in target_dates(asof)]
    steps = [business_days_between(asof, d) for d in target_dates(asof)]
    if not tickers:  # cash only
        v = float(cash_cad)
        return {"labels": labels, "target_dates": target_dates(asof), "steps": steps, "start_value": v,
                "p10": [v] * 4, "p50": [v] * 4, "p90": [v] * 4, "prob_loss": [0.0] * 4,
                "prob_drawdown": [0.0] * 4, "history_start": None, "n_days": 0, "n_paths": n_paths}
    px = aligned_prices({t: closes[t] for t in tickers})
    px = px.loc[px.index >= px.index[-1] - pd.DateOffset(years=history_years)]
    R = np.log(px / px.shift(1)).dropna().to_numpy()            # T x N joint daily log returns
    T, N = R.shape
    if T < 2 * block:
        raise ValueError(f"only {T} common daily returns; need {2 * block}+ to bootstrap")
    v0 = np.array([shares[t] * float(px[t].iloc[-1]) for t in tickers])
    start_value = float(v0.sum() + cash_cad)

    rng = np.random.default_rng(seed)
    cum = np.zeros((n_paths, N))
    start = rng.integers(0, T, n_paths)
    offset = np.zeros(n_paths, dtype=int)
    peak = np.full(n_paths, start_value)
    worst = np.zeros(n_paths)
    targets = {s: i for i, s in enumerate(steps)}
    vals = np.empty((n_paths, len(steps)))
    dd = np.empty(len(steps))
    for step in range(1, max(steps) + 1):
        renew = offset >= block
        if renew.any():
            start[renew] = rng.integers(0, T, int(renew.sum()))
            offset[renew] = 0
        cum += R[(start + offset) % T]
        offset += 1
        value = (v0 * np.exp(cum)).sum(axis=1) + cash_cad
        peak = np.maximum(peak, value)
        worst = np.minimum(worst, value / peak - 1)
        if step in targets:
            i = targets[step]
            vals[:, i] = value
            dd[i] = float(np.mean(worst < -drawdown_threshold))
    q = np.quantile(vals, [0.1, 0.5, 0.9], axis=0)
    return {"labels": labels, "target_dates": target_dates(asof), "steps": steps, "start_value": start_value,
            "p10": q[0].tolist(), "p50": q[1].tolist(), "p90": q[2].tolist(),
            "prob_loss": np.mean(vals < start_value, axis=0).tolist(), "prob_drawdown": dd.tolist(),
            "history_start": px.index[0], "n_days": int(T), "n_paths": n_paths}


def value_history(closes: dict[str, pd.Series], shares: dict[str, float], cash_cad: float,
                  years: int = 2) -> pd.Series:
    """Today's holdings valued over the past `years` (what they would have been worth)."""
    tickers = [t for t in shares if shares[t] > 0]
    if not tickers:
        return pd.Series(dtype=float)
    px = aligned_prices({t: closes[t] for t in tickers})
    px = px.loc[px.index >= px.index[-1] - pd.DateOffset(years=years)]
    v = sum(px[t] * shares[t] for t in tickers) + cash_cad
    return v.resample("W-FRI").last().dropna()


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #

def build_snapshot(pf: Portfolio, native: dict[str, pd.Series], usd_cad: pd.Series,
                   bench_native: dict[str, pd.Series], fetched_meta: dict[str, dict] | None = None,
                   asof: pd.Timestamp | None = None, n_paths: int = 10_000, seed: int = 42) -> dict:
    """Everything the Portfolio page shows. `native` = daily closes in trading currency."""
    fetched_meta = fetched_meta or {}
    cad = {t: to_cad(s.to_frame("Close"), t, usd_cad)["Close"] for t, s in native.items()}
    bench = {k: to_cad(s.to_frame("Close"), k, usd_cad)["Close"] for k, s in bench_native.items()}
    last_dates = [s.index[-1] for s in native.values()]
    asof = pd.Timestamp(asof or max(last_dates)).normalize()
    fx_now = float(rate_on_dates(usd_cad, pd.DatetimeIndex([asof])).iloc[0])
    native_last = {t: float(s.loc[:asof].iloc[-1]) for t, s in native.items()}
    val = valuation(pf, native_last, fx_now)
    meta = {h.ticker: classify(h, fetched_meta.get(h.ticker)) for h in pf.holdings}
    alloc = allocation(val, meta)
    flags = concentration_flags(val, meta, pf.concentration_limit)
    weights = val.groupby("ticker")["weight"].sum()
    shares = {t: float(sum(h.shares for h in pf.holdings if h.ticker == t)) for t in pf.tickers}
    cash_cad = float(val.loc[val["ticker"] == "CASH", "market_value"].sum())
    warnings = []
    try:
        risk = risk_metrics(cad, weights, bench)
    except ValueError as exc:
        risk, warnings = None, warnings + [f"risk metrics unavailable: {exc}"]
    try:
        sim = simulate_portfolio(cad, shares, cash_cad, asof, n_paths=n_paths, seed=seed)
    except ValueError as exc:
        sim, warnings = None, warnings + [f"forecast unavailable: {exc}"]
    hist = value_history(cad, shares, cash_cad)
    stale = [t for t, d in zip(native, last_dates) if (asof - pd.Timestamp(d)).days > 5]
    if stale:
        warnings.append(f"prices older than 5 days for: {', '.join(stale)}")
    no_sector = [t for t, m in meta.items() if not m["sectors"]]
    if no_sector:
        warnings.append(f"no sector data for {', '.join(no_sector)}: shown as Unclassified "
                        "(add `sectors:` in portfolio.yaml)")

    total = float(val["market_value"].sum())
    cost = float(val["cost"].sum())
    snap = {
        "kind": "portfolio", "currency": BASE_CURRENCY, "as_of": asof, "usd_cad": fx_now,
        "total_value": total, "total_cost": cost, "total_pnl": total - cost,
        "total_pnl_pct": (total / cost - 1) if cost else None,
        "holdings": val.assign(**{"class": val["ticker"].map(lambda t: meta.get(t, {}).get("asset_class", "Cash")),
                                  "name": val["ticker"].map(lambda t: meta.get(t, {}).get("name", "Cash"))}
                               ).to_dict("records"),
        "allocation": {k: [{"label": i, "weight": float(v)} for i, v in s.items()] for k, s in alloc.items()},
        "flags": flags,
        "risk": None if risk is None else {
            "volatility": risk["volatility"], "beta": risk["beta"], "benchmarks": BENCHMARKS,
            "n_days": risk["n_days"], "start": risk["start"], "end": risk["end"],
            "holdings": risk["holdings"],
            "correlation": {"tickers": list(risk["correlation"].columns),
                            "matrix": risk["correlation"].round(4).to_numpy().tolist()}},
        "planner": pf.planner,
        "forecast_method": ("joint block bootstrap of daily CAD returns (20-day blocks, "
                            f"{n_paths:,} buy-and-hold paths)"),
        "value_label": "Value", "history_label": "Weekly value",
        "warnings": warnings,
        # fan-chart compatible fields (dashboard.fan_chart)
        "last_close": total,
        "history": [{"date": d, "close": float(v)} for d, v in hist.items()],
        "forecast": [] if sim is None else [
            {"quarter": sim["labels"][i], "target_date": sim["target_dates"][i], "p10": sim["p10"][i],
             "p50": sim["p50"][i], "p90": sim["p90"][i], "prob_drawdown": sim["prob_drawdown"][i],
             "prob_loss": sim["prob_loss"][i]} for i in range(4)],
        "simulation": None if sim is None else {"history_start": sim["history_start"], "n_days": sim["n_days"],
                                                "n_paths": sim["n_paths"]},
    }
    return snap


def refresh_portfolio(settings, path: Path = PORTFOLIO_FILE, warn: Callable[[str], None] = print) -> dict:  # pragma: no cover
    """Fetch prices + metadata, build and write data/snapshots/portfolio.json."""
    from forecaster.cache import Cache
    from forecaster.market_data import ParquetStore, usd_cad_rates
    from forecaster.pipeline import snapshot_dir, write_json
    from forecaster.sources.fred import FredClient

    pf = load_portfolio(path)
    store = ParquetStore(Path(settings.cache_dir) / "ohlcv")
    cache = Cache(Path(settings.cache_dir), ttl_seconds=7 * 86400)
    fred = FredClient(settings.fred_api_key, cache) if settings.fred_api_key else None
    rates = usd_cad_rates(store, fred, on_warn=warn)
    native = {t: store.get(t)["Close"] for t in pf.tickers}
    bench = {k: store.get(k)["Close"] for k in BENCHMARKS}
    fetched = {}
    for t in pf.tickers:
        try:
            fetched[t] = cache.get_or_fetch(Cache.make_key("yf_meta", ticker=t), lambda t=t: yf_metadata(t))
        except Exception as exc:
            warn(f"metadata for {t} unavailable: {exc}")
    snap = build_snapshot(pf, native, rates, bench, fetched)
    write_json(snapshot_dir(settings) / "portfolio.json", snap)
    return snap
