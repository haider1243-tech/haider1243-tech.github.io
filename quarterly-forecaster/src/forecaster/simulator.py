"""Simulator: run any strategy on any instrument / period with money, sizing and costs, in CAD.

Reuses the Strategy Lab's data cache (intraday.IntradayStore), strategies and engine
(lab_engine.simulate), so a simulation and a lab backtest of the same setup trade identically.
On top of that it adds what a person deciding with real money needs:

  * filters: trading-hours window, days of week, skip CPI/PPI/FOMC days,
  * overrides: stop / target as % of the entry, time exit after N bars,
  * money: starting balance in CAD, sizing (fixed $ per trade, fixed units, or % of equity risked
    to the stop; never more than `max_leverage` x equity, default 1 = no leverage), max trades per
    day, max daily loss stop, commission per order + per unit with a minimum (IBKR-style),
  * results: CAD P&L, stats, a 95% CI and p-value vs 1,000 random-entry simulations, verdict
    (EDGE needs 100+ trades), equity + drawdown, buy-and-hold, a Monte Carlo fan of trade order,
    breakdowns by month / weekday / hour, a full trade log and a plain-English summary.

The verdict here is IN-SAMPLE: you chose the parameters looking at this same period. Use
"Forward test" (shadow trading on the live feed) or the Strategy Lab's walk-forward for an
out-of-sample check. Nothing in this module can place an order.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from forecaster import intraday as I
from forecaster import lab_engine as E
from forecaster import lab_stats as LS
from forecaster import strategies as S

TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": None}
RANGES = {"Last day": 1, "Last week": 7, "Last month": 31, "3 months": 92, "1 year": 365, "3 years": 3 * 365,
          "5 years": 5 * 365}
SIM_SYMBOLS = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "XAU_USD", "USD_CAD")


# --------------------------------------------------------------------------- #
# Costs and sizing
# --------------------------------------------------------------------------- #

@dataclass
class Costs:
    """Per ORDER (each side): flat + per unit, at least `min_commission` when any commission
    applies. Spread is the full bid/ask in price units (a bar's own Spread column wins when the
    data has one); slippage in price units on market and stop fills."""
    commission_per_order: float = 0.0
    commission_per_unit: float = 0.0
    min_commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0

    def commission(self, units: float) -> float:
        if units <= 0:
            return 0.0
        c = self.commission_per_order + self.commission_per_unit * units
        return max(c, self.min_commission) if (self.commission_per_unit or self.commission_per_order
                                                or self.min_commission) else 0.0


def default_costs(symbol: str) -> Costs:
    """Realistic defaults. US stocks: IBKR Pro fixed US$0.005/share, US$1 minimum per order, one
    cent spread and slippage. Gold and USD/CAD on OANDA: no commission, typical spread."""
    if symbol == "XAU_USD":
        return Costs(spread=0.35, slippage=0.05)
    if symbol == "USD_CAD":
        return Costs(spread=0.00020, slippage=0.00005)
    return Costs(commission_per_unit=0.005, min_commission=1.0, spread=0.01, slippage=0.01)


@dataclass
class Sizing:
    mode: str = "risk_pct"          # fixed_cash (CAD per trade) | fixed_units | risk_pct (% of equity)
    value: float = 1.0
    max_leverage: float = 1.0       # notional cap as a multiple of equity (1 = no leverage)


def size_units(sz: Sizing, equity_quote: float, cash_quote: float, entry: float, stop: float) -> int:
    """Whole units to trade. `equity_quote` and `cash_quote` (the fixed-cash amount) are already in
    the instrument's quote currency. Risk mode: units = (equity x pct) / |entry - stop|."""
    if entry <= 0 or equity_quote <= 0:
        return 0
    if sz.mode == "fixed_cash":
        units = math.floor(cash_quote / entry)
    elif sz.mode == "fixed_units":
        units = math.floor(sz.value)
    elif sz.mode == "risk_pct":
        per_unit = abs(entry - stop)
        units = math.floor(equity_quote * sz.value / 100 / per_unit) if per_unit > 0 else 0
    else:
        raise ValueError(f"unknown sizing mode {sz.mode!r}")
    cap = math.floor(equity_quote * sz.max_leverage / entry)
    return int(max(0, min(units, cap)))


def quote_ccy(symbol: str) -> str:
    return "CAD" if symbol.endswith("_CAD") or symbol.endswith(".TO") else "USD"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SimConfig:
    symbol: str = "SPY"
    strategy: str = "orb15"
    params: dict = field(default_factory=dict)
    timeframe: str = "5m"
    start: str = ""                 # YYYY-MM-DD (inclusive); empty = earliest data
    end: str = ""                   # YYYY-MM-DD (inclusive); empty = latest data
    hours: tuple | None = None      # (from, to) New York "HH:MM" for ENTRIES, e.g. ("09:30", "11:30")
    weekdays: tuple = (0, 1, 2, 3, 4)
    skip_events: bool = False
    balance_cad: float = 10_000.0
    sizing: Sizing = field(default_factory=Sizing)
    max_trades_per_day: int = 3
    max_daily_loss_cad: float | None = None
    costs: Costs | None = None
    stop_pct: float | None = None   # overrides, % of the entry reference price
    target_pct: float | None = None
    max_hold_bars: int | None = None
    name: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hours"] = list(self.hours) if self.hours else None
        d["weekdays"] = list(self.weekdays)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SimConfig:
        d = dict(d)
        d["sizing"] = Sizing(**(d.get("sizing") or {}))
        d["costs"] = Costs(**d["costs"]) if d.get("costs") else None
        d["hours"] = tuple(d["hours"]) if d.get("hours") else None
        d["weekdays"] = tuple(d.get("weekdays") or (0, 1, 2, 3, 4))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def strategy_obj(self) -> S.Strategy:
        return S.LIBRARY[self.strategy].with_params(**self.params)

    def resolved_costs(self) -> Costs:
        return self.costs or default_costs(self.symbol)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def daily_window_bars(m1: pd.DataFrame, spec: I.InstrumentSpec) -> tuple[pd.DataFrame, int]:
    """Daily bars of the day-trading window, stamped at the window start so the engine treats each
    as one whole-session bar. Returns (bars, window minutes)."""
    w = I.window_mask(m1.index, spec)
    d = m1[w]
    if d.empty:
        return d, 0
    day = I.trading_day(d.index, spec.vwap_anchor)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if "Spread" in d:
        agg["Spread"] = "mean"
    out = d.groupby(day).agg(agg)
    minutes = I._hm(spec.session_end) - I._hm(spec.session_start)
    out.index = (pd.DatetimeIndex(out.index).tz_localize(I.NY) + pd.Timedelta(minutes=I._hm(spec.session_start))
                 ).tz_convert("UTC")
    return out, minutes


def load_bars(store: I.IntradayStore, cfg: SimConfig) -> tuple[pd.DataFrame, int, I.InstrumentSpec]:
    spec = I.spec_for(cfg.symbol)
    if spec is None:
        raise ValueError(f"{cfg.symbol}: no intraday data source (US stocks, gold and USD/CAD only)")
    start = pd.Timestamp(cfg.start, tz=I.NY).tz_convert("UTC") - pd.Timedelta(days=3) if cfg.start else None
    end = (pd.Timestamp(cfg.end, tz=I.NY) + pd.Timedelta(days=1)).tz_convert("UTC") if cfg.end else None
    m1 = I.session_bars(store.read(spec.source, cfg.symbol, start, end), spec)
    if cfg.timeframe == "1d":
        bars, minutes = daily_window_bars(m1, spec)
    else:
        minutes = TIMEFRAMES[cfg.timeframe]
        bars = I.resample(m1, minutes, spec)
    return bars, minutes, spec


# --------------------------------------------------------------------------- #
# Signals: filters and overrides (also used by shadow trading for forward tests)
# --------------------------------------------------------------------------- #

def adjust_signals(sig: pd.DataFrame, bars: pd.DataFrame, cfg: SimConfig, events: dict[str, set] | None,
                   minutes: int) -> pd.DataFrame:
    """Filters act on the ENTRY bar (i + 1): hours window, weekdays, event days, date range.
    Overrides replace the strategy's stop/target with % distances from the signal close."""
    if sig.empty:
        return sig
    j = np.minimum(sig.index.to_numpy() + 1, len(bars) - 1)
    t = bars.index[j].tz_convert(I.NY)
    keep = np.isin(t.dayofweek, list(cfg.weekdays))
    if cfg.hours and cfg.timeframe != "1d":
        h0, h1 = I._hm(cfg.hours[0]), I._hm(cfg.hours[1])
        m = t.hour * 60 + t.minute
        keep &= (m >= h0) & (m < h1)
    if cfg.skip_events and events:
        bad = set().union(*(events.get(k, set()) for k in ("CPI", "PPI", "FOMC")))
        keep &= ~np.isin(t.strftime("%Y-%m-%d"), list(bad))
    if cfg.start:
        keep &= t.normalize().tz_localize(None) >= pd.Timestamp(cfg.start)
    if cfg.end:
        keep &= t.normalize().tz_localize(None) <= pd.Timestamp(cfg.end)
    out = sig[keep].copy()
    ref = bars["Close"].to_numpy()[out.index.to_numpy()]
    side = out["side"].to_numpy()
    if cfg.stop_pct:
        out["stop"] = ref * (1 - side * cfg.stop_pct / 100)
    if cfg.target_pct:
        out["target"] = ref * (1 + side * cfg.target_pct / 100)
    return out


# --------------------------------------------------------------------------- #
# Money
# --------------------------------------------------------------------------- #

def fx_lookup(fx: pd.Series | float | None, ccy: str):
    """CAD per one unit of quote currency at a date (last rate on or before it)."""
    if ccy == "CAD":
        return lambda d: 1.0
    if fx is None:
        fx = 1.37
    if isinstance(fx, (int, float)):
        return lambda d, v=float(fx): v
    s = fx.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    vals, keys = s.to_numpy(float), s.index

    def at(d):
        k = keys.searchsorted(pd.Timestamp(d).normalize(), side="right") - 1
        return float(vals[max(k, 0)])
    return at


def account(trades: pd.DataFrame, cfg: SimConfig, fx) -> tuple[pd.DataFrame, dict]:
    """Size, cost and book each engine trade in order (compounding equity). Engine prices already
    include spread and slippage; commission is applied here, per order, on the real unit count."""
    costs = cfg.resolved_costs()
    rate = fx_lookup(fx, quote_ccy(cfg.symbol))
    equity = float(cfg.balance_cad)
    day_pnl: dict = {}
    rows, skipped = [], {"too small": 0, "daily loss limit": 0}
    for t in trades.itertuples():
        d = pd.Timestamp(t.day)
        if cfg.max_daily_loss_cad and day_pnl.get(d, 0.0) <= -abs(cfg.max_daily_loss_cad):
            skipped["daily loss limit"] += 1
            continue
        r = rate(d)
        cash_quote = (cfg.sizing.value if cfg.sizing.mode == "fixed_cash" else 0.0) / r
        units = size_units(cfg.sizing, equity / r, cash_quote, t.entry, t.stop)
        if units <= 0:
            skipped["too small"] += 1
            continue
        comm = costs.commission(units) * 2                  # entry + exit orders
        pnl_q = t.side * (t.exit - t.entry) * units - comm
        pnl = pnl_q * r
        equity += pnl
        day_pnl[d] = day_pnl.get(d, 0.0) + pnl
        rows.append({"entry_time": t.entry_time, "exit_time": t.exit_time, "day": t.day, "side": int(t.side),
                     "entry": t.entry, "exit": t.exit, "stop": t.stop, "target": t.target, "reason": t.reason,
                     "units": units, "commission": comm, "pnl_quote": pnl_q, "fx": r, "pnl_cad": pnl,
                     "ret": pnl_q / (t.entry_ref * units), "net_ret": pnl_q / (t.entry_ref * units),
                     "equity": equity, "bars_held": t.bars_held, "entry_ref": t.entry_ref,
                     "hold_min": (t.exit_time - t.entry_time).total_seconds() / 60})
    cols = ["entry_time", "exit_time", "day", "side", "entry", "exit", "stop", "target", "reason", "units",
            "commission", "pnl_quote", "fx", "pnl_cad", "ret", "net_ret", "equity", "bars_held", "entry_ref", "hold_min"]
    return pd.DataFrame(rows, columns=cols), skipped


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def _breakdown(log: pd.DataFrame, key: pd.Series, order: list | None = None) -> list[dict]:
    if log.empty:
        return []
    g = log.groupby(key)
    out = [{"key": str(k), "trades": len(x), "win_rate": float((x["pnl_cad"] > 0).mean()),
            "pnl_cad": float(x["pnl_cad"].sum()), "avg_cad": float(x["pnl_cad"].mean())} for k, x in g]
    if order:
        rank = {k: i for i, k in enumerate(order)}
        out.sort(key=lambda r: rank.get(r["key"], 99))
    return out


def monte_carlo(pnls: np.ndarray, balance: float, n: int = 1000, seed: int = 0, points: int = 200) -> dict:
    """Reshuffle the trade ORDER n times: same trades, different luck of sequence."""
    if len(pnls) < 2:
        return {}
    rng = np.random.default_rng(seed)
    paths = balance + np.cumsum(np.array([rng.permutation(pnls) for _ in range(n)]), axis=1)
    paths = np.hstack([np.full((n, 1), balance), paths])
    step = max(1, paths.shape[1] // points)
    idx = np.unique(np.r_[np.arange(0, paths.shape[1], step), paths.shape[1] - 1])   # always keep the final trade
    finals = paths[:, -1]
    dd = (paths - np.maximum.accumulate(paths, axis=1)).min(axis=1)
    return {"x": idx.tolist(),
            "p5": np.quantile(paths, 0.05, axis=0)[idx].tolist(),
            "p50": np.quantile(paths, 0.50, axis=0)[idx].tolist(),
            "p95": np.quantile(paths, 0.95, axis=0)[idx].tolist(),
            "best": paths[int(np.argmax(finals))][idx].tolist(),
            "worst": paths[int(np.argmin(finals))][idx].tolist(),
            "dd_median": float(np.median(dd)), "dd_p95": float(np.quantile(dd, 0.05))}


def run(cfg: SimConfig, bars: pd.DataFrame, minutes: int, spec: I.InstrumentSpec, fx=None,
        events: dict[str, set] | None = None, n_random: int = 1000, seed: int = 0) -> dict:
    costs = cfg.resolved_costs()
    daily = cfg.timeframe == "1d"
    st = cfg.strategy_obj()
    cm = E.CostModel(spread=costs.spread, slippage=costs.slippage)
    pp = E.prepare(bars, spec, minutes, cm, last_minute_guard=0 if daily else 1, first_minute_guard=0 if daily else 1)
    feats = S.Features(bars, spec, minutes)
    sig = adjust_signals(st.signals(feats), bars, cfg, events, minutes)
    trades = E.simulate(pp, sig, cm, int(cfg.max_trades_per_day), max_hold=cfg.max_hold_bars, cross_day=daily)
    log, skipped = account(trades, cfg, fx)
    rate = fx_lookup(fx, quote_ccy(cfg.symbol))
    # performance (CAD) + edge (per-trade returns after every cost)
    n = len(log)
    pnl = log["pnl_cad"].to_numpy(float) if n else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = cfg.balance_cad + np.cumsum(pnl) if n else np.array([])
    peak = np.maximum.accumulate(np.r_[cfg.balance_cad, eq])[1:] if n else np.array([])
    dd_cad = eq - peak if n else np.array([])
    period = (bars.index[pp.entry_ok].min(), bars.index[pp.entry_ok].max()) if pp.entry_ok.any() else (None, None)
    headline = {
        "trades": n, "win_rate": float(len(wins) / n) if n else None,
        "pnl_cad": float(pnl.sum()), "pnl_pct": float(pnl.sum() / cfg.balance_cad),
        "expectancy_cad": float(pnl.mean()) if n else None,
        "expectancy_ret": float(log["ret"].mean()) if n else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if n and losses.sum() < 0 else None,
        "max_dd_cad": float(dd_cad.min()) if n else 0.0,
        "max_dd_pct": float((dd_cad / peak).min()) if n else 0.0,
        "avg_hold_min": float(log["hold_min"].mean()) if n else None,
        "losing_streak": LS.longest_losing_streak(pnl) if n else 0,
        "final_equity": float(eq[-1]) if n else cfg.balance_cad,
        "skipped": skipped, "signals": int(len(sig)),
    }
    # edge vs luck on per-trade returns: CI (whole days resampled), random entries with the same
    # days, holding times, sides, spread/slippage and the average commission per unit
    rets = log["ret"].to_numpy(float) if n else np.array([])
    ci = LS.bootstrap_ci(rets, days=log["day"].to_numpy(), seed=seed) if n > 1 else None
    rand = np.array([])
    if n:
        comm_unit = float((log["commission"] / 2 / log["units"]).mean())
        rand = E.random_entry_means(pp, log["bars_held"].to_numpy(int), log["side"].to_numpy(int),
                                    E.CostModel(commission=comm_unit, spread=costs.spread, slippage=costs.slippage),
                                    n_random, seed, trade_days=pd.to_datetime(log["day"]).to_numpy("datetime64[ns]"))
    p = LS.p_value(float(rets.mean()), rand) if len(rand) else None
    verdict, why = LS.verdict({"trades": n, "expectancy": float(rets.mean()) if n else None}, p)
    why = [w.replace("out-of-sample ", "").replace(" OOS trades", " trades") for w in why]   # this run is in-sample
    # buy and hold of the same instrument over the same period (CAD, whole units, one commission each way)
    bh = None
    if pp.entry_ok.any():
        first = int(np.flatnonzero(pp.entry_ok)[0])
        r0 = rate(pd.Timestamp(pp.day[first]))
        u = math.floor(cfg.balance_cad / r0 / bars["Open"].iloc[first])
        c = bars["Close"].iloc[first:]
        step = max(1, len(c) // 400)
        cc = c.iloc[::step]
        rates = np.array([rate(pd.Timestamp(d)) for d in cc.index.tz_convert(I.NY).normalize().tz_localize(None)])
        val = cfg.balance_cad + (cc.to_numpy() - bars["Open"].iloc[first]) * u * rates - 2 * costs.commission(u) * r0
        bh = {"t": [x.isoformat() for x in cc.index], "equity": val.tolist(), "units": u,
              "return_pct": float(val[-1] / cfg.balance_cad - 1) if len(val) else None}
    et = pd.DatetimeIndex(log["entry_time"]).tz_convert(I.NY) if n else pd.DatetimeIndex([])
    res = {
        "config": cfg.to_dict(), "costs": asdict(costs), "headline": headline,
        "edge": {"ci": ci, "p_value": p, "verdict": verdict, "why": why, "n_random": int(len(rand)),
                 "random_q": {q: float(np.quantile(rand, q / 100)) for q in (5, 50, 95)} if len(rand) else {},
                 "in_sample": True},
        "equity": {"t": [x.isoformat() for x in log["exit_time"]] if n else [], "equity": eq.tolist(),
                   "drawdown": dd_cad.tolist()},
        "buy_hold": bh, "mc": monte_carlo(pnl, cfg.balance_cad, seed=seed),
        "by_month": _breakdown(log, et.strftime("%Y-%m")) if n else [],
        "by_weekday": _breakdown(log, et.day_name().str[:3], ["Mon", "Tue", "Wed", "Thu", "Fri"]) if n else [],
        "by_hour": _breakdown(log, pd.Index([f"{h:02d}:00" for h in et.hour])) if n else [],
        "period": [None if x is None else x.isoformat() for x in period],
        "data_bars": len(bars),
    }
    res["summary"] = summary_text(res)
    return res, log


def summary_text(res: dict) -> str:
    h, e = res["headline"], res["edge"]
    p0, p1 = res["period"]
    if not p0:
        return "No data for this instrument and period."
    span = (pd.Timestamp(p1) - pd.Timestamp(p0)).days
    when = (f"over the last {span // 30} months" if 45 <= span < 700 else
            f"over the last {span / 365:.1f} years" if span >= 700 else f"over {max(span, 1)} days")
    if h["trades"] == 0:
        return f"{when.capitalize()} this strategy took no trades with these settings."
    v = e["verdict"]
    pv = "n/a" if e["p_value"] is None else f"{e['p_value']:.3f}"
    edge = {"EDGE": f"the edge looks statistically real (p = {pv} vs 1,000 random entries)",
            "WEAK": f"there may be an edge, but it isn't proven (p = {pv}"
                    + (f", only {h['trades']} trades" if h["trades"] < 100 else "") + ")",
            "NO EDGE": f"the result is consistent with luck (p = {pv})"}[v]
    money = f"{'made' if h['pnl_cad'] >= 0 else 'lost'} C${abs(h['pnl_cad']):,.0f} ({h['pnl_pct']:+.1%})"
    return (f"{when.capitalize()} this strategy made {h['trades']} trades, won {h['win_rate']:.0%}, {money} after "
            f"costs, and {edge}. In-sample: these settings were chosen on this same period; forward-test before "
            "trusting it.")


def compare(cfgs: list[SimConfig], store: I.IntradayStore, fx=None, events=None, n_random: int = 1000) -> list[dict]:
    out = []
    for c in cfgs[:4]:
        bars, minutes, spec = load_bars(store, c)
        res, _ = run(c, bars, minutes, spec, fx, events, n_random)
        out.append(res)
    return out


# --------------------------------------------------------------------------- #
# Scenarios + forward tests (shadow trading)
# --------------------------------------------------------------------------- #

def sim_dir(cache_dir: Path) -> Path:
    return Path(cache_dir) / "sim"


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _save(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(p)


def save_scenario(cache_dir: Path, cfg: SimConfig) -> str:
    name = cfg.name or f"{S.LIBRARY[cfg.strategy].name} · {cfg.symbol} {cfg.timeframe}"
    d = _load(sim_dir(cache_dir) / "scenarios.json")
    d[name] = {**cfg.to_dict(), "name": name, "saved_at": pd.Timestamp.now(tz="UTC").isoformat()}
    _save(sim_dir(cache_dir) / "scenarios.json", d)
    return name


def load_scenarios(cache_dir: Path) -> dict[str, SimConfig]:
    return {k: SimConfig.from_dict(v) for k, v in _load(sim_dir(cache_dir) / "scenarios.json").items()}


def delete_scenario(cache_dir: Path, name: str) -> None:
    d = _load(sim_dir(cache_dir) / "scenarios.json")
    d.pop(name, None)
    _save(sim_dir(cache_dir) / "scenarios.json", d)


def add_forward_test(cache_dir: Path, cfg: SimConfig, result: dict | None = None) -> str:
    """Register a scenario for shadow trading on the live feed (paper only)."""
    if cfg.timeframe == "1d":
        raise ValueError("forward tests run on intraday bars (1m-1h)")
    name = cfg.name or f"{S.LIBRARY[cfg.strategy].name} · {cfg.symbol} {cfg.timeframe}"
    key = f"fwd:{name}|{cfg.symbol}|{cfg.timeframe}"
    d = _load(sim_dir(cache_dir) / "forward.json")
    costs = cfg.resolved_costs()
    d[key] = {"key": key, "name": name, "strategy": cfg.strategy, "instrument": cfg.symbol,
              "timeframe": cfg.timeframe, "params": cfg.strategy_obj().params,
              "costs": asdict(E.CostModel(commission=costs.commission_per_unit, spread=costs.spread,
                                          slippage=costs.slippage)),
              "verdict": (result or {}).get("edge", {}).get("verdict", "NO EDGE"),
              "backtest_expectancy": (result or {}).get("headline", {}).get("expectancy_ret"),
              "backtest_ci": (result or {}).get("edge", {}).get("ci"),
              "sim": cfg.to_dict(), "added_at": pd.Timestamp.now(tz="UTC").isoformat()}
    _save(sim_dir(cache_dir) / "forward.json", d)
    return key


def forward_tests(cache_dir: Path) -> list[dict]:
    return list(_load(sim_dir(cache_dir) / "forward.json").values())


def remove_forward_test(cache_dir: Path, key: str) -> None:
    d = _load(sim_dir(cache_dir) / "forward.json")
    d.pop(key, None)
    _save(sim_dir(cache_dir) / "forward.json", d)


def trade_log_csv(log: pd.DataFrame) -> bytes:
    cols = ["entry_time", "exit_time", "side", "units", "entry", "exit", "stop", "target", "reason",
            "commission", "pnl_quote", "fx", "pnl_cad", "equity"]
    out = log[cols].copy() if len(log) else pd.DataFrame(columns=cols)
    out["side"] = out["side"].map({1: "long", -1: "short"})
    for c in ("entry_time", "exit_time"):
        out[c] = pd.to_datetime(out[c]).dt.tz_convert(I.NY).dt.strftime("%Y-%m-%d %H:%M")
    return out.to_csv(index=False).encode()


def load_fx(cache_dir: Path) -> pd.Series | None:
    """Daily USD/CAD (CAD per USD) from the forecaster's cache, if present."""
    try:
        from forecaster.market_data import ParquetStore

        df = ParquetStore(Path(cache_dir) / "ohlcv").read("CAD=X")
        return None if df is None or df.empty else df["Close"]
    except Exception:
        return None
