"""Strategy Lab: download history, backtest every strategy with walk-forward, judge edge vs luck.

    python -m forecaster.lab download --years 3        # OANDA (gold) + Alpaca (SPY, QQQ) 1m bars -> cache
    python -m forecaster.lab run                       # all strategies x SPY/QQQ/XAU_USD x 5m/15m
    python -m forecaster.lab run --synthetic           # same pipeline on random-walk data (offline demo:
                                                       #   every verdict SHOULD be NO EDGE)
    python -m forecaster.lab show                      # leaderboard in the terminal

Results: data/lab/results.json (read by the app's Signals tab: Signal Board, Leaderboard, Strategy
Lab) and data/lab/trades/<setup>.parquet (out-of-sample trades). Nothing here places orders.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecaster import intraday as I
from forecaster import lab_engine as E
from forecaster import lab_stats as LS
from forecaster import strategies as S
from forecaster.config import PROJECT_ROOT

log = logging.getLogger("lab")
INSTRUMENTS = ("SPY", "QQQ", "XAU_USD")
TIMEFRAMES = ("5m", "15m")
N_RANDOM = 1000


def lab_dir(cache_dir: Path) -> Path:
    return Path(cache_dir) / "lab"


def setup_key(strategy: str, symbol: str, tf: str) -> str:
    return f"{strategy}|{symbol}|{tf}"


def _safe(key: str) -> str:
    return key.replace("|", "__")


# --------------------------------------------------------------------------- #
# One setup
# --------------------------------------------------------------------------- #

def evaluate_setup(bars: pd.DataFrame, symbol: str, tf: str, strategy: S.Strategy,
                   costs: E.CostModel | None = None, vix: pd.Series | None = None,
                   events: dict[str, set] | None = None, n_random: int = N_RANDOM,
                   train_months: int = 12, test_months: int = 3, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    spec = I.INSTRUMENTS[symbol]
    minutes = I.TIMEFRAMES[tf]
    costs = costs or E.DEFAULT_COSTS[symbol]
    f = S.Features(bars, spec, minutes)
    pp = E.prepare(bars, spec, minutes, costs)

    def run_combo(params: dict) -> pd.DataFrame:
        st = strategy.with_params(**params)
        return E.simulate(pp, st.signals(f), costs, int(st.params.get("max_trades", 99)))

    days = pd.DatetimeIndex(sorted(set(pp.day[pp.entry_ok])))
    wf = LS.walk_forward(run_combo, strategy.grid, days, train_months, test_months)
    oos, is_ = wf["oos"], wf["is"]
    oos_days = sorted({d for fo in wf["folds"] if fo.best for d in days if fo.test[0] <= d < fo.test[1]})
    oos_s = LS.summary(oos, days=len(oos_days))
    is_s = LS.summary(is_)
    r = oos["net_ret"].to_numpy(float) if len(oos) else np.array([])
    rand = E.random_entry_means(pp, oos["bars_held"].to_numpy(int), oos["side"].to_numpy(int), costs, n_random, seed,
                                trade_days=pd.to_datetime(oos["day"]).to_numpy("datetime64[ns]")) \
        if len(oos) else np.array([])
    p = LS.p_value(oos_s["expectancy"], rand) if len(rand) else None
    verdict, why = LS.verdict(oos_s, p)
    sliced = LS.add_slice_columns(oos, vix, events)
    by_side = {name: LS.summary(oos[oos["side"] == sgn]) for name, sgn in (("long", 1), ("short", -1))} \
        if len(oos) else {}
    side_ci = {name: LS.bootstrap_ci(oos.loc[oos["side"] == sgn, "net_ret"].to_numpy(float),
                                     days=oos.loc[oos["side"] == sgn, "day"].to_numpy())
               for name, sgn in (("long", 1), ("short", -1))} if len(oos) else {}
    eq = np.cumsum(r)
    step = max(1, len(eq) // 300)
    hist_counts, hist_edges = (np.histogram(rand, bins=30) if len(rand) else (np.array([]), np.array([])))
    res = {
        "key": setup_key(strategy.key, symbol, tf), "strategy": strategy.key, "name": strategy.name,
        "instrument": symbol, "timeframe": tf, "description": (strategy.description or "").strip(),
        "params": {**strategy.params, **wf["latest_params"]}, "grid": strategy.grid, "tuned": strategy.tuned(),
        "combos": wf["combos"], "costs": costs.to_dict(),
        "folds": [{"train": [str(fo.train[0].date()), str(fo.train[1].date())],
                   "test": [str(fo.test[0].date()), str(fo.test[1].date())], "best": fo.best,
                   "is_expectancy": fo.is_stats.get("expectancy"), "oos_expectancy": fo.oos_stats.get("expectancy"),
                   "oos_trades": fo.oos_stats.get("trades", 0)} for fo in wf["folds"]],
        "is": is_s, "oos": oos_s, "oos_by_side": by_side, "ci": LS.bootstrap_ci(r, seed=seed, days=oos["day"].to_numpy() if len(oos) else None), "side_ci": side_ci,
        "mc": LS.order_monte_carlo(r, seed=seed), "p_value": p,
        "random": {"n": int(len(rand)), "quantiles": {q: float(np.quantile(rand, q / 100)) for q in (5, 25, 50, 75, 95)}
                   if len(rand) else {}, "hist": hist_counts.tolist(), "edges": hist_edges.tolist()},
        "verdict": verdict, "why": why,
        "warnings": LS.overfit_warnings(strategy.tuned(), wf["combos"], wf["folds"], is_s, oos_s),
        "slices": {k: LS.slice_stats(sliced, k) for k in LS.SLICES},
        "equity": eq[::step].tolist(),
        "data": {"start": str(bars.index.min()), "end": str(bars.index.max()), "bars": len(bars)},
    }
    return res, oos


# --------------------------------------------------------------------------- #
# All setups
# --------------------------------------------------------------------------- #

def run_lab(bars_by: dict[tuple[str, str], pd.DataFrame], out_dir: Path, strategies: list[str] | None = None,
            vix: pd.Series | None = None, events: dict[str, set] | None = None, synthetic: bool = False,
            n_random: int = N_RANDOM, progress=print, train_months: int = 12, test_months: int = 3) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "trades").mkdir(parents=True, exist_ok=True)
    results = []
    for (symbol, tf), bars in bars_by.items():
        minutes = I.TIMEFRAMES[tf]
        for key, st in S.LIBRARY.items():
            if strategies and key not in strategies or not S.applicable(st, minutes):
                continue
            t0 = time.time()
            res, oos = evaluate_setup(bars, symbol, tf, st, vix=vix, events=events, n_random=n_random,
                                      train_months=train_months, test_months=test_months)
            oos.to_parquet(out_dir / "trades" / f"{_safe(res['key'])}.parquet")
            results.append(res)
            e = res["oos"]["expectancy"]
            progress(f"{res['key']:<32} OOS {res['oos']['trades']:>5} trades  exp "
                     f"{'n/a' if e is None else f'{e * 1e4:+.2f} bps':>11}  p={res['p_value'] if res['p_value'] is None else round(res['p_value'], 3)}"
                     f"  {res['verdict']}  ({time.time() - t0:.1f}s)")
    m = max(1, len(results))
    for r in results:          # multiple-testing guard: Bonferroni over every setup tested
        r["bonferroni_ok"] = r["p_value"] is not None and r["p_value"] < LS.EDGE_P / m
        r["setups_tested"] = m
    doc = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "synthetic": synthetic,
           "n_random": n_random, "setups": results}
    tmp = out_dir / "results.json.tmp"
    tmp.write_text(json.dumps(doc, default=_json_default))
    tmp.replace(out_dir / "results.json")
    return doc


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    return str(o)


def load_results(cache_dir: Path) -> dict | None:
    try:
        return json.loads((lab_dir(cache_dir) / "results.json").read_text())
    except (FileNotFoundError, ValueError):
        return None


def load_oos_trades(cache_dir: Path, key: str) -> pd.DataFrame:
    p = lab_dir(cache_dir) / "trades" / f"{_safe(key)}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=E.TRADE_COLS)


def leaderboard(doc: dict | None, instrument: str | None = None, timeframe: str | None = None) -> list[dict]:
    """Setups ranked by out-of-sample expectancy after costs (best first)."""
    rows = [s for s in (doc or {}).get("setups", [])
            if (instrument in (None, s["instrument"])) and (timeframe in (None, s["timeframe"]))]
    return sorted(rows, key=lambda s: -np.inf if s["oos"]["expectancy"] is None else s["oos"]["expectancy"],
                  reverse=True)


# --------------------------------------------------------------------------- #
# Inputs: bars, VIX, event days
# --------------------------------------------------------------------------- #

def bars_from_cache(store: I.IntradayStore, instruments=INSTRUMENTS, timeframes=TIMEFRAMES) -> dict:
    out = {}
    for sym in instruments:
        for tf in timeframes:
            b = I.load_bars(store, sym, tf)
            if len(b):
                out[(sym, tf)] = b
    return out


def synthetic_bars(instruments=INSTRUMENTS, timeframes=TIMEFRAMES, start="2022-01-03", end="2024-12-31") -> dict:
    out = {}
    for n, sym in enumerate(instruments):
        spec = I.INSTRUMENTS[sym]
        m1 = I.session_bars(I.synthetic_m1(sym, start, end, seed=100 + n), spec)
        for tf in timeframes:
            out[(sym, tf)] = I.resample(m1, I.TIMEFRAMES[tf], spec)
    return out


def load_vix(cache_dir: Path) -> pd.Series | None:
    try:
        from forecaster.market_data import ParquetStore

        df = ParquetStore(Path(cache_dir) / "ohlcv").read("^VIX")
        return None if df is None else df["Close"]
    except Exception:
        return None


def load_event_days(settings, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, set] | None:  # pragma: no cover - network
    """Historical CPI / PPI release dates and FOMC decision dates from FRED (cached)."""
    if not settings.fred_api_key:
        return None
    from forecaster.cache import Cache
    from forecaster.sources.fred import FredClient

    fred = FredClient(settings.fred_api_key, Cache(settings.cache_dir, ttl_seconds=7 * 86400))
    out: dict[str, set] = {}
    for name, series in (("CPI", "CPIAUCSL"), ("PPI", "PPIACO"), ("FOMC", "DFEDTARU")):
        try:
            rid = int(fred.release_for_series(series)["id"])
            out[name] = {d.strftime("%Y-%m-%d") for d in fred.release_dates(rid, limit=1000) if start <= d <= end}
        except Exception as exc:
            log.warning("event dates %s unavailable: %s", name, exc)
    return out or None


def download(settings, years: float = 3.0, instruments=INSTRUMENTS, progress=print) -> None:  # pragma: no cover - network
    from dotenv import dotenv_values

    env = {**{k: v for k, v in dotenv_values(PROJECT_ROOT / ".env").items() if v}, **os.environ} \
        if (PROJECT_ROOT / ".env").exists() else dict(os.environ)
    store = I.IntradayStore(Path(settings.cache_dir) / "intraday")
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.DateOffset(years=years)
    for sym in instruments:
        spec = I.INSTRUMENTS[sym]
        if spec.source == "oanda":
            tok = env.get("OANDA_API_TOKEN")
            if not tok:
                progress(f"{sym}: skipped (OANDA_API_TOKEN missing from .env)")
                continue
            host = I.OANDA_REST.get((env.get("OANDA_ENVIRONMENT") or "practice").lower(), I.OANDA_REST["practice"])
            store.fill("oanda", sym, start, end, lambda a, b, s=sym: I.oanda_m1(s, a, b, tok, host), progress)
        else:
            kid, sec = env.get("ALPACA_API_KEY_ID"), env.get("ALPACA_API_SECRET_KEY")
            if kid and sec:
                store.fill("alpaca", sym, start, end, lambda a, b, s=sym: I.alpaca_m1(s, a, b, kid, sec), progress)
            elif env.get("FINNHUB_API_KEY"):
                try:
                    store.fill("alpaca", sym, start, end,
                               lambda a, b, s=sym: I.finnhub_m1(s, a, b, env["FINNHUB_API_KEY"]), progress)
                except I.DataUnavailable as exc:
                    progress(f"{sym}: {exc}")
            else:
                progress(f"{sym}: skipped (add ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY; free paper account)")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    from forecaster.config import load_settings

    ap = argparse.ArgumentParser(description="Strategy Lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--years", type=float, default=3.0)
    d.add_argument("instruments", nargs="*", default=list(INSTRUMENTS))
    r = sub.add_parser("run")
    r.add_argument("--synthetic", action="store_true", help="random-walk data (offline demo)")
    r.add_argument("--strategies", nargs="*")
    r.add_argument("--instruments", nargs="*", default=list(INSTRUMENTS))
    r.add_argument("--timeframes", nargs="*", default=list(TIMEFRAMES))
    r.add_argument("--random", type=int, default=N_RANDOM)
    sub.add_parser("show")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()
    if args.cmd == "download":
        download(settings, args.years, args.instruments)
        return 0
    if args.cmd == "run":
        if args.synthetic:
            bars = synthetic_bars(args.instruments, args.timeframes)
            vix, events = None, None
        else:
            bars = bars_from_cache(I.IntradayStore(Path(settings.cache_dir) / "intraday"), args.instruments,
                                   args.timeframes)
            if not bars:
                print("No cached intraday data: run `python -m forecaster.lab download` first.")
                return 1
            vix = load_vix(settings.cache_dir)
            starts = min(b.index.min() for b in bars.values()).tz_localize(None)
            events = load_event_days(settings, starts, pd.Timestamp.now())
        run_lab(bars, lab_dir(settings.cache_dir), args.strategies, vix, events, args.synthetic, args.random)
        return 0
    doc = load_results(settings.cache_dir)
    for s in leaderboard(doc):
        e = s["oos"]["expectancy"]
        print(f"{s['key']:<32} {s['oos']['trades']:>5}  {'n/a' if e is None else f'{e * 1e4:+.2f} bps':>11}  "
              f"{s['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
