"""Shadow (paper) trading of every lab setup on the live feed. NEVER places orders.

    python -m forecaster.shadow          # the launcher runs this while the app runs

Every 15 s, per instrument and timeframe: recent cached 1m history + live 1m bars (data/live.sqlite,
written by live_feed) -> session bars -> the same Features / signals / lab_engine.simulate as the
backtest, with each setup's latest walk-forward parameters (data/lab/results.json). Using the
identical code path is what makes live-vs-backtest comparable.

  * trades whose signal came after shadow mode started are upserted into data/shadow.sqlite
    (open until the engine exits them: stop, target or the time exit),
  * a signal on the latest completed bar (entry at the next bar's open) is a PENDING setup: the
    Signal Board shows it with that setup's historical stats,
  * weekly (Sunday job): live results vs the backtest's out-of-sample results -> DECAY flag when
    live expectancy falls below the backtest's 95% CI (20+ live trades) or a one-sided Welch test
    says live < backtest (p < 0.05).

This module deliberately has no access to forecaster.execution: shadow trades are bookkeeping.
"""

from __future__ import annotations

import json
import logging
import signal as _signal
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from forecaster import intraday as I
from forecaster import lab as L
from forecaster import lab_engine as E
from forecaster import strategies as S

log = logging.getLogger("shadow")
HISTORY_DAYS = 10
DECAY_MIN_TRADES = 20

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS shadow_trades (
    key TEXT, signal_time TEXT, strategy TEXT, instrument TEXT, timeframe TEXT, side INTEGER,
    entry_time TEXT, entry REAL, stop REAL, target REAL, exit_time TEXT, exit REAL, reason TEXT,
    net_ret REAL, r_mult REAL, status TEXT, verdict TEXT, params TEXT, updated_at TEXT,
    PRIMARY KEY (key, signal_time));
CREATE TABLE IF NOT EXISTS pending (
    key TEXT PRIMARY KEY, signal_time TEXT, strategy TEXT, instrument TEXT, timeframe TEXT, side INTEGER,
    ref REAL, stop REAL, target REAL, verdict TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS decay (
    week TEXT, key TEXT, live_trades INTEGER, live_exp REAL, bt_exp REAL, ci_low REAL, ci_high REAL,
    p_value REAL, flag TEXT, PRIMARY KEY (week, key));
"""


def shadow_db(cache_dir: Path) -> Path:
    return Path(cache_dir) / "shadow.sqlite"


def _connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def init_db(db: Path) -> None:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db)) as con, con:
        con.executescript(_SCHEMA)


def started_at(db: Path, now: pd.Timestamp) -> pd.Timestamp:
    """When shadow mode first started (persisted, so restarts don't re-label history as live)."""
    init_db(db)
    with closing(_connect(db)) as con, con:
        row = con.execute("SELECT v FROM meta WHERE k='started_at'").fetchone()
        if row:
            return pd.Timestamp(row[0])
        con.execute("INSERT INTO meta VALUES ('started_at', ?)", (now.isoformat(),))
        return now


# --------------------------------------------------------------------------- #
# Bars: cached history + live
# --------------------------------------------------------------------------- #

def live_1m(live_db: Path, symbol: str, since: pd.Timestamp) -> pd.DataFrame:
    from forecaster.live_feed import read_bars

    df = read_bars(live_db, symbol, int(since.value // 1_000_000))
    if df.empty:
        return df[["Open", "High", "Low", "Close", "Volume"]]
    vol = df["Ticks"] if I.INSTRUMENTS[symbol].kind == "fx" else df["Volume"]
    return df[["Open", "High", "Low", "Close"]].assign(Volume=vol.astype(float))


def combined_1m(store: I.IntradayStore | None, live_db: Path, symbol: str, now: pd.Timestamp,
                days: int = HISTORY_DAYS) -> pd.DataFrame:
    spec = I.spec_for(symbol)
    since = now - pd.Timedelta(days=days)
    hist = store.read(spec.source, symbol, since) if store is not None else pd.DataFrame()
    live = live_1m(live_db, symbol, since)
    parts = [p for p in (hist[["Open", "High", "Low", "Close", "Volume"]] if len(hist) else None, live)
             if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=I.COLS, index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]          # live wins where both exist
    df = df[df.index + pd.Timedelta(minutes=1) <= now]   # completed minutes only
    return I.session_bars(df, spec)


# --------------------------------------------------------------------------- #
# One pass
# --------------------------------------------------------------------------- #

def evaluate(bars_1m: pd.DataFrame, symbol: str, tf: str, setups: list[dict], since: pd.Timestamp,
             now: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    """-> (trades with signal >= since, pending setups on the latest completed bar)."""
    spec = I.spec_for(symbol)
    minutes = I.TIMEFRAMES[tf]
    bars = I.resample(bars_1m, minutes, spec)
    bars = bars[bars.index + pd.Timedelta(minutes=minutes) <= now]          # completed bars only
    if len(bars) < 3:
        return [], []
    f = S.Features(bars, spec, minutes)
    trades, pending = [], []
    for s in setups:
        st = S.LIBRARY[s["strategy"]].with_params(**s["params"])
        costs = E.CostModel(**s["costs"]) if s.get("costs") else E.DEFAULT_COSTS[symbol]
        pp = E.prepare(bars, spec, minutes, costs)
        sig = st.signals(f)
        max_hold, max_trades = None, int(st.params.get("max_trades", 99))
        if s.get("sim"):                    # a Simulator forward test: same filters and overrides as its backtest
            from forecaster.simulator import SimConfig, adjust_signals

            cfg = SimConfig.from_dict({**s["sim"], "start": "", "end": ""})
            sig = adjust_signals(sig, bars, cfg, None, minutes)
            max_hold, max_trades = cfg.max_hold_bars, int(cfg.max_trades_per_day)
        tr = E.simulate(pp, sig, costs, max_trades, allow_open=True, max_hold=max_hold)
        tr = tr[pd.DatetimeIndex(tr["signal_time"]) >= since] if len(tr) else tr
        base = {"key": s["key"], "strategy": s["strategy"], "instrument": symbol, "timeframe": tf,
                "verdict": s["verdict"], "params": json.dumps(s["params"], sort_keys=True)}
        for _, t in tr.iterrows():
            trades.append({**base, "signal_time": t["signal_time"].isoformat(), "side": int(t["side"]),
                           "entry_time": t["entry_time"].isoformat(), "entry": float(t["entry"]),
                           "stop": float(t["stop"]), "target": float(t["target"]),
                           "exit_time": None if t["reason"] == "open" else t["exit_time"].isoformat(),
                           "exit": None if t["reason"] == "open" else float(t["exit"]), "reason": t["reason"],
                           "net_ret": None if t["reason"] == "open" else float(t["net_ret"]),
                           "r_mult": None if t["reason"] == "open" else float(t["r_mult"]),
                           "status": "open" if t["reason"] == "open" else "closed"})
        last = len(bars) - 1
        if len(sig) and sig.index[-1] == last and f.in_window[last]:
            busy = bool(len(tr)) and bool((tr["reason"] == "open").any())
            if not busy and bars.index[-1] >= since:
                row = sig.iloc[-1]
                pending.append({**base, "signal_time": bars.index[last].isoformat(), "side": int(row["side"]),
                                "ref": float(bars["Close"].iloc[-1]), "stop": float(row["stop"]),
                                "target": float(row["target"])})
    return trades, pending


def write(db: Path, trades: list[dict], pending: list[dict], keys: list[str], now: pd.Timestamp) -> None:
    init_db(db)
    with closing(_connect(db)) as con, con:
        con.executemany("""INSERT INTO shadow_trades (key, signal_time, strategy, instrument, timeframe, side,
            entry_time, entry, stop, target, exit_time, exit, reason, net_ret, r_mult, status, verdict, params, updated_at)
            VALUES (:key,:signal_time,:strategy,:instrument,:timeframe,:side,:entry_time,:entry,:stop,:target,
            :exit_time,:exit,:reason,:net_ret,:r_mult,:status,:verdict,:params,:updated_at)
            ON CONFLICT(key, signal_time) DO UPDATE SET exit_time=excluded.exit_time, exit=excluded.exit,
            reason=excluded.reason, net_ret=excluded.net_ret, r_mult=excluded.r_mult, status=excluded.status,
            updated_at=excluded.updated_at WHERE shadow_trades.status='open'""",
                        [{**t, "updated_at": now.isoformat()} for t in trades])
        if keys:
            con.execute(f"DELETE FROM pending WHERE key IN ({','.join('?' * len(keys))})", keys)
        con.executemany("""INSERT OR REPLACE INTO pending (key, signal_time, strategy, instrument, timeframe, side,
            ref, stop, target, verdict, updated_at) VALUES (:key,:signal_time,:strategy,:instrument,:timeframe,:side,
            :ref,:stop,:target,:verdict,:updated_at)""", [{**p, "updated_at": now.isoformat()} for p in pending])
        con.execute("INSERT OR REPLACE INTO meta VALUES ('heartbeat', ?)", (now.isoformat(),))


def run_once(cache_dir: Path, now: pd.Timestamp | None = None, doc: dict | None = None,
             store: I.IntradayStore | None = None) -> dict:
    from forecaster.clock import now as _now

    now = now or _now()
    doc = doc if doc is not None else L.load_results(cache_dir)
    db = shadow_db(cache_dir)
    since = started_at(db, now)
    from forecaster.simulator import forward_tests as _fwd

    if (not doc or not doc.get("setups")) and not _fwd(cache_dir):
        write(db, [], [], [], now)
        return {"trades": 0, "pending": 0, "setups": 0}
    doc = doc or {"setups": []}
    from forecaster.live_feed import live_db

    store = store if store is not None else I.IntradayStore(Path(cache_dir) / "intraday")
    groups: dict[tuple[str, str], list[dict]] = {}
    from forecaster.simulator import forward_tests

    for s in [*doc["setups"], *forward_tests(cache_dir)]:
        groups.setdefault((s["instrument"], s["timeframe"]), []).append(s)
    all_tr, all_pend = [], []
    for sym in {k[0] for k in groups}:
        m1 = combined_1m(store, live_db(cache_dir), sym, now)
        for (s2, tf), setups in groups.items():
            if s2 != sym:
                continue
            tr, pend = evaluate(m1, sym, tf, setups, since, now)
            all_tr += tr
            all_pend += pend
    keys = [s["key"] for v in groups.values() for s in v]
    write(db, all_tr, all_pend, keys, now)
    return {"trades": len(all_tr), "pending": len(all_pend), "setups": len(keys)}


# --------------------------------------------------------------------------- #
# Readers + decay
# --------------------------------------------------------------------------- #

def _rows(db: Path, sql: str, params=()) -> list[dict]:
    if not Path(db).exists():
        return []
    try:
        with closing(_connect(db)) as con:
            return [dict(r) for r in con.execute(sql, params)]
    except sqlite3.Error:
        return []


def read_trades(db: Path, status: str | None = None) -> pd.DataFrame:
    rows = _rows(db, "SELECT * FROM shadow_trades" + (" WHERE status=?" if status else "") +
                 " ORDER BY signal_time DESC", (status,) if status else ())
    return pd.DataFrame(rows)


def read_pending(db: Path, max_age: pd.Timedelta, now: pd.Timestamp) -> list[dict]:
    rows = _rows(db, "SELECT * FROM pending ORDER BY signal_time DESC")
    return [r for r in rows if now - pd.Timestamp(r["signal_time"]) <= max_age]


def heartbeat(db: Path) -> pd.Timestamp | None:
    r = _rows(db, "SELECT v FROM meta WHERE k='heartbeat'")
    return pd.Timestamp(r[0]["v"]) if r else None


def decay_check(live: np.ndarray, bt: np.ndarray, ci: tuple | list | None) -> tuple[str, float | None]:
    """DECAY if live expectancy is below the backtest CI (20+ live trades) or a one-sided Welch
    t-test says live < backtest at p < 0.05. OK / TOO FEW otherwise."""
    from scipy import stats

    live, bt = np.asarray(live, float), np.asarray(bt, float)
    if len(live) < DECAY_MIN_TRADES or len(bt) < 2:
        return "TOO FEW", None
    p = float(stats.ttest_ind(live, bt, equal_var=False, alternative="less").pvalue)
    below = ci is not None and live.mean() < ci[0]
    return ("DECAY" if (below or p < 0.05) else "OK"), p


def weekly_decay(cache_dir: Path, now: pd.Timestamp | None = None, doc: dict | None = None) -> list[dict]:
    """Compare all closed shadow trades so far with the backtest's OOS trades, per setup; store a row
    per (week, setup)."""
    now = now or pd.Timestamp.now(tz="UTC")
    doc = doc if doc is not None else L.load_results(cache_dir)
    db = shadow_db(cache_dir)
    init_db(db)
    live = read_trades(db, "closed")
    week = now.tz_convert(I.NY).strftime("%G-W%V")
    out = []
    for s in (doc or {}).get("setups", []):
        lt = live[live["key"] == s["key"]]["net_ret"].dropna().to_numpy(float) if len(live) else np.array([])
        if len(lt) == 0:
            continue
        bt = L.load_oos_trades(cache_dir, s["key"])["net_ret"].dropna().to_numpy(float)
        flag, p = decay_check(lt, bt, s.get("ci"))
        out.append({"week": week, "key": s["key"], "live_trades": len(lt), "live_exp": float(lt.mean()),
                    "bt_exp": s["oos"]["expectancy"], "ci_low": (s.get("ci") or [None, None])[0],
                    "ci_high": (s.get("ci") or [None, None])[1], "p_value": p, "flag": flag})
    with closing(_connect(db)) as con, con:
        con.executemany("""INSERT OR REPLACE INTO decay VALUES (:week,:key,:live_trades,:live_exp,:bt_exp,:ci_low,
            :ci_high,:p_value,:flag)""", out)
    return out


def decay_text(rows: list[dict]) -> str | None:
    import html

    if not rows:
        return None
    lines = ["<b>Shadow trading vs backtest</b>"]
    for r in sorted(rows, key=lambda r: (r["flag"] != "DECAY", -r["live_trades"])):
        icon = {"DECAY": "🔻", "OK": "✅"}.get(r["flag"], "⏳")
        bt = "n/a" if r["bt_exp"] is None else f"{r['bt_exp'] * 1e4:+.1f}"
        lines.append(f"{icon} {html.escape(r['key'])}: live {r['live_exp'] * 1e4:+.1f} bps over "
                     f"{r['live_trades']} trades vs backtest {bt} bps ({r['flag']})")
    return "\n".join(lines[:25])


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - long-running
    import argparse

    from forecaster.config import PROJECT_ROOT, load_settings
    from forecaster.monitor import setup_logging

    ap = argparse.ArgumentParser(description="Shadow-trade every lab setup on the live feed (no orders)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=15.0)
    args = ap.parse_args(argv)
    setup_logging(PROJECT_ROOT / "logs", "shadow")
    settings = load_settings()
    stop = {"flag": False}
    for sig in (_signal.SIGTERM, _signal.SIGINT):
        _signal.signal(sig, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        try:
            r = run_once(settings.cache_dir)
            log.info("shadow pass: %s", r)
        except Exception as exc:
            log.exception("shadow pass failed: %s", exc)
        if args.once:
            break
        t_end = time.time() + args.every
        while time.time() < t_end and not stop["flag"]:
            time.sleep(0.5)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
