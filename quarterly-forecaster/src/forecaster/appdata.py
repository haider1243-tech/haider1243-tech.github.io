"""Plain data for the mobile app's tabs (pure functions; app.py only renders).

* outlook_lines: a 3-line plain-English outlook from the snapshots.
* calendar_items: next N days of CPI/PPI/FOMC/jobs/GDP/earnings, with countdowns.
* gold_journal: latest TradingView alerts + verdicts + journal win rates.
* list_reports: weekly reports, newest first.
"""

from __future__ import annotations

import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

NY = "America/New_York"
CAL_CATEGORIES = ("CPI", "PPI", "FOMC", "Jobs", "GDP", "Earnings")
REGIME_WORDS = {"bull": "calm, rising", "bear": "calm, weaker", "high_vol": "high-volatility"}


def _pct(x: float) -> str:
    return f"{x:+.0%}" if abs(x) >= 0.095 else f"{x:+.1%}"


def regime_of(snap: dict | None) -> tuple[str, float] | None:
    cur = ((snap or {}).get("regime") or {}).get("current") or {}
    if not cur:
        return None
    k = max(cur, key=cur.get)
    return k, float(cur[k])


def outlook_lines(snaps: dict[str, dict], market: dict | None, prices: dict | None,
                  now: pd.Timestamp) -> list[str]:
    """Three short sentences. Ranges, never single-point calls."""
    from forecaster.webhook_server import event_time

    lines: list[str] = []
    spy = snaps.get("SPY")
    reg = regime_of(spy)
    vix = ((prices or {}).get("quotes") or {}).get("^VIX")
    if reg:
        k, p = reg
        txt = f"Markets look {REGIME_WORDS.get(k, k)} ({p:.0%} likely regime)"
        if vix:
            level = "low" if vix["price"] < 15 else "normal" if vix["price"] < 20 else "elevated" if vix["price"] < 30 else "high"
            txt += f", with VIX {level} at {vix['price']:.1f}"
        lines.append(txt + ".")
    elif vix:
        lines.append(f"VIX is {vix['price']:.1f}; no regime estimate yet.")
    else:
        lines.append("No market regime or VIX data yet.")

    if spy and spy.get("forecast"):
        f = spy["forecast"][-1]
        last = float(spy["last_close"])
        lines.append(f"S&P 500 by {f['quarter']}: middle case {_pct(f['p50'] / last - 1)}, "
                     f"likely range {_pct(f['p10'] / last - 1)} to {_pct(f['p90'] / last - 1)}.")
    else:
        lines.append("No S&P 500 forecast yet (it rebuilds every Sunday evening).")

    week_end = now + pd.Timedelta(days=7)
    evs = []
    for e in (market or {}).get("events", []):
        if e.get("impact") != "high" or e.get("category") not in CAL_CATEGORIES:
            continue
        t = event_time(e)
        if now < t <= week_end:
            evs.append((t, e))
    evs.sort(key=lambda x: x[0])
    if evs:
        names = ", ".join(f"{e['category'] if e['category'] != 'Jobs' else 'jobs report'} "
                          f"{t.tz_convert(NY):%a}" for t, e in evs[:3])
        lines.append(f"Busy week ahead: {names}. Expect bigger moves around those releases.")
    else:
        lines.append("Quiet calendar: no high-impact releases in the next 7 days.")
    return lines


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalItem:
    when: pd.Timestamp          # UTC
    name: str
    category: str
    impact: str
    detail: str
    countdown: str
    ticker: str | None


def countdown(delta: pd.Timedelta) -> str:
    s = delta.total_seconds()
    if s <= 0:
        return "now"
    d, rem = divmod(int(s), 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"in {d}d {h}h"
    if h:
        return f"in {h}h {m:02d}m"
    return f"in {m}m"


def calendar_items(market: dict | None, now: pd.Timestamp, days: int = 14) -> list[CalItem]:
    from forecaster.webhook_server import event_time

    out = []
    for e in (market or {}).get("events", []):
        if e.get("category") not in CAL_CATEGORIES:
            continue
        t = event_time(e)
        if now - pd.Timedelta(minutes=30) <= t <= now + pd.Timedelta(days=days):
            out.append(CalItem(t, e["name"], e["category"], e.get("impact", "low"), e.get("detail") or "",
                               countdown(t - now), e.get("ticker")))
    return sorted(out, key=lambda c: c.when)


# --------------------------------------------------------------------------- #
# Gold VWAP journal
# --------------------------------------------------------------------------- #

def gold_journal(db: Path, limit: int = 20) -> dict:
    """Latest alerts and win rates by verdict and horizon (signed return > 0)."""
    if not Path(db).exists():
        return {"alerts": [], "stats": [], "n_alerts": 0}
    with closing(sqlite3.connect(db)) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(
            "SELECT id, alert_time, ticker, strategy, side, price, verdict, reasons FROM alerts "
            "ORDER BY alert_time DESC, id DESC LIMIT ?", (limit,))]
        n = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        out = [dict(r) for r in con.execute(
            "SELECT a.verdict, o.horizon, COUNT(*) n, AVG(o.signed_ret) avg_ret, "
            "AVG(CASE WHEN o.signed_ret > 0 THEN 1.0 ELSE 0.0 END) win_rate FROM outcomes o "
            "JOIN alerts a ON a.id = o.alert_id WHERE o.status = 'done' GROUP BY a.verdict, o.horizon")]
    order = {"TAKE": 0, "CAUTION": 1, "SKIP": 2}
    hz = {"1h": 0, "4h": 1, "1d": 2}
    out.sort(key=lambda r: (order.get(r["verdict"], 9), hz.get(r["horizon"], 9)))
    import json

    for r in rows:
        try:
            r["reasons"] = json.loads(r["reasons"])
        except (TypeError, ValueError):
            r["reasons"] = []
    return {"alerts": rows, "stats": out, "n_alerts": int(n)}


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

_REPORT = re.compile(r"weekly_(\d{4}-\d{2}-\d{2})\.md$")


def list_reports(reports_dir: Path) -> list[dict]:
    """[{date, md, pdf}] newest first."""
    d = Path(reports_dir)
    if not d.exists():
        return []
    out = []
    for p in d.glob("weekly_*.md"):
        m = _REPORT.search(p.name)
        if m:
            pdf = p.with_suffix(".pdf")
            out.append({"date": pd.Timestamp(m.group(1)), "md": p, "pdf": pdf if pdf.exists() else None})
    return sorted(out, key=lambda r: r["date"], reverse=True)


def ago(ts: pd.Timestamp | None, now: pd.Timestamp) -> str:
    if ts is None:
        return "never"
    s = (now - ts).total_seconds()
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{s / 60:.0f} min ago"
    if s < 172800:
        return f"{s / 3600:.0f} h ago"
    return f"{s / 86400:.0f} days ago"


def fmt_signed_pct(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:+.2%}"
