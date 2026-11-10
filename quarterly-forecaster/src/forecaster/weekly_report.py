"""Sunday weekly report: Markdown + PDF in reports/, short summary to Telegram.

    python -m forecaster.weekly_report            # refresh data, build, save, send
    python -m forecaster.weekly_report --no-refresh --no-send --date 2026-11-08

Sections
  1. Last week's moves: SPY, QQQ, gold (GC=F) in CAD, VIX in points, plus USD/CAD
     (Friday close vs the previous Friday's close; holidays use the last trading day).
  2. Macro data released last week: CPI/PPI/jobs/GDP release dates (FRED) and FOMC
     decisions, with the latest value of each series from the market snapshot.
  3. Next week's high-impact events, with dates and typical release times (ET).
  4. 4-quarter forecasts vs last week: P10/P50/P90 for each ticker snapshot,
     matched by quarter, compared with last week's saved state.
  5. Sentiment shifts: 7-day average news sentiment vs the week before, and shocks.
  6. Portfolio changes: value, P&L, holdings and weights, flags, forecast vs last week.

Each run saves reports/state/<date>.json, which the next run compares against.
The reports contain your portfolio, so reports/ is gitignored.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from forecaster.config import PROJECT_ROOT
from forecaster.market_data import USD_CAD_SYMBOL, rate_on_dates, ticker_currency, to_cad

REPORTS_DIR = PROJECT_ROOT / "reports"
NY = "America/New_York"
MOVERS: tuple[tuple[str, str], ...] = (("SPY", "S&P 500 (SPY)"), ("QQQ", "Nasdaq 100 (QQQ)"),
                                       ("GC=F", "Gold (GC=F)"), ("^VIX", "VIX"))
# Calendar category -> series in the market snapshot's macro table
RELEASE_SERIES = {"CPI": ("CPIAUCSL", "CPILFESL"), "PPI": ("PPIACO",), "Jobs": ("UNRATE",),
                  "GDP": ("GDP",), "FOMC": ("FEDFUNDS",)}
CAVEAT = ("Forecasts are probability ranges, not predictions. In the 2005-2022 backtest the ensemble "
          "did not reliably beat a naive historical baseline.")


# --------------------------------------------------------------------------- #
# Report model (rendered to both Markdown and PDF)
# --------------------------------------------------------------------------- #

@dataclass
class Table:
    headers: list[str]
    rows: list[list[Any]]


@dataclass
class Section:
    title: str
    blocks: list[Any] = field(default_factory=list)   # str paragraph | list[str] bullets | Table


@dataclass
class Report:
    run_date: pd.Timestamp
    week_start: pd.Timestamp
    week_end: pd.Timestamp
    sections: list[Section]
    state: dict
    summary: list[str]

    @property
    def title(self) -> str:
        return f"Weekly report: week of {self.week_start:%b %d} - {self.week_end:%b %d, %Y}"


def week_window(run_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(Monday, Friday) of the last complete trading week on or before run_date."""
    d = pd.Timestamp(run_date).normalize()
    fri = d - pd.Timedelta(days=(d.day_of_week - 4) % 7)   # Sun -> 2 days back, Mon -> 3, Fri -> same day
    return fri - pd.Timedelta(days=4), fri


def _fmt_pct(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:+.{nd}%}"


def _fmt_money(x: float | None, nd: int = 0) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"C${x:,.{nd}f}"


def _signed_money(x: float | None, nd: int = 0) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{'+' if x >= 0 else '-'}C${abs(x):,.{nd}f}"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def weekly_moves(closes: dict[str, pd.Series], usd_cad: pd.Series | None,
                 week_start: pd.Timestamp, week_end: pd.Timestamp) -> tuple[Section, dict]:
    """Close at the week's last trading day vs the last close before the week."""
    rows, state = [], {}
    series = dict(closes)
    for t, label in (*MOVERS, (USD_CAD_SYMBOL, "USD/CAD")):
        s = series.get(t)
        if s is None and t == USD_CAD_SYMBOL and usd_cad is not None:
            s = usd_cad
        if s is None or s.dropna().empty:
            rows.append([label, "no data", "", "", ""])
            continue
        s = s.dropna().sort_index()
        unit = ticker_currency(t)
        if unit == "USD":
            if usd_cad is None:
                rows.append([label, "no USD/CAD rate", "", "", ""])
                continue
            s = to_cad(s.to_frame("Close"), t, usd_cad)["Close"]
        before, during = s.loc[: week_start - pd.Timedelta(days=1)], s.loc[week_start:week_end]
        if before.empty or during.empty:
            rows.append([label, "no data for the week", "", "", ""])
            continue
        p0, p1 = float(before.iloc[-1]), float(during.iloc[-1])
        hi, lo = float(during.max()), float(during.min())
        if unit == "USD" or unit == "CAD":
            fmt = lambda v: f"C${v:,.2f}"  # noqa: E731
            chg = _fmt_pct(p1 / p0 - 1, 2)
        elif t == USD_CAD_SYMBOL:
            fmt = lambda v: f"{v:.4f}"  # noqa: E731
            chg = _fmt_pct(p1 / p0 - 1, 2)
        else:
            fmt = lambda v: f"{v:.2f}"  # noqa: E731
            chg = f"{p1 - p0:+.2f} pts"
        rows.append([label, fmt(p1), chg, fmt(lo), fmt(hi)])
        state[t] = {"close": p1, "prev_close": p0, "change": p1 / p0 - 1 if p0 else None}
    sec = Section("Last week's moves", [
        Table(["Market", "Friday close", "Week change", "Lowest close", "Highest close"], rows),
        "Prices in CAD (USD converted at each day's USD/CAD); VIX in index points. "
        "A falling USD/CAD lowers the CAD value of US holdings.",
    ])
    return sec, state


def macro_released(events: Iterable, macro_table: list[dict], names: dict[str, str] | None = None) -> Section:
    """Past-week release events, with the latest value of the matching series."""
    by_id = {r.get("series_id"): r for r in (macro_table or [])}
    rows = []
    for e in sorted(events, key=lambda e: _get(e, "date")):
        cat = _get(e, "category")
        for sid in RELEASE_SERIES.get(cat, ()) or (None,):
            r = by_id.get(sid) if sid else None
            if r is None:
                val = "value not in snapshot" if sid else ""
                rows.append([f"{pd.Timestamp(_get(e, 'date')):%a %b %d}", _get(e, "name"), sid or "", val, "", ""])
                continue
            unit = "%" if r.get("units") == "pct" else " pp"
            latest = (_num(r.get("value"), "{:,.2f}") if sid in ("FEDFUNDS", "UNRATE")
                      else _num(r.get("yoy"), "YoY {:+.2f}%"))
            obs = f" ({pd.Timestamp(r['date']):%b %Y})" if r.get("date") else ""
            rows.append([f"{pd.Timestamp(_get(e, 'date')):%a %b %d}", _get(e, "name"), r.get("name", sid),
                         latest + obs, _num(r.get("trend_3m"), "{:+.2f}" + unit), _num(r.get("surprise_z"), "{:+.1f}")])
    blocks: list[Any] = []
    if rows:
        blocks.append(Table(["Released", "Release", "Series", "Latest", "3-month trend", "Surprise z"], rows))
        blocks.append("Latest = most recent observation in the macro snapshot (refresh before the report). "
                      "Surprise z compares the latest change with its prior 3-month trend; no consensus data.")
    else:
        blocks.append("No CPI, PPI, jobs, GDP or FOMC releases last week (or FRED release dates unavailable).")
    return Section("Macro data released last week", blocks)


def _num(x: Any, fmt: str) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return fmt.format(v) if math.isfinite(v) else "n/a"


def _get(e: Any, k: str) -> Any:
    return e[k] if isinstance(e, dict) else getattr(e, k)


def next_week_events(market: dict | None, week_end: pd.Timestamp) -> tuple[Section, list[dict]]:
    from forecaster.webhook_server import event_time

    start = week_end + pd.Timedelta(days=3)               # next Monday
    end = start + pd.Timedelta(days=4)
    evs = []
    for e in (market or {}).get("events", []):
        d = pd.Timestamp(e["date"]).normalize()
        if start <= d <= end and e.get("impact") == "high":
            t = event_time(e).tz_convert(NY)
            evs.append({**e, "time_et": f"{t:%H:%M}"})
    evs.sort(key=lambda e: (e["date"], e["time_et"]))
    if not evs:
        msg = ("No high-impact events listed for next week." if market else
               "No events snapshot: refresh data to see next week's calendar.")
        return Section(f"Next week's high-impact events ({start:%b %d} - {end:%b %d})", [msg]), []
    rows = [[f"{pd.Timestamp(e['date']):%a %b %d}", f"{e['time_et']} ET", e["name"], e.get("detail") or ""]
            for e in evs]
    return Section(f"Next week's high-impact events ({start:%b %d} - {end:%b %d})", [
        Table(["Date", "Time", "Event", "Detail"], rows),
        "Times are typical release times (CPI/PPI/jobs/GDP 08:30, FOMC statement 14:00).",
    ]), evs


def forecast_changes(snaps: dict[str, dict], prev: dict | None) -> tuple[Section, dict]:
    prev_fc = (prev or {}).get("forecasts", {})
    rows, state, notable = [], {}, []
    for t in sorted(snaps):
        s = snaps[t]
        unit = "C$" if s.get("currency") == "CAD" else ""
        quarters = {f["quarter"]: f for f in s.get("forecast", [])}
        state[t] = {"as_of": str(s.get("as_of")), "last_close": s.get("last_close"), "currency": s.get("currency"),
                    "quarters": {q: {k: f[k] for k in ("p10", "p50", "p90")} for q, f in quarters.items()}}
        old = prev_fc.get(t, {}).get("quarters", {})
        for q, f in quarters.items():
            o = old.get(q)
            d50 = (f["p50"] / o["p50"] - 1) if o and o.get("p50") else None
            w_now = (f["p90"] - f["p10"]) / f["p50"] if f["p50"] else None
            w_old = (o["p90"] - o["p10"]) / o["p50"] if o and o.get("p50") else None
            rows.append([t, q, f"{unit}{f['p10']:,.2f}", f"{unit}{f['p50']:,.2f}", f"{unit}{f['p90']:,.2f}",
                         "new" if o is None else _fmt_pct(d50),
                         "" if w_old is None or w_now is None else f"{(w_now - w_old) * 100:+.1f} pts"])
            if d50 is not None:
                notable.append((abs(d50), t, q, d50))
    blocks: list[Any] = []
    if rows:
        blocks.append(Table(["Ticker", "Quarter", "P10", "P50", "P90", "P50 vs last week", "Band width change"], rows))
        blocks.append("Band width = (P90 - P10) / P50. 'new' = quarter not in last week's report "
                      "(the horizon rolls forward each quarter). " + CAVEAT)
    else:
        blocks.append("No forecast snapshots: run the pipeline (python -m forecaster.pipeline).")
    if not prev:
        blocks.append("First report: no previous week to compare with yet.")
    notable.sort(reverse=True)
    return Section("4-quarter forecasts vs last week", blocks), {"forecasts": state, "_notable": notable[:3]}


def _avg(rows: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, int]:
    w = [(r["sentiment"], r.get("headlines") or 0) for r in rows
         if start <= pd.Timestamp(r["date"]) <= end and r.get("sentiment") is not None]
    n = sum(h for _, h in w)
    return (sum(s * h for s, h in w) / n if n else float("nan")), int(n)


def sentiment_shifts(market: dict | None, week_start: pd.Timestamp, week_end: pd.Timestamp,
                     top: int = 8) -> tuple[Section, dict]:
    sent = (market or {}).get("sentiment") or {}
    if not sent:
        return Section("Sentiment shifts", ["No scored headlines yet (needs FINNHUB_API_KEY and FinBERT)."]), {}
    end, start = week_end + pd.Timedelta(days=2), week_start - pd.Timedelta(days=1)   # Sun..Sat window
    pstart, pend = start - pd.Timedelta(days=7), start - pd.Timedelta(days=1)
    rows, state = [], {}
    for tag, data in sent.items():
        now, n_now = _avg(data, start, end)
        before, n_before = _avg(data, pstart, pend)
        if n_now == 0:
            continue
        shift = now - before if math.isfinite(before) else float("nan")
        state[tag] = {"sentiment": now, "headlines": n_now}
        rows.append((abs(shift) if math.isfinite(shift) else -1, tag, now, before, shift, n_now))
    rows.sort(reverse=True)
    shocks = [s for s in (market or {}).get("shocks", []) if start <= pd.Timestamp(s["date"]) <= end]
    blocks: list[Any] = [Table(
        ["Source", "This week", "Week before", "Shift", "Headlines"],
        [["Market (general)" if t == "MARKET" else t, f"{n:+.2f}",
          "n/a" if not math.isfinite(b) else f"{b:+.2f}", "n/a" if not math.isfinite(s) else f"{s:+.2f}", h]
         for _, t, n, b, s, h in rows[:top]])]
    if shocks:
        blocks.append([f"Shock: {s['tag']} on {str(s['date'])[:10]} (sentiment {s['sentiment']:+.2f}, z {s['z']:+.1f})"
                       for s in shocks[:6]])
    blocks.append("Sentiment = FinBERT P(positive) - P(negative), -1 to +1, headline-weighted. "
                  "Shocks are |z| > 2; about 5% of quiet days trigger one by chance.")
    return Section("Sentiment shifts", blocks), {"sentiment": state, "_shocks": len(shocks)}


def portfolio_changes(pf: dict | None, prev: dict | None) -> tuple[Section, dict]:
    if not pf:
        return Section("Portfolio", ["No portfolio snapshot (create portfolio.yaml and refresh)."]), {}
    old = (prev or {}).get("portfolio")
    holdings = {}
    for h in pf["holdings"]:
        k = f"{h['ticker']}|{h['account']}"
        holdings[k] = {"ticker": h["ticker"], "account": h["account"], "shares": h.get("shares"),
                       "value": h["market_value"], "weight": h["weight"]}
    fc = pf.get("forecast") or []
    state = {"total_value": pf["total_value"], "total_cost": pf["total_cost"], "total_pnl": pf["total_pnl"],
             "holdings": holdings, "flags": [f["ticker"] for f in pf.get("flags", [])],
             "forecast": {f["quarter"]: {k: f[k] for k in ("p10", "p50", "p90")} for f in fc},
             "volatility": (pf.get("risk") or {}).get("volatility")}
    blocks: list[Any] = []
    dv = pf["total_value"] - old["total_value"] if old else None
    lines = [f"Market value {_fmt_money(pf['total_value'])}"
             + (f" ({_signed_money(dv)}, {_fmt_pct(pf['total_value'] / old['total_value'] - 1)} vs last week)" if old else ""),
             f"Unrealised P&L {_signed_money(pf['total_pnl'])} on cost {_fmt_money(pf['total_cost'])}"]
    if old:
        dc = pf["total_cost"] - old["total_cost"]
        if abs(dc) > 0.5:
            lines.append(f"Cost base changed {_signed_money(dc)} (buys/sells or edits to portfolio.yaml), "
                         f"so market moves explain {_signed_money(dv - dc)} of the change")
    if state["volatility"] is not None:
        lines.append(f"Volatility {state['volatility']:.1%}")
    blocks.append(lines)

    rows = []
    keys = sorted(set(holdings) | set((old or {}).get("holdings", {})))
    for k in keys:
        n, o = holdings.get(k), (old or {}).get("holdings", {}).get(k)
        t, acct = k.split("|")
        if n and o:
            ds = (n["shares"] or 0) - (o["shares"] or 0) if n["shares"] is not None and o["shares"] is not None else 0
            status = "" if abs(ds) < 1e-9 else f"{ds:+g} shares"
        else:
            status = "new" if n else "sold / removed"
        rows.append([t, acct, _fmt_money(n["value"] if n else 0.0), f"{(n or {'weight': 0})['weight']:.1%}",
                     "" if not (n and o) else f"{(n['weight'] - o['weight']) * 100:+.1f} pts",
                     "" if not (n and o) else _signed_money(n["value"] - o["value"]), status])
    blocks.append(Table(["Holding", "Account", "Value", "Weight", "Weight change", "Value change", "Change"], rows))

    flags_now = set(state["flags"])
    flags_old = set((old or {}).get("flags", []))
    fl = [f["message"] for f in pf.get("flags", [])]
    fl += [f"Resolved: {t} is back under the concentration limit." for t in sorted(flags_old - flags_now)]
    if fl:
        blocks.append(["Flag: " + m if not m.startswith("Resolved") else m for m in fl])

    if fc:
        oldfc = (old or {}).get("forecast", {})
        frows = [[f["quarter"], _fmt_money(f["p10"]), _fmt_money(f["p50"]), _fmt_money(f["p90"]),
                  _fmt_pct(f["p50"] / oldfc[f["quarter"]]["p50"] - 1) if f["quarter"] in oldfc else "new"]
                 for f in fc]
        blocks.append(Table(["Quarter", "P10", "P50", "P90", "P50 vs last week"], frows))
        blocks.append("Portfolio range: joint block bootstrap of daily CAD returns, buy-and-hold; not backtested.")
    return Section("Portfolio changes", blocks), {"portfolio": state, "_dv": dv}


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #

def build_report(run_date: pd.Timestamp, closes: dict[str, pd.Series], usd_cad: pd.Series | None,
                 market: dict | None, ticker_snaps: dict[str, dict], portfolio: dict | None,
                 past_events: Iterable, prev_state: dict | None) -> Report:
    ws, we = week_window(run_date)
    s_moves, st_moves = weekly_moves(closes, usd_cad, ws, we)
    s_macro = macro_released(past_events, (market or {}).get("macro_table", []))
    s_next, evs = next_week_events(market, we)
    s_fc, st_fc = forecast_changes(ticker_snaps, prev_state)
    s_sent, st_sent = sentiment_shifts(market, ws, we)
    s_pf, st_pf = portfolio_changes(portfolio, prev_state)

    summary = [f"<b>Weekly report</b>: {ws:%b %d} - {we:%b %d}"]
    mv = [f"{lbl.split(' (')[0]} {_fmt_pct(st_moves[t]['change'])}" if t != "^VIX" else
          f"VIX {st_moves[t]['close'] - st_moves[t]['prev_close']:+.1f}"
          for t, lbl in MOVERS if t in st_moves]
    if mv:
        summary.append("Moves (CAD): " + ", ".join(mv))
    if USD_CAD_SYMBOL in st_moves:
        summary.append(f"USD/CAD {st_moves[USD_CAD_SYMBOL]['close']:.4f} ({_fmt_pct(st_moves[USD_CAD_SYMBOL]['change'], 2)})")
    if evs:
        summary.append("Next week: " + "; ".join(f"{pd.Timestamp(e['date']):%a} {e['category']}" for e in evs[:5]))
    else:
        summary.append("Next week: no high-impact events listed")
    if st_pf.get("portfolio"):
        p = st_pf["portfolio"]
        summary.append(f"Portfolio {_fmt_money(p['total_value'])}"
                       + (f" ({_signed_money(st_pf['_dv'])})" if st_pf.get("_dv") is not None else ""))
        if p["flags"]:
            summary.append("Over 25%: " + ", ".join(p["flags"]))
    if st_fc.get("_notable"):
        _, t, q, d = st_fc["_notable"][0]
        summary.append(f"Biggest forecast change: {t} {q} P50 {_fmt_pct(d)}")
    if st_sent.get("_shocks"):
        summary.append(f"Sentiment shocks this week: {st_sent['_shocks']}")

    state = {"run_date": str(pd.Timestamp(run_date).date()), "week_start": str(ws.date()),
             "week_end": str(we.date()), "moves": st_moves,
             **{k: v for k, v in {**st_fc, **st_sent, **st_pf}.items() if not k.startswith("_")}}
    return Report(pd.Timestamp(run_date), ws, we, [s_moves, s_macro, s_next, s_fc, s_sent, s_pf], state, summary)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _md_cell(x: Any) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def render_markdown(r: Report) -> str:
    out = [f"# {r.title}", "", f"Generated {r.run_date:%A %b %d, %Y}. All money values in CAD.", ""]
    for s in r.sections:
        out += [f"## {s.title}", ""]
        for b in s.blocks:
            if isinstance(b, Table):
                out.append("| " + " | ".join(map(_md_cell, b.headers)) + " |")
                out.append("|" + "|".join("---" for _ in b.headers) + "|")
                out += ["| " + " | ".join(map(_md_cell, row)) + " |" for row in b.rows]
            elif isinstance(b, list):
                out += [f"- {x}" for x in b]
            else:
                out.append(str(b))
            out.append("")
    out += ["---", CAVEAT, ""]
    return "\n".join(out)


# Built-in PDF fonts (WinAnsi) lack these glyphs; map them to ASCII so nothing renders as a black box.
_PDF_SUBS = {"≈": "~", "⚠": "!", "→": "->", "←": "<-", "−": "-", "≥": ">=", "≤": "<=", "✅": "", "⛔": "",
             "•": "-", "’": "'", "“": '"', "”": '"'}


def pdf_safe(text: str) -> str:
    s = str(text)
    for a, b in _PDF_SUBS.items():
        s = s.replace(a, b)
    return s.encode("cp1252", errors="replace").decode("cp1252")


def render_pdf(r: Report, path: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table as RLTable, TableStyle

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8.5, leading=11)
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7.5, leading=9)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.HexColor("#52514e"))
    esc = lambda s: html.escape(pdf_safe(s))  # noqa: E731

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch, title=pdf_safe(r.title),
                            author="quarterly-forecaster")
    story = [Paragraph(esc(r.title), styles["Title"]),
             Paragraph(esc(f"Generated {r.run_date:%A %b %d, %Y}. All money values in CAD."), small),
             Spacer(1, 10)]
    width = letter[0] - 1.2 * inch
    for s in r.sections:
        story += [Paragraph(esc(s.title), styles["Heading2"]), Spacer(1, 3)]
        for b in s.blocks:
            if isinstance(b, Table):
                data = [[Paragraph(esc(h), head) for h in b.headers]] + \
                       [[Paragraph(esc(c), cell) for c in row] for row in b.rows]
                t = RLTable(data, colWidths=[width / len(b.headers)] * len(b.headers), repeatRows=1)
                t.setStyle(TableStyle([
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#c3c2b7")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e1e0d9")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
                story += [t, Spacer(1, 5)]
            elif isinstance(b, list):
                story += [Paragraph(esc("- " + x), small) for x in b] + [Spacer(1, 4)]
            else:
                story += [Paragraph(esc(b), small), Spacer(1, 4)]
    story += [Spacer(1, 8), Paragraph(esc(CAVEAT), small)]
    doc.build(story)
    return path


def summary_text(r: Report, md_path: Path | None = None) -> str:
    lines = list(r.summary)
    if md_path is not None:
        lines.append(f"Full report: {md_path.name}")
    return "\n".join(lines[:1] + [html.escape(x) for x in lines[1:]])


# --------------------------------------------------------------------------- #
# Saving / state
# --------------------------------------------------------------------------- #

def previous_state(reports_dir: Path, before: pd.Timestamp) -> dict | None:
    d = Path(reports_dir) / "state"
    if not d.exists():
        return None
    cands = sorted(p for p in d.glob("*.json") if p.stem < str(pd.Timestamp(before).date()))
    return json.loads(cands[-1].read_text()) if cands else None


def save_report(r: Report, reports_dir: Path = REPORTS_DIR, pdf: bool = True) -> dict[str, Path]:
    from forecaster.pipeline import write_json

    reports_dir = Path(reports_dir)
    (reports_dir / "state").mkdir(parents=True, exist_ok=True)
    stem = f"weekly_{r.run_date:%Y-%m-%d}"
    paths = {"markdown": reports_dir / f"{stem}.md", "state": reports_dir / "state" / f"{r.run_date:%Y-%m-%d}.json"}
    paths["markdown"].write_text(render_markdown(r))
    write_json(paths["state"], r.state)
    if pdf:
        paths["pdf"] = render_pdf(r, reports_dir / f"{stem}.pdf")
    return paths


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def gather_inputs(settings, run_date: pd.Timestamp, warn=print) -> dict:  # pragma: no cover - I/O
    """Read cached prices + snapshots; fetch last week's release dates from FRED if possible."""
    from forecaster.cache import Cache
    from forecaster.calendar import MACRO_RELEASES, fomc_events, fred_release_events
    from forecaster.market_data import ParquetStore
    from forecaster.pipeline import read_json, snapshot_dir
    from forecaster.sources.fred import FredClient

    store = ParquetStore(Path(settings.cache_dir) / "ohlcv")
    closes = {}
    for t, _ in (*MOVERS, (USD_CAD_SYMBOL, "")):
        df = store.read(t)
        if df is None:
            warn(f"no cached prices for {t}")
        else:
            closes[t] = df["Close"]
    usd_cad = closes.get(USD_CAD_SYMBOL)
    sd = snapshot_dir(settings)
    snaps = {s["ticker"]: s for p in sorted(sd.glob("ticker_*.json")) if (s := read_json(p))}
    ws, we = week_window(run_date)
    past = list(fomc_events(ws, we + pd.Timedelta(hours=23)))
    if settings.fred_api_key:
        try:
            fred = FredClient(settings.fred_api_key, Cache(settings.cache_dir, settings.cache_ttl_hours * 3600))
            past += fred_release_events(fred, ws, we, MACRO_RELEASES, on_error=lambda c, e: warn(f"{c}: {e}"))
        except Exception as exc:
            warn(f"FRED release dates unavailable: {exc}")
    else:
        warn("FRED_API_KEY not set: only FOMC decisions listed under macro releases")
    return {"closes": closes, "usd_cad": usd_cad, "market": read_json(sd / "market.json"),
            "ticker_snaps": snaps, "portfolio": read_json(sd / "portfolio.json"), "past_events": past}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import sys

    from forecaster.config import load_settings
    from forecaster.webhook_server import TelegramNotifier

    ap = argparse.ArgumentParser(description="Build the Sunday weekly report")
    ap.add_argument("--date", help="report date (default today)")
    ap.add_argument("--no-refresh", action="store_true", help="use cached data/snapshots as they are")
    ap.add_argument("--no-send", action="store_true", help="don't send the Telegram summary")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-status", action="store_true", help="skip the project status check (STATUS.md)")
    ap.add_argument("--out", default=str(REPORTS_DIR))
    args = ap.parse_args(argv)
    warn = lambda m: print(f"[warn] {m}", file=sys.stderr)  # noqa: E731

    settings = load_settings()
    run_date = pd.Timestamp(args.date or pd.Timestamp.today()).normalize()
    if not args.no_refresh:
        from forecaster.market_data import ParquetStore, market_universe
        from forecaster.pipeline import refresh_all

        store = ParquetStore(Path(settings.cache_dir) / "ohlcv")
        for t in market_universe(settings.tickers):
            try:
                store.get(t)
            except Exception as exc:
                warn(f"{t}: {exc}")
        refresh_all(settings, warn=warn)
    inputs = gather_inputs(settings, run_date, warn)
    report = build_report(run_date, prev_state=previous_state(Path(args.out), run_date), **inputs)
    paths = save_report(report, Path(args.out), pdf=not args.no_pdf)
    for k, p in paths.items():
        print(f"{k:>8}: {p}")
    status = None
    if not args.no_status:
        from forecaster import project_status

        try:   # imports + full test run, then STATUS.md
            status = project_status.refresh(settings.cache_dir)
            print("  status:", project_status.STATUS_MD)
        except Exception as exc:
            warn(f"project status: {exc}")
    if not args.no_send:
        tg = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        ok, err = tg.send(summary_text(report, paths["markdown"]))
        print("telegram:", "sent" if ok else f"not sent ({err})")
        if status is not None:
            from forecaster.project_status import telegram_text

            ok, err = tg.send(telegram_text(status))
            print("telegram status:", "sent" if ok else f"not sent ({err})")
    try:   # shadow trading vs backtest, per setup (strategy decay)
        from forecaster.shadow import decay_text, weekly_decay

        txt = decay_text(weekly_decay(settings.cache_dir))
        if txt:
            print(txt)
            if not args.no_send:
                ok, err = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id).send(txt)
                print("telegram decay:", "sent" if ok else f"not sent ({err})")
    except Exception as exc:
        warn(f"shadow decay report: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
