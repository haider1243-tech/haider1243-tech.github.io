import copy

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from forecaster.calendar import Event
from forecaster.weekly_report import (
    Table, build_report, forecast_changes, macro_released, next_week_events, pdf_safe, portfolio_changes,
    previous_state, render_markdown, save_report, sentiment_shifts, summary_text, week_window, weekly_moves,
)
from forecaster.webhook_server import TelegramNotifier

SUN = pd.Timestamp("2026-11-08")          # report date
WS, WE = pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-06")


def daily(values, end="2026-11-06"):
    return pd.Series(values, index=pd.bdate_range(end=end, periods=len(values)), dtype=float)


@pytest.fixture
def closes():
    # 10 business days: previous week Oct 26-30, this week Nov 2-6
    return {"SPY": daily([600] * 5 + [602, 605, 598, 610, 612]),
            "QQQ": daily([500] * 5 + [490, 495, 492, 497, 495]),
            "GC=F": daily([2600] * 5 + [2610, 2620, 2650, 2640, 2660]),
            "^VIX": daily([16] * 5 + [17, 18, 22, 19, 18.5]),
            "CAD=X": daily([1.40] * 5 + [1.39, 1.39, 1.38, 1.38, 1.38])}


def market(sentiment=True):
    days = pd.date_range("2026-10-19", "2026-11-07")
    sent = {"MARKET": [{"date": str(d.date()), "sentiment": (0.10 if d < pd.Timestamp("2026-11-01") else -0.05),
                        "headlines": 10, "z": 0.0, "shock": False} for d in days]} if sentiment else {}
    return {
        "events": [
            {"date": "2026-11-10", "category": "CPI", "name": "CPI release", "impact": "high", "detail": ""},
            {"date": "2026-11-12", "category": "PPI", "name": "PPI release", "impact": "medium", "detail": ""},
            {"date": "2026-11-13", "category": "Jobs", "name": "Employment Situation", "impact": "high", "detail": ""},
            {"date": "2026-11-18", "category": "FOMC", "name": "FOMC rate decision", "impact": "high", "detail": ""},
        ],
        "macro_table": [
            {"series_id": "CPIAUCSL", "name": "CPI", "units": "pct", "date": "2026-10-01", "value": 330.1,
             "yoy": 2.9, "trend_3m": 3.1, "surprise_z": 1.4},
            {"series_id": "CPILFESL", "name": "Core CPI", "units": "pct", "date": "2026-10-01", "value": 320.0,
             "yoy": 3.1, "trend_3m": None, "surprise_z": None},
            {"series_id": "UNRATE", "name": "Unemployment rate", "units": "pp", "date": "2026-10-01", "value": 4.3,
             "yoy": 0.2, "trend_3m": 0.1, "surprise_z": -0.5},
        ],
        "sentiment": sent,
        "shocks": [{"tag": "MARKET", "date": "2026-11-04", "sentiment": -0.6, "z": -2.8, "headlines": 9}],
    }


def tsnap(p50=650.0, quarters=("2027Q1", "2027Q2", "2027Q3", "2027Q4")):
    return {"ticker": "SPY", "currency": "CAD", "as_of": "2026-11-06", "last_close": 840.0,
            "forecast": [{"quarter": q, "p10": p50 * 0.85, "p50": p50 * (1 + 0.01 * i), "p90": p50 * 1.2}
                         for i, q in enumerate(quarters)]}


def pf(value=40_000.0, cost=31_000.0, flags=("XEQT.TO",), xeqt_shares=400, extra=True):
    h = [{"ticker": "XEQT.TO", "account": "TFSA", "shares": xeqt_shares, "market_value": value * 0.3, "weight": 0.3},
         {"ticker": "ZAG.TO", "account": "RRSP", "shares": 300, "market_value": value * 0.2, "weight": 0.2}]
    if extra:
        h.append({"ticker": "VFV.TO", "account": "RRSP", "shares": 60, "market_value": value * 0.5, "weight": 0.5})
    return {"total_value": value, "total_cost": cost, "total_pnl": value - cost,
            "holdings": h, "flags": [{"ticker": t, "message": f"{t} is 30% of the portfolio (limit 25%)."} for t in flags],
            "forecast": [{"quarter": "2027Q1", "p10": value * 0.9, "p50": value * 1.01, "p90": value * 1.1}],
            "risk": {"volatility": 0.11}}


PAST = [Event(pd.Timestamp("2026-11-03"), "CPI", "CPI release", "high", "FRED"),
        Event(pd.Timestamp("2026-11-06"), "Jobs", "Employment Situation", "high", "FRED")]


# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("d,mon,fri", [
    ("2026-11-08", "2026-11-02", "2026-11-06"),   # Sunday
    ("2026-11-07", "2026-11-02", "2026-11-06"),   # Saturday
    ("2026-11-09", "2026-11-02", "2026-11-06"),   # Monday -> previous week
    ("2026-11-06", "2026-11-02", "2026-11-06"),   # Friday -> this week
    ("2026-11-04", "2026-10-26", "2026-10-30"),   # Wednesday -> previous complete week
])
def test_week_window(d, mon, fri):
    assert week_window(pd.Timestamp(d)) == (pd.Timestamp(mon), pd.Timestamp(fri))


def test_weekly_moves_in_cad(closes):
    sec, st = weekly_moves(closes, closes["CAD=X"], WS, WE)
    t = sec.blocks[0]
    rows = {r[0]: r for r in t.rows}
    # SPY: 612 * 1.38 vs 600 * 1.40
    assert st["SPY"]["change"] == pytest.approx(612 * 1.38 / (600 * 1.40) - 1)
    assert rows["S&P 500 (SPY)"][1] == f"C${612 * 1.38:,.2f}"
    assert rows["VIX"][2] == "+2.50 pts" and rows["VIX"][1] == "18.50"            # points, not CAD
    assert rows["USD/CAD"][1] == "1.3800" and st["CAD=X"]["change"] == pytest.approx(1.38 / 1.40 - 1)
    assert rows["Gold (GC=F)"][4] == f"C${2660 * 1.38:,.2f}"                      # week high


def test_weekly_moves_holiday_and_missing(closes):
    c = {k: v.drop(pd.Timestamp("2026-11-06")) for k, v in closes.items()}      # Friday holiday
    _, st = weekly_moves(c, c["CAD=X"], WS, WE)
    assert st["SPY"]["close"] == pytest.approx(610 * 1.38)                       # Thursday close
    sec, st2 = weekly_moves({"^VIX": closes["^VIX"]}, None, WS, WE)
    rows = {r[0]: r for r in sec.blocks[0].rows}
    assert rows["S&P 500 (SPY)"][1] == "no data" and "SPY" not in st2


def test_macro_released():
    sec = macro_released(PAST, market()["macro_table"])
    t = sec.blocks[0]
    assert [r[2] for r in t.rows] == ["CPI", "Core CPI", "Unemployment rate"]
    assert t.rows[0][3].startswith("YoY +2.90%") and t.rows[0][5] == "+1.4"
    assert t.rows[1][4] == "n/a"                                                  # None-safe
    assert t.rows[2][3].startswith("4.30")
    assert "No CPI" in macro_released([], []).blocks[0]


def test_next_week_high_impact_only():
    sec, evs = next_week_events(market(), WE)
    assert [e["category"] for e in evs] == ["CPI", "Jobs"]                        # PPI medium, FOMC next-next week
    assert sec.blocks[0].rows[0][:2] == ["Tue Nov 10", "08:30 ET"]
    assert "Nov 09 - Nov 13" in sec.title
    assert "refresh" in next_week_events(None, WE)[0].blocks[0]


def test_forecast_changes_first_and_second_week():
    sec, st = forecast_changes({"SPY": tsnap()}, None)
    t = sec.blocks[0]
    assert all(r[5] == "new" for r in t.rows) and "First report" in sec.blocks[-1]
    prev = {"forecasts": st["forecasts"]}
    # next week: P50 up 2%, a quarter rolled off and a new one appeared
    sec2, st2 = forecast_changes({"SPY": tsnap(663.0, ("2027Q2", "2027Q3", "2027Q4", "2028Q1"))}, prev)
    rows = {r[1]: r for r in sec2.blocks[0].rows}
    assert rows["2028Q1"][5] == "new"
    assert rows["2027Q2"][5] == f"{663 * 1.00 / (650 * 1.01) - 1:+.1%}"
    assert st2["_notable"] and st2["_notable"][0][1] == "SPY"


def test_sentiment_shift():
    sec, st = sentiment_shifts(market(), WS, WE)
    row = sec.blocks[0].rows[0]
    assert row[0] == "Market (general)" and row[1] == "-0.05" and row[2] == "+0.10" and row[3] == "-0.15"
    assert any("Shock: MARKET on 2026-11-04" in x for x in sec.blocks[1])
    assert st["_shocks"] == 1
    assert "No scored headlines" in sentiment_shifts(market(sentiment=False), WS, WE)[0].blocks[0]


def test_portfolio_changes_vs_last_week():
    _, st = portfolio_changes(pf(), None)
    prev = {"portfolio": st["portfolio"]}
    now = pf(value=41_000.0, cost=31_500.0, flags=(), xeqt_shares=410, extra=False)
    sec, st2 = portfolio_changes(now, prev)
    summary = sec.blocks[0]
    assert summary[0].startswith("Market value C$41,000 (+C$1,000, +2.5% vs last week)")
    assert any("Cost base changed +C$500" in x and "market moves explain +C$500" in x for x in summary)
    rows = {r[0]: r for r in sec.blocks[1].rows}
    assert rows["XEQT.TO"][6] == "+10 shares" and rows["VFV.TO"][6] == "sold / removed"
    assert any("Resolved: XEQT.TO" in x for x in sec.blocks[2])
    assert st2["_dv"] == pytest.approx(1000)
    assert "No portfolio snapshot" in portfolio_changes(None, None)[0].blocks[0]


# --------------------------------------------------------------------------- #
# End to end: two Sundays, Markdown + PDF + state + Telegram
# --------------------------------------------------------------------------- #

def test_two_weeks_end_to_end(closes, tmp_path):
    r1 = build_report(SUN, closes, closes["CAD=X"], market(), {"SPY": tsnap()}, pf(), PAST, None)
    p1 = save_report(r1, tmp_path)
    assert set(p1) == {"markdown", "state", "pdf"}
    md = p1["markdown"].read_text()
    for title in ("Last week's moves", "Macro data released last week", "Next week's high-impact events",
                  "4-quarter forecasts vs last week", "Sentiment shifts", "Portfolio changes"):
        assert f"## {title}" in md
    assert "C$" in md and "not reliably beat" in md
    pdf = PdfReader(str(p1["pdf"]))
    text = "".join(pg.extract_text() for pg in pdf.pages)
    assert len(pdf.pages) >= 1 and "Weekly report" in text and "C$" in text and "Portfolio changes" in text
    assert pdf.metadata.title.startswith("Weekly report")

    # a week later: reads the previous state and shows changes
    sun2 = SUN + pd.Timedelta(days=7)
    c2 = {k: pd.concat([v, daily(v.iloc[-5:].to_numpy() * 1.01, end="2026-11-13")]) for k, v in closes.items()}
    prev = previous_state(tmp_path, sun2)
    assert prev["run_date"] == "2026-11-08"
    r2 = build_report(sun2, c2, c2["CAD=X"], market(), {"SPY": tsnap(663.0)}, pf(41_000.0), [], prev)
    md2 = render_markdown(r2)
    assert "First report" not in md2 and "vs last week)" in md2
    save_report(r2, tmp_path)
    assert previous_state(tmp_path, sun2 + pd.Timedelta(days=1))["run_date"] == "2026-11-15"
    assert previous_state(tmp_path, SUN) is None                                  # nothing before the first


def test_summary_is_short_escaped_and_sendable(closes):
    r = build_report(SUN, closes, closes["CAD=X"], market(), {"SPY": tsnap()}, pf(), PAST, None)
    txt = summary_text(r)
    assert len(txt) < 1000 and txt.startswith("<b>Weekly report</b>")
    assert "Moves (CAD): S&amp;P 500" in txt                                      # escaped for Telegram HTML
    assert "Next week: Tue CPI; Fri Jobs" in txt and "Portfolio C$40,000" in txt and "Over 25%: XEQT.TO" in txt
    sent = []
    class R:
        status_code = 200
    ok, err = TelegramNotifier("t", "c", post=lambda url, json, timeout: (sent.append(json), R())[1]).send(txt)
    assert ok and sent[0]["parse_mode"] == "HTML" and sent[0]["text"] == txt


def test_pdf_safe_replaces_unsupported_glyphs():
    assert pdf_safe("≈ C$3,657 ⚠ → ✅") == "~ C$3,657 ! -> "
    assert pdf_safe("Montréal – 5%") == "Montréal – 5%"                          # cp1252 keeps these


def test_markdown_escapes_pipes():
    from forecaster.weekly_report import Report, Section
    r = Report(SUN, WS, WE, [Section("S", [Table(["a"], [["x|y"]])])], {}, [])
    assert "x\\|y" in render_markdown(r)
