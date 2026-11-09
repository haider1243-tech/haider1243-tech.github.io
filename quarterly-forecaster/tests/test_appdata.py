import json

import pandas as pd
import pytest

from forecaster import appdata as ad
from forecaster.webhook_server import AlertStore

NOW = pd.Timestamp("2026-11-09 15:00", tz="UTC")       # Monday 10:00 ET


def _ev(day, cat, impact="high", name=None, **kw):
    return {"date": day, "category": cat, "name": name or f"{cat} release", "impact": impact, **kw}


MARKET = {"events": [
    _ev("2026-11-10", "CPI"), _ev("2026-11-12", "PPI", "medium"), _ev("2026-11-11", "FOMC"),
    _ev("2026-11-13", "Jobs"), _ev("2026-11-20", "Earnings", "low", "NVDA earnings", ticker="NVDA"),
    _ev("2026-11-30", "CPI"), _ev("2026-11-10", "Retail", "medium"),
]}


def test_countdown():
    assert ad.countdown(pd.Timedelta(days=2, hours=5, minutes=3)) == "in 2d 5h"
    assert ad.countdown(pd.Timedelta(hours=3, minutes=5)) == "in 3h 05m"
    assert ad.countdown(pd.Timedelta(minutes=12)) == "in 12m"
    assert ad.countdown(pd.Timedelta(minutes=-5)) == "now"


def test_calendar_items_window_categories_and_order():
    items = ad.calendar_items(MARKET, NOW, days=14)
    assert [i.category for i in items] == ["CPI", "FOMC", "PPI", "Jobs", "Earnings"]   # no Retail, no Nov 30
    cpi = items[0]
    assert cpi.when == pd.Timestamp("2026-11-10 08:30", tz=ad.NY).tz_convert("UTC")
    assert cpi.countdown == "in 22h 30m" and items[-1].ticker == "NVDA"


def test_outlook_lines_use_ranges():
    snaps = {"SPY": {"last_close": 100.0, "regime": {"current": {"bull": 0.7, "bear": 0.2, "high_vol": 0.1}},
                     "forecast": [{"quarter": "2027Q4", "p10": 85.0, "p50": 106.0, "p90": 121.0}]}}
    prices = {"quotes": {"^VIX": {"price": 22.4}}}
    lines = ad.outlook_lines(snaps, MARKET, prices, NOW)
    assert len(lines) == 3
    assert lines[0] == "Markets look calm, rising (70% likely regime), with VIX elevated at 22.4."
    assert lines[1] == "S&P 500 by 2027Q4: middle case +6.0%, likely range -15% to +21%."
    assert lines[2].startswith("Busy week ahead: CPI Tue, FOMC Wed, jobs report Fri")
    empty = ad.outlook_lines({}, None, None, NOW)
    assert len(empty) == 3 and "Quiet calendar" in empty[2] and "No S&P 500 forecast" in empty[1]


def test_gold_journal(tmp_path):
    assert ad.gold_journal(tmp_path / "none.sqlite") == {"alerts": [], "stats": [], "n_alerts": 0}
    store = AlertStore(tmp_path / "a.sqlite")
    for i, (v, r) in enumerate([("SKIP", -0.01), ("TAKE", 0.01), ("TAKE", 0.02), ("TAKE", -0.01)]):
        t = (NOW - pd.Timedelta(hours=10 - i)).isoformat()
        rid, _ = store.insert({"received_at": t, "alert_time": t, "time_from_tv": 1, "ticker": "XAUUSD",
                               "forecaster_ticker": "GC=F", "price_symbol": "GC=F", "price": 2600.0,
                               "strategy": "vwap_gold", "side": "long", "payload": "{}", "dedupe_key": str(i),
                               "verdict": v, "reasons": json.dumps(["r"]), "checks": "[]"})
        store.update_outcome(rid, "1h", status="done", ret=r, signed_ret=r)
    j = ad.gold_journal(tmp_path / "a.sqlite", limit=2)
    assert j["n_alerts"] == 4 and len(j["alerts"]) == 2 and j["alerts"][0]["id"] == 4
    assert j["alerts"][0]["reasons"] == ["r"]
    assert [(s["verdict"], s["horizon"]) for s in j["stats"]] == [("TAKE", "1h"), ("SKIP", "1h")]   # pending 4h/1d excluded
    assert j["stats"][0]["win_rate"] == pytest.approx(2 / 3) and j["stats"][0]["n"] == 3


def test_list_reports(tmp_path):
    assert ad.list_reports(tmp_path / "missing") == []
    for d in ("2026-10-25", "2026-11-08", "2026-11-01"):
        (tmp_path / f"weekly_{d}.md").write_text(d)
    (tmp_path / "weekly_2026-11-08.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.md").write_text("x")
    r = ad.list_reports(tmp_path)
    assert [x["date"].strftime("%Y-%m-%d") for x in r] == ["2026-11-08", "2026-11-01", "2026-10-25"]
    assert r[0]["pdf"] is not None and r[1]["pdf"] is None


def test_ago():
    assert ad.ago(None, NOW) == "never"
    assert ad.ago(NOW - pd.Timedelta(seconds=30), NOW) == "just now"
    assert ad.ago(NOW - pd.Timedelta(minutes=14), NOW) == "14 min ago"
    assert ad.ago(NOW - pd.Timedelta(hours=5), NOW) == "5 h ago"
    assert ad.ago(NOW - pd.Timedelta(days=3), NOW) == "3 days ago"
