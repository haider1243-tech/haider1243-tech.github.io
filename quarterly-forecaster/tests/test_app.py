"""Drive the mobile app (app.py) headless with Streamlit's AppTest against fixture data."""
import json
from pathlib import Path

import pandas as pd
import pytest

from app_fixtures import write_fixture_snapshots
from conftest import authed_app
from forecaster import pipeline, status
from forecaster.pipeline import write_json
from forecaster.webhook_server import AlertStore

APP = str(Path(__file__).resolve().parents[1] / "app.py")
NOW = pd.Timestamp.now(tz="UTC")


def _prices():
    return {"generated_at": NOW.isoformat(), "usd_cad": 1.38, "errors": [], "quotes": {
        "SPY": {"label": "SPY", "price": 600.0, "price_cad": 828.0, "change": 0.004, "as_of": NOW.isoformat()},
        "QQQ": {"label": "QQQ", "price": 520.0, "price_cad": 717.6, "change": -0.01, "as_of": NOW.isoformat()},
        "GC=F": {"label": "Gold", "price": 2700.0, "price_cad": 3726.0, "change": 0.002, "vwap": 2690.0,
                 "sd": 4.0, "z": 2.5, "as_of": NOW.isoformat()},
        "^VIX": {"label": "VIX", "price": 17.2, "price_cad": None, "change": 0.05, "as_of": NOW.isoformat()},
    }}


def _alerts(db: Path):
    store = AlertStore(db)
    for i, (verdict, ret) in enumerate([("TAKE", 0.004), ("TAKE", -0.002), ("SKIP", -0.003)]):
        t = (NOW - pd.Timedelta(hours=30 - i)).isoformat()
        rid, _ = store.insert({"received_at": t, "alert_time": t, "time_from_tv": 1, "ticker": "XAUUSD",
                               "forecaster_ticker": "GC=F", "price_symbol": "GC=F", "price": 2650.0 + i,
                               "strategy": "vwap_gold", "side": "long", "payload": "{}", "dedupe_key": f"k{i}",
                               "verdict": verdict, "reasons": json.dumps([f"reason {i}"]), "checks": "[]"})
        store.update_outcome(rid, "1h", status="done", ret=ret, signed_ret=ret)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    snaps = d / "snapshots"
    for t in ("SPY", "QQQ", "GC=F"):
        write_fixture_snapshots(snaps, ticker=t)
        s = pipeline.read_json(snaps / f"ticker_{t}.json")
        write_json(snaps / f"ticker_{pipeline.safe_name(t)}.json", s)
    write_json(snaps / "prices.json", _prices())
    _alerts(d / "alerts.sqlite")
    for src in ("prices", "market", "forecasts", "news"):
        status.record(d, src, True)
    reports = tmp_path_factory.mktemp("reports")
    (reports / "weekly_2026-11-01.md").write_text("# Weekly report 2026-11-01\n\nOlder report body")
    (reports / "weekly_2026-11-08.md").write_text("# Weekly report 2026-11-08\n\nNewest report body")
    (reports / "weekly_2026-11-08.pdf").write_bytes(b"%PDF-1.4 fake")
    return d, reports


@pytest.fixture
def app(data, monkeypatch):
    d, reports = data
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(d))
    monkeypatch.setenv("FORECASTER_REPORTS_DIR", str(reports))
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    at = authed_app(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _html(at) -> str:
    return "\n".join(str(getattr(e, "proto", e).body) for e in at.get("html"))


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [x.value for x in at.warning] + [x.value for x in at.info] + [x.value for x in at.success]
    parts += [x.value for x in at.error]
    return "\n".join(map(str, parts)) + "\n" + _html(at)


def _go(at, tab):
    at.radio(key="tab").set_value(tab).run()
    assert not at.exception, at.exception
    return at


def test_bottom_tabs_and_last_updated_on_every_tab(app):
    assert app.radio(key="tab").options == ["Week", "Calendar", "Gold", "Portfolio", "Reports", "Status"]
    for tab, title in (("week", "This Week"), ("calendar", "Calendar"), ("gold", "Gold VWAP"),
                       ("portfolio", "Portfolio"), ("reports", "Reports"), ("status", "Status")):
        at = _go(app, tab)
        txt = _text(at)
        assert f"### {title}" in txt, tab
        assert "Last updated · prices just now · forecasts just now" in txt, tab
        assert "Data problem" not in txt, tab                      # all sources fresh -> no red banner
        assert at.query_params["tab"] == [tab] or at.query_params["tab"] == tab


def test_pwa_head_and_manifest_link(app):
    h = _html(app)
    assert "/app/static/manifest.json" in h and "apple-touch-icon" in h and "theme-color" in h


def test_this_week(app):
    txt = _text(app)
    assert "Market regime (SPY)" in txt and "VIX" in txt and "17.2" in txt
    assert "middle case" in txt and "likely range" in txt      # 3-line outlook gives ranges
    assert "C$828.00" in txt                                   # quotes in CAD
    charts = [c for c in app.get("plotly_chart")]
    assert len(charts) == 3                                    # SPY, QQQ, gold fans
    assert "Model check" in txt and "within noise" in txt      # honest backtest note


def test_calendar(app):
    at = _go(app, "calendar")
    txt = _text(at)
    assert "CPI release" in txt and "PPI release" in txt
    assert "in 9d" in txt or "in 10d" in txt                   # countdown
    assert "#d03b3b" in txt and "High" in txt and "Medium" in txt   # impact colours + labels
    assert "AAPL earnings" not in txt                          # 20 days out: beyond 14


def test_gold_journal(app):
    at = _go(app, "gold")
    txt = _text(at)
    assert "vs session VWAP" in txt and "+2.5 st. dev." in txt
    assert "3 total" in txt and "TAKE" in txt and "SKIP" in txt and "reason 2" in txt
    stats = at.dataframe[0].value
    take = stats[stats["Verdict"] == "TAKE · 1h"].iloc[0]
    assert take["n"] == 2 and take["Win rate"] == pytest.approx(50)


def test_reports_newest_first(app):
    at = _go(app, "reports")
    assert "(latest)" in at.selectbox(key="report_pick").format_func(0)
    assert any("Newest report body" in m.value for m in at.markdown)
    assert at.get("download_button")
    at.selectbox(key="report_pick").set_value(1).run()
    assert any("Older report body" in m.value for m in at.markdown)


def test_red_banner_on_failure_and_stale(data, monkeypatch, tmp_path):
    d = tmp_path / "c"
    status.record(d, "prices", False, "Yahoo timed out")
    status.record(d, "market", True, now=NOW - pd.Timedelta(hours=30))
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(d))
    monkeypatch.setenv("FORECASTER_REPORTS_DIR", str(tmp_path / "none"))
    at = authed_app(APP).run()
    assert not at.exception, at.exception
    for tab in ("week", "calendar", "gold", "portfolio", "reports"):
        txt = _text(_go(at, tab))
        assert "role=\"alert\"" in txt and "Data problem" in txt
        assert "Prices: refresh failed (Yahoo timed out)" in txt
        assert "Macro &amp; events: data is 30 h old" in txt
        assert "Forecasts: no data yet" in txt


def test_empty_cache_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FORECASTER_REPORTS_DIR", str(tmp_path / "reports"))
    at = authed_app(APP).run()
    for tab in ("week", "calendar", "gold", "portfolio", "reports"):
        _go(at, tab)
        assert "Last updated · prices never" in _text(at)


def test_query_param_opens_tab(data, monkeypatch):
    d, reports = data
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(d))
    at = authed_app(APP)
    at.query_params["tab"] = "gold"
    at.run()
    assert at.radio(key="tab").value == "gold"


def test_refresh_button_calls_pipeline(app, monkeypatch):
    calls = []

    def fake_refresh_all(settings, tickers=None, warn=print, **kw):
        calls.append(kw)
        warn("fake refresh ran")
        return {"market": "ok"}

    monkeypatch.setattr(pipeline, "refresh_all", fake_refresh_all)
    app.button(key="refresh").click().run()
    assert not app.exception, app.exception
    assert calls == [{"forecasts": False}]


def test_status_tab(app, monkeypatch):
    from forecaster import project_status as ps

    at = _go(app, "status")
    txt = _text(at)
    assert "Next step" in txt and "Re-check now" in txt                     # no build check cached yet
    assert "Phase tracker" in txt and "1 · Build" in txt and "5 · Live" in txt and "current" in txt
    assert "market_data" in txt and "weekly_report" in txt and "not checked yet" in txt
    assert "Data health" in txt and "FRED macro" in txt and "FRED_API_KEY missing from .env" in txt
    assert "Model health" in txt and "^GSPC (reference run)" in txt and "no edge" in txt
    assert "Journal: 3 alerts, 3 with a resolved outcome" in txt
    calls = []

    def fake_refresh(cache_dir, **kw):
        calls.append(cache_dir)
        p = ps.status_json(cache_dir)
        p.write_text(json.dumps({"checked_at": NOW.isoformat(), "imports": {m: None for m in ps.MODULES},
                                 "tests": {"ran_at": NOW.isoformat(), "returncode": 0, "passed": 99, "failed": 0,
                                           "errors": 0, "skipped": 0, "seconds": 1.0, "note": "", "failures": [],
                                           "per_file": {f"test_{m}" if m != "webhook_server" else "test_webhook":
                                                        {"passed": 1, "failed": 0, "errors": 0, "skipped": 0}
                                                        for m in ps.MODULES}}}))

    monkeypatch.setattr(ps, "refresh", fake_refresh)
    at.button(key="recheck").click().run()
    assert not at.exception, at.exception
    txt = _text(at)
    assert calls and "99 tests passed" in txt and "12/12 modules ✅" in txt
    assert "Add FINNHUB_API_KEY to .env" in txt or "Add FRED_API_KEY to .env" in txt
    ps.status_json(calls[0]).unlink()
