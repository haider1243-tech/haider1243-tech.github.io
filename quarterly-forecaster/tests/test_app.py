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
    assert app.radio(key="tab").options == ["Week", "Live", "Signals", "Calendar", "Gold", "Portfolio", "Reports", "Status"]
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
    assert calls and "99 tests passed" in txt and "20/20 modules ✅" in txt
    assert "Add FINNHUB_API_KEY to .env" in txt or "Add FRED_API_KEY to .env" in txt
    ps.status_json(calls[0]).unlink()


def test_live_tab_tiles_chart_and_tradingview(app, data):
    from forecaster import live_feed as lf

    d, _ = data
    now_ms = int(NOW.value // 1_000_000)
    hub = lf.Hub(clock=lambda: now_ms)
    store = lf.LiveStore(lf.live_db(d))
    for i in range(30):                                  # 30 one-minute gold bars ending now
        hub.add(lf.Tick(lf.GOLD, 2640 + 0.3 * i, now_ms - (30 - i) * 60_000, 0, "oanda"))
    hub.add(lf.Tick("SPY", 601.25, now_ms - 400, 10, "finnhub"))
    for name in ("oanda", "finnhub"):
        hub.feed(name).status, hub.feed(name).last_msg_ms = "connected", now_ms - 300
    hub.flush(store)
    store.set_prev_close({"SPY": 600.0})
    at = _go(app, "live")
    txt = _text(at)
    assert "Gold (XAU/USD)" in txt and "601.25" in txt and "DXY (proxy)" in txt
    assert ".%03d ET" % ((now_ms - 400) % 1000) in txt                       # ms timestamp on the tile
    assert "Finnhub (stocks): connected" in txt and "OANDA (gold, FX): connected" in txt
    assert "silent for more than 10 s" in txt
    assert len(at.get("plotly_chart")) == 1 and "VWAP" in txt
    at.radio(key="live_view").set_value("tv").run()
    assert not at.exception, at.exception
    assert "TradingView decides which feeds are real-time" in _text(at)
    assert at.radio(key="tv_sym").options == ["S&P 500", "Nasdaq", "VIX", "Gold"]
    at.radio(key="tv_sym").set_value("OANDA:XAUUSD").run()
    assert not at.exception
    at.radio(key="live_view").set_value("feed").run()
    lf.live_db(d).unlink()


@pytest.fixture
def lab_app(data, monkeypatch, tmp_path_factory):
    """App with lab results (one EDGE, one NO EDGE) + a pending shadow setup and a closed shadow trade."""
    from forecaster import lab as L
    from forecaster import shadow as SH
    from forecaster.intraday import INSTRUMENTS, resample, session_bars, synthetic_m1
    from test_lab import plant_momentum

    d, reports = data
    spec = INSTRUMENTS["SPY"]
    m1 = synthetic_m1("SPY", "2022-01-03", "2023-12-29", seed=5)
    L.run_lab({("SPY", "5m"): resample(session_bars(plant_momentum(m1), spec), 5, spec)}, L.lab_dir(d),
              strategies=["orb30", "vwap_reversion"], n_random=200, progress=lambda m: None)
    db = SH.shadow_db(d)
    SH.init_db(db)
    now = NOW
    SH.write(db, [{"key": "orb30|SPY|5m", "signal_time": (now - pd.Timedelta(hours=3)).isoformat(), "strategy": "orb30",
                   "instrument": "SPY", "timeframe": "5m", "side": 1, "entry_time": (now - pd.Timedelta(hours=3)).isoformat(),
                   "entry": 600.0, "stop": 598.0, "target": 604.0, "exit_time": now.isoformat(), "exit": 604.0,
                   "reason": "target", "net_ret": 0.0066, "r_mult": 2.0, "status": "closed", "verdict": "EDGE",
                   "params": "{}"}],
             [{"key": "vwap_reversion|SPY|5m", "signal_time": (now - pd.Timedelta(minutes=2)).isoformat(),
               "strategy": "vwap_reversion", "instrument": "SPY", "timeframe": "5m", "side": -1, "ref": 601.0,
               "stop": 602.5, "target": 598.0, "verdict": "NO EDGE"},
              {"key": "orb30|SPY|5m", "signal_time": (now - pd.Timedelta(minutes=1)).isoformat(),
               "strategy": "orb30", "instrument": "SPY", "timeframe": "5m", "side": 1, "ref": 601.2,
               "stop": 599.0, "target": 605.6, "verdict": "EDGE"}], [], now)
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(d))
    at = authed_app(APP, default_timeout=60)
    at.run()
    yield at
    import shutil
    shutil.rmtree(L.lab_dir(d))
    db.unlink()


def test_signal_board_shows_setups_with_stats_and_greys_no_edge(lab_app):
    at = _go(lab_app, "signals")
    h = _html(at)
    assert "Opening range breakout 30m" in h and "VWAP mean reversion" in h
    assert "LONG" in h and "SHORT" in h and "stop 599.00" in h and "target 605.60" in h and "R:R 2.0" in h
    assert "Win rate" in h and "Expectancy" in h and "95% CI" in h and "p vs random" in h
    assert 'qf-setup noedge' in h                                     # NO EDGE card greyed out
    assert h.index("Opening range breakout 30m") < h.index("VWAP mean reversion")   # EDGE ranked first
    assert "Recent shadow trades" in _text(at) and "+66.0 bps" in h
    assert "nothing is ever ordered" in _text(at)
    assert not [b for b in at.button if "order" in (b.label or "").lower()]          # no order buttons anywhere


def test_leaderboard_and_lab_views(lab_app):
    at = _go(lab_app, "signals")
    at.radio(key="sig_view").set_value("leaders").run()
    assert not at.exception, at.exception
    h = _html(at)
    assert h.index("Opening range breakout 30m") < h.index("VWAP mean reversion")
    assert "2 setups tested" in _text(at) and "★" in h and "live +66.0 bps (1)" in h
    at.selectbox(key="lb_inst").set_value("QQQ").run()
    assert "No setups for this filter" in _html(at)
    at.radio(key="sig_view").set_value("lab").run()
    assert not at.exception, at.exception
    txt = _text(at)
    assert "EDGE" in _html(at) and "p &lt; 0.05 vs 1,000 random-entry strategies" in _html(at)
    assert "walk-forward" in txt and len(at.get("plotly_chart")) >= 3          # random hist, equity, slices
    for sl in ("weekday", "vix_band", "event_day", "month"):
        at.selectbox(key="lab_slice").set_value(sl).run()
        assert not at.exception, (sl, at.exception)
    at.selectbox(key="lab_pick").set_value("vwap_reversion|SPY|5m").run()
    assert not at.exception


def test_signals_tab_without_results(app):
    at = _go(app, "signals")
    assert any("python -m forecaster.lab download" in i.value for i in at.info)
