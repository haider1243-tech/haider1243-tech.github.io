import json
import sqlite3
from dataclasses import asdict

import pandas as pd
import pytest

from forecaster import project_status as ps
from forecaster import status
from forecaster.execution import OrderStore
from forecaster.webhook_server import AlertStore

NOW = pd.Timestamp("2026-11-09 15:00", tz="UTC")
SECRET = "sk-very-secret-value-123"
JUNIT = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.test_market_data" name="a"/><testcase classname="tests.test_market_data" name="b"/>
<testcase classname="tests.test_webhook" name="c"><failure message="x"/></testcase>
<testcase classname="tests.test_news" name="d"><skipped/></testcase>
<testcase classname="tests.test_macro" name="e"><error message="boom"/></testcase>
</testsuite></testsuites>"""


def _root(tmp_path, env_lines=()):
    root = tmp_path / "proj"
    (root / "src" / "forecaster").mkdir(parents=True)
    for m in ps.MODULES:
        (root / "src" / "forecaster" / f"{m}.py").write_text("")
    (root / ".env").write_text("\n".join(env_lines) + "\n")
    return root


def _all_keys():
    return [f"{k}={SECRET}" for k in ps.KEYS]


def _tests(failed=0, files=None):
    per = files or {f"test_{m}" if m != "webhook_server" else "test_webhook": {"passed": 3, "failed": 0, "errors": 0,
                                                                               "skipped": 0} for m in ps.MODULES}
    return asdict(ps.TestRun(NOW.isoformat(), 1 if failed else 0, 36, failed, 0, 0, 12.0, per,
                             ["test_news.py::test_x"] if failed else []))


def test_parse_junit():
    per, fails = ps.parse_junit(JUNIT)
    assert per["test_market_data"] == {"passed": 2, "failed": 0, "errors": 0, "skipped": 0}
    assert per["test_webhook"]["failed"] == 1 and per["test_macro"]["errors"] == 1 and per["test_news"]["skipped"] == 1
    assert fails == ["test_webhook.py::c", "test_macro.py::e (error)"]


def test_module_checks_states(tmp_path):
    root = _root(tmp_path)
    (root / "src" / "forecaster" / "news.py").unlink()
    per, _ = ps.parse_junit(JUNIT)
    imports = {m: None for m in ps.MODULES}
    imports["forecast"] = "ModuleNotFoundError: No module named 'lightgbm'"
    tests = {"per_file": per, "note": ""}
    mods = {m.name: m for m in ps.module_checks(imports, tests, root)}
    assert mods["market_data"].state == "built" and mods["market_data"].detail == "2 tests pass"
    assert mods["webhook_server"].state == "failing" and "1 of 1 tests failing" in mods["webhook_server"].detail
    assert mods["macro"].state == "failing"
    assert mods["news"].state == "missing" and mods["news"].detail == "file not found"
    assert mods["forecast"].state == "missing" and "lightgbm" in mods["forecast"].detail
    assert mods["calendar"].state == "failing" and "no tests found" in mods["calendar"].detail
    assert all(m.state == "unchecked" for m in ps.module_checks(None, None, root) if m.name != "news")
    assert ps.module_checks(imports, None, root)[0].detail == "imports OK; tests not run"


def test_key_presence_never_returns_values(tmp_path):
    root = _root(tmp_path, [f"FRED_API_KEY={SECRET}", "FINNHUB_API_KEY=", "# TELEGRAM_BOT_TOKEN=x"])
    keys = ps.key_presence(root, environ={"WEBHOOK_SECRET": SECRET})
    assert keys["FRED_API_KEY"] and keys["WEBHOOK_SECRET"]
    assert not keys["FINNHUB_API_KEY"] and not keys["TELEGRAM_BOT_TOKEN"]
    assert set(keys.values()) <= {True, False}


def test_data_health(tmp_path):
    cache = tmp_path / "c"
    status.record(cache, "prices", True, now=NOW - pd.Timedelta(hours=2))
    status.record(cache, "macro", True, now=NOW - pd.Timedelta(hours=30))
    status.record(cache, "calendar", True, now=NOW - pd.Timedelta(hours=1))
    status.record(cache, "calendar", False, "FRED 500", now=NOW)
    keys = {k: True for k in ps.KEYS} | {"FINNHUB_API_KEY": False}
    h = {d.source: d for d in ps.data_health(cache, keys, NOW)}
    assert h["prices"].ok and h["prices"].age_hours == 2
    assert not h["macro"].ok and h["macro"].message == "30 h old (limit 24 h)"
    assert not h["calendar"].ok and "FRED 500" in h["calendar"].message
    assert not h["news"].ok and h["news"].message == "FINNHUB_API_KEY missing from .env"
    keys["FRED_API_KEY"] = False
    assert ps.data_health(cache, keys, NOW)[1].message == "FRED_API_KEY missing from .env"


def _bt_json(ticker, beats, gen, skill=0.04):
    return {"ticker": ticker, "generated_at": gen, "first_target": "2005-03-31", "last_target": "2026-06-30",
            "model": {"coverage": 0.81, "hit_rate": 0.7}, "skill": skill, "beats_baseline": beats}


def test_model_health_prefers_own_spy_run(tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(_bt_json("^GSPC", False, "2026-11-08T00:00:00+00:00")))
    cache = tmp_path / "c"
    assert ps.primary_backtest(ps.model_health(cache, ref)).reference          # only the reference
    (cache / "snapshots").mkdir(parents=True)
    (cache / "snapshots" / "backtest_QQQ.json").write_text(json.dumps(_bt_json("QQQ", False, "2026-11-09T00:00:00+00:00")))
    (cache / "snapshots" / "backtest_SPY.json").write_text(json.dumps(_bt_json("SPY", True, "2026-10-01T00:00:00+00:00")))
    (cache / "snapshots" / "backtest_bad.json").write_text("{oops")
    bts = ps.model_health(cache, ref)
    assert [b.ticker for b in bts] == ["QQQ", "SPY", "^GSPC"] and bts[-1].reference
    p = ps.primary_backtest(bts)
    assert p.ticker == "SPY" and p.beats_baseline and p.period == "2005–2026"


def _alerts(db, n, resolved):
    store = AlertStore(db)
    for i in range(n):
        t = (NOW - pd.Timedelta(hours=i + 2)).isoformat()
        rid, _ = store.insert({"received_at": t, "alert_time": t, "time_from_tv": 1, "ticker": "XAUUSD",
                               "forecaster_ticker": "GC=F", "price_symbol": "GC=F", "price": 2600.0,
                               "strategy": "vwap_gold", "side": "long", "payload": "{}", "dedupe_key": str(i),
                               "verdict": "TAKE", "reasons": "[]", "checks": "[]"})
        if i < resolved:
            for h in ("1h", "4h"):                                  # counted once per alert
                store.update_outcome(rid, h, status="done", ret=0.001, signed_ret=0.001)


def test_journal_and_demo_stats(tmp_path):
    db = tmp_path / "alerts.sqlite"
    assert ps.journal_stats(db)["alerts"] == 0 and ps.demo_stats(db, NOW)["days"] == 0
    _alerts(db, 5, 3)
    j = ps.journal_stats(db)
    assert j["alerts"] == 5 and j["journaled"] == 3
    OrderStore(db)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO orders (alert_id, created_at, environment, status) VALUES (1, '2026-08-01', 'practice', 'filled')")
        con.execute("INSERT INTO orders (alert_id, created_at, environment, status) VALUES (2, '2026-08-02', 'practice', 'blocked')")
        con.executemany("INSERT INTO daily_nav VALUES (?,?,?)",
                        [("2026-08-01", 100000.0, "x"), ("2026-09-15", 100400.0, "x"), ("2026-10-05", 100250.0, "x")])
    d = ps.demo_stats(db, NOW)
    assert d["filled"] == 1 and d["days"] == 65 and d["pnl"] == pytest.approx(250.0)


def _gates(tests, bt_beats, journaled, demo_days, filled=1):
    mods = ps.module_checks({m: None for m in ps.MODULES}, tests, ps.PROJECT_ROOT)
    bt = ps.BacktestHealth("SPY", 0.8, 0.7, 0.05, bt_beats, NOW.isoformat(), False, "2005–2026")
    j = {"alerts": journaled, "journaled": journaled}
    d = {"filled": filled, "days": demo_days, "pnl": 120.0}
    return mods, bt, j, d, ps.phase_gates(mods, tests, bt, j, d, False)


def test_phase_progression():
    assert ps.current_phase(_gates(None, True, 40, 90)[-1]) == 1                    # tests not run
    assert ps.current_phase(_gates(_tests(failed=1), True, 40, 90)[-1]) == 1
    assert ps.current_phase(_gates(_tests(), False, 40, 90)[-1]) == 2               # strictly in order
    assert ps.current_phase(_gates(_tests(), True, 29, 90)[-1]) == 3
    assert ps.current_phase(_gates(_tests(), True, 30, 59)[-1]) == 4
    assert ps.current_phase(_gates(_tests(), True, 30, 90, filled=0)[-1]) == 4
    gates = _gates(_tests(), True, 30, 60)[-1]
    assert ps.current_phase(gates) == 5 and "Only you can turn it on" in gates[4].evidence


def test_next_step_order(tmp_path):
    keys = {k: True for k in (*ps.KEYS, *ps.DEMO_KEYS)}
    fresh = [ps.SourceHealth("prices", "Prices", NOW.isoformat(), 0.1, None, True, "fresh")]

    def step(tests=_tests(), keys=keys, data=fresh, beats=True, journaled=30, days=60, bt_ref=False, filled=1):
        mods, bt, j, d, gates = _gates(tests, beats, journaled, days, filled)
        bt.reference = bt_ref
        return ps.next_step(mods, tests, keys, data, gates, bt, j, d)

    assert "failing test" in step(tests=_tests(failed=1)) and "pytest tests/test_news.py" in step(tests=_tests(failed=1))
    assert "Re-check now" in step(tests=None)
    assert step(keys=keys | {"FRED_API_KEY": False}).startswith("Add FRED_API_KEY to .env")
    bad = [ps.SourceHealth("macro", "FRED macro", None, None, "x", False, "never refreshed")]
    assert step(data=bad).startswith("FRED macro data: never refreshed")
    assert "--backtest-stale-days 0" in step(beats=False, bt_ref=True)
    assert "fix #1" in step(beats=False)
    assert "journal 12 more" in step(journaled=18)
    assert "OANDA_API_TOKEN" in step(keys=keys | {"OANDA_API_TOKEN": False}, days=10)
    assert "50 more days" in step(days=10)
    final = step()
    assert "only you can set LIVE_TRADING=true" in final


def test_refresh_writes_cache_md_and_never_leaks_keys(tmp_path):
    root = _root(tmp_path, _all_keys())
    cache = tmp_path / "data"
    status.record(cache, "prices", True)
    md = tmp_path / "STATUS.md"
    s = ps.refresh(cache, md_path=md, runner=lambda r: ps.TestRun(NOW.isoformat(), 0, 36, 0, 0, 0, 5.0,
                                                                   _tests()["per_file"]),
                   importer=lambda: {m: None for m in ps.MODULES}, root=root)
    assert not (cache / "project_status.lock").exists()
    assert json.loads((cache / "project_status.json").read_text())["tests"]["passed"] == 36
    assert s["gates"][0]["passed"] is True and all(m["state"] == "built" for m in s["modules"])
    text = md.read_text()
    assert "# Project status" in text and "**Next step:**" in text and "| 👉 | **2 Validate** |" in text
    assert "FRED_API_KEY ✅" in text and "| ✅ | `market_data` | 3 tests pass |" in text
    tg = ps.telegram_text(s)
    assert "Phase 2 (Validate)" in tg and "<b>Next:</b>" in tg and "19/19 modules" in tg
    for out in (text, tg, json.dumps(s)):
        assert SECRET not in out                                     # key values never written or sent
    # --quick keeps the previous test result
    s2 = ps.refresh(cache, run_tests=False, md_path=None, runner=None,
                    importer=lambda: {m: None for m in ps.MODULES}, root=root)
    assert s2["tests"]["passed"] == 36


def test_child_env_scrubs_secrets(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", SECRET)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", SECRET)
    monkeypatch.setenv("FORECASTER_CACHE_DIR", "/real/data")
    env = ps._child_env(ps.PROJECT_ROOT)
    assert SECRET not in env.values() and "FORECASTER_CACHE_DIR" not in env
    assert env["FORECASTER_STATUS_CHILD"] == "1"


def test_should_start_guards(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()
    assert ps.should_start(cache, environ={})
    assert not ps.should_start(cache, environ={"PYTEST_CURRENT_TEST": "x"})       # never pytest-in-pytest
    assert not ps.should_start(cache, environ={"FORECASTER_STATUS_CHILD": "1"})
    (cache / "project_status.lock").write_text(str(__import__("os").getpid()))   # a check is running
    assert not ps.should_start(cache, environ={})
    (cache / "project_status.lock").unlink()
    recent = {"checked_at": pd.Timestamp.now(tz="UTC").isoformat(), "tests": {"passed": 1}}
    ps.status_json(cache).write_text(json.dumps(recent))
    assert not ps.should_start(cache, environ={})                                # launcher just ran it
    old = recent | {"checked_at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).isoformat()}
    ps.status_json(cache).write_text(json.dumps(old))
    assert ps.should_start(cache, environ={})
