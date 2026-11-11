"""Demo mode: shifted clock, simulated feeds, seeded data, launcher wiring."""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from forecaster import clock
from forecaster import intraday as I
from forecaster import launcher as L
from forecaster import live_feed as lf
from forecaster import simfeed as SF


def test_demo_starts_at_0944_on_the_latest_weekday():
    tue_night = pd.Timestamp("2026-11-10 19:00", tz=I.NY).tz_convert("UTC")
    assert SF.demo_market_start(tue_night) == pd.Timestamp("2026-11-10 09:44", tz=I.NY).tz_convert("UTC")
    sunday = pd.Timestamp("2026-11-15 12:00", tz=I.NY).tz_convert("UTC")
    assert SF.demo_market_start(sunday).tz_convert(I.NY).strftime("%a %H:%M") == "Fri 09:44"


def test_clock_offset(monkeypatch):
    monkeypatch.delenv(clock.OFFSET_ENV, raising=False)
    assert not clock.is_demo() and abs((clock.now() - pd.Timestamp.now(tz="UTC")).total_seconds()) < 1
    monkeypatch.setenv(clock.OFFSET_ENV, "-3600")
    assert abs((pd.Timestamp.now(tz="UTC") - clock.now()).total_seconds() - 3600) < 1
    assert abs(clock.now_ms() - (pd.Timestamp.now(tz="UTC").value // 1_000_000 - 3_600_000)) < 1000
    assert clock.is_demo()


def test_demo_env_points_only_at_local_simulators(tmp_path):
    real = pd.Timestamp("2026-11-10 23:00", tz="UTC")
    env = L.demo_env(tmp_path, real)
    assert float(env["FORECASTER_CLOCK_OFFSET_S"]) == (SF.demo_market_start(real) - real).total_seconds()
    assert env["FORECASTER_LIVE_FINNHUB_URL"].startswith("ws://127.0.0.1:")
    assert env["FORECASTER_LIVE_OANDA_URL"].startswith("http://127.0.0.1:")
    assert env["FORECASTER_CACHE_DIR"] == str(tmp_path / "data" / "demo")             # never the real data
    names = {s.name for s in L.build_demo_services(tmp_path, python="py")}
    assert names == {"simfeed", "live", "shadow", "dashboard"}                        # no monitor/webhook/tunnel


def test_prepare_demo_seeds_history_prev_closes_and_wipes_old_runs(tmp_path):
    mnow = pd.Timestamp("2026-11-10 09:44", tz=I.NY).tz_convert("UTC")
    (tmp_path / "shadow.sqlite").write_text("old")
    prices = SF.prepare_demo(tmp_path, mnow, run_lab=False, progress=lambda m: None)
    assert not (tmp_path / "shadow.sqlite").exists() or (tmp_path / "shadow.sqlite").read_bytes() != b"old"
    store = I.IntradayStore(tmp_path / "intraday")
    spy = store.read("alpaca", "SPY")
    assert spy.index.max() < mnow and spy.index.max() >= mnow - pd.Timedelta(minutes=2)   # history up to "now"
    assert (spy.index.tz_convert(I.NY).normalize().unique().size) >= 8
    assert prices["SPY"] == pytest.approx(spy["Close"].iloc[-1]) and SF.last_prices(tmp_path)["SPY"] == prices["SPY"]
    latest = lf.read_latest(lf.live_db(tmp_path))
    assert {"SPY", "QQQ", "XAU_USD", "USD_CAD", "DXY"} <= set(latest)
    assert all(r["prev_close"] and r["prev_close"] > 0 for r in latest.values())


def test_simulated_volatility_is_realistic():
    m = SF.Market({"SPY": 600.0}, seed=1)
    path = [m.step("SPY", 0.25) for _ in range(4 * 60 * 390)]                       # one session of 4 ticks/s
    daily = np.diff(np.log(path)).std() * np.sqrt(len(path))
    assert 0.004 < daily < 0.03                                                       # around 1% a day, not 30%


def test_live_feed_consumes_the_simulated_feeds(tmp_path, monkeypatch):
    """End to end: simulator servers -> live_feed -> live.sqlite, with the shifted clock."""
    monkeypatch.setenv(clock.OFFSET_ENV, str((SF.demo_market_start() - pd.Timestamp.now(tz="UTC")).total_seconds()))
    monkeypatch.setattr(SF, "FINNHUB_PORT", 28765)
    monkeypatch.setattr(SF, "OANDA_PORT", 28766)

    async def main():
        stop = asyncio.Event()
        srv = asyncio.create_task(SF.serve(SF.Market(dict(SF.START), seed=2), stop))
        await asyncio.sleep(0.8)
        cfg = lf.LiveConfig("demo", "demo", "demo", "practice", ["SPY", "QQQ"])
        asyncio.get_running_loop().call_later(3.0, stop.set)
        hub = await lf.run_service(cfg, tmp_path / "live.sqlite", stop, finnhub_url="ws://127.0.0.1:28765",
                                   oanda_host="http://127.0.0.1:28766", flush_every=0.2)
        await srv
        return hub

    hub = asyncio.run(main())
    assert hub.feeds["finnhub"].ticks > 3 and hub.feeds["oanda"].ticks > 3
    last = lf.read_latest(tmp_path / "live.sqlite")
    t = pd.Timestamp(last["SPY"]["ts_ms"], unit="ms", tz="UTC").tz_convert(I.NY)
    assert t.hour * 60 + t.minute in range(9 * 60 + 43, 9 * 60 + 46)                 # stamped in demo market time
    assert lf.session_open("stocks", clock.now())


def test_demo_banner_in_the_app(monkeypatch, tmp_path):
    from conftest import authed_app
    from pathlib import Path
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FORECASTER_DEMO", "1")
    at = authed_app(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    html = "\n".join(str(e.proto.body) for e in at.get("html"))
    assert "Demo mode: simulated market" in html
    monkeypatch.delenv("FORECASTER_DEMO")
    at = authed_app(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert "Demo mode: simulated market" not in "\n".join(str(e.proto.body) for e in at.get("html"))
