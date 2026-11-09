import json
import logging
import warnings

import httpx
import numpy as np
import pandas as pd
import pytest

from forecaster.config import load_settings
from forecaster.monitor import (
    build_quotes,
    Alert, Monitor, MonitorConfig, MonitorState, TelegramBot, check_events, check_gold_vwap, check_regimes,
    check_shocks, check_vix_spike, due, market_open, session_vwap, setup_logging, status_text,
)
from forecaster.pipeline import write_json
from forecaster.webhook_server import RateLimiter

NY = "America/New_York"


def ny(s):
    return pd.Timestamp(s, tz=NY).tz_convert("UTC")


@pytest.fixture
def state(tmp_path):
    return MonitorState(tmp_path / "m.sqlite")


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #

def test_market_hours():
    assert market_open(ny("2026-11-09 09:30")) and market_open(ny("2026-11-09 15:59"))
    assert not market_open(ny("2026-11-09 09:29")) and not market_open(ny("2026-11-09 16:00"))
    assert not market_open(ny("2026-11-08 12:00"))                     # Sunday


def test_due_every_15min_around_the_clock():
    t = ny("2026-11-09 10:00")
    assert due(t, None)
    assert not due(t, t - pd.Timedelta(minutes=10)) and due(t, t - pd.Timedelta(minutes=15))
    night = ny("2026-11-08 22:00")                                    # Sunday night: still 15 min
    assert not due(night, night - pd.Timedelta(minutes=10)) and due(night, night - pd.Timedelta(minutes=15))
    hourly = MonitorConfig(off_hours_every_min=60)                    # still configurable
    assert not due(night, night - pd.Timedelta(minutes=45), hourly) and due(night, night - pd.Timedelta(minutes=60), hourly)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

EVENTS = [{"date": "2026-11-10", "category": "CPI", "name": "CPI release", "impact": "high", "detail": ""},
          {"date": "2026-11-10", "category": "PPI", "name": "PPI release", "impact": "medium", "detail": ""},
          {"date": "2026-11-12", "category": "FOMC", "name": "FOMC rate decision", "impact": "high",
           "detail": "SEP + press conference"},
          {"date": "2026-11-10", "category": "Earnings", "name": "AAPL earnings", "impact": "high", "detail": ""}]


def test_event_warnings_24h_and_1h():
    # CPI 2026-11-10 08:30 ET
    assert [a.key for a in check_events(EVENTS, ny("2026-11-09 08:00"))] == []           # 24.5h away
    a24 = check_events(EVENTS, ny("2026-11-09 09:00"))
    assert [a.key for a in a24] == ["event:2026-11-10:CPI:24h"] and "08:30 ET" in a24[0].text
    assert check_events(EVENTS, ny("2026-11-10 07:00"))[0].key == "event:2026-11-10:CPI:24h"   # 1.5h: still 24h
    a1 = check_events(EVENTS, ny("2026-11-10 07:45"))
    assert [a.key for a in a1] == ["event:2026-11-10:CPI:1h"]
    assert check_events(EVENTS, ny("2026-11-10 08:31")) == []                            # passed
    fomc = check_events(EVENTS, ny("2026-11-12 13:10"))
    assert fomc[0].key == "event:2026-11-12:FOMC:1h" and "SEP" in fomc[0].text
    # medium PPI and earnings never warn (only CPI/PPI/FOMC/jobs at high impact)
    assert all("PPI" not in a.key and "Earnings" not in a.key for a in a24 + a1)


def test_vix_spike():
    daily = pd.Series([15.0, 16.0], index=pd.to_datetime(["2026-11-05", "2026-11-06"]))
    t = ny("2026-11-09 11:00")
    assert check_vix_spike(daily, 19.0, t) == []                                        # +18.75%
    a = check_vix_spike(daily, 19.3, t)
    assert a[0].key == "vix:2026-11-09" and "+21%" in a[0].text
    # today's own daily bar (if already cached) must not be the baseline
    daily2 = pd.concat([daily, pd.Series([19.5], index=pd.to_datetime(["2026-11-09"]))])
    assert check_vix_spike(daily2, 19.5, t)[0].key == "vix:2026-11-09"


def gold_bars(n=40, jump=0.0, start="2026-11-09 03:00"):
    idx = pd.date_range(pd.Timestamp(start, tz=NY), periods=n, freq="5min").tz_convert("UTC")
    rng = np.random.default_rng(0)
    c = 2650 + np.cumsum(rng.normal(0, 0.5, n))
    c[-1] += jump
    return pd.DataFrame({"Open": c, "High": c + 0.3, "Low": c - 0.3, "Close": c, "Volume": 100.0}, index=idx)


def test_session_vwap_resets_at_1800():
    b = gold_bars(n=6, start="2026-11-09 17:35")      # 17:35..18:00 ... crosses the 18:00 reset (no 17-18 bars in reality)
    df = session_vwap(b)
    assert df["session"].iloc[0] != df["session"].iloc[-1]
    assert df["bars_in_session"].iloc[-1] == 1


def test_gold_vwap_alert_beyond_2_sigma():
    assert check_gold_vwap(gold_bars()) == []
    up = check_gold_vwap(gold_bars(jump=15))
    assert up and up[0].key.endswith(":above") and "above today's session VWAP" in up[0].text
    down = check_gold_vwap(gold_bars(jump=-15))
    assert down[0].key.endswith(":below")
    assert check_gold_vwap(gold_bars(n=8, jump=15)) == []                               # too early in session
    assert check_gold_vwap(None) == [] and check_gold_vwap(pd.DataFrame()) == []


def test_regime_change(state):
    s1 = {"SPY": {"ticker": "SPY", "as_of": "2026-11-06", "regime": {"current": {"bull": .7, "bear": .2, "high_vol": .1}}}}
    assert check_regimes(s1, state) == []                                               # first sighting: just stored
    assert check_regimes(s1, state) == []                                               # unchanged
    s2 = {"SPY": {"ticker": "SPY", "as_of": "2026-11-09", "regime": {"current": {"bull": .2, "bear": .2, "high_vol": .6}}}}
    a = check_regimes(s2, state)
    assert a[0].key == "regime:SPY:2026-11-09:high_vol" and "bull -> high volatility" in a[0].text


def test_shocks_today_or_yesterday_only():
    sh = [{"tag": "MARKET", "date": "2026-11-08", "sentiment": -0.6, "z": -2.7, "headlines": 9},
          {"tag": "gold", "date": "2026-11-01", "sentiment": 0.6, "z": 2.4, "headlines": 5}]
    a = check_shocks(sh, ny("2026-11-09 10:00"))
    assert [x.key for x in a] == ["shock:MARKET:2026-11-08"] and "negative" in a[0].text


# --------------------------------------------------------------------------- #
# No duplicates
# --------------------------------------------------------------------------- #

def test_claim_is_once_only_and_survives_restart(tmp_path):
    s = MonitorState(tmp_path / "m.sqlite")
    assert s.claim("k", "t") and not s.claim("k", "t")
    assert not MonitorState(tmp_path / "m.sqlite").claim("k", "t")


def _monitor(tmp_path, now, sent, intraday=None, send_ok=True):
    settings = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path / "data")})
    snaps = tmp_path / "data" / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    write_json(snaps / "market.json", {"events": EVENTS, "shocks": []})

    class Store:
        def get(self, t):
            return pd.DataFrame({"Close": [1.0]})

        def read(self, t):
            return pd.DataFrame({"Close": [15.0, 16.0]}, index=pd.to_datetime(["2026-11-05", "2026-11-06"])) \
                if t == "^VIX" else None

    return Monitor(settings, MonitorState(tmp_path / "data" / "monitor.sqlite"),
                   send=lambda text: (sent.append(text), send_ok)[1], clock=lambda: now[0],
                   intraday=intraday or (lambda s: pd.DataFrame()), store=Store())


def test_monitor_tick_dedupes_across_runs(tmp_path):
    now = [ny("2026-11-09 10:00")]
    vix = pd.DataFrame({"Close": [20.0]}, index=[now[0]])
    sent = []
    m = _monitor(tmp_path, now, sent, intraday=lambda s: vix if s == "^VIX" else gold_bars(jump=15))
    assert m.run_checks() == 3                                    # CPI 24h, VIX spike, gold
    assert m.run_checks() == 0                                    # not due yet
    now[0] += pd.Timedelta(minutes=15)
    assert m.run_checks() == 0 and len(sent) == 3                 # due, but nothing new: no duplicates
    now[0] = ny("2026-11-10 07:45")
    m.run_checks()
    assert sum("CPI release" in t for t in sent) == 2             # 24h warning + 1h warning, once each
    assert (tmp_path / "data" / "monitor_heartbeat").exists()


def test_failed_send_is_retried(tmp_path):
    now = [ny("2026-11-09 10:00")]
    sent = []
    m = _monitor(tmp_path, now, sent, send_ok=False)
    m.run_checks()
    assert not m.state.was_sent("event:2026-11-10:CPI:24h")      # Telegram down: not marked sent
    m.send = lambda text: (sent.append(text), True)[1]
    now[0] += pd.Timedelta(minutes=15)
    m.run_checks()
    assert m.state.was_sent("event:2026-11-10:CPI:24h")


def test_tick_writes_prices_snapshot_and_health(tmp_path):
    from forecaster import status as health
    now = [ny("2026-11-09 10:00")]
    vix = pd.DataFrame({"Close": [17.6]}, index=[now[0]])
    cad = pd.DataFrame({"Close": [1.40]}, index=[now[0]])
    sent = []
    m = _monitor(tmp_path, now, sent, intraday=lambda s: {"^VIX": vix, "CAD=X": cad, "GC=F": gold_bars()}.get(s))
    m.run_checks()
    p = json.loads((tmp_path / "data" / "snapshots" / "prices.json").read_text())
    assert p["usd_cad"] == 1.40
    assert p["quotes"]["^VIX"]["price"] == 17.6 and p["quotes"]["^VIX"]["change"] == pytest.approx(17.6 / 16 - 1)
    assert p["quotes"]["^VIX"]["price_cad"] is None                          # points, never converted
    g = p["quotes"]["GC=F"]
    assert g["price_cad"] == pytest.approx(g["price"] * 1.40) and "z" in g and "vwap" in g
    probs = {x.source: x.message for x in health.problems(tmp_path / "data", now[0], ("prices",))}
    assert "no data for SPY, QQQ" in probs["prices"]         # no data for them -> failure recorded
    m.intraday = lambda s: {"^VIX": vix, "CAD=X": cad, "GC=F": gold_bars()}.get(s, gold_bars())
    now[0] += pd.Timedelta(minutes=15)
    m.run_checks()
    assert health.problems(tmp_path / "data", now[0], ("prices",)) == []


def test_build_quotes_cad_and_change():
    now = ny("2026-11-09 10:00")
    bars = pd.DataFrame({"Close": [100.0, 101.0]}, index=[now - pd.Timedelta(minutes=5), now])
    daily = pd.Series([95.0, 99.0, 100.5], index=pd.to_datetime(["2026-11-05", "2026-11-06", "2026-11-09"]))
    q = build_quotes({"SPY": bars, "CAD=X": pd.DataFrame({"Close": [1.38]}, index=[now])}, {"SPY": daily}, now)
    s = q["quotes"]["SPY"]
    assert s["prev_close"] == 99.0                      # today's partial daily bar is not "previous"
    assert s["change"] == pytest.approx(101 / 99 - 1) and s["price_cad"] == pytest.approx(101 * 1.38)
    assert "no intraday data for QQQ" in q["errors"]
    no_fx = build_quotes({"SPY": bars}, {}, now)["quotes"]["SPY"]
    assert no_fx["price_cad"] is None and no_fx["change"] is None


# --------------------------------------------------------------------------- #
# /status and the Telegram bot
# --------------------------------------------------------------------------- #

def test_status_text():
    snaps = {"SPY": {"ticker": "SPY", "currency": "CAD", "as_of": "2026-11-06",
                     "regime": {"current": {"bull": .7, "bear": .2, "high_vol": .1}},
                     "forecast": [{"quarter": "2027Q1", "p10": 800, "p50": 860, "p90": 920}]}}
    pf = {"total_value": 40000, "forecast": [{"quarter": "2027Q1", "p10": 37000, "p50": 40500, "p90": 44000}]}
    txt = status_text(ny("2026-11-09 07:00"), snaps, {"events": EVENTS}, pf, (17.2, ny("2026-11-09 06:55")))
    assert "Regime (SPY): bull 70%" in txt and "VIX: 17.2" in txt
    assert "Next big event: CPI release Tue Nov 10 08:30 ET (in 26h)" in txt
    assert "P10 C$800 · P50 C$860 · P90 C$920" in txt and "Portfolio C$40,000" in txt
    empty = status_text(ny("2026-11-09 07:00"), {}, None, None, None)
    assert "no forecast snapshot" in empty and "none in the events snapshot" in empty


class FakeTelegram:
    def __init__(self, updates):
        self.updates, self.sent, self.offsets = updates, [], []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/getUpdates"):
            off = int(req.url.params["offset"])
            self.offsets.append(off)
            return httpx.Response(200, json={"ok": True, "result": [u for u in self.updates if u["update_id"] >= off]})
        if req.url.path.endswith("/sendMessage"):
            self.sent.append(json.loads(req.content))
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


def upd(i, chat, text):
    return {"update_id": i, "message": {"chat": {"id": chat}, "text": text}}


def test_bot_answers_owner_only_and_persists_offset(state):
    fake = FakeTelegram([upd(10, 42, "/status"), upd(11, 999, "/status"), upd(12, 42, "/help"),
                         upd(13, 42, "hello"), upd(14, 42, "/status@MyBot")])
    bot = TelegramBot("TOKEN123", "42", state, status=lambda: "STATUS!", http=httpx.Client(transport=httpx.MockTransport(fake)))
    assert bot.poll_once(timeout=0) == 3
    assert [m["text"] for m in fake.sent] == ["STATUS!", "Commands: /status: regime, VIX, next big event, latest forecast.", "STATUS!"]
    assert all(m["chat_id"] == "42" for m in fake.sent)          # never replies to chat 999
    assert state.get("tg_offset") == "15"
    bot2 = TelegramBot("TOKEN123", "42", state, status=lambda: "x", http=httpx.Client(transport=httpx.MockTransport(fake)))
    assert bot2.poll_once(timeout=0) == 0 and fake.offsets[-1] == 15   # restart: old messages not re-answered


def test_bot_errors_never_log_token(state, caplog):
    def boom(req):
        raise httpx.ConnectError(f"failed {req.url}")
    bot = TelegramBot("SECRET-TOKEN", "42", state, status=lambda: "x", http=httpx.Client(transport=httpx.MockTransport(boom)))
    with caplog.at_level(logging.WARNING):
        assert bot.poll_once(timeout=0) == 0 and not bot.send("hi")
    assert "SECRET-TOKEN" not in caplog.text


def test_status_failure_is_reported_not_raised(state):
    fake = FakeTelegram([upd(1, 42, "/status")])
    def bad():
        raise RuntimeError("boom")
    bot = TelegramBot("T", "42", state, status=bad, http=httpx.Client(transport=httpx.MockTransport(fake)))
    assert bot.poll_once(timeout=0) == 1 and "Status unavailable" in fake.sent[0]["text"]


def test_setup_logging_writes_file_and_silences_httpx(tmp_path):
    setup_logging(tmp_path, "svc")
    logging.getLogger("forecaster.test").info("hello file")
    logging.getLogger("httpx").info("HTTP Request: POST https://api.telegram.org/botSECRET/sendMessage")
    for h in logging.getLogger().handlers:
        h.flush()
    text = (tmp_path / "svc.log").read_text()
    assert "hello file" in text and "SECRET" not in text
    for h in logging.getLogger().handlers:
        h.close()
    logging.getLogger().handlers[:] = []


# --------------------------------------------------------------------------- #
# Webhook rate limiting
# --------------------------------------------------------------------------- #

def test_rate_limiter_per_ip_and_total():
    rl = RateLimiter(per_ip=3, total=5, window_s=60)
    assert all(rl.allow("a", 0) for _ in range(3)) and not rl.allow("a", 1)
    assert rl.allow("b", 1) and rl.allow("b", 1) and not rl.allow("c", 1)                # total 5 reached
    assert rl.allow("a", 61)                                                             # window slid


def test_webhook_returns_429_before_checking_secret(tmp_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
        from fastapi.testclient import TestClient
    from forecaster.webhook_server import TelegramNotifier, create_app
    s = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path), "WEBHOOK_SECRET": "s"})
    c = TestClient(create_app(s, resolve_every_s=None, trust_proxy=True, notifier=TelegramNotifier(None, None),
                              rate_limiter=RateLimiter(per_ip=2, total=100)))
    h = {"CF-Connecting-IP": "203.0.113.9"}
    codes = [c.post("/tv-alert", json={"secret": "wrong"}, headers=h).status_code for _ in range(3)]
    assert codes == [401, 401, 429]
    assert c.post("/tv-alert", json={"secret": "wrong"}, headers={"CF-Connecting-IP": "198.51.100.1"}).status_code == 401
    assert c.get("/tv-alert").status_code == 405                                         # POST only
