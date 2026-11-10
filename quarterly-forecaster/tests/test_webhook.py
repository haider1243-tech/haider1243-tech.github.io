import json
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    # Starlette now prefers a package named httpx2; the established httpx works.
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

from forecaster.config import load_settings
from forecaster.pipeline import write_json
from forecaster.webhook_server import (
    Alert, AlertStore, Check, Context, RuleConfig, TelegramNotifier, check_events, check_regime,
    check_sentiment, check_vix, create_app, decide, event_time, format_message, outcome_stats,
    parse_alert_time, resolve_outcomes, resolve_symbol,
)

SECRET = "s3cret-value-123"
NOW = pd.Timestamp("2026-11-09 14:00", tz="UTC")       # Monday 09:00 New York
CFG = RuleConfig()


def regime_snap(bull=0.7, bear=0.2, hv=0.1, generated=NOW, as_of=None):
    return {"ticker": "SPY", "generated_at": generated.isoformat(),
            "as_of": (as_of or (NOW - pd.Timedelta(days=3)).normalize().tz_localize(None)).isoformat(),
            "regime": {"current": {"bull": bull, "bear": bear, "high_vol": hv}}}


def market_snap(events=(), shocks=(), sentiment=True, generated=NOW):
    return {"generated_at": generated.isoformat(), "events": list(events), "shocks": list(shocks),
            "sentiment": {"MARKET": [{"date": "2026-11-06", "sentiment": 0.0}]} if sentiment else {}}


def ev(date, category="CPI", impact="high", name=None, ticker=None, detail=""):
    return {"date": date, "category": category, "impact": impact, "name": name or f"{category} release",
            "ticker": ticker, "detail": detail, "points": 3}


def vix_series(last=16.0, prev=15.5, last_date="2026-11-06"):
    d = pd.Timestamp(last_date)
    return pd.Series([prev, last], index=[d - pd.offsets.BDay(1), d])


class FakePost:
    def __init__(self, status=200, raise_exc=None):
        self.calls, self.status, self.raise_exc = [], status, raise_exc

    def __call__(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if self.raise_exc:
            raise self.raise_exc
        return type("R", (), {"status_code": self.status, "json": lambda self: {"description": "Bad Request"}})()


# a healthy live feed: gold 2,650.50, 1.4 sigma above its session VWAP, tick 0.4 s old
LIVE_OK = {"price": 2650.5, "ts_ms": 0, "age_s": 0.4, "prev_close": 2640.0, "bars": 120,
           "vwap": 2645.0, "sd": 4.0, "z": 1.375}


def live_ok(now=None):
    return dict(LIVE_OK)


@pytest.fixture
def env(tmp_path):
    settings = load_settings(root=tmp_path, env={
        "FORECASTER_CACHE_DIR": str(tmp_path / "data"), "WEBHOOK_SECRET": SECRET,
        "TELEGRAM_BOT_TOKEN": "123:TOKEN", "TELEGRAM_CHAT_ID": "42"})
    snaps = tmp_path / "data" / "snapshots"
    snaps.mkdir(parents=True)
    write_json(snaps / "ticker_SPY.json", regime_snap())
    write_json(snaps / "market.json", market_snap())
    state = {"vix": vix_series()}
    ctx = Context(settings, vix_loader=lambda: state["vix"], live_loader=live_ok)
    post = FakePost()
    store = AlertStore(tmp_path / "data" / "alerts.sqlite")
    app = create_app(settings, store, ctx, TelegramNotifier("123:TOKEN", "42", post=post),
                     clock=lambda: NOW, resolve_every_s=None)
    return {"client": TestClient(app), "store": store, "post": post, "snaps": snaps, "state": state,
            "settings": settings, "tmp": tmp_path}


def body(**over):
    b = {"ticker": "XAUUSD", "price": "2650.40", "strategy": "vwap_gold", "side": "long",
         "time": "2026-11-09T14:00:00Z", "secret": SECRET}
    b.update(over)
    return b


# --------------------------------------------------------------------------- #
# Auth / validation
# --------------------------------------------------------------------------- #

def test_fails_closed_without_secret_configured(tmp_path):
    s = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path)})
    c = TestClient(create_app(s, resolve_every_s=None, clock=lambda: NOW))
    assert c.post("/tv-alert", json=body()).status_code == 503


@pytest.mark.parametrize("bad", ["wrong", "", None, 12345, SECRET + "x"])
def test_rejects_bad_secret(env, bad):
    b = body()
    if bad is None:
        del b["secret"]
    else:
        b["secret"] = bad
    r = env["client"].post("/tv-alert", json=b)
    assert r.status_code == 401
    assert env["store"].frame().empty and env["post"].calls == []


def test_secret_never_stored_or_echoed(env):
    r = env["client"].post("/tv-alert", json=body())
    assert r.status_code == 200 and SECRET not in r.text
    raw = (env["tmp"] / "data" / "alerts.sqlite").read_bytes()
    assert SECRET.encode() not in raw
    assert SECRET not in env["post"].calls[0][1]["text"]


def test_accepts_text_plain_json(env):
    r = env["client"].post("/tv-alert", content=json.dumps(body()), headers={"Content-Type": "text/plain"})
    assert r.status_code == 200 and r.json()["verdict"] in {"TAKE", "CAUTION", "SKIP"}


def test_rejects_non_json_and_bad_fields(env):
    c = env["client"]
    assert c.post("/tv-alert", content="not json").status_code == 400
    assert c.post("/tv-alert", content="[1,2]").status_code == 400
    assert c.post("/tv-alert", json=body(side="sideways")).status_code == 422
    assert c.post("/tv-alert", json=body(price="{{close}}")).status_code == 422   # unsubstituted
    assert c.post("/tv-alert", json=body(price="-3")).status_code == 422
    assert c.post("/tv-alert", content="x" * 20_000).status_code == 413


def test_ip_allowlist(tmp_path):
    s = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path), "WEBHOOK_SECRET": SECRET,
                                          "WEBHOOK_ALLOWED_IPS": "52.89.214.238, 10.0.0.1"})
    assert s.webhook_allowed_ips == ("52.89.214.238", "10.0.0.1")
    c = TestClient(create_app(s, resolve_every_s=None, clock=lambda: NOW,
                              notifier=TelegramNotifier(None, None)))
    assert c.post("/tv-alert", json=body()).status_code == 403          # TestClient host is "testclient"
    c2 = TestClient(create_app(s, resolve_every_s=None, clock=lambda: NOW, trust_proxy=True,
                               notifier=TelegramNotifier(None, None)))
    assert c2.post("/tv-alert", json=body(), headers={"X-Forwarded-For": "52.89.214.238"}).status_code == 200


def test_alert_model_normalises():
    a = Alert.model_validate(body(side="BUY", price="2,650.4", ticker="OANDA:XAUUSD"))
    assert a.side == "long" and a.price == 2650.4 and a.symbol == "XAUUSD"
    assert Alert.model_validate(body(side="sell")).side == "short"


def test_parse_alert_time():
    t, ok = parse_alert_time("2026-11-09T13:59:30Z", NOW)
    assert ok and t == pd.Timestamp("2026-11-09 13:59:30", tz="UTC")
    assert parse_alert_time("{{timenow}}", NOW) == (NOW, False)
    assert parse_alert_time("1999-01-01T00:00:00Z", NOW) == (NOW, False)    # implausible
    assert parse_alert_time(None, NOW) == (NOW, False)


def test_resolve_symbol():
    assert resolve_symbol("XAUUSD") == ("GC=F", "GC=F", ("gold",))
    assert resolve_symbol("AAPL") == ("AAPL", "AAPL", ("MARKET", "AAPL"))


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def test_event_time_uses_release_times():
    assert event_time(ev("2026-11-10", "CPI")) == pd.Timestamp("2026-11-10 13:30", tz="UTC")   # 08:30 EST
    assert event_time(ev("2026-12-09", "FOMC")) == pd.Timestamp("2026-12-09 19:00", tz="UTC")  # 14:00 EST
    assert event_time(ev("2026-07-29", "FOMC")) == pd.Timestamp("2026-07-29 18:00", tz="UTC")  # EDT
    assert event_time(ev("2026-11-10", "Earnings", detail="Q4 2026 after close")) == \
        pd.Timestamp("2026-11-10 21:05", tz="UTC")


def test_events_skip_caution_ok():
    skip = check_events(market_snap([ev("2026-11-09", "CPI")]), NOW, CFG)          # 13:30 UTC, just passed
    assert skip.level == "skip"
    soon = check_events(market_snap([ev("2026-11-09", "FOMC")]), NOW, CFG)         # 19:00 UTC = 5h away
    assert soon.level == "caution" and "FOMC" in soon.reason
    later = check_events(market_snap([ev("2026-11-10", "CPI")]), NOW, CFG)         # ~23.5h
    assert later.level == "caution"
    far = check_events(market_snap([ev("2026-11-12", "CPI")]), NOW, CFG)
    assert far.level == "ok"
    medium = check_events(market_snap([ev("2026-11-09", "PPI", impact="medium")]), NOW, CFG)
    assert medium.level == "ok"
    own = check_events(market_snap([ev("2026-11-09", "Earnings", impact="low", ticker="AAPL",
                                       detail="before open")]), NOW - pd.Timedelta(hours=2), CFG, ("AAPL",))
    assert own.level == "skip"                                                      # own earnings imminent
    assert check_events(None, NOW, CFG).level == "unknown"
    stale = check_events(market_snap(generated=NOW - pd.Timedelta(days=20)), NOW, CFG)
    assert stale.level == "unknown"


def test_regime_check():
    assert check_regime(regime_snap(), "SPY", False, "long", NOW, CFG).level == "ok"
    assert check_regime(regime_snap(0.2, 0.4, 0.4), "SPY", False, "long", NOW, CFG).level == "caution"
    assert check_regime(regime_snap(0.8, 0.1, 0.1), "SPY", False, "short", NOW, CFG).level == "caution"
    fb = check_regime(regime_snap(), "SPY", True, "long", NOW, CFG)
    assert "market fallback" in fb.reason
    old = check_regime(regime_snap(generated=NOW - pd.Timedelta(days=30)), "SPY", False, "long", NOW, CFG)
    assert old.level == "unknown" and "30 days old" in old.reason
    # generated today from 2022 prices: still stale
    old_data = check_regime(regime_snap(as_of=pd.Timestamp("2022-12-28")), "SPY", False, "long", NOW, CFG)
    assert old_data.level == "unknown" and "days old" in old_data.reason
    assert check_regime(None, "X", False, "long", NOW, CFG).level == "unknown"


def test_sentiment_check_direction_and_tags():
    neg_gold = {"tag": "gold", "date": "2026-11-08", "z": -2.6, "sentiment": -0.5, "headlines": 5}
    neg_mkt = {**neg_gold, "tag": "MARKET"}
    m = market_snap(shocks=[neg_gold])
    assert check_sentiment(m, "long", NOW, CFG, ("gold",)).level == "caution"
    assert check_sentiment(m, "short", NOW, CFG, ("gold",)).level == "ok"          # aligned with a short
    # equity-market shock doesn't flag a gold trade, but does flag an index trade
    m2 = market_snap(shocks=[neg_mkt])
    assert check_sentiment(m2, "long", NOW, CFG, ("gold",)).level == "ok"
    assert check_sentiment(m2, "long", NOW, CFG, ("MARKET",)).level == "caution"
    old = market_snap(shocks=[{**neg_gold, "date": "2026-10-20"}])
    assert check_sentiment(old, "long", NOW, CFG, ("gold",)).level == "ok"
    assert check_sentiment(market_snap(sentiment=False), "long", NOW, CFG, ("gold",)).level == "unknown"


def test_vix_check():
    assert check_vix(vix_series(16), NOW, CFG).level == "ok"
    assert check_vix(vix_series(32, 31), NOW, CFG).level == "caution"
    assert check_vix(vix_series(45, 40), NOW, CFG).level == "skip"
    spike = check_vix(vix_series(24, 19), NOW, CFG)
    assert spike.level == "caution" and "spiked" in spike.reason
    assert check_vix(vix_series(16, last_date="2026-10-20"), NOW, CFG).level == "unknown"
    assert check_vix(None, NOW, CFG).level == "unknown"


def test_decide_rules():
    ok = Check("a", "ok", "fine")
    c = Check("b", "caution", "careful")
    u = Check("c", "unknown", "no data")
    s = Check("d", "skip", "stop")
    assert decide([ok, ok], CFG)[0] == "TAKE"
    assert decide([ok, c], CFG)[0] == "CAUTION"
    assert decide([ok, u], CFG)[0] == "CAUTION"
    assert decide([u, u, u, u], CFG)[0] == "CAUTION"              # missing data never escalates to SKIP
    assert decide([c, c, c], CFG)[0] == "SKIP"
    v, reasons = decide([ok, s, u], CFG)
    assert v == "SKIP" and reasons[0] == "stop" and reasons[1].startswith("can't verify")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_take_alert_logged_and_sent(env):
    r = env["client"].post("/tv-alert", json=body())
    j = r.json()
    assert r.status_code == 200 and j["verdict"] == "TAKE" and not j["duplicate"]
    row = env["store"].get(j["id"])
    assert row["forecaster_ticker"] == "GC=F" and row["price_symbol"] == "GC=F" and row["time_from_tv"] == 1
    assert row["telegram_status"] == "sent"
    url, payload = env["post"].calls[0]
    assert url.endswith("/bot123:TOKEN/sendMessage") and payload["chat_id"] == "42"
    assert "<b>TAKE</b>" in payload["text"] and "XAUUSD" in payload["text"]
    f = env["store"].frame()
    assert sorted(f["horizon"]) == ["1d", "1h", "4h"] and (f["status"] == "pending").all()


def test_skip_on_fomc_and_vix(env):
    write_json(env["snaps"] / "market.json", market_snap([ev("2026-11-09", "FOMC")]))
    env["state"]["vix"] = vix_series(42, 35)
    j = env["client"].post("/tv-alert", json=body()).json()
    assert j["verdict"] == "SKIP"
    assert any("VIX 42.0" in r for r in j["reasons"]) and any("FOMC" in r for r in j["reasons"])
    assert "⛔" in env["post"].calls[0][1]["text"]


def test_duplicate_alert_not_resent(env):
    c = env["client"]
    a = c.post("/tv-alert", json=body()).json()
    b = c.post("/tv-alert", json=body()).json()
    assert b["duplicate"] and b["id"] == a["id"] and b["verdict"] == a["verdict"]
    assert len(env["post"].calls) == 1 and len(env["store"].frame()) == 3


def test_telegram_failure_still_logs_alert_without_token(env, tmp_path):
    for post in (FakePost(status=400), FakePost(raise_exc=RuntimeError("https://api.telegram.org/bot123:TOKEN/x"))):
        settings = env["settings"]
        store = AlertStore(tmp_path / f"a{id(post)}.sqlite")
        c = TestClient(create_app(settings, store, Context(settings, vix_loader=vix_series, live_loader=live_ok),
                                  TelegramNotifier("123:TOKEN", "42", post=post),
                                  clock=lambda: NOW, resolve_every_s=None))
        j = c.post("/tv-alert", json=body()).json()
        row = store.get(j["id"])
        assert row["telegram_status"] == "error" and "TOKEN" not in (row["telegram_error"] or "")


def test_telegram_unconfigured_is_recorded(env, tmp_path):
    settings = env["settings"]
    store = AlertStore(tmp_path / "u.sqlite")
    c = TestClient(create_app(settings, store, Context(settings, vix_loader=vix_series, live_loader=live_ok),
                              TelegramNotifier(None, None), clock=lambda: NOW, resolve_every_s=None))
    j = c.post("/tv-alert", json=body()).json()
    assert store.get(j["id"])["telegram_status"] == "error"


def test_no_snapshots_gives_caution_not_skip(tmp_path):
    s = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path / "empty"), "WEBHOOK_SECRET": SECRET})
    c = TestClient(create_app(s, resolve_every_s=None, clock=lambda: NOW, notifier=TelegramNotifier(None, None)))
    j = c.post("/tv-alert", json=body()).json()
    assert j["verdict"] == "CAUTION"
    assert all(r.startswith("can't verify") for r in j["reasons"])


def test_format_message_escapes_html():
    a = Alert.model_validate(body(strategy="<script>x</script>"))
    msg = format_message(a, NOW, "CAUTION", [], [Check("events", "caution", "CPI & PPI <soon>")])
    assert "<script>" not in msg and "&lt;script&gt;" in msg and "&amp;" in msg


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #

def _prices(start="2026-11-09 12:00", periods=600, step=0.5, base=100.0):
    idx = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    return pd.Series(base + step * np.arange(periods), index=idx)


def test_outcomes_resolve_when_due(env):
    c = env["client"]
    long_id = c.post("/tv-alert", json=body()).json()["id"]
    short_id = c.post("/tv-alert", json=body(side="short", time="2026-11-09T14:00:01Z")).json()["id"]
    px = _prices()
    fetch = lambda sym, a, b: px  # noqa: E731
    store = env["store"]
    assert resolve_outcomes(store, fetch, NOW + pd.Timedelta(minutes=30)) == 0         # nothing due
    assert resolve_outcomes(store, fetch, NOW + pd.Timedelta(hours=1, minutes=19)) == 0  # within grace
    assert resolve_outcomes(store, fetch, NOW + pd.Timedelta(hours=1, minutes=21)) == 2
    f = store.frame().set_index(["id", "horizon"])
    ref = px.loc[:NOW].iloc[-1]
    at = px.loc[:NOW + pd.Timedelta(hours=1)].iloc[-1]
    assert f.loc[(long_id, "1h"), "ret"] == pytest.approx(at / ref - 1)
    assert f.loc[(long_id, "1h"), "signed_ret"] > 0 > f.loc[(short_id, "1h"), "signed_ret"]
    assert f.loc[(long_id, "1h"), "mae"] >= 0 and f.loc[(short_id, "1h"), "mfe"] <= 0  # monotone rising price
    assert f.loc[(long_id, "4h"), "status"] == "pending"
    assert resolve_outcomes(store, fetch, NOW + pd.Timedelta(days=2)) == 4
    assert (store.frame()["status"] == "done").all()


def test_outcomes_use_source_price_not_alert_price(env):
    aid = env["client"].post("/tv-alert", json=body(price="2650.40")).json()["id"]  # spot; source ~100s
    px = _prices()
    resolve_outcomes(env["store"], lambda s, a, b: px, NOW + pd.Timedelta(days=2))
    import sqlite3
    con = sqlite3.connect(env["store"].path)
    refs = {r[0] for r in con.execute("SELECT ref_price FROM outcomes WHERE alert_id=?", (aid,))}
    con.close()
    assert refs == {px.loc[:NOW].iloc[-1]}           # source close at alert time, not 2650.40


def test_outcome_weekend_uses_last_close(env):
    fri = pd.Timestamp("2026-11-13 20:30", tz="UTC")
    c = TestClient(create_app(env["settings"], env["store"], Context(env["settings"], vix_loader=vix_series, live_loader=live_ok),
                              TelegramNotifier(None, None), clock=lambda: fri, resolve_every_s=None))
    c.post("/tv-alert", json=body(time="2026-11-13T20:30:00Z"))
    px = _prices("2026-11-13 18:00", periods=36)                     # trading stops 21:00 Friday
    resolve_outcomes(env["store"], lambda s, a, b: px, fri + pd.Timedelta(days=2))
    d = env["store"].frame().set_index("horizon")
    assert (d["status"] == "done").all()
    assert d.loc["1d", "ret"] == pytest.approx(px.iloc[-1] / px.loc[:fri].iloc[-1] - 1)


def test_outcome_gives_up_when_no_data(env):
    env["client"].post("/tv-alert", json=body())
    store = env["store"]
    empty = lambda s, a, b: pd.Series(dtype=float)  # noqa: E731
    resolve_outcomes(store, empty, NOW + pd.Timedelta(days=2))
    assert (store.frame()["status"] == "pending").all()
    resolve_outcomes(store, empty, NOW + pd.Timedelta(days=10))
    assert (store.frame()["status"] == "unavailable").all()
    def boom(s, a, b):
        raise ConnectionError("network")
    resolve_outcomes(store, boom, NOW + pd.Timedelta(days=10))       # doesn't raise


def test_outcome_stats(env):
    c = env["client"]
    c.post("/tv-alert", json=body())
    write_json(env["snaps"] / "market.json", market_snap([ev("2026-11-09", "CPI")]))
    c.post("/tv-alert", json=body(time="2026-11-09T14:00:02Z"))         # SKIP (CPI)
    resolve_outcomes(env["store"], lambda s, a, b: _prices(), NOW + pd.Timedelta(days=2))
    st = outcome_stats(env["store"])
    assert set(st["verdict"]) == {"TAKE", "SKIP"} and (st["n"] == 1).all()
    assert list(st["verdict"])[:3] == ["TAKE"] * 3
    assert list(st["horizon"])[:3] == ["1h", "4h", "1d"]
    assert outcome_stats(AlertStore(env["tmp"] / "empty.sqlite")).empty


def test_health(env):
    assert env["client"].get("/health").json() == {"ok": True}
    assert env["client"].get("/docs").status_code == 404


# --------------------------------------------------------------------------- #
# Live gold check (current price + session VWAP from live_feed)
# --------------------------------------------------------------------------- #

def _live(**over):
    d = dict(LIVE_OK)
    d.update(over)
    if "price" in over and "z" not in over:
        d["z"] = (d["price"] - d["vwap"]) / d["sd"]
    return d


@pytest.mark.parametrize("side,price,level,text", [
    ("long", 2650.5, "ok", "+1.4σ from session VWAP 2,645.00"),
    ("long", 2644.6, "caution", "wrong side of VWAP"),          # -0.1σ: noise around VWAP
    ("long", 2642.0, "skip", "already back below VWAP, the long setup has failed"),
    ("long", 2654.0, "caution", "stretched past the 2σ band"),
    ("short", 2640.0, "ok", "-1.2σ"),
    ("short", 2648.0, "skip", "already back above VWAP, the short setup has failed"),
    ("short", 2635.0, "caution", "stretched"),
])
def test_check_live_gold_levels(side, price, level, text):
    from forecaster.webhook_server import check_live_gold
    a = Alert.model_validate(body(side=side, price=str(price)))
    c = check_live_gold(a, _live(price=price), RuleConfig())
    assert c.level == level and text in c.reason, c.reason
    assert c.data["price"] == price


def test_check_live_gold_missing_stale_and_price_mismatch():
    from forecaster.webhook_server import check_live_gold
    cfg = RuleConfig()
    a = Alert.model_validate(body(price="2650.40"))
    assert check_live_gold(a, None, cfg).level == "unknown"                       # never escalates
    stale = check_live_gold(a, _live(age_s=42.0), cfg)
    assert stale.level == "unknown" and "42 s old" in stale.reason
    early = check_live_gold(a, _live(z=None, vwap=None, sd=None), cfg)
    assert early.level == "unknown" and "VWAP not formed yet" in early.reason
    off = check_live_gold(Alert.model_validate(body(price="2670.00")), _live(), cfg)
    assert off.level == "caution" and "+0.74% from live 2,650.50" in off.reason
    both = check_live_gold(Alert.model_validate(body(price="2660.00")), _live(price=2640.0), cfg)
    assert both.level == "skip" and "setup has failed" in both.reason and "from live 2,640.00" in both.reason
    fut = check_live_gold(Alert.model_validate(body(ticker="GC1!", price="2672.00")), _live(), cfg)
    assert fut.level == "ok"                                                       # futures basis: no price check


def test_live_gold_drives_the_verdict(env):
    c, state = env["client"], {"live": _live(price=2642.0)}
    ctx = Context(env["settings"], vix_loader=vix_series, live_loader=lambda now: state["live"])
    app = create_app(env["settings"], env["store"], ctx, TelegramNotifier(None, None),
                     clock=lambda: NOW, resolve_every_s=None)
    c = TestClient(app)
    r = c.post("/tv-alert", json=body(price="2642.00", time="2026-11-09T14:05:00Z")).json()
    assert r["verdict"] == "SKIP" and any("long setup has failed" in x for x in r["reasons"])
    state["live"] = None
    r = c.post("/tv-alert", json=body(time="2026-11-09T14:06:00Z")).json()
    assert r["verdict"] == "CAUTION" and any("no live gold price" in x for x in r["reasons"])
    state["live"] = _live()
    r = c.post("/tv-alert", json=body(time="2026-11-09T14:07:00Z")).json()
    assert r["verdict"] == "TAKE"
    stored = json.loads(env["store"].get(r["id"])["checks"])
    live = next(x for x in stored if x["name"] == "live_gold")
    assert live["data"]["price"] == 2650.5 and live["data"]["z"] == pytest.approx(1.375)   # journaled
    # non-gold alerts don't need the live feed
    state["live"] = None
    r = c.post("/tv-alert", json=body(ticker="SPY", price="600", time="2026-11-09T14:08:00Z")).json()
    assert not any("live gold" in x for x in r["reasons"])
