import json
import warnings

import httpx
import pandas as pd
import pytest

from forecaster.config import load_settings
from forecaster.execution import (
    HARD_MAX_RISK_PCT, HOSTS, Executor, LiveTradingNotAllowed, OandaClient, OrderRequest, OrderStore,
    load_execution_config, trading_day,
)
from forecaster.pipeline import write_json
from forecaster.webhook_server import AlertStore, Context, TelegramNotifier, create_app

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

NOW = pd.Timestamp("2026-11-09 15:00", tz="UTC")
TOKEN = "practice-token-abc123"
BASE_ENV = {"OANDA_API_TOKEN": TOKEN, "OANDA_ACCOUNT_ID": "101-001-1234567-001", "EXECUTION_ENABLED": "true",
            "MAX_POSITION_UNITS": "10", "MAX_DAILY_LOSS": "200", "RISK_PER_TRADE_PCT": "1"}


class FakeOanda:
    """In-memory OANDA v20 endpoints. Records every request."""

    def __init__(self, nav=10_000.0, bid=2650.0, ask=2650.5, positions=None, open_trades=0,
                 order_response=None, fail=None):
        self.nav, self.bid, self.ask = nav, bid, ask
        self.positions = positions or []
        self.open_trades = open_trades
        self.order_response = order_response
        self.fail = fail or set()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for frag in self.fail:
            if frag in path:
                return httpx.Response(503, json={"errorMessage": "Service unavailable"})
        if path.endswith("/summary"):
            return httpx.Response(200, json={"account": {"NAV": str(self.nav), "openTradeCount": self.open_trades,
                                                         "currency": "USD"}})
        if path.endswith("/openPositions"):
            return httpx.Response(200, json={"positions": self.positions})
        if path.endswith("/instruments"):
            return httpx.Response(200, json={"instruments": [{"name": "XAU_USD", "displayPrecision": 3,
                                                              "tradeUnitsPrecision": 0, "minimumTradeSize": "1",
                                                              "maximumOrderUnits": "5000"}]})
        if path.endswith("/pricing"):
            return httpx.Response(200, json={"prices": [{"bids": [{"price": str(self.bid)}],
                                                         "asks": [{"price": str(self.ask)}]}]})
        if path.endswith("/orders"):
            if self.order_response is not None:
                return self.order_response
            units = json.loads(request.content)["order"]["units"]
            return httpx.Response(201, json={"orderFillTransaction": {"id": "777", "price": str(self.ask),
                                                                      "units": units}})
        if "/positions/" in path and path.endswith("/close"):
            return httpx.Response(200, json={"longOrderFillTransaction": {"id": "888"}})
        return httpx.Response(404, json={"errorMessage": "not found"})

    def orders_sent(self):
        return [json.loads(r.content)["order"] for r in self.requests if r.url.path.endswith("/orders")]


@pytest.fixture
def make(tmp_path):
    def _make(env=None, fake=None, dotenv=None, clock=NOW):
        cfg = load_execution_config(tmp_path, env={**BASE_ENV, **(env or {})}, dotenv=dotenv or {},
                                    data_dir=tmp_path / "data")
        fake = fake or FakeOanda()
        client = OandaClient(cfg, transport=httpx.MockTransport(fake))
        notes = []
        ex = Executor(cfg, OrderStore(tmp_path / "data" / "alerts.sqlite"), client,
                      clock=lambda: clock, notify=notes.append)
        return ex, fake, notes
    return _make


def req(alert_id=1, side="long", price=2650.2, stop=2640.0, target=2670.0, verdict="TAKE", t=NOW, symbol="XAUUSD"):
    return OrderRequest(alert_id, symbol, side, price, stop, target, t, verdict)


# --------------------------------------------------------------------------- #
# Live-trading gate
# --------------------------------------------------------------------------- #

def test_default_is_practice_and_disabled(tmp_path):
    cfg = load_execution_config(tmp_path, env={}, dotenv={})
    assert cfg.environment == "practice" and cfg.host == HOSTS["practice"] == "https://api-fxpractice.oanda.com"
    assert not cfg.enabled and not cfg.is_live


def test_live_refused_without_dotenv_flag(tmp_path):
    with pytest.raises(LiveTradingNotAllowed):
        load_execution_config(tmp_path, env={"OANDA_ENVIRONMENT": "live"}, dotenv={})
    # exported shell variable does NOT count
    with pytest.raises(LiveTradingNotAllowed):
        load_execution_config(tmp_path, env={"OANDA_ENVIRONMENT": "live", "LIVE_TRADING": "true"}, dotenv={})
    # sloppy values don't count either
    for v in ("True", "1", "yes", " true"):
        with pytest.raises(LiveTradingNotAllowed):
            load_execution_config(tmp_path, env={"OANDA_ENVIRONMENT": "live"}, dotenv={"LIVE_TRADING": v})


def test_live_requires_flag_in_real_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OANDA_ENVIRONMENT", "live")
    monkeypatch.setenv("LIVE_TRADING", "true")                 # shell export: ignored
    (tmp_path / ".env").write_text("OANDA_API_TOKEN=x\n")
    with pytest.raises(LiveTradingNotAllowed):
        load_execution_config(tmp_path)
    (tmp_path / ".env").write_text("OANDA_API_TOKEN=x\nLIVE_TRADING=true\n")
    cfg = load_execution_config(tmp_path)
    assert cfg.is_live and cfg.host == "https://api-fxtrade.oanda.com"


def test_flag_alone_stays_on_practice(tmp_path):
    cfg = load_execution_config(tmp_path, env={}, dotenv={"LIVE_TRADING": "true"})
    assert cfg.environment == "practice" and cfg.host == HOSTS["practice"]


def test_limits_have_hard_ceilings(tmp_path):
    with pytest.raises(ValueError):
        load_execution_config(tmp_path, env={"RISK_PER_TRADE_PCT": str(HARD_MAX_RISK_PCT + 1)}, dotenv={})
    with pytest.raises(ValueError):
        load_execution_config(tmp_path, env={"MAX_POSITION_UNITS": "1000000"}, dotenv={})
    with pytest.raises(ValueError):
        load_execution_config(tmp_path, env={"OANDA_ENVIRONMENT": "paper"}, dotenv={})


# --------------------------------------------------------------------------- #
# Orders only on TAKE
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("verdict", ["CAUTION", "SKIP"])
def test_non_take_never_touches_oanda(make, verdict):
    ex, fake, notes = make()
    d = ex.handle(req(verdict=verdict))
    assert d.status == "not_taken" and fake.requests == [] and notes == []
    assert ex.store.orders().iloc[0]["status"] == "not_taken"


def test_disabled_never_touches_oanda(make):
    ex, fake, _ = make(env={"EXECUTION_ENABLED": "false"})
    assert ex.handle(req()).status == "blocked" and fake.requests == []


def test_take_places_market_order_with_stop_and_target(make):
    ex, fake, notes = make()
    d = ex.handle(req())
    assert d.status == "filled" and d.fill_price == 2650.5 and d.oanda_id == "777"
    (order,) = fake.orders_sent()
    # risk 1% of 10,000 = 100; entry at ask 2650.5, stop 2640 -> 10.5/unit -> 9 units (cap 10)
    assert order == {"type": "MARKET", "instrument": "XAU_USD", "units": "9", "timeInForce": "FOK",
                     "positionFill": "DEFAULT",
                     "stopLossOnFill": {"price": "2640.000", "timeInForce": "GTC"},
                     "takeProfitOnFill": {"price": "2670.000", "timeInForce": "GTC"},
                     "clientExtensions": {"id": "tv-alert-1", "tag": "vwap_gold"}}
    r = fake.requests[-1]
    assert r.url.host == "api-fxpractice.oanda.com"
    assert r.headers["authorization"] == f"Bearer {TOKEN}" and TOKEN not in str(r.url)
    assert "practice" in notes[-1] and "filled" in notes[-1]


def test_short_units_are_negative(make):
    ex, fake, _ = make()
    d = ex.handle(req(side="short", price=2650.2, stop=2660.0, target=2630.0))
    assert d.status == "filled"
    assert fake.orders_sent()[0]["units"] == "-10"       # risk 100 / 10 per unit = 10, = cap


def test_token_never_stored(make, tmp_path):
    ex, _, _ = make()
    ex.handle(req())
    assert TOKEN.encode() not in (tmp_path / "data" / "alerts.sqlite").read_bytes()


def test_mt4_account_omits_client_extensions(make):
    ex, fake, _ = make(env={"OANDA_MT4_ACCOUNT": "true"})
    ex.handle(req())
    assert "clientExtensions" not in fake.orders_sent()[0]


# --------------------------------------------------------------------------- #
# Hard limits
# --------------------------------------------------------------------------- #

def test_position_cap_counts_existing_position(make):
    held = [{"instrument": "XAU_USD", "long": {"units": "7"}, "short": {"units": "0"}}]
    ex, fake, _ = make(fake=FakeOanda(positions=held))
    d = ex.handle(req())
    assert d.status == "filled" and fake.orders_sent()[0]["units"] == "3"     # 10 cap - 7 held
    assert "capped from 9" in d.reason


def test_no_room_under_cap_blocks(make):
    held = [{"instrument": "XAU_USD", "long": {"units": "10"}, "short": {"units": "0"}}]
    ex, fake, _ = make(fake=FakeOanda(positions=held))
    d = ex.handle(req())
    assert d.status == "blocked" and "below minimum" in d.reason and fake.orders_sent() == []


def test_oversized_existing_position_trips_kill_switch(make):
    held = [{"instrument": "XAU_USD", "long": {"units": "25"}, "short": {"units": "0"}}]
    ex, fake, notes = make(fake=FakeOanda(positions=held))
    d = ex.handle(req())
    assert d.status == "blocked" and ex.kill_active() and fake.orders_sent() == []
    assert any("Kill switch" in n for n in notes)


def test_max_open_trades(make):
    ex, fake, _ = make(fake=FakeOanda(open_trades=1))
    assert "open trades" in ex.handle(req()).reason and fake.orders_sent() == []


def test_daily_loss_limit_and_reset(make):
    fake = FakeOanda(nav=10_000)
    ex, _, _ = make(fake=fake)
    ex.handle(req(alert_id=1, verdict="CAUTION"))                  # no API call, no NAV snapshot
    fake.nav = 10_000
    assert ex.handle(req(alert_id=2)).status == "filled"           # day start NAV recorded = 10,000
    fake.nav = 9_790                                               # -210 > 200 limit
    d = ex.handle(req(alert_id=3, t=NOW))
    assert d.status == "blocked" and "daily loss" in d.reason
    # next trading day (after 17:00 New York) the limit resets
    tomorrow = NOW + pd.Timedelta(days=1)
    ex.clock = lambda: tomorrow
    assert ex.handle(req(alert_id=4, t=tomorrow)).status == "filled"


def test_trading_day_rolls_at_1700_new_york():
    assert trading_day(pd.Timestamp("2026-11-09 21:59", tz="UTC")) == "2026-11-09"   # 16:59 EST
    assert trading_day(pd.Timestamp("2026-11-09 22:00", tz="UTC")) == "2026-11-10"   # 17:00 EST


def test_kill_switch_file_blocks_and_resume(make):
    ex, fake, _ = make()
    ex.trip_kill("test")
    assert ex.handle(req(alert_id=1)).reason == "kill switch active" and fake.requests == []
    ex.resume()
    assert not ex.kill_active() and ex.handle(req(alert_id=2)).status == "filled"


def test_kill_switch_env(make):
    ex, fake, _ = make(env={"KILL_SWITCH": "true"})
    assert ex.handle(req()).reason == "kill switch active" and fake.requests == []


def test_flatten_all_closes_positions(make):
    held = [{"instrument": "XAU_USD", "long": {"units": "5"}, "short": {"units": "0"}},
            {"instrument": "EUR_USD", "long": {"units": "0"}, "short": {"units": "0"}}]
    ex, fake, _ = make(fake=FakeOanda(positions=held))
    assert ex.flatten_all() == ["XAU_USD"]
    put = [r for r in fake.requests if r.method == "PUT"]
    assert len(put) == 1 and json.loads(put[0].content) == {"longUnits": "ALL"}


def test_consecutive_api_errors_trip_kill_switch(make):
    ex, fake, _ = make(fake=FakeOanda(fail={"/summary"}))
    for i in range(1, 4):
        d = ex.handle(req(alert_id=i))
        assert d.status == "error"
    assert ex.kill_active()
    assert ex.handle(req(alert_id=9)).reason == "kill switch active"


@pytest.mark.parametrize("kw,needle", [
    ({"stop": None}, "no stop-loss"),
    ({"stop": 2660.0}, "wrong side"),                         # long with stop above price
    ({"side": "short", "stop": 2640.0, "target": 2630.0}, "wrong side"),
    ({"target": 2600.0}, "target"),
    ({"t": NOW - pd.Timedelta(minutes=10)}, "old"),
    ({"price": 2700.0, "stop": 2690.0, "target": 2720.0}, "from OANDA mid"),
    ({"symbol": "EURUSD"}, "not in the allowed instruments"),
])
def test_order_validation_blocks(make, kw, needle):
    ex, fake, _ = make()
    d = ex.handle(req(**kw))
    assert d.status == "blocked" and needle in d.reason, d.reason
    assert fake.orders_sent() == []


def test_stop_already_crossed_by_market(make):
    ex, fake, _ = make(fake=FakeOanda(bid=2639.0, ask=2639.5))
    d = ex.handle(req(price=2641.0, stop=2640.0, target=2660.0))
    assert d.status == "blocked" and "already crossed" in d.reason


def test_duplicate_alert_not_resubmitted(make):
    ex, fake, _ = make()
    assert ex.handle(req(alert_id=5)).status == "filled"
    d = ex.handle(req(alert_id=5))
    assert d.reason == "already handled this alert" and len(fake.orders_sent()) == 1
    assert len(ex.store.orders()) == 1


@pytest.mark.parametrize("resp,status", [
    (httpx.Response(201, json={"orderCancelTransaction": {"id": "9", "reason": "MARKET_HALTED"}}), "cancelled"),
    (httpx.Response(400, json={"orderRejectTransaction": {"rejectReason": "INSUFFICIENT_MARGIN"},
                               "errorMessage": "Insufficient margin"}), "error"),
    (httpx.Response(201, json={"orderRejectTransaction": {"rejectReason": "STOP_LOSS_ON_FILL_LOSS"}}), "rejected"),
    (httpx.Response(201, json={"somethingElse": {}}), "error"),
])
def test_order_responses(make, resp, status):
    ex, _, _ = make(fake=FakeOanda(order_response=resp))
    d = ex.handle(req())
    assert d.status == status
    assert TOKEN not in d.reason


def test_missing_credentials(tmp_path):
    cfg = load_execution_config(tmp_path, env={"EXECUTION_ENABLED": "true"}, dotenv={})
    with pytest.raises(Exception, match="OANDA_API_TOKEN"):
        OandaClient(cfg)


# --------------------------------------------------------------------------- #
# Webhook integration
# --------------------------------------------------------------------------- #

def test_webhook_take_places_order_and_caution_does_not(make, tmp_path):
    ex, fake, _ = make()
    settings = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path / "data"),
                                                 "WEBHOOK_SECRET": "s"})
    snaps = tmp_path / "data" / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    write_json(snaps / "ticker_SPY.json", {"generated_at": NOW.isoformat(), "as_of": "2026-11-06",
                                           "regime": {"current": {"bull": 0.8, "bear": 0.1, "high_vol": 0.1}}})
    write_json(snaps / "market.json", {"generated_at": NOW.isoformat(), "events": [], "shocks": [],
                                       "sentiment": {"MARKET": [{"date": "2026-11-06"}]}})
    vix = pd.Series([15.0, 15.2], index=pd.to_datetime(["2026-11-05", "2026-11-06"]))
    app = create_app(settings, AlertStore(tmp_path / "data" / "alerts.sqlite"),
                     Context(settings, vix_loader=lambda: vix), TelegramNotifier(None, None),
                     clock=lambda: NOW, resolve_every_s=None, executor=ex)
    c = TestClient(app)
    body = {"ticker": "XAUUSD", "price": "2650.20", "strategy": "vwap_gold", "side": "long",
            "time": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "stop": "2640.00", "target": "2670.00", "secret": "s"}
    r = c.post("/tv-alert", json=body).json()
    assert r["verdict"] == "TAKE" and len(fake.orders_sent()) == 1
    # a CAUTION alert (VIX now high) is logged but never sent to OANDA
    vix.iloc[-1] = 33.0
    r2 = c.post("/tv-alert", json={**body, "time": (NOW + pd.Timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")}).json()
    assert r2["verdict"] == "CAUTION" and len(fake.orders_sent()) == 1
    o = ex.store.orders()
    assert list(o["status"]) == ["filled", "not_taken"]
