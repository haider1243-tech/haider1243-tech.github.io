"""live_feed: ring buffer, bars, SQLite, readers, and both feeds against local fake servers."""
import asyncio
import json

import httpx
import pandas as pd
import pytest

from forecaster import live_feed as L

T0 = 1_762_786_800_000            # 2025-11-10 15:00:00 UTC (10:00 ET Monday): stocks + gold open
NOW = pd.Timestamp(T0, unit="ms", tz="UTC")
KEY = "fh-secret-KEY-123"


def tick(sym, p, ms, v=1.0, src="finnhub"):
    return L.Tick(sym, p, ms, v, src)


def test_ring_buffer_bounds():
    rb = L.RingBuffer(maxlen=3)
    for i in range(5):
        rb.append(tick("SPY", 100 + i, T0 + i))
    assert len(rb) == 3 and rb.last("SPY").price == 104 and [t.price for t in rb.recent("SPY", 2)] == [103, 104]
    assert rb.last("QQQ") is None


def test_bar_builder_ohlc_out_of_order_and_late():
    b = L.BarBuilder(60, keep=1)
    for p, dt in ((100, 1_000), (103, 5_000), (99, 9_000), (101, 8_000)):     # 8 s tick arrives after 9 s
        b.add(tick("SPY", p, T0 + dt, v=2))
    (sym, start, bar), = b.take_dirty()
    assert (sym, start) == ("SPY", T0) and (bar.o, bar.h, bar.low, bar.c, bar.v, bar.n) == (100, 103, 99, 99, 8, 4)
    b.add(tick("SPY", 105, T0 + 3 * 60_000))                                  # two minutes later
    b.add(tick("SPY", 98, T0 + 2_000))                                        # older than keep -> late
    assert b.late == 1
    b.take_dirty()
    b.evict(T0 + 3 * 60_000)
    assert ("SPY", T0) not in b.bars and ("SPY", T0 + 180_000) in b.bars


def test_dxy_proxy_formula():
    mids = {"EUR_USD": 1.08, "USD_JPY": 150.0, "GBP_USD": 1.27, "USD_CAD": 1.38, "USD_SEK": 10.5, "USD_CHF": 0.88}
    v = L.dxy_value(mids)
    assert 100 < v < 110                                                       # ~104 for these rates
    assert L.dxy_value({**mids, "USD_SEK": 0}) is None
    hub = L.Hub(clock=lambda: T0)
    for k, p in mids.items():
        hub.add(tick(k, p, T0, 0, "oanda"))
    assert hub.ring.last("DXY").price == pytest.approx(v)


def test_sessions():
    ny = lambda s: pd.Timestamp(s, tz="America/New_York")  # noqa: E731
    assert L.session_open("stocks", ny("2026-11-10 10:00")) and not L.session_open("stocks", ny("2026-11-10 17:30"))
    assert not L.session_open("stocks", ny("2026-11-14 11:00"))                          # Saturday
    assert L.session_open("fx", ny("2026-11-10 17:30")) and not L.session_open("gold", ny("2026-11-10 17:30"))
    assert L.session_open("gold", ny("2026-11-10 18:05"))
    assert not L.session_open("fx", ny("2026-11-13 17:05")) and not L.session_open("fx", ny("2026-11-15 16:00"))
    assert L.session_open("fx", ny("2026-11-15 17:01"))                                # Sunday reopen
    assert L.session_start(ny("2026-11-10 17:30").tz_convert("UTC")) == ny("2026-11-09 18:00").tz_convert("UTC")
    assert L.session_start(ny("2026-11-10 18:30").tz_convert("UTC")) == ny("2026-11-10 18:00").tz_convert("UTC")


def test_finnhub_symbols_from_watchlist():
    assert L.finnhub_symbols(["XIU.TO", "^VIX", "GC=F", "spy", "AMD"]) == ["SPY", "QQQ", "NVDA", "AAPL", "MSFT",
                                                                         "XIU.TO", "AMD"]


def test_hub_gap_logging_and_flush(tmp_path):
    clock = [T0]
    hub = L.Hub(clock=lambda: clock[0])
    store = L.LiveStore(tmp_path / "live.sqlite")
    hub.message("finnhub", market_open=True)
    hub.add(tick("SPY", 600.0, T0))
    clock[0] = T0 + 25_000
    hub.message("finnhub", market_open=False)                  # quiet while closed: not a gap
    clock[0] = T0 + 60_000
    hub.feed("finnhub").drop_reason = "heartbeat timeout"
    hub.message("finnhub", market_open=True)                   # 35 s silence while open: gap
    hub.add(tick("SPY", 601.0, T0 + 60_000))
    hub.flush(store)
    gaps = L.read_gaps(tmp_path / "live.sqlite", 0)
    assert len(gaps) == 1 and gaps[0]["seconds"] == 35 and gaps[0]["reason"] == "disconnected: heartbeat timeout"
    latest = L.read_latest(tmp_path / "live.sqlite")
    assert latest["SPY"]["price"] == 601.0 and latest["SPY"]["ts_ms"] == T0 + 60_000
    bars = L.read_bars(tmp_path / "live.sqlite", "SPY", 0)
    assert list(bars["Close"]) == [600.0, 601.0] and bars.index[0] == NOW
    assert len(L.read_bars(tmp_path / "live.sqlite", "SPY", 0, "bars_1s")) == 2
    store.set_prev_close({"SPY": 590.0})
    hub.add(tick("SPY", 602.0, T0 + 61_000))
    hub.flush(store)                                            # a new tick keeps the prev close
    assert L.read_latest(tmp_path / "live.sqlite")["SPY"]["prev_close"] == 590.0
    store.prune(T0 + 2 * 86400_000)
    assert L.read_bars(tmp_path / "live.sqlite", "SPY", 0, "bars_1s").empty


def _gold_db(tmp_path, prices, start=T0, age_ms=300):
    hub = L.Hub(clock=lambda: start + len(prices) * 60_000)
    store = L.LiveStore(tmp_path / "live.sqlite")
    for i, p in enumerate(prices):
        for j in range(3):                                      # 3 ticks per minute
            hub.add(tick(L.GOLD, p + j * 0.1, start + i * 60_000 + j * 10_000, 0, "oanda"))
    hub.feed("oanda").last_msg_ms = start + len(prices) * 60_000 - age_ms
    hub.feed("oanda").status = "connected"
    hub.flush(store)
    return tmp_path / "live.sqlite"


def test_gold_vwap_matches_monitor_formula(tmp_path):
    from forecaster.monitor import session_vwap
    prices = [2640 + 0.5 * i for i in range(30)]
    db = _gold_db(tmp_path, prices)
    now = pd.Timestamp(T0 + 30 * 60_000, unit="ms", tz="UTC")
    g = L.gold_vwap(db, now)
    bars = L.read_bars(db, L.GOLD, 0)
    ref = session_vwap(bars.assign(Volume=bars["Ticks"].astype(float))).iloc[-1]
    assert g["vwap"] == pytest.approx(ref["vwap"]) and g["sd"] == pytest.approx(ref["sd"])
    assert g["price"] == pytest.approx(prices[-1] + 0.2) and g["z"] > 0 and g["bars"] == 30
    assert g["age_s"] == pytest.approx((now.value // 1_000_000 - (T0 + 29 * 60_000 + 20_000)) / 1000)
    assert L.gold_vwap(tmp_path / "none.sqlite", now) is None
    early = L.gold_vwap(_gold_db(tmp_path / "e", [2640.0]), now)
    assert early["vwap"] is None and early["z"] is None                      # VWAP needs 3 bars


def test_tiles_states(tmp_path):
    db = _gold_db(tmp_path, [2640.0, 2641.0, 2642.0])
    now = pd.Timestamp(T0 + 3 * 60_000, unit="ms", tz="UTC")
    t = {x.symbol: x for x in L.tiles(db, now, ["SPY"])}
    assert t[L.GOLD].state == "ok" and t["SPY"].state == "off"               # finnhub never ran
    assert "USD_CAD" in t and t["USD_CAD"].state == "waiting"                # connected, no tick yet
    later = now + pd.Timedelta(seconds=12)
    assert {x.symbol: x for x in L.tiles(db, later, ["SPY"])}[L.GOLD].state == "stale"   # feed silent 12 s
    closed = pd.Timestamp("2026-11-10 17:30", tz="America/New_York").tz_convert("UTC")
    assert {x.symbol: x for x in L.tiles(db, closed, ["SPY"])}[L.GOLD].state == "closed"
    L.LiveStore(db).set_prev_close({L.GOLD: 2600.0})
    assert {x.symbol: x for x in L.tiles(db, now, [])}[L.GOLD].change == pytest.approx(2642.2 / 2600 - 1)


def test_backoff_doubles_caps_and_resets():
    import random
    b = L.Backoff(rng=random.Random(0))
    raw = [b.next(1.0) for _ in range(8)]
    assert raw[0] == pytest.approx(1, rel=0.2) and raw[3] == pytest.approx(8, rel=0.2)
    assert max(raw) <= L.BACKOFF_MAX * 1.2
    assert b.next(120.0) == pytest.approx(1, rel=0.2)                        # stable connection resets it
    assert b.next(1.0, auth=True) >= L.BACKOFF_AUTH * 0.8


def test_redact():
    assert L.redact(f"failed wss://ws.finnhub.io?token={KEY}", (KEY, None)) == "failed wss://ws.finnhub.io?token=***"


# --------------------------------------------------------------------------- #
# Finnhub against a local websocket server
# --------------------------------------------------------------------------- #

def test_finnhub_stream_subscribes_parses_and_reconnects(tmp_path, monkeypatch, caplog):
    from websockets.asyncio.server import serve
    monkeypatch.setattr(L, "HEARTBEAT_FINNHUB_OPEN", 0.5)
    monkeypatch.setattr(L, "HEARTBEAT_FINNHUB_CLOSED", 0.5)
    monkeypatch.setattr(L.Backoff, "next", lambda self, c, auth=False: 0.05)
    seen = {"subs": [], "paths": [], "conns": 0}

    async def handler(ws):
        seen["conns"] += 1
        seen["paths"].append(ws.request.path)
        n = seen["conns"]
        async for raw in ws:
            seen["subs"].append(json.loads(raw)["symbol"])
            if len(seen["subs"]) % 3 == 0:
                break
        await ws.send(json.dumps({"type": "ping"}))
        await ws.send(json.dumps({"type": "trade", "data": [
            {"s": "SPY", "p": 600.0 + n, "t": L.now_ms(), "v": 10},
            {"s": "AAPL", "p": 230.5, "t": L.now_ms(), "v": 5}, {"s": "BAD"}]}))
        if n == 1:
            await ws.close()                                   # server drops -> reconnect
        else:
            await asyncio.sleep(5)                             # silence -> heartbeat timeout

    async def main():
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            cfg = L.LiveConfig(KEY, None, None, "practice", ["SPY", "QQQ", "AAPL"])
            stop = asyncio.Event()
            asyncio.get_running_loop().call_later(1.6, stop.set)
            return await L.run_service(cfg, tmp_path / "live.sqlite", stop, finnhub_url=f"ws://127.0.0.1:{port}",
                                       flush_every=0.1)

    hub = asyncio.run(main())
    assert seen["subs"][:3] == ["SPY", "QQQ", "AAPL"] and seen["conns"] >= 3
    assert seen["paths"][0] == f"/?token={KEY}"
    f = hub.feeds["finnhub"]
    assert f.reconnects >= 2 and f.ticks >= 4
    latest = L.read_latest(tmp_path / "live.sqlite")
    assert latest["SPY"]["price"] >= 601 and latest["AAPL"]["price"] == 230.5
    feeds = L.read_feeds(tmp_path / "live.sqlite")
    assert feeds["finnhub"]["status"] == "stopped" and feeds["oanda"]["status"] == "not configured"
    assert KEY not in caplog.text and KEY not in json.dumps(feeds)


def test_finnhub_auth_failure_backs_off_long(tmp_path, monkeypatch):
    from websockets.asyncio.server import serve
    from websockets.http11 import Response
    from websockets.datastructures import Headers
    delays = []
    monkeypatch.setattr(L.Backoff, "next", lambda self, c, auth=False: (delays.append(auth), 0.05)[1])

    def reject(conn, request):
        return Response(401, "Unauthorized", Headers(), b"")

    async def main():
        async with serve(lambda ws: None, "127.0.0.1", 0, process_request=reject) as server:
            port = server.sockets[0].getsockname()[1]
            hub = L.Hub()
            stop = asyncio.Event()
            asyncio.get_running_loop().call_later(0.3, stop.set)
            await L._supervise("finnhub", hub, lambda: L.finnhub_session(hub, KEY, ["SPY"], f"ws://127.0.0.1:{port}"),
                               (KEY,), stop)
            return hub

    hub = asyncio.run(main())
    f = hub.feeds["finnhub"]
    assert f.status == "auth failed" and "check FINNHUB_API_KEY" in f.detail and KEY not in f.detail
    assert delays and all(delays)


# --------------------------------------------------------------------------- #
# OANDA against a streaming mock transport
# --------------------------------------------------------------------------- #

def _price(inst, bid, ask, t="2025-11-10T15:00:00.123456789Z"):
    return json.dumps({"type": "PRICE", "instrument": inst, "time": t,
                       "bids": [{"price": str(bid), "liquidity": 1}], "asks": [{"price": str(ask), "liquidity": 1}]})


def test_oanda_stream_parses_heartbeat_timeout_and_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "HEARTBEAT_OANDA", 0.3)
    calls = []

    async def stream():
        yield (_price("XAU_USD", 2650.0, 2650.6) + "\n").encode()
        yield (json.dumps({"type": "HEARTBEAT", "time": "2025-11-10T15:00:05Z"}) + "\n").encode()
        yield (_price("USD_CAD", 1.3800, 1.3802) + "\n").encode()
        await asyncio.sleep(2)                                 # stalls -> heartbeat timeout
        yield b""

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=stream())

    hub = L.Hub()

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await L.oanda_session(hub, "tok-SECRET", "101-001-1", L.OANDA_INSTRUMENTS, "https://stream.test", client=client)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())
    r = calls[0]
    assert r.headers["authorization"] == "Bearer tok-SECRET" and "tok-SECRET" not in str(r.url)
    assert r.url.path == "/v3/accounts/101-001-1/pricing/stream" and "XAU_USD" in r.url.params["instruments"]
    g = hub.ring.last("XAU_USD")
    assert g.price == pytest.approx(2650.3) and g.ts_ms == T0 + 123                      # mid, ms precision
    assert hub.ring.last("USD_CAD").price == pytest.approx(1.3801) and hub.feeds["oanda"].status == "connected"

    async def denied():
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
        await L.oanda_session(L.Hub(), "t", "a", ["XAU_USD"], "https://stream.test", client=client)

    with pytest.raises(L.AuthError):
        asyncio.run(denied())


def test_live_config_reads_env_file(tmp_path):
    (tmp_path / ".env").write_text("FINNHUB_API_KEY=k\nOANDA_API_TOKEN=t\nOANDA_ACCOUNT_ID=a\nOANDA_ENVIRONMENT=bogus\n")
    (tmp_path / "watchlist.yaml").write_text("tickers: [XIU.TO]\n")
    cfg = L.LiveConfig.load(tmp_path, env={"FINNHUB_API_KEY": "k", "OANDA_API_TOKEN": "t", "OANDA_ACCOUNT_ID": "a",
                                            "OANDA_ENVIRONMENT": "bogus"})
    assert cfg.oanda_env == "practice" and cfg.stocks[-1] == "XIU.TO" and cfg.secrets == ("k", "t", "a")
