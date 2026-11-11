"""Real-time prices: Finnhub trades websocket + OANDA v20 pricing stream.

    python -m forecaster.live_feed            # run the feeds (start.bat / the launcher runs this)
    python -m forecaster.live_feed --check 30 # connect for 30 s and print ticks per symbol, then exit

Feeds
  Finnhub  wss://ws.finnhub.io (FINNHUB_API_KEY): trades for SPY, QQQ, NVDA, AAPL, MSFT + the
           watchlist. The free tier streams US stocks; TSX symbols (.TO) are subscribed but
           may never trade on it.
  OANDA    v20 pricing stream (OANDA_API_TOKEN + OANDA_ACCOUNT_ID, practice host unless
           OANDA_ENVIRONMENT=live; read-only, no orders): XAU_USD, USD_CAD and the six DXY
           components. "DXY" is the ICE US dollar index formula applied to OANDA mids (a proxy:
           OANDA has no DXY instrument). Prices are mids; OANDA has no traded volume, so bars
           carry tick counts (TradingView's OANDA:XAUUSD volume is tick volume too).

Each feed reconnects with exponential backoff (1 s doubling to 60 s, +-20% jitter; 5 min after
an auth failure; reset once a connection has lasted 60 s). Heartbeat: OANDA sends one every
5 s, so 15 s of silence forces a reconnect; Finnhub has no data heartbeat, so 15 s without a
trade *while US stocks are open* forces one (websocket pings check the socket itself). Any
silence over 10 s while the market is open is logged as a gap (table `gaps`, logs/live.log).

Storage (data/live.sqlite, WAL): ticks go to an in-memory ring buffer per symbol; 1-second and
1-minute OHLC bars (the forming bar too) are upserted once a second; `latest` holds each
symbol's last tick (ms timestamp) and previous close; `feeds` holds connection state. Old rows
are pruned (1 s bars after 24 h, 1 min bars after 30 days).

Readers (`read_*`, `gold_vwap`) are pure and used by the app's Live tab and by the webhook,
which checks TradingView gold alerts against the live price and session VWAP.

Keys: the Finnhub token has to go in the websocket URL (its API has no header option there);
URLs and exception texts are redacted before anything is logged or stored.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import signal
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from forecaster.config import PROJECT_ROOT

log = logging.getLogger("live")
NY = "America/New_York"

FINNHUB_WS = "wss://ws.finnhub.io"
FINNHUB_REST = "https://finnhub.io/api/v1"
OANDA_STREAM = {"practice": "https://stream-fxpractice.oanda.com", "live": "https://stream-fxtrade.oanda.com"}
OANDA_REST = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}

CORE_STOCKS = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT")
GOLD = "XAU_USD"
DXY = "DXY"
# ICE US Dollar Index: 50.14348112 x EURUSD^-0.576 x USDJPY^0.136 x GBPUSD^-0.119 x USDCAD^0.091
#                      x USDSEK^0.042 x USDCHF^0.036
DXY_WEIGHTS = {"EUR_USD": -0.576, "USD_JPY": 0.136, "GBP_USD": -0.119, "USD_CAD": 0.091,
               "USD_SEK": 0.042, "USD_CHF": 0.036}
DXY_CONST = 50.14348112
OANDA_INSTRUMENTS = (GOLD, "USD_CAD", *[k for k in DXY_WEIGHTS if k != "USD_CAD"])
OANDA_DISPLAY = (GOLD, "USD_CAD", DXY)
LABELS = {GOLD: "Gold (XAU/USD)", "USD_CAD": "USD/CAD", DXY: "DXY (proxy)"}

STALE_SECONDS = 10.0          # tile turns red: feed silent this long while its market is open
GAP_SECONDS = 10.0            # silence logged as a gap (market open only)
HEARTBEAT_OANDA = 15.0        # OANDA heartbeats every 5 s
HEARTBEAT_FINNHUB_OPEN = 15.0  # no trade at all on SPY/QQQ/... for 15 s in regular hours = dead feed
HEARTBEAT_FINNHUB_CLOSED = 300.0
BACKOFF_MAX = 60.0
BACKOFF_AUTH = 300.0
STABLE_AFTER = 60.0


def now_ms() -> int:
    from forecaster.clock import now_ms as _now_ms

    return _now_ms()


def _now() -> pd.Timestamp:
    from forecaster.clock import now

    return now()


def redact(text: str, secrets: Iterable[str | None]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def finnhub_symbols(watchlist: Iterable[str]) -> list[str]:
    """Core stocks + watchlist, minus indices/futures/FX (Finnhub trades stocks only)."""
    out = list(CORE_STOCKS)
    for t in watchlist:
        t = t.strip().upper()
        if t and not t.startswith("^") and "=" not in t and t not in out:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Market sessions
# --------------------------------------------------------------------------- #

def session_open(kind: str, now: pd.Timestamp) -> bool:
    """stocks: Mon-Fri 09:30-16:00 NY (holidays not excluded). fx: Sun 17:00 - Fri 17:00 NY.
    gold: fx hours minus the daily 17:00-18:00 NY break."""
    t = now.tz_convert(NY)
    m = t.hour * 60 + t.minute
    dow = t.day_of_week                                   # Mon=0 .. Sun=6
    if kind == "stocks":
        return dow < 5 and 9 * 60 + 30 <= m < 16 * 60
    fx = not (dow == 5 or (dow == 4 and m >= 17 * 60) or (dow == 6 and m < 17 * 60))
    if kind == "gold":
        return fx and not (17 * 60 <= m < 18 * 60)
    return fx


def symbol_kind(symbol: str) -> str:
    return "gold" if symbol == GOLD else ("fx" if symbol in OANDA_INSTRUMENTS or symbol == DXY else "stocks")


def feed_of(symbol: str) -> str:
    return "stocks" if symbol_kind(symbol) == "stocks" else "oanda"


FEED_NAME = {"stocks": "finnhub", "oanda": "oanda"}


# --------------------------------------------------------------------------- #
# Ticks, ring buffer, bars
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    price: float
    ts_ms: int
    size: float = 0.0
    source: str = ""


class RingBuffer:
    """Last `maxlen` ticks per symbol (thread-safe)."""

    def __init__(self, maxlen: int = 5000) -> None:
        self.maxlen = maxlen
        self._d: dict[str, deque[Tick]] = {}
        self._lock = threading.Lock()

    def append(self, t: Tick) -> None:
        with self._lock:
            q = self._d.get(t.symbol)
            if q is None:
                q = self._d[t.symbol] = deque(maxlen=self.maxlen)
            q.append(t)

    def last(self, symbol: str) -> Tick | None:
        with self._lock:
            q = self._d.get(symbol)
            return q[-1] if q else None

    def recent(self, symbol: str, n: int = 100) -> list[Tick]:
        with self._lock:
            q = self._d.get(symbol)
            return list(q)[-n:] if q else []

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._d)

    def __len__(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._d.values())


@dataclass
class Bar:
    o: float
    h: float
    low: float
    c: float
    v: float
    n: int
    last_ts: int


class BarBuilder:
    """OHLC bars of `seconds` length keyed by (symbol, bar start ms). The forming bar is kept
    (and written) too; bars older than `keep` periods are evicted; ticks for evicted bars are
    counted as late and dropped."""

    def __init__(self, seconds: int, keep: int = 3) -> None:
        self.ms = seconds * 1000
        self.keep = keep
        self.bars: dict[tuple[str, int], Bar] = {}
        self.dirty: set[tuple[str, int]] = set()
        self.late = 0
        self._newest = 0

    def add(self, t: Tick) -> None:
        start = t.ts_ms // self.ms * self.ms
        if start < self._newest - self.keep * self.ms:
            self.late += 1
            return
        k = (t.symbol, start)
        b = self.bars.get(k)
        if b is None:
            self.bars[k] = Bar(t.price, t.price, t.price, t.price, t.size, 1, t.ts_ms)
        else:
            b.h, b.low = max(b.h, t.price), min(b.low, t.price)
            if t.ts_ms >= b.last_ts:                       # out-of-order ticks don't move the close
                b.c, b.last_ts = t.price, t.ts_ms
            b.v += t.size
            b.n += 1
        self.dirty.add(k)
        self._newest = max(self._newest, start)

    def take_dirty(self) -> list[tuple[str, int, Bar]]:
        out = [(s, t, self.bars[(s, t)]) for s, t in sorted(self.dirty) if (s, t) in self.bars]
        self.dirty.clear()
        return out

    def evict(self, now: int) -> None:
        cutoff = now // self.ms * self.ms - self.keep * self.ms
        for k in [k for k in self.bars if k[1] < cutoff and k not in self.dirty]:
            del self.bars[k]


def dxy_value(mids: dict[str, float]) -> float | None:
    if not all(mids.get(k, 0) > 0 for k in DXY_WEIGHTS):
        return None
    return DXY_CONST * math.prod(mids[k] ** w for k, w in DXY_WEIGHTS.items())


# --------------------------------------------------------------------------- #
# SQLite store
# --------------------------------------------------------------------------- #

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS bars_1s (symbol TEXT, t INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, n INTEGER,
    PRIMARY KEY (symbol, t)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bars_1m (symbol TEXT, t INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, n INTEGER,
    PRIMARY KEY (symbol, t)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS latest (symbol TEXT PRIMARY KEY, price REAL, ts_ms INTEGER, prev_close REAL,
    source TEXT, updated_ms INTEGER);
CREATE TABLE IF NOT EXISTS feeds (feed TEXT PRIMARY KEY, status TEXT, detail TEXT, connected_since_ms INTEGER,
    last_msg_ms INTEGER, reconnects INTEGER DEFAULT 0, ticks INTEGER DEFAULT 0, updated_ms INTEGER);
CREATE TABLE IF NOT EXISTS gaps (id INTEGER PRIMARY KEY AUTOINCREMENT, feed TEXT, start_ms INTEGER,
    end_ms INTEGER, seconds REAL, reason TEXT);
"""
_UPSERT_BAR = """INSERT INTO {t} (symbol, t, o, h, l, c, v, n) VALUES (?,?,?,?,?,?,?,?)
    ON CONFLICT(symbol, t) DO UPDATE SET o=excluded.o, h=excluded.h, l=excluded.l, c=excluded.c,
    v=excluded.v, n=excluded.n"""


def live_db(cache_dir: Path) -> Path:
    return Path(cache_dir) / "live.sqlite"


def _connect(db: Path, timeout: float = 5.0) -> sqlite3.Connection:
    con = sqlite3.connect(db, timeout=timeout)
    con.row_factory = sqlite3.Row
    return con


class LiveStore:
    def __init__(self, db: Path) -> None:
        self.db = Path(db)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(self.db)) as con, con:
            con.executescript(_SCHEMA)

    def write(self, bars_1s: list, bars_1m: list, latest: list[tuple], feeds: list[dict]) -> None:
        with closing(_connect(self.db)) as con, con:
            for table, rows in (("bars_1s", bars_1s), ("bars_1m", bars_1m)):
                con.executemany(_UPSERT_BAR.format(t=table),
                                [(s, t, b.o, b.h, b.low, b.c, b.v, b.n) for s, t, b in rows])
            con.executemany("""INSERT INTO latest (symbol, price, ts_ms, prev_close, source, updated_ms)
                VALUES (?,?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET price=excluded.price, ts_ms=excluded.ts_ms,
                prev_close=COALESCE(excluded.prev_close, latest.prev_close), source=excluded.source,
                updated_ms=excluded.updated_ms""", latest)
            con.executemany("""INSERT INTO feeds (feed, status, detail, connected_since_ms, last_msg_ms, reconnects,
                ticks, updated_ms) VALUES (:feed,:status,:detail,:connected_since_ms,:last_msg_ms,:reconnects,:ticks,
                :updated_ms) ON CONFLICT(feed) DO UPDATE SET status=excluded.status, detail=excluded.detail,
                connected_since_ms=excluded.connected_since_ms, last_msg_ms=excluded.last_msg_ms,
                reconnects=excluded.reconnects, ticks=excluded.ticks, updated_ms=excluded.updated_ms""", feeds)

    def set_prev_close(self, values: dict[str, float]) -> None:
        with closing(_connect(self.db)) as con, con:
            con.executemany("""INSERT INTO latest (symbol, prev_close) VALUES (?, ?)
                ON CONFLICT(symbol) DO UPDATE SET prev_close=excluded.prev_close""", list(values.items()))

    def log_gap(self, feed: str, start_ms: int, end_ms: int, reason: str) -> None:
        with closing(_connect(self.db)) as con, con:
            con.execute("INSERT INTO gaps (feed, start_ms, end_ms, seconds, reason) VALUES (?,?,?,?,?)",
                        (feed, start_ms, end_ms, (end_ms - start_ms) / 1000, reason[:300]))

    def prune(self, now: int) -> None:
        with closing(_connect(self.db)) as con, con:
            con.execute("DELETE FROM bars_1s WHERE t < ?", (now - 24 * 3600_000,))
            con.execute("DELETE FROM bars_1m WHERE t < ?", (now - 30 * 86400_000,))
            con.execute("DELETE FROM gaps WHERE end_ms < ?", (now - 30 * 86400_000,))


# --------------------------------------------------------------------------- #
# Feed state + hub (sink for both feeds)
# --------------------------------------------------------------------------- #

@dataclass
class FeedState:
    feed: str
    status: str = "starting"          # starting | connected | reconnecting | auth failed | not configured
    detail: str = ""
    connected_since_ms: int | None = None
    last_msg_ms: int | None = None
    reconnects: int = 0
    ticks: int = 0
    gaps: list[tuple[int, int, str]] = field(default_factory=list)   # pending (start, end, reason)
    drop_reason: str | None = None    # why the last connection ended (used for the next gap)

    def row(self, now: int) -> dict:
        return {"feed": self.feed, "status": self.status, "detail": self.detail,
                "connected_since_ms": self.connected_since_ms, "last_msg_ms": self.last_msg_ms,
                "reconnects": self.reconnects, "ticks": self.ticks, "updated_ms": now}


class Hub:
    """Receives ticks from the feeds (all on one event loop) and turns them into bars and rows."""

    def __init__(self, ring: RingBuffer | None = None, clock: Callable[[], int] = now_ms) -> None:
        self.ring = ring or RingBuffer()
        self.b1s, self.b1m = BarBuilder(1, keep=3), BarBuilder(60, keep=2)
        self.feeds: dict[str, FeedState] = {}
        self.fx_mids: dict[str, float] = {}
        self.updated: dict[str, Tick] = {}
        self.clock = clock

    def feed(self, name: str) -> FeedState:
        return self.feeds.setdefault(name, FeedState(name))

    def message(self, name: str, market_open: bool, reason: str = "silence") -> None:
        """Any message (trade, price or heartbeat). Logs a gap if the feed was silent > GAP_SECONDS
        while its market was open."""
        f, now = self.feed(name), self.clock()
        if f.last_msg_ms is not None and market_open and now - f.last_msg_ms > GAP_SECONDS * 1000:
            if f.drop_reason:
                reason = f"disconnected: {f.drop_reason}"
            f.gaps.append((f.last_msg_ms, now, reason))
            log.warning("%s gap: %.1f s without data (%s)", name, (now - f.last_msg_ms) / 1000, reason)
        f.drop_reason = None
        f.last_msg_ms = now

    def add(self, t: Tick) -> None:
        self.ring.append(t)
        self.b1s.add(t)
        self.b1m.add(t)
        self.updated[t.symbol] = t
        if t.source:
            self.feed(t.source).ticks += 1
        if t.symbol in DXY_WEIGHTS:
            self.fx_mids[t.symbol] = t.price
            v = dxy_value(self.fx_mids)
            if v is not None:
                self.add(Tick(DXY, v, t.ts_ms, 0.0, ""))

    def flush(self, store: LiveStore) -> None:
        now = self.clock()
        latest = [(t.symbol, t.price, t.ts_ms, None, t.source or "oanda", now) for t in self.updated.values()]
        self.updated.clear()
        store.write(self.b1s.take_dirty(), self.b1m.take_dirty(), latest, [f.row(now) for f in self.feeds.values()])
        for f in self.feeds.values():
            for start, end, reason in f.gaps:
                store.log_gap(f.feed, start, end, reason)
            f.gaps.clear()
        self.b1s.evict(now)
        self.b1m.evict(now)


# --------------------------------------------------------------------------- #
# Feeds
# --------------------------------------------------------------------------- #

class AuthError(Exception):
    pass


class Backoff:
    def __init__(self, base: float = 1.0, cap: float = BACKOFF_MAX, rng: random.Random | None = None) -> None:
        self.base, self.cap, self.cur = base, cap, base
        self.rng = rng or random.Random()

    def next(self, connected_for: float, auth: bool = False) -> float:
        if connected_for >= STABLE_AFTER:
            self.cur = self.base
        delay = BACKOFF_AUTH if auth else self.cur
        self.cur = min(self.cap, self.cur * 2)
        return delay * (0.8 + 0.4 * self.rng.random())


async def _supervise(name: str, hub: Hub, session_once: Callable[[], object], secrets: tuple,
                     stop: asyncio.Event, sleep=asyncio.sleep, backoff: Backoff | None = None) -> None:
    """Run one connection at a time; reconnect with backoff until `stop` is set."""
    backoff = backoff or Backoff()
    f = hub.feed(name)
    while not stop.is_set():
        t0 = time.monotonic()
        auth = False
        try:
            await session_once()
            reason = "server closed the connection"
        except AuthError as exc:
            auth, reason = True, f"auth failed: {exc}"
        except asyncio.TimeoutError:
            reason = "heartbeat timeout"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        reason = redact(reason, secrets)[:300]
        if stop.is_set():
            break
        f.status, f.detail, f.connected_since_ms = ("auth failed" if auth else "reconnecting"), reason, None
        f.drop_reason = reason
        f.reconnects += 1
        delay = backoff.next(time.monotonic() - t0, auth)
        log.warning("%s disconnected (%s); reconnecting in %.1f s", name, reason, delay)
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


def _connected(hub: Hub, name: str) -> None:
    f = hub.feed(name)
    f.status, f.detail, f.connected_since_ms = "connected", "", hub.clock()
    log.info("%s connected", name)


async def finnhub_session(hub: Hub, key: str, symbols: list[str], url: str = FINNHUB_WS,
                          connect=None, clock: Callable[[], pd.Timestamp] | None = None) -> None:
    """One websocket session: subscribe, then read trades until it fails."""
    from websockets.asyncio.client import connect as ws_connect
    from websockets.exceptions import InvalidStatus

    connect = connect or ws_connect
    clock = clock or _now
    try:
        ws_cm = connect(f"{url}?token={key}", ping_interval=20, ping_timeout=20, open_timeout=15,
                        max_size=2**22)
        async with ws_cm as ws:
            _connected(hub, "finnhub")
            for s in symbols:
                await ws.send(json.dumps({"type": "subscribe", "symbol": s}))
            while True:
                is_open = session_open("stocks", clock())
                timeout = HEARTBEAT_FINNHUB_OPEN if is_open else HEARTBEAT_FINNHUB_CLOSED
                raw = await asyncio.wait_for(ws.recv(), timeout)
                hub.message("finnhub", session_open("stocks", clock()))
                msg = json.loads(raw)
                typ = msg.get("type")
                if typ == "trade":
                    for d in msg.get("data") or []:
                        try:
                            hub.add(Tick(str(d["s"]), float(d["p"]), int(d["t"]), float(d.get("v") or 0), "finnhub"))
                        except (KeyError, TypeError, ValueError):
                            continue
                elif typ == "error":
                    text = str(msg.get("msg", "error"))
                    if "token" in text.lower() or "auth" in text.lower():
                        raise AuthError(text)
                    raise ConnectionError(text)
    except InvalidStatus as exc:
        code = getattr(getattr(exc, "response", None), "status_code", 0)
        if code in (401, 403):
            raise AuthError(f"HTTP {code}: check FINNHUB_API_KEY") from None
        raise ConnectionError(f"HTTP {code}") from None


def _oanda_ms(ts: str) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


async def oanda_session(hub: Hub, token: str, account: str, instruments: Iterable[str], host: str,
                        client=None, clock: Callable[[], pd.Timestamp] | None = None) -> None:
    import httpx

    clock = clock or _now
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=None))
    try:
        url = f"{host}/v3/accounts/{account}/pricing/stream"
        async with client.stream("GET", url, params={"instruments": ",".join(instruments)},
                                 headers={"Authorization": f"Bearer {token}"}) as r:
            if r.status_code in (401, 403):
                raise AuthError(f"HTTP {r.status_code}: check OANDA_API_TOKEN / OANDA_ACCOUNT_ID")
            if r.status_code != 200:
                raise ConnectionError(f"HTTP {r.status_code}")
            _connected(hub, "oanda")
            lines = r.aiter_lines()
            while True:
                line = await asyncio.wait_for(lines.__anext__(), HEARTBEAT_OANDA)
                if not line.strip():
                    continue
                hub.message("oanda", session_open("fx", clock()))
                msg = json.loads(line)
                if msg.get("type") == "PRICE" and msg.get("bids") and msg.get("asks"):
                    mid = (float(msg["bids"][0]["price"]) + float(msg["asks"][0]["price"])) / 2
                    hub.add(Tick(msg["instrument"], mid, _oanda_ms(msg["time"]), 0.0, "oanda"))
    except StopAsyncIteration:
        return
    finally:
        if own:
            await client.aclose()


# --------------------------------------------------------------------------- #
# Previous closes (for % change on the day)
# --------------------------------------------------------------------------- #

def finnhub_prev_close(key: str, symbols: list[str], cache, session=None) -> dict[str, float]:  # pragma: no cover
    import httpx

    from forecaster.cache import Cache

    day = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
    out = {}
    with (session or httpx.Client(timeout=10)) as c:
        for s in symbols:
            def fetch(s=s):
                r = c.get(f"{FINNHUB_REST}/quote", params={"symbol": s}, headers={"X-Finnhub-Token": key})
                r.raise_for_status()
                return r.json()
            try:
                q = cache.get_or_fetch(Cache.make_key("live_quote", symbol=s, day=day), fetch)
                if q.get("pc"):
                    out[s] = float(q["pc"])
            except Exception as exc:
                log.info("prev close %s unavailable: %s", s, redact(str(exc), (key,)))
    return out


def oanda_prev_close(token: str, host: str, instruments: Iterable[str], cache) -> dict[str, float]:  # pragma: no cover
    """Close of the last complete daily candle (OANDA trading day ends 17:00 New York)."""
    import httpx

    from forecaster.cache import Cache

    day = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d-%H")
    out = {}
    with httpx.Client(timeout=10, headers={"Authorization": f"Bearer {token}"}) as c:
        for inst in instruments:
            def fetch(inst=inst):
                r = c.get(f"{host}/v3/instruments/{inst}/candles",
                          params={"count": 3, "granularity": "D", "price": "M", "dailyAlignment": 17,
                                  "alignmentTimezone": NY})
                r.raise_for_status()
                return r.json()
            try:
                js = cache.get_or_fetch(Cache.make_key("live_candles", inst=inst, day=day), fetch)
                done = [cd for cd in js.get("candles", []) if cd.get("complete")]
                if done:
                    out[inst] = float(done[-1]["mid"]["c"])
            except Exception as exc:
                log.info("prev close %s unavailable: %s", inst, redact(str(exc), (token,)))
    v = dxy_value(out)
    if v is not None:
        out[DXY] = v
    return out


# --------------------------------------------------------------------------- #
# Readers (app + webhook)
# --------------------------------------------------------------------------- #

def _rows(db: Path, sql: str, params=()) -> list[dict]:
    if not Path(db).exists():
        return []
    try:
        with closing(_connect(db, timeout=2.0)) as con:
            return [dict(r) for r in con.execute(sql, params)]
    except sqlite3.Error:
        return []


def read_latest(db: Path) -> dict[str, dict]:
    return {r["symbol"]: r for r in _rows(db, "SELECT * FROM latest")}


def read_feeds(db: Path) -> dict[str, dict]:
    return {r["feed"]: r for r in _rows(db, "SELECT * FROM feeds")}


def read_gaps(db: Path, since_ms: int) -> list[dict]:
    return _rows(db, "SELECT * FROM gaps WHERE end_ms >= ? ORDER BY end_ms DESC", (since_ms,))


def read_bars(db: Path, symbol: str, since_ms: int, table: str = "bars_1m") -> pd.DataFrame:
    assert table in ("bars_1s", "bars_1m")
    rows = _rows(db, f"SELECT t, o, h, l, c, v, n FROM {table} WHERE symbol=? AND t>=? ORDER BY t", (symbol, since_ms))
    cols = ["Open", "High", "Low", "Close", "Volume", "Ticks"]
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    df.columns = cols
    return df


def session_start(now: pd.Timestamp) -> pd.Timestamp:
    """Start of the gold session containing `now` (sessions reset 18:00 New York)."""
    t = now.tz_convert(NY)
    start = t.normalize() + pd.Timedelta(hours=18)
    if t < start:
        start -= pd.Timedelta(days=1)
    return start.tz_convert("UTC")


def gold_vwap(db: Path, now: pd.Timestamp, min_bars: int = 3) -> dict | None:
    """Live gold vs its session VWAP (tick-volume weighted 1-minute bars, 18:00 NY reset,
    same formula as monitor.session_vwap / the Pine strategy)."""
    from forecaster.monitor import session_vwap

    last = read_latest(db).get(GOLD)
    if not last or last.get("price") is None:
        return None
    bars = read_bars(db, GOLD, int(session_start(now).value // 1_000_000))
    out = {"price": float(last["price"]), "ts_ms": int(last["ts_ms"]),
           "age_s": (int(now.value // 1_000_000) - int(last["ts_ms"])) / 1000,
           "prev_close": last.get("prev_close"), "bars": len(bars), "vwap": None, "sd": None, "z": None}
    if len(bars) >= min_bars:
        vb = session_vwap(bars.assign(Volume=bars["Ticks"].astype(float)))
        v, sd = float(vb["vwap"].iloc[-1]), float(vb["sd"].iloc[-1])
        out.update(vwap=v, sd=sd, z=(out["price"] - v) / sd if sd > 0 else None)
    return out


def vwap_frame(db: Path, now: pd.Timestamp, hours: float = 6.0) -> pd.DataFrame:
    """Session 1-minute gold bars with vwap / sd columns, limited to the last `hours`."""
    from forecaster.monitor import session_vwap

    bars = read_bars(db, GOLD, int(session_start(now).value // 1_000_000))
    if bars.empty:
        return bars
    vb = session_vwap(bars.assign(Volume=bars["Ticks"].astype(float)))
    return vb[vb.index >= now - pd.Timedelta(hours=hours)]


@dataclass
class TileState:
    symbol: str
    label: str
    price: float | None
    change: float | None
    ts_ms: int | None
    state: str            # ok | stale | closed | waiting | off
    note: str


def tiles(db: Path, now: pd.Timestamp, stock_symbols: list[str]) -> list[TileState]:
    latest, feeds = read_latest(db), read_feeds(db)
    t_ms = int(now.value // 1_000_000)
    out = []
    for sym in [*stock_symbols, *OANDA_DISPLAY]:
        feed = FEED_NAME[feed_of(sym)]
        f = feeds.get(feed) or {}
        row = latest.get(sym) or {}
        price, ts, pc = row.get("price"), row.get("ts_ms"), row.get("prev_close")
        change = (price / pc - 1) if price and pc else None
        open_ = session_open(symbol_kind(sym), now)
        silent = (t_ms - f["last_msg_ms"]) / 1000 if f.get("last_msg_ms") else None
        if not f or f.get("status") in ("not configured",):
            state, note = "off", f.get("detail") or f"{feed} not running"
        elif not open_:
            state, note = "closed", "market closed"
        elif f.get("status") != "connected" or silent is None or silent > STALE_SECONDS:
            state = "stale"
            note = (f"feed silent {silent:.0f} s" if silent is not None else "no data yet") + (
                f" · {f['status']}" if f.get("status") != "connected" else "")
        elif price is None:
            state, note = "waiting", "no trades yet"
        else:
            state, note = "ok", "live"
        out.append(TileState(sym, LABELS.get(sym, sym), price, change, ts, state, note))
    return out


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

@dataclass
class LiveConfig:
    finnhub_key: str | None
    oanda_token: str | None
    oanda_account: str | None
    oanda_env: str
    stocks: list[str]

    @classmethod
    def load(cls, root: Path = PROJECT_ROOT, env: dict | None = None) -> LiveConfig:
        from dotenv import dotenv_values

        from forecaster.config import load_watchlist

        if env is None:
            env = {**{k: v for k, v in dotenv_values(root / ".env").items() if v}, **os.environ} \
                if (root / ".env").exists() else dict(os.environ)
        oenv = (env.get("OANDA_ENVIRONMENT") or "practice").strip().lower()
        return cls(env.get("FINNHUB_API_KEY") or None, env.get("OANDA_API_TOKEN") or None,
                   env.get("OANDA_ACCOUNT_ID") or None, oenv if oenv in OANDA_STREAM else "practice",
                   finnhub_symbols(load_watchlist(root / "watchlist.yaml")))

    @property
    def secrets(self) -> tuple:
        return (self.finnhub_key, self.oanda_token, self.oanda_account)


async def run_service(cfg: LiveConfig, db: Path, stop: asyncio.Event, hub: Hub | None = None,
                      finnhub_url: str = FINNHUB_WS, oanda_host: str | None = None,
                      oanda_client_factory=None, prev_close_loader=None, flush_every: float = 1.0) -> Hub:
    """Run both feeds + the 1-second flush until `stop` is set."""
    hub = hub or Hub()
    store = LiveStore(db)
    tasks = []
    if cfg.finnhub_key:
        tasks.append(asyncio.create_task(_supervise(
            "finnhub", hub, lambda: finnhub_session(hub, cfg.finnhub_key, cfg.stocks, finnhub_url),
            cfg.secrets, stop)))
    else:
        f = hub.feed("finnhub")
        f.status, f.detail = "not configured", "FINNHUB_API_KEY missing from .env"
    if cfg.oanda_token and cfg.oanda_account:
        host = oanda_host or OANDA_STREAM[cfg.oanda_env]
        tasks.append(asyncio.create_task(_supervise(
            "oanda", hub, lambda: oanda_session(hub, cfg.oanda_token, cfg.oanda_account, OANDA_INSTRUMENTS, host,
                                                client=oanda_client_factory() if oanda_client_factory else None),
            cfg.secrets, stop)))
    else:
        f = hub.feed("oanda")
        f.status, f.detail = "not configured", "OANDA_API_TOKEN / OANDA_ACCOUNT_ID missing from .env"
    if prev_close_loader is not None:
        async def prev_loop():
            day = None
            while not stop.is_set():
                d = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d-%H")
                if d != day:
                    try:
                        vals = await asyncio.to_thread(prev_close_loader)
                        store.set_prev_close(vals)
                        day = d
                    except Exception as exc:
                        log.info("prev close refresh failed: %s", redact(str(exc), cfg.secrets))
                try:
                    await asyncio.wait_for(stop.wait(), 300)
                except asyncio.TimeoutError:
                    pass
        tasks.append(asyncio.create_task(prev_loop()))
    last_prune = 0.0
    try:
        while not stop.is_set():
            try:
                hub.flush(store)
                if time.monotonic() - last_prune > 3600:
                    store.prune(hub.clock())
                    last_prune = time.monotonic()
            except sqlite3.Error as exc:
                log.warning("flush failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), flush_every)
            except asyncio.TimeoutError:
                pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for f in hub.feeds.values():
            if f.status == "connected":
                f.status, f.connected_since_ms = "stopped", None
        hub.flush(store)
    return hub


def _prev_close_loader(cfg: LiveConfig, cache_dir: Path):  # pragma: no cover - network
    from forecaster.cache import Cache

    cache = Cache(Path(cache_dir) / "live_http", ttl_seconds=6 * 3600)

    def load() -> dict[str, float]:
        out = {}
        if cfg.finnhub_key:
            out.update(finnhub_prev_close(cfg.finnhub_key, cfg.stocks, cache))
        if cfg.oanda_token:
            out.update(oanda_prev_close(cfg.oanda_token, OANDA_REST[cfg.oanda_env], OANDA_INSTRUMENTS, cache))
        return out
    return load


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - long-running
    import argparse

    from forecaster.config import load_settings
    from forecaster.monitor import setup_logging

    ap = argparse.ArgumentParser(description="Live Finnhub + OANDA price feeds")
    ap.add_argument("--check", type=float, metavar="SECONDS", help="connect, print ticks per symbol, exit")
    args = ap.parse_args(argv)
    settings = load_settings()
    setup_logging(PROJECT_ROOT / "logs", "live")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    cfg = LiveConfig.load()
    db = live_db(settings.cache_dir)
    if not cfg.finnhub_key and not (cfg.oanda_token and cfg.oanda_account):
        log.warning("no live feed configured (FINNHUB_API_KEY / OANDA_API_TOKEN + OANDA_ACCOUNT_ID); idling")

    async def runner() -> Hub:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):     # Windows: Ctrl+C / taskkill end the process
                pass
        if args.check:
            loop.call_later(args.check, stop.set)
        # test hooks: point the feeds at local simulators (defaults are the real hosts)
        from forecaster.clock import is_demo

        loader = None if is_demo() else _prev_close_loader(cfg, settings.cache_dir)
        return await run_service(cfg, db, stop, prev_close_loader=loader,
                                 finnhub_url=os.environ.get("FORECASTER_LIVE_FINNHUB_URL") or FINNHUB_WS,
                                 oanda_host=os.environ.get("FORECASTER_LIVE_OANDA_URL") or None)

    hub = asyncio.run(runner())
    if args.check:
        print(f"{'symbol':<10} {'ticks':>7} {'last price':>12}  last tick (ET)")
        for s in sorted(hub.ring.symbols()):
            q = hub.ring.recent(s, hub.ring.maxlen)
            t = pd.Timestamp(q[-1].ts_ms, unit="ms", tz="UTC").tz_convert(NY)
            print(f"{s:<10} {len(q):>7} {q[-1].price:>12,.4f}  {t:%H:%M:%S}.{t.microsecond // 1000:03d}")
        for f in hub.feeds.values():
            print(f"feed {f.feed}: {f.status} {f.detail} · {f.ticks} ticks · {f.reconnects} reconnects")
        return 0 if len(hub.ring) else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
