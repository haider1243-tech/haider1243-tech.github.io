"""Simulated live market for demo mode (`start.bat demo` / `python -m forecaster.launcher --demo`).

    python -m forecaster.simfeed          # the launcher runs this in demo mode

Serves the same protocols the real feeds use, on loopback only, so live_feed.py runs unchanged:
  * Finnhub-style trades websocket on ws://127.0.0.1:18765 (SPY, QQQ, NVDA, AAPL, MSFT, ...),
  * OANDA-style v20 pricing stream on http://127.0.0.1:18766 (XAU_USD, USD_CAD, DXY components).

Prices are random walks with realistic daily volatility and a U-shaped intraday activity
profile, continuing from the seeded history's last closes. Tick times use forecaster.clock, which
demo mode shifts so the market is open whenever you start it. Nothing here touches a real market.

`seed(cache_dir, market_now)` prepares data/demo: 10 days of 1-minute history plus this morning
(so indicators are warm), previous closes for "% today", and a Strategy Lab run on synthetic
data (so the Signal Board has stats to show).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecaster import intraday as I
from forecaster.clock import now as market_now
from forecaster.clock import now_ms as market_now_ms

log = logging.getLogger("simfeed")
FINNHUB_PORT, OANDA_PORT = 18765, 18766
DAILY_VOL = {"SPY": 0.011, "QQQ": 0.014, "NVDA": 0.026, "AAPL": 0.015, "MSFT": 0.013, "TSLA": 0.035, "XAU_USD": 0.010,
             "USD_CAD": 0.004, "EUR_USD": 0.005, "USD_JPY": 0.006, "GBP_USD": 0.005, "USD_SEK": 0.006,
             "USD_CHF": 0.005}
START = {"SPY": 600.0, "QQQ": 520.0, "NVDA": 190.0, "AAPL": 230.0, "MSFT": 510.0, "TSLA": 420.0, "XAU_USD": 2650.0,
         "USD_CAD": 1.38, "EUR_USD": 1.08, "USD_JPY": 150.0, "GBP_USD": 1.27, "USD_SEK": 10.5, "USD_CHF": 0.88}
TICKS_PER_SEC = 4
SIM_SYMBOLS = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "XAU_USD", "USD_CAD")
HISTORY_DAYS = 730          # demo history for the Simulator tab (2 years of 1-minute bars)


def demo_market_start(real_now: pd.Timestamp | None = None) -> pd.Timestamp:
    """09:44 New York on the latest weekday (today if a weekday): where the demo clock starts."""
    t = (real_now or pd.Timestamp.now(tz="UTC")).tz_convert(I.NY).normalize()
    while t.dayofweek >= 5:
        t -= pd.Timedelta(days=1)
    return (t + pd.Timedelta(hours=9, minutes=44)).tz_convert("UTC")


def _activity(ny_minutes: float) -> float:
    u = (min(max(ny_minutes, 570), 959) - 570) / 390 * 2 - 1
    return 0.6 + 1.2 * u * u


class Market:
    def __init__(self, prices: dict[str, float], seed: int | None = None) -> None:
        self.p = dict(prices)
        self.rng = random.Random(seed)

    def step(self, sym: str, dt_s: float) -> float:
        t = market_now().tz_convert(I.NY)
        act = _activity(t.hour * 60 + t.minute) if sym in ("SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA") else 1.0
        sigma = DAILY_VOL.get(sym, 0.01) * act / math.sqrt(390 * 60)
        self.p[sym] *= math.exp(sigma * math.sqrt(dt_s) * self.rng.gauss(0, 1))
        return self.p[sym]


# --------------------------------------------------------------------------- #
# Servers
# --------------------------------------------------------------------------- #

async def _finnhub_handler(ws, market: Market) -> None:
    subs: set[str] = set()

    async def reader():
        async for raw in ws:
            m = json.loads(raw)
            if m.get("type") == "subscribe":
                subs.add(m["symbol"])

    task = asyncio.create_task(reader())
    last_ping = time.monotonic()
    try:
        while True:
            data = []
            for s in list(subs):
                if s in START and market.rng.random() < 0.8:
                    data.append({"s": s, "p": round(market.step(s, 1 / TICKS_PER_SEC), 2), "t": market_now_ms(),
                                 "v": market.rng.randint(1, 400)})
            if data:
                await ws.send(json.dumps({"type": "trade", "data": data}))
            if time.monotonic() - last_ping > 10:
                await ws.send(json.dumps({"type": "ping"}))
                last_ping = time.monotonic()
            await asyncio.sleep(1 / TICKS_PER_SEC)
    finally:
        task.cancel()


def _rfc3339(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _oanda_app(market: Market):
    from starlette.applications import Starlette
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    async def stream(request):
        insts = request.query_params.get("instruments", "XAU_USD").split(",")

        async def gen():
            last_hb = time.monotonic()
            while True:
                for i in insts:
                    if i in START and market.rng.random() < 0.7:
                        mid = market.step(i, 1 / TICKS_PER_SEC)
                        half = (0.17 if i == "XAU_USD" else mid * 0.00004)
                        nd = 3 if i in ("XAU_USD", "USD_JPY") else 5
                        yield json.dumps({"type": "PRICE", "instrument": i, "time": _rfc3339(market_now_ms()),
                                          "bids": [{"price": f"{mid - half:.{nd}f}", "liquidity": 1000000}],
                                          "asks": [{"price": f"{mid + half:.{nd}f}", "liquidity": 1000000}]}) + "\n"
                if time.monotonic() - last_hb > 5:
                    yield json.dumps({"type": "HEARTBEAT", "time": _rfc3339(market_now_ms())}) + "\n"
                    last_hb = time.monotonic()
                await asyncio.sleep(1 / TICKS_PER_SEC)

        return StreamingResponse(gen(), media_type="application/octet-stream")

    return Starlette(routes=[Route("/v3/accounts/{acct}/pricing/stream", stream)])


async def serve(market: Market, stop: asyncio.Event | None = None) -> None:  # pragma: no cover - long-running
    import uvicorn
    from websockets.asyncio.server import serve as ws_serve

    stop = stop or asyncio.Event()
    server = uvicorn.Server(uvicorn.Config(_oanda_app(market), host="127.0.0.1", port=OANDA_PORT, log_level="warning"))
    async with ws_serve(lambda ws: _finnhub_handler(ws, market), "127.0.0.1", FINNHUB_PORT):
        log.info("simulated feeds: ws://127.0.0.1:%d (stocks), http://127.0.0.1:%d (gold/FX)", FINNHUB_PORT, OANDA_PORT)
        srv = asyncio.create_task(server.serve())
        await stop.wait()
        server.should_exit = True
        await srv


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #

def _scaled(m1: pd.DataFrame, target_last: float) -> pd.DataFrame:
    f = target_last / float(m1["Close"].iloc[-1])
    out = m1.copy()
    for c in ("Open", "High", "Low", "Close"):
        out[c] *= f
    if "Spread" in out:
        out["Spread"] = out["Spread"] * (f if f < 3 else 1)
    return out


def seed(cache_dir: Path, mnow: pd.Timestamp, run_lab: bool = True, progress=print,
         symbols=SIM_SYMBOLS, history_days: int = HISTORY_DAYS) -> dict[str, float]:
    """History up to `mnow` (market time) for SPY/QQQ/XAU_USD, previous closes, lab results.
    Returns the last prices the simulator continues from."""
    from forecaster import lab as L
    from forecaster.live_feed import LiveStore, live_db

    cache_dir = Path(cache_dir)
    store = I.IntradayStore(cache_dir / "intraday")
    rng = np.random.default_rng(int(mnow.timestamp()) // 86400)
    last, prev = {}, {}
    start = (mnow - pd.Timedelta(days=history_days)).tz_convert(I.NY).strftime("%Y-%m-%d")
    end = mnow.tz_convert(I.NY).strftime("%Y-%m-%d")
    for n, sym in enumerate(symbols):
        spec = I.INSTRUMENTS[sym]
        m1 = I.synthetic_m1(sym, start, end, seed=int(rng.integers(1e9)) + n)
        m1 = m1[m1.index < mnow]
        m1 = _scaled(m1, START[sym] * float(np.exp(rng.normal(0, 0.01))))
        for month, g in m1.groupby(m1.index.tz_convert("UTC").tz_localize(None).to_period("M")):
            store.write(spec.source, sym, month, g)
        last[sym] = float(m1["Close"].iloc[-1])
        days = I.daily_bars(I.session_bars(m1, spec), spec)
        today = pd.Timestamp(mnow.tz_convert(I.NY).date())
        before = days[days.index < today]
        prev[sym] = float(before["Close"].iloc[-1]) if len(before) else last[sym]
    for sym in START:
        if sym not in last:
            last[sym] = START[sym] * float(np.exp(rng.normal(0, 0.01)))
            prev[sym] = last[sym] * float(np.exp(rng.normal(0, DAILY_VOL[sym] * 0.6)))
    from forecaster.live_feed import DXY, dxy_value

    prev[DXY] = dxy_value(prev) or 104.0
    LiveStore(live_db(cache_dir)).set_prev_close(prev)
    if run_lab and not (L.lab_dir(cache_dir) / "results.json").exists():
        progress("Demo: running the Strategy Lab on synthetic history (about a minute, first demo only)...")
        L.run_lab(L.synthetic_bars(), L.lab_dir(cache_dir), synthetic=True, n_random=500, progress=lambda m: None)
    (cache_dir / "DEMO.txt").write_text("Simulated market data for demo mode. Safe to delete.\n")
    return last


def prepare_demo(cache_dir: Path, mnow: pd.Timestamp, run_lab: bool = True, progress=print,
                 symbols=SIM_SYMBOLS, history_days: int = HISTORY_DAYS) -> dict[str, float]:
    """Fresh simulated day: wipe the previous demo's live/shadow/intraday data (the demo clock
    restarts at 09:44, so old bars would overlap), keep the lab results, then seed."""
    import shutil

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ("live.sqlite", "shadow.sqlite"):
        for p in cache_dir.glob(name + "*"):
            p.unlink(missing_ok=True)
    shutil.rmtree(cache_dir / "intraday", ignore_errors=True)
    return seed(cache_dir, mnow, run_lab=run_lab, progress=progress, symbols=symbols, history_days=history_days)


def last_prices(cache_dir: Path) -> dict[str, float]:
    """Where the simulator continues from: the seeded history's last closes (START otherwise)."""
    store = I.IntradayStore(Path(cache_dir) / "intraday")
    out = dict(START)
    for sym in SIM_SYMBOLS:
        df = store.read(I.INSTRUMENTS[sym].source, sym, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=40))
        if len(df):
            out[sym] = float(df["Close"].iloc[-1])
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - long-running
    import argparse
    import signal

    from forecaster.config import PROJECT_ROOT, load_settings
    from forecaster.monitor import setup_logging

    ap = argparse.ArgumentParser(description="Simulated Finnhub + OANDA feeds for demo mode")
    ap.add_argument("--seed", action="store_true", help="re-seed demo data first (the launcher already does)")
    args = ap.parse_args(argv)
    setup_logging(PROJECT_ROOT / "logs", "simfeed")
    settings = load_settings()
    prices = prepare_demo(settings.cache_dir, market_now(), progress=log.info) if args.seed \
        else last_prices(settings.cache_dir)

    async def run():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        await serve(Market(prices), stop)

    asyncio.run(run())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
