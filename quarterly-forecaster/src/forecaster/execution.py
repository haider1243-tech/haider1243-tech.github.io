"""Order execution on OANDA v20, PRACTICE by default, behind hard risk limits.

Only alerts the webhook judged **TAKE** can become orders, and only when every
check below passes. Anything unexpected blocks the order; it never "tries anyway".

Live trading is off unless BOTH are true:
  * OANDA_ENVIRONMENT=live, and
  * LIVE_TRADING=true is written in the project's .env file. It is read from the
    file itself, so an exported shell variable or a typo'd value can't unlock live.
Asking for live without that flag fails at load time (LiveTradingNotAllowed).
The practice host is used in every other case.

Order placement is also opt-in on practice: EXECUTION_ENABLED=true.

Checks, in order (first failure wins, recorded in the `orders` table):
  verdict TAKE -> execution enabled -> kill switch off -> instrument allowed ->
  stop given and on the right side -> alert fresh -> not already handled ->
  account reachable -> today's loss within MAX_DAILY_LOSS -> open trades below
  MAX_OPEN_TRADES -> live price close to the alert price -> size within
  MAX_POSITION_UNITS (counting the existing position) and >= the minimum trade
  size -> submit MARKET FOK with stop-loss and take-profit attached on fill.

Kill switch: the file data/KILL_SWITCH (or KILL_SWITCH=true). While it exists, no
orders are placed. `--kill [--flatten]` creates it; `--resume --yes` removes it. It
trips by itself after MAX_CONSECUTIVE_ERRORS API failures, or when a position is
found larger than MAX_POSITION_UNITS.

Daily loss: NAV change since the first check of the OANDA trading day (17:00
New York). If the process starts after losses were already taken that day,
they're not counted, so keep the service running.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd
from dotenv import dotenv_values

from forecaster.config import PROJECT_ROOT

log = logging.getLogger("forecaster.execution")

HOSTS = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}
NY = "America/New_York"

# Absolute ceilings no setting can exceed.
HARD_MAX_RISK_PCT = 2.0
HARD_MAX_POSITION_UNITS = 1_000.0

# TradingView symbol -> OANDA instrument
DEFAULT_INSTRUMENTS = {"XAUUSD": "XAU_USD", "OANDA:XAUUSD": "XAU_USD"}


class LiveTradingNotAllowed(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionConfig:
    token: str | None
    account_id: str | None
    environment: str = "practice"
    enabled: bool = False
    instruments: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_INSTRUMENTS))
    risk_pct: float = 0.5
    max_position_units: float = 10.0
    max_daily_loss: float = 100.0          # account currency
    max_open_trades: int = 1
    max_alert_age_s: float = 120.0
    max_price_deviation: float = 0.005     # |alert price / OANDA mid - 1|
    max_consecutive_errors: int = 3
    use_client_extensions: bool = True     # must be False on MT4-linked accounts
    kill_file: Path = PROJECT_ROOT / "data" / "KILL_SWITCH"
    kill_env: bool = False

    @property
    def host(self) -> str:
        return HOSTS[self.environment]

    @property
    def is_live(self) -> bool:
        return self.environment == "live"


def _bool(v: str | None) -> bool:
    return str(v or "").strip().lower() == "true"


def load_execution_config(root: Path = PROJECT_ROOT, env: dict[str, str] | None = None,
                          dotenv: dict[str, str | None] | None = None,
                          data_dir: Path | None = None) -> ExecutionConfig:
    """Build the config. LIVE_TRADING is read ONLY from the .env file (`dotenv`)."""
    dotenv = dict(dotenv_values(root / ".env")) if dotenv is None else dotenv
    env = {**{k: v for k, v in dotenv.items() if v is not None}, **(dict(os.environ) if env is None else env)}

    environment = (env.get("OANDA_ENVIRONMENT") or "practice").strip().lower()
    if environment not in HOSTS:
        raise ValueError(f"OANDA_ENVIRONMENT must be 'practice' or 'live', got {environment!r}")
    if environment == "live" and dotenv.get("LIVE_TRADING") != "true":
        raise LiveTradingNotAllowed(
            "OANDA_ENVIRONMENT=live requires LIVE_TRADING=true written in the .env file "
            "(exactly 'true'; shell variables don't count). Refusing to start.")

    def num(key: str, default: float, lo: float, hi: float) -> float:
        raw = env.get(key)
        v = default if raw in (None, "") else float(raw)
        if not (lo <= v <= hi):
            raise ValueError(f"{key}={v} outside the allowed range [{lo}, {hi}]")
        return v

    inst = dict(DEFAULT_INSTRUMENTS)
    if env.get("OANDA_INSTRUMENTS"):
        inst = dict(pair.split("=", 1) for pair in env["OANDA_INSTRUMENTS"].split(",") if "=" in pair)
    data = Path(data_dir or env.get("FORECASTER_CACHE_DIR") or root / "data")
    if not data.is_absolute():
        data = root / data
    return ExecutionConfig(
        token=env.get("OANDA_API_TOKEN") or None,
        account_id=env.get("OANDA_ACCOUNT_ID") or None,
        environment=environment,
        enabled=_bool(env.get("EXECUTION_ENABLED")),
        instruments={k.strip().upper(): v.strip() for k, v in inst.items()},
        risk_pct=num("RISK_PER_TRADE_PCT", 0.5, 0.01, HARD_MAX_RISK_PCT),
        max_position_units=num("MAX_POSITION_UNITS", 10.0, 0.0, HARD_MAX_POSITION_UNITS),
        max_daily_loss=num("MAX_DAILY_LOSS", 100.0, 0.0, 1e9),
        max_open_trades=int(num("MAX_OPEN_TRADES", 1, 0, 20)),
        max_alert_age_s=num("MAX_ALERT_AGE_SECONDS", 120.0, 1.0, 3600.0),
        max_price_deviation=num("MAX_PRICE_DEVIATION", 0.005, 0.0001, 0.05),
        use_client_extensions=not _bool(env.get("OANDA_MT4_ACCOUNT")),
        kill_file=data / "KILL_SWITCH",
        kill_env=_bool(env.get("KILL_SWITCH")),
    )


# --------------------------------------------------------------------------- #
# OANDA client
# --------------------------------------------------------------------------- #

class OandaError(RuntimeError):
    pass


class OandaClient:
    """Minimal v20 REST client. The token goes in the Authorization header only."""

    def __init__(self, cfg: ExecutionConfig, transport: httpx.BaseTransport | None = None,
                 timeout: float = 10.0) -> None:
        if not cfg.token or not cfg.account_id:
            raise OandaError("OANDA_API_TOKEN and OANDA_ACCOUNT_ID must be set")
        self.cfg = cfg
        self._http = httpx.Client(base_url=cfg.host, transport=transport, timeout=timeout,
                                  headers={"Authorization": f"Bearer {cfg.token}",
                                           "Content-Type": "application/json",
                                           "Accept-Datetime-Format": "RFC3339"})

    def _req(self, method: str, path: str, **kw) -> dict:
        try:
            r = self._http.request(method, f"/v3/accounts/{self.cfg.account_id}{path}", **kw)
        except httpx.HTTPError as exc:
            raise OandaError(f"{method} {path}: {type(exc).__name__}") from None
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code >= 400:
            raise OandaError(f"{method} {path}: HTTP {r.status_code} {body.get('errorMessage', '')}".strip())
        return body

    def summary(self) -> dict:
        return self._req("GET", "/summary")["account"]

    def instrument(self, name: str) -> dict:
        items = self._req("GET", "/instruments", params={"instruments": name}).get("instruments", [])
        if not items:
            raise OandaError(f"instrument {name} not available on this account")
        return items[0]

    def price(self, name: str) -> dict:
        prices = self._req("GET", "/pricing", params={"instruments": name}).get("prices", [])
        if not prices:
            raise OandaError(f"no price for {name}")
        return prices[0]

    def open_positions(self) -> list[dict]:
        return self._req("GET", "/openPositions").get("positions", [])

    def market_order(self, order: dict) -> dict:
        return self._req("POST", "/orders", json={"order": order})

    def close_position(self, name: str, long_units: float, short_units: float) -> dict:
        body = {}
        if long_units:
            body["longUnits"] = "ALL"
        if short_units:
            body["shortUnits"] = "ALL"
        return self._req("PUT", f"/positions/{name}/close", json=body)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id      INTEGER UNIQUE,
    created_at    TEXT NOT NULL,
    environment   TEXT NOT NULL,
    instrument    TEXT,
    side          TEXT,
    units         REAL,
    stop          REAL,
    target        REAL,
    status        TEXT NOT NULL,   -- not_taken | blocked | filled | cancelled | rejected | error
    reason        TEXT,
    oanda_id      TEXT,
    fill_price    REAL,
    response      TEXT
);
CREATE TABLE IF NOT EXISTS daily_nav (
    day       TEXT PRIMARY KEY,   -- OANDA trading day (starts 17:00 New York)
    start_nav REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kill_log (
    at TEXT NOT NULL, action TEXT NOT NULL, reason TEXT
);
"""


class OrderStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as con, con:
            con.executescript(_SCHEMA)

    def _con(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def has_alert(self, alert_id: int) -> bool:
        with closing(self._con()) as con:
            return con.execute("SELECT 1 FROM orders WHERE alert_id=?", (alert_id,)).fetchone() is not None

    def record(self, **row) -> None:
        row.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        cols = ", ".join(row)
        with closing(self._con()) as con, con:
            con.execute(f"INSERT OR IGNORE INTO orders ({cols}) VALUES ({', '.join('?' * len(row))})",
                        list(row.values()))

    def orders(self) -> pd.DataFrame:
        with closing(self._con()) as con:
            return pd.read_sql("SELECT * FROM orders ORDER BY id", con)

    def start_nav(self, day: str, nav: float) -> float:
        with closing(self._con()) as con, con:
            con.execute("INSERT OR IGNORE INTO daily_nav VALUES (?,?,?)",
                        (day, nav, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            return float(con.execute("SELECT start_nav FROM daily_nav WHERE day=?", (day,)).fetchone()[0])

    def kill_event(self, action: str, reason: str) -> None:
        with closing(self._con()) as con, con:
            con.execute("INSERT INTO kill_log VALUES (?,?,?)",
                        (datetime.now(timezone.utc).isoformat(timespec="seconds"), action, reason))


def trading_day(now: pd.Timestamp) -> str:
    """OANDA's trading day rolls at 17:00 New York."""
    return (now.tz_convert(NY) + pd.Timedelta(hours=7)).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OrderRequest:
    alert_id: int
    symbol: str
    side: str            # long | short
    price: float         # alert (signal) price
    stop: float | None
    target: float | None
    alert_time: pd.Timestamp
    verdict: str


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    units: float = 0.0
    fill_price: float | None = None
    oanda_id: str | None = None


def _round_down(x: float, decimals: int) -> float:
    f = 10 ** decimals
    return math.floor(abs(x) * f + 1e-9) / f


class Executor:
    def __init__(self, cfg: ExecutionConfig, store: OrderStore, client: OandaClient | None = None,
                 clock: Callable[[], pd.Timestamp] | None = None,
                 notify: Callable[[str], Any] | None = None) -> None:
        self.cfg, self.store = cfg, store
        self._client = client
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.notify = notify or (lambda text: None)
        self._lock = threading.Lock()
        self._errors = 0
        if cfg.is_live:
            log.warning("LIVE TRADING ENABLED on %s (LIVE_TRADING=true in .env)", cfg.host)

    @property
    def client(self) -> OandaClient:
        if self._client is None:
            self._client = OandaClient(self.cfg)
        return self._client

    # -- kill switch -------------------------------------------------------- #
    def kill_active(self) -> bool:
        return self.cfg.kill_env or self.cfg.kill_file.exists()

    def trip_kill(self, reason: str) -> None:
        self.cfg.kill_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.kill_file.write_text(f"{self.clock().isoformat()} {reason}\n")
        self.store.kill_event("trip", reason)
        log.error("KILL SWITCH tripped: %s", reason)
        self.notify(f"🛑 Kill switch tripped: {reason}")

    def resume(self) -> None:
        if self.cfg.kill_file.exists():
            self.cfg.kill_file.unlink()
        self.store.kill_event("resume", "manual")

    def flatten_all(self) -> list[str]:
        closed = []
        for p in self.client.open_positions():
            lu, su = float(p["long"]["units"]), float(p["short"]["units"])
            if lu or su:
                self.client.close_position(p["instrument"], lu, su)
                closed.append(p["instrument"])
        return closed

    # -- main entry ---------------------------------------------------------- #
    def handle(self, req: OrderRequest) -> Decision:
        with self._lock:
            d = self._decide_and_submit(req)
        base = {"alert_id": req.alert_id, "environment": self.cfg.environment, "side": req.side,
                "instrument": self.cfg.instruments.get(req.symbol.upper()), "stop": req.stop,
                "target": req.target, "units": d.units, "status": d.status, "reason": d.reason,
                "oanda_id": d.oanda_id, "fill_price": d.fill_price}
        if not (d.status == "blocked" and d.reason.startswith("already handled")):
            self.store.record(**base)
        if d.status not in ("not_taken",) and not d.reason.startswith("already handled"):
            env = "LIVE" if self.cfg.is_live else "practice"
            icon = {"filled": "🟢", "blocked": "🚫", "cancelled": "⚪", "rejected": "🔴", "error": "🔴"}.get(d.status, "•")
            self.notify(f"{icon} OANDA {env}: {d.status} {req.side} {req.symbol} "
                        f"{d.units:g} units" + (f" @ {d.fill_price}" if d.fill_price else "") + f"\n{d.reason}")
        return d

    def _api_ok(self) -> None:
        self._errors = 0

    def _api_failed(self, exc: Exception) -> Decision:
        self._errors += 1
        if self._errors >= self.cfg.max_consecutive_errors:
            self.trip_kill(f"{self._errors} consecutive OANDA API errors (last: {exc})")
        return Decision("error", f"OANDA API error: {exc}")

    def _decide_and_submit(self, req: OrderRequest) -> Decision:
        cfg = self.cfg
        if req.verdict != "TAKE":
            return Decision("not_taken", f"verdict {req.verdict}: orders only on TAKE")
        if not cfg.enabled:
            return Decision("blocked", "execution disabled (set EXECUTION_ENABLED=true)")
        if self.kill_active():
            return Decision("blocked", "kill switch active")
        instrument = cfg.instruments.get(req.symbol.upper())
        if not instrument:
            return Decision("blocked", f"{req.symbol} is not in the allowed instruments")
        if req.stop is None or not math.isfinite(req.stop) or req.stop <= 0:
            return Decision("blocked", "no stop-loss in the alert: refusing to size or place the order")
        if (req.side == "long" and req.stop >= req.price) or (req.side == "short" and req.stop <= req.price):
            return Decision("blocked", f"stop {req.stop} is on the wrong side of {req.price} for a {req.side}")
        if req.target is not None and ((req.side == "long" and req.target <= req.price) or
                                       (req.side == "short" and req.target >= req.price)):
            return Decision("blocked", f"target {req.target} is on the wrong side for a {req.side}")
        age = (self.clock() - req.alert_time).total_seconds()
        if age > cfg.max_alert_age_s or age < -60:
            return Decision("blocked", f"alert is {age:.0f}s old (max {cfg.max_alert_age_s:.0f}s)")
        if self.store.has_alert(req.alert_id):
            return Decision("blocked", "already handled this alert")

        try:
            acct = self.client.summary()
            positions = self.client.open_positions()
            spec = self.client.instrument(instrument)
            px = self.client.price(instrument)
            self._api_ok()
        except OandaError as exc:
            return self._api_failed(exc)

        nav = float(acct["NAV"])
        start = self.store.start_nav(trading_day(self.clock()), nav)
        day_pl = nav - start
        if day_pl <= -cfg.max_daily_loss:
            return Decision("blocked", f"daily loss limit hit: {day_pl:,.2f} vs -{cfg.max_daily_loss:,.2f}")
        if int(acct.get("openTradeCount", 0)) >= cfg.max_open_trades:
            return Decision("blocked", f"{acct.get('openTradeCount')} open trades (max {cfg.max_open_trades})")

        held = 0.0
        for p in positions:
            if p["instrument"] == instrument:
                held = abs(float(p["long"]["units"])) + abs(float(p["short"]["units"]))
        if held > cfg.max_position_units:
            self.trip_kill(f"{instrument} position {held:g} exceeds MAX_POSITION_UNITS {cfg.max_position_units:g}")
            return Decision("blocked", "existing position above the limit: kill switch tripped")

        bid = float(px["bids"][0]["price"])
        ask = float(px["asks"][0]["price"])
        mid = (bid + ask) / 2
        if abs(req.price / mid - 1) > cfg.max_price_deviation:
            return Decision("blocked", f"alert price {req.price} is {abs(req.price / mid - 1):.2%} from OANDA mid {mid:.3f}")
        entry = ask if req.side == "long" else bid
        if (req.side == "long" and req.stop >= bid) or (req.side == "short" and req.stop <= ask):
            return Decision("blocked", f"stop {req.stop} already crossed by the market ({bid}/{ask})")

        unit_dec = int(spec.get("tradeUnitsPrecision", 0))
        price_dec = int(spec.get("displayPrecision", 3))
        min_units = float(spec.get("minimumTradeSize", 1))
        risk = nav * cfg.risk_pct / 100
        units = _round_down(risk / abs(entry - req.stop), unit_dec)
        room = max(cfg.max_position_units - held, 0.0)
        capped = _round_down(min(units, room, float(spec.get("maximumOrderUnits", units) or units)), unit_dec)
        if capped < min_units or capped <= 0:
            return Decision("blocked", f"size {capped:g} below minimum {min_units:g} "
                                       f"(risk-based {units:g}, room under the cap {room:g})")
        note = "" if capped == units else f" (capped from {units:g})"

        signed = capped if req.side == "long" else -capped
        order: dict = {
            "type": "MARKET", "instrument": instrument, "units": f"{signed:.{unit_dec}f}",
            "timeInForce": "FOK", "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{req.stop:.{price_dec}f}", "timeInForce": "GTC"},
        }
        if req.target is not None:
            order["takeProfitOnFill"] = {"price": f"{req.target:.{price_dec}f}", "timeInForce": "GTC"}
        if cfg.use_client_extensions:
            order["clientExtensions"] = {"id": f"tv-alert-{req.alert_id}", "tag": "vwap_gold"}
        try:
            resp = self.client.market_order(order)
            self._api_ok()
        except OandaError as exc:
            return self._api_failed(exc)

        if "orderFillTransaction" in resp:
            f = resp["orderFillTransaction"]
            return Decision("filled", f"risk {cfg.risk_pct}% of NAV {nav:,.2f}{note}", capped,
                            float(f.get("price", entry)), str(f.get("id")))
        if "orderCancelTransaction" in resp:
            return Decision("cancelled", f"OANDA cancelled: {resp['orderCancelTransaction'].get('reason', '?')}",
                            capped, oanda_id=str(resp["orderCancelTransaction"].get("id")))
        if "orderRejectTransaction" in resp:
            return Decision("rejected", f"OANDA rejected: {resp['orderRejectTransaction'].get('rejectReason', '?')}",
                            capped)
        return Decision("error", f"unexpected OANDA response keys: {sorted(resp)}", capped)


def request_from_alert(alert_id: int, alert, t: pd.Timestamp, verdict: str) -> OrderRequest:
    return OrderRequest(alert_id, alert.symbol, alert.side, float(alert.price),
                        getattr(alert, "stop", None), getattr(alert, "target", None), t, verdict)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="OANDA execution controls")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true", help="environment, limits, kill switch, account, positions")
    g.add_argument("--kill", action="store_true", help="activate the kill switch")
    g.add_argument("--resume", action="store_true", help="remove the kill switch (needs --yes)")
    ap.add_argument("--flatten", action="store_true", help="with --kill: also close all open positions")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cfg = load_execution_config()
    store = OrderStore(cfg.kill_file.parent / "alerts.sqlite")
    ex = Executor(cfg, store)
    if args.kill:
        ex.trip_kill("manual (--kill)")
        if args.flatten:
            print("closed:", ex.flatten_all() or "nothing open")
        return 0
    if args.resume:
        if not args.yes:
            print("Refusing: pass --yes to confirm you want orders to resume.")
            return 1
        ex.resume()
        print("kill switch removed")
        return 0
    print(f"environment : {cfg.environment} ({cfg.host}){'  <-- LIVE' if cfg.is_live else ''}")
    print(f"execution   : {'ENABLED' if cfg.enabled else 'disabled'}; kill switch {'ON' if ex.kill_active() else 'off'}")
    print(f"limits      : risk {cfg.risk_pct}%/trade, max {cfg.max_position_units:g} units, "
          f"max daily loss {cfg.max_daily_loss:,.2f}, max open trades {cfg.max_open_trades}")
    if cfg.token and cfg.account_id:
        a = ex.client.summary()
        nav, ccy = float(a["NAV"]), a.get("currency", "")
        cad = ""
        if ccy == "USD":
            from forecaster.config import load_settings
            from forecaster.market_data import USD_CAD_SYMBOL, ParquetStore

            fx = ParquetStore(load_settings().cache_dir / "ohlcv").read(USD_CAD_SYMBOL)
            cad = f" (≈ C${nav * float(fx['Close'].iloc[-1]):,.2f})" if fx is not None else " (no USD/CAD cached)"
        print(f"account     : NAV {nav:,.2f} {ccy}{cad}, open trades {a.get('openTradeCount')}")
        print("              limits (MAX_DAILY_LOSS) are in the account currency")
        for p in ex.client.open_positions():
            print(f"position    : {p['instrument']} long {p['long']['units']} short {p['short']['units']}")
    o = store.orders().tail(10)
    print(o[["created_at", "status", "side", "units", "reason"]].to_string(index=False) if len(o) else "no orders yet")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
