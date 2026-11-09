"""TradingView webhook -> forecaster risk checks -> TAKE / CAUTION / SKIP -> Telegram.

Run (needs a public HTTPS URL on port 443; TradingView only posts to 80/443):

    uvicorn forecaster.webhook_server:create_app --factory --host 127.0.0.1 --port 8000
    # then expose it, e.g. `cloudflared tunnel --url http://localhost:8000`
    # TradingView alert webhook URL: https://<your-host>/tv-alert

TradingView alert message:
    {"ticker":"{{ticker}}","price":"{{close}}","strategy":"vwap_gold","side":"long",
     "time":"{{timenow}}","secret":"<WEBHOOK_SECRET>"}

What the checks are, and are not
--------------------------------
They are *risk filters* built from the dashboard snapshots: HMM regime, high-impact
events in the next 24 h, recent news-sentiment shocks, and VIX level/spike. They
don't predict the trade. The forecaster itself does not reliably beat a naive
baseline (see results/), so a TAKE means "no red flags found", not "edge".
Outcomes at +1h/+4h/+1d are logged so you can measure whether the filters help
(`python -m forecaster.webhook_server --stats`).

Security
--------
* Fails closed: no WEBHOOK_SECRET configured -> 503 for every alert.
* Secret compared in constant time; never stored or logged (payload saved without it).
* Optional IP allowlist (WEBHOOK_ALLOWED_IPS); body size cap; JSON parsed from the
  raw body because TradingView sends text/plain unless the message is valid JSON.
* Duplicate alerts (same ticker/strategy/side/time) return the stored verdict and
  aren't re-sent to Telegram.
* The Telegram token is part of the API URL, so errors are reported without it.
"""

from __future__ import annotations

import asyncio
import hmac
import html
import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from forecaster.config import Settings, load_settings
from forecaster.pipeline import read_json, safe_name

log = logging.getLogger("forecaster.webhook")

MAX_BODY_BYTES = 10_000
HORIZONS: dict[str, pd.Timedelta] = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4),
                                     "1d": pd.Timedelta(days=1)}
NY = "America/New_York"

# TradingView symbol -> (forecaster ticker for regime, price symbol for outcomes,
# news tags whose sentiment shocks matter). Gold watches the "gold" news topic,
# not general market news: bad equity news is often good for gold.
TICKER_MAP: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "XAUUSD": ("GC=F", "GC=F", ("gold",)),
    "GOLD": ("GC=F", "GC=F", ("gold",)),
    "GC1!": ("GC=F", "GC=F", ("gold",)),
    "MGC1!": ("GC=F", "GC=F", ("gold",)),
    "GLD": ("GLD", "GLD", ("gold", "GLD")),
    "ES1!": ("^GSPC", "ES=F", ("MARKET",)),
    "MES1!": ("^GSPC", "ES=F", ("MARKET",)),
    "SPX": ("^GSPC", "^GSPC", ("MARKET",)),
    "SPY": ("SPY", "SPY", ("MARKET",)),
    "NQ1!": ("^IXIC", "NQ=F", ("MARKET",)),
    "MNQ1!": ("^IXIC", "NQ=F", ("MARKET",)),
    "QQQ": ("QQQ", "QQQ", ("MARKET",)),
}
MARKET_FALLBACK = ("SPY", "^GSPC")  # regime source when the ticker has no snapshot

# Typical release times (US Eastern); calendar events are stored as dates only.
EVENT_TIMES_ET: dict[str, tuple[int, int]] = {
    "CPI": (8, 30), "PPI": (8, 30), "Jobs": (8, 30), "GDP": (8, 30), "FOMC": (14, 0),
}
EARNINGS_TIMES_ET = {"before open": (7, 0), "after close": (16, 5), "during market": (12, 0)}


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #

class Alert(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    ticker: str = Field(min_length=1, max_length=40)
    price: float
    strategy: str = Field(min_length=1, max_length=80)
    side: Literal["long", "short"]
    time: str | None = None
    secret: str = Field(min_length=1, max_length=512)
    stop: float | None = None      # optional: needed for execution (forecaster.execution)
    target: float | None = None

    @field_validator("side", mode="before")
    @classmethod
    def _side(cls, v: Any) -> Any:
        v = str(v).strip().lower()
        return {"buy": "long", "sell": "short"}.get(v, v)

    @field_validator("price", mode="before")
    @classmethod
    def _price(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return v

    @field_validator("price")
    @classmethod
    def _positive(cls, v: float) -> float:
        if not v > 0 or v != v:
            raise ValueError("price must be a positive number")
        return v

    @field_validator("stop", "target", mode="before")
    @classmethod
    def _optional_price(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and v.strip() in ("", "na", "NaN")):
            return None
        return v.replace(",", "").strip() if isinstance(v, str) else v

    @field_validator("stop", "target")
    @classmethod
    def _optional_positive(cls, v: float | None) -> float | None:
        if v is not None and (not v > 0 or v != v):
            raise ValueError("stop/target must be positive numbers")
        return v

    @property
    def symbol(self) -> str:
        return self.ticker.split(":")[-1].upper()  # "OANDA:XAUUSD" -> "XAUUSD"


def parse_alert_time(raw: str | None, received: pd.Timestamp) -> tuple[pd.Timestamp, bool]:
    """TradingView {{timenow}} is ISO UTC ('2026-11-08T14:30:05Z'). Falls back to receipt time."""
    if raw:
        try:
            t = pd.Timestamp(raw)
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            if abs((t - received).total_seconds()) < 7 * 86400:  # reject nonsense/unsubstituted
                return t, True
        except (ValueError, TypeError):
            pass
    return received, False


def resolve_symbol(symbol: str) -> tuple[str, str, tuple[str, ...]]:
    if symbol in TICKER_MAP:
        return TICKER_MAP[symbol]
    return symbol, symbol, ("MARKET", symbol)  # unknown: assume an equity


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

Level = Literal["ok", "caution", "skip", "unknown"]


@dataclass(frozen=True)
class RuleConfig:
    regime_caution: float = 0.60       # P(adverse regimes) above this -> caution
    vix_caution: float = 30.0
    vix_skip: float = 40.0
    vix_spike: float = 0.20            # 1-day VIX change >= +20% -> caution
    event_skip_hours: float = 4.0      # high-impact event this close -> skip
    event_caution_hours: float = 24.0
    shock_lookback_days: int = 2
    snapshot_max_age_days: float = 8.0
    vix_max_age_days: float = 4.0
    cautions_to_skip: int = 3          # this many real cautions -> skip


@dataclass
class Check:
    name: str
    level: Level
    reason: str
    data: dict = field(default_factory=dict)


def _age_days(iso: str | None, now: pd.Timestamp) -> float:
    if not iso:
        return float("inf")
    t = pd.Timestamp(iso)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return (now - t).total_seconds() / 86400


def check_regime(snap: dict | None, source: str, is_fallback: bool, side: str,
                 now: pd.Timestamp, cfg: RuleConfig) -> Check:
    if snap is None:
        return Check("regime", "unknown", "no forecast snapshot to read the regime from")
    reg = snap.get("regime", {}).get("current", {})
    bull, bear, hv = (float(reg.get(k, 0.0)) for k in ("bull", "bear", "high_vol"))
    where = f"{source} regime" + (" (market fallback; no snapshot for this ticker)" if is_fallback else "")
    data = {"source": source, "bull": bull, "bear": bear, "high_vol": hv}
    # Freshness is about the *data*: a snapshot generated today from old prices is stale.
    age = max(_age_days(snap.get("generated_at"), now), _age_days(snap.get("as_of"), now))
    stale = f"; regime data is {age:.0f} days old" if age > cfg.snapshot_max_age_days else ""
    probs = f"bull {bull:.0%} / bear {bear:.0%} / high-vol {hv:.0%}"
    if side == "long" and bear + hv > cfg.regime_caution:
        return Check("regime", "caution", f"{where}: {probs}, adverse for a long{stale}", data)
    if side == "short" and bull > cfg.regime_caution:
        return Check("regime", "caution", f"{where}: {probs}, adverse for a short{stale}", data)
    if stale:
        return Check("regime", "unknown", f"{where}: {probs}{stale}", data)
    return Check("regime", "ok", f"{where}: {probs}", data)


def event_time(e: dict) -> pd.Timestamp:
    """Best-guess UTC time of a calendar event (dates are stored without times)."""
    d = pd.Timestamp(e["date"]).normalize()
    hm = EVENT_TIMES_ET.get(e.get("category", ""))
    if hm is None and e.get("category") == "Earnings":
        detail = e.get("detail") or ""
        hm = next((v for k, v in EARNINGS_TIMES_ET.items() if k in detail), (12, 0))
    hm = hm or (12, 0)
    return (d + pd.Timedelta(hours=hm[0], minutes=hm[1])).tz_localize(NY).tz_convert("UTC")


def check_events(market: dict | None, now: pd.Timestamp, cfg: RuleConfig,
                 tickers: tuple[str, ...] = ()) -> Check:
    if market is None:
        return Check("events", "unknown", "no events snapshot")
    soon, near = [], []
    for e in market.get("events", []):
        relevant = e["impact"] == "high" or (e.get("ticker") and e["ticker"] in tickers)
        if not relevant:
            continue
        hours = (event_time(e) - now).total_seconds() / 3600
        if -1 <= hours <= cfg.event_skip_hours:
            soon.append((e, hours))
        elif cfg.event_skip_hours < hours <= cfg.event_caution_hours:
            near.append((e, hours))

    def fmt(items):
        return ", ".join(f"{e['name']} in ~{max(h, 0):.0f}h" for e, h in items)

    data = {"within_skip": len(soon), "within_24h": len(near) + len(soon)}
    if soon:
        return Check("events", "skip", f"high-impact event imminent: {fmt(soon)}", data)
    if near:
        return Check("events", "caution", f"high-impact event in next 24h: {fmt(near)}", data)
    age = _age_days(market.get("generated_at"), now)
    if age > cfg.snapshot_max_age_days:
        return Check("events", "unknown", f"no high-impact events listed, but snapshot is {age:.0f} days old", data)
    return Check("events", "ok", "no high-impact events in the next 24h", data)


def check_sentiment(market: dict | None, side: str, now: pd.Timestamp, cfg: RuleConfig,
                    tags: tuple[str, ...]) -> Check:
    if market is None or not market.get("sentiment"):
        return Check("sentiment", "unknown", "no news-sentiment data")
    watch = set(tags)
    since = now.tz_convert(NY).tz_localize(None).normalize() - pd.Timedelta(days=cfg.shock_lookback_days)
    recent = [s for s in market.get("shocks", [])
              if s["tag"] in watch and pd.Timestamp(s["date"]) >= since]
    against = [s for s in recent if (s["z"] < 0 and side == "long") or (s["z"] > 0 and side == "short")]
    data = {"recent_shocks": len(recent), "against": len(against)}
    if against:
        txt = ", ".join(f"{s['tag']} {str(s['date'])[:10]} (z {s['z']:+.1f})" for s in against[:3])
        return Check("sentiment", "caution", f"news-sentiment shock against the trade: {txt}", data)
    if recent:
        return Check("sentiment", "ok", f"{len(recent)} recent sentiment shock(s), none against the trade", data)
    return Check("sentiment", "ok", f"no sentiment shocks in the last {cfg.shock_lookback_days} days", data)


def check_vix(vix: pd.Series | None, now: pd.Timestamp, cfg: RuleConfig) -> Check:
    if vix is None or vix.dropna().empty:
        return Check("vix", "unknown", "no cached VIX data")
    v = vix.dropna().sort_index()
    last, last_date = float(v.iloc[-1]), pd.Timestamp(v.index[-1])
    chg = float(v.iloc[-1] / v.iloc[-2] - 1) if len(v) > 1 else 0.0
    age = (now.tz_convert(NY).tz_localize(None).normalize() - last_date.normalize()).days
    data = {"vix": last, "change_1d": chg, "as_of": str(last_date.date())}
    stale = f" (as of {last_date:%b %d}, {age} days old)" if age > cfg.vix_max_age_days else ""
    if last >= cfg.vix_skip:
        return Check("vix", "skip", f"VIX {last:.1f} ≥ {cfg.vix_skip:.0f}: extreme volatility{stale}", data)
    if last >= cfg.vix_caution:
        return Check("vix", "caution", f"VIX {last:.1f} ≥ {cfg.vix_caution:.0f}: high volatility{stale}", data)
    if chg >= cfg.vix_spike:
        return Check("vix", "caution", f"VIX spiked {chg:+.0%} in a day to {last:.1f}{stale}", data)
    if stale:
        return Check("vix", "unknown", f"VIX {last:.1f}{stale}", data)
    return Check("vix", "ok", f"VIX {last:.1f} ({chg:+.0%} 1d)", data)


def decide(checks: list[Check], cfg: RuleConfig) -> tuple[str, list[str]]:
    """SKIP if any check says skip or >= cautions_to_skip real cautions; CAUTION if any
    caution or missing data; else TAKE. Missing data never escalates to SKIP."""
    skips = [c for c in checks if c.level == "skip"]
    cautions = [c for c in checks if c.level == "caution"]
    unknown = [c for c in checks if c.level == "unknown"]
    reasons = [c.reason for c in skips + cautions] + [f"can't verify: {c.reason}" for c in unknown]
    if skips or len(cautions) >= cfg.cautions_to_skip:
        return "SKIP", reasons
    if cautions or unknown:
        return "CAUTION", reasons
    return "TAKE", ["no red flags: " + "; ".join(c.reason for c in checks)]


# --------------------------------------------------------------------------- #
# Context (local files only: the request path never hits the network)
# --------------------------------------------------------------------------- #

class Context:
    def __init__(self, settings: Settings,
                 vix_loader: Callable[[], pd.Series | None] | None = None,
                 fx_loader: Callable[[], float | None] | None = None) -> None:
        self.settings = settings
        self.snap_dir = Path(settings.cache_dir) / "snapshots"
        self._vix_loader = vix_loader or self._cached_vix
        self._fx_loader = fx_loader

    def ticker_snapshot(self, ticker: str) -> tuple[dict | None, str, bool]:
        s = read_json(self.snap_dir / f"ticker_{safe_name(ticker)}.json")
        if s is not None:
            return s, ticker, False
        for t in MARKET_FALLBACK:
            s = read_json(self.snap_dir / f"ticker_{safe_name(t)}.json")
            if s is not None:
                return s, t, True
        return None, ticker, False

    def market(self) -> dict | None:
        return read_json(self.snap_dir / "market.json")

    def vix(self) -> pd.Series | None:
        return self._vix_loader()

    def usd_cad(self) -> float | None:
        """Latest cached USD/CAD (CAD per USD), or None. Local file only."""
        if self._fx_loader is not None:
            return self._fx_loader()
        try:
            from forecaster.market_data import USD_CAD_SYMBOL, ParquetStore, validate_usd_cad

            df = ParquetStore(Path(self.settings.cache_dir) / "ohlcv").read(USD_CAD_SYMBOL)
            return None if df is None else float(validate_usd_cad(df["Close"]).iloc[-1])
        except Exception:
            return None

    def _cached_vix(self) -> pd.Series | None:
        from forecaster.market_data import ParquetStore

        df = ParquetStore(Path(self.settings.cache_dir) / "ohlcv").read("^VIX")
        return None if df is None else df["Close"]


def evaluate_alert(alert: Alert, t: pd.Timestamp, ctx: Context, cfg: RuleConfig) -> tuple[str, list[str], list[Check]]:
    fticker, _, tags = resolve_symbol(alert.symbol)
    snap, source, fallback = ctx.ticker_snapshot(fticker)
    market = ctx.market()
    checks = [
        check_regime(snap, source, fallback, alert.side, t, cfg),
        check_events(market, t, cfg, tickers=(fticker, alert.symbol)),
        check_sentiment(market, alert.side, t, cfg, tags),
        check_vix(ctx.vix(), t, cfg),
    ]
    verdict, reasons = decide(checks, cfg)
    return verdict, reasons, checks


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at    TEXT NOT NULL,
    alert_time     TEXT NOT NULL,
    time_from_tv   INTEGER NOT NULL,
    ticker         TEXT NOT NULL,
    forecaster_ticker TEXT NOT NULL,
    price_symbol   TEXT NOT NULL,
    price          REAL NOT NULL,
    strategy       TEXT NOT NULL,
    side           TEXT NOT NULL,
    payload        TEXT NOT NULL,      -- JSON, secret removed
    dedupe_key     TEXT NOT NULL UNIQUE,
    verdict        TEXT NOT NULL,
    reasons        TEXT NOT NULL,      -- JSON list
    checks         TEXT NOT NULL,      -- JSON list
    telegram_status TEXT NOT NULL DEFAULT 'pending',
    telegram_error TEXT
);
CREATE TABLE IF NOT EXISTS outcomes (
    alert_id   INTEGER NOT NULL REFERENCES alerts(id),
    horizon    TEXT NOT NULL,
    due_at     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',   -- pending | done | unavailable
    ref_price  REAL,      -- price-source close at alert time (avoids spot/futures basis)
    price_at   REAL,
    price_time TEXT,
    ret        REAL,      -- raw return ref -> horizon
    signed_ret REAL,      -- ret for longs, -ret for shorts
    mfe        REAL,      -- best signed excursion within the window
    mae        REAL,      -- worst signed excursion within the window
    attempts   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (alert_id, horizon)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_due ON outcomes(status, due_at);
"""


class AlertStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._con()) as con, con:
            con.executescript(_SCHEMA)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def insert(self, row: dict) -> tuple[int, bool]:
        """Insert an alert + pending outcomes. Returns (id, is_new)."""
        with closing(self._con()) as con, con:
            cur = con.execute(
                """INSERT OR IGNORE INTO alerts (received_at, alert_time, time_from_tv, ticker,
                   forecaster_ticker, price_symbol, price, strategy, side, payload, dedupe_key,
                   verdict, reasons, checks) VALUES (:received_at, :alert_time, :time_from_tv, :ticker,
                   :forecaster_ticker, :price_symbol, :price, :strategy, :side, :payload, :dedupe_key,
                   :verdict, :reasons, :checks)""", row)
            if cur.rowcount == 0:
                rid = con.execute("SELECT id FROM alerts WHERE dedupe_key=?", (row["dedupe_key"],)).fetchone()[0]
                return rid, False
            rid = cur.lastrowid
            t = pd.Timestamp(row["alert_time"])
            con.executemany("INSERT INTO outcomes (alert_id, horizon, due_at) VALUES (?,?,?)",
                            [(rid, h, (t + d).isoformat()) for h, d in HORIZONS.items()])
            return rid, True

    def get(self, alert_id: int) -> dict:
        with closing(self._con()) as con:
            return dict(con.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone())

    def set_telegram(self, alert_id: int, status: str, error: str | None = None) -> None:
        with closing(self._con()) as con, con:
            con.execute("UPDATE alerts SET telegram_status=?, telegram_error=? WHERE id=?",
                        (status, error, alert_id))

    def due_outcomes(self, now: pd.Timestamp) -> list[dict]:
        with closing(self._con()) as con:
            rows = con.execute(
                """SELECT o.*, a.alert_time, a.price_symbol, a.side FROM outcomes o
                   JOIN alerts a ON a.id = o.alert_id
                   WHERE o.status='pending' AND o.due_at <= ? ORDER BY o.due_at""",
                (now.isoformat(),)).fetchall()
        return [dict(r) for r in rows]

    def update_outcome(self, alert_id: int, horizon: str, **values) -> None:
        cols = ", ".join(f"{k}=:{k}" for k in values)
        with closing(self._con()) as con, con:
            con.execute(f"UPDATE outcomes SET {cols}, attempts=attempts+1 "
                        "WHERE alert_id=:alert_id AND horizon=:horizon",
                        {**values, "alert_id": alert_id, "horizon": horizon})

    def frame(self) -> pd.DataFrame:
        with closing(self._con()) as con:
            return pd.read_sql("""SELECT a.id, a.alert_time, a.ticker, a.strategy, a.side, a.verdict,
                                  o.horizon, o.status, o.ret, o.signed_ret, o.mfe, o.mae
                                  FROM alerts a LEFT JOIN outcomes o ON o.alert_id = a.id""", con)


def outcome_stats(store: AlertStore) -> pd.DataFrame:
    """Signed return by verdict and horizon: do SKIPs really do worse than TAKEs?"""
    f = store.frame()
    f = f[f["status"] == "done"]
    if f.empty:
        return pd.DataFrame(columns=["verdict", "horizon", "n", "mean_signed_ret", "win_rate", "mean_mae"])
    g = f.groupby(["verdict", "horizon"])
    out = pd.DataFrame({"n": g.size(), "mean_signed_ret": g["signed_ret"].mean(),
                        "win_rate": g["signed_ret"].apply(lambda s: (s > 0).mean()),
                        "mean_mae": g["mae"].mean()}).reset_index()
    order = {"TAKE": 0, "CAUTION": 1, "SKIP": 2}
    hz = {h: i for i, h in enumerate(HORIZONS)}
    return out.sort_values(["verdict", "horizon"], key=lambda s: s.map(order if s.name == "verdict" else hz))


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #

PriceFetcher = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]


def yf_intraday(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:  # pragma: no cover
    """5-minute closes (yfinance keeps ~60 days of them), UTC index."""
    import yfinance as yf

    df = yf.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"),
                                   end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), interval="5m")
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].dropna()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    return s


def resolve_outcomes(store: AlertStore, fetch: PriceFetcher, now: pd.Timestamp,
                     grace: pd.Timedelta = pd.Timedelta(minutes=20),
                     give_up_after: pd.Timedelta = pd.Timedelta(days=5)) -> int:
    """Fill outcomes whose horizon has passed. Returns how many were completed.

    Reference = last price-source close at or before the alert time (not the
    TradingView price, which may be spot while the source is futures). Horizon
    price = last close at or before alert + horizon; over a weekend that's
    correctly the last close before the market shut.
    """
    done = 0
    for o in store.due_outcomes(now - grace):
        t, due = pd.Timestamp(o["alert_time"]), pd.Timestamp(o["due_at"])
        try:
            px = fetch(o["price_symbol"], t - pd.Timedelta(days=4), due)
        except Exception as exc:  # network etc.: retry next pass
            log.warning("outcome fetch failed for %s: %s", o["price_symbol"], exc)
            px = pd.Series(dtype=float)
        px = px.dropna().sort_index()
        ref, at = px.loc[:t], px.loc[:due]
        if ref.empty or at.empty:
            status = "unavailable" if now - due > give_up_after else "pending"
            store.update_outcome(o["alert_id"], o["horizon"], status=status)
            continue
        sign = 1.0 if o["side"] == "long" else -1.0
        r0 = float(ref.iloc[-1])
        window = px.loc[(px.index > t) & (px.index <= due)]
        path = sign * (window / r0 - 1) if len(window) else pd.Series([0.0])
        ret = float(at.iloc[-1] / r0 - 1)
        store.update_outcome(o["alert_id"], o["horizon"], status="done", ref_price=r0,
                             price_at=float(at.iloc[-1]), price_time=at.index[-1].isoformat(),
                             ret=ret, signed_ret=sign * ret, mfe=float(path.max()), mae=float(path.min()))
        done += 1
    return done


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #

class TelegramNotifier:
    def __init__(self, token: str | None, chat_id: str | None, post: Callable | None = None) -> None:
        self.token, self.chat_id = token, chat_id
        self._post = post

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> tuple[bool, str | None]:
        if not self.configured:
            return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
        post = self._post
        if post is None:  # pragma: no cover
            import httpx
            post = httpx.post
        try:
            r = post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                     json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                           "disable_web_page_preview": True}, timeout=10)
            if getattr(r, "status_code", 200) >= 400:
                desc = ""
                try:
                    desc = r.json().get("description", "")
                except Exception:
                    pass
                return False, f"Telegram HTTP {r.status_code} {desc}".strip()
            return True, None
        except Exception as exc:
            # str(exc) can include the request URL, which contains the token
            return False, f"Telegram request failed: {type(exc).__name__}"


ICONS = {"TAKE": "✅", "CAUTION": "⚠️", "SKIP": "⛔"}


def price_text(symbol: str, price: float, usd_cad: float | None) -> str:
    """Instrument price in its own currency, plus the CAD equivalent for USD instruments.

    Orders stay in the instrument's quote currency (XAU_USD is priced in USD), so
    the native price is never replaced, only annotated.
    """
    from forecaster.market_data import ticker_currency

    cur = ticker_currency(symbol)
    if cur == "CAD":
        return f"C${price:,.2f}"
    if cur is None:
        return f"{price:,.2f} pts"
    if usd_cad:
        return f"{price:,.2f} USD (≈ C${price * usd_cad:,.2f})"
    return f"{price:,.2f} USD (no USD/CAD rate cached)"


def format_message(alert: Alert, t: pd.Timestamp, verdict: str, reasons: list[str], checks: list[Check],
                   usd_cad: float | None = None) -> str:
    e = html.escape
    lines = [f"{ICONS[verdict]} <b>{verdict}</b>: {e(alert.side.upper())} {e(alert.ticker)} @ "
             f"{e(price_text(alert.symbol, alert.price, usd_cad))}",
             f"<i>{e(alert.strategy)}</i> · {t:%Y-%m-%d %H:%M} UTC", ""]
    for c in checks:
        mark = {"ok": "•", "caution": "⚠️", "skip": "⛔", "unknown": "❔"}[c.level]
        lines.append(f"{mark} {e(c.name)}: {e(c.reason)}")
    lines += ["", "<i>Risk filters, not a signal. The forecaster doesn't beat a naive baseline.</i>"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

def _client_ip(request: Request, trust_proxy: bool) -> str:
    """Real client IP. Behind Cloudflare Tunnel the connection comes from cloudflared,
    so the visitor's address is in CF-Connecting-IP (only trusted with --trust-proxy,
    i.e. when the server is reachable through the tunnel alone)."""
    if trust_proxy:
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.strip()
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


class RateLimiter:
    """Sliding-window limits per client IP and in total. In-memory, single process."""

    def __init__(self, per_ip: int = 30, total: int = 120, window_s: float = 60.0) -> None:
        self.per_ip, self.total, self.window_s = per_ip, total, window_s
        self._hits: dict[str, list[float]] = {}
        self._all: list[float] = []

    def allow(self, key: str, now: float) -> bool:
        cut = now - self.window_s
        self._all = [t for t in self._all if t > cut]
        mine = [t for t in self._hits.get(key, []) if t > cut]
        if len(mine) >= self.per_ip or len(self._all) >= self.total:
            self._hits[key] = mine
            return False
        mine.append(now)
        self._hits[key] = mine
        self._all.append(now)
        if len(self._hits) > 10_000:           # bound memory under a flood of distinct IPs
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > cut}
        return True


def create_app(settings: Settings | None = None, store: AlertStore | None = None,
               ctx: Context | None = None, notifier: TelegramNotifier | None = None,
               cfg: RuleConfig = RuleConfig(), clock: Callable[[], pd.Timestamp] | None = None,
               price_fetcher: PriceFetcher | None = None, resolve_every_s: float | None = 300.0,
               trust_proxy: bool = False, executor: Any = None,
               rate_limiter: RateLimiter | None = None) -> FastAPI:
    """`executor`: optional forecaster.execution.Executor. It gets every new alert and
    places an order only for TAKE verdicts that pass its own hard limits."""
    settings = settings or load_settings()
    store = store or AlertStore(Path(settings.cache_dir) / "alerts.sqlite")
    ctx = ctx or Context(settings)
    notifier = notifier or TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    fetch = price_fetcher or yf_intraday
    limiter = rate_limiter or RateLimiter(int(os.environ.get("WEBHOOK_RATE_PER_IP", 30)),
                                          int(os.environ.get("WEBHOOK_RATE_TOTAL", 120)))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = None
        if resolve_every_s:
            async def loop() -> None:
                while True:
                    try:
                        n = await asyncio.to_thread(resolve_outcomes, store, fetch, clock())
                        if n:
                            log.info("resolved %d outcomes", n)
                    except Exception:
                        log.exception("outcome resolver failed")
                    await asyncio.sleep(resolve_every_s)
            task = asyncio.create_task(loop())
        yield
        if task:
            task.cancel()

    app = FastAPI(title="quarterly-forecaster webhook", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store, app.state.ctx, app.state.notifier = store, ctx, notifier

    def notify(alert_id: int, text: str) -> None:
        ok, err = notifier.send(text)
        store.set_telegram(alert_id, "sent" if ok else "error", err)
        if not ok:
            log.warning("telegram not sent for alert %s: %s", alert_id, err)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/tv-alert")
    async def tv_alert(request: Request, background: BackgroundTasks) -> dict:
        ip = _client_ip(request, trust_proxy)
        # rate limit before anything else, so the secret can't be brute-forced quickly
        if not limiter.allow(ip or "unknown", clock().timestamp()):
            raise HTTPException(429, "too many requests")
        if not settings.webhook_secret:
            raise HTTPException(503, "WEBHOOK_SECRET is not configured")
        if settings.webhook_allowed_ips and ip not in settings.webhook_allowed_ips:
            raise HTTPException(403, "source IP not allowed")
        if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
            raise HTTPException(413, "body too large")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(413, "body too large")
        try:
            raw = json.loads(body)          # TradingView may send JSON as text/plain
            if not isinstance(raw, dict):
                raise ValueError
        except ValueError:
            raise HTTPException(400, "body must be a JSON object") from None
        given = raw.get("secret")
        if not isinstance(given, str) or not hmac.compare_digest(given.encode(), settings.webhook_secret.encode()):
            raise HTTPException(401, "bad secret")
        try:
            alert = Alert.model_validate(raw)
        except ValidationError as exc:
            errs = [{"field": ".".join(map(str, e["loc"])), "msg": e["msg"]} for e in exc.errors()]
            raise HTTPException(422, errs) from None

        received = clock()
        t, from_tv = parse_alert_time(alert.time, received)
        verdict, reasons, checks = evaluate_alert(alert, t, ctx, cfg)
        fticker, psym, _ = resolve_symbol(alert.symbol)
        payload = {k: v for k, v in raw.items() if k != "secret"}
        row = {
            "received_at": received.isoformat(), "alert_time": t.isoformat(), "time_from_tv": int(from_tv),
            "ticker": alert.ticker, "forecaster_ticker": fticker, "price_symbol": psym, "price": alert.price,
            "strategy": alert.strategy, "side": alert.side, "payload": json.dumps(payload),
            "dedupe_key": "|".join([alert.symbol, alert.strategy, alert.side,
                                    t.isoformat() if from_tv else f"{alert.price}@{received.floor('min').isoformat()}"]),
            "verdict": verdict, "reasons": json.dumps(reasons), "checks": json.dumps([asdict(c) for c in checks]),
        }
        alert_id, is_new = store.insert(row)
        if not is_new:
            prev = store.get(alert_id)
            return {"id": alert_id, "duplicate": True, "verdict": prev["verdict"],
                    "reasons": json.loads(prev["reasons"])}
        background.add_task(notify, alert_id, format_message(alert, t, verdict, reasons, checks, ctx.usd_cad()))
        if executor is not None:
            from forecaster.execution import request_from_alert

            background.add_task(executor.handle, request_from_alert(alert_id, alert, t, verdict))
        return {"id": alert_id, "duplicate": False, "verdict": verdict, "reasons": reasons,
                "checks": [asdict(c) for c in checks]}

    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="TradingView webhook server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--trust-proxy", action="store_true", help="use X-Forwarded-For for the IP allowlist")
    ap.add_argument("--resolve-outcomes", action="store_true", help="fill due outcomes once and exit")
    ap.add_argument("--stats", action="store_true", help="print outcome stats by verdict and exit")
    args = ap.parse_args(argv)
    from forecaster.config import PROJECT_ROOT
    from forecaster.monitor import setup_logging

    # rotating file logs; also silences httpx INFO lines, which would print the Telegram token URL
    setup_logging(PROJECT_ROOT / "logs", "webhook")

    settings = load_settings()
    store = AlertStore(Path(settings.cache_dir) / "alerts.sqlite")
    if args.resolve_outcomes:
        print(f"resolved {resolve_outcomes(store, yf_intraday, pd.Timestamp.now(tz='UTC'))} outcomes")
        return 0
    if args.stats:
        st = outcome_stats(store)
        print(st.to_string(index=False) if len(st) else "no resolved outcomes yet")
        return 0
    import uvicorn

    from forecaster.execution import Executor, OrderStore, load_execution_config

    xcfg = load_execution_config()          # raises if live is requested without LIVE_TRADING=true in .env
    executor = None
    if xcfg.enabled:
        notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        executor = Executor(xcfg, OrderStore(Path(settings.cache_dir) / "alerts.sqlite"),
                            notify=lambda text: notifier.send(html.escape(text)))
        log.warning("order execution ENABLED on OANDA %s", xcfg.environment.upper())
    # log_config=None: uvicorn's own loggers propagate to the root handlers (logs/webhook.log + stdout)
    uvicorn.run(create_app(settings, store, trust_proxy=args.trust_proxy, executor=executor),
                host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
