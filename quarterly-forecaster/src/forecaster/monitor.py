"""24/7 monitor: scheduled checks -> Telegram alerts (each sent once), plus /status.

    python -m forecaster.monitor          # what the `monitor` Docker service runs

Schedule (APScheduler, times in New York):
  * tick every 15 min; it does work every 15 min during US market hours
    (Mon-Fri 09:30-16:00 ET) and hourly otherwise.
  * daily data refresh Mon-Fri 06:15 (pipeline: prices, macro, events, news,
    forecasts, portfolio); Saturday 06:15 also re-runs backtests older than 28 days.
  * weekly report Sunday 07:23.
  * Telegram long-polling for /status and /help (only your TELEGRAM_CHAT_ID).

Checks on each tick (alert text always says why):
  * events: CPI, PPI, jobs and FOMC (high impact), 24 h and 1 h before.
  * VIX spike: latest VIX >= 20% above the previous daily close.
  * gold vs session VWAP: last 5-minute GC=F close beyond 2 standard deviations
    of the volume-weighted session (resets 18:00 New York).
  * regime change: argmax of the HMM regime in a forecast snapshot differs from
    the last one seen (updates with the daily refresh).
  * news shocks: |z| > 2 sentiment days for the market, topics or tickers, today or yesterday.
Prices refreshed each tick: SPY, QQQ, gold (GC=F), VIX, DXY (DX-Y.NYB), US10Y (^TNX).

No duplicates: every alert has a key (e.g. "event:2026-11-10:CPI:24h"), stored in
data/monitor.sqlite; a key is sent at most once, across restarts too.
"""

from __future__ import annotations

import html
import json
import logging
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger("forecaster.monitor")

NY = "America/New_York"
MONITORED: dict[str, str] = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "GC=F": "Gold",
                             "^VIX": "VIX", "DX-Y.NYB": "Dollar index (DXY)", "^TNX": "US 10Y yield"}
EVENT_CATEGORIES = ("CPI", "PPI", "FOMC", "Jobs")
REGIME_NAMES = {"bull": "bull", "bear": "bear", "high_vol": "high volatility"}


@dataclass(frozen=True)
class MonitorConfig:
    vix_spike: float = 0.20
    gold_sigma: float = 2.0
    gold_min_bars: int = 12            # 1 hour of 5-minute bars before the bands mean anything
    market_every_min: int = 15
    off_hours_every_min: int = 60
    shock_days: int = 1                # today and yesterday


@dataclass(frozen=True)
class Alert:
    key: str
    text: str


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class MonitorState:
    """Sent-alert keys + small key/value store (last run, Telegram offset, regimes)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as con, con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS sent (key TEXT PRIMARY KEY, sent_at TEXT NOT NULL, text TEXT);
                CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);""")

    def _con(self):
        return sqlite3.connect(self.path, timeout=10)

    def claim(self, key: str, text: str) -> bool:
        """Atomically mark `key` sent. False if it already was (-> don't send again)."""
        with closing(self._con()) as con, con:
            cur = con.execute("INSERT OR IGNORE INTO sent VALUES (?,?,?)",
                              (key, pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), text))
            return cur.rowcount == 1

    def unclaim(self, key: str) -> None:
        with closing(self._con()) as con, con:
            con.execute("DELETE FROM sent WHERE key=?", (key,))

    def was_sent(self, key: str) -> bool:
        with closing(self._con()) as con:
            return con.execute("SELECT 1 FROM sent WHERE key=?", (key,)).fetchone() is not None

    def get(self, k: str, default: str | None = None) -> str | None:
        with closing(self._con()) as con:
            r = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
            return r[0] if r else default

    def set(self, k: str, v: str) -> None:
        with closing(self._con()) as con, con:
            con.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, v))


# --------------------------------------------------------------------------- #
# Schedule gating
# --------------------------------------------------------------------------- #

def market_open(now: pd.Timestamp) -> bool:
    """US equity regular hours, Mon-Fri 09:30-16:00 New York (exchange holidays not excluded)."""
    t = now.tz_convert(NY)
    m = t.hour * 60 + t.minute
    return t.day_of_week < 5 and 9 * 60 + 30 <= m < 16 * 60


def due(now: pd.Timestamp, last_run: pd.Timestamp | None, cfg: MonitorConfig = MonitorConfig()) -> bool:
    if last_run is None:
        return True
    every = cfg.market_every_min if market_open(now) else cfg.off_hours_every_min
    return (now - last_run) >= pd.Timedelta(minutes=every) - pd.Timedelta(seconds=30)


# --------------------------------------------------------------------------- #
# Checks (pure: data in, alerts out)
# --------------------------------------------------------------------------- #

def check_events(events: list[dict], now: pd.Timestamp) -> list[Alert]:
    """24 h warning when 1 h < time-to-event <= 24 h; 1 h warning when 0 < time-to-event <= 1 h."""
    from forecaster.webhook_server import event_time

    out = []
    for e in events or []:
        if e.get("impact") != "high" or e.get("category") not in EVENT_CATEGORIES:
            continue
        t = event_time(e)
        left = (t - now).total_seconds() / 3600
        lead = "1h" if 0 < left <= 1 else ("24h" if 1 < left <= 24 else None)
        if lead is None:
            continue
        when = t.tz_convert(NY)
        detail = f" ({e['detail']})" if e.get("detail") else ""
        out.append(Alert(f"event:{pd.Timestamp(e['date']):%Y-%m-%d}:{e['category']}:{lead}",
                         f"📅 {e['name']}{detail} in ~{max(left, 0):.0f}h: {when:%a %b %d %H:%M} ET. "
                         "High-impact release: expect volatility; consider waiting before new entries."))
    return out


def check_vix_spike(daily_vix: pd.Series | None, latest: float | None, latest_time: pd.Timestamp | None,
                    cfg: MonitorConfig = MonitorConfig()) -> list[Alert]:
    if daily_vix is None or latest is None or latest_time is None:
        return []
    d = daily_vix.dropna()
    today = latest_time.tz_convert(NY).tz_localize(None).normalize()
    prev = d[d.index < today]
    if prev.empty:
        return []
    p0 = float(prev.iloc[-1])
    chg = latest / p0 - 1
    if chg < cfg.vix_spike:
        return []
    return [Alert(f"vix:{today:%Y-%m-%d}",
                  f"⚡ VIX spike: {latest:.1f}, {chg:+.0%} vs the previous close {p0:.1f}. "
                  "Markets are pricing more volatility.")]


def session_vwap(bars: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP and volume-weighted st. dev. (session resets 18:00 New York)."""
    df = bars.dropna(subset=["Close"]).copy()
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df["session"] = (idx + pd.Timedelta(hours=6)).tz_convert(NY).strftime("%Y-%m-%d")
    src = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].fillna(0).clip(lower=0)
    g = df["session"]
    sv, spv, sp2v = v.groupby(g).cumsum(), (src * v).groupby(g).cumsum(), (src * src * v).groupby(g).cumsum()
    df["vwap"] = np.where(sv > 0, spv / sv.where(sv > 0), src)
    df["sd"] = np.sqrt(np.maximum(np.where(sv > 0, sp2v / sv.where(sv > 0) - df["vwap"] ** 2, 0.0), 0.0))
    df["bars_in_session"] = g.groupby(g).cumcount() + 1
    return df


def check_gold_vwap(bars: pd.DataFrame | None, cfg: MonitorConfig = MonitorConfig()) -> list[Alert]:
    if bars is None or bars.empty:
        return []
    df = session_vwap(bars)
    last = df.iloc[-1]
    if last["bars_in_session"] < cfg.gold_min_bars or not last["sd"] > 0:
        return []
    z = (last["Close"] - last["vwap"]) / last["sd"]
    if abs(z) <= cfg.gold_sigma:
        return []
    side = "above" if z > 0 else "below"
    return [Alert(f"gold_vwap:{last['session']}:{side}",
                  f"🥇 Gold {last['Close']:,.2f} USD is {abs(z):.1f} st. dev. {side} today's session VWAP "
                  f"({last['vwap']:,.2f}): stretched move; VWAP setups may be unreliable.")]


def regime_label(snap: dict) -> str | None:
    cur = (snap.get("regime") or {}).get("current") or {}
    return max(cur, key=cur.get) if cur else None


def check_regimes(snaps: dict[str, dict], state: MonitorState) -> list[Alert]:
    out = []
    for t, s in sorted(snaps.items()):
        new = regime_label(s)
        if new is None:
            continue
        k = f"regime:{t}"
        old = state.get(k)
        state.set(k, new)
        if old is None or old == new:
            continue
        p = s["regime"]["current"][new]
        out.append(Alert(f"regime:{t}:{s.get('as_of')}:{new}",
                         f"🔄 {t} regime changed: {REGIME_NAMES.get(old, old)} -> {REGIME_NAMES.get(new, new)} "
                         f"({p:.0%} probability, HMM as of {str(s.get('as_of'))[:10]})."))
    return out


def check_shocks(shocks: list[dict], now: pd.Timestamp, cfg: MonitorConfig = MonitorConfig()) -> list[Alert]:
    today = now.tz_convert(NY).tz_localize(None).normalize()
    out = []
    for s in shocks or []:
        d = pd.Timestamp(s["date"]).normalize()
        if (today - d).days > cfg.shock_days:
            continue
        tone = "negative" if s["z"] < 0 else "positive"
        tag = "market news" if s["tag"] == "MARKET" else s["tag"]
        out.append(Alert(f"shock:{s['tag']}:{d:%Y-%m-%d}",
                         f"📰 News shock ({tone}) in {tag} on {d:%b %d}: sentiment {s['sentiment']:+.2f}, "
                         f"z {s['z']:+.1f} over {s.get('headlines', '?')} headlines. ~5% of quiet days trigger by chance."))
    return out


# --------------------------------------------------------------------------- #
# /status
# --------------------------------------------------------------------------- #

def status_text(now: pd.Timestamp, snaps: dict[str, dict], market: dict | None, portfolio: dict | None,
                vix_latest: tuple[float, pd.Timestamp] | None) -> str:
    from forecaster.webhook_server import event_time

    e = html.escape
    lines = [f"<b>Status</b> · {now.tz_convert(NY):%a %b %d %H:%M} ET"]
    spy = snaps.get("SPY") or next(iter(snaps.values()), None)
    if spy and regime_label(spy):
        r = regime_label(spy)
        lines.append(f"Regime ({e(spy['ticker'])}): {REGIME_NAMES.get(r, r)} "
                     f"{spy['regime']['current'][r]:.0%} (as of {str(spy.get('as_of'))[:10]})")
    else:
        lines.append("Regime: no forecast snapshot yet")
    if vix_latest:
        lines.append(f"VIX: {vix_latest[0]:.1f} ({vix_latest[1].tz_convert(NY):%b %d %H:%M} ET)")
    upcoming = sorted(((event_time(ev), ev) for ev in (market or {}).get("events", [])
                       if ev.get("impact") == "high" and event_time(ev) > now), key=lambda x: x[0])
    if upcoming:
        t, ev = upcoming[0]
        hrs = (t - now).total_seconds() / 3600
        lines.append(f"Next big event: {e(ev['name'])} {t.tz_convert(NY):%a %b %d %H:%M} ET "
                     f"(in {hrs:.0f}h)" if hrs < 72 else
                     f"Next big event: {e(ev['name'])} {t.tz_convert(NY):%a %b %d %H:%M} ET (in {hrs / 24:.0f} days)")
    else:
        lines.append("Next big event: none in the events snapshot")
    if spy and spy.get("forecast"):
        f = spy["forecast"][0]
        cur = "C$" if spy.get("currency") == "CAD" else ""
        lines.append(f"Forecast {e(spy['ticker'])} {f['quarter']}: P10 {cur}{f['p10']:,.0f} · "
                     f"P50 {cur}{f['p50']:,.0f} · P90 {cur}{f['p90']:,.0f}")
    if portfolio and portfolio.get("forecast"):
        f = portfolio["forecast"][0]
        lines.append(f"Portfolio C${portfolio['total_value']:,.0f}; {f['quarter']} P10 C${f['p10']:,.0f} · "
                     f"P50 C${f['p50']:,.0f} · P90 C${f['p90']:,.0f}")
    lines.append("<i>Ranges, not predictions; the ensemble doesn't beat a naive baseline.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Telegram bot (long polling: no inbound port, nothing public)
# --------------------------------------------------------------------------- #

class TelegramBot:
    """Answers /status and /help in TELEGRAM_CHAT_ID only; everything else is ignored."""

    def __init__(self, token: str | None, chat_id: str | None, state: MonitorState,
                 status: Callable[[], str], http=None) -> None:
        import httpx

        self.token, self.chat_id, self.state, self.status = token, str(chat_id or ""), state, status
        self._http = http or httpx.Client(timeout=70)

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send(self, text: str) -> bool:
        if not self.configured:
            return False
        try:
            r = self._http.post(self._url("sendMessage"), json={
                "chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True})
            return r.status_code < 400
        except Exception as exc:  # the URL holds the token: log the type only
            log.warning("telegram send failed: %s", type(exc).__name__)
            return False

    def poll_once(self, timeout: int = 50) -> int:
        """One getUpdates long-poll. Returns how many commands were answered."""
        if not self.configured:
            return 0
        offset = int(self.state.get("tg_offset", "0"))
        try:
            r = self._http.get(self._url("getUpdates"), params={"offset": offset, "timeout": timeout,
                                                                "allowed_updates": json.dumps(["message"])})
            updates = r.json().get("result", []) if r.status_code < 400 else []
        except Exception as exc:
            log.warning("telegram poll failed: %s", type(exc).__name__)
            return 0
        answered = 0
        for u in updates:
            self.state.set("tg_offset", str(int(u["update_id"]) + 1))
            msg = u.get("message") or {}
            chat = str((msg.get("chat") or {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if chat != self.chat_id:
                log.warning("ignored Telegram message from unknown chat %s", chat)
                continue
            cmd = text.split()[0].split("@")[0].lower() if text else ""
            if cmd == "/status":
                try:
                    reply = self.status()
                except Exception as exc:
                    log.exception("status failed")
                    reply = f"Status unavailable: {html.escape(type(exc).__name__)}"
                self.send(reply)
                answered += 1
            elif cmd in ("/help", "/start"):
                self.send("Commands: /status: regime, VIX, next big event, latest forecast.")
                answered += 1
        return answered


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #

def yf_intraday_5m(symbol: str) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf

    return yf.Ticker(symbol).history(period="5d", interval="5m")


class Monitor:
    def __init__(self, settings, state: MonitorState, send: Callable[[str], bool],
                 cfg: MonitorConfig = MonitorConfig(), clock: Callable[[], pd.Timestamp] | None = None,
                 intraday: Callable[[str], pd.DataFrame] | None = None, store=None,
                 refresh_news: Callable[[], None] | None = None) -> None:
        from forecaster.market_data import ParquetStore

        self.settings, self.state, self.send, self.cfg = settings, state, send, cfg
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.intraday = intraday or yf_intraday_5m
        self.store = store or ParquetStore(Path(settings.cache_dir) / "ohlcv")
        self.refresh_news = refresh_news
        self.snap_dir = Path(settings.cache_dir) / "snapshots"
        self._lock = threading.Lock()

    # -- data -------------------------------------------------------------- #
    def _json(self, name: str) -> dict | None:
        from forecaster.pipeline import read_json

        return read_json(self.snap_dir / name)

    def snapshots(self) -> dict[str, dict]:
        from forecaster.pipeline import read_json

        return {s["ticker"]: s for p in sorted(self.snap_dir.glob("ticker_*.json")) if (s := read_json(p))}

    def latest_vix(self) -> tuple[float, pd.Timestamp] | None:
        k = self.state.get("vix_latest")
        if not k:
            return None
        v, t = k.split("|")
        return float(v), pd.Timestamp(t)

    # -- one run ----------------------------------------------------------- #
    def deliver(self, alerts: list[Alert]) -> int:
        n = 0
        for a in alerts:
            if not self.state.claim(a.key, a.text):
                continue                                    # already sent: no duplicates
            if self.send(a.text):
                n += 1
                log.info("alert sent: %s", a.key)
            else:
                self.state.unclaim(a.key)                   # retry next tick
                log.warning("alert not delivered, will retry: %s", a.key)
        return n

    def run_checks(self, force: bool = False) -> int:
        with self._lock:
            now = self.clock()
            last = self.state.get("last_run")
            if not force and not due(now, pd.Timestamp(last) if last else None, self.cfg):
                return 0
            self.state.set("last_run", now.isoformat())
            alerts: list[Alert] = []
            for t in MONITORED:                               # daily cache: downloads only when stale
                try:
                    self.store.get(t)
                except Exception as exc:
                    log.warning("refresh %s failed: %s", t, exc)
            market = self._json("market.json")
            alerts += check_events((market or {}).get("events", []), now)
            try:
                vb = self.intraday("^VIX")
                if vb is not None and not vb.empty:
                    t_last = pd.Timestamp(vb.index[-1])
                    t_last = t_last.tz_localize("UTC") if t_last.tz is None else t_last.tz_convert("UTC")
                    v_last = float(vb["Close"].dropna().iloc[-1])
                    self.state.set("vix_latest", f"{v_last}|{t_last.isoformat()}")
                    daily = self.store.read("^VIX")
                    alerts += check_vix_spike(None if daily is None else daily["Close"], v_last, t_last, self.cfg)
            except Exception as exc:
                log.warning("VIX intraday failed: %s", exc)
            try:
                alerts += check_gold_vwap(self.intraday("GC=F"), self.cfg)
            except Exception as exc:
                log.warning("gold intraday failed: %s", exc)
            alerts += check_regimes(self.snapshots(), self.state)
            if self.refresh_news and (not market_open(now) or now.minute < 15):   # news hourly
                try:
                    self.refresh_news()
                except Exception as exc:
                    log.warning("news refresh failed: %s", exc)
            alerts += check_shocks(self.current_shocks(market), now, self.cfg)
            sent = self.deliver(alerts)
            self.heartbeat(now)
            return sent

    def current_shocks(self, market: dict | None) -> list[dict]:
        db = Path(self.settings.cache_dir) / "news.sqlite"
        if db.exists():
            try:
                from forecaster.news import NewsStore, daily_index, shocks

                s = shocks(daily_index(NewsStore(db).scored_frame()))
                return [{"tag": t, "date": d, "sentiment": r["sentiment"], "z": r["z"], "headlines": r["headlines"]}
                        for (t, d), r in s.iterrows()]
            except Exception as exc:
                log.warning("news index failed: %s", exc)
        return (market or {}).get("shocks", [])

    def status(self) -> str:
        return status_text(self.clock(), self.snapshots(), self._json("market.json"),
                           self._json("portfolio.json"), self.latest_vix())

    def heartbeat(self, now: pd.Timestamp) -> None:
        p = Path(self.settings.cache_dir) / "monitor_heartbeat"
        p.write_text(now.isoformat())


# --------------------------------------------------------------------------- #
# Process entry point
# --------------------------------------------------------------------------- #

def setup_logging(log_dir: Path, name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    for h in root.handlers:               # close replaced handlers (don't leak open log files)
        h.close()
    root.handlers[:] = [fh, sh]
    logging.getLogger("httpx").setLevel(logging.WARNING)   # its INFO lines would print the token URL


def _news_refresher(settings) -> Callable[[], None] | None:  # pragma: no cover - network
    if not settings.finnhub_api_key:
        return None

    def run() -> None:
        from forecaster.cache import Cache
        from forecaster.news import FinBertScorer, NewsStore, collect_news, score_pending
        from forecaster.sources.finnhub import FinnhubClient

        fh = FinnhubClient(settings.finnhub_api_key, Cache(Path(settings.cache_dir) / "news_http", ttl_seconds=3600))
        store = NewsStore(Path(settings.cache_dir) / "news.sqlite")
        today = pd.Timestamp.today().normalize()
        collect_news(fh, store, settings.tickers, today - pd.Timedelta(days=3), today,
                     on_error=lambda t, e: log.warning("news %s: %s", t, e))
        try:
            score_pending(store, FinBertScorer())
        except Exception as exc:
            log.info("FinBERT not available (%s); headlines collected but not scored", type(exc).__name__)
    return run


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - long-running service
    import argparse

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    from forecaster.config import PROJECT_ROOT, load_settings

    ap = argparse.ArgumentParser(description="24/7 monitor + Telegram /status")
    ap.add_argument("--once", action="store_true", help="run the checks once and exit")
    args = ap.parse_args(argv)

    settings = load_settings()
    setup_logging(PROJECT_ROOT / "logs", "monitor")
    state = MonitorState(Path(settings.cache_dir) / "monitor.sqlite")
    bot_holder: list[TelegramBot] = []
    mon = Monitor(settings, state, send=lambda text: bot_holder[0].send(text), refresh_news=_news_refresher(settings))
    bot = TelegramBot(settings.telegram_bot_token, settings.telegram_chat_id, state, mon.status)
    bot_holder.append(bot)
    if not bot.configured:
        log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set: alerts will be logged but not sent")
    if args.once:
        print(f"sent {mon.run_checks(force=True)} alerts")
        return 0

    def daily_refresh(backtests: bool = False) -> None:
        from forecaster.pipeline import refresh_all
        log.info("daily refresh start")
        status = refresh_all(settings, warn=lambda m: log.warning(m), backtest_stale_days=28 if backtests else None)
        log.info("daily refresh done: %s", status)

    def weekly_report() -> None:
        from forecaster import weekly_report as wr
        wr.main(["--no-refresh"])

    def safe(fn, *a):
        def run():
            try:
                fn(*a)
            except Exception:
                log.exception("job %s failed", getattr(fn, "__name__", fn))
        return run

    sched = BackgroundScheduler(timezone=NY, job_defaults={"coalesce": True, "max_instances": 1,
                                                          "misfire_grace_time": 900})
    sched.add_job(safe(mon.run_checks), "interval", minutes=15, next_run_time=pd.Timestamp.now(tz=NY).to_pydatetime(),
                  id="checks")
    sched.add_job(safe(daily_refresh), CronTrigger(day_of_week="mon-fri", hour=6, minute=15, timezone=NY), id="refresh")
    sched.add_job(safe(daily_refresh, True), CronTrigger(day_of_week="sat", hour=6, minute=15, timezone=NY), id="refresh_sat")
    sched.add_job(safe(weekly_report), CronTrigger(day_of_week="sun", hour=7, minute=23, timezone=NY), id="report")
    sched.start()
    log.info("monitor started (telegram %s)", "on" if bot.configured else "off")
    try:
        while True:
            if bot.configured:
                bot.poll_once()
            else:
                time.sleep(60)
    except KeyboardInterrupt:
        sched.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
