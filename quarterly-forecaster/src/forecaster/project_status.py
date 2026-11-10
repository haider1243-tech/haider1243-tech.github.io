"""Project status: build progress, data health, model health, phase gates, next step.

    python -m forecaster.project_status            # run imports + pytest, write data/project_status.json + STATUS.md
    python -m forecaster.project_status --quick    # skip pytest (imports + data only)
    python -m forecaster.project_status --send     # also send the summary to Telegram

The app's Status tab shows the same thing live. The expensive part (importing every
module and running pytest, ~1 min) runs in a subprocess when the app or launcher starts,
on the Sunday weekly job, and on demand ("Re-check now"); its result is cached in
data/project_status.json. Everything else (data freshness, backtests, journal, demo P&L)
is cheap and recomputed on every view.

Phases and their gates (checked automatically, strictly in order):
  1 Build     every module imports and the whole test suite passes
  2 Validate  the latest backtest beats the naive baseline (bootstrap CI excludes zero)
  3 Journal   30+ TradingView alerts with a resolved outcome in the journal
  4 Demo      2+ months of OANDA practice-account P&L (NAV records spanning >= 60 days)
  5 Live      all gates passed. Nothing here ever enables live trading: only you can,
              by setting LIVE_TRADING=true in .env yourself.

API keys are only reported as set / missing. Their values are never read into the
output, printed, written or sent.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from forecaster.config import PROJECT_ROOT

MODULES = ("market_data", "macro", "calendar", "news", "forecast", "backtest", "webhook_server",
           "vwap_backtest", "execution", "monitor", "portfolio", "weekly_report", "live_feed",
           "intraday", "strategies", "lab_engine", "lab_stats", "lab", "shadow")
TEST_FILES = {"webhook_server": "test_webhook", "strategies": "test_lab_engine"}          # module -> test file stem when not test_<module>
DATA_SOURCES = (("prices", "Prices"), ("macro", "FRED macro"), ("news", "News"), ("calendar", "Calendar"))
MAX_AGE = pd.Timedelta(hours=24)
# key -> (what needs it, source flag it unlocks); values are never read into the output
KEYS = {
    "FRED_API_KEY": "FRED macro data and CPI/PPI/jobs/GDP dates",
    "FINNHUB_API_KEY": "news sentiment and earnings dates",
    "TELEGRAM_BOT_TOKEN": "Telegram alerts and /status",
    "TELEGRAM_CHAT_ID": "Telegram alerts and /status",
    "WEBHOOK_SECRET": "TradingView webhook",
    "APP_PASSWORD_HASH": "dashboard login",
}
DEMO_KEYS = {"OANDA_API_TOKEN": "OANDA practice orders (Phase 4)",
             "OANDA_ACCOUNT_ID": "OANDA practice orders (Phase 4)"}
SOURCE_KEY = {"macro": "FRED_API_KEY", "calendar": "FRED_API_KEY", "news": "FINNHUB_API_KEY"}
MIN_JOURNAL = 30
MIN_DEMO_DAYS = 60
PHASES = ("Build", "Validate", "Journal", "Demo", "Live")
_SCRUB = ("FORECASTER_", "TELEGRAM_", "OANDA_")          # env prefixes kept away from the test run


def status_json(cache_dir: Path) -> Path:
    return Path(cache_dir) / "project_status.json"


STATUS_MD = Path(os.environ.get("FORECASTER_STATUS_MD") or PROJECT_ROOT / "STATUS.md")   # Docker: data/STATUS.md


# --------------------------------------------------------------------------- #
# 1. Build progress (subprocess: imports + pytest)
# --------------------------------------------------------------------------- #

@dataclass
class TestRun:
    ran_at: str
    returncode: int
    passed: int
    failed: int
    errors: int
    skipped: int
    seconds: float
    per_file: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)      # "test_x.py::test_name"
    note: str = ""


def parse_junit(xml_text: str) -> tuple[dict[str, dict[str, int]], list[str]]:
    per_file: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    root = ET.fromstring(xml_text)
    for tc in root.iter("testcase"):
        cls = tc.get("classname", "")
        stem = next((p for p in cls.split(".") if p.startswith("test_")), cls.split(".")[-1])
        d = per_file.setdefault(stem, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0})
        if tc.find("failure") is not None:
            d["failed"] += 1
            failures.append(f"{stem}.py::{tc.get('name')}")
        elif tc.find("error") is not None:
            d["errors"] += 1
            failures.append(f"{stem}.py::{tc.get('name')} (error)")
        elif tc.find("skipped") is not None:
            d["skipped"] += 1
        else:
            d["passed"] += 1
    return per_file, failures


def _child_env(root: Path) -> dict[str, str]:
    """The test run must not see real keys, paths or caches (tests use fakes and tmp dirs)."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(_SCRUB) and k not in KEYS and k not in DEMO_KEYS
           and k not in ("LIVE_TRADING", "EXECUTION_ENABLED", "KILL_SWITCH", "PYTEST_CURRENT_TEST")}
    env["FORECASTER_STATUS_CHILD"] = "1"                 # the app won't start another check from inside
    env["PYTHONPATH"] = str(root / "src")
    return env


def run_pytest(root: Path = PROJECT_ROOT, timeout: float = 1800) -> TestRun:  # pragma: no cover - spawns pytest
    now = pd.Timestamp.now(tz="UTC").isoformat()
    if not (root / "tests").is_dir():
        return TestRun(now, -1, 0, 0, 0, 0, 0.0, note="tests/ folder not found, tests not run")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider",
                 f"--junitxml={xml}", "tests"],
                cwd=root, env=_child_env(root), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return TestRun(now, -1, 0, 0, 0, 0, timeout, note=f"pytest timed out after {timeout:.0f}s")
        secs = time.time() - t0
        if not xml.exists():
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
            return TestRun(now, proc.returncode, 0, 0, 1, 0, secs, note="pytest didn't run: " + " | ".join(tail))
        per_file, failures = parse_junit(xml.read_text())
    tot = {k: sum(d[k] for d in per_file.values()) for k in ("passed", "failed", "errors", "skipped")}
    return TestRun(now, proc.returncode, tot["passed"], tot["failed"], tot["errors"], tot["skipped"], secs,
                   per_file, failures[:50])


def check_imports(modules=MODULES) -> dict[str, str | None]:  # pragma: no cover - heavy imports
    """module -> None if it imports, else the error (one line)."""
    import importlib

    out = {}
    for m in modules:
        try:
            importlib.import_module(f"forecaster.{m}")
            out[m] = None
        except Exception as exc:                          # missing file, syntax error, missing dependency...
            out[m] = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
    return out


@dataclass
class ModuleCheck:
    name: str
    state: str          # built | failing | missing | unchecked
    detail: str


ICON = {"built": "✅", "failing": "⚠️", "missing": "❌", "unchecked": "⏳"}


def module_checks(imports: dict[str, str | None] | None, tests: dict | None,
                  root: Path = PROJECT_ROOT) -> list[ModuleCheck]:
    per_file = (tests or {}).get("per_file") or {}
    ran = bool(tests) and not (tests or {}).get("note")
    out = []
    for m in MODULES:
        if not (root / "src" / "forecaster" / f"{m}.py").exists():
            out.append(ModuleCheck(m, "missing", "file not found"))
            continue
        if imports is None or m not in imports:
            out.append(ModuleCheck(m, "unchecked", "not checked yet"))
            continue
        if imports[m]:
            out.append(ModuleCheck(m, "missing", f"doesn't import: {imports[m]}"))
            continue
        stem = TEST_FILES.get(m, f"test_{m}")
        d = per_file.get(stem)
        if not ran:
            out.append(ModuleCheck(m, "built", "imports OK; tests not run"))
        elif d is None:
            out.append(ModuleCheck(m, "failing", f"no tests found ({stem}.py)"))
        elif d["failed"] or d["errors"]:
            out.append(ModuleCheck(m, "failing", f"{d['failed'] + d['errors']} of "
                                                 f"{d['passed'] + d['failed'] + d['errors']} tests failing"))
        else:
            out.append(ModuleCheck(m, "built", f"{d['passed']} tests pass"))
    return out


# --------------------------------------------------------------------------- #
# 2. Data health + keys
# --------------------------------------------------------------------------- #

def key_presence(root: Path = PROJECT_ROOT, environ: dict | None = None) -> dict[str, bool]:
    """KEY -> set? (from .env or the process environment). Values never leave this function."""
    from dotenv import dotenv_values

    env_file = root / ".env"
    file_vals = dotenv_values(env_file) if env_file.exists() else {}
    environ = os.environ if environ is None else environ
    return {k: bool((file_vals.get(k) or "").strip() or (environ.get(k) or "").strip())
            for k in (*KEYS, *DEMO_KEYS)}


@dataclass
class SourceHealth:
    source: str
    label: str
    last_success: str | None
    age_hours: float | None
    error: str | None
    ok: bool
    message: str


def data_health(cache_dir: Path, keys: dict[str, bool], now: pd.Timestamp) -> list[SourceHealth]:
    from forecaster import status

    data = status.load(cache_dir)
    out = []
    for src, label in DATA_SOURCES:
        s = data.get(src) or {}
        last = s.get("last_success")
        age = (now - pd.Timestamp(last)).total_seconds() / 3600 if last else None
        need = SOURCE_KEY.get(src)
        if need and not keys.get(need):
            msg, ok = f"{need} missing from .env", False
        elif last is None:
            msg, ok = ("refresh failed: " + s["error"]) if s.get("error") else "never refreshed", False
        elif s.get("error"):
            msg, ok = f"last refresh failed: {s['error']}", False
        elif age > MAX_AGE.total_seconds() / 3600:
            msg, ok = f"{age:.0f} h old (limit 24 h)", False
        else:
            msg, ok = "fresh", True
        out.append(SourceHealth(src, label, last, None if age is None else round(age, 2), s.get("error"), ok, msg))
    return out


# --------------------------------------------------------------------------- #
# 3. Model health
# --------------------------------------------------------------------------- #

@dataclass
class BacktestHealth:
    ticker: str
    coverage: float
    hit_rate: float
    skill: float | None
    beats_baseline: bool
    run_at: str | None
    reference: bool
    period: str


def _bt(d: dict, reference: bool, mtime: float | None) -> BacktestHealth:
    m = d["model"]
    run = d.get("generated_at") or (pd.Timestamp(mtime, unit="s", tz="UTC").isoformat() if mtime else None)
    period = f"{str(d.get('first_target', ''))[:4]}–{str(d.get('last_target', ''))[:4]}"
    return BacktestHealth(d.get("ticker", "?"), float(m["coverage"]), float(m["hit_rate"]),
                          d.get("skill"), bool(d.get("beats_baseline")), run, reference, period)


def model_health(cache_dir: Path, reference: Path | None = None) -> list[BacktestHealth]:
    """Own backtests (newest first), then the committed reference run."""
    from forecaster.pipeline import REFERENCE_BACKTEST

    reference = reference or REFERENCE_BACKTEST
    own = []
    for p in sorted((Path(cache_dir) / "snapshots").glob("backtest_*.json")):
        try:
            d = json.loads(p.read_text())
            own.append(_bt(d, False, p.stat().st_mtime))
        except (ValueError, KeyError, OSError):
            continue
    own.sort(key=lambda b: b.run_at or "", reverse=True)
    if reference.exists():
        try:
            own.append(_bt(json.loads(reference.read_text()), True, reference.stat().st_mtime))
        except (ValueError, KeyError):
            pass
    return own


def primary_backtest(bts: list[BacktestHealth]) -> BacktestHealth | None:
    """The one the Validate gate uses: your own S&P 500 run if there is one, else the newest own run,
    else the committed reference."""
    for want in ("SPY", "^GSPC"):
        for b in bts:
            if not b.reference and b.ticker == want:
                return b
    return next((b for b in bts if not b.reference), bts[0] if bts else None)


# --------------------------------------------------------------------------- #
# 4. Journal + demo
# --------------------------------------------------------------------------- #

def journal_stats(db: Path) -> dict:
    out = {"alerts": 0, "journaled": 0, "first": None, "last": None}
    if not Path(db).exists():
        return out
    with closing(sqlite3.connect(db)) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "alerts" not in tables:
            return out
        out["alerts"], out["first"], out["last"] = con.execute(
            "SELECT COUNT(*), MIN(alert_time), MAX(alert_time) FROM alerts").fetchone()
        if "outcomes" in tables:
            out["journaled"] = con.execute(
                "SELECT COUNT(DISTINCT alert_id) FROM outcomes WHERE status='done'").fetchone()[0]
    return out


def demo_stats(db: Path, now: pd.Timestamp) -> dict:
    """Practice-account history from execution.py's tables (orders + daily_nav, in alerts.sqlite)."""
    out = {"filled": 0, "first_fill": None, "nav_first_day": None, "nav_last_day": None,
           "nav_start": None, "nav_last": None, "pnl": None, "days": 0, "live_orders": 0}
    if not Path(db).exists():
        return out
    with closing(sqlite3.connect(db)) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "orders" in tables:
            out["filled"], out["first_fill"] = con.execute(
                "SELECT COUNT(*), MIN(created_at) FROM orders WHERE status='filled' AND environment='practice'"
            ).fetchone()
            out["live_orders"] = con.execute(
                "SELECT COUNT(*) FROM orders WHERE environment='live' AND status='filled'").fetchone()[0]
        if "daily_nav" in tables:
            rows = con.execute("SELECT day, start_nav FROM daily_nav ORDER BY day").fetchall()
            if rows:
                out.update(nav_first_day=rows[0][0], nav_last_day=rows[-1][0],
                           nav_start=float(rows[0][1]), nav_last=float(rows[-1][1]),
                           pnl=float(rows[-1][1]) - float(rows[0][1]))
                out["days"] = int((pd.Timestamp(rows[-1][0]) - pd.Timestamp(rows[0][0])).days)
    return out


# --------------------------------------------------------------------------- #
# Phases + next step
# --------------------------------------------------------------------------- #

@dataclass
class Gate:
    phase: int
    name: str
    gate: str
    passed: bool | None          # None = can't tell yet
    evidence: str


def phase_gates(modules: list[ModuleCheck], tests: dict | None, bt: BacktestHealth | None,
                journal: dict, demo: dict, live_flag: bool) -> list[Gate]:
    n_ok = sum(m.state == "built" for m in modules)
    if tests is None or tests.get("note") or any(m.state == "unchecked" for m in modules):
        g1 = None
        ev1 = f"{n_ok}/{len(MODULES)} modules import; " + ((tests or {}).get("note") or "tests not run yet")
    else:
        bad = tests["failed"] + tests["errors"]
        g1 = n_ok == len(MODULES) and bad == 0 and tests["returncode"] == 0
        ev1 = f"{n_ok}/{len(MODULES)} modules built; {tests['passed']} tests pass" + (f", {bad} failing" if bad else "")
    if bt is None:
        g2, ev2 = False, "no backtest yet"
    else:
        g2 = bt.beats_baseline
        ev2 = (f"{bt.ticker} {bt.period}{' (reference run)' if bt.reference else ''}: "
               f"{'beats' if g2 else 'does not beat'} the baseline (skill {bt.skill:+.1%})"
               if bt.skill is not None else f"{bt.ticker}: beats baseline {'yes' if g2 else 'no'}")
    g3 = journal["journaled"] >= MIN_JOURNAL
    ev3 = f"{journal['journaled']}/{MIN_JOURNAL} alerts with a resolved outcome ({journal['alerts']} received)"
    g4 = demo["filled"] > 0 and demo["days"] >= MIN_DEMO_DAYS
    pnl = "" if demo["pnl"] is None else f", P&L {demo['pnl']:+,.2f} (account currency)"
    ev4 = f"{demo['days']}/{MIN_DEMO_DAYS} days of practice P&L, {demo['filled']} filled practice orders{pnl}"
    ev5 = ("LIVE_TRADING=true is set in .env" if live_flag else
           "LIVE_TRADING is off. Only you can turn it on, in .env")
    return [Gate(1, "Build", "all modules import and every test passes", g1, ev1),
            Gate(2, "Validate", "backtest beats the naive baseline", g2, ev2),
            Gate(3, "Journal", f"{MIN_JOURNAL}+ journaled TradingView alerts", g3, ev3),
            Gate(4, "Demo", "2+ months of practice-account P&L", g4, ev4),
            Gate(5, "Live", "all gates passed; your decision", None, ev5)]


def current_phase(gates: list[Gate]) -> int:
    for g in gates[:4]:
        if not g.passed:
            return g.phase
    return 5


def next_step(modules: list[ModuleCheck], tests: dict | None, keys: dict[str, bool], data: list[SourceHealth],
              gates: list[Gate], bt: BacktestHealth | None, journal: dict, demo: dict) -> str:
    missing = [m for m in modules if m.state == "missing"]
    if missing:
        return f"Fix `{missing[0].name}`: it {missing[0].detail}. Then press Re-check on the Status tab."
    if tests and not tests.get("note") and (tests["failed"] or tests["errors"]):
        f = (tests.get("failures") or ["?"])[0]
        file = f.split("::")[0]
        return (f"Fix the {tests['failed'] + tests['errors']} failing test(s), starting with {f}: "
                f"run `pytest tests/{file} -q` to see why.")
    if tests is None or tests.get("note"):
        return "Run the build check: press Re-check now on the Status tab (runs every test, about a minute)."
    for k in KEYS:
        if not keys.get(k):
            how = {"APP_PASSWORD_HASH": "run setup.bat (or `python -m forecaster.auth --write-env`)",
                   "FRED_API_KEY": "get a free key at fred.stlouisfed.org/docs/api/api_key.html",
                   "FINNHUB_API_KEY": "get a free key at finnhub.io/register",
                   "TELEGRAM_BOT_TOKEN": "create a bot with @BotFather",
                   "TELEGRAM_CHAT_ID": "message your bot, then read the id from getUpdates (README)",
                   "WEBHOOK_SECRET": "generate one with `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`",
                   }[k]
            return f"Add {k} to .env ({KEYS[k]}): {how}, then restart with start.bat."
    bad = [d for d in data if not d.ok]
    if bad:
        d = bad[0]
        return (f"{d.label} data: {d.message}. Check that the app is running (start.bat) and look at "
                f"logs/monitor.log for the error.")
    ph = current_phase(gates)
    if ph == 1:
        return "Run the build check again: press Re-check now on the Status tab."
    if ph == 2:
        if bt is not None and bt.reference:
            return ("Run your own SPY backtest with macro data: `python -m forecaster.pipeline SPY "
                    "--backtest-stale-days 0` (about 5 min). The reference run doesn't beat the baseline.")
        return ("Improve the model before trusting it: shrink regime drifts and the P50 toward the historical "
                "drift (fix #1 in CLAUDE.md), then rerun `python -m forecaster.pipeline SPY --backtest-stale-days 0`.")
    if ph == 3:
        need = MIN_JOURNAL - journal["journaled"]
        return (f"Keep the TradingView VWAP alerts running and journal {need} more resolved alerts "
                "(watch the Gold VWAP tab; outcomes resolve 1 h / 4 h / 1 d after each alert).")
    if ph == 4:
        if not (keys.get("OANDA_API_TOKEN") and keys.get("OANDA_ACCOUNT_ID")):
            return ("Start the demo: add OANDA_API_TOKEN and OANDA_ACCOUNT_ID for a PRACTICE account to .env "
                    "and set EXECUTION_ENABLED=true (orders stay on practice).")
        if demo["filled"] == 0:
            return "Wait for the first TAKE alert to place a practice order (check `python -m forecaster.execution --status`)."
        return f"Keep the practice account trading: {MIN_DEMO_DAYS - demo['days']} more days of demo P&L needed."
    pnl = "" if demo["pnl"] is None else f" (demo P&L {demo['pnl']:+,.2f})"
    return (f"All gates passed{pnl}. Review the demo results; going live is your decision and only you can set "
            "LIVE_TRADING=true in .env. Start small.")


# --------------------------------------------------------------------------- #
# Assemble, render, persist
# --------------------------------------------------------------------------- #

def load_cached(cache_dir: Path) -> dict | None:
    try:
        return json.loads(status_json(cache_dir).read_text())
    except (FileNotFoundError, ValueError):
        return None


def build(cache_dir: Path, now: pd.Timestamp | None = None, cached: dict | None = None,
          root: Path = PROJECT_ROOT, environ: dict | None = None, reference: Path | None = None) -> dict:
    """Full status from the cached build check (imports + tests) plus live data/model/phase checks."""
    now = now or pd.Timestamp.now(tz="UTC")
    cached = cached if cached is not None else (load_cached(cache_dir) or {})
    tests = cached.get("tests")
    mods = module_checks(cached.get("imports"), tests, root)
    keys = key_presence(root, environ)
    data = data_health(cache_dir, keys, now)
    bts = model_health(cache_dir, reference)
    bt = primary_backtest(bts)
    db = Path(cache_dir) / "alerts.sqlite"
    journal, demo = journal_stats(db), demo_stats(db, now)
    from dotenv import dotenv_values

    env_file = root / ".env"
    live = str((dotenv_values(env_file) if env_file.exists() else {}).get("LIVE_TRADING", "")).lower() == "true"
    gates = phase_gates(mods, tests, bt, journal, demo, live)
    return {
        "generated_at": now.isoformat(),
        "checked_at": cached.get("checked_at"),
        "modules": [asdict(m) for m in mods],
        "tests": tests,
        "keys": keys,
        "data": [asdict(d) for d in data],
        "backtests": [asdict(b) for b in bts],
        "primary_backtest": None if bt is None else asdict(bt),
        "journal": journal,
        "demo": demo,
        "gates": [asdict(g) for g in gates],
        "phase": current_phase(gates),
        "next_step": next_step(mods, tests, keys, data, gates, bt, journal, demo),
    }


def _ts(iso: str | None, fmt: str = "%Y-%m-%d %H:%M UTC") -> str:
    return "never" if not iso else pd.Timestamp(iso).tz_convert("UTC").strftime(fmt)


def gate_icon(g: dict, current: int) -> str:
    if g["phase"] == current:
        return "👉"
    return "✅" if g["passed"] else ("⏳" if g["passed"] is None else "⬜")


def render_markdown(s: dict) -> str:
    L = ["# Project status", "",
         f"_Auto-generated {_ts(s['generated_at'])} by `python -m forecaster.project_status` "
         "(on app start and the Sunday job). Don't edit by hand._", "",
         f"**Next step:** {s['next_step']}", "",
         f"## Phase tracker: Phase {s['phase']} ({PHASES[s['phase'] - 1]})", "",
         "| | Phase | Gate | Status |", "|---|---|---|---|"]
    for g in s["gates"]:
        name = f"**{g['phase']} {g['name']}**" if g["phase"] == s["phase"] else f"{g['phase']} {g['name']}"
        L.append(f"| {gate_icon(g, s['phase'])} | {name} | {g['gate']} | {g['evidence']} |")
    t = s["tests"]
    L += ["", "## Build progress", ""]
    if t and not t.get("note"):
        L.append(f"Tests: {t['passed']} passed, {t['failed']} failed, {t['errors']} errors, {t['skipped']} skipped "
                 f"({t['seconds']:.0f}s, run {_ts(t['ran_at'])}).")
    else:
        L.append(f"Tests: {(t or {}).get('note') or 'not run yet'}.")
    L += ["", "| | Module | Detail |", "|---|---|---|"]
    L += [f"| {ICON[m['state']]} | `{m['name']}` | {m['detail']} |" for m in s["modules"]]
    if t and t.get("failures"):
        L += ["", "Failing tests:", ""] + [f"- `{f}`" for f in t["failures"][:15]]
    L += ["", "## Data health", "", "| | Source | Last successful refresh | Status |", "|---|---|---|---|"]
    L += [f"| {'✅' if d['ok'] else '🔴'} | {d['label']} | {_ts(d['last_success'])} | {d['message']} |"
          for d in s["data"]]
    L += ["", "API keys in .env (values never shown): " +
          ", ".join(f"{k} {'✅' if v else '🔴 missing'}" for k, v in s["keys"].items() if k in KEYS),
          "", "Demo-phase keys: " + ", ".join(f"{k} {'✅' if v else '— not set'}"
                                          for k, v in s["keys"].items() if k in DEMO_KEYS)]
    L += ["", "## Model health", "", "| Backtest | Period | P10–P90 coverage | Hit rate | Beats baseline | Last run |",
          "|---|---|---|---|---|---|"]
    for b in s["backtests"]:
        L.append(f"| {b['ticker']}{' (reference)' if b['reference'] else ''} | {b['period']} | {b['coverage']:.0%} "
                 f"(target 80%) | {b['hit_rate']:.0%} | {'yes' if b['beats_baseline'] else 'no'} | "
                 f"{_ts(b['run_at'], '%Y-%m-%d')} |")
    if not s["backtests"]:
        L.append("| none yet | | | | | |")
    j, d = s["journal"], s["demo"]
    L += ["", "## Journal & demo", "",
          f"- TradingView alerts: {j['alerts']} received, {j['journaled']} with a resolved outcome.",
          f"- Practice account: {d['filled']} filled orders, {d['days']} days of NAV history"
          + ("" if d["pnl"] is None else f", P&L {d['pnl']:+,.2f} (account currency, from daily start NAVs)") + "."]
    return "\n".join(L) + "\n"


def telegram_text(s: dict) -> str:
    import html

    e = html.escape
    lines = [f"<b>Project status</b> · Phase {s['phase']} ({PHASES[s['phase'] - 1]})"]
    lines += [f"{gate_icon(g, s['phase'])} {g['phase']} {g['name']}: {e(g['evidence'])}" for g in s["gates"][:4]]
    n = sum(m["state"] == "built" for m in s["modules"])
    t = s["tests"]
    tests = (f"{t['passed']} tests pass" + (f", {t['failed'] + t['errors']} failing" if t["failed"] + t["errors"] else "")
             if t and not t.get("note") else "tests not run")
    lines.append(f"Build: {n}/{len(s['modules'])} modules ✅, {tests}")
    bad = [d for d in s["data"] if not d["ok"]]
    lines.append("Data: all fresh" if not bad else "Data 🔴 " + "; ".join(e(f"{d['label']}: {d['message']}") for d in bad))
    missing = [k for k, v in s["keys"].items() if k in KEYS and not v]
    if missing:
        lines.append("Keys missing: " + ", ".join(missing))
    bt = s["primary_backtest"]
    if bt:
        lines.append(f"Model: {e(bt['ticker'])} coverage {bt['coverage']:.0%}, hit rate {bt['hit_rate']:.0%}, "
                     f"beats baseline {'yes' if bt['beats_baseline'] else 'no'} ({_ts(bt['run_at'], '%Y-%m-%d')})")
    lines.append(f"<b>Next:</b> {e(s['next_step'])}")
    return "\n".join(lines)


def _pid_alive(pid: int) -> bool:
    from forecaster.launcher import pid_alive

    return pid_alive(pid)


def running(cache_dir: Path) -> bool:
    try:
        return _pid_alive(int((Path(cache_dir) / "project_status.lock").read_text().strip()))
    except (FileNotFoundError, ValueError):
        return False


def refresh(cache_dir: Path, run_tests: bool = True, md_path: Path | None = STATUS_MD,
            runner=run_pytest, importer=check_imports, root: Path = PROJECT_ROOT) -> dict:
    """Run the build check, cache it, and write STATUS.md. Single-flight via a lock file."""
    lock = Path(cache_dir) / "project_status.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if running(cache_dir) and int(lock.read_text().strip()) != os.getpid():
        return build(cache_dir, root=root)                       # another check is running: use the cache
    lock.write_text(str(os.getpid()))
    try:
        prev = load_cached(cache_dir) or {}
        cached = {"checked_at": pd.Timestamp.now(tz="UTC").isoformat(), "imports": importer(),
                  "tests": asdict(runner(root)) if run_tests else prev.get("tests")}
        p = status_json(cache_dir)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cached, indent=1))
        tmp.replace(p)
        s = build(cache_dir, cached=cached, root=root)
        if md_path is not None:
            write_md(s, md_path)
        return s
    finally:
        lock.unlink(missing_ok=True)


def write_md(s: dict, path: Path = STATUS_MD) -> None:
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(render_markdown(s))
    tmp.replace(path)


def recently_checked(cache_dir: Path, within: pd.Timedelta = pd.Timedelta(minutes=10),
                     now: pd.Timestamp | None = None) -> bool:
    c = load_cached(cache_dir) or {}
    if not c.get("checked_at") or not c.get("tests"):
        return False
    return (now or pd.Timestamp.now(tz="UTC")) - pd.Timestamp(c["checked_at"]) < within


def should_start(cache_dir: Path, environ: dict | None = None) -> bool:
    """Start-up check: not from inside a test run, not twice at once, not right after another one
    (the launcher and the app both start it; the second one is skipped)."""
    environ = os.environ if environ is None else environ
    if environ.get("FORECASTER_STATUS_CHILD") or environ.get("PYTEST_CURRENT_TEST"):
        return False
    return not running(cache_dir) and not recently_checked(cache_dir)


def start_background(cache_dir: Path, root: Path = PROJECT_ROOT) -> bool:  # pragma: no cover - spawns
    """Kick off `python -m forecaster.project_status` detached (app/launcher start). False if skipped."""
    if not should_start(cache_dir):
        return False
    log = Path(root) / "logs"
    log.mkdir(exist_ok=True)
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(root) / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with open(log / "project_status.log", "ab") as f:
        subprocess.Popen([sys.executable, "-m", "forecaster.project_status"], cwd=root, env=env,
                         stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
    return True


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    from forecaster.config import load_settings

    ap = argparse.ArgumentParser(description="Project status: build, data, model, phase, next step")
    ap.add_argument("--quick", action="store_true", help="don't run pytest (keep the last test result)")
    ap.add_argument("--send", action="store_true", help="send the summary to Telegram")
    args = ap.parse_args(argv)
    settings = load_settings()
    s = refresh(settings.cache_dir, run_tests=not args.quick)
    print(re.sub(r"\*\*|`", "", render_markdown(s)))
    if args.send:
        from forecaster.webhook_server import TelegramNotifier

        ok, err = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id).send(telegram_text(s))
        print("telegram:", "sent" if ok else f"not sent ({err})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
