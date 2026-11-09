"""Data health: when each source last refreshed, and whether its last attempt failed.

Writers (monitor, pipeline) call `record(source, ok, error)`; the app reads
`problems()` to decide whether to show the red banner on every page.

A source is a problem when its last attempt failed, or its last success is older
than its limit (24 h for prices, news, macro/events and the portfolio; 8 days for
the weekly forecasts). A source that has never run is reported as missing.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# source -> (label, max age)
SOURCES: dict[str, tuple[str, pd.Timedelta]] = {
    "prices": ("Prices", pd.Timedelta(hours=24)),
    "news": ("News", pd.Timedelta(hours=24)),
    "market": ("Macro & events", pd.Timedelta(hours=24)),
    "macro": ("FRED macro", pd.Timedelta(hours=24)),
    "calendar": ("Calendar", pd.Timedelta(hours=24)),
    "portfolio": ("Portfolio", pd.Timedelta(hours=24)),
    "forecasts": ("Forecasts", pd.Timedelta(days=8)),
}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Problem:
    source: str
    message: str


def _path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "status.json"


def load(cache_dir: Path) -> dict:
    try:
        return json.loads(_path(cache_dir).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def record(cache_dir: Path, source: str, ok: bool, error: str | None = None,
           now: pd.Timestamp | None = None) -> None:
    now = now or pd.Timestamp.now(tz="UTC")
    with _LOCK:
        data = load(cache_dir)
        s = data.get(source, {})
        s["last_attempt"] = now.isoformat()
        if ok:
            s["last_success"] = now.isoformat()
            s["error"] = None
        else:
            s["error"] = (error or "failed")[:300]
        data[source] = s
        p = _path(cache_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(p)


def _ago(td: pd.Timedelta) -> str:
    h = td.total_seconds() / 3600
    if h < 1:
        return f"{max(1, round(h * 60))} min"
    return f"{h:.0f} h" if h < 48 else f"{h / 24:.0f} days"


def problems(cache_dir: Path, now: pd.Timestamp | None = None,
             expected: tuple[str, ...] | None = None) -> list[Problem]:
    """Failures and stale data. `expected` limits which sources must exist
    (e.g. news only when a Finnhub key is configured)."""
    now = now or pd.Timestamp.now(tz="UTC")
    data = load(cache_dir)
    out = []
    for src in expected or tuple(SOURCES):
        label, limit = SOURCES[src]
        s = data.get(src)
        if not s or not s.get("last_success"):
            if s and s.get("error"):
                out.append(Problem(src, f"{label}: refresh failed ({s['error']}) and has never succeeded"))
            else:
                out.append(Problem(src, f"{label}: no data yet"))
            continue
        age = now - pd.Timestamp(s["last_success"])
        if s.get("error"):
            out.append(Problem(src, f"{label}: last refresh failed ({s['error']}); data is {_ago(age)} old"))
        elif age > limit:
            out.append(Problem(src, f"{label}: data is {_ago(age)} old (limit {_ago(limit)})"))
    return out


def last_updated(cache_dir: Path, source: str) -> pd.Timestamp | None:
    s = load(cache_dir).get(source) or {}
    return pd.Timestamp(s["last_success"]) if s.get("last_success") else None


def expected_sources(settings) -> tuple[str, ...]:
    """Which sources should exist given the configuration."""
    out = ["prices", "market", "forecasts"]
    if settings.finnhub_api_key:
        out.append("news")
    from forecaster.portfolio import PORTFOLIO_FILE

    if PORTFOLIO_FILE.exists():
        out.append("portfolio")
    return tuple(out)
