"""Event calendar and per-quarter risk density.

Sources
-------
* FRED release dates (scheduled, incl. future) for CPI, PPI, jobs, GDP.
* Hardcoded FOMC decision dates for 2026-2027 (federalreserve.gov schedule;
  the Fed marks future meetings tentative until the prior meeting confirms them).
* Finnhub free-tier earnings calendar for watchlist tickers.

Impact tags
-----------
high:   FOMC decisions, CPI, Employment Situation (jobs)
medium: PPI, GDP, earnings of mega-cap index movers (MEGA_CAPS)
low:    other watchlist earnings

Risk density
------------
Each event scores IMPACT_POINTS (high 3, medium 2, low 1); FOMC meetings with
a Summary of Economic Projections get +1. A quarter's density is its points
per business day, rescaled to a 63-business-day quarter so a partial current
quarter is comparable to full ones. `missing` flags full quarters with no
events for a scheduled macro category — usually because the agency hasn't
published that far ahead — so the score is a lower bound there.

Note: imported as `forecaster.calendar`; don't run this file by path
(`python src/forecaster/calendar.py`), which would shadow the stdlib module.
Use `python -m forecaster.calendar`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Literal

import pandas as pd

from forecaster.config import HORIZON_QUARTERS
from forecaster.sources.finnhub import FinnhubClient
from forecaster.sources.fred import FredClient

Impact = Literal["high", "medium", "low"]
IMPACT_POINTS: dict[str, int] = {"high": 3, "medium": 2, "low": 1}
SEP_BONUS = 1
STANDARD_QUARTER_BDAYS = 63

# Decision (second) day of each scheduled FOMC meeting.
FOMC_DECISION_DATES: tuple[str, ...] = (
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    # 2027 (tentative)
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
)
SEP_MONTHS = {3, 6, 9, 12}  # meetings with a Summary of Economic Projections


@dataclass(frozen=True)
class MacroRelease:
    category: str
    name: str
    series_id: str  # used to look up the FRED release id
    impact: Impact


MACRO_RELEASES: tuple[MacroRelease, ...] = (
    MacroRelease("CPI", "CPI release", "CPIAUCSL", "high"),
    MacroRelease("Jobs", "Employment Situation", "UNRATE", "high"),
    MacroRelease("PPI", "PPI release", "PPIACO", "medium"),
    MacroRelease("GDP", "GDP release", "GDP", "medium"),
)
SCHEDULED_CATEGORIES = ("FOMC", *(r.category for r in MACRO_RELEASES))

MEGA_CAPS = frozenset(
    {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK.B", "BRK-B"}
)


@dataclass(frozen=True)
class Event:
    date: pd.Timestamp
    category: str
    name: str
    impact: Impact
    source: str
    ticker: str | None = None
    detail: str = ""

    @property
    def points(self) -> int:
        bonus = SEP_BONUS if self.category == "FOMC" and "SEP" in self.detail else 0
        return IMPACT_POINTS[self.impact] + bonus


# --------------------------------------------------------------------------- #
# Event sources
# --------------------------------------------------------------------------- #

def _in_window(d: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return start <= d <= end


def fomc_events(start: pd.Timestamp, end: pd.Timestamp) -> list[Event]:
    out = []
    for s in FOMC_DECISION_DATES:
        d = pd.Timestamp(s)
        if _in_window(d, start, end):
            sep = d.month in SEP_MONTHS
            out.append(
                Event(d, "FOMC", "FOMC rate decision", "high", "hardcoded",
                      detail="SEP + press conference" if sep else "press conference")
            )
    return out


def fred_release_events(
    fred: FredClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    releases: tuple[MacroRelease, ...] = MACRO_RELEASES,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[Event]:
    out = []
    for rel in releases:
        try:
            info = fred.release_for_series(rel.series_id)
            for d in fred.release_dates(int(info["id"])):
                if _in_window(d, start, end):
                    out.append(Event(d, rel.category, rel.name, rel.impact, "FRED",
                                     detail=info.get("name", "")))
        except Exception as exc:
            if on_error is None:
                raise
            on_error(rel.category, exc)
    return out


def earnings_impact(ticker: str) -> Impact:
    return "medium" if ticker.upper() in MEGA_CAPS else "low"


def _has_earnings(ticker: str) -> bool:
    # Indices (^GSPC), futures (GC=F) and FX/index symbols (DX-Y.NYB) don't report.
    return not (ticker.startswith("^") or "=" in ticker or ticker.endswith(".NYB"))


def earnings_events(
    finnhub: FinnhubClient,
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[Event]:
    out = []
    for t in tickers:
        if not _has_earnings(t):
            continue
        try:
            rows = finnhub.earnings_calendar(t, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        except Exception as exc:
            if on_error is None:
                raise
            on_error(t, exc)
            continue
        for r in rows:
            if not r.get("date"):
                continue
            d = pd.Timestamp(r["date"])
            if not _in_window(d, start, end):
                continue
            detail = " ".join(
                str(x) for x in (
                    f"Q{r['quarter']} {r['year']}" if r.get("quarter") and r.get("year") else "",
                    {"bmo": "before open", "amc": "after close", "dmh": "during market"}.get(r.get("hour", ""), ""),
                ) if x
            )
            out.append(Event(d, "Earnings", f"{t} earnings", earnings_impact(t), "Finnhub",
                             ticker=t, detail=detail))
    return out


# --------------------------------------------------------------------------- #
# Calendar + risk density
# --------------------------------------------------------------------------- #

def quarter_windows(today: date | pd.Timestamp, n_ahead: int = HORIZON_QUARTERS) -> list[pd.Period]:
    """Current quarter plus the next `n_ahead` quarters."""
    q0 = pd.Period(pd.Timestamp(today), freq="Q")
    return [q0 + i for i in range(n_ahead + 1)]


def build_calendar(
    today: date | pd.Timestamp,
    fred: FredClient | None = None,
    finnhub: FinnhubClient | None = None,
    watchlist: Iterable[str] = (),
    n_ahead: int = HORIZON_QUARTERS,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[Event]:
    """All events from `today` through the end of the last horizon quarter."""
    quarters = quarter_windows(today, n_ahead)
    start = pd.Timestamp(today).normalize()
    end = quarters[-1].end_time.normalize()

    events = fomc_events(start, end)
    if fred is not None:
        events += fred_release_events(fred, start, end, on_error=on_error)
    if finnhub is not None:
        events += earnings_events(finnhub, watchlist, start, end, on_error=on_error)

    unique = {(e.date, e.category, e.ticker): e for e in events}
    return sorted(unique.values(), key=lambda e: (e.date, -e.points, e.category, e.ticker or ""))


def events_frame(events: list[Event]) -> pd.DataFrame:
    cols = ["date", "category", "name", "impact", "points", "ticker", "source", "detail"]
    rows = [{**{c: getattr(e, c) for c in cols if c != "points"}, "points": e.points} for e in events]
    return pd.DataFrame(rows, columns=cols)


def risk_density(
    events: list[Event],
    today: date | pd.Timestamp,
    n_ahead: int = HORIZON_QUARTERS,
) -> pd.DataFrame:
    """One row per quarter (current + n_ahead): counts, points, density, gaps."""
    today = pd.Timestamp(today).normalize()
    rows = []
    for q in quarter_windows(today, n_ahead):
        q_start, q_end = q.start_time.normalize(), q.end_time.normalize()
        start = max(q_start, today)
        bdays = len(pd.bdate_range(start, q_end))
        in_q = [e for e in events if start <= e.date <= q_end]
        points = sum(e.points for e in in_q)

        if in_q:
            weekly = pd.Series([e.points for e in in_q], index=[e.date for e in in_q])
            busiest = float(weekly.groupby(pd.Grouper(freq="W-FRI")).sum().max())
        else:
            busiest = 0.0

        partial = start > q_start
        present = {e.category for e in in_q}
        missing = [] if partial else [c for c in SCHEDULED_CATEGORIES if c not in present]

        rows.append(
            {
                "quarter": str(q),
                "start": start,
                "end": q_end,
                "partial": partial,
                "business_days": bdays,
                "n_high": sum(e.impact == "high" for e in in_q),
                "n_medium": sum(e.impact == "medium" for e in in_q),
                "n_low": sum(e.impact == "low" for e in in_q),
                "points": points,
                "density": points / bdays * STANDARD_QUARTER_BDAYS if bdays else 0.0,
                "busiest_week_points": busiest,
                "missing": ",".join(missing),
            }
        )
    return pd.DataFrame(rows).set_index("quarter")


def main() -> int:  # pragma: no cover - network wrapper
    import sys

    from forecaster.cache import Cache
    from forecaster.config import PROJECT_ROOT, load_settings, load_watchlist

    settings = load_settings()
    cache = Cache(settings.cache_dir, ttl_seconds=settings.cache_ttl_hours * 3600)
    warn = lambda s, e: print(f"[warn] {s}: {e}", file=sys.stderr)  # noqa: E731
    fred = FredClient(settings.fred_api_key, cache) if settings.fred_api_key else None
    fh = FinnhubClient(settings.finnhub_api_key, cache) if settings.finnhub_api_key else None
    if fred is None:
        warn("FRED", "FRED_API_KEY not set; macro release dates skipped")
    if fh is None:
        warn("Finnhub", "FINNHUB_API_KEY not set; earnings skipped")

    today = pd.Timestamp.today().normalize()
    events = build_calendar(today, fred, fh, load_watchlist(PROJECT_ROOT / "watchlist.yaml"),
                            on_error=warn)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 200):
        print(events_frame(events).to_string(index=False))
        print()
        print(risk_density(events, today).round({"density": 1}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
