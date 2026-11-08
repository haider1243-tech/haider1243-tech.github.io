import pandas as pd
import pytest

from conftest import RoutingSession
from forecaster.calendar import (
    FOMC_DECISION_DATES,
    MEGA_CAPS,
    Event,
    build_calendar,
    earnings_events,
    earnings_impact,
    events_frame,
    fomc_events,
    fred_release_events,
    quarter_windows,
    risk_density,
)
from forecaster.sources.finnhub import FinnhubClient
from forecaster.sources.fred import FredClient

TODAY = pd.Timestamp("2026-11-08")

RELEASE_IDS = {"CPIAUCSL": (10, "Consumer Price Index"), "UNRATE": (50, "Employment Situation"),
               "PPIACO": (46, "Producer Price Index"), "GDP": (53, "Gross Domestic Product")}
RELEASE_DATES = {
    10: ["2027-12-10", "2027-01-13", "2026-12-10", "2026-11-13", "2026-10-15"],  # incl. one past
    50: ["2027-01-08", "2026-12-04"],
    46: ["2026-11-14"],
    53: ["2027-01-28"],
}


def fred_handler(url, params):
    if url.endswith("series/release"):
        rid, name = RELEASE_IDS[params["series_id"]]
        return {"releases": [{"id": rid, "name": name}]}
    if url.endswith("release/dates"):
        assert params["include_release_dates_with_no_data"] == "true"
        return {"release_dates": [{"release_id": params["release_id"], "date": d}
                                  for d in RELEASE_DATES[params["release_id"]]]}
    raise AssertionError(url)


EARNINGS = {
    "AAPL": [{"date": "2027-01-28", "symbol": "AAPL", "hour": "amc", "quarter": 1, "year": 2027}],
    "ACME": [{"date": "2026-12-01", "symbol": "ACME", "hour": "bmo", "quarter": 4, "year": 2026},
             {"date": "2030-01-01", "symbol": "ACME"},  # outside window
             {"date": None, "symbol": "ACME"}],
}


def finnhub_handler(url, params):
    assert url.endswith("calendar/earnings")
    if params["symbol"] == "BOOM":
        return 429, {}
    return {"earningsCalendar": EARNINGS.get(params["symbol"], [])}


@pytest.fixture
def fred(cache):
    return FredClient("k", cache, session=RoutingSession(fred_handler))


@pytest.fixture
def finnhub(cache):
    return FinnhubClient("k", cache, session=RoutingSession(finnhub_handler))


# --------------------------------------------------------------------------- #
# FOMC
# --------------------------------------------------------------------------- #

def test_fomc_dates_shape():
    dates = [pd.Timestamp(d) for d in FOMC_DECISION_DATES]
    assert len(dates) == 16
    assert sum(d.year == 2026 for d in dates) == 8
    assert sum(d.year == 2027 for d in dates) == 8
    assert all(d.day_of_week == 2 for d in dates)  # decisions on Wednesdays
    assert dates == sorted(dates)


def test_fomc_sep_meetings_score_higher():
    ev = fomc_events(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))
    by_month = {e.date.month: e for e in ev}
    assert by_month[12].points == 4 and "SEP" in by_month[12].detail
    assert by_month[10].points == 3
    assert all(e.impact == "high" for e in ev)


def test_fomc_window():
    ev = fomc_events(TODAY, pd.Timestamp("2027-03-31"))
    assert [e.date for e in ev] == [pd.Timestamp(d) for d in ("2026-12-09", "2027-01-27", "2027-03-17")]


# --------------------------------------------------------------------------- #
# FRED releases
# --------------------------------------------------------------------------- #

def test_fred_release_events(fred):
    ev = fred_release_events(fred, TODAY, pd.Timestamp("2027-12-31"))
    cpi = [e for e in ev if e.category == "CPI"]
    assert [e.date.strftime("%Y-%m-%d") for e in sorted(cpi, key=lambda e: e.date)] == [
        "2026-11-13", "2026-12-10", "2027-01-13", "2027-12-10"
    ]  # past 2026-10-15 excluded
    assert all(e.impact == "high" for e in cpi)
    assert {e.impact for e in ev if e.category == "PPI"} == {"medium"}
    assert {e.impact for e in ev if e.category == "Jobs"} == {"high"}
    assert {e.impact for e in ev if e.category == "GDP"} == {"medium"}


def test_fred_release_events_cached(cache):
    session = RoutingSession(fred_handler)
    fred = FredClient("k", cache, session=session)
    fred_release_events(fred, TODAY, pd.Timestamp("2027-12-31"))
    n = len(session.calls)
    fred_release_events(fred, TODAY, pd.Timestamp("2027-12-31"))
    assert len(session.calls) == n == 8  # 4 release lookups + 4 date lists, once


def test_fred_release_events_on_error(cache):
    fred = FredClient("k", cache, session=RoutingSession(lambda u, p: (500, {})))
    errs = []
    assert fred_release_events(fred, TODAY, pd.Timestamp("2027-12-31"),
                               on_error=lambda c, e: errs.append(c)) == []
    assert errs == ["CPI", "Jobs", "PPI", "GDP"]


# --------------------------------------------------------------------------- #
# Earnings
# --------------------------------------------------------------------------- #

def test_earnings_impact():
    assert "AAPL" in MEGA_CAPS
    assert earnings_impact("aapl") == "medium"
    assert earnings_impact("ACME") == "low"


def test_earnings_events(finnhub):
    errs = []
    ev = earnings_events(finnhub, ["AAPL", "ACME", "^VIX", "GC=F", "BOOM"],
                         TODAY, pd.Timestamp("2027-12-31"), on_error=lambda t, e: errs.append(t))
    assert {(e.ticker, e.date.strftime("%Y-%m-%d"), e.impact) for e in ev} == {
        ("AAPL", "2027-01-28", "medium"), ("ACME", "2026-12-01", "low"),
    }
    acme = next(e for e in ev if e.ticker == "ACME")
    assert acme.detail == "Q4 2026 before open"
    assert errs == ["BOOM"]
    calls = [p["symbol"] for _, p, _ in finnhub.session.calls]
    assert "^VIX" not in calls and "GC=F" not in calls


def test_finnhub_token_sent_in_header_not_params(finnhub):
    earnings_events(finnhub, ["AAPL"], TODAY, pd.Timestamp("2027-12-31"))
    _, params, kwargs = finnhub.session.calls[0]
    assert "token" not in params
    assert kwargs["headers"]["X-Finnhub-Token"] == "k"
    assert params["from"] == "2026-11-08" and params["to"] == "2027-12-31"


# --------------------------------------------------------------------------- #
# Calendar + risk density
# --------------------------------------------------------------------------- #

def test_quarter_windows():
    assert [str(q) for q in quarter_windows(TODAY)] == ["2026Q4", "2027Q1", "2027Q2", "2027Q3", "2027Q4"]


def test_build_calendar_sorted_deduped(fred, finnhub):
    ev = build_calendar(TODAY, fred, finnhub, ["AAPL", "ACME", "AAPL"])
    assert [e.date for e in ev] == sorted(e.date for e in ev)
    assert len({(e.date, e.category, e.ticker) for e in ev}) == len(ev)
    assert ev[0].date >= TODAY
    assert ev[-1].date <= pd.Timestamp("2027-12-31")
    assert any(e.category == "FOMC" and e.date == pd.Timestamp("2026-12-09") for e in ev)
    # same-day ordering: higher impact first (2027-01-28 has AAPL earnings + GDP)
    same_day = [e for e in ev if e.date == pd.Timestamp("2027-01-28")]
    assert [e.points for e in same_day] == sorted((e.points for e in same_day), reverse=True)


def test_build_calendar_fomc_only_without_clients():
    ev = build_calendar(TODAY)
    assert {e.category for e in ev} == {"FOMC"}
    assert len(ev) == 9  # Dec 2026 + 8 in 2027


def test_events_frame(fred, finnhub):
    df = events_frame(build_calendar(TODAY, fred, finnhub, ["AAPL"]))
    assert list(df.columns) == ["date", "category", "name", "impact", "points", "ticker", "source", "detail"]
    assert events_frame([]).empty


def test_risk_density_shape_and_partial():
    rd = risk_density(build_calendar(TODAY), TODAY)
    assert list(rd.index) == ["2026Q4", "2027Q1", "2027Q2", "2027Q3", "2027Q4"]
    assert rd.loc["2026Q4", "partial"]
    assert not rd.loc["2027Q1", "partial"]
    assert rd.loc["2026Q4", "start"] == TODAY


def test_risk_density_normalizes_partial_quarter():
    # 1 high event in the remaining 2026Q4 window vs the same in a full quarter.
    ev = [Event(pd.Timestamp("2026-12-01"), "CPI", "CPI", "high", "t"),
          Event(pd.Timestamp("2027-02-01"), "CPI", "CPI", "high", "t")]
    rd = risk_density(ev, TODAY)
    q4_bdays = len(pd.bdate_range(TODAY, "2026-12-31"))
    assert rd.loc["2026Q4", "business_days"] == q4_bdays
    assert rd.loc["2026Q4", "points"] == rd.loc["2027Q1", "points"] == 3
    assert rd.loc["2026Q4", "density"] == pytest.approx(3 / q4_bdays * 63)
    assert rd.loc["2026Q4", "density"] > rd.loc["2027Q1", "density"]


def test_risk_density_counts_busiest_week_and_missing():
    ev = [
        Event(pd.Timestamp("2027-01-12"), "CPI", "CPI", "high", "t"),
        Event(pd.Timestamp("2027-01-13"), "PPI", "PPI", "medium", "t"),
        Event(pd.Timestamp("2027-01-13"), "Earnings", "X", "low", "t", ticker="X"),
        Event(pd.Timestamp("2027-03-17"), "FOMC", "FOMC", "high", "t", detail="SEP + press conference"),
    ]
    row = risk_density(ev, TODAY).loc["2027Q1"]
    assert (row["n_high"], row["n_medium"], row["n_low"]) == (2, 1, 1)
    assert row["points"] == 3 + 2 + 1 + 4
    assert row["busiest_week_points"] == 6
    assert row["missing"] == "Jobs,GDP"
    assert risk_density(ev, TODAY).loc["2027Q2", "missing"] == "FOMC,CPI,Jobs,PPI,GDP"
    assert risk_density(ev, TODAY).loc["2026Q4", "missing"] == ""  # partial: not flagged


def test_risk_density_empty():
    rd = risk_density([], TODAY)
    assert (rd["points"] == 0).all() and (rd["density"] == 0).all()
