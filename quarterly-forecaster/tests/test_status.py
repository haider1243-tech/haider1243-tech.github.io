import pandas as pd

from forecaster import status
from forecaster.config import load_settings

NOW = pd.Timestamp("2026-11-09 15:00", tz="UTC")


def test_fresh_sources_have_no_problems(tmp_path):
    for s in ("prices", "market", "forecasts"):
        status.record(tmp_path, s, True, now=NOW - pd.Timedelta(hours=1))
    assert status.problems(tmp_path, NOW, ("prices", "market", "forecasts")) == []
    assert status.last_updated(tmp_path, "prices") == NOW - pd.Timedelta(hours=1)
    assert status.last_updated(tmp_path, "news") is None


def test_failure_stale_and_missing(tmp_path):
    status.record(tmp_path, "prices", True, now=NOW - pd.Timedelta(hours=2))
    status.record(tmp_path, "prices", False, "timeout", now=NOW)
    status.record(tmp_path, "market", True, now=NOW - pd.Timedelta(hours=25))
    status.record(tmp_path, "forecasts", True, now=NOW - pd.Timedelta(days=7))     # weekly: 8-day limit
    status.record(tmp_path, "news", False, "401", now=NOW)
    msgs = {p.source: p.message for p in status.problems(tmp_path, NOW)}
    assert msgs["prices"] == "Prices: last refresh failed (timeout); data is 2 h old"
    assert msgs["market"] == "Macro & events: data is 25 h old (limit 24 h)"
    assert "forecasts" not in msgs
    assert msgs["news"] == "News: refresh failed (401) and has never succeeded"
    assert msgs["portfolio"] == "Portfolio: no data yet"
    status.record(tmp_path, "prices", True, now=NOW)                                # success clears the error
    assert "prices" not in {p.source for p in status.problems(tmp_path, NOW)}


def test_corrupt_file_is_treated_as_empty(tmp_path):
    (tmp_path / "status.json").write_text("{not json")
    assert [p.message for p in status.problems(tmp_path, NOW, ("prices",))] == ["Prices: no data yet"]


def test_expected_sources_follow_config(tmp_path, monkeypatch):
    import forecaster.portfolio as pf
    monkeypatch.setattr(pf, "PORTFOLIO_FILE", tmp_path / "portfolio.yaml")
    s = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path)})
    assert status.expected_sources(s) == ("prices", "market", "forecasts")
    (tmp_path / "portfolio.yaml").write_text("holdings: []")
    s2 = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path), "FINNHUB_API_KEY": "x"})
    assert status.expected_sources(s2) == ("prices", "market", "forecasts", "news", "portfolio")
