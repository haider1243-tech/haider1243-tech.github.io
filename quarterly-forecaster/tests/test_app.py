"""Run app.py headless with Streamlit's AppTest against fixture snapshots."""
from pathlib import Path

import pytest

from conftest import authed_app
from streamlit.testing.v1 import AppTest

from app_fixtures import write_fixture_snapshots
from forecaster import pipeline

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    write_fixture_snapshots(d / "snapshots", ticker="SYN")
    return d


@pytest.fixture
def app(cache_dir, monkeypatch):
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(cache_dir))
    at = authed_app(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader]
    parts += [x.value for x in at.warning] + [x.value for x in at.info] + [x.value for x in at.success]
    parts += [x.value for x in at.error] + [m.label for m in at.metric]
    return "\n".join(map(str, parts))


def _select(at, ticker="SYN"):
    at.sidebar.selectbox(key="ticker").set_value(ticker).run()
    assert not at.exception, at.exception
    return at


def test_banner_on_every_page(app):
    at = _select(app)
    for page in ("Forecast", "Macro & events", "Sentiment & news"):
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, at.exception
        txt = _text(at)
        assert "Backtest accuracy" in txt, page
        assert "Outcomes inside P10–P90" in txt
        assert "Not reliably better than a naive historical-average baseline" in txt  # reference verdict
        assert "reference S&P 500 run" in txt


def test_forecast_page(app):
    at = _select(app)
    txt = _text(at)
    assert "SYN: 4-quarter forecast range" in txt
    assert "Ranges, not predictions" in txt
    assert "C$" in txt and "(CAD)" in txt and "Priced in CAD" in txt     # every price shown in CAD
    assert len(at.dataframe) >= 1
    tbl = at.dataframe[0].value
    assert list(tbl["Quarter"]) == ["2021Q3", "2021Q4", "2022Q1", "2022Q2"]
    assert (tbl["P10 (CAD)"] < tbl["P50 (CAD)"]).all() and (tbl["P50 (CAD)"] < tbl["P90 (CAD)"]).all()
    assert "Price data ends" in txt      # fixture as-of is 2021 -> staleness warning
    # SHAP selectors work
    at.selectbox(key="shap_level").set_value(0.1).run()
    at.selectbox(key="shap_q").set_value("2022Q2").run()
    assert not at.exception


def test_macro_page(app):
    at = _select(app)
    at.sidebar.radio(key="page").set_value("Macro & events").run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert {"CPI, YoY", "PPI, YoY", "Fed funds rate"} <= set(labels)
    events = at.dataframe[0].value
    assert set(events["Impact"]) == {"High"}
    at.toggle(key="high_only").set_value(False).run()
    assert {"High", "Medium"} <= set(at.dataframe[0].value["Impact"])


def test_sentiment_page(app):
    at = _select(app)
    at.sidebar.radio(key="page").set_value("Sentiment & news").run()
    assert not at.exception, at.exception
    txt = _text(at)
    assert "News sentiment (FinBERT)" in txt and "Top headlines" in txt
    assert "Recent shocks" in txt
    for order in ("Most negative", "Most positive", "Most recent"):
        at.radio(key="h_sort").set_value(order).run()
        assert not at.exception
    h = at.dataframe[-1].value
    assert len(h) == 15


def test_missing_snapshots_degrade_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(tmp_path))
    at = authed_app(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    for page in ("Forecast", "Macro & events", "Sentiment & news"):
        at.sidebar.radio(key="page").set_value(page).run()
        assert not at.exception, at.exception
        assert any("Refresh data" in i.value for i in at.info), page


def test_refresh_button_calls_pipeline(app, monkeypatch, cache_dir):
    calls = []

    def fake_refresh_all(settings, tickers, warn, backtest_stale_days=None, **kw):
        calls.append((tuple(tickers), backtest_stale_days))
        warn("fake refresh ran")
        return {"market": "ok", **{t: "ok" for t in tickers}}

    monkeypatch.setattr(pipeline, "refresh_all", fake_refresh_all)
    at = _select(app)
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert calls == [(("SYN",), None)]
    assert "fake refresh ran" in at.session_state["refresh_log"]
    at.sidebar.checkbox(key="with_bt").check().run()
    at.sidebar.radio(key="scope").set_value("All tickers + market").run()
    at.sidebar.button[0].click().run()
    assert calls[-1][1] == 0 and "SYN" in calls[-1][0] and "SPY" in calls[-1][0]
