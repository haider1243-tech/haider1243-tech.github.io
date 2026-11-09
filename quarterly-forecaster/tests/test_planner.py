import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import authed_app

from forecaster.pipeline import write_json
from forecaster.planner import (
    PlannerConfig, PlannerError, check_instrument, months_to_target, normalise_targets, plan_contribution,
    risk_flags, tranche_schedule, waterfill,
)

TODAY = pd.Timestamp("2026-11-09")   # Monday


# --------------------------------------------------------------------------- #
# Never leverage / options
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("t,name", [
    ("TQQQ", None), ("SQQQ", None), ("HXU.TO", None), ("HQU.TO", None), ("UVXY", None),
    ("ABC", "BetaPro S&P 500 2x Daily Bull ETF"), ("DEF", "ProShares UltraPro QQQ"),
    ("GHI", "Direxion Daily Semiconductor Bear 3X"), ("JKL", "Some Leveraged Gold Fund"),
    ("MNO", "XYZ -1x Inverse S&P ETF"),
    ("SPY241220C00500000", None), ("XIU261218P00030000", None),
    ("GC=F", None), ("CAD=X", None), ("^GSPC", None),
])
def test_refuses_leverage_options_futures(t, name):
    with pytest.raises(PlannerError):
        check_instrument(t, name)


@pytest.mark.parametrize("t,name", [
    ("XEQT.TO", "iShares Core Equity ETF Portfolio"), ("XSB.TO", "iShares Core Canadian Short Term Bond Index ETF"),
    ("JPST", "JPMorgan Ultra-Short Income ETF"), ("ZAG.TO", "BMO Aggregate Bond Index ETF"),
    ("XBB.TO", "iShares Core Canadian Universe Bond"), ("VFV.TO", None),
])
def test_ordinary_funds_allowed(t, name):
    check_instrument(t, name)


def test_plan_rejects_leveraged_target():
    with pytest.raises(PlannerError, match="leverage"):
        plan_contribution({"XEQT.TO": 1000}, {"XEQT.TO": 50, "TQQQ": 50}, 500)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #

def test_normalise_targets_percent_or_fraction():
    assert normalise_targets({"a": 60, "b": 40}) == pytest.approx({"A": 0.6, "B": 0.4})
    assert normalise_targets({"a": 0.6, "b": 0.4}) == pytest.approx({"A": 0.6, "B": 0.4})
    assert normalise_targets({"a": 100, "b": 0}) == {"A": 1.0}
    for bad in ({}, {"a": 60, "b": 30}, {"a": -10, "b": 110}, {"a": 0.5}):
        with pytest.raises(PlannerError):
            normalise_targets(bad)


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #

def test_waterfill_least_squares_buy_only():
    x = waterfill(np.array([300.0, -100.0, 50.0]), 200.0)
    # level 100 > the 50 gap, so only the biggest gap gets money
    assert list(x) == pytest.approx([200.0, 0.0, 0.0])
    x2 = waterfill(np.array([300.0, -100.0, 50.0]), 300.0)
    assert list(x2) == pytest.approx([275.0, 0.0, 25.0])           # level 25: both buys end equally short


def test_split_fills_the_most_underweight_first():
    # 60/20/20 target, total after = 10,000 -> targets 6000/2000/2000
    p = plan_contribution({"XEQT.TO": 5000, "VFV.TO": 2500, "ZAG.TO": 1500}, {"XEQT.TO": 60, "VFV.TO": 20, "ZAG.TO": 20}, 1000)
    buy = dict(zip(p.rows["ticker"], p.rows["buy"]))
    assert sum(buy.values()) == pytest.approx(1000) and min(buy.values()) >= 0
    assert buy["VFV.TO"] == 0                                       # overweight: never bought, never sold
    # least squares: gaps 1000 / 500 -> level 250 -> buys 750 / 250
    assert buy["XEQT.TO"] == pytest.approx(750) and buy["ZAG.TO"] == pytest.approx(250)
    assert p.max_drift_after < p.max_drift_before
    # and it beats the naive "split the deficits" 500/500 on squared distance to target
    sq = lambda b: sum((v - t) ** 2 for v, t in zip([5000 + b[0], 2500, 1500 + b[1]], [6000, 2000, 2000]))  # noqa: E731
    assert sq((750, 250)) < sq((500, 500))


def test_contribution_larger_than_gaps_spreads_by_target():
    p = plan_contribution({"A": 600, "B": 400}, {"A": 60, "B": 40}, 1000)       # already on target
    buy = dict(zip(p.rows["ticker"], p.rows["buy"]))
    assert buy["A"] == pytest.approx(600) and buy["B"] == pytest.approx(400)
    assert p.max_drift_after == pytest.approx(0, abs=1e-9) and p.months_to_target == 0


def test_new_ticker_and_holdings_outside_targets():
    p = plan_contribution({"XEQT.TO": 9000, "OLD": 1000}, {"XEQT.TO": 80, "ZAG.TO": 20}, 1000)
    buy = dict(zip(p.rows["ticker"], p.rows["buy"]))
    assert buy["ZAG.TO"] == pytest.approx(1000)                     # target 20% of 10,000 = 2,000 > 1,000
    assert p.outside_targets == ["OLD"] and "never sells" in p.notes[0]


@pytest.mark.parametrize("amount", [1.0, 123.45, 999.99, 25_000.0])
@pytest.mark.parametrize("seed", range(5))
def test_buys_never_negative_and_sum_exactly(amount, seed):
    rng = np.random.default_rng(seed)
    cur = {f"T{i}": float(v) for i, v in enumerate(rng.uniform(0, 20_000, 5))}
    w = rng.dirichlet(np.ones(5))
    tg = {f"T{i}": float(x) for i, x in enumerate(w)}
    p = plan_contribution(cur, tg, amount)
    assert (p.rows["buy"] >= -1e-9).all()
    assert p.rows["buy"].sum() == pytest.approx(amount, rel=1e-9)


def test_whole_shares():
    p = plan_contribution({"XEQT.TO": 5000, "ZAG.TO": 1500}, {"XEQT.TO": 70, "ZAG.TO": 30}, 1000,
                          prices={"XEQT.TO": 33.0, "ZAG.TO": 14.0}, whole_shares=True)
    r = p.rows.set_index("ticker")
    assert all(float(s).is_integer() for s in r["shares"])
    assert (r["buy"] == r["shares"] * r["price"]).all()
    assert r["buy"].sum() + p.leftover == pytest.approx(1000)
    assert 0 <= p.leftover < 14.0                                   # can't afford even the cheapest share
    with pytest.raises(PlannerError, match="price"):
        plan_contribution({"A": 1}, {"A": 100}, 100, prices={}, whole_shares=True)


def test_invalid_amount():
    for a in (0, -5, float("nan")):
        with pytest.raises(PlannerError):
            plan_contribution({"A": 1}, {"A": 100}, a)


def test_months_to_target():
    m = months_to_target(np.array([9000.0, 1000.0]), np.array([0.5, 0.5]), 1000, 0.01)
    assert m == 8                                                   # B needs ~8 months of full contributions
    assert months_to_target(np.array([1e9, 0.0]), np.array([0.5, 0.5]), 1, 0.01) is None


# --------------------------------------------------------------------------- #
# Flags and tranches
# --------------------------------------------------------------------------- #

def _snap(t, dd=0.1, hv=0.1, close=100.0):
    return {"ticker": t, "last_close": close, "forecast": [{"quarter": "2027Q1", "prob_drawdown": dd}],
            "regime": {"current": {"bull": 0.8, "bear": 0.1, "high_vol": hv}}}


def _market(*evs):
    return {"events": [{"date": d, "name": n, "category": n.split()[0], "impact": "high", "points": p}
                       for d, n, p in evs]}


def test_no_flags_when_calm():
    snaps = {"XEQT.TO": _snap("XEQT.TO"), "^VIX": _snap("^VIX", close=15)}
    assert risk_flags(["XEQT.TO"], snaps, {"forecast": [{"quarter": "2027Q1", "prob_drawdown": 0.1}]},
                      _market(("2026-11-11", "CPI release", 3)), TODAY) == []


def test_flags_drawdown_regime_vix_events():
    snaps = {"XEQT.TO": _snap("XEQT.TO", dd=0.4, hv=0.6), "^VIX": _snap("^VIX", close=28)}
    pf = {"forecast": [{"quarter": "2027Q1", "prob_drawdown": 0.36}]}
    mkt = _market(("2026-11-10", "CPI release", 3), ("2026-11-13", "Employment Situation", 3),
                  ("2026-11-30", "FOMC rate decision", 4))
    kinds = [f["kind"] for f in risk_flags(["XEQT.TO"], snaps, pf, mkt, TODAY)]
    assert kinds == ["drawdown", "drawdown", "regime", "vix", "events"]
    ev = next(f for f in risk_flags(["XEQT.TO"], snaps, pf, mkt, TODAY) if f["kind"] == "events")
    assert "6 risk points" in ev["message"] and "FOMC" not in ev["message"]       # FOMC is outside 7 days


def test_tranches_avoid_event_days_and_sum_exactly():
    mkt = _market(("2026-11-10", "CPI release", 3), ("2026-11-17", "PPI release", 3))
    t = tranche_schedule(1000.0, TODAY, mkt, n=3)
    assert [d["date"] for d in t] == [pd.Timestamp("2026-11-11"), pd.Timestamp("2026-11-18"), pd.Timestamp("2026-11-24")]
    assert sum(d["amount"] for d in t) == pytest.approx(1000.0)
    t2 = tranche_schedule(100.0, TODAY, None, n=3)
    assert [d["amount"] for d in t2] == [33.34, 33.33, 33.33]


# --------------------------------------------------------------------------- #
# App page
# --------------------------------------------------------------------------- #

@pytest.fixture
def app_with_portfolio(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from streamlit.testing.v1 import AppTest
    holdings = [
        {"ticker": "XEQT.TO", "account": "TFSA", "market_value": 5000.0, "price_cad": 33.0, "name": "XEQT", "weight": 0.5},
        {"ticker": "ZAG.TO", "account": "RRSP", "market_value": 1500.0, "price_cad": 14.0, "name": "ZAG", "weight": 0.15},
        {"ticker": "VFV.TO", "account": "RRSP", "market_value": 3500.0, "price_cad": 140.0, "name": "VFV", "weight": 0.35},
    ]
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    write_json(snaps / "portfolio.json", {"holdings": holdings, "planner": {"monthly_amount": 1000.0,
               "targets": {"XEQT.TO": 60, "ZAG.TO": 20, "VFV.TO": 20}},
               "forecast": [{"quarter": "2027Q1", "prob_drawdown": 0.12}]})
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(tmp_path))
    at = authed_app(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    at.sidebar.radio(key="page").set_value("Contribution planner").run()
    assert not at.exception, at.exception
    return at, snaps


def test_planner_page(app_with_portfolio):
    at, _ = app_with_portfolio
    assert "Contribution planner (CAD)" in [s.value for s in at.subheader]
    t = next(d.value for d in at.dataframe if "Buy (CAD)" in d.value.columns)
    buy = dict(zip(t["Ticker"], t["Buy (CAD)"]))
    assert sum(buy.values()) == pytest.approx(1000) and buy["VFV.TO"] == 0
    assert any("never uses leverage" in c.value for c in at.caption)
    assert at.success and "one buy is fine" in at.success[0].value


def test_planner_page_flags_and_tranches(app_with_portfolio):
    at, snaps = app_with_portfolio
    write_json(snaps / "ticker_VIX.json", _snap("^VIX", close=31.0))
    at.run()
    assert any("VIX 31.0" in w.value for w in at.warning)
    assert any("weekly parts" in m.value for m in at.markdown)
    sched = at.dataframe[-1].value
    assert len(sched) == 3 and sched["Amount (CAD)"].sum() == pytest.approx(1000)


def test_planner_page_bad_targets(app_with_portfolio):
    at, _ = app_with_portfolio
    at.number_input(key="pl_amount").set_value(0.0).run()
    assert at.error and "positive" in at.error[0].value
