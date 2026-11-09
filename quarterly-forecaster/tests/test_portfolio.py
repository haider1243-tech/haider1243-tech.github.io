import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import authed_app

from forecaster.pipeline import _jsonable, write_json
from forecaster.portfolio import (
    CashBalance, Holding, Portfolio, PortfolioError, aligned_prices, allocation, build_snapshot, classify,
    concentration_flags, load_portfolio, parse_portfolio, risk_metrics, simulate_portfolio, valuation,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "portfolio.example.yaml"
ASOF = pd.Timestamp("2026-11-06")


def series(seed, n=1500, mu=0.0003, vol=0.01, start=100.0, shocks=None, end=ASOF):
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, vol, n) if shocks is None else mu + vol * shocks
    return pd.Series(start * np.exp(np.cumsum(r)), index=idx)


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(0)
    n = 1500
    common = rng.normal(0, 1, n)
    def eq(k, w=0.8):
        return w * common + math.sqrt(1 - w * w) * np.random.default_rng(k).normal(0, 1, n)
    native = {
        "XEQT.TO": series(1, shocks=eq(1), start=30), "VFV.TO": series(2, shocks=eq(2, 0.9), start=130),
        "XIU.TO": series(3, shocks=eq(3), start=34), "ZAG.TO": series(4, vol=0.003, start=14),
        "SPY": series(5, shocks=eq(5, 0.9), start=600),
    }
    rates = pd.Series(1.35 + 0.02 * np.sin(np.arange(n) / 50), index=native["SPY"].index)
    bench = {"XIU.TO": native["XIU.TO"], "SPY": native["SPY"]}
    return native, rates, bench


def h(t, shares=10, cost=10.0, account="TFSA", **kw):
    from forecaster.market_data import ticker_currency
    return Holding(t, shares, cost, account, kw.pop("cost_currency", ticker_currency(t)), **kw)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def test_example_file_loads():
    pf = load_portfolio(EXAMPLE)
    assert {x.account for x in pf.holdings} == {"TFSA", "RRSP", "FHSA", "cash"}
    spy = next(x for x in pf.holdings if x.ticker == "SPY")
    assert spy.cost_currency == "USD" and spy.cost_fx == 1.37
    assert next(x for x in pf.holdings if x.ticker == "XEQT.TO").cost_currency == "CAD"
    assert pf.cash[0] == CashBalance("TFSA", 750.0, "CAD") and pf.concentration_limit == 0.25


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="portfolio.example.yaml"):
        load_portfolio(tmp_path / "portfolio.yaml")


@pytest.mark.parametrize("bad,msg", [
    ({"holdings": [{"ticker": "XEQT.TO", "shares": 1, "avg_cost": 1, "account": "LIRA"}]}, "account must be"),
    ({"holdings": [{"ticker": "XEQT.TO", "shares": -1, "avg_cost": 1, "account": "TFSA"}]}, "shares must be > 0"),
    ({"holdings": [{"ticker": "XEQT.TO", "shares": "x", "avg_cost": 1, "account": "TFSA"}]}, "numeric"),
    ({"holdings": [{"ticker": "^VIX", "shares": 1, "avg_cost": 1, "account": "TFSA"}]}, "not something you can hold"),
    ({"holdings": [{"ticker": "SPY", "shares": 1, "avg_cost": 1, "account": "TFSA", "cost_fx": 0.73}]}, "cost_fx"),
    ({"holdings": [{"ticker": "SPY", "shares": 1, "avg_cost": 1, "account": "TFSA",
                    "sectors": {"Tech": 0.8, "Energy": 0.5}}]}, "sectors"),
    ({"holdings": []}, "empty"),
    ({}, "holdings"),
])
def test_validation(bad, msg):
    with pytest.raises(PortfolioError, match=msg):
        parse_portfolio(bad)


def test_account_names_case_insensitive():
    pf = parse_portfolio({"holdings": [{"ticker": "xeqt.to", "shares": 1, "avg_cost": 1, "account": "tfsa"}]})
    assert pf.holdings[0].account == "TFSA" and pf.holdings[0].ticker == "XEQT.TO"


# --------------------------------------------------------------------------- #
# Valuation
# --------------------------------------------------------------------------- #

def test_valuation_cad_and_usd_cases():
    pf = Portfolio((h("XEQT.TO", 100, 25.0), h("SPY", 2, 500.0, cost_fx=1.30),
                    h("SPY", 1, 500.0, account="RRSP"), h("SPY", 1, 700.0, account="cash", cost_currency="CAD")),
                   (CashBalance("TFSA", 100.0), CashBalance("cash", 100.0, "USD")))
    v = valuation(pf, {"XEQT.TO": 30.0, "SPY": 600.0}, usd_cad=1.40).set_index(["ticker", "account"])
    assert v.loc[("XEQT.TO", "TFSA"), "market_value"] == 3000 and v.loc[("XEQT.TO", "TFSA"), "pnl"] == 500
    # USD cost with purchase FX: full CAD P&L incl. currency gain
    assert v.loc[("SPY", "TFSA"), "market_value"] == pytest.approx(2 * 600 * 1.40)
    assert v.loc[("SPY", "TFSA"), "pnl"] == pytest.approx(2 * 600 * 1.40 - 2 * 500 * 1.30)
    assert v.loc[("SPY", "TFSA"), "note"] == ""
    # USD cost without FX: converted at today's rate, labelled
    assert v.loc[("SPY", "RRSP"), "pnl"] == pytest.approx((600 - 500) * 1.40)
    assert "excludes FX" in v.loc[("SPY", "RRSP"), "note"]
    # CAD ACB for a US holding
    assert v.loc[("SPY", "cash"), "pnl"] == pytest.approx(600 * 1.40 - 700)
    cash = v.loc["CASH"]
    assert sorted(cash["market_value"]) == [100.0, 140.0]
    assert v["weight"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Allocation & flags
# --------------------------------------------------------------------------- #

def test_classify_priority():
    m = classify(h("XEQT.TO"), {"asset_class": "Fixed income", "sectors": {"Technology": 0.25}})
    assert m["asset_class"] == "Equity"                       # built-in beats fetched
    assert m["sectors"] == {"Technology": 0.25} and m["diversified"]
    o = classify(h("XEQT.TO", sectors={"Financials": 1.0}, region="Mars"))
    assert o["sectors"] == {"Financials": 1.0} and o["region"] == "Mars"  # overrides win
    u = classify(h("ACME"))
    assert u["asset_class"] == "Unclassified" and u["sectors"] == {}
    assert classify(h("ZAG.TO"))["sectors"] == {"Fixed income": 1.0}


def test_allocation_sums_to_one_and_looks_through():
    pf = Portfolio((h("XEQT.TO", 100, 1), h("ZAG.TO", 100, 1), h("XLK", 1, 1)), (CashBalance("TFSA", 1000.0),))
    v = valuation(pf, {"XEQT.TO": 30, "ZAG.TO": 15, "XLK": 1000}, 1.0)
    meta = {x.ticker: classify(x, {"sectors": {"Technology": 0.3, "Financials": 0.2}} if x.ticker == "XEQT.TO" else None)
            for x in pf.holdings}
    a = allocation(v, meta)
    for k in ("asset_class", "region", "sector", "account"):
        assert a[k].sum() == pytest.approx(1.0), k
    total = 3000 + 1500 + 1000 + 1000
    assert a["asset_class"]["Equity"] == pytest.approx(4000 / total)
    assert a["asset_class"]["Cash"] == pytest.approx(1000 / total)
    assert a["sector"]["Technology"] == pytest.approx((3000 * 0.3 + 1000) / total)
    assert a["sector"]["Unclassified"] == pytest.approx(3000 * 0.5 / total)
    assert a["sector"]["Fixed income"] == pytest.approx(1500 / total)


def test_concentration_flags():
    pf = Portfolio((h("XEQT.TO", 50, 1, "TFSA"), h("XEQT.TO", 50, 1, "RRSP"), h("ACME", 25, 1), h("ZAG.TO", 25, 1)))
    v = valuation(pf, {"XEQT.TO": 1, "ACME": 1, "ZAG.TO": 1}, 1.0)         # 100 / 25 / 25 of 150
    meta = {x.ticker: classify(x) for x in pf.holdings}
    f = concentration_flags(v, meta, 0.25)
    assert [x["ticker"] for x in f] == ["XEQT.TO"]                         # summed across accounts
    assert "diversified fund" in f[0]["message"] and f[0]["weight"] == pytest.approx(2 / 3)
    # exactly at the limit is not flagged
    pf2 = Portfolio((h("ACME", 25, 1), h("ZAG.TO", 75, 1)))
    v2 = valuation(pf2, {"ACME": 1, "ZAG.TO": 1}, 1.0)
    f2 = concentration_flags(v2, {x.ticker: classify(x) for x in pf2.holdings}, 0.25)
    assert [x["ticker"] for x in f2] == ["ZAG.TO"] and "diversified" in f2[0]["message"]
    pf3 = Portfolio((h("ACME", 30, 1), h("ZAG.TO", 70, 1)))
    f3 = concentration_flags(valuation(pf3, {"ACME": 1, "ZAG.TO": 1}, 1.0),
                             {x.ticker: classify(x) for x in pf3.holdings}, 0.25)
    acme = next(x for x in f3 if x["ticker"] == "ACME")
    assert "diversified" not in acme["message"]


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #

def test_aligned_prices_inner_join():
    a = pd.Series([1.0, 2, 3], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    b = pd.Series([1.0, 2], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))   # holiday on the 3rd
    assert list(aligned_prices({"a": a, "b": b}).index) == list(pd.to_datetime(["2024-01-02", "2024-01-04"]))


def test_risk_metrics_vol_formula_and_beta(market):
    native, rates, bench = market
    w = pd.Series({"XEQT.TO": 0.5, "ZAG.TO": 0.3, "CASH": 0.2})
    r = risk_metrics(native, w, bench)
    rets = aligned_prices({t: native[t] for t in ("XEQT.TO", "ZAG.TO")}).pct_change().dropna().tail(756)
    ww = np.array([0.5, 0.3])
    assert r["volatility"] == pytest.approx(math.sqrt(ww @ (rets.cov().to_numpy() * 252) @ ww), rel=1e-9)
    assert sum(x["risk_share"] for x in r["holdings"]) == pytest.approx(1.0)
    # holding the benchmark outright -> beta exactly 1
    one = risk_metrics(native, pd.Series({"XIU.TO": 1.0}), bench)
    assert one["beta"]["XIU.TO"] == pytest.approx(1.0)
    # cash dilutes beta proportionally
    half = risk_metrics(native, pd.Series({"XIU.TO": 0.5, "CASH": 0.5}), bench)
    assert half["beta"]["XIU.TO"] == pytest.approx(0.5) and half["volatility"] == pytest.approx(one["volatility"] / 2)
    assert all(math.copysign(1, x["beta_SPY"]) == 1 or x["beta_SPY"] != 0 for x in r["holdings"])  # no -0.0


def test_risk_needs_enough_history(market):
    native, _, bench = market
    short = {"XEQT.TO": native["XEQT.TO"].tail(30)}
    with pytest.raises(ValueError, match="60"):
        risk_metrics(short, pd.Series({"XEQT.TO": 1.0}), bench)


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #

def test_simulation_contract(market):
    native, _, _ = market
    sim = simulate_portfolio(native, {"XEQT.TO": 100, "ZAG.TO": 200}, 500.0, ASOF, n_paths=3000)
    assert sim["labels"] == ["2027Q1", "2027Q2", "2027Q3", "2027Q4"]
    assert all(a < b < c for a, b, c in zip(sim["p10"], sim["p50"], sim["p90"]))
    widths = np.subtract(sim["p90"], sim["p10"])
    assert (np.diff(widths) > 0).all()
    assert (np.diff(sim["prob_drawdown"]) >= 0).all() and all(0 <= p <= 1 for p in sim["prob_loss"])
    again = simulate_portfolio(native, {"XEQT.TO": 100, "ZAG.TO": 200}, 500.0, ASOF, n_paths=3000)
    assert again["p50"] == sim["p50"]                                       # seeded


def test_cash_only_is_constant():
    sim = simulate_portfolio({}, {}, 1000.0, ASOF)
    assert sim["p10"] == sim["p90"] == [1000.0] * 4 and sim["prob_drawdown"] == [0.0] * 4


def test_correlation_matters():
    """Two perfectly correlated assets spread more than two independent ones."""
    n = 2000
    z1 = np.random.default_rng(11).normal(0, 1, n)
    z2 = np.random.default_rng(12).normal(0, 1, n)
    a = series(0, n=n, mu=0, shocks=z1)
    same = {"A": a, "B": series(0, n=n, mu=0, shocks=z1)}
    indep = {"A": a, "B": series(0, n=n, mu=0, shocks=z2)}
    s1 = simulate_portfolio(same, {"A": 50, "B": 50}, 0.0, ASOF, n_paths=4000)
    s2 = simulate_portfolio(indep, {"A": 50, "B": 50}, 0.0, ASOF, n_paths=4000)
    w1 = (s1["p90"][-1] - s1["p10"][-1]) / s1["start_value"]
    w2 = (s2["p90"][-1] - s2["p10"][-1]) / s2["start_value"]
    assert w1 > 1.25 * w2                                                    # ~sqrt(2) in theory


# --------------------------------------------------------------------------- #
# Snapshot + app
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def snapshot(market):
    native, rates, bench = market
    pf = load_portfolio(EXAMPLE)
    nat = {t: native[t] for t in pf.tickers}
    return build_snapshot(pf, nat, rates, bench, {"XEQT.TO": {"sectors": {"Technology": 0.25, "Financials": 0.2}}},
                          n_paths=2000)


def test_snapshot_contents(snapshot):
    from forecaster.dashboard import PALETTES, fan_chart
    s = snapshot
    fig = fan_chart(s, PALETTES["light"])
    assert fig.layout.yaxis.title.text == "Value (CAD)" and fig.data[0].name == "Weekly value"
    json.dumps(_jsonable(s), allow_nan=False)
    assert s["currency"] == "CAD" and s["total_value"] == pytest.approx(sum(x["market_value"] for x in s["holdings"]))
    spy = next(x for x in s["holdings"] if x["ticker"] == "SPY")
    assert spy["market_value"] == pytest.approx(spy["shares"] * spy["price"] * s["usd_cad"])
    assert len(s["forecast"]) == 4 and s["last_close"] == s["total_value"]
    assert s["risk"]["volatility"] > 0 and set(s["risk"]["beta"]) == {"XIU.TO", "SPY"}
    assert any(f["ticker"] == "XEQT.TO" for f in s["flags"])                 # 400 x ~30 is > 25%
    assert any("no sector data" in w for w in s["warnings"])                 # XIU/VFV/SPY have none here
    total = sum(a["weight"] for a in s["allocation"]["sector"])
    assert total == pytest.approx(1.0)


def test_portfolio_page_renders(snapshot, tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from streamlit.testing.v1 import AppTest
    (tmp_path / "snapshots").mkdir()
    write_json(tmp_path / "snapshots" / "portfolio.json", snapshot)
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(tmp_path))
    app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = authed_app(app, default_timeout=60).run()
    at.sidebar.radio(key="page").set_value("Portfolio").run()
    assert not at.exception, at.exception
    txt = "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning]
                    + [m.label for m in at.metric] + [m.value for m in at.metric])
    assert "Portfolio (CAD)" in [x.value for x in at.subheader]
    assert "Market value" in txt and "C$" in txt and "Beta vs S&P/TSX 60" in txt
    assert "XEQT.TO is" in txt and "limit 25%" in txt                         # concentration flag
    assert "Backtest accuracy" in txt                                          # banner on this page too
    assert "hasn't been backtested" in txt
    tables = [d.value for d in at.dataframe]
    assert any("Value (CAD)" in t.columns for t in tables) and any("P50 (CAD)" in t.columns for t in tables)


def test_portfolio_page_without_snapshot(tmp_path, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FORECASTER_CACHE_DIR", str(tmp_path))
    app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = authed_app(app, default_timeout=60).run()
    at.sidebar.radio(key="page").set_value("Portfolio").run()
    assert not at.exception
    assert any("portfolio.example.yaml" in i.value for i in at.info)
