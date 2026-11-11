"""Simulator: position sizing, cost math, the random-entry comparison, filters/overrides, scenarios."""
import numpy as np
import pandas as pd
import pytest

from forecaster import intraday as I
from forecaster import lab_engine as E
from forecaster import simulator as SIM
from forecaster import strategies as S

SPY = I.INSTRUMENTS["SPY"]


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #

def test_fixed_cash_units_floor_and_cap():
    sz = SIM.Sizing("fixed_cash", 5000)
    assert SIM.size_units(sz, 100_000, 5000, 600.0, 590.0) == 8                    # floor(5000 / 600)
    assert SIM.size_units(sz, 3000, 5000, 600.0, 590.0) == 5                       # capped by equity (no leverage)
    assert SIM.size_units(sz, 100_000, 500, 600.0, 590.0) == 0                     # too small for one share


def test_fixed_units_and_cap():
    assert SIM.size_units(SIM.Sizing("fixed_units", 10.7), 100_000, 0, 50.0, 49.0) == 10
    assert SIM.size_units(SIM.Sizing("fixed_units", 100), 1000, 0, 50.0, 49.0) == 20


def test_risk_pct_sizes_to_the_stop():
    sz = SIM.Sizing("risk_pct", 1.0)
    # 1% of 10,000 = 100 at risk; 2.00 per share to the stop -> 50 shares (5,000 notional, under equity)
    assert SIM.size_units(sz, 10_000, 0, 100.0, 98.0) == 50
    assert SIM.size_units(sz, 10_000, 0, 100.0, 102.0) == 50                       # short: same distance
    assert SIM.size_units(sz, 10_000, 0, 100.0, 99.9) == 100                       # tight stop -> capped at 1x equity
    assert SIM.size_units(SIM.Sizing("risk_pct", 1.0, max_leverage=2.0), 10_000, 0, 100.0, 99.9) == 200
    assert SIM.size_units(sz, 10_000, 0, 100.0, 100.0) == 0                        # no distance, no trade
    with pytest.raises(ValueError):
        SIM.size_units(SIM.Sizing("martingale", 1), 10_000, 0, 100.0, 99.0)


# --------------------------------------------------------------------------- #
# Cost math
# --------------------------------------------------------------------------- #

def test_commission_ibkr_style_minimum():
    c = SIM.default_costs("SPY")
    assert c.commission(100) == 1.0                    # 100 x 0.005 = 0.50 -> US$1 minimum
    assert c.commission(1000) == pytest.approx(5.0)
    assert c.commission(0) == 0.0
    flat = SIM.Costs(commission_per_order=4.95)
    assert flat.commission(1) == 4.95 and flat.commission(500) == 4.95
    assert SIM.default_costs("XAU_USD").commission(10) == 0.0 and SIM.default_costs("XAU_USD").spread == 0.35


def engine_trade(**kw):
    t = {"day": pd.Timestamp("2024-03-04"), "side": 1, "entry_ref": 100.0, "entry": 100.02, "exit": 101.98,
         "stop": 98.0, "target": 102.0, "reason": "target", "bars_held": 4,
         "entry_time": pd.Timestamp("2024-03-04 15:00", tz="UTC"), "exit_time": pd.Timestamp("2024-03-04 15:20", tz="UTC")}
    t.update(kw)
    return t


def test_account_books_commission_spread_and_fx():
    trades = pd.DataFrame([engine_trade()])
    cfg = SIM.SimConfig(symbol="SPY", balance_cad=50_000, sizing=SIM.Sizing("fixed_units", 100),
                        costs=SIM.Costs(commission_per_unit=0.005, min_commission=1.0))
    log, skipped = SIM.account(trades, cfg, fx=1.37)
    r = log.iloc[0]
    assert r.units == 100 and r.commission == pytest.approx(2.0)                   # US$1 min, both orders
    assert r.pnl_quote == pytest.approx((101.98 - 100.02) * 100 - 2.0)            # fills already include spread+slip
    assert r.pnl_cad == pytest.approx(r.pnl_quote * 1.37)
    assert r.equity == pytest.approx(50_000 + r.pnl_cad)
    assert r.ret == pytest.approx(r.pnl_quote / (100.0 * 100))


def test_account_cad_quoted_instrument_needs_no_fx():
    t = pd.DataFrame([engine_trade(entry_ref=1.38, entry=1.3801, exit=1.3851, stop=1.37, target=1.39)])
    cfg = SIM.SimConfig(symbol="USD_CAD", balance_cad=100_000, sizing=SIM.Sizing("fixed_units", 10_000))
    log, _ = SIM.account(t, cfg, fx=1.37)
    assert log.iloc[0].fx == 1.0 and log.iloc[0].pnl_cad == pytest.approx(50.0)    # 0.005 x 10,000 CAD


def test_account_risk_sizing_compounds_and_converts_cad_to_usd():
    trades = pd.DataFrame([engine_trade(), engine_trade(entry_time=pd.Timestamp("2024-03-04 16:00", tz="UTC"))])
    cfg = SIM.SimConfig(symbol="SPY", balance_cad=13_700, sizing=SIM.Sizing("risk_pct", 1.0), costs=SIM.Costs())
    log, _ = SIM.account(trades, cfg, fx=1.37)
    # equity 13,700 CAD = 10,000 USD; 1% = 100 USD risk; 2.02 per share to the stop -> 49 shares
    assert log.iloc[0].units == 49
    assert log.iloc[1].units == int(log.iloc[0].equity / 1.37 * 0.01 / 2.02)       # sized on the new equity


def test_daily_loss_limit_stops_the_day():
    loss = dict(exit=99.0, reason="stop")
    rows = [engine_trade(**loss, entry_time=pd.Timestamp(f"2024-03-04 1{h}:00", tz="UTC")) for h in (4, 5, 6)]
    rows.append(engine_trade(day=pd.Timestamp("2024-03-05")))
    cfg = SIM.SimConfig(symbol="SPY", balance_cad=50_000, sizing=SIM.Sizing("fixed_units", 100), costs=SIM.Costs(),
                        max_daily_loss_cad=150)
    log, skipped = SIM.account(pd.DataFrame(rows), cfg, fx=1.0)
    assert len(log) == 3 and skipped["daily loss limit"] == 1                       # -102, -204: third blocked
    assert log.iloc[-1].day == pd.Timestamp("2024-03-05")                          # next day trades again


def test_fx_lookup_uses_last_rate_on_or_before():
    fx = pd.Series([1.30, 1.40], index=pd.to_datetime(["2024-03-01", "2024-03-05"]))
    at = SIM.fx_lookup(fx, "USD")
    assert at("2024-03-04") == 1.30 and at("2024-03-05") == 1.40 and at("2024-02-01") == 1.30
    assert SIM.fx_lookup(fx, "CAD")("2024-03-04") == 1.0


# --------------------------------------------------------------------------- #
# Engine extensions, filters, overrides
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def spy5():
    m1 = I.session_bars(I.synthetic_m1("SPY", "2023-01-02", "2024-06-28", seed=11), SPY)
    return m1, I.resample(m1, 5, SPY)


def test_max_hold_and_overrides(spy5):
    _, bars = spy5
    f = S.Features(bars, SPY, 5)
    sig = S.LIBRARY["ema_cross"].signals(f)
    pp = E.prepare(bars, SPY, 5, E.CostModel())
    held = E.simulate(pp, sig, E.CostModel(), max_hold=3)
    assert len(held) and (held.bars_held <= 3).all() and set(held.reason) <= {"stop", "target", "time"}
    cfg = SIM.SimConfig(stop_pct=0.5, target_pct=1.0)
    adj = SIM.adjust_signals(sig, bars, cfg, None, 5)
    ref = bars["Close"].to_numpy()[adj.index]
    np.testing.assert_allclose(adj["stop"], ref * (1 - adj["side"] * 0.005))
    np.testing.assert_allclose(adj["target"], ref * (1 + adj["side"] * 0.01))


def test_hours_weekday_event_and_date_filters(spy5):
    _, bars = spy5
    sig = S.LIBRARY["ema_cross"].signals(S.Features(bars, SPY, 5))
    cfg = SIM.SimConfig(hours=("09:30", "11:30"), weekdays=(0, 4), start="2023-06-01", end="2023-12-31")
    events = {"CPI": {"2023-06-13"}, "FOMC": set(), "PPI": set()}
    out = SIM.adjust_signals(sig, bars, SIM.SimConfig(**{**cfg.__dict__, "skip_events": True}), events, 5)
    t = bars.index[out.index + 1].tz_convert(I.NY)
    m = t.hour * 60 + t.minute
    assert len(out) > 10 and ((m >= 570) & (m < 690)).all() and set(t.dayofweek) <= {0, 4}
    assert t.min() >= pd.Timestamp("2023-06-01", tz=I.NY) and "2023-06-13" not in set(t.strftime("%Y-%m-%d"))


def test_daily_bars_trade_next_day_open_to_close(spy5):
    m1, _ = spy5
    daily, minutes = SIM.daily_window_bars(m1, SPY)
    assert minutes == 390 and daily.index[0].tz_convert(I.NY).strftime("%H:%M") == "09:30"
    cfg = SIM.SimConfig(symbol="SPY", strategy="ema_cross", timeframe="1d", params={"fast": 3, "slow": 8})
    res, log = SIM.run(cfg, daily, minutes, SPY, fx=1.0, n_random=200)
    assert len(log) > 5
    et, xt = pd.DatetimeIndex(log.entry_time).tz_convert(I.NY), pd.DatetimeIndex(log.exit_time).tz_convert(I.NY)
    assert (et.date == xt.date).all()                                               # in and out the same day


# --------------------------------------------------------------------------- #
# Random-entry comparison and the full run
# --------------------------------------------------------------------------- #

def test_random_walk_is_not_an_edge_and_mc_is_consistent(spy5):
    _, bars = spy5
    cfg = SIM.SimConfig(symbol="SPY", strategy="ema_cross", timeframe="5m", sizing=SIM.Sizing("fixed_units", 10))
    res, log = SIM.run(cfg, bars, 5, SPY, fx=1.37, n_random=500)
    h, e = res["headline"], res["edge"]
    assert h["trades"] == len(log) > 100 and e["verdict"] != "EDGE" and e["n_random"] == 500
    assert h["pnl_cad"] == pytest.approx(log["pnl_cad"].sum())
    assert res["equity"]["equity"][-1] == pytest.approx(cfg.balance_cad + h["pnl_cad"])
    mc = res["mc"]
    assert all(a <= b <= c for a, b, c in zip(mc["p5"], mc["p50"], mc["p95"]))
    assert mc["best"][-1] == pytest.approx(mc["worst"][-1])                         # same trades, any order: same total
    assert sum(r["trades"] for r in res["by_weekday"]) == h["trades"] == sum(r["trades"] for r in res["by_hour"])
    assert res["buy_hold"]["units"] == int(10_000 / 1.37 // bars["Open"].iloc[np.flatnonzero(
        E.prepare(bars, SPY, 5, E.CostModel()).entry_ok)[0]])
    assert "consistent with luck" in res["summary"] or "isn't proven" in res["summary"]


def test_planted_pattern_is_an_edge():
    from test_lab import plant_momentum
    m1 = I.session_bars(plant_momentum(I.synthetic_m1("SPY", "2023-01-02", "2024-06-28", seed=5)), SPY)
    bars = I.resample(m1, 5, SPY)
    cfg = SIM.SimConfig(symbol="SPY", strategy="orb30", timeframe="5m", sizing=SIM.Sizing("risk_pct", 0.5))
    res, log = SIM.run(cfg, bars, 5, SPY, fx=1.37, n_random=500)
    assert res["edge"]["verdict"] == "EDGE" and res["edge"]["p_value"] < 0.01 and res["edge"]["ci"][1] > 0
    assert res["headline"]["pnl_cad"] > 0 and "statistically real" in res["summary"]


def test_random_entries_pay_the_same_commission(spy5):
    """The benchmark must carry the strategy's own costs: raising commission lowers both."""
    _, bars = spy5
    base = dict(symbol="SPY", strategy="ema_cross", timeframe="5m", sizing=SIM.Sizing("fixed_units", 10))
    cheap, _ = SIM.run(SIM.SimConfig(**base, costs=SIM.Costs(spread=0.01, slippage=0.01)), bars, 5, SPY, 1.0, n_random=400)
    dear, _ = SIM.run(SIM.SimConfig(**base, costs=SIM.Costs(commission_per_order=5, spread=0.01, slippage=0.01)),
                      bars, 5, SPY, 1.0, n_random=400)
    shift = 2 * 5 / 10 / 450                                                        # 2 orders x $5 / 10 shares / ~price
    d_strat = cheap["headline"]["expectancy_ret"] - dear["headline"]["expectancy_ret"]
    d_rand = cheap["edge"]["random_q"][50] - dear["edge"]["random_q"][50]
    assert d_strat == pytest.approx(d_rand, rel=0.25) and d_strat == pytest.approx(shift, rel=0.5)
    assert abs(cheap["edge"]["p_value"] - dear["edge"]["p_value"]) < 0.15           # costs don't fake an edge


def test_needs_100_trades_for_edge(spy5):
    _, bars = spy5
    cfg = SIM.SimConfig(symbol="SPY", strategy="gap_go", timeframe="5m", start="2024-01-01", end="2024-03-31")
    res, log = SIM.run(cfg, bars, 5, SPY, fx=1.0, n_random=200)
    assert len(log) < 100 and res["edge"]["verdict"] != "EDGE"


# --------------------------------------------------------------------------- #
# Scenarios, forward tests, CSV
# --------------------------------------------------------------------------- #

def test_scenarios_roundtrip_and_forward_tests(tmp_path):
    cfg = SIM.SimConfig(symbol="QQQ", strategy="vwap_breakout", params={"k": 1.5}, timeframe="15m",
                        hours=("09:30", "11:30"), weekdays=(1, 2), sizing=SIM.Sizing("fixed_cash", 2500),
                        costs=SIM.Costs(commission_per_order=1), stop_pct=0.4, max_hold_bars=6, name="QQQ morning")
    assert SIM.save_scenario(tmp_path, cfg) == "QQQ morning"
    back = SIM.load_scenarios(tmp_path)["QQQ morning"]
    assert back == cfg
    key = SIM.add_forward_test(tmp_path, cfg, {"edge": {"verdict": "WEAK", "ci": [0.0, 0.001]},
                                              "headline": {"expectancy_ret": 0.0004}})
    f = SIM.forward_tests(tmp_path)[0]
    assert f["key"] == key == "fwd:QQQ morning|QQQ|15m" and f["verdict"] == "WEAK" and f["sim"]["stop_pct"] == 0.4
    with pytest.raises(ValueError):
        SIM.add_forward_test(tmp_path, SIM.SimConfig(timeframe="1d"))
    SIM.remove_forward_test(tmp_path, key)
    SIM.delete_scenario(tmp_path, "QQQ morning")
    assert SIM.forward_tests(tmp_path) == [] and SIM.load_scenarios(tmp_path) == {}


def test_shadow_applies_forward_test_filters(tmp_path, spy5):
    from forecaster import shadow as SH
    m1, _ = spy5
    m1 = m1.loc["2024-05-01":]
    cfg = SIM.SimConfig(symbol="SPY", strategy="ema_cross", timeframe="5m", hours=("13:00", "15:00"), max_hold_bars=2)
    key = SIM.add_forward_test(tmp_path, cfg)
    setup = SIM.forward_tests(tmp_path)[0]
    trades, _ = SH.evaluate(m1, "SPY", "5m", [setup], m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1))
    assert trades and all(t["key"] == key for t in trades)
    et = pd.DatetimeIndex([t["entry_time"] for t in trades]).tz_convert(I.NY)
    assert ((et.hour >= 13) & (et.hour < 15)).all()
    held = (pd.DatetimeIndex([t["exit_time"] for t in trades]) - pd.DatetimeIndex([t["entry_time"] for t in trades]))
    assert (held <= pd.Timedelta(minutes=5)).all()                                   # 2 bars max: entry bar + 1


def test_trade_log_csv(spy5):
    _, bars = spy5
    res, log = SIM.run(SIM.SimConfig(strategy="ema_cross"), bars.iloc[:3000], 5, SPY, 1.0, n_random=50)
    csv = SIM.trade_log_csv(log).decode()
    head = csv.splitlines()[0]
    assert head.startswith("entry_time,exit_time,side,units") and "pnl_cad" in head
    assert len(csv.splitlines()) == len(log) + 1 and ",long," in csv or ",short," in csv


def test_simulator_never_touches_execution():
    from pathlib import Path
    src = Path(SIM.__file__).read_text().split('"""', 2)[2]
    assert "execution" not in src and "place_order" not in src
