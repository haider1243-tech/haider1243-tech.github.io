import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecaster.calendar import FOMC_DECISION_DATES
from forecaster.vwap_backtest import (
    FOMC_DECISIONS_NY, NY, VwapParams, compare_with_tradingview, compute_indicators,
    default_event_lines, execute, infer_timeframe, load_tradingview_csv, load_tv_trades,
    macro_flags, parse_events, pine_events_text, run_backtest, summarize, tv_atr, tv_ema, tv_rma,
)
from forecaster.webhook_server import Alert

PINE = Path(__file__).resolve().parents[1] / "pine" / "vwap_gold.pine"
OFF = dict(use_macro=False, use_events=False, use_volume=False, atr_len=2)


def bars(rows, start="2024-03-05 08:00", freq="15min", tz=NY):
    idx = pd.date_range(pd.Timestamp(start, tz=tz), periods=len(rows), freq=freq).tz_convert("UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx, dtype=float)


def frame(rows, sig=None, atr=1.0, flat=None, start="2024-03-05 08:00"):
    """Hand-built execution frame: rows of (o,h,l,c); sig = {i: 'long'|'short'}."""
    b = bars([(*r, 100) for r in rows], start=start)
    b["t_close"] = b.index + pd.Timedelta(minutes=15)
    b["atr"] = atr
    b["long_sig"] = [sig.get(i) == "long" for i in range(len(rows))] if sig else False
    b["short_sig"] = [sig.get(i) == "short" for i in range(len(rows))] if sig else False
    b["flat_window"] = [i in (flat or ()) for i in range(len(rows))]
    b["setup"] = "reclaim"
    return b


# --------------------------------------------------------------------------- #
# Pine <-> Python parity
# --------------------------------------------------------------------------- #

def test_pine_default_events_match_python():
    src = PINE.read_text()
    consts = dict(re.findall(r'const string (FOMC_\d{4}) = "([^"]*)"', src))
    order = re.search(r"const string FOMC_ALL = (.*)", src).group(1)
    names = re.findall(r"FOMC_\d{4}", order)
    pine_text = ",".join(consts[n] for n in names)
    assert len(names) == 7
    pd.testing.assert_index_equal(parse_events([pine_text]), parse_events(default_event_lines()))


def test_fomc_list_agrees_with_calendar_module():
    cal = {d for d in FOMC_DECISION_DATES}
    ours = {d for d in FOMC_DECISIONS_NY if d[:4] in ("2026", "2027")}
    assert ours == cal
    assert len(FOMC_DECISIONS_NY) == 56
    assert all(pd.Timestamp(d).day_of_week == 2 or d == "2024-11-07" for d in FOMC_DECISIONS_NY)  # Wed; 2024 election week Thu


def test_pine_script_has_required_parts():
    src = PINE.read_text()
    assert src.startswith("//@version=6")
    for needle in ('request.economic("US", cpiField', 'request.economic("US", ppiField',
                   "request.security(dxySym", "request.security(yldSym", "ta.atr(atrLen)",
                   "strategy.exit(", "alert(f_json(\"long\", close - stopDist", "alert(f_json(\"short\", close + stopDist",
                   "riskPct / 100.0 / (stopDist * syminfo.pointvalue)", "process_orders_on_close = false",
                   "immediately = true", '"DXY up"', '"US10Y up"', '"Econ release"'):
        assert needle in src, needle
    # session key formula identical to the Python port
    assert 'str.format_time(time + 6 * 3600000, "yyyy-MM-dd", "America/New_York")' in src


def test_pine_alert_json_matches_webhook_format():
    """Rebuild the Pine f_json message in Python and validate it with the server's model."""
    line = next(l for l in PINE.read_text().splitlines() if l.strip().startswith("'{\"ticker\""))
    keys = re.findall(r'"(\w+)":"', line)
    assert keys == ["ticker", "price", "strategy", "side", "time", "stop", "target", "secret"]
    msg = ('{"ticker":"' + "XAUUSD" + '","price":"' + "2650.43" + '","strategy":"' + "vwap_gold"
           + '","side":"' + "long" + '","time":"' + "2026-11-09" + "T" + "14:15:00" + "Z"
           + '","stop":"' + "2641.10" + '","target":"' + "2669.10" + '","secret":"' + "abc" + '"}')
    a = Alert.model_validate_json(msg)
    assert (a.ticker, a.price, a.side, a.time) == ("XAUUSD", 2650.43, "long", "2026-11-09T14:15:00Z")
    assert (a.stop, a.target) == (2641.10, 2669.10)


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #

def test_parse_events_dst_and_format_tolerance():
    ev = parse_events(["2024-06-12 14:00 FOMC, 2024-01-31 14:00", "2024-06-12 08:30 CPI\n bad-line ,2024-06-12 14:00"])
    assert list(ev) == [pd.Timestamp("2024-01-31 19:00", tz="UTC"),    # EST
                        pd.Timestamp("2024-06-12 12:30", tz="UTC"),    # EDT
                        pd.Timestamp("2024-06-12 18:00", tz="UTC")]
    assert pine_events_text(["2024-06-12 14:00 FOMC"]) == "2024-06-12 14:00"


def test_blackout_window_pre_and_post():
    p = VwapParams(**{**OFF, "use_events": True, "block_before_min": 30, "block_after_min": 60})
    b = bars([(100, 101, 99, 100, 1)] * 16, start="2024-06-12 07:00")      # closes 07:15 .. 11:00
    df = compute_indicators(b, p, parse_events(["2024-06-12 08:30"]))
    closes = pd.DatetimeIndex(df["t_close"]).tz_convert(NY).strftime("%H:%M")
    blocked = dict(zip(closes, df["blocked"]))
    assert not blocked["07:45"] and blocked["08:00"] and blocked["09:30"] and not blocked["09:45"]


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #

def test_tv_rma_and_ema():
    x = pd.Series([1.0, 2, 3, 4, 5])
    r = tv_rma(x, 3)
    assert np.isnan(r[0]) and np.isnan(r[1]) and r[2] == 2.0
    assert r[3] == pytest.approx(2 + (4 - 2) / 3) and r[4] == pytest.approx(r[3] + (5 - r[3]) / 3)
    e = tv_ema(x, 3)
    assert e[0] == 1.0 and e[1] == pytest.approx(0.5 * 2 + 0.5 * 1)


def test_atr_true_range():
    b = bars([(10, 11, 9, 10, 1), (12, 13, 11.5, 12, 1), (12, 12.5, 8, 9, 1)])
    a = tv_atr(b, 2)
    # TR: 2 (h-l, first bar), 3 (13-10), 4.5 (12.5-8 vs |8-12|=4)
    assert a.iloc[1] == pytest.approx(2.5) and a.iloc[2] == pytest.approx(2.5 + (4.5 - 2.5) / 2)


def test_vwap_and_bands_match_manual():
    rows = [(100, 101, 99, 100, 10), (100, 102, 99, 101, 30), (101, 103, 100, 102, 20)]
    df = compute_indicators(bars(rows), VwapParams(**OFF), None)
    src = np.array([(r[1] + r[2] + r[3]) / 3 for r in rows])
    v = np.array([r[4] for r in rows])
    for i in range(3):
        w = v[: i + 1]
        vw = (src[: i + 1] * w).sum() / w.sum()
        sd = math.sqrt(max((src[: i + 1] ** 2 * w).sum() / w.sum() - vw ** 2, 0))
        assert df["vwap"].iloc[i] == pytest.approx(vw)
        assert df["up2"].iloc[i] == pytest.approx(vw + 2 * sd)


@pytest.mark.parametrize("day", ["2024-01-09", "2024-07-09"])  # EST and EDT
def test_vwap_resets_at_1800_new_york(day):
    b = bars([(100, 101, 99, 100, 1)] * 2 + [(200, 201, 199, 200, 1)] * 2, start=f"{day} 16:30", freq="45min")
    df = compute_indicators(b, VwapParams(**OFF), None)
    opens = df.index.tz_convert(NY).strftime("%H:%M").tolist()
    assert opens == ["16:30", "17:15", "18:00", "18:45"]
    assert df["session"].iloc[1] != df["session"].iloc[2]
    assert df["vwap"].iloc[2] == pytest.approx(200)                         # fresh session


def test_entry_window_and_flatten_window():
    b = bars([(100, 101, 99, 100, 1)] * 4, start="2024-03-05 15:30")      # opens 15:30..16:15
    df = compute_indicators(b, VwapParams(**OFF, flatten_at="16:10"), None)
    assert df["in_window"].tolist() == [True, True, False, False]           # open < 16:00
    assert df["flat_window"].tolist() == [False, False, True, True]         # close 15:45, 16:00 | 16:15, 16:30


def test_signals_reclaim_rejection_and_filters():
    rows = [(100, 100.5, 99.5, 100, 100), (100, 100.2, 98.8, 99, 100), (99, 101.2, 98.9, 100.5, 100),
            (100.5, 101, 99.7, 100.6, 100)]
    df = compute_indicators(bars(rows), VwapParams(**OFF), None)
    # bar 2: close[1] < vwap[1] and close > vwap -> long reclaim, inside +2 sd
    assert df["close"].iloc[1] < df["vwap"].iloc[1] and df["close"].iloc[2] > df["vwap"].iloc[2]
    assert df["long_sig"].iloc[2] and df["setup"].iloc[2] == "reclaim"
    # bar 3: was above, low tests vwap, holds -> long rejection
    assert df["low"].iloc[3] <= df["vwap"].iloc[3] < df["close"].iloc[3]
    assert df["long_sig"].iloc[3] and df["setup"].iloc[3] == "rejection"
    # volume filter: equal volumes never exceed 1.2x their SMA
    p = VwapParams(**{**OFF, "use_volume": True, "vol_len": 2})
    assert not compute_indicators(bars(rows), p, None)["long_sig"].any()
    # longs disabled
    assert not compute_indicators(bars(rows), VwapParams(**OFF, allow_long=False), None)["long_sig"].any()


def test_no_chasing_beyond_band2():
    rows = [(100, 100.1, 99.9, 100, 100)] * 5 + [(100, 100.1, 99.0, 99.2, 100), (99.2, 110, 99.2, 109, 100)]
    df = compute_indicators(bars(rows), VwapParams(**OFF), None)
    assert df["close"].iloc[6] > df["up2"].iloc[6] and not df["long_sig"].iloc[6]


def test_macro_filter_alignment_no_lookahead():
    b = bars([(100, 101, 99, 100, 1)] * 6)
    other = pd.Series([1.0, 1, 1, 5, 5, 5], index=b.index + pd.Timedelta(minutes=1))  # stamped 1 min after each bar
    f = macro_flags(b, other, 2)
    assert np.isnan(f.iloc[0])                         # nothing known at/before the first bar's time
    assert f.iloc[4] == 1.0 and f.iloc[3] == 0.0       # value stamped after bar 3 shows up at bar 4
    p = VwapParams(**{**OFF, "use_macro": True})
    df = compute_indicators(b.assign(dxy_up=[0, 0, 1, 1, 0, np.nan], us10y_up=[0, 1, 0, 1, 0, 0]), p, None)
    assert df["macro_long_ok"].tolist() == [True, False, False, False, True, False]
    assert df["macro_short_ok"].tolist() == [False, False, False, True, False, False]
    loose = compute_indicators(b.assign(dxy_up=[0, 0, 1, 1, 0, 0], us10y_up=[0, 1, 0, 1, 0, 0]),
                               VwapParams(**{**OFF, "use_macro": True, "macro_strict": False}), None)
    assert loose["macro_long_ok"].tolist() == [True, True, True, False, True, True]


def test_filters_off_means_off():
    """Regression: `~True` is -2 in Python; disabling a filter must give all-True masks."""
    df = compute_indicators(bars([(100, 101, 99, 100, 0)] * 3), VwapParams(**OFF), None)
    assert df["vol_ok"].dtype == bool and df["vol_ok"].all()
    assert df["macro_long_ok"].all() and df["macro_short_ok"].all()


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

P = VwapParams(**OFF, atr_mult=1.0, tp_r=2.0, qty_step=1.0, max_leverage=1000)


def test_fill_next_open_with_slippage_and_target():
    f = frame([(100, 100, 100, 100), (100.5, 103, 100.4, 102.5), (102.5, 102.6, 102, 102)], sig={0: "long"}, atr=1.0)
    p = VwapParams(**{**P.__dict__, "slippage_ticks": 10, "mintick": 0.01})
    t, eq, open_ = execute(f, p)
    tr = t.iloc[0]
    assert tr["entry_price"] == pytest.approx(100.6)                     # next open + 10 ticks
    assert tr["stop"] == 99.0 and tr["target"] == 102.0                  # from the signal close
    assert tr["exit_price"] == 102.0 and tr["exit_reason"] == "target"   # limit: no slippage, same bar as fill
    assert tr["qty"] == 100                                              # 1% of 10k / 1.0 stop
    assert tr["pnl"] == pytest.approx((102 - 100.6) * 100)


def test_intrabar_path_rule_when_both_levels_hit():
    # open 100, high 103 (3 away), low 98.5 (1.5 away): low closer -> open->low->high: stop first
    f = frame([(100, 100, 100, 100), (100, 103, 98.5, 101)], sig={0: "long"})
    assert execute(f, P)[0].iloc[0]["exit_reason"] == "stop"
    # high closer: open->high->low: target first
    f2 = frame([(100, 100, 100, 100), (100, 102.2, 96, 101)], sig={0: "long"})
    assert execute(f2, P)[0].iloc[0]["exit_reason"] == "target"


def test_gap_through_stop_fills_at_open():
    f = frame([(100, 100, 100, 100), (100, 100.5, 99.5, 100), (97, 98, 96, 97.5)], sig={0: "long"})
    tr = execute(f, P)[0].iloc[0]
    assert tr["exit_reason"] == "stop (gap)" and tr["exit_price"] == 97.0
    assert tr["pnl"] == pytest.approx(-300)                               # worse than the 1% risk (gap)


def test_short_trade_and_sizing_risk_is_one_percent():
    f = frame([(100, 100, 100, 100), (100, 100.2, 99.8, 100), (100, 101.5, 99.9, 101)], sig={0: "short"}, atr=0.5)
    p = VwapParams(**{**P.__dict__, "atr_mult": 2.0})
    tr = execute(f, p)[0].iloc[0]
    assert tr["side"] == "short" and tr["stop"] == 101.0 and tr["target"] == 98.0
    assert tr["qty"] == 100 and tr["exit_reason"] == "stop"
    assert tr["pnl"] == pytest.approx(-0.01 * 10_000)                     # exactly 1% at the stop, no costs
    assert tr["r_multiple"] == pytest.approx(-1)


def test_leverage_cap_and_qty_step():
    f = frame([(100, 100, 100, 100), (100, 100, 100, 100)], sig={0: "long"}, atr=0.01)
    p = VwapParams(**{**P.__dict__, "max_leverage": 5, "qty_step": 7})
    tr = execute(f, p)[2]                                                  # still open
    assert tr["qty"] == 497                                                # min(10000, 500) -> floor to 7s


def test_end_of_day_flatten_at_close():
    f = frame([(100, 100, 100, 100), (100, 100.5, 99.5, 100.2), (100.2, 100.8, 99.6, 100.7)], sig={0: "long"}, flat={2})
    tr = execute(f, P)[0].iloc[0]
    assert tr["exit_reason"] == "end of day" and tr["exit_price"] == 100.7
    assert tr["exit_time"] == f["t_close"].iloc[2]


def test_no_entry_while_in_position_and_equity_compounds():
    # bar1 fills (no exit), bar1 signal ignored (in position), bar2 hits target 102,
    # bar2 signal -> bar3 fill sized off the new equity
    rows = [(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 102.5, 100.4, 102), (102, 102.2, 101.9, 102),
            (102, 102, 102, 102)]
    f = frame(rows, sig={0: "long", 1: "long", 2: "long"})
    t, eq, open_ = execute(f, P)
    assert len(t) == 1 and t.iloc[0]["exit_time"] == f.index[2] and t.iloc[0]["exit_reason"] == "target"
    assert open_ is not None and open_["signal_time"] == f["t_close"].iloc[2]
    assert open_["qty"] == math.floor((10_000 + t.iloc[0]["pnl"]) * 0.01 / 1.0)
    assert open_["qty"] > t.iloc[0]["qty"]
    assert eq.iloc[2] == pytest.approx(10_000 + t["pnl"].sum())


def test_new_entry_allowed_on_bar_that_exited():
    # bar1 fills and hits target 102 intrabar; the bar-1 signal enters at bar2's open (TV does the same)
    f = frame([(100, 100, 100, 100), (100, 102.5, 100, 101), (101, 102.5, 101, 102)], sig={0: "long", 1: "long"})
    t, _, open_ = execute(f, P)
    assert len(t) == 1 and open_ is not None and open_["entry_time"] == f.index[2]


def test_summarize():
    tr = pd.DataFrame({"pnl": [100.0, -50, 200, -50], "side": ["long", "short", "long", "long"],
                       "exit_time": pd.to_datetime(["2024-01-02", "2024-02-02", "2025-01-02", "2025-02-02"], utc=True),
                       "r_multiple": [2, -1, 4, -1], "commission": 0.0, "bars": 3, "exit_reason": "x"})
    s = summarize(tr, pd.Series([10_000.0, 10_100, 10_050, 10_250, 10_200]), 10_000)
    assert s["net_profit"] == 200 and s["profit_factor"] == pytest.approx(3.0)
    assert s["percent_profitable"] == 0.5 and s["max_drawdown_closed"] == 50
    assert s["by_year"] == {2024: {"trades": 2, "net_profit": 50.0}, 2025: {"trades": 2, "net_profit": 150.0}}
    assert s["long"]["trades"] == 3 and s["max_consecutive_losses"] == 1


# --------------------------------------------------------------------------- #
# I/O and TradingView comparison
# --------------------------------------------------------------------------- #

def test_load_tradingview_csv_unix_and_iso(tmp_path):
    t0 = int(pd.Timestamp("2024-03-05 13:00", tz="UTC").timestamp())
    p = tmp_path / "a.csv"
    p.write_text("time,open,high,low,close,Volume,VWAP,DXY up,US10Y up,Econ release\n"
                 f"{t0},1,2,0.5,1.5,10,1,1,0,0\n{t0 + 900},1.5,2,1,1.8,12,1,0,0,1\n")
    df = load_tradingview_csv(p)
    assert df.index[0] == pd.Timestamp("2024-03-05 13:00", tz="UTC")
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "dxy_up", "us10y_up", "econ_release"]
    assert infer_timeframe(df.index) == pd.Timedelta(minutes=15)
    q = tmp_path / "b.csv"
    q.write_text("time,open,high,low,close\n2024-03-05T08:00:00-05:00,1,2,0.5,1.5\n")
    assert load_tradingview_csv(q).index[0] == pd.Timestamp("2024-03-05 13:00", tz="UTC")


def synthetic_gold(years=5, seed=1):
    """15-minute gold-like bars, 23 h/day, 5 days/week, with intraday volume pattern."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=years * 252)
    idx = []
    for d in days:
        start = pd.Timestamp(d, tz=NY) - pd.Timedelta(hours=6)             # 18:00 previous evening
        idx.append(pd.date_range(start, periods=92, freq="15min"))         # until 17:00
    idx = pd.DatetimeIndex(np.concatenate(idx)).tz_convert("UTC")
    n = len(idx)
    r = rng.normal(0, 0.0012, n)
    c = 1800 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0006, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0006, n)))
    hour = idx.tz_convert(NY).hour
    vol = rng.gamma(2, 50, n) * np.where((hour >= 8) & (hour < 12), 3, 1)
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol}, index=idx)
    df["dxy_up"] = (rng.random(n) < 0.5).astype(float)
    df["us10y_up"] = (rng.random(n) < 0.5).astype(float)
    return df


@pytest.fixture(scope="module")
def five_year_run():
    t0 = time.time()
    run = run_backtest(synthetic_gold(), VwapParams(slippage_ticks=20))
    return run, time.time() - t0


def test_five_year_backtest_runs_fast_and_sane(five_year_run):
    run, secs = five_year_run
    assert len(run.bars) > 100_000 and secs < 60
    s = run.stats
    assert s["total_closed_trades"] > 50
    t = run.trades
    assert (t["entry_time"] > t["signal_time"] - pd.Timedelta(minutes=1)).all()
    ny = pd.DatetimeIndex(t["signal_time"]).tz_convert(NY)
    assert ((ny.hour * 60 + ny.minute >= 3 * 60 + 15) & (ny.hour * 60 + ny.minute <= 16 * 60)).all()
    stops = t[t["exit_reason"] == "stop"]
    # at the stop, loss = risk + slippage + rounding; never far beyond 1% of the then-equity
    eq_before = 10_000 + t["pnl"].cumsum().shift(fill_value=0)
    assert (-stops["pnl"] <= 0.0125 * eq_before[stops.index]).all()
    assert set(t["exit_reason"]) <= {"stop", "target", "end of day", "stop (gap)", "target (gap)"}
    fomc = parse_events(default_event_lines())
    sig = pd.DatetimeIndex(t["signal_time"])
    near = [(abs(sig - e) <= pd.Timedelta(minutes=60)).any() for e in fomc]
    assert not any(near)                                                   # nothing traded around FOMC


def test_compare_with_tradingview_roundtrip(five_year_run, tmp_path):
    run, _ = five_year_run
    t = run.trades.head(30)
    rows = []
    for k, r in enumerate(t.itertuples(), start=1):
        side = r.side.capitalize()
        rows.append({"Trade #": k, "Type": f"Exit {side}", "Signal": "XL",
                     "Date/Time": pd.Timestamp(r.exit_time).tz_convert(NY).strftime("%Y-%m-%d %H:%M"),
                     "Price USD": r.exit_price, "Contracts": r.qty, "Profit USD": r.pnl})
        rows.append({"Trade #": k, "Type": f"Entry {side}", "Signal": "L",
                     "Date/Time": pd.Timestamp(r.entry_time).tz_convert(NY).strftime("%Y-%m-%d %H:%M"),
                     "Price USD": r.entry_price, "Contracts": r.qty, "Profit USD": r.pnl})
    p = tmp_path / "tv.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    tv = load_tv_trades(p, tz=NY)
    assert len(tv) == 30
    c = compare_with_tradingview(run, tv)
    assert c["matched"] == 30 and c["mean_abs_entry_diff"] == pytest.approx(0, abs=1e-9)
    assert c["mean_abs_pnl_diff"] == pytest.approx(0, abs=1e-6)
    tv2 = tv.drop(index=[0]).assign(entry_price=tv["entry_price"].drop(index=[0]) + 0.5)
    c2 = compare_with_tradingview(run, tv2)
    assert c2["matched"] == 29 and c2["mean_abs_entry_diff"] == pytest.approx(0.5)
