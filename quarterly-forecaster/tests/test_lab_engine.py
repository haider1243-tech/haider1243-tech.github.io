"""Backtest engine: known trades, cost math, no look-ahead, parity with backtesting.py."""
import numpy as np
import pandas as pd
import pytest

from forecaster import intraday as I
from forecaster import lab_engine as E
from forecaster import strategies as S

SPY = I.INSTRUMENTS["SPY"]
ZERO = E.CostModel()


def bars_at(start_ny: str, ohlc: list[tuple], minutes: int = 1) -> pd.DataFrame:
    idx = (pd.Timestamp(start_ny, tz=I.NY) + pd.to_timedelta(np.arange(len(ohlc)) * minutes, unit="m")).tz_convert("UTC")
    df = pd.DataFrame(ohlc, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 100.0
    return df


def sig(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame([r[1:] for r in rows], columns=["side", "stop", "target"],
                        index=pd.Index([r[0] for r in rows], name="i"))


def pad_day(df: pd.DataFrame) -> pd.DataFrame:
    """Extend with flat bars (last close) to 16:00 so the day counts as complete."""
    last = df.index[-1].tz_convert(I.NY)
    end = last.normalize() + pd.Timedelta(hours=16)
    n = int((end - last) / pd.Timedelta(minutes=1)) - 1
    if n <= 0:
        return df
    c = df["Close"].iloc[-1]
    extra = bars_at(str((last + pd.Timedelta(minutes=1)).tz_localize(None)), [(c, c, c, c)] * n)
    if "Spread" in df:
        extra["Spread"] = df["Spread"].iloc[-1]
    return pd.concat([df, extra])


def run(df, signals, costs=ZERO, minutes=1, pad=True, **kw):
    return E.run(pad_day(df) if pad else df, SPY, minutes, signals, costs, **kw)


FLAT = [(100, 100.2, 99.8, 100)] * 400       # 10:00 -> 16:40 (window ends 16:00)


# --------------------------------------------------------------------------- #
# Known trades
# --------------------------------------------------------------------------- #

def test_target_fill_next_open_and_returns():
    b = FLAT[:]
    b[1] = (100.0, 100.3, 99.9, 100.2)
    b[2] = (100.2, 101.2, 100.1, 101.0)                       # target 101 touched
    t = run(bars_at("2024-03-04 10:00", b), sig([(0, 1, 99.0, 101.0)])).iloc[0]
    assert (t.entry_i, t.exit_i, t.reason) == (1, 2, "target")
    assert t.entry == 100.0 and t.exit == 101.0                 # filled at bar 1's open, target exact
    assert t.gross_ret == pytest.approx(0.01) and t.net_ret == pytest.approx(0.01) and t.cost_ret == pytest.approx(0)
    assert t.r_mult == pytest.approx(1.0) and t.bars_held == 2
    assert t.entry_time == pd.Timestamp("2024-03-04 10:01", tz=I.NY)


def test_stop_first_when_both_hit_and_gap_through_stop():
    b = FLAT[:]
    b[2] = (100.0, 101.5, 98.5, 100.0)                        # touches both -> stop (conservative)
    t = run(bars_at("2024-03-04 10:00", b), sig([(0, 1, 99.0, 101.0)])).iloc[0]
    assert t.reason == "stop" and t.exit == 99.0 and t.exit_i == 2
    g = FLAT[:]
    g[3] = (98.0, 98.5, 97.5, 98.2)                           # opens below the stop -> filled at the open
    t = run(bars_at("2024-03-04 10:00", g), sig([(0, 1, 99.0, 101.0)])).iloc[0]
    assert t.reason == "stop" and t.exit == 98.0 and t.net_ret == pytest.approx(-0.02)


def test_short_side_mirrors():
    b = FLAT[:]
    b[3] = (100.0, 100.1, 98.9, 99.0)
    t = run(bars_at("2024-03-04 10:00", b), sig([(0, -1, 101.0, 99.0)])).iloc[0]
    assert t.side == -1 and t.reason == "target" and t.exit == 99.0 and t.net_ret == pytest.approx(0.01)


def test_time_exit_flat_before_last_minute():
    df = bars_at("2024-03-04 15:50", [(100, 100.1, 99.9, 100 + k * 0.01) for k in range(10)])   # 15:50..15:59
    t = run(df, sig([(0, 1, 90.0, 110.0)])).iloc[0]
    assert t.reason == "time"
    assert t.exit_time == pd.Timestamp("2024-03-04 15:58", tz=I.NY)       # bar 15:58-15:59 is the last allowed
    assert t.exit == df["Close"].iloc[8]


def test_no_entry_in_first_or_last_minute_and_next_day_is_separate():
    a = bars_at("2024-03-04 09:30", [(100, 100.1, 99.9, 100)] * 400)
    assert run(a, sig([(0, 1, 99, 101)])).iloc[0].entry_time == pd.Timestamp("2024-03-04 09:31", tz=I.NY)
    first = bars_at("2024-03-04 09:29", [(100, 100.1, 99.9, 100)] * 400)  # 09:29 bar isn't in the window
    assert run(first, sig([(0, 1, 99, 101)])).empty                      # would fill at 09:30: first minute
    late = bars_at("2024-03-04 15:57", [(100, 100.1, 99.9, 100)] * 3)
    assert run(late, sig([(1, 1, 99, 101)])).empty                       # fill 15:59: last minute


def test_one_position_at_a_time_and_max_trades_per_day():
    b = FLAT[:60]
    b[1:4] = [(100, 100.1, 99.9, 100)] * 3                    # first trade stays open over bars 1-3
    df = bars_at("2024-03-04 10:00", b)                       # ... and hits 100.15 on bar 4
    s = sig([(0, 1, 99.0, 100.15), (1, 1, 99.0, 100.15), (5, 1, 99.0, 100.15), (9, -1, 101, 99.85)])
    t = run(df, s)
    assert list(t.signal_i) == [0, 5, 9]                      # signal 1 arrives while in a trade
    assert list(run(df, s, max_trades=2).signal_i) == [0, 5]


def test_entry_gapped_past_stop_or_target_is_skipped():
    b = FLAT[:]
    b[1] = (98.5, 98.6, 98.4, 98.5)
    assert run(bars_at("2024-03-04 10:00", b), sig([(0, 1, 99.0, 101.0)])).empty


def test_incomplete_day_skipped_in_backtest_open_in_live():
    df = bars_at("2024-03-04 10:00", FLAT[:30])               # data stops at 10:29
    s = sig([(0, 1, 90.0, 110.0)])
    assert run(df, s, pad=False).empty
    t = run(df, s, pad=False, allow_open=True).iloc[0]
    assert t.reason == "open" and np.isnan(t.net_ret)


# --------------------------------------------------------------------------- #
# Cost math
# --------------------------------------------------------------------------- #

def test_cost_math_spread_slippage_commission():
    df = bars_at("2024-03-04 15:50", [(100, 100.1, 99.9, 100.5)] * 10)
    costs = E.CostModel(commission=0.005, spread=0.02, slippage=0.01)
    t = run(df, sig([(0, 1, 90.0, 110.0)]), costs).iloc[0]
    # buy 100 + 0.01 (half spread) + 0.01 slip; time exit sell 100.5 - 0.01 - 0.01; commission 2 x 0.005
    assert t.entry == pytest.approx(100.02) and t.exit == pytest.approx(100.48)
    assert t.gross_ret == pytest.approx(0.005)
    assert t.net_ret == pytest.approx((100.48 - 100.02 - 0.01) / 100)
    assert t.cost_ret == pytest.approx((0.02 + 0.02 + 0.01) / 100)


def test_stop_slips_but_target_does_not_and_spread_column_wins():
    df = bars_at("2024-03-04 10:00", [(100, 100.1, 99.9, 100)] * 2 + [(100, 101.5, 100, 101)] + [(100, 100, 100, 100)] * 5)
    df["Spread"] = 0.10
    costs = E.CostModel(spread=9.99, slippage=0.03)            # bar spread overrides the model's
    t = run(df, sig([(0, 1, 98.0, 101.0)]), costs).iloc[0]
    assert t.entry == pytest.approx(100.08) and t.exit == pytest.approx(101.0) and t.reason == "target"
    d2 = bars_at("2024-03-04 10:00", [(100, 100.1, 99.9, 100)] * 2 + [(100, 100, 98.5, 99)] + [(99, 99, 99, 99)] * 5)
    d2["Spread"] = 0.10
    s = run(d2, sig([(0, 1, 99.0, 101.0)]), costs).iloc[0]
    assert s.reason == "stop" and s.exit == pytest.approx(99.0 - 0.03)   # bid hit the stop, then slippage


def test_target_requires_the_bid_to_reach_it_on_mid_bars():
    df = bars_at("2024-03-04 10:00", [(100, 100.1, 99.9, 100)] * 2 + [(100, 101.04, 100, 101)] + [(100, 100.1, 99.9, 100)] * 4)
    df["Spread"] = 0.10                                         # high mid 101.04 -> bid 100.99 < 101
    t = run(df, sig([(0, 1, 98.0, 101.0)]), E.CostModel()).iloc[0]
    assert t.reason != "target"


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def m5():
    out = {}
    for sym in ("SPY", "XAU_USD"):
        spec = I.INSTRUMENTS[sym]
        m1 = I.session_bars(I.synthetic_m1(sym, "2024-01-02", "2024-03-29", seed=7), spec)
        out[sym] = I.resample(m1, 5, spec)
    return out


@pytest.mark.parametrize("key", list(S.LIBRARY))
def test_signals_never_use_future_bars(m5, key):
    st = S.LIBRARY[key]
    if not S.applicable(st, 5):
        st = st.with_params(minutes=5) if "minutes" in st.params else st
    for sym, bars in m5.items():
        full = st.signals(S.Features(bars, I.INSTRUMENTS[sym], 5))
        for k in (len(bars) // 3, len(bars) // 2, len(bars) - 7):
            part = st.signals(S.Features(bars.iloc[:k], I.INSTRUMENTS[sym], 5))
            ref = full[full.index < k]
            pd.testing.assert_frame_equal(part, ref, check_exact=False, rtol=1e-9, obj=f"{key} {sym} k={k}")


def test_trades_unchanged_when_future_is_altered(m5):
    bars = m5["SPY"]
    st = S.LIBRARY["ema_cross"]
    f = S.Features(bars, SPY, 5)
    full = E.run(bars, SPY, 5, st.signals(f), E.DEFAULT_COSTS["SPY"])
    cut = len(bars) // 2
    cut_time = bars.index[cut]
    noisy = bars.copy()
    noisy.iloc[cut:, :4] *= 1.07                               # rewrite the future
    alt = E.run(noisy, SPY, 5, st.signals(S.Features(noisy, SPY, 5)), E.DEFAULT_COSTS["SPY"])
    done = full[full.exit_time < cut_time].reset_index(drop=True)
    pd.testing.assert_frame_equal(done, alt[alt.exit_time < cut_time].reset_index(drop=True))
    assert len(done) > 20


# --------------------------------------------------------------------------- #
# Parity with backtesting.py
# --------------------------------------------------------------------------- #

def test_matches_backtesting_py_trade_for_trade(m5):
    bt = pytest.importorskip("backtesting")
    bars = m5["SPY"]
    st = S.LIBRARY["ema_cross"].with_params(max_trades=99)
    signals = st.signals(S.Features(bars, SPY, 5))
    ours = E.run(bars, SPY, 5, signals, ZERO)
    ours = ours[ours.reason != "time"]
    # feed backtesting.py exactly the signals our engine took (it has no session/time-exit rules)
    taken = signals.loc[signals.index.isin(ours.signal_i)]
    plan = {int(i): (int(r.side), float(r.stop), float(r.target)) for i, r in taken.iterrows()}

    class Fixed(bt.Strategy):
        def init(self):
            pass

        def next(self):
            i = len(self.data) - 1
            if i in plan and not self.position:
                side, stop, tgt = plan[i]
                (self.buy if side > 0 else self.sell)(size=1, sl=stop, tp=tgt)

    data = bars.copy()
    data.index = data.index.tz_localize(None)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bt.Backtest(data, Fixed, cash=1e7, commission=0, trade_on_close=False, exclusive_orders=False,
                          finalize_trades=True).run()
    theirs = res._trades
    theirs = theirs[theirs.EntryTime.isin(ours.entry_time.dt.tz_localize(None))].reset_index(drop=True)
    ours = ours.reset_index(drop=True)
    assert len(ours) > 30 and len(theirs) == len(ours)
    np.testing.assert_allclose(theirs.EntryPrice, ours.entry, rtol=1e-12)
    np.testing.assert_allclose(theirs.ExitPrice, ours.exit, rtol=1e-12)
    assert (pd.DatetimeIndex(theirs.ExitTime) == pd.DatetimeIndex(ours.exit_time.dt.tz_localize(None))).all()


# --------------------------------------------------------------------------- #
# Random-entry benchmark
# --------------------------------------------------------------------------- #

def test_random_entries_respect_holds_costs_and_windows(m5):
    bars = m5["SPY"]
    pp = E.prepare(bars, SPY, 5, ZERO)
    means = E.random_entry_means(pp, np.array([3, 6, 12]), np.array([1, -1, 1]), ZERO, 500, seed=1)
    assert means.shape == (500,) and abs(means.mean()) < 5e-4            # random walk, no costs: ~0
    again = E.random_entry_means(pp, np.array([3, 6, 12]), np.array([1, -1, 1]), ZERO, 500, seed=1)
    np.testing.assert_array_equal(means, again)
    costly = E.random_entry_means(E.prepare(bars, SPY, 5, E.DEFAULT_COSTS["SPY"]), np.array([3, 6, 12]),
                                  np.array([1, -1, 1]), E.DEFAULT_COSTS["SPY"], 500, seed=1)
    assert costly.mean() < means.mean()                                   # costs always hurt


def test_every_timeframe_trades_on_both_instruments():
    """Regression: gold's window ends 16:55, which 15m bars never end on; days must still count."""
    for sym in ("SPY", "XAU_USD"):
        spec = I.INSTRUMENTS[sym]
        m1 = I.session_bars(I.synthetic_m1(sym, "2024-01-02", "2024-02-29", seed=3), spec)
        for tf, minutes in I.TIMEFRAMES.items():
            bars = I.resample(m1, minutes, spec)
            st = S.LIBRARY["ema_cross"]
            t = E.run(bars, spec, minutes, st.signals(S.Features(bars, spec, minutes)), E.DEFAULT_COSTS[sym])
            assert len(t) > 5, (sym, tf)
            exit_ny = pd.DatetimeIndex(t.exit_time).tz_convert(I.NY)
            assert ((exit_ny.hour * 60 + exit_ny.minute) <= I._hm(spec.session_end) - 1).all(), (sym, tf)


def test_random_entries_keep_the_real_trades_days(m5):
    bars = m5["SPY"]
    pp = E.prepare(bars, SPY, 5, ZERO)
    days = np.unique(pp.day[pp.entry_ok])[:3]
    trade_days = np.repeat(days, [5, 1, 2])
    rng_means = E.random_entry_means(pp, np.full(8, 4), np.ones(8, int), ZERO, 50, seed=0, trade_days=trade_days)
    assert rng_means.shape == (50,)
    # with every trade forced onto one day, all random strategies only sample that day's bars
    one = E.random_entry_means(pp, np.full(4, 2), np.ones(4, int), ZERO, 200, seed=0, trade_days=np.repeat(days[:1], 4))
    d0 = pp.entry_ok & (pp.day == days[0])
    rets = (pp.c[np.flatnonzero(d0) + 1] - pp.o[np.flatnonzero(d0)]) / pp.o[np.flatnonzero(d0)]
    assert one.min() >= rets.min() - 1e-12 and one.max() <= rets.max() + 1e-12
