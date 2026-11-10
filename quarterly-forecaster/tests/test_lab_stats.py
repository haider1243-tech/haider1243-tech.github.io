import numpy as np
import pandas as pd
import pytest

from forecaster import lab_stats as LS


def trades(rets, days=None, sides=None):
    n = len(rets)
    days = days or [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)]
    return pd.DataFrame({"net_ret": rets, "day": pd.to_datetime(days), "side": sides or [1] * n,
                         "r_mult": np.array(rets) * 100, "bars_held": [3] * n,
                         "entry_time": pd.to_datetime(days).tz_localize("America/New_York") + pd.Timedelta(hours=10)})


def test_summary_known_numbers():
    r = [0.02, -0.01, -0.01, 0.03, -0.01, -0.01, -0.01]
    s = LS.summary(trades(r))
    assert s["trades"] == 7 and s["win_rate"] == pytest.approx(2 / 7)
    assert s["avg_win"] == pytest.approx(0.025) and s["avg_loss"] == pytest.approx(-0.01)
    assert s["expectancy"] == pytest.approx(0.0) and s["profit_factor"] == pytest.approx(0.05 / 0.05)
    assert s["max_dd"] == pytest.approx(-0.03) and s["losing_streak"] == 3
    assert LS.summary(trades([]).iloc[0:0])["trades"] == 0


def test_max_drawdown_and_streak_edge_cases():
    assert LS.max_drawdown(np.array([-0.01, -0.02, 0.05])) == pytest.approx(-0.03)   # from the 0 start
    assert LS.max_drawdown(np.array([0.01, 0.02])) == 0.0
    assert LS.longest_losing_streak(np.array([1, -1, 0, -1, 2, -1])) == 3            # 0 counts as a loss


def test_sharpe_counts_flat_days():
    t = trades([0.01, -0.005, 0.01], days=["2024-01-02", "2024-01-03", "2024-01-04"])
    s3, s10 = LS.summary(t)["sharpe"], LS.summary(t, days=10)["sharpe"]
    d = np.array([0.01, -0.005, 0.01] + [0] * 7)
    assert s10 == pytest.approx(d.mean() / d.std(ddof=1) * np.sqrt(252)) and s10 != s3


def test_bootstrap_ci_and_order_mc():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 400)
    lo, hi = LS.bootstrap_ci(r)
    assert lo < r.mean() < hi and hi - lo == pytest.approx(2 * 1.96 * r.std() / 20, rel=0.15)
    mc = LS.order_monte_carlo(r, n=200)
    assert mc["dd_p95"] <= mc["dd_median"] <= 0
    assert LS.bootstrap_ci([0.1]) is None


def test_p_value_smoothing():
    rand = np.linspace(-1, 1, 999)
    assert LS.p_value(2.0, rand) == pytest.approx(1 / 1000)
    assert LS.p_value(0.5, rand) == pytest.approx((1 + 250) / 1000)
    assert LS.p_value(0.1, np.array([])) is None


@pytest.mark.parametrize("n,e,p,label", [
    (150, 0.001, 0.01, "EDGE"),
    (99, 0.001, 0.01, "WEAK"),             # too few OOS trades
    (150, 0.001, 0.30, "WEAK"),            # positive with 100+ trades but not significant
    (40, 0.001, 0.30, "NO EDGE"),
    (500, -0.0001, 0.01, "NO EDGE"),       # negative after costs
    (500, 0.001, None, "WEAK"),
])
def test_verdict_rules(n, e, p, label):
    v, why = LS.verdict({"trades": n, "expectancy": e}, p)
    assert v == label
    assert (v == "EDGE") == (why == ["p < 0.05 vs 1,000 random-entry strategies, 100+ OOS trades, positive expectancy after costs"])


def test_folds_roll_without_overlap():
    days = pd.bdate_range("2022-01-03", "2024-12-31")
    folds = LS.make_folds(pd.DatetimeIndex(days), 12, 3)
    assert len(folds) == 8
    for f in folds:
        assert f.train[1] == f.test[0]                       # test starts where training ends
    for a, b in zip(folds, folds[1:]):
        assert a.test[1] == b.test[0]                        # test windows tile the period, no overlap


def test_walk_forward_picks_on_train_and_scores_only_the_next_window():
    days = pd.DatetimeIndex(pd.bdate_range("2022-01-03", "2023-12-29"))
    # combo A wins in-sample in 2022 but loses in 2023; combo B is the reverse
    def run_combo(p):
        r = np.where(days.year == 2022, 0.002, -0.002) * (1 if p["x"] == "A" else -1)
        return pd.DataFrame({"day": days, "net_ret": r, "side": 1, "r_mult": r, "bars_held": 3,
                             "entry_time": days.tz_localize("America/New_York")})
    wf = LS.walk_forward(run_combo, {"x": ["A", "B"]}, days, train_months=12, test_months=3)
    f0 = wf["folds"][0]
    assert f0.best == {"x": "A"}                              # chosen on 2022 only
    assert f0.oos_stats["expectancy"] == pytest.approx(-0.002)   # ...and judged on unseen 2023Q1
    assert (pd.to_datetime(wf["oos"]["day"]) >= pd.Timestamp("2023-01-03")).all()
    w = LS.overfit_warnings(["x"], wf["combos"], wf["folds"], LS.summary(wf["is"]), LS.summary(wf["oos"]))
    assert any("fitted noise" in x for x in w) or any("decay" in x for x in w)


def test_overfit_warnings_for_many_params():
    w = LS.overfit_warnings(["a", "b", "c"], 27, [], {"expectancy": 0.001}, {"expectancy": 0.0004})
    assert any("3 parameters tuned" in x for x in w) and any("27 parameter combinations" in x for x in w)
    assert any("40% of in-sample" in x for x in w)


def test_slices_use_previous_day_vix_and_event_days():
    t = trades([0.01, -0.02, 0.03], days=["2024-03-11", "2024-03-12", "2024-03-20"])
    vix = pd.Series([14.0, 22.0, 35.0, 18.0], index=pd.to_datetime(["2024-03-08", "2024-03-11", "2024-03-12", "2024-03-19"]))
    ev = {"CPI": {"2024-03-12"}, "FOMC": {"2024-03-20"}, "PPI": set()}
    s = LS.add_slice_columns(t, vix, ev)
    assert list(s["vix_band"]) == ["<15", "20-30", "15-20"]          # Mar 11 uses Mar 8's 14.0, etc.
    assert list(s["event_day"]) == ["normal", "CPI", "FOMC"]
    assert list(s["weekday"]) == ["Mon", "Tue", "Wed"] and set(s["time_of_day"]) == {"10:00"}
    by = {r["key"]: r for r in LS.slice_stats(s, "event_day")}
    assert by["CPI"]["expectancy"] == pytest.approx(-0.02) and by["normal"]["trades"] == 1
    assert LS.add_slice_columns(t)["event_day"].eq("unknown").all()


def test_day_bootstrap_is_wider_for_clustered_trades():
    rng = np.random.default_rng(1)
    day_effect = rng.normal(0, 0.01, 60)                     # 60 days, 10 correlated trades each
    days = np.repeat(np.arange(60), 10)
    r = day_effect[days] + rng.normal(0, 0.002, 600)
    naive = LS.bootstrap_ci(r)
    clustered = LS.bootstrap_ci(r, days=days)
    assert (clustered[1] - clustered[0]) > 2 * (naive[1] - naive[0])
