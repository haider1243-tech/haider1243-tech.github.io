import math

import numpy as np
import pandas as pd
import pytest

from forecaster.backtest import (
    BacktestConfig,
    BacktestResult,
    baseline_drawdown_prob,
    baseline_quantiles,
    block_bootstrap_ci,
    evaluate,
    origins,
    per_origin_loss_diff,
    pinball,
    realized_close,
    realized_drawdown,
    run_backtest,
    score,
)
from forecaster.forecast import ensemble_forecast, regime_feature_table, regime_observations
from test_forecast import FAST, simulate_world


@pytest.fixture(scope="module")
def world():
    return simulate_world()[0]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def test_pinball_known_values():
    assert pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    assert pinball(np.array([-1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.1)
    assert pinball(np.array([0.3]), np.array([0.3]), 0.5)[0] == 0


def test_pinball_minimised_by_true_quantile():
    y = np.random.default_rng(0).normal(size=20000)
    best = min(np.linspace(-2, 2, 81), key=lambda f: pinball(y, np.full_like(y, f), 0.9).mean())
    assert best == pytest.approx(1.2816, abs=0.06)


def test_origins_are_quarter_ends():
    idx = pd.bdate_range("2004-06-01", "2006-02-15")
    o = origins(pd.Series(1.0, index=idx), "2004-12-31")
    assert o[0] == pd.Timestamp("2004-12-31") and o[-1] == pd.Timestamp("2005-12-31")
    assert all(t.is_quarter_end for t in o)


def test_realized_close_requires_complete_quarter():
    idx = pd.bdate_range("2022-01-03", "2022-12-28")
    p = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    assert math.isnan(realized_close(p, pd.Timestamp("2022-12-31")))  # data stops Dec 28
    assert realized_close(p, pd.Timestamp("2022-09-30")) == p.loc["2022-09-30"]


def test_realized_drawdown_counts_start_as_peak():
    idx = pd.bdate_range("2020-01-01", periods=4)
    p = pd.Series([100.0, 95, 80, 120], index=idx)
    assert realized_drawdown(p, idx[0], idx[-1], 100.0) == pytest.approx(-0.20)


def test_baseline_uses_only_given_history():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-01", periods=3000)
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 3000))), index=idx)
    T = idx[2000]
    b1 = baseline_quantiles(p.loc[:T], 63)
    tampered = p.copy()
    tampered[tampered.index > T] *= 5
    assert np.array_equal(b1, baseline_quantiles(tampered.loc[:T], 63))
    # roughly normal theory for this path's own drift / vol (one path's drift
    # differs from the generator's by ~0.014 over 63 days, so compare to the sample)
    lr = np.log(p.loc[:T]).diff().dropna()
    assert b1[1] == pytest.approx(lr.mean() * 63, abs=0.01)
    assert (b1[2] - b1[0]) == pytest.approx(2 * 1.2816 * lr.std() * math.sqrt(63), rel=0.15)


def test_baseline_drawdown_prob_bounds():
    idx = pd.bdate_range("2000-01-01", periods=1000)
    up = pd.Series(np.linspace(100, 200, 1000), index=idx)
    assert baseline_drawdown_prob(up, 63, 0.10) == 0.0
    crash = pd.Series(np.r_[np.full(500, 100.0), np.full(500, 50.0)], index=idx)
    assert 0 < baseline_drawdown_prob(crash, 63, 0.10) < 0.2


def test_block_bootstrap_ci_width_matches_theory_on_average():
    # A single sample can have chance autocorrelation (which legitimately
    # changes a block CI), so check the average over many iid samples.
    ratios = []
    for seed in range(20):
        x = np.random.default_rng(100 + seed).normal(0, 1, 400)
        lo, hi = block_bootstrap_ci(x, n_boot=500, block=4, seed=seed)
        assert lo < x.mean() < hi
        ratios.append((hi - lo) / (2 * 1.645 * x.std(ddof=1) / math.sqrt(400)))
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.1)


def test_block_bootstrap_widens_for_autocorrelated_data():
    rng = np.random.default_rng(7)
    e = rng.normal(0, 1, 400)
    y = np.empty(400)
    y[0] = e[0]
    for t in range(1, 400):
        y[t] = 0.6 * y[t - 1] + e[t]
    w1 = np.subtract(*block_bootstrap_ci(y, 1000, block=1)[::-1])
    w4 = np.subtract(*block_bootstrap_ci(y, 1000, block=4)[::-1])
    assert w4 > 1.2 * w1


# --------------------------------------------------------------------------- #
# Scoring on hand-built rows
# --------------------------------------------------------------------------- #

def _rows(y, m, b, origin_step=True):
    n = len(y)
    o = pd.date_range("2005-03-31", periods=n, freq="QE")
    return pd.DataFrame({
        "origin": o, "horizon": 1, "target": o + pd.offsets.QuarterEnd(1), "y": y,
        "realized_dd": np.minimum(y, 0), "dd_event": (np.asarray(y) < -0.1).astype(float),
        "m_q10": m[0], "m_q50": m[1], "m_q90": m[2], "m_dd": 0.2,
        "b_q10": b[0], "b_q50": b[1], "b_q90": b[2], "b_dd": 0.2,
    })


def test_score_coverage_hit_rate():
    y = np.array([0.05, -0.02, 0.2, -0.3, 0.01])
    r = _rows(y, (-0.1, 0.02, 0.1), (-0.1, 0.02, 0.1))
    s = score(r, "m")
    assert s["coverage"] == pytest.approx(3 / 5)
    assert s["below_p10"] == pytest.approx(1 / 5) and s["above_p90"] == pytest.approx(1 / 5)
    assert s["hit_rate"] == pytest.approx(3 / 5)  # P50 > 0, three positive outcomes
    assert s["width"] == pytest.approx(0.2)


def test_evaluate_says_so_when_model_loses():
    rng = np.random.default_rng(2)
    y = rng.normal(0.02, 0.08, 80)
    good = (np.quantile(y, 0.1), np.quantile(y, 0.5), np.quantile(y, 0.9))
    bad = (0.10, 0.12, 0.14)  # confidently wrong, far too narrow
    rep = evaluate(BacktestResult(_rows(y, bad, good), [], BacktestConfig(n_boot=500)), windows={})
    assert rep.skill < 0
    assert rep.diff_ci[0] > 0
    assert "WORSE" in rep.verdict[0]
    assert any("too narrow" in v for v in rep.verdict)


def test_evaluate_says_so_when_model_wins():
    rng = np.random.default_rng(3)
    mu = rng.normal(0, 0.05, 80)
    y = mu + rng.normal(0, 0.02, 80)                 # model knows mu, baseline doesn't
    r = _rows(y, (mu - 0.026, mu, mu + 0.026), (np.full(80, -0.07), np.zeros(80), np.full(80, 0.07)))
    rep = evaluate(BacktestResult(r, [], BacktestConfig(n_boot=500)), windows={})
    assert rep.skill > 0.3 and rep.diff_ci[1] < 0
    assert "BEATS" in rep.verdict[0]


def test_evaluate_inconclusive_when_equal_ish():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 0.05, 40)
    q = (np.full(40, -0.064), np.zeros(40), np.full(40, 0.064))
    q2 = (np.full(40, -0.066), np.full(40, 0.001), np.full(40, 0.065))
    rep = evaluate(BacktestResult(_rows(y, q, q2), [], BacktestConfig(n_boot=500)), windows={})
    assert "does NOT reliably beat" in rep.verdict[0]


def test_per_origin_loss_diff_sign():
    y = np.zeros(4)
    r = _rows(y, (-0.1, 0.0, 0.1), (-0.1, 0.05, 0.1))
    assert (per_origin_loss_diff(r) < 0).all()


# --------------------------------------------------------------------------- #
# End to end on synthetic data
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def bt(world):
    cfg = BacktestConfig(start="2018-12-31", end="2021-12-31", ensemble=FAST, n_boot=200)
    calls = []

    def spy(data, asof, config, regimes):
        calls.append((asof, regimes.index.max()))
        return ensemble_forecast(data, asof=asof, config=config, regimes=regimes)

    return run_backtest(world, cfg, forecaster=spy), calls


def test_backtest_retrains_every_quarter(bt):
    res, calls = bt
    asofs = [a for a, _ in calls]
    assert asofs == [pd.Timestamp(x) for x in pd.date_range("2018-12-31", "2021-12-31", freq="QE")]
    assert res.skipped == []
    assert len(res.rows) == 4 * len(asofs)


def test_backtest_rows_consistent(bt, world):
    res, _ = bt
    r = res.rows
    assert set(r["horizon"]) == {1, 2, 3, 4}
    assert (r["target"] > r["origin"]).all()
    s = res.scored
    assert (s["target"] <= world.prices.index.max()).all()
    assert r.loc[r["target"] > world.prices.index.max(), "y"].isna().all()
    row = s.iloc[5]
    assert row["y"] == pytest.approx(math.log(world.prices.loc[:row["target"]].iloc[-1] / row["close_T"]))
    assert (s["m_q10"] <= s["m_q50"]).all() and (s["m_q50"] <= s["m_q90"]).all()


def test_cached_regimes_identical_to_fresh(world):
    T = pd.Timestamp("2020-03-31")
    obs = regime_observations(world.prices, world.vix)
    table = regime_feature_table(obs, world.prices.index.max(), FAST.hmm_refit_every, FAST.hmm_min_obs,
                                 FAST.hmm_n_init, FAST.hmm_n_iter, FAST.seed)
    a = ensemble_forecast(world, asof=T, config=FAST, regimes=table)
    b = ensemble_forecast(world, asof=T, config=FAST)
    pd.testing.assert_frame_equal(a.table, b.table)


def test_evaluate_end_to_end(bt):
    res, _ = bt
    rep = evaluate(res, windows={"2020 COVID": ("2020-01-01", "2020-12-31"), "never": ("1990-01-01", "1990-12-31")})
    assert set(rep.overall.index) == {"model", "baseline"}
    assert set(rep.by_horizon.index.get_level_values("horizon")) <= {1, 2, 3, 4}
    assert 0 <= rep.overall.loc["model", "coverage"] <= 1
    assert any("never: no scored forecasts" in v for v in rep.verdict)
    md = rep.to_markdown()
    assert "## Verdict" in md and "Pinball skill" in md and "## Components" in md
    assert set(rep.components.index) == {"blend", "monte_carlo", "lightgbm", "baseline"}
    # blend quantiles are the sorted 50/50 average of the components
    s = res.scored.dropna(subset=["lgb_q50"])
    np.testing.assert_allclose(s["m_q50"], np.sort(0.5 * s[["mc_q10", "mc_q50", "mc_q90"]].to_numpy()
                                                   + 0.5 * s[["lgb_q10", "lgb_q50", "lgb_q90"]].to_numpy(), axis=1)[:, 1])
