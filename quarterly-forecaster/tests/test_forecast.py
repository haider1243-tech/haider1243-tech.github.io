import numpy as np
import pandas as pd
import pytest

from forecaster.forecast import (
    QUANTILES,
    REGIME_FEATURES,
    EnsembleConfig,
    GarchParams,
    PointInTimeData,
    RegimeModel,
    build_training_set,
    business_days_between,
    ensemble_forecast,
    feature_frame,
    feature_names,
    fit_garch,
    macro_features,
    pit_indicator,
    price_features,
    regime_feature_table,
    regime_observations,
    refit_dates,
    sentiment_feature,
    simulate_paths,
    target_dates,
    to_yield_percent,
)
from forecaster.macro import SeriesSpec
from forecaster.model import future_quarter_labels

CPI = SeriesSpec("CPIAUCSL", "CPI", "index", "M", annualize_trend=True)
UNRATE = SeriesSpec("UNRATE", "Unemployment", "rate", "M")
SPREAD = SeriesSpec("T10Y2Y", "Spread", "rate", "D")

FAST = EnsembleConfig(n_paths=2000, hmm_n_init=1, hmm_n_iter=50, hmm_refit_every=8,
                      hmm_min_obs=500, min_train_rows=40,
                      lgb_params=dict(n_estimators=60, learning_rate=0.1, num_leaves=4,
                                      min_child_samples=10))

# Regime-switching "true" world: bull / bear / high-vol.
TRUE_P = np.array([[0.995, 0.004, 0.001], [0.010, 0.985, 0.005], [0.010, 0.010, 0.980]])
TRUE_MU = np.array([0.0008, -0.0006, -0.0010])
TRUE_SIG = np.array([0.007, 0.012, 0.030])


def simulate_world(n_days=252 * 12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    s = np.empty(n_days, dtype=int)
    s[0] = 0
    for t in range(1, n_days):
        s[t] = rng.choice(3, p=TRUE_P[s[t - 1]])
    r = TRUE_MU[s] + TRUE_SIG[s] * rng.standard_normal(n_days)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx, name="SYN")
    vix = pd.Series(100 * TRUE_SIG[s] * np.sqrt(252) * np.exp(0.08 * rng.standard_normal(n_days)), index=idx)
    tnx = pd.Series(np.clip(3 + np.cumsum(rng.normal(0, 0.02, n_days)), 0.5, 8), index=idx)

    months = pd.date_range("2008-01-01", "2024-12-01", freq="MS")
    cpi = pd.DataFrame({"date": months, "value": 100 * 1.002 ** np.arange(len(months)),
                        "available": months + pd.Timedelta(days=45)})
    un = pd.DataFrame({"date": months, "value": 5 + 0.3 * np.sin(np.arange(len(months)) / 9),
                       "available": months + pd.Timedelta(days=38)})
    macro = {"CPIAUCSL": pit_indicator(cpi, CPI), "UNRATE": pit_indicator(un, UNRATE)}
    q = pd.period_range("2011Q1", "2026Q4", freq="Q")
    risk = pd.Series(30 + rng.integers(-3, 4, len(q)), index=q, dtype=float)
    sent_idx = idx[-1500:]  # overlaps T_MID
    sentiment = pd.Series(rng.normal(0, 0.2, len(sent_idx)), index=sent_idx)
    world = PointInTimeData(close, vix, tnx, macro, sentiment, risk, "SYN")
    return world, pd.Series(s, index=idx)


@pytest.fixture(scope="module")
def world():
    return simulate_world()


T_MID = pd.Timestamp("2021-06-15")  # mid-quarter as-of date


@pytest.fixture(scope="module")
def result(world):
    return ensemble_forecast(world[0], asof=T_MID, config=FAST)


# --------------------------------------------------------------------------- #
# Point-in-time macro
# --------------------------------------------------------------------------- #

def test_pit_indicator_respects_publication_date():
    months = pd.date_range("2020-01-01", periods=30, freq="MS")
    fr = pd.DataFrame({"date": months, "value": np.arange(30, dtype=float) + 100,
                       "available": months + pd.Timedelta(days=45)})
    pit = pit_indicator(fr, CPI)
    # On 2021-06-10: May 2021 obs (published Jun 15) is NOT known; April's (May 16) is.
    f = macro_features({"CPIAUCSL": pit}, pd.DatetimeIndex(["2021-06-10", "2021-06-16"]))
    apr = pit.loc["2021-04-01", "yoy"]
    may = pit.loc["2021-05-01", "yoy"]
    assert f["cpi_yoy"].iloc[0] == pytest.approx(apr)
    assert f["cpi_yoy"].iloc[1] == pytest.approx(may)


def test_pit_availability_is_cummaxed():
    months = pd.date_range("2020-01-01", periods=4, freq="MS")
    late = pd.Timestamp("2020-12-01")  # Feb published very late (e.g. shutdown)
    fr = pd.DataFrame({"date": months, "value": [1.0, 2, 3, 4],
                       "available": [months[0] + pd.Timedelta(days=40), late,
                                     months[2] + pd.Timedelta(days=40), months[3] + pd.Timedelta(days=40)]})
    pit = pit_indicator(fr, UNRATE)
    assert pit["available"].is_monotonic_increasing
    assert pit.loc["2020-03-01", "available"] == late  # March can't be used before Feb


def test_pit_fallback_lag_without_vintages():
    months = pd.date_range("2020-01-01", periods=3, freq="MS")
    pit = pit_indicator(pd.DataFrame({"date": months, "value": [1.0, 2, 3]}), UNRATE)
    assert (pit["available"] == months + pd.Timedelta(days=45)).all()


def test_pit_daily_series_available_at_month_end():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    pit = pit_indicator(pd.DataFrame({"date": idx, "value": 1.0}), SPREAD)
    assert list(pit["available"]) == list(pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))


def test_asof_drops_unpublished_and_future(world):
    w, _ = world
    d = w.asof(T_MID)
    assert d.prices.index.max() <= T_MID and d.vix.index.max() <= T_MID
    assert all((m["available"] <= T_MID).all() for m in d.macro.values())
    assert len(d.sentiment) > 0 and d.sentiment.index.max() < T_MID
    assert d.risk_points.index.max() <= pd.Period(T_MID, "Q") + 4


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def test_price_features_ignore_future(world):
    w, _ = world
    full = price_features(w.prices, w.vix, w.tnx)
    cut = price_features(w.prices.loc[:T_MID], w.vix.loc[:T_MID], w.tnx.loc[:T_MID])
    pd.testing.assert_frame_equal(full.loc[:T_MID], cut)


def test_sentiment_feature_strictly_before_t():
    idx = pd.date_range("2024-01-01", periods=40)
    s = pd.Series(0.0, index=idx)
    s.iloc[-1] = 10.0  # the decision day itself
    t = idx[-1]
    assert sentiment_feature(s, pd.DatetimeIndex([t])).iloc[0] == 0.0
    assert np.isnan(sentiment_feature(None, pd.DatetimeIndex([t])).iloc[0])


def test_yield_scale_detection():
    assert to_yield_percent(pd.Series([42.0, 45.0])).iloc[0] == pytest.approx(4.2)
    assert to_yield_percent(pd.Series([4.2, 4.5])).iloc[0] == pytest.approx(4.2)


def test_target_dates_are_next_four_quarter_ends():
    assert target_dates(pd.Timestamp("2026-11-08")) == list(pd.to_datetime(
        ["2027-03-31", "2027-06-30", "2027-09-30", "2027-12-31"]))
    assert [str(pd.Period(d, "Q")) for d in target_dates(T_MID)] == future_quarter_labels(T_MID)


def test_business_days_between():
    assert business_days_between(pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-13")) == 5


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #

def test_regime_model_recovers_true_regimes(world):
    w, true_s = world
    obs = regime_observations(w.prices, w.vix)
    m = RegimeModel(n_init=2, n_iter=100).fit(obs.to_numpy())
    np.testing.assert_allclose(m.transmat_.sum(1), 1)
    assert m.covars_[2, 0, 0] == max(m.covars_[:, 0, 0])        # high_vol = largest variance
    assert m.means_[0, 0] > m.means_[1, 0]                      # bull mean > bear mean
    filt = m.filter(obs.to_numpy())
    np.testing.assert_allclose(filt.sum(1), 1)
    pred = filt.argmax(1)
    truth = true_s.reindex(obs.index).to_numpy()
    assert (pred == truth).mean() > 0.7
    assert filt[truth == 2, 2].mean() > 0.7                     # high-vol detected


def test_filter_is_causal(world):
    w, _ = world
    obs = regime_observations(w.prices, w.vix).to_numpy()
    m = RegimeModel(n_init=1, n_iter=30).fit(obs[:1500])
    full = m.filter(obs)
    np.testing.assert_allclose(m.filter(obs[:1000]), full[:1000])  # future obs change nothing


def test_refit_dates_only_completed_quarters():
    idx = pd.bdate_range("2020-01-01", "2021-05-14")
    r = refit_dates(idx, pd.Timestamp("2021-05-14"), every=1, min_obs=1)
    assert r[-1] == pd.Timestamp("2021-03-31")  # 2021Q2 not finished
    assert all(pd.Period(d, "Q").end_time.normalize() <= pd.Timestamp("2021-05-14") for d in r)


def test_regime_table_has_no_lookahead(world):
    w, _ = world
    obs = regime_observations(w.prices, w.vix)
    kw = dict(every=8, min_obs=500, n_init=1, n_iter=30)
    full = regime_feature_table(obs, obs.index[-1], **kw)
    cut = regime_feature_table(obs.loc[:T_MID], T_MID, **kw)
    pd.testing.assert_frame_equal(full.loc[:T_MID], cut)
    assert full.iloc[:499].isna().all().all()                  # nothing before first fit


# --------------------------------------------------------------------------- #
# GARCH + Monte Carlo
# --------------------------------------------------------------------------- #

def test_fit_garch_recovers_persistence():
    rng = np.random.default_rng(1)
    n, w, a, b = 4000, 2e-6, 0.08, 0.90
    var, eps = w / (1 - a - b), np.empty(n)
    for t in range(n):
        eps[t] = np.sqrt(var) * rng.standard_normal()
        var = w + a * eps[t] ** 2 + b * var
    g = fit_garch(eps)
    assert 0.93 < g.persistence < 0.995
    assert g.next_var > 0 and g.nu > 2


def test_mc_constant_vol_matches_normal_theory():
    g = GarchParams(omega=0.01 ** 2, alpha=0.0, beta=0.0, nu=1e6, next_var=0.01 ** 2)
    eye = np.eye(3)
    rets, dd = simulate_paths(np.zeros(3), eye, np.array([1, 0, 0.0]), g, [63, 252], 20000,
                              np.random.default_rng(0))
    assert np.std(rets[:, 1]) == pytest.approx(0.01 * np.sqrt(252), rel=0.03)
    assert abs(np.median(rets[:, 0])) < 0.005
    assert 0 < dd[0] <= dd[1] < 1                               # drawdown prob grows with horizon


def test_mc_regime_drift_direction():
    g = GarchParams(omega=1e-8, alpha=0.0, beta=0.0, nu=10, next_var=1e-6)
    drift = np.array([0.001, -0.001, 0.0])
    eye = np.eye(3)
    up, _ = simulate_paths(drift, eye, np.array([1, 0, 0.0]), g, [100], 2000, np.random.default_rng(0))
    dn, _ = simulate_paths(drift, eye, np.array([0, 1, 0.0]), g, [100], 2000, np.random.default_rng(0))
    assert np.median(up) == pytest.approx(0.1, abs=0.01)
    assert np.median(dn) == pytest.approx(-0.1, abs=0.01)


def test_mc_reproducible():
    g = GarchParams(1e-6, 0.05, 0.9, 6, 1e-4)
    P = np.full((3, 3), 1 / 3)
    a = simulate_paths(np.zeros(3), P, np.ones(3) / 3, g, [10, 20], 500, np.random.default_rng(7))
    b = simulate_paths(np.zeros(3), P, np.ones(3) / 3, g, [10, 20], 500, np.random.default_rng(7))
    np.testing.assert_array_equal(a[0], b[0])


# --------------------------------------------------------------------------- #
# Training set purging
# --------------------------------------------------------------------------- #

def test_training_set_is_purged(world):
    w, _ = world
    d = w.asof(T_MID)
    obs = regime_observations(d.prices, d.vix)
    regimes = regime_feature_table(obs, T_MID, every=8, min_obs=500, n_init=1, n_iter=30)
    X, y, meta = build_training_set(d, T_MID, regimes)
    assert len(X) == len(y) == len(meta) > 100
    assert (meta["target_date"] <= T_MID).all()
    assert (meta["decision_date"] < meta["target_date"]).all()
    assert list(X.columns) == feature_names()
    assert (X["days_to_target"] > 0).all()
    # last decision rows exist only for targets already realised
    last = meta[meta["decision_date"] == meta["decision_date"].max()]
    assert (last["target_date"] <= T_MID).all()


def test_training_targets_match_prices(world):
    w, _ = world
    d = w.asof(T_MID)
    obs = regime_observations(d.prices, d.vix)
    regimes = regime_feature_table(obs, T_MID, every=8, min_obs=500, n_init=1, n_iter=30)
    _, y, meta = build_training_set(d, T_MID, regimes)
    i = 17
    t, D = meta.loc[i, "decision_date"], meta.loc[i, "target_date"]
    assert y[i] == pytest.approx(np.log(d.prices.loc[:D].iloc[-1] / d.prices.loc[t]))


# --------------------------------------------------------------------------- #
# Ensemble end to end
# --------------------------------------------------------------------------- #

def test_output_contract(result):
    fc = result.forecast
    assert [q.quarter for q in fc.quarters] == future_quarter_labels(T_MID)
    assert all(q.p10 < q.p50 < q.p90 for q in fc.quarters)
    assert result.weights == {"mc": 0.5, "lgb": 0.5}
    assert result.n_train_rows >= FAST.min_train_rows
    dd = result.table["prob_drawdown"].to_numpy()
    assert ((0 <= dd) & (dd <= 1)).all() and (np.diff(dd) >= 0).all()
    assert result.prob_drawdown == dd[-1]
    np.testing.assert_allclose(sum(result.regime["current"].values()), 1)
    # uncertainty widens with horizon
    width = (result.table["blend_q90"] - result.table["blend_q10"]).to_numpy()
    assert (np.diff(width) > 0).all()


def test_blend_is_weighted_quantile_average(result):
    t = result.table
    raw = 0.5 * t[["mc_q10", "mc_q50", "mc_q90"]].to_numpy() + 0.5 * t[["lgb_q10", "lgb_q50", "lgb_q90"]].to_numpy()
    np.testing.assert_allclose(t[["blend_q10", "blend_q50", "blend_q90"]].to_numpy(), np.sort(raw, axis=1))
    np.testing.assert_allclose(t["p50"], result.forecast.last_close * np.exp(t["blend_q50"]))


def test_shap_additivity(result):
    d = result.drivers
    assert set(d["quantile"]) == set(QUANTILES)
    assert set(d["quarter"]) == set(future_quarter_labels(T_MID))
    for (lab, q), g in d.groupby(["quarter", "quantile"]):
        # SHAP is exact for trees: base value + contributions == model output
        assert g["base"].iloc[0] + g["shap"].sum() == pytest.approx(g["prediction"].iloc[0], abs=1e-9)
    # table holds the same predictions, sorted per quarter to remove crossing
    raw = d.groupby(["quarter", "quantile"])["prediction"].first().unstack()
    np.testing.assert_allclose(np.sort(raw.loc[result.table.index].to_numpy(), axis=1),
                               result.table[["lgb_q10", "lgb_q50", "lgb_q90"]].to_numpy())
    top = result.top_drivers(3)
    assert len(top) == 12 and set(top["feature"]) <= set(feature_names())


def test_no_lookahead_future_data_changes_nothing(world, result):
    """The core guarantee: forecasting at T with full history, with history cut
    at T, or with wildly altered post-T data must give identical output."""
    w, _ = world
    cut = ensemble_forecast(w.asof(T_MID), asof=T_MID, config=FAST)

    after = w.prices.index > T_MID
    macro = {k: v.assign(yoy=np.where(v["available"] > T_MID, 999.0, v["yoy"]))
             for k, v in w.macro.items()}
    tampered = PointInTimeData(
        prices=w.prices.where(~after, w.prices * 3),
        vix=w.vix.where(~after, 80.0),
        tnx=w.tnx.where(~after, 0.1),
        macro=macro,
        sentiment=w.sentiment.where(w.sentiment.index < T_MID, -1.0),
        risk_points=w.risk_points,
        ticker="SYN",
    )
    tam = ensemble_forecast(tampered, asof=T_MID, config=FAST)

    for other in (cut, tam):
        pd.testing.assert_frame_equal(result.table, other.table)
        pd.testing.assert_frame_equal(result.drivers, other.drivers)
        assert result.forecast == other.forecast
        assert result.regime["current"] == other.regime["current"]


def test_falls_back_to_mc_when_little_training_data(world):
    w, _ = world
    T = w.prices.index[560]  # ~22 purged training rows < min_train_rows (40)
    res = ensemble_forecast(w, asof=T, config=FAST)
    assert res.n_train_rows < FAST.min_train_rows
    assert res.weights == {"mc": 1.0, "lgb": 0.0}
    assert res.drivers.empty
    np.testing.assert_allclose(res.table["blend_q50"], res.table["mc_q50"])


def test_requires_enough_history(world):
    w, _ = world
    with pytest.raises(ValueError):
        ensemble_forecast(w, asof=w.prices.index[100], config=FAST)
