"""Lab pipeline: random walk -> NO EDGE; a planted pattern -> EDGE; results file + leaderboard."""
import json

import numpy as np
import pandas as pd
import pytest

from forecaster import intraday as I
from forecaster import lab
from forecaster import strategies as S

SPY = I.INSTRUMENTS["SPY"]


def plant_momentum(df: pd.DataFrame, size: float = 0.004) -> pd.DataFrame:
    """The 09:30-10:00 direction continues 10:00-11:00 (a real, tradeable pattern)."""
    out = df.copy()
    t = df.index.tz_convert(I.NY)
    for _, g in df.groupby(t.normalize()):
        mm = g.index.tz_convert(I.NY).hour * 60 + g.index.tz_convert(I.NY).minute
        sgn = np.sign(g["Close"][mm < 600].iloc[-1] / g["Open"].iloc[0] - 1)
        f = np.exp(np.clip((mm - 600) / 60, 0, 1) * size * sgn)
        for c in ("Open", "High", "Low", "Close"):
            out.loc[g.index, c] = g[c].to_numpy() * f
    return out


@pytest.fixture(scope="module")
def spy_m1():
    return I.synthetic_m1("SPY", "2022-01-03", "2024-12-31", seed=5)


def _bars(m1):
    return I.resample(I.session_bars(m1, SPY), 5, SPY)


def test_random_walk_has_no_edge(spy_m1):
    for key in ("orb30", "vwap_reversion", "ema_cross"):
        res, oos = lab.evaluate_setup(_bars(spy_m1), "SPY", "5m", S.LIBRARY[key], n_random=500)
        assert res["verdict"] != "EDGE", (key, res["p_value"])
        assert res["oos"]["trades"] == len(oos) > 100


def test_planted_pattern_is_found(spy_m1):
    res, oos = lab.evaluate_setup(_bars(plant_momentum(spy_m1)), "SPY", "5m", S.LIBRARY["orb30"], n_random=500)
    assert res["verdict"] == "EDGE" and res["p_value"] < 0.01
    assert res["ci"][0] > 0 and res["oos"]["trades"] >= 100
    assert res["random"]["quantiles"][95] < res["oos"]["expectancy"]
    assert res["oos_by_side"]["long"]["trades"] + res["oos_by_side"]["short"]["trades"] == res["oos"]["trades"]
    assert len(res["folds"]) == 8 and all(f["oos_trades"] > 0 for f in res["folds"])


def test_run_lab_writes_results_bonferroni_and_leaderboard(tmp_path, spy_m1):
    bars = {("SPY", "5m"): _bars(plant_momentum(spy_m1))}
    doc = lab.run_lab(bars, tmp_path, strategies=["orb30", "ema_cross"], n_random=200, progress=lambda m: None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert [s["key"] for s in saved["setups"]] == ["orb30|SPY|5m", "ema_cross|SPY|5m"]
    assert all(s["setups_tested"] == 2 for s in saved["setups"])
    top = lab.leaderboard(saved)
    assert top[0]["strategy"] == "orb30" and top[0]["bonferroni_ok"]
    assert lab.leaderboard(saved, instrument="QQQ") == []
    assert len(pd.read_parquet(tmp_path / "trades" / "orb30__SPY__5m.parquet")) == doc["setups"][0]["oos"]["trades"]


def test_orb_not_run_on_too_coarse_timeframes(tmp_path, spy_m1):
    m15 = I.resample(I.session_bars(spy_m1.loc[:"2022-06-30"], SPY), 15, SPY)
    doc = lab.run_lab({("SPY", "15m"): m15}, tmp_path, strategies=["orb5", "orb15", "orb30"], n_random=50,
                      progress=lambda m: None, train_months=3, test_months=1)
    assert [s["strategy"] for s in doc["setups"]] == ["orb15", "orb30"]


def test_luck_test_is_calibrated_on_random_walks():
    """On pure noise p-values should be ~uniform: p < 0.05 about 5% of the time (not 20%)."""
    ps = []
    for seed in range(24):
        m1 = I.session_bars(I.synthetic_m1("SPY", "2023-01-02", "2024-06-28", seed=2000 + seed), SPY)
        res, _ = lab.evaluate_setup(I.resample(m1, 15, SPY), "SPY", "15m", S.LIBRARY["ema_cross"], n_random=200,
                                    train_months=6, test_months=3)
        ps.append(1.0 if res["p_value"] is None else res["p_value"])
    ps = np.array(ps)
    assert np.mean(ps < 0.05) <= 0.15 and 0.25 < np.median(ps) < 0.75, ps
