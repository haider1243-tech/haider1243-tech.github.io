import json
import math

import numpy as np
import pandas as pd
import pytest

from app_fixtures import TODAY, make_market_snapshot, make_ticker_snapshot
from forecaster.backtest import BacktestConfig, BacktestResult, evaluate
from forecaster.config import load_settings
from forecaster.pipeline import (
    REFERENCE_BACKTEST, _jsonable, backtest_is_stale, backtest_summary, load_backtest,
    read_json, safe_name, snapshot_dir, write_json,
)


@pytest.fixture(scope="module")
def tsnap():
    return make_ticker_snapshot()


def test_safe_name():
    assert safe_name("^GSPC") == "GSPC" and safe_name("GC=F") == "GC_F" and safe_name("dx-y.nyb") == "DX-Y_NYB"


def test_jsonable_handles_nan_timestamps_numpy():
    out = _jsonable({"a": np.float64("nan"), "b": pd.Timestamp("2026-01-02"), "c": np.int64(3),
                     "d": [np.bool_(True), float("inf")], "e": pd.Period("2026Q1")})
    assert out == {"a": None, "b": "2026-01-02T00:00:00", "c": 3, "d": [True, None], "e": "2026Q1"}
    json.dumps(out, allow_nan=False)


def test_ticker_snapshot_contract(tsnap, tmp_path):
    snap, res = tsnap
    assert [f["quarter"] for f in snap["forecast"]] == [q.quarter for q in res.forecast.quarters]
    assert all(f["p10"] < f["p50"] < f["p90"] for f in snap["forecast"])
    assert snap["drivers"] and {d["quantile"] for d in snap["drivers"]} == {0.1, 0.5, 0.9}
    assert len({(d["quarter"], d["quantile"]) for d in snap["drivers"]}) == 12
    hist_dates = pd.to_datetime([h["date"] for h in snap["history"]])
    assert hist_dates.max() <= res.as_of + pd.Timedelta(days=6)     # weekly bar ending after as_of only partially
    assert all(h["close"] is not None for h in snap["history"])
    p = tmp_path / "t.json"
    write_json(p, snap)
    back = read_json(p)
    assert back["forecast"][0]["p50"] == pytest.approx(snap["forecast"][0]["p50"])


def test_history_never_after_asof(tsnap):
    snap, res = tsnap
    # weekly resample labels bars on Friday; values must come from closes <= as_of
    world_last = max(pd.Timestamp(h["date"]) for h in snap["history"])
    assert world_last - res.as_of <= pd.Timedelta(days=6)


def test_market_snapshot_contents():
    m = make_market_snapshot()
    json.dumps(_jsonable(m), allow_nan=False)
    assert set(m["macro_history"]) == {"CPIAUCSL", "CPILFESL", "PPIACO", "FEDFUNDS"}
    first = pd.Timestamp(m["macro_history"]["CPIAUCSL"][0]["date"])
    assert first >= TODAY - pd.DateOffset(years=5)
    assert any(e["impact"] == "high" for e in m["events"])
    assert {"MARKET", "AAPL"} <= set(m["sentiment"])
    assert any(s["tag"] == "MARKET" for s in m["shocks"])
    assert m["headlines"] and all(pd.Timestamp(h["date"]) >= TODAY - pd.Timedelta(days=14) for h in m["headlines"])
    assert m["warnings"] == ["test warning"]


def test_market_snapshot_all_optional():
    from forecaster.pipeline import market_snapshot
    empty = market_snapshot(TODAY)
    assert empty["events"] == [] and empty["sentiment"] == {} and empty["macro_history"] == {}


def test_backtest_summary_flags():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.05, 40)
    o = pd.date_range("2005-03-31", periods=40, freq="QE")
    rows = pd.DataFrame({"origin": o, "horizon": 1, "target": o + pd.offsets.QuarterEnd(1), "y": y,
                         "realized_dd": 0.0, "dd_event": 0.0, "m_dd": 0.1, "b_dd": 0.1,
                         "m_q10": -0.2, "m_q50": 0.2, "m_q90": 0.25,     # bad model
                         "b_q10": -0.064, "b_q50": 0.0, "b_q90": 0.064})
    res = BacktestResult(rows, [], BacktestConfig(n_boot=300))
    s = backtest_summary(evaluate(res, windows={}), res, "X")
    assert s["worse_than_baseline"] and not s["beats_baseline"]
    assert s["n_forecasts"] == 40 and s["first_origin"] == o[0]
    json.dumps(_jsonable(s), allow_nan=False)


def test_reference_backtest_committed_and_honest():
    ref = read_json(REFERENCE_BACKTEST)
    assert ref is not None and ref["ticker"] == "^GSPC"
    lo, hi = ref["diff_ci"]
    assert ref["beats_baseline"] == (hi < 0) and ref["worse_than_baseline"] == (lo > 0)
    assert 0 <= ref["model"]["coverage"] <= 1 and ref["n_origins"] > 50


def test_load_backtest_falls_back_to_reference(tmp_path):
    s = load_settings(root=tmp_path, env={"FORECASTER_CACHE_DIR": str(tmp_path / "data")})
    bt, is_ref = load_backtest(s, "SPY")
    assert is_ref and bt["ticker"] == "^GSPC"
    assert backtest_is_stale(s, "SPY", 28)
    write_json(snapshot_dir(s) / "backtest_SPY.json", {**bt, "ticker": "SPY",
                                                       "generated_at": "2000-01-01T00:00:00+00:00"})
    bt2, is_ref2 = load_backtest(s, "SPY")
    assert not is_ref2 and bt2["ticker"] == "SPY"
    assert backtest_is_stale(s, "SPY", 28)
