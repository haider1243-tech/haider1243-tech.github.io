"""Shadow trading: same engine on live bars, upserts, pending setups, decay, never orders."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecaster import intraday as I
from forecaster import lab_engine as E
from forecaster import live_feed as lf
from forecaster import shadow as SH
from forecaster import strategies as S

SPY = I.INSTRUMENTS["SPY"]


def write_live(cache: Path, m1: pd.DataFrame, symbol="SPY"):
    store = lf.LiveStore(lf.live_db(cache))
    rows = [(symbol, int(t.value // 1_000_000), lf.Bar(r.Open, r.High, r.Low, r.Close, r.Volume, 10, 0))
            for t, r in m1.iterrows()]
    store.write([], rows, [], [])


def setup(params=None, verdict="NO EDGE"):
    st = S.LIBRARY["ema_cross"]
    return {"key": "ema_cross|SPY|5m", "strategy": "ema_cross", "instrument": "SPY", "timeframe": "5m",
            "params": {**st.params, **(params or {})}, "costs": E.DEFAULT_COSTS["SPY"].to_dict(),
            "verdict": verdict, "oos": {"expectancy": 0.0005, "trades": 300}, "ci": [0.0002, 0.0008]}


@pytest.fixture
def live(tmp_path):
    m1 = I.synthetic_m1("SPY", "2024-03-04", "2024-03-06", seed=3)
    write_live(tmp_path, m1)
    return tmp_path, m1


def test_shadow_matches_the_backtest_engine_on_the_same_bars(live):
    cache, m1 = live
    now = m1.index[-1] + pd.Timedelta(minutes=1)
    SH.started_at(SH.shadow_db(cache), m1.index[0])                     # shadow mode began before the data
    r = SH.run_once(cache, now=now, doc={"setups": [setup()]}, store=I.IntradayStore(cache / "none"))
    got = SH.read_trades(SH.shadow_db(cache))
    bars = I.resample(I.session_bars(m1, SPY), 5, SPY)
    ref = E.run(bars, SPY, 5, S.LIBRARY["ema_cross"].signals(S.Features(bars, SPY, 5)), E.DEFAULT_COSTS["SPY"],
                S.LIBRARY["ema_cross"].params["max_trades"])
    assert r["trades"] == len(ref) == len(got) >= 3
    got = got.sort_values("signal_time").reset_index(drop=True)
    np.testing.assert_allclose(got["net_ret"].to_numpy(float), ref["net_ret"].to_numpy(float))
    assert (got["status"] == "closed").all()


def test_only_signals_after_shadow_start_and_open_trades_close_later(live):
    cache, m1 = live
    day2 = pd.Timestamp("2024-03-05 09:30", tz=I.NY).tz_convert("UTC")
    db = SH.shadow_db(cache)
    SH.started_at(db, day2)
    noon = pd.Timestamp("2024-03-05 12:00", tz=I.NY).tz_convert("UTC")
    SH.run_once(cache, now=noon, doc={"setups": [setup({"rr": 50.0, "atr_mult": 50.0})]},   # can only time-exit
                store=I.IntradayStore(cache / "none"))
    t = SH.read_trades(db)
    assert len(t) >= 1 and (pd.to_datetime(t["signal_time"]) >= day2).all()
    assert set(t["status"]) == {"open"} and t["net_ret"].isna().all()
    end = m1.index[-1] + pd.Timedelta(minutes=1)
    SH.run_once(cache, now=end, doc={"setups": [setup({"rr": 50.0, "atr_mult": 50.0})]},
                store=I.IntradayStore(cache / "none"))
    t2 = SH.read_trades(db)
    first = t2[t2["signal_time"] == t["signal_time"].iloc[-1]].iloc[0]
    assert first["status"] == "closed" and first["reason"] == "time" and np.isfinite(first["net_ret"])
    # closed trades are immutable: a later pass with different params can't rewrite them
    with sqlite3.connect(db) as con:
        before = con.execute("SELECT exit, net_ret FROM shadow_trades WHERE status='closed'").fetchall()
    SH.run_once(cache, now=end, doc={"setups": [setup({"rr": 1.0})]}, store=I.IntradayStore(cache / "none"))
    with sqlite3.connect(db) as con:
        after = con.execute("SELECT exit, net_ret FROM shadow_trades WHERE status='closed'").fetchall()
    assert set(before) <= set(after)


def test_pending_setup_on_the_latest_bar():
    m1 = I.synthetic_m1("SPY", "2024-03-04", "2024-03-05", seed=11)
    bars5 = I.resample(m1, 5, SPY)
    sig = S.LIBRARY["ema_cross"].signals(S.Features(bars5, SPY, 5))
    i = int(sig.index[sig.index > 100][0])                               # a signal on day 2
    now = bars5.index[i] + pd.Timedelta(minutes=5)                       # bar i just completed
    trades, pending = SH.evaluate(m1[m1.index < now], "SPY", "5m", [setup()], m1.index[0], now)
    p = [x for x in pending if x["key"] == "ema_cross|SPY|5m"]
    assert len(p) == 1 and p[0]["signal_time"] == bars5.index[i].isoformat()
    assert p[0]["side"] == int(sig.loc[i, "side"]) and p[0]["stop"] == pytest.approx(sig.loc[i, "stop"])


def test_decay_check_and_weekly_rows(tmp_path):
    rng = np.random.default_rng(0)
    bt = rng.normal(0.001, 0.004, 400)
    assert SH.decay_check(rng.normal(0.001, 0.004, 10), bt, (0.0006, 0.0014)) == ("TOO FEW", None)
    flag, p = SH.decay_check(rng.normal(-0.002, 0.004, 40), bt, (0.0006, 0.0014))
    assert flag == "DECAY" and p < 0.05
    flag, _ = SH.decay_check(rng.normal(0.0011, 0.004, 60), bt, (0.0, 0.002))
    assert flag == "OK"
    # weekly rows from closed shadow trades vs saved OOS trades
    lab_dir = tmp_path / "lab" / "trades"
    lab_dir.mkdir(parents=True)
    pd.DataFrame({"net_ret": bt}).to_parquet(lab_dir / "ema_cross__SPY__5m.parquet")
    db = SH.shadow_db(tmp_path)
    SH.init_db(db)
    rows = [{"key": "ema_cross|SPY|5m", "signal_time": f"2026-11-0{1 + i % 9}T15:{i:02d}:00+00:00", "strategy": "ema_cross",
             "instrument": "SPY", "timeframe": "5m", "side": 1, "entry_time": "x", "entry": 1, "stop": 0.9, "target": 1.1,
             "exit_time": "y", "exit": 1, "reason": "stop", "net_ret": float(r), "r_mult": -1, "status": "closed",
             "verdict": "WEAK", "params": "{}", "updated_at": "z"} for i, r in enumerate(rng.normal(-0.003, 0.003, 25))]
    SH.write(db, rows, [], [], pd.Timestamp("2026-11-09", tz="UTC"))
    out = SH.weekly_decay(tmp_path, pd.Timestamp("2026-11-09 12:00", tz="UTC"), doc={"setups": [setup()]})
    assert len(out) == 1 and out[0]["flag"] == "DECAY" and out[0]["live_trades"] == 25 and out[0]["week"] == "2026-W46"
    txt = SH.decay_text(out)
    assert "🔻 ema_cross|SPY|5m" in txt and "DECAY" in txt


def test_shadow_can_never_place_orders():
    src = Path(SH.__file__).read_text()
    code = src.split('"""', 2)[2]                                          # skip the module docstring
    assert "execution" not in code and "OandaExecutor" not in code and "place_order" not in code
