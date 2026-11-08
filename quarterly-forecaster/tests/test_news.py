from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from conftest import RoutingSession
from forecaster.news import (
    MARKET_TAG,
    FinBertScorer,
    NewsStore,
    ShockConfig,
    collect_news,
    daily_index,
    match_topics,
    score_pending,
    shocks,
    text_key,
    weekly_index,
)
from forecaster.sources.finnhub import FinnhubClient


def ts(s: str) -> int:
    """'2026-11-05 15:00' New York time -> unix seconds."""
    return int(pd.Timestamp(s, tz="America/New_York").timestamp())


def item(i, headline, when="2026-11-05 10:00", summary=""):
    return {"id": i, "datetime": ts(when), "headline": headline, "summary": summary,
            "source": "X", "url": f"https://x/{i}"}


class FakeScorer:
    """Positive if 'beat'/'surge' in text, negative if 'miss'/'plunge', else neutral."""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts):
        self.seen += texts
        out = []
        for t in texts:
            low = t.lower()
            if "beat" in low or "surge" in low:
                out.append({"positive": 0.9, "negative": 0.05, "neutral": 0.05})
            elif "miss" in low or "plunge" in low:
                out.append({"positive": 0.05, "negative": 0.9, "neutral": 0.05})
            else:
                out.append({"positive": 0.1, "negative": 0.1, "neutral": 0.8})
        return out


@pytest.fixture
def store(tmp_path):
    return NewsStore(tmp_path / "news.sqlite")


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("Fed holds rates steady as Powell signals patience", ["fed"]),
    ("Investors are fed up with volatility", []),               # lowercase 'fed' is not the Fed
    ("Federal Reserve weighs rate cuts amid cooling CPI", ["fed", "inflation"]),
    ("New tariffs spark fears of a trade war and recession", ["tariffs", "recession"]),
    ("Gold hits record as bullion demand climbs", ["gold"]),
    ("Apple earnings beat; guidance raised", ["earnings"]),
    ("Inflationary pressures ease", ["inflation"]),
    ("Goldman Sachs upgrades bank stocks", []),                  # 'Goldman' is not gold
])
def test_match_topics(text, expected):
    assert match_topics(text) == expected


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #

def test_store_dedupes_and_tags_many_to_many(store):
    assert store.add([item(1, "Apple beats"), item(2, "Apple misses")], [("AAPL", "ticker")]) == 2
    assert store.add([item(1, "Apple beats")], [("AAPL", "ticker")]) == 0
    assert store.add([item(1, "Apple beats")], [("MSFT", "ticker")]) == 0  # same story, new tag
    c = store.counts()
    assert c["headlines"] == 2 and c["tags"] == 3


def test_store_skips_malformed(store):
    bad = [{"id": None, "datetime": 1, "headline": "x"}, {"id": 5, "datetime": 1, "headline": " "},
           {"id": 6, "headline": "no time"}]
    assert store.add(bad, [("AAPL", "ticker")]) == 0


def test_topics_only_tagged_on_market_news(store):
    store.add([item(1, "Fed cuts rates")], [("AAPL", "ticker")])
    store.add([item(2, "Fed cuts rates again")], [(MARKET_TAG, "market")])
    score_pending(store, FakeScorer())
    f = store.scored_frame()
    assert set(f.loc[f["id"] == "1", "tag"]) == {"AAPL"}
    assert set(f.loc[f["id"] == "2", "tag"]) == {MARKET_TAG, "fed"}


# --------------------------------------------------------------------------- #
# Scoring cache
# --------------------------------------------------------------------------- #

def test_old_headlines_not_rescored(store):
    scorer = FakeScorer()
    store.add([item(1, "Stocks surge"), item(2, "Stocks plunge")], [(MARKET_TAG, "market")])
    assert score_pending(store, scorer) == 2
    assert score_pending(store, scorer) == 0
    store.add([item(3, "Stocks flat")], [(MARKET_TAG, "market")])
    assert score_pending(store, scorer) == 1
    assert sorted(scorer.seen[:2]) == ["Stocks plunge", "Stocks surge"]
    assert scorer.seen[2:] == ["Stocks flat"]     # each text scored exactly once


def test_same_text_different_id_reuses_score(store):
    scorer = FakeScorer()
    store.add([item(1, "Nvidia beats estimates")], [("NVDA", "ticker")])
    score_pending(store, scorer)
    store.add([item(99, "Nvidia beats estimates")], [("AMD", "ticker")])
    assert score_pending(store, scorer) == 0
    assert set(store.scored_frame()["tag"]) == {"NVDA", "AMD"}


def test_scores_persist_across_store_instances(tmp_path):
    p = tmp_path / "n.sqlite"
    s1 = NewsStore(p)
    s1.add([item(1, "Stocks surge")], [(MARKET_TAG, "market")])
    score_pending(s1, FakeScorer())
    assert score_pending(NewsStore(p), FakeScorer()) == 0


def test_model_name_partitions_cache(store):
    store.add([item(1, "Stocks surge")], [(MARKET_TAG, "market")])
    score_pending(store, FakeScorer())
    assert score_pending(store, FakeScorer(), model="other/model") == 1
    assert text_key("a") != text_key("a", "other/model")


def test_score_pending_chunks_and_saves_progress(store):
    store.add([item(i, f"headline {i}") for i in range(10)], [(MARKET_TAG, "market")])

    class Boom(FakeScorer):
        def __call__(self, texts):
            if len(self.seen) >= 4:
                raise RuntimeError("crash")
            return super().__call__(texts)

    with pytest.raises(RuntimeError):
        score_pending(store, Boom(), chunk_size=4)
    assert store.counts()["scores"] == 4          # first chunk kept
    assert score_pending(store, FakeScorer()) == 6


def test_score_pending_rejects_wrong_length(store):
    store.add([item(1, "a"), item(2, "b")], [(MARKET_TAG, "market")])
    with pytest.raises(ValueError):
        score_pending(store, lambda texts: [{"positive": 1, "negative": 0, "neutral": 0}])


# --------------------------------------------------------------------------- #
# FinBERT wrapper (fake transformers pipeline)
# --------------------------------------------------------------------------- #

def test_finbert_scorer_maps_labels_and_loads_lazily():
    created = []

    def factory(task, **kwargs):
        created.append((task, kwargs))

        def pipe(texts, batch_size):
            return [[{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.2},
                     {"label": "neutral", "score": 0.1}] for _ in texts]
        return pipe

    s = FinBertScorer(pipeline_factory=factory, batch_size=8)
    assert created == []                         # lazy
    assert s([]) == []
    assert created == []
    out = s(["a", "b"])
    s(["c"])
    assert len(created) == 1                     # loaded once
    task, kw = created[0]
    assert task == "text-classification"
    assert kw["model"] == "ProsusAI/finbert" and kw["top_k"] is None and kw["truncation"]
    assert out == [{"positive": 0.7, "negative": 0.2, "neutral": 0.1}] * 2


def test_finbert_scorer_rejects_unexpected_labels():
    s = FinBertScorer(pipeline_factory=lambda *a, **k: (lambda t, batch_size: [[{"label": "LABEL_0", "score": 1}]]))
    with pytest.raises(ValueError):
        s(["x"])


# --------------------------------------------------------------------------- #
# Collection (mocked Finnhub)
# --------------------------------------------------------------------------- #

def finnhub_handler(url, params):
    if url.endswith("company-news"):
        if params["symbol"] == "BOOM":
            return 429, {}
        return [item(f"{params['symbol']}-1", f"{params['symbol']} beats estimates")]
    if url.endswith("/news"):
        assert params["category"] == "general"
        return [item(100, "Fed signals rate cuts as inflation cools"),
                item(101, "Gold surges on tariff fears")]
    raise AssertionError(url)


def test_collect_news(store, cache):
    session = RoutingSession(finnhub_handler)
    fh = FinnhubClient("k", cache, session=session)
    errs = []
    added = collect_news(fh, store, ["AAPL", "aapl", "^VIX", "GC=F", "BOOM"],
                         pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-05"),
                         on_error=lambda t, e: errs.append(t))
    assert added == {"AAPL": 1, MARKET_TAG: 2}
    assert errs == ["BOOM"]
    symbols = [p.get("symbol") for _, p, _ in session.calls if "symbol" in p]
    assert symbols == ["AAPL", "BOOM"]           # deduped, non-equities skipped

    score_pending(store, FakeScorer())
    tags = set(store.scored_frame()["tag"])
    assert tags == {"AAPL", MARKET_TAG, "fed", "inflation", "gold", "tariffs"}

    # second run: HTTP cached, nothing new stored
    n = len(session.calls)
    added2 = collect_news(fh, store, ["AAPL"], pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-05"))
    assert len(session.calls) == n
    assert added2 == {"AAPL": 0, MARKET_TAG: 0}


# --------------------------------------------------------------------------- #
# Indices
# --------------------------------------------------------------------------- #

def test_dates_use_new_york_time(store):
    # 2026-11-05 22:30 New York == 2026-11-06 03:30 UTC -> should be Nov 5.
    store.add([item(1, "Stocks surge", when="2026-11-05 22:30")], [(MARKET_TAG, "market")])
    score_pending(store, FakeScorer())
    assert store.scored_frame()["date"].iloc[0] == pd.Timestamp("2026-11-05")


def test_daily_and_weekly_index_values(store):
    store.add([
        item(1, "AAPL beats", "2026-11-02 09:00"),
        item(2, "AAPL misses", "2026-11-02 12:00"),
        item(3, "AAPL beats again", "2026-11-03 09:00"),
        item(4, "AAPL quiet day", "2026-11-09 09:00"),
    ], [("AAPL", "ticker")])
    score_pending(store, FakeScorer())
    d = daily_index(store.scored_frame()).loc["AAPL"]
    assert d.loc["2026-11-02", "sentiment"] == pytest.approx(0.0)
    assert d.loc["2026-11-02", "headlines"] == 2
    assert d.loc["2026-11-03", "sentiment"] == pytest.approx(0.85)
    assert len(d) == 3                            # no rows for days without news

    w = weekly_index(store.scored_frame()).loc["AAPL"]
    assert list(w.index) == [pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-13")]
    assert w.iloc[0]["headlines"] == 3
    assert w.iloc[0]["sentiment"] == pytest.approx((0.85 - 0.85 + 0.85) / 3)  # headline-weighted


def test_empty_index():
    assert daily_index(pd.DataFrame()).empty


def _synthetic_frame(daily_values, tag="MARKET", per_day=5, start="2026-01-01"):
    rows, k = [], 0
    for d, v in zip(pd.date_range(start, periods=len(daily_values)), daily_values):
        for _ in range(per_day):
            rows.append({"id": str(k), "tag": tag, "date": d, "sentiment": v})
            k += 1
    return pd.DataFrame(rows)


def test_shock_detected_on_sentiment_spike():
    rng = np.random.default_rng(0)
    vals = list(rng.normal(0.05, 0.05, 40)) + [-0.8] + list(rng.normal(0.05, 0.05, 5))
    d = daily_index(_synthetic_frame(vals)).loc["MARKET"]
    spike_day = pd.date_range("2026-01-01", periods=46)[40]
    assert d.loc[spike_day, "shock"]
    assert d.loc[spike_day, "shock_direction"] == -1
    assert d.loc[spike_day, "z"] < -10
    s = shocks(daily_index(_synthetic_frame(vals)))
    assert spike_day in s.index.get_level_values("date")
    assert s["z"].idxmin() == ("MARKET", spike_day)


def test_false_alarm_rate_on_pure_noise_is_about_five_percent():
    # |z| > 2 is a 2-sigma rule: on news-free noise it still fires ~5% of days
    # (a bit more with short baselines). Documented so nobody reads every
    # flag as a real event.
    rng = np.random.default_rng(3)
    d = daily_index(_synthetic_frame(rng.normal(0, 0.1, 2000)))
    rate = d["shock"].mean()
    assert 0.03 < rate < 0.09


def test_id_zero_is_kept(store):
    assert store.add([item(0, "Stocks surge")], [(MARKET_TAG, "market")]) == 1


def test_z_uses_only_prior_days():
    rng = np.random.default_rng(1)
    vals = rng.normal(0, 0.1, 50)
    d = daily_index(_synthetic_frame(vals)).loc["MARKET"]
    base = pd.Series(vals[10:40])                 # 30 days before day 40
    assert d["z"].iloc[40] == pytest.approx((vals[40] - base.mean()) / base.std())


def test_no_z_without_enough_history_and_thin_days_not_flagged():
    rng = np.random.default_rng(2)
    vals = list(rng.normal(0, 0.02, 9)) + [0.9]
    d = daily_index(_synthetic_frame(vals)).loc["MARKET"]
    assert d["z"].isna().all() and not d["shock"].any()

    vals = list(rng.normal(0, 0.02, 30)) + [0.9]
    thin = daily_index(_synthetic_frame(vals, per_day=1)).loc["MARKET"]
    assert thin["z"].iloc[-1] > 2 and not thin["shock"].iloc[-1]
    assert daily_index(_synthetic_frame(vals, per_day=1), ShockConfig(min_headlines=1)).loc["MARKET"]["shock"].iloc[-1]


def test_duplicate_id_tag_counted_once():
    f = _synthetic_frame([0.5, 0.5], per_day=2)
    f = pd.concat([f, f])
    assert (daily_index(f)["headlines"] == 2).all()
