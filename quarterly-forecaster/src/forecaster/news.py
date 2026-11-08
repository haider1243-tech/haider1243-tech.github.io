"""News sentiment: Finnhub headlines -> FinBERT scores -> daily/weekly index + shocks.

Pipeline
--------
1. collect_news(): company headlines per equity ticker (Finnhub `company-news`)
   plus market headlines (`news?category=general`). Every market headline is
   tagged MARKET, and also with any macro topic (Fed, inflation, tariffs,
   recession, earnings, gold) its headline/summary matches. Finnhub's free
   tier has no keyword search, so topics come from regex matching here.
   Headlines are stored in SQLite (data/news.sqlite) and accumulate across
   runs. That matters because the free market-news endpoint only returns the
   latest ~100 items, so history only exists if you keep running this.
2. score_pending(): FinBERT (ProsusAI/finbert, run locally) scores only
   headlines whose text hash isn't in the score table yet. Old headlines
   are never re-scored, and the same text under another id or ticker reuses
   its score.
3. daily_index() / weekly_index(): mean of (P(positive) - P(negative)) per
   tag, a value in [-1, 1], with headline counts.
4. Shock flags: a day is an event shock when |z| > 2, where
   z = (today - mean of the prior `window` news days) / their std, and the
   day has at least `min_headlines` headlines. The current day is never part
   of its own baseline.

Dates are America/New_York calendar dates of publication.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd

from forecaster.config import is_equity_symbol
from forecaster.sources.finnhub import FinnhubClient

MODEL_NAME = "ProsusAI/finbert"
MARKET_TAG = "MARKET"
NEWS_TZ = "America/New_York"

# Topic -> regex over headline + summary (case-insensitive unless scoped).
TOPICS: dict[str, re.Pattern] = {
    "fed": re.compile(
        r"(?-i:\bFed\b)|\bfederal reserve\b|\bFOMC\b|\bpowell\b|\brate (?:cut|hike)s?\b", re.I
    ),
    "inflation": re.compile(r"\binflation\w*|\bdisinflation|\bCPI\b|\bPPI\b|\bPCE\b", re.I),
    "tariffs": re.compile(r"\btariffs?\b|\btrade war\b|\bimport dut(?:y|ies)\b", re.I),
    "recession": re.compile(r"\brecession\w*|\beconomic downturn\b|\bhard landing\b", re.I),
    "earnings": re.compile(r"\bearnings\b|\bEPS\b|\bquarterly (?:results|profit)\b|\bguidance\b", re.I),
    "gold": re.compile(r"\bgold\b|\bbullion\b", re.I),
}


def match_topics(text: str) -> list[str]:
    return [name for name, pat in TOPICS.items() if pat.search(text or "")]


def text_key(text: str, model: str = MODEL_NAME) -> str:
    return hashlib.sha256(f"{model}\x00{text.strip()}".encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id        TEXT PRIMARY KEY,
    ts        INTEGER NOT NULL,     -- unix seconds (UTC)
    headline  TEXT NOT NULL,
    summary   TEXT,
    source    TEXT,
    url       TEXT
);
CREATE TABLE IF NOT EXISTS tags (
    id   TEXT NOT NULL,
    tag  TEXT NOT NULL,
    kind TEXT NOT NULL,             -- ticker | market | topic
    PRIMARY KEY (id, tag)
);
CREATE TABLE IF NOT EXISTS scores (
    text_hash TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    positive  REAL NOT NULL,
    negative  REAL NOT NULL,
    neutral   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
"""


class NewsStore:
    """SQLite store for headlines, their tags, and FinBERT scores."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, items: Iterable[dict], tags: Iterable[tuple[str, str]]) -> int:
        """Insert headlines (ignored if id exists) and attach (tag, kind) to each.

        Returns the number of new headlines.
        """
        tags = list(tags)
        # Macro topics are tagged on market-wide news only, so topic indices
        # aren't skewed by whichever companies happen to be on the watchlist.
        tag_topics = any(kind == "market" for _, kind in tags)
        new = 0
        with closing(self._connect()) as con, con:
            for it in items:
                raw_id = it.get("id")
                hid = "" if raw_id is None else str(raw_id).strip()  # id 0 is valid
                headline = (it.get("headline") or "").strip()
                if not hid or not headline or not it.get("datetime"):
                    continue
                cur = con.execute(
                    "INSERT OR IGNORE INTO headlines VALUES (?,?,?,?,?,?)",
                    (hid, int(it["datetime"]), headline, it.get("summary") or "",
                     it.get("source") or "", it.get("url") or ""),
                )
                new += cur.rowcount
                all_tags = list(tags)
                if tag_topics:
                    text = f"{headline} {it.get('summary') or ''}"
                    all_tags += [(t, "topic") for t in match_topics(text)]
                con.executemany(
                    "INSERT OR IGNORE INTO tags VALUES (?,?,?)",
                    [(hid, tag, kind) for tag, kind in all_tags],
                )
        return new

    def unscored(self, model: str = MODEL_NAME) -> list[str]:
        """Distinct headline texts with no score for `model`."""
        with closing(self._connect()) as con:
            texts = [r[0] for r in con.execute("SELECT DISTINCT headline FROM headlines")]
            have = {r[0] for r in con.execute("SELECT text_hash FROM scores WHERE model=?", (model,))}
        return [t for t in texts if text_key(t, model) not in have]

    def put_scores(self, rows: Iterable[tuple[str, dict[str, float]]], model: str = MODEL_NAME) -> None:
        with closing(self._connect()) as con, con:
            con.executemany(
                "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)",
                [(text_key(t, model), model, p["positive"], p["negative"], p["neutral"])
                 for t, p in rows],
            )

    def scored_frame(self, model: str = MODEL_NAME) -> pd.DataFrame:
        """One row per (headline, tag) that has a score."""
        with closing(self._connect()) as con:
            h = pd.read_sql("SELECT id, ts, headline, source, url FROM headlines", con)
            t = pd.read_sql("SELECT id, tag, kind FROM tags", con)
            s = pd.read_sql("SELECT text_hash, positive, negative, neutral FROM scores WHERE model=?",
                            con, params=(model,))
        if h.empty or s.empty:
            return pd.DataFrame(columns=["id", "ts", "headline", "tag", "kind", "date",
                                         "positive", "negative", "neutral", "sentiment"])
        h["text_hash"] = [text_key(x, model) for x in h["headline"]]
        df = h.merge(s, on="text_hash").merge(t, on="id")
        df["date"] = (pd.to_datetime(df["ts"], unit="s", utc=True)
                      .dt.tz_convert(NEWS_TZ).dt.tz_localize(None).dt.normalize())
        df["sentiment"] = df["positive"] - df["negative"]
        return df.drop(columns="text_hash")

    def counts(self) -> dict[str, int]:
        with closing(self._connect()) as con:
            return {
                tbl: con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                for tbl in ("headlines", "tags", "scores")
            }


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def collect_news(
    finnhub: FinnhubClient,
    store: NewsStore,
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    on_error: Callable[[str, Exception], None] | None = None,
) -> dict[str, int]:
    """Fetch (cached) and store headlines. Returns {source tag: new headlines}."""
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    added: dict[str, int] = {}

    def run(tag: str, fetch: Callable[[], list[dict]], tags: list[tuple[str, str]]) -> None:
        try:
            added[tag] = store.add(fetch(), tags)
        except Exception as exc:
            if on_error is None:
                raise
            on_error(tag, exc)

    for t in dict.fromkeys(x.upper() for x in tickers):
        if is_equity_symbol(t):
            run(t, lambda t=t: finnhub.company_news(t, s, e), [(t, "ticker")])
    run(MARKET_TAG, lambda: finnhub.market_news("general"), [(MARKET_TAG, "market")])
    return added


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

class Scorer(Protocol):
    def __call__(self, texts: list[str]) -> list[dict[str, float]]: ...


class FinBertScorer:
    """Local FinBERT via transformers. Loads the model lazily on first call.

    `pipeline_factory` exists so tests can inject a fake; by default it builds
    a transformers text-classification pipeline (CPU unless device is given).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 32,
        device: int | str | None = None,
        pipeline_factory: Callable[..., Callable] | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._factory = pipeline_factory
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            factory = self._factory
            if factory is None:
                try:
                    from transformers import pipeline as factory
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "FinBERT needs transformers + torch: pip install -r requirements-ml.txt"
                    ) from exc
            self._pipe = factory(
                "text-classification", model=self.model_name, top_k=None,
                truncation=True, device=self.device,
            )
        return self._pipe

    def __call__(self, texts: list[str]) -> list[dict[str, float]]:
        if not texts:
            return []
        raw = self._load()(texts, batch_size=self.batch_size)
        out = []
        for labels in raw:
            probs = {d["label"].lower(): float(d["score"]) for d in labels}
            missing = {"positive", "negative", "neutral"} - probs.keys()
            if missing:
                raise ValueError(f"Unexpected FinBERT labels, missing {missing}: {labels}")
            out.append({k: probs[k] for k in ("positive", "negative", "neutral")})
        return out


def score_pending(
    store: NewsStore,
    scorer: Scorer,
    model: str = MODEL_NAME,
    chunk_size: int = 256,
) -> int:
    """Score headlines that have no cached score yet. Returns number scored.

    Writes each chunk as it finishes, so an interrupted run keeps its progress.
    """
    pending = store.unscored(model)
    for i in range(0, len(pending), chunk_size):
        chunk = pending[i : i + chunk_size]
        probs = scorer(chunk)
        if len(probs) != len(chunk):
            raise ValueError(f"Scorer returned {len(probs)} results for {len(chunk)} texts")
        store.put_scores(zip(chunk, probs), model)
    return len(pending)


# --------------------------------------------------------------------------- #
# Indices + shocks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ShockConfig:
    window: int = 30          # prior news days in the baseline
    min_periods: int = 10     # baseline days needed before z is defined
    threshold: float = 2.0    # |z| above this is a shock
    min_headlines: int = 3    # ignore thin days


def _add_shocks(df: pd.DataFrame, cfg: ShockConfig) -> pd.DataFrame:
    """Add z / headline_z / shock columns to a single-tag index frame."""
    base = df["sentiment"].shift(1).rolling(cfg.window, min_periods=cfg.min_periods)
    mu, sd = base.mean(), base.std()
    df["z"] = (df["sentiment"] - mu) / sd.replace(0, np.nan)
    nb = df["headlines"].shift(1).rolling(cfg.window, min_periods=cfg.min_periods)
    df["headline_z"] = (df["headlines"] - nb.mean()) / nb.std().replace(0, np.nan)
    df["shock"] = (df["z"].abs() > cfg.threshold) & (df["headlines"] >= cfg.min_headlines)
    df["shock_direction"] = np.where(df["shock"], np.sign(df["z"]), 0).astype(int)
    return df


def _index(frame: pd.DataFrame, rule: str | None, cfg: ShockConfig) -> pd.DataFrame:
    cols = ["tag", "date", "sentiment", "headlines", "z", "headline_z", "shock", "shock_direction"]
    if frame.empty:
        return pd.DataFrame(columns=cols).set_index(["tag", "date"])
    # A headline tagged twice under the same tag (re-fetch) counts once.
    f = frame.drop_duplicates(["id", "tag"])
    parts = []
    for tag, g in f.groupby("tag"):
        grouper = pd.Grouper(key="date", freq=rule) if rule else "date"
        agg = g.groupby(grouper)["sentiment"].agg(["mean", "size"])
        agg = agg[agg["size"] > 0].rename(columns={"mean": "sentiment", "size": "headlines"})
        agg = _add_shocks(agg, cfg)
        agg["tag"] = tag
        parts.append(agg.reset_index())
    return pd.concat(parts, ignore_index=True)[cols].set_index(["tag", "date"]).sort_index()


def daily_index(frame: pd.DataFrame, cfg: ShockConfig = ShockConfig()) -> pd.DataFrame:
    """Per (tag, date): mean sentiment, headline count, z vs prior days, shock flag."""
    return _index(frame, None, cfg)


def weekly_index(
    frame: pd.DataFrame,
    cfg: ShockConfig = ShockConfig(window=12, min_periods=6, min_headlines=5),
) -> pd.DataFrame:
    """Same as daily_index on Friday-ending weeks (mean over all headlines in the week)."""
    return _index(frame, "W-FRI", cfg)


def shocks(index: pd.DataFrame) -> pd.DataFrame:
    return index[index["shock"]].copy()


def main() -> int:  # pragma: no cover - network + model wrapper
    import argparse
    import sys

    from forecaster.cache import Cache
    from forecaster.config import load_settings

    parser = argparse.ArgumentParser(description="News sentiment index")
    parser.add_argument("--days", type=int, default=30, help="company-news lookback")
    parser.add_argument("--no-score", action="store_true", help="collect only")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.finnhub_api_key:
        print("FINNHUB_API_KEY is not set: copy .env.example to .env and add a key", file=sys.stderr)
        return 1
    http_cache = Cache(settings.cache_dir / "news_http", ttl_seconds=3600)  # news goes stale fast
    fh = FinnhubClient(settings.finnhub_api_key, http_cache)
    store = NewsStore(settings.cache_dir / "news.sqlite")

    end = pd.Timestamp.today().normalize()
    added = collect_news(fh, store, settings.tickers, end - pd.Timedelta(days=args.days), end,
                         on_error=lambda t, e: print(f"[warn] {t}: {e}", file=sys.stderr))
    print(f"New headlines: {sum(added.values())} {added}")
    if not args.no_score:
        print(f"Scored {score_pending(store, FinBertScorer())} new headline texts")

    frame = store.scored_frame()
    d = daily_index(frame)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(d.groupby(level="tag").tail(1).round(3))
        s = shocks(d)
        print(f"\nShocks (|z| > 2): {len(s)}")
        if len(s):
            print(s.tail(20).round(2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
