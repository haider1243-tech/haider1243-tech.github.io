"""Market data: daily OHLCV via yfinance, parquet cache, resampling and metrics.

Pipeline:
    store = ParquetStore(settings.cache_dir / "ohlcv")
    frames = fetch_universe(market_universe(settings.tickers), store)
    metrics = compute_metrics(frames)

Caching: one parquet file per ticker. On each request the store checks whether
the newest bar is older than the most recent completed business day. Only then
does it download — incrementally, from a few days before the last stored bar —
and merge. A sidecar JSON records when each ticker was last checked so that
holidays (no new bar exists) don't trigger a re-download on every run.

All I/O is injectable (`downloader`, `now`) so tests run offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from forecaster.config import merge_tickers

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TRADING_DAYS = 252

BENCHMARK = "SPY"
MACRO_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "^VIX",
    "^TNX",       # 10Y Treasury yield (x10)
    "DX-Y.NYB",   # US dollar index
    "GC=F",       # gold futures
)
SECTOR_ETFS: tuple[str, ...] = ("XLK", "XLF", "XLE", "XLV", "XLY")
CANADIAN_ETFS: tuple[str, ...] = ("XIU.TO", "XEQT.TO", "VFV.TO", "ZAG.TO")

# --------------------------------------------------------------------------- #
# Currency: every money value is reported in CAD
# --------------------------------------------------------------------------- #
BASE_CURRENCY = "CAD"
USD_CAD_SYMBOL = "CAD=X"          # Yahoo: CAD per 1 USD (~1.35)
FRED_USD_CAD = "DEXCAUS"          # FRED daily noon rate, CAD per 1 USD, since 1971
CAD_SUFFIXES = (".TO", ".V", ".NE", ".CN")
# Not money: volatility, yields, the dollar index and the FX rate itself stay as-is.
NON_MONETARY: frozenset[str] = frozenset({"^VIX", "^VVIX", "^TNX", "^IRX", "^FVX", "^TYX",
                                          "DX-Y.NYB", USD_CAD_SYMBOL})
USD_CAD_RANGE = (1.0, 2.0)        # sanity band: anything outside is inverted or broken


def ticker_currency(ticker: str, overrides: dict[str, str] | None = None) -> str | None:
    """'CAD', 'USD', or None for non-monetary series (index points, yields)."""
    t = ticker.upper()
    if overrides and t in overrides:
        v = overrides[t]
        return None if v in (None, "", "none", "NONE") else v.upper()
    if t in NON_MONETARY:
        return None
    if t.endswith(CAD_SUFFIXES):
        return "CAD"
    return "USD"


def validate_usd_cad(rates: pd.Series) -> pd.Series:
    r = rates.dropna().sort_index()
    if r.empty:
        raise ValueError("no USD/CAD rates")
    med = float(r.median())
    if not (USD_CAD_RANGE[0] <= med <= USD_CAD_RANGE[1]):
        raise ValueError(f"USD/CAD median {med:.4f} outside {USD_CAD_RANGE}: series looks inverted "
                         "(CAD/USD) or isn't USD/CAD at all; refusing to convert")
    if (r <= 0).any():
        raise ValueError("USD/CAD contains non-positive rates")
    return r


def usd_cad_rates(store: "ParquetStore", fred=None,
                  on_warn: Callable[[str], None] | None = None) -> pd.Series:
    """Daily USD/CAD (CAD per USD) from Yahoo CAD=X, back-filled before its first
    date with FRED DEXCAUS when a FRED client is given. Date index, tz-naive."""
    yahoo = store.get(USD_CAD_SYMBOL)["Close"]
    yahoo.index = pd.DatetimeIndex(yahoo.index).normalize()
    rates = yahoo
    if fred is not None:
        try:
            older = fred.get_series(FRED_USD_CAD)
            older = older[older.index < yahoo.index.min()]
            rates = pd.concat([older, yahoo])
        except Exception as exc:  # FRED is only a back-fill; Yahoo alone still works
            if on_warn:
                on_warn(f"DEXCAUS back-fill failed, USD/CAD starts {yahoo.index.min():%Y-%m-%d}: {exc}")
    rates = rates[~rates.index.duplicated(keep="last")]
    return validate_usd_cad(rates).rename("usd_cad")


def rate_on_dates(rates: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Rate for each date = last known rate on or before that date (never a later one)."""
    r = validate_usd_cad(rates)
    d = pd.DatetimeIndex(dates)
    key = d.tz_convert(None) if d.tz is not None else d
    pos = r.index.searchsorted(key.normalize(), side="right") - 1
    vals = np.where(pos >= 0, r.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=dates, name="usd_cad")


def to_cad(df: pd.DataFrame, ticker: str, rates: pd.Series | None,
           overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Convert a USD OHLCV frame (or a Close-only frame) to CAD with daily USD/CAD.

    CAD and non-monetary tickers come back unchanged. Bars before the first
    available rate are dropped rather than guessed. Volume is not converted.
    The result carries attrs: currency, converted_from.
    """
    cur = ticker_currency(ticker, overrides)
    out = df.copy()
    if cur is None:
        out.attrs.update(currency=None, converted_from=None)
        return out
    if cur == BASE_CURRENCY:
        out.attrs.update(currency=BASE_CURRENCY, converted_from=None)
        return out
    if cur != "USD":
        raise ValueError(f"{ticker}: no conversion from {cur} to {BASE_CURRENCY}")
    if rates is None:
        raise ValueError(f"{ticker} is priced in USD and no USD/CAD rates are available")
    fx = rate_on_dates(rates, out.index)
    keep = fx.notna().to_numpy()
    out = out[keep]
    fx = fx[keep]
    for c in ("Open", "High", "Low", "Close", "open", "high", "low", "close"):
        if c in out:
            out[c] = out[c] * fx.to_numpy()
    out.attrs.update(currency=BASE_CURRENCY, converted_from="USD")
    return out


def close_in_cad(store: "ParquetStore", ticker: str, rates: pd.Series | None,
                 overrides: dict[str, str] | None = None) -> pd.Series:
    """Daily close in CAD (or unconverted for non-monetary tickers)."""
    df = to_cad(store.get(ticker)[["Close"]], ticker, rates, overrides)
    s = df["Close"].rename(ticker)
    s.attrs.update(df.attrs)
    return s

# Downloader(ticker, start) -> daily OHLCV DataFrame. start=None means full history.
Downloader = Callable[[str, "pd.Timestamp | None"], pd.DataFrame]


def market_universe(watchlist: tuple[str, ...] | list[str] = ()) -> tuple[str, ...]:
    """Watchlist (incl. core tickers) + macro tickers + sector ETFs + USD/CAD, de-duplicated."""
    return merge_tickers(tuple(watchlist), [*MACRO_TICKERS, *SECTOR_ETFS, USD_CAD_SYMBOL])


# --------------------------------------------------------------------------- #
# Download + parquet cache
# --------------------------------------------------------------------------- #

def yf_downloader(ticker: str, start: pd.Timestamp | None) -> pd.DataFrame:
    import yfinance as yf

    t = yf.Ticker(ticker)
    if start is None:
        return t.history(period="max", interval="1d", auto_adjust=True)
    return t.history(start=start.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Keep OHLCV columns, tz-naive date index, sorted, de-duplicated."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="Date"))
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data missing columns: {missing}")
    out = df[OHLCV_COLUMNS].copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["Close"]).astype(float)


def last_completed_business_day(now: datetime) -> pd.Timestamp:
    """Most recent weekday strictly before `now`'s date (today's bar may be partial)."""
    return (pd.Timestamp(now).normalize() - pd.offsets.BDay(1)).normalize()


class ParquetStore:
    """Per-ticker parquet cache of daily OHLCV with stale-only refresh."""

    def __init__(
        self,
        directory: Path,
        downloader: Downloader = yf_downloader,
        now: Callable[[], datetime] = datetime.now,
        recheck_after: timedelta = timedelta(hours=12),
        overlap_days: int = 5,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader
        self.now = now
        self.recheck_after = recheck_after
        self.overlap_days = overlap_days
        self._meta_path = self.directory / "_meta.json"

    # -- paths / metadata --------------------------------------------------- #
    @staticmethod
    def _safe(ticker: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in ticker.upper())

    def path(self, ticker: str) -> Path:
        return self.directory / f"{self._safe(ticker)}.parquet"

    def _load_meta(self) -> dict[str, str]:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _mark_checked(self, ticker: str) -> None:
        meta = self._load_meta()
        meta[ticker.upper()] = pd.Timestamp(self.now()).isoformat()
        self._meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    # -- read / write ------------------------------------------------------- #
    def read(self, ticker: str) -> pd.DataFrame | None:
        p = self.path(ticker)
        return pd.read_parquet(p) if p.exists() else None

    def _write(self, ticker: str, df: pd.DataFrame) -> None:
        tmp = self.path(ticker).with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(self.path(ticker))

    def is_stale(self, ticker: str, cached: pd.DataFrame | None = None) -> bool:
        cached = self.read(ticker) if cached is None else cached
        if cached is None or cached.empty:
            return True
        if cached.index.max() >= last_completed_business_day(self.now()):
            return False
        checked = self._load_meta().get(ticker.upper())
        if checked and pd.Timestamp(self.now()) - pd.Timestamp(checked) < self.recheck_after:
            return False  # checked recently; probably a holiday / no new bar
        return True

    def get(self, ticker: str, force: bool = False) -> pd.DataFrame:
        """Daily OHLCV for `ticker`, refreshing from the source only if stale."""
        cached = self.read(ticker)
        if not force and not self.is_stale(ticker, cached):
            return cached

        start = None
        if cached is not None and not cached.empty and not force:
            start = cached.index.max() - pd.Timedelta(days=self.overlap_days)

        fresh = normalize_ohlcv(self.downloader(ticker, start))
        self._mark_checked(ticker)

        if fresh.empty:
            if cached is None or cached.empty:
                raise ValueError(f"No OHLCV data returned for {ticker!r}")
            return cached  # keep what we have; don't overwrite with nothing

        if start is not None and cached is not None:
            merged = pd.concat([cached[cached.index < fresh.index.min()], fresh])
        else:
            merged = fresh
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self._write(ticker, merged)
        return merged


def fetch_universe(
    tickers: tuple[str, ...] | list[str],
    store: ParquetStore,
    on_error: Callable[[str, Exception], None] | None = None,
    rates: pd.Series | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch all tickers; failures are reported via `on_error` and skipped.

    With `rates` (USD/CAD), USD-priced frames are converted to CAD (to_cad).
    """
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = store.get(t)
            out[t] = df if rates is None else to_cad(df, t, rates, overrides)
        except Exception as exc:
            if on_error is None:
                raise
            on_error(t, exc)
    return out


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #

_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate daily OHLCV to a coarser bar ('W-FRI', 'QE', ...)."""
    return df.resample(rule).agg(_AGG).dropna(subset=["Close"])


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlcv(df, "W-FRI")


def to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlcv(df, "QE")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def quarterly_returns(close: pd.Series) -> pd.Series:
    """Simple return from one quarter-end close to the next."""
    q = close.resample("QE").last().dropna()
    return q.pct_change().dropna().rename("return")


def realized_volatility(close: pd.Series, rule: str = "QE") -> pd.Series:
    """Annualized std-dev of daily log returns within each period."""
    r = log_returns(close)
    return (r.resample(rule).std(ddof=1) * np.sqrt(TRADING_DAYS)).dropna().rename("realized_vol")


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (0 if none)."""
    if close.empty:
        return float("nan")
    dd = close / close.cummax() - 1.0
    return float(dd.min())


def quarterly_max_drawdown(close: pd.Series) -> pd.Series:
    """Max drawdown within each calendar quarter."""
    return close.resample("QE").apply(max_drawdown).dropna().rename("max_drawdown")


def moving_average_trend(close: pd.Series, short: int = 50, long: int = 200) -> pd.DataFrame:
    """50/200-day moving averages with a trend label.

    up:    close > MA50 > MA200
    down:  close < MA50 < MA200
    mixed: anything else; NaN until both MAs exist.
    """
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long).mean()
    up = (close > ma_s) & (ma_s > ma_l)
    down = (close < ma_s) & (ma_s < ma_l)
    trend = pd.Series(np.where(up, "up", np.where(down, "down", "mixed")), index=close.index, dtype=object)
    trend[ma_l.isna()] = None
    return pd.DataFrame(
        {
            "close": close,
            f"ma{short}": ma_s,
            f"ma{long}": ma_l,
            "golden_cross": ma_s > ma_l,
            "trend": trend,
        }
    )


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    delta = close.diff().to_numpy()
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    n = len(close)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > period:
        # Seed with the simple mean of the first `period` changes, then
        # Wilder smoothing: avg_t = (avg_{t-1} * (period - 1) + x_t) / period.
        avg_gain[period] = gain[1 : period + 1].mean()
        avg_loss[period] = loss[1 : period + 1].mean()
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    avg_gain = pd.Series(avg_gain, index=close.index)
    avg_loss = pd.Series(avg_loss, index=close.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses -> 100; no gains -> 0; flat -> 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss != 0)), 0.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    out[avg_gain.isna()] = np.nan
    return out.rename("rsi")


def _aligned_returns(asset: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Align prices on common dates *first*, then take returns.

    Taking returns before aligning would pair a multi-day return (across a gap
    in one series) with a single-day return in the other, biasing beta.
    """
    prices = pd.concat({"asset": asset, "bench": benchmark}, axis=1, join="inner").dropna()
    return np.log(prices / prices.shift(1)).dropna()


def beta(asset: pd.Series, benchmark: pd.Series) -> float:
    """OLS beta of daily log returns vs benchmark over the overlapping period."""
    r = _aligned_returns(asset, benchmark)
    if len(r) < 2:
        return float("nan")
    var = r["bench"].var(ddof=1)
    return float(r["asset"].cov(r["bench"]) / var) if var > 0 else float("nan")


def quarterly_beta(asset: pd.Series, benchmark: pd.Series, min_obs: int = 20) -> pd.Series:
    """Beta computed from daily returns within each quarter."""
    r = _aligned_returns(asset, benchmark)

    def _b(g: pd.DataFrame) -> float:
        if len(g) < min_obs:
            return np.nan
        v = g["bench"].var(ddof=1)
        return g["asset"].cov(g["bench"]) / v if v > 0 else np.nan

    return r.groupby(pd.Grouper(freq="QE")).apply(_b).dropna().rename("beta")


def correlation_matrix(
    closes: dict[str, pd.Series] | pd.DataFrame,
    rule: str | None = None,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Pearson correlation of log returns.

    rule: resample closes first ('W-FRI', 'QE'); None = daily.
    lookback: keep only the last N return observations.
    Uses pairwise-complete observations, so mismatched calendars (e.g. futures
    vs equities) don't drop whole rows.
    """
    df = pd.DataFrame(closes) if isinstance(closes, dict) else closes.copy()
    if rule:
        df = df.resample(rule).last()
    rets = np.log(df / df.shift(1))
    rets = rets.dropna(how="all")
    if lookback:
        rets = rets.tail(lookback)
    return rets.corr()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MarketMetrics:
    quarterly: pd.DataFrame      # index (ticker, quarter_end): return, realized_vol, max_drawdown, beta
    snapshot: pd.DataFrame       # index ticker: latest close, ma50, ma200, trend, rsi, beta, max_drawdown
    correlation: pd.DataFrame    # ticker x ticker, weekly log returns


def compute_metrics(
    frames: dict[str, pd.DataFrame],
    benchmark: str = BENCHMARK,
    corr_rule: str = "W-FRI",
    corr_lookback: int | None = 156,  # ~3 years of weeks
) -> MarketMetrics:
    closes = {t: df["Close"] for t, df in frames.items() if not df.empty}
    bench = closes.get(benchmark)

    q_rows, snap_rows = [], []
    for t, c in closes.items():
        q = pd.concat(
            [quarterly_returns(c), realized_volatility(c), quarterly_max_drawdown(c)],
            axis=1,
            sort=True,
        )
        q["beta"] = quarterly_beta(c, bench) if bench is not None else np.nan
        q.index.name = "quarter_end"
        q_rows.append(q.assign(ticker=t).set_index("ticker", append=True).swaplevel())

        trend = moving_average_trend(c).iloc[-1]
        snap_rows.append(
            {
                "ticker": t,
                "currency": frames[t].attrs.get("currency", ticker_currency(t)) or "index/yield",
                "as_of": c.index[-1],
                "close": c.iloc[-1],
                "ma50": trend["ma50"],
                "ma200": trend["ma200"],
                "trend": trend["trend"],
                "rsi14": rsi(c).iloc[-1],
                "beta_vs_" + benchmark: beta(c, bench) if bench is not None else np.nan,
                "max_drawdown": max_drawdown(c),
            }
        )

    quarterly = pd.concat(q_rows) if q_rows else pd.DataFrame()
    snapshot = pd.DataFrame(snap_rows).set_index("ticker") if snap_rows else pd.DataFrame()
    corr = correlation_matrix(closes, rule=corr_rule, lookback=corr_lookback) if closes else pd.DataFrame()
    return MarketMetrics(quarterly=quarterly, snapshot=snapshot, correlation=corr)


def main() -> int:  # pragma: no cover - thin wrapper around network I/O
    import sys

    from forecaster.config import load_settings

    from forecaster.config import load_currency_overrides

    settings = load_settings()
    store = ParquetStore(settings.cache_dir / "ohlcv")
    warn = lambda m: print(f"[warn] {m}", file=sys.stderr)  # noqa: E731
    fred = None
    if settings.fred_api_key:
        from forecaster.cache import Cache
        from forecaster.sources.fred import FredClient

        fred = FredClient(settings.fred_api_key, Cache(settings.cache_dir, settings.cache_ttl_hours * 3600))
    rates = usd_cad_rates(store, fred, on_warn=warn)
    frames = fetch_universe(
        market_universe(settings.tickers),
        store,
        on_error=lambda t, e: warn(f"{t}: {e}"),
        rates=rates,
        overrides=load_currency_overrides(),
    )
    if not frames:
        return 1
    m = compute_metrics(frames)
    print(f"Prices in {BASE_CURRENCY}; USD converted at daily USD/CAD "
          f"(last {rates.iloc[-1]:.4f} on {rates.index[-1]:%Y-%m-%d}). Returns, vol, beta and "
          f"correlations include the currency effect. Index/yield series are not converted.")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(m.snapshot.round(3))
        print()
        print(m.correlation.round(2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
