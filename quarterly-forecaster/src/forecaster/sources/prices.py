"""Price history from yfinance (no API key needed)."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from forecaster.cache import Cache

Downloader = Callable[[str, str], pd.DataFrame]


def _yf_download(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)


def get_daily_closes(
    ticker: str,
    cache: Cache,
    period: str = "max",
    downloader: Downloader = _yf_download,
) -> pd.Series:
    """Daily adjusted close prices for `ticker`, cached."""
    key = Cache.make_key("yf_daily", ticker=ticker, period=period)
    df = cache.get_or_fetch(key, lambda: downloader(ticker, period))
    if df is None or df.empty or "Close" not in df:
        raise ValueError(f"No price data returned for {ticker!r}")
    closes = df["Close"].dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes.name = ticker
    return closes


def to_quarterly(closes: pd.Series) -> pd.Series:
    """Last close of each calendar quarter."""
    return closes.resample("QE").last().dropna()
