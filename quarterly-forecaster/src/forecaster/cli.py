"""Command line entry point: `python -m forecaster`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from forecaster.cache import Cache
from forecaster.config import load_currency_overrides, load_settings
from forecaster.market_data import BASE_CURRENCY, USD_CAD_SYMBOL, ticker_currency, to_cad
from forecaster.model import block_bootstrap_forecast
from forecaster.sources.prices import get_daily_closes, to_quarterly


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="4-quarter probabilistic forecasts")
    parser.add_argument("tickers", nargs="*", help="Override tickers (default: core + watchlist)")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    cache = Cache(settings.cache_dir, ttl_seconds=settings.cache_ttl_hours * 3600)
    if args.clear_cache:
        print(f"Cleared {cache.clear()} cached responses")

    overrides = load_currency_overrides()
    rates: list = []  # lazily fetched USD/CAD

    def in_cad(ticker: str, closes: pd.Series) -> tuple[pd.Series, str]:
        cur = ticker_currency(ticker, overrides)
        if cur != "USD":
            return closes, cur or "points"
        if not rates:
            rates.append(get_daily_closes(USD_CAD_SYMBOL, cache))
        return to_cad(closes.to_frame("Close"), ticker, rates[0])["Close"], BASE_CURRENCY

    frames, failed = [], []
    for ticker in args.tickers or settings.tickers:
        try:
            closes, unit = in_cad(ticker, get_daily_closes(ticker, cache))
            quarterly = to_quarterly(closes)
            frames.append(block_bootstrap_forecast(ticker, quarterly).to_frame().assign(unit=unit))
        except Exception as exc:  # keep going for other tickers
            failed.append(ticker)
            print(f"[warn] {ticker}: {exc}", file=sys.stderr)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        print(f"Prices in {BASE_CURRENCY} (USD tickers converted at daily USD/CAD, {USD_CAD_SYMBOL}); "
              "index/yield series in points.")
        print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    return 1 if failed and not frames else 0


if __name__ == "__main__":
    raise SystemExit(main())
