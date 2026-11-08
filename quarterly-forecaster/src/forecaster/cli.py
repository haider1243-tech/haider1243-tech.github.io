"""Command line entry point: `python -m forecaster`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from forecaster.cache import Cache
from forecaster.config import load_settings
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

    frames, failed = [], []
    for ticker in args.tickers or settings.tickers:
        try:
            quarterly = to_quarterly(get_daily_closes(ticker, cache))
            frames.append(block_bootstrap_forecast(ticker, quarterly).to_frame())
        except Exception as exc:  # keep going for other tickers
            failed.append(ticker)
            print(f"[warn] {ticker}: {exc}", file=sys.stderr)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    return 1 if failed and not frames else 0


if __name__ == "__main__":
    raise SystemExit(main())
