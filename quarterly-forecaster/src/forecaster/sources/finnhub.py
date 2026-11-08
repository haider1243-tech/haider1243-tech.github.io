"""Finnhub free tier: quotes, company metrics, news sentiment context.

Free tier is ~60 calls/minute, so everything goes through the cache.
"""

from __future__ import annotations

from typing import Any

import requests

from forecaster.cache import Cache

FINNHUB_BASE = "https://finnhub.io/api/v1"


class FinnhubClient:
    def __init__(
        self,
        api_key: str | None,
        cache: Cache,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.session = session or requests.Session()

    def _get(self, endpoint: str, **params: Any) -> Any:
        if not self.api_key:
            raise RuntimeError("FINNHUB_API_KEY is not set (see .env.example)")
        key = Cache.make_key(f"finnhub_{endpoint}", **params)

        def fetch() -> Any:
            r = self.session.get(
                f"{FINNHUB_BASE}/{endpoint}",
                params=params,
                headers={"X-Finnhub-Token": self.api_key},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()

        return self.cache.get_or_fetch(key, fetch)

    def quote(self, symbol: str) -> dict:
        return self._get("quote", symbol=symbol)

    def basic_financials(self, symbol: str) -> dict:
        return self._get("stock/metric", symbol=symbol, metric="all")
