"""FRED macro series (free API key required)."""

from __future__ import annotations

import pandas as pd
import requests

from forecaster.cache import Cache

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# Useful defaults for equity-regime context.
DEFAULT_SERIES = {
    "DGS10": "10Y Treasury yield",
    "DGS2": "2Y Treasury yield",
    "T10Y2Y": "10Y-2Y spread",
    "FEDFUNDS": "Fed funds rate",
    "CPIAUCSL": "CPI",
    "UNRATE": "Unemployment rate",
}


class FredClient:
    def __init__(
        self,
        api_key: str | None,
        cache: Cache,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.session = session or requests.Session()

    def get_series(self, series_id: str) -> pd.Series:
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is not set (see .env.example)")
        key = Cache.make_key("fred", series_id=series_id)

        def fetch() -> list[dict]:
            r = self.session.get(
                FRED_URL,
                params={"series_id": series_id, "api_key": self.api_key, "file_type": "json"},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["observations"]

        obs = self.cache.get_or_fetch(key, fetch)
        s = pd.Series(
            pd.to_numeric([o["value"] for o in obs], errors="coerce"),  # "." = missing
            index=pd.to_datetime([o["date"] for o in obs]),
            name=series_id,
        )
        return s.dropna()
