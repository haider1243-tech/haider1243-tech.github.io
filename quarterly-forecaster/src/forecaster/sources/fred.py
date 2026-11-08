"""FRED API client (free API key required). Every request is cached.

FRED only accepts the key as a query parameter, so it is never included in
cache keys, logs or error messages produced here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from forecaster.cache import Cache

FRED_BASE = "https://api.stlouisfed.org/fred"


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

    def _get(self, endpoint: str, **params: Any) -> dict:
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is not set (see .env.example)")
        key = Cache.make_key("fred_" + endpoint.replace("/", "_"), **params)

        def fetch() -> dict:
            r = self.session.get(
                f"{FRED_BASE}/{endpoint}",
                params={**params, "api_key": self.api_key, "file_type": "json"},
                timeout=30,
            )
            try:
                r.raise_for_status()
            except requests.HTTPError as exc:
                # Re-raise without the URL, which contains the API key.
                raise RuntimeError(
                    f"FRED {endpoint} failed: HTTP {r.status_code} for params {params}"
                ) from None
            return r.json()

        return self.cache.get_or_fetch(key, fetch)

    # -- observations ------------------------------------------------------ #
    def get_series(self, series_id: str) -> pd.Series:
        """Full observation history for a series ("." values dropped)."""
        obs = self._get("series/observations", series_id=series_id)["observations"]
        s = pd.Series(
            pd.to_numeric([o["value"] for o in obs], errors="coerce"),
            index=pd.to_datetime([o["date"] for o in obs]),
            name=series_id,
        )
        return s.dropna()

    def get_first_release(self, series_id: str) -> pd.DataFrame:
        """Point-in-time history: each observation's *initial* published value
        and the date it was first published (ALFRED `output_type=4`).

        Columns: date (observation period), value, available (first release
        date). Use this, not get_series, for backtests and model training:
        get_series returns today's revised values, which weren't known then.
        """
        obs = self._get(
            "series/observations",
            series_id=series_id,
            realtime_start="1776-07-04",
            realtime_end="9999-12-31",
            output_type=4,
        )["observations"]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime([o["date"] for o in obs]),
                "value": pd.to_numeric([o["value"] for o in obs], errors="coerce"),
                "available": pd.to_datetime([o["realtime_start"] for o in obs]),
            }
        )
        return df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)

    # -- releases ---------------------------------------------------------- #
    def release_for_series(self, series_id: str) -> dict:
        """The release (id, name) a series is published in."""
        releases = self._get("series/release", series_id=series_id)["releases"]
        if not releases:
            raise ValueError(f"No FRED release found for series {series_id}")
        return releases[0]

    def release_dates(self, release_id: int, limit: int = 100) -> list[pd.Timestamp]:
        """Release dates, including scheduled future dates, newest first."""
        data = self._get(
            "release/dates",
            release_id=release_id,
            include_release_dates_with_no_data="true",
            sort_order="desc",
            limit=limit,
        )
        return [pd.Timestamp(d["date"]) for d in data["release_dates"]]
