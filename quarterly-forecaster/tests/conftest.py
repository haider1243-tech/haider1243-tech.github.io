import numpy as np
import pandas as pd
import pytest

from forecaster.cache import Cache


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "cache", ttl_seconds=3600)


@pytest.fixture
def daily_prices():
    """~15 years of synthetic daily closes with upward drift."""
    idx = pd.bdate_range("2010-01-01", "2025-09-30")
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0003, 0.01, len(idx))
    return pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rets))}, index=idx)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """Records calls; returns a canned payload. No network."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)
