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


class RoutedResponse:
    def __init__(self, payload, status=200, url=""):
        self._payload = payload
        self.status_code = status
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Error for url: {self.url}")

    def json(self):
        return self._payload


class RoutingSession:
    """Fake requests.Session: handler(url, params) -> payload or (status, payload)."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, **kwargs):
        params = params or {}
        self.calls.append((url, params, kwargs))
        out = self.handler(url, params)
        status, payload = out if isinstance(out, tuple) else (200, out)
        full = url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        return RoutedResponse(payload, status, full)


# --------------------------------------------------------------------------- #
# Dashboard login: every test gets a known password hash; `authed_app` starts
# an AppTest already logged in (the login page itself is tested in test_auth.py).
# --------------------------------------------------------------------------- #
TEST_PASSWORD = "correct horse battery staple"
_TEST_HASH = None


def _test_hash():
    global _TEST_HASH
    if _TEST_HASH is None:
        from forecaster.auth import hash_password
        _TEST_HASH = hash_password(TEST_PASSWORD, iterations=1000)
    return _TEST_HASH


@pytest.fixture(autouse=True)
def _dashboard_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD_HASH", _test_hash())


def authed_app(path, default_timeout=60):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(path, default_timeout=default_timeout)
    at.session_state["auth_ok"] = True
    return at
