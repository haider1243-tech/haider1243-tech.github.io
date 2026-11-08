import pandas as pd
import pytest

from conftest import FakeSession
from forecaster.sources.finnhub import FinnhubClient
from forecaster.sources.fred import FredClient
from forecaster.sources.prices import get_daily_closes, to_quarterly


def test_prices_cached(cache, daily_prices):
    calls = []

    def downloader(ticker, period):
        calls.append(ticker)
        return daily_prices

    s1 = get_daily_closes("SPY", cache, downloader=downloader)
    s2 = get_daily_closes("SPY", cache, downloader=downloader)
    assert calls == ["SPY"]
    pd.testing.assert_series_equal(s1, s2)


def test_prices_empty_raises(cache):
    with pytest.raises(ValueError):
        get_daily_closes("BAD", cache, downloader=lambda t, p: pd.DataFrame())


def test_to_quarterly(daily_prices):
    q = to_quarterly(daily_prices["Close"])
    assert q.index[-1] == pd.Timestamp("2025-09-30")
    assert all(d.is_quarter_end for d in q.index)


def test_fred_parses_and_caches(cache):
    session = FakeSession(
        {"observations": [{"date": "2025-01-01", "value": "4.5"}, {"date": "2025-02-01", "value": "."}]}
    )
    fred = FredClient("key", cache, session=session)
    s = fred.get_series("DGS10")
    fred.get_series("DGS10")
    assert len(session.calls) == 1
    assert s.tolist() == [4.5]  # "." treated as missing


def test_fred_requires_key(cache):
    with pytest.raises(RuntimeError):
        FredClient(None, cache).get_series("DGS10")


def test_finnhub_quote_cached_and_key_in_header(cache):
    session = FakeSession({"c": 500.0})
    fh = FinnhubClient("secret", cache, session=session)
    assert fh.quote("SPY") == {"c": 500.0}
    fh.quote("SPY")
    assert len(session.calls) == 1
    _, kwargs = session.calls[0]
    assert kwargs["headers"]["X-Finnhub-Token"] == "secret"
    assert "token" not in kwargs["params"]


def test_finnhub_requires_key(cache):
    with pytest.raises(RuntimeError):
        FinnhubClient(None, cache).quote("SPY")
