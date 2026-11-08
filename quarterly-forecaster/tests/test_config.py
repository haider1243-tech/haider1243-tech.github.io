import pytest

from forecaster.config import CORE_TICKERS, load_settings, load_watchlist, merge_tickers


def test_core_tickers():
    assert CORE_TICKERS == ("SPY", "QQQ", "^GSPC", "^IXIC", "^VIX")


def test_merge_dedupes_and_keeps_core_first():
    assert merge_tickers(("SPY", "QQQ"), ["aapl", "SPY"]) == ("SPY", "QQQ", "AAPL")


def test_watchlist_missing_file(tmp_path):
    assert load_watchlist(tmp_path / "nope.yaml") == []


def test_watchlist_empty_list(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text("tickers:\n")
    assert load_watchlist(p) == []


def test_watchlist_rejects_non_list(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text("tickers: AAPL\n")
    with pytest.raises(ValueError):
        load_watchlist(p)


def test_load_settings_with_injected_env(tmp_path):
    (tmp_path / "watchlist.yaml").write_text("tickers: [msft]\n")
    s = load_settings(root=tmp_path, env={"FRED_API_KEY": "abc", "FORECASTER_CACHE_TTL_HOURS": "6"})
    assert s.fred_api_key == "abc"
    assert s.finnhub_api_key is None
    assert s.cache_ttl_hours == 6
    assert s.cache_dir == tmp_path / "data"
    assert s.tickers[-1] == "MSFT"
