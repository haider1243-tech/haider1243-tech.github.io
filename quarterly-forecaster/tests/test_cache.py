from forecaster.cache import Cache


def test_miss_then_hit(cache):
    calls = []
    fetch = lambda: calls.append(1) or {"x": 1}
    key = Cache.make_key("t", a=1)
    assert cache.get_or_fetch(key, fetch) == {"x": 1}
    assert cache.get_or_fetch(key, fetch) == {"x": 1}
    assert len(calls) == 1


def test_ttl_expiry(tmp_path):
    now = [1000.0]
    c = Cache(tmp_path, ttl_seconds=10, clock=lambda: now[0])
    c.set("k", 5)
    assert c.get("k") == 5
    now[0] += 11
    assert c.get("k") is None


def test_key_is_stable_and_param_order_independent():
    assert Cache.make_key("ns", a=1, b=2) == Cache.make_key("ns", b=2, a=1)
    assert Cache.make_key("ns", a=1) != Cache.make_key("ns", a=2)


def test_key_sanitizes_namespace():
    assert "/" not in Cache.make_key("a/b^c", x=1)


def test_clear(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.clear() == 2
    assert cache.get("a") is None


def test_empty_results_not_cached(cache):
    import pandas as pd

    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame()

    cache.get_or_fetch("k", fetch)
    cache.get_or_fetch("k", fetch)
    assert len(calls) == 2
