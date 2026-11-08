"""File-based cache for API responses, so repeated runs don't hit rate limits.

Every data source takes a `Cache` instance; tests pass one pointed at a tmp dir.
Values are pickled (pandas DataFrames, dicts, lists). The cache is local-only
and lives in /data, which is gitignored — never load cache files from elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    empty = getattr(value, "empty", None)  # pandas objects
    if isinstance(empty, bool):
        return not empty
    try:
        return len(value) > 0
    except TypeError:
        return True


class Cache:
    def __init__(
        self,
        directory: Path,
        ttl_seconds: float = 24 * 3600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    @staticmethod
    def make_key(namespace: str, **params: Any) -> str:
        blob = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        safe_ns = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
        return f"{safe_ns}-{digest}"

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("rb") as f:
            stored_at, value = pickle.load(f)
        if self._clock() - stored_at > self.ttl_seconds:
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump((self._clock(), value), f)
        tmp.replace(self._path(key))

    def get_or_fetch(
        self,
        key: str,
        fetch: Callable[[], T],
        should_cache: Callable[[T], bool] | None = None,
    ) -> T:
        """Return cached value, or fetch and store it.

        Empty results (None, empty DataFrame/list/dict) are not cached by
        default, so a transient API failure doesn't poison the cache.
        """
        value = self.get(key)
        if value is None:
            value = fetch()
            if (should_cache or _is_non_empty)(value):
                self.set(key, value)
        return value

    def clear(self) -> int:
        n = 0
        for p in self.directory.glob("*.pkl"):
            p.unlink()
            n += 1
        return n
