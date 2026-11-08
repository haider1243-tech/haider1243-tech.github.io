"""Configuration: API keys from .env, tickers from watchlist.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORE_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "^GSPC", "^IXIC", "^VIX")
HORIZON_QUARTERS = 4
PERCENTILES: tuple[int, ...] = (10, 50, 90)


@dataclass(frozen=True)
class Settings:
    fred_api_key: str | None
    finnhub_api_key: str | None
    cache_dir: Path
    cache_ttl_hours: float
    tickers: tuple[str, ...] = field(default=CORE_TICKERS)


def is_equity_symbol(ticker: str) -> bool:
    """False for indices (^GSPC), futures (GC=F) and ICE index symbols (DX-Y.NYB),
    which have no earnings or company news."""
    return not (ticker.startswith("^") or "=" in ticker or ticker.endswith(".NYB"))


def load_watchlist(path: Path) -> list[str]:
    """Read extra tickers from a watchlist YAML file. Missing file -> []."""
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    raw = data.get("tickers") or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'tickers' must be a list")
    return [str(t).strip().upper() for t in raw if t and str(t).strip()]


def merge_tickers(core: tuple[str, ...], extra: list[str]) -> tuple[str, ...]:
    """Core tickers first, then watchlist additions, de-duplicated in order."""
    seen: dict[str, None] = {}
    for t in (*core, *extra):
        seen.setdefault(t.upper(), None)
    return tuple(seen)


def load_settings(
    root: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> Settings:
    """Build Settings. Pass `env` in tests to avoid touching os.environ/.env."""
    if env is None:
        load_dotenv(root / ".env")
        env = dict(os.environ)

    cache_dir = Path(env.get("FORECASTER_CACHE_DIR") or root / "data")
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir

    return Settings(
        fred_api_key=env.get("FRED_API_KEY") or None,
        finnhub_api_key=env.get("FINNHUB_API_KEY") or None,
        cache_dir=cache_dir,
        cache_ttl_hours=float(env.get("FORECASTER_CACHE_TTL_HOURS") or 24),
        tickers=merge_tickers(CORE_TICKERS, load_watchlist(root / "watchlist.yaml")),
    )
