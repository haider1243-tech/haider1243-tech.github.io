"""One clock for the app. Normally real time; in demo mode (`start.bat demo`) it runs shifted so
the simulated market is open whenever the demo is started.

FORECASTER_CLOCK_OFFSET_S (seconds) is set only by the launcher's demo mode. Every consumer of
"now" that cares about market sessions (live feed, shadow trading, the app, the webhook's live
check) uses this module, so they all agree on what time it is in the market.
"""

from __future__ import annotations

import os
import time

import pandas as pd

OFFSET_ENV = "FORECASTER_CLOCK_OFFSET_S"


def offset_s() -> float:
    try:
        return float(os.environ.get(OFFSET_ENV) or 0.0)
    except ValueError:
        return 0.0


def now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=offset_s())


def now_ms() -> int:
    return time.time_ns() // 1_000_000 + int(offset_s() * 1000)


def is_demo() -> bool:
    return offset_s() != 0.0 or os.environ.get("FORECASTER_DEMO") == "1"
