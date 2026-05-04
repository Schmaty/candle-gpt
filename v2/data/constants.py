"""Canonical constants for v2 data pipeline.

One source of truth for asset universe, timeframes, kline schema, and history defaults.
Importing modules MUST NOT redefine these.
"""
from __future__ import annotations
from enum import Enum


class Asset(str, Enum):
    BTC = "BTCUSDT"
    ETH = "ETHUSDT"
    SOL = "SOLUSDT"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"


INTERVAL_MS: dict[Timeframe, int] = {
    Timeframe.M1: 60_000,
    Timeframe.M5: 300_000,
}

# Order matters: open_time is the sort + join key. We drop the trailing Binance
# fields (quote_vol, n_trades, taker_buy_base, taker_buy_quote, ignore) — none
# are used by the v2 model and they bloat parquet by ~40%.
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
)

KLINE_DTYPES: dict[str, str] = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
}

DEFAULT_HISTORY_DAYS: int = 4 * 365  # 4 years; spec-locked
