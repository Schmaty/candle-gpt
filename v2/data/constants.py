"""Canonical constants for v2 data pipeline (v2.0.0).

One source of truth for asset universe, timeframes, kline + funding + liquidation
schemas, and history defaults. Every parquet write embeds DATA_VERSION + the
matching SCHEMA_HASH in pyarrow file metadata; mismatches surface via
v2.data.store.assert_schema_compatible.
"""
from __future__ import annotations
import hashlib
from enum import Enum


DATA_VERSION = "2.0.0"


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

# --- Kline schema (on-disk, post-v2.0.0) ---------------------------------
# Order matters: open_time is sort+join key; regime is appended in v2.0.0.
# regime sentinel: -1 = untagged (post-fetch, pre-tag_regimes); 0/1/2 are the
# three documented buckets (see v2.data.regime).
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open", "high", "low", "close", "volume",
    "close_time",
    "regime",
)

KLINE_DTYPES: dict[str, str] = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "regime": "int8",
}

DEFAULT_HISTORY_DAYS: int = 4 * 365  # 4 years; spec-locked

# --- Funding rate schema -------------------------------------------------
FUNDING_COLUMNS: tuple[str, ...] = (
    "funding_time",
    "funding_rate",
    "mark_price",
)

FUNDING_DTYPES: dict[str, str] = {
    "funding_time": "int64",
    "funding_rate": "float64",
    "mark_price": "float64",
}

# --- Liquidations: rolled-up per-minute schema --------------------------
# This is the schema the dataloader joins on. The live collector and the
# Tardis backfill both flow through rollup.py to produce this exact shape.
LIQ_BUCKETED_COLUMNS: tuple[str, ...] = (
    "bucket_time",
    "count",
    "sum_notional",
    "max_single",
    "long_liq_count",
    "long_liq_notional",
    "short_liq_count",
    "short_liq_notional",
)

LIQ_BUCKETED_DTYPES: dict[str, str] = {
    "bucket_time": "int64",
    "count": "int64",
    "sum_notional": "float64",
    "max_single": "float64",
    "long_liq_count": "int64",
    "long_liq_notional": "float64",
    "short_liq_count": "int64",
    "short_liq_notional": "float64",
}


def _hash_schema(columns: tuple[str, ...], dtypes: dict[str, str]) -> str:
    """Deterministic sha256 over the (column, dtype) tuple list in order."""
    h = hashlib.sha256()
    for col in columns:
        h.update(col.encode("utf-8"))
        h.update(b"\x00")
        h.update(dtypes[col].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


KLINE_SCHEMA_HASH: str = _hash_schema(KLINE_COLUMNS, KLINE_DTYPES)
FUNDING_SCHEMA_HASH: str = _hash_schema(FUNDING_COLUMNS, FUNDING_DTYPES)
LIQ_BUCKETED_SCHEMA_HASH: str = _hash_schema(LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES)
