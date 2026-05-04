"""Schema version + hash + metadata round-trip + guardrail behavior."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from v2.data.constants import (
    DATA_VERSION,
    KLINE_COLUMNS, KLINE_DTYPES, KLINE_SCHEMA_HASH,
    FUNDING_COLUMNS, FUNDING_DTYPES, FUNDING_SCHEMA_HASH,
    LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES, LIQ_BUCKETED_SCHEMA_HASH,
)
from v2.data.store import (
    write_klines, read_klines,
    write_funding, read_funding,
    write_liq_bucketed, read_liq_bucketed,
    assert_schema_compatible,
)
from v2.data.validate import SchemaViolation


def _kline_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "open_time":   pd.array([i * 60_000 for i in range(n)], dtype="int64"),
        "open":        [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
        "close":       [1.0] * n, "volume": [1.0] * n,
        "close_time":  pd.array([i * 60_000 + 59_999 for i in range(n)], dtype="int64"),
        "regime":      pd.array([-1] * n, dtype="int8"),
    })


def _funding_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "funding_time": pd.array([i * 8 * 3_600_000 for i in range(n)], dtype="int64"),
        "funding_rate": [0.0001] * n,
        "mark_price":   [30000.0] * n,
    })


def _liq_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "bucket_time":         pd.array([i * 60_000 for i in range(n)], dtype="int64"),
        "count":               pd.array([1] * n, dtype="int64"),
        "sum_notional":        [100.0] * n,
        "max_single":          [50.0] * n,
        "long_liq_count":      pd.array([1] * n, dtype="int64"),
        "long_liq_notional":   [60.0] * n,
        "short_liq_count":     pd.array([0] * n, dtype="int64"),
        "short_liq_notional":  [0.0] * n,
    })


def test_data_version_is_2_0_0():
    assert DATA_VERSION == "2.0.0"


def test_schema_hashes_are_distinct():
    hashes = {KLINE_SCHEMA_HASH, FUNDING_SCHEMA_HASH, LIQ_BUCKETED_SCHEMA_HASH}
    assert len(hashes) == 3


def test_kline_round_trip_embeds_metadata(tmp_path: Path):
    p = tmp_path / "k.parquet"
    write_klines(_kline_df(), p)
    md = pq.read_schema(p).metadata or {}
    assert md[b"data_version"] == DATA_VERSION.encode()
    assert md[b"schema_hash"] == KLINE_SCHEMA_HASH.encode()
    out = read_klines(p)
    assert list(out.columns) == list(KLINE_COLUMNS)


def test_funding_round_trip_embeds_metadata(tmp_path: Path):
    p = tmp_path / "f.parquet"
    write_funding(_funding_df(), p)
    md = pq.read_schema(p).metadata or {}
    assert md[b"schema_hash"] == FUNDING_SCHEMA_HASH.encode()
    out = read_funding(p)
    assert list(out.columns) == list(FUNDING_COLUMNS)


def test_liq_round_trip_embeds_metadata(tmp_path: Path):
    p = tmp_path / "l.parquet"
    write_liq_bucketed(_liq_df(), p)
    md = pq.read_schema(p).metadata or {}
    assert md[b"schema_hash"] == LIQ_BUCKETED_SCHEMA_HASH.encode()
    out = read_liq_bucketed(p)
    assert list(out.columns) == list(LIQ_BUCKETED_COLUMNS)


def test_assert_schema_compatible_passes_on_match(tmp_path: Path):
    p = tmp_path / "k.parquet"
    write_klines(_kline_df(), p)
    assert_schema_compatible(p, KLINE_SCHEMA_HASH)  # must not raise


def test_assert_schema_compatible_fails_on_mismatch(tmp_path: Path):
    p = tmp_path / "k.parquet"
    write_klines(_kline_df(), p)
    with pytest.raises(SchemaViolation, match="schema_hash mismatch"):
        assert_schema_compatible(p, "not-the-real-hash")


def test_assert_schema_compatible_fails_on_legacy_file(tmp_path: Path):
    """Plan-1 parquets had no metadata; must surface clearly."""
    p = tmp_path / "legacy.parquet"
    df = pd.DataFrame({"x": [1]})
    df.to_parquet(p, index=False)  # plain pandas, no metadata
    with pytest.raises(SchemaViolation, match="no schema_hash"):
        assert_schema_compatible(p, KLINE_SCHEMA_HASH)


def test_kline_schema_hash_changes_if_columns_change(tmp_path: Path):
    """Hash is deterministic over (column, dtype) ordered list — sanity check."""
    from v2.data.constants import _hash_schema, KLINE_DTYPES
    altered_cols = KLINE_COLUMNS + ("extra",)
    altered_dtypes = {**KLINE_DTYPES, "extra": "float64"}
    other = _hash_schema(altered_cols, altered_dtypes)
    assert other != KLINE_SCHEMA_HASH
