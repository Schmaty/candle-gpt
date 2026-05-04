"""Parquet round-trip with schema enforcement and standard path resolution."""
from pathlib import Path

import pandas as pd
import pytest

from v2.data.constants import Asset, Timeframe, KLINE_COLUMNS
from v2.data.store import parquet_path, write_klines, read_klines
from v2.data.validate import SchemaViolation


def _good_df() -> pd.DataFrame:
    return pd.DataFrame({
        "open_time":   pd.array([0, 60_000], dtype="int64"),
        "open":        [1.0, 1.1],
        "high":        [1.2, 1.3],
        "low":         [0.9, 1.0],
        "close":       [1.1, 1.2],
        "volume":      [10.0, 11.0],
        "close_time":  pd.array([59_999, 119_999], dtype="int64"),
        "regime":      pd.array([-1, -1], dtype="int8"),
    })


def test_parquet_path_uses_lowercase_symbol_and_interval(tmp_path: Path):
    p = parquet_path(tmp_path, Asset.BTC, Timeframe.M1)
    assert p == tmp_path / "btcusdt_1m.parquet"


def test_parquet_path_eth_5m(tmp_path: Path):
    p = parquet_path(tmp_path, Asset.ETH, Timeframe.M5)
    assert p == tmp_path / "ethusdt_5m.parquet"


def test_write_then_read_roundtrips(tmp_path: Path):
    df = _good_df()
    p = parquet_path(tmp_path, Asset.BTC, Timeframe.M1)
    write_klines(df, p)
    out = read_klines(p)
    pd.testing.assert_frame_equal(out, df)


def test_write_creates_parent_dir(tmp_path: Path):
    df = _good_df()
    p = tmp_path / "deep" / "nested" / "btc_1m.parquet"
    write_klines(df, p)
    assert p.exists()


def test_write_rejects_bad_schema(tmp_path: Path):
    df = _good_df().drop(columns=["volume"])
    p = parquet_path(tmp_path, Asset.BTC, Timeframe.M1)
    with pytest.raises(SchemaViolation):
        write_klines(df, p)


def test_read_rejects_bad_schema(tmp_path: Path):
    # Sneak in a parquet file that doesn't match — read should reject.
    bad = pd.DataFrame({"foo": [1, 2]})
    p = tmp_path / "bad.parquet"
    bad.to_parquet(p, index=False)
    with pytest.raises(SchemaViolation):
        read_klines(p)
