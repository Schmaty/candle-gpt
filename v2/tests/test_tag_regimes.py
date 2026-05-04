"""tag_regimes CLI: round-trip in-place mutation, atomicity, schema metadata preserved."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from v2.data.constants import (
    Asset, Timeframe, KLINE_SCHEMA_HASH, FUNDING_COLUMNS, LIQ_BUCKETED_COLUMNS,
    LIQ_BUCKETED_DTYPES,
)
from v2.data.regime import PERCENTILE_WINDOW_BARS, REGIME_UNTAGGED
from v2.data.store import (
    parquet_path, funding_parquet_path, liq_bucketed_parquet_path,
    write_klines, read_klines, write_funding, write_liq_bucketed,
)
from v2.data.tag_regimes import tag_kline_parquet


def _setup_inputs(root: Path) -> tuple[Path, Path, Path]:
    n = PERCENTILE_WINDOW_BARS + 200
    closes = 100.0 + 0.05 * np.arange(n)
    klines = pd.DataFrame({
        "open_time":  pd.array([i * 60_000 for i in range(n)], dtype="int64"),
        "open":       closes - 0.1,
        "high":       closes + 0.5,
        "low":        closes - 0.5,
        "close":      closes,
        "volume":     np.full(n, 10.0),
        "close_time": pd.array([i * 60_000 + 59_999 for i in range(n)], dtype="int64"),
        "regime":     pd.array([-1] * n, dtype="int8"),
    })
    kp = parquet_path(root, Asset.BTC, Timeframe.M1)
    write_klines(klines, kp)

    fdf = pd.DataFrame({
        "funding_time": pd.array([i * 8 * 3_600_000 for i in range(50)], dtype="int64"),
        "funding_rate": [0.0001] * 50,
        "mark_price":   [100.0] * 50,
    })
    fp = funding_parquet_path(root, Asset.BTC)
    write_funding(fdf, fp)

    liq = pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })
    lp = liq_bucketed_parquet_path(root, Asset.BTC)
    write_liq_bucketed(liq, lp)

    return kp, fp, lp


def test_tag_overwrites_regime_column_in_place(tmp_path: Path):
    kp, _, _ = _setup_inputs(tmp_path)
    tag_kline_parquet(asset=Asset.BTC, timeframe=Timeframe.M1, root=tmp_path)
    out = read_klines(kp)
    # Pre-warmup bars stay UNTAGGED, post-warmup bars get tagged values.
    pre = out.iloc[:PERCENTILE_WINDOW_BARS]
    post = out.iloc[PERCENTILE_WINDOW_BARS:]
    assert (pre["regime"] == REGIME_UNTAGGED).all()
    assert (post["regime"] != REGIME_UNTAGGED).any()


def test_tag_preserves_schema_metadata(tmp_path: Path):
    kp, _, _ = _setup_inputs(tmp_path)
    tag_kline_parquet(asset=Asset.BTC, timeframe=Timeframe.M1, root=tmp_path)
    md = pq.read_schema(kp).metadata or {}
    assert md[b"schema_hash"] == KLINE_SCHEMA_HASH.encode()


def test_tag_is_idempotent(tmp_path: Path):
    kp, _, _ = _setup_inputs(tmp_path)
    tag_kline_parquet(asset=Asset.BTC, timeframe=Timeframe.M1, root=tmp_path)
    first = read_klines(kp)
    tag_kline_parquet(asset=Asset.BTC, timeframe=Timeframe.M1, root=tmp_path)
    second = read_klines(kp)
    pd.testing.assert_frame_equal(first, second)


def test_tag_is_atomic_no_tmp_file_left(tmp_path: Path):
    kp, _, _ = _setup_inputs(tmp_path)
    tag_kline_parquet(asset=Asset.BTC, timeframe=Timeframe.M1, root=tmp_path)
    leftovers = list(kp.parent.glob(f"{kp.name}.tmp*"))
    assert leftovers == []
