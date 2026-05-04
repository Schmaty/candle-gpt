"""Parquet I/O for kline data with schema enforcement on both read and write."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from v2.data.constants import Asset, Timeframe, KLINE_COLUMNS, KLINE_DTYPES
from v2.data.validate import assert_schema


def parquet_path(root: Path, asset: Asset, timeframe: Timeframe) -> Path:
    """Canonical path for a (asset, timeframe) parquet file under `root`."""
    return root / f"{asset.value.lower()}_{timeframe.value}.parquet"


def write_klines(df: pd.DataFrame, path: Path) -> None:
    """Validate then write. Creates parent dirs."""
    assert_schema(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[list(KLINE_COLUMNS)].to_parquet(path, index=False)


def read_klines(path: Path) -> pd.DataFrame:
    """Read and validate. Coerces dtypes to the canonical schema before checking."""
    df = pd.read_parquet(path)
    # Cast first, then validate — parquet sometimes returns float32/int32 depending on engine.
    for col, dtype in KLINE_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    df = df.reindex(columns=list(KLINE_COLUMNS))
    assert_schema(df)
    return df
