"""Parquet I/O for v2 data with schema enforcement + version metadata.

Every write embeds {b"data_version", b"schema_hash"} in the parquet schema
metadata. Every read can be verified against an expected hash via
assert_schema_compatible(path, expected_hash).
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from v2.data.constants import (
    Asset,
    Timeframe,
    DATA_VERSION,
    KLINE_COLUMNS, KLINE_DTYPES, KLINE_SCHEMA_HASH,
    FUNDING_COLUMNS, FUNDING_DTYPES, FUNDING_SCHEMA_HASH,
    LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES, LIQ_BUCKETED_SCHEMA_HASH,
)
from v2.data.validate import assert_schema, SchemaViolation


_META_KEY_VERSION = b"data_version"
_META_KEY_HASH = b"schema_hash"


def parquet_path(root: Path, asset: Asset, timeframe: Timeframe) -> Path:
    return root / f"{asset.value.lower()}_{timeframe.value}.parquet"


def funding_parquet_path(root: Path, asset: Asset) -> Path:
    return root / f"funding_{asset.value.lower()}.parquet"


def liq_bucketed_parquet_path(root: Path, asset: Asset) -> Path:
    return root / f"liq_{asset.value.lower()}_per_minute.parquet"


def _write_with_metadata(
    df: pd.DataFrame,
    path: Path,
    columns: tuple[str, ...],
    dtypes: dict[str, str],
    schema_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[list(columns)].copy()
    for col, dtype in dtypes.items():
        out[col] = out[col].astype(dtype)
    table = pa.Table.from_pandas(out, preserve_index=False)
    new_md = {
        _META_KEY_VERSION: DATA_VERSION.encode("utf-8"),
        _META_KEY_HASH: schema_hash.encode("utf-8"),
    }
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, **new_md})
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _read_and_validate(
    path: Path,
    columns: tuple[str, ...],
    dtypes: dict[str, str],
    schema_hash: str,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    df = df.reindex(columns=list(columns))
    assert_schema(df, columns=columns, dtypes=dtypes)
    assert_schema_compatible(path, schema_hash)
    return df


# --- Kline ---------------------------------------------------------------
def write_klines(df: pd.DataFrame, path: Path) -> None:
    assert_schema(df, columns=KLINE_COLUMNS, dtypes=KLINE_DTYPES)
    _write_with_metadata(df, path, KLINE_COLUMNS, KLINE_DTYPES, KLINE_SCHEMA_HASH)


def read_klines(path: Path) -> pd.DataFrame:
    return _read_and_validate(path, KLINE_COLUMNS, KLINE_DTYPES, KLINE_SCHEMA_HASH)


# --- Funding -------------------------------------------------------------
def write_funding(df: pd.DataFrame, path: Path) -> None:
    assert_schema(df, columns=FUNDING_COLUMNS, dtypes=FUNDING_DTYPES)
    _write_with_metadata(df, path, FUNDING_COLUMNS, FUNDING_DTYPES, FUNDING_SCHEMA_HASH)


def read_funding(path: Path) -> pd.DataFrame:
    return _read_and_validate(path, FUNDING_COLUMNS, FUNDING_DTYPES, FUNDING_SCHEMA_HASH)


# --- Liquidations (per-minute rollup) ------------------------------------
def write_liq_bucketed(df: pd.DataFrame, path: Path) -> None:
    assert_schema(df, columns=LIQ_BUCKETED_COLUMNS, dtypes=LIQ_BUCKETED_DTYPES)
    _write_with_metadata(
        df, path, LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES, LIQ_BUCKETED_SCHEMA_HASH
    )


def read_liq_bucketed(path: Path) -> pd.DataFrame:
    return _read_and_validate(
        path, LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES, LIQ_BUCKETED_SCHEMA_HASH
    )


# --- Schema-compat guardrail --------------------------------------------
def assert_schema_compatible(path: Path, expected_hash: str) -> None:
    """Read pyarrow file metadata and verify the embedded SCHEMA_HASH matches.

    Raises SchemaViolation with a clear migration hint on mismatch / missing.
    """
    schema = pq.read_schema(path)
    md = schema.metadata or {}
    on_disk = md.get(_META_KEY_HASH)
    on_disk_version = md.get(_META_KEY_VERSION, b"<absent>").decode("utf-8")
    if on_disk is None:
        raise SchemaViolation(
            f"{path}: no schema_hash in parquet metadata. File predates v2.0.0 "
            f"versioning; rerun the writer or run tag_regimes (for kline files)."
        )
    if on_disk.decode("utf-8") != expected_hash:
        raise SchemaViolation(
            f"{path}: schema_hash mismatch. on_disk={on_disk.decode()} "
            f"version={on_disk_version} expected={expected_hash}. "
            f"Migration required."
        )
