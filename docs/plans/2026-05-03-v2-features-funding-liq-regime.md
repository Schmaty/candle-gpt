# Candle-GPT v2 — Features Pipeline: Funding, Liquidations, Regime (Plan 1.5 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the v2 data pipeline (Plan 1) before the Day 3–4 feature engineering phase begins by adding three feature streams — perpetual **funding rate**, **liquidation events** (rolled to per-minute aggregates), and offline **regime tagging** — plus a **schema-versioning guardrail** so future schema drift fails loudly instead of silently. After this plan, the dataloader produces fully-joined per-bar feature rows, with `minutes_until_funding` baked in and a `regime` column available for filtering/weighting.

**Architecture:** New modules under `v2/data/`:
- `funding.py` — Binance fapi `/fapi/v1/fundingRate` fetcher → `v2/data/raw/funding_btcusdt.parquet`.
- `liquidations/` — package with three pieces: `collect.py` (live WebSocket → dated raw parquet), `tardis_backfill.py` (stub interface for future Tardis fill), `rollup.py` (raw events → per-minute aggregates → `v2/data/raw/liq_btcusdt_per_minute.parquet`).
- `regime.py` — pure compute: kline + funding + liq → INT8 regime label per bar.
- `tag_regimes.py` — CLI that reads kline + funding + liq parquets, computes regimes, and **mutates the kline parquet in place** with an atomic temp-write-then-rename.

The kline parquet schema gains one column (`regime: int8`) in v2.0.0. Funding and liquidation data live in **separate raw parquets**; the dataloader joins them at load-time, never at storage time. Regime is the **one** feature written into the kline parquet directly so dataloaders can `.filter()` or weight by regime cheaply.

**Tech Stack:** Python 3.11+, `uv`, `pandas` + `pyarrow`, `requests` (REST), `websockets` (live liq stream), `pytest` + `responses` for tests.

**Scope notes:**
- This plan ends at "feature parquets exist, the kline parquet has a populated regime column, and the dataloader can consume the joined feature row." No model code, no 41-dim feature vector engineering — those are Plan 2.
- **Tardis backfill is a stub.** The interface and target schema are defined; the actual subscription / fetch is `NotImplementedError`. A fresh Tardis fill drops in later **without** a schema migration.
- **L2/L3 order book depth is explicitly Plan N+1**, contingent on funding+liq features showing edge in walk-forward validation.
- The live liquidation WebSocket runner is **specified but not started** by this plan. It's a separate operational concern.
- The existing BTCUSDT 1m parquet from Plan 1 is **schema-incompatible** after Task 1 (regime column added). Migration = run `tag_regimes` once. Task 8 is exactly that.
- v1 code in `data/`, `model/`, `server/`, `web/` remains untouched.

---

## File Structure

**To create:**
- `projects/candle-gpt/v2/data/funding.py` — funding-rate fetcher (CLI + library).
- `projects/candle-gpt/v2/data/liquidations/__init__.py` — empty marker.
- `projects/candle-gpt/v2/data/liquidations/collect.py` — live WebSocket collector (runnable but not started here).
- `projects/candle-gpt/v2/data/liquidations/tardis_backfill.py` — backfill stub: empty parquet writer + `NotImplementedError` real fetcher.
- `projects/candle-gpt/v2/data/liquidations/rollup.py` — raw events → per-minute aggregate parquet.
- `projects/candle-gpt/v2/data/regime.py` — pure compute: classify bars into the three-bucket regime taxonomy.
- `projects/candle-gpt/v2/data/tag_regimes.py` — CLI: load kline + funding + liq, compute regimes, atomically write back.
- `projects/candle-gpt/v2/data/raw/liquidations/.gitkeep` — placeholder so `raw/liquidations/` exists.
- `projects/candle-gpt/v2/tests/test_schema_versioning.py`
- `projects/candle-gpt/v2/tests/test_funding.py`
- `projects/candle-gpt/v2/tests/test_liquidations_rollup.py`
- `projects/candle-gpt/v2/tests/test_liquidations_tardis_stub.py`
- `projects/candle-gpt/v2/tests/test_liquidations_collect.py`
- `projects/candle-gpt/v2/tests/test_regime.py`
- `projects/candle-gpt/v2/tests/test_tag_regimes.py`
- `projects/candle-gpt/v2/tests/test_dataloader_join.py`

**To modify:**
- `projects/candle-gpt/v2/data/constants.py` — add `DATA_VERSION`, three `*_SCHEMA_HASH` constants, `regime` column on kline schema, funding & liq schemas.
- `projects/candle-gpt/v2/data/store.py` — embed schema metadata via pyarrow on every write, add `assert_schema_compatible(path, expected_hash)`, mirror `write_klines`/`read_klines` for funding + liq parquets.
- `projects/candle-gpt/v2/data/fetch.py` — append `regime: int8 = -1` (untagged sentinel) to fetched DataFrame before `write_klines`.
- `projects/candle-gpt/v2/data/dataset.py` — extend `KlineWindowDataset` to optionally accept `funding_path` and `liq_path`; emit `funding_rate`, `mark_price`, `minutes_until_funding`, and 6 liq aggregates as joined columns.
- `projects/candle-gpt/v2/tests/test_constants.py` — update for new columns + new constants.
- `projects/candle-gpt/v2/tests/test_store.py` — fixtures gain `regime` column, new tests for funding/liq round-trip + metadata.
- `projects/candle-gpt/v2/tests/test_validate.py` — fixture gains `regime` column.
- `projects/candle-gpt/v2/tests/test_dataset.py` — fixture gains `regime` column; shape assertions auto-update via `len(KLINE_COLUMNS)`.
- `projects/candle-gpt/v2/tests/test_fetch.py` — assertions account for the regime sentinel column on disk.
- `projects/candle-gpt/pyproject.toml` — add `websockets` runtime dep.

**Not touched:**
- `projects/candle-gpt/data/` (v1) / `model/` / `server/` / `web/`.

---

## Task 1: Schema versioning — DATA_VERSION, schema hashes, metadata, guardrail

This is the foundation. Every other parquet written in this plan embeds `data_version` + `schema_hash` in pyarrow metadata, and `assert_schema_compatible` is the single tripwire that fires on any drift. The kline schema bumps to v2.0.0 (adds `regime: int8`); funding and liq parquets get their own hashes (new in v2.0.0).

**Files:**
- Modify: `projects/candle-gpt/v2/data/constants.py`
- Modify: `projects/candle-gpt/v2/data/store.py`
- Modify: `projects/candle-gpt/v2/data/fetch.py`
- Modify: `projects/candle-gpt/v2/tests/test_constants.py`, `test_validate.py`, `test_store.py`, `test_dataset.py`, `test_fetch.py`
- Create: `projects/candle-gpt/v2/tests/test_schema_versioning.py`

- [ ] **Step 1: Update `constants.py` to v2.0.0**

Path: `v2/data/constants.py` — replace the file body with:

```python
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
    """Deterministic sha256 over the (column, dtype) tuple list in order.

    Computed at import time; embedded in parquet metadata; checked at read time.
    """
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
```

- [ ] **Step 2: Update `store.py` with metadata embedding + the guardrail**

Path: `v2/data/store.py` — replace file body with:

```python
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
    # Coerce to canonical column order + dtypes, then enforce.
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
    # Atomic write: temp file in same dir, then rename.
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
    df = df.reindex(columns=list(columns), copy=False)
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
```

- [ ] **Step 3: Extend `validate.assert_schema` to take optional schema args**

The Plan-1 `assert_schema(df)` validated against `KLINE_COLUMNS`/`KLINE_DTYPES` only. Make it parameterizable so funding + liq writers can reuse it. In `v2/data/validate.py`, replace the existing `assert_schema` with:

```python
def assert_schema(
    df: pd.DataFrame,
    columns: tuple[str, ...] | None = None,
    dtypes: dict[str, str] | None = None,
) -> None:
    """Raise SchemaViolation if df doesn't match the given schema exactly.

    Defaults to the kline schema for backward compatibility with Plan-1 callers.
    """
    from v2.data.constants import KLINE_COLUMNS, KLINE_DTYPES
    cols = columns if columns is not None else KLINE_COLUMNS
    dts = dtypes if dtypes is not None else KLINE_DTYPES

    actual_cols = tuple(df.columns)
    if actual_cols != cols:
        missing = set(cols) - set(actual_cols)
        extra = set(actual_cols) - set(cols)
        problems: list[str] = []
        if missing:
            problems.append(f"missing={sorted(missing)}")
        if extra:
            problems.append(f"unexpected={sorted(extra)}")
        if not problems:
            problems.append(f"wrong_order: got {actual_cols}, want {cols}")
        raise SchemaViolation("schema mismatch: " + "; ".join(problems))

    for col, want_dtype in dts.items():
        got_dtype = str(df[col].dtype)
        if got_dtype != want_dtype:
            raise SchemaViolation(
                f"schema mismatch: column {col} dtype={got_dtype}, want {want_dtype}"
            )
```

- [ ] **Step 4: Update `fetch.py` to append the untagged regime sentinel**

In `v2/data/fetch.py`, inside `fetch_to_parquet`, **after** `df = dedupe_open_time(df)` and **before** `write_klines(df, out_path)`, insert:

```python
    # v2.0.0: kline schema includes regime (int8). Fetcher writes -1 sentinel;
    # tag_regimes overwrites with {0,1,2} once funding + liq are available.
    df = df.copy()
    df["regime"] = pd.Series([-1] * len(df), dtype="int8")
```

- [ ] **Step 5: Update Plan-1 test fixtures for the new kline schema**

Each Plan-1 test file has a fixture-builder that constructs 7-column kline DataFrames. Add a `regime` column (`int8`, value `-1`) to each so the fixtures are schema-valid under v2.0.0.

In `v2/tests/test_validate.py`, in `_make_df`, append:
```python
        "regime": pd.array([-1] * len(open_times_ms), dtype="int8"),
```

In `v2/tests/test_store.py`, in `_good_df`, append:
```python
        "regime": pd.array([-1, -1], dtype="int8"),
```

In `v2/tests/test_dataset.py`, in `_write_synthetic`, append:
```python
        "regime": pd.array([-1] * n_bars, dtype="int8"),
```

In `v2/tests/test_fetch.py`, no fixture change is needed because the test-side helpers don't construct kline DataFrames directly — but **add an assertion** to `test_fetch_walks_backward_until_target_start` that checks the on-disk file contains `regime == -1`:

```python
    # Schema-version sanity: fetcher writes the untagged sentinel.
    out_df = pd.read_parquet(out_path)
    assert "regime" in out_df.columns
    assert (out_df["regime"] == -1).all()
    assert out_df["regime"].dtype == "int8"
```

In `v2/tests/test_constants.py`, replace `test_kline_columns_canonical_order` and `test_kline_dtypes_match_columns` with:

```python
def test_kline_columns_canonical_order():
    assert KLINE_COLUMNS == (
        "open_time", "open", "high", "low", "close", "volume", "close_time", "regime",
    )


def test_kline_dtypes_match_columns():
    assert set(KLINE_DTYPES.keys()) == set(KLINE_COLUMNS)
    assert KLINE_DTYPES["open_time"] == "int64"
    assert KLINE_DTYPES["close"] == "float64"
    assert KLINE_DTYPES["regime"] == "int8"
```

- [ ] **Step 6: Write the new `test_schema_versioning.py`**

Path: `v2/tests/test_schema_versioning.py`

```python
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
```

- [ ] **Step 7: Run all tests; expect Plan-1 + new schema tests passing**

```bash
cd projects/candle-gpt
uv run pytest v2/tests/ -v
```

Expected: Plan-1's 31 tests still pass (with their now-8-col fixtures), plus 9 new schema-versioning tests = 40 passing.

- [ ] **Step 8: Commit**

```bash
git add v2/data/constants.py v2/data/store.py v2/data/validate.py v2/data/fetch.py \
        v2/tests/test_constants.py v2/tests/test_validate.py v2/tests/test_store.py \
        v2/tests/test_dataset.py v2/tests/test_fetch.py v2/tests/test_schema_versioning.py
git commit -m "v2: schema versioning — DATA_VERSION 2.0.0, schema hashes, metadata guardrail"
```

---

## Task 2: Funding rate fetcher

Pulls historical funding rate + mark price from Binance perpetual futures (`/fapi/v1/fundingRate`). Walks **forward** in time from `target_start_ms` to `end_ms`, max 1000 records per request. ~6.5 years × ~3 events/day = ~7,000 rows total — a single mid-sized parquet.

**Files:**
- Create: `projects/candle-gpt/v2/data/funding.py`
- Create: `projects/candle-gpt/v2/tests/test_funding.py`

- [ ] **Step 1: Write the failing tests**

Path: `v2/tests/test_funding.py`

```python
"""Funding-rate fetcher: pagination, schema, dedup, stop conditions."""
from pathlib import Path

import pandas as pd
import pytest
import responses

from v2.data.constants import Asset, FUNDING_COLUMNS
from v2.data.funding import (
    BINANCE_FUNDING_URL,
    chunk_to_funding_rows,
    fetch_funding_to_parquet,
)


def _payload_row(funding_time_ms: int, rate: str = "0.00010000",
                 mark_price: str = "30000.00") -> dict:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": funding_time_ms,
        "fundingRate": rate,
        "markPrice": mark_price,
    }


def test_chunk_to_funding_rows_keeps_canonical_columns():
    rows = [_payload_row(0), _payload_row(8 * 3_600_000)]
    df = chunk_to_funding_rows(rows)
    assert list(df.columns) == list(FUNDING_COLUMNS)
    assert df["funding_time"].tolist() == [0, 8 * 3_600_000]
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert df["mark_price"].iloc[0] == pytest.approx(30000.0)


def test_chunk_to_funding_rows_handles_missing_mark_price():
    """Some early historical rows lack markPrice — fill NaN, do not drop."""
    rows = [{"symbol": "BTCUSDT", "fundingTime": 0, "fundingRate": "0.0001"}]
    df = chunk_to_funding_rows(rows)
    assert pd.isna(df["mark_price"].iloc[0])


@responses.activate
def test_fetch_walks_forward_until_end_ms(tmp_path: Path):
    # Three pages, each with two rows, walking forward.
    eight_h = 8 * 3_600_000
    page1 = [_payload_row(0), _payload_row(eight_h)]
    page2 = [_payload_row(2 * eight_h), _payload_row(3 * eight_h)]
    page3 = [_payload_row(4 * eight_h)]
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page3, status=200)

    out_path = tmp_path / "funding_btcusdt.parquet"
    df = fetch_funding_to_parquet(
        asset=Asset.BTC,
        target_start_ms=0,
        end_ms=4 * eight_h + 1,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["funding_time"].tolist() == [0, eight_h, 2 * eight_h, 3 * eight_h, 4 * eight_h]
    assert out_path.exists()


@responses.activate
def test_fetch_stops_on_empty_chunk(tmp_path: Path):
    eight_h = 8 * 3_600_000
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=[_payload_row(0)], status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=[], status=200)
    out_path = tmp_path / "funding_btcusdt.parquet"
    df = fetch_funding_to_parquet(
        asset=Asset.BTC,
        target_start_ms=0,
        end_ms=10 * eight_h,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["funding_time"].tolist() == [0]


@responses.activate
def test_fetch_dedupes_overlap(tmp_path: Path):
    eight_h = 8 * 3_600_000
    # API sometimes echoes the cursor row at the head of next page.
    page1 = [_payload_row(0), _payload_row(eight_h)]
    page2 = [_payload_row(eight_h), _payload_row(2 * eight_h)]
    page3 = []
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_FUNDING_URL, json=page3, status=200)
    out_path = tmp_path / "funding_btcusdt.parquet"
    df = fetch_funding_to_parquet(
        asset=Asset.BTC,
        target_start_ms=0,
        end_ms=10 * eight_h,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["funding_time"].tolist() == [0, eight_h, 2 * eight_h]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest v2/tests/test_funding.py -v
```

Expected: `ModuleNotFoundError: No module named 'v2.data.funding'`.

- [ ] **Step 3: Implement `funding.py`**

Path: `v2/data/funding.py`

```python
"""Binance perpetual funding-rate fetcher.

GET /fapi/v1/fundingRate — paginated, max 1000 rows per request, walks forward
in time. Stores to parquet via store.write_funding (which embeds schema metadata).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from v2.data.constants import (
    Asset,
    FUNDING_COLUMNS,
    FUNDING_DTYPES,
)
from v2.data.store import funding_parquet_path, write_funding


BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
MAX_LIMIT = 1000  # Binance hard cap


def chunk_to_funding_rows(rows: list[dict]) -> pd.DataFrame:
    """Convert a /fapi/v1/fundingRate response into the canonical funding schema."""
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=FUNDING_DTYPES[c]) for c in FUNDING_COLUMNS})
    df = pd.DataFrame({
        "funding_time": pd.array([int(r["fundingTime"]) for r in rows], dtype="int64"),
        "funding_rate": [float(r["fundingRate"]) for r in rows],
        "mark_price":   [float(r["markPrice"]) if r.get("markPrice") is not None else float("nan")
                         for r in rows],
    })
    return df


def _fetch_chunk(symbol: str, start_ms: int, limit: int = MAX_LIMIT) -> list[dict]:
    params = {"symbol": symbol, "startTime": start_ms, "limit": limit}
    r = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_funding_to_parquet(
    asset: Asset,
    target_start_ms: int,
    end_ms: int,
    out_path: Path,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    all_chunks: list[pd.DataFrame] = []
    seen: set[int] = set()
    cursor = target_start_ms
    while cursor < end_ms:
        rows = _fetch_chunk(asset.value, cursor)
        if not rows:
            break
        df_chunk = chunk_to_funding_rows(rows)
        new_mask = ~df_chunk["funding_time"].isin(seen)
        if not new_mask.any():
            break
        new_df = df_chunk[new_mask & (df_chunk["funding_time"] < end_ms)]
        if new_df.empty:
            break
        seen.update(int(t) for t in new_df["funding_time"].tolist())
        all_chunks.append(new_df)
        latest = int(new_df["funding_time"].max())
        cursor = latest + 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not all_chunks:
        raise RuntimeError(
            f"No funding data returned for {asset.value} in window "
            f"[{target_start_ms}, {end_ms}]"
        )
    df = (
        pd.concat(all_chunks, ignore_index=True)
          .sort_values("funding_time", kind="mergesort")
          .drop_duplicates(subset=["funding_time"], keep="first")
          .reset_index(drop=True)
    )
    write_funding(df, out_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Binance perpetual funding rates.")
    ap.add_argument("--asset", required=True, choices=[a.value for a in Asset])
    ap.add_argument("--years", type=float, default=6.5,
                    help="history depth in years (default: 6.5)")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "raw",
                    help="output dir (parquet path = root / funding_<symbol>.parquet)")
    args = ap.parse_args()

    asset = Asset(args.asset)
    out_path = funding_parquet_path(args.root, asset)
    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - int(args.years * 365 * 24 * 60 * 60 * 1000)
    print(f"Fetching funding for {asset.value} ({args.years} years) → {out_path}")
    df = fetch_funding_to_parquet(
        asset=asset,
        target_start_ms=target_start_ms,
        end_ms=now_ms,
        out_path=out_path,
    )
    print(f"Done. {len(df)} funding events saved.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest v2/tests/test_funding.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/data/funding.py v2/tests/test_funding.py
git commit -m "v2: funding — Binance perpetual fundingRate fetcher (CLI + lib)"
```

---

## Task 3: Run funding fetch end-to-end for BTCUSDT

Real network operation, single shot. Idempotent — re-running overwrites the parquet from scratch. Expected runtime: ~10–20 s (7 round-trips at 0.05 s sleep + network latency).

- [ ] **Step 1: Run the fetch**

```bash
cd projects/candle-gpt
uv run python -m v2.data.funding --asset BTCUSDT --years 6.5
```

Expected stdout:
```
Fetching funding for BTCUSDT (6.5 years) → .../v2/data/raw/funding_btcusdt.parquet
Done. ~7100 funding events saved.
```

- [ ] **Step 2: Sanity-check the output**

```bash
uv run python -c "
import pandas as pd
from pathlib import Path
from v2.data.store import read_funding
df = read_funding(Path('v2/data/raw/funding_btcusdt.parquet'))
print(f'rows={len(df):,}')
print(f'first={pd.Timestamp(df.funding_time.iloc[0], unit=\"ms\", tz=\"UTC\")}')
print(f'last ={pd.Timestamp(df.funding_time.iloc[-1], unit=\"ms\", tz=\"UTC\")}')
print(f'rate range: [{df.funding_rate.min():.6f}, {df.funding_rate.max():.6f}]')
print(f'cadence: median Δt = {df.funding_time.diff().median()/3_600_000:.1f} h')
"
```

Expected:
- `rows` between 6,500 and 7,500.
- `first` ≈ 2019-09 (when Binance perpetual launched).
- Median Δt ≈ 8.0 h.
- Rate range plausibly [-0.005, +0.005] (most rates are tiny, but extremes exist).

- [ ] **Step 3: Marker commit**

```bash
echo "BTCUSDT funding fetched $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > v2/data/raw/FUNDING_FETCHED.md
git add v2/data/raw/FUNDING_FETCHED.md
git commit -m "v2: data — fetched BTCUSDT funding history (~6.5y)"
```

---

## Task 4: Liquidations — per-minute rollup module + Tardis backfill stub

This task defines the **dataloader-facing** liquidation parquet schema (`LIQ_BUCKETED_COLUMNS`) and the rollup function that produces it from raw per-event input. It also stubs the Tardis interface so downstream code (regime, dataloader) can be written today against real column names without waiting on a Tardis subscription.

**Files:**
- Create: `projects/candle-gpt/v2/data/liquidations/__init__.py`
- Create: `projects/candle-gpt/v2/data/liquidations/rollup.py`
- Create: `projects/candle-gpt/v2/data/liquidations/tardis_backfill.py`
- Create: `projects/candle-gpt/v2/data/raw/liquidations/.gitkeep`
- Create: `projects/candle-gpt/v2/tests/test_liquidations_rollup.py`
- Create: `projects/candle-gpt/v2/tests/test_liquidations_tardis_stub.py`

- [ ] **Step 1: Create the package directory + marker**

```bash
mkdir -p v2/data/liquidations v2/data/raw/liquidations
touch v2/data/liquidations/__init__.py v2/data/raw/liquidations/.gitkeep
```

- [ ] **Step 2: Write the failing rollup tests**

Path: `v2/tests/test_liquidations_rollup.py`

```python
"""Per-minute liquidation rollup: bucketing, side-aware aggregation, schema."""
from pathlib import Path

import pandas as pd
import pytest

from v2.data.constants import LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES
from v2.data.liquidations.rollup import (
    EVENT_COLUMNS,
    EVENT_DTYPES,
    roll_to_per_minute,
)
from v2.data.store import write_liq_bucketed, read_liq_bucketed


def _events(rows: list[dict]) -> pd.DataFrame:
    """Build a per-event DataFrame matching EVENT_COLUMNS/EVENT_DTYPES."""
    df = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    for col, dtype in EVENT_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def test_rollup_returns_canonical_schema():
    df = _events([
        {"event_time": 0, "side": "long",  "price": 100.0, "qty": 1.0, "notional": 100.0},
    ])
    out = roll_to_per_minute(df)
    assert list(out.columns) == list(LIQ_BUCKETED_COLUMNS)
    for col, dtype in LIQ_BUCKETED_DTYPES.items():
        assert str(out[col].dtype) == dtype


def test_rollup_floors_to_minute_buckets():
    df = _events([
        {"event_time": 30_000,  "side": "long",  "price": 1.0, "qty": 1.0, "notional": 50.0},
        {"event_time": 45_000,  "side": "short", "price": 1.0, "qty": 1.0, "notional": 70.0},
        {"event_time": 60_000,  "side": "long",  "price": 1.0, "qty": 1.0, "notional": 80.0},
    ])
    out = roll_to_per_minute(df)
    assert out["bucket_time"].tolist() == [0, 60_000]
    # Bucket 0 has the two events totaling 120; bucket 60_000 has one of 80.
    assert out.loc[out.bucket_time == 0, "count"].iloc[0] == 2
    assert out.loc[out.bucket_time == 0, "sum_notional"].iloc[0] == pytest.approx(120.0)
    assert out.loc[out.bucket_time == 60_000, "sum_notional"].iloc[0] == pytest.approx(80.0)


def test_rollup_side_split():
    df = _events([
        {"event_time": 0, "side": "long",  "price": 1.0, "qty": 1.0, "notional": 30.0},
        {"event_time": 0, "side": "long",  "price": 1.0, "qty": 1.0, "notional": 50.0},
        {"event_time": 0, "side": "short", "price": 1.0, "qty": 1.0, "notional": 200.0},
    ])
    out = roll_to_per_minute(df)
    row = out.iloc[0]
    assert row["long_liq_count"] == 2
    assert row["long_liq_notional"] == pytest.approx(80.0)
    assert row["short_liq_count"] == 1
    assert row["short_liq_notional"] == pytest.approx(200.0)
    assert row["count"] == 3
    assert row["max_single"] == pytest.approx(200.0)


def test_rollup_empty_events_returns_empty_canonical():
    df = _events([])
    out = roll_to_per_minute(df)
    assert list(out.columns) == list(LIQ_BUCKETED_COLUMNS)
    assert len(out) == 0


def test_rollup_rejects_unknown_side():
    df = _events([
        {"event_time": 0, "side": "diagonal", "price": 1.0, "qty": 1.0, "notional": 10.0},
    ])
    with pytest.raises(ValueError, match="side"):
        roll_to_per_minute(df)


def test_rollup_round_trips_through_store(tmp_path: Path):
    df = _events([
        {"event_time": 0, "side": "long",  "price": 1.0, "qty": 1.0, "notional": 30.0},
        {"event_time": 60_000, "side": "short", "price": 1.0, "qty": 1.0, "notional": 70.0},
    ])
    rolled = roll_to_per_minute(df)
    p = tmp_path / "liq.parquet"
    write_liq_bucketed(rolled, p)
    back = read_liq_bucketed(p)
    pd.testing.assert_frame_equal(back, rolled)
```

- [ ] **Step 3: Write the failing Tardis stub tests**

Path: `v2/tests/test_liquidations_tardis_stub.py`

```python
"""Tardis backfill stub: writes a valid empty parquet matching the rollup schema."""
from pathlib import Path

import pytest

from v2.data.constants import LIQ_BUCKETED_COLUMNS
from v2.data.store import read_liq_bucketed
from v2.data.liquidations.tardis_backfill import (
    write_empty_backfill,
    fetch_tardis_backfill,
)


def test_write_empty_backfill_creates_canonical_parquet(tmp_path: Path):
    out = tmp_path / "liq_backfill.parquet"
    write_empty_backfill(out)
    df = read_liq_bucketed(out)
    assert list(df.columns) == list(LIQ_BUCKETED_COLUMNS)
    assert len(df) == 0


def test_fetch_tardis_backfill_raises_until_subscription(tmp_path: Path):
    out = tmp_path / "liq.parquet"
    with pytest.raises(NotImplementedError, match="Tardis"):
        fetch_tardis_backfill(start_ms=0, end_ms=1, out_path=out)
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
uv run pytest v2/tests/test_liquidations_rollup.py v2/tests/test_liquidations_tardis_stub.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Implement `rollup.py`**

Path: `v2/data/liquidations/rollup.py`

```python
"""Per-event → per-minute liquidation rollup.

Per-event schema (input from live collector / Tardis):
    event_time : int64 (ms epoch)
    side       : str ("long" | "short")  long = long position liquidated (market sell)
    price      : float64
    qty        : float64
    notional   : float64 (price * qty)

Per-minute schema (output, joined by dataloader): see LIQ_BUCKETED_COLUMNS.
"""
from __future__ import annotations
import pandas as pd

from v2.data.constants import LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES


EVENT_COLUMNS: tuple[str, ...] = ("event_time", "side", "price", "qty", "notional")
EVENT_DTYPES: dict[str, str] = {
    "event_time": "int64",
    "side":       "object",
    "price":      "float64",
    "qty":        "float64",
    "notional":   "float64",
}

_MINUTE_MS = 60_000
_VALID_SIDES = frozenset({"long", "short"})


def _empty_bucketed() -> pd.DataFrame:
    return pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })


def roll_to_per_minute(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event rows into per-minute buckets matching LIQ_BUCKETED_COLUMNS."""
    if events.empty:
        return _empty_bucketed()

    bad = ~events["side"].isin(_VALID_SIDES)
    if bad.any():
        offenders = events.loc[bad, "side"].unique().tolist()
        raise ValueError(f"unknown side(s): {offenders}; expected one of {_VALID_SIDES}")

    df = events.copy()
    df["bucket_time"] = (df["event_time"] // _MINUTE_MS) * _MINUTE_MS
    df["is_long"] = df["side"] == "long"
    df["is_short"] = df["side"] == "short"
    df["long_notional"] = df["notional"].where(df["is_long"], 0.0)
    df["short_notional"] = df["notional"].where(df["is_short"], 0.0)

    grouped = df.groupby("bucket_time", sort=True).agg(
        count=("event_time", "count"),
        sum_notional=("notional", "sum"),
        max_single=("notional", "max"),
        long_liq_count=("is_long", "sum"),
        long_liq_notional=("long_notional", "sum"),
        short_liq_count=("is_short", "sum"),
        short_liq_notional=("short_notional", "sum"),
    ).reset_index()

    # Coerce to the canonical dtypes; groupby may produce int64 for sums of bool, etc.
    for col, dtype in LIQ_BUCKETED_DTYPES.items():
        grouped[col] = grouped[col].astype(dtype)
    return grouped[list(LIQ_BUCKETED_COLUMNS)]
```

- [ ] **Step 6: Implement `tardis_backfill.py`**

Path: `v2/data/liquidations/tardis_backfill.py`

```python
"""Tardis.dev liquidations backfill stub.

We don't have a Tardis subscription as of plan-write time. This module exposes:

  - write_empty_backfill(out_path): writes a valid parquet with zero rows but
    the canonical LIQ_BUCKETED_COLUMNS + schema metadata. The dataloader can
    join against this today and downstream code (regime, model) is written
    against real column names. When the real fill drops in later, no schema
    migration is required.

  - fetch_tardis_backfill(...): raises NotImplementedError. Wire this up once
    a Tardis API key is available; signature is committed.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from v2.data.constants import LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES
from v2.data.store import write_liq_bucketed


def write_empty_backfill(out_path: Path) -> None:
    """Write a zero-row parquet matching LIQ_BUCKETED_COLUMNS, with metadata."""
    df = pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })
    write_liq_bucketed(df, out_path)


def fetch_tardis_backfill(
    *,
    start_ms: int,
    end_ms: int,
    out_path: Path,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Pull `liquidations` channel from Tardis (binance-futures BTCUSDT), roll up, write.

    Pending Tardis subscription. When implemented:
      1. Request `liquidations` channel for `binance-futures BTCUSDT` over the window.
      2. Map each event {timestamp, side="sell"|"buy", price, amount} →
         {event_time, side="long" if "sell" else "short", price, qty, notional}.
      3. Pass to rollup.roll_to_per_minute.
      4. Persist via store.write_liq_bucketed(out_path).
    """
    raise NotImplementedError(
        "Tardis backfill not yet wired up. Use write_empty_backfill() to scaffold an "
        "empty parquet against the canonical schema; the dataloader and regime tagger "
        "tolerate empty liq parquets (zero counts everywhere)."
    )
```

- [ ] **Step 7: Run tests; expect green**

```bash
uv run pytest v2/tests/test_liquidations_rollup.py v2/tests/test_liquidations_tardis_stub.py -v
```

Expected: 6 + 2 = 8 passed.

- [ ] **Step 8: Write an empty backfill parquet for downstream tasks**

```bash
uv run python -c "
from pathlib import Path
from v2.data.liquidations.tardis_backfill import write_empty_backfill
write_empty_backfill(Path('v2/data/raw/liq_btcusdt_per_minute.parquet'))
print('empty backfill written')
"
```

This produces a zero-row, schema-valid parquet at the canonical path so Tasks 7–9 can run against it.

- [ ] **Step 9: Commit**

```bash
git add v2/data/liquidations/ v2/data/raw/liquidations/.gitkeep \
        v2/tests/test_liquidations_rollup.py v2/tests/test_liquidations_tardis_stub.py
git commit -m "v2: liquidations — per-minute rollup + Tardis backfill stub"
```

---

## Task 5: Liquidations — live WebSocket collector (specified, not started)

Subscribes to Binance USDT-M futures `forceOrder@arr` stream, parses each force-liquidation event into the per-event schema, and appends it to a dated parquet under `v2/data/raw/liquidations/YYYY-MM-DD.parquet`. The collector is **runnable** but **not started** in this plan — it's an operational concern.

**Files:**
- Create: `projects/candle-gpt/v2/data/liquidations/collect.py`
- Create: `projects/candle-gpt/v2/tests/test_liquidations_collect.py`
- Modify: `projects/candle-gpt/pyproject.toml`

- [ ] **Step 1: Add `websockets` runtime dep**

```bash
cd projects/candle-gpt
uv add websockets
```

- [ ] **Step 2: Write the failing collector tests**

Path: `v2/tests/test_liquidations_collect.py`

```python
"""Live liquidation collector parsing + restart-safe append (no real socket here)."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from v2.data.liquidations.rollup import EVENT_COLUMNS, EVENT_DTYPES
from v2.data.liquidations.collect import (
    parse_force_order_event,
    daily_parquet_path,
    append_events,
)


# Sample forceOrder@arr payload from Binance docs.
SAMPLE_PAYLOAD = {
    "e": "forceOrder",
    "E": 1700000000123,
    "o": {
        "s": "BTCUSDT",
        "S": "SELL",         # SELL = a long position got liquidated → side="long"
        "o": "LIMIT",
        "f": "IOC",
        "q": "0.5",          # qty
        "p": "30000.5",      # price (limit)
        "ap": "30001.2",     # average filled price
        "X": "FILLED",
        "l": "0.5",
        "z": "0.5",
        "T": 1700000000456,
    },
}


def test_parse_force_order_event_maps_sell_to_long():
    ev = parse_force_order_event(SAMPLE_PAYLOAD)
    assert ev["side"] == "long"
    assert ev["event_time"] == 1700000000456
    assert ev["price"] == pytest.approx(30001.2)  # uses average price, not limit
    assert ev["qty"] == pytest.approx(0.5)
    assert ev["notional"] == pytest.approx(0.5 * 30001.2)


def test_parse_force_order_event_maps_buy_to_short():
    payload = {**SAMPLE_PAYLOAD, "o": {**SAMPLE_PAYLOAD["o"], "S": "BUY"}}
    ev = parse_force_order_event(payload)
    assert ev["side"] == "short"


def test_parse_rejects_non_btc_symbol():
    payload = {**SAMPLE_PAYLOAD, "o": {**SAMPLE_PAYLOAD["o"], "s": "ETHUSDT"}}
    with pytest.raises(ValueError, match="symbol"):
        parse_force_order_event(payload, expected_symbol="BTCUSDT")


def test_daily_parquet_path_rolls_at_utc_midnight(tmp_path: Path):
    p = daily_parquet_path(tmp_path, event_time_ms=1700006400000)
    # 2023-11-15 00:00:00 UTC → "2023-11-15.parquet"
    assert p.name == "2023-11-15.parquet"


def test_append_events_creates_then_appends(tmp_path: Path):
    p = tmp_path / "2023-11-15.parquet"
    e1 = parse_force_order_event(SAMPLE_PAYLOAD)
    append_events(p, [e1])
    df1 = pd.read_parquet(p)
    assert len(df1) == 1
    assert list(df1.columns) == list(EVENT_COLUMNS)

    e2 = parse_force_order_event(
        {**SAMPLE_PAYLOAD, "o": {**SAMPLE_PAYLOAD["o"], "S": "BUY"}}
    )
    append_events(p, [e2])
    df2 = pd.read_parquet(p)
    assert len(df2) == 2  # restart-safe: existing rows preserved
```

- [ ] **Step 3: Run tests; expect failure**

```bash
uv run pytest v2/tests/test_liquidations_collect.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `collect.py`**

Path: `v2/data/liquidations/collect.py`

```python
"""Live liquidation collector for Binance USDT-M futures.

Subscribes to the `forceOrder@arr` stream (all symbols), filters for the
target symbol (BTCUSDT by default), parses events into the per-event schema,
and appends to a dated parquet at `<root>/YYYY-MM-DD.parquet` (UTC).

Restart-safe: appending re-reads the existing dated file (if any) and rewrites
atomically. The script is meant to run as a long-lived background process;
this plan does NOT start it. Wire up via systemd / launchd separately.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from v2.data.liquidations.rollup import EVENT_COLUMNS, EVENT_DTYPES


WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"


def parse_force_order_event(payload: dict, expected_symbol: str = "BTCUSDT") -> dict:
    """Map a Binance forceOrder payload to the per-event schema."""
    o = payload["o"]
    if o["s"] != expected_symbol:
        raise ValueError(f"unexpected symbol: {o['s']!r}, want {expected_symbol!r}")
    side = "long" if o["S"] == "SELL" else "short" if o["S"] == "BUY" else None
    if side is None:
        raise ValueError(f"unexpected order side: {o['S']!r}")
    qty = float(o["q"])
    price = float(o["ap"])  # use average filled price, not the limit `p`
    return {
        "event_time": int(o["T"]),
        "side": side,
        "price": price,
        "qty": qty,
        "notional": price * qty,
    }


def daily_parquet_path(root: Path, event_time_ms: int) -> Path:
    dt = datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc)
    return root / f"{dt.strftime('%Y-%m-%d')}.parquet"


def append_events(path: Path, events: list[dict]) -> None:
    if not events:
        return
    new_df = pd.DataFrame(events, columns=list(EVENT_COLUMNS))
    for col, dtype in EVENT_DTYPES.items():
        new_df[col] = new_df[col].astype(dtype)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = new_df
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


async def _run(root: Path, expected_symbol: str) -> None:  # pragma: no cover - I/O
    import websockets  # local import keeps test imports fast

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                buffer: list[dict] = []
                last_flush_ms = 0
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("e") != "forceOrder":
                        continue
                    if msg["o"]["s"] != expected_symbol:
                        continue
                    try:
                        ev = parse_force_order_event(msg, expected_symbol=expected_symbol)
                    except ValueError:
                        continue
                    buffer.append(ev)
                    # Flush every ~5 seconds OR when buffer grows.
                    if len(buffer) >= 16 or ev["event_time"] - last_flush_ms > 5000:
                        path = daily_parquet_path(root, ev["event_time"])
                        append_events(path, buffer)
                        buffer.clear()
                        last_flush_ms = ev["event_time"]
        except Exception as e:
            print(f"[collect] reconnecting after error: {e}")
            await asyncio.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Binance forceOrder collector.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data" / "raw" / "liquidations")
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"[collect] subscribing to {WS_URL} for {args.symbol}; writing to {args.root}")
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(_run(args.root, args.symbol))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests; expect green**

```bash
uv run pytest v2/tests/test_liquidations_collect.py -v
```

Expected: 5 passed. (The `_run` coroutine is excluded from coverage by the `pragma: no cover` comment; we test only the parser/append helpers.)

- [ ] **Step 6: Commit**

```bash
git add v2/data/liquidations/collect.py v2/tests/test_liquidations_collect.py \
        pyproject.toml uv.lock
git commit -m "v2: liquidations — live forceOrder@arr collector (parser + append, runner stub)"
```

---

## Task 6: Regime tagging module — pure compute

Three-bucket taxonomy with **explicit numeric thresholds** and a documented **priority order**. Pure function: `compute_regimes(klines, funding, liq) -> pd.Series[int8]`. No I/O. Tests use synthetic data.

**Files:**
- Create: `projects/candle-gpt/v2/data/regime.py`
- Create: `projects/candle-gpt/v2/tests/test_regime.py`

- [ ] **Step 1: Write the failing tests**

Path: `v2/tests/test_regime.py`

```python
"""Regime classifier: three-bucket taxonomy with priority high_vol > mean_revert > trend."""
import numpy as np
import pandas as pd
import pytest

from v2.data.regime import (
    REGIME_TREND,
    REGIME_MEAN_REVERT,
    REGIME_HIGH_VOL_SQUEEZE,
    REGIME_UNTAGGED,
    PERCENTILE_WINDOW_BARS,
    FUNDING_NEAR_ZERO,
    compute_regimes,
)
from v2.data.constants import (
    LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES,
    FUNDING_COLUMNS,
)


def _kline_frame(n: int, *, base: float = 100.0, step: float = 0.0) -> pd.DataFrame:
    """Synthetic 1m klines starting at t=0; close walks up by `step` per bar."""
    closes = base + step * np.arange(n)
    return pd.DataFrame({
        "open_time":  pd.array([i * 60_000 for i in range(n)], dtype="int64"),
        "open":       closes - 0.1,
        "high":       closes + 0.5,
        "low":        closes - 0.5,
        "close":      closes,
        "volume":     np.full(n, 10.0),
        "close_time": pd.array([i * 60_000 + 59_999 for i in range(n)], dtype="int64"),
        "regime":     pd.array([-1] * n, dtype="int8"),
    })


def _funding_frame(rate_per_event: list[float]) -> pd.DataFrame:
    n = len(rate_per_event)
    return pd.DataFrame({
        "funding_time": pd.array([i * 8 * 3_600_000 for i in range(n)], dtype="int64"),
        "funding_rate": rate_per_event,
        "mark_price":   [100.0] * n,
    })


def _empty_liq() -> pd.DataFrame:
    return pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })


def test_returns_int8_series_aligned_to_klines():
    n = PERCENTILE_WINDOW_BARS + 100
    klines = _kline_frame(n, step=0.05)
    funding = _funding_frame([0.0001] * 5)
    out = compute_regimes(klines, funding, _empty_liq())
    assert isinstance(out, pd.Series)
    assert out.dtype == np.int8
    assert len(out) == len(klines)


def test_pre_warmup_bars_get_untagged_sentinel():
    """Bars that lack PERCENTILE_WINDOW_BARS of trailing history default to UNTAGGED (-1)."""
    n = PERCENTILE_WINDOW_BARS // 2
    klines = _kline_frame(n, step=0.01)
    out = compute_regimes(klines, _funding_frame([0.0001] * 5), _empty_liq())
    assert (out == REGIME_UNTAGGED).all()


def test_trend_is_default_when_price_above_ma_and_funding_positive():
    n = PERCENTILE_WINDOW_BARS + 100
    klines = _kline_frame(n, base=100.0, step=0.10)  # uptrend
    # Persistent positive funding well above NEAR_ZERO.
    funding = _funding_frame([5 * FUNDING_NEAR_ZERO] * 50)
    out = compute_regimes(klines, funding, _empty_liq())
    # Last 50 bars (after warmup, well into the trend) should mostly be TREND.
    tail = out.iloc[-50:]
    assert (tail == REGIME_TREND).sum() >= 30


def test_high_vol_squeeze_beats_trend_when_funding_extreme():
    """Priority: a bar matching both trend AND high_vol gets HIGH_VOL_SQUEEZE."""
    n = PERCENTILE_WINDOW_BARS + 50
    klines = _kline_frame(n, step=0.05)
    # Funding spikes at the END so the latest bars sit at extreme percentile.
    rates = [0.00005] * 100 + [0.005] * 5
    funding = _funding_frame(rates)
    out = compute_regimes(klines, funding, _empty_liq())
    last = out.iloc[-1]
    assert last == REGIME_HIGH_VOL_SQUEEZE


def test_mean_revert_when_compressed_and_funding_near_zero():
    n = PERCENTILE_WINDOW_BARS + 200
    # First 1000 bars are volatile; last 200 are compressed (constant close, tight range).
    closes_volatile = 100.0 + 5.0 * np.sin(np.arange(PERCENTILE_WINDOW_BARS) / 10.0)
    closes_compressed = np.full(200, 100.0)
    closes = np.concatenate([closes_volatile, closes_compressed])
    klines = pd.DataFrame({
        "open_time":  pd.array([i * 60_000 for i in range(n)], dtype="int64"),
        "open":       closes,
        "high":       closes + 0.05,   # tight range in compressed phase
        "low":        closes - 0.05,
        "close":      closes,
        "volume":     np.full(n, 10.0),
        "close_time": pd.array([i * 60_000 + 59_999 for i in range(n)], dtype="int64"),
        "regime":     pd.array([-1] * n, dtype="int8"),
    })
    # Funding hovers near zero throughout.
    funding = _funding_frame([0.0] * 50)
    out = compute_regimes(klines, funding, _empty_liq())
    # The compressed tail should classify mean-revert in at least the latter half.
    tail = out.iloc[-100:]
    assert (tail == REGIME_MEAN_REVERT).sum() >= 50


def test_liq_spike_triggers_high_vol_squeeze():
    n = PERCENTILE_WINDOW_BARS + 50
    klines = _kline_frame(n, step=0.05)
    funding = _funding_frame([FUNDING_NEAR_ZERO * 0.5] * 50)  # benign funding
    # Liq spike at the LAST bar far above any reasonable trailing percentile.
    spike_bucket = klines["open_time"].iloc[-1]
    liq = pd.DataFrame({
        "bucket_time":         pd.array([spike_bucket], dtype="int64"),
        "count":               pd.array([5_000], dtype="int64"),
        "sum_notional":        [5_000_000.0],
        "max_single":          [1_000_000.0],
        "long_liq_count":      pd.array([5_000], dtype="int64"),
        "long_liq_notional":   [5_000_000.0],
        "short_liq_count":     pd.array([0], dtype="int64"),
        "short_liq_notional":  [0.0],
    })
    out = compute_regimes(klines, funding, liq)
    assert out.iloc[-1] == REGIME_HIGH_VOL_SQUEEZE
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest v2/tests/test_regime.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `regime.py`**

Path: `v2/data/regime.py`

```python
"""Three-bucket regime classifier (offline).

Buckets (and integer codes — see also tag_regimes CLI):
    0 = trend             — price > MA20 AND funding_rate > FUNDING_NEAR_ZERO
    1 = mean_revert       — BB width < 25th pct trailing 1000 bars
                            AND |funding_rate| < FUNDING_NEAR_ZERO
                            AND ATR14 < 25th pct trailing 1000 bars
    2 = high_vol_squeeze  — |funding_rate| > 95th pct trailing 720 funding obs
                            OR liq.count > 95th pct trailing 1000 bars
                            OR ATR14 > 90th pct trailing 1000 bars
   -1 = untagged          — insufficient trailing history (first
                            PERCENTILE_WINDOW_BARS bars of file)

Priority where multiple buckets match:
    high_vol_squeeze (2)  >  mean_revert (1)  >  trend (0)

Computed entirely from klines + funding + liq parquets — no model state.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


REGIME_TREND: int = 0
REGIME_MEAN_REVERT: int = 1
REGIME_HIGH_VOL_SQUEEZE: int = 2
REGIME_UNTAGGED: int = -1

# --- Thresholds (frozen for v2.0.0) -------------------------------------
MA_PERIOD: int = 20                # bars
BB_PERIOD: int = 20                # bars (k = 2 stddev)
BB_K: float = 2.0
ATR_PERIOD: int = 14               # bars
PERCENTILE_WINDOW_BARS: int = 1000  # trailing kline-bar window for BB / ATR / liq pcts
FUNDING_PERCENTILE_LOOKBACK: int = 720  # trailing funding observations (~240 days)

BB_WIDTH_LOW_PCTILE: float = 0.25
ATR_LOW_PCTILE: float = 0.25
ATR_HIGH_PCTILE: float = 0.90
FUNDING_EXTREME_PCTILE: float = 0.95
LIQ_COUNT_HIGH_PCTILE: float = 0.95

FUNDING_NEAR_ZERO: float = 1e-4    # |rate| < 0.01% per 8h interval


def _atr(klines: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = klines["high"]
    low = klines["low"]
    close_prev = klines["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bb_width(klines: pd.DataFrame, period: int = BB_PERIOD, k: float = BB_K) -> pd.Series:
    mid = klines["close"].rolling(period, min_periods=period).mean()
    sd = klines["close"].rolling(period, min_periods=period).std()
    return (2 * k * sd) / mid  # relative width


def _ffill_funding_to_klines(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Forward-fill funding_rate from funding observations to each kline open_time."""
    if funding.empty:
        return pd.Series(0.0, index=klines.index)
    f = funding[["funding_time", "funding_rate"]].sort_values("funding_time")
    merged = pd.merge_asof(
        klines[["open_time"]].sort_values("open_time"),
        f, left_on="open_time", right_on="funding_time", direction="backward",
    )
    return merged["funding_rate"].fillna(0.0).reset_index(drop=True)


def _liq_count_to_klines(klines: pd.DataFrame, liq: pd.DataFrame) -> pd.Series:
    """Join per-minute liq counts onto kline open_time. Empty liq → all zeros."""
    if liq.empty:
        return pd.Series(0, index=klines.index, dtype="int64")
    merged = klines[["open_time"]].merge(
        liq[["bucket_time", "count"]],
        left_on="open_time", right_on="bucket_time", how="left",
    )
    return merged["count"].fillna(0).astype("int64")


def _funding_extreme_per_bar(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Boolean per kline-bar: is the current funding_rate above the trailing
    FUNDING_EXTREME_PCTILE percentile of |rate| over the last
    FUNDING_PERCENTILE_LOOKBACK funding observations?
    """
    if funding.empty:
        return pd.Series(False, index=klines.index)
    f = funding.sort_values("funding_time").reset_index(drop=True)
    # Per-funding-event extreme flag.
    abs_rate = f["funding_rate"].abs()
    pctile = abs_rate.rolling(FUNDING_PERCENTILE_LOOKBACK, min_periods=FUNDING_PERCENTILE_LOOKBACK)\
                     .quantile(FUNDING_EXTREME_PCTILE)
    f_extreme = (abs_rate > pctile).fillna(False)
    f_lookup = pd.DataFrame({"funding_time": f["funding_time"], "extreme": f_extreme})
    merged = pd.merge_asof(
        klines[["open_time"]].sort_values("open_time"),
        f_lookup, left_on="open_time", right_on="funding_time", direction="backward",
    )
    return merged["extreme"].fillna(False).astype(bool).reset_index(drop=True)


def compute_regimes(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    liq: pd.DataFrame,
) -> pd.Series:
    """Classify each kline bar into one of {-1, 0, 1, 2}. See module docstring."""
    n = len(klines)
    out = np.full(n, REGIME_UNTAGGED, dtype=np.int8)

    if n < PERCENTILE_WINDOW_BARS:
        return pd.Series(out, index=klines.index, dtype="int8")

    close = klines["close"].reset_index(drop=True)
    ma = close.rolling(MA_PERIOD, min_periods=MA_PERIOD).mean()
    atr = _atr(klines.reset_index(drop=True))
    bbw = _bb_width(klines.reset_index(drop=True))

    bbw_low = bbw.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                 .quantile(BB_WIDTH_LOW_PCTILE)
    atr_low = atr.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                 .quantile(ATR_LOW_PCTILE)
    atr_high = atr.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                  .quantile(ATR_HIGH_PCTILE)

    funding_per_bar = _ffill_funding_to_klines(klines.reset_index(drop=True), funding)
    funding_extreme = _funding_extreme_per_bar(klines.reset_index(drop=True), funding)
    liq_count = _liq_count_to_klines(klines.reset_index(drop=True), liq)
    liq_high = liq_count.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                        .quantile(LIQ_COUNT_HIGH_PCTILE)

    is_compressed = bbw < bbw_low
    is_atr_low = atr < atr_low
    is_atr_high = atr > atr_high
    is_liq_spike = liq_count > liq_high
    funding_pos = funding_per_bar > FUNDING_NEAR_ZERO
    funding_near_zero = funding_per_bar.abs() < FUNDING_NEAR_ZERO

    high_vol = funding_extreme | is_liq_spike | is_atr_high
    mean_revert = is_compressed & funding_near_zero & is_atr_low
    trend = (close > ma) & funding_pos

    valid = (
        ~bbw_low.isna() & ~atr_low.isna() & ~atr_high.isna() & ~liq_high.isna()
        & ~ma.isna()
    )
    valid = valid.to_numpy()

    # Apply priority: high_vol > mean_revert > trend > default(trend).
    out_arr = np.where(high_vol.to_numpy(), REGIME_HIGH_VOL_SQUEEZE,
              np.where(mean_revert.to_numpy(), REGIME_MEAN_REVERT,
              np.where(trend.to_numpy(), REGIME_TREND,
                       REGIME_TREND)))  # default = trend (neutral baseline)
    out_arr = out_arr.astype(np.int8)
    out_arr[~valid] = REGIME_UNTAGGED
    return pd.Series(out_arr, index=klines.index, dtype="int8")
```

- [ ] **Step 4: Run tests; expect green**

```bash
uv run pytest v2/tests/test_regime.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/data/regime.py v2/tests/test_regime.py
git commit -m "v2: regime — 3-bucket classifier (trend / mean_revert / high_vol_squeeze)"
```

---

## Task 7: `tag_regimes` CLI — atomic in-place mutation of kline parquet

CLI: `python -m v2.data.tag_regimes --asset BTC --timeframe 1m`. Reads kline + funding + liq, computes regime, writes back to the SAME parquet path atomically (temp file in same dir, then `rename`). Tolerant of the Plan-1 untagged sentinel (`regime == -1`) on input.

**Files:**
- Create: `projects/candle-gpt/v2/data/tag_regimes.py`
- Create: `projects/candle-gpt/v2/tests/test_tag_regimes.py`

- [ ] **Step 1: Write the failing tests**

Path: `v2/tests/test_tag_regimes.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest v2/tests/test_tag_regimes.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tag_regimes.py`**

Path: `v2/data/tag_regimes.py`

```python
"""CLI: compute regime labels and write them back into the kline parquet, in place.

Usage:
    python -m v2.data.tag_regimes --asset BTC --timeframe 1m

Requires the funding and liq-bucketed parquets to exist at the canonical paths.
The liq parquet may be empty (Tardis-stub state) — that's fine; high_vol_squeeze
just won't trigger via liq spikes until a real fill drops in.

Atomicity: write_klines (in store.py) writes to <path>.tmp then rename()s. If the
process dies mid-write, the original parquet is intact.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from v2.data.constants import Asset, Timeframe
from v2.data.regime import compute_regimes
from v2.data.store import (
    parquet_path, funding_parquet_path, liq_bucketed_parquet_path,
    read_klines, read_funding, read_liq_bucketed, write_klines,
)


def tag_kline_parquet(*, asset: Asset, timeframe: Timeframe, root: Path) -> int:
    """Read the three parquets, compute regime, write klines back. Returns row count."""
    kp = parquet_path(root, asset, timeframe)
    fp = funding_parquet_path(root, asset)
    lp = liq_bucketed_parquet_path(root, asset)

    klines = read_klines(kp)
    funding = read_funding(fp)
    liq = read_liq_bucketed(lp)

    regimes = compute_regimes(klines, funding, liq)
    klines = klines.copy()
    klines["regime"] = regimes.astype("int8").to_numpy()
    write_klines(klines, kp)  # write_klines does atomic temp+rename internally
    return len(klines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag regime labels into a kline parquet in place.")
    ap.add_argument("--asset", required=True, choices=[a.value for a in Asset])
    ap.add_argument("--timeframe", required=True, choices=[t.value for t in Timeframe])
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "raw",
                    help="parquet root dir (default: v2/data/raw)")
    args = ap.parse_args()
    n = tag_kline_parquet(
        asset=Asset(args.asset),
        timeframe=Timeframe(args.timeframe),
        root=args.root,
    )
    print(f"Tagged {n:,} bars in {parquet_path(args.root, Asset(args.asset), Timeframe(args.timeframe))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests; expect green**

```bash
uv run pytest v2/tests/test_tag_regimes.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/data/tag_regimes.py v2/tests/test_tag_regimes.py
git commit -m "v2: tag_regimes — CLI for in-place regime column population"
```

---

## Task 8: Run regime tagging end-to-end on BTCUSDT 1m

The Plan-1 BTCUSDT 1m parquet currently has `regime == -1` (sentinel after the Task 1 fetcher modification re-fetched, OR — if the file pre-dates Task 1 — a manual one-time migration is needed; see Step 1 fallback). After this task, `regime ∈ {-1, 0, 1, 2}` with `-1` only on the first ~1000 bars.

- [ ] **Step 1: Ensure the kline parquet is at v2.0.0 (re-fetch if needed)**

If the Plan-1 fetch ran *before* Task 1, the on-disk parquet has 7 columns and no metadata — `read_klines` will refuse it with a clear `SchemaViolation` from `assert_schema_compatible`. In that case, re-fetch:

```bash
cd projects/candle-gpt
uv run python -m v2.data.fetch --asset BTCUSDT --timeframe 1m --days 1460
```

Verify the post-fetch state:

```bash
uv run python -c "
from pathlib import Path
from v2.data.store import read_klines
df = read_klines(Path('v2/data/raw/btcusdt_1m.parquet'))
print('rows:', len(df))
print('regime distribution before tagging:', df['regime'].value_counts().to_dict())
"
```

Expected: all rows have `regime == -1`.

- [ ] **Step 2: Run tag_regimes**

```bash
uv run python -m v2.data.tag_regimes --asset BTC --timeframe 1m
```

Expected stdout:
```
Tagged 2,100,000 bars in .../v2/data/raw/btcusdt_1m.parquet
```

(Or whatever the row count is.)

- [ ] **Step 3: Sanity-check the regime distribution**

```bash
uv run python -c "
from pathlib import Path
from v2.data.store import read_klines
df = read_klines(Path('v2/data/raw/btcusdt_1m.parquet'))
print('regime distribution:', df['regime'].value_counts().to_dict())
print('untagged-pct:', (df['regime'] == -1).mean())
print('trend-pct:   ', (df['regime'] == 0).mean())
print('mean_revert: ', (df['regime'] == 1).mean())
print('high_vol:    ', (df['regime'] == 2).mean())
"
```

Plausible distribution (rough guidance — don't fail on exact pcts):
- `-1` (untagged) ≈ `1000 / total_rows` ≈ 0.05%.
- `0` (trend) ≈ 50–80% — most bars in crypto are within trending or quiet drift.
- `1` (mean_revert) ≈ 5–25%.
- `2` (high_vol_squeeze) ≈ 5–15%.

If any bucket is 0% or >95%, the thresholds need a sanity-check before continuing.

- [ ] **Step 4: Marker commit**

```bash
echo "BTCUSDT 1m regime-tagged $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > v2/data/raw/REGIME_TAGGED.md
git add v2/data/raw/REGIME_TAGGED.md
git commit -m "v2: data — tagged BTCUSDT 1m regimes"
```

---

## Task 9: Dataloader extension — funding + liq join with `minutes_until_funding`

Extend `KlineWindowDataset` (Plan 1, `v2/data/dataset.py`) to optionally take `funding_path` and `liq_path`. When provided, the window tensor is widened from shape `(window, 8)` (kline cols + regime) to `(window, 17)` with the joined feature columns appended in **canonical order**.

**Joined columns appended (in this order, after the 8 kline cols):**
1. `funding_rate` — ffill from funding parquet; pre-first-funding bars get `0.0`.
2. `mark_price` — ffill; pre-first-funding bars get NaN-filled with the bar's `close` (sane proxy).
3. `minutes_until_funding` — `(next_funding_time - bar_open_time) / 60_000`; clamped to `[0, 480]`; bars after the last funding event get `480.0`.
4. `liq_count`, `liq_sum_notional`, `liq_max_single`, `long_liq_count`, `long_liq_notional`, `short_liq_count`, `short_liq_notional` — joined on `bucket_time == open_time`, missing → 0.

`minutes_until_funding` is the single non-trivial computed column the user explicitly called out as the **pre-squeeze signal** that must be in the canonical schema *now*, not retrofitted.

**Files:**
- Modify: `projects/candle-gpt/v2/data/dataset.py`
- Create: `projects/candle-gpt/v2/tests/test_dataloader_join.py`

- [x] **Step 1: Write the failing tests**

Path: `v2/tests/test_dataloader_join.py`

```python
"""Dataloader extension: funding + liq join + minutes_until_funding."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from v2.data.constants import (
    Asset, Timeframe, KLINE_COLUMNS, LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES,
)
from v2.data.dataset import KlineWindowDataset, FEATURE_COLUMNS_WITH_JOIN
from v2.data.store import (
    parquet_path, funding_parquet_path, liq_bucketed_parquet_path,
    write_klines, write_funding, write_liq_bucketed,
)


def _setup(tmp_path: Path, n_bars: int = 200) -> tuple[Path, Path, Path]:
    klines = pd.DataFrame({
        "open_time":  pd.array([i * 60_000 for i in range(n_bars)], dtype="int64"),
        "open":       np.arange(n_bars, dtype="float64"),
        "high":       np.arange(n_bars, dtype="float64") + 0.5,
        "low":        np.arange(n_bars, dtype="float64") - 0.5,
        "close":      np.arange(n_bars, dtype="float64") + 0.1,
        "volume":     np.full(n_bars, 10.0),
        "close_time": pd.array([i * 60_000 + 59_999 for i in range(n_bars)], dtype="int64"),
        "regime":     pd.array([0] * n_bars, dtype="int8"),
    })
    kp = parquet_path(tmp_path, Asset.BTC, Timeframe.M1)
    write_klines(klines, kp)

    # Funding events at t=0 and t=480min (8h apart).
    fdf = pd.DataFrame({
        "funding_time": pd.array([0, 480 * 60_000], dtype="int64"),
        "funding_rate": [0.0001, 0.0002],
        "mark_price":   [100.0, 101.0],
    })
    fp = funding_parquet_path(tmp_path, Asset.BTC)
    write_funding(fdf, fp)

    # One liq event at bar 50.
    liq = pd.DataFrame({
        "bucket_time":         pd.array([50 * 60_000], dtype="int64"),
        "count":               pd.array([3], dtype="int64"),
        "sum_notional":        [300.0],
        "max_single":          [200.0],
        "long_liq_count":      pd.array([2], dtype="int64"),
        "long_liq_notional":   [200.0],
        "short_liq_count":     pd.array([1], dtype="int64"),
        "short_liq_notional":  [100.0],
    })
    lp = liq_bucketed_parquet_path(tmp_path, Asset.BTC)
    write_liq_bucketed(liq, lp)
    return kp, fp, lp


def test_feature_columns_with_join_canonical_order():
    """Stable, documented column order — frozen for downstream feature-engineering."""
    assert FEATURE_COLUMNS_WITH_JOIN == (
        "open_time", "open", "high", "low", "close", "volume", "close_time", "regime",
        "funding_rate", "mark_price", "minutes_until_funding",
        "liq_count", "liq_sum_notional", "liq_max_single",
        "long_liq_count", "long_liq_notional",
        "short_liq_count", "short_liq_notional",
    )


def test_dataset_without_join_returns_8_cols(tmp_path: Path):
    kp, _, _ = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=10, stride=1)
    item = ds[0]
    assert item.shape == (10, len(KLINE_COLUMNS))


def test_dataset_with_join_returns_17_cols(tmp_path: Path):
    kp, fp, lp = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=10, stride=1, funding_path=fp, liq_path=lp)
    item = ds[0]
    assert item.shape == (10, len(FEATURE_COLUMNS_WITH_JOIN))
    assert item.dtype == torch.float32


def test_minutes_until_funding_decreases_then_resets(tmp_path: Path):
    kp, fp, lp = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=200, stride=1, funding_path=fp, liq_path=lp)
    item = ds[0].numpy()
    muf_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("minutes_until_funding")
    muf = item[:, muf_idx]
    # Bar 0 is exactly at funding time → 0; or with backward-direction match,
    # next_funding is at minute 480, so muf at bar 0 = 480.
    assert muf[0] == pytest.approx(480.0)
    # Bar 100 (100 min after t=0) → next funding at 480 → 380 min until.
    assert muf[100] == pytest.approx(380.0)
    # Bar 480 onward — past the second funding event, no future event → clamp to 480.
    # We only have 200 bars in this fixture; verify monotonic decrease in available range.
    assert muf[100] < muf[50]


def test_funding_rate_ffilled(tmp_path: Path):
    kp, fp, lp = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=200, stride=1, funding_path=fp, liq_path=lp)
    item = ds[0].numpy()
    fr_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("funding_rate")
    rates = item[:, fr_idx]
    # Bar 0 is at funding_time=0 → uses rate 0.0001.
    assert rates[0] == pytest.approx(0.0001)
    # Bars 1..479 still see the t=0 funding (no new event).
    assert rates[100] == pytest.approx(0.0001)


def test_liq_aggregates_zero_filled(tmp_path: Path):
    kp, fp, lp = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=200, stride=1, funding_path=fp, liq_path=lp)
    item = ds[0].numpy()
    cnt_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("liq_count")
    sum_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("liq_sum_notional")
    counts = item[:, cnt_idx]
    sums = item[:, sum_idx]
    assert counts[50] == pytest.approx(3.0)
    assert sums[50] == pytest.approx(300.0)
    assert counts[0] == 0.0
    assert sums[0] == 0.0


def test_dataloader_handles_empty_liq_parquet(tmp_path: Path):
    """Tardis-stub state: liq parquet has 0 rows. All liq cols → zeros."""
    kp, fp, lp = _setup(tmp_path)
    # Overwrite liq parquet with zero rows.
    empty = pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })
    write_liq_bucketed(empty, lp)
    ds = KlineWindowDataset(kp, window=10, stride=1, funding_path=fp, liq_path=lp)
    item = ds[0].numpy()
    cnt_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("liq_count")
    assert (item[:, cnt_idx] == 0).all()
```

- [x] **Step 2: Run tests to verify they fail**

```bash
uv run pytest v2/tests/test_dataloader_join.py -v
```

Expected: `ImportError: cannot import name 'FEATURE_COLUMNS_WITH_JOIN'` or test failures.

- [x] **Step 3: Modify `dataset.py`**

Path: `v2/data/dataset.py` — replace file body with:

```python
"""PyTorch Dataset over a (asset, timeframe) parquet, with optional feature joins.

When `funding_path` and/or `liq_path` are provided, the per-bar tensor is
widened with the joined feature columns in canonical order:

    [ ...kline cols (incl. regime), funding_rate, mark_price,
      minutes_until_funding, liq aggregates(7) ]
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from v2.data.constants import (
    KLINE_COLUMNS, LIQ_BUCKETED_COLUMNS,
)
from v2.data.store import read_klines, read_funding, read_liq_bucketed


# Canonical post-join feature row. Frozen for v2.0.0 — downstream feature
# engineering will index by name into this tuple.
FEATURE_COLUMNS_WITH_JOIN: tuple[str, ...] = KLINE_COLUMNS + (
    "funding_rate",
    "mark_price",
    "minutes_until_funding",
    "liq_count",
    "liq_sum_notional",
    "liq_max_single",
    "long_liq_count",
    "long_liq_notional",
    "short_liq_count",
    "short_liq_notional",
)

_MAX_MINUTES_UNTIL_FUNDING: float = 480.0  # 8h * 60min


def _join_features(
    klines: pd.DataFrame,
    funding: pd.DataFrame | None,
    liq: pd.DataFrame | None,
) -> pd.DataFrame:
    df = klines.copy()
    open_times = df["open_time"]

    # Funding ffill + minutes_until_funding.
    if funding is not None and not funding.empty:
        f = funding.sort_values("funding_time").reset_index(drop=True)
        merged = pd.merge_asof(
            df[["open_time"]].sort_values("open_time"),
            f, left_on="open_time", right_on="funding_time", direction="backward",
        )
        df["funding_rate"] = merged["funding_rate"].fillna(0.0).to_numpy()
        df["mark_price"] = merged["mark_price"].fillna(df["close"]).to_numpy()
        # Forward search to find next funding_time >= open_time.
        next_idx = np.searchsorted(f["funding_time"].to_numpy(), open_times.to_numpy(),
                                   side="left")
        next_t = np.where(
            next_idx < len(f),
            f["funding_time"].to_numpy()[np.clip(next_idx, 0, len(f) - 1)],
            -1,
        )
        muf = np.where(
            next_t >= 0,
            (next_t - open_times.to_numpy()) / 60_000.0,
            _MAX_MINUTES_UNTIL_FUNDING,
        )
        muf = np.clip(muf, 0.0, _MAX_MINUTES_UNTIL_FUNDING)
        df["minutes_until_funding"] = muf
    else:
        df["funding_rate"] = 0.0
        df["mark_price"] = df["close"].to_numpy()
        df["minutes_until_funding"] = _MAX_MINUTES_UNTIL_FUNDING

    # Liq aggregates joined on bucket_time == open_time.
    liq_cols_out = [
        "liq_count", "liq_sum_notional", "liq_max_single",
        "long_liq_count", "long_liq_notional",
        "short_liq_count", "short_liq_notional",
    ]
    if liq is not None and not liq.empty:
        liq_renamed = liq.rename(columns={
            "count": "liq_count",
            "sum_notional": "liq_sum_notional",
            "max_single": "liq_max_single",
        })[["bucket_time"] + liq_cols_out]
        joined = df.merge(liq_renamed, left_on="open_time", right_on="bucket_time",
                          how="left")
        for c in liq_cols_out:
            df[c] = joined[c].fillna(0).to_numpy()
        df = df.drop(columns=["bucket_time"], errors="ignore")
    else:
        for c in liq_cols_out:
            df[c] = 0.0

    return df[list(FEATURE_COLUMNS_WITH_JOIN)]


class KlineWindowDataset(Dataset):
    """Windowed access over a kline parquet, with optional funding+liq join."""

    def __init__(
        self,
        path: Path,
        window: int,
        stride: int = 1,
        *,
        funding_path: Path | None = None,
        liq_path: Path | None = None,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        df = read_klines(path)
        if len(df) < window:
            raise ValueError(
                f"window={window} larger than available bars={len(df)} in {path}"
            )

        funding = read_funding(funding_path) if funding_path is not None else None
        liq = read_liq_bucketed(liq_path) if liq_path is not None else None

        if funding is not None or liq is not None:
            features = _join_features(df, funding, liq)
            self._columns = FEATURE_COLUMNS_WITH_JOIN
        else:
            features = df[list(KLINE_COLUMNS)]
            self._columns = KLINE_COLUMNS

        self._bars = np.ascontiguousarray(features.to_numpy(dtype=np.float32))
        self._window = window
        self._stride = stride
        self._n_windows = (len(self._bars) - window) // stride + 1

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._n_windows:
            raise IndexError(idx)
        start = idx * self._stride
        return torch.from_numpy(self._bars[start : start + self._window])
```

- [x] **Step 4: Run new + existing dataset tests; expect green**

```bash
uv run pytest v2/tests/test_dataset.py v2/tests/test_dataloader_join.py -v
```

Expected: 8 (existing, after Task 1's fixture update) + 7 (new) = 15 passed.

- [x] **Step 5: Commit**

```bash
git add v2/data/dataset.py v2/tests/test_dataloader_join.py
git commit -m "v2: dataset — funding+liq join with minutes_until_funding"
```

---

## Task 10: Final verification — full milestone gate

- [x] **Step 1: Run the entire v2 test suite**

```bash
cd projects/candle-gpt
uv run pytest v2/tests/ -v
```

Expected (all should pass):
- Plan-1 tests: 31 (5 + 8 + 6 + 4 + 8).
- Schema versioning: 9.
- Funding fetcher: 5.
- Liq rollup: 6.
- Liq tardis stub: 2.
- Liq collect parser: 5.
- Regime: 6.
- tag_regimes: 4.
- Dataloader join: 7.

Approximate total: **75 passed** (drift ±2 acceptable depending on parametrize splits). 0 failures, 0 errors.

- [x] **Step 2: Confirm all on-disk parquets validate against expected hashes**

```bash
uv run python -c "
from pathlib import Path
from v2.data.constants import (
    KLINE_SCHEMA_HASH, FUNDING_SCHEMA_HASH, LIQ_BUCKETED_SCHEMA_HASH,
)
from v2.data.store import assert_schema_compatible
root = Path('v2/data/raw')
assert_schema_compatible(root / 'btcusdt_1m.parquet', KLINE_SCHEMA_HASH)
assert_schema_compatible(root / 'funding_btcusdt.parquet', FUNDING_SCHEMA_HASH)
assert_schema_compatible(root / 'liq_btcusdt_per_minute.parquet', LIQ_BUCKETED_SCHEMA_HASH)
print('OK — all parquets at v2.0.0 hashes')
"
```

Expected: `OK — all parquets at v2.0.0 hashes`.

- [x] **Step 3: Spot-check the joined dataloader on real data**

```bash
uv run python -c "
from pathlib import Path
from v2.data.constants import Asset, Timeframe
from v2.data.dataset import KlineWindowDataset, FEATURE_COLUMNS_WITH_JOIN
from v2.data.store import parquet_path, funding_parquet_path, liq_bucketed_parquet_path
root = Path('v2/data/raw')
kp = parquet_path(root, Asset.BTC, Timeframe.M1)
fp = funding_parquet_path(root, Asset.BTC)
lp = liq_bucketed_parquet_path(root, Asset.BTC)
ds = KlineWindowDataset(kp, window=512, stride=1, funding_path=fp, liq_path=lp)
print(f'len(ds)={len(ds):,}; window shape={tuple(ds[0].shape)}; cols={len(ds.columns)}')
print('first row joined cols:', dict(zip(FEATURE_COLUMNS_WITH_JOIN, ds[0][0].tolist())))
"
```

Expected:
- `len(ds)` ≈ 2,099,489 (rows minus 511).
- Window shape `(512, 17)`.
- Sample row shows real funding rate, real `minutes_until_funding ∈ [0, 480]`, real regime ∈ {0,1,2}, plausible OHLCV values.

- [x] **Step 4: Final commit + tag the milestone**

```bash
git tag v2-features-funding-liq-regime
git log --oneline -20
```

Expected: clean commit history; tag points at HEAD.

---

## What's next (out of scope for this plan)

- **Plan 2 (Day 3–4): Feature engineering.** Convert the 17-column joined window into the 41-dim feature vector (log returns, realized vol, ATR z-score, volume z-score, time-of-day sin/cos, regime one-hot, funding-rate ranks, asset/timeframe embeddings, etc.). The `FEATURE_COLUMNS_WITH_JOIN` tuple is the input contract.
- **Plan 3: Tardis backfill, real fill.** Wire up `fetch_tardis_backfill(...)` once a Tardis subscription is available. No schema migration required — the `LIQ_BUCKETED_SCHEMA_HASH` is identical for stub and real data.
- **Plan 4: Live-collector deployment.** systemd / launchd unit for `python -m v2.data.liquidations.collect`, plus a small `compact_dailies.py` cron job that rolls `v2/data/raw/liquidations/YYYY-MM-DD.parquet` files into the canonical per-minute parquet via `roll_to_per_minute`.
- **Plan N+1: L2/L3 order book depth.** Explicitly contingent on funding+liq features showing edge in walk-forward validation.

Multi-asset extension (ETH, SOL × 1m, 5m): re-run `fetch.py` + `funding.py` + `tag_regimes.py` per pair. The constants and code already accept all 6 cross-products; only the historical fetches are pending.
