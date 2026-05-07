# Candle-GPT v2 — Data Pipeline Implementation Plan (Plan 1 of N)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline for v2 — multi-asset, multi-timeframe Binance klines stored as parquet, validated for gaps/dupes, with a windowed PyTorch Dataset and a notebook to plot any window. Day 1–2 milestone scope: end-to-end working for ONE (asset, timeframe) pair (BTCUSDT 1m), but the abstractions must already accept (BTC|ETH|SOL, 1m|5m).

**Architecture:** New `v2/` subpackage inside `projects/candle-gpt/`, kept separate from v1 so the existing dashboard keeps running. Layers: `fetch` (download chunks) → `validate` (gap/dupe checks) → `store` (parquet schema enforcement) → `dataset` (PyTorch windowed access). All driven by a single canonical schema in `v2/data/constants.py`.

**Tech Stack:** Python 3.11+, `uv` for env, `pandas` + `pyarrow` for parquet, `requests` for Binance public REST API, `torch` for Dataset, `pytest` for tests, `matplotlib` + `jupyter` for the notebook.

**Scope notes:**
- This plan covers ONLY the data pipeline (Day 1–2 milestone).
- Subsequent plans will cover: feature engineering (Day 3–4), model code (Day 5), full training (Day 6–7), dashboard upgrade.
- Multi-asset coverage (ETH, SOL) and 5m timeframe are scoped here for the abstractions but only BTCUSDT 1m is fetched in this plan to keep the milestone tight; the same `fetch.py` CLI will be re-run for the other 5 pairs at the start of Plan 2.
- v1 code in `data/`, `model/`, `server/`, `web/` is **untouched**. v1 dashboard continues serving v1 model.

---

## File Structure

**To create:**
- `projects/candle-gpt/v2/__init__.py` — empty marker
- `projects/candle-gpt/v2/data/__init__.py` — empty marker
- `projects/candle-gpt/v2/data/constants.py` — `Asset`, `Timeframe` enums, `KLINE_SCHEMA`, interval→ms map, default 4-year history boundary
- `projects/candle-gpt/v2/data/validate.py` — `find_gaps(df, interval_ms)`, `dedupe_open_time(df)`, `assert_schema(df)`
- `projects/candle-gpt/v2/data/store.py` — `parquet_path(root, asset, timeframe)`, `write_klines(df, path)`, `read_klines(path)`
- `projects/candle-gpt/v2/data/fetch.py` — Binance kline fetcher, paginated, resume-friendly, CLI entry-point
- `projects/candle-gpt/v2/data/dataset.py` — `KlineWindowDataset` (PyTorch `Dataset`)
- `projects/candle-gpt/v2/data/raw/.gitkeep` — placeholder so `raw/` exists even when empty
- `projects/candle-gpt/v2/notebooks/01_data_explore.ipynb` — load a parquet, plot any window
- `projects/candle-gpt/v2/tests/__init__.py` — empty marker
- `projects/candle-gpt/v2/tests/test_constants.py`
- `projects/candle-gpt/v2/tests/test_validate.py`
- `projects/candle-gpt/v2/tests/test_store.py`
- `projects/candle-gpt/v2/tests/test_fetch.py`
- `projects/candle-gpt/v2/tests/test_dataset.py`

**To modify:**
- `projects/candle-gpt/pyproject.toml` — add dev deps: `pytest`, `matplotlib`, `jupyter`, `responses` (HTTP mocking)
- `projects/candle-gpt/.gitignore` (create if missing) — ignore `v2/data/raw/*.parquet`, `__pycache__`, `.ipynb_checkpoints`

**Not touched:**
- `projects/candle-gpt/data/` (v1)
- `projects/candle-gpt/model/` (v1)
- `projects/candle-gpt/server/` (v1)
- `projects/candle-gpt/web/` (v1)

---

## Task 1: Project skeleton, deps, and git baseline

**Files:**
- Create: `projects/candle-gpt/v2/__init__.py`
- Create: `projects/candle-gpt/v2/data/__init__.py`
- Create: `projects/candle-gpt/v2/data/raw/.gitkeep`
- Create: `projects/candle-gpt/v2/notebooks/.gitkeep`
- Create: `projects/candle-gpt/v2/tests/__init__.py`
- Create: `projects/candle-gpt/.gitignore`
- Modify: `projects/candle-gpt/pyproject.toml`

- [x] **Step 1: Initialize a git repo inside `projects/candle-gpt/`**

The workspace's outer git is empty and noisy. v2 work gets its own clean history.

```bash
cd /Users/kazkeller/.openclaw/workspace/projects/candle-gpt
git init
git config user.name "Kaz Keller"
git config user.email "kaz.keller20@gmail.com"
```

- [x] **Step 2: Write `.gitignore`**

Path: `projects/candle-gpt/.gitignore`

```
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
*.egg-info/

# Data — too large for git, regenerate via fetch.py
v2/data/raw/*.parquet

# v1 artifacts (kept on disk, not tracked)
checkpoints/
runs/
data/*.parquet

# Notebooks
.ipynb_checkpoints/

# OS
.DS_Store
```

- [x] **Step 3: Create empty package markers and `.gitkeep` files**

```bash
mkdir -p v2/data/raw v2/notebooks v2/tests
touch v2/__init__.py v2/data/__init__.py v2/tests/__init__.py
touch v2/data/raw/.gitkeep v2/notebooks/.gitkeep
```

- [x] **Step 4: Add dev dependencies via uv**

Run from `projects/candle-gpt/`:

```bash
uv add --dev pytest pytest-mock responses matplotlib jupyter jupytext
```

This updates `pyproject.toml` and `uv.lock`. Verify `pyproject.toml` now has a `[dependency-groups]` (or `[tool.uv.dev-dependencies]`) block listing those packages.

- [x] **Step 4b: Configure pytest to put the project root on `sys.path`**

Append the following block to `projects/candle-gpt/pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["v2/tests"]
```

Without this, `from v2.data.constants import Asset` fails to resolve when pytest runs.

- [x] **Step 5: Sanity-check the env**

```bash
uv run python -c "import torch, pandas, pyarrow, requests, pytest; print('ok')"
```

Expected: `ok`

- [x] **Step 6: Initial commit**

```bash
git add .gitignore pyproject.toml uv.lock v2/
git commit -m "v2: project skeleton + dev deps"
```

---

## Task 2: Constants module — assets, timeframes, schema

**Files:**
- Create: `projects/candle-gpt/v2/data/constants.py`
- Test: `projects/candle-gpt/v2/tests/test_constants.py`

- [x] **Step 1: Write the failing test**

Path: `v2/tests/test_constants.py`

```python
"""Constants must define the full v2 (asset, timeframe) cross-product cleanly."""
from v2.data.constants import (
    Asset,
    Timeframe,
    INTERVAL_MS,
    KLINE_COLUMNS,
    KLINE_DTYPES,
    DEFAULT_HISTORY_DAYS,
)


def test_assets_are_btc_eth_sol():
    assert {a.value for a in Asset} == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def test_timeframes_are_1m_and_5m():
    assert {t.value for t in Timeframe} == {"1m", "5m"}


def test_interval_ms_matches_timeframes():
    assert INTERVAL_MS[Timeframe.M1] == 60_000
    assert INTERVAL_MS[Timeframe.M5] == 300_000


def test_kline_columns_canonical_order():
    # open_time first (used as join/sort key), close_time last in the kept set
    assert KLINE_COLUMNS == (
        "open_time", "open", "high", "low", "close", "volume", "close_time",
    )


def test_kline_dtypes_match_columns():
    assert set(KLINE_DTYPES.keys()) == set(KLINE_COLUMNS)
    assert KLINE_DTYPES["open_time"] == "int64"
    assert KLINE_DTYPES["close"] == "float64"


def test_default_history_is_four_years():
    assert DEFAULT_HISTORY_DAYS == 4 * 365
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd projects/candle-gpt && uv run pytest v2/tests/test_constants.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'v2.data.constants'`

- [x] **Step 3: Implement `constants.py`**

Path: `v2/data/constants.py`

```python
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
```

- [x] **Step 4: Run test to verify it passes**

```bash
uv run pytest v2/tests/test_constants.py -v
```

Expected: 5 passed.

- [x] **Step 5: Commit**

```bash
git add v2/data/constants.py v2/tests/test_constants.py
git commit -m "v2: constants — Asset/Timeframe enums + canonical kline schema"
```

---

## Task 3: Validation — gap detection, dedup, schema check

**Files:**
- Create: `projects/candle-gpt/v2/data/validate.py`
- Test: `projects/candle-gpt/v2/tests/test_validate.py`

- [x] **Step 1: Write the failing test**

Path: `v2/tests/test_validate.py`

```python
"""Validation: gap detection, dedup, schema enforcement on kline DataFrames."""
import pandas as pd
import pytest

from v2.data.constants import KLINE_COLUMNS, KLINE_DTYPES, INTERVAL_MS, Timeframe
from v2.data.validate import (
    find_gaps,
    dedupe_open_time,
    assert_schema,
    SchemaViolation,
)


def _make_df(open_times_ms: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "open_time": pd.array(open_times_ms, dtype="int64"),
        "open":  [1.0] * len(open_times_ms),
        "high":  [1.0] * len(open_times_ms),
        "low":   [1.0] * len(open_times_ms),
        "close": [1.0] * len(open_times_ms),
        "volume":[1.0] * len(open_times_ms),
        "close_time": pd.array([t + 59_999 for t in open_times_ms], dtype="int64"),
    })


def test_find_gaps_returns_empty_when_contiguous():
    interval_ms = INTERVAL_MS[Timeframe.M1]
    df = _make_df([0, 60_000, 120_000, 180_000])
    assert find_gaps(df, interval_ms) == []


def test_find_gaps_reports_each_missing_run():
    interval_ms = INTERVAL_MS[Timeframe.M1]
    # missing 60_000; missing 180_000 and 240_000
    df = _make_df([0, 120_000, 300_000, 360_000])
    gaps = find_gaps(df, interval_ms)
    assert gaps == [
        (0, 120_000, 1),           # one missing bar between 0 and 120_000
        (120_000, 300_000, 2),     # two missing bars between 120_000 and 300_000
    ]


def test_dedupe_open_time_keeps_first():
    df = _make_df([0, 60_000, 60_000, 120_000])
    df.loc[2, "close"] = 999.0  # mark the duplicate
    out = dedupe_open_time(df)
    assert list(out["open_time"]) == [0, 60_000, 120_000]
    # the kept row at 60_000 is the FIRST one, not the duplicate with close=999
    assert out.loc[out["open_time"] == 60_000, "close"].iloc[0] == 1.0


def test_dedupe_returns_sorted():
    df = _make_df([120_000, 0, 60_000])
    out = dedupe_open_time(df)
    assert list(out["open_time"]) == [0, 60_000, 120_000]


def test_assert_schema_passes_on_valid_frame():
    df = _make_df([0, 60_000])
    assert_schema(df)  # should not raise


def test_assert_schema_rejects_missing_column():
    df = _make_df([0, 60_000]).drop(columns=["volume"])
    with pytest.raises(SchemaViolation, match="volume"):
        assert_schema(df)


def test_assert_schema_rejects_wrong_dtype():
    df = _make_df([0, 60_000])
    df["close"] = df["close"].astype("float32")
    with pytest.raises(SchemaViolation, match="close"):
        assert_schema(df)


def test_assert_schema_rejects_extra_column():
    df = _make_df([0, 60_000])
    df["unexpected"] = 1
    with pytest.raises(SchemaViolation, match="unexpected"):
        assert_schema(df)
```

- [x] **Step 2: Run test to verify it fails**

```bash
uv run pytest v2/tests/test_validate.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement `validate.py`**

Path: `v2/data/validate.py`

```python
"""Kline DataFrame validation: gaps, duplicates, schema."""
from __future__ import annotations
import pandas as pd

from v2.data.constants import KLINE_COLUMNS, KLINE_DTYPES


class SchemaViolation(ValueError):
    """Raised when a DataFrame does not match the canonical kline schema."""


def find_gaps(df: pd.DataFrame, interval_ms: int) -> list[tuple[int, int, int]]:
    """Return list of (prev_open_time, next_open_time, n_missing_bars).

    A "gap" is any consecutive pair of `open_time` values whose difference
    exceeds one interval. The DataFrame must already be sorted ascending.
    """
    if len(df) < 2:
        return []
    times = df["open_time"].to_numpy()
    diffs = times[1:] - times[:-1]
    gap_idx = (diffs > interval_ms).nonzero()[0]
    out: list[tuple[int, int, int]] = []
    for i in gap_idx:
        prev_t = int(times[i])
        next_t = int(times[i + 1])
        n_missing = (next_t - prev_t) // interval_ms - 1
        out.append((prev_t, next_t, int(n_missing)))
    return out


def dedupe_open_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by open_time ascending and drop duplicate open_time rows (keep first)."""
    return (
        df.sort_values("open_time", kind="mergesort")
          .drop_duplicates(subset=["open_time"], keep="first")
          .reset_index(drop=True)
    )


def assert_schema(df: pd.DataFrame) -> None:
    """Raise SchemaViolation if df doesn't match the canonical kline schema exactly."""
    actual_cols = tuple(df.columns)
    if actual_cols != KLINE_COLUMNS:
        missing = set(KLINE_COLUMNS) - set(actual_cols)
        extra = set(actual_cols) - set(KLINE_COLUMNS)
        problems: list[str] = []
        if missing:
            problems.append(f"missing={sorted(missing)}")
        if extra:
            problems.append(f"unexpected={sorted(extra)}")
        if not problems:
            problems.append(f"wrong_order: got {actual_cols}, want {KLINE_COLUMNS}")
        raise SchemaViolation("kline schema mismatch: " + "; ".join(problems))

    for col, want_dtype in KLINE_DTYPES.items():
        got_dtype = str(df[col].dtype)
        if got_dtype != want_dtype:
            raise SchemaViolation(
                f"kline schema mismatch: column {col} dtype={got_dtype}, want {want_dtype}"
            )
```

- [x] **Step 4: Run tests to verify they pass**

```bash
uv run pytest v2/tests/test_validate.py -v
```

Expected: 8 passed.

- [x] **Step 5: Commit**

```bash
git add v2/data/validate.py v2/tests/test_validate.py
git commit -m "v2: validate — gap detection, dedup, schema enforcement"
```

---

## Task 4: Store — parquet read/write with schema enforcement

**Files:**
- Create: `projects/candle-gpt/v2/data/store.py`
- Test: `projects/candle-gpt/v2/tests/test_store.py`

- [x] **Step 1: Write the failing test**

Path: `v2/tests/test_store.py`

```python
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
```

- [x] **Step 2: Run test to verify it fails**

```bash
uv run pytest v2/tests/test_store.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement `store.py`**

Path: `v2/data/store.py`

```python
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
    df = df.reindex(columns=list(KLINE_COLUMNS), copy=False)
    assert_schema(df)
    return df
```

- [x] **Step 4: Run tests to verify they pass**

```bash
uv run pytest v2/tests/test_store.py -v
```

Expected: 6 passed.

- [x] **Step 5: Commit**

```bash
git add v2/data/store.py v2/tests/test_store.py
git commit -m "v2: store — parquet read/write with schema enforcement"
```

---

## Task 5: Fetch — Binance kline downloader (CLI + library)

**Files:**
- Create: `projects/candle-gpt/v2/data/fetch.py`
- Test: `projects/candle-gpt/v2/tests/test_fetch.py`

This rewrite of v1's `fetch_binance.py` does three things v1 didn't: (1) takes the asset/timeframe as `Asset`/`Timeframe` enums, (2) writes through the schema-enforcing `store.write_klines`, (3) is unit-testable via mocked HTTP.

- [x] **Step 1: Write the failing tests**

Path: `v2/tests/test_fetch.py`

```python
"""Fetcher pagination, dedup, and stop-condition logic, with HTTP mocked."""
from pathlib import Path

import pandas as pd
import pytest
import responses

from v2.data.constants import Asset, Timeframe
from v2.data.fetch import (
    BINANCE_KLINES_URL,
    chunk_to_rows,
    fetch_to_parquet,
)


def _kline(open_ms: int, interval_ms: int = 60_000) -> list:
    """Build one Binance-shaped kline row."""
    return [
        open_ms,
        "1.0", "1.2", "0.9", "1.1", "10.0",
        open_ms + interval_ms - 1,
        "11.0", 5, "5.0", "5.5", "0",
    ]


def test_chunk_to_rows_keeps_only_canonical_columns():
    rows = [_kline(0), _kline(60_000)]
    df = chunk_to_rows(rows)
    assert list(df.columns) == [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
    ]
    assert df["open_time"].tolist() == [0, 60_000]
    assert df["close"].tolist() == [1.1, 1.1]


@responses.activate
def test_fetch_walks_backward_until_target_start(tmp_path: Path):
    # Three pages, each with 2 candles, working backward from now.
    # Page 1 (newest): open_times 240_000, 300_000
    # Page 2:          open_times 120_000, 180_000
    # Page 3 (oldest): open_times 0, 60_000     <-- target_start_ms = 0, stop here
    page1 = [_kline(240_000), _kline(300_000)]
    page2 = [_kline(120_000), _kline(180_000)]
    page3 = [_kline(0), _kline(60_000)]
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page3, status=200)

    out_path = tmp_path / "btcusdt_1m.parquet"
    df = fetch_to_parquet(
        asset=Asset.BTC,
        timeframe=Timeframe.M1,
        end_ms=300_000 + 60_000,  # "now" is just after the newest bar
        target_start_ms=0,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["open_time"].tolist() == [0, 60_000, 120_000, 180_000, 240_000, 300_000]
    assert out_path.exists()


@responses.activate
def test_fetch_stops_on_empty_chunk(tmp_path: Path):
    responses.add(responses.GET, BINANCE_KLINES_URL, json=[_kline(60_000)], status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=[], status=200)
    out_path = tmp_path / "btcusdt_1m.parquet"
    df = fetch_to_parquet(
        asset=Asset.BTC,
        timeframe=Timeframe.M1,
        end_ms=120_000,
        target_start_ms=0,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["open_time"].tolist() == [60_000]


@responses.activate
def test_fetch_dedupes_repeated_open_times(tmp_path: Path):
    # Binance occasionally returns overlapping pages; fetcher must dedupe.
    page1 = [_kline(120_000), _kline(180_000)]
    page2 = [_kline(60_000), _kline(120_000)]  # 120_000 repeated
    page3 = [_kline(0), _kline(60_000)]        # 60_000 repeated
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page1, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page2, status=200)
    responses.add(responses.GET, BINANCE_KLINES_URL, json=page3, status=200)
    out_path = tmp_path / "btcusdt_1m.parquet"
    df = fetch_to_parquet(
        asset=Asset.BTC,
        timeframe=Timeframe.M1,
        end_ms=240_000,
        target_start_ms=0,
        out_path=out_path,
        sleep_s=0,
    )
    assert df["open_time"].tolist() == [0, 60_000, 120_000, 180_000]
```

- [x] **Step 2: Run test to verify it fails**

```bash
uv run pytest v2/tests/test_fetch.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement `fetch.py`**

Path: `v2/data/fetch.py`

```python
"""Binance public kline fetcher.

Walks backward in time, MAX_LIMIT bars per request, until either:
  - target_start_ms reached, or
  - Binance returns an empty page (start of available history).

All writes go through `store.write_klines`, which enforces the canonical schema.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from v2.data.constants import (
    Asset,
    Timeframe,
    INTERVAL_MS,
    KLINE_COLUMNS,
    KLINE_DTYPES,
    DEFAULT_HISTORY_DAYS,
)
from v2.data.store import parquet_path, write_klines
from v2.data.validate import dedupe_open_time

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance hard cap per request

# Binance returns 12 fields per kline; we keep 7.
_RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def chunk_to_rows(chunk: list[list]) -> pd.DataFrame:
    """Convert a raw Binance kline payload into a canonical-schema DataFrame."""
    df = pd.DataFrame(chunk, columns=_RAW_COLUMNS)
    df = df[list(KLINE_COLUMNS)].copy()
    for col, dtype in KLINE_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def _fetch_chunk(symbol: str, interval: str, end_ms: int, limit: int = MAX_LIMIT) -> list[list]:
    params = {"symbol": symbol, "interval": interval, "endTime": end_ms, "limit": limit}
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_to_parquet(
    asset: Asset,
    timeframe: Timeframe,
    end_ms: int,
    target_start_ms: int,
    out_path: Path,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """Fetch klines from `target_start_ms` to `end_ms` and write to `out_path`."""
    all_rows: list[pd.DataFrame] = []
    seen: set[int] = set()
    cursor_end = end_ms
    while True:
        chunk = _fetch_chunk(asset.value, timeframe.value, cursor_end)
        if not chunk:
            break
        df_chunk = chunk_to_rows(chunk)
        new_mask = ~df_chunk["open_time"].isin(seen)
        if not new_mask.any():
            break
        new_df = df_chunk[new_mask]
        seen.update(int(t) for t in new_df["open_time"].tolist())
        all_rows.append(new_df)
        oldest_open = int(new_df["open_time"].min())
        if oldest_open <= target_start_ms:
            break
        cursor_end = oldest_open - 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not all_rows:
        raise RuntimeError(
            f"No data returned for {asset.value} {timeframe.value} in window "
            f"[{target_start_ms}, {end_ms}]"
        )
    df = pd.concat(all_rows, ignore_index=True)
    df = dedupe_open_time(df)
    write_klines(df, out_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Binance klines for a (asset, timeframe) pair.")
    ap.add_argument("--asset", required=True, choices=[a.value for a in Asset])
    ap.add_argument("--timeframe", required=True, choices=[t.value for t in Timeframe])
    ap.add_argument("--days", type=int, default=DEFAULT_HISTORY_DAYS,
                    help=f"history depth in days (default: {DEFAULT_HISTORY_DAYS})")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent / "raw",
                    help="output dir (parquet path = root / <symbol>_<interval>.parquet)")
    args = ap.parse_args()

    asset = Asset(args.asset)
    timeframe = Timeframe(args.timeframe)
    out_path = parquet_path(args.root, asset, timeframe)

    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - args.days * 24 * 60 * 60 * 1000

    print(f"Fetching {asset.value} {timeframe.value} for {args.days} days → {out_path}")
    df = fetch_to_parquet(
        asset=asset,
        timeframe=timeframe,
        end_ms=now_ms,
        target_start_ms=target_start_ms,
        out_path=out_path,
    )
    print(f"Done. {len(df)} bars saved.")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run tests to verify they pass**

```bash
uv run pytest v2/tests/test_fetch.py -v
```

Expected: 4 passed.

- [x] **Step 5: Commit**

```bash
git add v2/data/fetch.py v2/tests/test_fetch.py
git commit -m "v2: fetch — Binance kline downloader (paginated, dedup, schema-checked)"
```

---

## Task 6: Run fetch end-to-end for BTCUSDT 1m (4 years)

This task is a real network operation, not a unit test. Treat it as one shot. If it fails partway, the script is idempotent — re-running with the same args overwrites the parquet from scratch. (We'll add resume support in a later plan if it proves needed.)

- [ ] **Step 1: Run the fetch**

```bash
cd projects/candle-gpt
uv run python -m v2.data.fetch --asset BTCUSDT --timeframe 1m --days 1460
```

Expected runtime: ~10 min on a residential connection (4 years × ~525,600 bars/year ÷ 1000 bars/req × 0.05 s/req ≈ 105 s minimum, plus network latency typically 5–10× that).

Expected stdout:
```
Fetching BTCUSDT 1m for 1460 days → .../v2/data/raw/btcusdt_1m.parquet
Done. ~2_100_000 bars saved.
```

- [ ] **Step 2: Sanity-check the output**

```bash
uv run python -c "
from pathlib import Path
import pandas as pd
from v2.data.constants import INTERVAL_MS, Timeframe
from v2.data.store import read_klines
from v2.data.validate import find_gaps

p = Path('v2/data/raw/btcusdt_1m.parquet')
df = read_klines(p)
print(f'rows={len(df):,}')
print(f'first={pd.Timestamp(df.open_time.iloc[0], unit=\"ms\", tz=\"UTC\")}')
print(f'last ={pd.Timestamp(df.open_time.iloc[-1], unit=\"ms\", tz=\"UTC\")}')
gaps = find_gaps(df, INTERVAL_MS[Timeframe.M1])
print(f'gaps={len(gaps)} (sum_missing={sum(g[2] for g in gaps)} bars)')
"
```

Expected:
- `rows` between 2,000,000 and 2,200,000.
- `first` ≈ 4 years before today (e.g. 2022-05).
- `last` within the past few minutes.
- Some gaps are normal (Binance maintenance windows), but `sum_missing / rows` should be < 0.5%. If it's higher, dig in before moving on.

- [ ] **Step 3: Commit the validation script (NOT the parquet — it's gitignored)**

If you saved the snippet to a script (optional), commit it. Otherwise just commit a marker note:

```bash
echo "BTCUSDT 1m fetched $(date -u +%Y-%m-%dT%H:%M:%SZ)" > v2/data/raw/FETCHED.md
git add v2/data/raw/FETCHED.md
git commit -m "v2: data — fetched BTCUSDT 1m, 4 years"
```

---

## Task 7: Dataset — windowed PyTorch access

**Files:**
- Create: `projects/candle-gpt/v2/data/dataset.py`
- Test: `projects/candle-gpt/v2/tests/test_dataset.py`

For Day 1–2 the Dataset returns *raw* OHLCV windows (shape `(window, 7)`, the canonical kline columns). Feature engineering (the 41-dim vector) is a separate layer in the next plan.

- [x] **Step 1: Write the failing tests**

Path: `v2/tests/test_dataset.py`

```python
"""KlineWindowDataset: windowed access semantics over a single parquet file."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from v2.data.constants import Asset, Timeframe, KLINE_COLUMNS
from v2.data.dataset import KlineWindowDataset
from v2.data.store import parquet_path, write_klines


def _write_synthetic(tmp_path: Path, n_bars: int) -> Path:
    df = pd.DataFrame({
        "open_time":  pd.array([i * 60_000 for i in range(n_bars)], dtype="int64"),
        "open":       np.arange(n_bars, dtype="float64"),
        "high":       np.arange(n_bars, dtype="float64") + 0.5,
        "low":        np.arange(n_bars, dtype="float64") - 0.5,
        "close":      np.arange(n_bars, dtype="float64") + 0.1,
        "volume":     np.arange(n_bars, dtype="float64") * 10.0,
        "close_time": pd.array([i * 60_000 + 59_999 for i in range(n_bars)], dtype="int64"),
    })
    p = parquet_path(tmp_path, Asset.BTC, Timeframe.M1)
    write_klines(df, p)
    return p


def test_dataset_length_with_default_stride(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=100)
    ds = KlineWindowDataset(p, window=10, stride=1)
    # 100 bars, window 10, stride 1 → 91 windows
    assert len(ds) == 91


def test_dataset_length_with_stride(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=100)
    ds = KlineWindowDataset(p, window=10, stride=5)
    # 100 bars, window 10, stride 5 → floor((100-10)/5)+1 = 19
    assert len(ds) == 19


def test_dataset_returns_correct_shape_and_dtype(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=100)
    ds = KlineWindowDataset(p, window=10, stride=1)
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (10, len(KLINE_COLUMNS))
    assert item.dtype == torch.float32


def test_dataset_first_window_starts_at_bar_zero(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=100)
    ds = KlineWindowDataset(p, window=10, stride=1)
    item = ds[0]
    # Column index 1 is "open"; synthetic data has open[i]=i.
    open_col_idx = list(KLINE_COLUMNS).index("open")
    assert item[:, open_col_idx].tolist() == list(range(10))


def test_dataset_index_into_middle_window(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=100)
    ds = KlineWindowDataset(p, window=10, stride=5)
    item = ds[3]  # 3rd window @ stride 5 starts at bar 15
    open_col_idx = list(KLINE_COLUMNS).index("open")
    assert item[:, open_col_idx].tolist() == list(range(15, 25))


def test_dataset_last_window_does_not_overflow(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=100)
    ds = KlineWindowDataset(p, window=10, stride=1)
    item = ds[len(ds) - 1]
    open_col_idx = list(KLINE_COLUMNS).index("open")
    assert item[:, open_col_idx].tolist() == list(range(90, 100))


def test_dataset_out_of_range_index_raises(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=100)
    ds = KlineWindowDataset(p, window=10, stride=1)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_dataset_rejects_window_larger_than_data(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=5)
    with pytest.raises(ValueError, match="window"):
        KlineWindowDataset(p, window=10, stride=1)
```

- [x] **Step 2: Run test to verify it fails**

```bash
uv run pytest v2/tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement `dataset.py`**

Path: `v2/data/dataset.py`

```python
"""PyTorch Dataset over a single (asset, timeframe) parquet file.

Returns raw OHLCV-shaped windows. Feature engineering happens in a separate
module (next plan) — this layer's job is just bar-array slicing.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from v2.data.constants import KLINE_COLUMNS
from v2.data.store import read_klines


class KlineWindowDataset(Dataset):
    def __init__(self, path: Path, window: int, stride: int = 1) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        df = read_klines(path)
        if len(df) < window:
            raise ValueError(
                f"window={window} larger than available bars={len(df)} in {path}"
            )
        # Materialize once as a contiguous float32 array for fast slicing.
        self._bars = np.ascontiguousarray(
            df[list(KLINE_COLUMNS)].to_numpy(dtype=np.float32)
        )
        self._window = window
        self._stride = stride
        self._n_windows = (len(self._bars) - window) // stride + 1

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._n_windows:
            raise IndexError(idx)
        start = idx * self._stride
        return torch.from_numpy(self._bars[start : start + self._window])
```

- [x] **Step 4: Run tests to verify they pass**

```bash
uv run pytest v2/tests/test_dataset.py -v
```

Expected: 8 passed.

- [x] **Step 5: Run the whole test suite**

```bash
uv run pytest v2/tests/ -v
```

Expected: 31 passed (5 + 8 + 6 + 4 + 8).

- [x] **Step 6: Commit**

```bash
git add v2/data/dataset.py v2/tests/test_dataset.py
git commit -m "v2: dataset — KlineWindowDataset for windowed PyTorch access"
```

---

## Task 8: Notebook — plot any window of BTCUSDT 1m

**Files:**
- Create: `projects/candle-gpt/v2/notebooks/01_data_explore.ipynb`

The notebook is the Day 1–2 milestone's "plot any window" deliverable. Build it as a Python script first, then convert to `.ipynb` so it's deterministic and re-runnable.

- [x] **Step 1: Write the source script**

Path: `projects/candle-gpt/v2/notebooks/_01_data_explore.py` (temporary, deleted after conversion)

```python
# %% [markdown]
# # Candle-GPT v2 — Data Exploration
#
# Loads BTCUSDT 1m from the v2 raw store and plots an arbitrary window.

# %%
import sys
from pathlib import Path

# When run via `nbconvert --execute`, cwd is the notebook's directory
# (`v2/notebooks/`); the project root (which contains the `v2/` package)
# is two levels up. When run interactively from project root, cwd is
# already correct. Handle both.
_HERE = Path.cwd()
_PROJECT_ROOT = _HERE.parent.parent if _HERE.name == "notebooks" else _HERE
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from v2.data.constants import Asset, Timeframe, KLINE_COLUMNS
from v2.data.store import parquet_path, read_klines
from v2.data.dataset import KlineWindowDataset

ROOT = _PROJECT_ROOT / "v2" / "data" / "raw"
ASSET = Asset.BTC
TIMEFRAME = Timeframe.M1

# %%
path = parquet_path(ROOT, ASSET, TIMEFRAME)
df = read_klines(path)
print(f"loaded {len(df):,} bars from {path}")
print(f"first: {pd.Timestamp(df.open_time.iloc[0], unit='ms', tz='UTC')}")
print(f"last:  {pd.Timestamp(df.open_time.iloc[-1], unit='ms', tz='UTC')}")

# %% [markdown]
# ## Plot any window
#
# Set `start_idx` and `window_len` to whatever you want.

# %%
start_idx = len(df) - 1024  # last 1024 bars by default
window_len = 1024

window = df.iloc[start_idx : start_idx + window_len]
ts = pd.to_datetime(window["open_time"], unit="ms", utc=True)

fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
ax_price.plot(ts, window["close"], lw=0.8)
ax_price.fill_between(ts, window["low"], window["high"], alpha=0.15, lw=0)
ax_price.set_ylabel(f"{ASSET.value} close")
ax_price.set_title(f"{ASSET.value} {TIMEFRAME.value} — bars [{start_idx}:{start_idx + window_len}]")
ax_vol.bar(ts, window["volume"], width=4e-4)
ax_vol.set_ylabel("volume")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Same window via `KlineWindowDataset`
#
# Verifies the Dataset returns the same bars we just plotted.

# %%
ds = KlineWindowDataset(path, window=window_len, stride=1)
ds_idx = start_idx  # stride=1 so dataset index == bar index
tensor = ds[ds_idx]
print(f"tensor shape: {tuple(tensor.shape)}")
close_col = list(KLINE_COLUMNS).index("close")
assert (tensor[:, close_col].numpy() == window["close"].to_numpy()).all(), \
    "Dataset slice disagrees with raw DataFrame slice"
print("Dataset and raw DataFrame match.")
```

- [ ] **Step 2: Convert to `.ipynb` and execute**

`jupytext` was already added in Task 1 Step 4. Convert and execute:

```bash
uv run jupytext --to ipynb v2/notebooks/_01_data_explore.py -o v2/notebooks/01_data_explore.ipynb
uv run jupyter nbconvert --to notebook --execute v2/notebooks/01_data_explore.ipynb --output 01_data_explore.ipynb
rm v2/notebooks/_01_data_explore.py
```

Expected: the notebook executes top-to-bottom without error and the plot cell produces an embedded PNG. If `jupytext` complains, the fallback is to open `_01_data_explore.py` in Jupyter Lab, which understands the `# %%` cell markers natively, save as `.ipynb`, then run "Restart & Run All".

- [ ] **Step 3: Verify the notebook runs cleanly**

```bash
uv run jupyter nbconvert --to notebook --execute v2/notebooks/01_data_explore.ipynb --output 01_data_explore.ipynb
```

Expected: exit code 0, no traceback in any cell. If a cell errored, the conversion will fail loudly.

- [ ] **Step 4: Commit**

```bash
git add v2/notebooks/01_data_explore.ipynb pyproject.toml uv.lock
git commit -m "v2: notebook — explore + plot any window"
```

---

## Task 9: Final verification — full milestone gate

- [ ] **Step 1: Run the entire test suite**

```bash
cd projects/candle-gpt
uv run pytest v2/tests/ -v
```

Expected: 31 passed, 0 failed.

- [ ] **Step 2: Confirm parquet exists and is well-formed**

```bash
uv run python -c "
from pathlib import Path
from v2.data.store import read_klines
df = read_klines(Path('v2/data/raw/btcusdt_1m.parquet'))
assert len(df) > 2_000_000, f'too few bars: {len(df)}'
print(f'OK — {len(df):,} bars')
"
```

- [ ] **Step 3: Confirm notebook runs end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute v2/notebooks/01_data_explore.ipynb --output 01_data_explore.ipynb
```

Expected: exit code 0.

- [ ] **Step 4: Final commit + tag the milestone**

```bash
git tag v2-day1-2-data-pipeline
git log --oneline
```

Expected: clean commit history, ~9 commits, tag points at HEAD.

---

## What's next (out of scope for this plan)

Plan 2 will cover Day 3–4: **feature engineering** — turning the raw 7-column kline windows into the 41-dim feature vectors specified in the v2 spec (log returns, realized vol, ATR, volume z-score, time-of-day sin/cos, asset/timeframe embeddings).

Before starting Plan 2, I'll need to re-run `fetch.py` for the other 5 (asset, timeframe) pairs:

```bash
for asset in BTCUSDT ETHUSDT SOLUSDT; do
  for tf in 1m 5m; do
    [ "$asset" = "BTCUSDT" ] && [ "$tf" = "1m" ] && continue
    uv run python -m v2.data.fetch --asset $asset --timeframe $tf --days 1460
  done
done
```

That ~50 minutes of fetching is a fine background task; we can kick it off at the start of Plan 2.
