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
