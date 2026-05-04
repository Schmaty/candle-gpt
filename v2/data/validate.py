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


def assert_schema(
    df: pd.DataFrame,
    columns: tuple[str, ...] | None = None,
    dtypes: dict[str, str] | None = None,
) -> None:
    """Raise SchemaViolation if df doesn't match the given schema exactly.

    Defaults to the kline schema for backward compatibility with Plan-1 callers.
    """
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
