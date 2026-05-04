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
