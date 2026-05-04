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
