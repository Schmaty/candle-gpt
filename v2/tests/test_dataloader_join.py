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
    ds = KlineWindowDataset(kp, window=10, stride=1, funding_path=fp, liq_path=lp, apply_features=False)
    item = ds[0]
    assert item.shape == (10, len(FEATURE_COLUMNS_WITH_JOIN))
    assert item.dtype == torch.float32


def test_minutes_until_funding_decreases_then_resets(tmp_path: Path):
    kp, fp, lp = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=200, stride=1, funding_path=fp, liq_path=lp, apply_features=False)
    item = ds[0].numpy()
    muf_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("minutes_until_funding")
    muf = item[:, muf_idx]
    # Bar 0 is exactly at funding time → next funding is at minute 480, so muf = 480.
    assert muf[0] == pytest.approx(480.0)
    # Bar 100 (100 min after t=0) → next funding at 480 → 380 min until.
    assert muf[100] == pytest.approx(380.0)
    # Verify monotonic decrease in available range.
    assert muf[100] < muf[50]


def test_funding_rate_ffilled(tmp_path: Path):
    kp, fp, lp = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=200, stride=1, funding_path=fp, liq_path=lp, apply_features=False)
    item = ds[0].numpy()
    fr_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("funding_rate")
    rates = item[:, fr_idx]
    # Bar 0 is at funding_time=0 → uses rate 0.0001.
    assert rates[0] == pytest.approx(0.0001)
    # Bars 1..479 still see the t=0 funding (no new event).
    assert rates[100] == pytest.approx(0.0001)


def test_liq_aggregates_zero_filled(tmp_path: Path):
    kp, fp, lp = _setup(tmp_path)
    ds = KlineWindowDataset(kp, window=200, stride=1, funding_path=fp, liq_path=lp, apply_features=False)
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
    ds = KlineWindowDataset(kp, window=10, stride=1, funding_path=fp, liq_path=lp, apply_features=False)
    item = ds[0].numpy()
    cnt_idx = list(FEATURE_COLUMNS_WITH_JOIN).index("liq_count")
    assert (item[:, cnt_idx] == 0).all()
