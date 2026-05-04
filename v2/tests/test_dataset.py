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
