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
