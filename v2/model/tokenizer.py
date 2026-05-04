"""ReturnTokenizerV2: quantile-based discretization of log-return values."""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np


class ReturnTokenizerV2:
    def __init__(self, n_bins: int = 256) -> None:
        self._n_bins = n_bins
        self.breakpoints: np.ndarray | None = None

    @property
    def n_bins(self) -> int:
        return self._n_bins

    def fit(self, log_returns: np.ndarray) -> None:
        data = np.asarray(log_returns, dtype=np.float64)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            raise ValueError("fit() received no finite values; all inputs were NaN or Inf.")
        q = np.linspace(0.0, 1.0, self._n_bins + 1)
        breaks = np.quantile(data, q)
        # Do NOT deduplicate or mutate _n_bins — that causes model vocab mismatch.
        # Duplicate breakpoints are handled correctly by np.digitize (ties go right).
        self.breakpoints = breaks

    def encode(self, log_returns: np.ndarray) -> np.ndarray:
        if self.breakpoints is None:
            raise RuntimeError("Tokenizer is not fitted; call fit() first.")
        data = np.asarray(log_returns, dtype=np.float64)
        ids = np.digitize(data, self.breakpoints[1:-1])
        return np.clip(ids, 0, self._n_bins - 1).astype(np.int64)

    def decode(self, bin_ids: np.ndarray) -> np.ndarray:
        if self.breakpoints is None:
            raise RuntimeError("Tokenizer is not fitted; call fit() first.")
        centers = (self.breakpoints[:-1] + self.breakpoints[1:]) / 2.0
        ids = np.clip(np.asarray(bin_ids, dtype=np.int64), 0, len(centers) - 1)
        return centers[ids]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "ReturnTokenizerV2":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ReturnTokenizerV2, got {type(obj)}")
        return obj
