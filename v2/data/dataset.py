"""PyTorch Dataset over a (asset, timeframe) parquet, with optional feature joins.

When `funding_path` and/or `liq_path` are provided, the per-bar tensor is
widened with the joined feature columns. When `apply_features=True` (default)
and a join was performed, the 18 raw join columns are further transformed into
the 41-dim engineered feature vector via v2.features.engineer.compute_features.

When `return_targets=True`, __getitem__ returns (features, log_returns) where
log_returns[i] = log(close[i+1] / close[i]), 0.0 for the final bar.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from v2.data.constants import KLINE_COLUMNS
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

    if funding is not None and not funding.empty:
        f = funding.sort_values("funding_time").reset_index(drop=True)
        merged = pd.merge_asof(
            df[["open_time"]].sort_values("open_time"),
            f, left_on="open_time", right_on="funding_time", direction="backward",
        )
        df["funding_rate"] = merged["funding_rate"].fillna(0.0).to_numpy()
        df["mark_price"] = merged["mark_price"].fillna(df["close"]).to_numpy()
        next_idx = np.searchsorted(f["funding_time"].to_numpy(), open_times.to_numpy(),
                                   side="right")
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
    """Windowed access over a kline parquet, with optional funding+liq join.

    Kwargs:
        apply_features: If True (default) AND a join was performed, transform
            the 18 raw join columns into the 41-dim engineered feature vector.
            Has no effect when no join paths are provided.
        return_targets: If True, __getitem__ returns (features, log_returns)
            where log_returns[i] = log(close[i+1]/close[i]), 0.0 for last bar.
    """

    def __init__(
        self,
        path: Path,
        window: int,
        stride: int = 1,
        *,
        funding_path: Path | None = None,
        liq_path: Path | None = None,
        apply_features: bool = True,
        return_targets: bool = False,
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
            joined = _join_features(df, funding, liq)
            if apply_features:
                from v2.features.engineer import compute_features
                from v2.features.constants import FEATURE_COLUMNS
                features = compute_features(joined)
                self._columns = FEATURE_COLUMNS
            else:
                features = joined
                self._columns = FEATURE_COLUMNS_WITH_JOIN
        else:
            features = df[list(KLINE_COLUMNS)]
            self._columns = KLINE_COLUMNS

        if return_targets:
            close_arr = df["close"].to_numpy(dtype=np.float64)
            log_returns = np.zeros(len(close_arr), dtype=np.float32)
            log_returns[:-1] = np.log(
                close_arr[1:] / np.maximum(close_arr[:-1], 1e-12)
            )
            self._log_returns = log_returns
        else:
            self._log_returns = None

        self._return_targets = return_targets
        self._bars = np.ascontiguousarray(features.to_numpy(dtype=np.float32))
        self._window = window
        self._stride = stride
        self._n_windows = (len(self._bars) - window) // stride + 1

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(
        self, idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._n_windows:
            raise IndexError(idx)
        start = idx * self._stride
        feats = torch.from_numpy(self._bars[start : start + self._window])
        if self._return_targets:
            assert self._log_returns is not None
            targets = torch.from_numpy(
                self._log_returns[start : start + self._window]
            )
            return feats, targets
        return feats
