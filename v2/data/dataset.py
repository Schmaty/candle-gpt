"""PyTorch Dataset over a (asset, timeframe) parquet, with optional feature joins.

When `funding_path` and/or `liq_path` are provided, the per-bar tensor is
widened with the joined feature columns. When `apply_features=True` (default)
and a join was performed, the raw join columns are further transformed into
the engineered feature vector via v2.features.engineer.compute_features (see
v2.features.constants.FEATURE_COLUMNS for the canonical order).

`interval` selects the bar timeframe. The on-disk parquet is always 1m; the
dataset OHLCV-resamples on the fly to 5m / 15m / 1h / etc. Per-minute liq
buckets are sum-aggregated to the same timeframe so coarse bars do not lose
intra-bucket liquidation activity.

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

# Pandas resample rules. Keep this aligned with v2.data.constants.Timeframe and
# the inference-server INTERVAL_MS map. Adding a new entry here is sufficient
# to enable training on that timeframe (we OHLCV-resample 1m bars on the fly
# rather than fetching new parquets).
_RESAMPLE_RULES: dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}


def _resample_klines(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """OHLCV resample of a 1m kline DataFrame to a coarser timeframe.

    Buckets are left-labelled, left-closed (the 5m bar at minute T spans
    minutes [T, T+5)). Empty buckets (gaps in the 1m source) are dropped.
    """
    if interval == "1m":
        return df
    rule = _RESAMPLE_RULES.get(interval)
    if rule is None:
        raise ValueError(
            f"Unsupported interval {interval!r}; supported: {sorted(_RESAMPLE_RULES)}"
        )
    idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    indexed = df.set_index(idx)
    agg = indexed.resample(rule, label="left", closed="left").agg({
        "open_time":  "first",
        "open":       "first",
        "high":       "max",
        "low":        "min",
        "close":      "last",
        "volume":     "sum",
        "close_time": "last",
        "regime":     "last",
    }).dropna(subset=["close"]).reset_index(drop=True)
    agg["open_time"] = agg["open_time"].astype("int64")
    agg["close_time"] = agg["close_time"].astype("int64")
    agg["regime"] = agg["regime"].astype("int8")
    return agg


def _resample_liq(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Sum-aggregate per-minute liq buckets up to the requested timeframe.

    Without this, joining per-minute liq counts onto e.g. 5m klines would
    drop 4 of every 5 minutes of activity (only the bucket whose timestamp
    happens to land on a 5m boundary survives the exact-match join).
    """
    if interval == "1m" or df is None or df.empty:
        return df
    rule = _RESAMPLE_RULES.get(interval)
    if rule is None:
        raise ValueError(
            f"Unsupported interval {interval!r}; supported: {sorted(_RESAMPLE_RULES)}"
        )
    idx = pd.to_datetime(df["bucket_time"], unit="ms", utc=True)
    indexed = df.set_index(idx)
    agg = indexed.resample(rule, label="left", closed="left").agg({
        "bucket_time":         "first",
        "count":               "sum",
        "sum_notional":        "sum",
        "max_single":          "max",
        "long_liq_count":      "sum",
        "long_liq_notional":   "sum",
        "short_liq_count":     "sum",
        "short_liq_notional":  "sum",
    }).dropna(subset=["bucket_time"]).reset_index(drop=True)
    agg["bucket_time"] = agg["bucket_time"].astype("int64")
    for c in ("count", "long_liq_count", "short_liq_count"):
        agg[c] = agg[c].astype("int64")
    return agg


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
            the raw join columns into the engineered feature vector
            (FEATURE_COLUMNS). Has no effect when no join paths are provided.
        return_targets: If True, __getitem__ returns (features, log_returns)
            where log_returns[i] = log(close[i+1]/close[i]), 0.0 for last bar.
        interval: Bar timeframe. Defaults to "1m" (no resampling). Pass "5m"
            etc. to OHLCV-resample the 1m parquet on the fly.
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
        interval: str = "1m",
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        df = read_klines(path)
        df = _resample_klines(df, interval)
        if len(df) < window:
            raise ValueError(
                f"window={window} larger than available bars={len(df)} in {path}"
                f" (interval={interval})"
            )

        funding = read_funding(funding_path) if funding_path is not None else None
        liq = read_liq_bucketed(liq_path) if liq_path is not None else None
        liq = _resample_liq(liq, interval) if liq is not None else None

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
