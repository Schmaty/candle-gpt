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
        "regime":     pd.array([-1] * n_bars, dtype="int8"),
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


# ---- New tests for apply_features and return_targets (Plan 2) ----

from v2.data.store import write_funding, write_liq_bucketed
from v2.features.constants import N_FEATURES


def _write_synthetic_funding(tmp_path: Path, n_events: int = 10, start_ms: int = 0) -> Path:
    import pandas as pd
    interval_ms = 8 * 60 * 60 * 1000  # 8h
    df = pd.DataFrame({
        "funding_time": pd.array(
            [start_ms + i * interval_ms for i in range(n_events)], dtype="int64"
        ),
        "funding_rate": [0.0001] * n_events,
        "mark_price":   [100.0] * n_events,
    })
    p = tmp_path / "funding_btcusdt.parquet"
    write_funding(df, p)
    return p


def _write_synthetic_liq(tmp_path: Path, n_bars: int, start_ms: int = 0) -> Path:
    import pandas as pd
    df = pd.DataFrame({
        "bucket_time":       pd.array([start_ms + i * 60_000 for i in range(n_bars)], dtype="int64"),
        "count":             pd.array([0] * n_bars, dtype="int64"),
        "sum_notional":      [0.0] * n_bars,
        "max_single":        [0.0] * n_bars,
        "long_liq_count":    pd.array([0] * n_bars, dtype="int64"),
        "long_liq_notional": [0.0] * n_bars,
        "short_liq_count":   pd.array([0] * n_bars, dtype="int64"),
        "short_liq_notional":[0.0] * n_bars,
    })
    p = tmp_path / "liq_btcusdt_per_minute.parquet"
    write_liq_bucketed(df, p)
    return p


def test_apply_features_emits_41_cols(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=200)
    fp = _write_synthetic_funding(tmp_path, n_events=10)
    lp = _write_synthetic_liq(tmp_path, n_bars=200)
    ds = KlineWindowDataset(p, window=10, stride=1, funding_path=fp, liq_path=lp,
                            apply_features=True)
    item = ds[0]
    assert item.shape == (10, N_FEATURES)
    assert item.dtype == torch.float32


def test_apply_features_false_emits_raw_join_cols(tmp_path: Path):
    from v2.data.dataset import FEATURE_COLUMNS_WITH_JOIN
    p = _write_synthetic(tmp_path, n_bars=200)
    fp = _write_synthetic_funding(tmp_path, n_events=10)
    lp = _write_synthetic_liq(tmp_path, n_bars=200)
    ds = KlineWindowDataset(p, window=10, stride=1, funding_path=fp, liq_path=lp,
                            apply_features=False)
    item = ds[0]
    assert item.shape == (10, len(FEATURE_COLUMNS_WITH_JOIN))


def test_return_targets_gives_tuple(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=200)
    fp = _write_synthetic_funding(tmp_path, n_events=10)
    lp = _write_synthetic_liq(tmp_path, n_bars=200)
    ds = KlineWindowDataset(p, window=10, stride=1, funding_path=fp, liq_path=lp,
                            apply_features=True, return_targets=True)
    result = ds[0]
    assert isinstance(result, tuple)
    feats, log_rets = result
    assert feats.shape == (10, N_FEATURES)
    assert log_rets.shape == (10,)
    assert log_rets.dtype == torch.float32


def test_return_targets_last_bar_is_zero(tmp_path: Path):
    p = _write_synthetic(tmp_path, n_bars=50)
    fp = _write_synthetic_funding(tmp_path, n_events=5)
    lp = _write_synthetic_liq(tmp_path, n_bars=50)
    ds = KlineWindowDataset(p, window=10, stride=1, funding_path=fp, liq_path=lp,
                            apply_features=True, return_targets=True)
    _, log_rets = ds[len(ds) - 1]
    # Last bar in the file (bar 49 in a 50-bar dataset) has no next close → target = 0
    assert float(log_rets[-1]) == pytest.approx(0.0, abs=1e-7)


def test_columns_property_with_features(tmp_path: Path):
    from v2.features.constants import FEATURE_COLUMNS
    p = _write_synthetic(tmp_path, n_bars=200)
    fp = _write_synthetic_funding(tmp_path, n_events=10)
    lp = _write_synthetic_liq(tmp_path, n_bars=200)
    ds = KlineWindowDataset(p, window=10, stride=1, funding_path=fp, liq_path=lp,
                            apply_features=True)
    assert ds.columns == FEATURE_COLUMNS
