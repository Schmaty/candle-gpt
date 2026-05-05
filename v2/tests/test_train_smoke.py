"""Smoke test: 10-step training on synthetic data."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from v2.data.constants import Asset, Timeframe, FUNDING_COLUMNS, FUNDING_DTYPES
from v2.data.store import write_klines, write_funding, write_liq_bucketed
from v2.features.constants import N_FEATURES
from v2.model.config import ModelConfig
from v2.train.config import TrainConfig
from v2.train.loop import train


def _write_synthetic_klines(path: Path, n: int = 2000) -> None:
    rng = np.random.default_rng(1)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    regime_arr = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)), dtype=np.int8)
    df = pd.DataFrame({
        "open_time":  np.arange(n, dtype=np.int64) * 60_000,
        "open":       close * (1 + rng.normal(0, 0.001, n)),
        "high":       close * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low":        close * (1 - np.abs(rng.normal(0, 0.002, n))),
        "close":      close,
        "volume":     np.abs(rng.normal(1000, 200, n)),
        "close_time": np.arange(n, dtype=np.int64) * 60_000 + 59_999,
        "regime":     regime_arr,
    })
    write_klines(df, path)


def _write_synthetic_funding(path: Path, n: int = 10) -> None:
    df = pd.DataFrame({
        "funding_time": np.array([i * 8 * 60 * 60 * 1000 for i in range(n)], dtype=np.int64),
        "funding_rate": np.full(n, 0.0001, dtype=np.float64),
        "mark_price":   np.full(n, 100.0, dtype=np.float64),
    })
    write_funding(df, path)


def _write_synthetic_liq(path: Path, n: int = 2000) -> None:
    df = pd.DataFrame({
        "bucket_time":       np.arange(n, dtype=np.int64) * 60_000,
        "count":             np.zeros(n, dtype=np.int64),
        "sum_notional":      np.zeros(n, dtype=np.float64),
        "max_single":        np.zeros(n, dtype=np.float64),
        "long_liq_count":    np.zeros(n, dtype=np.int64),
        "long_liq_notional": np.zeros(n, dtype=np.float64),
        "short_liq_count":   np.zeros(n, dtype=np.int64),
        "short_liq_notional":np.zeros(n, dtype=np.float64),
    })
    write_liq_bucketed(df, path)


def test_smoke_10_steps(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # 20k 1m bars resamples to 4k 5m bars — enough for window=32 + a train/val
    # split. Default cfg.interval is "5m", so the dataset OHLCV-resamples on
    # the fly and the tokenizer fits on 5m returns.
    _write_synthetic_klines(raw_dir / "btcusdt_1m.parquet", n=20_000)
    _write_synthetic_funding(raw_dir / "funding_btcusdt.parquet")
    _write_synthetic_liq(raw_dir / "liq_btcusdt_per_minute.parquet", n=20_000)

    cfg = TrainConfig(
        raw_dir=raw_dir,
        runs_dir=tmp_path / "runs",
        model=ModelConfig(
            n_features=N_FEATURES, d_model=32, n_heads=4, n_layers=1,
            block_size=32, n_bins=16, dropout=0.0,
        ),
        window=32,
        stride_train=16,
        stride_val=32,
        batch_size=4,
        max_steps=10,
        warmup_steps=2,
        val_interval_steps=5,
        val_batches=2,
        log_interval_steps=5,
        n_bins=16,
        max_wall_clock_s=3600.0,
        checkpoint_interval_s=3600.0,
        progress_interval_s=0.0,  # don't throttle in tests
    )

    run_id = train(cfg)
    assert run_id == cfg.run_id
    assert (cfg.run_dir / "train.log").exists()
    assert cfg.tokenizer_path.exists()
    assert cfg.run_dir.exists()
    # status.json and events.jsonl must be present (Task 2A)
    assert (cfg.run_dir / "status.json").exists()
    assert (cfg.run_dir / "events.jsonl").exists()
    assert cfg.best_ckpt_path.exists(), "best_val.pt was not written — val checkpoint missing"
