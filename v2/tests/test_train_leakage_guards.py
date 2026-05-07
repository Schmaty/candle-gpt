"""Regression checks for CandleGPT training leakage guards."""
from pathlib import Path

from v2.model.config import ModelConfig
from v2.train.config import TrainConfig
from v2.train.loop import _build_datasets
from v2.features.constants import N_FEATURES
from v2.tests.test_train_smoke import (
    _write_synthetic_klines,
    _write_synthetic_funding,
    _write_synthetic_liq,
)


def _cfg(tmp_path: Path) -> TrainConfig:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_synthetic_klines(raw_dir / "btcusdt_1m.parquet", n=5_000)
    _write_synthetic_funding(raw_dir / "funding_btcusdt.parquet")
    _write_synthetic_liq(raw_dir / "liq_btcusdt_per_minute.parquet", n=5_000)
    return TrainConfig(
        raw_dir=raw_dir,
        runs_dir=tmp_path / "runs",
        model=ModelConfig(
            n_features=N_FEATURES, d_model=32, n_heads=4, n_layers=1,
            block_size=32, n_bins=16, dropout=0.0,
        ),
        interval="1m",
        window=32,
        stride_train=16,
        stride_val=32,
        n_bins=16,
        split_gap_bars=32,
    )


def test_clean_split_indices_supervise_only_their_own_target_ranges(tmp_path: Path):
    cfg = _cfg(tmp_path)
    full_ds, train_ds, val_ds, test_ds = _build_datasets(cfg)
    n_bars = full_ds._bars.shape[0]
    train_end_bar = int(n_bars * cfg.train_frac)
    val_end_bar = int(n_bars * (cfg.train_frac + cfg.val_frac))
    gap = cfg.split_gap_bars

    def target_bar(window_start: int) -> int:
        return window_start + cfg.window - 1

    train_targets = [target_bar(i) for i in train_ds.indices]
    val_targets = [target_bar(i) for i in val_ds.indices]
    test_targets = [target_bar(i) for i in test_ds.indices]

    assert max(train_targets) < train_end_bar - gap
    assert min(val_targets) >= train_end_bar + gap
    assert max(val_targets) < val_end_bar - gap
    assert min(test_targets) >= val_end_bar + gap
    # The file-level final log-return is a sentinel 0.0 and must never be used.
    assert max(test_targets) <= n_bars - 2


def test_absolute_price_and_timeline_features_are_zeroed_in_training_dataset(tmp_path: Path):
    cfg = _cfg(tmp_path)
    full_ds, *_ = _build_datasets(cfg)
    cols = list(full_ds.columns)
    for name in ("log_close", "time_index_norm"):
        idx = cols.index(name)
        assert float(abs(full_ds._bars[:, idx]).max()) == 0.0
