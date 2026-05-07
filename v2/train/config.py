"""Training hyperparameters and path configuration for CandleGPTv2."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path

from v2.model.config import ModelConfig


@dataclass
class TrainConfig:
    # --- Paths ---
    raw_dir: Path = field(default_factory=lambda: Path("v2/data/raw"))
    runs_dir: Path = field(default_factory=lambda: Path("v2/runs"))
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    # --- Model ---
    model: ModelConfig = field(default_factory=ModelConfig)

    # --- Data ---
    kline_file: str = "btcusdt_1m.parquet"
    funding_file: str = "funding_btcusdt.parquet"
    liq_file: str = "liq_btcusdt_per_minute.parquet"
    # Source parquet is always 1m; we OHLCV-resample on the fly to the
    # training timeframe. Bin centers, feature stats, and run-time hyperparams
    # all re-derive from the resampled stream.
    interval: str = "5m"
    window: int = 1024
    stride_train: int = 16
    stride_val: int = 1024
    # Keep target labels away from split boundaries. With forecast-only loss,
    # each window's supervised target is the final bar in that window; this gap
    # prevents train/val/test targets from sitting immediately adjacent while
    # still allowing validation windows to use past context.
    split_gap_bars: int | None = None

    # --- Split (by bar index) ---
    train_frac: float = 0.72
    val_frac: float = 0.12

    # --- Optimiser ---
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 32

    # --- Schedule ---
    warmup_steps: int = 500
    max_steps: int = 200_000

    # --- Wall-clock cap ---
    max_wall_clock_s: float = 6 * 3600.0
    resume_from: Path | None = None
    early_stop_patience_evals: int | None = None
    early_stop_min_delta: float = 0.001
    checkpoint_interval_s: float = 30 * 60.0

    # --- Eval ---
    val_interval_steps: int = 500
    val_batches: int = 100
    log_interval_steps: int = 50
    # If True, compute loss only on the final timestep in each window. This
    # turns every sample into one clean next-bar forecast instead of scoring all
    # heavily-overlapping timesteps inside the context window.
    forecast_only_loss: bool = True

    # --- Loss ---
    # "ce" keeps the old hard-label cross entropy. "soft_ce" keeps the exact
    # same model output shape (n_bins logits), but trains against a Gaussian
    # distribution around the true ordinal return bin so near misses are less
    # wrong than far misses.
    loss_type: str = "ce"
    soft_label_sigma_bins: float = 2.0

    # --- Tokenizer ---
    n_bins: int = 256

    # --- Progress emission (min interval between status.json rewrites) ---
    progress_interval_s: float = 5.0

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def kline_path(self) -> Path:
        return self.raw_dir / self.kline_file

    @property
    def funding_path(self) -> Path:
        return self.raw_dir / self.funding_file

    @property
    def liq_path(self) -> Path:
        return self.raw_dir / self.liq_file

    @property
    def tokenizer_path(self) -> Path:
        return self.run_dir / "tokenizer.pkl"

    @property
    def best_ckpt_path(self) -> Path:
        return self.ckpt_dir / "best_val.pt"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "REPORT.md"
