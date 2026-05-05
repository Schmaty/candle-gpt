"""Canonical names for the 41-dimensional v2 engineered feature vector.

Index positions are stable — downstream code (model, training, dashboard) may
use FEATURE_COLUMNS.index("name") to retrieve the column position.
"""
from __future__ import annotations

FEATURE_COLUMNS: tuple[str, ...] = (
    # Group A — Candle structure (5)
    "log_return",
    "log_return_open",
    "high_low_range",
    "close_open_range",
    "candle_body_ratio",
    # Group B — Volatility (4)
    "realized_vol_5",
    "realized_vol_20",
    "realized_vol_60",
    "atr_14_norm",
    # Group C — Volume (3)
    "log_volume",
    "volume_z_5",
    "volume_z_20",
    # Group D — Momentum / MA (8)
    "ema12_ratio",
    "ema26_ratio",
    "macd_norm",
    "macd_signal_norm",
    "rsi_14_norm",
    "close_vs_ma20",
    "close_vs_ma60",
    "vwap_bar_ratio",
    # Group E — Rolling extremes (2)
    "high_vs_max20",
    "low_vs_min20",
    # Group F — Time cyclical (4)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Group G — Regime one-hot (3)
    "regime_0",
    "regime_1",
    "regime_2",
    # Group H — Funding (3)
    "funding_rate_norm",
    "minutes_until_funding_norm",
    "mark_premium",
    # Group I — Liquidations (7)
    "log_liq_count",
    "log_liq_notional",
    "log_liq_max",
    "long_liq_frac",
    "short_liq_frac",
    "log_long_liq_notional",
    "log_short_liq_notional",
    # Group J — Absolute level / time (2)
    "log_close",
    "time_index_norm",
    # Group K — Multi-lag returns / signed volume / extra RV (4) [v2.1]
    # Appended at the tail so older 41-feature checkpoints can still be
    # served by slicing the first 41 columns of compute_features output.
    "log_return_3",
    "log_return_10",
    "signed_log_volume",
    "realized_vol_12",
)

N_FEATURES: int = len(FEATURE_COLUMNS)
N_FEATURES_LEGACY_V20: int = 41  # frozen position of the v2.0.0 prefix
assert N_FEATURES == 45, f"Expected 45 features, got {N_FEATURES}"
