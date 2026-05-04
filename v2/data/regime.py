"""Three-bucket regime classifier (offline).

Buckets (and integer codes — see also tag_regimes CLI):
    0 = trend             — price > MA20 AND funding_rate > FUNDING_NEAR_ZERO
    1 = mean_revert       — BB width < 25th pct trailing 1000 bars
                            AND |funding_rate| < FUNDING_NEAR_ZERO
                            AND ATR14 < 25th pct trailing 1000 bars
    2 = high_vol_squeeze  — |funding_rate| > 95th pct trailing 720 funding obs
                            OR liq.count > 95th pct trailing 1000 bars
                            OR ATR14 > 90th pct trailing 1000 bars
   -1 = untagged          — insufficient trailing history (first
                            PERCENTILE_WINDOW_BARS bars of file)

Priority where multiple buckets match:
    high_vol_squeeze (2)  >  mean_revert (1)  >  trend (0)

Computed entirely from klines + funding + liq parquets — no model state.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


REGIME_TREND: int = 0
REGIME_MEAN_REVERT: int = 1
REGIME_HIGH_VOL_SQUEEZE: int = 2
REGIME_UNTAGGED: int = -1

# --- Thresholds (frozen for v2.0.0) -------------------------------------
MA_PERIOD: int = 20                # bars
BB_PERIOD: int = 20                # bars (k = 2 stddev)
BB_K: float = 2.0
ATR_PERIOD: int = 14               # bars
PERCENTILE_WINDOW_BARS: int = 1000  # trailing kline-bar window for BB / ATR / liq pcts
FUNDING_PERCENTILE_LOOKBACK: int = 720  # trailing funding observations (~240 days)

BB_WIDTH_LOW_PCTILE: float = 0.25
ATR_LOW_PCTILE: float = 0.25
ATR_HIGH_PCTILE: float = 0.90
FUNDING_EXTREME_PCTILE: float = 0.95
LIQ_COUNT_HIGH_PCTILE: float = 0.95

FUNDING_NEAR_ZERO: float = 1e-4    # |rate| < 0.01% per 8h interval


def _atr(klines: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = klines["high"]
    low = klines["low"]
    close_prev = klines["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bb_width(klines: pd.DataFrame, period: int = BB_PERIOD, k: float = BB_K) -> pd.Series:
    mid = klines["close"].rolling(period, min_periods=period).mean()
    sd = klines["close"].rolling(period, min_periods=period).std()
    return (2 * k * sd) / mid  # relative width


def _ffill_funding_to_klines(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Forward-fill funding_rate from funding observations to each kline open_time."""
    if funding.empty:
        return pd.Series(0.0, index=klines.index)
    f = funding[["funding_time", "funding_rate"]].sort_values("funding_time")
    merged = pd.merge_asof(
        klines[["open_time"]].sort_values("open_time"),
        f, left_on="open_time", right_on="funding_time", direction="backward",
    )
    return merged["funding_rate"].fillna(0.0).reset_index(drop=True)


def _liq_count_to_klines(klines: pd.DataFrame, liq: pd.DataFrame) -> pd.Series:
    """Join per-minute liq counts onto kline open_time. Empty liq → all zeros."""
    if liq.empty:
        return pd.Series(0, index=klines.index, dtype="int64")
    merged = klines[["open_time"]].merge(
        liq[["bucket_time", "count"]],
        left_on="open_time", right_on="bucket_time", how="left",
    )
    return merged["count"].fillna(0).astype("int64")


def _funding_extreme_per_bar(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Boolean per kline-bar: is the current funding_rate above the trailing
    FUNDING_EXTREME_PCTILE percentile of |rate| over the last
    FUNDING_PERCENTILE_LOOKBACK funding observations?
    """
    if funding.empty:
        return pd.Series(False, index=klines.index)
    f = funding.sort_values("funding_time").reset_index(drop=True)
    # Per-funding-event extreme flag.
    abs_rate = f["funding_rate"].abs()
    pctile = abs_rate.rolling(FUNDING_PERCENTILE_LOOKBACK, min_periods=FUNDING_PERCENTILE_LOOKBACK)\
                     .quantile(FUNDING_EXTREME_PCTILE)
    f_extreme = (abs_rate > pctile).fillna(False)
    f_lookup = pd.DataFrame({"funding_time": f["funding_time"], "extreme": f_extreme})
    merged = pd.merge_asof(
        klines[["open_time"]].sort_values("open_time"),
        f_lookup, left_on="open_time", right_on="funding_time", direction="backward",
    )
    return merged["extreme"].fillna(False).astype(bool).reset_index(drop=True)


def compute_regimes(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    liq: pd.DataFrame,
) -> pd.Series:
    """Classify each kline bar into one of {-1, 0, 1, 2}. See module docstring."""
    n = len(klines)
    out = np.full(n, REGIME_UNTAGGED, dtype=np.int8)

    if n < PERCENTILE_WINDOW_BARS:
        return pd.Series(out, index=klines.index, dtype="int8")

    close = klines["close"].reset_index(drop=True)
    ma = close.rolling(MA_PERIOD, min_periods=MA_PERIOD).mean()
    atr = _atr(klines.reset_index(drop=True))
    bbw = _bb_width(klines.reset_index(drop=True))

    bbw_low = bbw.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                 .quantile(BB_WIDTH_LOW_PCTILE)
    atr_low = atr.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                 .quantile(ATR_LOW_PCTILE)
    atr_high = atr.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                  .quantile(ATR_HIGH_PCTILE)

    funding_per_bar = _ffill_funding_to_klines(klines.reset_index(drop=True), funding)
    funding_extreme = _funding_extreme_per_bar(klines.reset_index(drop=True), funding)
    liq_count = _liq_count_to_klines(klines.reset_index(drop=True), liq)
    liq_high = liq_count.rolling(PERCENTILE_WINDOW_BARS, min_periods=PERCENTILE_WINDOW_BARS)\
                        .quantile(LIQ_COUNT_HIGH_PCTILE)

    is_compressed = bbw < bbw_low
    is_atr_low = atr < atr_low
    is_atr_high = atr > atr_high
    is_liq_spike = liq_count > liq_high
    funding_pos = funding_per_bar > FUNDING_NEAR_ZERO
    funding_near_zero = funding_per_bar.abs() < FUNDING_NEAR_ZERO

    high_vol = funding_extreme | is_liq_spike | is_atr_high
    mean_revert = is_compressed & funding_near_zero & is_atr_low
    trend = (close > ma) & funding_pos

    valid = (
        ~bbw_low.isna() & ~atr_low.isna() & ~atr_high.isna() & ~liq_high.isna()
        & ~ma.isna()
    )
    valid = valid.to_numpy()

    # Apply priority: high_vol > mean_revert > trend > default(trend).
    out_arr = np.where(high_vol.to_numpy(), REGIME_HIGH_VOL_SQUEEZE,
              np.where(mean_revert.to_numpy(), REGIME_MEAN_REVERT,
              np.where(trend.to_numpy(), REGIME_TREND,
                       REGIME_TREND)))  # default = trend (neutral baseline)
    out_arr = out_arr.astype(np.int8)
    out_arr[~valid] = REGIME_UNTAGGED
    return pd.Series(out_arr, index=klines.index, dtype="int8")
