"""Regime classifier: three-bucket taxonomy with priority high_vol > mean_revert > trend."""
import numpy as np
import pandas as pd
import pytest

from v2.data.regime import (
    REGIME_TREND,
    REGIME_MEAN_REVERT,
    REGIME_HIGH_VOL_SQUEEZE,
    REGIME_UNTAGGED,
    PERCENTILE_WINDOW_BARS,
    FUNDING_NEAR_ZERO,
    compute_regimes,
)
from v2.data.constants import (
    LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES,
    FUNDING_COLUMNS,
)


def _kline_frame(n: int, *, base: float = 100.0, step: float = 0.0) -> pd.DataFrame:
    """Synthetic 1m klines starting at t=0; close walks up by `step` per bar."""
    closes = base + step * np.arange(n)
    return pd.DataFrame({
        "open_time":  pd.array([i * 60_000 for i in range(n)], dtype="int64"),
        "open":       closes - 0.1,
        "high":       closes + 0.5,
        "low":        closes - 0.5,
        "close":      closes,
        "volume":     np.full(n, 10.0),
        "close_time": pd.array([i * 60_000 + 59_999 for i in range(n)], dtype="int64"),
        "regime":     pd.array([-1] * n, dtype="int8"),
    })


def _funding_frame(rate_per_event: list[float]) -> pd.DataFrame:
    n = len(rate_per_event)
    return pd.DataFrame({
        "funding_time": pd.array([i * 8 * 3_600_000 for i in range(n)], dtype="int64"),
        "funding_rate": rate_per_event,
        "mark_price":   [100.0] * n,
    })


def _empty_liq() -> pd.DataFrame:
    return pd.DataFrame({
        c: pd.Series(dtype=LIQ_BUCKETED_DTYPES[c]) for c in LIQ_BUCKETED_COLUMNS
    })


def test_returns_int8_series_aligned_to_klines():
    n = PERCENTILE_WINDOW_BARS + 100
    klines = _kline_frame(n, step=0.05)
    funding = _funding_frame([0.0001] * 5)
    out = compute_regimes(klines, funding, _empty_liq())
    assert isinstance(out, pd.Series)
    assert out.dtype == np.int8
    assert len(out) == len(klines)


def test_pre_warmup_bars_get_untagged_sentinel():
    """Bars that lack PERCENTILE_WINDOW_BARS of trailing history default to UNTAGGED (-1)."""
    n = PERCENTILE_WINDOW_BARS // 2
    klines = _kline_frame(n, step=0.01)
    out = compute_regimes(klines, _funding_frame([0.0001] * 5), _empty_liq())
    assert (out == REGIME_UNTAGGED).all()


def test_trend_is_default_when_price_above_ma_and_funding_positive():
    n = PERCENTILE_WINDOW_BARS + 100
    klines = _kline_frame(n, base=100.0, step=0.10)  # uptrend
    # Persistent positive funding well above NEAR_ZERO.
    funding = _funding_frame([5 * FUNDING_NEAR_ZERO] * 50)
    out = compute_regimes(klines, funding, _empty_liq())
    # Last 50 bars (after warmup, well into the trend) should mostly be TREND.
    tail = out.iloc[-50:]
    assert (tail == REGIME_TREND).sum() >= 30


def test_high_vol_squeeze_beats_trend_when_liq_spike():
    """Priority: a bar matching both trend AND high_vol (via liq spike) → HIGH_VOL_SQUEEZE."""
    n = PERCENTILE_WINDOW_BARS + 50
    klines = _kline_frame(n, step=0.05)  # uptrend → would normally be TREND
    # Positive funding → trend condition satisfied
    funding = _funding_frame([5 * FUNDING_NEAR_ZERO] * 50)
    # Massive liq spike at the last bar forces high_vol over trend (priority test).
    spike_bucket = klines["open_time"].iloc[-1]
    liq = pd.DataFrame({
        "bucket_time":         pd.array([spike_bucket], dtype="int64"),
        "count":               pd.array([5_000], dtype="int64"),
        "sum_notional":        [5_000_000.0],
        "max_single":          [1_000_000.0],
        "long_liq_count":      pd.array([5_000], dtype="int64"),
        "long_liq_notional":   [5_000_000.0],
        "short_liq_count":     pd.array([0], dtype="int64"),
        "short_liq_notional":  [0.0],
    })
    out = compute_regimes(klines, funding, liq)
    assert out.iloc[-1] == REGIME_HIGH_VOL_SQUEEZE


def test_mean_revert_when_compressed_and_funding_near_zero():
    n = PERCENTILE_WINDOW_BARS + 200
    # First 1000 bars are volatile; last 200 are compressed (constant close, tight range).
    closes_volatile = 100.0 + 5.0 * np.sin(np.arange(PERCENTILE_WINDOW_BARS) / 10.0)
    closes_compressed = np.full(200, 100.0)
    closes = np.concatenate([closes_volatile, closes_compressed])
    klines = pd.DataFrame({
        "open_time":  pd.array([i * 60_000 for i in range(n)], dtype="int64"),
        "open":       closes,
        "high":       closes + 0.05,   # tight range in compressed phase
        "low":        closes - 0.05,
        "close":      closes,
        "volume":     np.full(n, 10.0),
        "close_time": pd.array([i * 60_000 + 59_999 for i in range(n)], dtype="int64"),
        "regime":     pd.array([-1] * n, dtype="int8"),
    })
    # Funding hovers near zero throughout.
    funding = _funding_frame([0.0] * 50)
    out = compute_regimes(klines, funding, _empty_liq())
    # The compressed tail should classify mean-revert in at least the latter half.
    tail = out.iloc[-100:]
    assert (tail == REGIME_MEAN_REVERT).sum() >= 50


def test_liq_spike_triggers_high_vol_squeeze():
    n = PERCENTILE_WINDOW_BARS + 50
    klines = _kline_frame(n, step=0.05)
    funding = _funding_frame([FUNDING_NEAR_ZERO * 0.5] * 50)  # benign funding
    # Liq spike at the LAST bar far above any reasonable trailing percentile.
    spike_bucket = klines["open_time"].iloc[-1]
    liq = pd.DataFrame({
        "bucket_time":         pd.array([spike_bucket], dtype="int64"),
        "count":               pd.array([5_000], dtype="int64"),
        "sum_notional":        [5_000_000.0],
        "max_single":          [1_000_000.0],
        "long_liq_count":      pd.array([5_000], dtype="int64"),
        "long_liq_notional":   [5_000_000.0],
        "short_liq_count":     pd.array([0], dtype="int64"),
        "short_liq_notional":  [0.0],
    })
    out = compute_regimes(klines, funding, liq)
    assert out.iloc[-1] == REGIME_HIGH_VOL_SQUEEZE
