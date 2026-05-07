"""compute_features: 18-col joined DataFrame → 41-col engineered DataFrame."""
import numpy as np
import pandas as pd
import pytest

from v2.features.constants import FEATURE_COLUMNS, N_FEATURES
from v2.features.engineer import compute_features


def _make_raw_df(n: int = 200) -> pd.DataFrame:
    """Synthetic joined DataFrame with 18 FEATURE_COLUMNS_WITH_JOIN columns."""
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    return pd.DataFrame({
        "open_time":            np.arange(n, dtype="int64") * 60_000,
        "open":                 open_.astype("float64"),
        "high":                 high.astype("float64"),
        "low":                  low.astype("float64"),
        "close":                close.astype("float64"),
        "volume":               np.abs(rng.normal(1000, 200, n)).astype("float64"),
        "close_time":           (np.arange(n, dtype="int64") * 60_000 + 59_999),
        "regime":               np.full(n, 1, dtype="int8"),
        "funding_rate":         np.full(n, 0.0001, dtype="float64"),
        "mark_price":           (close * (1 + rng.normal(0, 0.0005, n))).astype("float64"),
        "minutes_until_funding": np.full(n, 240.0, dtype="float64"),
        "liq_count":            rng.integers(0, 5, n).astype("int64"),
        "liq_sum_notional":     rng.uniform(0, 50_000, n).astype("float64"),
        "liq_max_single":       rng.uniform(0, 10_000, n).astype("float64"),
        "long_liq_count":       rng.integers(0, 3, n).astype("int64"),
        "long_liq_notional":    rng.uniform(0, 25_000, n).astype("float64"),
        "short_liq_count":      rng.integers(0, 3, n).astype("int64"),
        "short_liq_notional":   rng.uniform(0, 25_000, n).astype("float64"),
    })


def test_output_shape():
    df = _make_raw_df(200)
    out = compute_features(df)
    assert out.shape == (200, N_FEATURES)


def test_output_columns_match_feature_columns():
    df = _make_raw_df(200)
    out = compute_features(df)
    assert tuple(out.columns) == FEATURE_COLUMNS


def test_no_nan_in_output():
    df = _make_raw_df(200)
    out = compute_features(df)
    nan_cols = [c for c in out.columns if out[c].isna().any()]
    assert not nan_cols, f"NaN in columns: {nan_cols}"


def test_no_inf_in_output():
    df = _make_raw_df(200)
    out = compute_features(df)
    inf_cols = [c for c in out.columns if np.isinf(out[c].to_numpy()).any()]
    assert not inf_cols, f"Inf in columns: {inf_cols}"


def test_group_a_log_return_correct():
    df = _make_raw_df(100)
    out = compute_features(df)
    # log_return[0] should be 0 (no previous bar)
    assert out["log_return"].iloc[0] == pytest.approx(0.0)
    # log_return[1] = log(close[1] / close[0])
    expected = float(np.log(df["close"].iloc[1] / df["close"].iloc[0]))
    assert out["log_return"].iloc[1] == pytest.approx(expected, rel=1e-5)


def test_group_a_high_low_range_positive():
    df = _make_raw_df(100)
    out = compute_features(df)
    assert (out["high_low_range"] >= 0).all()


def test_group_a_candle_body_ratio_in_unit_interval():
    df = _make_raw_df(100)
    out = compute_features(df)
    assert (out["candle_body_ratio"] >= 0).all()
    assert (out["candle_body_ratio"] <= 1.0 + 1e-6).all()


def test_group_b_vol_nonnegative():
    df = _make_raw_df(100)
    out = compute_features(df)
    for col in ("realized_vol_5", "realized_vol_20", "realized_vol_60", "atr_14_norm"):
        assert (out[col] >= 0).all(), f"{col} has negative values"


def test_group_d_rsi_in_unit_interval():
    df = _make_raw_df(200)
    out = compute_features(df)
    assert (out["rsi_14_norm"] >= 0).all()
    assert (out["rsi_14_norm"] <= 1.0 + 1e-6).all()


def test_group_f_time_features_in_neg1_pos1():
    df = _make_raw_df(200)
    out = compute_features(df)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        vals = out[col].to_numpy()
        assert (vals >= -1.0 - 1e-6).all() and (vals <= 1.0 + 1e-6).all(), \
            f"{col} out of [-1,1]"


def test_group_g_regime_onehot_sums_to_one_or_zero():
    # regime=1 → regime_1=1, regime_0=0, regime_2=0
    df = _make_raw_df(50)
    df["regime"] = pd.array([1] * 50, dtype="int8")
    out = compute_features(df)
    assert (out["regime_0"] == 0).all()
    assert (out["regime_1"] == 1).all()
    assert (out["regime_2"] == 0).all()


def test_group_g_untagged_regime_all_zeros():
    # regime=-1 (untagged sentinel) → all three regime cols are 0
    df = _make_raw_df(50)
    df["regime"] = pd.array([-1] * 50, dtype="int8")
    out = compute_features(df)
    assert (out["regime_0"] == 0).all()
    assert (out["regime_1"] == 0).all()
    assert (out["regime_2"] == 0).all()


def test_group_h_funding_rate_norm_bounded():
    df = _make_raw_df(100)
    out = compute_features(df)
    assert (out["funding_rate_norm"].abs() <= 1.0 + 1e-6).all()


def test_group_h_minutes_until_funding_norm_in_01():
    df = _make_raw_df(100)
    out = compute_features(df)
    # 240 / 480 = 0.5
    assert out["minutes_until_funding_norm"].iloc[0] == pytest.approx(0.5, rel=1e-5)


def test_group_i_log_liq_count_nonnegative():
    df = _make_raw_df(100)
    out = compute_features(df)
    assert (out["log_liq_count"] >= 0).all()


def test_group_j_absolute_keys_are_zeroed():
    df = _make_raw_df(100)
    out = compute_features(df)
    # Schema positions are retained for checkpoint compatibility, but the
    # values are zeroed so models cannot memorize absolute price/time.
    assert (out["time_index_norm"] == 0.0).all()
    assert (out["log_close"] == 0.0).all()


def test_single_bar_does_not_crash():
    df = _make_raw_df(1)
    out = compute_features(df)
    assert out.shape == (1, N_FEATURES)
    assert not out.isna().any().any()
