# Candle-GPT v2 — Feature Engineering: 41-Dim Feature Vector (Plan 2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `v2/features/` package that transforms the 18-column joined DataFrame (produced by `KlineWindowDataset` after the funding+liq join) into a 41-dimensional per-bar feature vector, and extend `KlineWindowDataset` to emit `(window, 41)` tensors by default when the join is active, plus `(features, log_returns)` pairs for supervised training.

**Architecture:** New `v2/features/` subpackage with two modules: `constants.py` (canonical 41-name tuple + `N_FEATURES=41`) and `engineer.py` (single public function `compute_features(df) -> pd.DataFrame`). `KlineWindowDataset` gains two new kwargs — `apply_features: bool = True` (only honoured when funding+liq join is performed) and `return_targets: bool = False` — and imports `compute_features` lazily to avoid circular imports. No other module is aware of the 18→41 mapping.

**Tech Stack:** Python 3.11+, numpy, pandas, torch, pytest.

---

## Feature Vector Specification (41 dims)

| # | Name | Formula | Group |
|---|------|---------|-------|
| 0 | `log_return` | log(close_t / close_{t-1}); 0 for t=0 | A |
| 1 | `log_return_open` | log(open_t / close_{t-1}); 0 for t=0 | A |
| 2 | `high_low_range` | (high - low) / close | A |
| 3 | `close_open_range` | (close - open) / open | A |
| 4 | `candle_body_ratio` | \|close - open\| / (high - low + 1e-8) | A |
| 5 | `realized_vol_5` | rolling(5).std(log_return); 0 if NaN | B |
| 6 | `realized_vol_20` | rolling(20).std(log_return); 0 if NaN | B |
| 7 | `realized_vol_60` | rolling(60).std(log_return); 0 if NaN | B |
| 8 | `atr_14_norm` | EWM-ATR(14) / close; 0 if NaN | B |
| 9 | `log_volume` | log1p(volume) | C |
| 10 | `volume_z_5` | (vol - roll5_mean) / (roll5_std + 1e-8); 0 if NaN | C |
| 11 | `volume_z_20` | (vol - roll20_mean) / (roll20_std + 1e-8); 0 if NaN | C |
| 12 | `ema12_ratio` | ema(span=12)/close - 1; 0 if NaN | D |
| 13 | `ema26_ratio` | ema(span=26)/close - 1; 0 if NaN | D |
| 14 | `macd_norm` | (ema12 - ema26) / close; 0 if NaN | D |
| 15 | `macd_signal_norm` | ema9(macd_line) / close; 0 if NaN | D |
| 16 | `rsi_14_norm` | RSI(14)/100 ∈ (0,1]; 0.5 if NaN | D |
| 17 | `close_vs_ma20` | close / rolling(20).mean() - 1; 0 if NaN | D |
| 18 | `close_vs_ma60` | close / rolling(60).mean() - 1; 0 if NaN | D |
| 19 | `vwap_bar_ratio` | (H+L+C)/3/close - 1 | D |
| 20 | `high_vs_max20` | high / rolling_max(20) - 1; 0 if NaN | E |
| 21 | `low_vs_min20` | low / rolling_min(20) - 1; 0 if NaN | E |
| 22 | `hour_sin` | sin(2π × hour_utc/24) | F |
| 23 | `hour_cos` | cos(2π × hour_utc/24) | F |
| 24 | `dow_sin` | sin(2π × day_of_week/7) | F |
| 25 | `dow_cos` | cos(2π × day_of_week/7) | F |
| 26 | `regime_0` | (regime == 0).float() | G |
| 27 | `regime_1` | (regime == 1).float() | G |
| 28 | `regime_2` | (regime == 2).float() | G |
| 29 | `funding_rate_norm` | tanh(funding_rate × 1000) | H |
| 30 | `minutes_until_funding_norm` | minutes_until_funding / 480.0 | H |
| 31 | `mark_premium` | (mark_price / close) - 1; 0 if NaN | H |
| 32 | `log_liq_count` | log1p(liq_count) | I |
| 33 | `log_liq_notional` | log1p(liq_sum_notional) | I |
| 34 | `log_liq_max` | log1p(liq_max_single) | I |
| 35 | `long_liq_frac` | long_liq_notional / (liq_sum_notional + 1e-8) | I |
| 36 | `short_liq_frac` | short_liq_notional / (liq_sum_notional + 1e-8) | I |
| 37 | `log_long_liq_notional` | log1p(long_liq_notional) | I |
| 38 | `log_short_liq_notional` | log1p(short_liq_notional) | I |
| 39 | `log_close` | log(close) | J |
| 40 | `time_index_norm` | (open_time - open_time.min()) / ms_per_day | J |

---

## File Structure

**Create:**
- `projects/candle-gpt/v2/features/__init__.py` — empty marker
- `projects/candle-gpt/v2/features/constants.py` — `FEATURE_COLUMNS` tuple (41 names), `N_FEATURES = 41`
- `projects/candle-gpt/v2/features/engineer.py` — `compute_features(df: pd.DataFrame) -> pd.DataFrame`; all private helpers internal

**Test:**
- `projects/candle-gpt/v2/tests/test_features_constants.py` — count, uniqueness, no overlap with dataset raw names
- `projects/candle-gpt/v2/tests/test_features_engineer.py` — shape, NaN-free, per-group spot checks

**Modify:**
- `projects/candle-gpt/v2/data/dataset.py` — add `apply_features: bool = True`, `return_targets: bool = False`; call `compute_features` when join was performed and `apply_features=True`; update `columns` property; `__getitem__` returns tuple when `return_targets=True`
- `projects/candle-gpt/v2/tests/test_dataset.py` — add tests for (window, 41) shape and `(features, log_returns)` pair; existing raw-column tests remain unchanged (they use no join paths so feature engineering is not triggered)

**Not touched:** v1 code, `v2/data/constants.py`, `v2/data/store.py`, `v2/data/fetch.py`, existing Plan 1.5 modules.

---

## Task 1: Feature constants module

**Files:**
- Create: `v2/features/__init__.py`
- Create: `v2/features/constants.py`
- Test: `v2/tests/test_features_constants.py`

- [ ] **Step 1: Write the failing test**

Path: `v2/tests/test_features_constants.py`

```python
"""Feature constants: 41 names, all unique, no conflict with raw join column names."""
from v2.features.constants import FEATURE_COLUMNS, N_FEATURES
from v2.data.dataset import FEATURE_COLUMNS_WITH_JOIN


def test_n_features_is_41():
    assert N_FEATURES == 41


def test_feature_columns_length():
    assert len(FEATURE_COLUMNS) == 41


def test_feature_columns_all_unique():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


def test_feature_columns_no_raw_names():
    # Engineered features should not shadow raw column names (confuses downstream indexing).
    raw = set(FEATURE_COLUMNS_WITH_JOIN)
    overlap = raw & set(FEATURE_COLUMNS)
    assert not overlap, f"Overlap between raw and engineered columns: {overlap}"


def test_feature_columns_contains_expected_groups():
    names = set(FEATURE_COLUMNS)
    # Spot-check one name from each group
    assert "log_return" in names          # A
    assert "realized_vol_20" in names     # B
    assert "volume_z_20" in names         # C
    assert "rsi_14_norm" in names         # D
    assert "high_vs_max20" in names       # E
    assert "hour_sin" in names            # F
    assert "regime_0" in names            # G
    assert "funding_rate_norm" in names   # H
    assert "log_liq_count" in names       # I
    assert "log_close" in names           # J
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd projects/candle-gpt && uv run pytest v2/tests/test_features_constants.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'v2.features'`

- [ ] **Step 3: Create package marker and constants module**

Path: `v2/features/__init__.py` — empty file.

Path: `v2/features/constants.py`:

```python
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
)

N_FEATURES: int = len(FEATURE_COLUMNS)
assert N_FEATURES == 41, f"Expected 41 features, got {N_FEATURES}"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest v2/tests/test_features_constants.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/features/__init__.py v2/features/constants.py v2/tests/test_features_constants.py
git commit -m "v2: feature constants — 41-dim FEATURE_COLUMNS"
```

---

## Task 2: Feature engineer — `compute_features`

**Files:**
- Create: `v2/features/engineer.py`
- Test: `v2/tests/test_features_engineer.py`

- [ ] **Step 1: Write the failing tests**

Path: `v2/tests/test_features_engineer.py`

```python
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


def test_group_j_time_index_norm_starts_at_zero():
    df = _make_raw_df(100)
    out = compute_features(df)
    assert out["time_index_norm"].iloc[0] == pytest.approx(0.0, abs=1e-8)


def test_single_bar_does_not_crash():
    df = _make_raw_df(1)
    out = compute_features(df)
    assert out.shape == (1, N_FEATURES)
    assert not out.isna().any().any()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest v2/tests/test_features_engineer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'v2.features.engineer'`

- [ ] **Step 3: Implement `engineer.py`**

Path: `v2/features/engineer.py`:

```python
"""Feature engineering: 18-col joined DataFrame → 41-dim feature vector.

Called by KlineWindowDataset when apply_features=True and a funding/liq join
has been performed. All NaN produced by rolling warm-up are replaced with 0.0
(or 0.5 for RSI) so the model sees consistent values for the first ~60 bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.features.constants import FEATURE_COLUMNS


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 1.0 / (1.0 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (len(df), 41) DataFrame with columns == FEATURE_COLUMNS.

    Input must have all 18 columns from FEATURE_COLUMNS_WITH_JOIN.
    NaN values (rolling warm-up) are filled with 0.0; RSI warm-up fills to 0.5.
    Does not modify the input DataFrame.
    """
    c = df["close"].astype("float64")
    o = df["open"].astype("float64")
    h = df["high"].astype("float64")
    l = df["low"].astype("float64")
    v = df["volume"].astype("float64")
    t = df["open_time"].astype("int64")

    prev_close = c.shift(1)

    # --- Group A: Candle structure ---
    log_return = np.log(c / prev_close).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    log_return_open = np.log(o / prev_close).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    high_low_range = (h - l) / c
    close_open_range = (c - o) / o
    candle_body_ratio = (c - o).abs() / (h - l + 1e-8)

    # --- Group B: Volatility ---
    realized_vol_5 = log_return.rolling(5, min_periods=1).std().fillna(0.0)
    realized_vol_20 = log_return.rolling(20, min_periods=1).std().fillna(0.0)
    realized_vol_60 = log_return.rolling(60, min_periods=1).std().fillna(0.0)
    atr14 = _atr(h, l, c, period=14)
    atr_14_norm = (atr14 / c).fillna(0.0)

    # --- Group C: Volume ---
    log_volume = np.log1p(v)
    vol_mean5 = v.rolling(5, min_periods=1).mean()
    vol_std5 = v.rolling(5, min_periods=1).std().fillna(1.0)
    volume_z_5 = ((v - vol_mean5) / (vol_std5 + 1e-8)).fillna(0.0)
    vol_mean20 = v.rolling(20, min_periods=1).mean()
    vol_std20 = v.rolling(20, min_periods=1).std().fillna(1.0)
    volume_z_20 = ((v - vol_mean20) / (vol_std20 + 1e-8)).fillna(0.0)

    # --- Group D: Momentum / MA ---
    ema12 = c.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = c.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    ema12_ratio = (ema12 / c - 1).fillna(0.0)
    ema26_ratio = (ema26 / c - 1).fillna(0.0)
    macd_norm = (macd_line / c).fillna(0.0)
    macd_signal_norm = (signal_line / c).fillna(0.0)
    rsi_14_norm = _rsi(c, 14).fillna(0.5)
    ma20 = c.rolling(20, min_periods=1).mean()
    ma60 = c.rolling(60, min_periods=1).mean()
    close_vs_ma20 = (c / ma20 - 1).fillna(0.0)
    close_vs_ma60 = (c / ma60 - 1).fillna(0.0)
    vwap_bar_ratio = (h + l + c) / 3 / c - 1

    # --- Group E: Rolling extremes ---
    high_vs_max20 = (h / h.rolling(20, min_periods=1).max() - 1).fillna(0.0)
    low_vs_min20 = (l / l.rolling(20, min_periods=1).min() - 1).fillna(0.0)

    # --- Group F: Time cyclical ---
    ts = pd.to_datetime(t, unit="ms", utc=True)
    hour = (ts.dt.hour + ts.dt.minute / 60.0).astype("float64")
    dow = ts.dt.dayofweek.astype("float64")
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    # --- Group G: Regime one-hot ---
    reg = df["regime"].astype("int8")
    regime_0 = (reg == 0).astype("float64")
    regime_1 = (reg == 1).astype("float64")
    regime_2 = (reg == 2).astype("float64")

    # --- Group H: Funding ---
    fr = df["funding_rate"].astype("float64")
    mp = df["mark_price"].astype("float64")
    muf = df["minutes_until_funding"].astype("float64")
    funding_rate_norm = np.tanh(fr * 1000)
    minutes_until_funding_norm = muf / 480.0
    mark_premium = (mp / c - 1).fillna(0.0)

    # --- Group I: Liquidations ---
    liq_notional = df["liq_sum_notional"].astype("float64")
    ll_notional = df["long_liq_notional"].astype("float64")
    sl_notional = df["short_liq_notional"].astype("float64")
    log_liq_count = np.log1p(df["liq_count"].astype("float64"))
    log_liq_notional = np.log1p(liq_notional)
    log_liq_max = np.log1p(df["liq_max_single"].astype("float64"))
    long_liq_frac = ll_notional / (liq_notional + 1e-8)
    short_liq_frac = sl_notional / (liq_notional + 1e-8)
    log_long_liq_notional = np.log1p(ll_notional)
    log_short_liq_notional = np.log1p(sl_notional)

    # --- Group J: Absolute level / time ---
    log_close = np.log(c)
    time_index_norm = (t - t.min()) / (24.0 * 60 * 60 * 1000)

    out = pd.DataFrame(
        {
            "log_return": log_return,
            "log_return_open": log_return_open,
            "high_low_range": high_low_range,
            "close_open_range": close_open_range,
            "candle_body_ratio": candle_body_ratio,
            "realized_vol_5": realized_vol_5,
            "realized_vol_20": realized_vol_20,
            "realized_vol_60": realized_vol_60,
            "atr_14_norm": atr_14_norm,
            "log_volume": log_volume,
            "volume_z_5": volume_z_5,
            "volume_z_20": volume_z_20,
            "ema12_ratio": ema12_ratio,
            "ema26_ratio": ema26_ratio,
            "macd_norm": macd_norm,
            "macd_signal_norm": macd_signal_norm,
            "rsi_14_norm": rsi_14_norm,
            "close_vs_ma20": close_vs_ma20,
            "close_vs_ma60": close_vs_ma60,
            "vwap_bar_ratio": vwap_bar_ratio,
            "high_vs_max20": high_vs_max20,
            "low_vs_min20": low_vs_min20,
            "hour_sin": hour_sin.to_numpy() if hasattr(hour_sin, "to_numpy") else np.asarray(hour_sin),
            "hour_cos": hour_cos.to_numpy() if hasattr(hour_cos, "to_numpy") else np.asarray(hour_cos),
            "dow_sin": dow_sin.to_numpy() if hasattr(dow_sin, "to_numpy") else np.asarray(dow_sin),
            "dow_cos": dow_cos.to_numpy() if hasattr(dow_cos, "to_numpy") else np.asarray(dow_cos),
            "regime_0": regime_0,
            "regime_1": regime_1,
            "regime_2": regime_2,
            "funding_rate_norm": funding_rate_norm,
            "minutes_until_funding_norm": minutes_until_funding_norm,
            "mark_premium": mark_premium,
            "log_liq_count": log_liq_count,
            "log_liq_notional": log_liq_notional,
            "log_liq_max": log_liq_max,
            "long_liq_frac": long_liq_frac,
            "short_liq_frac": short_liq_frac,
            "log_long_liq_notional": log_long_liq_notional,
            "log_short_liq_notional": log_short_liq_notional,
            "log_close": log_close,
            "time_index_norm": time_index_norm,
        },
        index=df.index,
    )

    return out.fillna(0.0).replace([np.inf, -np.inf], 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest v2/tests/test_features_engineer.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add v2/features/engineer.py v2/tests/test_features_engineer.py
git commit -m "v2: feature engineer — compute_features, 41-dim output"
```

---

## Task 3: Update `KlineWindowDataset` — `apply_features` and `return_targets`

**Files:**
- Modify: `v2/data/dataset.py`
- Modify: `v2/tests/test_dataset.py`

- [ ] **Step 1: Write new failing tests (append to existing test_dataset.py)**

Add these tests at the end of `v2/tests/test_dataset.py`. The existing tests remain unchanged — they pass no `funding_path`/`liq_path`, so the feature engineering path is never triggered.

```python
# ---- New tests for apply_features and return_targets (Plan 2) ----

import pandas as pd as _pd  # already imported above as pd — re-use existing import
from v2.data.constants import FUNDING_COLUMNS, FUNDING_DTYPES, LIQ_BUCKETED_COLUMNS, LIQ_BUCKETED_DTYPES
from v2.data.store import write_funding, write_liq_bucketed
from v2.features.constants import N_FEATURES


def _write_synthetic_funding(tmp_path: Path, n_events: int, start_ms: int = 0) -> Path:
    interval_ms = 8 * 60 * 60 * 1000  # 8h
    df = pd.DataFrame({
        "funding_time": pd.array(
            [start_ms + i * interval_ms for i in range(n_events)], dtype="int64"
        ),
        "funding_rate": [0.0001] * n_events,
        "mark_price": [100.0] * n_events,
    })
    p = tmp_path / "funding_btcusdt.parquet"
    write_funding(df, p)
    return p


def _write_synthetic_liq(tmp_path: Path, n_bars: int, start_ms: int = 0) -> Path:
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
    # Last bar in the dataset has no "next close", so its target log-return = 0.
    p = _write_synthetic(tmp_path, n_bars=50)
    fp = _write_synthetic_funding(tmp_path, n_events=5)
    lp = _write_synthetic_liq(tmp_path, n_bars=50)
    ds = KlineWindowDataset(p, window=10, stride=1, funding_path=fp, liq_path=lp,
                            apply_features=True, return_targets=True)
    _, log_rets = ds[len(ds) - 1]
    # The last bar of the last window (bar index 49 in the 50-bar file) has target 0.
    assert float(log_rets[-1]) == pytest.approx(0.0, abs=1e-7)


def test_columns_property_with_features(tmp_path: Path):
    from v2.features.constants import FEATURE_COLUMNS
    p = _write_synthetic(tmp_path, n_bars=200)
    fp = _write_synthetic_funding(tmp_path, n_events=10)
    lp = _write_synthetic_liq(tmp_path, n_bars=200)
    ds = KlineWindowDataset(p, window=10, stride=1, funding_path=fp, liq_path=lp,
                            apply_features=True)
    assert ds.columns == FEATURE_COLUMNS
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
uv run pytest v2/tests/test_dataset.py -k "apply_features or return_targets or columns_property" -v
```

Expected: FAIL with `TypeError` (unexpected keyword argument `apply_features`).

- [ ] **Step 3: Update `v2/data/dataset.py`**

Replace the file with:

```python
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
```

- [ ] **Step 4: Run full test suite to verify everything passes**

```bash
uv run pytest v2/tests/ -v
```

Expected: all existing tests pass (no regression) + 5 new dataset tests pass.

- [ ] **Step 5: Commit**

```bash
git add v2/data/dataset.py v2/tests/test_dataset.py
git commit -m "v2: dataset — apply_features=True→41-dim, return_targets for training"
```

---

## Task 4: Final verification — full test suite gate

- [ ] **Step 1: Run every test**

```bash
cd projects/candle-gpt
uv run pytest v2/tests/ -v --tb=short
```

Expected: all tests pass (76 prior + ~21 new = ~97 total), 0 failed.

- [ ] **Step 2: Smoke test — verify the full dataset pipeline on real data**

```bash
uv run python -c "
from pathlib import Path
from v2.data.dataset import KlineWindowDataset

RAW = Path('v2/data/raw')
ds = KlineWindowDataset(
    path=RAW / 'btcusdt_1m.parquet',
    window=512,
    stride=512,
    funding_path=RAW / 'funding_btcusdt.parquet',
    liq_path=RAW / 'liq_btcusdt_per_minute.parquet',
    apply_features=True,
    return_targets=True,
)
feats, rets = ds[0]
print(f'Dataset length: {len(ds)}')
print(f'features shape: {tuple(feats.shape)}')   # (512, 41)
print(f'targets shape:  {tuple(rets.shape)}')    # (512,)
print(f'Any NaN: {feats.isnan().any().item()}')  # False
print('OK')
"
```

Expected output:
```
Dataset length: ~4100
features shape: (512, 41)
targets shape: (512,)
Any NaN: False
OK
```

- [ ] **Step 3: Tag milestone**

```bash
git tag v2-features-vector
```

---

## What's next (Plan 3)

Plan 3 defines the `CandleGPTv2` transformer model that consumes the (batch, window, 41) feature tensor and predicts the next-bar return distribution via a 256-bin softmax. No training in Plan 3 — only the forward pass, a `ReturnTokenizerV2`, and tests confirming output shapes + ~10.9M param count.
