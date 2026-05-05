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

    # --- Group K (multi-lag returns; depend on log_return) ---
    # log_return_k = log(c_t / c_{t-k}) — explicit short-horizon momentum
    # signal at the input layer, instead of forcing the model to derive it
    # via attention. Lags 3 and 10 complement the lag-1 already in `log_return`.
    log_return_3 = np.log(c / c.shift(3)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    log_return_10 = np.log(c / c.shift(10)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    high_low_range = (h - l) / c
    close_open_range = (c - o) / o
    candle_body_ratio = ((c - o).abs() / (h - l + 1e-8)).clip(upper=1.0)

    # --- Group B: Volatility ---
    realized_vol_5 = log_return.rolling(5, min_periods=1).std().fillna(0.0)
    realized_vol_12 = log_return.rolling(12, min_periods=1).std().fillna(0.0)
    realized_vol_20 = log_return.rolling(20, min_periods=1).std().fillna(0.0)
    realized_vol_60 = log_return.rolling(60, min_periods=1).std().fillna(0.0)
    atr14 = _atr(h, l, c, period=14)
    atr_14_norm = (atr14 / c).fillna(0.0)

    # --- Group C: Volume ---
    log_volume = np.log1p(v)
    # signed_log_volume: log_volume * sign(log_return_1) — captures whether
    # moves are happening on real volume or thin tape.
    signed_log_volume = log_volume * np.sign(log_return)
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

    def _arr(s):
        if hasattr(s, "to_numpy"):
            return s.to_numpy()
        return np.asarray(s)

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
            "hour_sin": _arr(hour_sin),
            "hour_cos": _arr(hour_cos),
            "dow_sin": _arr(dow_sin),
            "dow_cos": _arr(dow_cos),
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
            "log_return_3": log_return_3,
            "log_return_10": log_return_10,
            "signed_log_volume": signed_log_volume,
            "realized_vol_12": realized_vol_12,
        },
        index=df.index,
    )

    return out.fillna(0.0).replace([np.inf, -np.inf], 0.0)
