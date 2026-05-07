"""Synthetic market-data generator for CandleGPT curriculum pretraining.

This does not change the model output. It creates messy, regime-switching 1m
OHLCV/funding/liquidation parquets with the same schemas as real v2 data, so a
normal CandleGPT training run can pretrain on synthetic data and then fine-tune
on real data via --resume-from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v2.data.store import write_klines, write_funding, write_liq_bucketed


_REGIME_PARAMS = {
    # mu/min, sigma/min, mean-reversion coefficient, jump probability
    0: (0.0, 0.00075, 0.08, 0.0005),   # chop / mean-revert
    1: (0.00003, 0.00115, 0.01, 0.0010),  # trend / drift
    2: (0.0, 0.00240, 0.02, 0.0040),   # high-vol / liquidation regime
}


def _sample_regimes(n: int, rng: np.random.Generator) -> np.ndarray:
    regimes = np.empty(n, dtype=np.int8)
    t = 0
    current = int(rng.choice([0, 1, 2], p=[0.62, 0.24, 0.14]))
    while t < n:
        # Regime dwell time: 2h to ~2d, heavy-tailed enough to create trends.
        dwell = int(np.clip(rng.lognormal(mean=6.2, sigma=0.9), 120, 2880))
        end = min(n, t + dwell)
        regimes[t:end] = current
        t = end
        if current == 0:
            current = int(rng.choice([1, 2], p=[0.72, 0.28]))
        elif current == 1:
            current = int(rng.choice([0, 2], p=[0.75, 0.25]))
        else:
            current = int(rng.choice([0, 1], p=[0.82, 0.18]))
    return regimes


def generate_synthetic_market(
    *,
    n_minutes: int,
    seed: int = 7,
    start_price: float = 80_000.0,
    start_time_ms: int = 1_735_689_600_000,  # 2025-01-01 UTC
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic 1m kline, funding, and liquidation DataFrames."""
    rng = np.random.default_rng(seed)
    regimes = _sample_regimes(n_minutes, rng)

    open_time = start_time_ms + np.arange(n_minutes, dtype=np.int64) * 60_000
    close_time = open_time + 59_999

    log_price = np.empty(n_minutes, dtype=np.float64)
    log_price[0] = np.log(start_price)
    returns = np.zeros(n_minutes, dtype=np.float64)
    trend_state = 0.0

    for i in range(1, n_minutes):
        r = int(regimes[i])
        mu, sigma, mr, jump_p = _REGIME_PARAMS[r]
        if r == 1:
            # Persistent but noisy trend state; sign flips are possible.
            trend_state = 0.997 * trend_state + rng.normal(mu, sigma * 0.08)
        else:
            trend_state *= 0.96
        prev_ret = returns[i - 1]
        noise = rng.normal(0.0, sigma)
        mean_revert = -mr * prev_ret
        jump = 0.0
        if rng.random() < jump_p:
            jump = rng.normal(0.0, sigma * rng.uniform(6.0, 18.0))
        returns[i] = trend_state + mean_revert + noise + jump
        # Keep paths messy but bounded enough for stable synthetic candles.
        returns[i] = float(np.clip(returns[i], -0.06, 0.06))
        log_price[i] = log_price[i - 1] + returns[i]

    close = np.exp(log_price)
    open_ = np.concatenate([[close[0]], close[:-1]])
    abs_ret = np.abs(returns)
    wick_scale = rng.gamma(shape=1.5, scale=0.0009, size=n_minutes) + abs_ret * rng.uniform(0.2, 1.8, n_minutes)
    high = np.maximum(open_, close) * (1.0 + wick_scale)
    low = np.minimum(open_, close) * np.maximum(0.01, 1.0 - wick_scale * rng.uniform(0.6, 1.4, n_minutes))

    base_vol = rng.lognormal(mean=6.8, sigma=0.55, size=n_minutes)
    regime_vol_mult = np.choose(regimes, [0.8, 1.15, 2.5])
    volume = base_vol * regime_vol_mult * (1.0 + 55.0 * abs_ret)

    klines = pd.DataFrame({
        "open_time": open_time.astype("int64"),
        "open": open_.astype("float64"),
        "high": high.astype("float64"),
        "low": low.astype("float64"),
        "close": close.astype("float64"),
        "volume": volume.astype("float64"),
        "close_time": close_time.astype("int64"),
        "regime": regimes.astype("int8"),
    })

    funding_step = 8 * 60
    funding_idx = np.arange(0, n_minutes, funding_step, dtype=np.int64)
    # Funding is loosely tied to recent trend and volatility, but noisy.
    roll_ret = pd.Series(returns).rolling(funding_step, min_periods=1).mean().to_numpy()[funding_idx]
    funding_rate = np.clip(roll_ret * 8.0 + rng.normal(0.0, 0.00008, len(funding_idx)), -0.003, 0.003)
    mark_price = close[funding_idx] * (1.0 + rng.normal(0.0, 0.0004, len(funding_idx)))
    funding = pd.DataFrame({
        "funding_time": open_time[funding_idx].astype("int64"),
        "funding_rate": funding_rate.astype("float64"),
        "mark_price": mark_price.astype("float64"),
    })

    high_vol = regimes == 2
    liq_intensity = 0.04 + 80.0 * abs_ret + high_vol.astype(float) * 0.35
    count = rng.poisson(liq_intensity).astype(np.int64)
    side_bias_long = 1.0 / (1.0 + np.exp(-returns * 3500.0))
    long_count = rng.binomial(count, np.clip(1.0 - side_bias_long, 0.05, 0.95)).astype(np.int64)
    short_count = (count - long_count).astype(np.int64)
    avg_notional = rng.lognormal(mean=9.0, sigma=1.0, size=n_minutes) * (1.0 + 120.0 * abs_ret)
    sum_notional = count * avg_notional
    long_notional = sum_notional * np.divide(long_count, count, out=np.zeros_like(sum_notional), where=count > 0)
    short_notional = sum_notional - long_notional
    max_single = np.where(count > 0, sum_notional / np.maximum(1, count) * rng.uniform(0.7, 2.3, n_minutes), 0.0)
    liq = pd.DataFrame({
        "bucket_time": open_time.astype("int64"),
        "count": count.astype("int64"),
        "sum_notional": sum_notional.astype("float64"),
        "max_single": max_single.astype("float64"),
        "long_liq_count": long_count.astype("int64"),
        "long_liq_notional": long_notional.astype("float64"),
        "short_liq_count": short_count.astype("int64"),
        "short_liq_notional": short_notional.astype("float64"),
    })

    return klines, funding, liq


def write_synthetic_raw_dir(raw_dir: Path, *, n_minutes: int, seed: int = 7) -> None:
    klines, funding, liq = generate_synthetic_market(n_minutes=n_minutes, seed=seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_klines(klines, raw_dir / "btcusdt_1m.parquet")
    write_funding(funding, raw_dir / "funding_btcusdt.parquet")
    write_liq_bucketed(liq, raw_dir / "liq_btcusdt_per_minute.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic CandleGPT v2 raw parquets.")
    ap.add_argument("--raw-dir", type=Path, default=Path("v2/data/synthetic_raw"))
    ap.add_argument("--n-minutes", type=int, default=180 * 24 * 60,
                    help="Number of 1m bars to generate; default is 180 days.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    write_synthetic_raw_dir(args.raw_dir, n_minutes=args.n_minutes, seed=args.seed)
    print(f"Wrote synthetic raw parquets to {args.raw_dir} ({args.n_minutes:,} minutes)")


if __name__ == "__main__":
    main()
