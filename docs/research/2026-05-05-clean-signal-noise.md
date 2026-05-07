# Candle GPT — Cleanest Signal-to-Noise Research

Date: 2026-05-05

## Executive conclusion

For BTC next-candle prediction, the cleanest SNR will not come from a bigger transformer. It will come from changing the **sampling, target, and filtering problem**:

1. Stop treating every fixed-time candle as equally informative.
2. Stop training only on next-bar log-return bins.
3. Add trade-aware labels: volatility-scaled triple-barrier / meta-labeling.
4. Add microstructure/order-flow features if available.
5. Train regime-conditional heads and only trade high-confidence events.

The current Candle GPT target is mostly market micro-noise: next-bar return bin over 1m/5m bars. Bigger model size amplifies the noise unless the target is cleaned.

## What external research says

### 1. Information-driven bars beat arbitrary time bars

Recent crypto research explicitly argues that time bars miss market activity and that information-driven bars — volume, dollar, range, CUSUM/event bars — better align samples with actual market information. The 2025 Financial Innovation paper compares time sampling against information-driven sampling and triple-barrier labeling for crypto algorithmic trading.

Takeaway for Candle GPT: implement dollar/volume/range bars, not only 1m/5m OHLCV resampling.

### 2. Triple-barrier labels are cleaner than next-bar labels

Lopez de Prado-style triple-barrier labels use:
- upper profit barrier,
- lower stop barrier,
- vertical time barrier.

This labels actual tradable events instead of asking “what is the next candle close?” Research and practice repeatedly use this to reduce label noise and make targets match execution.

Takeaway: Candle GPT should have a second target/head: `barrier_label ∈ {up_hit, down_hit, no_hit}` plus optional time-to-hit / expected return.

### 3. Meta-labeling improves precision by filtering false positives

Meta-labeling asks a second model: “Given a candidate primary signal, should we take it?” This often improves precision/Sharpe even when the primary signal is noisy. But it needs a non-awful primary signal and contextual features.

Takeaway: use the transformer distribution as the primary signal, then train a meta-filter over model confidence, entropy, regime, vol, funding, liquidation, trend/momentum, and order-flow features.

### 4. Fractional differentiation can preserve memory while making features more stationary

Financial price series are non-stationary; raw prices produce spurious relationships. Fractional differentiation is used to make series closer to stationary without fully destroying memory like simple returns can.

Takeaway: add fracdiff close/log_close features and test ADF/correlation tradeoff.

### 5. Order book / order-flow imbalance is probably the highest-value missing feature

Crypto short-horizon price movement is heavily microstructure-driven. Recent LOB papers and order-flow work emphasize that supply/demand imbalance, depth, trade direction, and order flow often dominate macro/technical features for short horizon prediction.

Takeaway: if we want truly clean short-horizon SNR, Candle GPT needs Binance futures book/trade features:
- bid/ask depth imbalance over top 1/5/10/20 levels,
- order-flow imbalance (OFI),
- trade aggression: buyer/seller initiated volume,
- spread, microprice, mid-price return,
- depth slope / liquidity walls,
- cancellation/refresh rates if possible.

## What Candle GPT data shows

Quick diagnostics on current BTCUSDT data:

- 1m returns: std ≈ 7.13 bps, mean/std ≈ 0.00047. Direction is ~48.5% up, essentially coin flip.
- 5m returns: std ≈ 16.0 bps, direction ~49.9% up.
- 15m/1h horizons increase volatility, but directional edge remains tiny.
- Return autocorrelations are near zero. Example 5m: lag-1 ≈ -0.0229, lag-2 ≈ -0.0130, then mostly tiny.
- Current engineered features have univariate IC mostly around 1–2%, which is weak but not zero.

Top simple 5m IC features:
- H=1: lag return / candle body mean-reversion; IC around -0.02.
- H=12–24: realized volatility and range become more predictive; IC around +0.016 to +0.019.

Interpretation: there is weak signal, but it is not clean enough for raw next-bar classification to shine. We need event filtering and better labels.

## Recommended SNR roadmap

### Phase 1 — Label cleanup (highest ROI)

Add a target module that can generate:

1. `future_return_h`: cumulative return over h ∈ {3,6,12,24} bars.
2. `direction_h`: sign of volatility-normalized return, with neutral deadzone.
3. `triple_barrier_label`: +1 / -1 / 0 using trailing realized vol.
4. `time_to_barrier`: bars until first barrier hit.
5. `realized_vol_h`: for risk-adjusted sizing.

Default suggested barrier grid for 5m bars:
- horizon H = 12 or 24 bars (1–2h),
- barrier k = 0.5σ for frequent events, k = 1.0σ for higher precision,
- neutral/no-trade class retained.

From quick sampling:
- 5m, H=24, k=1.0 → ~20.4% up, ~20.4% down, ~59.2% neutral.
- This is much cleaner as a trading/no-trade problem than 256-bin next-bar prediction.

### Phase 2 — Event sampling

Implement at least:
- volume bars,
- dollar bars,
- range bars,
- CUSUM event bars.

Train/evaluate the same labels on each. Pick the sampling method by out-of-sample expected value / calibration, not by validation loss alone.

### Phase 3 — Model objective upgrade

Replace single next-token return-bin loss with multitask heads:

- return distribution head, existing 256-bin model,
- barrier direction head: up/down/neutral,
- meta-label head: trade/no-trade,
- volatility head: expected realized vol / uncertainty,
- optional regime head.

Loss:
- cross entropy for barrier labels,
- focal loss or class-weighted CE for imbalanced event labels,
- KL/calibration regularization for distribution head,
- maybe ordinal loss for return bins instead of treating bins as unrelated classes.

### Phase 4 — Microstructure features

Backfill/live collect:
- Binance futures aggTrades,
- bookTicker,
- partial depth snapshots or diff depth,
- open interest if available,
- long/short ratio if available.

Feature groups:
- OFI: Δbid_size - Δask_size adjusted for price moves,
- OBI: (bid_depth - ask_depth)/(bid_depth + ask_depth), by depth bucket,
- microprice - midprice,
- spread bps,
- market buy/sell volume imbalance,
- large trade imbalance,
- liquidation shock decays.

### Phase 5 — Regime-specific modeling

The same signal has different meaning by regime:
- trend: continuation/momentum may matter,
- mean-revert: fades may matter,
- high-vol/liquidation: order-flow + volatility dominates.

Implement either:
- separate heads per regime, or
- a gating network conditioned on regime.

### Phase 6 — Evaluation that matches trading

Do not optimize only top-1 bin accuracy. Track:
- directional accuracy conditional on confidence,
- precision/recall for up/down barrier events,
- no-trade rejection quality,
- expected return after fees/slippage,
- calibration/ECE,
- profit factor,
- max drawdown,
- Sharpe/Sortino,
- per-regime metrics.

## Concrete next build plan

1. Add `v2/labels/barrier.py` with fixed-horizon + triple-barrier labels.
2. Add tests for label correctness/no lookahead.
3. Add label distribution report for 1m/5m/15m.
4. Add barrier-head model or simple baseline first: LightGBM/logistic/MLP on engineered features.
5. Compare against current transformer:
   - current next-bin model,
   - barrier classifier,
   - meta-label filter on top of transformer confidence.
6. Only then train the 32M transformer on the cleaned multitask objective.

## Sources consulted

- Hudson & Thames / MLFinLab: fractional differentiation, CUSUM filters, triple-barrier labeling.
- Hudson & Thames: meta-labeling and event-based sampling discussion.
- Financial Innovation 2025: Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning.
- arXiv 2411.12753: Supervised autoencoders with fractionally differentiated features and triple barrier labeling on noisy crypto data.
- Recent crypto LOB/order-flow search results: order book imbalance and order flow are central for short-horizon crypto predictability.
