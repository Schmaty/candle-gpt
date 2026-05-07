# Lowering CandleGPT Loss Asymptote While Keeping Output Shape

Date: 2026-05-07

Goal: lower train/validation loss without changing the general inference output: `CandleGPTv2.forward(features) -> logits over n_bins tokenized future return bins`.

Current reference run: `20260506_32m_fixed_clean_forecast`
- Objective: final-timestep cross entropy over 256 quantile bins.
- Random/uniform CE baseline: `ln(256) = 5.545`.
- Best val: `5.4624` at step 9000.
- Plateau stopped at step 24000.
- Top-1 bin accuracy is not very informative for 256 ordered bins; report currently scores many non-final positions, so eval/report should be aligned with forecast-only validation.

## Key diagnosis

The model output treats return bins as unrelated categories, but bins are ordered samples from a continuous return distribution. Standard hard CE gives the same penalty shape for predicting the adjacent return bin versus a far-tail crash bin except through the target class probability. This wastes structure and encourages unstable overconfidence. The first asymptote-lowering experiments should preserve the 256-bin output but train it with ordinal/distribution-aware losses and better regularization/evaluation.

## Ranked experiments

### 1. Soft ordinal labels / Gaussian-smoothed CE

**Idea:** keep 256 logits, but replace one-hot target with a small distribution around the true bin. Adjacent bins get partial credit. Use KL/soft CE:

`loss = -sum(target_dist * log_softmax(logits))`

**Why:** return bins are ordered; a miss by 1-2 bins is not the same as a miss by 100 bins. Literature on ordinal losses/soft labels supports using probabilistic target distributions for ordinal categories.

**Expected impact:** medium-high. Likely smoother validation and lower asymptote; may reduce exact top-1 accuracy but improve NLL/calibration/directional metrics.

**Risk:** if smoothing is too wide, model learns the unconditional prior and becomes mushy. Tune sigma/eta.

**Implementation sketch:**
- Add `loss_type: str = "ce"` and `soft_label_sigma_bins: float = 1.5` to `TrainConfig`.
- Add helper in `v2/train/losses.py`:
  - `ordinal_soft_targets(ids, n_bins, sigma)`
  - `soft_cross_entropy(logits, soft_targets)`
- Replace CE in train/eval with this helper when enabled.
- Keep output logits unchanged.

**First settings:** sigma 1.0, 2.0, 3.0; compare hard CE validation as a reported metric too.

### 2. Add auxiliary regression heads only during training

**Idea:** keep final output as 256-bin logits, but add temporary training losses from hidden state:
- decoded expected return MSE/Huber
- direction BCE
- absolute return / volatility bucket

At inference, the API can keep returning the same bin logits. Auxiliary heads can remain unused or be ignored.

**Why:** CE alone gives sparse gradient to one bin. Auxiliary continuous/directional losses teach the representation smoother market structure.

**Expected impact:** medium. Especially useful if model is learning the return distribution but not sign/magnitude structure.

**Risk:** too much aux weight can optimize direction at the expense of NLL. Keep aux weight small.

**Implementation sketch:**
- Add optional heads to `CandleGPTv2`: `direction_head`, `return_head`, maybe `vol_head` from final hidden state. Or expose `forward(..., return_hidden=True)`.
- Compute true continuous final log return from batch before tokenization.
- Loss: `ce + 0.05*huber(expected_return, ret) + 0.05*bce(direction)`.
- Output interface remains logits unless a debug flag asks for aux outputs.

### 3. Multi-timeframe features, same target/head

**Idea:** keep target as next 5m return bin, but append causal features from 15m/1h contexts: longer RV, trend slope, distance to higher-timeframe VWAP/MA, rolling high/low breakout state.

**Why:** current features are mostly same-timeframe. BTC predictability often depends on regime/volatility context more than local candle geometry.

**Expected impact:** medium. Better regime awareness can lower the real asymptote more than optimizer tweaks.

**Risk:** easy to introduce accidental future leakage during resampling. Must use only closed higher-timeframe bars or features lagged by one coarse bucket.

**Implementation sketch:**
- In dataset/feature engineering, compute higher-timeframe bars from 1m source.
- Join as-of with `direction="backward"` and shift closed HTF bars if needed.
- Append features: `rv_15m_20`, `rv_1h_20`, `trend_1h`, `close_vs_1h_ma`, `htf_range_position`.
- Increase `ModelConfig.n_features` accordingly.

### 4. Regime-conditioned output bias / adapters

**Idea:** same 256 logits, but condition head on regime/volatility bucket:
- add learned regime embedding to every token
- or add small per-regime logit bias vector
- or mixture-of-experts head gated by regime features

**Why:** unconditional quantile bins hide different distributions. Chop/trend/high-vol have different next-return priors.

**Expected impact:** medium. May lower NLL by matching priors per regime even when directional signal is weak.

**Risk:** regime labels are currently simple; bad regimes can overfit. Need walk-forward evaluation.

**Implementation sketch:**
- Derive regime id from existing one-hot/regime features or volatility quantile.
- Add `regime_embed = nn.Embedding(n_regimes, d_model)` or `logit_bias = nn.Embedding(n_regimes, n_bins)`.
- Forward still returns `(B,T,256)` logits.

### 5. Self-supervised / synthetic curriculum pretraining

**Idea:** pretrain same transformer body on either:
- masked feature reconstruction / denoising time-series objective, or
- synthetic market regimes, then fine-tune on real 256-bin CE.

**Why:** time-series masked autoencoding is used to learn stronger representations before forecasting. Synthetic data can teach candle/regime mechanics if noisy and realistic.

**Expected impact:** uncertain-medium. Helps representation; won’t create real BTC edge by itself.

**Risk:** synthetic data that is too clean teaches fake predictability; validation loss can get worse after transfer.

**Implementation sketch:**
- Create pretrain mode with same input projection/transformer, temporary reconstruction head.
- Pretrain on noisy synthetic + real masked windows.
- Fine-tune with normal CandleGPT head and real validation only.

## Evaluation fixes before trusting results

1. Align `v2/train/eval.py` with forecast-only objective. Current report evaluates all non-last positions, while validation trains/scores only final timestep. Add final-token NLL, directional accuracy, expected-return MAE, EMD/bin-distance.
2. Always report hard CE even when training with soft CE, so experiments are comparable.
3. Add naive baselines:
   - unconditional train-bin prior NLL
   - per-regime prior NLL
   - last-return sign/magnitude simple baseline
4. Use walk-forward splits or multiple date folds; a single split can overfit market period.
5. Keep leakage guard tests and add a higher-timeframe causal-join test before adding HTF features.

## Recommended next experiment

Start with the smallest code change: **soft ordinal labels + better final-token eval metrics**.

Proposed run after implementation:

```bash
uv run python -m v2.train.run \
  --run-id 20260507_32m_softordinal_sigma2 \
  --batch-size 8 \
  --window 768 \
  --interval 5m \
  --stride-train 16 \
  --lr-max 8e-5 \
  --max-steps 50000 \
  --max-wall-clock-h 0 \
  --early-stop-patience-evals 20 \
  --early-stop-min-delta 0.0005 \
  --loss-type soft_ce \
  --soft-label-sigma-bins 2.0
```

If this improves final-token hard CE or bin-distance without hurting calibration, then layer in auxiliary regression/direction losses.
