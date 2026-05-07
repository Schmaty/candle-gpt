# CandleGPT Synthetic Curriculum Path

This repo now supports a synthetic pretraining stage without changing the model output shape.

## Why

The model still predicts the same tokenized return-bin logits. Synthetic data is only a curriculum: it can teach candle mechanics, volatility clustering, trends, mean reversion, jumps, and liquidation-like events before fine-tuning on real BTC data.

Synthetic metrics are not success metrics. Only real validation/test metrics matter.

## Generate synthetic raw data

```bash
uv run python -m v2.data.synthetic \
  --raw-dir v2/data/synthetic_raw \
  --n-minutes 259200 \
  --seed 7
```

`259200` minutes is ~180 days of 1m data. The generator writes the same files expected by normal training:

- `btcusdt_1m.parquet`
- `funding_btcusdt.parquet`
- `liq_btcusdt_per_minute.parquet`

## Synthetic pretrain command

Use the same output objective and optional improvements:

```bash
uv run python -m v2.train.run \
  --run-id 20260507_synth_pretrain_small \
  --raw-dir v2/data/synthetic_raw \
  --batch-size 16 \
  --window 512 \
  --interval 5m \
  --stride-train 8 \
  --lr-max 1e-4 \
  --max-steps 20000 \
  --max-wall-clock-h 0 \
  --loss-type soft_ce \
  --soft-label-sigma-bins 2.0 \
  --aux-return-loss-weight 0.02 \
  --aux-direction-loss-weight 0.02 \
  --regime-conditioning
```

## Real fine-tune command

```bash
uv run python -m v2.train.run \
  --run-id 20260507_real_finetune_from_synth_small \
  --resume-from v2/runs/20260507_synth_pretrain_small/checkpoints/best_val.pt \
  --batch-size 16 \
  --window 512 \
  --interval 5m \
  --stride-train 16 \
  --lr-max 5e-5 \
  --max-steps 50000 \
  --max-wall-clock-h 0 \
  --early-stop-patience-evals 20 \
  --early-stop-min-delta 0.0005 \
  --loss-type soft_ce \
  --soft-label-sigma-bins 2.0 \
  --aux-return-loss-weight 0.02 \
  --aux-direction-loss-weight 0.02 \
  --regime-conditioning
```

## Guardrails

- Do not mix synthetic and real validation metrics.
- Synthetic pretraining should be judged only by whether real fine-tune improves hard CE/NLL, bin-distance, direction accuracy, and calibration.
- Keep synthetic data messy. If it becomes too clean, it will teach fake predictability.
