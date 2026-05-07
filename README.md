# CandleGPT

**CandleGPT** is a compact transformer research project for predicting tokenized BTC candle movements from market microstructure features. It includes a FastAPI inference server, a Vite/React dashboard, data pipelines, training loops, leakage-guard tests, and checkpoint reload tooling.

> Research/education project — not financial advice and not a production trading system.

## What it does

- Builds candle datasets from BTC market data at configurable intervals such as `5m`.
- Engineers causal-only features for forecasting future candle movement.
- Trains a ~32M parameter GPT-style transformer on tokenized candle targets.
- Tracks training progress, validation loss, checkpoints, and live inference state.
- Serves predictions through a FastAPI backend.
- Provides a web dashboard for live candles, training progress, backtesting, calibration, regimes, and history.

## Current model family

The active v2 model is a **32M parameter CandleGPT** variant:

- Transformer decoder architecture
- 10 layers
- 8 attention heads
- 512 hidden dimension
- 256 target bins
- 768-candle context window
- Forecast-only final-timestep training loss
- Clean target split gaps to reduce leakage
- Causal feature engineering guards

Large checkpoint files are intentionally not committed to git. They should be shared through GitHub Releases, external storage, or Git LFS.

## Repository layout

```text
v2/
  data/        Dataset fetching, storage, validation, regimes, liquidations
  features/    Causal feature engineering
  model/       Transformer config and model code
  train/       Training loop, eval, progress tracking
  server/      FastAPI inference/training endpoints
  tests/       Pytest suite, including leakage guards
  web/         Vite + React dashboard

docs/
  plans/       Design notes and implementation plans
  refs/        Reference UI/architecture material
```

## Quick start

```bash
uv sync
uv run pytest v2/tests
```

Run the API server:

```bash
uv run uvicorn v2.server.main:app --reload --port 8766
```

Run the web dashboard:

```bash
cd v2/web
npm install
npm run dev
```

Train a v2 model example:

```bash
uv run python -m v2.train.run \
  --run-id candle-gpt-32m \
  --batch-size 8 \
  --window 768 \
  --interval 5m \
  --stride-train 16 \
  --lr-max 1e-4 \
  --max-steps 80000
```

## Safety checks

The project includes tests to guard against target leakage and accidental future information in features/training splits:

```bash
uv run pytest v2/tests/test_train_leakage_guards.py v2/tests/test_features_engineer.py
```

## Notes

- Raw market data, checkpoints, runs, and `node_modules` are ignored.
- Checkpoints can be hundreds of MB each, so use Releases/LFS instead of normal git commits.
- This repo is for experimentation and learning, not automated financial decision-making.
