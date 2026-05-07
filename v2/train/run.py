"""CLI entry point for CandleGPTv2 training.

Usage:
    nohup uv run python -m v2.train.run [--run-id ID] [--batch-size N] > v2/runs/latest_launch.log 2>&1 &
    echo $! > v2/runs/train.pid
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from v2.train.config import TrainConfig
from v2.train.loop import train
from v2.train.eval import evaluate, write_report

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CandleGPTv2")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--lr-max", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--interval", type=str, default="5m",
                    help="Bar timeframe; 1m parquet is OHLCV-resampled on the fly.")
    ap.add_argument("--stride-train", type=int, default=None,
                    help="Override training-window stride (default from TrainConfig).")
    ap.add_argument("--raw-dir", type=Path, default=Path("v2/data/raw"))
    ap.add_argument("--runs-dir", type=Path, default=Path("v2/runs"))
    ap.add_argument("--eval-only", type=str, default=None)
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="Resume model+optimizer state from a checkpoint.")
    ap.add_argument("--early-stop-patience-evals", type=int, default=None,
                    help="Stop after N validation checks without meaningful best-val improvement.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.001,
                    help="Minimum val-loss improvement counted for plateau stopping.")
    ap.add_argument("--max-wall-clock-h", type=float, default=None,
                    help="Override wall-clock cap in hours; 0 disables wall-clock stopping.")
    ap.add_argument("--loss-type", choices=["ce", "soft_ce"], default="ce",
                    help="Training loss: hard CE or ordinal Gaussian-smoothed soft CE.")
    ap.add_argument("--soft-label-sigma-bins", type=float, default=2.0,
                    help="Gaussian soft-label width in return-bin units for --loss-type soft_ce.")
    ap.add_argument("--aux-return-loss-weight", type=float, default=0.0,
                    help="Weight for training-only final-hidden-state return Huber loss.")
    ap.add_argument("--aux-direction-loss-weight", type=float, default=0.0,
                    help="Weight for training-only final-hidden-state direction BCE loss.")
    ap.add_argument("--regime-conditioning", action="store_true",
                    help="Add learned per-regime logit biases while preserving n_bins output shape.")
    args = ap.parse_args()

    cfg = TrainConfig(
        raw_dir=args.raw_dir,
        runs_dir=args.runs_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr_max=args.lr_max,
        window=args.window,
        interval=args.interval,
        loss_type=args.loss_type,
        soft_label_sigma_bins=args.soft_label_sigma_bins,
        aux_return_loss_weight=args.aux_return_loss_weight,
        aux_direction_loss_weight=args.aux_direction_loss_weight,
    )
    cfg.model.regime_conditioning = args.regime_conditioning
    if args.stride_train is not None:
        cfg.stride_train = args.stride_train
    # Keep model context length in sync with the training window so reductions
    # in --window (e.g. to fit memory) shrink positional embeddings too.
    if cfg.window != cfg.model.block_size:
        cfg.model.block_size = cfg.window
    if args.max_wall_clock_h is not None:
        # Use 0 as the internal "no wall-clock cap" sentinel. Avoid inf so
        # status.json remains strict JSON for the dashboard.
        cfg.max_wall_clock_s = 0.0 if args.max_wall_clock_h <= 0 else args.max_wall_clock_h * 3600.0
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.early_stop_patience_evals is not None:
        cfg.early_stop_patience_evals = args.early_stop_patience_evals
        cfg.early_stop_min_delta = args.early_stop_min_delta
    if args.run_id:
        cfg.run_id = args.run_id

    if args.eval_only:
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate(cfg, Path(args.eval_only))
        write_report(metrics, cfg.run_dir)
        print(f"Evaluation complete. Report: {cfg.report_path}")
        return

    print(f"Starting training run: {cfg.run_id}")
    print(f"Run dir: {cfg.run_dir}")
    wall = "none" if cfg.max_wall_clock_s <= 0 else f"{cfg.max_wall_clock_s/3600:.1f}h"
    print(f"Max wall clock: {wall}")

    run_id = train(cfg)
    print(f"Training finished. run_id={run_id}")

    if cfg.best_ckpt_path.exists():
        print(f"Evaluating best checkpoint: {cfg.best_ckpt_path}")
        metrics = evaluate(cfg, cfg.best_ckpt_path)
        write_report(metrics, cfg.run_dir)
        print(f"Report: {cfg.report_path}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"ECE: {metrics['ece']:.4f}")
        for r_id, stats in sorted(metrics["per_regime_accuracy"].items()):
            print(f"  Regime {r_id}: acc={stats['accuracy']:.4f}  n={stats['n']:,}")
    else:
        print("WARNING: No best_val checkpoint found — evaluation skipped.")


if __name__ == "__main__":
    main()
