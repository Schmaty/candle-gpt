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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--lr-max", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--raw-dir", type=Path, default=Path("v2/data/raw"))
    ap.add_argument("--runs-dir", type=Path, default=Path("v2/runs"))
    ap.add_argument("--eval-only", type=str, default=None)
    ap.add_argument("--max-wall-clock-h", type=float, default=None,
                    help="Override wall-clock cap in hours (default: TrainConfig's 6h)")
    args = ap.parse_args()

    cfg = TrainConfig(
        raw_dir=args.raw_dir,
        runs_dir=args.runs_dir,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr_max=args.lr_max,
        window=args.window,
    )
    if args.max_wall_clock_h is not None:
        cfg.max_wall_clock_s = args.max_wall_clock_h * 3600.0
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
    print(f"Max wall clock: {cfg.max_wall_clock_s/3600:.1f}h")

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
