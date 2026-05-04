"""Background poller: watches a training run's checkpoint directory and runs
a small sweep + backtest on every new step_*.pt that appears, appending
results to v2/runs/<run_id>/eval_history.jsonl.

Designed to run as a long-lived sidecar to a training process. Each iteration:
  1. Discover any step_NNNN.pt that hasn't been evaluated yet.
  2. Build a one-off SweepService over that checkpoint.
  3. Run a sweep over a small (T, H) grid → pick best (T, H).
  4. Run a backtest at that (T, H) over the full test set.
  5. Append a JSON line with all the numbers + a wall-clock ts.

Usage:
    python -m v2.train.poll_eval --run-id 20260504_xxxxxx --runs-dir v2/runs

Stops automatically when the training process dies and no new checkpoints
appear for `--idle-timeout` seconds (default 1800s = 30 min).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from v2.server.sweep import SweepService

log = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"step_0*(\d+)\.pt$")
DEFAULT_TEMPS = [0.5, 0.8, 1.0, 1.5, 2.0]
DEFAULT_HORIZONS = [1, 3, 5, 10, 20, 30]
DEFAULT_N_SAMPLES = 150


def _checkpoints_in(ckpt_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = CHECKPOINT_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def _read_seen_steps(history_path: Path) -> set[int]:
    seen: set[int] = set()
    if not history_path.exists():
        return seen
    with history_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(int(json.loads(line)["step"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def _evaluate_one(svc: SweepService, step: int, ckpt_path: Path,
                  temperatures: list[float], horizons: list[int],
                  n_samples: int) -> dict:
    sweep_t0 = time.time()
    sweep_res = svc.sweep(temperatures=temperatures, horizons=horizons, n_samples=n_samples)
    sweep_dt = time.time() - sweep_t0

    best = sweep_res.get("best") or {}
    best_T = float(best.get("temperature", 1.0))
    best_H = int(best.get("horizon", 30))
    best_dir_acc = best.get("dir_acc")

    bt_t0 = time.time()
    bt = svc.backtest(temperature=best_T, horizon=best_H, z_threshold=0.0,
                      start_frac=0.0, end_frac=1.0, fee_bps=1.0)
    bt_dt = time.time() - bt_t0

    return {
        "ts": time.time(),
        "step": int(step),
        "ckpt": str(ckpt_path),
        "sweep_seconds": round(sweep_dt, 2),
        "backtest_seconds": round(bt_dt, 2),
        "best_T": best_T,
        "best_H": best_H,
        "best_dir_acc": best_dir_acc,
        "all_sweep_results": sweep_res.get("results", []),
        "backtest": {
            "trades": bt["trades"],
            "longs": bt["longs"],
            "shorts": bt["shorts"],
            "win_rate": bt["win_rate"],
            "total_return_pct": bt["total_return_pct"],
            "sharpe_per_trade": bt["sharpe_per_trade"],
            "max_drawdown_pct": bt["max_drawdown_pct"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Periodic sweep+backtest on new training checkpoints")
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--runs-dir", type=Path, default=Path("v2/runs"))
    ap.add_argument("--raw-dir", type=Path, default=Path("v2/data/raw"))
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--idle-timeout", type=int, default=1800,
                    help="Exit if no new checkpoint shows up for this many seconds")
    ap.add_argument("--temperatures", type=str, default=",".join(str(t) for t in DEFAULT_TEMPS))
    ap.add_argument("--horizons", type=str, default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [poll_eval] %(message)s")
    run_dir = args.runs_dir / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    history_path = run_dir / "eval_history.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    temps = [float(x) for x in args.temperatures.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    log.info(f"watching {ckpt_dir} → writing history to {history_path}")
    log.info(f"sweep grid: temps={temps} horizons={horizons} n_samples={args.n_samples}")

    seen = _read_seen_steps(history_path)
    log.info(f"already-evaluated steps: {len(seen)}")
    last_progress = time.time()

    while True:
        if not ckpt_dir.exists():
            time.sleep(args.poll_seconds)
            continue
        ckpts = _checkpoints_in(ckpt_dir)
        new = [(s, p) for s, p in ckpts if s not in seen]
        if new:
            log.info(f"found {len(new)} new checkpoint(s)")
        for step, path in new:
            try:
                # Fresh SweepService per checkpoint so it loads the new weights.
                svc = SweepService(
                    run_dir=run_dir,
                    kline_path=args.raw_dir / "btcusdt_1m.parquet",
                    funding_path=args.raw_dir / "funding_btcusdt.parquet",
                    liq_path=args.raw_dir / "liq_btcusdt_per_minute.parquet",
                    ckpt_filename=path.name,
                )
                rec = _evaluate_one(svc, step, path, temps, horizons, args.n_samples)
                with history_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                seen.add(step)
                last_progress = time.time()
                bt = rec["backtest"]
                log.info(
                    f"step {step}  best T={rec['best_T']} H={rec['best_H']} "
                    f"dir_acc={rec['best_dir_acc']}  "
                    f"backtest: trades={bt['trades']} win={bt['win_rate']:.3f} "
                    f"ret={bt['total_return_pct']:.2f}% sharpe={bt['sharpe_per_trade']:.3f}"
                )
            except Exception as e:
                log.error(f"eval failed for step {step}: {e}", exc_info=True)
                time.sleep(5)

        # Idle-timeout: if no new checkpoints for a long time, assume training ended.
        if time.time() - last_progress > args.idle_timeout:
            log.info(f"no progress for {args.idle_timeout}s — exiting")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
