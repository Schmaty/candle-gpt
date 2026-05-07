"""Post-training evaluation of the best checkpoint on the held-out test split."""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from v2.data.dataset import KlineWindowDataset
from v2.features.constants import FEATURE_COLUMNS
from v2.model.model import CandleGPTv2
from v2.model.config import ModelConfig
from v2.model.tokenizer import ReturnTokenizerV2
from v2.train.config import TrainConfig

log = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _collate(batch, tokenizer: ReturnTokenizerV2, device: torch.device):
    # batch is the DataLoader-collated result: (feats_batch, rets_batch)
    # feats_batch: (B, T, F), rets_batch: (B, T)
    feats_batch, rets_batch = batch
    ids_list = []
    for i in range(rets_batch.shape[0]):
        ids_list.append(torch.from_numpy(tokenizer.encode(rets_batch[i].numpy())))
    return feats_batch.to(device), torch.stack(ids_list).to(device)


def _clean_test_indices(cfg: TrainConfig, full_ds: KlineWindowDataset) -> list[int]:
    """Mirror v2.train.loop clean forecast split construction for test eval."""
    n_bars = full_ds._bars.shape[0]
    val_end_bar = int(n_bars * (cfg.train_frac + cfg.val_frac))
    gap = cfg.window if cfg.split_gap_bars is None else max(0, cfg.split_gap_bars)
    last_valid_target = n_bars - 2
    lo = max(cfg.window - 1, val_end_bar + gap)
    hi = last_valid_target + 1
    if hi <= lo:
        return []
    return [target_bar - cfg.window + 1 for target_bar in range(lo, hi, cfg.stride_val)]


@torch.no_grad()
def evaluate(cfg: TrainConfig, ckpt_path: Path) -> dict:
    device = _select_device()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = CandleGPTv2(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"Loaded checkpoint: {ckpt_path}  step={ckpt['step']}")

    tok = ReturnTokenizerV2.load(cfg.tokenizer_path)
    centers = torch.as_tensor(tok.decode(np.arange(tok.n_bins)), dtype=torch.float32, device=device)

    full_ds = KlineWindowDataset(
        path=cfg.kline_path,
        window=cfg.window,
        stride=1,
        funding_path=cfg.funding_path,
        liq_path=cfg.liq_path,
        apply_features=True,
        return_targets=True,
        interval=cfg.interval,
    )
    test_indices = _clean_test_indices(cfg, full_ds)
    test_ds = Subset(full_ds, test_indices)
    log.info(f"Test windows: {len(test_ds)}")

    regime_0_idx = list(FEATURE_COLUMNS).index("regime_0")
    regime_1_idx = list(FEATURE_COLUMNS).index("regime_1")
    regime_2_idx = list(FEATURE_COLUMNS).index("regime_2")

    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    all_correct: list[np.ndarray] = []
    all_top1_conf: list[np.ndarray] = []
    all_nll: list[np.ndarray] = []
    all_abs_bin_error: list[np.ndarray] = []
    all_within_1: list[np.ndarray] = []
    all_within_3: list[np.ndarray] = []
    all_within_5: list[np.ndarray] = []
    all_direction_correct: list[np.ndarray] = []
    all_return_abs_error: list[np.ndarray] = []
    regime_stats = {r: {"correct": 0, "total": 0} for r in [-1, 0, 1, 2]}
    sample_windows = []

    for batch in loader:
        feats, ids = _collate(batch, tok, device)
        logits = model(feats)

        # Forecast-only evaluation: one prediction per window, matching the
        # training/validation objective and avoiding internal overlapping labels.
        logits_last = logits[:, -1, :]
        ids_last = ids[:, -1]
        feats_last = feats[:, -1, :]
        probs_last = F.softmax(logits_last, dim=-1)
        pred_ids = logits_last.argmax(dim=-1)
        correct = pred_ids == ids_last
        nll = F.cross_entropy(logits_last, ids_last, reduction="none")
        abs_bin_error = (pred_ids - ids_last).abs()
        top1_conf = probs_last.gather(1, pred_ids.unsqueeze(1)).squeeze(1)

        pred_ret = centers[pred_ids]
        true_ret = centers[ids_last]
        expected_ret = probs_last @ centers
        direction_correct = (torch.sign(expected_ret) == torch.sign(true_ret))
        return_abs_error = (expected_ret - true_ret).abs()

        correct_np = correct.cpu().numpy()
        pred_ids_np = pred_ids.cpu().numpy()
        ids_np = ids_last.cpu().numpy()
        feats_np = feats_last.cpu().numpy()

        all_correct.append(correct_np)
        all_top1_conf.append(top1_conf.cpu().numpy())
        all_nll.append(nll.cpu().numpy())
        all_abs_bin_error.append(abs_bin_error.cpu().numpy())
        all_within_1.append((abs_bin_error <= 1).cpu().numpy())
        all_within_3.append((abs_bin_error <= 3).cpu().numpy())
        all_within_5.append((abs_bin_error <= 5).cpu().numpy())
        all_direction_correct.append(direction_correct.cpu().numpy())
        all_return_abs_error.append(return_abs_error.cpu().numpy())

        r0 = feats_np[:, regime_0_idx] > 0.5
        r1 = feats_np[:, regime_1_idx] > 0.5
        r2 = feats_np[:, regime_2_idx] > 0.5
        r_none = ~(r0 | r1 | r2)
        for mask, r_id in [(r0, 0), (r1, 1), (r2, 2), (r_none, -1)]:
            if mask.any():
                regime_stats[r_id]["correct"] += int(correct_np[mask].sum())
                regime_stats[r_id]["total"] += int(mask.sum())

        if len(sample_windows) < 10:
            take = min(10 - len(sample_windows), len(pred_ids_np))
            for i in range(take):
                sample_windows.append({
                    "true_id": int(ids_np[i]),
                    "pred_id": int(pred_ids_np[i]),
                    "true_ret": float(true_ret[i].cpu().item()),
                    "pred_ret": float(pred_ret[i].cpu().item()),
                    "expected_ret": float(expected_ret[i].cpu().item()),
                    "top1_conf": float(top1_conf[i].cpu().item()),
                    "nll": float(nll[i].cpu().item()),
                })

    if not all_correct:
        raise ValueError("No test windows available for evaluation.")

    correct_arr = np.concatenate(all_correct)
    top1_conf_arr = np.concatenate(all_top1_conf)
    nll_arr = np.concatenate(all_nll)
    abs_bin_error_arr = np.concatenate(all_abs_bin_error)
    within_1_arr = np.concatenate(all_within_1)
    within_3_arr = np.concatenate(all_within_3)
    within_5_arr = np.concatenate(all_within_5)
    direction_arr = np.concatenate(all_direction_correct)
    return_abs_error_arr = np.concatenate(all_return_abs_error)

    n_buckets = 10
    bucket_edges = np.linspace(0.0, 1.0, n_buckets + 1)
    ece_buckets = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (top1_conf_arr >= lo) & (top1_conf_arr < hi)
        if mask.sum() > 0:
            avg_conf = float(top1_conf_arr[mask].mean())
            avg_acc = float(correct_arr[mask].mean())
            frac = float(mask.sum()) / len(top1_conf_arr)
            ece_buckets.append({"lo": lo, "hi": hi, "conf": avg_conf, "acc": avg_acc, "frac": frac})
    ece = float(sum(b["frac"] * abs(b["conf"] - b["acc"]) for b in ece_buckets))

    per_regime = {}
    for r_id, stats in regime_stats.items():
        if stats["total"] > 0:
            per_regime[str(r_id)] = {
                "accuracy": stats["correct"] / stats["total"],
                "n": stats["total"],
            }

    return {
        "run_id": cfg.run_id,
        "ckpt_path": str(ckpt_path),
        "ckpt_step": ckpt.get("step"),
        "best_val_loss": ckpt.get("val_loss"),
        "forecast_only_eval": True,
        "hard_ce_nll": float(nll_arr.mean()),
        "overall_accuracy": float(correct_arr.mean()),
        "ece": ece,
        "mean_abs_bin_error": float(abs_bin_error_arr.mean()),
        "within_1_bin_accuracy": float(within_1_arr.mean()),
        "within_3_bin_accuracy": float(within_3_arr.mean()),
        "within_5_bin_accuracy": float(within_5_arr.mean()),
        "direction_accuracy_expected_return": float(direction_arr.mean()),
        "expected_return_mae": float(return_abs_error_arr.mean()),
        "calibration_buckets": ece_buckets,
        "per_regime_accuracy": per_regime,
        "sample_predictions": sample_windows,
        "n_test_bars": int(correct_arr.shape[0]),
    }


def write_report(metrics: dict, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# CandleGPTv2 Training Report — {metrics['run_id']}",
        "",
        f"**Checkpoint:** step {metrics['ckpt_step']}  ",
        f"**Best val loss:** {metrics['best_val_loss']:.4f}  ",
        f"**Test forecasts evaluated:** {metrics['n_test_bars']:,}  ",
        f"**Forecast-only eval:** {metrics.get('forecast_only_eval', False)}  ",
        "",
        "## Final-Token Forecast Metrics",
        "",
        f"- Hard CE / NLL: **{metrics['hard_ce_nll']:.4f}**",
        f"- Top-1 bin accuracy: **{metrics['overall_accuracy']:.4f}** "
        f"({metrics['overall_accuracy']*100:.2f}%)",
        f"- Within ±1 bin: **{metrics['within_1_bin_accuracy']:.4f}** "
        f"({metrics['within_1_bin_accuracy']*100:.2f}%)",
        f"- Within ±3 bins: **{metrics['within_3_bin_accuracy']:.4f}** "
        f"({metrics['within_3_bin_accuracy']*100:.2f}%)",
        f"- Within ±5 bins: **{metrics['within_5_bin_accuracy']:.4f}** "
        f"({metrics['within_5_bin_accuracy']*100:.2f}%)",
        f"- Mean absolute bin error: **{metrics['mean_abs_bin_error']:.2f}**",
        f"- Direction accuracy from expected return: **{metrics['direction_accuracy_expected_return']:.4f}** "
        f"({metrics['direction_accuracy_expected_return']*100:.2f}%)",
        f"- Expected-return MAE: **{metrics['expected_return_mae']:.6f}**",
        f"- ECE (10-bucket): **{metrics['ece']:.4f}**",
        "",
        "## Per-Regime Top-1 Accuracy",
        "",
        "| Regime | Accuracy | N forecasts |",
        "|--------|----------|-------------|",
    ]
    regime_names = {"-1": "Untagged", "0": "Regime 0", "1": "Regime 1", "2": "Regime 2"}
    for r_id, stats in sorted(metrics["per_regime_accuracy"].items()):
        name = regime_names.get(r_id, f"Regime {r_id}")
        lines.append(
            f"| {name} | {stats['accuracy']:.4f} ({stats['accuracy']*100:.1f}%) "
            f"| {stats['n']:,} |"
        )
    lines += [
        "",
        "## Calibration (10 confidence buckets)",
        "",
        "| Conf range | Avg conf | Avg acc | Fraction |",
        "|-----------|---------|---------|----------|",
    ]
    for b in metrics["calibration_buckets"]:
        lines.append(
            f"| [{b['lo']:.1f}, {b['hi']:.1f}) "
            f"| {b['conf']:.3f} | {b['acc']:.3f} | {b['frac']:.3f} |"
        )
    lines += [
        "",
        "## Sample Final Forecasts",
        "",
        "Each row: predicted bin center | expected return | actual bin center.",
        "",
        "| # | Pred bin | Exp return | Actual bin | Top1 conf | NLL |",
        "|---|----------|------------|------------|-----------|-----|",
    ]
    for i, s in enumerate(metrics["sample_predictions"]):
        lines.append(
            f"| {i+1} | {s['pred_ret']:+.5f} | {s['expected_ret']:+.5f} "
            f"| {s['true_ret']:+.5f} | {s['top1_conf']:.3f} | {s['nll']:.3f} |"
        )
    report_path = run_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report written: {report_path}")
    json_path = run_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(f"Metrics JSON: {json_path}")
