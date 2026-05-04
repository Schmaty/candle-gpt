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
    B = rets_batch.shape[0]
    ids_list = []
    for i in range(B):
        ids_list.append(torch.from_numpy(tokenizer.encode(rets_batch[i].numpy())))
    return feats_batch.to(device), torch.stack(ids_list).to(device)


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

    full_ds = KlineWindowDataset(
        path=cfg.kline_path,
        window=cfg.window,
        stride=1,
        funding_path=cfg.funding_path,
        liq_path=cfg.liq_path,
        apply_features=True,
        return_targets=True,
    )
    n_bars = full_ds._bars.shape[0]
    train_end_bar = int(n_bars * cfg.train_frac)
    val_end_bar = int(n_bars * (cfg.train_frac + cfg.val_frac))

    def bar_to_window_idx(bar: int) -> int:
        return max(0, bar - cfg.window + 1)

    val_end_win = bar_to_window_idx(val_end_bar)
    test_indices = list(range(val_end_win, len(full_ds), cfg.stride_val))
    test_ds = Subset(full_ds, test_indices)
    log.info(f"Test windows: {len(test_ds)}")

    regime_0_idx = list(FEATURE_COLUMNS).index("regime_0")
    regime_1_idx = list(FEATURE_COLUMNS).index("regime_1")
    regime_2_idx = list(FEATURE_COLUMNS).index("regime_2")

    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    all_correct = []
    all_probs_top1 = []
    regime_stats = {r: {"correct": 0, "total": 0} for r in [-1, 0, 1, 2]}
    sample_windows = []

    for batch in loader:
        feats, ids = _collate(batch, tok, device)
        logits = model(feats)
        probs = F.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)
        correct = (pred_ids == ids)

        # Create valid mask — exclude last position per window (sentinel target)
        valid_mask = torch.ones_like(ids, dtype=torch.bool)  # (B, T)
        valid_mask[:, -1] = False
        valid_flat = valid_mask.view(-1).cpu().numpy()

        correct_flat = correct.view(-1).cpu().numpy()[valid_flat]
        pred_ids_flat = pred_ids.view(-1).cpu().numpy()[valid_flat]
        ids_flat = ids.view(-1).cpu().numpy()[valid_flat]
        feats_flat = feats.view(-1, feats.size(-1)).cpu().numpy()[valid_flat]
        probs_flat = probs.view(-1, tok.n_bins).cpu().numpy()[valid_flat]
        top1_conf = probs_flat[np.arange(len(pred_ids_flat)), pred_ids_flat]

        all_correct.append(correct_flat)
        all_probs_top1.append(top1_conf)

        r0 = feats_flat[:, regime_0_idx] > 0.5
        r1 = feats_flat[:, regime_1_idx] > 0.5
        r2 = feats_flat[:, regime_2_idx] > 0.5
        r_none = ~(r0 | r1 | r2)
        for mask, r_id in [(r0, 0), (r1, 1), (r2, 2), (r_none, -1)]:
            if mask.any():
                regime_stats[r_id]["correct"] += int(correct_flat[mask].sum())
                regime_stats[r_id]["total"] += int(mask.sum())

        if len(sample_windows) < 5:
            b = 0
            sample_windows.append({
                "true_ids": ids[b, :10].cpu().tolist(),
                "pred_ids": pred_ids[b, :10].cpu().tolist(),
                "true_rets": tok.decode(ids[b, :10].cpu().numpy()).tolist(),
                "pred_rets": tok.decode(pred_ids[b, :10].cpu().numpy()).tolist(),
            })

    all_correct_arr = np.concatenate(all_correct)
    all_probs_arr = np.concatenate(all_probs_top1)
    overall_accuracy = float(all_correct_arr.mean())

    n_buckets = 10
    bucket_edges = np.linspace(0.0, 1.0, n_buckets + 1)
    ece_buckets = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (all_probs_arr >= lo) & (all_probs_arr < hi)
        if mask.sum() > 0:
            avg_conf = float(all_probs_arr[mask].mean())
            avg_acc = float(all_correct_arr[mask].mean())
            frac = float(mask.sum()) / len(all_probs_arr)
            ece_buckets.append({"lo": lo, "hi": hi, "conf": avg_conf,
                                 "acc": avg_acc, "frac": frac})
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
        "overall_accuracy": overall_accuracy,
        "ece": ece,
        "calibration_buckets": ece_buckets,
        "per_regime_accuracy": per_regime,
        "sample_predictions": sample_windows,
        "n_test_bars": int(all_correct_arr.shape[0]),
    }


def write_report(metrics: dict, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# CandleGPTv2 Training Report — {metrics['run_id']}",
        "",
        f"**Checkpoint:** step {metrics['ckpt_step']}  ",
        f"**Best val loss:** {metrics['best_val_loss']:.4f}  ",
        f"**Test bars evaluated:** {metrics['n_test_bars']:,}  ",
        "",
        "## Accuracy",
        "",
        f"- Overall top-1 bin accuracy: **{metrics['overall_accuracy']:.4f}** "
        f"({metrics['overall_accuracy']*100:.2f}%)",
        f"- ECE (10-bucket): **{metrics['ece']:.4f}**",
        "",
        "## Per-Regime Accuracy",
        "",
        "| Regime | Accuracy | N bars |",
        "|--------|----------|--------|",
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
        "## Sample Predictions (first 10 positions of first 5 test windows)",
        "",
        "Each row: predicted return | actual return (log-return, e.g. 0.0012 = +0.12%)",
        "",
    ]
    for i, s in enumerate(metrics["sample_predictions"]):
        lines.append(f"**Window {i+1}:**")
        rows = ["| # | Predicted | Actual |", "|---|---------|--------|"]
        for j, (p, a) in enumerate(zip(s["pred_rets"], s["true_rets"])):
            rows.append(f"| {j} | {p:+.5f} | {a:+.5f} |")
        lines.extend(rows)
        lines.append("")
    report_path = run_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report written: {report_path}")
    json_path = run_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(f"Metrics JSON: {json_path}")
