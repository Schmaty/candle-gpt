"""Training loop for CandleGPTv2."""
from __future__ import annotations
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from v2.data.dataset import KlineWindowDataset
from v2.model.model import CandleGPTv2
from v2.model.tokenizer import ReturnTokenizerV2
from v2.train.config import TrainConfig
from v2.train.losses import candle_loss
from v2.train.progress import HardwareSpecs, ModelSpecs, ProgressEmitter

log = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr_max * step / max(cfg.warmup_steps, 1)
    if cfg.max_steps <= 0:
        return cfg.lr_max
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.lr_min + cosine_decay * (cfg.lr_max - cfg.lr_min)


def _save_checkpoint(
    model: CandleGPTv2,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: Optional[float],
    cfg: TrainConfig,
    tag: str,
) -> Path:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.ckpt_dir / f"{tag}.pt"
    torch.save(
        {
            "step": step,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": cfg.model.to_dict(),
            "run_id": cfg.run_id,
            "tokenizer_path": str(cfg.tokenizer_path),
        },
        path,
    )
    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "None"
    log.info(f"Checkpoint saved: {path}  (step={step}, val_loss={val_loss_str})")
    return path


def _build_datasets(cfg: TrainConfig):
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
    n = len(full_ds)
    n_bars = full_ds._bars.shape[0]
    train_end_bar = int(n_bars * cfg.train_frac)
    val_end_bar = int(n_bars * (cfg.train_frac + cfg.val_frac))

    # Clean forecast splits: a dataset index is a window start. The supervised
    # target for that window is the final timestep's next return, whose bar
    # index is start + window - 1. Build subsets by target-bar ranges so no
    # train loss is computed on val/test-era labels. Also leave a configurable
    # gap around boundaries to make leakage/near-boundary overlap harder.
    gap = cfg.window if cfg.split_gap_bars is None else max(0, cfg.split_gap_bars)
    last_valid_target = n_bars - 2  # log_returns[-1] is sentinel 0.0; never train on it

    def target_to_window_idx(target_bar: int) -> int:
        return target_bar - cfg.window + 1

    def target_range_to_window_indices(start_target: int, end_target: int, stride: int) -> list[int]:
        lo = max(cfg.window - 1, start_target)
        hi = min(last_valid_target + 1, end_target)
        if hi <= lo:
            return []
        return [target_to_window_idx(t) for t in range(lo, hi, stride)]

    train_indices = target_range_to_window_indices(
        cfg.window - 1,
        train_end_bar - gap,
        cfg.stride_train,
    )
    val_indices = target_range_to_window_indices(
        train_end_bar + gap,
        val_end_bar - gap,
        cfg.stride_val,
    )
    test_indices = target_range_to_window_indices(
        val_end_bar + gap,
        last_valid_target + 1,
        cfg.stride_val,
    )

    if not train_indices or not val_indices:
        raise ValueError(
            "Empty train/val split after applying clean target boundaries; "
            f"n_bars={n_bars} window={cfg.window} gap={gap} "
            f"train={len(train_indices)} val={len(val_indices)}"
        )

    return (
        full_ds,
        Subset(full_ds, train_indices),
        Subset(full_ds, val_indices),
        Subset(full_ds, test_indices),
    )


def _fit_tokenizer(cfg: TrainConfig, train_subset, full_ds) -> ReturnTokenizerV2:
    """Fit tokenizer from raw log_returns array on the training bar range."""
    log.info("Fitting tokenizer on training split log returns...")
    n_bars = full_ds._bars.shape[0]
    train_end_bar = int(n_bars * cfg.train_frac)
    train_rets = full_ds._log_returns[:train_end_bar].astype(np.float64)
    train_rets = train_rets[np.isfinite(train_rets)]
    tok = ReturnTokenizerV2(n_bins=cfg.n_bins)
    tok.fit(train_rets)
    tok.save(cfg.tokenizer_path)
    log.info(f"Tokenizer fitted: n_bins={tok.n_bins}, saved to {cfg.tokenizer_path}")
    return tok


def _collate(batch, tokenizer: ReturnTokenizerV2, device: torch.device):
    # batch is the DataLoader-collated result: (feats_batch, rets_batch)
    # feats_batch: (B, T, F), rets_batch: (B, T)
    feats_batch, rets_batch = batch
    B = rets_batch.shape[0]
    ids_list = []
    for i in range(B):
        ids = tokenizer.encode(rets_batch[i].numpy())
        ids_list.append(torch.from_numpy(ids))
    feats = feats_batch.to(device)
    ids = torch.stack(ids_list).to(device)
    return feats, ids


@torch.no_grad()
def _eval_loss(
    model: CandleGPTv2,
    val_subset,
    tokenizer: ReturnTokenizerV2,
    device: torch.device,
    n_batches: int,
    batch_size: int,
) -> float:
    model.eval()
    loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)
    losses = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        feats, ids = _collate(batch, tokenizer, device)
        logits = model(feats)
        B, T, V = logits.shape
        # Validation stays hard-label CE so runs remain comparable even when
        # training uses soft ordinal labels.
        loss = nn.functional.cross_entropy(logits[:, -1, :], ids[:, -1])
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def train(cfg: TrainConfig) -> str:
    """Run the training loop. Returns run_id."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(cfg.run_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    log.info(f"=== CandleGPTv2 Training Run: {cfg.run_id} ===")

    device = _select_device()
    log.info(f"Device: {device}")

    full_ds, train_ds, val_ds, test_ds = _build_datasets(cfg)
    log.info(f"Split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    tokenizer = _fit_tokenizer(cfg, train_ds, full_ds)

    model = CandleGPTv2(cfg.model).to(device)
    log.info(f"Model params: {model.num_params():,}")

    # Build emitter
    hw = ProgressEmitter.collect_hardware()
    model_specs = ModelSpecs(
        n_params=model.num_params(),
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model,
        n_bins=cfg.model.n_bins,
        window=cfg.window,
        interval=cfg.interval,
        n_features=cfg.model.n_features,
    )
    emitter = ProgressEmitter(
        cfg.run_dir, hw, model_specs,
        cfg.max_wall_clock_s, cfg.max_steps,
        min_interval_s=cfg.progress_interval_s,
    )
    emitter.update(step=0, state="starting", lr=cfg.lr_max, force=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    resume_step = 0
    resume_val_loss: Optional[float] = None
    if cfg.resume_from is not None:
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        resume_step = int(ckpt.get("step", 0))
        resume_val_loss = ckpt.get("val_loss")
        log.info(f"Resumed from {cfg.resume_from} at step={resume_step} val_loss={resume_val_loss}")
        if resume_val_loss is not None:
            emitter._last_val_loss = float(resume_val_loss)
            emitter._best_val_loss = float(resume_val_loss)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    best_val_loss = float(resume_val_loss) if resume_val_loss is not None else float("inf")
    best_ckpt_path: Optional[Path] = cfg.resume_from if cfg.resume_from is not None and resume_val_loss is not None else None
    has_validated = resume_val_loss is not None
    plateau_evals = 0
    step = resume_step
    train_start = time.time()
    last_ckpt_time = train_start
    tokens_processed = 0
    last_log_time = train_start
    recent_losses = []
    last_checkpoint_step: Optional[int] = None
    last_eval_step: Optional[int] = None

    log.info("Starting training loop...")
    model.train()

    try:
        while True:
            for batch in train_loader:
                # Wall-clock cap
                elapsed = time.time() - train_start
                if cfg.max_wall_clock_s > 0 and math.isfinite(cfg.max_wall_clock_s) and elapsed >= cfg.max_wall_clock_s:
                    log.info(f"[STOP] wall-clock cap at step {step} ({elapsed/3600:.2f}h)")
                    tag = f"step_{step:07d}"
                    ckpt_path = _save_checkpoint(model, optimizer, step, best_val_loss if has_validated else None, cfg, tag)
                    flush_loss = float(np.mean(recent_losses)) if recent_losses else None
                    emitter.update(step=step, state="done",
                                   train_loss=flush_loss,
                                   last_checkpoint_step=step, force=True)
                    emitter.event("checkpoint", {"step": step, "path": str(ckpt_path)})
                    return cfg.run_id

                if cfg.max_steps > 0 and step >= cfg.max_steps:
                    log.info(f"[STOP] max_steps={cfg.max_steps} reached.")
                    emitter.update(step=step, state="done", force=True)
                    return cfg.run_id

                lr = _get_lr(step, cfg)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                step_start = time.time()
                feats, ids = _collate(batch, tokenizer, device)
                logits = model(feats)
                B, T, V = logits.shape
                if cfg.forecast_only_loss:
                    # One supervised forecast per sampled window: use all
                    # context up to the final bar to predict that bar's next
                    # return. This avoids rewarding the model for repeatedly
                    # reconstructing the same overlapping historical labels.
                    loss = candle_loss(
                        logits[:, -1, :], ids[:, -1],
                        loss_type=cfg.loss_type,
                        soft_label_sigma_bins=cfg.soft_label_sigma_bins,
                    )
                else:
                    # Legacy sequence loss: valid for ablations only. The last
                    # position is now safe because clean split construction
                    # excludes the file-level sentinel target.
                    loss = candle_loss(
                        logits.view(B * T, V),
                        ids.view(B * T),
                        loss_type=cfg.loss_type,
                        soft_label_sigma_bins=cfg.soft_label_sigma_bins,
                    )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
                optimizer.step()

                step_dt = time.time() - step_start
                tokens_processed += B * T
                step += 1
                recent_losses.append(loss.item())

                # Logging + progress emission
                if step % cfg.log_interval_steps == 0:
                    elapsed_h = (time.time() - train_start) / 3600
                    smooth_loss = float(np.mean(recent_losses))
                    tps = (B * T) / max(step_dt, 1e-9)
                    log.info(
                        f"step={step:6d}  loss={smooth_loss:.4f}  "
                        f"lr={lr:.2e}  elapsed={elapsed_h:.2f}h  tps={tps:.0f}"
                    )
                    emitter.update(
                        step=step, state="training",
                        train_loss=smooth_loss, lr=lr,
                        throughput_tok_per_s=tps, grad_norm=grad_norm,
                        last_checkpoint_step=last_checkpoint_step,
                        last_eval_step=last_eval_step,
                    )
                    emitter.event("step", {"step": step, "loss": smooth_loss,
                                           "lr": lr, "grad_norm": grad_norm,
                                           "throughput_tok_per_s": tps})
                    recent_losses = []

                # Validation
                if step % cfg.val_interval_steps == 0:
                    emitter.update(step=step, state="evaluating",
                                   last_eval_step=step, force=True)
                    val_loss = _eval_loss(
                        model, val_ds, tokenizer, device,
                        cfg.val_batches, cfg.batch_size,
                    )
                    log.info(f"  [val] step={step}  val_loss={val_loss:.4f}")
                    emitter.update(step=step, state="training", val_loss=val_loss,
                                   lr=lr, last_eval_step=step, force=True)
                    emitter.event("val", {"step": step, "val_loss": val_loss})
                    last_eval_step = step
                    has_validated = True
                    if val_loss < best_val_loss - cfg.early_stop_min_delta:
                        best_val_loss = val_loss
                        plateau_evals = 0
                        best_ckpt_path = _save_checkpoint(
                            model, optimizer, step, val_loss, cfg, "best_val"
                        )
                        log.info(f"  [val] NEW BEST: {val_loss:.4f}")
                    elif cfg.early_stop_patience_evals is not None:
                        plateau_evals += 1
                        log.info(
                            f"  [val] no meaningful improvement: {plateau_evals}/"
                            f"{cfg.early_stop_patience_evals} evals"
                        )
                        if plateau_evals >= cfg.early_stop_patience_evals:
                            log.info(
                                f"[STOP] validation plateau: best_val_loss={best_val_loss:.4f} "
                                f"patience={cfg.early_stop_patience_evals} min_delta={cfg.early_stop_min_delta}"
                            )
                            tag = f"step_{step:07d}"
                            ckpt_path = _save_checkpoint(model, optimizer, step, best_val_loss if has_validated else None, cfg, tag)
                            emitter.update(step=step, state="done", last_checkpoint_step=step, force=True)
                            emitter.event("checkpoint", {"step": step, "path": str(ckpt_path)})
                            return cfg.run_id

                # Periodic checkpoint
                now = time.time()
                if now - last_ckpt_time >= cfg.checkpoint_interval_s:
                    tag = f"step_{step:07d}"
                    emitter.update(step=step, state="checkpointing",
                                   last_checkpoint_step=step, force=True)
                    ckpt_path = _save_checkpoint(model, optimizer, step, best_val_loss if has_validated else None, cfg, tag)
                    emitter.event("checkpoint", {"step": step, "path": str(ckpt_path)})
                    last_checkpoint_step = step
                    last_ckpt_time = now
                    emitter.update(step=step, state="training",
                                   last_checkpoint_step=last_checkpoint_step, force=True)

    except Exception:
        emitter.update(step=step, state="failed", force=True)
        raise

    emitter.update(step=step, state="done", force=True)
    return cfg.run_id
