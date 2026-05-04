"""Calibration sweep + backtest helpers for the v2 dashboard.

Loads the model + test dataset once on first use, then re-uses the cached
arrays for every sweep / backtest call. Designed for interactive latency
budgets: a 5×6 sweep over 200 sampled windows runs in <2s on MPS.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from v2.data.dataset import KlineWindowDataset
from v2.features.constants import FEATURE_COLUMNS  # noqa: F401  (kept for downstream regime indexing)
from v2.model.config import ModelConfig
from v2.model.model import CandleGPTv2
from v2.model.tokenizer import ReturnTokenizerV2
from v2.train.config import TrainConfig

log = logging.getLogger(__name__)


@dataclass
class _Cache:
    model: CandleGPTv2
    tokenizer: ReturnTokenizerV2
    bin_centers: np.ndarray  # (n_bins,)
    test_ds: Subset
    full_ds: KlineWindowDataset
    val_end_bar: int
    device: torch.device


class SweepService:
    """Lazy singleton that owns the model + test set for sweep / backtest."""

    def __init__(self, run_dir: Path, kline_path: Path, funding_path: Path, liq_path: Path,
                 ckpt_filename: str = "best_val.pt"):
        self.run_dir = run_dir
        self.kline_path = kline_path
        self.funding_path = funding_path
        self.liq_path = liq_path
        self.ckpt_filename = ckpt_filename
        self._cache: Optional[_Cache] = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> _Cache:
        if self._cache is not None:
            return self._cache
        with self._lock:
            if self._cache is not None:
                return self._cache
            ckpt_path = self.run_dir / "checkpoints" / self.ckpt_filename
            tokenizer_path = self.run_dir / "tokenizer.pkl"
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_cfg = ModelConfig.from_dict(ckpt["model_config"])
            model = CandleGPTv2(model_cfg).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            tok = ReturnTokenizerV2.load(tokenizer_path)
            bin_centers = tok.decode(np.arange(tok.n_bins)).astype(np.float64)

            cfg = TrainConfig()  # only used to know window/strides — defaults match training
            full_ds = KlineWindowDataset(
                path=self.kline_path,
                window=cfg.window,
                stride=1,
                funding_path=self.funding_path,
                liq_path=self.liq_path,
                apply_features=True,
                return_targets=True,
            )
            n_bars = full_ds._bars.shape[0]
            train_end_bar = int(n_bars * cfg.train_frac)
            val_end_bar = int(n_bars * (cfg.train_frac + cfg.val_frac))
            val_end_win = max(0, val_end_bar - cfg.window + 1)
            test_indices = list(range(val_end_win, len(full_ds), cfg.stride_val))
            test_ds = Subset(full_ds, test_indices)
            log.info(f"[sweep] loaded model + {len(test_ds)} test windows on {device}")

            self._cache = _Cache(
                model=model,
                tokenizer=tok,
                bin_centers=bin_centers,
                test_ds=test_ds,
                full_ds=full_ds,
                val_end_bar=val_end_bar,
                device=device,
            )
            return self._cache

    @torch.no_grad()
    def _logits_for_indices(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on the given test-set indices. Returns:
            logits  (N, n_bins) at the last valid position
            true_id (N,)  bin id of the actual next bar at that position
            start_bar (N,)  absolute bar index of the *starting close* —
                            i.e. the close from which the predicted return
                            is measured. close[start_bar+H] / close[start_bar]
                            gives the H-bar realised return.
        """
        c = self._ensure_loaded()
        full_ds = c.full_ds
        window = full_ds._window
        # Resolve the test-set selections to absolute window indices in full_ds.
        abs_window_idx = np.array([c.test_ds.indices[i] for i in indices], dtype=np.int64)
        # Run the model in mini-batches to keep memory bounded.
        BATCH = 32
        logits_chunks: list[np.ndarray] = []
        true_chunks: list[np.ndarray] = []
        for k0 in range(0, len(abs_window_idx), BATCH):
            chunk = abs_window_idx[k0: k0 + BATCH]
            feats_list, rets_list = [], []
            for w in chunk:
                feats, rets = full_ds[int(w)]
                feats_list.append(feats)
                rets_list.append(rets)
            feats_batch = torch.stack(feats_list).to(c.device)
            rets_batch = torch.stack(rets_list).numpy()  # (B, T)
            ids_batch = np.stack([c.tokenizer.encode(rets_batch[b]) for b in range(rets_batch.shape[0])])
            out = c.model(feats_batch)  # (B, T, n_bins)
            T = out.shape[1]
            last_valid = T - 2
            logits_chunks.append(out[:, last_valid, :].cpu().numpy())
            true_chunks.append(ids_batch[:, last_valid])
        logits = np.concatenate(logits_chunks, axis=0)  # (N, n_bins)
        true_ids = np.concatenate(true_chunks, axis=0)
        # start_bar: absolute bar index of close from which we measure forward return.
        # window i covers bars [i*stride .. i*stride+window-1] with stride=1, and
        # the last valid prediction position is T-2, whose target is bar (i + T - 1).
        # The "from" close is close[i + T - 2].
        start_bar = abs_window_idx + (window - 2)
        return logits, true_ids, start_bar

    def sweep(self, temperatures: list[float], horizons: list[int],
              n_samples: int = 200, seed: int = 17) -> dict:
        """Sample N test windows once, then for each (T, H) compute:
            - dir_acc: sign(predicted E[r]) == sign(actual cumulative return over H bars)
            - ece (10 buckets) of top-1 confidence vs. correctness on the immediate-next bar
            - mean confidence on next bar
        """
        c = self._ensure_loaded()
        rng = np.random.default_rng(seed)
        N = min(n_samples, len(c.test_ds))
        indices = rng.choice(len(c.test_ds), size=N, replace=False).tolist()
        logits, true_ids, start_bar = self._logits_for_indices(indices)

        # The dataset already exposes per-bar log-returns: log_returns[i] = log(close[i+1]/close[i]).
        log_ret = c.full_ds._log_returns  # (n_bars,) — last entry is 0.0 sentinel.

        results = []
        max_h = max(horizons)
        for T in temperatures:
            scaled = logits / max(T, 1e-6)
            scaled -= scaled.max(axis=1, keepdims=True)
            probs = np.exp(scaled)
            probs /= probs.sum(axis=1, keepdims=True)
            expected_ret = (probs * c.bin_centers[None, :]).sum(axis=1)  # (N,)
            top1 = probs.argmax(axis=1)
            top1_conf = probs[np.arange(N), top1]
            top1_correct = (top1 == true_ids).astype(np.float64)
            mean_top1_acc = float(top1_correct.mean())
            mean_conf = float(top1_conf.mean())

            # ECE on top-1 bin accuracy of the immediate next bar
            n_buckets = 10
            edges = np.linspace(0.0, 1.0, n_buckets + 1)
            ece = 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (top1_conf >= lo) & (top1_conf < hi)
                if m.any():
                    ece += float(m.mean()) * abs(top1_conf[m].mean() - top1_correct[m].mean())

            for H in horizons:
                # Actual cumulative log-return over the next H bars from the
                # starting close. log_ret[j] = log(close[j+1]/close[j]); H bars
                # forward = sum of log_ret[start_bar : start_bar + H].
                start = start_bar
                ok = (start >= 0) & (start + H <= len(log_ret))
                valid = ok
                if valid.sum() == 0:
                    results.append({
                        "temperature": T, "horizon": H,
                        "dir_acc": None, "n_valid": 0,
                        "ece": ece, "mean_conf": mean_conf, "top1_acc": mean_top1_acc,
                    })
                    continue
                cum_actual = np.array([
                    float(log_ret[start[i]: start[i] + H].sum()) if valid[i] else 0.0
                    for i in range(N)
                ])
                pred_sign = np.sign(expected_ret[valid])
                act_sign = np.sign(cum_actual[valid])
                # Treat zero as "flat" — drop those from accuracy denominator.
                dir_mask = (pred_sign != 0) & (act_sign != 0)
                if dir_mask.sum() == 0:
                    dir_acc = None
                else:
                    dir_acc = float((pred_sign[dir_mask] == act_sign[dir_mask]).mean())
                results.append({
                    "temperature": T,
                    "horizon": H,
                    "dir_acc": dir_acc,
                    "n_valid": int(dir_mask.sum()),
                    "ece": ece,
                    "mean_conf": mean_conf,
                    "top1_acc": mean_top1_acc,
                })
        # Best by dir_acc (ignore None)
        scored = [r for r in results if r["dir_acc"] is not None and r["n_valid"] > 0]
        best = max(scored, key=lambda r: r["dir_acc"]) if scored else None
        return {
            "n_samples_requested": n_samples,
            "n_samples_used": N,
            "results": results,
            "best": best,
        }

    def backtest(self, temperature: float, horizon: int, z_threshold: float,
                 start_frac: float = 0.0, end_frac: float = 1.0,
                 fee_bps: float = 1.0) -> dict:
        """Run a long/short backtest over a slice of the test set using the
        chosen (temperature, horizon, z_threshold). Returns equity curve plus
        summary stats."""
        c = self._ensure_loaded()
        n_test = len(c.test_ds)
        i0 = int(max(0.0, min(1.0, start_frac)) * n_test)
        i1 = int(max(0.0, min(1.0, end_frac)) * n_test)
        if i1 <= i0:
            return {"error": "end_frac must be > start_frac"}
        indices = list(range(i0, i1))
        # Limit to a sane number of windows so the backtest stays interactive.
        if len(indices) > 500:
            stride = max(1, len(indices) // 500)
            indices = indices[::stride]

        logits, true_ids, start_bar = self._logits_for_indices(indices)
        log_ret = c.full_ds._log_returns

        scaled = logits / max(temperature, 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True)
        expected_ret = (probs * c.bin_centers[None, :]).sum(axis=1)
        # Per-prediction std-dev of returns
        var = (probs * (c.bin_centers[None, :] - expected_ret[:, None]) ** 2).sum(axis=1)
        std = np.sqrt(np.maximum(var, 1e-24))
        # Cumulative-horizon z-score using independent-step assumption:
        #   sum of H draws → mean = H * E[r], var = H * Var[r] (per-bar)
        # We use the *single-bar* expected_ret as a proxy for per-bar drift.
        cum_ret_pred = horizon * expected_ret
        cum_std = std * np.sqrt(horizon)
        z = cum_ret_pred / np.maximum(cum_std, 1e-12)

        # Decision: long if z > +threshold, short if z < -threshold, else flat.
        # Hold for `horizon` bars, then close. Realized return = actual cumulative
        # log-return over those H bars, minus fee on entry+exit (2*fee_bps in bps).
        fee = (2 * fee_bps) / 10000.0
        equity = []
        cum_log_pnl = 0.0
        wins = 0
        trades = 0
        position_taken = []
        for i, b_i in enumerate(start_bar):
            s = int(b_i)
            if s < 0 or s + horizon > len(log_ret):
                continue
            actual_cum = float(log_ret[s: s + horizon].sum())
            if z[i] > z_threshold:
                pnl = actual_cum - fee
                position_taken.append(1)
                trades += 1
                if actual_cum > 0:
                    wins += 1
            elif z[i] < -z_threshold:
                pnl = -actual_cum - fee
                position_taken.append(-1)
                trades += 1
                if actual_cum < 0:
                    wins += 1
            else:
                pnl = 0.0
                position_taken.append(0)
            cum_log_pnl += pnl
            equity.append({
                "idx": int(b_i),
                "cum_log_pnl": cum_log_pnl,
                "cum_ret_pct": float(np.exp(cum_log_pnl) - 1) * 100,
                "position": int(position_taken[-1]),
            })

        win_rate = (wins / trades) if trades > 0 else 0.0
        total_log = cum_log_pnl
        total_ret = float(np.exp(total_log) - 1) * 100
        # Per-trade Sharpe approximation
        per_trade = [r for r in [(equity[i]["cum_log_pnl"] - (equity[i-1]["cum_log_pnl"] if i > 0 else 0))
                                  for i in range(len(equity))] if r != 0]
        if len(per_trade) > 1:
            mu = float(np.mean(per_trade))
            sigma = float(np.std(per_trade))
            sharpe_per_trade = mu / sigma if sigma > 0 else 0.0
        else:
            sharpe_per_trade = 0.0
        peak = -float("inf")
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e["cum_log_pnl"])
            dd = peak - e["cum_log_pnl"]
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = float(np.exp(-max_dd) - 1) * 100  # negative number

        return {
            "temperature": temperature,
            "horizon": horizon,
            "z_threshold": z_threshold,
            "fee_bps": fee_bps,
            "n_windows": len(equity),
            "trades": trades,
            "longs": int(sum(1 for p in position_taken if p > 0)),
            "shorts": int(sum(1 for p in position_taken if p < 0)),
            "flats": int(sum(1 for p in position_taken if p == 0)),
            "win_rate": win_rate,
            "total_return_pct": total_ret,
            "total_log_pnl": total_log,
            "sharpe_per_trade": sharpe_per_trade,
            "max_drawdown_pct": max_dd_pct,
            "equity": equity,
        }
