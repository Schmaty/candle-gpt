"""Auto-reload test for V2InferenceModel: when training writes a fresh
best_val.pt, predict_live() should pick up the new weights without a
server restart."""
from __future__ import annotations
import os
import time
from pathlib import Path

import numpy as np
import torch

from v2.model.config import ModelConfig
from v2.model.model import CandleGPTv2
from v2.model.tokenizer import ReturnTokenizerV2
from v2.server.inference import V2InferenceModel


def _save_dummy_ckpt(path: Path, run_id: str, step: int, seed: int) -> None:
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, ffn_mult=2,
                      block_size=64, n_bins=16, n_features=41, dropout=0.0)
    torch.manual_seed(seed)
    model = CandleGPTv2(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "val_loss": 1.0 + 0.001 * step,
        "model_state": model.state_dict(),
        "model_config": cfg.to_dict(),
        "run_id": run_id,
        "tokenizer_path": "tok.pkl",
    }, path)


def _save_dummy_tokenizer(path: Path) -> None:
    tok = ReturnTokenizerV2(n_bins=16)
    rng = np.random.default_rng(0)
    tok.fit(rng.standard_normal(2000).astype(np.float64) * 0.001)
    tok.save(path)


def test_inference_auto_reloads_when_ckpt_replaced(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_val.pt"
    tok = tmp_path / "tokenizer.pkl"
    _save_dummy_tokenizer(tok)

    _save_dummy_ckpt(ckpt, run_id="run_a", step=100, seed=11)
    inf = V2InferenceModel()
    assert inf.load(ckpt, tok) is True
    assert inf.ckpt_step == 100

    # Simulate the training process replacing best_val.pt mid-flight.
    # mtime resolution can be coarse, so bump it explicitly.
    _save_dummy_ckpt(ckpt, run_id="run_a", step=2500, seed=22)
    new_t = time.time() + 5.0
    os.utime(ckpt, (new_t, new_t))

    # _maybe_reload sleeps ~0.5s while it confirms the file is settled.
    inf._maybe_reload()
    assert inf.ckpt_step == 2500

    # Subsequent calls without further writes are no-ops.
    inf._maybe_reload()
    assert inf.ckpt_step == 2500


def test_inference_load_without_path_is_safe(tmp_path: Path) -> None:
    inf = V2InferenceModel()
    # No load() yet — _maybe_reload should not raise.
    inf._maybe_reload()
    assert inf.model is None
