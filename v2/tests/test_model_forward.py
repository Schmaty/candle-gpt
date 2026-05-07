"""CandleGPTv2: forward pass shapes, causal masking, parameter count."""
import torch
import pytest

from v2.model.config import ModelConfig
from v2.model.model import CandleGPTv2


def _small_cfg() -> ModelConfig:
    return ModelConfig(
        n_features=41, d_model=64, n_heads=4, n_layers=2,
        ffn_mult=4, block_size=16, n_bins=32, dropout=0.0,
    )


def test_forward_output_shape():
    cfg = _small_cfg()
    model = CandleGPTv2(cfg)
    x = torch.randn(2, 16, 41)
    logits = model(x)
    assert logits.shape == (2, 16, 32), logits.shape


def test_forward_single_token():
    cfg = _small_cfg()
    model = CandleGPTv2(cfg)
    x = torch.randn(1, 1, 41)
    logits = model(x)
    assert logits.shape == (1, 1, 32)


def test_forward_shorter_than_block_size():
    cfg = _small_cfg()
    model = CandleGPTv2(cfg)
    x = torch.randn(3, 8, 41)
    logits = model(x)
    assert logits.shape == (3, 8, 32)


def test_causal_masking_future_independence():
    """Changing future tokens must not affect past predictions."""
    cfg = _small_cfg()
    model = CandleGPTv2(cfg)
    model.eval()
    x = torch.randn(1, 16, 41)
    x_perturbed = x.clone()
    x_perturbed[0, 8:, :] += 100.0

    with torch.no_grad():
        logits_orig = model(x)
        logits_pert = model(x_perturbed)

    assert torch.allclose(logits_orig[0, :8], logits_pert[0, :8], atol=1e-5), \
        "Causal masking broken: past positions affected by future tokens"


def test_parameter_count_approx_30m():
    # v2.1 default config targets ~30M params (d_model=512, 10L, 8H, ctx=1024).
    cfg = ModelConfig()
    model = CandleGPTv2(cfg)
    n_params = model.num_params()
    assert 27_000_000 < n_params < 36_000_000, \
        f"Unexpected param count: {n_params:,}"


def test_num_params_excludes_position_embedding():
    cfg = ModelConfig()
    model = CandleGPTv2(cfg)
    n_all = model.num_params(exclude_pos_embed=False)
    n_no_pos = model.num_params(exclude_pos_embed=True)
    assert n_no_pos < n_all


def test_logits_sum_to_reasonable_softmax():
    cfg = _small_cfg()
    model = CandleGPTv2(cfg)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 4, 41)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_generate_returns_bin_ids():
    cfg = _small_cfg()
    model = CandleGPTv2(cfg)
    model.eval()
    x = torch.randn(1, 4, 41)
    with torch.no_grad():
        ids = model.generate_ids(x, n_steps=3, temperature=1.0)
    assert ids.shape == (1, 3)
    assert (ids >= 0).all() and (ids < cfg.n_bins).all()


def test_generate_ids_preserves_training_mode():
    """generate_ids() must not permanently flip the model to eval mode."""
    cfg = _small_cfg()
    model = CandleGPTv2(cfg)
    model.train()
    assert model.training
    x = torch.randn(1, 4, 41)
    with torch.no_grad():
        model.generate_ids(x, n_steps=2)
    assert model.training, "generate_ids() leaked eval mode"


def test_regime_conditioning_preserves_shape_and_changes_logits_by_regime():
    cfg = ModelConfig(
        n_features=52, d_model=64, n_heads=4, n_layers=2,
        ffn_mult=4, block_size=16, n_bins=32, dropout=0.0,
        regime_conditioning=True,
    )
    model = CandleGPTv2(cfg)
    model.eval()
    x0 = torch.randn(1, 16, 52)
    x1 = x0.clone()
    # Clear the regime one-hot slots, then set different regimes. Other
    # features are identical, so logits should differ only from regime bias.
    x0[..., 26:29] = 0.0
    x1[..., 26:29] = 0.0
    x0[..., 26] = 1.0  # regime_0
    x1[..., 27] = 1.0  # regime_1
    with torch.no_grad():
        logits0 = model(x0)
        logits1 = model(x1)
    assert logits0.shape == (1, 16, 32)
    assert logits1.shape == logits0.shape
    assert not torch.allclose(logits0, logits1)
