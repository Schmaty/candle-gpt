"""ModelConfig: construction defaults, validation, dict round-trip."""
import pytest
from v2.model.config import ModelConfig


def test_defaults():
    cfg = ModelConfig()
    assert cfg.n_features == 41
    assert cfg.d_model == 384
    assert cfg.n_heads == 6
    assert cfg.n_layers == 6
    assert cfg.ffn_mult == 4
    assert cfg.block_size == 512
    assert cfg.n_bins == 256
    assert cfg.dropout == 0.1


def test_to_dict_from_dict_roundtrip():
    cfg = ModelConfig(d_model=128, n_layers=2, n_heads=2, block_size=64, n_bins=32)
    d = cfg.to_dict()
    cfg2 = ModelConfig.from_dict(d)
    assert cfg2.d_model == 128
    assert cfg2.n_layers == 2
    assert cfg2.block_size == 64


def test_head_dim_must_divide_evenly():
    with pytest.raises(ValueError, match="head_dim"):
        ModelConfig(d_model=100, n_heads=6)


def test_ffn_dim():
    cfg = ModelConfig(d_model=384, ffn_mult=4)
    assert cfg.ffn_dim == 1536
