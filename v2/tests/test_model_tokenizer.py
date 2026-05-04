"""ReturnTokenizerV2: quantile binning, encode/decode, save/load."""
from pathlib import Path

import numpy as np
import pytest

from v2.model.tokenizer import ReturnTokenizerV2


def _sample_returns(n: int = 10_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.002, n).astype(np.float64)


def test_fit_sets_n_bins():
    tok = ReturnTokenizerV2(n_bins=256)
    tok.fit(_sample_returns())
    assert tok.n_bins == 256


def test_fit_creates_breakpoints():
    tok = ReturnTokenizerV2(n_bins=256)
    tok.fit(_sample_returns())
    assert len(tok.breakpoints) >= 2


def test_encode_returns_int64_in_range():
    tok = ReturnTokenizerV2(n_bins=256)
    tok.fit(_sample_returns())
    rets = np.array([-0.01, 0.0, 0.001, 0.005])
    ids = tok.encode(rets)
    assert ids.dtype == np.int64
    assert (ids >= 0).all() and (ids < tok.n_bins).all()


def test_encode_monotone():
    tok = ReturnTokenizerV2(n_bins=256)
    tok.fit(_sample_returns())
    rets = np.array([-0.01, -0.001, 0.0, 0.001, 0.01])
    ids = tok.encode(rets)
    assert (ids == np.sort(ids)).all(), f"Non-monotone: {ids}"


def test_decode_roundtrip_approximate():
    tok = ReturnTokenizerV2(n_bins=256)
    data = _sample_returns(50_000)
    tok.fit(data)
    rets = np.array([-0.005, -0.001, 0.0, 0.001, 0.005])
    ids = tok.encode(rets)
    reconstructed = tok.decode(ids)
    bin_width_approx = np.diff(tok.breakpoints).mean()
    np.testing.assert_allclose(reconstructed, rets, atol=3 * bin_width_approx)


def test_encode_scalar_array():
    tok = ReturnTokenizerV2(n_bins=32)
    tok.fit(_sample_returns())
    ids = tok.encode(np.array([0.001]))
    assert ids.shape == (1,)


def test_encode_not_fitted_raises():
    tok = ReturnTokenizerV2()
    with pytest.raises(RuntimeError, match="not fitted"):
        tok.encode(np.array([0.0]))


def test_save_load_roundtrip(tmp_path: Path):
    tok = ReturnTokenizerV2(n_bins=64)
    tok.fit(_sample_returns())
    p = tmp_path / "tok.pkl"
    tok.save(p)
    tok2 = ReturnTokenizerV2.load(p)
    assert tok2.n_bins == 64
    rets = np.array([-0.002, 0.0, 0.002])
    np.testing.assert_array_equal(tok.encode(rets), tok2.encode(rets))
