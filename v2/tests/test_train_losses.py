import torch
import torch.nn.functional as F

from v2.train.losses import candle_loss, ordinal_soft_targets, soft_cross_entropy


def test_ordinal_soft_targets_sum_to_one_and_peak_at_target():
    ids = torch.tensor([0, 3, 7])
    targets = ordinal_soft_targets(ids, n_bins=8, sigma_bins=2.0)
    assert targets.shape == (3, 8)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert targets.argmax(dim=-1).tolist() == ids.tolist()


def test_zero_sigma_matches_one_hot_ce():
    logits = torch.randn(4, 6)
    ids = torch.tensor([0, 1, 2, 5])
    hard = F.cross_entropy(logits, ids)
    soft = soft_cross_entropy(logits, ordinal_soft_targets(ids, n_bins=6, sigma_bins=0.0))
    assert torch.allclose(hard, soft, atol=1e-6)


def test_candle_loss_supports_soft_ce():
    logits = torch.randn(5, 16, requires_grad=True)
    ids = torch.tensor([0, 3, 7, 10, 15])
    loss = candle_loss(logits, ids, loss_type="soft_ce", soft_label_sigma_bins=1.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
