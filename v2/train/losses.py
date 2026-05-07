"""Training losses for CandleGPTv2.

The model still outputs ordinary logits over return bins. These helpers only
change how training scores those logits.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ordinal_soft_targets(
    target_ids: torch.Tensor,
    n_bins: int,
    sigma_bins: float,
) -> torch.Tensor:
    """Return Gaussian-smoothed ordinal target distributions.

    Args:
        target_ids: integer class ids with any shape.
        n_bins: number of ordered return bins.
        sigma_bins: Gaussian width measured in bin units. Smaller values are
            closer to one-hot labels; larger values give nearby bins more
            partial credit.

    Returns:
        Tensor shaped ``target_ids.shape + (n_bins,)`` whose last dimension sums
        to 1.0.
    """
    if sigma_bins <= 0:
        return F.one_hot(target_ids, num_classes=n_bins).to(torch.float32)

    ids = target_ids.to(torch.float32).unsqueeze(-1)
    bins = torch.arange(n_bins, device=target_ids.device, dtype=torch.float32)
    dist2 = (bins - ids).pow(2)
    targets = torch.exp(-0.5 * dist2 / (sigma_bins * sigma_bins))
    return targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy against probabilistic targets."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def candle_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    loss_type: str = "ce",
    soft_label_sigma_bins: float = 2.0,
) -> torch.Tensor:
    """Compute CandleGPT loss while preserving the same output logits.

    Supported loss types:
    - ``ce``: standard hard-label cross entropy.
    - ``soft_ce``: Gaussian-smoothed ordinal soft-label cross entropy.
    """
    if loss_type == "ce":
        return F.cross_entropy(logits, target_ids)
    if loss_type == "soft_ce":
        targets = ordinal_soft_targets(
            target_ids,
            n_bins=logits.shape[-1],
            sigma_bins=soft_label_sigma_bins,
        )
        return soft_cross_entropy(logits, targets)
    raise ValueError(f"Unsupported loss_type={loss_type!r}; expected 'ce' or 'soft_ce'.")
