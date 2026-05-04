"""Pre-LN transformer block: LayerNorm → Attention → residual, LayerNorm → FFN → residual."""
from __future__ import annotations

import torch
import torch.nn as nn

from v2.model.attention import CausalSelfAttention
from v2.model.config import ModelConfig


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
