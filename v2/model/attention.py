"""Causal multi-head self-attention."""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.full((cfg.block_size, cfg.block_size), float("-inf")),
                diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.cfg.d_model
        q, k, v = self.qkv(x).split(C, dim=2)
        h = self.cfg.n_heads
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)

        scale = 1.0 / math.sqrt(k.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale
        att = att + self.mask[:T, :T]
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))
