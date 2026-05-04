"""CandleGPTv2: continuous-input causal transformer for next-bar return prediction.

Input:  (batch, seq_len, n_features=41) float32
Output: (batch, seq_len, n_bins=256) float32 logits
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.model.config import ModelConfig
from v2.model.transformer import TransformerBlock


class CandleGPTv2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model, bias=False)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_bins, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
            )
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.input_proj(x) + self.pos_embed(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

    def num_params(self, exclude_pos_embed: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_pos_embed:
            total -= self.pos_embed.weight.numel()
        return total

    @torch.no_grad()
    def generate_ids(
        self,
        x: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            out_ids = []
            ctx = x
            for _ in range(n_steps):
                ctx_crop = ctx[:, -self.cfg.block_size:, :]
                logits = self.forward(ctx_crop)
                logits_last = logits[:, -1, :] / max(temperature, 1e-8)
                if top_k is not None:
                    v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                    logits_last[logits_last < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits_last, dim=-1)
                ids = torch.multinomial(probs, num_samples=1)
                out_ids.append(ids)
                ctx = torch.cat([ctx, ctx[:, -1:, :]], dim=1)
            return torch.cat(out_ids, dim=1)
        finally:
            self.train(was_training)
