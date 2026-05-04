"""Model hyperparameter configuration for CandleGPTv2."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ModelConfig:
    n_features: int = 41
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    ffn_mult: int = 4
    block_size: int = 512
    n_bins: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"head_dim = d_model/n_heads = {self.d_model}/{self.n_heads} "
                f"is not an integer"
            )

    @property
    def ffn_dim(self) -> int:
        return self.d_model * self.ffn_mult

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
