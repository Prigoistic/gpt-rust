import torch
from torch import nn

from blt.attention import WindowedSelfAttention


class TransformerBlock(nn.Module):
    """Pre-norm block: windowed causal self-attention then a GELU MLP."""

    def __init__(self, dim: int, heads: int, window: int, mlp_mult: int = 4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowedSelfAttention(dim, heads, window)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_mult * dim), nn.GELU(), nn.Linear(mlp_mult * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))
