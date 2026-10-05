from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from blt.masks import sliding_window_mask
from blt.norm import RMSNorm
from blt.rope import apply_rope, rope_angles
from blt.tokens import EOS_ID, VOCAB_SIZE


@dataclass(frozen=True)
class EntropyModelConfig:
    dim: int = 768
    n_layers: int = 14
    n_heads: int = 12
    vocab_size: int = VOCAB_SIZE
    ffn_dim_multiplier: float | None = 1.0
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    sliding_window: int = 512
    max_seqlen: int = 8192
    eos_id: int = EOS_ID

    @classmethod
    def from_hf_args(cls, args: dict) -> "EntropyModelConfig":
        return cls(
            dim=args["dim"],
            n_layers=args["n_layers"],
            n_heads=args["n_heads"],
            vocab_size=args["vocab_size"],
            ffn_dim_multiplier=args["ffn_dim_multiplier"],
            multiple_of=args["multiple_of"],
            norm_eps=args["norm_eps"],
            rope_theta=args["rope_theta"],
            sliding_window=args["sliding_window"],
            max_seqlen=args["max_seqlen"],
            eos_id=args["eos_id"],
        )

    @property
    def ffn_hidden_dim(self) -> int:
        hidden = int(2 * (4 * self.dim) / 3)
        if self.ffn_dim_multiplier is not None:
            hidden = int(self.ffn_dim_multiplier * hidden)
        return self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        shape = (batch, length, self.n_heads, dim // self.n_heads)
        q = apply_rope(self.wq(x).view(shape), cos, sin)
        k = apply_rope(self.wk(x).view(shape), cos, sin)
        v = self.wv(x).view(shape)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        return self.wo(out.transpose(1, 2).reshape(batch, length, dim))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: EntropyModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention = SelfAttention(cfg.dim, cfg.n_heads)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.feed_forward = FeedForward(cfg.dim, cfg.ffn_hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), cos, sin, mask)
        return x + self.feed_forward(self.ffn_norm(x))


class EntropyModel(nn.Module):
    """Small byte-level language model whose next-byte entropy decides patch boundaries."""

    def __init__(self, cfg: EntropyModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, S) token ids -> (B, S, vocab) logits; position i predicts token i + 1."""
        cfg = self.cfg
        if tokens.shape[1] > cfg.max_seqlen:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seqlen {cfg.max_seqlen}")
        mask = sliding_window_mask(tokens, cfg.sliding_window, cfg.eos_id)
        cos, sin = rope_angles(cfg.dim // cfg.n_heads, tokens.shape[1], cfg.rope_theta, tokens.device)
        hidden = self.tok_embeddings(tokens)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask)
        return self.output(self.norm(hidden))
