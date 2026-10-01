import torch
import torch.nn.functional as F
from torch import nn

ROPE_THETA = 10_000.0


def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on (B, H, T, D) using positions 0..T-1."""
    half = x.shape[-1] // 2
    freqs = ROPE_THETA ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * freqs
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


def window_causal_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    """(T, T) bool: position i attends to the `window` positions ending at i."""
    i = torch.arange(length, device=device)[:, None]
    j = torch.arange(length, device=device)[None, :]
    return (j <= i) & (i - j < window)


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, dim = x.shape
    return x.view(batch, length, heads, dim // heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_dim)


class WindowedSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.heads = heads
        self.window = window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (split_heads(t, self.heads) for t in self.qkv(x).chunk(3, dim=-1))
        mask = window_causal_mask(x.shape[1], self.window, x.device)
        out = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=mask)
        return self.out(merge_heads(out))


class CrossAttention(nn.Module):
    """Queries attend to keys/values under a bool mask; no positional encoding.

    A learned null key/value is always attendable, so a query whose mask row is
    empty falls back to it instead of producing NaN.
    """

    def __init__(self, dim_q: int, dim_kv: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim_q, dim_q, bias=False)
        self.kv = nn.Linear(dim_kv, 2 * dim_q, bias=False)
        self.null_kv = nn.Parameter(torch.zeros(2, dim_q))
        self.out = nn.Linear(dim_q, dim_q, bias=False)

    def forward(self, queries: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = queries.shape[0]
        k, v = self.kv(context).chunk(2, dim=-1)
        null_k, null_v = self.null_kv.expand(batch, -1, -1).chunk(2, dim=1)
        k, v = torch.cat([null_k, k], dim=1), torch.cat([null_v, v], dim=1)
        mask = F.pad(mask, (1, 0), value=True).unsqueeze(1)
        out = F.scaled_dot_product_attention(
            split_heads(self.q(queries), self.heads),
            split_heads(k, self.heads),
            split_heads(v, self.heads),
            attn_mask=mask,
        )
        return self.out(merge_heads(out))
