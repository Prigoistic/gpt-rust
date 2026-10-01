import torch

from blt.attention import CrossAttention, WindowedSelfAttention, rope, window_causal_mask
from blt.blocks import TransformerBlock

torch.manual_seed(0)


def test_rope_preserves_norm():
    x = torch.randn(2, 4, 10, 16)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_relative_offset():
    vec = torch.randn(1, 1, 1, 16)
    x = vec.expand(1, 1, 12, 16).contiguous()
    rotated = rope(x)[0, 0]

    def score(m: int, n: int) -> torch.Tensor:
        return (rotated[m] * rotated[n]).sum()

    assert torch.allclose(score(5, 3), score(9, 7), atol=1e-4)


def test_window_mask_shape_and_band():
    mask = window_causal_mask(6, 3, torch.device("cpu")).int()
    assert mask.tolist() == [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ]


def test_self_attention_ignores_future_positions():
    attn = WindowedSelfAttention(32, 4, window=8).eval()
    x = torch.randn(1, 12, 32)
    y = x.clone()
    y[:, 7:] = torch.randn(1, 5, 32)
    assert torch.allclose(attn(x)[:, :7], attn(y)[:, :7], atol=1e-5)


def test_self_attention_ignores_positions_outside_window():
    attn = WindowedSelfAttention(32, 4, window=4).eval()
    x = torch.randn(1, 12, 32)
    y = x.clone()
    y[:, :5] = torch.randn(1, 5, 32)
    assert torch.allclose(attn(x)[:, 9:], attn(y)[:, 9:], atol=1e-5)


def test_cross_attention_respects_the_mask():
    cross = CrossAttention(16, 24, heads=4).eval()
    q = torch.randn(1, 3, 16)
    ctx = torch.randn(1, 5, 24)
    mask = torch.tensor([[[1, 1, 0, 0, 0]] * 3], dtype=torch.bool)
    ctx2 = ctx.clone()
    ctx2[:, 2:] = torch.randn(1, 3, 24)
    assert torch.allclose(cross(q, ctx, mask), cross(q, ctx2, mask), atol=1e-5)


def test_cross_attention_with_empty_mask_row_is_finite():
    cross = CrossAttention(16, 24, heads=4)
    out = cross(torch.randn(1, 2, 16), torch.randn(1, 4, 24), torch.zeros(1, 2, 4, dtype=torch.bool))
    assert torch.isfinite(out).all()


def test_block_preserves_shape_and_backpropagates():
    block = TransformerBlock(32, 4, window=8)
    x = torch.randn(2, 10, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert out.shape == x.shape and torch.isfinite(x.grad).all()
