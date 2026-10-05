import torch

from blt.masks import document_ids, sliding_window_mask
from blt.norm import RMSNorm
from blt.rope import apply_rope, rope_angles

torch.manual_seed(0)
CPU = torch.device("cpu")


def test_rope_matches_complex_rotation_of_adjacent_pairs():
    head_dim, length, theta = 8, 6, 10_000.0
    x = torch.randn(1, length, 2, head_dim)
    cos, sin = rope_angles(head_dim, length, theta, CPU)
    out = apply_rope(x, cos, sin)

    pairs = torch.view_as_complex(x.reshape(1, length, 2, head_dim // 2, 2).contiguous())
    freqs = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    phase = torch.polar(torch.ones(length, head_dim // 2), torch.outer(torch.arange(length).float(), freqs))
    expected = torch.view_as_real(pairs * phase[None, :, None, :]).flatten(-2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_rope_position_zero_is_identity():
    x = torch.randn(1, 4, 3, 16)
    cos, sin = rope_angles(16, 4, 10_000.0, CPU)
    assert torch.allclose(apply_rope(x, cos, sin)[:, 0], x[:, 0], atol=1e-6)


def test_rope_scores_depend_only_on_relative_offset():
    vec = torch.randn(1, 1, 1, 16).expand(1, 12, 1, 16).contiguous()
    cos, sin = rope_angles(16, 12, 10_000.0, CPU)
    rotated = apply_rope(vec, cos, sin)[0, :, 0]
    assert torch.allclose((rotated[5] * rotated[3]).sum(), (rotated[9] * rotated[7]).sum(), atol=1e-4)


def test_rmsnorm_matches_torch():
    norm = RMSNorm(32, eps=1e-5)
    reference = torch.nn.RMSNorm(32, eps=1e-5)
    weight = torch.randn(32)
    norm.weight.data.copy_(weight)
    reference.weight.data.copy_(weight)
    x = torch.randn(3, 7, 32)
    assert torch.allclose(norm(x), reference(x), atol=1e-6)


def test_document_ids_end_documents_at_eos():
    tokens = torch.tensor([[5, 6, 2, 7, 2, 8]])
    assert document_ids(tokens, eos_id=2).tolist() == [[0, 0, 0, 1, 1, 2]]


def reference_mask(tokens: list[int], window: int, eos_id: int) -> torch.Tensor:
    """Per-document tril/triu construction, following xformers' block-diagonal local mask."""
    lengths, start = [], 0
    for i, t in enumerate(tokens):
        if t == eos_id or i == len(tokens) - 1:
            lengths.append(i + 1 - start)
            start = i + 1
    size = len(tokens)
    mask = torch.zeros(size, size, dtype=torch.bool)
    offset = 0
    for n in lengths:
        block = torch.triu(torch.tril(torch.ones(n, n)), diagonal=-window + 1).bool()
        mask[offset : offset + n, offset : offset + n] = block
        offset += n
    return mask


def test_mask_matches_block_diagonal_local_reference():
    cases = [([5] * 9, 4), ([5, 6, 2, 7, 8, 2, 9, 5, 5], 3), ([1, 2, 3, 4, 5, 2], 512), ([5, 5, 5, 2, 2, 5], 2)]
    for toks, window in cases:
        mask = sliding_window_mask(torch.tensor([toks]), window, eos_id=2)
        assert mask.shape == (1, 1, len(toks), len(toks))
        assert torch.equal(mask[0, 0], reference_mask(toks, window, eos_id=2)), (toks, window)


def test_window_counts_the_query_itself():
    mask = sliding_window_mask(torch.tensor([[5] * 6]), 3, eos_id=2)[0, 0]
    assert mask[5].int().tolist() == [0, 0, 0, 1, 1, 1]
