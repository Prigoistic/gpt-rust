import torch

from blt.config import LocalConfig
from blt.local import LocalEncoder
from blt.patching import patch_ids_from_starts

CFG = LocalConfig(local_dim=32, global_dim=48, heads=4, encoder_layers=2, window=16,
                  ngram_sizes=(3, 4), ngram_table_size=512)
STARTS = torch.tensor([[1, 0, 0, 1, 0, 1, 0, 0]], dtype=torch.bool)
IDS = patch_ids_from_starts(STARTS)


def encoder() -> LocalEncoder:
    torch.manual_seed(0)
    return LocalEncoder(CFG).eval()


def test_output_shapes():
    patches, hidden = encoder()(torch.randint(0, 256, (1, 8)), IDS, 3)
    assert patches.shape == (1, 3, 48)
    assert hidden.shape == (1, 8, 32)


def test_patches_ignore_bytes_after_their_end():
    enc = encoder()
    a = torch.randint(0, 256, (1, 8))
    b = a.clone()
    b[:, 5:] = torch.randint(0, 256, (1, 3))
    pa, ha = enc(a, IDS, 3)
    pb, hb = enc(b, IDS, 3)
    assert torch.allclose(pa[:, :2], pb[:, :2], atol=1e-5)
    assert torch.allclose(ha[:, :5], hb[:, :5], atol=1e-5)


def test_patch_depends_on_its_own_bytes():
    enc = encoder()
    a = torch.randint(0, 256, (1, 8))
    b = a.clone()
    b[:, 4] = (b[:, 4] + 1) % 256
    pa, _ = enc(a, IDS, 3)
    pb, _ = enc(b, IDS, 3)
    assert not torch.allclose(pa[:, 1], pb[:, 1], atol=1e-5)


def test_empty_patch_slots_stay_finite():
    patches, _ = encoder()(torch.randint(0, 256, (1, 8)), IDS, 5)
    assert torch.isfinite(patches).all()


def test_gradients_are_finite():
    enc = LocalEncoder(CFG)
    patches, hidden = enc(torch.randint(0, 256, (2, 8)), IDS.expand(2, -1), 3)
    (patches.sum() + hidden.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None)
