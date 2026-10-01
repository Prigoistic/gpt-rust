import torch

from blt.blocks import TransformerBlock
from blt.config import LocalConfig
from blt.local import LocalDecoder, LocalEncoder
from blt.patching import patch_ids_from_starts

CFG = LocalConfig(local_dim=32, global_dim=48, heads=4, encoder_layers=1, decoder_layers=2,
                  window=16, ngram_sizes=(3, 4), ngram_table_size=512)
STARTS = torch.tensor([[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.bool)
IDS = patch_ids_from_starts(STARTS)
NUM_PATCHES = 4


class Pipeline(torch.nn.Module):
    """Encoder -> causal stand-in for the global transformer -> decoder."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = LocalEncoder(CFG)
        self.global_model = TransformerBlock(CFG.global_dim, CFG.heads, window=64)
        self.decoder = LocalDecoder(CFG)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        patches, hidden = self.encoder(byte_ids, IDS, NUM_PATCHES)
        return self.decoder(hidden, self.global_model(patches), IDS)


def test_logits_shape():
    assert Pipeline().eval()(torch.randint(0, 256, (1, 10))).shape == (1, 10, 256)


def test_future_bytes_never_change_earlier_logits():
    model = Pipeline().eval()
    original = torch.randint(0, 256, (1, 10))
    base = model(original)
    for k in range(1, 10):
        changed = original.clone()
        changed[:, k:] = torch.randint(0, 256, (1, 10 - k))
        assert torch.allclose(model(changed)[:, :k], base[:, :k], atol=1e-5), f"leak at k={k}"


def test_first_patch_bytes_have_finite_logits():
    assert torch.isfinite(Pipeline().eval()(torch.randint(0, 256, (1, 10)))).all()


def test_decoder_uses_the_patch_path():
    model = Pipeline().eval()
    ids = torch.randint(0, 256, (1, 10))
    patches, hidden = model.encoder(ids, IDS, NUM_PATCHES)
    real = model.decoder(hidden, model.global_model(patches), IDS)
    zeroed = model.decoder(hidden, torch.zeros_like(patches), IDS)
    assert not torch.allclose(real[:, 3:], zeroed[:, 3:], atol=1e-5)


def test_end_to_end_gradients_are_finite():
    model = Pipeline()
    ids = torch.randint(0, 256, (1, 10))
    model(ids).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
