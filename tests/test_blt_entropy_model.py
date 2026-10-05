import torch

from blt.entropy_model import EntropyModel, EntropyModelConfig

SMALL = EntropyModelConfig(dim=32, n_layers=2, n_heads=4, sliding_window=8, max_seqlen=64)


def model(cfg: EntropyModelConfig = SMALL) -> EntropyModel:
    torch.manual_seed(0)
    return EntropyModel(cfg).eval()


def test_ffn_hidden_dim_follows_the_llama_rule():
    assert EntropyModelConfig(dim=768).ffn_hidden_dim == 2048
    assert EntropyModelConfig(dim=2048).ffn_hidden_dim == 5632


def test_output_shape():
    assert model()(torch.randint(0, 260, (2, 20))).shape == (2, 20, 260)


def test_future_tokens_never_change_earlier_logits():
    m = model()
    a = torch.randint(3, 260, (1, 24))
    b = a.clone()
    b[:, 15:] = torch.randint(3, 260, (1, 9))
    assert torch.allclose(m(a)[:, :15], m(b)[:, :15], atol=1e-5)


def test_other_documents_cannot_influence_a_document():
    m = model()
    a = torch.tensor([[1, 9, 9, 2, 5, 6, 7, 8]])
    b = a.clone()
    b[:, 1:3] = torch.tensor([[40, 41]])
    assert torch.allclose(m(a)[:, 4:], m(b)[:, 4:], atol=1e-5)
    assert not torch.allclose(m(a)[:, 1:3], m(b)[:, 1:3], atol=1e-5)


def test_single_layer_ignores_tokens_beyond_the_window():
    cfg = EntropyModelConfig(dim=32, n_layers=1, n_heads=4, sliding_window=4, max_seqlen=64)
    m = model(cfg)
    a = torch.randint(3, 260, (1, 16))
    b = a.clone()
    b[:, :6] = torch.randint(3, 260, (1, 6))
    assert torch.allclose(m(a)[:, 10:], m(b)[:, 10:], atol=1e-5)


def test_rejects_sequences_longer_than_max_seqlen():
    try:
        model()(torch.randint(3, 260, (1, 65)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
