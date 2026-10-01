import torch

from blt.ngram import ByteEmbedding


def make(dim=16, sizes=(3, 4, 5), table_size=1024):
    torch.manual_seed(0)
    return ByteEmbedding(dim, sizes, table_size)


def test_output_shape():
    emb = make()
    out = emb(torch.randint(0, 256, (2, 20)))
    assert out.shape == (2, 20, 16)


def test_hash_is_in_table_range():
    emb = make(table_size=97)
    ids = torch.randint(0, 256, (4, 64))
    for n in emb.ngram_sizes:
        hashes = emb.hash_ngrams(ids, n)
        assert hashes.min() >= 0 and hashes.max() < 97


def test_same_ngram_hashes_equal_at_any_position():
    emb = make()
    ids = torch.tensor([[1, 2, 3, 9, 9, 1, 2, 3]])
    hashes = emb.hash_ngrams(ids, 3)
    assert hashes[0, 2] == hashes[0, 7]


def test_different_ngrams_hash_differently():
    emb = make(table_size=1 << 20)
    ids = torch.tensor([[1, 2, 3, 3, 2, 1]])
    hashes = emb.hash_ngrams(ids, 3)
    assert hashes[0, 2] != hashes[0, 5]


def test_hash_ignores_bytes_after_the_position():
    emb = make()
    a = torch.tensor([[5, 6, 7, 8, 9]])
    b = torch.tensor([[5, 6, 7, 1, 2]])
    for n in emb.ngram_sizes:
        assert torch.equal(emb.hash_ngrams(a, n)[:, :3], emb.hash_ngrams(b, n)[:, :3])


def test_gradients_reach_byte_and_ngram_tables():
    emb = make()
    emb(torch.randint(0, 256, (2, 12))).sum().backward()
    assert emb.bytes.weight.grad.abs().sum() > 0
    assert all(t.weight.grad.abs().sum() > 0 for t in emb.tables)
