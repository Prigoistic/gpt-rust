import torch
import torch.nn.functional as F
from torch import nn

HASH_BASE = 1_000_003


class ByteEmbedding(nn.Module):
    """Byte embedding plus hashed byte n-gram embeddings, averaged over all terms."""

    def __init__(self, dim: int, ngram_sizes: tuple[int, ...], table_size: int):
        super().__init__()
        self.ngram_sizes = ngram_sizes
        self.table_size = table_size
        self.bytes = nn.Embedding(256, dim)
        self.tables = nn.ModuleList(nn.Embedding(table_size, dim) for _ in ngram_sizes)
        powers = [pow(HASH_BASE, k, table_size) for k in range(max(ngram_sizes))]
        self.register_buffer("powers", torch.tensor(powers), persistent=False)

    def hash_ngrams(self, byte_ids: torch.Tensor, n: int) -> torch.Tensor:
        """Polynomial hash of the n bytes ending at each position; missing history hashes as zero."""
        padded = F.pad(byte_ids + 1, (n - 1, 0))
        windows = padded.unfold(1, n, 1)
        return (windows * self.powers[:n]).sum(-1) % self.table_size

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        total = self.bytes(byte_ids)
        for n, table in zip(self.ngram_sizes, self.tables):
            total = total + table(self.hash_ngrams(byte_ids, n))
        return total / (len(self.ngram_sizes) + 1)
