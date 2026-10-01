from dataclasses import dataclass


@dataclass(frozen=True)
class LocalConfig:
    local_dim: int = 256
    global_dim: int = 512
    heads: int = 4
    encoder_layers: int = 1
    decoder_layers: int = 3
    window: int = 512
    ngram_sizes: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    ngram_table_size: int = 1 << 16
