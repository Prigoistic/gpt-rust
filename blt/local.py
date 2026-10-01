import torch
from torch import nn

from blt.attention import CrossAttention
from blt.blocks import TransformerBlock
from blt.config import LocalConfig
from blt.ngram import ByteEmbedding
from blt.patching import decoder_cross_mask, encoder_cross_mask, max_pool_patches


class LocalEncoder(nn.Module):
    """Bytes -> patch representations.

    Patch queries start as the max-pool of the byte embeddings in each patch and
    are refined after every byte layer by cross-attending to that patch's bytes.
    """

    def __init__(self, cfg: LocalConfig):
        super().__init__()
        self.embed = ByteEmbedding(cfg.local_dim, cfg.ngram_sizes, cfg.ngram_table_size)
        self.patch_init = nn.Linear(cfg.local_dim, cfg.local_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.local_dim, cfg.heads, cfg.window) for _ in range(cfg.encoder_layers)
        )
        self.cross = nn.ModuleList(
            CrossAttention(cfg.local_dim, cfg.local_dim, cfg.heads) for _ in range(cfg.encoder_layers)
        )
        self.query_norms = nn.ModuleList(nn.LayerNorm(cfg.local_dim) for _ in range(cfg.encoder_layers))
        self.context_norms = nn.ModuleList(nn.LayerNorm(cfg.local_dim) for _ in range(cfg.encoder_layers))
        self.to_global = nn.Linear(cfg.local_dim, cfg.global_dim)

    def forward(
        self, byte_ids: torch.Tensor, patch_ids: torch.Tensor, num_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (patches (B, P, global_dim), byte hiddens (B, T, local_dim))."""
        hidden = self.embed(byte_ids)
        patches = self.patch_init(max_pool_patches(hidden, patch_ids, num_patches))
        mask = encoder_cross_mask(patch_ids, num_patches)
        for block, cross, q_norm, c_norm in zip(
            self.blocks, self.cross, self.query_norms, self.context_norms
        ):
            hidden = block(hidden)
            patches = patches + cross(q_norm(patches), c_norm(hidden), mask)
        return self.to_global(patches), hidden


class LocalDecoder(nn.Module):
    """Byte hiddens + global patch outputs -> next-byte logits.

    Each layer cross-attends bytes (queries) to the patches that finished at or
    before them, then runs a windowed causal byte block.
    """

    def __init__(self, cfg: LocalConfig):
        super().__init__()
        self.context_norm = nn.LayerNorm(cfg.global_dim)
        self.cross = nn.ModuleList(
            CrossAttention(cfg.local_dim, cfg.global_dim, cfg.heads) for _ in range(cfg.decoder_layers)
        )
        self.query_norms = nn.ModuleList(nn.LayerNorm(cfg.local_dim) for _ in range(cfg.decoder_layers))
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.local_dim, cfg.heads, cfg.window) for _ in range(cfg.decoder_layers)
        )
        self.final_norm = nn.LayerNorm(cfg.local_dim)
        self.head = nn.Linear(cfg.local_dim, 256, bias=False)

    def forward(
        self, byte_hidden: torch.Tensor, patches: torch.Tensor, patch_ids: torch.Tensor
    ) -> torch.Tensor:
        """Logits (B, T, 256); position i predicts byte i + 1."""
        mask = decoder_cross_mask(patch_ids, patches.shape[1])
        context = self.context_norm(patches)
        hidden = byte_hidden
        for cross, q_norm, block in zip(self.cross, self.query_norms, self.blocks):
            hidden = hidden + cross(q_norm(hidden), context, mask)
            hidden = block(hidden)
        return self.head(self.final_norm(hidden))
