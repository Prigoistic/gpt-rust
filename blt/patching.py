import torch


def patch_ids_from_starts(starts: torch.Tensor) -> torch.Tensor:
    """Patch index per byte from a bool mask marking the first byte of each patch."""
    return starts.long().cumsum(dim=1) - 1


def max_pool_patches(x: torch.Tensor, patch_ids: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Elementwise max over the bytes of each patch; patches with no bytes pool to zero."""
    batch, _, dim = x.shape
    index = patch_ids.unsqueeze(-1).expand(-1, -1, dim)
    out = x.new_zeros(batch, num_patches, dim)
    return out.scatter_reduce(1, index, x, reduce="amax", include_self=False)


def encoder_cross_mask(patch_ids: torch.Tensor, num_patches: int) -> torch.Tensor:
    """(B, P, T): patch j may attend only to the bytes of patch j."""
    patches = torch.arange(num_patches, device=patch_ids.device)
    return patches[None, :, None] == patch_ids[:, None, :]


def completed_patch(patch_ids: torch.Tensor) -> torch.Tensor:
    """Latest patch whose last byte is at or before each position, or -1 if none."""
    ends = torch.ones_like(patch_ids, dtype=torch.bool)
    ends[:, :-1] = patch_ids[:, 1:] != patch_ids[:, :-1]
    return torch.where(ends, patch_ids, patch_ids - 1)


def decoder_cross_mask(patch_ids: torch.Tensor, num_patches: int) -> torch.Tensor:
    """(B, T, P): a byte may attend only to patches that finished at or before it.

    Patch outputs summarise every byte up to the end of their patch, so any later
    patch would leak bytes the position is supposed to predict.
    """
    patches = torch.arange(num_patches, device=patch_ids.device)
    return patches[None, None, :] <= completed_patch(patch_ids)[:, :, None]
