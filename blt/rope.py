import torch


def rope_angles(head_dim: int, length: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (length, head_dim // 2) for positions 0..length-1."""
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(length, device=device).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent element pairs (2i, 2i+1) of (B, S, H, D) by position-dependent angles."""
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).type_as(x)
