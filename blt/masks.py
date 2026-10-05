import torch


def document_ids(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """(B, S) document index per position; an EOS token is the last token of its document."""
    is_eos = (tokens == eos_id).long()
    return is_eos.cumsum(dim=1) - is_eos


def sliding_window_mask(tokens: torch.Tensor, window: int, eos_id: int) -> torch.Tensor:
    """(B, 1, S, S) bool, True where query q may attend key k.

    Causal, within the same document, and at most `window` positions back
    (the query itself counts as one of them).
    """
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    distance = positions[:, None] - positions[None, :]
    in_window = (distance >= 0) & (distance < window)
    docs = document_ids(tokens, eos_id)
    same_doc = docs[:, :, None] == docs[:, None, :]
    return (in_window[None] & same_doc).unsqueeze(1)
