import torch
import torch.nn.functional as F

from blt.entropy_model import EntropyModel

DEFAULT_THRESHOLD = 1.335442066192627


def next_byte_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy in nats of the predicted next-byte distribution at every position."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@torch.no_grad()
def sequence_entropies(model: EntropyModel, tokens: torch.Tensor) -> torch.Tensor:
    """(S,) entropies for one sequence; sequences beyond max_seqlen are scored in independent chunks."""
    chunks = tokens.split(model.cfg.max_seqlen)
    return torch.cat([next_byte_entropy(model(chunk[None])[0]) for chunk in chunks])


def patch_start_ids(entropies: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> torch.Tensor:
    """Token 0 and 1 always start patches; token p >= 2 starts one when the entropy at p - 1 exceeds the threshold."""
    length = entropies.shape[0]
    if length <= 2:
        return torch.arange(length, device=entropies.device)
    over = (entropies[1:-1] > threshold).nonzero().flatten() + 2
    return torch.cat([torch.arange(2, device=entropies.device), over])


def patch_lengths_from_starts(starts: torch.Tensor, length: int) -> torch.Tensor:
    ends = torch.cat([starts[1:] - 1, starts.new_tensor([length - 1])])
    return ends - starts + 1


def patch_lengths(entropies: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> torch.Tensor:
    return patch_lengths_from_starts(patch_start_ids(entropies, threshold), entropies.shape[0])
