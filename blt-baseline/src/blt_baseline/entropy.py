"""Length-aware execution of the pinned entropy model, with native patch rules.

Only trailing padding is removed. Each request and each native max-length chunk
retains its own positional origin and causal mask. No logits or entropies are
cached between calls, so request reuse cannot expose stale state.
"""

from __future__ import annotations

import torch


class LengthAwarePatcher:
    """Compute scores without padding; delegate boundary selection to upstream.

    ``microbatch_size=1`` isolates the padding-removal intervention. Larger
    values also vectorize equal-length requests along the model's batch axis;
    they never flatten requests onto the sequence axis. Unequal-length requests
    must be bucketed by the caller, rather than padded with context bytes.
    """

    # Wider than the admitted fp32 score tolerance (including differences).
    # This is a numerical guard, not a universal forward-error proof.
    boundary_margin_nats = 0.002

    def __init__(self, native, *, microbatch_size: int = 1):
        if type(microbatch_size) is not int or microbatch_size < 1:
            raise ValueError("microbatch_size must be a positive integer")
        if native.patching_mode != "entropy":
            raise ValueError("Length-aware execution requires the entropy patcher")
        self.native = native
        self.microbatch_size = microbatch_size
        self.fallback_rows = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    @torch.inference_mode()
    def scores(self, tokens):
        from bytelatent.data.patcher import entropy

        if tokens.ndim != 2 or min(tokens.shape) < 1:
            raise ValueError("Patcher requires nonempty, equal-length prefix rows")
        model = self.native.entropy_model
        if model.training:
            raise ValueError("Length-aware entropy model must be in eval mode")
        # This is the exact native chunk/reset boundary, not a new context cap.
        max_length = getattr(model, "max_length", 8192)
        rows = []
        for first in range(0, tokens.shape[0], self.microbatch_size):
            chunks = []
            for start in range(0, tokens.shape[1], max_length):
                chunk = tokens[first:first + self.microbatch_size, start:start + max_length]
                logits = model(chunk.to(self.native.device))
                chunks.append(entropy(logits))
            rows.append(torch.cat(chunks, dim=1))
        return torch.cat(rows, dim=0)

    @torch.inference_mode()
    def patch(self, tokens, include_next_token=False, preds=None, entropies=None, threshold=None):
        # Preserve the native precedence of supplied entropies over predictions.
        if entropies is None and preds is None:
            entropies = self.scores(tokens)
            cutoff = self.native.threshold if threshold is None else threshold
            if cutoff is None:
                # Global top-k ordering has no fixed local decision margin.
                uncertain = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
            else:
                values = entropies[:, 1:]
                if self.native.monotonicity:
                    values = values[:, 1:] - values[:, :-1]
                uncertain = ((values - cutoff).abs() <= self.boundary_margin_nats).any(dim=1)
                if self.native.threshold_add is not None and not self.native.monotonicity:
                    jumps = entropies[:, 2:] - entropies[:, 1:-1]
                    uncertain |= ((jumps - self.native.threshold_add).abs() <= self.boundary_margin_nats).any(dim=1)
            # Only ambiguous requests pay the original padded reference cost.
            # The synchronization and any fallback are included in timings.
            for row in uncertain.nonzero(as_tuple=False).flatten().tolist():
                _, reference = self.native.patch(tokens[row:row + 1],
                                                  include_next_token=include_next_token,
                                                  threshold=threshold)
                entropies[row:row + 1] = reference
                self.fallback_rows += 1
        return self.native.patch(tokens, include_next_token=include_next_token,
                                 preds=preds, entropies=entropies, threshold=threshold)
