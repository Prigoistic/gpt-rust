"""State/shape controls; real-weight numerical parity is a separate GPU gate."""

from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from blt_baseline.entropy import LengthAwarePatcher
from blt_baseline.evidence import Config


@pytest.fixture
def native(monkeypatch):
    module = ModuleType("bytelatent.data.patcher")
    module.entropy = lambda scores: -(scores.log_softmax(-1).exp() * scores.log_softmax(-1)).sum(-1)
    monkeypatch.setitem(sys.modules, "bytelatent.data.patcher", module)

    class CausalModel(torch.nn.Module):
        max_length = 8

        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, tokens):
            self.seen.append(tokens.clone())
            values = tokens.float().cumsum(1) / 100
            return torch.stack((values, -values, values * 0), dim=-1)

    model = CausalModel().eval()
    def patch(tokens, **kwargs):
        return kwargs, kwargs["entropies"]
    return SimpleNamespace(entropy_model=model, device="cpu", patching_mode="entropy", patch=patch,
                           threshold=1.335, threshold_add=None, monotonicity=False)


@pytest.mark.parametrize("width", [1, 7, 8, 9, 17])
def test_exact_causal_scores_match_row_padded_reference_and_native_chunk_resets(native, width):
    tokens = torch.arange(3 * width).reshape(3, width)
    expected = []
    entropy = sys.modules["bytelatent.data.patcher"].entropy
    for row in tokens:
        chunks = []
        for start in range(0, width, 8):
            chunk = row[start:start + 8]
            padded = torch.nn.functional.pad(chunk, (0, 8 - len(chunk)))
            chunks.append(entropy(native.entropy_model(padded[None]))[:, :len(chunk)])
        expected.append(torch.cat(chunks, 1))
    native.entropy_model.seen.clear()
    wrapper = LengthAwarePatcher(native, microbatch_size=2)
    scores = wrapper.scores(tokens)
    torch.testing.assert_close(scores, torch.cat(expected), atol=0, rtol=0)
    assert all(call.shape[0] <= 2 and call.shape[1] <= 8 for call in native.entropy_model.seen)
    assert sum(call.numel() for call in native.entropy_model.seen) == tokens.numel()
    # A changed preceding request cannot affect the last request's scores.
    changed = tokens.clone()
    changed[0] += 100
    assert torch.equal(wrapper.scores(changed)[-1], scores[-1])


def test_patch_delegates_supplied_values_and_all_boundary_arguments(native):
    wrapper = LengthAwarePatcher(native)
    tokens = torch.tensor([[1, 2, 3]])
    supplied = torch.tensor([[.1, .2, .3]])
    preds = torch.zeros(1, 3, 3)
    args, scores = wrapper.patch(tokens, True, preds=preds, entropies=supplied, threshold=1.7)
    assert scores is supplied and args["preds"] is preds
    assert args["include_next_token"] is True and args["threshold"] == 1.7
    wrapper.patch(tokens, preds=preds)
    assert not native.entropy_model.seen


def test_execution_is_stateless_under_request_replacement_and_reordering(native):
    wrapper = LengthAwarePatcher(native, microbatch_size=4)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    before = wrapper.scores(tokens)
    wrapper.scores(torch.tensor([[9, 8, 7, 6, 5]]))
    assert torch.equal(wrapper.scores(tokens.flip(0)), before.flip(0))
    assert not before.requires_grad


def test_rejects_training_empty_inputs_and_invalid_microbatch(native):
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            LengthAwarePatcher(native, microbatch_size=invalid)
    wrapper = LengthAwarePatcher(native)
    for shape in ((0, 2), (2, 0), (2,)):
        with pytest.raises(ValueError, match="nonempty"):
            wrapper.scores(torch.empty(shape))
    native.entropy_model.train()
    with pytest.raises(ValueError, match="eval"):
        wrapper.scores(torch.ones(1, 2, dtype=torch.long))


def test_config_requires_explicit_isolated_semantics():
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/local.json")
    with pytest.raises(ValueError, match="isolated"):
        replace(cfg, entropy_execution="length_aware_batch").validate()
    replace(cfg, entropy_execution="length_aware_batch", patcher_batch_policy="isolated_rows").validate()
    with pytest.raises(ValueError, match="positive integer"):
        replace(cfg, entropy_microbatch_size=0).validate()


def test_near_threshold_row_uses_reference_scores_only_for_that_request(native):
    wrapper = LengthAwarePatcher(native, microbatch_size=4)
    native.threshold = 1.0
    wrapper.scores = lambda tokens: torch.tensor([[.5, 1.00001, .7], [.4, .3, .2]])
    seen = []
    def patch(tokens, **kwargs):
        if kwargs.get("entropies") is None:
            seen.append(tokens.clone())
            return None, torch.tensor([[.5, .99999, .7]])
        return kwargs, kwargs["entropies"]
    native.patch = patch
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    _, scores = wrapper.patch(tokens, include_next_token=True)
    assert torch.equal(seen[0], tokens[:1]) and len(seen) == 1
    assert scores[0, 1] < native.threshold
    assert wrapper.fallback_rows == 1


def test_guard_is_safe_when_called_outside_an_outer_inference_context(native):
    wrapper = LengthAwarePatcher(native)
    tokens = torch.tensor([[1, 2, 3]])
    native.threshold = float(wrapper.scores(tokens)[0, 1])
    entropy = sys.modules["bytelatent.data.patcher"].entropy
    def patch(tokens, **kwargs):
        scores = kwargs.get("entropies")
        if scores is None:
            scores = entropy(native.entropy_model(tokens))
        return None, scores
    native.patch = patch
    _, scores = wrapper.patch(tokens, include_next_token=True)
    assert wrapper.fallback_rows == 1 and scores.shape == tokens.shape
