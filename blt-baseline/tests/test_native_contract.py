"""Behavioral checks of the harness using a deterministic CPU model, never BLT evidence."""

import math
from types import SimpleNamespace

import pytest
import torch

from blt_baseline.native import NativeBLT, upstream_import_compatibility, move_parameters_preserving_buffers
from blt_baseline.native import generic_flex_call
from blt_baseline.native import IsolatedRowPatcher


class ByteTokenizer:
    bos_id = 1

    def encode(self, text, add_bos, add_eos):
        assert not add_eos
        return ([1] if add_bos else []) + [b + 4 for b in text.encode()]


class Patcher:
    def __init__(self):
        self.seen = []

    def patch(self, tokens, include_next_token):
        assert include_next_token
        self.seen.append(tokens.clone())
        return torch.ones((tokens.shape[0], tokens.shape[1] + 1), dtype=torch.long), None


class DeterministicModel:
    def __call__(self, tokens, patch_lengths):
        assert patch_lengths.sum() == tokens.numel() + tokens.shape[0]
        # Predict 'A' with a fixed full-vocabulary distribution.
        logits = torch.zeros((*tokens.shape, 260))
        logits[:, :, ord("A") + 4] = 2
        return logits


@pytest.fixture
def adapter():
    instance = NativeBLT.__new__(NativeBLT)
    instance.tokenizer = ByteTokenizer()
    instance.model = DeterministicModel()
    instance.patcher = Patcher()
    instance.device = torch.device("cpu")
    instance.max_prefix_tokens = 128
    instance.tracing = False
    return instance


def test_generation_reuses_prefix_and_counts_only_committed_bytes(adapter):
    result = adapter.generate(["é", "ab"], 3, diagnose=True)
    assert result["ids"] == [[69, 69, 69], [69, 69, 69]]
    assert [o["raw_byte_count"] for o in result["outputs"]] == [3, 3]
    assert [t.shape[1] for t in adapter.patcher.seen] == [3, 4, 5]
    assert result["diagnostics"]["model_forward_calls"] == 3
    assert result["diagnostics"]["prefix_byte_or_bos_positions_reprocessed"] == 24


def test_generation_batch_isolation(adapter):
    alone = adapter.generate(["ab"], 3)["ids"][0]
    batch = adapter.generate(["ab", "cd"], 3)["ids"]
    assert batch == [alone, alone]


def test_rejects_unequal_length_no_padding(adapter):
    with pytest.raises(ValueError, match="equal-byte-length"):
        adapter.generate(["a", "é"], 3)
    assert not adapter.patcher.seen


def test_generation_context_limit_checked_before_compute(adapter):
    adapter.max_prefix_tokens = 4
    with pytest.raises(ValueError, match="context"):
        adapter.generate(["abc"], 2)
    assert not adapter.patcher.seen


def test_scoring_targets_are_shifted_and_never_leak(adapter):
    result = adapter.score("p", "AB")
    seen = [row[0].tolist() for row in adapter.patcher.seen]
    assert seen == [[1, ord("p") + 4], [1, ord("p") + 4, ord("A") + 4]]
    normalizer = math.log(math.exp(2) + 259)
    assert result["nll_nats"] == pytest.approx(2 * normalizer - 2, abs=1e-6)
    assert result["scored_bytes"] == 2
    assert result["bpb"] == pytest.approx((normalizer - 1) / math.log(2), abs=1e-6)


def test_score_counts_multibyte_continuation_bytes(adapter):
    assert adapter.score("p", "é")["scored_bytes"] == 2


def test_score_context_limit_does_not_truncate(adapter):
    adapter.max_prefix_tokens = 3
    with pytest.raises(ValueError, match="context"):
        adapter.score("ab", "AA")
    assert not adapter.patcher.seen


def test_no_finite_loss_no_quality_result(adapter):
    class BrokenModel:
        def __call__(self, tokens, patch_lengths):
            return torch.full((*tokens.shape, 260), float("nan"))
    adapter.model = BrokenModel()
    with pytest.raises(RuntimeError, match="Nonfinite"):
        adapter.score("p", "A")


def test_eos_is_not_implicitly_counted_as_output_byte(adapter):
    class EOSModel:
        def __call__(self, tokens, patch_lengths):
            logits = torch.zeros((*tokens.shape, 260))
            logits[:, :, 2] = 1
            return logits
    adapter.model = EOSModel()
    result = adapter.generate(["p"], 3)
    assert result["ids"] == [[2, 2, 2]]
    assert result["outputs"][0]["raw_byte_count"] == 0
    assert result["outputs"][0]["special_token_count"] == 3


def test_trace_hooks_removed_after_error(adapter, monkeypatch):
    import sys
    from types import ModuleType

    root = ModuleType("bytelatent")
    model = ModuleType("bytelatent.model")
    blt = ModuleType("bytelatent.model.blt")
    original = lambda: None
    blt.compute_hash_embeddings = original
    root.model = model
    model.blt = blt
    monkeypatch.setitem(sys.modules, "bytelatent", root)
    monkeypatch.setitem(sys.modules, "bytelatent.model", model)
    monkeypatch.setitem(sys.modules, "bytelatent.model.blt", blt)
    modules = [torch.nn.Identity() for _ in range(4)]
    adapter.model = SimpleNamespace(local_encoder=modules[0], global_transformer=modules[1], local_decoder=modules[2])
    adapter.entropy_model = modules[3]
    with pytest.raises(RuntimeError, match="interrupted"):
        with adapter.profile_regions():
            assert blt.compute_hash_embeddings is not original
            raise RuntimeError("interrupted")
    assert blt.compute_hash_embeddings is original
    assert not adapter.tracing
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in modules)


def pipeline_setup(adapter, tmp_path, monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    from blt_baseline.evidence import Config

    root = Path(__file__).resolve().parents[1]
    cfg = replace(Config.load(root / "configs/local.json"), context_bytes=(8,), batch_sizes=(1,),
                  output_steps=3, warmup=1, repeats=2, profile=False)
    state = {"blockers": [], "upstream": {"revision": "pinned"}, "cuda": {"available": True}}
    monkeypatch.setattr("blt_baseline.runner.doctor", lambda _cfg: state)
    monkeypatch.setattr("blt_baseline.runner.collect_manifest", lambda _cfg: {"test_double": True})
    monkeypatch.setattr("blt_baseline.native.NativeBLT", lambda _cfg: adapter)
    monkeypatch.setattr("blt_baseline.runner.load_workload", lambda _path: [
        {"id": "test", "category": "control", "prompt": "p", "continuation": "A", "exact_match": True}
    ])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    for name in ("memory_allocated", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 10)
    return cfg


def test_pipeline_writes_complete_artifacts_with_cpu_test_double(adapter, tmp_path, monkeypatch):
    import json
    from blt_baseline.runner import execute

    cfg = pipeline_setup(adapter, tmp_path, monkeypatch)
    adapter.check_parity = lambda _prompt: {"status": "PASS", "test_double": True}
    out = tmp_path / "run"
    assert execute(cfg, out)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "COMPLETED"
    assert result["quality_summary"]["fixed_length_correct"] == 1
    assert len(result["performance"][0]["samples"]) == 2
    assert result["performance"][0]["samples"][0]["raw_byte_count"] == 3
    assert result["performance"][0]["diagnostics"]["model_forward_calls"] == 3
    assert (out / "manifest.json").exists()
    assert (out / "harness-source.tar.gz").exists()
    from blt_baseline.evidence import sha256
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["harness_archive_sha256"] == sha256(out / "harness-source.tar.gz")
    assert "not whole-answer accuracy" in (out / "report.md").read_text()


def test_parity_failure_prevents_quality_and_performance(adapter, tmp_path, monkeypatch):
    import json
    from blt_baseline.runner import execute

    cfg = pipeline_setup(adapter, tmp_path, monkeypatch)
    def fail(_prompt):
        raise RuntimeError("parity mismatch")
    adapter.check_parity = fail
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="parity mismatch"):
        execute(cfg, out)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "FAILED"
    assert result["performance"] == []
    assert result["quality"] == []
    assert "parity mismatch" in result["error"]


@pytest.mark.parametrize("torch_version,xf_version,accepted", [
    ("2.10.0+cu128", "0.0.35", True),
    ("2.6.0+cu124", "0.0.29.post3", False),
])
def test_training_registry_alias_is_exact_and_removed_on_error(monkeypatch, torch_version, xf_version, accepted):
    import sys
    from types import ModuleType

    package = ModuleType("xformers")
    package.__version__ = xf_version
    ops = ModuleType("xformers.ops")
    fmha = ModuleType("xformers.ops.fmha")
    fmha.cutlass = SimpleNamespace(FwOp=SimpleNamespace(OPERATOR=torch.ops.aten._efficient_attention_forward))
    monkeypatch.setitem(sys.modules, "xformers", package)
    monkeypatch.setitem(sys.modules, "xformers.ops", ops)
    monkeypatch.setitem(sys.modules, "xformers.ops.fmha", fmha)
    monkeypatch.setattr(torch, "__version__", torch_version)
    assert not hasattr(torch.ops.xformers, "efficient_attention_forward_cutlass")
    if not accepted:
        with pytest.raises(RuntimeError, match="admitted only"):
            with upstream_import_compatibility(True):
                pass
        assert not hasattr(torch.ops.xformers, "efficient_attention_forward_cutlass")
        return
    with pytest.raises(RuntimeError, match="test cleanup"):
        with upstream_import_compatibility(True):
            assert torch.ops.xformers.efficient_attention_forward_cutlass is torch.ops.aten._efficient_attention_forward
            raise RuntimeError("test cleanup")
    assert not hasattr(torch.ops.xformers, "efficient_attention_forward_cutlass")


def test_main_precision_conversion_preserves_rope_and_complex_buffers():
    model = torch.nn.Linear(2, 2)
    rope = torch.tensor([0.123456789, 0.987654321], dtype=torch.float32)
    complex_table = torch.tensor([1 + 2j], dtype=torch.complex64)
    model.register_buffer("rope", rope.clone())
    model.register_buffer("complex_table", complex_table.clone())
    move_parameters_preserving_buffers(model, device="cpu", dtype=torch.float16)
    assert model.weight.dtype == torch.float16
    assert model.rope.dtype == torch.float32
    assert torch.equal(model.rope, rope)
    assert model.complex_table.dtype == torch.complex64
    assert torch.equal(model.complex_table, complex_table)


def test_generic_flex_selects_kernel_without_dropping_mask_or_other_options():
    seen = {}
    def compiled(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return "result"
    q, k, v, mask = object(), object(), object(), object()
    assert generic_flex_call(compiled)(q, k, v, block_mask=mask, kernel_options={"SOME_OTHER_OPTION": 7}) == "result"
    assert seen["args"] == (q, k, v)
    assert seen["kwargs"]["block_mask"] is mask
    assert seen["kwargs"]["kernel_options"] == {"SOME_OTHER_OPTION": 7, "FORCE_USE_FLEX_ATTENTION": True}


class LeakyBatchPatcher:
    realtime_patching = True

    def patch(self, tokens, include_next_token=False, preds=None, entropies=None, threshold=None):
        assert include_next_token
        scores = tokens.flatten().cumsum(0).view_as(tokens).float() if entropies is None else entropies
        rows = [torch.tensor([1, 1, 2]) if row[0] < 10 else torch.tensor([1, 3]) for row in tokens]
        return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True), scores


def test_row_patcher_prevents_cross_request_dependence_and_pads_lengths():
    native = LeakyBatchPatcher()
    wrapper = IsolatedRowPatcher(native)
    tokens = torch.tensor([[1, 2, 3], [11, 12, 13]])
    actual_lengths, actual_scores = wrapper.patch(tokens, include_next_token=True)
    _, leaky = native.patch(tokens, include_next_token=True)
    assert actual_lengths.tolist() == [[1, 1, 2], [1, 3, 0]]
    assert actual_scores.tolist() == [[1, 3, 6], [11, 23, 36]]
    assert not torch.equal(actual_scores, leaky)
    assert wrapper.realtime_patching
    for i in range(2):
        lengths, scores = wrapper.patch(tokens[i:i + 1], include_next_token=True)
        assert torch.equal(scores[0], actual_scores[i])
        assert lengths.sum() == tokens.shape[1] + 1


def test_row_patcher_slices_precomputed_scores_and_rejects_empty_batch():
    wrapper = IsolatedRowPatcher(LeakyBatchPatcher())
    tokens = torch.tensor([[1, 2, 3], [11, 12, 13]])
    supplied = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
    _, actual = wrapper.patch(tokens, include_next_token=True, entropies=supplied)
    assert torch.equal(actual, supplied)
    with pytest.raises(ValueError, match="nonempty"):
        wrapper.patch(torch.empty((0, 3)), include_next_token=True)
