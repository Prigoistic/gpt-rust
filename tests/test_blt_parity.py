from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download

from blt.patcher import patch_lengths, sequence_entropies
from blt.weights import ENTROPY_REPO, load_entropy_model

FIXTURE = Path(__file__).parent / "fixtures" / "blt_entropy_reference.npz"
CASES = ["ascii", "unicode", "repeated_700", "multi_doc", "zen", "code"]


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def model():
    try:
        hf_hub_download(ENTROPY_REPO, "model.safetensors", local_files_only=True)
    except Exception:
        pytest.skip("blt-entropy weights not in the local Hugging Face cache")
    return load_entropy_model()


@pytest.mark.parametrize("case", CASES)
def test_logits_match_official_model(model, reference, case):
    if f"{case}/logits" not in reference:
        pytest.skip("logits only stored for short cases")
    tokens = torch.from_numpy(reference[f"{case}/tokens"].astype(np.int64))
    with torch.no_grad():
        logits = model(tokens[None])[0]
    assert torch.allclose(logits, torch.from_numpy(reference[f"{case}/logits"]), atol=1e-4)


@pytest.mark.parametrize("case", CASES)
def test_entropies_and_patch_lengths_match_official(model, reference, case):
    tokens = torch.from_numpy(reference[f"{case}/tokens"].astype(np.int64))
    entropies = sequence_entropies(model, tokens)
    assert torch.allclose(entropies, torch.from_numpy(reference[f"{case}/entropies"]), atol=1e-4)
    assert patch_lengths(entropies).tolist() == reference[f"{case}/patch_lengths"].tolist()


def test_a_one_position_window_error_would_be_detected(model, reference):
    tokens = torch.from_numpy(reference["repeated_700/tokens"].astype(np.int64))
    expected = torch.from_numpy(reference["repeated_700/entropies"])
    original = model.cfg
    model.cfg = replace(original, sliding_window=511)
    try:
        wrong = sequence_entropies(model, tokens)
    finally:
        model.cfg = original
    assert not torch.allclose(wrong, expected, atol=1e-3)
