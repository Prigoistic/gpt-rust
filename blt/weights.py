import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from blt.entropy_model import EntropyModel, EntropyModelConfig

ENTROPY_REPO = "facebook/blt-entropy"


def load_entropy_model(
    repo_id: str = ENTROPY_REPO, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> EntropyModel:
    with open(hf_hub_download(repo_id, "config.json")) as f:
        cfg = EntropyModelConfig.from_hf_args(json.load(f)["args"])
    model = EntropyModel(cfg)
    model.load_state_dict(load_file(hf_hub_download(repo_id, "model.safetensors")))
    return model.to(dtype=dtype, device=device).eval()
