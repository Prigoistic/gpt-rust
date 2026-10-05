"""Generate reference fixtures from Meta's official BLT code (facebookresearch/blt).

Run with the official repo on PYTHONPATH; if xformers is not installed, put a stub
`xformers` package on PYTHONPATH too (only the dense-mask SDPA path is used):

    PYTHONPATH=/path/to/blt:/path/to/stubs uv run --with pydantic python scripts/make_blt_fixtures.py
"""

import codecs
import contextlib
import io
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    import this  # Zen of Python, used as a mid-size English text

# the official patcher imports distributed-training code that needs xformers kernels
sys.modules.setdefault("bytelatent.distributed", types.SimpleNamespace(get_local_rank=lambda: 0))

from bytelatent.data.patcher import (  # noqa: E402
    entropy,
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)
from bytelatent.transformer import LMTransformer  # noqa: E402

from blt import tokens  # noqa: E402
from blt.masks import sliding_window_mask  # noqa: E402
from blt.patcher import DEFAULT_THRESHOLD  # noqa: E402

LOGIT_MAX_LEN = 128
OUT = ROOT / "tests" / "fixtures" / "blt_entropy_reference.npz"


def cases() -> dict[str, list[int]]:
    zen = codecs.decode(this.s, "rot13")
    code = Path(__file__).read_text()[:1500]
    return {
        "ascii": tokens.encode("Hello, world! This is a test of the entropy model."),
        "unicode": tokens.encode("naïve café 日本語 😀 Привет мир"),
        "repeated_700": tokens.encode(("The quick brown fox jumps over the lazy dog. " * 16)[:700]),
        "multi_doc": tokens.encode("first doc here", eos=True) + tokens.encode("second doc follows"),
        "zen": tokens.encode(zen),
        "code": tokens.encode(code),
    }


if __name__ == "__main__":
    model = LMTransformer.from_pretrained("facebook/blt-entropy").float().eval()
    arrays: dict[str, np.ndarray] = {}
    for name, ids in cases().items():
        t = torch.tensor([ids])
        mask = sliding_window_mask(t, model.sliding_window, tokens.EOS_ID)[0, 0]
        with torch.no_grad():
            logits = model(t, mask=mask, attn_impl="sdpa")
        scores = entropy(logits)
        starts = find_entropy_patch_start_ids(scores, threshold=DEFAULT_THRESHOLD, include_next_token=False)
        lengths = patch_lengths_from_start_ids(starts, t.shape[1])
        arrays[f"{name}/tokens"] = np.array(ids, dtype=np.int16)
        arrays[f"{name}/entropies"] = scores[0].numpy()
        arrays[f"{name}/patch_lengths"] = lengths[0].numpy().astype(np.int32)
        if len(ids) <= LOGIT_MAX_LEN:
            arrays[f"{name}/logits"] = logits[0].numpy()
        print(f"{name:14s} bytes={len(ids):5d} patches={lengths.shape[1]:4d} avg={len(ids) / lengths.shape[1]:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")
