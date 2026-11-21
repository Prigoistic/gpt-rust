"""Test native entropy patching for cross-request dependence on actual weights."""

import argparse
import json
import sys
from pathlib import Path

import torch

from blt_baseline.evidence import Config, write_json
from blt_baseline.native import IsolatedRowPatcher, upstream_import_compatibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    cfg = Config.load(args.config.resolve())
    sys.path.insert(0, cfg.upstream)
    with upstream_import_compatibility(cfg.legacy_cutlass_alias):
        from bytelatent.data.patcher import PatcherArgs
        from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
        from bytelatent.transformer import LMTransformer
    tokenizer = BltTokenizer()
    texts = ["The rain stopped before dawn. ", "def f(x): return x + 17; # ok!!"]
    width = len(texts[0].encode())
    texts[1] = (texts[1] * 2)[:width]
    rows = [tokenizer.encode(t, add_bos=True, add_eos=False) for t in texts]
    tokens = torch.tensor(rows, dtype=torch.long, device="cuda")
    model = LMTransformer.from_pretrained(cfg.entropy_snapshot, local_files_only=True, strict=True).float().cuda().eval()
    train = json.loads((Path(cfg.blt_snapshot) / "train_args.json").read_text())
    patch_args = dict(train["data"]["patcher_args"])
    patch_args.update(realtime_patching=False, device="cuda", patching_device="cuda", log_time=False)
    patcher = PatcherArgs(**patch_args).build()
    patcher.entropy_model = model
    patcher.realtime_patching = True
    if args.isolated:
        patcher = IsolatedRowPatcher(patcher)
    with torch.inference_mode():
        batched_lengths, batched_scores = patcher.patch(tokens, include_next_token=True)
        isolated = [patcher.patch(tokens[i:i + 1], include_next_token=True) for i in range(len(rows))]
    result = {"status": "PASS", "torch": torch.__version__, "texts": texts,
              "entropy_snapshot": cfg.entropy_snapshot, "source_revision": cfg.upstream_revision,
              "patcher_batch_policy": "isolated_rows" if args.isolated else "native",
              "include_next_token": True, "atol": 1e-4, "rtol": 1e-4, "rows": []}
    for i, (lengths, scores) in enumerate(isolated):
        matches = torch.allclose(batched_scores[i], scores[0], atol=1e-4, rtol=1e-4)
        actual = batched_lengths[i][batched_lengths[i] > 0].tolist()
        expected = lengths[0][lengths[0] > 0].tolist()
        result["rows"].append({"row": i, "entropy_equal": matches,
                               "max_entropy_error_nats": float((batched_scores[i] - scores[0]).abs().max().item()),
                               "batched_patch_lengths": actual, "isolated_patch_lengths": expected,
                               "patches_equal": actual == expected})
        if not matches or actual != expected:
            result["status"] = "FAIL"
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
