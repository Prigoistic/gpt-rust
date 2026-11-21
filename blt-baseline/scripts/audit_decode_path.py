"""Audit patch decisions on recorded generated prefixes, outside timed samples."""

import argparse
import json
from pathlib import Path
import sys

import torch

from blt_baseline.entropy import LengthAwarePatcher
from blt_baseline.evidence import Config, sha256, write_json
from blt_baseline.native import IsolatedRowPatcher, upstream_import_compatibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = Config.load(args.config.resolve())
    measured = json.loads((args.run / "result.json").read_text())
    if measured["status"] != "COMPLETED":
        raise ValueError("Audit requires a completed measured run")
    sys.path.insert(0, cfg.upstream)
    with upstream_import_compatibility(cfg.legacy_cutlass_alias):
        from bytelatent.data.patcher import PatcherArgs
        from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
        from bytelatent.transformer import LMTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LMTransformer.from_pretrained(cfg.entropy_snapshot, local_files_only=True, strict=True).float().cuda().eval()
    train = json.loads((Path(cfg.blt_snapshot) / "train_args.json").read_text())
    params = dict(train["data"]["patcher_args"])
    params.update(realtime_patching=False, device="cuda:0", patching_device="cuda:0", log_time=False)
    native = PatcherArgs(**params).build()
    native.entropy_model = model
    native.realtime_patching = True
    reference = IsolatedRowPatcher(native)
    policies = {"length_aware_rows": LengthAwarePatcher(native),
                "length_aware_batch": LengthAwarePatcher(native, microbatch_size=4)}
    tokenizer = BltTokenizer()
    result = {"status": "RUNNING", "measured_result_sha256": sha256(args.run / "result.json"),
              "audit_script_sha256": sha256(Path(__file__)),
              "entropy_source_sha256": sha256(Path(sys.modules[LengthAwarePatcher.__module__].__file__)),
              "entropy_weight_sha256": sha256(Path(cfg.entropy_snapshot) / "model.safetensors"),
              "entropy_atol": 1e-4, "entropy_rtol": 1e-4, "states": []}
    def save():
        write_json(args.out / "result.json", result)
    save()
    try:
        with torch.inference_mode():
            for case in measured["performance"]:
                generated = case["samples"][0]["padded"]["generated_ids"]
                for step in sorted({0, 1, 2, case["steps"] // 2, case["steps"] - 2, case["steps"] - 1}):
                    rows = [tokenizer.encode(prompt, add_bos=True, add_eos=False) + ids[:step]
                            for prompt, ids in zip(case["prompts"], generated)]
                    tokens = torch.tensor(rows, device="cuda")
                    expected, expected_scores = reference.patch(tokens, include_next_token=True)
                    for name, policy in policies.items():
                        actual, scores = policy.patch(tokens, include_next_token=True)
                        obs = {"case": case["id"], "step": step, "policy": name,
                               "patches_equal": torch.equal(actual, expected),
                               "entropy_max_abs_error_nats": float((scores - expected_scores).abs().max())}
                        result["states"].append(obs)
                        save()
                        assert obs["patches_equal"], obs
                        torch.testing.assert_close(scores, expected_scores, atol=1e-4, rtol=1e-4)
                print(json.dumps({"case": case["id"], "status": "PASS"}), flush=True)
        result["status"] = "PASS"
    except Exception as exc:
        result.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
