"""Real-weight padding-removal/batch/chunk-seam gate before timing full BLT."""

import argparse
from dataclasses import asdict
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = Config.load(args.config.resolve())
    sys.path.insert(0, cfg.upstream)
    with upstream_import_compatibility(cfg.legacy_cutlass_alias):
        from bytelatent.data.patcher import PatcherArgs
        from bytelatent.transformer import LMTransformer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg.seed)
    model = LMTransformer.from_pretrained(cfg.entropy_snapshot, local_files_only=True, strict=True).float().cuda().eval()
    train = json.loads((Path(cfg.blt_snapshot) / "train_args.json").read_text())
    params = dict(train["data"]["patcher_args"])
    params.update(realtime_patching=False, device="cuda:0", patching_device="cuda:0", log_time=False)
    native = PatcherArgs(**params).build()
    native.entropy_model = model
    native.realtime_patching = True
    policies = {"padded": IsolatedRowPatcher(native),
                "length_aware_rows": LengthAwarePatcher(native),
                "length_aware_batch": LengthAwarePatcher(native, microbatch_size=4)}
    result = {"status": "RUNNING", "config": asdict(cfg), "torch": torch.__version__,
              "entropy_weight_sha256": sha256(Path(cfg.entropy_snapshot) / "model.safetensors"),
              "entropy_atol": 1e-4, "entropy_rtol": 1e-4, "cases": []}
    def save():
        write_json(args.out / "result.json", result)
    save()
    try:
        with torch.inference_mode():
            for length in (3, 31, 129, 513, 2049, 4097, 8191, 8192, 8193):
                for batch in (1, 4):
                    tokens = torch.randint(4, 260, (batch, length), device="cuda")
                    tokens[:, 0] = 1
                    if length > 31:
                        # Include native document-reset handling, distinct per row.
                        for row in range(batch):
                            tokens[row, 7 + row] = 2
                    expected, expected_scores = policies["padded"].patch(tokens, include_next_token=True)
                    for name in ("length_aware_rows", "length_aware_batch"):
                        patcher = policies[name]
                        actual, scores = patcher.patch(tokens, include_next_token=True)
                        error = float((scores - expected_scores).abs().max())
                        case = {"length": length, "batch": batch, "policy": name,
                                "entropy_max_abs_error_nats": error,
                                "patches_equal": torch.equal(actual, expected),
                                "min_threshold_distance_nats": float((expected_scores - native.threshold).abs().min())}
                        result["cases"].append(case)
                        save()
                        torch.testing.assert_close(scores, expected_scores, atol=1e-4, rtol=1e-4)
                        assert case["patches_equal"], case
                        without_next, _ = patcher.patch(tokens, entropies=scores, include_next_token=False)
                        reference_without, _ = policies["padded"].patch(tokens, entropies=expected_scores, include_next_token=False)
                        assert torch.equal(without_next, reference_without)
                        if batch > 1:
                            isolated = torch.cat([patcher.scores(tokens[i:i + 1]) for i in range(batch)])
                            torch.testing.assert_close(scores, isolated, atol=1e-4, rtol=1e-4)
                        print(json.dumps(case), flush=True)
            # Explicit causal suffix perturbation: earlier entropy must not change.
            tokens = torch.randint(4, 260, (4, 257), device="cuda")
            altered = tokens.clone()
            altered[:, 129:] = torch.randint(4, 260, altered[:, 129:].shape, device="cuda")
            for name in ("length_aware_rows", "length_aware_batch"):
                a, b = policies[name].scores(tokens), policies[name].scores(altered)
                torch.testing.assert_close(a[:, :129], b[:, :129], atol=0, rtol=0)
            result["causal_suffix_invariance"] = True
            # Construct an adversarial cutoff between two fp32 score results.
            # The guard must recover reference boundaries, rather than trusting
            # a passing allclose alone at a discontinuous routing decision.
            _, reference_scores = policies["padded"].patch(tokens, include_next_token=True)
            raw_scores = policies["length_aware_batch"].scores(tokens)
            differences = (raw_scores[:, 1:] - reference_scores[:, 1:]).abs()
            flat = int(differences.argmax())
            row, column = divmod(flat, differences.shape[1])
            column += 1
            cutoff = float((raw_scores[row, column].double() + reference_scores[row, column].double()) / 2)
            expected, _ = policies["padded"].patch(tokens, include_next_token=True, threshold=cutoff)
            unguarded, _ = native.patch(tokens, include_next_token=True, entropies=raw_scores, threshold=cutoff)
            guarded, _ = policies["length_aware_batch"].patch(tokens, include_next_token=True, threshold=cutoff)
            result["threshold_witness"] = {"cutoff": cutoff,
                "unguarded_patches_equal": torch.equal(unguarded, expected),
                "guarded_patches_equal": torch.equal(guarded, expected),
                "fallback_rows": policies["length_aware_batch"].fallback_rows}
            assert result["threshold_witness"]["guarded_patches_equal"]
            print(json.dumps(result["threshold_witness"]), flush=True)
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = repr(exc)
        raise
    finally:
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
