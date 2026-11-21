"""Explicit preparation step, separate from offline benchmark execution.

Use local Hugging Face authentication. Only revision-pinned model artifacts are
downloaded. Credentials stay on the machine executing this preparation step.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from blt_baseline.evidence import write_json

REVISIONS = {
    "facebook/blt-1b": "8134b32f0b1d25d1248c30e8c7bdfd442d3bb380",
    "facebook/blt-entropy": "f2aae511e44e2086b1204bc4ddec6ac6c9651332",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--token-stdin", action="store_true", help="Use only with explicit authorization to send a token to this host; never saves the token")
    args = parser.parse_args()
    token = json.load(sys.stdin).pop("token") if args.token_stdin else None
    paths = {}
    for repo, revision in REVISIONS.items():
        names = ["config.json", "model.safetensors"]
        if repo.endswith("1b"):
            names.insert(1, "train_args.json")
        for name in names:
            kwargs = {"token": token} if args.token_stdin else {}
            path = hf_hub_download(repo_id=repo, filename=name, revision=revision,
                                   cache_dir=args.cache, **kwargs)
            paths[repo] = str(Path(path).parent)
            print(f"Prepared {repo}@{revision}: {name}", flush=True)
    cfg = {
        "upstream": str(args.upstream.resolve()),
        "blt_snapshot": paths["facebook/blt-1b"],
        "entropy_snapshot": paths["facebook/blt-entropy"],
        "workload": str(args.workload.resolve()),
        "dtype": "fp16", "context_bytes": [128], "batch_sizes": [1],
        "output_steps": 4, "warmup": 2, "repeats": 5,
        "quality_max_bytes": 128, "profile": True,
        "legacy_cutlass_alias": True,
    }
    write_json(args.config_out, cfg)


if __name__ == "__main__":
    main()
