"""Direct official generate_nocache versus M1 on identical single requests.

This is the public uncached path, not a claim to reproduce a cached paper engine.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time

import torch
import torch.distributed as dist

from blt_baseline.comparison import paired_summary
from blt_baseline.entropy import LengthAwarePatcher
from blt_baseline.evidence import Config, sha256, write_json
from blt_baseline.native import NativeBLT
from blt_baseline.runner import capture_source, collect_manifest
from blt_baseline.workloads import load_workload, output_accounting, systems_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = Config.load(args.config.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "RUNNING", "config": asdict(cfg), "cases": [],
              "boundary": "Single-request complete function calls plus raw-byte accounting; loading/warmup excluded. Official single-rank NCCL bookkeeping retained.",
              "upstream_generator": "bytelatent.generate_blt.generate_nocache",
              "note": "Uncached official generator, adapted only for admitted T4 runtime/precision; not published cached or H200 performance."}
    def save():
        write_json(args.out / "result.json", result)
    save()
    try:
        manifest = collect_manifest(cfg)
        manifest["harness_archive_sha256"] = capture_source(args.out, cfg)
        write_json(args.out / "manifest.json", manifest)
        model = NativeBLT(cfg)
        from bytelatent.generate_blt import generate_nocache
        result["generator_sha256"] = sha256(Path(cfg.upstream) / "bytelatent/generate_blt.py")
        native = model.patcher.native
        candidate = LengthAwarePatcher(native, microbatch_size=4)
        workload = load_workload(Path(cfg.workload))
        with tempfile.TemporaryDirectory(prefix="blt-official-dist-") as rendezvous:
            dist.init_process_group("nccl", init_method=f"file://{rendezvous}/init", rank=0, world_size=1)
            try:
                for width in (128, 512, 2048):
                    prompt = systems_prompt(workload[0]["prompt"], width)
                    steps = 8
                    def official():
                        ids = generate_nocache([prompt], model=model.model, tokenizer=model.tokenizer,
                            patcher=native, max_prompt_len=len(model.encode(prompt)), max_gen_len=steps,
                            use_sampling=False, remove_prompts=True)
                        return {"ids": ids, "outputs": [output_accounting(row) for row in ids]}
                    def ours():
                        model.patcher = candidate
                        return model.generate([prompt], steps)
                    calls = {"official_generate_nocache": official, "m1": ours}
                    expected = None
                    for call in calls.values():
                        for _ in range(2):
                            output = call()
                            if expected is None:
                                expected = output["ids"]
                            assert output["ids"] == expected
                    case = {"context_bytes": len(prompt.encode()), "batch": 1, "steps": steps,
                            "prompt": prompt, "samples": [], "orders": []}
                    result["cases"].append(case)
                    for repetition in range(5):
                        order = list(calls) if repetition % 2 == 0 else list(reversed(calls))
                        sample = {}
                        for name in order:
                            before = candidate.fallback_rows
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter()
                            output = calls[name]()
                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - started
                            assert output["ids"] == expected
                            sample[name] = {"elapsed_seconds": elapsed, "generated_ids": output["ids"],
                                "outputs": output["outputs"],
                                "raw_byte_count": sum(row["raw_byte_count"] for row in output["outputs"]),
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "fallback_rows": candidate.fallback_rows - before if name == "m1" else 0}
                        case["orders"].append(order)
                        case["samples"].append(sample)
                        save()
                        print(json.dumps({"context_bytes": width, "repetition": repetition,
                            "seconds": {k: v["elapsed_seconds"] for k, v in sample.items()}}), flush=True)
                    case["summary"] = paired_summary(case["samples"], reference="official_generate_nocache")
                    save()
            finally:
                dist.destroy_process_group()
        result["status"] = "COMPLETED"
    except Exception as exc:
        result.update(status="FAILED", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
