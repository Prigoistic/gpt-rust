"""Frozen, interleaved real-weight A/B/C experiment; no inference downloads."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from blt_baseline.comparison import paired_summary
from blt_baseline.entropy import LengthAwarePatcher
from blt_baseline.evidence import Config, doctor, sha256, write_json
from blt_baseline.native import NativeBLT
from blt_baseline.runner import capture_source, collect_manifest
from blt_baseline.workloads import load_workload, systems_prompt


def fixed_width(text, width):
    result = systems_prompt(text, width)
    return result + " " * (width - len(result.encode()))


def performance_cases(workload):
    cases = []
    for width in (128, 512, 2048):
        for batch in (1, 4):
            cases.append({"id": f"primary-c{width}-b{batch}", "primary": True,
                          "prompts": [fixed_width(workload[0]["prompt"], width)] * batch,
                          "steps": 8, "repeats": 7})
    by_id = {row["id"]: row for row in workload}
    for name in ("code-1", "unicode-2"):
        cases.append({"id": f"long-{name}-c512-b1", "primary": False,
                      "prompts": [fixed_width(by_id[name]["prompt"], 512)],
                      "steps": 64, "repeats": 5})
    mixed = [fixed_width(by_id[name]["prompt"], 384) for name in ("prose-1", "code-1", "unicode-1", "unicode-2")]
    cases.append({"id": "mixed-c384-b4", "primary": False, "prompts": mixed, "steps": 32, "repeats": 5})
    return cases


def render(result):
    lines = ["# M1: entropy execution ablation", "", f"Status: **{result['status']}**", "",
             "Pinned native BLT/T4; uncached greedy generation, original fp32 entropy model.",
             "A/B/C differ only in entropy execution. Timings include the numerical boundary safeguard.", "",
             f"Stage: {result.get('stage')}", "",
             "| Workload | Policy | Raw bytes/s | Median batch s | Paired median speedup | Bootstrap 95% | Fallback rows |",
             "|---|---|---:|---:|---:|---|---:|"]
    for case in result["performance"]:
        for name, summary in case.get("summary", {}).items():
            lo, hi = summary["paired_speedup_bootstrap_95"]
            lines.append(f"| {case['id']} | {name} | {summary['raw_bytes_per_second_per_gpu']:.2f} | "
                         f"{summary['batch_latency_seconds_median']:.4f} | {summary['paired_speedup_median']:.2f}x | "
                         f"{lo:.2f}–{hi:.2f} | {summary['fallback_rows']} |")
    lines += ["", "Intervals resample paired repetitions and describe runtime variation only. Synthetic/authored workloads, not a representative quality benchmark.",
              "No model-wide KV cache, BPE comparison, production serving SLO or new modeling contribution is established.", ""]
    if "error" in result:
        lines += [f"Failure: `{result['error']}`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--entropy-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = Config.load(args.config.resolve())
    if cfg.entropy_execution != "padded" or cfg.patcher_batch_policy != "isolated_rows":
        raise ValueError("Ablation reference must be the explicit M0 padded/isolated policy")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "RUNNING", "stage": "preflight", "config": asdict(cfg),
              "started_utc": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke,
              "correctness": [], "quality": [], "performance": [], "profiles": []}
    def save():
        write_json(args.out / "result.json", result)
        (args.out / "report.md").write_text(render(result))
    def stage(name):
        result["stage"] = name
        save()
        print(json.dumps({"stage": name, "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
    save()
    try:
        readiness = doctor(cfg)
        result["readiness"] = readiness
        if readiness["blockers"]:
            raise RuntimeError(readiness["blockers"])
        gate = json.loads(args.entropy_gate.read_text())
        if gate["status"] != "PASS":
            raise RuntimeError("Real-weight entropy gate has not passed")
        result["entropy_gate_sha256"] = sha256(args.entropy_gate)
        workload = load_workload(Path(cfg.workload))
        cases = performance_cases(workload)
        if args.smoke:
            cases = [dict(cases[0], repeats=3, steps=4)]
        write_json(args.out / "workloads.json", cases)
        manifest = collect_manifest(cfg)
        manifest["harness_archive_sha256"] = capture_source(args.out, cfg)
        manifest["experiment_workloads_sha256"] = sha256(args.out / "workloads.json")
        write_json(args.out / "manifest.json", manifest)
        stage("load")
        adapter = NativeBLT(cfg)
        native = adapter.patcher.native
        policies = {"padded": adapter.patcher,
                    "length_aware_rows": LengthAwarePatcher(native),
                    "length_aware_batch": LengthAwarePatcher(native, microbatch_size=cfg.entropy_microbatch_size)}
        result["boundary_margin_nats"] = LengthAwarePatcher.boundary_margin_nats
        stage("correctness")
        result["native_parity"] = adapter.check_parity(workload[0]["prompt"])
        # Cross-policy comparison: this gate does not merely compare a candidate
        # to an upstream generator using the very same candidate patcher.
        with torch.inference_mode():
            for case in cases:
                tokens = torch.tensor([adapter.encode(p) for p in case["prompts"]], device="cuda")
                adapter.patcher = policies["padded"]
                reference_logits, reference_lengths = adapter.next_logits(tokens, diagnose=True)
                for name, patcher in policies.items():
                    adapter.patcher = patcher
                    logits, lengths = adapter.next_logits(tokens, diagnose=True)
                    error = float((logits - reference_logits).abs().max())
                    assert torch.equal(lengths, reference_lengths), (case["id"], name, "patches")
                    torch.testing.assert_close(logits, reference_logits, atol=0, rtol=0)
                    result["correctness"].append({"case": case["id"], "policy": name,
                                                  "patches_equal": True, "logits_max_abs_error": error})
                save()
        stage("likelihood")
        for row in workload[:1] if args.smoke else workload:
            observed = {}
            for name, patcher in policies.items():
                adapter.patcher = patcher
                observed[name] = adapter.score(row["prompt"], row["continuation"])
            for name in policies:
                if abs(observed[name]["nll_nats"] - observed["padded"]["nll_nats"]) > 1e-5:
                    raise RuntimeError((row["id"], name, "likelihood changed", observed))
            result["quality"].append({"id": row["id"], "category": row["category"], "policies": observed})
            print(json.dumps({"quality": row["id"], "policies": observed}), flush=True)
            save()
        stage("paired_performance")
        rng = random.Random(cfg.seed)
        orders = list(itertools.permutations(policies))
        rng.shuffle(orders)
        for specification in cases:
            case = {**specification, "samples": [], "orders": []}
            case["gpu_before"] = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu",
                "--format=csv"], text=True)
            # Pre-warm every policy/shape, then compare identical greedy IDs.
            expected = None
            for name, patcher in policies.items():
                adapter.patcher = patcher
                for _ in range(2):
                    output = adapter.generate(case["prompts"], case["steps"])
                    if expected is None:
                        expected = output["ids"]
                    assert output["ids"] == expected, (case["id"], name, "warmup IDs")
            result["performance"].append(case)
            for repetition in range(case["repeats"]):
                sample = {}
                order = orders[repetition % len(orders)]
                for name in order:
                    patcher = policies[name]
                    adapter.patcher = patcher
                    before = getattr(patcher, "fallback_rows", 0)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    output = adapter.generate(case["prompts"], case["steps"])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    assert output["ids"] == expected, (case["id"], name, "timed IDs")
                    sample[name] = {"elapsed_seconds": elapsed, "generated_ids": output["ids"],
                                    "outputs": output["outputs"],
                                    "raw_byte_count": sum(o["raw_byte_count"] for o in output["outputs"]),
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                    "fallback_rows": getattr(patcher, "fallback_rows", 0) - before}
                case["orders"].append(order)
                case["samples"].append(sample)
                save()
                print(json.dumps({"case": case["id"], "repetition": repetition,
                                  "seconds": {name: obs["elapsed_seconds"] for name, obs in sample.items()}}), flush=True)
            case["summary"] = paired_summary(case["samples"])
            case["gpu_after"] = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu",
                "--format=csv"], text=True)
            save()
        stage("post_change_profile")
        adapter.patcher = policies["length_aware_batch"]
        for case in (cases[:1] if args.smoke else [cases[0], cases[5]]):
            with adapter.profile_regions(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                adapter.generate(case["prompts"], case["steps"])
                torch.cuda.synchronize()
            trace = f"trace-{case['id']}.json"
            prof.export_chrome_trace(str(args.out / trace))
            ranges = [{"name": event.key, "calls": event.count, "cpu_total_us": event.cpu_time_total,
                       "device_total_us": event.device_time_total}
                      for event in prof.key_averages() if event.key.startswith("blt::")
                      and str(event.device_type) == "DeviceType.CPU" and event.cpu_time_total > 0]
            result["profiles"].append({"case": case["id"], "trace": trace, "ranges": ranges})
            save()
        primary = [case for case in result["performance"] if case["primary"]]
        result["admission"] = {name: all(
            case["summary"][name]["paired_speedup_median"] >= 1.25 and
            case["summary"][name]["paired_speedup_bootstrap_95"][0] > 1.0
            for case in primary) for name in ("length_aware_rows", "length_aware_batch")}
        result["status"] = "SMOKE_COMPLETED" if args.smoke else "COMPLETED"
        stage("finished")
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
