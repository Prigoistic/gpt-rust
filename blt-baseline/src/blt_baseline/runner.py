"""Separate correctness, quality, uninstrumented timing, and trace passes."""

from __future__ import annotations

import importlib.metadata
import math
import os
import statistics
import subprocess
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

from .evidence import Config, doctor, file_manifest, sha256, write_json
from .workloads import load_workload, systems_prompt


def percentile(values: list[float], p: float) -> float:
    if not values or not 0 <= p <= 1:
        raise ValueError("Nonempty values and p in [0, 1] required")
    ordered = sorted(values)
    i = (len(ordered) - 1) * p
    lower = int(i)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (i - lower)


def aggregate_quality(rows: list[dict]) -> dict:
    total_nats = math.fsum(row["nll_nats"] for row in rows)
    total_bytes = sum(row["scored_bytes"] for row in rows)
    if not total_bytes:
        raise ValueError("Cannot report BPB without scored bytes")
    eligible = [row for row in rows if row["fixed_length_match"] is not None]
    return {"scored_bytes": total_bytes, "nll_nats": total_nats,
            "bpb": total_nats / (total_bytes * math.log(2)),
            "fixed_length_correct": sum(row["fixed_length_match"] for row in eligible),
            "fixed_length_total": len(eligible),
            "note": "Authored diagnostic continuations; not a representative downstream benchmark."}


def timing_summary(samples: list[dict], hourly_usd: float | None) -> dict:
    seconds = [s["elapsed_seconds"] for s in samples]
    if not seconds or any(not math.isfinite(s) or s <= 0 for s in seconds):
        raise ValueError("Invalid elapsed time")
    total_bytes = sum(s["raw_byte_count"] for s in samples)
    rate = total_bytes / math.fsum(seconds)
    return {"samples": len(samples), "raw_bytes_per_second_per_gpu": rate,
            "batch_latency_seconds_median": statistics.median(seconds),
            "batch_latency_seconds_p95": percentile(seconds, 0.95),
            "batch_latency_seconds_min": min(seconds),
            "usd_per_million_raw_bytes": hourly_usd * 1e6 / (3600 * rate) if hourly_usd is not None and rate > 0 else None,
            "note": "Single-GPU closed-loop equal-length uncached batches; p95 is batch latency, not a serving SLO. Raw bytes may be invalid UTF-8."}


def render_report(result: dict) -> str:
    readiness = result["readiness"]
    random_smoke = readiness.get("config", {}).get("checkpoint_kind") == "random_smoke"
    lines = ["# BLT baseline report", "", f"Status: **{result['status']}**", "",
             ("**Random-weight tiny-model integration smoke, not a pretrained BLT baseline or useful-quality/speed evidence.** " if random_smoke else "Native pretrained BLT. ") +
             "Greedy fixed-step, full-prefix recomputation. No KV cache, speculative decoding, or serving scheduler.", "",
             f"Official revision: `{readiness['upstream']['revision']}`.",
             f"CUDA available: `{readiness['cuda']['available']}`.", ""]
    if readiness.get("config", {}).get("legacy_cutlass_alias"):
        lines += ["Runtime compatibility: temporary legacy CUTLASS name for the unused upstream training registry; inference dispatch is unchanged.", ""]
    lines += [f"FlexAttention kernel policy: `{readiness.get('config', {}).get('flex_kernel_policy', 'upstream')}`.", ""]
    lines += [f"Entropy batch policy: `{readiness.get('config', {}).get('patcher_batch_policy', 'native')}`.", ""]
    lines += [f"Entropy execution: `{readiness.get('config', {}).get('entropy_execution', 'padded')}`.", ""]
    if readiness["blockers"]:
        lines += ["## Execution prerequisites", ""] + [f"- {item}" for item in readiness["blockers"]] + [""]
    if "error" in result:
        lines += ["## Failure", "", f"`{result['error'].splitlines()[0]}`", "",
                  "The complete diagnostic is retained in result.json and the execution log.", ""]
    parity = result.get("parity")
    lines += ["## Correctness", "", f"Native generation / batch parity: **{parity['status'] if parity else 'NOT_RUN'}**.", ""]
    quality = result.get("quality_summary")
    lines += ["## Quality and transfer", ""]
    if quality:
        lines += [f"Conditional BPB: {quality['bpb']:.6f} over {quality['scored_bytes']} continuation bytes.",
                  f"Diagnostic fixed-length continuation match: {quality['fixed_length_correct']}/{quality['fixed_length_total']} (not whole-answer accuracy).", ""]
    elif result.get("quality"):
        lines += [f"Continuation scoring: **INCOMPLETE**; {len(result['quality'])} examples were scored before the attempt stopped. Partial observations remain in result.json; no full score is reported.", ""]
    else:
        lines += ["Pretrained continuation scoring: **NOT_RUN**.", ""]
    lines += ["The included authored fixtures check evaluation behavior. Generalization, reasoning quality, compute-to-loss scaling, and validation-to-downstream transfer remain **NOT_ESTABLISHED**. Those require held-out datasets and controlled model/training ladders.", "",
              "## Performance", ""]
    cases = result.get("performance", [])
    if cases:
        lines += ["| Prompt bytes | Batch | Steps | Raw bytes/s/GPU | Median batch s | p95 batch s |", "|---:|---:|---:|---:|---:|---:|"]
        for case in cases:
            s = case["summary"]
            lines += [f"| {case['actual_context_bytes']} | {case['batch_size']} | {case['output_steps']} | {s['raw_bytes_per_second_per_gpu']:.2f} | {s['batch_latency_seconds_median']:.6f} | {s['batch_latency_seconds_p95']:.6f} |"]
        lines += ["", "Timings include UTF-8 encoding, input transfer/allocation, patching, model, argmax/commit, output transfer and byte accounting; model loading and warmup are excluded. Trace and synchronized step-diagnostic passes are separate. Five samples provide a preliminary p95, not a reliable tail-latency claim.", ""]
    else:
        lines += ["GPU performance: **NOT_RUN**. No speedup or cost advantage established.", ""]
    lines += ["## Bottleneck atlas", "",
              "Profiler traces, when present, separate entropy-model/patcher, hash embeddings, local encoder, global transformer, local decoder and sampling. Cross-attention and bookkeeping appear as operators within these ranges. Nested totals overlap and must not be summed. PyTorch traces do not establish SM occupancy or achieved HBM bandwidth; use a separate Nsight pass for those.", ""]
    profiled = [case for case in cases if case.get("profile_inclusive_ranges")]
    if profiled:
        lines += ["Totals below are for each separate profiling pass, with instrumentation overhead. GPU columns sum associated kernel durations, not wall-clock critical-path fractions.", "",
                  "| Prompt bytes | Batch | Range | Calls | CPU inclusive ms | GPU inclusive ms |",
                  "|---:|---:|---|---:|---:|---:|"]
        for case in profiled:
            for event in case["profile_inclusive_ranges"]:
                # Modern Kineto also exports GPU stream annotations with the
                # same name. Those have zero CPU duration and include queue
                # gaps; do not duplicate them as CPU-range kernel attribution.
                if event.get("device_type", "CPU") not in {"CPU", "DeviceType.CPU"} or event["cpu_total_us"] <= 0:
                    continue
                name = event["name"].removeprefix("blt::")
                lines += [f"| {case['actual_context_bytes']} | {case['batch_size']} | {name} | {event['calls']} | {event['cpu_total_us'] / 1000:.3f} | {event['device_total_us'] / 1000:.3f} |"]
        lines += [""]
    lines += ["Optimization admission: measured substantial cost → causal explanation → end-to-end headroom → smallest controlled intervention. Amdahl upper bound for removable fraction f is 1/(1−f); estimate f from attributable critical-path time, not overlapping profiler totals.", "",
              "## Reproduction and limits", "",
              "See `result.json`, `manifest.json`, per-case raw samples, and trace files. A blocked preflight has no model manifest or timings. All model loading is offline; missing weights fail rather than download. The frozen project charter and prior component PDF are unchanged.", ""]
    return "\n".join(lines)


def save_result(out: Path, result: dict) -> None:
    write_json(out / "result.json", result)
    (out / "report.md").write_text(render_report(result), encoding="utf-8")


def source_locations() -> tuple[Path, Path | None]:
    package = Path(__file__).resolve().parent
    candidate = package.parents[1]
    checkout = candidate if (candidate / "pyproject.toml").is_file() and (candidate / "src/blt_baseline").resolve() == package else None
    return package, checkout


def collect_manifest(cfg: Config) -> dict:
    package, root = source_locations()
    # A reused runtime can expose multiple dist-info versions. Record the first
    # effective version importlib resolves, not an impossible duplicate locklist.
    names = {d.metadata["Name"] for d in importlib.metadata.distributions()}
    packages = sorted(f"{name}=={importlib.metadata.version(name)}" for name in names)
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,power.limit", "--format=csv"],
            text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = None
    return {"schema_version": 1, "config": asdict(cfg), "packages": packages,
            "nvidia_smi": gpu,
            "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "BLT_ALLOW_MISSING_FLEX_ATTENTION", "PYTHONHASHSEED")},
            "harness": {"package": file_manifest(package, ("*.py",)),
                        "source_checkout": str(root) if root else None,
                        "tests": file_manifest(root / "tests", ("*.py",)) if root else [],
                        "scripts": file_manifest(root / "scripts", ("*.py",)) if root else [],
                        "pyproject_sha256": sha256(root / "pyproject.toml") if root else None},
            "upstream_source": file_manifest(Path(cfg.upstream) / "bytelatent", ("*.py",)),
            "blt_snapshot": file_manifest(Path(cfg.blt_snapshot), ("*.json", "*.safetensors")),
            "entropy_snapshot": file_manifest(Path(cfg.entropy_snapshot), ("*.json", "*.safetensors")),
            "workload_sha256": sha256(Path(cfg.workload)),
            "policy": {"kv_cache": False, "sampling": "greedy", "eos_stop": False,
                       "entropy_dtype": "fp32", "tf32": False, "implicit_truncation": False}}


def capture_source(out: Path, cfg: Config) -> str:
    package, root = source_locations()
    archive = out / "harness-source.tar.gz"
    paths = []
    if root:
        paths = [root / "pyproject.toml", root / "README.md"]
        for directory, pattern in (("tests", "*.py"), ("scripts", "*.py"), ("docs", "*.md")):
            paths.extend(sorted((root / directory).rglob(pattern)))
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(package.rglob("*.py")):
            stream.add(path, arcname="src/blt_baseline/" + str(path.relative_to(package)), recursive=False)
        for path in paths:
            stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
        stream.add(Path(cfg.workload), arcname="workload.jsonl", recursive=False)
    return sha256(archive)


def execute(cfg: Config, out: Path) -> bool:
    # Each directory represents one immutable attempt. Never mix data across retries.
    out.mkdir(parents=True, exist_ok=False)
    readiness = doctor(cfg)
    result = {"schema_version": 1, "status": "BLOCKED" if readiness["blockers"] else "RUNNING",
              "readiness": readiness, "performance": [], "quality": []}
    save_result(out, result)
    if readiness["blockers"]:
        return False
    try:
        import torch
        from .native import NativeBLT

        workload = load_workload(Path(cfg.workload))
        for row in workload:
            if len(row["continuation"].encode("utf-8")) > cfg.quality_max_bytes:
                raise ValueError(f"{row['id']}: continuation exceeds quality_max_bytes; no silent truncation")
        result["stage"] = "provenance"
        save_result(out, result)
        manifest = collect_manifest(cfg)
        manifest["harness_archive_sha256"] = capture_source(out, cfg)
        write_json(out / "manifest.json", manifest)
        result["stage"] = "model_load"
        save_result(out, result)
        adapter = NativeBLT(cfg)
        # Validate requested shapes before spending time on quality/performance sweeps.
        if max(cfg.context_bytes) + cfg.output_steps > adapter.max_prefix_tokens:
            raise ValueError("Matrix exceeds native max_seqlen; shorten context/output steps explicitly")
        result["stage"] = "correctness"
        save_result(out, result)
        result["parity"] = adapter.check_parity(workload[0]["prompt"])
        result["stage"] = "quality"
        save_result(out, result)
        for row in workload:
            score = adapter.score(row["prompt"], row["continuation"])
            exact = None
            if row["exact_match"]:
                expected = adapter.encode(row["continuation"], bos=False)
                exact = adapter.generate([row["prompt"]], len(expected))["ids"][0] == expected
            result["quality"].append({"id": row["id"], "category": row["category"], **score, "fixed_length_match": exact})
            save_result(out, result)
        result["quality_summary"] = aggregate_quality(result["quality"])
        result["quality_by_category"] = {
            name: aggregate_quality([r for r in result["quality"] if r["category"] == name])
            for name in sorted({r["category"] for r in result["quality"]})
        }
        result["stage"] = "performance"
        save_result(out, result)
        for context in cfg.context_bytes:
            prompt = systems_prompt(workload[0]["prompt"], context)
            for batch in cfg.batch_sizes:
                prompts = [prompt] * batch
                for _ in range(cfg.warmup):
                    adapter.generate(prompts, cfg.output_steps)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()
                samples = []
                for _ in range(cfg.repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    generation = adapter.generate(prompts, cfg.output_steps)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    samples.append({"elapsed_seconds": elapsed,
                                    "raw_byte_count": sum(o["raw_byte_count"] for o in generation["outputs"]),
                                    "outputs": generation["outputs"], "generated_ids": generation["ids"]})
                memory = {"resident_allocated_bytes": resident,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                torch.cuda.synchronize()
                diagnostic = adapter.generate(prompts, cfg.output_steps, diagnose=True, observe_steps=True)
                times = diagnostic["step_ready_seconds"]
                case = {"requested_context_bytes": context, "actual_context_bytes": len(prompt.encode("utf-8")),
                        "batch_size": batch, "output_steps": cfg.output_steps,
                        "warmup": cfg.warmup, "samples": samples,
                        "summary": timing_summary(samples, cfg.gpu_hourly_usd), "memory": memory,
                        "diagnostics": diagnostic["diagnostics"],
                        "synchronized_diagnostic": {
                            "first_step_gpu_ready_seconds": times[0],
                            "inter_step_seconds": [b - a for a, b in zip(times, times[1:])],
                            "note": "Separate synchronized pass; observer overhead included. Not network TTFT or production inter-output latency."}}
                if cfg.profile:
                    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    with adapter.profile_regions(), torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                        adapter.generate(prompts, cfg.output_steps)
                        torch.cuda.synchronize()
                    trace = f"trace-c{context}-b{batch}.json"
                    prof.export_chrome_trace(str(out / trace))
                    table = f"operators-c{context}-b{batch}.txt"
                    averages = prof.key_averages()
                    (out / table).write_text(averages.table(sort_by="self_cuda_time_total", row_limit=60))
                    case["profile_inclusive_ranges"] = [
                        {"name": event.key, "device_type": str(event.device_type), "calls": event.count,
                         "cpu_total_us": event.cpu_time_total, "cpu_self_us": event.self_cpu_time_total,
                         "device_total_us": event.device_time_total,
                         "device_self_us": event.self_device_time_total}
                        for event in averages if event.key.startswith("blt::")
                        and str(event.device_type) == "DeviceType.CPU"
                    ]
                    case.update(trace=trace, operator_table=table)
                result["performance"].append(case)
                save_result(out, result)
        result["status"] = "COMPLETED"
        result["stage"] = "finished"
        save_result(out, result)
        return True
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save_result(out, result)
        raise
