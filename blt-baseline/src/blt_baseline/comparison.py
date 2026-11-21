"""Paired runtime summaries; uncertainty concerns run noise, not task quality."""

import math
import random
import statistics

from .runner import percentile, timing_summary


def paired_summary(samples, reference="padded", *, seed=42):
    names = list(samples[0])
    if reference not in names or any(set(row) != set(names) for row in samples):
        raise ValueError("Each repetition must contain the same policies")
    if any(row[name]["generated_ids"] != row[reference]["generated_ids"]
           for row in samples for name in names):
        raise ValueError("Paired policies generated different IDs")
    result = {}
    for name in names:
        ratios = [row[reference]["elapsed_seconds"] / row[name]["elapsed_seconds"] for row in samples]
        logs = [math.log(ratio) for ratio in ratios]
        rng = random.Random(seed)
        boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(10000))
        result[name] = {**timing_summary([row[name] for row in samples], None),
                        "paired_speedup_median": statistics.median(ratios),
                        "paired_speedup_geomean": math.exp(statistics.mean(logs)),
                        "paired_speedup_bootstrap_95": [percentile(boot, .025), percentile(boot, .975)],
                        "fallback_rows": sum(row[name].get("fallback_rows", 0) for row in samples),
                        "peak_allocated_bytes": max(row[name]["peak_allocated_bytes"] for row in samples),
                        "all_repeated_ids_equal": all(row[name]["generated_ids"] == samples[0][name]["generated_ids"] for row in samples)}
    return result
