# BLT M1 — measured entropy-execution improvement

17 November 2026 · Azure Tesla T4 · pinned BLT-1B

**Status: completed paired experiment.** Main model, weights, patch policy and greedy decoding are unchanged. All model calls remain uncached.

## Result

| Input bytes | Batch | Padded bytes/s | Length-aware rows bytes/s | Length-aware batch bytes/s | Batch speedup | Paired bootstrap 95% |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 1 | 1.90 | 21.83 | 21.81 | 11.49x | 11.43–11.51x |
| 128 | 4 | 1.91 | 42.31 | 52.37 | 27.41x | 27.31–27.47x |
| 512 | 1 | 1.80 | 13.87 | 13.89 | 7.71x | 7.70–7.74x |
| 512 | 4 | 1.83 | 17.57 | 18.39 | 10.07x | 10.03–10.07x |
| 2048 | 1 | 1.64 | 4.60 | 4.61 | 2.81x | 2.80–2.81x |
| 2048 | 4 | 1.65 | 4.83 | 4.81 | 2.92x | 2.91–2.92x |

Seven paired repetitions per primary shape, eight output steps, two warmups per policy. All six padded controls reproduce the stored M0 greedy IDs. The reference is rerun in the same process; speedups are not calculated against old timings.

## Longer-output and mixed-request controls

| Workload | Steps | Batch | Padded bytes/s | Optimized bytes/s | Paired median speedup | Guard fallback rows |
|---|---:|---:|---:|---:|---:|---:|
| long-code-1-c512-b1 | 64 | 1 | 1.77 | 13.46 | 7.63x | 0 |
| long-unicode-2-c512-b1 | 64 | 1 | 1.76 | 13.40 | 7.60x | 0 |
| mixed-c384-b4 | 32 | 4 | 1.81 | 22.45 | 12.40x | 0 |

Five paired repetitions per longer control. Code/Hindi prompts are repeated authored fixtures. The mixed batch contains prose, code, Japanese and Hindi, padded to equal *input byte length* with explicit spaces in the recorded prompts; there is no hidden model-attention padding.

## Implementation and correctness

- 59 harness tests pass locally and on the Azure runtime. The integrated length-aware CLI configuration also completes with unchanged diagnostic BPB. The updated wheel builds/installs cleanly and matches the measurement-time package source.
- Removed unnecessary trailing entropy-model padding; retained native 8192-position chunk resets, document/EOS masks, fp32 entropy and all boundary-selection rules.
- Batched independent equal-length requests along the batch axis, bounded at four requests; no request flattening.
- A 0.002-nat decision-margin safeguard re-evaluates ambiguous requests with the padded reference. All timed controls recorded zero fallback rows, so these rates include the guard check but do not estimate real-world fallback frequency. Inputs triggering frequent fallback can lose the gain. The separate threshold witness exercises fallback correctness. The margin is not a universal floating-point error proof.
- 36 real-weight entropy controls pass at absolute/relative tolerance 1e-4, with exact patch boundaries: odd lengths, batch 1/4, embedded EOS, 512-window transitions and 8191/8192/8193 lengths. Causal suffix invariance passes.
- An adversarial cutoff changes boundaries without the guard; the guarded result matches the reference.
- A separate generated-prefix audit passes 108 policy/state comparisons across first, middle and final decode positions in all nine workloads, with exact patches and close entropy scores.
- All 27 cross-policy full-model comparisons have exact logits and patch lengths. Every timed/warmup output is checked for exact greedy equality.
- Conditional scoring is identical for all policies: **1.195740823 BPB over 144 authored continuation bytes**. This is a preservation check, not broad quality certification.

## Comparison with official BLT baselines

The additional direct experiment calls the actual public `generate_nocache` function with the same released 1B weights on T4. See [direct measurements and published-table audit](PUBLISHED_BASELINES.md). This establishes a comparison with the public uncached path, not with the best cached BLT execution or the published FLOP/estimated-runtime results.

## New bottleneck

- primary-c128-b1: entropy 35.3% and main BLT 64.1% of associated GPU activity duration in a separate profile.
- primary-c2048-b4: entropy 73.0% and main BLT 27.0% of associated GPU activity duration in a separate profile.

These fractions sum associated kernel durations under disjoint parent ranges, not wall-clock critical-path fractions. Profiling overhead is excluded from performance samples. No occupancy or HBM-bandwidth claim follows from these traces.

## Measurement boundary and limits

The ablation driver uses the pinned-model settings from the inherited config and its own recorded per-case schedule. `workloads.json` specifies the actual 7/5 repetitions and 8/64/32 output steps; those per-case settings override the inherited generic timing fields.

171 timed policy batches, 6,360 generated raw bytes across all three policies. Timings include UTF-8 encoding, H2D/allocation, entropy prediction, patch bookkeeping, full BLT, argmax/commit, D2H and byte accounting. Loading, warmup and profiling are excluded. Policy order uses all six permutations over each first six repetitions; the seventh repeats one order. Bootstrap intervals describe paired run noise, not quality/generalization uncertainty.

This is a substantial engineering improvement over the explicit padded, uncached reference. It does not establish a production-ready engine, model-wide KV-cache performance, representative downstream quality, BPE competitiveness, request-serving p95, or useful-output dollar cost. The architecture and model parameters were not redesigned.

## Decision and next gate

Registered >=1.25x per-shape median/positive lower-bound admission: `{'length_aware_rows': True, 'length_aware_batch': True}`.

Keep the guarded length-aware execution path as the M1 candidate and retain padded execution as a selectable reference. The next inference gate is correct incremental state/caching, guided by the post-change profile. Upstream local cache arguments are passed through unchanged; enabling a flag cannot establish working BLT caching. Validate stable/dirty patch state, positions, EOS, request reuse and rollback before performance claims. Multi-byte decoding and an operation-based serving scheduler follow demonstrated headroom.

## Public evidence export

[Raw paired observations](evidence/m1.json), [entropy gate](evidence/entropy-gate.json), [generated-prefix audit](evidence/decode-audit.json), [direct official comparison](evidence/official.json), and [provenance](evidence/provenance.json) are included. They preserve measured numerical values and generated IDs while removing machine-specific metadata. See [reproduction and evidence boundaries](README.md). Original source archives, full profiler traces and checkpoints are retained outside Git.

Run `python scripts/verify_published.py` from `blt-baseline/` to recompute the published numerical checks and verify the measured package source hashes.
