# BLT inference decisions — M1, 17 November 2026

The north star is competitive useful quality with better real throughput and cost
for raw-byte modeling. The near-term user priority is to settle an evidence-backed
inference design within the next few days. This record distinguishes implemented
execution from cache/serving designs that still need validation. It does not
revise the frozen project-wide charter or claim that model architecture is solved.

## Implemented execution contract

| Part | Decision | Verification boundary |
|---|---|---|
| Input | Native byte mapping, BOS, exact UTF-8 bytes; no silent truncation | Harness mapping/context checks |
| Model | Pinned released BLT-1B and entropy weights; no parameter changes | Content hashes and strict loads |
| Precision | fp16 main parameters, original positional buffers, fp32 entropy; TF32 disabled | Existing precision regression and real-weight comparisons |
| Reference | Retain padded, request-isolated native entropy and uncached greedy model | M0 immutable report; fresh paired reference in M1 |
| Candidate entropy | Length-aware prefixes/chunks, bounded batch-axis execution of equal-length requests | Real-weight score, patch, EOS, chunk and isolation gates |
| Boundary stability | Native boundary rules, include-next-token semantics; numerical decision-margin fallback | Deliberate threshold witness and full-model A/B/C |
| Kernels | Existing T4-admitted native attention configuration | Independent controls from M0; full-model comparison in M1 |
| Timing | Encoding through output byte accounting; separate loading/warmup/profile | Raw paired samples and recomputed accounting |
| Claim | Execution speedup at the measured policy/workload/hardware | No cached-engine/BPE/serving-quality claim |

The candidate's fixed decision margin is empirical numerical protection, not a
universal proof that all floating-point executions produce identical patches.
The original padded policy remains selectable for investigation and fallback.
Do not silently switch precision, thresholds, sampling, EOS policy or weights.

## M1 measured decision

The full paired experiment completed: 2.81–27.41x primary median speedups, about
7.6x on 64-step code/Hindi controls, and 12.4x on the mixed four-request control.
Retain `length_aware_batch` with a four-request cap as the recommended explicit
M1 configuration; keep padded execution selectable. Extra batching helps short
controls most and provides little extra benefit over per-row execution at 2048
bytes. There is no evidence here for an input-independent optimal batch policy.

Separate post-change profiles attribute 35.3% entropy / 64.1% main-model GPU
activity at 128 bytes, batch 1, and 73.0% entropy / 27.0% main-model activity at
2048 bytes, batch 4. These are associated kernel durations, not critical-path
wall-time shares. They support testing incremental **entropy state** as well as
main-model state; optimizing only the main transformer misses the larger measured
long-control cost. No occupancy or bandwidth bottleneck is claimed.

All timed controls had zero reference-fallback rows. Threshold-sensitive real
inputs can pay both candidate and reference work, so the reported speedups are
not guarantees for arbitrary request distributions. Keep fallback counts visible
in subsequent evaluation.

## Next implementation gate: incremental inference state

The pinned `generate_blt.py` implements `generate_nocache`. The local encoder and
decoder accept a `cache` argument, then return it without updating or consuming it.
The base attention implementation has a pluggable `kv_cache`, but the BLT forward
path must manage more than ordinary autoregressive token KV. A cache flag alone
is insufficient evidence of functioning cached BLT generation.

Design the request state explicitly:

1. **Input/history:** byte/BOS positions and the history required by the existing
   rolling hash groups. Preserve absolute positions and document boundaries.
2. **Entropy state:** per-layer history, local attention/document boundaries and
   native 8192-position chunk reset behavior. A 512-position attention window in
   each of 14 layers does not justify recomputing only the last 512 raw inputs:
   deeper-layer representations depend on earlier-layer histories.
3. **Local encoder state:** causal byte representations plus cross-attention and
   pooled summaries. Keep the currently extending patch separate from committed
   patches; prove which representations remain invariant when one byte is added.
4. **Global state:** committed patch representations/KV and an explicitly
   replaceable tail where inputs may change. Derive invalidation from native
   encoder pooling/cross-attention and decoder patch-ID shifting.
5. **Local decoder/cross-attention state:** byte positions and visible global
   context with the same causal masks as the full-prefix reference.
6. **Lifecycle:** request creation, completion, cancellation, reuse and any
   speculative rollback must release or invalidate the correct state. Cross-row
   or cross-request state leakage is a correctness failure.

Start with one request, prefill plus one-byte decode, against the M1 full-prefix
reference. Validate logits, likelihood and patches at every step, not only final
text. Include patch-open/patch-close transitions, EOS, 512/8192 boundaries,
request reuse, context limits and batch permutations. Record precision tolerances
before the run; investigate greedy flips instead of hiding them in tolerances.
Measure prefill and decode separately, then full-request latency/memory. Reject
cache variants whose maintenance/recomputation cost erases useful improvement.

## Gates after correct cached execution

- Re-profile. Use actual critical-path cost to choose kernel/launch/batching work.
  An operation-based serving scheduler and Rust control plane are candidates;
  changing implementation language does not establish GPU acceleration.
- Study supported multi-byte draft/verify/commit generation against the competent
  cached reference. Include draft, rejection, verification, temporary state and
  commit costs; preserve the target distribution or label the quality change.
- Compare strong BPE systems at controlled useful quality and hardware using
  bytes/s, fixed-SLO requests/s, latency, memory and useful-output cost.
- Admit learned routing only after a target-aware oracle shows material headroom.
  Learned chunking/model changes require separate compute/quality/transfer tests.

## What can be fixed now

Fix the evidence contracts, checkpoint/precision policy, reference semantics and
request-isolation requirements. The guarded entropy candidate passed its registered M1 matrix; keep its exact
measurements and validation in the M1 report.
Keep whole-model cache layout and scheduler policy provisional until their own
state-equivalence and end-to-end gates pass. The next-few-days horizon is a
priority and work sequence, not a promise to claim success before those checks.

## Published-baseline audit and direct source comparison

The user requested checking the open source and published BLT results before
locking the design. The current official HEAD is exactly our pin. A separate
single-request timing of its actual `generate_nocache` measured 10.57x / 7.49x /
2.78x improvements at 128/512/2048 bytes with identical outputs.

The original ACL paper's quality comparison is at 8B and its main efficiency
claim is about FLOPs. The final ICML Fast BLT paper also has H200 estimated
runtimes in Appendix G, extrapolated from isolated component timings and NFEs.
The arXiv v1 tables do not contain those later columns. No direct wall-clock
ranking against those published results is established here. See
`reports/PUBLISHED_BASELINES.md` for verified values and
protocol differences. The next competent cached baseline remains necessary.
