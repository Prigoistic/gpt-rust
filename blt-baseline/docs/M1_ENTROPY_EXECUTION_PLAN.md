# M1: entropy execution, registered 17 November 2026

The immediate scope is an inference design that can be fixed on evidence within
the user's next-few-days horizon. M0 remains an immutable diagnostic reference.

## Mechanism and isolated interventions

1. `padded`: M0 native entropy model with per-request native 8192-position padding.
2. `length_aware_rows`: same weights, fp32 precision, attention backend, native
   causal/document mask and 8192-position chunk resets; remove only trailing pad.
3. `length_aware_batch`: additionally execute equal-length independent requests
   along the batch axis, with bounded microbatches. Never flatten requests.

All policies delegate thresholding, next-byte boundary semantics and maximum
patch-length processing to the same unmodified native Patcher. Main BLT execution
and greedy generation remain unchanged and uncached. No distributional/model
quality improvement is claimed by preserving an execution policy.

Numerical guard: if computed scores (or the native relevant score differences)
are within 0.002 nats of a boundary decision, re-evaluate that request using the
padded reference. Report fallback counts and include its synchronization/cost.
The margin exceeds the admitted score tolerance but is not a formal error bound
for every possible input. Fixed-top-k policy, if requested, falls back entirely.

## Admission before speed claims

- Real-weight entropy at absolute/relative tolerance 1e-4; exact patch lengths.
- Test short/odd lengths, 512-window edges, 8192 native chunk seams, batch 1/4,
  embedded EOS, causal suffix invariance and request independence.
- Full-model paired logits, exact greedy continuations and continuation NLL.
- Freeze source/workload/configuration hashes in each immutable attempt.
- Stop timing on a failed gate. Keep the failure and investigate its mechanism;
  do not relax tolerances to admit the result.

## Timing and useful outcome

Run paired, balanced-order comparisons in one process on the same T4, with
per-policy warmup, no profiling in samples, and identical prompts/steps. Include
encoding, transfers, patching, full BLT and output accounting. Record raw samples,
IDs, bytes, memory and paired latency ratios. Use at least seven repetitions for
the primary matrix and document-level/category controls beside repeated prose.

Admission requires correctness plus at least 1.25x median end-to-end speedup in
every primary M0 shape and a paired bootstrap lower 95% bound above 1.0. Treat
this as a practical engineering threshold, not a publication significance test.
Also run longer-output controls and a separate post-change profile. Report
absolute rates and all regressions; a fast entropy microbenchmark alone fails.

## After M1

Use the new critical path to select cached model execution/state work. Audit
prefill/decode equivalence, patch-boundary invalidation, entropy-window/chunk
resets, and request lifecycle before admitting cache reuse. Keep whole-model
cache work distinct from entropy score reuse. Multi-byte generation follows a
competent execution baseline. The Rust serving control plane and optional
oracle-gated routing remain dependent on measured headroom. Broader BPE quality,
transfer and serving-SLO claims require separate representative evaluation.
