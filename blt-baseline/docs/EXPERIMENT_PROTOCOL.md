# Baseline protocol v0.1 — 7 November 2026

This protocol adds an executable measurement harness to the frozen research scope. Implementation status and empirical evidence are separate. A completed harness is not a completed GPU baseline.

## Gate 0: reproducibility and supported execution

Pin official source, immutable BLT and entropy snapshots, package versions, GPU/driver/runtime, main/entropy precision, patching policy, seeds, input/output limits and workload hashes. Preserve all failed attempts. Do not silently switch kernels, precision, quantization, chunking, cache policy or hardware between comparisons.

The first adapter uses official uncached generation. Its full-prefix recomputation is a reference behavior and an important limitation, not evidence that a good BLT serving engine must do the same. It cannot establish superiority over a mature cached BPE server.

Probe required kernels before loading large weights on a new GPU. If a checkpoint requires unsupported kernels, record the failure. A compatible fallback needs its own named configuration, verified masks/logits/patches, and separate performance claims.

## Gate 1: correctness

Required before timed measurements:

1. Byte encoding round-trips and special-symbol policy agree with the official mapping.
2. Greedy generation agrees with the unmodified upstream generator under identical prefix, output length, precision and patcher settings.
3. Batched and isolated requests agree on patch boundaries and generated IDs; logit discrepancies must fit recorded tolerances. Greedy ties are investigated rather than hidden by loosening tolerances.
4. Scoring uses only causal prefixes and normalizes all model symbols. UTF-8 bytes, code points and token IDs remain distinct.
5. Context limits fail explicitly. No truncation, padding treated as text, or hidden workload substitution.

The existing `gpt-rust` entropy-model tests remain component checks against their historical fixtures. Three absent long-case logit arrays do not invalidate the separately run entropy/patch checks, but they do limit coverage. Full-model numerical parity is not implied.

## Gate 2: first baseline matrix

Start with 128 prompt bytes, batch 1, 4 generated symbols. After successful execution, sweep 128/512/2048 prompt bytes × batch 1/4 × 32 generated symbols. Record actual byte lengths, all samples, warmups, memory and special/invalid-UTF-8 outputs. Five repetitions are a smoke measurement; expand repetitions and independent runs for credible tail latency and comparison uncertainty.

T4 implementation decision, made before the matrix run: the initial diagnostic sweep uses **8 output symbols** at all six shapes. The measured 4-step uncached smoke latency was about 2.04 seconds, so this reduces measurement cost while retaining all planned context/batch shapes. The generic CUDA example retains 32 output steps. Long-output steady-state behavior is not established by the short T4 sweep. Its repeated prose prompt is a shape-control workload, not a representative distribution of prose, code, language, or patch entropy.

Keep four passes separate:

| Pass | Purpose | Instrumentation |
|---|---|---|
| Correctness | Faithful generation and isolation | Reference comparisons and assertions |
| Quality | Conditional continuation likelihood and small controls | Prefix-only sequential scoring; no speed claim |
| Throughput | Whole-request execution cost | CUDA synchronization only around each batch; no trace hooks |
| Diagnostic | Operation attribution and temporal structure | Named profiler ranges, operator table, per-step synchronization in a separate run |

Do not add nested CPU/GPU profiler totals into an end-to-end denominator. Inspect the timeline for synchronization and overlap. Report launch counts and GPU/CPU operator activity from traces; use Nsight Systems/Compute in a separate pass for critical-path attribution, SM occupancy and HBM bandwidth. The harness does not invent those hardware counters from timings.

## Gate 3: meaningful quality and transfer

The Cartesia discussion changes the evaluation standard: compression ratio, validation BPB, downstream transfer and runtime efficiency are separate axes. The included fixtures validate mechanics only.

Before a model-quality claim:

- Choose versioned, held-out text/code/multilingual continuation data and task suites with provenance, split definitions, licenses and stable prompt templates. Record byte-level prompt/continuation boundaries.
- Add whole-answer task scoring with explicit stop/EOS rules, answer extraction and denominators. Fixed-target-length continuation matching cannot substitute for whole-answer accuracy.
- Use copy/lookup controls beside arithmetic/execution tasks; use controlled random continuations to distinguish irreducible uncertainty from recoverable computation. The current random-looking string is only a fixture, not a statistical irreducibility test.
- For tokenized comparisons, audit prompt-boundary tokenization and sequence-probability conventions. A candidate boundary correction is an explicit evaluation choice, never a silent score adjustment.
- Report BPB and downstream results separately, by category/language as well as aggregate. Retain confidence intervals at document/example level where the dataset supports them.

A single inference checkpoint cannot reproduce compute-to-loss or loss-to-loss scaling laws. Those need a controlled ladder with training-compute/data metadata and multiple checkpoints. H-Net is a comparator, not an automatic pivot; Cartesia's BLT-like controlled architecture is not actual Meta BLT.

## Gate 4: admit one intervention

| Candidate | Required observation | First discriminating experiment | Stop condition |
|---|---|---|---|
| Execution/fusion/batching | Material critical-path overhead in compatible operations | Change one substantial path at fixed model, patching, decoding and quality | No meaningful end-to-end gain after overhead |
| Multi-byte draft–verify | Sequential byte events dominate after reasonable execution | Supported draft/verify scheme; measure accepted bytes, rejections, verification and cache work | Acceptance or verification cost erases useful speedup |
| Gain routing | Supported action semantics and material target-aware oracle headroom | Offline paired branches from identical causal state; signed gains and cheap causal features | Oracle headroom negligible, leakage, or quality/cost regression |
| Learned chunking/representation | Controlled evidence of recoverable quality/transfer limitation | Matched data/compute and downstream evaluation against relevant comparators | Compression changes without useful-quality improvement |

For an attributed removable cost fraction f, 1/(1−f) is the idealized speedup ceiling. It is not valid to insert overlapping profiler percentages as f. Thresholds for material improvement should be chosen before running the intervention and justified by measurement variance and the north star.

## Report acceptance

Each report names status, code/weight/workload versions, correctness checks, scored denominators, execution policy, raw observations, uncertainty, failures and remaining gates. A blocked run contains no fabricated timing, quality score, or claimed bottleneck. A negative result is retained. Broader routing/runtime redesign remains gated on evidence.
