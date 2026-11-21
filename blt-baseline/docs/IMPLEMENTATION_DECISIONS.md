# Baseline implementation decisions — 7 November 2026

These are additive execution and measurement decisions. The frozen research charter, architecture scope and earlier routing PDF are unchanged.

## Model and execution boundary

Use the official BLT architecture and released checkpoints, not the earlier partial entropy-model reimplementation. The reference policy is greedy, fixed-length, uncached full-prefix generation. Preserve raw UTF-8 input bytes, checkpoint patching settings, fp32 entropy prediction, and original positional-buffer precision. Convert only main-model parameters to fp16 on the T4. Load both checkpoints strictly and offline.

The canonical source remains an unmodified checkout at `9774ed4fcc78313f9f218295f3d7e4decdadf2ae`. The adapter records its runtime policies separately. Native single-request generation is the reference for the request-isolation correction.

## Admitted runtime and rejected configurations

| Observation | Decision |
|---|---|
| The existing VM's PyTorch 2.13 runtime and the selected xFormers wheel had incompatible extensions | Use a matched pair in an isolated environment; preserve the existing environment |
| Current xFormers removed an old operator name used by an import-time training registry | Temporarily expose the identical ATen overload for that registry, then remove the alias; no inference kernel replacement |
| PyTorch 2.6 specialized FlexAttention lowering failed on symbolic split-K computation | Preserve failure; test alternatives independently |
| The 2.6 generic kernel passed shorter shapes but failed the 129×127 numerical control, including with static compilation | Reject this older runtime; do not loosen numerical tolerances |
| PyTorch 2.10 / xFormers 0.0.35 passed the three-shape math-reference control | Admit this matched pair; tighten the compatibility guard to this pair |
| Default native kernel configuration exceeded T4 shared memory during a later full-model shape | Use explicit native generic FlexAttention with 32×32 tiles, one stage and four warps |
| That T4 policy passed seven math-reference checks, including patch-like masks, long sequences and batch 4 | Admit it for the diagnostic baseline, with its exact options recorded |

The numerical probes use fp16 and compare with the independent SDPA math backend at absolute/relative tolerances of 0.01. They are controlled checks, not proof for every possible shape. Full-model generation and isolation checks still gate each run.

## Correctness correction: entropy batching

The pinned native `calculate_entropies` flattens short batch rows before reshaping into 8,192-position entropy-model inputs. For independent requests without EOS separators, the second row can attend to the preceding row's bytes.

The pretrained witness used a prose prefix followed by an equal-length code prefix. The first row agreed with isolated execution. The second row differed by up to **2.2747759819 nats**, and its patch boundaries changed. This is a request-independence failure, not a demonstration of a downstream-quality improvement.

The conservative adapter runs the unmodified patcher separately per row and right-pads only the resulting patch-length metadata. Repeating the witness with this policy produced identical entropies and patch lengths for both rows, with maximum observed entropy difference **0**. The original entropy padding cost is retained and measured. Future vectorization must preserve this invariant.

The generation gate now checks both requests' logits and patch boundaries and compares entropy scores using a varied code distractor. Repeated `x` alone can fail to expose the defect because a particular patch pattern may stay unchanged.

## Measurement and artifact integrity

- Preallocate the output buffer. Use inference mode. Keep profiler hooks and per-step synchronization outside timed throughput samples.
- Record generated IDs, raw byte counts, special symbols and UTF-8 validity. Fixed output steps can cut a UTF-8 sequence; this is not itself a language-quality diagnosis.
- Weight BPB by continuation bytes, not examples; normalize the full model vocabulary. Score each target from a causal prefix.
- Call the small generation control **fixed-length continuation match**, not whole-answer accuracy.
- Record all raw timings, CPU/GPU memory boundaries, package/source/checkpoint/workload hashes, and a harness-source archive. New attempts use new directories.
- Modern Kineto exports CPU-side ranges and GPU stream markers with the same names. The reviewed atlas uses CPU-side range entries for kernel attribution and excludes duplicate stream-marker rows. Nested inclusive ranges are never added together.
- The clean wheel test verifies CLI operation and source archiving without a source checkout or PyTorch installation. Missing compute dependencies produce a blocked preflight, not fabricated results.

## Limits of the first report

The T4 shape sweep uses a repeated prose prompt, three context sizes, two batch sizes, eight output steps and five timed repetitions. It is a diagnostic of this execution path. It does not establish representative language quality, robust p95 serving latency, long-output performance, BPE competitiveness, H-Net scaling, or useful-output cost. GPU price is intentionally unset.

Cartesia-informed compute-to-loss and loss-to-downstream-transfer studies remain separate experiments requiring appropriate datasets, model ladders and training-compute metadata. No routing, speculative generation, Rust scheduler or architecture redesign is admitted by this baseline alone.
