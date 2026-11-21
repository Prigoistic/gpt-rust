# BLT baseline comparison — checked 17 November 2026

M1 is a measured improvement to the released model's padded, uncached execution
path. It does **not** establish superiority over a competent cached BLT engine,
the published 8B quality results, or Fast BLT's generation methods.

## Public code and released model identity

`git ls-remote https://github.com/facebookresearch/blt.git HEAD` returned
`9774ed4fcc78313f9f218295f3d7e4decdadf2ae`, exactly our source pin. The public
README links released 1B/7B weights and says its instructions were tested on H100.
Our measurements use the actual released 1B snapshot on T4 with recorded runtime
compatibility and kernel settings; 7B was not run.
[Official repository](https://github.com/facebookresearch/blt).

The public `bytelatent/generate_blt.py:generate_nocache` recomputes patching and
calls the full model at every byte. The model's forward path recomputes local
encoding, all patch representations, the global transformer and local decoding.
Local-model `cache` arguments are passed through, not consumed. This is the
public uncached implementation we optimize, not proof of the most efficient
inference algorithm described by the architecture.

The released checkpoint has a roughly 1.285B global transformer, 19.4M local
encoder and 157.9M local decoder, plus large hash tables. These module sizes align
with the rounded 1B model description in Fast BLT. Architecture size does not
establish identical trained weights, training data or evaluation policy. The
name “1B” is not a count of all stored hash-embedding elements.

## Direct official-function timing on our GPU

The additional `official-generation-001` experiment invokes the unmodified
`generate_nocache` function with the same weights, fp16 main parameters, original
positional buffers, fp32 entropy, T4 attention settings, prompts and eight output
steps. Batch 1 avoids the public short-batch cross-request entropy defect.
Five paired repetitions alternate order after two warmups per path. The official
function retains its single-rank NCCL bookkeeping; raw-byte accounting is inside
the timing boundary for both paths. Loading/warmup are excluded. Generated IDs
must match exactly on every call.

<!-- DIRECT_OFFICIAL_TABLE_START -->
| Input bytes | Official bytes/s | M1 bytes/s | Paired median speedup | Bootstrap 95% |
|---:|---:|---:|---:|---|
| 128 | 2.08 | 21.92 | 10.57x | 10.49–10.61x |
| 512 | 2.00 | 14.95 | 7.49x | 7.46–7.50x |
| 2048 | 1.81 | 5.03 | 2.78x | 2.78–2.78x |
<!-- DIRECT_OFFICIAL_TABLE_END -->

This is a direct comparison with the public **uncached function**, not with a
cached paper implementation or a different BLT checkpoint.

## Published quality context: original BLT

The final ACL paper's Table 1 compares FLOP-matched **8B** models. Its average
across seven tasks is:

| Model | Published average |
|---|---:|
| Llama 3 BPE | 60.0 |
| BLT-Space | 58.0 |
| BLT-Global | 59.7 |
| BLT-Mono | 61.1 |

Its “up to 50%” efficiency statement concerns inference **FLOPs**, not a measured
2x T4 wall-clock result. These scores cannot be assigned to our 1B checkpoint.
[ACL 2025 paper, Table 1 and §5](https://aclanthology.org/2025.acl-long.453.pdf).

## Published 1B quality context: Fast BLT

The final ICML paper reports the following **BLT 1B autoregressive baseline**:

| Benchmark | Published score | M1 matched benchmark reproduction |
|---|---:|---|
| ARC-Easy | 63.21 | NOT_RUN |
| ARC-Challenge | 34.94 | NOT_RUN |
| PIQA | 75.46 | NOT_RUN |
| HellaSwag | 60.17 | NOT_RUN |
| MMLU | 33.90 | NOT_RUN |
| HumanEval pass@1 | 12.80 | NOT_RUN |
| MBPP pass@1 | 14.00 | NOT_RUN |

These are literature targets, not locally reproduced scores. Our 144 authored
continuation bytes validate likelihood preservation, not these benchmarks.
[ICML 2026 paper, Tables 2, 5 and 6](https://proceedings.mlr.press/v306/kallini26a.html).

## Fast BLT efficiency: distinguish estimates from measured requests

The final ICML Appendix G adds **H200 estimated runtime**: each component is timed
with five warmups and ten measured passes, then its mean latency is multiplied
by its NFE count. It is not a direct complete-request timing. Table 10's 3B MBPP
examples are:

| 3B method | pass@1 | Estimated H200 seconds |
|---|---:|---:|
| BLT AR | 29.60 | 13.78 |
| BLT-S, k=16 | 29.60 | 5.52 |
| BLT-D-8, confidence 0.7 | 20.80 | 2.31 |
| BLT-DV-8, one-step draft | 27.00 | 3.73 |

The quality tradeoff is visible. We cannot divide these estimates into our T4
request times: model size, workloads, generation lengths, implementation and
measurement boundary differ. Use the final publisher PDF, not only arXiv v1,
which omits these later runtime columns.
[ICML publisher PDF, Appendix G/Table 10](https://raw.githubusercontent.com/mlresearch/v306/main/assets/kallini26a/kallini26a.pdf).

## Comparability gaps to close before locking the full inference design

1. **Cached execution:** establish correct incremental entropy, local and global
   state on the same released checkpoint. Count new/recomputed work separately
   from reused state; a full-prefix module call is not equivalent to an
   incremental patch update in a paper NFE table.
2. **Patching policy:** Fast BLT's trained models use average patch size 4 and
   maximum 8. Our checkpoint config uses threshold 1.335442066192627,
   `monotonicity=false`, no maximum patch length. Its `patch_size=4.5` field is
   not a measured average under threshold patching. Match policy or label the
   quality/compute intervention explicitly.
3. **Task protocol:** version datasets/prompts; match few-shot templates, byte
   boundaries, decoding and stop/EOS rules; report whole-answer accuracy and
   document/example-level uncertainty. The paper uses greedy generation with
   0-shot HumanEval, 3-shot MBPP and 4-shot FLORES translation.
4. **Hardware and units:** reproduce on the same GPU/precision/batch/context and
   report bytes/s, complete-request latency, memory and work counts. Keep measured
   times, extrapolated times, FLOPs, NFEs and estimated transferred GB separate.
5. **Evidence status:** source/weights were verified for original BLT. No official
   Fast BLT implementation/checkpoint was verified in the checked paper,
   publisher, author page and original repository; this bounded search is not a
   proof of worldwide unavailability. Community code needs its own parity audit.

The appropriate next priority remains correct incremental execution, followed by
standard-task reproduction and supported multi-byte verification. M1 gives a
useful implementation improvement; research/serving superiority remains unproven.

## Public source record

The cited publisher URLs identify the papers and exact tables checked on 17 November 2026. Publisher PDFs are not rehosted here. [Portable raw observations](evidence/official.json) and [provenance](evidence/provenance.json) record the direct official-function comparison and original artifact hashes. The full original archives and PDFs remain outside this repository.
