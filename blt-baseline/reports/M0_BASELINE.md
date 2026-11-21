# BLT diagnostic baseline — Azure T4, 7 November 2026

Status: **COMPLETED**

Implementation: official checkpoint at fp16, original positional buffers, fp32 entropy model, row-isolated native patching and explicitly bounded native FlexAttention tiles. No KV cache.

All six shapes completed. Entropy prediction accounts for 90.1–98.5% of associated GPU kernel duration across these repeated-prose controls. This is an active-kernel attribution, not a wall-clock fraction or a general BLT scaling result.


Native pretrained BLT. Greedy fixed-step, full-prefix recomputation. No KV cache, speculative decoding, or serving scheduler.

Official revision: `9774ed4fcc78313f9f218295f3d7e4decdadf2ae`.
CUDA available: `True`.

Runtime compatibility: temporary legacy CUTLASS name for the unused upstream training registry; inference dispatch is unchanged.

FlexAttention kernel policy: `t4`.

Entropy batch policy: `isolated_rows`.

## Correctness

Native generation / batch parity: **PASS**.

## Quality and transfer

Conditional BPB: 1.195741 over 144 continuation bytes.
Diagnostic fixed-length continuation match: 1/3 (not whole-answer accuracy).

The included authored fixtures check evaluation behavior. Generalization, reasoning quality, compute-to-loss scaling, and validation-to-downstream transfer remain **NOT_ESTABLISHED**. Those require held-out datasets and controlled model/training ladders.

## Performance

| Prompt bytes | Batch | Steps | Raw bytes/s/GPU | Median batch s | p95 batch s |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 8 | 1.95 | 4.103755 | 4.127016 |
| 128 | 4 | 8 | 1.96 | 16.359472 | 16.527126 |
| 512 | 1 | 8 | 1.86 | 4.311930 | 4.326476 |
| 512 | 4 | 8 | 1.86 | 17.244100 | 17.365950 |
| 2048 | 1 | 8 | 1.70 | 4.707863 | 4.736464 |
| 2048 | 4 | 8 | 1.71 | 18.776956 | 18.816497 |

Timings include UTF-8 encoding, input transfer/allocation, patching, model, argmax/commit, output transfer and byte accounting; model loading and warmup are excluded. Trace and synchronized step-diagnostic passes are separate. Five samples provide a preliminary p95, not a reliable tail-latency claim.

## Bottleneck atlas

Profiler traces, when present, separate entropy-model/patcher, hash embeddings, local encoder, global transformer, local decoder and sampling. Cross-attention and bookkeeping appear as operators within these ranges. Nested totals overlap and must not be summed. PyTorch traces do not establish SM occupancy or achieved HBM bandwidth; use a separate Nsight pass for those.

Totals below are for each separate profiling pass, with instrumentation overhead. GPU columns sum associated kernel durations, not wall-clock critical-path fractions.

| Prompt bytes | Batch | Range | Calls | CPU inclusive ms | GPU inclusive ms |
|---:|---:|---|---:|---:|---:|
| 128 | 1 | frontend | 1 | 0.683 | 0.003 |
| 128 | 1 | patcher | 8 | 3870.525 | 3849.012 |
| 128 | 1 | entropy_model | 8 | 117.302 | 3847.454 |
| 128 | 1 | model | 8 | 455.363 | 157.342 |
| 128 | 1 | hash_embeddings | 8 | 23.688 | 1.891 |
| 128 | 1 | local_encoder | 8 | 21.292 | 7.060 |
| 128 | 1 | global_transformer | 8 | 211.107 | 111.060 |
| 128 | 1 | local_decoder | 8 | 132.487 | 34.817 |
| 128 | 1 | sampling_commit | 8 | 0.769 | 0.059 |
| 128 | 1 | output_transfer | 1 | 0.094 | 0.002 |
| 128 | 4 | frontend | 1 | 1.546 | 0.004 |
| 128 | 4 | patcher | 8 | 16392.649 | 16309.874 |
| 128 | 4 | entropy_model | 32 | 479.629 | 16303.224 |
| 128 | 4 | model | 8 | 483.880 | 246.293 |
| 128 | 4 | hash_embeddings | 8 | 24.963 | 2.386 |
| 128 | 4 | local_encoder | 8 | 22.855 | 14.000 |
| 128 | 4 | global_transformer | 8 | 229.487 | 143.767 |
| 128 | 4 | local_decoder | 8 | 136.383 | 83.273 |
| 128 | 4 | sampling_commit | 8 | 0.751 | 0.070 |
| 128 | 4 | output_transfer | 1 | 0.155 | 0.004 |
| 512 | 1 | frontend | 1 | 0.634 | 0.003 |
| 512 | 1 | patcher | 8 | 4017.436 | 3996.443 |
| 512 | 1 | entropy_model | 8 | 118.079 | 3994.730 |
| 512 | 1 | model | 8 | 456.389 | 211.477 |
| 512 | 1 | hash_embeddings | 8 | 23.975 | 2.308 |
| 512 | 1 | local_encoder | 8 | 21.370 | 14.693 |
| 512 | 1 | global_transformer | 8 | 212.255 | 113.338 |
| 512 | 1 | local_decoder | 8 | 132.093 | 78.476 |
| 512 | 1 | sampling_commit | 8 | 0.832 | 0.059 |
| 512 | 1 | output_transfer | 1 | 0.095 | 0.002 |
| 512 | 4 | frontend | 1 | 1.355 | 0.010 |
| 512 | 4 | patcher | 8 | 16995.025 | 16767.423 |
| 512 | 4 | entropy_model | 32 | 615.399 | 16760.119 |
| 512 | 4 | model | 8 | 503.904 | 496.270 |
| 512 | 4 | hash_embeddings | 8 | 24.538 | 5.064 |
| 512 | 4 | local_encoder | 8 | 22.517 | 47.776 |
| 512 | 4 | global_transformer | 8 | 222.266 | 144.155 |
| 512 | 4 | local_decoder | 8 | 135.350 | 296.234 |
| 512 | 4 | sampling_commit | 8 | 0.775 | 0.084 |
| 512 | 4 | output_transfer | 1 | 21.419 | 0.004 |
| 2048 | 1 | frontend | 1 | 1.271 | 0.009 |
| 2048 | 1 | patcher | 8 | 4354.114 | 4192.412 |
| 2048 | 1 | entropy_model | 8 | 258.909 | 4190.307 |
| 2048 | 1 | model | 8 | 485.483 | 456.745 |
| 2048 | 1 | hash_embeddings | 8 | 23.857 | 4.947 |
| 2048 | 1 | local_encoder | 8 | 21.154 | 46.713 |
| 2048 | 1 | global_transformer | 8 | 211.464 | 114.338 |
| 2048 | 1 | local_decoder | 8 | 133.245 | 287.961 |
| 2048 | 1 | sampling_commit | 8 | 0.777 | 0.075 |
| 2048 | 1 | output_transfer | 1 | 19.533 | 0.002 |
| 2048 | 4 | frontend | 1 | 2.624 | 0.019 |
| 2048 | 4 | patcher | 8 | 18306.118 | 17110.641 |
| 2048 | 4 | entropy_model | 32 | 1597.065 | 17101.918 |
| 2048 | 4 | model | 8 | 629.251 | 1731.743 |
| 2048 | 4 | hash_embeddings | 8 | 29.447 | 13.599 |
| 2048 | 4 | local_encoder | 8 | 23.222 | 158.736 |
| 2048 | 4 | global_transformer | 8 | 226.589 | 146.701 |
| 2048 | 4 | local_decoder | 8 | 138.884 | 1409.229 |
| 2048 | 4 | sampling_commit | 8 | 0.789 | 0.112 |
| 2048 | 4 | output_transfer | 1 | 158.298 | 0.005 |

Optimization admission: measured substantial cost → causal explanation → end-to-end headroom → smallest controlled intervention. Amdahl upper bound for removable fraction f is 1/(1−f); estimate f from attributable critical-path time, not overlapping profiler totals.

## Memory and saved-output checks

| Prompt bytes | Batch | Resident allocated GiB | Peak allocated GiB | Peak reserved GiB |
|---:|---:|---:|---:|---:|
| 128 | 1 | 8.859 | 9.226 | 9.406 |
| 128 | 4 | 8.859 | 9.227 | 9.408 |
| 512 | 1 | 8.859 | 9.227 | 9.408 |
| 512 | 4 | 8.859 | 9.228 | 9.408 |
| 2048 | 1 | 8.859 | 9.228 | 9.408 |
| 2048 | 4 | 8.859 | 9.234 | 9.473 |

Byte accounting and throughput were recomputed from 30 timed batches, totaling 600 raw output bytes. Stored greedy IDs agree across all repeats and matched-context batch sizes: **True**. This covers these stored controls, not general downstream quality. Driver allocations outside PyTorch are excluded from the memory table.

## First justified engineering experiment

Test length-aware execution of the same entropy model against this padded reference. The native routine pads short rows to 8,192 positions at each decode step. Preserve entropy estimates, patch boundaries, causal isolation and scored likelihood; then compare end-to-end time with all overhead included. This is baseline engineering, not a demonstrated new modeling contribution. Representative prose/code/multilingual workloads and a competent cached BLT reference are needed before claiming gains over BPE or generalizing this bottleneck.

## Public evidence export

This report describes the historical measurement. See [raw numerical observations](evidence/m0.json), [export provenance](evidence/provenance.json), and [evidence/reproduction notes](README.md). Public evidence omits machine-specific metadata; original hashes are retained. Large traces, original source archives and checkpoints are not committed.
