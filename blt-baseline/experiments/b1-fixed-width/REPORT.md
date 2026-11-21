# Fixed-width multi-byte generation on released BLT-1B

22 November 2026. Single Tesla T4, batch one, greedy output, uncached M1 configuration.

## Decision

Carry greedy block verification and an immediate first-byte path into the cached comparison. Keep width selectable. The released model can amortize full/global calls across several accepted bytes. Its local decoder still drafts sequentially. This experiment establishes neither a trained parallel output head nor a production width.

## Five-width sweep

Each cell is the median of three within-repetition latency ratios relative to the width-1 streaming control. Each request produces 40 byte IDs. Warmups are excluded.

|Prompt|2|4|8|16|32|
|---|---:|---:|---:|---:|---:|
|prose|1.47×|1.57×|1.45×|0.96×|0.61×|
|code|1.77×|2.05×|2.29×|2.49×|2.60×|
|hindi|1.57×|1.71×|1.55×|1.17×|0.83×|

At width 4 the model performed the following work per 40-symbol response. The full-call count includes initial prefill. A local-decoder evaluation occurs on each proposal and each full pass.

|Prompt|Accepted / proposed|Full calls (AR = 40)|Local decoder calls|AR → block bytes/s|AR → block first byte, ms|
|---|---:|---:|---:|---:|---:|
|prose|29 / 41|12|53|22.9 → 36.0|43.9 → 44.0|
|code|32 / 32|9|41|22.9 → 46.8|44.1 → 44.0|
|hindi|30 / 38|11|49|22.8 → 39.1|43.9 → 44.2|

Every sweep output starts with ASCII space. First byte therefore measures early host availability, including prompt encoding and prefill. It does not measure first visible word, network delivery, or a serving queue.

## Fresh prompts and direct M1 control

Two widths chosen before the second sweep finished were compared with the unmodified M1 generator. Requests generated 64 symbols in each case. Times below are complete-call medians in seconds; speedups are median paired ratios.

|Prompt|Input bytes|M1 AR|Width 4|Width 8|Speedup at 4 / 8|
|---|---:|---:|---:|---:|---:|
|prose|507|4.589|1.885|1.826|2.42× / 2.51×|
|code|213|3.209|1.431|1.204|2.24× / 2.67×|
|hindi|413|4.619|2.003|2.028|2.29× / 2.28×|

## First readable output

The default block policy releases its first verified byte immediately. That byte was a space on each prompt. Waiting for the next verified block delayed the first Latin character on prose and code. On the Hindi prompt, width 4 could verify enough bytes for the first Devanagari character sooner than sequential decoding. The visible-first variant runs ordinary target steps through that character, then uses width 4. Each latency below is a warm median over three observations. The speedup uses paired complete-request timings in the visible-first process.

|Prompt|Ordinary AR, first visible (ms)|Immediate blocks, first visible (ms)|Visible-first, first visible (ms)|Visible-first end-to-end speedup|
|---|---:|---:|---:|---:|
|prose|87.7|145.4|87.5|1.66×|
|code|87.7|145.6|87.5|1.98×|
|hindi|174.5|145.6|175.4|1.66×|

The visible-first policy adds no parallelism to startup. It preserves ordinary AR startup on these controls. It can sacrifice an early win when a multibyte character would fit in a verified block. A three-byte Hindi character changes how many sequential target steps startup needs. Wider blocks can improve time to larger chunks while increasing time to first text. [Stream metrics](stream/summary.json) include 4- and 16-byte timestamps. Both streaming replays matched the original greedy IDs on all 60 executions including warmups.

## Evidence and limitations

All 72 fixed-width sweep requests (54 timed) and 36 direct-comparison requests (27 timed) matched the corresponding M1 greedy IDs. These are repeated runs on six distinct prompts. Equality on those prompts is a regression check. It does not establish a standard-task score, exactness for every near-tie under fp16, or sampling-distribution equivalence.

For this implementation, wide blocks trade lower full-call counts against rejected proposals, whose causal local encoder and decoder work has already been paid. The initial anchor-per-round control is separately archived. Its nominal width includes an anchor, so its width labels are not directly interchangeable with the additional proposal widths in the main sweep.

Next compare identical incremental/cached AR and block paths, paying for verification, rejected suffixes, cache rollback, and time to first non-whitespace character. The full-prefix baseline repeats more work per target call than a cached engine. Broader task quality, batch/load behavior, p95 latency, optimized BPE comparison and cost per useful response remain open.

Upstream BLT revision `9774ed4fcc78313f9f218295f3d7e4decdadf2ae`. BLT and entropy revisions `8134b32f0b1d25d1248c30e8c7bdfd442d3bb380 / f2aae511e44e2086b1204bc4ddec6ac6c9651332`. CUDA 12.8, PyTorch 2.10.0+cu128, xFormers 0.0.35, NVIDIA driver 595.91.07. See [protocol](PROTOCOL.md), [primary raw data](fixed.jsonl), [fresh-prompt data](holdout.jsonl), [anchored control](anchored.jsonl), and [recomputed metrics](summary.json).
