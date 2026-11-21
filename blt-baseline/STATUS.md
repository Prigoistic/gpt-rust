# Native BLT status

M0/M1 measurement milestones are complete. This public package was imported on
21 November 2026; it preserves the 7/17 November measurements and exact M1
numerical package source.

Read [the package overview](README.md), [M1 report](reports/M1_ENTROPY.md),
[official/published baseline comparison](reports/PUBLISHED_BASELINES.md) and
[evidence boundary](reports/README.md). The offline evidence verifier recomputes
the recorded results without weights, CUDA or access to the research machine.

The admitted change is guarded length-aware entropy execution. Whole-model
cache/state layout, a serving engine and superiority over competent cached BLT
remain unestablished. Incremental entropy and main-model state are the next
correctness/measurement gate. The north star remains competitive useful quality,
real throughput and cost per useful output for raw-byte modeling.


## Multi-byte pilot, 22 November 2026

Track B now has a measured uncached BLT-1B implementation of greedy local
drafting and full-model block verification. Width 4 completed three fresh
64-symbol requests 2.24–2.42× faster than the unmodified M1 generator on one
T4, with identical output IDs. The best width varied by prompt. Initial bytes
and readable UTF-8 text have separate timing measurements. See the
[B1 report](experiments/b1-fixed-width/REPORT.md) and
[reproduction protocol](experiments/b1-fixed-width/PROTOCOL.md).
The cached comparison remains open.
