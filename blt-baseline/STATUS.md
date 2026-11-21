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
