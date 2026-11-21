# Published evidence

This is a portable public export made on 21 November 2026 from the M0/M1 research
workspace. Measurement dates are 7 and 17 November; publication is not a new GPU
run or a new performance result.

## Included

`evidence/` preserves numerical observations, timing samples, authored prompts,
generated IDs, byte accounting, likelihoods, parity witnesses and profile ranges.
`provenance.json` records both original-file SHA256 identifiers and projected-file
SHA256 identifiers, measured numerical package/test source hashes, weight pins,
and selected runtime versions. The measured package source is copied verbatim.

The JSON files are explicitly **projections**, not byte-identical copies of the
original run files. Machine-specific paths, intermediate trace names and raw GPU
command snapshots were removed. Numerical values and generated IDs were not
changed. Never treat an export hash as the original-file hash.

Run `python scripts/verify_published.py` from `blt-baseline/`. It recomputes M0,
M1 and direct-official summaries, validates the 201 timed policy batches / 6,600
raw output bytes from the two M1 experiments, checks stored parity evidence and
verifies exact measured package-source hashes. Original M0 timing samples are
also checked separately. These offline checks do not rerun CUDA.

## Outside this public repository

- Model weights, Hugging Face credentials and SSH/private infrastructure details.
- Large profiler traces, full source/transport archives, wheel builds and logs.
- Publisher PDFs: the reports link directly to the publishers instead.
- Earlier failed attempts beyond the report's description; original evidence
  remains preserved in the research workspace, not silently removed there.

The original source/archive hashes are retained as provenance; this export does
not claim independently to verify unavailable archive contents. The runnable
harness and pinned input identifiers support a fresh GPU reproduction.

## Reading order

1. [M1 result and limits](M1_ENTROPY.md).
2. [Direct official-generation and published-baseline comparison](PUBLISHED_BASELINES.md).
3. [Historical M0 reference](M0_BASELINE.md).
4. [Inference decisions](../docs/INFERENCE_DESIGN.md).
