# T4 runtime and execution notes

The historical experiments used an existing Azure Tesla T4 with 16 GiB VRAM.
Hostnames, addresses, users, private keys and deployment paths are deliberately
not part of the public package. Configure your own host and paths outside Git.

Use an isolated environment. The measured execution pair was PyTorch
2.10.0+cu128 / xFormers 0.0.35, with fp16 main parameters, original positional
buffers, fp32 entropy, TF32 disabled and the recorded native FlexAttention T4
options. The repository's dev dependencies use a different Torch version; do
not downgrade that environment to reproduce this run.

The import-only compatibility bridge exposes the identical existing ATen CUTLASS
operator for an obsolete upstream training registry name, then removes it. It
is admitted only for the verified package pair; it does not replace inference
kernels. T4 FlexAttention uses 32x32 tiles, one stage and four warps. Earlier
compiler/resource/numerical failures were excluded from admitted measurements.

Before loading full weights on another runtime:

1. Read the official source's setup instructions and package requirements.
2. Run `scripts/probe_flex.py --help`, then its independent math-reference probes
   using the appropriate T4 policy. Package imports alone do not prove kernels.
3. Run the real-weight entropy gate, then a short full-model parity/control run.
4. Expand to the recorded workload only after correctness succeeds.

Copy `configs/t4-padded.example.json` or `configs/t4-length-aware.example.json`
to an ignored `*.local.json` config for host-specific source/checkpoint paths.
Set the offline environment flags shown in the package README. Run attempts
have unique directories and preserve failures; do not merge partial retries.

Authentication is an explicit model-preparation step. No benchmark command
logs in to Hugging Face or downloads missing snapshots. Any helper that forwards
a credential to another host requires separate authorization for that exact
credential and destination; repository publication is not such authorization.

Azure temporary-disk weights/environments can disappear. Keep pinned revisions,
hashes, source and observations in persistent storage. Do not assume a historical
VM deployment is a durable artifact or leave GPU jobs running after checks.
