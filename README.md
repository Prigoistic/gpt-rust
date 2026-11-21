# gpt-rust

Experiments in tokenization and raw-byte language-model execution.

| Directory | Scope |
|---|---|
| `crates/tokeniser-core`, `crates/tokeniser-py`, `src/tokeniser.py` | Rust tokenizer and Python implementation/bindings |
| `blt/` | Byte tokens, RoPE/norm/masks, entropy-model and patcher components |
| [`blt-baseline/`](blt-baseline/README.md) | Pinned official full-BLT adapter, M0 baseline, guarded M1 entropy execution, tests and measured reports |

The component implementation is not a complete BLT inference engine. The native
baseline package uses the official full model through a separately pinned source
checkout and local authorized checkpoints.

## BLT measurements and reproduction

Start with [the native BLT package](blt-baseline/README.md) and its
[public evidence boundary](blt-baseline/reports/README.md). Numerical code and
portable timing/output observations are included. Weights, credentials,
machine-specific configs and large traces are excluded.

The recorded M1 gains concern padded uncached execution. They do not establish
superiority over a competent cached BLT engine or the papers' quality/scaling
results. The reports distinguish measurements, published results and open gates.

## Local verification

Using this repository's existing Torch-capable Python environment:

```sh
.venv/bin/python -m pytest -q tests
.venv/bin/python -m pytest -c blt-baseline/pyproject.toml blt-baseline/tests -q
python blt-baseline/scripts/verify_published.py
```

Historical component parity fixtures omit three long-case logit arrays; those
specific checks skip while separate entropy/patch checks still run. The native
GPU package has additional explicit runtime and real-weight gates. Its measured
T4 Torch/xFormers environment is separate from the repository's CPU/dev runtime.

To prepare official source for new CUDA runs, follow the package README. The
`vendor/blt` checkout and run outputs are ignored and never implicitly downloaded
by the benchmark itself.
