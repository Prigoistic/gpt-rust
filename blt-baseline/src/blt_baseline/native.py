"""Thin adapter around pinned, unmodified Meta BLT; no alternate attention kernels."""

from __future__ import annotations

import json
import math
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from .evidence import Config, FLEX_KERNEL_OPTIONS
from .workloads import output_accounting


@contextmanager
def upstream_import_compatibility(enabled: bool):
    """Bridge only an obsolete *training registry* name; never replace a kernel.

    Meta's distributed.py adds an xFormers CUTLASS overload to a no-recompute set
    at import time. xFormers 0.0.29 delegates CUTLASS to ATen and removes that
    old namespace name. Temporarily expose the identical actual overload for
    that registry, then restore the namespace. Inference dispatch is unchanged.
    This is explicit, narrowly versioned, and recorded in the run configuration.
    """
    created = False
    name = "efficient_attention_forward_cutlass"
    if enabled:
        import xformers
        from xformers.ops.fmha import cutlass
        pair = (torch.__version__.split("+")[0], xformers.__version__)
        if pair != ("2.10.0", "0.0.35"):
            raise RuntimeError("Legacy registry compatibility is admitted only for verified torch 2.10.0 / xformers 0.0.35")
        if not hasattr(torch.ops.xformers, name):
            if cutlass.FwOp.OPERATOR is not torch.ops.aten._efficient_attention_forward:
                raise RuntimeError("Unexpected CUTLASS operator; refusing compatibility alias")
            setattr(torch.ops.xformers, name, cutlass.FwOp.OPERATOR)
            created = True
    try:
        yield
    finally:
        if created:
            delattr(torch.ops.xformers, name)


def move_parameters_preserving_buffers(model, *, device, dtype):
    """Match upstream's parameter-only precision conversion, retaining fp32 RoPE.

    Module.to(dtype=...) would also downcast floating-point buffers. Positional
    tables intentionally retain their original precision in the reference demo.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=dtype)
    return model.to(device=device)


def generic_flex_call(compiled, policy="generic"):
    """Select native FlexAttention's generic kernel, preserving all mask/math args.

    torch 2.6's specialized decoding lowering computes split-K from unguarded
    symbolic shapes. FORCE_USE_FLEX_ATTENTION is its supported kernel selector;
    this avoids that compiler path without suppressing errors or using eager.
    """
    def call(*args, **kwargs):
        options = dict(kwargs.pop("kernel_options", None) or {})
        options.update(FLEX_KERNEL_OPTIONS[policy])
        return compiled(*args, **kwargs, kernel_options=options)
    return call


class IsolatedRowPatcher:
    """Apply the unmodified native patcher separately to independent requests.

    Native calculate_entropies flattens short batch rows into one long sequence.
    Without EOS separators that changes the second request's entropy and patches.
    This conservative correction preserves single-request semantics, including
    the native padding overhead; it makes no vectorization speedup claim.
    """
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def patch(self, tokens, include_next_token=False, preds=None, entropies=None, threshold=None):
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError("Patcher requires a nonempty batch of prefix rows")
        if tokens.shape[0] == 1:
            return self.native.patch(tokens, include_next_token=include_next_token,
                                     preds=preds, entropies=entropies, threshold=threshold)
        results = [self.native.patch(
            tokens[i:i + 1], include_next_token=include_next_token,
            preds=preds[i:i + 1] if preds is not None else None,
            entropies=entropies[i:i + 1] if entropies is not None else None, threshold=threshold,
        ) for i in range(tokens.shape[0])]
        lengths = torch.nn.utils.rnn.pad_sequence(
            [result[0][0] for result in results], batch_first=True, padding_value=0
        )
        scores = None if results[0][1] is None else torch.cat([r[1] for r in results], dim=0)
        return lengths, scores


class NativeBLT:
    def __init__(self, cfg: Config):
        sys.path.insert(0, cfg.upstream)
        with upstream_import_compatibility(cfg.legacy_cutlass_alias):
            from bytelatent.data.patcher import PatcherArgs
            from bytelatent.model.blt import ByteLatentTransformer
            from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
            from bytelatent.transformer import LMTransformer

        if cfg.flex_kernel_policy != "upstream":
            import bytelatent.base_transformer as base
            import bytelatent.model.latent_transformer as latent
            compiled = generic_flex_call(base.flex_attention_comp, cfg.flex_kernel_policy)
            base.flex_attention_comp = compiled
            latent.flex_attention_comp = compiled

        imported = Path(sys.modules["bytelatent.model.blt"].__file__).resolve()
        if not imported.is_relative_to(Path(cfg.upstream).resolve()):
            raise RuntimeError(f"Wrong bytelatent imported: {imported}")
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.tracing = False
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.dtype]
        torch.manual_seed(cfg.seed)
        # Record backend flags in manifest. Avoid silent TF32 changes between runs.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Configured bf16 is unsupported on this GPU")
        train = json.loads((Path(cfg.blt_snapshot) / "train_args.json").read_text())
        tok = train["data"]["tokenizer_args"]
        if tok["name"] != "blt":
            raise ValueError("Baseline requires the native BLT byte tokenizer")
        self.tokenizer = BltTokenizer(**(tok.get("init_kwargs") or {}))
        if self.tokenizer.bpe_delim:
            raise ValueError("BPE delimiters are not enabled in this raw-byte baseline")
        # Explicit local directories: these calls must never resolve mutable hub 'main'.
        self.model = ByteLatentTransformer.from_pretrained(
            cfg.blt_snapshot, local_files_only=True, strict=True
        ).eval()
        move_parameters_preserving_buffers(self.model, device=self.device, dtype=self.dtype)
        # Keep entropy model fp32: changing its precision can move patch boundaries.
        self.entropy_model = LMTransformer.from_pretrained(
            cfg.entropy_snapshot, local_files_only=True, strict=True
        ).to(device=self.device, dtype=torch.float32).eval()
        args = dict(train["data"]["patcher_args"])
        if args["patching_mode"] != "entropy":
            raise ValueError("This baseline is scoped to pretrained entropy-patched BLT")
        args.update(realtime_patching=False, device=str(self.device), patching_device=str(self.device), log_time=False)
        self.patcher = PatcherArgs(**args).build()
        # Upstream's consolidated-checkpoint loader differs from HF snapshots.
        # Attach the officially loaded entropy model without changing patch policy.
        self.patcher.entropy_model = self.entropy_model
        self.patcher.realtime_patching = True
        if cfg.entropy_execution != "padded":
            from .entropy import LengthAwarePatcher
            size = cfg.entropy_microbatch_size if cfg.entropy_execution == "length_aware_batch" else 1
            self.patcher = LengthAwarePatcher(self.patcher, microbatch_size=size)
        elif cfg.patcher_batch_policy == "isolated_rows":
            self.patcher = IsolatedRowPatcher(self.patcher)
        self.model_args = json.loads((Path(cfg.blt_snapshot) / "config.json").read_text())["args"]
        self.max_prefix_tokens = self.model_args["max_seqlen"]

    def region(self, name: str):
        return torch.profiler.record_function(f"blt::{name}") if self.tracing else nullcontext()

    def encode(self, text: str, *, bos: bool = True) -> list[int]:
        raw = text.encode("utf-8", errors="strict")
        tokens = self.tokenizer.encode(text, add_bos=bos, add_eos=False)
        expected = ([self.tokenizer.bos_id] if bos else []) + [b + 4 for b in raw]
        if tokens != expected:
            raise ValueError("Native tokenizer no longer satisfies the pinned byte mapping")
        return tokens

    @torch.inference_mode()
    def next_logits(self, tokens: torch.Tensor, *, diagnose: bool = False):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.max_prefix_tokens:
            raise ValueError("Prefix exceeds model context; implicit truncation is forbidden")
        with self.region("patcher"):
            lengths, _ = self.patcher.patch(tokens, include_next_token=True)
        with self.region("model"):
            logits = self.model(tokens, patch_lengths=lengths)[:, -1, :]
        return logits, lengths if diagnose else None

    @torch.inference_mode()
    def generate(self, prompts: list[str], steps: int, *, diagnose: bool = False, observe_steps: bool = False) -> dict:
        """Equal-length batch; fixed steps, greedy, full-prefix recomputation, no KV cache.

        Never treats special symbols as bytes. Does not silently stop at EOS: this
        matches upstream generate_nocache's fixed-step policy. Quality is separate.
        """
        if not prompts or steps < 1:
            raise ValueError("Nonempty batch and positive step count required")
        started = time.perf_counter()
        step_ready_seconds = []
        with self.region("frontend"):
            encoded = [self.encode(prompt) for prompt in prompts]
            lengths = {len(ids) for ids in encoded}
            if len(lengths) != 1:
                raise ValueError("Use equal-byte-length batches; padding is not a valid context byte")
            size = len(encoded[0])
            if size + steps - 1 > self.max_prefix_tokens:
                raise ValueError("Requested generation exceeds context; reduce context/output_steps")
            # Allocate once; avoid torch.cat per generated byte.
            buffer = torch.empty((len(prompts), size + steps), dtype=torch.long, device=self.device)
            buffer[:, :size] = torch.tensor(encoded, dtype=torch.long, device=self.device)
        patches = []
        prefix_positions = 0
        for i in range(steps):
            logits, patch_lengths = self.next_logits(buffer[:, :size + i], diagnose=diagnose)
            with self.region("sampling_commit"):
                buffer[:, size + i] = logits.argmax(dim=-1)
            if observe_steps:
                torch.cuda.synchronize(self.device)
                step_ready_seconds.append(time.perf_counter() - started)
            if diagnose:
                patches.append((patch_lengths > 0).sum())
                prefix_positions += len(prompts) * (size + i)
        with self.region("output_transfer"):
            ids = buffer[:, size:].cpu().tolist()
            outputs = [output_accounting(row) for row in ids]
        diagnostics = None
        if diagnose:
            diagnostics = {
                "model_forward_calls": steps,
                "patcher_calls": steps,
                "prefix_byte_or_bos_positions_reprocessed": prefix_positions,
                "patch_positions_including_next_slot": int(torch.stack(patches).sum().item()),
                "note": "Uncached work summed across steps; not unique committed patches or per-layer NFEs.",
            }
        return {"ids": ids, "outputs": outputs, "diagnostics": diagnostics,
                "step_ready_seconds": step_ready_seconds}

    @torch.inference_mode()
    def score(self, prompt: str, continuation: str) -> dict:
        """Exact prefix-only chain-rule scoring; slower than teacher-forced full-sequence evaluation.

        The current target/future bytes never enter patching or model input. Count
        continuation bytes only, excluding BOS, prompt, EOS and padding.
        """
        prefix = self.encode(prompt)
        targets = self.encode(continuation, bos=False)
        if not targets or len(prefix) + len(targets) - 1 > self.max_prefix_tokens:
            raise ValueError("Empty continuation or quality example exceeds context")
        buffer = torch.empty((1, len(prefix) + len(targets)), dtype=torch.long, device=self.device)
        buffer[0, :len(prefix)] = torch.tensor(prefix, device=self.device)
        target_tensor = torch.tensor(targets, device=self.device)
        losses = []
        for i in range(len(targets)):
            logits, _ = self.next_logits(buffer[:, :len(prefix) + i])
            # FP32 log-softmax, full 260-symbol normalizer, no byte-only renormalization.
            losses.append(-torch.log_softmax(logits.float(), dim=-1)[0, target_tensor[i]])
            buffer[0, len(prefix) + i] = target_tensor[i]
        nll = float(torch.stack(losses).double().sum().item())
        if not math.isfinite(nll):
            raise RuntimeError("Nonfinite likelihood; quality measurement is invalid")
        return {"nll_nats": nll, "scored_bytes": len(targets), "bpb": nll / (len(targets) * math.log(2))}

    @torch.inference_mode()
    def check_parity(self, prompt: str, steps: int = 4) -> dict:
        """Compare to unmodified upstream greedy generation and test batch isolation."""
        from bytelatent.generate_blt import generate_nocache
        import torch.distributed as dist
        import tempfile

        # Official generate_nocache calls all_reduce even for world_size=1.
        owned_group = not dist.is_initialized()
        if not owned_group and dist.get_world_size() != 1:
            raise ValueError("Baseline is single GPU; launch as one Python process, not multi-rank torchrun")
        if owned_group:
            rendezvous = tempfile.TemporaryDirectory(prefix="blt-baseline-dist-")
            try:
                dist.init_process_group("nccl", init_method=f"file://{rendezvous.name}/init", rank=0, world_size=1)
            except Exception:
                rendezvous.cleanup()
                raise
        try:
            ours = self.generate([prompt], steps)["ids"]
            reference = generate_nocache(
                [prompt], model=self.model, tokenizer=self.tokenizer, patcher=self.patcher,
                max_prompt_len=len(self.encode(prompt)), max_gen_len=steps,
                use_sampling=False, remove_prompts=True,
            )
            if ours != reference:
                raise RuntimeError("Greedy output differs from official generate_nocache")
            # Equal byte lengths with different content detect cross-request coupling.
            other = "x" * len(prompt.encode("utf-8"))
            batch = self.generate([prompt, other], steps)["ids"]
            other_alone = self.generate([other], steps)["ids"][0]
            if batch != [ours[0], other_alone]:
                raise RuntimeError("Batch isolation failed under greedy decoding")
            a = torch.tensor([self.encode(prompt)], device=self.device)
            b = torch.tensor([self.encode(other)], device=self.device)
            single, single_lengths = self.next_logits(a, diagnose=True)
            other_logits, other_lengths = self.next_logits(b, diagnose=True)
            together, batch_lengths = self.next_logits(torch.cat((a, b)), diagnose=True)
            singles = torch.cat((single, other_logits))
            error = float((singles.float() - together.float()).abs().max().item())
            atol = 1e-4 if self.dtype == torch.float32 else 0.05
            torch.testing.assert_close(singles, together, atol=atol, rtol=atol)
            for i, lengths in enumerate((single_lengths, other_lengths)):
                if lengths[0][lengths[0] > 0].tolist() != batch_lengths[i][batch_lengths[i] > 0].tolist():
                    raise RuntimeError("Patch boundaries depend on another request")
            # Check a varied distractor too: repeated 'x' can mask entropy leakage
            # because its realized patch boundaries may happen to stay unchanged.
            seed = "def f(x): return x + 17; "
            byte_count = len(prompt.encode())
            distractor = (seed * (1 + byte_count // len(seed)))[:byte_count]
            c = torch.tensor([self.encode(distractor)], device=self.device)
            _, scores_a = self.patcher.patch(a, include_next_token=True)
            _, scores_c = self.patcher.patch(c, include_next_token=True)
            _, batch_scores = self.patcher.patch(torch.cat((a, c)), include_next_token=True)
            isolated_scores = torch.cat((scores_a, scores_c))
            torch.testing.assert_close(batch_scores, isolated_scores, atol=1e-4, rtol=1e-4)
            entropy_error = float((batch_scores - isolated_scores).abs().max().item())
            return {"status": "PASS", "upstream_greedy_equal": True, "batch_greedy_equal": True,
                    "patches_batch_independent": True, "batch_logit_max_abs_error": error,
                    "entropy_batch_max_abs_error_nats": entropy_error,
                    "logit_atol": atol, "logit_rtol": atol, "steps": steps,
                    "note": "One prompt and an equal-length distractor; not exhaustive numerical certification."}
        finally:
            if owned_group:
                dist.destroy_process_group()
                rendezvous.cleanup()

    @contextmanager
    def profile_regions(self):
        """Diagnostic-only ranges. Hooks and wrappers are removed even on failure."""
        import bytelatent.model.blt as upstream_blt
        from contextlib import ExitStack

        handles = []
        stacks = {}
        original_hash = upstream_blt.compute_hash_embeddings
        def hashed(*args, **kwargs):
            with self.region("hash_embeddings"):
                return original_hash(*args, **kwargs)
        try:
            self.tracing = True
            upstream_blt.compute_hash_embeddings = hashed
            modules = {"local_encoder": self.model.local_encoder,
                       "global_transformer": self.model.global_transformer,
                       "local_decoder": self.model.local_decoder,
                       "entropy_model": self.entropy_model}
            for name, module in modules.items():
                stacks[name] = ExitStack()
                def before(_module, _args, key=name):
                    stacks[key].enter_context(self.region(key))
                def after(_module, _args, _output, key=name):
                    stacks[key].close()
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after, always_call=True))
            yield
        finally:
            for handle in handles:
                handle.remove()
            for stack in stacks.values():
                stack.close()
            upstream_blt.compute_hash_embeddings = original_hash
            self.tracing = False
