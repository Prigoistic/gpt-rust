"""Verify native generic FlexAttention against dense masked SDPA on actual CUDA."""

import argparse
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from blt_baseline.evidence import FLEX_KERNEL_OPTIONS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--upstream-kernel", action="store_true")
    parser.add_argument("--policy", choices=tuple(FLEX_KERNEL_OPTIONS))
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.upstream_kernel and args.policy:
        raise ValueError("Select --policy or --upstream-kernel")
    policy = "upstream" if args.upstream_kernel else (args.policy or "generic")
    options = dict(FLEX_KERNEL_OPTIONS[policy])
    compiled = torch.compile(flex_attention, dynamic=False if args.static else None)
    result = {"status": "RUNNING", "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "kernel_options": options,
              "compile_dynamic": False if args.static else None, "cases": []}
    try:
        cases = [(31, 29, 2, "causal"), (32, 30, 2, "causal"), (129, 127, 2, "causal")]
        if policy == "t4":
            cases += [(65, 257, 2, "causal"), (256, 1024, 1, "encoder"),
                      (2048, 512, 1, "decoder"), (512, 256, 4, "decoder")]
        for q_len, kv_len, batch, pattern in cases:
            q = torch.randn(batch, q_len, 16, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
            k = torch.randn(batch, kv_len, 16, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
            v = torch.randn_like(k)
            def causal(b, h, qi, ki):
                if pattern == "encoder":
                    return qi // 2 == ki // 4
                if pattern == "decoder":
                    return qi // 8 == ki // 2
                return qi >= ki
            mask = create_block_mask(causal, B=batch, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cuda", _compile=True)
            with torch.inference_mode():
                actual = compiled(q, k, v, block_mask=mask, kernel_options=options)
                dense = causal(None, None, torch.arange(q_len, device="cuda")[:, None], torch.arange(kv_len, device="cuda")[None, :])
                # Force the math backend as an independent reference, not another fused kernel.
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense)
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            result["cases"].append({"q_len": q_len, "kv_len": kv_len, "batch": batch, "heads": 16, "mask": pattern,
                                    "head_dim": 64, "max_abs_error": float((actual - expected).abs().max().item()),
                                    "atol": 0.01, "rtol": 0.01})
            write_json(args.out, result)
        result["status"] = "PASS"
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
