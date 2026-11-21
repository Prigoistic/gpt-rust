"""Single-request, uncached, anchored local self-speculation on released BLT-1B."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from blt_baseline.evidence import Config
from blt_baseline.native import NativeBLT

PROMPTS = {
    "prose": "The train pulled into the station just before dawn. On the empty platform,",
    "code": "def sum_even(numbers):\n    total = 0\n    for number in numbers:\n        if",
    "hindi": "आज सुबह जब मैंने खिड़की खोली तो",
}
WIDTHS = (1, 2, 4, 8, 16, 32)


def draft_logits(model, tokens, patch_lengths, frozen, anchor_ids):
    """Extend the last visible decoder patch; reuse its verified global latents."""
    from bytelatent.model.blt import compute_hash_embeddings, cross_attn_mask
    from bytelatent.model.local_models import create_causal_mask

    encoder = model.local_encoder
    embeds = compute_hash_embeddings(tokens, encoder, model.encoder_hash_tok_embedding,
                                     model.encoder_hash_byte_group_nb_functions,
                                     model.encoder_hash_byte_group_size,
                                     model.encoder_hash_byte_group_vocab)
    h = encoder.apply_embedding(tokens, embeds)
    length = tokens.shape[1]
    mask = create_causal_mask(length, encoder.attn_impl, "local_block_causal",
                              sliding_window=encoder.sliding_window, tokens=tokens,
                              eos_id=encoder.eos_id)
    freqs = encoder.rope(seqlen=length)
    for layer in encoder.layers:
        h = layer(h, mask=mask, freq_cis=freqs, attn_impl=encoder.attn_impl)
    extra = length - anchor_ids.shape[1]
    ids = torch.cat([anchor_ids, anchor_ids[:, -1:].expand(-1, extra)], dim=1)
    mask = cross_attn_mask(ids, patch_lengths, length, cross_attn_k=model.cross_attn_k,
                           window=model.cross_attn_window_decoder,
                           block_mask=model.cross_attn_use_flex_attention)
    logits, _ = model.local_decoder(tokens=tokens, embeds=h, patch_embeds=frozen, cross_mask=mask)
    return logits[:, -1].argmax(-1).item()


@torch.inference_mode()
def run(native, prompt, width, steps):
    from bytelatent.model.blt import decoder_patch_ids_from_lengths
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    ids = native.encode(prompt)
    if len(ids) + steps > native.max_prefix_tokens:
        raise ValueError("context limit")
    buf = torch.empty((1, len(ids) + steps), device=native.device, dtype=torch.long)
    buf[0, :len(ids)] = torch.tensor(ids, device=native.device)
    pos = len(ids)
    end = buf.shape[1]
    rounds = []
    output = []
    first = None
    while pos < end:
        prefix = buf[:, :pos]
        count = min(width, end - pos)
        frozen = []
        def capture(_module, _args, kwargs):
            frozen.append(kwargs["patch_embeds"])
        handle = None
        if count > 1:
            handle = native.model.local_decoder.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            anchor, lengths = native.next_logits(prefix, diagnose=True)
        finally:
            if handle is not None:
                handle.remove()
        token = anchor.argmax(-1).item()
        output.append(token)
        if first is None and token >= 4:
            first = time.perf_counter() - start
        buf[0, pos] = token
        pos += 1
        if count == 1:
            rounds.append(dict(proposed=0, accepted=0, verify_s=0, bonus=0))
            continue
        anchor_ids = decoder_patch_ids_from_lengths(lengths, 0, pos - 1)
        proposed = []
        for j in range(count - 1):
            token = draft_logits(native.model, buf[:, :pos + j], lengths, frozen[0], anchor_ids)
            proposed.append(token)
            buf[0, pos + j] = token
        torch.cuda.synchronize()
        before = time.perf_counter()
        block = buf[:, :pos + len(proposed)]
        block_lengths, _ = native.patcher.patch(block, include_next_token=True)
        pred = native.model(block, patch_lengths=block_lengths)[0, pos-1:].argmax(-1).tolist()
        verify_s = time.perf_counter() - before
        accepted = next((i for i, (a, b) in enumerate(zip(proposed, pred)) if a != b), len(proposed))
        committed = proposed[:accepted]
        bonus = int(accepted == len(proposed) and pos + accepted < end)
        if accepted < len(proposed) or bonus:
            committed.append(pred[accepted])
        buf[0, pos:pos + len(committed)] = torch.tensor(committed, device=native.device)
        output.extend(committed)
        pos += len(committed)
        rounds.append(dict(proposed=len(proposed), accepted=accepted, verify_s=verify_s, bonus=bonus))
    torch.cuda.synchronize()
    return dict(ids=output, seconds=time.perf_counter()-start, first_byte_s=first,
                raw_bytes=sum(x >= 4 for x in output), rounds=rounds,
                peak_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = Config.load(args.config.resolve())
    native = NativeBLT(cfg)
    assert native.model.cross_attn_decoder and native.model.patching_mode == "entropy"
    rows = []
    with args.out.open("x") as f:
        def save(obj):
            f.write(json.dumps(obj) + "\n")
            f.flush()
        save(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  upstream=cfg.upstream_revision, checkpoints=[Path(cfg.blt_snapshot).name, Path(cfg.entropy_snapshot).name],
                  prompts=PROMPTS, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), steps=40, widths=WIDTHS))
        for name, prompt in PROMPTS.items():
            gold = native.generate([prompt], 40)["ids"][0]
            for rep in range(4):
                order = list(WIDTHS[rep:] + WIDTHS[:rep])
                if rep % 2:
                    order.reverse()
                for width in order:
                    obs = run(native, prompt, width, 40)
                    row = dict(prompt=name, width=width, rep=rep, warmup=rep==0,
                               exact=obs["ids"]==gold, reference_ids=gold, **obs)
                    save(row)
                    print(name, rep, width, row["exact"], round(row["seconds"], 3), flush=True)
                    if not row["exact"]:
                        raise RuntimeError("verified greedy divergence; inspect saved IDs before timing claims")


if __name__ == "__main__":
    main()
