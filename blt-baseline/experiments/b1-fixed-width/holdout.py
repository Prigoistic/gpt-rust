"""Compare two prespecified draft widths to unmodified M1 on fresh longer prompts."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from fixed_width import run
from blt_baseline.evidence import Config
from blt_baseline.native import NativeBLT

PROMPTS = {
    "prose": ("On Monday we walked down to the harbor to take measurements. "
              "The tide had risen above the old mark on the wall. " * 4 +
              "When we returned on Tuesday, the first thing we noticed was"),
    "code": ('def mean(values):\n    if not values:\n        return 0\n'
             '    return sum(values) / len(values)\n\n'
             'def median(values):\n    ordered = sorted(values)\n    n = len(ordered)\n'
             '    if n == 0:\n        return 0\n    if n % 2 == 1:\n'),
    "hindi": ("हम नदी के किनारे पुराने पुल तक पैदल गए। वहाँ से गाँव दिखाई देता था। " * 2 +
              "जब बारिश शुरू हुई तो हमने"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = Config.load(args.config.resolve())
    native = NativeBLT(cfg)
    orders = ((0, 4, 8), (0, 4, 8), (4, 8, 0), (8, 0, 4))
    with args.out.open("x") as f:
        def save(row):
            f.write(json.dumps(row) + "\n")
            f.flush()
        save(dict(prompts=PROMPTS, steps=64, upstream=cfg.upstream_revision,
                  sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(__file__).with_name("fixed_width.py"))}))
        for name, prompt in PROMPTS.items():
            gold = None
            for rep, order in enumerate(orders):
                for w in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if w == 0:
                        generated = native.generate([prompt], 64)
                        ids = generated["ids"][0]
                        obs = {}
                    else:
                        obs = run(native, prompt, w, 64)
                        ids = obs["ids"]
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    if gold is None:
                        gold = ids
                    save(dict(prompt=name, width=w, rep=rep, elapsed=elapsed,
                              ids=ids, exact=ids==gold, candidate=obs))
                    print(name, rep, w, ids==gold, round(elapsed, 3), flush=True)
                    if ids != gold:
                        raise RuntimeError("held-out greedy divergence")
if __name__ == "__main__":
    main()
