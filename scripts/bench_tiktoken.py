"""Measure tiktoken gpt2 encode throughput (single thread) on bench-data/."""

import time
from pathlib import Path

import tiktoken

DATA = Path(__file__).resolve().parent.parent / "bench-data"
RUNS = 5


def best_seconds(enc: tiktoken.Encoding, text: str) -> float:
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        enc.encode(text)
        times.append(time.perf_counter() - start)
    return min(times)


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    for name in ("varied", "repetitive"):
        text = (DATA / f"{name}.txt").read_text(encoding="utf-8")
        mib = len(text.encode("utf-8")) / 2**20
        print(f"tiktoken/{name}: {mib / best_seconds(enc, text):.1f} MiB/s (best of {RUNS})")
