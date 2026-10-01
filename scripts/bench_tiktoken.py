"""Measure tiktoken gpt2 encode throughput (single thread) on bench-data/."""

import time
from pathlib import Path

import tiktoken

DATA = Path(__file__).resolve().parent.parent / "bench-data"
RUNS = 5
DOC_BYTES = 16 * 1024


def split_docs(text: str, size: int) -> list[str]:
    data = text.encode("utf-8")
    docs, pos = [], 0
    while pos < len(data):
        cut = min(pos + size, len(data))
        while cut < len(data) and (data[cut] & 0xC0) == 0x80:
            cut -= 1
        docs.append(data[pos:cut].decode("utf-8"))
        pos = cut
    return docs


def best_seconds(fn) -> float:
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    for name in ("varied", "repetitive"):
        text = (DATA / f"{name}.txt").read_text(encoding="utf-8")
        mib = len(text.encode("utf-8")) / 2**20
        docs = split_docs(text, DOC_BYTES)
        single = mib / best_seconds(lambda: enc.encode(text))
        batch = mib / best_seconds(lambda: enc.encode_batch(docs))
        print(f"tiktoken/{name}: {single:.1f} MiB/s single, {batch:.1f} MiB/s batch (best of {RUNS})")
