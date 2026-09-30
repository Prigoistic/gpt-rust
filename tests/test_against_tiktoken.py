import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken

from tokeniser import Tokeniser

TEST_STRINGS = [
    "Hello, world!",
    "don't stop believing",
    "I'll be there in 2026, promise.",
    "   leading and trailing whitespace   ",
    "unicode: café, naïve, 日本語, 😀🚀",
    "numbers 123456 and 3.14159 mixed with text42",
    "Repeated repeated repeated words words words.",
    "",
    "a",
    "\n\nmultiple\n\nnewlines\n\n",
]


def check_against_tiktoken(tokeniser: Tokeniser, enc, strings: list[str]) -> bool:
    all_match = True
    for s in strings:
        ours = tokeniser.encode(s)
        theirs = enc.encode(s)
        match = ours == theirs and tokeniser.decode(ours) == s
        all_match &= match
        print(("OK  " if match else "FAIL"), repr(s))
        if not match:
            print(f"    ours:   {ours}")
            print(f"    theirs: {theirs}")
    return all_match


def benchmark_cache_benefit(tokeniser: Tokeniser, text: str, n: int = 5) -> None:
    start = time.perf_counter()
    for _ in range(n):
        tokeniser.encode(text)
    elapsed = time.perf_counter() - start
    print(f"cached run: {elapsed / n * 1000:.2f} ms/encode (avg over {n} runs)")


if __name__ == "__main__":
    tok = Tokeniser.from_gpt2_files()
    enc = tiktoken.get_encoding("gpt2")

    print("=== correctness vs tiktoken ===")
    ok = check_against_tiktoken(tok, enc, TEST_STRINGS)

    print("\n=== word-level cache benefit on repetitive text ===")
    sample_text = ("the quick brown fox jumps over the lazy dog. " * 200).strip()
    benchmark_cache_benefit(Tokeniser.from_gpt2_files(), sample_text)

    print("\n" + ("ALL MATCH" if ok else "MISMATCH FOUND"))
