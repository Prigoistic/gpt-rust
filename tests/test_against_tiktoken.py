import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tiktoken

from tokeniser import Tokeniser
from tokeniser_rs import Tokeniser as RustTokeniser

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
    "a" * 2000,
    "=" * 500 + " tail",
]


def check_against_tiktoken(name: str, tokeniser, enc, strings: list[str]) -> bool:
    print(f"\n=== {name}: correctness vs tiktoken ===")
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
    print(f"{name}: {'ALL MATCH' if all_match else 'MISMATCH FOUND'}")
    return all_match


def benchmark(name: str, fn, n: int = 20) -> None:
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    print(f"{name:28s}: {elapsed / n * 1000:.3f} ms/encode (avg over {n} runs)")


if __name__ == "__main__":
    py_tok = Tokeniser.from_gpt2_files()
    rs_tok = RustTokeniser.from_gpt2_files(str(ROOT / "data"))
    enc = tiktoken.get_encoding("gpt2")

    ok_py = check_against_tiktoken("Python", py_tok, enc, TEST_STRINGS)
    ok_rs = check_against_tiktoken("Rust (PyO3, debug build)", rs_tok, enc, TEST_STRINGS)

    sample_text = ("the quick brown fox jumps over the lazy dog. " * 200).strip()
    print(f"\n=== speed on {len(sample_text)}-char repetitive text ===")
    print("NOTE: Rust binding is a DEBUG build (release build hits an unresolved")
    print("macOS/Rust-toolchain linker bug) - not a fair optimized-vs-optimized number yet.")
    benchmark("python (regex + cache)", lambda: py_tok.encode(sample_text))
    benchmark("rust (debug, cache + rayon)", lambda: rs_tok.encode(sample_text))
    benchmark("tiktoken (release, Rust core)", lambda: enc.encode(sample_text))

    print("\n" + ("ALL MATCH" if ok_py and ok_rs else "MISMATCH FOUND ABOVE"))
