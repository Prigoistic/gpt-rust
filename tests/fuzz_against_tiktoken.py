"""Fuzz the Rust tokeniser against tiktoken on unicode-heavy random strings."""

import random
import sys
from pathlib import Path

import tiktoken
from tokeniser_rs import Tokeniser

ROOT = Path(__file__).resolve().parent.parent
CASES = 100_000
SEED = 20261001

PIECES = [
    "a", "Zed", "é", "ß", "日本", "0", "42", "٣", "²", " ", "  ", "\n", "\t", "\r\n",
    " ", " ", "　", "\u0085", "'", "'s", "'re", "'ll", "'d", "'m", "'t",
    "'ve", ".", ",", "!", "-", "_", "ः", "😀", "​", "‍", "́",
    "the", "ing", "tion", "http://", "def ", "    ", "==", "\\n",
]


def random_char(rng: random.Random) -> str:
    while True:
        cp = rng.randrange(0x110000)
        if not 0xD800 <= cp <= 0xDFFF:
            return chr(cp)


def random_text(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randrange(40)):
        parts.append(random_char(rng) if rng.random() < 0.2 else rng.choice(PIECES))
    return "".join(parts)


if __name__ == "__main__":
    rng = random.Random(SEED)
    enc = tiktoken.get_encoding("gpt2")
    tok = Tokeniser.from_gpt2_files(str(ROOT / "data"))

    failures = 0
    for _ in range(CASES):
        text = random_text(rng)
        if tok.encode(text) != enc.encode(text, disallowed_special=()):
            failures += 1
            if failures <= 5:
                print(f"MISMATCH: {text!r}")
    print(f"{CASES - failures}/{CASES} match tiktoken")
    sys.exit(1 if failures else 0)
