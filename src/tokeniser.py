"""GPT-2-compatible byte-level BPE tokeniser, built from scratch."""

import json
from functools import lru_cache
from pathlib import Path

import regex as re

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# GPT-2's fixed pre-split. Merges never cross a match boundary, so this must match the
# original exactly. Needs the third-party `regex` module for \p{L}/\p{N}.
PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable-char table, so every byte has a visible stand-in."""
    # Printable bytes map to themselves.
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = printable[:]
    unicode_points = printable[:]

    # Control/whitespace bytes are shifted to unused code points from U+0100 up.

    n = 0
    for b in range(2**8):
        if b not in byte_values:
            byte_values.append(b)
            unicode_points.append(2**8 + n)
            n += 1

    return dict(zip(byte_values, (chr(c) for c in unicode_points)))


class Tokeniser:
    def __init__(self, encoder: dict[str, int], bpe_merges: list[tuple[str, str]]):
        self.encoder = encoder
        self.decoder = {v: k for k, v in encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # Earlier merges (lower rank) have priority when several pairs could merge.
        self.bpe_ranks = {pair: i for i, pair in enumerate(bpe_merges)}
        # Real text repeats words constantly, so memoizing per chunk skips most merge loops.
        self.cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_gpt2_files(cls, data_dir: Path = DATA_DIR) -> "Tokeniser":
        encoder = json.loads((data_dir / "encoder.json").read_text())
        # vocab.bpe starts with a version header and ends with a blank line; neither is a merge.
        lines = (data_dir / "vocab.bpe").read_text(encoding="utf-8").split("\n")[1:-1]
        merges = [tuple(line.split()) for line in lines]
        return cls(encoder, merges)

    def _merge_symbols(self, byte_str: str) -> tuple[str, ...]:
        """BPE one chunk: repeatedly merge the lowest-rank adjacent pair until none apply.

        Rescanning all pairs each round is fine here: chunks are single words, not whole texts.
        """
        cached = self.cache.get(byte_str)
        if cached is not None:
            return cached

        symbols = tuple(byte_str)
        while len(symbols) > 1:
            pairs = set(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break

            first, second = best
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = tuple(merged)

        self.cache[byte_str] = symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in re.findall(PRETOKENIZE_PATTERN, text):
            byte_str = "".join(self.byte_encoder[b] for b in chunk.encode("utf-8"))
            ids.extend(self.encoder[s] for s in self._merge_symbols(byte_str))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[i] for i in ids)
        raw = bytearray(self.byte_decoder[c] for c in text)
        # A token boundary can split a multi-byte character, so tolerate invalid UTF-8.
        return raw.decode("utf-8", errors="replace")
