"""Build deterministic benchmark corpora under bench-data/."""

import sysconfig
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "bench-data"
VARIED_BYTES = 12_000_000
REPETITIVE_UNIT = "the quick brown fox jumps over the lazy dog. "
REPETITIVE_REPEATS = 200_000


def build_varied() -> str:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    parts: list[str] = []
    size = 0
    for path in sorted(stdlib.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        parts.append(text)
        size += len(text.encode("utf-8"))
        if size >= VARIED_BYTES:
            break
    return "".join(parts)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "varied.txt").write_text(build_varied(), encoding="utf-8")
    (OUT / "repetitive.txt").write_text(REPETITIVE_UNIT * REPETITIVE_REPEATS, encoding="utf-8")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
