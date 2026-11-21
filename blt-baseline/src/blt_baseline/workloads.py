"""Small, authored diagnostics. These are not a held-out research benchmark."""

import json
from pathlib import Path


def load_workload(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"id", "category", "prompt", "continuation", "exact_match"}:
            raise ValueError(f"Invalid workload keys on line {line_number}")
        if any(not isinstance(row[key], str) or not row[key] for key in ("id", "category", "prompt", "continuation")):
            raise ValueError(f"Empty/non-string workload field on line {line_number}")
        if type(row["exact_match"]) is not bool or row["id"] in seen:
            raise ValueError(f"Invalid exact_match or duplicate id on line {line_number}")
        # Reject lone surrogates rather than silently dropping bytes as upstream encode can.
        row["prompt"].encode("utf-8", errors="strict")
        row["continuation"].encode("utf-8", errors="strict")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("Workload must not be empty")
    return rows


def systems_prompt(text: str, byte_budget: int) -> str:
    """Repeat a fixture for shape sweeps, retaining a valid UTF-8 prefix <= budget."""
    raw = text.encode("utf-8", errors="strict")
    if not raw or byte_budget < 1:
        raise ValueError("Nonempty text and positive byte budget required")
    repeated = (raw * ((byte_budget + len(raw) - 1) // len(raw)))[:byte_budget]
    result = repeated.decode("utf-8", errors="ignore")
    if not result:
        raise ValueError("Byte budget cannot hold the first UTF-8 code point")
    return result


def output_accounting(ids: list[int], *, offset: int = 4) -> dict:
    raw = bytes(token - offset for token in ids if offset <= token < offset + 256)
    try:
        raw.decode("utf-8", errors="strict")
        valid = True
    except UnicodeDecodeError:
        valid = False
    return {
        "steps": len(ids), "raw_byte_count": len(raw),
        "special_token_count": sum(not offset <= t < offset + 256 for t in ids),
        "valid_utf8": valid, "raw_hex": raw.hex(),
        "display_text": raw.decode("utf-8", errors="replace"),
    }
