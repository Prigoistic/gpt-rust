"""Validate stream trace parity, timing order and script identity, then print medians."""
import hashlib
import json
import statistics
from pathlib import Path

def summarize(directory, filename, script):
    header, *rows = map(json.loads, (directory / filename).read_text().splitlines())
    assert header["script_sha256"] == hashlib.sha256((directory / script).read_bytes()).hexdigest()
    assert len(rows) == len(header["prompts"]) * len(header["widths"]) * 4
    result = {}
    fields = ["first_byte_s", "first_visible_s", "four_s", "sixteen_s", "seconds"]
    for name in header["prompts"]:
        result[name] = {}
        for width in header["widths"]:
            samples = [r for r in rows if r["prompt"] == name and r["width"] == width]
            assert sorted(r["rep"] for r in samples) == [0,1,2,3]
            for r in samples:
                assert r["ids"] == r["reference_ids"] and r["ids"][0] == 32+4
                assert all(r[f] > 0 for f in fields)
                assert r["first_byte_s"] <= r["first_visible_s"] <= r["seconds"]
                assert r["first_byte_s"] <= r["four_s"] <= r["sixteen_s"] <= r["seconds"]
            result[name][width] = {f: statistics.median(r[f] for r in samples if r["rep"]) for f in fields}
    return result

if __name__ == "__main__":
    directory = Path(__file__).parent
    print(json.dumps(dict(block=summarize(directory, "stream.jsonl", "stream_probe.py"),
                          visible_first=summarize(directory, "visible.jsonl", "visible_first.py")), indent=2))
