"""Recompute pilot measurements and validate block accounting from JSON lines."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

def summarize(path):
    header, *rows = map(json.loads, path.read_text().splitlines())
    assert hashlib.sha256(path.with_name("fixed_width.py").read_bytes()).hexdigest() == header["script_sha256"]
    assert len(rows) == len(header["prompts"])*len(header["widths"])*4
    stats = {}
    widths = header["widths"]
    for name in header["prompts"]:
        by_width = {}
        for w in widths:
            records = [r for r in rows if r["prompt"] == name and r["width"] == w]
            assert sorted(r["rep"] for r in records) == [0, 1, 2, 3]
            for r in records:
                assert r["ids"] == r["reference_ids"] and len(r["ids"]) == header["steps"]
                assert r["ids"][0] >= 4 and 0 < r["first_byte_s"] <= r["seconds"]
                if w != 1:
                    total = 1
                    for b in r["rounds"]:
                        p, a = b["proposed"], b["accepted"]
                        assert 0 <= a <= p <= w and b["bonus"] in (0, 1)
                        assert not (b["bonus"] and a < p)
                        total += a + int(a < p) + b["bonus"]
                    assert total == len(r["ids"])
            samples = sorted((r for r in records if not r["warmup"]), key=lambda r:r["rep"])
            elapsed = [r["seconds"] for r in samples]
            if w == 1:
                reference = elapsed
            first = statistics.median(r["first_byte_s"] for r in samples)
            r = samples[0]
            rounds = r["rounds"]
            count = len(rounds)
            proposed = sum(b["proposed"] for b in rounds)
            accepted = sum(b["accepted"] for b in rounds)
            full = 1 + count if w > 1 else header["steps"]
            by_width[w] = dict(bytes_s=statistics.median(x["raw_bytes"]/x["seconds"] for x in samples),
                               speedup=statistics.median(a/b for a,b in zip(reference, elapsed)),
                               first_ms=first*1000, full_calls=full, local_decoder_calls=full+proposed,
                               accepted=accepted, proposed=proposed, verifications=count,
                               verify_ms=statistics.median(sum(b["verify_s"] for b in x["rounds"])*1000 for x in samples),
                               peak_mib=max(x["peak_bytes"] for x in samples)/2**20)
        stats[name] = by_width
    return stats

def summarize_holdout(path):
    header, *rows = map(json.loads, path.read_text().splitlines())
    for script, expected in header["sha256"].items():
        assert hashlib.sha256(path.with_name(script).read_bytes()).hexdigest() == expected
    assert len(rows) == len(header["prompts"])*12
    result = {}
    for name in header["prompts"]:
        group = [r for r in rows if r["prompt"] == name]
        gold = next(r["ids"] for r in group if r["width"] == 0)
        assert len(gold) == header["steps"]
        assert all(r["exact"] and r["ids"] == gold and r["elapsed"] > 0 for r in group)
        by_width = {}
        for width in (0, 4, 8):
            samples = sorted((r for r in group if r["width"] == width), key=lambda r:r["rep"])
            assert [r["rep"] for r in samples] == list(range(4))
            elapsed = [x["elapsed"] for x in samples[1:]]
            if width == 0:
                reference = elapsed
            by_width[width] = dict(
                median_seconds=statistics.median(elapsed),
                speedup=statistics.median(a/b for a,b in zip(reference,elapsed)),
                raw_bytes=sum(t>=4 for t in gold))
        result[name] = by_width
    return result

if __name__ == "__main__":
    result = dict(primary=summarize(Path(sys.argv[1])))
    if len(sys.argv) > 2:
        result["holdout"] = summarize_holdout(Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
