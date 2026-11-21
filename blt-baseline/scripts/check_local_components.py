"""Run the user's existing entropy/patcher checks offline, preserving raw evidence.

This is component regression evidence against historical fixtures, not current
full-model parity against the newly pinned official source.
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from blt_baseline.evidence import file_manifest, git_state, sha256, write_json

TESTS = ["tests/test_blt_tokens.py", "tests/test_blt_rope_norm_masks.py",
         "tests/test_blt_entropy_model.py", "tests/test_blt_patcher.py", "tests/test_blt_parity.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = args.repo.resolve()
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-m", "pytest", *TESTS, "-q", "-ra", "-p", "no:cacheprovider", f"--junitxml={out / 'junit.xml'}"]
    completed = subprocess.run(command, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "pytest.txt").write_text(completed.stdout)
    suites = ET.parse(out / "junit.xml").getroot().iter("testsuite")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, 0))
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    from huggingface_hub import try_to_load_from_cache
    checkpoint = []
    for filename in ("config.json", "model.safetensors"):
        cached = try_to_load_from_cache("facebook/blt-entropy", filename)
        if isinstance(cached, str) and Path(cached).is_file():
            checkpoint.append({"filename": filename, "cache_path": cached,
                               "sha256": sha256(Path(cached)), "bytes": Path(cached).stat().st_size})
    result = {"status": "PASS" if completed.returncode == 0 and counts["passed"] > 0 else "FAIL",
              "exit_code": completed.returncode, "counts": counts,
              "command": command, "python": sys.executable, "repo": str(repo), "git": git_state(repo),
              "implementation": file_manifest(repo / "blt", ("*.py",)),
              "checkpoint": checkpoint,
              "fixtures": file_manifest(repo / "tests/fixtures", ("*.npz",)),
              "test_sources": [{"path": name, **file_manifest(repo, (name,))[0]} for name in TESTS],
              "limitation": "Existing entropy checkpoint and historical official-reference fixtures only; no full BLT checkpoint, generation, or GPU performance validated."}
    write_json(out / "result.json", result)
    (out / "report.md").write_text(
        f"# Local BLT component regression\n\nStatus: **{result['status']}**. "
        f"{counts['passed']} passed, {counts['skipped']} skipped, {counts['failures']} failures, {counts['errors']} errors.\n\n"
        "Three long-case logit checks have no stored logit fixture; entropy/patch-length checks still run. "
        "See pytest.txt for actual skip reasons.\n\n" + result["limitation"] + "\n"
    )
    print((out / "report.md").read_text())
    return completed.returncode if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
