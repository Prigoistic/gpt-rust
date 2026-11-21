import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from blt_baseline.evidence import Config, file_manifest, sha256, snapshot_status, write_json
from blt_baseline.runner import aggregate_quality, execute, percentile, render_report, timing_summary
from blt_baseline.workloads import load_workload, output_accounting, systems_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_config_paths_resolve_from_config_not_cwd(tmp_path, monkeypatch):
    path = ROOT / "configs/local.json"
    monkeypatch.chdir(tmp_path)
    cfg = Config.load(path)
    assert Path(cfg.workload) == ROOT / "data/diagnostics.jsonl"
    assert Path(cfg.upstream).name == "blt"
    assert Path(cfg.blt_snapshot).is_absolute()


@pytest.mark.parametrize("field,value", [
    ("warmup", 0), ("repeats", False), ("output_steps", -1), ("dtype", "auto"),
    ("gpu_hourly_usd", -1), ("gpu_hourly_usd", float("nan")),
    ("gpu_hourly_usd", True), ("context_bytes", (1, 1)),
    ("batch_sizes", ()), ("context_bytes", (1.5,)), ("seed", -1), ("profile", 1),
])
def test_invalid_config_rejected(field, value):
    cfg = Config.load(ROOT / "configs/local.json")
    with pytest.raises(ValueError):
        replace(cfg, **{field: value}).validate()


def test_unknown_config_keys_fail(tmp_path):
    obj = json.loads((ROOT / "configs/local.json").read_text())
    obj["silent_cast"] = True
    path = tmp_path / "config.json"
    write_json(path, obj)
    with pytest.raises(ValueError, match="Unknown"):
        Config.load(path)


def test_raw_bytes_are_not_characters_or_special_tokens():
    ids = [b + 4 for b in "😀".encode()] + [1, 2, 3]
    result = output_accounting(ids)
    assert result["steps"] == 7
    assert result["raw_byte_count"] == 4
    assert result["special_token_count"] == 3
    assert result["valid_utf8"]
    assert result["display_text"] == "😀"


def test_invalid_utf8_retains_raw_bytes_and_flags_invalid():
    result = output_accounting([0xF0 + 4, 0x9F + 4])
    assert result["raw_byte_count"] == 2
    assert not result["valid_utf8"]
    assert result["raw_hex"] == "f09f"


def test_systems_prompt_does_not_split_unicode():
    prompt = systems_prompt("é", 5)
    assert prompt == "éé"
    assert len(prompt.encode()) == 4
    with pytest.raises(ValueError):
        systems_prompt("😀", 1)


def test_bpb_weights_bytes_not_documents():
    rows = [dict(nll_nats=math.log(2), scored_bytes=1, fixed_length_match=True),
            dict(nll_nats=27 * math.log(2), scored_bytes=9, fixed_length_match=None)]
    result = aggregate_quality(rows)
    assert result["bpb"] == pytest.approx(2.8)
    assert result["fixed_length_correct"] == 1
    assert result["fixed_length_total"] == 1


def test_throughput_uses_total_bytes_over_total_time():
    samples = [dict(elapsed_seconds=1, raw_byte_count=100),
               dict(elapsed_seconds=3, raw_byte_count=100)]
    result = timing_summary(samples, 3.6)
    assert result["raw_bytes_per_second_per_gpu"] == 50
    assert result["usd_per_million_raw_bytes"] == 20
    assert result["batch_latency_seconds_p95"] == pytest.approx(2.9)
    assert timing_summary(samples, None)["usd_per_million_raw_bytes"] is None
    with pytest.raises(ValueError):
        timing_summary([dict(elapsed_seconds=0, raw_byte_count=100)], None)


def test_percentile_single_sample():
    assert percentile([3], .95) == 3
    with pytest.raises(ValueError):
        percentile([], .95)


def test_broken_checkpoint_symlink_is_missing(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").symlink_to(tmp_path / "absent")
    assert snapshot_status(tmp_path)["missing"] == ["model.safetensors"]


def test_hashes_detect_changed_artifacts(tmp_path):
    path = tmp_path / "x.py"
    path.write_text("a")
    first = file_manifest(tmp_path, ("*.py",))
    path.write_text("b")
    assert first[0]["sha256"] != sha256(path)


def test_json_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "x.json", {"measurement": float("nan")})
    assert not (tmp_path / "x.json").exists()


def test_duplicate_workload_fails(tmp_path):
    row = (ROOT / "data/diagnostics.jsonl").read_text().splitlines()[0]
    path = tmp_path / "rows.jsonl"
    path.write_text(row + "\n" + row)
    with pytest.raises(ValueError, match="duplicate"):
        load_workload(path)


def test_workload_contains_separate_controls():
    rows = load_workload(ROOT / "data/diagnostics.jsonl")
    categories = {r["category"] for r in rows}
    assert {"code", "computed_value", "multilingual", "copy_control", "boundary_control"} <= categories


def test_blocked_attempt_writes_report_and_refuses_overwrite(tmp_path, monkeypatch):
    state = {"blockers": ["No CUDA"], "upstream": {"revision": "pinned"}, "cuda": {"available": False}}
    monkeypatch.setattr("blt_baseline.runner.doctor", lambda _cfg: state)
    cfg = Config.load(ROOT / "configs/local.json")
    out = tmp_path / "run"
    assert execute(cfg, out) is False
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "BLOCKED"
    assert "NOT_RUN" in (out / "report.md").read_text()
    assert result["performance"] == []
    assert not (out / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        execute(cfg, out)


def test_report_preserves_failure_status():
    result = {"status": "FAILED", "readiness": {"blockers": [], "upstream": {"revision": "abc"},
              "cuda": {"available": True}}, "error": "parity mismatch"}
    report = render_report(result)
    assert "**FAILED**" in report
    assert "parity mismatch" in report
    assert "speedup" in report


def test_installed_package_archives_its_code_without_checkout(tmp_path, monkeypatch):
    import shutil
    import tarfile
    import blt_baseline.runner as runner

    package = tmp_path / "lib/python3.12/site-packages/blt_baseline"
    shutil.copytree(ROOT / "src/blt_baseline", package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(runner, "__file__", str(package / "runner.py"))
    assert runner.source_locations() == (package, None)
    cfg = Config.load(ROOT / "configs/local.json")
    out = tmp_path / "artifacts"
    out.mkdir()
    digest = runner.capture_source(out, cfg)
    assert digest == sha256(out / "harness-source.tar.gz")
    with tarfile.open(out / "harness-source.tar.gz") as stream:
        names = stream.getnames()
    assert "src/blt_baseline/runner.py" in names
    assert "workload.jsonl" in names
    assert "pyproject.toml" not in names
