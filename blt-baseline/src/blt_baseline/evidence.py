"""Content-addressed provenance, strict configuration, and offline preflight."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_REVISION = "9774ed4fcc78313f9f218295f3d7e4decdadf2ae"
FLEX_KERNEL_OPTIONS = {
    "upstream": {},
    "generic": {"FORCE_USE_FLEX_ATTENTION": True},
    "t4": {"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": 32, "BLOCK_N": 32,
           "num_stages": 1, "num_warps": 4},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"revision": None, "dirty": None}


def write_json(path: Path, value: object) -> None:
    """Replace atomically; refuse NaN/Infinity so missing evidence is explicit null."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class Config:
    upstream: str
    blt_snapshot: str
    entropy_snapshot: str
    workload: str
    upstream_revision: str = UPSTREAM_REVISION
    dtype: str = "bf16"
    seed: int = 42
    context_bytes: tuple[int, ...] = (128, 512, 2048)
    batch_sizes: tuple[int, ...] = (1, 4)
    output_steps: int = 32
    warmup: int = 2
    repeats: int = 5
    quality_max_bytes: int = 128
    profile: bool = True
    gpu_hourly_usd: float | None = None
    legacy_cutlass_alias: bool = False
    checkpoint_kind: str = "pretrained"
    flex_kernel_policy: str = "upstream"
    patcher_batch_policy: str = "native"
    entropy_execution: str = "padded"
    entropy_microbatch_size: int = 4

    @classmethod
    def load(cls, path: Path) -> "Config":
        obj = json.loads(path.read_text(encoding="utf-8"))
        unknown = obj.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        for name in ("upstream", "blt_snapshot", "entropy_snapshot", "workload"):
            value = Path(obj[name]).expanduser()
            obj[name] = str((path.parent / value).resolve())
        for name in ("context_bytes", "batch_sizes"):
            if name in obj:
                obj[name] = tuple(obj[name])
        cfg = cls(**obj)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError("dtype must be fp32, bf16, or fp16")
        for name in ("output_steps", "warmup", "repeats", "quality_max_bytes", "entropy_microbatch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.profile) is not bool:
            raise ValueError("profile must be boolean")
        if type(self.legacy_cutlass_alias) is not bool:
            raise ValueError("legacy_cutlass_alias must be boolean")
        if self.checkpoint_kind not in {"pretrained", "random_smoke"}:
            raise ValueError("checkpoint_kind must be pretrained or random_smoke")
        if self.flex_kernel_policy not in FLEX_KERNEL_OPTIONS:
            raise ValueError("Unknown flex_kernel_policy")
        if self.patcher_batch_policy not in {"native", "isolated_rows"}:
            raise ValueError("patcher_batch_policy must be native or isolated_rows")
        if self.entropy_execution not in {"padded", "length_aware_rows", "length_aware_batch"}:
            raise ValueError("Unknown entropy_execution")
        if self.entropy_execution != "padded" and self.patcher_batch_policy != "isolated_rows":
            raise ValueError("Length-aware execution requires isolated request semantics")
        for name in ("context_bytes", "batch_sizes"):
            values = getattr(self, name)
            if not values or any(type(v) is not int or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if self.gpu_hourly_usd is not None and (
            isinstance(self.gpu_hourly_usd, bool)
            or not isinstance(self.gpu_hourly_usd, (float, int))
            or not math.isfinite(self.gpu_hourly_usd)
            or self.gpu_hourly_usd <= 0
        ):
            raise ValueError("gpu_hourly_usd must be a finite positive number")
        if len(self.upstream_revision) != 40 or any(c not in "0123456789abcdef" for c in self.upstream_revision):
            raise ValueError("upstream_revision must be a full lowercase Git SHA")


def snapshot_status(path: Path, *, train_args: bool = False) -> dict:
    required = ["config.json", "model.safetensors"]
    if train_args:
        required.append("train_args.json")
    missing = [name for name in required if not (path / name).is_file()]
    return {"path": str(path), "missing": missing, "ready": not missing}


def doctor(cfg: Config) -> dict:
    packages = {}
    for name in ("torch", "xformers", "pydantic", "safetensors", "huggingface-hub", "numpy", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    blockers = []
    upstream = git_state(Path(cfg.upstream))
    if upstream["revision"] != cfg.upstream_revision or upstream["dirty"] is not False:
        blockers.append("Official source must be clean and match the configured revision")
    snapshots = {
        "blt": snapshot_status(Path(cfg.blt_snapshot), train_args=True),
        "entropy": snapshot_status(Path(cfg.entropy_snapshot)),
    }
    for name, status in snapshots.items():
        if not status["ready"]:
            blockers.append(f"{name} snapshot missing: {', '.join(status['missing'])}")
    cuda = {"available": False}
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
        if cuda["available"]:
            cuda.update(name=torch.cuda.get_device_name(0), count=torch.cuda.device_count(),
                        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                        capability=list(torch.cuda.get_device_capability(0)),
                        native_bf16=torch.cuda.is_bf16_supported(including_emulation=False))
            if cfg.dtype == "bf16" and not cuda["native_bf16"]:
                blockers.append("Configured bf16 lacks native GPU support; explicitly select fp16 or fp32")
    except ImportError:
        blockers.append("PyTorch is not installed")
    if not cuda["available"]:
        blockers.append("Native performance baseline requires CUDA; CPU/MPS is not substituted")
    for name in ("xformers", "pydantic", "safetensors", "huggingface-hub", "numpy"):
        if packages[name] is None:
            blockers.append(f"Missing runtime dependency: {name}")
    if not Path(cfg.workload).is_file():
        blockers.append("Workload is missing")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_LOAD_CHECK" if not blockers else "BLOCKED",
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "cuda": cuda, "upstream": upstream, "snapshots": snapshots,
        "config": asdict(cfg), "blockers": blockers,
        "note": "Readiness is not successful import, model parity, quality, or performance evidence.",
    }


def file_manifest(root: Path, patterns: tuple[str, ...]) -> list[dict]:
    paths = sorted({p for pattern in patterns for p in root.rglob(pattern) if p.is_file()})
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths]
