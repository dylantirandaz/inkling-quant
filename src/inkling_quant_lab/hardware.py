"""Portable environment and hardware metadata probes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from inkling_quant_lab.benchmarking.energy import probe_default_energy_sensor
from inkling_quant_lab.distributed import probe_gloo_capability
from inkling_quant_lab.security import run_command


def _torch_accelerators() -> tuple[tuple[str, ...], dict[str, Any]]:
    """Probe accelerators through core PyTorch without optional backend imports."""

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return (), {"probe": "torch_unavailable"}

    devices: list[str] = []
    details: dict[str, Any] = {"torch_version": str(torch.__version__)}
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        cuda_available = bool(cuda.is_available())
        details["cuda_available"] = cuda_available
        details["cuda_device_count"] = int(cuda.device_count()) if cuda_available else 0
        if cuda_available:
            devices.append("cuda")
            details["cuda_devices"] = tuple(
                {
                    "index": index,
                    "name": str(cuda.get_device_name(index)),
                    "compute_capability": tuple(
                        int(value) for value in cuda.get_device_capability(index)
                    ),
                    "total_memory_bytes": int(cuda.get_device_properties(index).total_memory),
                }
                for index in range(int(cuda.device_count()))
            )
    backends = getattr(torch, "backends", None)
    mps = None if backends is None else getattr(backends, "mps", None)
    if mps is not None:
        details["mps_built"] = bool(mps.is_built())
        details["mps_available"] = bool(mps.is_available())
        if details["mps_available"]:
            devices.append("mps")
            get_name = getattr(mps, "get_name", None)
            get_core_count = getattr(mps, "get_core_count", None)
            if callable(get_name):
                details["mps_device_name"] = str(get_name())
            if callable(get_core_count):
                details["mps_core_count"] = int(get_core_count())
            recommended_max_memory = getattr(
                getattr(torch, "mps", None), "recommended_max_memory", None
            )
            if callable(recommended_max_memory):
                details["mps_recommended_max_memory_bytes"] = int(recommended_max_memory())
    return tuple(devices), details


def _package_versions() -> dict[str, str]:
    packages = (
        "inkling-quant-lab",
        "accelerate",
        "defuser",
        "gptqmodel",
        "huggingface-hub",
        "kernels",
        "mlx",
        "mlx-lm",
        "optimum",
        "pydantic",
        "PyYAML",
        "safetensors",
        "sglang",
        "torch",
        "torchao",
        "transformers",
        "typer",
        "vllm",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def git_provenance(directory: Path) -> dict[str, str | bool | None]:
    """Return Git commit/dirty state, or explicit unavailability."""

    if not (directory / ".git").exists():
        return {"commit": None, "dirty": None}
    revision = run_command(("git", "rev-parse", "HEAD"), cwd=directory)
    status = run_command(("git", "status", "--porcelain"), cwd=directory)
    if revision.returncode != 0 or status.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": revision.stdout.strip(), "dirty": bool(status.stdout.strip())}


def project_source_provenance(directory: Path) -> dict[str, Any]:
    """Hash the runnable project source when Git provenance is unavailable or dirty."""

    source_root = directory / "src" / "inkling_quant_lab"
    required = (directory / "pyproject.toml", directory / "uv.lock", source_root)
    if not all(path.exists() for path in required):
        return {
            "kind": "unavailable",
            "reason": "project source tree, pyproject.toml, or uv.lock is missing",
        }

    source_entries = tuple(sorted(source_root.rglob("*"), key=lambda path: path.as_posix()))
    candidates = (required[0], required[1], *source_entries)
    symlinks = tuple(path for path in (*required, *source_entries) if path.is_symlink())
    if symlinks:
        symlink_names = ", ".join(path.relative_to(directory).as_posix() for path in symlinks)
        raise ValueError(f"project source provenance must not contain symlinks: {symlink_names}")
    unsafe_bytecode = tuple(
        path
        for path in source_entries
        if path.is_file()
        and path.suffix in {".pyc", ".pyo"}
        and "__pycache__" not in path.relative_to(directory).parts
    )
    if unsafe_bytecode:
        bytecode_names = ", ".join(
            path.relative_to(directory).as_posix() for path in unsafe_bytecode
        )
        raise ValueError(
            f"project source provenance rejects importable sourceless bytecode: {bytecode_names}"
        )

    files: list[dict[str, str | int]] = []
    for path in candidates:
        relative_path = path.relative_to(directory)
        if not path.is_file() or "__pycache__" in relative_path.parts or path.name == ".DS_Store":
            continue
        payload = path.read_bytes()
        files.append(
            {
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    files.sort(key=lambda item: str(item["path"]))
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "kind": "filesystem_sha256_manifest_v1",
        "tree_sha256": hashlib.sha256(b"inkling-quant-project-source-v1\0" + canonical).hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def probe_environment(project_root: Path | None = None) -> dict[str, Any]:
    """Collect reproducibility metadata without importing optional GPU libraries."""

    root = Path.cwd() if project_root is None else project_root
    accelerators, accelerator_details = _torch_accelerators()
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hardware": {
            "logical_cpu_count": os.cpu_count(),
            "device": "cpu",
            "accelerator": accelerators[0] if len(accelerators) == 1 else None,
            "available_devices": ("cpu", *accelerators),
            "accelerators": accelerators,
            "accelerator_probe": accelerator_details,
        },
        "packages": _package_versions(),
        "energy": probe_default_energy_sensor().model_dump(mode="json"),
        "distributed": {"gloo": probe_gloo_capability().model_dump(mode="json")},
        "git": git_provenance(root),
        "project_source": project_source_provenance(root),
    }
