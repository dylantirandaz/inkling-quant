"""Environment provenance contracts for optional research backends."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from inkling_quant_lab.hardware import probe_environment, project_source_provenance

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_environment_records_optional_backend_package_versions() -> None:
    environment = probe_environment()

    packages = environment["packages"]
    assert {
        "accelerate",
        "defuser",
        "gptqmodel",
        "huggingface-hub",
        "kernels",
        "mlx",
        "mlx-lm",
        "optimum",
        "sglang",
        "torchao",
        "transformers",
        "vllm",
    }.issubset(packages)
    assert all(isinstance(packages[name], str) for name in packages)
    source = environment["project_source"]
    assert source["kind"] == "filesystem_sha256_manifest_v1"
    assert source["file_count"] == len(source["files"])
    assert len(source["tree_sha256"]) == 64


def test_project_source_provenance_is_content_addressed_and_stable(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "[project]\nname='fixture'\n")
    _write(tmp_path / "uv.lock", "version = 1\n")
    _write(tmp_path / "src/inkling_quant_lab/__init__.py", 'VERSION = "1"\n')
    _write(tmp_path / "src/inkling_quant_lab/data/fixture.jsonl", '{"id":"one"}\n')
    _write(tmp_path / "src/inkling_quant_lab/__pycache__/ignored.pyc", "runtime-cache")

    first = project_source_provenance(tmp_path)
    second = project_source_provenance(tmp_path)

    assert first == second
    assert first["kind"] == "filesystem_sha256_manifest_v1"
    assert first["file_count"] == 4
    assert [item["path"] for item in first["files"]] == [
        "pyproject.toml",
        "src/inkling_quant_lab/__init__.py",
        "src/inkling_quant_lab/data/fixture.jsonl",
        "uv.lock",
    ]
    assert all(len(item["sha256"]) == 64 for item in first["files"])

    _write(tmp_path / "src/inkling_quant_lab/__init__.py", 'VERSION = "2"\n')
    changed = project_source_provenance(tmp_path)
    assert changed["tree_sha256"] != first["tree_sha256"]


def test_project_source_provenance_reports_missing_project(tmp_path: Path) -> None:
    assert project_source_provenance(tmp_path) == {
        "kind": "unavailable",
        "reason": "project source tree, pyproject.toml, or uv.lock is missing",
    }


def test_project_source_provenance_rejects_symlinks(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "[project]\nname='fixture'\n")
    _write(tmp_path / "uv.lock", "version = 1\n")
    _write(tmp_path / "src/inkling_quant_lab/__init__.py", "")
    target = tmp_path / "outside.py"
    _write(target, "SECRET = True\n")
    (tmp_path / "src/inkling_quant_lab/link.py").symlink_to(target)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        project_source_provenance(tmp_path)


def test_project_source_provenance_rejects_symlinked_source_root(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "[project]\nname='fixture'\n")
    _write(tmp_path / "uv.lock", "version = 1\n")
    package = tmp_path / "package"
    _write(package / "__init__.py", "")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/inkling_quant_lab").symlink_to(package, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        project_source_provenance(tmp_path)


def test_project_source_provenance_rejects_sourceless_bytecode(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "[project]\nname='fixture'\n")
    _write(tmp_path / "uv.lock", "version = 1\n")
    _write(tmp_path / "src/inkling_quant_lab/__init__.py", "")
    _write(tmp_path / "src/inkling_quant_lab/plugin.pyc", "importable-bytecode")

    with pytest.raises(ValueError, match="rejects importable sourceless bytecode"):
        project_source_provenance(tmp_path)


def test_source_manifest_file_hash_is_raw_sha256(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "project")
    _write(tmp_path / "uv.lock", "lock")
    _write(tmp_path / "src/inkling_quant_lab/__init__.py", "payload")

    record = project_source_provenance(tmp_path)
    files = {item["path"]: item for item in record["files"]}
    assert (
        files["src/inkling_quant_lab/__init__.py"]["sha256"]
        == hashlib.sha256(b"payload").hexdigest()
    )
