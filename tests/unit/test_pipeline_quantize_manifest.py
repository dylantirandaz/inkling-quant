from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.pipeline import runner
from inkling_quant_lab.quantization.base import QuantizationManifest

pytestmark = pytest.mark.unit


def _manifest(*, backend_version: str = "1.0") -> QuantizationManifest:
    return QuantizationManifest(
        backend="candidate_fixture",
        backend_version=backend_version,
        method="fixture",
        source_model_checksum="source-checksum",
        module_precision_map={"linear": "int8"},
        excluded_modules=(),
        serialized_size_bytes=1,
    )


def _candidate_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: QuantizationManifest,
) -> tuple[Any, Any]:
    candidate = SimpleNamespace(manifest=manifest, loaded=object())
    context = SimpleNamespace(
        config=SimpleNamespace(
            benchmark=SimpleNamespace(host_memory_mode="boundary_samples"),
            quantization=SimpleNamespace(
                backend="candidate_fixture",
                method="fixture",
                parameters={},
                export=SimpleNamespace(enabled=True, destination="candidate"),
            ),
        ),
        run_directory=tmp_path,
    )
    components = SimpleNamespace()
    monkeypatch.setattr(runner, "_components", lambda _context: components)
    monkeypatch.setattr(runner, "load_baseline", lambda _config, _components: object())
    monkeypatch.setattr(runner, "_load_policy", lambda _context: object())
    monkeypatch.setattr(runner, "_load_statistics", lambda _context: object())
    monkeypatch.setattr(
        runner,
        "build_candidate",
        lambda _config, _components, _baseline, _policy, _statistics: candidate,
    )
    return context, candidate


def _write_manifest(run_directory: Path, manifest: QuantizationManifest) -> None:
    path = run_directory / "checkpoints/candidate/quantization_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(manifest.model_dump_json(), encoding="utf-8")


def test_candidate_reconstruction_requires_persisted_quantization_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context, _candidate = _candidate_context(monkeypatch, tmp_path, _manifest())

    with pytest.raises(ArtifactIntegrityError, match="manifest is missing or unsafe"):
        runner._candidate(context)


def test_candidate_reconstruction_must_match_persisted_quantization_manifest_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context, _candidate = _candidate_context(monkeypatch, tmp_path, _manifest())
    _write_manifest(tmp_path, _manifest(backend_version="2.0"))
    monkeypatch.setattr(
        runner,
        "evaluate_model",
        lambda *_args: pytest.fail("evaluation ran before candidate integrity validation"),
    )

    with pytest.raises(ArtifactIntegrityError, match="candidate reconstruction differs"):
        runner._execute_stage(
            context,
            SimpleNamespace(name="evaluate_candidate"),
        )


def test_candidate_reconstruction_accepts_exact_persisted_quantization_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    context, candidate = _candidate_context(monkeypatch, tmp_path, manifest)
    _write_manifest(tmp_path, manifest)

    assert runner._candidate(context) is candidate


def test_candidate_reconstruction_accepts_only_export_derived_size_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_manifest = _manifest()
    persisted_manifest = runtime_manifest.model_copy(update={"serialized_size_bytes": 9876})
    context, candidate = _candidate_context(monkeypatch, tmp_path, runtime_manifest)
    _write_manifest(tmp_path, persisted_manifest)

    assert runner._candidate(context) is candidate


def test_cpu_gptq_candidate_is_reloaded_from_governed_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    source = object()
    reloaded = SimpleNamespace(manifest=manifest, loaded=object())
    quantization = SimpleNamespace(
        backend="gptq",
        method="gptq",
        parameters={"device": "cpu"},
        export=SimpleNamespace(enabled=True, destination="candidate"),
    )
    context = SimpleNamespace(
        config=SimpleNamespace(
            benchmark=SimpleNamespace(host_memory_mode="boundary_samples"),
            quantization=quantization,
        ),
        run_directory=tmp_path,
    )
    components = SimpleNamespace()
    _write_manifest(tmp_path, manifest)
    monkeypatch.setattr(runner, "_components", lambda _context: components)
    monkeypatch.setattr(runner, "load_baseline", lambda _config, _components: source)
    monkeypatch.setattr(
        runner,
        "build_candidate",
        lambda *_args: pytest.fail("CPU GPTQ candidate was requantized instead of reloaded"),
    )

    def reload(path: Path, loaded_source: Any, persisted: Any, config: Any) -> Any:
        assert path == tmp_path / "checkpoints/candidate/candidate"
        assert loaded_source is source
        assert persisted == manifest
        assert config is quantization
        return reloaded

    monkeypatch.setattr(runner, "reload_gptqmodel_cpu_export", reload)

    assert runner._candidate(context) is reloaded


def test_quantize_stage_persists_manifest_after_mutating_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    manifest = QuantizationManifest(
        backend="mutating_export_fixture",
        backend_version="1.0",
        method="fixture",
        source_model_checksum="source-checksum",
        module_precision_map={"linear": "int8"},
        excluded_modules=(),
        serialized_size_bytes=1,
    )
    candidate = SimpleNamespace(manifest=manifest)
    persisted_size = 9876

    class MutatingExportQuantizer:
        def export(self, model: Any, destination, config: Any) -> None:
            del config
            assert not (destination.parent / "quantization_manifest.json").exists()
            destination.mkdir()
            model.manifest = model.manifest.model_copy(
                update={"serialized_size_bytes": persisted_size}
            )

    components = SimpleNamespace(quantizer=MutatingExportQuantizer())
    context = SimpleNamespace(
        config=SimpleNamespace(
            quantization=SimpleNamespace(
                export=SimpleNamespace(enabled=True, destination="candidate")
            )
        ),
        store=object(),
    )
    sentinel_result = object()

    monkeypatch.setattr(runner, "_components", lambda _context: components)
    monkeypatch.setattr(runner, "load_baseline", lambda _config, _components: object())
    monkeypatch.setattr(runner, "_load_policy", lambda _context: object())
    monkeypatch.setattr(runner, "_load_statistics", lambda _context: object())
    monkeypatch.setattr(
        runner,
        "build_candidate",
        lambda _config, _components, _baseline, _policy, _statistics: candidate,
    )

    def commit_stage_result(
        store: Any,
        name: str,
        producer: Any,
        *,
        relative_directory: str,
    ) -> object:
        del store, name, relative_directory
        stage = tmp_path / "stage"
        stage.mkdir()
        producer(stage)
        return sentinel_result

    monkeypatch.setattr(runner, "commit_stage_result", commit_stage_result)

    result = runner._quantize_stage(context, "quantize")

    persisted = json.loads(
        (tmp_path / "stage/quantization_manifest.json").read_text(encoding="utf-8")
    )
    assert result is sentinel_result
    assert persisted["serialized_size_bytes"] == persisted_size
