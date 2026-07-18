from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.artifacts import sha256_file
from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.manifests import (
    ArtifactChecksum,
    ModelProvenance,
    RunManifest,
    RunStatus,
    StageRecord,
    StageStatus,
)
from inkling_quant_lab.pipeline import candidate_artifact
from inkling_quant_lab.pipeline.candidate_artifact import (
    load_governed_candidate_artifact,
    load_verified_baseline_descriptor,
)
from inkling_quant_lab.quantization.base import QuantizationManifest

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    _PROJECT_ROOT / "configs/experiments/hf_stories15m_native_int8_source_weight_free_peak.yaml"
)
_NOW = datetime(2026, 7, 16, tzinfo=UTC)
_BASELINE_PATH = "checkpoints/baseline/descriptor.json"
_METADATA_PATH = "checkpoints/candidate/candidate/metadata.json"
_TENSOR_PATH = "checkpoints/candidate/candidate/model.safetensors"
_QUANTIZATION_MANIFEST_PATH = "checkpoints/candidate/quantization_manifest.json"


def _config() -> ExperimentConfig:
    return load_config(_CONFIG_PATH)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _checksum(root: Path, relative: str) -> ArtifactChecksum:
    path = root / relative
    return ArtifactChecksum(
        path=relative,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _stage(name: str, outputs: tuple[ArtifactChecksum, ...]) -> StageRecord:
    return StageRecord(
        name=name,
        status=StageStatus.SUCCESS,
        fingerprint=f"{name}-fingerprint",
        started_at=_NOW,
        completed_at=_NOW,
        outputs=outputs,
    )


def _bundle_sha256(metadata_sha256: str, tensor_sha256: str) -> str:
    digest = hashlib.sha256()
    for name, checksum in (
        ("metadata.json", metadata_sha256),
        ("model.safetensors", tensor_sha256),
    ):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _build_run(root: Path) -> tuple[ExperimentConfig, QuantizationManifest]:
    config = _config()
    source_checksum = "9" * 64
    descriptor = {
        "architecture": "MixtralForCausalLM",
        "capabilities": {
            "is_moe": True,
            "max_context_length": 256,
            "requires_remote_code": False,
            "supported_device_maps": ["single"],
            "supported_dtypes": ["float32"],
            "supports_audio": False,
            "supports_images": False,
            "supports_router_logits": True,
            "supports_text": True,
            "supports_token_level_routes": True,
        },
        "checksum": source_checksum,
        "model_id": config.model.model_id,
        "resolved_class": ("transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM"),
        "revision": config.model.revision,
        "serialized_size_bytes": 145_000_000,
    }
    _write(root / _BASELINE_PATH, _json_bytes(descriptor))

    quantization_manifest = QuantizationManifest(
        backend=config.quantization.backend,
        backend_version="2.13.0",
        method=config.quantization.method,
        source_model_checksum=source_checksum,
        module_precision_map={"model.layers.0.self_attn.q_proj": "int8"},
        excluded_modules=(),
        quantization_parameters={"quantized_engine": "qnnpack"},
        serialized_size_bytes=1,
    )
    tensor_payload = b"fixture-safetensors-payload"
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    for _ in range(16):
        metadata = {
            "model": {
                "architecture": descriptor["architecture"],
                "model_id": descriptor["model_id"],
                "resolved_class": descriptor["resolved_class"],
                "revision": descriptor["revision"],
                "source_checksum": source_checksum,
            },
            "quantization": quantization_manifest.model_dump(mode="json"),
            "reload": {
                "adapter": "hf_causal_lm_source_weight_free_v1",
                "backend": quantization_manifest.backend,
                "format": "safetensors",
                "metadata_file": "metadata.json",
                "tensor_file": "model.safetensors",
                "tensor_sha256": tensor_sha256,
            },
            "schema_version": "2.0",
        }
        metadata_payload = _json_bytes(metadata)
        size = len(metadata_payload) + len(tensor_payload)
        if size == quantization_manifest.serialized_size_bytes:
            break
        quantization_manifest = quantization_manifest.model_copy(
            update={"serialized_size_bytes": size}
        )
    else:
        raise AssertionError("fixture serialized size did not converge")

    _write(root / _METADATA_PATH, metadata_payload)
    _write(root / _TENSOR_PATH, tensor_payload)
    _write(
        root / _QUANTIZATION_MANIFEST_PATH,
        _json_bytes(quantization_manifest.model_dump(mode="json")),
    )

    manifest = RunManifest(
        run_id="source-weight-free-fixture",
        config_hash=config.config_hash(),
        model=ModelProvenance(
            id=config.model.model_id,
            revision=config.model.revision,
            resolved_class=str(descriptor["resolved_class"]),
            architecture=str(descriptor["architecture"]),
            checksum=source_checksum,
        ),
        stages={
            "load_baseline": _stage("load_baseline", (_checksum(root, _BASELINE_PATH),)),
            "quantize": _stage(
                "quantize",
                tuple(
                    _checksum(root, path)
                    for path in (
                        _METADATA_PATH,
                        _TENSOR_PATH,
                        _QUANTIZATION_MANIFEST_PATH,
                    )
                ),
            ),
        },
        status=RunStatus.RUNNING,
        started_at=_NOW,
    )
    _write(root / "manifest.json", _json_bytes(manifest.model_dump(mode="json")))
    return config, quantization_manifest


def _rewrite_manifest(root: Path, update: Any) -> None:
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    update(raw)
    _write(root / "manifest.json", _json_bytes(raw))


def _fake_reload_result(
    root: Path, manifest: QuantizationManifest
) -> tuple[SimpleNamespace, SimpleNamespace]:
    metadata_sha256 = sha256_file(root / _METADATA_PATH)
    tensor_sha256 = sha256_file(root / _TENSOR_PATH)
    raw = json.loads((root / _METADATA_PATH).read_text(encoding="utf-8"))
    model = raw["model"]
    provenance = SimpleNamespace(
        reload_adapter="hf_causal_lm_source_weight_free_v1",
        backend=manifest.backend,
        model_id=model["model_id"],
        revision=model["revision"],
        resolved_class=model["resolved_class"],
        architecture=model["architecture"],
        source_model_checksum=model["source_checksum"],
        metadata_file="metadata.json",
        metadata_sha256=metadata_sha256,
        tensor_file="model.safetensors",
        tensor_sha256=tensor_sha256,
        bundle_sha256=_bundle_sha256(metadata_sha256, tensor_sha256),
        strict_load=True,
        assign=True,
        missing_keys=(),
        unexpected_keys=(),
        meta_tensor_names=(),
        source_weights_loaded=False,
    )
    return SimpleNamespace(manifest=manifest), provenance


def test_verified_baseline_descriptor_is_hash_bound_and_typed(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)

    verified = load_verified_baseline_descriptor(tmp_path, config)

    assert verified.descriptor.model_id == config.model.model_id
    assert verified.descriptor.checksum == "9" * 64
    assert verified.serialized_size_bytes == 145_000_000
    assert verified.artifact_sha256 == sha256_file(tmp_path / _BASELINE_PATH)
    assert verified.artifact_size_bytes == (tmp_path / _BASELINE_PATH).stat().st_size


def test_governed_candidate_verifies_all_artifacts_before_core_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _build_run(tmp_path)
    expected = _fake_reload_result(tmp_path, manifest)
    calls: list[tuple[Path, object, ExperimentConfig, object]] = []

    def fake_reload(
        path: Path, adapter: object, active_config: ExperimentConfig, runtime: object
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        calls.append((path, adapter, active_config, runtime))
        return expected

    monkeypatch.setattr(candidate_artifact, "reload_exported_model_source_weight_free", fake_reload)
    adapter = object()
    runtime = object()

    actual = load_governed_candidate_artifact(tmp_path, config, adapter=adapter, runtime=runtime)

    assert actual == expected
    assert calls == [(tmp_path / "checkpoints/candidate/candidate", adapter, config, runtime)]


def test_candidate_tensor_tampering_fails_before_core_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _build_run(tmp_path)
    (tmp_path / _TENSOR_PATH).write_bytes(b"tampered")
    monkeypatch.setattr(
        candidate_artifact,
        "reload_exported_model_source_weight_free",
        lambda *_args, **_kwargs: pytest.fail("core loader must not run after tampering"),
    )

    with pytest.raises(ArtifactIntegrityError, match=r"Checksum mismatch.*model\.safetensors"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_requires_successful_required_stages(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)
    _rewrite_manifest(tmp_path, lambda raw: raw["stages"].pop("quantize"))

    with pytest.raises(ArtifactIntegrityError, match=r"missing required stage.*quantize"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_non_successful_required_stage(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)

    def mark_running(raw: dict[str, Any]) -> None:
        stage = raw["stages"]["quantize"]
        stage["status"] = "running"
        stage["completed_at"] = None

    _rewrite_manifest(tmp_path, mark_running)

    with pytest.raises(ArtifactIntegrityError, match=r"stage 'quantize' is not successful"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_missing_declared_output_file(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)
    (tmp_path / _METADATA_PATH).unlink()

    with pytest.raises(ArtifactIntegrityError, match="output is missing"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_missing_and_duplicate_declared_outputs(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)

    def remove_metadata(raw: dict[str, Any]) -> None:
        raw["stages"]["quantize"]["outputs"] = [
            output
            for output in raw["stages"]["quantize"]["outputs"]
            if output["path"] != _METADATA_PATH
        ]

    _rewrite_manifest(tmp_path, remove_metadata)
    with pytest.raises(ArtifactIntegrityError, match="output path set does not match"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())

    _build_run(tmp_path)

    def duplicate_metadata(raw: dict[str, Any]) -> None:
        output = next(
            item for item in raw["stages"]["quantize"]["outputs"] if item["path"] == _METADATA_PATH
        )
        raw["stages"]["quantize"]["outputs"].append(dict(output))

    _rewrite_manifest(tmp_path, duplicate_metadata)
    with pytest.raises(ArtifactIntegrityError, match="duplicate output"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_config_hash_mismatch(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)
    _rewrite_manifest(tmp_path, lambda raw: raw.update({"config_hash": "a" * 64}))

    with pytest.raises(ArtifactIntegrityError, match="config hash"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_declared_path_escape(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)

    def escape(raw: dict[str, Any]) -> None:
        raw["stages"]["quantize"]["outputs"][0]["path"] = "../metadata.json"

    _rewrite_manifest(tmp_path, escape)

    with pytest.raises(ArtifactIntegrityError, match="unsafe output path"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_symlinked_declared_output(tmp_path: Path) -> None:
    config, _ = _build_run(tmp_path)
    metadata = tmp_path / _METADATA_PATH
    target = tmp_path / "untrusted-metadata.json"
    metadata.replace(target)
    metadata.symlink_to(target)

    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_embedded_manifest_drift_before_core_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _build_run(tmp_path)
    canonical_path = tmp_path / _QUANTIZATION_MANIFEST_PATH
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["serialized_size_bytes"] += 1
    _write(canonical_path, _json_bytes(canonical))

    def refresh_checksum(raw: dict[str, Any]) -> None:
        output = next(
            item
            for item in raw["stages"]["quantize"]["outputs"]
            if item["path"] == _QUANTIZATION_MANIFEST_PATH
        )
        output["sha256"] = sha256_file(canonical_path)
        output["size_bytes"] = canonical_path.stat().st_size

    _rewrite_manifest(tmp_path, refresh_checksum)
    monkeypatch.setattr(
        candidate_artifact,
        "reload_exported_model_source_weight_free",
        lambda *_args, **_kwargs: pytest.fail("core loader must not run after manifest drift"),
    )

    with pytest.raises(ArtifactIntegrityError, match="embedded quantization manifest"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())


def test_candidate_rejects_reload_provenance_that_loaded_source_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest = _build_run(tmp_path)
    candidate, provenance = _fake_reload_result(tmp_path, manifest)
    provenance.source_weights_loaded = True
    monkeypatch.setattr(
        candidate_artifact,
        "reload_exported_model_source_weight_free",
        lambda *_args, **_kwargs: (candidate, provenance),
    )

    with pytest.raises(ArtifactIntegrityError, match="reload provenance"):
        load_governed_candidate_artifact(tmp_path, config, object(), object())
