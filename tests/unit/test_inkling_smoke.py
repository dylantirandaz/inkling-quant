from __future__ import annotations

import copy
import ctypes
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling_smoke import (
    B300_CMAKE_ARCHITECTURE,
    B300_COMPUTE_CAPABILITY,
    EXPECTED_PROJECTOR_BYTES,
    EXPECTED_PROJECTOR_PATH,
    EXPECTED_PROJECTOR_SHA256,
    EXPECTED_Q3_INVENTORY_SHA256,
    EXPECTED_Q3_SHARD_COUNT,
    EXPECTED_Q3_TOTAL_BYTES,
    EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256,
    INSTRUMENTATION_PATCH_RELATIVE_PATH,
    INSTRUMENTATION_PATCH_SHA256,
    INSTRUMENTATION_SCHEMA_VERSION,
    LLAMA_SERVER_AUDIT_LOG_VERBOSITY,
    PINNED_CUDA_IMAGE,
    PINNED_CUDA_IMAGE_DIGEST,
    PINNED_CUDA_PLATFORM,
    SMOKE_CONFIG_RELATIVE_PATH,
    SUBJECT_CONFIG_HASH,
    SUBJECT_CONTROL_PLANE_SHA256,
    SUBJECT_MMPROJ_CALL_ID,
    SUBJECT_QUANTIZE_CALL_ID,
    SUBJECT_RUN_ID,
    SUBJECT_VERIFY_CALL_ID,
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
    CudaDriverGpuEvidence,
    CudaDriverPeerLinkEvidence,
    CudaDriverPeerTopologyEvidence,
    InklingSmokeConfig,
    InklingVerifiedExportReference,
    combine_gpu_identity,
    enumerate_cuda_driver_gpus,
    enumerate_cuda_driver_peer_topology,
    load_inkling_smoke_config,
    load_verified_export_reference,
    parse_artifact_load_evidence,
    parse_backend_audit_evidence,
    parse_cuda_driver_linkage,
    parse_loader_offload_evidence,
    parse_nvidia_smi_csv,
    parse_nvidia_smi_monitor_csv,
    parse_raw_logit_audit_evidence,
    parse_server_completion,
    redacted_smoke_config_record,
    verified_export_reference_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH
CONFIG_PATH = PROJECT_ROOT / SMOKE_CONFIG_RELATIVE_PATH
GPU_UUID_0 = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU_UUID_1 = "GPU-ffffffff-1111-2222-3333-444444444444"
GPU_UUID_OTHER = "GPU-01234567-89ab-cdef-0123-456789abcdef"
GPU_UUID_0_UPPER_HEX = "GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
GPU_UUID_1_UPPER_HEX = "GPU-FFFFFFFF-1111-2222-3333-444444444444"


class _FakeCudaFunction:
    def __init__(self, callback: Callable[..., int]) -> None:
        self._callback = callback
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self, *arguments: object) -> int:
        return self._callback(*arguments)


class _FakeCudaDriver:
    def __init__(self) -> None:
        uuid_bytes = (
            UUID(GPU_UUID_0.removeprefix("GPU-")).bytes,
            UUID(GPU_UUID_1.removeprefix("GPU-")).bytes,
        )
        self.peer_access_values: dict[tuple[int, int], int] = {
            (0, 1): 1,
            (1, 0): 1,
        }
        self.peer_access_error_codes: dict[tuple[int, int], int] = {}
        self.peer_attribute_values: dict[tuple[int, int, int], int] = {
            (source, destination, attribute): value
            for source, destination in ((0, 1), (1, 0))
            for attribute, value in (
                (1, 0),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 0),
            )
        }
        self.peer_attribute_error_codes: dict[tuple[int, int, int], int] = {}
        self.peer_access_calls: list[tuple[int, int]] = []
        self.peer_attribute_calls: list[tuple[int, int, int]] = []

        def set_int(pointer: Any, value: int) -> None:
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = value

        def device_ordinal(device: object) -> int:
            if not isinstance(device, ctypes.c_int):
                raise AssertionError("fake CUDA device must be a c_int")
            return int(device.value)

        def device_get(pointer: object, ordinal: object) -> int:
            if not isinstance(ordinal, int):
                raise AssertionError("CUDA ordinal must be an int")
            set_int(pointer, ordinal)
            return 0

        def device_uuid(pointer: Any, device: object) -> int:
            target = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte * 16)).contents
            for index, value in enumerate(uuid_bytes[device_ordinal(device)]):
                target[index] = value
            return 0

        def device_name(buffer: object, _length: object, _device: object) -> int:
            if not isinstance(buffer, ctypes.Array):
                raise AssertionError("CUDA name target must be a ctypes array")
            buffer.value = b"NVIDIA B300"
            return 0

        def device_attribute(pointer: object, attribute: object, _device: object) -> int:
            if attribute == 75:
                set_int(pointer, 10)
            elif attribute == 76:
                set_int(pointer, 3)
            else:
                raise AssertionError(f"unexpected CUDA attribute {attribute}")
            return 0

        def total_memory(pointer: Any, _device: object) -> int:
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t)).contents.value = 200 * 1024**3
            return 0

        def pci_bus_id(buffer: object, _length: object, device: object) -> int:
            if device_ordinal(device) == 0:
                if not isinstance(buffer, ctypes.Array):
                    raise AssertionError("CUDA PCI target must be a ctypes array")
                buffer.value = b"0000:45:00.0"
                return 0
            return 999

        def can_access_peer(
            pointer: object,
            source_device: object,
            destination_device: object,
        ) -> int:
            pair = (
                device_ordinal(source_device),
                device_ordinal(destination_device),
            )
            self.peer_access_calls.append(pair)
            error_code = self.peer_access_error_codes.get(pair, 0)
            if error_code != 0:
                return error_code
            set_int(pointer, self.peer_access_values[pair])
            return 0

        def peer_attribute(
            pointer: object,
            attribute: object,
            source_device: object,
            destination_device: object,
        ) -> int:
            if not isinstance(attribute, int):
                raise AssertionError("CUDA peer attribute must be an int")
            key = (
                device_ordinal(source_device),
                device_ordinal(destination_device),
                attribute,
            )
            self.peer_attribute_calls.append(key)
            error_code = self.peer_attribute_error_codes.get(key, 0)
            if error_code != 0:
                return error_code
            set_int(pointer, self.peer_attribute_values[key])
            return 0

        def driver_get_version(pointer: Any) -> int:
            set_int(pointer, 13_010)
            return 0

        def device_get_count(pointer: Any) -> int:
            set_int(pointer, 2)
            return 0

        self.cuInit = _FakeCudaFunction(lambda _flags: 0)
        self.cuDriverGetVersion = _FakeCudaFunction(driver_get_version)
        self.cuDeviceGetCount = _FakeCudaFunction(device_get_count)
        self.cuDeviceGet = _FakeCudaFunction(device_get)
        self.cuDeviceGetUuid_v2 = _FakeCudaFunction(device_uuid)
        self.cuDeviceGetName = _FakeCudaFunction(device_name)
        self.cuDeviceGetAttribute = _FakeCudaFunction(device_attribute)
        self.cuDeviceTotalMem_v2 = _FakeCudaFunction(total_memory)
        self.cuDeviceGetPCIBusId = _FakeCudaFunction(pci_bus_id)
        self.cuDeviceCanAccessPeer = _FakeCudaFunction(can_access_peer)
        self.cuDeviceGetP2PAttribute = _FakeCudaFunction(peer_attribute)


def _reference_mapping() -> dict[str, object]:
    raw = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AssertionError("checked reference must be a JSON object")
    return raw


def _rehash_reference(raw: dict[str, object]) -> dict[str, object]:
    raw["reference_sha256"] = verified_export_reference_sha256(raw)
    return raw


def _config_mapping() -> dict[str, object]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AssertionError("checked smoke config must be a mapping")
    return raw


def _top_probability(token_id: int, logprob: float) -> dict[str, object]:
    token = f"token-{token_id}"
    return {
        "id": token_id,
        "token": token,
        "bytes": list(token.encode("utf-8")),
        "logprob": logprob,
    }


def _probability(token_id: int, logprob: float) -> dict[str, object]:
    top = [
        _top_probability(token_id, logprob),
        _top_probability(token_id + 10, -2.0),
        _top_probability(token_id + 20, -3.0),
        _top_probability(token_id + 30, -4.0),
        _top_probability(token_id + 40, -5.0),
    ]
    return {
        **_top_probability(token_id, logprob),
        "top_logprobs": top,
    }


def _completion_payload() -> dict[str, object]:
    return {
        "content": "redacted after validation",
        "tokens": [17, 19],
        "tokens_predicted": 2,
        "completion_probabilities": [
            _probability(17, -0.25),
            _probability(19, -0.5),
        ],
    }


def test_checked_verified_export_reference_is_exact_and_canonical() -> None:
    reference = load_verified_export_reference(REFERENCE_PATH)

    assert REFERENCE_PATH.read_bytes() == (reference.canonical_json() + "\n").encode()
    assert reference.reference_sha256 == EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256
    assert reference.reference_sha256 == reference.computed_reference_sha256()
    assert reference.subject_run_id == SUBJECT_RUN_ID
    assert reference.subject_config_hash == SUBJECT_CONFIG_HASH
    assert reference.subject_control_plane_sha256 == SUBJECT_CONTROL_PLANE_SHA256
    assert reference.verify_call_id == SUBJECT_VERIFY_CALL_ID
    assert reference.quantize_call_id == SUBJECT_QUANTIZE_CALL_ID
    assert reference.mmproj_call_id == SUBJECT_MMPROJ_CALL_ID
    assert reference.q3_shard_count == EXPECTED_Q3_SHARD_COUNT
    assert len(reference.q3_shards) == EXPECTED_Q3_SHARD_COUNT
    assert reference.q3_total_bytes == EXPECTED_Q3_TOTAL_BYTES
    assert sum(shard.size_bytes for shard in reference.q3_shards) == EXPECTED_Q3_TOTAL_BYTES
    assert reference.q3_inventory_sha256 == EXPECTED_Q3_INVENTORY_SHA256
    assert reference.projector.path == EXPECTED_PROJECTOR_PATH
    assert reference.projector.sha256 == EXPECTED_PROJECTOR_SHA256
    assert reference.projector.size_bytes == EXPECTED_PROJECTOR_BYTES
    assert reference.mtp == "omitted_unsupported"
    assert reference.quality_measured is False
    assert reference.deployment_benchmark_measured is False


def test_reference_rejects_semantic_tampering_even_after_rehash() -> None:
    raw = _reference_mapping()
    shards = raw["q3_shards"]
    assert isinstance(shards, list)
    first = shards[0]
    assert isinstance(first, dict)
    first["sha256"] = "f" * 64

    with pytest.raises(ValidationError, match="inventory differs"):
        InklingVerifiedExportReference.model_validate(_rehash_reference(raw))

    raw = _reference_mapping()
    shards = raw["q3_shards"]
    assert isinstance(shards, list)
    shards[0], shards[1] = shards[1], shards[0]
    with pytest.raises(ValidationError, match="ordered 49-file set"):
        InklingVerifiedExportReference.model_validate(_rehash_reference(raw))


def test_reference_rejects_identity_path_and_self_hash_tampering() -> None:
    raw = _reference_mapping()
    raw["subject_config_hash"] = "f" * 64
    with pytest.raises(ValidationError, match="subject_config_hash"):
        InklingVerifiedExportReference.model_validate(_rehash_reference(raw))

    raw = _reference_mapping()
    projector = raw["projector"]
    assert isinstance(projector, dict)
    projector["path"] = "../mmproj.gguf"
    with pytest.raises(ValidationError, match="traversal"):
        InklingVerifiedExportReference.model_validate(_rehash_reference(raw))

    raw = _reference_mapping()
    raw["reference_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="self-hash"):
        InklingVerifiedExportReference.model_validate(raw)


def test_reference_loader_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference_mapping(), indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="canonical JSON"):
        load_verified_export_reference(path)


def test_checked_smoke_config_binds_runtime_hardware_probes_and_claims() -> None:
    config = load_inkling_smoke_config(CONFIG_PATH)

    assert config.verified_export_reference_sha256 == (EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256)
    assert config.runtime.image.image == PINNED_CUDA_IMAGE
    assert config.runtime.image.digest == PINNED_CUDA_IMAGE_DIGEST
    assert config.runtime.image.platform == PINNED_CUDA_PLATFORM
    assert config.runtime.instrumentation_schema_version == INSTRUMENTATION_SCHEMA_VERSION
    assert config.runtime.instrumentation_patch_path == INSTRUMENTATION_PATCH_RELATIVE_PATH
    assert config.runtime.instrumentation_patch_sha256 == INSTRUMENTATION_PATCH_SHA256
    assert config.runtime.log_verbosity == LLAMA_SERVER_AUDIT_LOG_VERBOSITY == 4
    assert f"CMAKE_CUDA_ARCHITECTURES={B300_CMAKE_ARCHITECTURE}" in (
        config.runtime.cmake_definitions
    )
    assert (
        "CMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/iql-cuda-driver-link"
        in config.runtime.cmake_definitions
    )
    assert "LLAMA_BUILD_UI=OFF" in config.runtime.cmake_definitions
    assert "LLAMA_USE_PREBUILT_UI=OFF" in config.runtime.cmake_definitions
    assert config.resources.gpu_type == "B300"
    assert config.resources.gpu_count == 2
    assert config.resources.compute_capability == B300_COMPUTE_CAPABILITY
    assert tuple(probe.modality for probe in config.probes) == ("text", "image", "audio")
    assert all(probe.temperature == 0.0 for probe in config.probes)
    assert all(probe.seed == 42 and probe.trials == 2 for probe in config.probes)
    assert all(probe.post_sampling_probs is False for probe in config.probes)
    assert config.evidence.record_prompt_text is False
    assert config.evidence.record_output_text is False
    assert config.evidence.record_token_ids is True
    assert config.claims.mtp_included is False
    assert config.claims.quality_measured is False
    assert config.claims.benchmark_measured is False
    assert len(config.config_hash()) == 64


def test_cuda_driver_linkage_parser_requires_one_non_stub_absolute_path() -> None:
    output = "libcuda.so.1 => /usr/local/nvidia/lib64/libcuda.so.1 (0x00007f1234000000)\n"

    assert parse_cuda_driver_linkage(output) == ("/usr/local/nvidia/lib64/libcuda.so.1")

    with pytest.raises(ConfigurationError, match="driver stub"):
        parse_cuda_driver_linkage(
            "libcuda.so.1 => /usr/local/cuda/lib64/stubs/libcuda.so.1 (0x00007f1234000000)\n"
        )
    with pytest.raises(ConfigurationError, match="driver stub"):
        parse_cuda_driver_linkage(
            "libcuda.so.1 => "
            "/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs/libcuda.so.1 "
            "(0x00007f1234000000)\n"
        )
    with pytest.raises(ConfigurationError, match="driver stub"):
        parse_cuda_driver_linkage(
            "libcuda.so.1 => /opt/iql-cuda-driver-link/libcuda.so.1 (0x00007f1234000000)\n"
        )
    with pytest.raises(ConfigurationError, match="exactly one"):
        parse_cuda_driver_linkage("libcuda.so.1 => not found\n")


def test_resolved_smoke_config_record_redacts_all_prompt_text() -> None:
    config = load_inkling_smoke_config(CONFIG_PATH)
    record = redacted_smoke_config_record(config)
    encoded = json.dumps(record, sort_keys=True)

    assert record["smoke_config_hash"] == config.config_hash()
    assert record["prompt_text_recorded"] is False
    assert all(probe.prompt not in encoded for probe in config.probes)
    resolved = record["resolved_config"]
    assert isinstance(resolved, dict)
    probes = resolved["probes"]
    assert isinstance(probes, list)
    assert all("prompt" not in probe for probe in probes)
    assert [probe["prompt_sha256"] for probe in probes] == [
        probe.prompt_sha256 for probe in config.probes
    ]


@pytest.mark.parametrize(
    ("section", "field", "bad_value", "match"),
    (
        ("runtime.image", "digest", "sha256:" + "0" * 64, "digest"),
        ("runtime.image", "platform", "linux/arm64", "platform"),
        ("runtime", "cmake_definitions", ["GGML_CUDA=OFF"], "CMake"),
        ("runtime", "log_verbosity", 3, "log_verbosity"),
        ("runtime", "log_verbosity", "4", "log_verbosity"),
        ("resources", "gpu_count", 1, "gpu_count"),
        ("resources", "compute_capability", "9.0", "compute_capability"),
        ("evidence", "record_output_text", True, "record_output_text"),
        ("claims", "quality_measured", True, "quality_measured"),
    ),
)
def test_smoke_config_rejects_contract_drift(
    section: str,
    field: str,
    bad_value: object,
    match: str,
) -> None:
    raw = _config_mapping()
    target: dict[str, object] = raw
    for part in section.split("."):
        child = target[part]
        assert isinstance(child, dict)
        target = child
    target[field] = bad_value

    with pytest.raises(ValidationError, match=match):
        InklingSmokeConfig.model_validate(raw)


def test_smoke_config_requires_explicit_log_verbosity() -> None:
    raw = _config_mapping()
    runtime = raw["runtime"]
    assert isinstance(runtime, dict)
    runtime.pop("log_verbosity")

    with pytest.raises(ValidationError, match="log_verbosity"):
        InklingSmokeConfig.model_validate(raw)


def test_smoke_config_rejects_probe_order_fixture_and_prompt_drift() -> None:
    raw = _config_mapping()
    probes = raw["probes"]
    assert isinstance(probes, list)
    probes[0], probes[1] = probes[1], probes[0]
    with pytest.raises(ValidationError, match="ordered text/image/audio"):
        InklingSmokeConfig.model_validate(raw)

    raw = _config_mapping()
    probes = raw["probes"]
    assert isinstance(probes, list)
    image_probe = probes[1]
    assert isinstance(image_probe, dict)
    image_probe["fixture"] = "none"
    with pytest.raises(ValidationError, match="fixture"):
        InklingSmokeConfig.model_validate(raw)

    raw = _config_mapping()
    probes = raw["probes"]
    assert isinstance(probes, list)
    text_probe = probes[0]
    assert isinstance(text_probe, dict)
    text_probe["prompt"] = "changed prompt"
    with pytest.raises(ValidationError, match="prompt SHA-256"):
        InklingSmokeConfig.model_validate(raw)


def test_server_completion_parser_returns_only_redacted_finite_evidence() -> None:
    evidence = parse_server_completion(_completion_payload(), vocab_size=1000, expected_n_probs=5)

    assert evidence.token_ids == (17, 19)
    assert evidence.tokens_predicted == 2
    assert evidence.minimum_logprob == -0.5
    assert evidence.maximum_logprob == -0.25
    assert evidence.all_returned_logprobs_finite is True
    assert evidence.prompt_text_recorded is False
    assert evidence.output_text_recorded is False
    assert "content" not in evidence.model_dump()


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (lambda value: value.update(tokens_predicted=1), "tokens_predicted"),
        (lambda value: value.update(tokens=[True, 19]), "integer"),
        (
            lambda value: value["completion_probabilities"][0].update(logprob=float("nan")),
            "finite",
        ),
        (
            lambda value: value["completion_probabilities"][0].update(id=18),
            "does not match",
        ),
        (
            lambda value: value["completion_probabilities"][0]["top_logprobs"].pop(),
            "n_probs",
        ),
        (
            lambda value: value["completion_probabilities"][0]["top_logprobs"].__setitem__(
                0, _top_probability(18, -0.1)
            ),
            "top-ranked",
        ),
    ),
)
def test_server_completion_parser_rejects_invalid_evidence(
    mutator: object,
    match: str,
) -> None:
    payload = copy.deepcopy(_completion_payload())
    assert callable(mutator)
    mutator(payload)

    with pytest.raises(ValueError, match=match):
        parse_server_completion(payload, vocab_size=1000, expected_n_probs=5)


def test_loader_parser_requires_two_cuda_devices_and_full_offload() -> None:
    log = "\n".join(
        (
            "ggml_cuda_init: found 2 CUDA devices (Total VRAM: 376 GiB):",
            "load_tensors: offloading output layer to GPU",
            "load_tensors: offloaded 128/128 layers to GPU",
        )
    )

    evidence = parse_loader_offload_evidence(log)

    assert evidence.cuda_device_count == 2
    assert evidence.offloaded_layers == evidence.offloadable_layers == 128
    assert evidence.output_layer_offloaded is True
    assert evidence.all_offloadable_layers_on_gpu is True


@pytest.mark.parametrize(
    ("log", "match"),
    (
        (
            "warning: no usable GPU found, --gpu-layers option will be ignored",
            "no usable GPU",
        ),
        (
            "ggml_cuda_init: found 1 CUDA devices\n"
            "load_tensors: offloading output layer to GPU\n"
            "load_tensors: offloaded 128/128 layers to GPU",
            "device count",
        ),
        (
            "ggml_cuda_init: found 2 CUDA devices\n"
            "load_tensors: offloading output layer to GPU\n"
            "load_tensors: offloaded 127/128 layers to GPU",
            "every offloadable layer",
        ),
        (
            "ggml_cuda_init: found 2 CUDA devices\nload_tensors: offloaded 128/128 layers to GPU",
            "output layer",
        ),
        (
            "ggml_cuda_init: found 2 CUDA devices\n"
            "ggml_cuda_init: found 1 CUDA devices\n"
            "load_tensors: offloading output layer to GPU\n"
            "load_tensors: offloaded 128/128 layers to GPU",
            "conflicting CUDA device-count evidence",
        ),
    ),
)
def test_loader_parser_rejects_cpu_or_partial_offload(log: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_loader_offload_evidence(log)


def test_raw_logit_audit_requires_every_generated_vector_to_be_finite() -> None:
    log = "\n".join(
        f"IQL_SMOKE_RAW_LOGITS_V1 task_id={index} slot_id=0 "
        f"completion_index={index + 1} batch_index=0 count=1000 finite=1000 "
        "nan=0 pos_inf=0 neg_inf=0"
        for index in range(4)
    )

    evidence = parse_raw_logit_audit_evidence(
        log,
        expected_generated_token_vectors=4,
        vocab_size=1000,
    )

    assert evidence.observed_generated_token_vectors == 4
    assert evidence.all_rows_complete is True
    assert evidence.all_values_finite is True


@pytest.mark.parametrize(
    ("log", "expected", "match"),
    (
        (
            "IQL_SMOKE_RAW_LOGITS_V1 task_id=0 slot_id=0 completion_index=1 "
            "batch_index=0 count=1000 finite=999 nan=1 pos_inf=0 neg_inf=0",
            1,
            "non-finite",
        ),
        (
            "IQL_SMOKE_RAW_LOGITS_V1 task_id=0 slot_id=0 completion_index=1 "
            "batch_index=0 count=999 finite=999 nan=0 pos_inf=0 neg_inf=0",
            1,
            "vocabulary",
        ),
        (
            "IQL_SMOKE_RAW_LOGITS_V1 task_id=0 slot_id=0 completion_index=1 "
            "batch_index=0 count=1000 finite=1000 nan=0 pos_inf=0 neg_inf=0",
            2,
            "vector count",
        ),
    ),
)
def test_raw_logit_audit_rejects_incomplete_or_nonfinite_rows(
    log: str,
    expected: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_raw_logit_audit_evidence(
            log,
            expected_generated_token_vectors=expected,
            vocab_size=1000,
        )


def test_backend_audit_rejects_any_cpu_or_unassigned_operation() -> None:
    good = "\n".join(
        (
            "IQL_SMOKE_BACKEND_IDENTITY_V1 graph_uid=1 backend_index=0 "
            "backend_name=CUDA0 device_name=CUDA0 device_type=gpu compute=7",
            "IQL_SMOKE_BACKEND_IDENTITY_V1 graph_uid=1 backend_index=1 "
            "backend_name=CUDA1 device_name=CUDA1 device_type=gpu compute=5",
            "IQL_SMOKE_BACKEND_GRAPH_V1 graph_uid=1 "
            "phase=post_assignment_pre_split scope=non_view_compute "
            "compute=12 gpu=12 cpu=0 accel=0 other=0 unassigned=0",
            "IQL_SMOKE_BACKEND_IDENTITY_V1 graph_uid=2 backend_index=0 "
            "backend_name=CUDA0 device_name=CUDA0 device_type=gpu compute=8",
            "IQL_SMOKE_BACKEND_GRAPH_V1 graph_uid=2 "
            "phase=post_assignment_pre_split scope=non_view_compute "
            "compute=8 gpu=8 cpu=0 accel=0 other=0 unassigned=0",
        )
    )
    evidence = parse_backend_audit_evidence(good)
    assert evidence.observed_graphs == 2
    assert evidence.compute_operations == evidence.gpu_operations == 20
    assert evidence.accelerator_operations == 0
    assert len(evidence.identities) == 3
    assert evidence.no_cpu_model_graph_fallback is True

    bad = "\n".join(
        (
            "IQL_SMOKE_BACKEND_IDENTITY_V1 graph_uid=1 backend_index=0 "
            "backend_name=CUDA0 device_name=CUDA0 device_type=gpu compute=11",
            "IQL_SMOKE_BACKEND_IDENTITY_V1 graph_uid=1 backend_index=2 "
            "backend_name=CPU device_name=CPU device_type=cpu compute=1",
            "IQL_SMOKE_CPU_NODE_V1 graph_uid=1 ordinal=7 op=MUL_MAT name=ffn",
            "IQL_SMOKE_BACKEND_GRAPH_V1 graph_uid=1 "
            "phase=post_assignment_pre_split scope=non_view_compute "
            "compute=12 gpu=11 cpu=1 accel=0 other=0 unassigned=0",
        )
    )
    with pytest.raises(ValueError, match="CPU"):
        parse_backend_audit_evidence(bad)


def test_artifact_load_parser_binds_all_49_shards_and_projector() -> None:
    first = "/subject/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf"
    projector = "/subject/mmproj/mmproj-BF16.gguf"
    log = "\n".join(
        (
            f"llama_model_loader: loaded meta data with 57 key-value pairs from {first}",
            "llama_model_loader: additional 48 GGUFs metadata loaded.",
            f"srv load_model: loaded multimodal model, '{projector}'",
            "IQL_SMOKE_TEXT_SHARDS_V1 expected=49 opened=49 contexts=49 tensors=1200",
            "IQL_SMOKE_TEXT_LOAD_V1 opened=49 accounted=49 tensors=1200 "
            "bytes=451035400288 size_done=451035400288 size_data=451035400288 mmap=1",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=vision projector=inkling "
            "tensors=10 bytes=90000000",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=audio projector=inkling "
            "tensors=12 bytes=93264288",
            "IQL_SMOKE_PROJECTOR_READY_V1 opened=1 vision=1 audio=1 "
            "vision_type=inkling audio_type=inkling n_embd=7168",
        )
    )

    evidence = parse_artifact_load_evidence(log)

    assert evidence.first_shard_path == first
    assert evidence.total_shards_loaded == 49
    assert evidence.projector_path == projector
    assert evidence.text_shards.opened == 49
    assert evidence.text_load.accounted == 49
    assert tuple(row.modality for row in evidence.projector_tensors) == ("vision", "audio")
    assert evidence.all_expected_artifacts_loaded is True


@pytest.mark.parametrize(
    ("old", "new", "match"),
    (
        ("additional 48", "additional 47", "additional shard"),
        ("00001-of-00049", "00002-of-00049", "first shard"),
        ("mmproj-BF16.gguf", "other.gguf", "projector"),
    ),
)
def test_artifact_load_parser_rejects_wrong_artifact_set(
    old: str,
    new: str,
    match: str,
) -> None:
    log = "\n".join(
        (
            "llama_model_loader: loaded meta data with 57 key-value pairs from "
            "/subject/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf",
            "llama_model_loader: additional 48 GGUFs metadata loaded.",
            "loaded multimodal model, '/subject/mmproj/mmproj-BF16.gguf'",
            "IQL_SMOKE_TEXT_SHARDS_V1 expected=49 opened=49 contexts=49 tensors=1200",
            "IQL_SMOKE_TEXT_LOAD_V1 opened=49 accounted=49 tensors=1200 "
            "bytes=451035400288 size_done=451035400288 size_data=451035400288 mmap=1",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=vision projector=inkling "
            "tensors=10 bytes=90000000",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=audio projector=inkling "
            "tensors=12 bytes=93264288",
            "IQL_SMOKE_PROJECTOR_READY_V1 opened=1 vision=1 audio=1 "
            "vision_type=inkling audio_type=inkling n_embd=7168",
        )
    ).replace(old, new)
    with pytest.raises(ValueError, match=match):
        parse_artifact_load_evidence(log)


def _cuda_gpu(
    ordinal: int,
    uuid: str,
    *,
    pci_bus_id: str | None,
    pci_error_code: int | None,
) -> CudaDriverGpuEvidence:
    return CudaDriverGpuEvidence(
        cuda_ordinal=ordinal,
        uuid=uuid,
        cuda_driver_name="NVIDIA B300",
        cuda_compute_capability=B300_COMPUTE_CAPABILITY,
        cuda_total_memory_bytes=200 * 1024**3,
        pci_bus_id=pci_bus_id,
        pci_bus_id_status="available" if pci_bus_id is not None else "unavailable",
        pci_bus_id_source="cuda_driver_api",
        pci_bus_id_error_code=pci_error_code,
    )


def _cuda_peer_link(
    source_ordinal: int,
    destination_ordinal: int,
    *,
    source_uuid: str | None = None,
    destination_uuid: str | None = None,
    can_access_peer: bool = True,
    access_supported: bool = True,
) -> CudaDriverPeerLinkEvidence:
    uuids = (GPU_UUID_0, GPU_UUID_1)
    return CudaDriverPeerLinkEvidence(
        source_cuda_ordinal=source_ordinal,
        destination_cuda_ordinal=destination_ordinal,
        source_uuid=uuids[source_ordinal] if source_uuid is None else source_uuid,
        destination_uuid=(
            uuids[destination_ordinal] if destination_uuid is None else destination_uuid
        ),
        can_access_peer=can_access_peer,
        performance_rank=0,
        access_supported=access_supported,
        native_atomic_supported=True,
        cuda_array_access_supported=True,
        only_partial_native_atomic_supported=False,
    )


def _nvidia_smi_row(
    uuid: str,
    *,
    name: str = "NVIDIA B300",
    compute_capability: str = B300_COMPUTE_CAPABILITY,
) -> str:
    return f"{uuid}, {name}, 191456, 590.44.01, {compute_capability}"


def test_cuda_driver_enumeration_accepts_optional_pci_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    driver_api_version, evidence = enumerate_cuda_driver_gpus(
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    )

    assert driver_api_version == 13_010
    assert tuple(gpu.cuda_ordinal for gpu in evidence) == (0, 1)
    assert tuple(gpu.uuid for gpu in evidence) == (GPU_UUID_0, GPU_UUID_1)
    assert all(gpu.cuda_driver_name == "NVIDIA B300" for gpu in evidence)
    assert all(gpu.cuda_compute_capability == B300_COMPUTE_CAPABILITY for gpu in evidence)
    assert evidence[0].pci_bus_id == "00000000:45:00.0"
    assert evidence[0].pci_bus_id_status == "available"
    assert evidence[0].pci_bus_id_error_code is None
    assert evidence[1].pci_bus_id is None
    assert evidence[1].pci_bus_id_status == "unavailable"
    assert evidence[1].pci_bus_id_error_code == 999


def test_cuda_driver_enumeration_requires_uuid_v2_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    del fake_driver.cuDeviceGetUuid_v2
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match="required function cuDeviceGetUuid_v2"):
        enumerate_cuda_driver_gpus("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_topology_enumerates_both_ordered_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    evidence = enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")

    assert evidence.topology_protocol == "cuda-driver-p2p-attributes-v1"
    assert evidence.cuda_driver_api_version == 13_010
    assert tuple(
        (link.source_cuda_ordinal, link.destination_cuda_ordinal) for link in evidence.links
    ) == ((0, 1), (1, 0))
    assert tuple((link.source_uuid, link.destination_uuid) for link in evidence.links) == (
        (GPU_UUID_0, GPU_UUID_1),
        (GPU_UUID_1, GPU_UUID_0),
    )
    assert all(link.can_access_peer and link.access_supported for link in evidence.links)
    assert all(link.performance_rank == 0 for link in evidence.links)
    assert all(link.native_atomic_supported for link in evidence.links)
    assert all(link.cuda_array_access_supported for link in evidence.links)
    assert all(not link.only_partial_native_atomic_supported for link in evidence.links)
    assert fake_driver.peer_access_calls == [(0, 1), (1, 0)]
    assert fake_driver.peer_attribute_calls == [
        (source, destination, attribute)
        for source, destination in ((0, 1), (1, 0))
        for attribute in range(1, 6)
    ]
    assert fake_driver.cuDeviceCanAccessPeer.argtypes == [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    assert fake_driver.cuDeviceGetP2PAttribute.argtypes == [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]


@pytest.mark.parametrize(
    "symbol",
    ("cuDeviceCanAccessPeer", "cuDeviceGetP2PAttribute"),
)
def test_cuda_driver_peer_topology_requires_p2p_symbols(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    fake_driver = _FakeCudaDriver()
    delattr(fake_driver, symbol)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match=rf"required function {symbol}"):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_topology_rejects_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    fake_driver.peer_attribute_error_codes[(0, 1, 3)] = 701
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match=r"native-atomic-supported query failed.*701"):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_topology_rejects_peer_access_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    fake_driver.peer_access_error_codes[(0, 1)] = 702
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match=r"peer 0->1 access query failed.*702"):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


@pytest.mark.parametrize(
    ("attribute", "match"),
    (
        (2, "access-supported result must be zero or one"),
        (3, "native-atomic-supported result must be zero or one"),
        (4, "CUDA-array-access-supported result must be zero or one"),
        (5, "only-partial-native-atomic-supported result must be zero or one"),
    ),
)
def test_cuda_driver_peer_topology_rejects_nonbinary_attributes(
    monkeypatch: pytest.MonkeyPatch,
    attribute: int,
    match: str,
) -> None:
    fake_driver = _FakeCudaDriver()
    fake_driver.peer_attribute_values[(0, 1, attribute)] = 2
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match=match):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_topology_rejects_nonbinary_can_access_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    fake_driver.peer_access_values[(0, 1)] = 2
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match="peer 0->1 access result must be zero or one"):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_topology_rejects_negative_performance_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    fake_driver.peer_attribute_values[(0, 1, 1)] = -1
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match="performance rank must be non-negative"):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_topology_rejects_access_query_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    fake_driver.peer_access_values[(0, 1)] = 0
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match="peer 0->1 access queries disagree"):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_topology_rejects_full_and_partial_atomic_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_driver = _FakeCudaDriver()
    fake_driver.peer_attribute_values[(0, 1, 5)] = 1
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: fake_driver)

    with pytest.raises(RuntimeError, match="atomic support cannot be both full and partial"):
        enumerate_cuda_driver_peer_topology("/usr/lib/x86_64-linux-gnu/libcuda.so.1")


def test_cuda_driver_peer_link_evidence_is_strict_and_consistent() -> None:
    raw = _cuda_peer_link(0, 1).model_dump()
    raw["can_access_peer"] = 1
    with pytest.raises(ValidationError, match="valid boolean"):
        CudaDriverPeerLinkEvidence.model_validate(raw)

    raw = _cuda_peer_link(0, 1).model_dump()
    raw["access_supported"] = False
    with pytest.raises(ValidationError, match="peer access queries disagree"):
        CudaDriverPeerLinkEvidence.model_validate(raw)

    raw = _cuda_peer_link(0, 1).model_dump()
    raw["destination_cuda_ordinal"] = 0
    with pytest.raises(ValidationError, match="distinct ordinals"):
        CudaDriverPeerLinkEvidence.model_validate(raw)

    raw = _cuda_peer_link(0, 1).model_dump()
    raw["only_partial_native_atomic_supported"] = True
    with pytest.raises(ValidationError, match="atomic support cannot be both full and partial"):
        CudaDriverPeerLinkEvidence.model_validate(raw)


def test_cuda_driver_peer_topology_evidence_requires_exact_reversed_pair() -> None:
    forward = _cuda_peer_link(0, 1)
    reverse = _cuda_peer_link(1, 0)
    evidence = CudaDriverPeerTopologyEvidence(
        topology_protocol="cuda-driver-p2p-attributes-v1",
        cuda_driver_api_version=13_010,
        links=(forward, reverse),
    )
    assert evidence.links == (forward, reverse)

    with pytest.raises(ValidationError, match="ordered pairs 0->1 and 1->0"):
        CudaDriverPeerTopologyEvidence(
            topology_protocol="cuda-driver-p2p-attributes-v1",
            cuda_driver_api_version=13_010,
            links=(forward, forward),
        )

    wrong_reverse = _cuda_peer_link(1, 0, source_uuid=GPU_UUID_OTHER)
    with pytest.raises(ValidationError, match="UUID directions do not reverse exactly"):
        CudaDriverPeerTopologyEvidence(
            topology_protocol="cuda-driver-p2p-attributes-v1",
            cuda_driver_api_version=13_010,
            links=(forward, wrong_reverse),
        )


def test_nvidia_smi_parser_uses_five_full_uuid_fields() -> None:
    evidence = parse_nvidia_smi_csv(
        "\n".join((_nvidia_smi_row(GPU_UUID_0), _nvidia_smi_row(GPU_UUID_1)))
    )

    assert tuple(gpu.uuid for gpu in evidence) == (GPU_UUID_0, GPU_UUID_1)
    assert all("B300" in gpu.nvidia_smi_name for gpu in evidence)
    assert all(gpu.nvidia_smi_compute_capability == B300_COMPUTE_CAPABILITY for gpu in evidence)
    assert all(gpu.nvidia_smi_memory_total_mib == 191456 for gpu in evidence)


def test_nvidia_smi_parser_normalizes_full_uuid_hex_case() -> None:
    evidence = parse_nvidia_smi_csv(
        "\n".join(
            (
                _nvidia_smi_row(GPU_UUID_0_UPPER_HEX),
                _nvidia_smi_row(GPU_UUID_1_UPPER_HEX),
            )
        )
    )

    assert tuple(gpu.uuid for gpu in evidence) == (GPU_UUID_0, GPU_UUID_1)


def test_gpu_identity_join_reorders_nvidia_smi_rows_by_cuda_ordinal() -> None:
    cuda_gpus = (
        _cuda_gpu(
            0,
            GPU_UUID_0,
            pci_bus_id="00000000:45:00.0",
            pci_error_code=None,
        ),
        _cuda_gpu(1, GPU_UUID_1, pci_bus_id=None, pci_error_code=999),
    )
    reversed_smi_rows = parse_nvidia_smi_csv(
        "\n".join((_nvidia_smi_row(GPU_UUID_1), _nvidia_smi_row(GPU_UUID_0)))
    )

    evidence = combine_gpu_identity(13_010, cuda_gpus, reversed_smi_rows)

    assert tuple(gpu.cuda_ordinal for gpu in evidence) == (0, 1)
    assert tuple(gpu.uuid for gpu in evidence) == (GPU_UUID_0, GPU_UUID_1)
    assert evidence[0].pci_bus_id_status == "available"
    assert evidence[0].pci_bus_id == "00000000:45:00.0"
    assert evidence[1].pci_bus_id_status == "unavailable"
    assert evidence[1].pci_bus_id is None
    assert evidence[1].pci_bus_id_error_code == 999


def test_gpu_identity_join_tolerates_only_uuid_hex_case_differences() -> None:
    cuda_gpus = (
        _cuda_gpu(0, GPU_UUID_0_UPPER_HEX, pci_bus_id=None, pci_error_code=999),
        _cuda_gpu(1, GPU_UUID_1, pci_bus_id=None, pci_error_code=999),
    )
    smi_gpus = parse_nvidia_smi_csv(
        "\n".join(
            (
                _nvidia_smi_row(GPU_UUID_1_UPPER_HEX),
                _nvidia_smi_row(GPU_UUID_0),
            )
        )
    )

    evidence = combine_gpu_identity(13_010, cuda_gpus, smi_gpus)

    assert tuple(gpu.uuid for gpu in evidence) == (GPU_UUID_0, GPU_UUID_1)


def test_gpu_identity_join_rejects_duplicate_cuda_uuid() -> None:
    cuda_gpus = (
        _cuda_gpu(0, GPU_UUID_0, pci_bus_id=None, pci_error_code=999),
        _cuda_gpu(1, GPU_UUID_0, pci_bus_id=None, pci_error_code=999),
    )
    smi_gpus = parse_nvidia_smi_csv(
        "\n".join((_nvidia_smi_row(GPU_UUID_0), _nvidia_smi_row(GPU_UUID_1)))
    )

    with pytest.raises(ValueError, match="CUDA GPU UUIDs must be unique"):
        combine_gpu_identity(13_010, cuda_gpus, smi_gpus)


@pytest.mark.parametrize("kind", ("missing", "mismatched"))
def test_gpu_identity_join_rejects_nvidia_uuid_inventory(kind: str) -> None:
    cuda_gpus = (
        _cuda_gpu(0, GPU_UUID_0, pci_bus_id=None, pci_error_code=999),
        _cuda_gpu(1, GPU_UUID_1, pci_bus_id=None, pci_error_code=999),
    )
    parsed = parse_nvidia_smi_csv(
        "\n".join((_nvidia_smi_row(GPU_UUID_0), _nvidia_smi_row(GPU_UUID_OTHER)))
    )
    smi_gpus = parsed[:1] if kind == "missing" else parsed

    with pytest.raises(ValueError, match="GPU UUID inventories differ"):
        combine_gpu_identity(13_010, cuda_gpus, smi_gpus)


def test_nvidia_smi_parser_rejects_duplicate_full_uuid() -> None:
    payload = "\n".join((_nvidia_smi_row(GPU_UUID_0), _nvidia_smi_row(GPU_UUID_0_UPPER_HEX)))

    with pytest.raises(ValueError, match="GPU UUIDs must be unique"):
        parse_nvidia_smi_csv(payload)


@pytest.mark.parametrize(
    "malformed_uuid",
    (
        GPU_UUID_0.removeprefix("GPU-"),
        GPU_UUID_0.lower(),
        GPU_UUID_0.replace("-", ""),
        GPU_UUID_0[:-1],
        f"{GPU_UUID_0}0",
        GPU_UUID_0.replace("a", "g"),
    ),
)
def test_nvidia_smi_parser_rejects_malformed_full_uuid(malformed_uuid: str) -> None:
    payload = "\n".join((_nvidia_smi_row(malformed_uuid), _nvidia_smi_row(GPU_UUID_1)))

    with pytest.raises(ValueError, match="must be a full GPU UUID"):
        parse_nvidia_smi_csv(payload)


@pytest.mark.parametrize(
    ("payload", "match"),
    (
        (_nvidia_smi_row(GPU_UUID_0), "exactly two"),
        (
            "\n".join(
                (
                    _nvidia_smi_row(GPU_UUID_0, name="NVIDIA H100"),
                    _nvidia_smi_row(GPU_UUID_1),
                )
            ),
            "not B300",
        ),
        (
            "\n".join(
                (
                    _nvidia_smi_row(GPU_UUID_0, compute_capability="9.0"),
                    _nvidia_smi_row(GPU_UUID_1),
                )
            ),
            "nvidia_smi_compute_capability",
        ),
        (
            f"0, {GPU_UUID_0}, NVIDIA B300, 191456, 590.44.01, 10.3\n"
            f"1, {GPU_UUID_1}, NVIDIA B300, 191456, 590.44.01, 10.3",
            "five populated fields",
        ),
    ),
)
def test_nvidia_smi_parser_rejects_wrong_hardware(payload: str, match: str) -> None:
    with pytest.raises((ValueError, ValidationError), match=match):
        parse_nvidia_smi_csv(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("cuda_driver_name", "NVIDIA H100", "not B300"),
        ("cuda_compute_capability", "9.0", "cuda_compute_capability"),
    ),
)
def test_cuda_driver_evidence_rejects_wrong_hardware(
    field: str,
    value: str,
    match: str,
) -> None:
    raw = _cuda_gpu(0, GPU_UUID_0, pci_bus_id=None, pci_error_code=999).model_dump()
    raw[field] = value

    with pytest.raises(ValidationError, match=match):
        CudaDriverGpuEvidence.model_validate(raw)


@pytest.mark.parametrize(
    ("pci_bus_id", "status", "error_code", "match"),
    (
        (None, "available", None, "available CUDA PCI evidence is incomplete"),
        (
            "00000000:45:00.0",
            "available",
            999,
            "available CUDA PCI evidence is incomplete",
        ),
        (
            "00000000:45:00.0",
            "unavailable",
            999,
            "unavailable CUDA PCI evidence lacks its error code",
        ),
        (None, "unavailable", None, "unavailable CUDA PCI evidence lacks its error code"),
    ),
)
def test_cuda_driver_evidence_rejects_inconsistent_pci_state(
    pci_bus_id: str | None,
    status: str,
    error_code: int | None,
    match: str,
) -> None:
    raw = _cuda_gpu(0, GPU_UUID_0, pci_bus_id=None, pci_error_code=999).model_dump()
    raw.update(
        pci_bus_id=pci_bus_id,
        pci_bus_id_status=status,
        pci_bus_id_error_code=error_code,
    )

    with pytest.raises(ValidationError, match=match):
        CudaDriverGpuEvidence.model_validate(raw)


def test_resource_monitor_reorders_rows_by_expected_uuid() -> None:
    samples = parse_nvidia_smi_monitor_csv(
        f"{GPU_UUID_1}, 222, 88\n{GPU_UUID_0}, 111, 44",
        expected_uuids=(GPU_UUID_0, GPU_UUID_1),
    )

    assert tuple(sample.uuid for sample in samples) == (GPU_UUID_0, GPU_UUID_1)
    assert tuple(sample.memory_used_mib for sample in samples) == (111, 222)
    assert tuple(sample.utilization_percent for sample in samples) == (44, 88)


def test_resource_monitor_normalizes_observed_and_expected_uuid_hex_case() -> None:
    samples = parse_nvidia_smi_monitor_csv(
        f"{GPU_UUID_1_UPPER_HEX}, 222, 88\n{GPU_UUID_0}, 111, 44",
        expected_uuids=(GPU_UUID_0_UPPER_HEX, GPU_UUID_1),
    )

    assert tuple(sample.uuid for sample in samples) == (GPU_UUID_0, GPU_UUID_1)
    assert tuple(sample.memory_used_mib for sample in samples) == (111, 222)


def test_resource_monitor_rejects_malformed_expected_uuid() -> None:
    with pytest.raises(ValueError, match="must be a full GPU UUID"):
        parse_nvidia_smi_monitor_csv(
            f"{GPU_UUID_0}, 111, 44\n{GPU_UUID_1}, 222, 88",
            expected_uuids=(GPU_UUID_0.lower(), GPU_UUID_1),
        )


@pytest.mark.parametrize(
    "payload",
    (
        f"{GPU_UUID_0}, 111, 44\n{GPU_UUID_0_UPPER_HEX}, 222, 88",
        f"{GPU_UUID_0}, 111, 44\n{GPU_UUID_OTHER}, 222, 88",
    ),
)
def test_resource_monitor_rejects_duplicate_or_drifted_uuid(payload: str) -> None:
    with pytest.raises(ValueError, match="GPU UUID inventory drifted"):
        parse_nvidia_smi_monitor_csv(
            payload,
            expected_uuids=(GPU_UUID_0, GPU_UUID_1),
        )


@pytest.mark.parametrize(
    "payload",
    (
        f"{GPU_UUID_0}, 111, 44",
        f"not-a-full-uuid, 111, 44\n{GPU_UUID_1}, 222, 88",
        f"{GPU_UUID_0}, 01, 44\n{GPU_UUID_1}, 222, 88",
        f"{GPU_UUID_0}, 111, 44, 9\n{GPU_UUID_1}, 222, 88",
    ),
)
def test_resource_monitor_rejects_malformed_rows(payload: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        parse_nvidia_smi_monitor_csv(
            payload,
            expected_uuids=(GPU_UUID_0, GPU_UUID_1),
        )
