from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_smoke import (
    HISTORICAL_INSTRUMENTATION_PATCH_SHA256,
    INSTRUMENTATION_SCHEMA_VERSION,
    LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256,
    load_inkling_smoke_config,
    load_verified_export_reference,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SmokeControlPlaneProvenance,
    SmokeNvidiaSmiTopologyDiagnostic,
    SmokeRawLogitAuditV2,
    smoke_control_plane_provenance,
    smoke_control_plane_tree_sha256,
    smoke_hardware_topology_sha256,
    smoke_package_manifest_sha256,
    smoke_run_id,
    smoke_terminal_receipt_sha256,
    strict_json_object,
    validate_smoke_terminal_receipt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_smoke_modal.yaml"
REFERENCE_PATH = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_verified_export.json"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _timings(tokens: int) -> dict[str, int | float]:
    return {
        "prompt_n": 4,
        "predicted_n": tokens,
        "cache_n": 0,
        "prompt_ms": 2.0,
        "prompt_per_token_ms": 0.5,
        "prompt_per_second": 2000.0,
        "predicted_ms": 4.0,
        "predicted_per_token_ms": 2.0,
        "predicted_per_second": 500.0,
    }


def _trial(trial: int) -> dict[str, Any]:
    return {
        "trial": trial,
        "token_ids": [1, 2],
        "tokens_predicted": 2,
        "minimum_sampled_token_logprob": -2.0,
        "maximum_sampled_token_logprob": -0.25,
        "all_returned_logprobs_finite": True,
        "response_sha256": f"{trial}" * 64,
        "timings": _timings(2),
    }


def _receipt_context() -> tuple[Any, Any, SmokeControlPlaneProvenance, str]:
    config = load_inkling_smoke_config(CONFIG_PATH)
    reference = load_verified_export_reference(REFERENCE_PATH)
    control_plane = smoke_control_plane_provenance(PROJECT_ROOT)
    return (
        config,
        reference,
        control_plane,
        smoke_run_id(config, control_plane.tree_sha256),
    )


def _version_one_receipt_context(
    patch_sha256: str,
    patch_size_bytes: int,
) -> tuple[
    Any,
    Any,
    SmokeControlPlaneProvenance,
    str,
]:
    current_config, reference, current_control_plane, _run_id = _receipt_context()
    config_payload = current_config.model_dump(mode="json")
    config_payload["schema_version"] = "inkling-smoke-config-v1"
    config_payload.pop("output_vocabulary")
    config_payload["runtime"]["instrumentation_schema_version"] = (
        "inkling-llama-smoke-instrumentation-v1"
    )
    config_payload["runtime"]["instrumentation_patch_sha256"] = patch_sha256
    config = type(current_config).model_validate(config_payload)
    files = tuple(
        item.model_copy(
            update={
                "sha256": patch_sha256,
                "size_bytes": patch_size_bytes,
            }
        )
        if item.path == "patches/inkling-smoke-a015409.patch"
        else item
        for item in current_control_plane.files
    )
    control_plane = SmokeControlPlaneProvenance(
        file_count=len(files),
        files=files,
        tree_sha256=smoke_control_plane_tree_sha256(files),
    )
    return config, reference, control_plane, smoke_run_id(config, control_plane.tree_sha256)


def _historical_receipt_context() -> tuple[
    Any,
    Any,
    SmokeControlPlaneProvenance,
    str,
]:
    return _version_one_receipt_context(
        HISTORICAL_INSTRUMENTATION_PATCH_SHA256,
        14_870,
    )


def _legacy_current_receipt_context() -> tuple[
    Any,
    Any,
    SmokeControlPlaneProvenance,
    str,
]:
    return _version_one_receipt_context(
        LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256,
        15_179,
    )


def _gpu_topology(
    hardware: list[dict[str, Any]],
    *,
    diagnostic_status: str = "command_failed",
    diagnostic_return_code: int = 255,
    diagnostic_stdout_size_bytes: int = 0,
) -> dict[str, Any]:
    first, second = hardware
    return {
        "schema_version": "inkling-smoke-gpu-topology-v1",
        "protocol": "cuda-driver-p2p-v1+nvidia-smi-topo-diagnostic-v1",
        "cuda_driver_api_version": first["cuda_driver_api_version"],
        "edges": [
            {
                "source_cuda_ordinal": first["cuda_ordinal"],
                "source_uuid": first["uuid"],
                "destination_cuda_ordinal": second["cuda_ordinal"],
                "destination_uuid": second["uuid"],
                "can_access_peer": True,
                "performance_rank": 0,
                "access_supported": True,
                "native_atomic_supported": True,
                "cuda_array_access_supported": True,
                "only_partial_native_atomic_supported": False,
            },
            {
                "source_cuda_ordinal": second["cuda_ordinal"],
                "source_uuid": second["uuid"],
                "destination_cuda_ordinal": first["cuda_ordinal"],
                "destination_uuid": first["uuid"],
                "can_access_peer": True,
                "performance_rank": 0,
                "access_supported": True,
                "native_atomic_supported": True,
                "cuda_array_access_supported": True,
                "only_partial_native_atomic_supported": False,
            },
        ],
        "nvidia_smi_topology": {
            "schema_version": "inkling-smoke-nvidia-smi-topology-diagnostic-v1",
            "argv": ["nvidia-smi", "topo", "-m"],
            "status": diagnostic_status,
            "return_code": diagnostic_return_code,
            "stdout_size_bytes": diagnostic_stdout_size_bytes,
            "stdout_sha256": (EMPTY_SHA256 if diagnostic_stdout_size_bytes == 0 else "7" * 64),
            "stderr_size_bytes": 32,
            "stderr_sha256": "8" * 64,
            "stdout_recorded": False,
            "stderr_recorded": False,
        },
    }


def _valid_receipt() -> tuple[dict[str, Any], Any, Any, SmokeControlPlaneProvenance, str]:
    config, reference, control_plane, run_id = _historical_receipt_context()
    artifacts = [
        artifact.model_dump(mode="json")
        for artifact in (
            *reference.q3_shards,
            reference.projector,
            reference.export_manifest,
            reference.verify_receipt,
            reference.quantize_receipt,
            reference.mmproj_receipt,
        )
    ]
    patch = next(
        item for item in control_plane.files if item.path == "patches/inkling-smoke-a015409.patch"
    )
    probes: list[dict[str, Any]] = []
    fixture_identities = {
        "none": (None, None),
        "synthetic_rgb8_png_16x16_checkerboard_v1": (
            "95b4e645a67edfb972c4ca1f2a0b8ed97e60988adfaa020d015d6a334576c2d7",
            86,
        ),
        "synthetic_pcm_s16le_wav_16000hz_mono_silence_250ms_v1": (
            "59460d5690616336b990fc7b1629428e3bd825e422da84469d2c8c8ecfaff43b",
            8_044,
        ),
    }
    for probe in config.probes:
        fixture_sha256, fixture_size_bytes = fixture_identities[probe.fixture]
        probes.append(
            {
                "probe_id": probe.probe_id,
                "modality": probe.modality,
                "prompt_sha256": probe.prompt_sha256,
                "prompt_recorded": False,
                "output_text_recorded": False,
                "fixture": probe.fixture,
                "fixture_sha256": fixture_sha256,
                "fixture_size_bytes": fixture_size_bytes,
                "seed": probe.seed,
                "temperature": probe.temperature,
                "repeatable_greedy_token_ids": True,
                "trials": [_trial(1), _trial(2)],
            }
        )
    raw_rows = [
        {
            "task_id": index // 2,
            "slot_id": 0,
            "completion_index": index + 1,
            "batch_index": 0,
            "count": 128,
            "finite": 128,
            "nan": 0,
            "pos_inf": 0,
            "neg_inf": 0,
        }
        for index in range(12)
    ]
    command = [
        "/opt/llama.cpp/build/bin/llama-server",
        "--model",
        f"/subject/{reference.q3_shards[0].path}",
        "--mmproj",
        f"/subject/{reference.projector.path}",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--ctx-size",
        "8192",
        "--n-gpu-layers",
        "all",
        "--n-cpu-moe",
        "0",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--flash-attn",
        "on",
        "--mmap",
        "--mmproj-offload",
        "--parallel",
        "1",
        "--threads",
        "16",
        "--threads-batch",
        "16",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--log-verbosity",
        "4",
        "--no-webui",
    ]
    invocation = {
        "schema_version": "inkling-smoke-invocation-v3",
        "run_id": run_id,
        "stage": "smoke_test",
        "sequence": 1,
        "limit": 1,
        "call_id": "fc-CALL",
        "input_id": "in-INPUT",
        "task_id": "ta-TASK",
        "launch_intent_sha256": "b" * 64,
        "smoke_config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "post_spawn_acceptance_path": ("control/post-spawn-acceptances/" + "b" * 64 + ".json"),
        "post_spawn_acceptance_sha256": "f" * 64,
        "attempt_registry_name": "inkling-smoke-attempt-registry-v1",
        "attempt_registry_id": "di-Attempt123",
        "attempt_registry_created_at_utc": "2026-07-22T12:00:00.000000Z",
        "attempt_registry_key": f"{run_id}:smoke_test",
        "attempt_registry_claim_sha256": "c" * 64,
        "attempt_claim_path": "control/smoke_test.attempt.claim.json",
        "attempt_claim_sha256": "c" * 64,
        "invocation_history_path": ("control/history/smoke_test.attempt.1." + "d" * 64 + ".json"),
        "invocation_history_sha256": "e" * 64,
    }
    python_packages = {"modal": "1.5.0", "pip": "25.0"}
    dpkg_packages = {"libc6:amd64": "2.39", "python3": "3.12"}
    package_manifest_sha256 = smoke_package_manifest_sha256(
        python_implementation="CPython",
        python_version="3.12.11",
        python_executable_path="/usr/local/bin/python3.12",
        python_executable_sha256="0" * 64,
        python_inventory_scope="image_sysconfig_purelib_v1",
        python_purelib="/usr/local/lib/python3.12/site-packages",
        python_inventory_sha256="5" * 64,
        python_packages=python_packages,
        modal_runtime_version="1.5.0",
        modal_package_root="/pkg/modal",
        modal_package_tree_schema_version="inkling-smoke-source-tree-v1",
        modal_package_file_count=42,
        modal_package_tree_sha256="3" * 64,
        dpkg_inventory_sha256="6" * 64,
        dpkg_packages=dpkg_packages,
        nvcc_version="V13.1.115",
        nvcc_version_sha256="7" * 64,
    )
    hardware = [
        {
            "identity_protocol": "cuda-driver-uuid+nvidia-smi-uuid-v1",
            "cuda_driver_api_version": 13_010,
            "cuda_ordinal": index,
            "uuid": (
                "GPU-00000000-0000-0000-0000-000000000001"
                if index == 0
                else "GPU-00000000-0000-0000-0000-000000000002"
            ),
            "cuda_driver_name": "NVIDIA B300",
            "nvidia_smi_name": "NVIDIA B300",
            "cuda_compute_capability": "10.3",
            "nvidia_smi_compute_capability": "10.3",
            "cuda_total_memory_bytes": 262144 * 1024**2,
            "nvidia_smi_memory_total_mib": 262144,
            "driver_version": "580.1",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "pci_bus_id_status": "available",
            "pci_bus_id_source": "cuda_driver_api",
            "pci_bus_id_error_code": None,
        }
        for index in range(2)
    ]
    host: dict[str, Any] = {
        "provider": "Modal",
        "cpu_model": "AMD EPYC test CPU",
        "host_logical_cpu_count": 32,
        "host_logical_cpu_count_scope": "host_online_os_cpu_count",
        "logical_cpu_count": 16,
        "logical_cpu_count_scope": "container_cgroup_cpu_quota",
        "requested_cpu_cores": 16,
        "requested_cpu_scope": "modal_physical_cores_hard_request_and_limit",
        "cgroup_membership_sha256": "8" * 64,
        "cgroup_mountinfo_sha256": "9" * 64,
        "cgroup_visibility_scope": "process_mount_namespace_visible_hierarchy",
        "cgroup_process_pid": 1234,
        "cgroup_cpu_leaf_path_sha256": "1" * 64,
        "cgroup_cpu_leaf_pid_verified": True,
        "cgroup_cpu_leaf_cgroup_procs_sha256": "2" * 64,
        "cgroup_cpu_quota_millicores": 16_000,
        "cgroup_cpu_quota_source": "cgroup_v2_visible_hierarchy_cpu.max",
        "cgroup_cpu_limit_path_sha256s": ["3" * 64],
        "cgroup_cpu_limit_values_millicores": [16_000],
        "cpu_affinity_ids": list(range(32)),
        "cpu_affinity_scope": "container_effective_sched_getaffinity",
        "host_ram_bytes": 512 * 1024**3,
        "host_ram_scope": "host_physical_proc_meminfo_memtotal",
        "ram_bytes": 64 * 1024**3,
        "ram_scope": "container_cgroup_memory_limit",
        "requested_ram_bytes": 64 * 1024**3,
        "requested_ram_scope": "modal_bytes_hard_request_and_limit",
        "cgroup_memory_leaf_path_sha256": "4" * 64,
        "cgroup_memory_leaf_pid_verified": True,
        "cgroup_memory_leaf_cgroup_procs_sha256": "5" * 64,
        "cgroup_memory_limit_bytes": 64 * 1024**3,
        "cgroup_memory_limit_source": "cgroup_v2_visible_hierarchy_memory.max",
        "cgroup_memory_limit_path_sha256s": ["6" * 64],
        "cgroup_memory_limit_values_bytes": [64 * 1024**3],
        "nvidia_smi_topo_m_sha256": "a" * 64,
        "topology_schema_version": "inkling-smoke-hardware-topology-v3",
    }
    host["topology_sha256"] = smoke_hardware_topology_sha256(host, hardware)
    receipt: dict[str, Any] = {
        "schema_version": "inkling-smoke-terminal-v3",
        "status": "passed",
        "stage": "smoke_test",
        "run_id": run_id,
        "subject": {
            "run_id": reference.subject_run_id,
            "model_id": reference.model_id,
            "revision": reference.revision,
            "architecture": reference.architecture,
            "quant_type": reference.quant_type,
            "mtp": reference.mtp,
            "verified_export_reference_sha256": reference.reference_sha256,
            "q3_shard_count": reference.q3_shard_count,
            "q3_total_bytes": reference.q3_total_bytes,
            "projector_sha256": reference.projector.sha256,
        },
        "smoke_config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "control_plane_file_count": control_plane.file_count,
        "launch_intent_sha256": invocation["launch_intent_sha256"],
        "invocation": invocation,
        "artifact_rehash": {
            "algorithm": "sha256",
            "worker_count": 8,
            "elapsed_seconds": 1.0,
            "artifact_count": 54,
            "artifacts": artifacts,
        },
        "runtime": {
            "llama_cpp_repository": config.runtime.repository,
            "llama_cpp_commit": config.runtime.commit,
            "cuda_image": (f"{config.runtime.image.image}@{config.runtime.image.digest}"),
            "cuda_driver_library_path": ("/usr/lib/x86_64-linux-gnu/libcuda.so.580.1"),
            "cuda_driver_library_sha256": "4" * 64,
            "cmake_definitions": list(config.runtime.cmake_definitions),
            "binaries": [
                {
                    "name": name,
                    "path": f"/opt/llama.cpp/build/bin/{name}",
                    "sha256": f"{index}" * 64,
                    "size_bytes": 1024,
                }
                for index, name in enumerate(config.runtime.build_targets, start=1)
            ],
            "python_implementation": "CPython",
            "python_version": "3.12.11",
            "python_executable_path": "/usr/local/bin/python3.12",
            "python_executable_sha256": "0" * 64,
            "python_inventory_scope": "image_sysconfig_purelib_v1",
            "python_purelib": "/usr/local/lib/python3.12/site-packages",
            "python_inventory_sha256": "5" * 64,
            "python_packages": python_packages,
            "modal_runtime_version": "1.5.0",
            "modal_package_root": "/pkg/modal",
            "modal_package_tree_schema_version": "inkling-smoke-source-tree-v1",
            "modal_package_file_count": 42,
            "modal_package_tree_sha256": "3" * 64,
            "dpkg_inventory_sha256": "6" * 64,
            "dpkg_packages": dpkg_packages,
            "package_manifest_schema_version": ("inkling-smoke-package-manifest-v2"),
            "package_manifest_sha256": package_manifest_sha256,
            "nvcc_version": "V13.1.115",
            "nvcc_version_sha256": "7" * 64,
            "instrumentation_schema_version": ("inkling-llama-smoke-instrumentation-v1"),
            "instrumentation_patch_path": "/root/inkling-smoke-a015409.patch",
            "instrumentation_patch_sha256": patch.sha256,
            "patched_source_paths": [
                "ggml/src/ggml-backend.cpp",
                "src/llama-model-loader.cpp",
                "src/llama-model-loader.h",
                "tools/mtmd/clip.cpp",
                "tools/mtmd/mtmd.cpp",
                "tools/server/server-context.cpp",
            ],
            "patched_diff_sha256": "f" * 64,
            "base_source_blob_ids": [
                {
                    "path": "ggml/src/ggml-backend.cpp",
                    "git_blob_id": "87615921c09be5ef8c4996faa70fb3f49c385031",
                },
                {
                    "path": "src/llama-model-loader.cpp",
                    "git_blob_id": "28f8bb7934bbc807a08dc13ad58724ec77281903",
                },
                {
                    "path": "src/llama-model-loader.h",
                    "git_blob_id": "c476026d3e510ad03d3e6f0d619ecea7fc95319c",
                },
                {
                    "path": "tools/mtmd/clip.cpp",
                    "git_blob_id": "dbd07081bf73f336a17bd3b8d8359830128c424b",
                },
                {
                    "path": "tools/mtmd/mtmd.cpp",
                    "git_blob_id": "3e81e44143fa635e56e0a757ce1ba33d34d107e4",
                },
                {
                    "path": "tools/server/server-context.cpp",
                    "git_blob_id": "7564ad4e9cfb8e77d610e90c7530121214a4c483",
                },
            ],
        },
        "host": host,
        "hardware": hardware,
        "server": {
            "command": command,
            "audit_environment": {
                "IQL_SMOKE_BACKEND_AUDIT": "1",
                "IQL_SMOKE_RAW_LOGIT_AUDIT": "1",
                "LLAMA_MEDIA_MARKER": "<__media_iql_smoke_v1__>",
            },
            "network_scope": "loopback_only_with_modal_external_network_blocked",
            "post_rehash_load_to_health_seconds": 10.0,
            "loader_offload": {
                "cuda_device_count": 2,
                "offloaded_layers": 1,
                "offloadable_layers": 1,
                "output_layer_offloaded": True,
                "all_offloadable_layers_on_gpu": True,
                "no_gpu_warning_observed": True,
            },
            "artifact_load": {
                "schema_version": "inkling-artifact-load-v1",
                "first_shard_path": ("/subject/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf"),
                "additional_shards_loaded": 48,
                "total_shards_loaded": 49,
                "projector_path": "/subject/mmproj/mmproj-BF16.gguf",
                "text_shards": {
                    "expected": 49,
                    "opened": 49,
                    "contexts": 49,
                    "tensors": 1_000,
                },
                "text_load": {
                    "opened": 49,
                    "accounted": 49,
                    "tensors": 1_000,
                    "bytes": 123_456,
                    "size_done": 123_456,
                    "size_data": 123_456,
                    "mmap": True,
                },
                "projector_tensors": [
                    {
                        "modality": "vision",
                        "projector": "inkling",
                        "tensors": 10,
                        "bytes": 1_000,
                    },
                    {
                        "modality": "audio",
                        "projector": "inkling",
                        "tensors": 20,
                        "bytes": 2_000,
                    },
                ],
                "projector_ready": {
                    "opened": True,
                    "vision": True,
                    "audio": True,
                    "vision_type": "inkling",
                    "audio_type": "inkling",
                    "n_embd": 1_024,
                },
                "all_expected_artifacts_loaded": True,
            },
            "raw_logit_audit": {
                "schema_version": "inkling-raw-logit-audit-v1",
                "expected_generated_token_vectors": len(raw_rows),
                "observed_generated_token_vectors": len(raw_rows),
                "vocab_size": 128,
                "rows": raw_rows,
                "all_rows_complete": True,
                "all_values_finite": True,
            },
            "backend_audit": {
                "schema_version": "inkling-backend-audit-v1",
                "graphs": [
                    {
                        "graph_uid": 1,
                        "phase": "post_assignment_pre_split",
                        "scope": "non_view_compute",
                        "compute": 100,
                        "gpu": 100,
                        "cpu": 0,
                        "accel": 0,
                        "other": 0,
                        "unassigned": 0,
                    }
                ],
                "identities": [
                    {
                        "graph_uid": 1,
                        "backend_index": 0,
                        "backend_name": "B300-0",
                        "device_name": "B300-0",
                        "device_type": "gpu",
                        "compute": 50,
                    },
                    {
                        "graph_uid": 1,
                        "backend_index": 1,
                        "backend_name": "B300-1",
                        "device_name": "B300-1",
                        "device_type": "gpu",
                        "compute": 50,
                    },
                ],
                "observed_graphs": 1,
                "compute_operations": 100,
                "gpu_operations": 100,
                "accelerator_operations": 0,
                "cpu_operations": 0,
                "other_operations": 0,
                "unassigned_operations": 0,
                "all_compute_operations_accelerated": True,
                "no_cpu_model_graph_fallback": True,
            },
            "properties": {
                "props_sha256": "8" * 64,
                "models_sha256": "9" * 64,
                "build_info": f"build {config.runtime.commit[:7]}",
                "modalities": {"text": True, "vision": True, "audio": True},
                "vocab_size": 128,
                "media_marker_sha256": "a" * 64,
            },
            "server_log_sha256": "b" * 64,
            "cleanup": {"method": "SIGTERM", "return_code": -15, "clean": True},
        },
        "probes": probes,
        "resources": {
            "sampling_interval_seconds": 1.0,
            "sample_count": 2,
            "server_peak_host_rss_mib": 1024,
            "gpu_peak_memory_used_mib": [1024, 1024],
            "gpu_peak_utilization_percent": [50, 50],
        },
        "claims": config.claims.model_dump(mode="json"),
        "evidence_policy": config.evidence.model_dump(mode="json"),
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "completed_at_utc": "2026-07-22T12:00:00+00:00",
    }
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)
    return receipt, config, reference, control_plane, run_id


def _valid_v4_receipt() -> tuple[
    dict[str, Any],
    Any,
    Any,
    SmokeControlPlaneProvenance,
    str,
]:
    receipt, _historical_config, reference, _historical_control_plane, _run_id = _valid_receipt()
    config, current_reference, control_plane, run_id = _legacy_current_receipt_context()
    assert current_reference == reference
    receipt["schema_version"] = "inkling-smoke-terminal-v4"
    receipt["run_id"] = run_id
    receipt["smoke_config_hash"] = config.config_hash()
    receipt["control_plane_sha256"] = control_plane.tree_sha256
    receipt["control_plane_file_count"] = control_plane.file_count
    receipt["invocation"]["run_id"] = run_id
    receipt["invocation"]["smoke_config_hash"] = config.config_hash()
    receipt["invocation"]["control_plane_sha256"] = control_plane.tree_sha256
    receipt["invocation"]["attempt_registry_key"] = f"{run_id}:smoke_test"
    receipt["runtime"]["instrumentation_patch_sha256"] = config.runtime.instrumentation_patch_sha256
    command = receipt["server"]["command"]
    verbosity_index = command.index("--log-verbosity")
    verbosity_arguments = command[verbosity_index : verbosity_index + 2]
    del command[verbosity_index : verbosity_index + 2]
    command[1:1] = verbosity_arguments
    receipt["runtime"]["patched_source_paths"].append("tools/server/server.cpp")
    receipt["runtime"]["patched_source_paths"].sort()
    receipt["runtime"]["base_source_blob_ids"].append(
        {
            "path": "tools/server/server.cpp",
            "git_blob_id": "20effbb14851b201118843bf14fa5bc51de1e304",
        }
    )
    for ordinal, identity in enumerate(receipt["server"]["backend_audit"]["identities"]):
        identity["backend_name"] = f"CUDA{ordinal}"
        identity["device_name"] = f"CUDA{ordinal}"
    receipt["host"].pop("nvidia_smi_topo_m_sha256")
    receipt["host"]["topology_schema_version"] = "inkling-smoke-hardware-topology-v4"
    receipt["gpu_topology"] = _gpu_topology(receipt["hardware"])
    receipt["host"]["topology_sha256"] = smoke_hardware_topology_sha256(
        receipt["host"],
        receipt["hardware"],
        receipt["gpu_topology"],
    )
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)
    return receipt, config, reference, control_plane, run_id


def _valid_v5_receipt() -> tuple[
    dict[str, Any],
    Any,
    Any,
    SmokeControlPlaneProvenance,
    str,
]:
    receipt, _legacy_config, reference, _legacy_control_plane, _run_id = _valid_v4_receipt()
    config, current_reference, control_plane, run_id = _receipt_context()
    assert current_reference == reference
    assert config.output_vocabulary is not None
    output_vocabulary = config.output_vocabulary
    receipt["schema_version"] = "inkling-smoke-terminal-v5"
    receipt["run_id"] = run_id
    receipt["smoke_config_hash"] = config.config_hash()
    receipt["control_plane_sha256"] = control_plane.tree_sha256
    receipt["control_plane_file_count"] = control_plane.file_count
    receipt["invocation"]["run_id"] = run_id
    receipt["invocation"]["smoke_config_hash"] = config.config_hash()
    receipt["invocation"]["control_plane_sha256"] = control_plane.tree_sha256
    receipt["invocation"]["attempt_registry_key"] = f"{run_id}:smoke_test"
    receipt["runtime"]["instrumentation_schema_version"] = (
        config.runtime.instrumentation_schema_version
    )
    receipt["runtime"]["instrumentation_patch_sha256"] = config.runtime.instrumentation_patch_sha256
    raw_rows = [
        {
            "task_id": index // 2,
            "slot_id": 0,
            "completion_index": index + 1,
            "batch_index": 0,
            "count": output_vocabulary.vocab_size,
            "unpadded_count": output_vocabulary.unpadded_vocab_size,
            "padded_count": output_vocabulary.padded_vocab_size,
            "unpadded_finite": output_vocabulary.unpadded_vocab_size,
            "unpadded_nan": 0,
            "unpadded_pos_inf": 0,
            "unpadded_neg_inf": 0,
            "padded_finite": 0,
            "padded_nan": 0,
            "padded_pos_inf": 0,
            "padded_neg_inf": output_vocabulary.padded_vocab_size,
        }
        for index in range(12)
    ]
    receipt["server"]["raw_logit_audit"] = {
        "schema_version": "inkling-raw-logit-audit-v2",
        "expected_generated_token_vectors": len(raw_rows),
        "observed_generated_token_vectors": len(raw_rows),
        "vocab_size": output_vocabulary.vocab_size,
        "unpadded_vocab_size": output_vocabulary.unpadded_vocab_size,
        "padded_vocab_size": output_vocabulary.padded_vocab_size,
        "rows": raw_rows,
        "all_rows_complete": True,
        "all_unpadded_values_finite": True,
        "all_padded_values_negative_infinity": True,
    }
    receipt["server"]["properties"]["vocab_size"] = output_vocabulary.vocab_size
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)
    return receipt, config, reference, control_plane, run_id


def _reseal(value: dict[str, Any]) -> None:
    value["receipt_sha256"] = smoke_terminal_receipt_sha256(value)


def _reseal_topology_and_receipt(value: dict[str, Any]) -> None:
    value["host"]["topology_sha256"] = smoke_hardware_topology_sha256(
        value["host"],
        value["hardware"],
        value.get("gpu_topology"),
    )
    _reseal(value)


def test_strict_json_object_rejects_duplicate_nested_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key 'value'"):
        strict_json_object('{"outer":{"value":1,"value":2}}')


def test_strict_json_object_rejects_overflowed_number() -> None:
    with pytest.raises(ValueError, match="non-finite number"):
        strict_json_object('{"value":1e9999}')


def test_complete_terminal_receipt_validates() -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.receipt_sha256 == receipt["receipt_sha256"]
    assert len(observed.probes) == 3
    assert all(len(probe.trials) == 2 for probe in observed.probes)
    assert observed.host.logical_cpu_count == 16
    assert observed.host.logical_cpu_count_scope == "container_cgroup_cpu_quota"
    assert observed.host.ram_bytes == 64 * 1024**3
    assert observed.host.ram_scope == "container_cgroup_memory_limit"


def test_historical_v3_receipt_keeps_legacy_command_and_backend_labels_readable() -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.server.command[-3:] == ("--log-verbosity", "4", "--no-webui")
    assert {row.backend_name for row in observed.server.backend_audit.identities} == {
        "B300-0",
        "B300-1",
    }
    assert "tools/server/server.cpp" not in observed.runtime.patched_source_paths


def test_historical_v3_receipt_keeps_legacy_accelerator_identity_semantics() -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    backend_audit = receipt["server"]["backend_audit"]
    backend_audit["identities"][0]["device_type"] = "accel"
    backend_audit["graphs"][0]["gpu"] = 50
    backend_audit["graphs"][0]["accel"] = 50
    backend_audit["gpu_operations"] = 50
    backend_audit["accelerator_operations"] = 50
    _reseal(receipt)

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.server.backend_audit.accelerator_operations == 50


def test_complete_v4_terminal_receipt_binds_cuda_peer_topology() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.schema_version == "inkling-smoke-terminal-v4"
    assert observed.host.topology_schema_version == "inkling-smoke-hardware-topology-v4"
    assert observed.runtime.instrumentation_schema_version == (
        "inkling-llama-smoke-instrumentation-v1"
    )
    assert observed.host.nvidia_smi_topo_m_sha256 is None
    assert observed.gpu_topology is not None
    assert tuple(
        (edge.source_cuda_ordinal, edge.destination_cuda_ordinal)
        for edge in observed.gpu_topology.edges
    ) == ((0, 1), (1, 0))
    assert observed.gpu_topology.nvidia_smi_topology.status == "command_failed"
    assert observed.gpu_topology.nvidia_smi_topology.return_code == 255
    assert observed.host.topology_sha256 == smoke_hardware_topology_sha256(
        observed.host,
        observed.hardware,
        observed.gpu_topology,
    )


def test_complete_v5_terminal_receipt_binds_padded_vocabulary_evidence() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v5_receipt()

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.schema_version == "inkling-smoke-terminal-v5"
    assert observed.runtime.instrumentation_schema_version == INSTRUMENTATION_SCHEMA_VERSION
    raw_logit_audit = observed.server.raw_logit_audit
    assert isinstance(raw_logit_audit, SmokeRawLogitAuditV2)
    assert raw_logit_audit.vocab_size == 201_024
    assert raw_logit_audit.unpadded_vocab_size == 200_058
    assert raw_logit_audit.padded_vocab_size == 966


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["server"]["raw_logit_audit"]["rows"][0].__setitem__(
            "padded_finite",
            1,
        ),
        lambda value: value["server"]["raw_logit_audit"]["rows"][0].__setitem__(
            "padded_neg_inf",
            965,
        ),
        lambda value: value["server"]["raw_logit_audit"]["rows"][1].update(
            {
                field: value["server"]["raw_logit_audit"]["rows"][0][field]
                for field in (
                    "task_id",
                    "slot_id",
                    "completion_index",
                    "batch_index",
                )
            }
        ),
        lambda value: value["server"]["raw_logit_audit"]["rows"].pop(),
    ),
    ids=(
        "finite-padded-value",
        "missing-padded-negative-infinity",
        "duplicate-vector-identity",
        "missing-vector",
    ),
)
def test_v5_terminal_receipt_rejects_invalid_partitioned_logit_evidence(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_v5_receipt()
    mutate(receipt)
    _reseal(receipt)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_v5_terminal_receipt_rejects_a_config_incompatible_vocabulary_boundary() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v5_receipt()
    audit = receipt["server"]["raw_logit_audit"]
    audit["unpadded_vocab_size"] = 200_057
    audit["padded_vocab_size"] = 967
    for row in audit["rows"]:
        row["unpadded_count"] = 200_057
        row["unpadded_finite"] = 200_057
        row["padded_count"] = 967
        row["padded_neg_inf"] = 967
    _reseal(receipt)

    with pytest.raises(ValueError, match="vocabulary differs from the exact config"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_v5_terminal_receipt_rejects_a_token_from_the_padded_suffix() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v5_receipt()
    receipt["probes"][0]["trials"][0]["token_ids"][0] = 200_058
    _reseal(receipt)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_terminal_receipt_versions_reject_the_other_raw_logit_schema() -> None:
    v5, v5_config, reference, v5_control_plane, v5_run_id = _valid_v5_receipt()
    v3, _v3_config, _reference, _v3_control_plane, _v3_run_id = _valid_receipt()
    v1_audit = copy.deepcopy(v3["server"]["raw_logit_audit"])
    v1_audit["vocab_size"] = 201_024
    for row in v1_audit["rows"]:
        row["count"] = 201_024
        row["finite"] = 201_024
    v5["server"]["raw_logit_audit"] = v1_audit
    _reseal(v5)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_terminal_receipt(
            v5,
            config=v5_config,
            reference=reference,
            control_plane=v5_control_plane,
            run_id=v5_run_id,
        )

    v4, v4_config, _reference, v4_control_plane, v4_run_id = _valid_v4_receipt()
    valid_v2_audit = _valid_v5_receipt()[0]["server"]["raw_logit_audit"]
    v4["server"]["raw_logit_audit"] = valid_v2_audit
    v4["server"]["properties"]["vocab_size"] = 201_024
    _reseal(v4)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_terminal_receipt(
            v4,
            config=v4_config,
            reference=reference,
            control_plane=v4_control_plane,
            run_id=v4_run_id,
        )


def test_nvidia_smi_topology_diagnostic_accepts_rc255_only_as_command_failed() -> None:
    receipt, _, _, _, _ = _valid_v4_receipt()
    diagnostic = receipt["gpu_topology"]["nvidia_smi_topology"]

    observed = SmokeNvidiaSmiTopologyDiagnostic.model_validate(diagnostic)

    assert observed.status == "command_failed"
    assert observed.return_code == 255
    changed = copy.deepcopy(diagnostic)
    changed["status"] = "available"
    with pytest.raises(ValidationError, match="available nvidia-smi topology evidence"):
        SmokeNvidiaSmiTopologyDiagnostic.model_validate(changed)


def test_nvidia_smi_topology_diagnostic_accepts_available_nonempty_stdout() -> None:
    receipt, _, _, _, _ = _valid_v4_receipt()
    diagnostic = copy.deepcopy(receipt["gpu_topology"]["nvidia_smi_topology"])
    diagnostic.update(
        {
            "status": "available",
            "return_code": 0,
            "stdout_size_bytes": 128,
            "stdout_sha256": "7" * 64,
            "stderr_size_bytes": 0,
            "stderr_sha256": EMPTY_SHA256,
        }
    )

    observed = SmokeNvidiaSmiTopologyDiagnostic.model_validate(diagnostic)

    assert observed.status == "available"
    assert observed.return_code == 0
    assert observed.stdout_size_bytes == 128


@pytest.mark.parametrize(
    ("status", "return_code", "stdout_size_bytes", "stderr_size_bytes"),
    (
        ("empty_output", 0, 0, 0),
        ("timed_out", None, 17, 23),
        ("unavailable", None, 0, 0),
        ("invalid_result", None, 0, 0),
    ),
)
def test_nvidia_smi_topology_diagnostic_accepts_nonfatal_statuses(
    status: str,
    return_code: int | None,
    stdout_size_bytes: int,
    stderr_size_bytes: int,
) -> None:
    receipt, _, _, _, _ = _valid_v4_receipt()
    diagnostic = copy.deepcopy(receipt["gpu_topology"]["nvidia_smi_topology"])
    diagnostic.update(
        {
            "status": status,
            "return_code": return_code,
            "stdout_size_bytes": stdout_size_bytes,
            "stdout_sha256": EMPTY_SHA256 if stdout_size_bytes == 0 else "7" * 64,
            "stderr_size_bytes": stderr_size_bytes,
            "stderr_sha256": EMPTY_SHA256 if stderr_size_bytes == 0 else "8" * 64,
        }
    )

    observed = SmokeNvidiaSmiTopologyDiagnostic.model_validate(diagnostic)

    assert observed.status == status
    assert observed.return_code == return_code


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update(
            {
                "status": "available",
                "return_code": 0,
                "stdout_size_bytes": 0,
                "stdout_sha256": EMPTY_SHA256,
            }
        ),
        lambda value: value.update(
            {
                "status": "command_failed",
                "return_code": 0,
                "stdout_size_bytes": 128,
                "stdout_sha256": "7" * 64,
            }
        ),
        lambda value: value.update(
            {
                "status": "timed_out",
                "return_code": 255,
            }
        ),
        lambda value: value.update(
            {
                "status": "unavailable",
                "return_code": None,
                "stderr_size_bytes": 1,
                "stderr_sha256": "8" * 64,
            }
        ),
        lambda value: value.__setitem__("argv", ["nvidia-smi", "-L"]),
        lambda value: value.__setitem__("stdout_recorded", True),
        lambda value: value.__setitem__("raw_stdout", "GPU0 GPU1 NV18"),
    ),
    ids=(
        "available-empty-stdout",
        "command-failed-zero-return-code",
        "timed-out-with-return-code",
        "unavailable-with-output",
        "different-command",
        "stdout-recorded",
        "raw-extra-field",
    ),
)
def test_nvidia_smi_topology_diagnostic_rejects_invalid_outcomes(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    receipt, _, _, _, _ = _valid_v4_receipt()
    diagnostic = copy.deepcopy(receipt["gpu_topology"]["nvidia_smi_topology"])
    mutate(diagnostic)

    with pytest.raises(ValidationError):
        SmokeNvidiaSmiTopologyDiagnostic.model_validate(diagnostic)


@pytest.mark.parametrize(
    ("mutate", "expected_detail"),
    (
        (
            lambda value: value["gpu_topology"]["edges"].reverse(),
            "exact ordered zero-to-one edges",
        ),
        (
            lambda value: (
                value["gpu_topology"]["edges"][0].__setitem__(
                    "source_uuid",
                    "GPU-00000000-0000-0000-0000-000000000003",
                ),
                value["gpu_topology"]["edges"][1].__setitem__(
                    "destination_uuid",
                    "GPU-00000000-0000-0000-0000-000000000003",
                ),
            ),
            "differs from the receipt GPU inventory",
        ),
        (
            lambda value: value["gpu_topology"].__setitem__(
                "cuda_driver_api_version",
                13_020,
            ),
            "different driver API version",
        ),
        (
            lambda value: value["gpu_topology"]["edges"][0].__setitem__(
                "can_access_peer",
                False,
            ),
            "returned different support values",
        ),
        (
            lambda value: value["gpu_topology"]["edges"][0].__setitem__(
                "only_partial_native_atomic_supported",
                True,
            ),
            "atomic support cannot be both full and partial",
        ),
        (
            lambda value: value["gpu_topology"]["nvidia_smi_topology"].__setitem__(
                "stdout_sha256",
                "6" * 64,
            ),
            "topology hash differs from its inventory",
        ),
    ),
    ids=(
        "edge-order",
        "edge-uuid",
        "driver-api",
        "access-api-mismatch",
        "atomic-support-contradiction",
        "diagnostic-hash",
    ),
)
def test_v4_terminal_receipt_rejects_topology_tampering(
    mutate: Callable[[dict[str, Any]], object],
    expected_detail: str,
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    mutate(receipt)
    _reseal(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid") as error:
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )

    assert expected_detail in str(error.value.__cause__)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.pop("gpu_topology"),
        lambda value: value["host"].__setitem__(
            "nvidia_smi_topo_m_sha256",
            "a" * 64,
        ),
    ),
    ids=("missing-peer-topology", "legacy-topology-field"),
)
def test_v4_terminal_receipt_rejects_missing_or_legacy_topology(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    mutate(receipt)
    _reseal(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_terminal_receipt_accepts_observed_modal_input_suffix() -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    receipt["invocation"]["input_id"] = "in-01KY5YWSRQG1AR8BEWRJ2A797D:1784759084850-0"
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.invocation.input_id.endswith(":1784759084850-0")


@pytest.mark.parametrize(
    ("mode", "replacement"),
    (
        ("replace", "3"),
        ("replace", "trace"),
        ("remove", None),
        ("duplicate", "4"),
    ),
)
def test_terminal_receipt_rejects_log_verbosity_command_drift(
    mode: str,
    replacement: str | None,
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    command = receipt["server"]["command"]
    assert isinstance(command, list)
    index = command.index("--log-verbosity")
    if mode == "replace":
        assert replacement is not None
        command[index + 1] = replacement
    elif mode == "remove":
        command.pop(index + 1)
    else:
        assert mode == "duplicate"
        assert replacement is not None
        command[index + 2 : index + 2] = ["--log-verbosity", replacement]
    _reseal(receipt)

    with pytest.raises(ValueError, match="server command differs"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_terminal_receipt_rejects_late_log_verbosity_even_at_level_four() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    command = receipt["server"]["command"]
    assert isinstance(command, list)
    assert command[1:3] == ["--log-verbosity", "4"]
    del command[1:3]
    command[-1:-1] = ["--log-verbosity", "4"]
    _reseal(receipt)

    with pytest.raises(ValueError, match="server command differs"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_terminal_receipt_rejects_compute_on_only_one_cuda_device() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    backend_audit = receipt["server"]["backend_audit"]
    backend_audit["identities"] = [backend_audit["identities"][0]]
    backend_audit["identities"][0]["compute"] = 100
    _reseal(receipt)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_v4_terminal_receipt_accepts_cuda0_auxiliary_graph() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    backend_audit = receipt["server"]["backend_audit"]
    backend_audit["graphs"].append(
        {
            "graph_uid": 2,
            "phase": "post_assignment_pre_split",
            "scope": "non_view_compute",
            "compute": 25,
            "gpu": 25,
            "cpu": 0,
            "accel": 0,
            "other": 0,
            "unassigned": 0,
        }
    )
    backend_audit["identities"].append(
        {
            "graph_uid": 2,
            "backend_index": 0,
            "backend_name": "CUDA0",
            "device_name": "CUDA0",
            "device_type": "gpu",
            "compute": 25,
        }
    )
    backend_audit["observed_graphs"] = 2
    backend_audit["compute_operations"] = 125
    backend_audit["gpu_operations"] = 125
    _reseal(receipt)

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.server.backend_audit.observed_graphs == 2
    assert len(observed.server.backend_audit.identities) == 3


def _add_v4_auxiliary_backend_graph(
    receipt: dict[str, Any],
    identity_rows: tuple[dict[str, Any], ...],
) -> None:
    backend_audit = receipt["server"]["backend_audit"]
    backend_audit["graphs"].append(
        {
            "graph_uid": 2,
            "phase": "post_assignment_pre_split",
            "scope": "non_view_compute",
            "compute": 8,
            "gpu": 8,
            "cpu": 0,
            "accel": 0,
            "other": 0,
            "unassigned": 0,
        }
    )
    backend_audit["identities"].extend(copy.deepcopy(identity_rows))
    backend_audit["observed_graphs"] = 2
    backend_audit["compute_operations"] = 108
    backend_audit["gpu_operations"] = 108


@pytest.mark.parametrize(
    ("identity_rows", "cause_match"),
    (
        pytest.param(
            (
                {
                    "graph_uid": 2,
                    "backend_index": 0,
                    "backend_name": "CUDA0",
                    "device_name": "CUDA0",
                    "device_type": "gpu",
                    "compute": 7,
                },
                {
                    "graph_uid": 2,
                    "backend_index": 2,
                    "backend_name": "CUDA2",
                    "device_name": "CUDA2",
                    "device_type": "gpu",
                    "compute": 1,
                },
            ),
            "exact CUDA index and device identities",
            id="third-positive-identity",
        ),
        pytest.param(
            (
                {
                    "graph_uid": 2,
                    "backend_index": 0,
                    "backend_name": "CUDA0",
                    "device_name": "CUDA0",
                    "device_type": "gpu",
                    "compute": 4,
                },
                {
                    "graph_uid": 2,
                    "backend_index": 0,
                    "backend_name": "CUDA0",
                    "device_name": "CUDA0",
                    "device_type": "gpu",
                    "compute": 4,
                },
            ),
            "backend device identities are duplicated",
            id="exact-duplicate-graph-backend-index",
        ),
        pytest.param(
            (
                {
                    "graph_uid": 2,
                    "backend_index": 0,
                    "backend_name": "CUDA0-drift",
                    "device_name": "CUDA0",
                    "device_type": "gpu",
                    "compute": 8,
                },
            ),
            "exact CUDA index and device identities",
            id="backend-name-only-drift",
        ),
        pytest.param(
            (
                {
                    "graph_uid": 2,
                    "backend_index": 0,
                    "backend_name": "CUDA0",
                    "device_name": "CUDA0",
                    "device_type": "gpu",
                    "compute": 0,
                },
            ),
            "greater than 0",
            id="zero-compute",
        ),
        *(
            pytest.param(
                (
                    {
                        "graph_uid": 2,
                        "backend_index": 0,
                        "backend_name": "CUDA0",
                        "device_name": "CUDA0",
                        "device_type": device_type,
                        "compute": 8,
                    },
                ),
                (
                    "current backend audit used a non-CUDA accelerator"
                    if device_type == "igpu"
                    else "backend identity categories differ from graph evidence"
                ),
                id=f"device-type-{device_type}",
            )
            for device_type in ("igpu", "accel", "meta", "unassigned", "cpu")
        ),
    ),
)
def test_v4_terminal_receipt_rejects_invalid_cuda_auxiliary_identity(
    identity_rows: tuple[dict[str, Any], ...],
    cause_match: str,
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    _add_v4_auxiliary_backend_graph(receipt, identity_rows)
    _reseal(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid") as error:
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )

    assert cause_match in str(error.value.__cause__)


def test_v4_terminal_receipt_rejects_cuda1_only_auxiliary_graph() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    backend_audit = receipt["server"]["backend_audit"]
    backend_audit["graphs"].append(
        {
            "graph_uid": 2,
            "phase": "post_assignment_pre_split",
            "scope": "non_view_compute",
            "compute": 25,
            "gpu": 25,
            "cpu": 0,
            "accel": 0,
            "other": 0,
            "unassigned": 0,
        }
    )
    backend_audit["identities"].append(
        {
            "graph_uid": 2,
            "backend_index": 1,
            "backend_name": "CUDA1",
            "device_name": "CUDA1",
            "device_type": "gpu",
            "compute": 25,
        }
    )
    backend_audit["observed_graphs"] = 2
    backend_audit["compute_operations"] = 125
    backend_audit["gpu_operations"] = 125
    _reseal(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid") as error:
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )

    assert "exact CUDA index and device identities" in str(error.value.__cause__)


@pytest.mark.parametrize("mode", ("swapped", "duplicated"))
def test_v4_terminal_receipt_rejects_invalid_cuda_identity_pairs(mode: str) -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    identities = receipt["server"]["backend_audit"]["identities"]
    if mode == "swapped":
        identities[0]["device_name"] = "CUDA1"
        identities[1]["device_name"] = "CUDA0"
    else:
        identities[1]["backend_name"] = "CUDA0"
        identities[1]["device_name"] = "CUDA0"
    _reseal(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid") as error:
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )

    assert "exact CUDA index and device identities" in str(error.value.__cause__)


def test_v4_terminal_receipt_rejects_backend_index_identity_drift() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    backend_audit = receipt["server"]["backend_audit"]
    second_graph = copy.deepcopy(backend_audit["graphs"][0])
    second_graph["graph_uid"] = 2
    backend_audit["graphs"].append(second_graph)
    for identity in copy.deepcopy(backend_audit["identities"]):
        identity["graph_uid"] = 2
        identity["backend_index"] = 1 - identity["backend_index"]
        backend_audit["identities"].append(identity)
    backend_audit["observed_graphs"] = 2
    backend_audit["compute_operations"] = 200
    backend_audit["gpu_operations"] = 200
    _reseal(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid") as error:
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )

    assert "exact CUDA index and device identities" in str(error.value.__cause__)


def test_v4_terminal_receipt_rejects_historical_source_patch_inventory() -> None:
    receipt, config, reference, control_plane, run_id = _valid_v4_receipt()
    receipt["runtime"]["patched_source_paths"].remove("tools/server/server.cpp")
    receipt["runtime"]["base_source_blob_ids"] = [
        row
        for row in receipt["runtime"]["base_source_blob_ids"]
        if row["path"] != "tools/server/server.cpp"
    ]
    _reseal(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid") as error:
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )

    assert "base-source blob identities differ from their patch" in str(error.value.__cause__)


def test_terminal_receipt_accepts_explicitly_unavailable_optional_pci_evidence() -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    receipt["hardware"][0].update(
        {
            "pci_bus_id": None,
            "pci_bus_id_status": "unavailable",
            "pci_bus_id_error_code": 999,
        }
    )
    _reseal_topology_and_receipt(receipt)

    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.hardware[0].pci_bus_id is None
    assert observed.hardware[0].pci_bus_id_error_code == 999


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("cuda_ordinal", 1, "CUDA ordinals"),
        ("uuid", "GPU-00000000-0000-0000-0000-000000000002", "UUIDs"),
    ),
)
def test_terminal_receipt_rejects_duplicate_cuda_identity(
    field: str,
    replacement: object,
    match: str,
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    receipt["hardware"][0][field] = replacement
    _reseal_topology_and_receipt(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid") as error:
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )

    assert match in str(error.value.__cause__)


def test_terminal_receipt_rejects_cpu_quota_above_the_hard_request() -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    changed = copy.deepcopy(receipt)
    changed["host"].update(
        {
            "logical_cpu_count": 16,
            "logical_cpu_count_scope": "container_effective_sched_getaffinity",
            "cgroup_cpu_quota_millicores": 32_000,
            "cgroup_cpu_limit_values_millicores": [32_000],
            "cpu_affinity_ids": list(range(16)),
        }
    )
    _reseal_topology_and_receipt(changed)

    with pytest.raises(ValueError, match="receipt schema is invalid"):
        validate_smoke_terminal_receipt(
            changed,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["host"].update(
            {
                "logical_cpu_count": 15,
                "logical_cpu_count_scope": "container_cgroup_cpu_quota",
                "cgroup_cpu_quota_millicores": 15_000,
                "cgroup_cpu_limit_values_millicores": [15_000],
            }
        ),
        lambda value: value["host"].update(
            {
                "logical_cpu_count": 8,
                "logical_cpu_count_scope": "container_effective_sched_getaffinity",
                "cpu_affinity_ids": list(range(8)),
            }
        ),
        lambda value: value["host"].update(
            {
                "ram_bytes": 32 * 1024**3,
                "ram_scope": "container_cgroup_memory_limit",
                "cgroup_memory_limit_bytes": 32 * 1024**3,
                "cgroup_memory_limit_values_bytes": [32 * 1024**3],
            }
        ),
    ],
    ids=("cpu-quota-below-request", "cpu-affinity-below-request", "ram-below-request"),
)
def test_terminal_receipt_rejects_effective_capacity_below_request(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    changed = copy.deepcopy(receipt)
    mutate(changed)
    _reseal_topology_and_receipt(changed)

    with pytest.raises(ValueError):
        validate_smoke_terminal_receipt(
            changed,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["host"].update(
            {
                "ram_bytes": 128 * 1024**3,
                "cgroup_memory_limit_bytes": 128 * 1024**3,
                "cgroup_memory_limit_values_bytes": [128 * 1024**3],
            }
        ),
        lambda value: value["host"].update({"cgroup_cpu_limit_values_millicores": [None]}),
        lambda value: value["host"].update({"cgroup_memory_limit_values_bytes": [None]}),
    ],
    ids=("memory-above-hard-limit", "cpu-unlimited", "memory-unlimited"),
)
def test_terminal_receipt_rejects_non_exact_or_unlimited_hard_limits(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    changed = copy.deepcopy(receipt)
    mutate(changed)
    _reseal_topology_and_receipt(changed)

    with pytest.raises(ValueError, match="receipt schema is invalid"):
        validate_smoke_terminal_receipt(
            changed,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("artifact_rehash"),
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value["artifact_rehash"]["artifacts"][0].__setitem__("sha256", "0" * 64),
        lambda value: value["runtime"]["binaries"][0].__setitem__("sha256", "0" * 63),
        lambda value: value["runtime"].__setitem__(
            "cuda_driver_library_path",
            "/usr/local/cuda/lib64/stubs/libcuda.so.1",
        ),
        lambda value: value["runtime"]["python_packages"].__setitem__("modal", "1.5.1"),
        lambda value: value["runtime"].__setitem__("python_version", "3.13.0"),
        lambda value: value["runtime"].__setitem__("modal_runtime_version", "1.5.1"),
        lambda value: value["runtime"].__setitem__("modal_package_tree_sha256", "4" * 64),
        lambda value: value["runtime"].__setitem__(
            "package_manifest_schema_version", "inkling-smoke-package-manifest-v1"
        ),
        lambda value: value["host"].__setitem__("ram_bytes", value["host"]["ram_bytes"] + 1024),
        lambda value: value["hardware"][0].__setitem__("cuda_driver_name", "NVIDIA H100"),
        lambda value: value["probes"].pop(),
        lambda value: value["probes"][0].__setitem__("repeatable_greedy_token_ids", False),
        lambda value: value["probes"][0]["trials"][0]["timings"].__setitem__("predicted_ms", 0.0),
        lambda value: value["probes"][0]["trials"][0].__setitem__(
            "all_returned_logprobs_finite", False
        ),
        lambda value: value["server"]["raw_logit_audit"].__setitem__(
            "observed_generated_token_vectors", 11
        ),
        lambda value: value["server"]["raw_logit_audit"]["rows"][0].__setitem__("nan", 1),
        lambda value: value["server"]["raw_logit_audit"]["rows"][1].update(
            value["server"]["raw_logit_audit"]["rows"][0]
        ),
        lambda value: value["server"]["backend_audit"].__setitem__("cpu_operations", 1),
        lambda value: value["server"]["backend_audit"]["graphs"].append(
            copy.deepcopy(value["server"]["backend_audit"]["graphs"][0])
        ),
        lambda value: value["server"]["backend_audit"]["identities"][0].__setitem__("compute", 49),
        lambda value: value["server"]["artifact_load"].__setitem__("total_shards_loaded", 48),
        lambda value: value["server"]["artifact_load"]["text_load"].__setitem__(
            "size_done", 123_455
        ),
        lambda value: value["server"]["artifact_load"]["projector_tensors"].reverse(),
        lambda value: value["server"].__setitem__("post_rehash_load_to_health_seconds", 0.0),
        lambda value: value["resources"].pop("server_peak_host_rss_mib"),
        lambda value: value["claims"].__setitem__("quality_measured", True),
        lambda value: value["claims"].__setitem__("quality_measured", 0),
        lambda value: value["runtime"].pop("base_source_blob_ids"),
        lambda value: value["server"]["audit_environment"].__setitem__(
            "IQL_SMOKE_BACKEND_AUDIT", "0"
        ),
        lambda value: value["probes"][1].__setitem__("fixture_size_bytes", 87),
        lambda value: value["server"]["command"].__setitem__(4, "/wrong/model.gguf"),
        lambda value: value.__setitem__("prompt_text_recorded", True),
    ],
    ids=(
        "missing-provenance",
        "extra-field",
        "artifact-rehash-identity",
        "runtime-binary-identity",
        "cuda-driver-stub",
        "package-manifest",
        "python-runtime-identity",
        "modal-runtime-version",
        "modal-source-tree-identity",
        "package-manifest-schema",
        "hardware-topology-hash",
        "hardware-matrix",
        "missing-modality",
        "repeatability",
        "timing",
        "finite-returned-logprobs",
        "raw-logit-coverage",
        "raw-logit-nonfinite",
        "duplicate-raw-logit-identity",
        "cpu-fallback",
        "duplicate-backend-graph",
        "backend-identity-mismatch",
        "artifact-load",
        "text-byte-accounting",
        "projector-order",
        "load-timing",
        "missing-memory",
        "claim-limit",
        "claim-flag-type",
        "missing-runtime-source-provenance",
        "audit-environment",
        "fixture-identity",
        "command",
        "prompt-privacy",
    ),
)
def test_terminal_receipt_rejects_incomplete_or_contradictory_evidence(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    receipt, config, reference, control_plane, run_id = _valid_receipt()
    changed = copy.deepcopy(receipt)
    mutate(changed)
    _reseal(changed)

    with pytest.raises(ValueError):
        validate_smoke_terminal_receipt(
            changed,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )
