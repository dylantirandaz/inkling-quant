"""CPU-only runtime contracts for one matched BF16-to-Q3 smoke run."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_matched_execution import (
    MATCHED_SUBJECT_ORDER,
    MatchedArtifactHashObservation,
    MatchedCudaPeerEdgeEvidence,
    MatchedCudaPeerTopologyEvidence,
    MatchedFailureCauseCode,
    MatchedFailureReceipt,
    MatchedGpuResourceEvidence,
    MatchedProbeEvidence,
    MatchedProbeTrialEvidence,
    MatchedPublicationState,
    MatchedSanitizedFailureDiagnostic,
    MatchedServerCleanupEvidence,
    MatchedServerCommandSpec,
    MatchedSubject,
    MatchedSubjectArtifactRehashEvidence,
    MatchedSubjectReceiptReference,
    MatchedSubjectSmokeReceipt,
    build_matched_capacity_inputs,
    build_matched_rollup_receipt,
    build_matched_server_command,
    build_matched_server_environment,
    matched_failure_receipt_sha256,
    matched_shard_inventory_sha256,
    matched_subject_smoke_receipt_sha256,
    parse_exact_cuda_backend_audit,
    parse_matched_nvidia_smi_identity_csv,
    parse_matched_nvidia_smi_monitor_csv,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    PINNED_UNPADDED_VOCAB_SIZE,
    PINNED_VOCAB_SIZE,
    parse_artifact_load_evidence,
    parse_loader_offload_evidence,
    parse_raw_logit_audit_evidence,
)

pytestmark = pytest.mark.unit

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
RUN_ID = "inkling-matched-test"
ALLOCATION_SHA256 = hashlib.sha256(b"allocation").hexdigest()
RUNTIME_SHA256 = hashlib.sha256(b"runtime").hexdigest()
PROBE_CONTROL_SHA256 = hashlib.sha256(b"probe-control").hexdigest()
COMPLETED_AT = datetime(2026, 7, 28, 12, tzinfo=UTC).isoformat()


def _uuid(index: int) -> str:
    return f"GPU-{index + 1:08x}-0000-0000-0000-{index + 1:012x}"


def _subject_prefix(subject: MatchedSubject) -> str:
    return "inkling-BF16" if subject is MatchedSubject.BF16 else "inkling-Q3_K_M"


def _artifact(
    subject: MatchedSubject,
    *,
    kind: str,
    relative_path: str,
    sha256: str,
    size_bytes: int,
    shard_ordinal: int | None = None,
) -> MatchedArtifactHashObservation:
    mount = "/baseline" if subject is MatchedSubject.BF16 else "/final"
    return MatchedArtifactHashObservation.model_validate(
        {
            "subject": subject,
            "kind": kind,
            "relative_path": relative_path,
            "absolute_path": f"{mount}/{relative_path}",
            "shard_ordinal": shard_ordinal,
            "expected_sha256": sha256,
            "observed_sha256": sha256,
            "expected_size_bytes": size_bytes,
            "observed_size_bytes": size_bytes,
            "hash_matches": True,
            "size_matches": True,
        }
    )


def _artifact_evidence(subject: MatchedSubject) -> MatchedSubjectArtifactRehashEvidence:
    prefix = _subject_prefix(subject)
    base = "bf16" if subject is MatchedSubject.BF16 else "q3_k_m"
    shards = tuple(
        _artifact(
            subject,
            kind="text_shard",
            relative_path=f"{base}/{prefix}-{ordinal:05d}-of-00049.gguf",
            sha256=f"{ordinal:064x}",
            size_bytes=ordinal * 100,
            shard_ordinal=ordinal,
        )
        for ordinal in range(1, 50)
    )
    projector = _artifact(
        MatchedSubject.Q3,
        kind="projector",
        relative_path="mmproj/mmproj-BF16.gguf",
        sha256=SHA_A,
        size_bytes=10_000,
    )
    if subject is MatchedSubject.BF16:
        assignments = (
            *shards,
            _artifact(
                subject,
                kind="receipt",
                relative_path="convert_text_bf16.success.json",
                sha256=SHA_B,
                size_bytes=200,
            ),
        )
    else:
        assignments = (
            *shards,
            projector,
            _artifact(
                subject,
                kind="manifest",
                relative_path="q3_k_m/export_manifest.json",
                sha256=SHA_B,
                size_bytes=201,
            ),
            *(
                _artifact(
                    subject,
                    kind="receipt",
                    relative_path=f"receipt-{index}.json",
                    sha256=f"{60 + index:064x}",
                    size_bytes=210 + index,
                )
                for index in range(3)
            ),
            *(
                _artifact(
                    subject,
                    kind="tokenizer",
                    relative_path=f"tokenizer-{index}.json",
                    sha256=f"{70 + index:064x}",
                    size_bytes=220 + index,
                )
                for index in range(6)
            ),
        )
    inventory_sha256 = matched_shard_inventory_sha256(subject, shards)
    return MatchedSubjectArtifactRehashEvidence(
        schema_version="inkling-matched-artifact-rehash-v1",
        subject=subject,
        subject_reference_sha256=SHA_C if subject is MatchedSubject.BF16 else SHA_D,
        assignments=assignments,
        assignment_count=len(assignments),
        text_shard_count=49,
        text_shard_total_bytes=sum(item.observed_size_bytes for item in shards),
        expected_text_shard_inventory_sha256=inventory_sha256,
        observed_text_shard_inventory_sha256=inventory_sha256,
        first_shard_path=shards[0].absolute_path,
        metadata_only_first_shard=True,
        shared_projector=projector,
        rehash_completed=True,
        all_hashes_match=True,
    )


def _graph_line(
    graph_uid: int,
    owner: str,
    *,
    compute: int,
) -> str:
    return (
        f"IQL_SMOKE_BACKEND_GRAPH_V2 graph_uid={graph_uid} graph_owner={owner} "
        "phase=post_assignment_pre_split scope=non_view_compute "
        f"compute={compute} gpu={compute} cpu=0 accel=0 other=0 unassigned=0"
    )


def _identity_line(
    graph_uid: int,
    owner: str,
    ordinal: int,
    *,
    compute: int,
) -> str:
    return (
        f"IQL_SMOKE_BACKEND_IDENTITY_V2 graph_uid={graph_uid} graph_owner={owner} "
        f"backend_index={ordinal} backend_name=CUDA{ordinal} device_name=CUDA{ordinal} "
        f"device_type=gpu compute={compute}"
    )


def _backend_log() -> str:
    lines: list[str] = []
    for ordinal in range(8):
        lines.append(_identity_line(1, "text", ordinal, compute=ordinal + 1))
    lines.append(_graph_line(1, "text", compute=sum(range(1, 9))))
    for graph_uid, owner in ((2, "vision"), (3, "audio")):
        lines.append(_identity_line(graph_uid, owner, 0, compute=2))
        lines.append(_graph_line(graph_uid, owner, compute=2))
    return "\n".join(lines)


def _loader_log(subject: MatchedSubject) -> str:
    artifact = _artifact_evidence(subject)
    first_shard = artifact.first_shard_path
    projector = artifact.shared_projector.absolute_path
    return "\n".join(
        (
            "ggml_cuda_init: found 8 CUDA devices (Total VRAM: 2140 GiB):",
            "load_tensors: offloading output layer to GPU",
            "load_tensors: offloaded 128/128 layers to GPU",
            f"llama_model_loader: loaded meta data with 57 key-value pairs from {first_shard}",
            "llama_model_loader: additional 48 GGUFs metadata loaded.",
            f"srv load_model: loaded multimodal model, '{projector}'",
            "IQL_SMOKE_TEXT_SHARDS_V1 expected=49 opened=49 contexts=49 tensors=1200",
            "IQL_SMOKE_TEXT_LOAD_V2 opened=49 accounted=49 tensors=1200 "
            "bytes=451035400288 size_done=451035400288 "
            "size_data=451035400288 mmap=1",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=vision projector=inkling "
            "tensors=10 bytes=90000000",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=audio projector=inkling "
            "tensors=12 bytes=93264288",
            "IQL_SMOKE_PROJECTOR_READY_V1 opened=1 vision=1 audio=1 "
            "vision_type=inkling audio_type=inkling n_embd=7168",
        )
    )


def _raw_logit_marker(
    index: int,
    *,
    vocab_size: int = PINNED_VOCAB_SIZE,
    unpadded_vocab_size: int = PINNED_UNPADDED_VOCAB_SIZE,
) -> str:
    padded_vocab_size = vocab_size - unpadded_vocab_size
    return (
        f"IQL_SMOKE_RAW_LOGITS_V2 task_id={index} slot_id=0 "
        f"completion_index={index + 1} batch_index=0 count={vocab_size} "
        f"unpadded_count={unpadded_vocab_size} padded_count={padded_vocab_size} "
        f"unpadded_finite={unpadded_vocab_size} unpadded_nan=0 "
        "unpadded_pos_inf=0 unpadded_neg_inf=0 padded_finite=0 padded_nan=0 "
        f"padded_pos_inf=0 padded_neg_inf={padded_vocab_size}"
    )


def _raw_logit_evidence(
    generated_vectors: int,
    *,
    vocab_size: int = PINNED_VOCAB_SIZE,
    unpadded_vocab_size: int = PINNED_UNPADDED_VOCAB_SIZE,
):
    return parse_raw_logit_audit_evidence(
        "\n".join(
            _raw_logit_marker(
                index,
                vocab_size=vocab_size,
                unpadded_vocab_size=unpadded_vocab_size,
            )
            for index in range(generated_vectors)
        ),
        expected_generated_token_vectors=generated_vectors,
        vocab_size=vocab_size,
        unpadded_vocab_size=unpadded_vocab_size,
    )


def _trial(index: int, token_ids: tuple[int, ...] = (10, 11, 12)) -> MatchedProbeTrialEvidence:
    return MatchedProbeTrialEvidence(
        trial_index=index,
        token_ids=token_ids,
        generated_token_count=len(token_ids),
        minimum_logprob=-2.0,
        maximum_logprob=-0.1,
        mean_logprob=-0.7,
        prompt_processing_ms=8.0,
        decode_ms=20.0,
        response_sha256=f"{100 + index:064x}",
        finite_logits=True,
        valid_token_ids=True,
    )


def _probe(modality: str, index: int) -> MatchedProbeEvidence:
    fixture_sha256 = None if modality == "text" else f"{120 + index:064x}"
    fixture_size = None if modality == "text" else 100 + index
    return MatchedProbeEvidence.model_validate(
        {
            "probe_id": f"{modality}_greedy_v1",
            "modality": modality,
            "prompt_sha256": f"{130 + index:064x}",
            "fixture_sha256": fixture_sha256,
            "fixture_size_bytes": fixture_size,
            "seed": 42,
            "temperature": 0.0,
            "n_predict": 8,
            "n_probs": 5,
            "usable_vocab_size": 200_058,
            "trials": (_trial(1), _trial(2)),
            "repeatable_greedy_token_ids": True,
            "prompt_text_recorded": False,
            "output_text_recorded": False,
        }
    )


def _resource_evidence() -> MatchedGpuResourceEvidence:
    return MatchedGpuResourceEvidence(
        schema_version="inkling-matched-resource-evidence-v1",
        sampling_interval_seconds=1.0,
        sample_count=2,
        server_peak_host_rss_mib=8_192,
        gpu_peak_memory_used_mib=(100,) * 8,
        gpu_peak_utilization_percent=(50,) * 8,
    )


def _server_spec(subject: MatchedSubject) -> MatchedServerCommandSpec:
    artifact = _artifact_evidence(subject)
    return MatchedServerCommandSpec(
        schema_version="inkling-matched-server-command-v1",
        subject=subject,
        server_binary="/opt/llama.cpp/build/bin/llama-server",
        first_shard_path=artifact.first_shard_path,
        projector_path=artifact.shared_projector.absolute_path,
        host="127.0.0.1",
        port=18_080 if subject is MatchedSubject.BF16 else 18_081,
        server_log_path=f"/tmp/inkling-matched-{subject.value}-llama-server.log",
        endpoint="/completion",
        log_verbosity=4,
        context_size=8_192,
    )


def _subject_receipt(
    subject: MatchedSubject,
    *,
    process_id: int,
    allocation_sha256: str = ALLOCATION_SHA256,
    runtime_sha256: str = RUNTIME_SHA256,
    probe_control_sha256: str = PROBE_CONTROL_SHA256,
) -> MatchedSubjectSmokeReceipt:
    spec = _server_spec(subject)
    loader_log = _loader_log(subject)
    probes = (_probe("text", 1), _probe("image", 2), _probe("audio", 3))
    generated_vectors = sum(
        trial.generated_token_count for probe in probes for trial in probe.trials
    )
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-subject-smoke-v1",
        "status": "passed",
        "stage": "matched_smoke",
        "run_id": RUN_ID,
        "subject": subject,
        "subject_ordinal": 0 if subject is MatchedSubject.BF16 else 1,
        "allocation_identity_sha256": allocation_sha256,
        "runtime_identity_sha256": runtime_sha256,
        "probe_control_sha256": probe_control_sha256,
        "server_process_id": process_id,
        "server_command": build_matched_server_command(spec),
        "server_log_sha256": SHA_E,
        "load_time_seconds": 10.0,
        "artifact_rehash": _artifact_evidence(subject),
        "loader_offload": parse_loader_offload_evidence(
            loader_log,
            expected_gpu_count=8,
        ),
        "artifact_load": parse_artifact_load_evidence(
            loader_log,
            expected_first_shard_path=spec.first_shard_path,
            expected_projector_path=spec.projector_path,
        ),
        "raw_logit_audit": _raw_logit_evidence(generated_vectors),
        "backend_audit": parse_exact_cuda_backend_audit(
            _backend_log(),
            policy=_eight_gpu_policy(),
        ),
        "probes": probes,
        "resources": _resource_evidence(),
        "cleanup": MatchedServerCleanupEvidence(
            monitor_stopped_before_server=True,
            monitor_joined=True,
            server_stopped=True,
            process_exit_observed=True,
            log_scan_completed=True,
        ),
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "raw_server_log_recorded": False,
        "quality_measured": False,
        "benchmark_measured": False,
        "completed_at_utc": COMPLETED_AT,
    }
    payload["receipt_sha256"] = matched_subject_smoke_receipt_sha256(payload)
    return MatchedSubjectSmokeReceipt.model_validate(payload)


def _subject_reference(
    subject: MatchedSubject,
    *,
    allocation_sha256: str = ALLOCATION_SHA256,
    runtime_sha256: str = RUNTIME_SHA256,
    probe_control_sha256: str = PROBE_CONTROL_SHA256,
) -> MatchedSubjectReceiptReference:
    receipt = _subject_receipt(
        subject,
        process_id=101 if subject is MatchedSubject.BF16 else 202,
        allocation_sha256=allocation_sha256,
        runtime_sha256=runtime_sha256,
        probe_control_sha256=probe_control_sha256,
    )
    projector = receipt.artifact_rehash.shared_projector
    return MatchedSubjectReceiptReference(
        subject=receipt.subject,
        subject_ordinal=receipt.subject_ordinal,
        receipt_sha256=receipt.receipt_sha256,
        server_process_id=receipt.server_process_id,
        allocation_identity_sha256=receipt.allocation_identity_sha256,
        runtime_identity_sha256=receipt.runtime_identity_sha256,
        probe_control_sha256=receipt.probe_control_sha256,
        projector_path=projector.absolute_path,
        projector_sha256=projector.observed_sha256,
    )


def _eight_gpu_policy():
    from inkling_quant_lab.gguf.inkling_matched_execution import ExactCudaPlacementPolicy

    return ExactCudaPlacementPolicy(
        schema_version="iql-exact-cuda-placement-policy-v1",
        gpu_count=8,
        tensor_split=(1,) * 8,
        split_mode="layer",
        text_graph_policy="at_least_one_all_expected_cuda",
        vision_graph_policy="cuda0_only",
        audio_graph_policy="cuda0_only",
    )


def test_subject_order_and_server_commands_are_exact_and_isolated() -> None:
    assert MATCHED_SUBJECT_ORDER == (MatchedSubject.BF16, MatchedSubject.Q3)

    bf16_spec = _server_spec(MatchedSubject.BF16)
    q3_spec = _server_spec(MatchedSubject.Q3)
    bf16 = build_matched_server_command(bf16_spec)
    q3 = build_matched_server_command(q3_spec)

    assert bf16[0] == q3[0] == "/opt/llama.cpp/build/bin/llama-server"
    assert bf16[bf16.index("--model") + 1].endswith("/bf16/inkling-BF16-00001-of-00049.gguf")
    assert q3[q3.index("--model") + 1].endswith("/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf")
    assert bf16[bf16.index("--mmproj") + 1] == q3[q3.index("--mmproj") + 1]
    assert bf16[bf16.index("--port") + 1] == "18080"
    assert q3[q3.index("--port") + 1] == "18081"
    assert bf16[bf16.index("--n-gpu-layers") + 1] == "all"
    assert bf16[bf16.index("--n-cpu-moe") + 1] == "0"
    assert bf16[bf16.index("--tensor-split") + 1] == "1,1,1,1,1,1,1,1"
    assert bf16_spec.server_log_path != q3_spec.server_log_path


def test_server_spec_rejects_wrong_port_log_or_non_first_shard() -> None:
    payload = _server_spec(MatchedSubject.BF16).model_dump(mode="python")
    payload["port"] = 18_081
    with pytest.raises(ValidationError, match="port"):
        MatchedServerCommandSpec.model_validate(payload)

    payload = _server_spec(MatchedSubject.BF16).model_dump(mode="python")
    payload["first_shard_path"] = payload["first_shard_path"].replace("00001", "00002")
    with pytest.raises(ValidationError, match="first shard"):
        MatchedServerCommandSpec.model_validate(payload)

    payload = _server_spec(MatchedSubject.Q3).model_dump(mode="python")
    payload["server_log_path"] = "/tmp/inkling-matched-bf16-llama-server.log"
    with pytest.raises(ValidationError, match="log path"):
        MatchedServerCommandSpec.model_validate(payload)


def test_server_environment_scrubs_all_inherited_llama_arguments() -> None:
    environment = build_matched_server_environment(
        {
            "PATH": "/usr/bin",
            "LLAMA_ARG_MODEL": "/forged.gguf",
            "LLAMA_ARG_N_GPU_LAYERS": "0",
        },
        audit_environment={"IQL_SMOKE_GRAPH_OWNER": "text"},
    )

    assert environment == {
        "PATH": "/usr/bin",
        "IQL_SMOKE_GRAPH_OWNER": "text",
    }


def test_artifact_rehash_requires_exact_hashes_inventory_and_cardinality() -> None:
    bf16 = _artifact_evidence(MatchedSubject.BF16)
    q3 = _artifact_evidence(MatchedSubject.Q3)

    assert bf16.assignment_count == 50
    assert q3.assignment_count == 60
    assert bf16.metadata_only_first_shard is True
    assert q3.metadata_only_first_shard is True
    assert bf16.shared_projector == q3.shared_projector

    payload = bf16.model_dump(mode="python")
    observations = list(payload["assignments"])
    observations[0] = {**observations[0], "observed_sha256": SHA_A}
    payload["assignments"] = observations
    with pytest.raises(ValidationError, match="hash"):
        MatchedSubjectArtifactRehashEvidence.model_validate(payload)

    payload = q3.model_dump(mode="python")
    payload["assignments"] = payload["assignments"][:-1]
    payload["assignment_count"] -= 1
    with pytest.raises(ValidationError, match="60"):
        MatchedSubjectArtifactRehashEvidence.model_validate(payload)


def test_artifact_load_parser_binds_the_runtime_subject_paths() -> None:
    artifact = _artifact_evidence(MatchedSubject.BF16)
    evidence = parse_artifact_load_evidence(
        _loader_log(MatchedSubject.BF16),
        expected_first_shard_path=artifact.first_shard_path,
        expected_projector_path=artifact.shared_projector.absolute_path,
    )

    assert evidence.first_shard_path == artifact.first_shard_path
    assert evidence.projector_path == artifact.shared_projector.absolute_path
    assert evidence.total_shards_loaded == 49


def _identity_csv() -> str:
    return "\n".join(
        f"{_uuid(index)}, NVIDIA B300 SXM6 AC, 275040, 590.44, 10.3" for index in range(8)
    )


def test_eight_gpu_identity_monitor_and_capacity_inputs_are_exact() -> None:
    gpus = parse_matched_nvidia_smi_identity_csv(_identity_csv())
    assert tuple(gpu.cuda_ordinal for gpu in gpus) == tuple(range(8))
    assert len({gpu.uuid for gpu in gpus}) == 8

    monitor = "\n".join(
        f"{_uuid(index)}, {100 + index}, {20 + index}" for index in reversed(range(8))
    )
    samples = parse_matched_nvidia_smi_monitor_csv(
        monitor,
        expected_uuids=tuple(gpu.uuid for gpu in gpus),
    )
    assert tuple(sample.uuid for sample in samples) == tuple(gpu.uuid for gpu in gpus)
    assert tuple(sample.memory_used_mib for sample in samples) == tuple(
        100 + index for index in range(8)
    )

    capacity = build_matched_capacity_inputs(gpus)
    assert capacity.gpu_count == 8
    assert capacity.observed_gpu_memory_bytes == (275_040 * 1024 * 1024,) * 8
    assert capacity.observed_total_gpu_memory_bytes == sum(capacity.observed_gpu_memory_bytes)


@pytest.mark.parametrize(
    "payload",
    (
        "\n".join(_identity_csv().splitlines()[:7]),
        _identity_csv().replace("NVIDIA B300 SXM6 AC", "NVIDIA H100", 1),
        _identity_csv().replace("275040", "274113", 1),
        _identity_csv().replace("10.3", "9.0", 1),
    ),
)
def test_eight_gpu_identity_parser_rejects_matrix_drift(payload: str) -> None:
    with pytest.raises((ValidationError, ValueError)):
        parse_matched_nvidia_smi_identity_csv(payload)


def _peer_edges() -> tuple[MatchedCudaPeerEdgeEvidence, ...]:
    return tuple(
        MatchedCudaPeerEdgeEvidence(
            source_cuda_ordinal=source,
            source_uuid=_uuid(source),
            destination_cuda_ordinal=destination,
            destination_uuid=_uuid(destination),
            can_access_peer=True,
            performance_rank=0,
            access_supported=True,
            native_atomic_supported=True,
            cuda_array_access_supported=True,
            only_partial_native_atomic_supported=False,
        )
        for source in range(8)
        for destination in range(8)
        if source != destination
    )


def test_eight_gpu_topology_requires_all_56_ordered_directed_edges() -> None:
    topology = MatchedCudaPeerTopologyEvidence(
        schema_version="inkling-matched-cuda-peer-topology-v1",
        protocol="cuda-driver-p2p-attributes-v1",
        cuda_driver_api_version=13_100,
        gpu_uuids=tuple(_uuid(index) for index in range(8)),
        edges=_peer_edges(),
    )
    assert len(topology.edges) == 56

    with pytest.raises(ValidationError, match="56"):
        MatchedCudaPeerTopologyEvidence(
            schema_version="inkling-matched-cuda-peer-topology-v1",
            protocol="cuda-driver-p2p-attributes-v1",
            cuda_driver_api_version=13_100,
            gpu_uuids=tuple(_uuid(index) for index in range(8)),
            edges=_peer_edges()[:-1],
        )


def test_probe_evidence_requires_repeatable_tokens_and_finite_summaries() -> None:
    probe = _probe("text", 1)
    assert probe.trials[0].token_ids == probe.trials[1].token_ids

    payload = probe.model_dump(mode="python")
    trials = list(payload["trials"])
    trials[1] = {**trials[1], "token_ids": (99,), "generated_token_count": 1}
    payload["trials"] = trials
    with pytest.raises(ValidationError, match="repeatable"):
        MatchedProbeEvidence.model_validate(payload)

    payload = _trial(1).model_dump(mode="python")
    payload["mean_logprob"] = float("nan")
    with pytest.raises(ValidationError):
        MatchedProbeTrialEvidence.model_validate(payload)


def test_subject_receipts_and_rollup_enforce_same_cell_and_fresh_processes() -> None:
    bf16 = _subject_receipt(MatchedSubject.BF16, process_id=101)
    q3 = _subject_receipt(MatchedSubject.Q3, process_id=202)
    rollup = build_matched_rollup_receipt(
        run_id=RUN_ID,
        subject_receipts=(bf16, q3),
        completed_at_utc=COMPLETED_AT,
    )

    assert rollup.status == "passed"
    assert tuple(item.subject for item in rollup.subjects) == MATCHED_SUBJECT_ORDER
    assert rollup.both_subjects_passed is True
    assert rollup.prompt_text_recorded is False
    assert rollup.output_text_recorded is False
    assert rollup.quality_measured is False
    assert rollup.benchmark_measured is False

    q3_wrong_runtime = _subject_receipt(
        MatchedSubject.Q3,
        process_id=202,
        runtime_sha256=SHA_A,
    )
    with pytest.raises(ValueError, match="runtime"):
        build_matched_rollup_receipt(
            run_id=RUN_ID,
            subject_receipts=(bf16, q3_wrong_runtime),
            completed_at_utc=COMPLETED_AT,
        )

    q3_reused_process = _subject_receipt(MatchedSubject.Q3, process_id=101)
    with pytest.raises(ValueError, match="fresh"):
        build_matched_rollup_receipt(
            run_id=RUN_ID,
            subject_receipts=(bf16, q3_reused_process),
            completed_at_utc=COMPLETED_AT,
        )


def test_subject_receipt_hash_detects_tampering() -> None:
    receipt = _subject_receipt(MatchedSubject.BF16, process_id=101)
    payload = receipt.model_dump(mode="python")
    payload["load_time_seconds"] = 11.0

    with pytest.raises(ValidationError, match="hash"):
        MatchedSubjectSmokeReceipt.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "replacement", "match"),
    (
        (
            "loader_offload",
            parse_loader_offload_evidence(
                "\n".join(
                    (
                        "ggml_cuda_init: found 2 CUDA devices",
                        "load_tensors: offloading output layer to GPU",
                        "load_tensors: offloaded 128/128 layers to GPU",
                    )
                )
            ),
            "eight-GPU",
        ),
        (
            "raw_logit_audit",
            _raw_logit_evidence(17),
            "every generated token vector",
        ),
        (
            "raw_logit_audit",
            _raw_logit_evidence(18, vocab_size=1_000, unpadded_vocab_size=900),
            "exact vocabulary",
        ),
    ),
)
def test_subject_receipt_cross_checks_loader_and_raw_logit_evidence(
    field_name: str,
    replacement: object,
    match: str,
) -> None:
    receipt = _subject_receipt(MatchedSubject.BF16, process_id=101)
    payload = receipt.model_dump(mode="python")
    payload[field_name] = replacement
    payload["receipt_sha256"] = matched_subject_smoke_receipt_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ValidationError, match=match):
        MatchedSubjectSmokeReceipt.model_validate(payload)


@pytest.mark.parametrize("path_field", ("first_shard_path", "projector_path"))
def test_subject_receipt_cross_checks_loaded_artifact_paths(path_field: str) -> None:
    receipt = _subject_receipt(MatchedSubject.Q3, process_id=202)
    payload = receipt.model_dump(mode="python")
    artifact_load = dict(payload["artifact_load"])
    artifact_load[path_field] = f"/final/other/{path_field}.gguf"
    payload["artifact_load"] = artifact_load
    payload["receipt_sha256"] = matched_subject_smoke_receipt_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ValidationError, match=r"first shard|projector"):
        MatchedSubjectSmokeReceipt.model_validate(payload)


def test_subject_receipt_rejects_historical_raw_logit_evidence() -> None:
    receipt = _subject_receipt(MatchedSubject.BF16, process_id=101)
    payload = receipt.model_dump(mode="python")
    payload["raw_logit_audit"] = parse_raw_logit_audit_evidence(
        "\n".join(
            f"IQL_SMOKE_RAW_LOGITS_V1 task_id={index} slot_id=0 "
            f"completion_index={index + 1} batch_index=0 count={PINNED_VOCAB_SIZE} "
            f"finite={PINNED_VOCAB_SIZE} nan=0 pos_inf=0 neg_inf=0"
            for index in range(18)
        ),
        expected_generated_token_vectors=18,
        vocab_size=PINNED_VOCAB_SIZE,
    )
    payload["receipt_sha256"] = matched_subject_smoke_receipt_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ValidationError, match="RawLogitAuditEvidence"):
        MatchedSubjectSmokeReceipt.model_validate(payload)


def test_publication_lifecycle_suppresses_conflicting_failure_receipts() -> None:
    not_started = MatchedPublicationState(
        state="not_started",
        success_files_installed=False,
        commit_requested=False,
        volume_reloaded=False,
        mounted_readback_verified=False,
        terminal_success_proven=False,
        failure_receipt_allowed=True,
    )
    assert not_started.failure_receipt_allowed is True

    unknown = MatchedPublicationState(
        state="unknown",
        success_files_installed=True,
        commit_requested=True,
        volume_reloaded=False,
        mounted_readback_verified=False,
        terminal_success_proven=False,
        failure_receipt_allowed=False,
    )
    assert unknown.failure_receipt_allowed is False

    with pytest.raises(ValidationError):
        MatchedPublicationState(
            state="verified",
            success_files_installed=True,
            commit_requested=True,
            volume_reloaded=True,
            mounted_readback_verified=True,
            terminal_success_proven=True,
            failure_receipt_allowed=True,
        )


def test_failure_receipt_is_sanitized_self_hashed_and_pre_publication_only() -> None:
    publication = MatchedPublicationState(
        state="not_started",
        success_files_installed=False,
        commit_requested=False,
        volume_reloaded=False,
        mounted_readback_verified=False,
        terminal_success_proven=False,
        failure_receipt_allowed=True,
    )
    diagnostic = MatchedSanitizedFailureDiagnostic(
        schema_version="inkling-matched-sanitized-failure-v1",
        category="server_start",
        failure_type="RuntimeError",
        subject=MatchedSubject.BF16,
        cause_code=MatchedFailureCauseCode.SERVER_START_FAILED,
        message_sha256=hashlib.sha256(b"server failed").hexdigest(),
        raw_message_recorded=False,
        traceback_recorded=False,
        raw_server_log_recorded=False,
    )
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-failure-v1",
        "status": "failed",
        "stage": "matched_smoke",
        "run_id": RUN_ID,
        "allocation_identity_sha256": ALLOCATION_SHA256,
        "runtime_identity_sha256": RUNTIME_SHA256,
        "probe_control_sha256": PROBE_CONTROL_SHA256,
        "subject_at_failure": MatchedSubject.BF16,
        "completed_subject_receipts": (),
        "diagnostic": diagnostic,
        "publication": publication,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "completed_at_utc": COMPLETED_AT,
    }
    payload["receipt_sha256"] = matched_failure_receipt_sha256(payload)
    receipt = MatchedFailureReceipt.model_validate(payload)
    serialized = json.dumps(receipt.model_dump(mode="json"), sort_keys=True)
    assert "server failed" not in serialized

    tampered = receipt.model_dump(mode="python")
    tampered["subject_at_failure"] = MatchedSubject.Q3
    with pytest.raises(ValidationError, match="hash"):
        MatchedFailureReceipt.model_validate(tampered)


def test_failure_diagnostic_retains_only_allowlisted_cause_and_safe_artifact_path() -> None:
    diagnostic = MatchedSanitizedFailureDiagnostic(
        schema_version="inkling-matched-sanitized-failure-v1",
        category="artifact_rehash",
        failure_type="OSError",
        subject=MatchedSubject.Q3,
        cause_code=MatchedFailureCauseCode.ARTIFACT_HASH_MISMATCH,
        artifact_path="q3_k_m/inkling-Q3_K_M-00015-of-00049.gguf",
        message_sha256=SHA_A,
        raw_message_recorded=False,
        traceback_recorded=False,
        raw_server_log_recorded=False,
    )
    assert diagnostic.artifact_path == "q3_k_m/inkling-Q3_K_M-00015-of-00049.gguf"

    for unsafe in (
        "/final/q3_k_m/model.gguf",
        "../q3_k_m/model.gguf",
        "q3_k_m\\model.gguf",
    ):
        with pytest.raises(ValidationError, match="relative"):
            MatchedSanitizedFailureDiagnostic.model_validate(
                {
                    **diagnostic.model_dump(mode="python"),
                    "artifact_path": unsafe,
                }
            )

    with pytest.raises(ValidationError, match="requires a safe relative artifact path"):
        MatchedSanitizedFailureDiagnostic.model_validate(
            {
                **diagnostic.model_dump(mode="python"),
                "artifact_path": None,
            }
        )

    with pytest.raises(ValidationError, match="only artifact-rehash failures"):
        MatchedSanitizedFailureDiagnostic.model_validate(
            {
                **diagnostic.model_dump(mode="python"),
                "category": "server_start",
                "cause_code": MatchedFailureCauseCode.SERVER_START_FAILED,
            }
        )

    with pytest.raises(ValidationError, match="incompatible with its category"):
        MatchedSanitizedFailureDiagnostic.model_validate(
            {
                **diagnostic.model_dump(mode="python"),
                "artifact_path": None,
                "category": "server_start",
                "cause_code": MatchedFailureCauseCode.COMPLETION_CONTRACT_FAILED,
            }
        )


def _failure_payload(
    *,
    category: str,
    completed_subject_receipts: tuple[MatchedSubjectReceiptReference, ...],
) -> dict[str, object]:
    publication = MatchedPublicationState(
        state="not_started",
        success_files_installed=False,
        commit_requested=False,
        volume_reloaded=False,
        mounted_readback_verified=False,
        terminal_success_proven=False,
        failure_receipt_allowed=True,
    )
    diagnostic = MatchedSanitizedFailureDiagnostic(
        schema_version="inkling-matched-sanitized-failure-v1",
        category=category,
        failure_type="RuntimeError",
        subject=MatchedSubject.Q3,
        cause_code=(
            MatchedFailureCauseCode.PUBLICATION_FAILED
            if category == "publication"
            else MatchedFailureCauseCode.COMPLETION_CONTRACT_FAILED
        ),
        message_sha256=hashlib.sha256(b"matched failure").hexdigest(),
        raw_message_recorded=False,
        traceback_recorded=False,
        raw_server_log_recorded=False,
    )
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-failure-v1",
        "status": "failed",
        "stage": "matched_smoke",
        "run_id": RUN_ID,
        "allocation_identity_sha256": ALLOCATION_SHA256,
        "runtime_identity_sha256": RUNTIME_SHA256,
        "probe_control_sha256": PROBE_CONTROL_SHA256,
        "subject_at_failure": MatchedSubject.Q3,
        "completed_subject_receipts": completed_subject_receipts,
        "diagnostic": diagnostic,
        "publication": publication,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "completed_at_utc": COMPLETED_AT,
    }
    payload["receipt_sha256"] = matched_failure_receipt_sha256(payload)
    return payload


def test_failure_receipt_allows_both_subjects_only_for_publication_failure() -> None:
    references = (
        _subject_reference(MatchedSubject.BF16),
        _subject_reference(MatchedSubject.Q3),
    )
    receipt = MatchedFailureReceipt.model_validate(
        _failure_payload(
            category="publication",
            completed_subject_receipts=references,
        )
    )
    assert tuple(reference.subject for reference in receipt.completed_subject_receipts) == (
        MatchedSubject.BF16,
        MatchedSubject.Q3,
    )

    with pytest.raises(
        ValidationError,
        match="two completed subject receipts are valid only for publication failure",
    ):
        MatchedFailureReceipt.model_validate(
            _failure_payload(
                category="probe",
                completed_subject_receipts=references,
            )
        )


def test_failure_receipt_rejects_wrong_two_subject_order_and_cell() -> None:
    references = (
        _subject_reference(MatchedSubject.BF16),
        _subject_reference(MatchedSubject.Q3),
    )
    with pytest.raises(ValidationError, match="fixed execution order"):
        MatchedFailureReceipt.model_validate(
            _failure_payload(
                category="publication",
                completed_subject_receipts=tuple(reversed(references)),
            )
        )

    wrong_cell = (
        references[0],
        _subject_reference(
            MatchedSubject.Q3,
            runtime_sha256=hashlib.sha256(b"other-runtime").hexdigest(),
        ),
    )
    with pytest.raises(ValidationError, match="different matched cell"):
        MatchedFailureReceipt.model_validate(
            _failure_payload(
                category="publication",
                completed_subject_receipts=wrong_cell,
            )
        )
