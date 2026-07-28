"""Offline contracts for the exact matched Inkling runtime cell.

This module does not start Modal or execute llama.cpp. It binds the retained
BF16 control, verified Q3 candidate, tokenizer files, runtime binaries, and the
first eight-B300 capacity screen. A passing capacity screen is necessary, but
it does not prove that llama.cpp can place or run either model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import CapabilityError, ConfigurationError
from inkling_quant_lab.gguf.inkling import (
    InklingSourceAdoptionReference,
    load_inkling_source_adoption_reference,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    EXPECTED_PROJECTOR_BYTES,
    EXPECTED_PROJECTOR_PATH,
    EXPECTED_PROJECTOR_SHA256,
    EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256,
    InklingVerifiedExportReference,
    VerifiedExportArtifact,
    load_verified_export_reference,
)
from inkling_quant_lab.security import sensitive_literal_path

BF16_SUBJECT_REFERENCE_RELATIVE_PATH: Final = (
    "configs/experiments/inkling_bf16_subject_reference.json"
)
MATCHED_CELL_CONFIG_RELATIVE_PATH: Final = (
    "configs/experiments/inkling_q3_k_m_matched_cell_modal.yaml"
)
BF16_SUBJECT_REFERENCE_HASH_DOMAIN: Final = b"inkling-bf16-subject-reference-v1\0"
EXPECTED_BF16_SUBJECT_REFERENCE_SHA256: Final = (
    "3615928f79ae24cdeee9164ac2c4dc3e3ba1999bde74c5b0d631e3aae4f1a53d"
)

EXPECTED_BF16_SHARD_COUNT: Final = 49
EXPECTED_BF16_TOTAL_BYTES: Final = 1_894_278_547_552
EXPECTED_BF16_INVENTORY_SHA256: Final = (
    "4feca6069b09a33962a5321e63e18736c451bbc3270c8713871dd4bd94de007e"
)
EXPECTED_BF16_CONVERSION_RECEIPT_SHA256: Final = (
    "9b31468e3013214be8434fc26ab61a468bf8e687d8850c4efa823f1fb9826f36"
)
EXPECTED_SOURCE_INVENTORY_SHA256: Final = (
    "a8aa37efec2b12c5d584c8163111d3a8a22d9568ef01886343755a8af6ace571"
)
EXPECTED_SOURCE_ADOPTION_REFERENCE_SHA256: Final = (
    "4d808b2fcd1c8a0398c6aa15371b6522d6cfb276fe0df395ca9a8beb2293f146"
)

_EXPECTED_RUNTIME_BINARIES: Final = (
    {
        "name": "llama-cli",
        "path": "/opt/llama.cpp/build/bin/llama-cli",
        "sha256": "098d8b9c6e57f25b846c5b5b43ded5bb1194cbb3d1ce985f17bbd09c87a82dbc",
        "size_bytes": 1_246_680,
    },
    {
        "name": "llama-server",
        "path": "/opt/llama.cpp/build/bin/llama-server",
        "sha256": "e960cfe4dcb2f7e541fc0b15bf97a4c1f6feb5fc304267796ef2bdd004cd1b93",
        "size_bytes": 17_920,
    },
    {
        "name": "llama-bench",
        "path": "/opt/llama.cpp/build/bin/llama-bench",
        "sha256": "e0844ac337c419ebd8b6cee4902ba13e210a067d6fe47cb652429c71ae97382b",
        "size_bytes": 17_920,
    },
    {
        "name": "llama-perplexity",
        "path": "/opt/llama.cpp/build/bin/llama-perplexity",
        "sha256": "d04051888a157ee50a7d6286cffcc78da3a9ca5295c79aa99ea2d92672ebf733",
        "size_bytes": 15_968,
    },
)
_EXPECTED_CMAKE_DEFINITIONS: Final = (
    "GGML_CUDA=ON",
    "GGML_NATIVE=OFF",
    "LLAMA_CURL=OFF",
    "LLAMA_BUILD_UI=OFF",
    "LLAMA_USE_PREBUILT_UI=OFF",
    "CMAKE_CUDA_ARCHITECTURES=103",
    "CMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/iql-cuda-driver-link",
)
_EXPECTED_TOKENIZER_ASSETS: Final = (
    {
        "path": "chat_template.jinja",
        "sha256": "0aa1aa0c729d90176dcaa00c440c8faffca2957ffb2cc4b79456ee6d02bcf43b",
        "size_bytes": 6_294,
    },
    {
        "path": "processor_config.json",
        "sha256": "b4a3962ea5f7ec39f40b5cf14e57ce99776c3dcce4756a110f7a169809e3a04c",
        "size_bytes": 1_110,
    },
    {
        "path": "special_tokens_map.json",
        "sha256": "abc97715b4b3b30eb65ea6895afd7b529c32bd10c28901f9ddae7edc39723b0f",
        "size_bytes": 517,
    },
    {
        "path": "tiktoken/tokenizer.model",
        "sha256": "bc253fd2b702f7a6da7105eaa8f3463b2f1247e83614f23e5323b921088bed2a",
        "size_bytes": 3_615_874,
    },
    {
        "path": "tokenizer.json",
        "sha256": "9fb6333a7db8fe5da90728e741e4a3ee4ac2ae12c5dd4958cc6f31688787d3c2",
        "size_bytes": 27_875_797,
    },
    {
        "path": "tokenizer_config.json",
        "sha256": "2e36c9748a2081abb935b2e745ee22e82efa32589c2500df7e5bc0f93145cd77",
        "size_bytes": 12_111,
    },
)
_EXPECTED_PLANNED_PREFLIGHT_STAGES: Final = (
    "verify_subject_references",
    "rehash_bf16_subject",
    "rehash_q3_subject",
    "screen_aggregate_capacity",
    "smoke_bf16_subject",
    "smoke_q3_subject",
    "verify_matched_smoke_evidence",
)
CAPACITY_SCREEN_LIMITATION: Final = (
    "The aggregate capacity screen does not prove per-device tensor placement, "
    "runtime workspace fit, or successful inference."
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class BF16ConversionToolchain(StrictFrozenModel):
    """Exact toolchain identity copied from the retained conversion receipt."""

    base_image: Literal[
        "debian:bookworm-slim@sha256:"
        "7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818"
    ]
    repository: Literal["https://github.com/danielhanchen/llama.cpp.git"]
    commit: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"]
    converter_sha256: Literal["a263b251619daceb9b1832797104dc4fe04e037517839c69a5b31521d5644251"]
    gguf_split_sha256: Literal["0236c09112e0b9d56c833bffbd52bd3304a6bca76f34dd7a456b25aacf40d390"]
    python_inventory_sha256: Literal[
        "16e916c14131dc9813269706634a07d9dc1d7b3e82a60379cf9109b70df8a1d7"
    ]
    dpkg_inventory_sha256: Literal[
        "2c2f76ed3f9f7ecddbb8ae0202522050c3c2affb7793a13a5ddcec84ed1a1c5e"
    ]


class InklingBF16SubjectReference(StrictFrozenModel):
    """Self-hashed expected inventory for the retained BF16 control."""

    schema_version: Literal["inkling-bf16-subject-reference-v1"]
    recorded: Literal[True]
    conversion_status: Literal["success"]
    subject_run_id: Literal["inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    subject_config_hash: Literal["ffd466dd934005fa64d36e79e591f6351ccad709c5808828bbf0b65b90ae17fd"]
    subject_control_plane_sha256: Literal[
        "8083cf41e104b3f7164c02a1ad50ab027f630167970c4eb7e0589a6d079c1037"
    ]
    convert_call_id: Literal["fc-01KXZVFDC1CGXMBFXKTZ1ACH2F"]
    launch_intent_sha256: Literal[
        "15108c11662735a3d6d7bfe76cd2e5c4f04ae6aca6cb14a19aa993bb00607ee9"
    ]
    model_id: Literal["thinkingmachines/Inkling"]
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    license: Literal["apache-2.0"]
    work_volume: Literal["inkling-work-v1"]
    work_volume_version: Literal[1]
    work_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    storage_mutability: Literal["mutable_modal_volume"]
    dtype: Literal["BF16"]
    mtp: Literal["omitted_unsupported"]
    fresh_artifact_rehash_required: Literal[True]
    measurement_ready_without_rehash: Literal[False]
    quality_measured: Literal[False]
    deployment_benchmark_measured: Literal[False]
    bf16_shard_count: Literal[49]
    bf16_total_bytes: Literal[1894278547552]
    bf16_inventory_sha256: Literal[
        "4feca6069b09a33962a5321e63e18736c451bbc3270c8713871dd4bd94de007e"
    ]
    bf16_shards: tuple[VerifiedExportArtifact, ...]
    projector: VerifiedExportArtifact
    conversion_receipt: VerifiedExportArtifact
    source_adoption_reference_sha256: Literal[
        "4d808b2fcd1c8a0398c6aa15371b6522d6cfb276fe0df395ca9a8beb2293f146"
    ]
    source_inventory_sha256: Literal[
        "a8aa37efec2b12c5d584c8163111d3a8a22d9568ef01886343755a8af6ace571"
    ]
    source_inventory_size_bytes: Literal[25106]
    source_receipt_sha256: Literal[
        "9d95c928dd47e75a41ab9c493613d44d9bcfc1a344350d3b5a35cc226d4b9fdf"
    ]
    publication_intent_sha256: Literal[
        "b3789eb41f8ef6752020830c94c9fec45fe010f27c16ac2eae0341a2af272c24"
    ]
    publication_receipt_sha256: Literal[
        "fa603f6fc0c20774b23046bd1e9b4703f2044e6d9e758a67023d67a69a34e45b"
    ]
    q3_verified_export_reference_sha256: Literal[
        "9f0fae0a48058e73aab38c2b4f6c86916b69fd32343e0f7b821c7faac5b33198"
    ]
    q3_quantize_receipt_sha256: Literal[
        "c823baccc7f124ac4c8c05d01e19f0ad4c1bc0b499619840eb365edf2efaa6a5"
    ]
    toolchain: BF16ConversionToolchain
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def canonical_payload_dict(self) -> dict[str, Any]:
        """Return the content covered by the self-hash."""

        return self.model_dump(mode="json", exclude={"reference_sha256"})

    def computed_reference_sha256(self) -> str:
        """Compute the domain-separated reference hash."""

        return bf16_subject_reference_sha256(self.canonical_payload_dict())

    def canonical_json(self) -> str:
        """Serialize the complete reference as canonical JSON."""

        return _canonical_json(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def matches_retained_conversion_receipt(self) -> InklingBF16SubjectReference:
        expected_paths = tuple(
            f"bf16/inkling-BF16-{index:05d}-of-00049.gguf"
            for index in range(1, EXPECTED_BF16_SHARD_COUNT + 1)
        )
        observed_paths = tuple(item.path for item in self.bf16_shards)
        if observed_paths != expected_paths:
            raise ValueError("BF16 shard paths must be the exact ordered 49-file set")
        if len(set(observed_paths)) != EXPECTED_BF16_SHARD_COUNT:
            raise ValueError("BF16 shard paths must be unique")
        if sum(item.size_bytes for item in self.bf16_shards) != self.bf16_total_bytes:
            raise ValueError("BF16 shard sizes do not equal the retained total")
        inventory = [item.model_dump(mode="json") for item in self.bf16_shards]
        # The retained conversion receipt hashed its canonical inventory file,
        # including that file's one required trailing newline.
        inventory_bytes = (_canonical_json(inventory) + "\n").encode("utf-8")
        inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
        if inventory_sha256 != self.bf16_inventory_sha256:
            raise ValueError("BF16 shard inventory differs from the retained receipt")
        if self.projector.model_dump(mode="json") != {
            "path": EXPECTED_PROJECTOR_PATH,
            "sha256": EXPECTED_PROJECTOR_SHA256,
            "size_bytes": EXPECTED_PROJECTOR_BYTES,
        }:
            raise ValueError("BF16 subject projector differs from the verified Q3 projector")
        if self.conversion_receipt.model_dump(mode="json") != {
            "path": "convert_text_bf16.success.json",
            "sha256": EXPECTED_BF16_CONVERSION_RECEIPT_SHA256,
            "size_bytes": 10_852,
        }:
            raise ValueError("BF16 conversion receipt identity differs")
        if self.reference_sha256 != self.computed_reference_sha256():
            raise ValueError("BF16 subject reference self-hash does not match its payload")
        return self


def bf16_subject_reference_sha256(value: Mapping[str, Any]) -> str:
    """Hash a BF16 subject reference without trusting its hash field."""

    payload = dict(value)
    payload.pop("reference_sha256", None)
    return hashlib.sha256(
        BF16_SUBJECT_REFERENCE_HASH_DOMAIN + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def load_bf16_subject_reference(path: str | Path) -> InklingBF16SubjectReference:
    """Load the byte-canonical BF16 subject reference."""

    reference_path = Path(path)
    try:
        raw_bytes = reference_path.read_bytes()
        raw = json.loads(raw_bytes)
        if not isinstance(raw, Mapping):
            raise ValueError("reference root must be a JSON object")
        reference = InklingBF16SubjectReference.model_validate(raw)
    except (OSError, ValueError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load Inkling BF16 subject reference {reference_path}: {error}",
            component="inkling_matched_bf16_subject",
        ) from error
    if raw_bytes != (reference.canonical_json() + "\n").encode("utf-8"):
        raise ConfigurationError(
            "Inkling BF16 subject reference must use canonical JSON plus one newline",
            component="inkling_matched_bf16_subject",
        )
    return reference


class MatchedRuntimeBinary(StrictFrozenModel):
    """One exact executable from the accepted Q3 smoke runtime."""

    name: Literal["llama-cli", "llama-server", "llama-bench", "llama-perplexity"]
    path: str = Field(pattern=r"^/opt/llama\.cpp/build/bin/[a-z-]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


class MatchedRuntimeConfig(StrictFrozenModel):
    """Exact runtime build and placement rules for both subjects."""

    repository: Literal["https://github.com/danielhanchen/llama.cpp.git"]
    commit: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"]
    instrumentation_schema_version: Literal["inkling-llama-smoke-instrumentation-v5"]
    instrumentation_patch_path: Literal["patches/inkling-smoke-a015409.patch"]
    instrumentation_patch_sha256: Literal[
        "005f1f342511fc3fc843bdcc7be814ed8a60e67033b733eb7e7e4af53925be04"
    ]
    cuda_image: Literal["nvidia/cuda:13.1.2-devel-ubuntu24.04"]
    cuda_image_digest: Literal[
        "sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"
    ]
    platform: Literal["linux/amd64"]
    binaries: tuple[MatchedRuntimeBinary, ...]
    cmake_definitions: tuple[str, ...]
    tensor_split: tuple[
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
    ]
    split_mode: Literal["layer"]
    gpu_layers: Literal["all"]
    no_cpu_fallback: Literal[True]
    network_access: Literal[False]
    trust_remote_code: Literal[False]

    @model_validator(mode="after")
    def exact_runtime(self) -> MatchedRuntimeConfig:
        binaries = tuple(item.model_dump(mode="json") for item in self.binaries)
        if binaries != _EXPECTED_RUNTIME_BINARIES:
            raise ValueError("matched runtime binaries differ from the accepted smoke runtime")
        if self.cmake_definitions != _EXPECTED_CMAKE_DEFINITIONS:
            raise ValueError("matched runtime CMake definitions differ from the accepted build")
        return self


class MatchedResourcesConfig(StrictFrozenModel):
    """Exact provider cell and conservative aggregate capacity policy."""

    provider: Literal["modal"]
    gpu_type: Literal["B300"]
    gpu_count: Literal[8]
    compute_capability: Literal["10.3"]
    minimum_gpu_memory_bytes: Literal[287000000000]
    capacity_reserve_basis_points: Literal[1000]
    capacity_strategy: Literal["sequential_peak_plus_reserve"]
    cpu_cores: Literal[16]
    memory_gib: Literal[64]
    ephemeral_disk_mib: Literal[524288]
    startup_timeout_seconds: Literal[1800]
    max_attempts: Literal[1]


class MatchedStorageConfig(StrictFrozenModel):
    """Read-only subject mounts and isolated evidence output."""

    bf16_volume: Literal["inkling-work-v1"]
    bf16_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    bf16_mount_path: Literal["/baseline"]
    bf16_read_only: Literal[True]
    final_volume: Literal["inkling-final-v1"]
    final_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    final_mount_path: Literal["/final"]
    final_read_only: Literal[True]
    source_volume: Literal["inkling-source-v1"]
    source_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-551ab8f240-bcc168525e"]
    source_snapshot_subpath: Literal["snapshot"]
    source_mount_path: Literal["/source"]
    source_read_only: Literal[True]
    evidence_volume: Literal["inkling-matched-evidence-v1"]
    evidence_mount_path: Literal["/evidence"]
    evidence_append_only_after_success: Literal[True]


class MatchedExecutionConfig(StrictFrozenModel):
    """Reviewed plan that cannot start remote work."""

    record_status: Literal["planning_only"]
    runner_implemented: Literal[False]
    remote_execution_allowed: Literal[False]
    subject_mode: Literal["sequential_same_allocation"]
    subject_order: tuple[Literal["bf16"], Literal["q3"]]
    fresh_process_per_subject: Literal[True]
    rehash_all_subject_files: Literal[True]
    require_both_smoke_receipts: Literal[True]
    measurement_execution_allowed: Literal[False]
    planned_stages: tuple[str, ...]

    @model_validator(mode="after")
    def exact_preflight_order(self) -> MatchedExecutionConfig:
        if self.subject_order != ("bf16", "q3"):
            raise ValueError("matched subjects must run in the checked BF16 then Q3 order")
        if self.planned_stages != _EXPECTED_PLANNED_PREFLIGHT_STAGES:
            raise ValueError("matched preflight stages must use the checked fail-closed order")
        return self


class MatchedClaimLimits(StrictFrozenModel):
    """Claims forbidden before both real subject smokes pass."""

    purpose: Literal["dual_subject_runtime_smoke_prerequisite"]
    compatibility_scope: Literal["single_exact_matrix_cell"]
    runtime_fit_proven: Literal[False]
    quality_measured: Literal[False]
    performance_measured: Literal[False]
    quality_retention_claim_allowed: Literal[False]
    speedup_claim_allowed: Literal[False]
    mtp_included: Literal[False]
    mtp_supported: Literal[False]


class InklingMatchedCellConfig(StrictFrozenModel):
    """Checked data contract for the first matched eight-B300 cell."""

    schema_version: Literal["inkling-matched-cell-config-v1"]
    model_id: Literal["thinkingmachines/Inkling"]
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    bf16_subject_reference_path: Literal["configs/experiments/inkling_bf16_subject_reference.json"]
    bf16_subject_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    q3_verified_export_reference_path: Literal[
        "configs/experiments/inkling_q3_k_m_verified_export.json"
    ]
    q3_verified_export_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_adoption_reference_path: Literal[
        "configs/experiments/inkling_q3_k_m_source_adoption.json"
    ]
    source_adoption_reference_sha256: Literal[
        "4d808b2fcd1c8a0398c6aa15371b6522d6cfb276fe0df395ca9a8beb2293f146"
    ]
    source_inventory_sha256: Literal[
        "a8aa37efec2b12c5d584c8163111d3a8a22d9568ef01886343755a8af6ace571"
    ]
    tokenizer_assets: tuple[VerifiedExportArtifact, ...]
    runtime: MatchedRuntimeConfig
    resources: MatchedResourcesConfig
    storage: MatchedStorageConfig
    execution: MatchedExecutionConfig
    claims: MatchedClaimLimits

    @model_validator(mode="after")
    def exact_contract(self) -> InklingMatchedCellConfig:
        if self.bf16_subject_reference_sha256 != EXPECTED_BF16_SUBJECT_REFERENCE_SHA256:
            raise ValueError("matched config does not bind the checked BF16 subject reference")
        if self.q3_verified_export_reference_sha256 != (EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256):
            raise ValueError("matched config does not bind the checked Q3 export reference")
        tokenizer_assets = tuple(item.model_dump(mode="json") for item in self.tokenizer_assets)
        if tokenizer_assets != _EXPECTED_TOKENIZER_ASSETS:
            raise ValueError("matched tokenizer assets differ from the source inventory")
        literal_secret = sensitive_literal_path(self.model_dump(mode="json"))
        if literal_secret is not None:
            raise ValueError(
                "matched configuration contains literal credential material at "
                + ".".join(literal_secret)
            )
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Return the resolved configuration used for hashing."""

        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        """Serialize the resolved configuration deterministically."""

        return _canonical_json(self.canonical_dict())

    def config_hash(self) -> str:
        """Hash the complete resolved matched-cell configuration."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_matched_cell_config(path: str | Path) -> InklingMatchedCellConfig:
    """Load the checked matched-cell YAML without starting external work."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("matched-cell config root must be a mapping")
        return InklingMatchedCellConfig.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load Inkling matched-cell config {config_path}: {error}",
            component="inkling_matched_cell_config",
        ) from error


class MatchedSubjectPaths(StrictFrozenModel):
    """Resolved read-only paths for all files in the matched plan."""

    bf16_shards: tuple[str, ...]
    bf16_conversion_receipt: str
    q3_shards: tuple[str, ...]
    shared_projector: str
    q3_export_manifest: str
    q3_verify_receipt: str
    q3_quantize_receipt: str
    projector_conversion_receipt: str
    tokenizer_assets: tuple[str, ...]


class InklingMatchedCellBundle(StrictFrozenModel):
    """One matched plan with every referenced record loaded and checked."""

    config: InklingMatchedCellConfig
    bf16: InklingBF16SubjectReference
    q3: InklingVerifiedExportReference
    source: InklingSourceAdoptionReference
    paths: MatchedSubjectPaths


def _values_differ(left: object, right: object) -> bool:
    """Compare values whose exact types are checked at runtime."""

    return left != right


def _mount_path(mount_path: str, relative_path: str) -> str:
    """Join one checked relative artifact path to an absolute mount."""

    mount = PurePosixPath(mount_path)
    relative = PurePosixPath(relative_path)
    if (
        not mount.is_absolute()
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ConfigurationError(
            "Matched artifact path is not a contained canonical POSIX path",
            component="inkling_matched_artifact_paths",
            details={
                "mount_path": mount_path,
                "relative_path": relative_path,
            },
        )
    resolved = mount / relative
    if resolved.as_posix() != f"{mount_path.rstrip('/')}/{relative_path}":
        raise ConfigurationError(
            "Matched artifact path is not canonical",
            component="inkling_matched_artifact_paths",
        )
    return resolved.as_posix()


def resolve_matched_subject_paths(
    config: InklingMatchedCellConfig,
    bf16: InklingBF16SubjectReference,
    q3: InklingVerifiedExportReference,
) -> MatchedSubjectPaths:
    """Resolve every matched subject file against its declared read-only mount."""

    storage = config.storage
    snapshot_mount = _mount_path(
        storage.source_mount_path,
        storage.source_snapshot_subpath,
    )
    return MatchedSubjectPaths(
        bf16_shards=tuple(
            _mount_path(storage.bf16_mount_path, artifact.path) for artifact in bf16.bf16_shards
        ),
        bf16_conversion_receipt=_mount_path(
            storage.bf16_mount_path,
            bf16.conversion_receipt.path,
        ),
        q3_shards=tuple(
            _mount_path(storage.final_mount_path, artifact.path) for artifact in q3.q3_shards
        ),
        shared_projector=_mount_path(
            storage.final_mount_path,
            q3.projector.path,
        ),
        q3_export_manifest=_mount_path(
            storage.final_mount_path,
            q3.export_manifest.path,
        ),
        q3_verify_receipt=_mount_path(
            storage.final_mount_path,
            q3.verify_receipt.path,
        ),
        q3_quantize_receipt=_mount_path(
            storage.final_mount_path,
            q3.quantize_receipt.path,
        ),
        projector_conversion_receipt=_mount_path(
            storage.final_mount_path,
            q3.mmproj_receipt.path,
        ),
        tokenizer_assets=tuple(
            _mount_path(snapshot_mount, artifact.path) for artifact in config.tokenizer_assets
        ),
    )


def _project_file(project_root: Path, relative_path: str) -> Path:
    """Resolve one checked project file without permitting root escape."""

    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ConfigurationError(
            "Matched project reference path must be relative and contained",
            component="inkling_matched_cell_bundle",
            details={"relative_path": relative_path},
        )
    candidate = (project_root / Path(relative.as_posix())).resolve()
    if not candidate.is_relative_to(project_root):
        raise ConfigurationError(
            "Matched project reference path resolves outside the project root",
            component="inkling_matched_cell_bundle",
            details={"relative_path": relative_path},
        )
    return candidate


def load_matched_cell_bundle(
    project_root: str | Path,
    *,
    config_relative_path: str = MATCHED_CELL_CONFIG_RELATIVE_PATH,
) -> InklingMatchedCellBundle:
    """Load and cross-check the matched plan and all three subject records."""

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(
            f"Matched project root is not a directory: {root}",
            component="inkling_matched_cell_bundle",
        )

    config = load_matched_cell_config(_project_file(root, config_relative_path))
    bf16 = load_bf16_subject_reference(_project_file(root, config.bf16_subject_reference_path))
    q3 = load_verified_export_reference(
        _project_file(root, config.q3_verified_export_reference_path)
    )
    source = load_inkling_source_adoption_reference(
        _project_file(root, config.source_adoption_reference_path)
    )

    mismatches: list[str] = []
    if config.bf16_subject_reference_sha256 != bf16.reference_sha256:
        mismatches.append("bf16_subject_reference_sha256")
    if config.q3_verified_export_reference_sha256 != q3.reference_sha256:
        mismatches.append("q3_verified_export_reference_sha256")
    if config.source_adoption_reference_sha256 != source.reference_sha256:
        mismatches.append("source_adoption_reference_sha256")
    if bf16.q3_verified_export_reference_sha256 != q3.reference_sha256:
        mismatches.append("bf16_q3_reference_link")
    if bf16.source_adoption_reference_sha256 != source.reference_sha256:
        mismatches.append("bf16_source_reference_link")
    if (
        config.source_inventory_sha256 != source.source_inventory.sha256
        or bf16.source_inventory_sha256 != source.source_inventory.sha256
        or bf16.source_inventory_size_bytes != source.source_inventory.size_bytes
    ):
        mismatches.append("source_inventory")
    if bf16.projector != q3.projector:
        mismatches.append("shared_projector")
    if (
        bf16.model_id != q3.model_id
        or bf16.model_id != source.model_id
        or bf16.revision != q3.revision
        or bf16.revision != source.revision
        or bf16.architecture != q3.architecture
        or bf16.architecture != source.architecture
    ):
        mismatches.append("model_identity")
    if (
        _values_differ(config.model_id, bf16.model_id)
        or _values_differ(config.revision, bf16.revision)
        or _values_differ(config.architecture, bf16.architecture)
    ):
        mismatches.append("config_model_identity")
    if _values_differ(config.storage.bf16_volume, bf16.work_volume) or _values_differ(
        config.storage.bf16_run_subpath, bf16.work_run_subpath
    ):
        mismatches.append("bf16_storage")
    if _values_differ(config.storage.final_volume, q3.final_volume) or _values_differ(
        config.storage.final_run_subpath, q3.final_run_subpath
    ):
        mismatches.append("final_storage")
    source_run_subpath = (
        PurePosixPath(source.source_run_root)
        .relative_to(PurePosixPath(source.source_mount_path))
        .as_posix()
    )
    if (
        config.storage.source_volume != source.source_volume
        or config.storage.source_run_subpath != source_run_subpath
        or source.snapshot_path
        != f"{source.source_run_root}/{config.storage.source_snapshot_subpath}"
    ):
        mismatches.append("source_storage")
    if (
        config.runtime.repository != q3.llama_cpp_repository
        or config.runtime.commit != q3.llama_cpp_commit
        or config.runtime.repository != bf16.toolchain.repository
        or config.runtime.commit != bf16.toolchain.commit
        or config.runtime.repository != source.toolchain_repository
        or config.runtime.commit != source.toolchain_commit
    ):
        mismatches.append("runtime_identity")
    if mismatches:
        raise ConfigurationError(
            "Matched plan references are not one exact subject bundle",
            component="inkling_matched_cell_bundle",
            details={"mismatches": sorted(set(mismatches))},
        )

    return InklingMatchedCellBundle(
        config=config,
        bf16=bf16,
        q3=q3,
        source=source,
        paths=resolve_matched_subject_paths(config, bf16, q3),
    )


class MatchedCapacityResult(StrictFrozenModel):
    """Necessary aggregate capacity result for sequential subject loading."""

    schema_version: Literal["inkling-matched-capacity-v1"]
    status: Literal["pass"]
    gpu_count: Literal[8]
    observed_gpu_memory_bytes: tuple[int, ...]
    observed_total_gpu_memory_bytes: int
    required_headroom_bytes: int
    usable_gpu_memory_bytes: int
    bf16_subject_bytes: int
    q3_subject_bytes: int
    sequential_peak_subject_bytes: int
    aggregate_remaining_bytes: int
    runtime_fit_proven: Literal[False]
    limitation: Literal[
        "The aggregate capacity screen does not prove per-device tensor placement, "
        "runtime workspace fit, or successful inference."
    ]

    @field_validator("observed_gpu_memory_bytes")
    @classmethod
    def exact_observed_memory_set(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != 8 or any(type(item) is not int or item <= 0 for item in value):
            raise ValueError("capacity evidence requires eight positive integer memory values")
        return value


def screen_matched_capacity(
    config: InklingMatchedCellConfig,
    bf16: InklingBF16SubjectReference,
    q3: InklingVerifiedExportReference,
    *,
    observed_gpu_memory_bytes: Sequence[int],
) -> MatchedCapacityResult:
    """Apply the conservative aggregate screen to one observed GPU allocation."""

    observed = tuple(observed_gpu_memory_bytes)
    expected_count = config.resources.gpu_count
    if len(observed) != expected_count:
        raise CapabilityError(
            "Matched capacity screen observed the wrong GPU count",
            component="inkling_matched_capacity",
            remediation="Request the exact checked eight-B300 cell.",
            details={
                "expected_gpu_count": expected_count,
                "observed_gpu_count": len(observed),
            },
        )
    if any(type(item) is not int or item <= 0 for item in observed):
        raise CapabilityError(
            "Matched capacity screen requires positive integer GPU memory values",
            component="inkling_matched_capacity",
        )
    if config.bf16_subject_reference_sha256 != bf16.reference_sha256:
        raise ConfigurationError(
            "Matched config and BF16 subject reference hashes differ",
            component="inkling_matched_capacity",
        )
    if config.q3_verified_export_reference_sha256 != q3.reference_sha256:
        raise ConfigurationError(
            "Matched config and Q3 export reference hashes differ",
            component="inkling_matched_capacity",
        )
    if (
        bf16.model_id != q3.model_id
        or bf16.revision != q3.revision
        or bf16.architecture != q3.architecture
    ):
        raise ConfigurationError(
            "BF16 and Q3 subject model identities differ",
            component="inkling_matched_capacity",
        )

    observed_total = sum(observed)
    reserve_numerator = observed_total * config.resources.capacity_reserve_basis_points
    required_headroom = (reserve_numerator + 9_999) // 10_000
    usable_capacity = observed_total - required_headroom
    bf16_subject_bytes = bf16.bf16_total_bytes + bf16.projector.size_bytes
    q3_subject_bytes = q3.q3_total_bytes + q3.projector.size_bytes
    sequential_peak = max(bf16_subject_bytes, q3_subject_bytes)
    remaining = usable_capacity - sequential_peak
    below_minimum = [
        index
        for index, size_bytes in enumerate(observed)
        if size_bytes < config.resources.minimum_gpu_memory_bytes
    ]
    if below_minimum or remaining < 0:
        raise CapabilityError(
            "Observed GPU memory does not satisfy the matched capacity policy",
            component="inkling_matched_capacity",
            remediation=(
                "Use the exact checked cell or revise the protocol in a separately reviewed change."
            ),
            details={
                "expected_gpu_count": expected_count,
                "observed_gpu_count": len(observed),
                "minimum_gpu_memory_bytes": config.resources.minimum_gpu_memory_bytes,
                "below_minimum_ordinals": below_minimum,
                "observed_total_gpu_memory_bytes": observed_total,
                "required_headroom_bytes": required_headroom,
                "usable_gpu_memory_bytes": usable_capacity,
                "sequential_peak_subject_bytes": sequential_peak,
                "aggregate_remaining_bytes": remaining,
            },
        )

    return MatchedCapacityResult(
        schema_version="inkling-matched-capacity-v1",
        status="pass",
        gpu_count=8,
        observed_gpu_memory_bytes=observed,
        observed_total_gpu_memory_bytes=observed_total,
        required_headroom_bytes=required_headroom,
        usable_gpu_memory_bytes=usable_capacity,
        bf16_subject_bytes=bf16_subject_bytes,
        q3_subject_bytes=q3_subject_bytes,
        sequential_peak_subject_bytes=sequential_peak,
        aggregate_remaining_bytes=remaining,
        runtime_fit_proven=False,
        limitation=CAPACITY_SCREEN_LIMITATION,
    )


__all__ = [
    "BF16_SUBJECT_REFERENCE_RELATIVE_PATH",
    "EXPECTED_BF16_SUBJECT_REFERENCE_SHA256",
    "MATCHED_CELL_CONFIG_RELATIVE_PATH",
    "InklingBF16SubjectReference",
    "InklingMatchedCellBundle",
    "InklingMatchedCellConfig",
    "MatchedCapacityResult",
    "MatchedSubjectPaths",
    "bf16_subject_reference_sha256",
    "load_bf16_subject_reference",
    "load_matched_cell_bundle",
    "load_matched_cell_config",
    "resolve_matched_subject_paths",
    "screen_matched_capacity",
]
