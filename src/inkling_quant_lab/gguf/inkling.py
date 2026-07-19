"""Exact-source planning for the experimental Inkling-to-GGUF workflow.

This module deliberately has no Modal, Hugging Face Hub, transformers, or llama.cpp
imports.  It is the CPU-only, offline-testable control plane that proves which model
will be converted, records known component omissions, constructs shell-free command
vectors, and rejects plans whose configured startup-plus-body windows exceed the cap.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import CapabilityError, ConfigurationError
from inkling_quant_lab.security import sensitive_literal_path

PINNED_INKLING_MODEL_ID: Final = "thinkingmachines/Inkling"
PINNED_INKLING_REVISION: Final = "86b4d430ab871652a707666b89203a866888c5e5"
PINNED_LLAMA_CPP_REPOSITORY: Final = "https://github.com/danielhanchen/llama.cpp.git"
PINNED_LLAMA_CPP_COMMIT: Final = "a015409e6c27b84f60d688823d4c0126a11571fd"
EXPECTED_ARCHITECTURE: Final = "InklingForConditionalGeneration"
EXPECTED_MODEL_TYPE: Final = "inkling_mm_model"
EXPECTED_LICENSE: Final = "apache-2.0"
EXPECTED_MODEL_BYTES: Final = 1_904_604_285_204
EXPECTED_TEXT_TENSORS: Final = 1_382
EXPECTED_VISION_TENSORS: Final = 8
EXPECTED_AUDIO_TENSORS: Final = 2
EXPECTED_MTP_TENSORS: Final = 160
EXPECTED_SOURCE_TENSORS: Final = 1_552
EXPECTED_SOURCE_SHARDS: Final = 109
EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256: Final = (
    "22e9760da68bfe0cf6f86c554ca5514936be0f5dfee0b63aaecf01d87b38cd95"
)
INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH: Final = (
    "configs/experiments/inkling_q3_k_m_source_adoption.json"
)
ORIGIN_SOURCE_RUN_ID: Final = "inkling-q3km-86b4d430-a015409e-551ab8f240-bcc168525e"
ORIGIN_SOURCE_CONFIG_SHA256: Final = (
    "551ab8f240269edbdc19efb61afc73e8b8b50e128e15781cf2248c674a8c4562"
)
ORIGIN_SOURCE_CONTROL_PLANE_SHA256: Final = (
    "bcc168525e8392944f4d19b8119fd888ab86f1cca620bbfd1c0d9e5dc5461ca3"
)
SOURCE_ADOPTION_HASH_DOMAIN: Final = b"inkling-source-adoption-reference-v1\0"
MODAL_CPU_CORE_HOUR_USD: Final = Decimal("0.04716")
MODAL_MEMORY_GIB_HOUR_USD: Final = Decimal("0.007992")
MODAL_B300_GPU_HOUR_USD: Final = Decimal("7.0992")
MODAL_STORAGE_DELETION_LAG_DAYS: Final = 4
FULL_INITIAL_BILLING_WINDOW_POLICY: Final = "full_workflow_plus_storage_lag_v1"
SHORT_INITIAL_BILLING_WINDOW_POLICY: Final = "operator_accepted_short_initial_window_v1"
DASHBOARD_EXACT_UTC_SOURCE: Final = "dashboard_exact_utc"
USER_CONFIRMED_ASSUMED_UTC_SOURCE: Final = "user_confirmed_date_assumed_utc_midnight"
BILLING_CYCLE_END_PATTERN: Final = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
)


class InklingSourceConfig(StrictFrozenModel):
    """Immutable identity and safe-loading contract for the only allowed source."""

    model_id: Literal["thinkingmachines/Inkling"] = PINNED_INKLING_MODEL_ID
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"] = PINNED_INKLING_REVISION
    architecture: Literal["InklingForConditionalGeneration"] = EXPECTED_ARCHITECTURE
    model_type: Literal["inkling_mm_model"] = EXPECTED_MODEL_TYPE
    license: Literal["apache-2.0"] = EXPECTED_LICENSE
    checkpoint_format: Literal["safetensors"] = "safetensors"
    trust_remote_code: Literal[False] = False


class InklingToolchainConfig(StrictFrozenModel):
    """Experimental llama.cpp implementation pinned by immutable commit."""

    repository: Literal["https://github.com/danielhanchen/llama.cpp.git"] = (
        PINNED_LLAMA_CPP_REPOSITORY
    )
    commit: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"] = PINNED_LLAMA_CPP_COMMIT
    support_status: Literal["experimental_unmerged_pr_25731"] = "experimental_unmerged_pr_25731"


class InklingCoverageConfig(StrictFrozenModel):
    """Explicit output coverage so omitted components cannot be hidden."""

    text: Literal["converted"] = "converted"
    multimodal: Literal["separate_bf16_mmproj"] = "separate_bf16_mmproj"
    mtp: Literal["omitted_unsupported"] = "omitted_unsupported"


class InklingQuantizationConfig(StrictFrozenModel):
    """Reproducible stock quantization recipe requiring no private imatrix."""

    quant_type: Literal["Q3_K_M"] = "Q3_K_M"
    output_label: str = Field(
        default="inkling-Q3_K_M", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$"
    )
    use_importance_matrix: Literal[False] = False
    split_max_size: str = Field(default="48G", pattern=r"^[1-9][0-9]*G$")
    threads: int = Field(default=32, ge=1, le=256)

    @field_validator("split_max_size")
    @classmethod
    def split_must_fit_volume_file_contract(cls, value: str) -> str:
        """Keep every output shard well below Modal Volume v2's 1 TiB limit."""

        if int(value[:-1]) >= 1_000:
            raise ValueError("split_max_size must be below 1000G")
        return value

    @model_validator(mode="after")
    def reject_unreproducible_dynamic_label(self) -> InklingQuantizationConfig:
        """Never present a stock quant as Unsloth Dynamic without its private inputs."""

        if self.output_label.upper().startswith("UD-") or "UD-Q3_K_XL" in self.output_label.upper():
            raise CapabilityError(
                "UD-Q3_K_XL requires an unpublished importance matrix and calibration corpus",
                component="inkling_gguf_recipe",
                remediation="Use the checked stock Q3_K_M recipe or supply and pin a new imatrix.",
            )
        if self.output_label != "inkling-Q3_K_M":
            raise CapabilityError(
                "output_label must truthfully identify the checked inkling-Q3_K_M recipe",
                component="inkling_gguf_recipe",
            )
        return self


class ModalRateCard(StrictFrozenModel):
    """Explicit list rates used only to calculate a launch ceiling."""

    as_of: Literal["2026-07-17"] = "2026-07-17"
    cpu_core_hour_usd: Decimal = Field(default=MODAL_CPU_CORE_HOUR_USD, gt=0)
    memory_gib_hour_usd: Decimal = Field(default=MODAL_MEMORY_GIB_HOUR_USD, gt=0)
    b300_gpu_hour_usd: Decimal = Field(default=MODAL_B300_GPU_HOUR_USD, gt=0)

    @model_validator(mode="after")
    def rates_match_verified_list_prices(self) -> ModalRateCard:
        actual = (
            self.cpu_core_hour_usd,
            self.memory_gib_hour_usd,
            self.b300_gpu_hour_usd,
        )
        expected = (
            MODAL_CPU_CORE_HOUR_USD,
            MODAL_MEMORY_GIB_HOUR_USD,
            MODAL_B300_GPU_HOUR_USD,
        )
        if actual != expected:
            raise ValueError(f"Modal rate card must equal the verified {self.as_of} list rates")
        return self


class ModalStageResources(StrictFrozenModel):
    """Maximum billable shape and configured startup/body windows for one stage."""

    name: Literal[
        "materialize_source",
        "convert_text_bf16",
        "convert_multimodal_projector",
        "quantize_text",
        "verify_export",
        "smoke_test",
    ]
    cpu_cores: int = Field(ge=1, le=128)
    memory_gib: int = Field(ge=1, le=512)
    gpu_type: Literal["B300"] | None = None
    gpu_count: int = Field(default=0, ge=0, le=8)
    startup_timeout_seconds: int = Field(ge=60, le=30 * 60)
    max_hours: Decimal = Field(gt=0, le=Decimal("23"))
    max_attempts: int = Field(default=1, ge=1, le=24)
    max_recovery_attempts: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def gpu_fields_are_consistent(self) -> ModalStageResources:
        if (self.gpu_type is None) != (self.gpu_count == 0):
            raise ValueError("gpu_type and gpu_count must be set together")
        return self


def _default_stages() -> tuple[ModalStageResources, ...]:
    return (
        ModalStageResources(
            name="materialize_source",
            cpu_cores=8,
            memory_gib=32,
            startup_timeout_seconds=15 * 60,
            max_hours=Decimal("4"),
            max_attempts=12,
        ),
        ModalStageResources(
            name="convert_text_bf16",
            cpu_cores=32,
            memory_gib=192,
            startup_timeout_seconds=15 * 60,
            max_hours=Decimal("23"),
            max_attempts=2,
            max_recovery_attempts=1,
        ),
        ModalStageResources(
            name="convert_multimodal_projector",
            cpu_cores=8,
            memory_gib=32,
            startup_timeout_seconds=15 * 60,
            max_hours=Decimal("12"),
            max_attempts=2,
            max_recovery_attempts=1,
        ),
        ModalStageResources(
            name="quantize_text",
            cpu_cores=32,
            memory_gib=192,
            startup_timeout_seconds=15 * 60,
            max_hours=Decimal("23"),
            max_attempts=2,
            max_recovery_attempts=1,
        ),
        ModalStageResources(
            name="verify_export",
            cpu_cores=16,
            memory_gib=64,
            startup_timeout_seconds=15 * 60,
            max_hours=Decimal("12"),
            max_attempts=2,
            max_recovery_attempts=1,
        ),
        ModalStageResources(
            name="smoke_test",
            cpu_cores=16,
            memory_gib=64,
            gpu_type="B300",
            gpu_count=2,
            startup_timeout_seconds=15 * 60,
            max_hours=Decimal("2"),
            max_attempts=1,
        ),
    )


class InklingBudgetConfig(StrictFrozenModel):
    """Planned category envelopes plus the enforceable total Workspace cap."""

    user_limit_usd: Decimal = Field(default=Decimal("1000"), gt=0, le=Decimal("1000"))
    planned_compute_usd: Decimal = Field(default=Decimal("600"), gt=0)
    planned_storage_usd: Decimal = Field(default=Decimal("150"), ge=0)
    workspace_contingency_usd: Decimal = Field(default=Decimal("50"), ge=0)
    workspace_hard_budget_usd: Decimal = Field(default=Decimal("800"), gt=0)
    external_contingency_usd: Decimal = Field(default=Decimal("150"), ge=0)
    max_total_usd: Decimal = Field(default=Decimal("950"), gt=0)
    require_workspace_budget: Literal[True] = True

    @model_validator(mode="after")
    def remain_below_user_limit(self) -> InklingBudgetConfig:
        if self.max_total_usd >= self.user_limit_usd:
            raise ValueError("max_total_usd must remain strictly below the user limit")
        workspace_envelopes = (
            self.planned_compute_usd + self.planned_storage_usd + self.workspace_contingency_usd
        )
        if workspace_envelopes != self.workspace_hard_budget_usd:
            raise ValueError("planned Modal envelopes must equal workspace_hard_budget_usd")
        if self.workspace_hard_budget_usd + self.external_contingency_usd != self.max_total_usd:
            raise ValueError("workspace budget and external contingency must equal max_total_usd")
        return self


class InklingModalConfig(StrictFrozenModel):
    """Names and resource ceilings for the isolated paid environment."""

    app_name: Literal["inkling-q3-k-m"] = "inkling-q3-k-m"
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    source_volume: Literal["inkling-source-v1"] = "inkling-source-v1"
    work_volume: Literal["inkling-work-v1"] = "inkling-work-v1"
    final_volume: Literal["inkling-final-v1"] = "inkling-final-v1"
    hf_secret: Literal["huggingface-secret"] = "huggingface-secret"
    hf_secret_environment: Literal["main"] = "main"
    volume_version: Literal[1] = 1
    rate_card: ModalRateCard = Field(default_factory=ModalRateCard)
    stages: tuple[ModalStageResources, ...] = Field(default_factory=_default_stages)

    @model_validator(mode="after")
    def require_unique_complete_stage_budgets(self) -> InklingModalConfig:
        expected_stages = _default_stages()
        names = tuple(stage.name for stage in expected_stages)
        if self.stages != expected_stages:
            raise ValueError(
                f"modal resource/timeout/attempt matrix must equal the checked stages {names}"
            )
        return self


class InklingGGUFConfig(StrictFrozenModel):
    """Complete checked configuration for a manual-large Inkling conversion."""

    schema_version: Literal["1.1"] = "1.1"
    source: InklingSourceConfig = Field(default_factory=InklingSourceConfig)
    toolchain: InklingToolchainConfig = Field(default_factory=InklingToolchainConfig)
    coverage: InklingCoverageConfig = Field(default_factory=InklingCoverageConfig)
    quantization: InklingQuantizationConfig = Field(default_factory=InklingQuantizationConfig)
    modal: InklingModalConfig = Field(default_factory=InklingModalConfig)
    budget: InklingBudgetConfig = Field(default_factory=InklingBudgetConfig)

    @model_validator(mode="after")
    def validate_secrets_and_cost(self) -> InklingGGUFConfig:
        literal_secret = sensitive_literal_path(self.model_dump(mode="json"))
        if literal_secret is not None:
            raise ValueError(
                "configuration contains literal credential material at " + ".".join(literal_secret)
            )
        ceiling = compute_cost_ceiling_usd(self)
        if ceiling > self.budget.planned_compute_usd:
            raise ValueError(
                f"configured compute ceiling {ceiling} exceeds {self.budget.planned_compute_usd}"
            )
        return self

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class SourceAdoptionArtifact(StrictFrozenModel):
    """One immutable file needed to authenticate an existing source snapshot."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_canonical_and_contained(cls, value: str) -> str:
        if "\x00" in value or "\\" in value or "//" in value:
            raise ValueError("adoption artifact path must be canonical POSIX text")
        parsed = PurePosixPath(value)
        if any(part in {"", ".", ".."} for part in parsed.parts):
            raise ValueError("adoption artifact path must not contain traversal")
        if parsed.as_posix() != value:
            raise ValueError("adoption artifact path must be canonical")
        return value


class InklingSourceAdoptionReference(StrictFrozenModel):
    """Checked direct-source lineage used to avoid a second multi-terabyte download.

    This record is deliberately specific to the already verified materialization. It
    is not an alias and does not itself authorize adoption: the paid boundary must
    still rehash every referenced remote file and the complete source inventory.
    """

    schema_version: Literal["inkling-source-adoption-reference-v1"] = (
        "inkling-source-adoption-reference-v1"
    )
    adoption_kind: Literal["rehash_verified_source_into_distinct_run_v1"] = (
        "rehash_verified_source_into_distinct_run_v1"
    )
    origin_materialization_kind: Literal["direct_huggingface_snapshot_v1"] = (
        "direct_huggingface_snapshot_v1"
    )
    origin_parent_adoption_reference_sha256: None = None
    verified: Literal[True] = True
    origin_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    origin_app_name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    origin_app_id: str = Field(pattern=r"^ap-[A-Za-z0-9]+$")
    origin_app_required_state: Literal["stopped"] = "stopped"
    origin_app_required_active_tasks: Literal[0] = 0
    origin_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin_control_plane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin_control_plane_file_count: int = Field(gt=0)
    materialize_function_call_id: str = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    materialize_input_id: str = Field(min_length=1, pattern=r"^in-[A-Za-z0-9:-]+$")
    materialize_task_id: str = Field(pattern=r"^ta-[A-Za-z0-9]+$")
    materialize_call_terminal_status: Literal["SUCCESS"] = "SUCCESS"
    materialize_call_child_count: Literal[0] = 0
    materialize_continuation_call_ids: tuple[()] = ()
    materialize_launch_intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_volume: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    source_mount_path: str
    source_run_root: str
    snapshot_path: str
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    architecture: str
    model_type: str
    license: str
    checkpoint_format: Literal["safetensors"] = "safetensors"
    trust_remote_code: Literal[False] = False
    toolchain_repository: str
    toolchain_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    indexed_tensor_bytes: int = Field(gt=0)
    materialized_weight_file_bytes: int = Field(gt=0)
    source_tensor_count: int = Field(gt=0)
    source_shard_count: int = Field(gt=0)
    materialized_file_count: int = Field(gt=0)
    text_tensor_count: int = Field(gt=0)
    vision_tensor_count: int = Field(gt=0)
    audio_tensor_count: int = Field(gt=0)
    mtp_tensor_count: int = Field(gt=0)
    converted_source_tensor_count: int = Field(gt=0)
    omitted_source_tensor_count: int = Field(gt=0)
    weight_index_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_success_receipt: SourceAdoptionArtifact
    source_inventory: SourceAdoptionArtifact
    origin_resolved_config: SourceAdoptionArtifact
    origin_control_plane: SourceAdoptionArtifact
    origin_materialize_attempt_ledger: SourceAdoptionArtifact
    origin_materialize_invocation_history: SourceAdoptionArtifact
    snapshot_config: SourceAdoptionArtifact
    snapshot_weight_index: SourceAdoptionArtifact
    local_materialize_call_receipt: SourceAdoptionArtifact
    local_materialize_launch_intent: SourceAdoptionArtifact
    local_deployment_receipt: SourceAdoptionArtifact
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_mount_path", "source_run_root", "snapshot_path")
    @classmethod
    def source_paths_are_canonical_absolute_posix(cls, value: str) -> str:
        if "\x00" in value or "\\" in value or "//" in value:
            raise ValueError("source path must be canonical absolute POSIX text")
        parsed = PurePosixPath(value)
        if not parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
            raise ValueError("source path must be absolute and contain no traversal")
        if parsed.as_posix() != value:
            raise ValueError("source path must be canonical")
        return value

    def canonical_payload_dict(self) -> dict[str, Any]:
        """Return the self-hashed payload without its hash field."""

        return self.model_dump(mode="json", exclude={"reference_sha256"})

    def computed_reference_sha256(self) -> str:
        """Compute the domain-separated hash over the canonical payload."""

        return inkling_source_adoption_reference_sha256(self.canonical_payload_dict())

    def canonical_json(self) -> str:
        """Serialize the complete checked record as canonical JSON."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @model_validator(mode="after")
    def matches_exact_verified_direct_source(self) -> InklingSourceAdoptionReference:
        source_root = f"/source/runs/{ORIGIN_SOURCE_RUN_ID}"
        local_root = f"artifacts/inkling-modal/{ORIGIN_SOURCE_RUN_ID}"
        expected: dict[str, object] = {
            "origin_run_id": ORIGIN_SOURCE_RUN_ID,
            "origin_app_name": "inkling-q3-k-m-bcc168525e83",
            "origin_app_id": "ap-zhL7JVGVfoVSSeiKxrj1k3",
            "origin_config_hash": ORIGIN_SOURCE_CONFIG_SHA256,
            "origin_control_plane_sha256": ORIGIN_SOURCE_CONTROL_PLANE_SHA256,
            "origin_control_plane_file_count": 102,
            "materialize_function_call_id": "fc-01KXVB8CYN7EA0S0Z3650W0C53",
            "materialize_input_id": "in-01KXVB8CZ0V6E777SP8Z02JTGE:1784402949113-0",
            "materialize_task_id": "ta-01KXVB8EEZGNFCTYXHPJT612GR",
            "materialize_launch_intent_sha256": (
                "864b14663f506bc5aa6bd9d0ea24dffbbfc05779f82b1b3c380c4449ca4fa339"
            ),
            "source_volume": "inkling-source-v1",
            "source_mount_path": "/source",
            "source_run_root": source_root,
            "snapshot_path": f"{source_root}/snapshot",
            "model_id": PINNED_INKLING_MODEL_ID,
            "revision": PINNED_INKLING_REVISION,
            "architecture": EXPECTED_ARCHITECTURE,
            "model_type": EXPECTED_MODEL_TYPE,
            "license": EXPECTED_LICENSE,
            "toolchain_repository": PINNED_LLAMA_CPP_REPOSITORY,
            "toolchain_commit": PINNED_LLAMA_CPP_COMMIT,
            "indexed_tensor_bytes": EXPECTED_MODEL_BYTES,
            "materialized_weight_file_bytes": 1_904_755_463_940,
            "source_tensor_count": EXPECTED_SOURCE_TENSORS,
            "source_shard_count": EXPECTED_SOURCE_SHARDS,
            "materialized_file_count": 117,
            "text_tensor_count": EXPECTED_TEXT_TENSORS,
            "vision_tensor_count": EXPECTED_VISION_TENSORS,
            "audio_tensor_count": EXPECTED_AUDIO_TENSORS,
            "mtp_tensor_count": EXPECTED_MTP_TENSORS,
            "converted_source_tensor_count": (
                EXPECTED_TEXT_TENSORS + EXPECTED_VISION_TENSORS + EXPECTED_AUDIO_TENSORS
            ),
            "omitted_source_tensor_count": EXPECTED_MTP_TENSORS,
            "weight_index_canonical_sha256": EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256,
        }
        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]
        expected_artifacts = {
            "source_success_receipt": {
                "path": f"{source_root}/source.success.json",
                "sha256": "06937bc535fb703da6adc9d11e1e804ce15f67b39ffbbe98a9a51a7dc70edbbc",
                "size_bytes": 1193,
            },
            "source_inventory": {
                "path": f"{source_root}/source_inventory.json",
                "sha256": "a8aa37efec2b12c5d584c8163111d3a8a22d9568ef01886343755a8af6ace571",
                "size_bytes": 25106,
            },
            "origin_resolved_config": {
                "path": f"{source_root}/resolved_config.json",
                "sha256": "f0527ddb6b475e20771c231e42329cf51fc3326c87453593418b062d1b3aac96",
                "size_bytes": 3351,
            },
            "origin_control_plane": {
                "path": f"{source_root}/control_plane.json",
                "sha256": "135bf3af048e8e6fe6e4e9032c94da364a5e362dcb012e4b5fbaf36ececda380",
                "size_bytes": 18718,
            },
            "origin_materialize_invocation_history": {
                "path": (
                    f"{source_root}/control/history/materialize_source.attempt.1."
                    "896fb36e8aadc52a0d15fe190e464ff33e4910f2fa163bb00e64d712fd990581.json"
                ),
                "sha256": "42bb7cd1338485ea81c3c15346b29304f1c135167ea205877c72d1b902956456",
                "size_bytes": 632,
            },
            "origin_materialize_attempt_ledger": {
                "path": f"{source_root}/control/materialize_source.attempts.json",
                "sha256": "3d30e78f8f2f8ee70a5fb7fe53109b17b8bb2c39bc64ecd2e9f58e9607fa51f3",
                "size_bytes": 712,
            },
            "snapshot_config": {
                "path": f"{source_root}/snapshot/config.json",
                "sha256": "58720f145bcecef9a7ab2b419ab346e7c634af8d2f3e7362e900d00f789ea46c",
                "size_bytes": 2415,
            },
            "snapshot_weight_index": {
                "path": f"{source_root}/snapshot/model.safetensors.index.json",
                "sha256": "6bdebc2a928b1be96e1666b40704a4222ee0c764c2611247bb8ad4d485ea9a97",
                "size_bytes": 128600,
            },
            "local_materialize_call_receipt": {
                "path": (
                    f"{local_root}/calls/2026-07-18T192909.115192+0000-materialize_source.json"
                ),
                "sha256": "eba7fc186b0150e8e43dba50dd96c8bca865c8b6aa79587a1661e13f8b522a90",
                "size_bytes": 1054,
            },
            "local_materialize_launch_intent": {
                "path": (
                    f"{local_root}/launch-intents/"
                    "2026-07-18T192908.5039760000-materialize_source-"
                    "dd18a2109a926e7eae54a3f4f6890136.json"
                ),
                "sha256": "864b14663f506bc5aa6bd9d0ea24dffbbfc05779f82b1b3c380c4449ca4fa339",
                "size_bytes": 853,
            },
            "local_deployment_receipt": {
                "path": f"{local_root}/deployment.json",
                "sha256": "40eb0bf204115a58c7bd196fff601c5184c9479c82c52ba4d3891b876c85d6e6",
                "size_bytes": 1248,
            },
        }
        mismatches.extend(
            name
            for name, value in expected_artifacts.items()
            if getattr(self, name).model_dump(mode="json") != value
        )
        if mismatches:
            raise ValueError(
                "source adoption reference differs from exact verified source evidence: "
                + ", ".join(sorted(mismatches))
            )
        if self.reference_sha256 != self.computed_reference_sha256():
            raise ValueError("source adoption reference self-hash does not match its payload")
        return self


def inkling_source_adoption_reference_sha256(value: Mapping[str, Any]) -> str:
    """Hash a reference payload, ignoring its own hash field if present."""

    payload = dict(value)
    payload.pop("reference_sha256", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(SOURCE_ADOPTION_HASH_DOMAIN + canonical).hexdigest()


def load_inkling_source_adoption_reference(
    path: str | Path,
) -> InklingSourceAdoptionReference:
    """Load one byte-canonical, self-hashed direct-source adoption reference."""

    reference_path = Path(path)
    try:
        raw_bytes = reference_path.read_bytes()
        raw = json.loads(raw_bytes)
        if not isinstance(raw, Mapping):
            raise ValueError("reference root must be a JSON object")
        reference = InklingSourceAdoptionReference.model_validate(raw)
    except (OSError, ValueError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load Inkling source adoption reference {reference_path}: {error}",
            component="inkling_source_adoption",
        ) from error
    if raw_bytes != (reference.canonical_json() + "\n").encode("utf-8"):
        raise ConfigurationError(
            "Inkling source adoption reference must use canonical JSON plus one newline",
            component="inkling_source_adoption",
        )
    return reference


def validate_inkling_source_adoption_reference(
    reference: InklingSourceAdoptionReference,
    *,
    target_config: InklingGGUFConfig,
    target_control_plane_sha256: str,
) -> InklingSourceAdoptionReference:
    """Bind a direct source reference to a distinct, config-identical target run."""

    if re.fullmatch(r"[0-9a-f]{64}", target_control_plane_sha256) is None:
        raise ConfigurationError(
            "target_control_plane_sha256 must be SHA-256",
            component="inkling_source_adoption",
        )
    if target_config.config_hash() != reference.origin_config_hash:
        raise ConfigurationError(
            "Source adoption requires the exact origin configuration",
            component="inkling_source_adoption",
        )
    expected_origin_run_id = inkling_run_id(
        target_config,
        reference.origin_control_plane_sha256,
    )
    if expected_origin_run_id != reference.origin_run_id:
        raise ConfigurationError(
            "Source adoption origin run is not derived from its config and control plane",
            component="inkling_source_adoption",
        )
    target_run_id = inkling_run_id(target_config, target_control_plane_sha256)
    if (
        target_control_plane_sha256 == reference.origin_control_plane_sha256
        or target_run_id == reference.origin_run_id
    ):
        raise ConfigurationError(
            "Source self-adoption is forbidden; the target run must be distinct",
            component="inkling_source_adoption",
        )
    return reference


class ControlPlaneFile(StrictFrozenModel):
    """One file in the content-addressed local orchestration source."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ControlPlaneProvenance(StrictFrozenModel):
    """Exact source bytes allowed to coordinate one multi-invocation Modal run."""

    kind: Literal["inkling_gguf_control_plane_v1"] = "inkling_gguf_control_plane_v1"
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(ge=1)
    files: tuple[ControlPlaneFile, ...]

    @model_validator(mode="after")
    def count_matches_files(self) -> ControlPlaneProvenance:
        if self.file_count != len(self.files):
            raise ValueError("control-plane file_count does not match its file manifest")
        if tuple(sorted(item.path for item in self.files)) != tuple(
            item.path for item in self.files
        ):
            raise ValueError("control-plane files must be canonically ordered")
        expected_tree_sha256 = _control_plane_tree_sha256(self.files)
        if self.tree_sha256 != expected_tree_sha256:
            raise ValueError("control-plane tree hash does not match its file manifest")
        return self

    def canonical_json(self) -> str:
        """Serialize the exact manifest passed to a deployed Modal Function."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


class PaidLaunchAcknowledgement(StrictFrozenModel):
    """Explicit operator acknowledgement repeated at every paid remote boundary."""

    schema_version: Literal["1.3"] = "1.3"
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    launch_intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    workspace_budget_usd: Decimal
    billing_cycle_end_utc: str
    initial_billing_window_policy: Literal[
        "full_workflow_plus_storage_lag_v1",
        "operator_accepted_short_initial_window_v1",
    ] = FULL_INITIAL_BILLING_WINDOW_POLICY
    billing_cycle_end_source: Literal[
        "dashboard_exact_utc",
        "user_confirmed_date_assumed_utc_midnight",
    ] = DASHBOARD_EXACT_UTC_SOURCE
    confirmation: Literal["dashboard-workspace-hard-budget-confirmed"] = (
        "dashboard-workspace-hard-budget-confirmed"
    )

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @field_validator("billing_cycle_end_utc")
    @classmethod
    def cycle_end_is_canonical_utc(cls, value: str) -> str:
        _parse_billing_cycle_end(value)
        return value

    @model_validator(mode="after")
    def billing_window_evidence_is_consistent(self) -> PaidLaunchAcknowledgement:
        expected_source = (
            USER_CONFIRMED_ASSUMED_UTC_SOURCE
            if self.initial_billing_window_policy == SHORT_INITIAL_BILLING_WINDOW_POLICY
            else DASHBOARD_EXACT_UTC_SOURCE
        )
        if self.billing_cycle_end_source != expected_source:
            raise ValueError("billing-cycle source does not match the initial-window policy")
        return self


def inkling_control_plane_provenance(project_root: Path) -> ControlPlaneProvenance:
    """Hash all runnable package bytes plus the exact scripts, config, and dependency lock."""

    source_root = project_root / "src" / "inkling_quant_lab"
    required_files = (
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "configs" / "experiments" / "inkling_q3_k_m_modal.yaml",
        project_root / "configs" / "experiments" / "inkling_q3_k_m_source_adoption.json",
        project_root / "scripts" / "preflight_inkling_gguf.py",
        project_root / "scripts" / "manage_inkling_modal.py",
        project_root / "scripts" / "quantize_inkling_modal.py",
    )
    if not source_root.is_dir() or any(not path.is_file() for path in required_files):
        raise ConfigurationError("Inkling control-plane source tree is incomplete")
    source_entries = tuple(sorted(source_root.rglob("*"), key=lambda path: path.as_posix()))
    candidates = (*required_files, *source_entries)
    if source_root.is_symlink() or any(path.is_symlink() for path in candidates):
        raise ConfigurationError("Inkling control-plane provenance rejects symlinks")
    files: list[ControlPlaneFile] = []
    for path in candidates:
        relative = path.relative_to(project_root)
        if not path.is_file() or "__pycache__" in relative.parts or path.name == ".DS_Store":
            continue
        if path.suffix in {".pyc", ".pyo"}:
            raise ConfigurationError("Inkling control-plane provenance rejects bytecode")
        payload = path.read_bytes()
        files.append(
            ControlPlaneFile(
                path=relative.as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    files.sort(key=lambda item: item.path)
    return ControlPlaneProvenance(
        tree_sha256=_control_plane_tree_sha256(files),
        file_count=len(files),
        files=tuple(files),
    )


def _control_plane_tree_sha256(files: tuple[ControlPlaneFile, ...] | list[ControlPlaneFile]) -> str:
    canonical = json.dumps(
        [item.model_dump(mode="json") for item in files],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(b"inkling-gguf-control-plane-v1\0" + canonical).hexdigest()


def validate_deployed_control_plane(
    provenance_json: str,
    *,
    deployment_script: Path,
    deployed_package_root: Path,
) -> ControlPlaneProvenance:
    """Prove that deployed script/package bytes match the caller's full source manifest.

    Modal mounts a file entrypoint at a different container path, while the package is
    copied into the image.  This check maps those deployed paths back to their canonical
    repository-relative names and rejects missing, additional, symlinked, or changed
    runtime source files.  Local-only lock/config/preflight files remain part of the
    manifest and therefore the deterministic run ID, but are not claimed as container
    runtime inputs.
    """

    try:
        provenance = ControlPlaneProvenance.model_validate_json(provenance_json)
    except ValidationError as error:
        raise ConfigurationError(
            "Deployed control-plane manifest is invalid",
            component="inkling_control_plane",
        ) from error

    expected_by_path = {item.path: item for item in provenance.files}
    local_only_required = {
        "pyproject.toml",
        "uv.lock",
        "configs/experiments/inkling_q3_k_m_modal.yaml",
        INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
        "scripts/preflight_inkling_gguf.py",
        "scripts/manage_inkling_modal.py",
    }
    runtime_script_name = "scripts/quantize_inkling_modal.py"
    missing_required = sorted((local_only_required | {runtime_script_name}) - set(expected_by_path))
    if missing_required:
        raise ConfigurationError(
            "Control-plane manifest lacks required files: " + ", ".join(missing_required),
            component="inkling_control_plane",
        )

    script = Path(deployment_script)
    package_root = Path(deployed_package_root)
    if script.is_symlink() or not script.is_file():
        raise ConfigurationError(
            "Deployed Modal entrypoint is missing or a symlink",
            component="inkling_control_plane",
        )
    if package_root.is_symlink() or not package_root.is_dir():
        raise ConfigurationError(
            "Deployed Inkling Quant Lab package is missing or a symlink",
            component="inkling_control_plane",
        )

    observed: dict[str, tuple[str, int]] = {runtime_script_name: _deployed_file_identity(script)}
    for path in sorted(package_root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or path.name == ".DS_Store":
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ConfigurationError(
                f"Deployed package contains a symlink: {relative.as_posix()}",
                component="inkling_control_plane",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ConfigurationError(
                f"Deployed package contains a special file: {relative.as_posix()}",
                component="inkling_control_plane",
            )
        canonical_path = (Path("src/inkling_quant_lab") / relative).as_posix()
        observed[canonical_path] = _deployed_file_identity(path)

    expected_runtime = {
        path: item
        for path, item in expected_by_path.items()
        if path == runtime_script_name or path.startswith("src/inkling_quant_lab/")
    }
    if set(observed) != set(expected_runtime):
        missing = sorted(set(expected_runtime) - set(observed))
        additional = sorted(set(observed) - set(expected_runtime))
        raise ConfigurationError(
            f"Deployed source inventory differs (missing={missing}, additional={additional})",
            component="inkling_control_plane",
        )
    drifted = sorted(
        path
        for path, identity in observed.items()
        if identity != (expected_runtime[path].sha256, expected_runtime[path].size_bytes)
    )
    if drifted:
        raise ConfigurationError(
            "Deployed source bytes differ from the control plane: " + ", ".join(drifted[:5]),
            component="inkling_control_plane",
        )
    return provenance


def _deployed_file_identity(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def inkling_run_id(config: InklingGGUFConfig, control_plane_sha256: str) -> str:
    """Derive one run namespace from source, config, toolchain, and orchestration bytes."""

    if re.fullmatch(r"[0-9a-f]{64}", control_plane_sha256) is None:
        raise ConfigurationError("control_plane_sha256 must be SHA-256")
    return (
        f"inkling-q3km-{config.source.revision[:8]}-{config.toolchain.commit[:8]}-"
        f"{config.config_hash()[:10]}-{control_plane_sha256[:10]}"
    )


def validate_paid_launch_acknowledgement(
    config: InklingGGUFConfig,
    acknowledgement_json: str,
    *,
    control_plane_sha256: str,
) -> PaidLaunchAcknowledgement:
    """Reject remote work lacking the exact local budget/config/source acknowledgement."""

    try:
        acknowledgement = PaidLaunchAcknowledgement.model_validate_json(acknowledgement_json)
    except ValidationError as error:
        raise ConfigurationError(
            "Paid launch acknowledgement is invalid", component="inkling_paid_gate"
        ) from error
    if (
        acknowledgement.config_hash != config.config_hash()
        or acknowledgement.control_plane_sha256 != control_plane_sha256
        or acknowledgement.environment_name != config.modal.environment_name
        or acknowledgement.workspace_budget_usd != config.budget.workspace_hard_budget_usd
    ):
        raise ConfigurationError(
            "Paid launch acknowledgement does not bind this exact run",
            component="inkling_paid_gate",
        )
    return acknowledgement


class InklingSourceAudit(StrictFrozenModel):
    """Evidence that the source is the pinned official Inkling checkpoint."""

    model_id: str
    revision: str
    architecture: str
    model_type: str
    license: str
    source_bytes: int = Field(ge=0)
    source_tensor_count: int = Field(ge=0)
    source_shard_count: int = Field(ge=0)
    text_tensor_count: int = Field(ge=0)
    vision_tensor_count: int = Field(ge=0)
    audio_tensor_count: int = Field(ge=0)
    mtp_tensor_count: int = Field(ge=0)
    converted_source_tensor_count: int = Field(ge=0)
    omitted_source_tensor_count: int = Field(ge=0)
    weight_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified: bool
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def matches_exact_pinned_audit(self) -> InklingSourceAudit:
        actual = (
            self.model_id,
            self.revision,
            self.architecture,
            self.model_type,
            self.license,
            self.source_bytes,
            self.source_tensor_count,
            self.source_shard_count,
            self.text_tensor_count,
            self.vision_tensor_count,
            self.audio_tensor_count,
            self.mtp_tensor_count,
            self.converted_source_tensor_count,
            self.omitted_source_tensor_count,
            self.weight_index_sha256,
            self.verified,
        )
        expected = (
            PINNED_INKLING_MODEL_ID,
            PINNED_INKLING_REVISION,
            EXPECTED_ARCHITECTURE,
            EXPECTED_MODEL_TYPE,
            EXPECTED_LICENSE,
            EXPECTED_MODEL_BYTES,
            EXPECTED_SOURCE_TENSORS,
            EXPECTED_SOURCE_SHARDS,
            EXPECTED_TEXT_TENSORS,
            EXPECTED_VISION_TENSORS,
            EXPECTED_AUDIO_TENSORS,
            EXPECTED_MTP_TENSORS,
            EXPECTED_TEXT_TENSORS + EXPECTED_VISION_TENSORS + EXPECTED_AUDIO_TENSORS,
            EXPECTED_MTP_TENSORS,
            EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256,
            True,
        )
        if actual != expected:
            raise ValueError("source audit does not match the exact pinned Inkling inventory")
        return self


class WorkflowPaths(StrictFrozenModel):
    """Absolute mount paths passed to fixed-argument subprocesses."""

    source_dir: Path
    work_dir: Path
    final_dir: Path
    llama_cpp_dir: Path = Path("/opt/llama.cpp")

    @field_validator("source_dir", "work_dir", "final_dir", "llama_cpp_dir")
    @classmethod
    def paths_are_absolute(cls, value: Path) -> Path:
        if not value.is_absolute() or "\x00" in str(value):
            raise ValueError("workflow paths must be absolute and NUL-free")
        return value


class CommandSpec(StrictFrozenModel):
    """One shell-free subprocess invocation and its application timeout."""

    name: str
    argv: tuple[str, ...]
    timeout_seconds: int = Field(ge=1, le=23 * 3600)

    @field_validator("argv")
    @classmethod
    def argv_is_safe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not argument or "\x00" in argument for argument in value):
            raise ValueError("argv must contain only non-empty NUL-free arguments")
        return value


class ConversionPlan(StrictFrozenModel):
    """BF16 text/mmproj conversions; quantization is bound after split discovery."""

    text_conversion: CommandSpec
    mmproj_conversion: CommandSpec


class VerificationPlan(StrictFrozenModel):
    """Structural parsing commands for the complete split set and projector."""

    q3_split_set: CommandSpec
    mmproj: CommandSpec


class ExecutionBindingEvidence(StrictFrozenModel):
    """Proof that command paths resolve to the audited snapshot and toolchain."""

    source_dir: str
    source_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    revision: str
    weight_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    llama_cpp_dir: str
    llama_cpp_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


def compute_cost_ceiling_usd(config: InklingGGUFConfig) -> Decimal:
    """Calculate all configured startup plus body windows across all attempts.

    This is not a bound on Modal infrastructure rescheduling before user code;
    the externally activated workspace hard budget is that broader backstop.
    """

    rates = config.modal.rate_card
    total = Decimal("0")
    for stage in config.modal.stages:
        hourly = (
            Decimal(stage.cpu_cores) * rates.cpu_core_hour_usd
            + Decimal(stage.memory_gib) * rates.memory_gib_hour_usd
        )
        if stage.gpu_type == "B300":
            hourly += Decimal(stage.gpu_count) * rates.b300_gpu_hour_usd
        startup_hours = Decimal(stage.startup_timeout_seconds) / Decimal(3600)
        configured_invocations = stage.max_attempts + stage.max_recovery_attempts
        total += hourly * (startup_hours + stage.max_hours) * Decimal(configured_invocations)
    return total.quantize(Decimal("0.01"))


def modal_stage_resources(config: InklingGGUFConfig, name: str) -> ModalStageResources:
    """Return the one checked resource record that also controls command timeouts."""

    matches = [stage for stage in config.modal.stages if stage.name == name]
    if len(matches) != 1:
        raise ConfigurationError(f"Expected one Modal resource record for {name}")
    return matches[0]


def _parse_billing_cycle_end(value: str) -> datetime:
    """Parse the one canonical UTC format accepted at the paid boundary."""

    if BILLING_CYCLE_END_PATTERN.fullmatch(value) is None:
        raise ValueError("billing_cycle_end_utc must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("billing_cycle_end_utc is not a real UTC timestamp") from error
    return parsed


def configured_deployable_wall_time_hours(config: InklingGGUFConfig) -> Decimal:
    """Return all deployable startup/body windows, including recovery reserves."""

    total = Decimal("0")
    for stage in config.modal.stages:
        if stage.name == "smoke_test":
            continue
        invocations = stage.max_attempts + stage.max_recovery_attempts
        total += (
            Decimal(stage.startup_timeout_seconds) / Decimal(3600) + stage.max_hours
        ) * Decimal(invocations)
    return total.quantize(Decimal("0.01"))


def _require_billing_window(
    cycle_end_utc: str,
    *,
    required_hours: Decimal,
    now: datetime | None,
) -> None:
    observed_now = datetime.now(UTC) if now is None else now
    if observed_now.tzinfo is None or observed_now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    cycle_end = _parse_billing_cycle_end(cycle_end_utc)
    required_seconds = int(required_hours * Decimal(3600))
    deletion_lag = timedelta(days=MODAL_STORAGE_DELETION_LAG_DAYS)
    if (
        observed_now.astimezone(UTC) + timedelta(seconds=required_seconds) + deletion_lag
        > cycle_end
    ):
        raise ConfigurationError(
            "The confirmed monthly billing-cycle window is too short for the frozen work "
            "plus Modal's documented four-day storage deletion lag",
            component="inkling_paid_gate",
        )


def require_initial_billing_window(
    config: InklingGGUFConfig,
    cycle_end_utc: str,
    *,
    now: datetime | None = None,
) -> None:
    """Require the full sequential reserve before the first paid deployment."""

    _require_billing_window(
        cycle_end_utc,
        required_hours=configured_deployable_wall_time_hours(config),
        now=now,
    )


def require_materialize_initial_billing_window(
    config: InklingGGUFConfig,
    cycle_end_utc: str,
    *,
    now: datetime | None = None,
) -> None:
    """Require every frozen materialization invocation plus deletion lag."""

    resources = modal_stage_resources(config, "materialize_source")
    require_stage_billing_window(
        config,
        cycle_end_utc,
        "materialize_source",
        include_startup=True,
        invocations=resources.max_attempts + resources.max_recovery_attempts,
        now=now,
    )


def require_stage_billing_window(
    config: InklingGGUFConfig,
    cycle_end_utc: str,
    stage: str,
    *,
    include_startup: bool,
    invocations: int = 1,
    now: datetime | None = None,
) -> None:
    """Refuse a paid invocation that could overrun the cleanup cutoff."""

    if invocations < 1:
        raise ValueError("invocations must be positive")
    resources = modal_stage_resources(config, stage)
    required_hours = resources.max_hours
    if include_startup:
        required_hours += Decimal(resources.startup_timeout_seconds) / Decimal(3600)
    _require_billing_window(
        cycle_end_utc,
        required_hours=required_hours * Decimal(invocations),
        now=now,
    )


def load_inkling_gguf_config(path: str | Path) -> InklingGGUFConfig:
    """Load the dedicated checked YAML without invoking the standard experiment schema."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ConfigurationError("Inkling GGUF config root must be a mapping")
        return InklingGGUFConfig.model_validate(raw)
    except ConfigurationError:
        raise
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load Inkling GGUF config {config_path}: {error}",
            component="inkling_gguf_config",
        ) from error


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping", component="inkling_preflight")
    return value


def audit_inkling_source(
    config: InklingGGUFConfig,
    *,
    model_info: Mapping[str, Any],
    model_config: Mapping[str, Any],
    weight_index: Mapping[str, Any],
) -> InklingSourceAudit:
    """Fail closed unless Hub metadata proves the exact official checkpoint inventory."""

    if model_info.get("id") != config.source.model_id:
        raise ConfigurationError("Hub model id does not match pinned Inkling model id")
    if model_info.get("sha") != config.source.revision:
        raise ConfigurationError("Resolved Hub revision does not match pinned Inkling revision")
    card_data = _mapping(model_info.get("cardData"), label="Hub model card data")
    if card_data.get("license") != config.source.license:
        raise ConfigurationError("Resolved Hub license is not the pinned Apache-2.0 license")
    architectures = model_config.get("architectures")
    if not isinstance(architectures, list) or config.source.architecture not in architectures:
        raise ConfigurationError("Resolved model architecture is not Inkling")
    if model_config.get("model_type") != config.source.model_type:
        raise ConfigurationError("Resolved model type is not inkling_mm_model")

    metadata = _mapping(weight_index.get("metadata"), label="weight index metadata")
    if metadata.get("total_size") != EXPECTED_MODEL_BYTES:
        raise ConfigurationError(
            f"weight index total_size must be {EXPECTED_MODEL_BYTES}",
            component="inkling_preflight",
        )
    raw_weight_map = _mapping(weight_index.get("weight_map"), label="weight_map")
    weight_map = {str(name): str(filename) for name, filename in raw_weight_map.items()}
    if len(weight_map) != EXPECTED_SOURCE_TENSORS:
        raise ConfigurationError(
            f"weight_map must contain {EXPECTED_SOURCE_TENSORS} tensors, got {len(weight_map)}"
        )
    unsafe = sorted(
        {
            filename
            for filename in weight_map.values()
            if not filename.endswith(".safetensors")
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
        }
    )
    if unsafe:
        raise ConfigurationError(
            "weight index references non-safetensors or unsafe files: " + ", ".join(unsafe[:3])
        )
    shard_count = len(set(weight_map.values()))
    if shard_count != EXPECTED_SOURCE_SHARDS:
        raise ConfigurationError(
            f"weight index must reference {EXPECTED_SOURCE_SHARDS} shards, got {shard_count}"
        )

    vision = sum(name.startswith("model.visual.") for name in weight_map)
    audio = sum(name.startswith("model.audio.") for name in weight_map)
    mtp = sum(name.startswith("model.mtp.") for name in weight_map)
    text = len(weight_map) - vision - audio - mtp
    actual_counts = (text, vision, audio, mtp)
    expected_counts = (
        EXPECTED_TEXT_TENSORS,
        EXPECTED_VISION_TENSORS,
        EXPECTED_AUDIO_TENSORS,
        EXPECTED_MTP_TENSORS,
    )
    if actual_counts != expected_counts:
        raise ConfigurationError(
            f"Inkling component tensor counts must be {expected_counts}, got {actual_counts}"
        )

    index_sha256 = _canonical_sha256(weight_index)
    if index_sha256 != EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256:
        raise ConfigurationError(
            "weight index canonical SHA-256 does not match the pinned Inkling revision",
            component="inkling_preflight",
            details={"actual": index_sha256},
        )

    return InklingSourceAudit(
        model_id=config.source.model_id,
        revision=config.source.revision,
        architecture=config.source.architecture,
        model_type=config.source.model_type,
        license=config.source.license,
        source_bytes=EXPECTED_MODEL_BYTES,
        source_tensor_count=len(weight_map),
        source_shard_count=shard_count,
        text_tensor_count=text,
        vision_tensor_count=vision,
        audio_tensor_count=audio,
        mtp_tensor_count=mtp,
        converted_source_tensor_count=text + vision + audio,
        omitted_source_tensor_count=mtp,
        weight_index_sha256=index_sha256,
        verified=True,
        warnings=(
            "Inkling MTP is omitted because PR #25731 has no MTP converter/runtime support.",
            "llama.cpp support is experimental and pinned to unmerged PR #25731.",
        ),
    )


def build_conversion_plan(config: InklingGGUFConfig, paths: WorkflowPaths) -> ConversionPlan:
    """Build fixed local-snapshot converter commands; remote branch resolution is forbidden."""

    converter = paths.llama_cpp_dir / "convert_hf_to_gguf.py"
    text_output = paths.work_dir / "inkling-BF16.gguf"
    mmproj_output = paths.final_dir / "mmproj-BF16.gguf"
    common = ("python", str(converter), str(paths.source_dir))
    text = CommandSpec(
        name="convert_text_bf16",
        argv=(
            *common,
            "--outtype",
            "bf16",
            "--outfile",
            str(text_output),
            "--split-max-size",
            config.quantization.split_max_size,
            "--no-tensor-first-split",
        ),
        timeout_seconds=int(modal_stage_resources(config, "convert_text_bf16").max_hours * 3600),
    )
    mmproj = CommandSpec(
        name="convert_multimodal_projector",
        argv=(
            *common,
            "--mmproj",
            "--outtype",
            "bf16",
            "--outfile",
            str(mmproj_output),
        ),
        timeout_seconds=int(
            modal_stage_resources(config, "convert_multimodal_projector").max_hours * 3600
        ),
    )
    return ConversionPlan(text_conversion=text, mmproj_conversion=mmproj)


def build_quantize_command(
    config: InklingGGUFConfig,
    paths: WorkflowPaths,
    *,
    first_bf16_split: Path,
) -> CommandSpec:
    """Bind stock Q3_K_M quantization after discovering the converter's first split."""

    work_root = paths.work_dir.resolve()
    resolved_input = first_bf16_split.resolve()
    if (
        not resolved_input.is_relative_to(work_root)
        or not resolved_input.name.startswith("inkling-BF16-00001-of-")
        or resolved_input.suffix != ".gguf"
    ):
        raise ConfigurationError("first BF16 split is not a safe converter output")
    quantizer = paths.llama_cpp_dir / "build" / "bin" / "llama-quantize"
    output = paths.final_dir / f"{config.quantization.output_label}.gguf"
    return CommandSpec(
        name="quantize_text",
        argv=(
            str(quantizer),
            "--keep-split",
            str(resolved_input),
            str(output),
            config.quantization.quant_type,
            str(config.quantization.threads),
        ),
        timeout_seconds=int(modal_stage_resources(config, "quantize_text").max_hours * 3600),
    )


def build_verification_plan(
    config: InklingGGUFConfig,
    paths: WorkflowPaths,
    *,
    first_q3_split: Path,
    mmproj_file: Path,
) -> VerificationPlan:
    """Parse every Q3 split through merge dry-run and parse the separate mmproj."""

    final_root = paths.final_dir.resolve()
    resolved_q3 = first_q3_split.resolve()
    resolved_mmproj = mmproj_file.resolve()
    if (
        not resolved_q3.is_relative_to(final_root)
        or resolved_q3.parent != final_root / "q3_k_m"
        or not resolved_q3.name.startswith("inkling-Q3_K_M-00001-of-")
        or resolved_q3.suffix != ".gguf"
    ):
        raise ConfigurationError("first Q3 split is not a safe final output")
    expected_mmproj = (final_root / "mmproj" / "mmproj-BF16.gguf").resolve()
    if resolved_mmproj != expected_mmproj:
        raise ConfigurationError("mmproj path is not the exact final projector output")
    splitter = paths.llama_cpp_dir / "build" / "bin" / "llama-gguf-split"
    timeout_seconds = int(modal_stage_resources(config, "verify_export").max_hours * 3600)
    return VerificationPlan(
        q3_split_set=CommandSpec(
            name="verify_q3_split_set",
            argv=(
                str(splitter),
                "--merge",
                "--dry-run",
                str(resolved_q3),
                str(final_root / ".verify-merge-plan.gguf"),
            ),
            timeout_seconds=timeout_seconds,
        ),
        mmproj=CommandSpec(
            name="verify_mmproj",
            argv=(
                str(splitter),
                "--split",
                "--dry-run",
                str(resolved_mmproj),
                str(final_root / ".verify-mmproj-plan.gguf"),
            ),
            timeout_seconds=timeout_seconds,
        ),
    )


def verify_execution_bindings(
    config: InklingGGUFConfig,
    paths: WorkflowPaths,
    *,
    source_receipt: Mapping[str, Any],
    actual_llama_cpp_commit: str,
) -> ExecutionBindingEvidence:
    """Bind runtime mount paths to the source/toolchain identities recorded in the plan."""

    expected = {
        "verified": True,
        "config_hash": config.config_hash(),
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "license": config.source.license,
        "source_dir": str(paths.source_dir),
    }
    mismatches = [key for key, value in expected.items() if source_receipt.get(key) != value]
    index_sha = source_receipt.get("weight_index_sha256")
    if index_sha != EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256:
        mismatches.append("weight_index_sha256")
    inventory_sha = source_receipt.get("inventory_sha256")
    if not isinstance(inventory_sha, str) or re.fullmatch(r"[0-9a-f]{64}", inventory_sha) is None:
        mismatches.append("inventory_sha256")
    source_config_sha = source_receipt.get("source_config_sha256")
    if (
        not isinstance(source_config_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_config_sha) is None
    ):
        mismatches.append("source_config_sha256")
    if mismatches:
        raise ConfigurationError(
            "materialized source receipt does not bind the configured source: "
            + ", ".join(sorted(set(mismatches))),
            component="inkling_execution_binding",
        )
    if actual_llama_cpp_commit != config.toolchain.commit:
        raise ConfigurationError(
            "llama.cpp checkout does not match the pinned Inkling support commit",
            component="inkling_execution_binding",
        )
    return ExecutionBindingEvidence(
        source_dir=str(paths.source_dir),
        source_config_hash=config.config_hash(),
        model_id=config.source.model_id,
        revision=config.source.revision,
        weight_index_sha256=index_sha,
        source_inventory_sha256=inventory_sha,
        source_config_sha256=source_config_sha,
        llama_cpp_dir=str(paths.llama_cpp_dir),
        llama_cpp_commit=actual_llama_cpp_commit,
    )


JsonFetcher = Callable[[str, str | None], Mapping[str, Any]]


def fetch_public_json(url: str, token: str | None = None) -> Mapping[str, Any]:
    """Fetch one public JSON document without persisting credentials or model code."""

    headers = {"Accept": "application/json", "User-Agent": "inkling-quant-lab/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            value = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"Unable to fetch required Inkling metadata from {url}: {error}",
            component="inkling_preflight",
        ) from error
    return _mapping(value, label=f"JSON response from {url}")


def audit_pinned_inkling_online(
    config: InklingGGUFConfig,
    *,
    token: str | None = None,
    fetcher: JsonFetcher = fetch_public_json,
) -> InklingSourceAudit:
    """Fetch only pinned metadata and return the exact-source audit."""

    model = config.source.model_id
    revision = config.source.revision
    base = f"https://huggingface.co/{model}"
    model_info = fetcher(f"https://huggingface.co/api/models/{model}/revision/{revision}", token)
    model_config = fetcher(f"{base}/resolve/{revision}/config.json", token)
    weight_index = fetcher(f"{base}/resolve/{revision}/model.safetensors.index.json", token)
    return audit_inkling_source(
        config,
        model_info=model_info,
        model_config=model_config,
        weight_index=weight_index,
    )


__all__ = [
    "DASHBOARD_EXACT_UTC_SOURCE",
    "EXPECTED_AUDIO_TENSORS",
    "EXPECTED_LICENSE",
    "EXPECTED_MODEL_BYTES",
    "EXPECTED_MTP_TENSORS",
    "EXPECTED_TEXT_TENSORS",
    "EXPECTED_VISION_TENSORS",
    "EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256",
    "FULL_INITIAL_BILLING_WINDOW_POLICY",
    "INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH",
    "PINNED_INKLING_REVISION",
    "PINNED_LLAMA_CPP_COMMIT",
    "SHORT_INITIAL_BILLING_WINDOW_POLICY",
    "USER_CONFIRMED_ASSUMED_UTC_SOURCE",
    "CommandSpec",
    "ControlPlaneFile",
    "ControlPlaneProvenance",
    "ConversionPlan",
    "ExecutionBindingEvidence",
    "InklingGGUFConfig",
    "InklingSourceAdoptionReference",
    "InklingSourceAudit",
    "PaidLaunchAcknowledgement",
    "SourceAdoptionArtifact",
    "VerificationPlan",
    "WorkflowPaths",
    "audit_inkling_source",
    "audit_pinned_inkling_online",
    "build_conversion_plan",
    "build_quantize_command",
    "build_verification_plan",
    "compute_cost_ceiling_usd",
    "configured_deployable_wall_time_hours",
    "inkling_control_plane_provenance",
    "inkling_run_id",
    "inkling_source_adoption_reference_sha256",
    "load_inkling_gguf_config",
    "load_inkling_source_adoption_reference",
    "modal_stage_resources",
    "require_initial_billing_window",
    "require_materialize_initial_billing_window",
    "require_stage_billing_window",
    "validate_deployed_control_plane",
    "validate_inkling_source_adoption_reference",
    "validate_paid_launch_acknowledgement",
    "verify_execution_bindings",
]
