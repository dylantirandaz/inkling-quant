"""Contracts for the public-model native CPU quality evidence script."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast

import pytest
import yaml
from torch import nn

from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.models.base import (
    LoadedModel,
    LossOutput,
    ModelCapabilities,
    ModelDescriptor,
    ModelOutput,
    ModuleInfo,
    MoEDescriptor,
    MoELayerDescriptor,
)
from inkling_quant_lab.models.hf_causal_lm import _module_supported_precisions
from inkling_quant_lab.quantization.policies import resolve_precision_policy
from scripts import evaluate_public_moe_native_quality as quality_script
from scripts.evaluate_public_moe_native_quality import (
    OFFICIAL_TINYSTORIES_VALID_SHA256,
    ConfirmatorySample,
    ConfirmatorySelection,
    evaluate,
    generation_retention,
    load_confirmatory_protocol,
    prepare_public_data,
    require_hardware_label,
    require_official_dataset_sha256,
    resolve_quality_output_path,
    validate_confirmatory_experiment_contracts,
    validate_confirmatory_selection,
    validate_experiment_contracts,
    validate_native_policy,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "experiments"
PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "evaluations" / "stories15m_native_int8_confirmatory_256.yaml"
)


def _configs() -> tuple[ExperimentConfig, ExperimentConfig, ExperimentConfig]:
    baseline = load_config(CONFIG_ROOT / "hf_stories15m_tinystories_native_quality.yaml")
    int8 = load_config(CONFIG_ROOT / "hf_stories15m_tinystories_native_int8_quality.yaml")
    int4 = load_config(CONFIG_ROOT / "hf_stories15m_tinystories_native_int4_quality.yaml")
    return baseline, int8, int4


def test_checked_public_native_quality_configs_share_exact_contract() -> None:
    baseline, int8, int4 = _configs()

    validate_experiment_contracts(baseline, (int8, int4))

    perplexity = next(suite for suite in baseline.evaluation.suites if suite.type == "perplexity")
    generation = next(
        suite for suite in baseline.evaluation.suites if suite.type == "generation_regression"
    )
    assert len(perplexity.sample_ids) == 32
    assert generation.sample_ids == (
        "story-000001",
        "story-000008",
        "story-000016",
        "story-000024",
    )
    assert not baseline.security.log_prompts
    assert not baseline.security.log_model_outputs
    assert int8.quantization.policy.default_precision == "float32"
    assert int4.quantization.policy.default_precision == "float32"


def _confirmatory_configs() -> tuple[ExperimentConfig, ExperimentConfig]:
    baseline = load_config(CONFIG_ROOT / "hf_stories15m_tinystories_native_confirmatory_256.yaml")
    int8 = load_config(CONFIG_ROOT / "hf_stories15m_tinystories_native_int8_confirmatory_256.yaml")
    return baseline, int8


def test_checked_confirmatory_configs_and_protocol_are_hash_exact_and_disjoint() -> None:
    baseline, int8 = _confirmatory_configs()
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)

    validate_confirmatory_experiment_contracts(baseline, (int8,), protocol)

    quality = next(suite for suite in baseline.evaluation.suites if suite.type == "perplexity")
    descriptive = next(
        suite for suite in baseline.evaluation.suites if suite.type == "generation_regression"
    )
    excluded = {f"story-{index:06d}" for index in range(1, 33)}
    assert len(quality.sample_ids) == 256
    assert len(descriptive.sample_ids) == 16
    assert not excluded.intersection(quality.sample_ids)
    assert set(descriptive.sample_ids).issubset(quality.sample_ids)
    assert quality.sample_ids == tuple(sorted(quality.sample_ids))
    assert descriptive.sample_ids == tuple(sorted(descriptive.sample_ids))
    assert quality_script._canonical_json_sha256(list(quality.sample_ids)) == (
        protocol.holdout_selection.quality_sample_id_list_sha256
    )
    assert quality_script._canonical_json_sha256(list(descriptive.sample_ids)) == (
        protocol.generation_and_routing.sample_id_list_sha256
    )
    assert protocol.noninferiority.margin_relative_perplexity == 0.005
    assert protocol.noninferiority.margin_nats_per_token == pytest.approx(
        0.004987541511039074, rel=0.0, abs=0.0
    )
    assert protocol.execution_contract.baseline_resolved_config_sha256 == baseline.config_hash()
    assert protocol.execution_contract.candidate_resolved_config_sha256 == int8.config_hash()
    assert protocol.model.generation_config_json_sha256 == (
        "295aa491adda22ab9fbdecdda9e8121e8348fd0eea0529d8802993426ab0892c"
    )


def test_confirmatory_protocol_loader_rejects_unknown_fields_and_scalar_coercion(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["undeclared"] = True
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_confirmatory_protocol(unknown)

    raw.pop("undeclared")
    cast(dict[str, Any], raw["holdout_selection"])["seed"] = "20260715"
    coerced = tmp_path / "coerced.yaml"
    coerced.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="20260715"):
        load_confirmatory_protocol(coerced)


def test_confirmatory_contract_rejects_tampered_checked_ids_before_execution() -> None:
    baseline, int8 = _confirmatory_configs()
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)

    def insert_prior_sample(config: ExperimentConfig) -> ExperimentConfig:
        raw = config.canonical_dict()
        sample_ids = raw["evaluation"]["suites"][0]["sample_ids"]
        sample_ids[0] = "story-000001"
        return ExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="exclude all prior"):
        validate_confirmatory_experiment_contracts(
            insert_prior_sample(baseline), (insert_prior_sample(int8),), protocol
        )


def test_confirmatory_selection_validation_rejects_tampered_digest() -> None:
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)
    empty = ConfirmatorySelection(
        quality=(),
        generation_and_routing=(),
        excluded_selection_sha256="0" * 64,
        eligible_evaluated_token_count=0,
        quality_sample_id_list_sha256="0" * 64,
        quality_ids_content_sha256="0" * 64,
        quality_manifest_sha256="0" * 64,
        quality_input_token_ids_manifest_sha256="0" * 64,
        generation_sample_id_list_sha256="0" * 64,
        generation_ids_content_sha256="0" * 64,
        generation_manifest_sha256="0" * 64,
        routing_input_token_ids_manifest_sha256="0" * 64,
        generation_prompt_token_ids_manifest_sha256="0" * 64,
        strata=(),
    )

    with pytest.raises(ValueError, match="excluded_selection_sha256"):
        validate_confirmatory_selection(empty, protocol)


def test_confirmatory_cli_requires_exactly_one_candidate() -> None:
    common = [
        "baseline.yaml",
        "candidate.yaml",
        "--dataset",
        "TinyStories-valid.txt",
        "--expected-dataset-sha256",
        OFFICIAL_TINYSTORIES_VALID_SHA256,
        "--hardware-label",
        "test CPU",
        "--confirmatory-protocol",
        str(PROTOCOL_PATH),
        "--execution-ordinal",
        "1",
    ]
    parsed = quality_script._arguments(common)
    assert parsed.confirmatory_protocol == PROTOCOL_PATH
    assert parsed.execution_ordinal == 1

    with pytest.raises(SystemExit):
        quality_script._arguments(["baseline.yaml", "candidate.yaml", "second.yaml", *common[2:]])
    without_ordinal = common[:-2]
    with pytest.raises(SystemExit):
        quality_script._arguments(without_ordinal)
    with pytest.raises(SystemExit):
        quality_script._arguments(
            [
                "baseline.yaml",
                "candidate.yaml",
                "--dataset",
                "TinyStories-valid.txt",
                "--expected-dataset-sha256",
                OFFICIAL_TINYSTORIES_VALID_SHA256,
                "--hardware-label",
                "test CPU",
                "--execution-ordinal",
                "1",
            ]
        )


def _paired_quality_records(*, candidate_delta: float = 0.0) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index in range(256):
        sample_id = f"story-{index + 1000:06d}"
        records.append(
            {
                "sample_id": sample_id,
                "content_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
                "evaluated_token_count": 10,
                "length_stratum": index // 64,
                "mean_nll": 1.0 + candidate_delta,
            }
        )
    return {"samples": records}


def test_confirmatory_noninferiority_uses_exact_prospective_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)
    observed: dict[str, Any] = {}

    class _Result:
        def as_dict(self) -> dict[str, object]:
            return {"passed": True, "upper_confidence_bound_nll": 0.004}

    def fake_gate(
        observations: object,
        strata: object,
        **kwargs: object,
    ) -> _Result:
        observed["observations"] = observations
        observed["strata"] = strata
        observed.update(kwargs)
        return _Result()

    monkeypatch.setattr(quality_script, "paired_stratified_nll_noninferiority", fake_gate)

    result = quality_script._confirmatory_noninferiority(
        _paired_quality_records(),
        _paired_quality_records(candidate_delta=0.001),
        protocol,
    )

    assert result["passed"] is True
    assert len(observed["observations"]) == 256
    assert [item.population_size for item in observed["strata"]] == [5490, 5489, 5490, 5489]
    assert observed["exact_population_token_total"] == 4_441_967
    assert observed["margin_nll"] == 0.004987541511039074
    assert observed["confidence"] == 0.95
    assert observed["bootstrap_replicates"] == 100_000
    assert observed["seed"] == 20260715


def test_confirmatory_noninferiority_rejects_partial_samples() -> None:
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)
    candidate = _paired_quality_records()
    candidate["samples"].pop()

    with pytest.raises(ValueError, match="all 256 paired"):
        quality_script._confirmatory_noninferiority(_paired_quality_records(), candidate, protocol)


def test_single_confirmatory_execution_is_always_provisional() -> None:
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)

    status = quality_script._confirmatory_status_record(
        protocol, {"passed": True}, execution_ordinal=2
    )

    assert status == {
        "status": "provisional_until_2_clean_executions",
        "confirmatory_claim_ready": False,
        "required_clean_process_executions": 2,
        "execution_ordinal": 2,
        "within_execution_noninferiority_passed": True,
        "within_execution_decision_is_overall_confirmatory_claim": False,
        "pair_verification_required": True,
    }
    assert "overall_confirmatory_passed" not in status


def _observer_selection() -> ConfirmatorySelection:
    generation_indexes = tuple(
        index for stratum in range(4) for index in range(stratum * 64, stratum * 64 + 4)
    )
    generation_set = set(generation_indexes)
    remaining = [index for index in range(256) if index not in generation_set]
    evaluated_counts: dict[int, int] = {}
    for position, index in enumerate(generation_indexes):
        evaluated_counts[index] = 205 if position < 14 else 204
    for position, index in enumerate(remaining):
        evaluated_counts[index] = 204 if position < 40 else 203
    assert sum(evaluated_counts.values()) == 52_038

    samples: list[ConfirmatorySample] = []
    for index in range(256):
        sample_id = f"story-{index + 1000:06d}"
        count = evaluated_counts[index]
        samples.append(
            ConfirmatorySample(
                sample_id=sample_id,
                story=sample_id,
                content_sha256=hashlib.sha256(sample_id.encode()).hexdigest(),
                token_ids=tuple(range(count + 1)),
                source_token_count=count + 1,
                evaluated_token_count=count,
                truncated=False,
                length_stratum=index // 64,
            )
        )
    quality = tuple(samples)
    descriptive = tuple(samples[index] for index in generation_indexes)
    assert sum(len(sample.token_ids) for sample in descriptive) == 3_294
    return ConfirmatorySelection(
        quality=quality,
        generation_and_routing=descriptive,
        excluded_selection_sha256="a" * 64,
        eligible_evaluated_token_count=4_441_967,
        quality_sample_id_list_sha256="b" * 64,
        quality_ids_content_sha256="c" * 64,
        quality_manifest_sha256="d" * 64,
        quality_input_token_ids_manifest_sha256="2" * 64,
        generation_sample_id_list_sha256="e" * 64,
        generation_ids_content_sha256="f" * 64,
        generation_manifest_sha256="1" * 64,
        routing_input_token_ids_manifest_sha256="3" * 64,
        generation_prompt_token_ids_manifest_sha256="4" * 64,
        strata=(),
    )


def test_confirmatory_measurement_scopes_hooks_to_exact_reforward_and_proves_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _confirmatory_configs()
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)
    selection = _observer_selection()
    data = quality_script.PreparedPublicData(
        dataset_sha256=OFFICIAL_TINYSTORIES_VALID_SHA256,
        story_count=21_990,
        perplexity_ids=tuple(sample.sample_id for sample in selection.quality),
        perplexity_stories=tuple(sample.story for sample in selection.quality),
        perplexity_selection_sha256="c" * 64,
        generation_ids=tuple(sample.sample_id for sample in selection.generation_and_routing),
        generation_stories=tuple(sample.story for sample in selection.generation_and_routing),
        generation_selection_sha256="f" * 64,
        confirmatory_selection=selection,
    )

    class _Tokenizer:
        def __init__(self) -> None:
            self.tokens = {sample.story: list(sample.token_ids) for sample in selection.quality}

        def encode(self, text: str, *, special_tokens: bool = True) -> list[int]:
            assert special_tokens
            return self.tokens[text]

    capabilities = ModelCapabilities(
        supports_text=True,
        supports_images=False,
        supports_audio=False,
        is_moe=True,
        supports_router_logits=False,
        supports_token_level_routes=True,
        supported_dtypes=("float32",),
        supported_device_maps=("single",),
        max_context_length=256,
    )
    loaded = LoadedModel(
        model=nn.Linear(1, 1),
        tokenizer=_Tokenizer(),
        descriptor=ModelDescriptor(
            model_id="fixture",
            revision="fixture",
            resolved_class="fixture",
            architecture="fixture",
            checksum="fixture",
            capabilities=capabilities,
        ),
        load_time_seconds=0.0,
    )

    class _Handle:
        def __init__(self, adapter: _Adapter) -> None:
            self.adapter = adapter

        def remove(self) -> None:
            self.adapter.active = False
            self.adapter.timeline.append("hooks_removed")

    class _Adapter:
        def __init__(self) -> None:
            self.active = False
            self.timeline: list[str] = []

        def discover_moe(self, model: LoadedModel) -> MoEDescriptor:
            del model
            return MoEDescriptor(
                layers=tuple(
                    MoELayerDescriptor(
                        layer_id=f"layer-{index}",
                        module_name=f"layer-{index}",
                        router_module_name=f"layer-{index}.gate",
                        expert_module_names=(f"layer-{index}.expert",),
                        expert_count=1,
                        top_k=1,
                    )
                    for index in range(6)
                ),
                supports_router_logits=False,
                supports_token_level_routes=True,
            )

        def attach_routing_hooks(
            self, model: LoadedModel, sink: object, experiment: ExperimentConfig
        ) -> _Handle:
            del model, sink, experiment
            assert not self.active
            self.active = True
            self.timeline.append("hooks_attached")
            return _Handle(self)

        def forward_loss(self, model: LoadedModel, batch: object) -> LossOutput:
            del model
            typed = cast(Any, batch)
            sample_id = typed.sample_ids[0]
            count = int(typed.input_ids.shape[1]) - 1
            self.timeline.append("routing_loss" if self.active else "quality_loss")
            return LossOutput(
                sample_ids=(sample_id,),
                negative_log_likelihoods=(1.0 + int(sample_id[-2:]) / 1000.0,),
                token_counts=(count,),
            )

        def generate(self, model: LoadedModel, batch: object, decode: object) -> ModelOutput:
            del model, decode
            assert not self.active
            self.timeline.append("generation")
            sample_id = cast(Any, batch).sample_ids[0]
            return ModelOutput(sample_ids=(sample_id,), token_ids=((7, 8),))

    class _Sink:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start_batch(self, batch: object) -> None:
            del batch

        def end_batch(self) -> None:
            pass

        def close(self) -> object:
            return SimpleNamespace(
                observed_event_count=19_764,
                recorded_event_count=19_764,
                batch_count=16,
                raw_traces=tuple(
                    SimpleNamespace(
                        alignment_key=(sample.sample_id, token_position, f"layer-{layer}")
                    )
                    for sample in selection.generation_and_routing
                    for token_position in range(len(sample.token_ids))
                    for layer in range(6)
                ),
            )

    monkeypatch.setattr(quality_script, "InMemoryRoutingSink", _Sink)
    adapter = _Adapter()

    measurement = quality_script._measure_confirmatory_loaded_model(
        config, data, cast(Any, adapter), loaded, protocol
    )

    assert adapter.timeline[:256] == ["quality_loss"] * 256
    assert adapter.timeline[256] == "hooks_attached"
    assert adapter.timeline[257:273] == ["routing_loss"] * 16
    assert adapter.timeline[273] == "hooks_removed"
    assert adapter.timeline[274:] == ["generation"] * 16
    assert measurement.quality["sample_count"] == 256
    assert measurement.quality["evaluated_token_count"] == 52_038
    assert measurement.routing_observer_proof is not None
    assert measurement.routing_observer_proof["model_state_unchanged"] is True
    assert measurement.routing_observer_proof["routing_hooks_removed_before_generation"] is True
    assert measurement.routing_observer_proof["observed_layer_token_event_count"] == 19_764
    serialized = json.dumps(measurement.routing_observer_proof)
    assert "token_ids" not in serialized
    assert '"text"' not in serialized
    assert "story_text" not in serialized


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("IQL_RUN_CACHED_CONFIRMATORY_SELECTION") != "1",
    reason="set IQL_RUN_CACHED_CONFIRMATORY_SELECTION=1 for the pinned tokenizer-only audit",
)
def test_cached_tokenizer_rederives_confirmatory_golden_manifests() -> None:
    from transformers import AutoTokenizer

    from inkling_quant_lab.models.hf_causal_lm import HFTokenizerAdapter

    dataset = Path("/private/tmp/TinyStories-valid-f54c09fd.txt")
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--ggml-org--stories15M_MOE/snapshots"
        / "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    )
    if not dataset.is_file() or not snapshot.is_dir():
        pytest.skip("pinned dataset/tokenizer cache is unavailable")
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)
    tokenizer = HFTokenizerAdapter(
        AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    )
    stories = quality_script.split_stories(dataset.read_text(encoding="utf-8"))

    selection = quality_script.derive_confirmatory_selection(stories, tokenizer, protocol)

    assert selection.quality_sample_id_list_sha256 == (
        "1eb2da103e38898e93be80cec9a41cd28bf3a4dcc964d886905423c05e7d0203"
    )
    assert selection.quality_manifest_sha256 == (
        "3e290e440cb6c6c4e917682793c9af6039695d10305188da0da821277c41b91b"
    )
    assert selection.generation_sample_id_list_sha256 == (
        "a7ee74daec91ec377392f9f8a10606ecfb86ef47b8371391e24e97cf12febe83"
    )
    assert selection.generation_manifest_sha256 == (
        "69a6dbca6a03d621063b154c64ef3fa02a4c86b78f4a7bff3fc11fefdc63cc11"
    )
    assert selection.quality_input_token_ids_manifest_sha256 == (
        "4a93334e9ad18e4508fa58c368bcb2c2875d8f324b1f3c54efa2214bf3ea8bdb"
    )
    assert selection.routing_input_token_ids_manifest_sha256 == (
        "8a0059b59e4ab04d674fc9122c070017ca31f44600f690b4973b567fcdae1055"
    )
    assert selection.generation_prompt_token_ids_manifest_sha256 == (
        "923060f0f3dc78919b5b7356a7adcc5b3c362e119e31a99f6459570962b0293c"
    )


def test_public_data_selection_is_checksum_verified_and_declared(tmp_path: Path) -> None:
    baseline, _, _ = _configs()
    dataset = tmp_path / "valid.txt"
    dataset.write_text(
        "<|endoftext|>".join(f"story {index}" for index in range(33)),
        encoding="utf-8",
    )
    expected = hashlib.sha256(dataset.read_bytes()).hexdigest()

    prepared = prepare_public_data(
        baseline,
        dataset,
        expected_dataset_sha256=expected,
    )

    assert prepared.story_count == 33
    assert prepared.perplexity_stories[0] == "story 1"
    assert prepared.perplexity_stories[-1] == "story 32"
    assert prepared.generation_stories == (
        "story 1",
        "story 8",
        "story 16",
        "story 24",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_public_data(baseline, dataset, expected_dataset_sha256="0" * 64)


def test_pinned_evaluator_rejects_a_caller_selected_dataset_digest(tmp_path: Path) -> None:
    baseline, int8, int4 = _configs()

    with pytest.raises(ValueError, match="official TinyStories validation SHA-256"):
        evaluate(
            baseline,
            (int8, int4),
            tmp_path / "not-opened.txt",
            expected_dataset_sha256="0" * 64,
            project_root=tmp_path,
            hardware_label="test CPU (unit fixture)",
        )

    assert require_official_dataset_sha256(OFFICIAL_TINYSTORIES_VALID_SHA256) == (
        OFFICIAL_TINYSTORIES_VALID_SHA256
    )
    with pytest.raises(ValueError, match="official TinyStories validation SHA-256"):
        require_official_dataset_sha256("0" * 64)


@pytest.mark.parametrize("value", ("", " ", "\t", " Apple M3"))
def test_hardware_label_must_be_exact_and_non_empty(value: str) -> None:
    with pytest.raises(ValueError, match="exact non-empty hardware label"):
        require_hardware_label(value)


def test_public_quality_cli_requires_hardware_label() -> None:
    arguments = [
        "baseline.yaml",
        "candidate.yaml",
        "--dataset",
        "TinyStories-valid.txt",
        "--expected-dataset-sha256",
        OFFICIAL_TINYSTORIES_VALID_SHA256,
    ]

    with pytest.raises(SystemExit):
        quality_script._arguments(arguments)


def test_quality_output_must_be_below_configured_project_artifact_root(
    tmp_path: Path,
) -> None:
    baseline, _, _ = _configs()

    output = resolve_quality_output_path(
        baseline,
        project_root=tmp_path,
        requested=Path("artifacts/research-slices/result.json"),
    )

    assert output == (tmp_path / "artifacts/research-slices/result.json").resolve()
    with pytest.raises(ArtifactIntegrityError, match=r"cannot contain '\.\.'"):
        resolve_quality_output_path(
            baseline,
            project_root=tmp_path,
            requested=Path("artifacts/../outside.json"),
        )
    with pytest.raises(ArtifactIntegrityError, match="outside the configured artifact root"):
        resolve_quality_output_path(
            baseline,
            project_root=tmp_path,
            requested=Path("docs/outside.json"),
        )


def test_quality_output_rejects_symlink_escapes(tmp_path: Path) -> None:
    baseline, _, _ = _configs()
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifact_root.mkdir()
    outside.mkdir()
    (artifact_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="outside the configured artifact root"):
        resolve_quality_output_path(
            baseline,
            project_root=tmp_path,
            requested=Path("artifacts/escape/result.json"),
        )


def test_quality_record_publication_never_replaces_a_racing_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "record.json"
    winner = b"concurrent immutable winner\n"
    real_link = quality_script.os.link

    def racing_link(source: object, destination: object) -> None:
        Path(cast(Any, destination)).write_bytes(winner)
        real_link(source, destination)

    monkeypatch.setattr(quality_script.os, "link", racing_link)

    with pytest.raises(FileExistsError, match="refusing to overwrite quality record"):
        quality_script._write_record(output, {"schema_version": "loser"})

    assert output.read_bytes() == winner
    assert tuple(tmp_path.glob(".record.json.*.tmp")) == ()


def test_main_rejects_unsafe_output_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = argparse.Namespace(
        baseline_config=CONFIG_ROOT / "hf_stories15m_tinystories_native_quality.yaml",
        candidate_configs=(
            CONFIG_ROOT / "hf_stories15m_tinystories_native_int8_quality.yaml",
            CONFIG_ROOT / "hf_stories15m_tinystories_native_int4_quality.yaml",
        ),
        dataset=tmp_path / "not-opened.txt",
        expected_dataset_sha256=OFFICIAL_TINYSTORIES_VALID_SHA256,
        hardware_label="test CPU (unit fixture)",
        output=Path("../escape.json"),
    )
    evaluation_calls: list[bool] = []

    def forbidden_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        evaluation_calls.append(True)
        return {}

    monkeypatch.setattr(quality_script, "_arguments", lambda: arguments)
    monkeypatch.setattr(quality_script, "evaluate", forbidden_evaluate)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ArtifactIntegrityError, match=r"cannot contain '\.\.'"):
        quality_script.main()

    assert evaluation_calls == []


@pytest.mark.parametrize("suite_index", (0, 1))
def test_raw_story_evaluation_requires_exact_text_prompt_template(suite_index: int) -> None:
    baseline, int8, int4 = _configs()

    def changed(config: ExperimentConfig) -> ExperimentConfig:
        raw = config.canonical_dict()
        raw["evaluation"]["suites"][suite_index]["prompt_template"] = "Story: {text}"
        return ExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="exact '\\{text\\}' prompt template"):
        validate_experiment_contracts(changed(baseline), (changed(int8), changed(int4)))


@pytest.mark.parametrize("local_files_only", (False, True))
def test_source_weight_verification_honors_configured_network_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_files_only: bool,
) -> None:
    baseline, _, _ = _configs()
    raw = baseline.canonical_dict()
    raw["model"]["local_files_only"] = local_files_only
    config = ExperimentConfig.model_validate(raw)
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"safe-weight-fixture")
    observed: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        observed.update(kwargs)
        return str(weight)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.__dict__["hf_hub_download"] = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(quality_script, "_MODEL_WEIGHT_SIZE_BYTES", weight.stat().st_size)
    monkeypatch.setattr(quality_script, "_MODEL_WEIGHT_SHA256", quality_script.file_sha256(weight))

    quality_script._verified_cached_source_weight(config)

    assert observed["repo_id"] == "ggml-org/stories15M_MOE"
    assert observed["revision"] == "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    assert observed["local_files_only"] is local_files_only


def test_confirmatory_metadata_verifier_hashes_every_preload_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline, _ = _confirmatory_configs()
    names = (
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    paths: dict[str, Path] = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(f"pinned-{name}".encode())
        paths[name] = path
    calls: list[str] = []

    def fake_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        calls.append(filename)
        assert kwargs["repo_id"] == "ggml-org/stories15M_MOE"
        assert kwargs["revision"] == "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
        assert kwargs["local_files_only"] is True
        return str(paths[filename])

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.__dict__["hf_hub_download"] = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    expected = {name: quality_script.file_sha256(paths[name]) for name in names}
    monkeypatch.setattr(quality_script, "_TOKENIZER_FILE_SHA256", expected)

    assert quality_script._verified_cached_tokenizer_files(baseline) == expected
    assert calls == list(names)


def test_checked_curated_full_evidence_is_hash_linked_and_content_redacted() -> None:
    summary = json.loads(
        (PROJECT_ROOT / "docs/experiments/stories15m-moe-native-linear-quality.json").read_text(
            encoding="utf-8"
        )
    )
    full_path = PROJECT_ROOT / summary["full_record"]["path"]
    full_bytes = full_path.read_bytes()
    full = json.loads(full_bytes)

    assert hashlib.sha256(full_bytes).hexdigest() == summary["full_record"]["sha256"]
    assert full["schema_version"] == "public-moe-native-linear-quality-curated-v2"
    assert full["evidence_curation"]["prompt_or_output_content_present"] is False
    assert full["protocol"]["prompt_or_output_content_persisted"] is False
    assert summary["reproducibility_check"]["independent_execution_count"] == 3
    assert len(summary["reproducibility_check"]["execution_records"]) == 3
    assert (
        summary["reproducibility_check"]["post_gate_reexecution"][
            "quality_generation_routing_model_state_and_tensor_storage_match_originals"
        ]
        is True
    )
    assert (
        summary["reproducibility_check"]["canonical_scientific_projection"]["sha256_each_execution"]
        == "392352902162c583574ccc479f97f4d8220006c40e5c04cf9d6427d92bec50a4"
    )

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not {"prompt", "text", "output_ids", "token_ids"}.intersection(keys(full))


class _PolicyFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.gate = nn.Linear(8, 4)
        self.lm_head = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(8)


def _policy_inventory() -> tuple[ModuleInfo, ...]:
    return (
        ModuleInfo(
            name="proj",
            class_name="torch.nn.modules.linear.Linear",
            parameter_count=72,
            size_bytes=288,
            supported_precisions=("float32", "int8", "int4"),
        ),
        ModuleInfo(
            name="gate",
            class_name="torch.nn.modules.linear.Linear",
            parameter_count=36,
            size_bytes=144,
            is_router=True,
            supported_precisions=("float32", "int8", "int4"),
        ),
        ModuleInfo(
            name="lm_head",
            class_name="torch.nn.modules.linear.Linear",
            parameter_count=144,
            size_bytes=576,
            is_output_head=True,
            supported_precisions=("float32", "int8", "int4"),
        ),
        ModuleInfo(
            name="norm",
            class_name="torch.nn.modules.normalization.LayerNorm",
            parameter_count=16,
            size_bytes=64,
            supported_precisions=("float32",),
        ),
        ModuleInfo(
            name="experts.0",
            class_name="transformers.models.mixtral.modeling_mixtral.MixtralExperts.ExpertSlice",
            parameter_count=128,
            size_bytes=512,
            is_expert=True,
            layer_id="moe",
            expert_id=0,
            supported_precisions=("float32",),
        ),
    )


@pytest.mark.parametrize(
    ("config_name", "precision"),
    (
        ("hf_stories15m_tinystories_native_int8_quality.yaml", "int8"),
        ("hf_stories15m_tinystories_native_int4_quality.yaml", "int4"),
    ),
)
def test_native_policy_selects_only_unprotected_concrete_linears(
    config_name: str, precision: str
) -> None:
    config = load_config(CONFIG_ROOT / config_name)
    inventory = _policy_inventory()
    policy = resolve_precision_policy(inventory, config.quantization.policy)

    selected = validate_native_policy(
        _PolicyFixture(),
        inventory,
        policy,
        target_precision=cast(Literal["int8", "int4"], precision),
    )

    assert selected == ("proj",)
    assert policy.precision_map["gate"] == "float32"
    assert policy.precision_map["lm_head"] == "float32"
    assert policy.precision_map["experts.0"] == "float32"


def test_hf_inventory_precision_advertising_is_concrete_linear_only() -> None:
    assert _module_supported_precisions(nn.Linear(4, 4)) == ("float32", "int8", "int4")
    assert _module_supported_precisions(nn.LayerNorm(4)) == ("float32",)
    assert _module_supported_precisions(nn.Module()) == ("float32",)


def test_generation_retention_persists_hashes_not_output_tokens() -> None:
    result = generation_retention(
        {"story-1": (1, 2), "story-2": (3,)},
        {"story-1": (1, 2), "story-2": (4,)},
    )

    assert result["exact_match_count"] == 1
    assert result["exact_match_rate"] == pytest.approx(0.5)
    assert result["output_content_persisted"] is False
    assert "token_ids" not in result["samples"][0]
    assert len(result["samples"][0]["baseline_output_sha256"]) == 64


def test_checked_quality_summary_retains_scope_and_no_output_content() -> None:
    summary = json.loads(
        (
            PROJECT_ROOT / "docs" / "experiments" / "stories15m-moe-native-linear-quality.json"
        ).read_text(encoding="utf-8")
    )
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["protocol"]["prompt_or_output_content_persisted"] is False
    assert summary["quantization_scope"]["selected_attention_linear_count"] == 24
    assert summary["quantization_scope"]["fused_expert_slices_quantized"] == 0
    assert summary["quantization_scope"]["fused_expert_slice_count"] == 24
    assert summary["reproducibility_check"]["scientific_fields_exact_across_executions"]
    assert "token_ids" not in serialized
    assert "prompt_text" not in serialized
