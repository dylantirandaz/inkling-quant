from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_mlx_public_moe_quant import (
    _atomic_write_json,
    _expected_quantized_scale_keys,
    _expected_runtime_quantized_expert_names,
    _expected_runtime_quantized_linear_names,
    _validate_worker_evaluator_identity,
    audit_source_snapshot,
    compare_routes,
    percentile,
    resolve_output_path,
    select_stories,
    selection_sha256,
    summarize_samples,
    validate_checkpoint_tensor_facts,
    validate_hardware_label,
    validate_model_config_security,
    validate_tokenizer_config_security,
    validate_worker_timeout,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKED_EVIDENCE = PROJECT_ROOT / "docs/experiments/stories15m-moe-mlx-full-quantization-m3.json"
EVIDENCE_SHA256 = "09df5adfc2ed1d3fde896091a05c8a6098f1a7fce0793bca180d9d63202b7efa"
EVALUATOR_SHA256 = "343ceea8db1e0b9c0076ee2e7bf968d68ef916acc195d6f52e41a1bacb1b71d7"


def _safe_config() -> dict[str, object]:
    return {
        "architectures": ["MixtralForCausalLM"],
        "auto_map": None,
        "model_file": None,
        "model_type": "mixtral",
    }


def _quantized_facts(bits: int) -> dict[str, tuple[str, tuple[int, ...], int]]:
    facts: dict[str, tuple[str, tuple[int, ...], int]] = {}
    for scale_key in _expected_quantized_scale_keys():
        prefix = scale_key.removesuffix(".scales")
        facts[f"{prefix}.weight"] = ("uint32", (1, 1), 4)
        facts[scale_key] = ("float32", (1, 1), 4)
        facts[f"{prefix}.biases"] = ("float32", (1, 1), 4)
    for layer in range(6):
        for projection, output_dims, input_dims in (
            ("gate_proj", 768, 288),
            ("up_proj", 768, 288),
            ("down_proj", 288, 768),
        ):
            prefix = f"model.layers.{layer}.block_sparse_moe.switch_mlp.{projection}"
            packed_shape = (4, output_dims, input_dims * bits // 32)
            auxiliary_shape = (4, output_dims, input_dims // 32)
            facts[f"{prefix}.weight"] = ("uint32", packed_shape, 4)
            facts[f"{prefix}.scales"] = ("float32", auxiliary_shape, 4)
            facts[f"{prefix}.biases"] = ("float32", auxiliary_shape, 4)
    return facts


def _routes(route_by_layer: dict[int, list[list[int]]]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": "story-000001",
            "layer_id": layer,
            "routes": route_by_layer.get(layer, [[0, 1], [0, 1]]),
        }
        for layer in range(6)
    ]


def test_model_config_guard_rejects_every_custom_model_selector() -> None:
    validate_model_config_security(_safe_config(), source="fixture")

    model_file = _safe_config()
    model_file["model_file"] = "modeling.py"
    with pytest.raises(ValueError, match="model_file"):
        validate_model_config_security(model_file, source="fixture")

    auto_map = _safe_config()
    auto_map["auto_map"] = {"AutoModel": "modeling.Custom"}
    with pytest.raises(ValueError, match="auto_map"):
        validate_model_config_security(auto_map, source="fixture")

    nested = _safe_config()
    nested["text_config"] = {"model_file": "nested.py"}
    with pytest.raises(ValueError, match="nested"):
        validate_model_config_security(nested, source="fixture")


def test_tokenizer_guard_requires_exact_builtin_class_and_no_helper_selectors() -> None:
    config = {"tokenizer_class": "LlamaTokenizer"}
    validate_tokenizer_config_security(
        config,
        source="fixture",
        expected_class="LlamaTokenizer",
    )

    custom = {"tokenizer_class": "CustomTokenizer"}
    with pytest.raises(ValueError, match="tokenizer_class"):
        validate_tokenizer_config_security(
            custom,
            source="fixture",
            expected_class="LlamaTokenizer",
        )

    auto_map = {
        "tokenizer_class": "LlamaTokenizer",
        "auto_map": {"AutoTokenizer": "tokenizer.Custom"},
    }
    with pytest.raises(ValueError, match="auto_map"):
        validate_tokenizer_config_security(
            auto_map,
            source="fixture",
            expected_class="LlamaTokenizer",
        )


@pytest.mark.parametrize("bits", [4, 8])
def test_tensor_facts_prove_every_fused_expert_projection_is_packed(bits: int) -> None:
    result = validate_checkpoint_tensor_facts(_quantized_facts(bits), bits=bits)

    assert result["quantized_leaf_count"] == 50
    assert result["quantized_fused_expert_projection_count"] == 18
    assert result["bits"] == bits
    assert result["packed_weight_dtype"] == "uint32"


def test_tensor_facts_fail_when_one_expert_scale_is_absent() -> None:
    facts = _quantized_facts(4)
    facts.pop("model.layers.5.block_sparse_moe.switch_mlp.down_proj.scales")

    with pytest.raises(ValueError, match="scale-key mismatch"):
        validate_checkpoint_tensor_facts(facts, bits=4)


def test_runtime_quantized_module_name_contract_is_exact() -> None:
    linears = _expected_runtime_quantized_linear_names()
    experts = _expected_runtime_quantized_expert_names()

    assert len(linears) == 31
    assert "lm_head" in linears
    assert "model.layers.5.block_sparse_moe.gate" in linears
    assert "model.layers.5.self_attn.o_proj" in linears
    assert len(experts) == 18
    assert "model.layers.0.block_sparse_moe.switch_mlp.gate_proj" in experts
    assert "model.layers.5.block_sparse_moe.switch_mlp.down_proj" in experts


def test_aligned_route_comparison_distinguishes_order_set_and_overlap() -> None:
    baseline = _routes({})
    candidate = _routes(
        {
            0: [[1, 0], [1, 0]],
            1: [[2, 3], [2, 3]],
        }
    )

    result = compare_routes(baseline, candidate)

    assert result["alignment_key_count"] == 12
    assert result["ordered_top2_agreement"] == pytest.approx(8 / 12)
    assert result["unordered_top2_agreement"] == pytest.approx(10 / 12)
    assert result["mean_top2_overlap"] == pytest.approx(10 / 12)
    assert result["per_layer"][0]["selection_js_divergence"] == pytest.approx(0.0)
    assert result["per_layer"][1]["selection_js_divergence"] > 0.0


def test_route_comparison_rejects_misalignment_and_invalid_top2() -> None:
    baseline = _routes({})
    misaligned = _routes({})
    misaligned[0]["routes"] = [[0, 1]]
    with pytest.raises(ValueError, match="not exactly aligned"):
        compare_routes(baseline, misaligned)

    invalid = _routes({})
    invalid[0]["routes"] = [[0, 0], [0, 1]]
    with pytest.raises(ValueError, match="invalid top-2 route"):
        compare_routes(baseline, invalid)


def test_story_selection_and_digest_use_declared_ids_in_order() -> None:
    stories = ("zero", "one", "two", "three")
    sample_ids = ("story-000003", "story-000001")

    selected = select_stories(stories, sample_ids)

    assert selected == ("three", "one")
    expected = [
        {
            "sample_id": sample_id,
            "content_sha256": hashlib.sha256(story.encode()).hexdigest(),
        }
        for sample_id, story in zip(sample_ids, selected, strict=True)
    ]
    payload = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    assert selection_sha256(sample_ids, selected) == hashlib.sha256(payload).hexdigest()


def test_benchmark_statistics_use_linear_percentiles() -> None:
    values = [1.0, 2.0, 10.0, 20.0, 30.0]

    assert percentile(values, 0.1) == pytest.approx(1.4)
    assert percentile(values, 0.9) == pytest.approx(26.0)
    summary = summarize_samples(values)
    assert summary["median"] == 10.0
    assert summary["mean"] == pytest.approx(12.6)
    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 30.0
    with pytest.raises(ValueError, match="finite"):
        summarize_samples([1.0, float("nan")])
    with pytest.raises(ValueError, match="greater than zero"):
        summarize_samples([1.0, 0.0])


def test_source_audit_accepts_safe_local_files_and_rejects_python(tmp_path: Path) -> None:
    snapshot = tmp_path / "model-cache" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    config = json.dumps(_safe_config(), sort_keys=True).encode()
    weight = b"safe-tensor-placeholder"
    tokenizer_config = json.dumps({"tokenizer_class": "LlamaTokenizer"}).encode()
    tokenizer = b'{"version":"1.0"}'
    (snapshot / "config.json").write_bytes(config)
    (snapshot / "model.safetensors").write_bytes(weight)
    (snapshot / "tokenizer_config.json").write_bytes(tokenizer_config)
    (snapshot / "tokenizer.json").write_bytes(tokenizer)
    file_set = frozenset(
        {"config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"}
    )

    result = audit_source_snapshot(
        snapshot,
        expected_revision="revision",
        expected_config_sha256=hashlib.sha256(config).hexdigest(),
        expected_tokenizer_config_sha256=hashlib.sha256(tokenizer_config).hexdigest(),
        expected_tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(),
        expected_weight_sha256=hashlib.sha256(weight).hexdigest(),
        expected_weight_size_bytes=len(weight),
        expected_file_set=file_set,
        expected_safetensors_tensor_count=None,
    )

    assert result["python_file_count"] == 0
    assert result["model_file"] is None

    directory_target = tmp_path / "directory-target"
    directory_target.mkdir()
    directory_link = snapshot / "linked-directory"
    directory_link.symlink_to(directory_target, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlinks"):
        audit_source_snapshot(
            snapshot,
            expected_revision="revision",
            expected_config_sha256=hashlib.sha256(config).hexdigest(),
            expected_tokenizer_config_sha256=hashlib.sha256(tokenizer_config).hexdigest(),
            expected_tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(),
            expected_weight_sha256=hashlib.sha256(weight).hexdigest(),
            expected_weight_size_bytes=len(weight),
            expected_file_set=file_set,
            expected_safetensors_tensor_count=None,
        )
    directory_link.unlink()

    (snapshot / "modeling.py").write_text("raise RuntimeError", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe or unsupported"):
        audit_source_snapshot(
            snapshot,
            expected_revision="revision",
            expected_config_sha256=hashlib.sha256(config).hexdigest(),
            expected_tokenizer_config_sha256=hashlib.sha256(tokenizer_config).hexdigest(),
            expected_tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(),
            expected_weight_sha256=hashlib.sha256(weight).hexdigest(),
            expected_weight_size_bytes=len(weight),
            expected_file_set=file_set,
            expected_safetensors_tensor_count=None,
        )


def test_output_must_be_new_and_below_artifacts(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    output = resolve_output_path(tmp_path, Path("artifacts/research/record.json"))
    assert output == tmp_path / "artifacts" / "research" / "record.json"

    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable output"):
        resolve_output_path(tmp_path, Path("artifacts/research/record.json"))
    with pytest.raises(ValueError, match="below artifacts"):
        resolve_output_path(tmp_path, Path("docs/record.json"))
    with pytest.raises(ValueError, match="cannot contain"):
        resolve_output_path(tmp_path, Path("artifacts/../escape.json"))


def test_atomic_json_publish_never_replaces_an_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    _atomic_write_json(output, {"finite": 1.0})
    original = output.read_bytes()

    with pytest.raises(FileExistsError, match="immutable output"):
        _atomic_write_json(output, {"replacement": 2.0})

    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="Out of range float values"):
        _atomic_write_json(tmp_path / "nan.json", {"bad": float("nan")})


def test_hardware_label_and_worker_timeout_fail_closed() -> None:
    assert validate_hardware_label("Apple M3, 16 GB") == "Apple M3, 16 GB"
    with pytest.raises(ValueError, match="credential"):
        validate_hardware_label("Apple M3 token=hf_abcdefgh1234")
    with pytest.raises(ValueError, match="whitespace"):
        validate_hardware_label(" Apple M3")

    assert validate_worker_timeout(300.0) == 300.0
    for invalid in (float("nan"), float("inf"), 29.9, 3600.1):
        with pytest.raises(ValueError, match="finite and between"):
            validate_worker_timeout(invalid)


def test_worker_evaluator_identity_must_match_parent_at_both_boundaries() -> None:
    expected_sha256 = "a" * 64
    worker = {
        "evaluator": {
            "path": "scripts/evaluate_mlx_public_moe_quant.py",
            "sha256_at_worker_start": expected_sha256,
            "sha256_at_worker_end": expected_sha256,
        }
    }

    _validate_worker_evaluator_identity(
        worker,
        label="float32",
        expected_sha256=expected_sha256,
    )
    worker["evaluator"]["sha256_at_worker_end"] = "b" * 64
    with pytest.raises(ValueError, match="differs from the parent"):
        _validate_worker_evaluator_identity(
            worker,
            label="float32",
            expected_sha256=expected_sha256,
        )


def test_checked_full_mlx_evidence_is_exact_safe_and_evaluator_bound() -> None:
    evidence_bytes = CHECKED_EVIDENCE.read_bytes()
    evidence = json.loads(evidence_bytes)
    evaluator_bytes = (PROJECT_ROOT / "scripts/evaluate_mlx_public_moe_quant.py").read_bytes()

    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert hashlib.sha256(evaluator_bytes).hexdigest() == EVALUATOR_SHA256
    assert evidence["schema_version"] == "public-moe-mlx-full-quantization-evidence-v3"
    assert evidence["evaluator"]["sha256"] == EVALUATOR_SHA256
    assert evidence["evaluator"]["sha256_at_parent_start"] == EVALUATOR_SHA256
    assert evidence["evaluator"]["sha256_at_parent_end"] == EVALUATOR_SHA256
    assert evidence["model"]["revision"] == "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    assert evidence["dataset"]["file_sha256"] == (
        "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
    )
    assert evidence["model"]["source"]["expert_tensor_identity"]["group_count"] == 18

    models = [evidence["float32_control"], *evidence["candidates"]]
    for model in models:
        assert model["evaluator"]["sha256_at_worker_start"] == EVALUATOR_SHA256
        assert model["evaluator"]["sha256_at_worker_end"] == EVALUATOR_SHA256
        assert model["quality"]["sample_count"] == 32
        assert model["quality"]["evaluated_token_count"] == 5_826
        assert model["routing"]["token_layer_event_count"] == 35_148
        assert model["routing"]["raw_routes_persisted"] is False

    by_label = {candidate["label"]: candidate for candidate in evidence["candidates"]}
    assert by_label["mlx_affine_q8_g32"]["generation_vs_float32"]["exact_match_count"] == 4
    assert by_label["mlx_affine_q4_g32"]["generation_vs_float32"]["exact_match_count"] == 0
    for candidate in by_label.values():
        proof = candidate["load"]["runtime_quantization_proof"]
        assert len(proof["quantized_linear_names"]) == 31
        assert proof["quantized_embedding_names"] == ["model.embed_tokens"]
        assert len(proof["quantized_fused_expert_projection_names"]) == 18
        assert len(candidate["routing_vs_float32"]["per_layer"]) == 6

    serialized = evidence_bytes.decode("utf-8")
    assert "/Users/" not in serialized
    assert "/private/" not in serialized
    assert '"_ephemeral_routes"' not in serialized
    assert '"routes"' not in serialized
    assert '"output_token_ids"' not in serialized
