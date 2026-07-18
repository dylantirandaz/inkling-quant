from __future__ import annotations

import copy
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

import scripts.run_mlx_router_preflight as preflight_script
from inkling_quant_lab.post_training import (
    CorpusCollectionContract,
    CorpusContract,
    CorpusSampleContract,
    CorpusSourceContract,
    RouterOverlayLineage,
)
from scripts.run_mlx_router_preflight import (
    DEFAULT_CONFIG,
    PINNED_CORPORA,
    ROUTER_NAMES,
    AuditedCorpus,
    ParsedDocument,
    PreflightAcceptanceFailure,
    _assert_no_raw_corpus_text,
    _build_typed_lineage,
    _failure_output_path,
    _make_artifact_read_only,
    _publish_read_only_tree,
    _restore_artifact_permissions_for_cleanup,
    _write_completion_record,
    _write_failure_record,
    assess_held_out_acceptance,
    load_audited_overlay,
    load_preflight_config,
    main,
    parse_gutenberg_documents,
    probe_metal_subprocess,
    prove_identical_expert_tensors,
    resolve_output_path,
    validate_global_document_disjointness,
    verify_completion_record,
    verify_failure_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_checked_config_pins_model_corpora_objective_and_claim_boundary() -> None:
    config = load_preflight_config(DEFAULT_CONFIG)

    assert config["model"]["revision"] == "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    assert config["model"]["weights_sha256"] == (
        "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
    )
    assert config["model"]["device"] == "metal"
    assert config["model"]["dtype"] == "float32"
    assert config["training"]["steps"] == 10
    assert config["training"]["betas"] == [0.9, 0.999]
    assert config["training"]["epsilon"] == 1e-8
    assert config["training"]["weight_decay"] == 0.0
    assert config["training"]["bias_correction"] is False
    assert config["training"]["expected_router_tensor_count"] == len(ROUTER_NAMES) == 6
    assert config["training"]["expected_trainable_parameter_count"] == 6912
    assert config["acceptance"] == {
        "minimum_validation_cross_entropy_reduction": 0.05,
        "minimum_validation_exact_top2_pair_accuracy": 0.60,
        "minimum_accuracy_gain_over_source_router": 0.20,
        "minimum_per_domain_cross_entropy_reduction": 0.02,
        "minimum_per_domain_exact_top2_pair_accuracy": 0.50,
        "minimum_per_domain_accuracy_gain_over_source_router": 0.15,
    }
    assert [item["target_expert_pair"] for item in config["corpora"]] == [[0, 1], [2, 3]]
    assert [item["sha256"] for item in config["corpora"]] == [
        "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
        "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
    ]
    assert [item["size_bytes"] for item in config["corpora"]] == [174311, 607606]
    assert all("gutenberg.org" in item["provenance_url"] for item in config["corpora"])
    assert config["claim_boundary"] == {
        "learned_domain_supervised_routing": True,
        "causal_lm_specialization_claimed": False,
        "output_quality_retention_claimed": False,
        "raw_text_or_tokens_persisted": False,
    }


def test_config_fails_closed_on_checksum_steps_or_pair_change(tmp_path: Path) -> None:
    base = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    mutations = []
    checksum = copy.deepcopy(base)
    checksum["corpora"][0]["sha256"] = "0" * 64
    mutations.append(checksum)
    steps = copy.deepcopy(base)
    steps["training"]["steps"] = 9
    mutations.append(steps)
    pair = copy.deepcopy(base)
    pair["corpora"][1]["target_expert_pair"] = [1, 2]
    mutations.append(pair)
    device = copy.deepcopy(base)
    device["model"]["device"] = "cpu"
    mutations.append(device)
    betas = copy.deepcopy(base)
    betas["training"]["betas"] = [0.8, 0.999]
    mutations.append(betas)

    for index, payload in enumerate(mutations):
        path = tmp_path / f"bad-{index}.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="must be exactly"):
            load_preflight_config(path)


def test_document_split_is_deterministic_and_content_disjoint() -> None:
    corpus = {"domain_id": "alice", **PINNED_CORPORA["alice"]}
    start = corpus["start_marker"]
    end = corpus["end_marker"]
    paragraphs = "\n\n".join(
        f"Document {index} contains enough deterministic fixture prose to be retained safely."
        for index in range(80)
    )
    text = (
        "The Project Gutenberg eBook fixture\n"
        f"Title: {corpus['title']}\nAuthor: {corpus['author']}\n\n"
        f"{start}\n\n{paragraphs}\n\n{end}\n"
    )
    data = {"split_modulus": 5, "validation_bucket": 0}

    first = parse_gutenberg_documents(text, corpus, seed=20260716, data=data)
    second = parse_gutenberg_documents(text, corpus, seed=20260716, data=data)

    assert first == second
    assert {item.split for item in first} == {"train", "validation"}
    train = {item.content_sha256 for item in first if item.split == "train"}
    validation = {item.content_sha256 for item in first if item.split == "validation"}
    assert train.isdisjoint(validation)


def test_duplicate_document_content_is_forced_into_one_split() -> None:
    corpus = {"domain_id": "alice", **PINNED_CORPORA["alice"]}
    start = corpus["start_marker"]
    end = corpus["end_marker"]
    repeated = "Repeated document content is long enough to remain in the parsed selection."
    paragraphs = [repeated, repeated]
    paragraphs.extend(
        f"Unique document {index} contains enough deterministic prose for stable splitting."
        for index in range(80)
    )
    text = (
        "The Project Gutenberg eBook fixture\n"
        f"Title: {corpus['title']}\nAuthor: {corpus['author']}\n\n"
        f"{start}\n\n" + "\n\n".join(paragraphs) + f"\n\n{end}\n"
    )

    documents = parse_gutenberg_documents(
        text,
        corpus,
        seed=20260716,
        data={"split_modulus": 5, "validation_bucket": 0},
    )
    repeated_records = [item for item in documents if item.text == repeated]

    assert len(repeated_records) == 2
    assert len({item.split for item in repeated_records}) == 1

    sherlock = {"domain_id": "sherlock", **PINNED_CORPORA["sherlock"]}
    sherlock_text = (
        "The Project Gutenberg eBook fixture\n"
        f"Title: {sherlock['title']}\nAuthor: {sherlock['author']}\n\n"
        f"{sherlock['start_marker']}\n\n"
        + "\n\n".join(paragraphs)
        + f"\n\n{sherlock['end_marker']}\n"
    )
    sherlock_documents = parse_gutenberg_documents(
        sherlock_text,
        sherlock,
        seed=20260716,
        data={"split_modulus": 5, "validation_bucket": 0},
    )
    sherlock_repeated = [item for item in sherlock_documents if item.text == repeated]
    assert {item.split for item in sherlock_repeated} == {repeated_records[0].split}


def test_global_document_disjointness_rejects_cross_source_train_validation_overlap() -> None:
    shared_digest = "a" * 64

    def audited(domain: str, split: str, digest: str) -> AuditedCorpus:
        source = CorpusSourceContract(
            dataset_id=f"fixture/{domain}",
            revision="v1",
            declared_license="fixture-license",
            size_bytes=1,
            sha256="b" * 64,
            parser_version="fixture-parser-v1",
        )
        document = ParsedDocument(
            sample_id=f"{domain}-document-00000",
            domain_id=domain,
            split=split,
            text="ephemeral fixture text",
            content_sha256=digest,
            split_key_sha256="c" * 64,
        )
        return AuditedCorpus(
            config={"domain_id": domain},
            source=source,
            documents=(document,),
        )

    alice = audited("alice", "train", shared_digest)
    sherlock_overlap = audited("sherlock", "validation", shared_digest)
    with pytest.raises(ValueError, match="cross-source training/evaluation"):
        validate_global_document_disjointness((alice, sherlock_overlap))

    sherlock_disjoint = audited("sherlock", "validation", "d" * 64)
    validate_global_document_disjointness((alice, sherlock_disjoint))


def _expert_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for layer in range(6):
        for expert in range(4):
            for projection, value in (("w1", 1.0), ("w2", 2.0), ("w3", 3.0)):
                name = f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"
                tensors[name] = np.full((2, 3), value + layer, dtype=np.float32)
    return tensors


def test_identical_expert_proof_covers_all_layers_and_rejects_one_changed_byte() -> None:
    tensors = _expert_tensors()
    proof = prove_identical_expert_tensors(tensors)

    assert proof["layer_count"] == 6
    assert proof["projection_group_count"] == 18
    assert proof["causal_lm_router_specialization_identifiable"] is False
    assert all(item["four_expert_bundles_byte_identical"] for item in proof["layers"])

    tensors["model.layers.5.block_sparse_moe.experts.3.w3.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="not byte-identical"):
        prove_identical_expert_tensors(tensors)


def test_import_and_default_cli_refusal_do_not_import_mlx() -> None:
    code = (
        "import sys; import scripts.run_mlx_router_preflight; "
        "assert not any(n == 'mlx' or n.startswith('mlx.') or n.startswith('mlx_lm') "
        "for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(SystemExit, match="explicit --execute-metal"):
        main([])


def test_main_returns_distinct_failure_status_and_prints_published_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "failed-attempt"

    def fail(*_args: object, **_kwargs: object) -> Path:
        raise PreflightAcceptanceFailure(artifact, ("overall.validation_exact_top2_pair_accuracy",))

    monkeypatch.setattr(preflight_script, "run_preflight", fail)

    assert main(["--execute-metal"]) == 2
    output = capsys.readouterr()
    assert str(artifact) in output.out
    assert "missed predeclared" in output.err


def _acceptance_thresholds() -> dict[str, float]:
    return {
        "minimum_validation_cross_entropy_reduction": 0.05,
        "minimum_validation_exact_top2_pair_accuracy": 0.60,
        "minimum_accuracy_gain_over_source_router": 0.20,
        "minimum_per_domain_cross_entropy_reduction": 0.02,
        "minimum_per_domain_exact_top2_pair_accuracy": 0.50,
        "minimum_per_domain_accuracy_gain_over_source_router": 0.15,
    }


def test_held_out_acceptance_retains_every_check_instead_of_raising_early() -> None:
    before = {
        "mean_cross_entropy": 1.5,
        "exact_top2_pair_accuracy": 0.10,
        "per_domain": {
            "alice": {"mean_cross_entropy": 1.4, "exact_top2_pair_accuracy": 0.10},
            "sherlock": {"mean_cross_entropy": 1.6, "exact_top2_pair_accuracy": 0.10},
        },
    }
    after = {
        "mean_cross_entropy": 1.2,
        "exact_top2_pair_accuracy": 0.55,
        "per_domain": {
            "alice": {"mean_cross_entropy": 1.0, "exact_top2_pair_accuracy": 0.70},
            "sherlock": {"mean_cross_entropy": 1.4, "exact_top2_pair_accuracy": 0.40},
        },
    }

    result = assess_held_out_acceptance(before, after, _acceptance_thresholds())

    assert result["passed"] is False
    assert len(result["checks"]) == 9
    assert result["failed_checks"] == [
        "overall.validation_exact_top2_pair_accuracy",
        "domain.sherlock.validation_exact_top2_pair_accuracy",
    ]
    assert result["per_domain"]["alice"]["validation_accuracy_gain_over_source_router"] == (
        pytest.approx(0.60)
    )


def test_held_out_acceptance_passes_only_when_all_overall_and_domain_gates_pass() -> None:
    before = {
        "mean_cross_entropy": 1.5,
        "exact_top2_pair_accuracy": 0.10,
        "per_domain": {
            "alice": {"mean_cross_entropy": 1.4, "exact_top2_pair_accuracy": 0.10},
            "sherlock": {"mean_cross_entropy": 1.6, "exact_top2_pair_accuracy": 0.10},
        },
    }
    after = {
        "mean_cross_entropy": 1.0,
        "exact_top2_pair_accuracy": 0.75,
        "per_domain": {
            "alice": {"mean_cross_entropy": 0.9, "exact_top2_pair_accuracy": 0.80},
            "sherlock": {"mean_cross_entropy": 1.1, "exact_top2_pair_accuracy": 0.70},
        },
    }

    result = assess_held_out_acceptance(before, after, _acceptance_thresholds())

    assert result["passed"] is True
    assert result["failed_checks"] == []
    assert all(check["passed"] for check in result["checks"])


def test_output_is_constrained_to_artifact_root() -> None:
    output = resolve_output_path("artifacts/research-slices/test-router-preflight")
    assert output.is_relative_to((PROJECT_ROOT / "artifacts").resolve())
    with pytest.raises(ValueError, match="artifacts"):
        resolve_output_path("outside-artifacts")


def test_source_bundle_rejects_a_symlink_before_resolving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    link.symlink_to(source)
    monkeypatch.setattr(preflight_script, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="must not be a symlink"):
        preflight_script._source_bundle_record(("linked.py",))


def test_metal_probe_converts_native_child_failure_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], -6),
    )

    with pytest.raises(RuntimeError, match="exit code -6"):
        probe_metal_subprocess()


def test_saved_overlay_audit_requires_exact_six_float32_router_tensors(
    tmp_path: Path,
) -> None:
    from scripts.run_mlx_router_preflight import _OVERLAY_METADATA

    path = tmp_path / "router_overlay.safetensors"
    arrays = {
        name: np.full((4, 288), layer, dtype=np.float32) for layer, name in enumerate(ROUTER_NAMES)
    }
    save_file(arrays, path, metadata=_OVERLAY_METADATA)

    loaded = load_audited_overlay(path)
    assert tuple(loaded) == ROUTER_NAMES
    assert all(array.shape == (4, 288) for array in loaded.values())

    save_file({**arrays, "unexpected.weight": np.zeros(1, dtype=np.float32)}, path)
    with pytest.raises(ValueError, match=r"metadata|exactly"):
        load_audited_overlay(path)


def _ephemeral_audited_corpus(text: str) -> AuditedCorpus:
    source = CorpusSourceContract(
        dataset_id="fixture/alice",
        revision="fixture-v1",
        declared_license="fixture-license",
        size_bytes=1,
        sha256="a" * 64,
        parser_version="fixture-parser-v1",
    )
    return AuditedCorpus(
        config={"domain_id": "alice"},
        source=source,
        documents=(
            ParsedDocument(
                sample_id="alice-document-00000",
                domain_id="alice",
                split="train",
                text=text,
                content_sha256="b" * 64,
                split_key_sha256="c" * 64,
            ),
        ),
    )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "resolved_config.yaml",
        "corpus_contract.json",
        "environment.json",
        "lineage.json",
        "metrics.json",
        "evidence.json",
        "completion.json",
        "failure.json",
    ],
)
def test_raw_corpus_scan_covers_every_generated_text_artifact(
    tmp_path: Path, artifact_name: str
) -> None:
    raw_text = (
        "This exact ephemeral corpus paragraph is deliberately long enough for the scanner "
        "to identify any persisted contiguous snippet without storing token identifiers."
    )
    corpus = _ephemeral_audited_corpus(raw_text)
    (tmp_path / "safe.json").write_text('{"safe": true}\n', encoding="utf-8")
    (tmp_path / "router_overlay.safetensors").write_bytes(raw_text.encode("utf-8"))
    _assert_no_raw_corpus_text(tmp_path, (corpus,))

    (tmp_path / artifact_name).write_text(raw_text[12:90], encoding="utf-8")
    with pytest.raises(ValueError, match=artifact_name):
        _assert_no_raw_corpus_text(tmp_path, (corpus,))


def test_completion_record_seals_evidence_and_rejects_changed_or_extra_files(
    tmp_path: Path,
) -> None:
    evidence_bytes = '{"status": "complete"}\n'
    (tmp_path / "evidence.json").write_text(evidence_bytes, encoding="utf-8")
    (tmp_path / "metrics.json").write_text('{"metric": 1}\n', encoding="utf-8")

    completion = _write_completion_record(
        tmp_path,
        run_id="fixture-run",
        bindings={"fixture_sha256": "d" * 64},
    )
    verified = verify_completion_record(tmp_path)

    assert verified == completion
    assert "evidence.json" in completion["files"]
    assert completion["file_count"] == 2

    (tmp_path / "evidence.json").write_text('{"status": "changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ledger differs"):
        verify_completion_record(tmp_path)

    (tmp_path / "evidence.json").write_text(evidence_bytes, encoding="utf-8")
    (tmp_path / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger differs"):
        verify_completion_record(tmp_path)


def test_failure_record_seals_negative_evidence_and_keeps_success_path_free(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "successful-run"
    failed_path = _failure_output_path(
        requested,
        resolved_config_sha256="a" * 64,
        created_at=datetime(2026, 7, 16, 18, 30, tzinfo=UTC),
    )
    assert failed_path == tmp_path / "successful-run.failed-20260716T183000000000Z-aaaaaaaa"
    assert failed_path != requested

    (tmp_path / "evidence.json").write_text('{"status": "failed"}\n', encoding="utf-8")
    (tmp_path / "metrics.json").write_text('{"acceptance": {"passed": false}}\n', encoding="utf-8")
    failure = {
        "failure_code": "held_out_acceptance_failed",
        "failed_checks": ["overall.validation_exact_top2_pair_accuracy"],
    }
    record = _write_failure_record(
        tmp_path,
        run_id=failed_path.name,
        bindings={"resolved_config_sha256": "b" * 64},
        failure=failure,
    )

    assert verify_failure_record(tmp_path) == record
    assert record["status"] == "failed"
    assert record["failed_checks"] == failure["failed_checks"]

    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger differs"):
        verify_failure_record(tmp_path)


def test_artifact_tree_is_read_only_and_can_be_restored_for_failed_publication_cleanup(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "metrics"
    nested.mkdir()
    artifact = nested / "metrics.json"
    artifact.write_text("{}\n", encoding="utf-8")

    _make_artifact_read_only(tmp_path)

    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o555
    assert stat.S_IMODE(nested.stat().st_mode) == 0o555
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o444

    _restore_artifact_permissions_for_cleanup(tmp_path)
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(nested.stat().st_mode) == 0o700
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_publication_renames_before_sealing_root_and_leaves_final_tree_read_only(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    artifact = temporary / "evidence.json"
    artifact.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "published"

    _publish_read_only_tree(temporary, output)

    assert not temporary.exists()
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert stat.S_IMODE((output / "evidence.json").stat().st_mode) == 0o444


def test_typed_router_lineage_is_built_from_saved_and_reloaded_full_states() -> None:
    source = CorpusSourceContract(
        dataset_id="fixture/alice",
        revision="fixture-v1",
        declared_license="fixture-license",
        size_bytes=1,
        sha256="e" * 64,
        parser_version="fixture-parser-v1",
    )
    train = CorpusSampleContract.from_text(
        sample_id="alice-train",
        text="training fixture content",
        split="train",
        labels=("alice", "domain-supervised-router"),
        token_count=64,
    )
    validation = CorpusSampleContract.from_text(
        sample_id="alice-validation",
        text="validation fixture content",
        split="validation",
        labels=("alice", "domain-supervised-router"),
        token_count=64,
    )
    collection = CorpusCollectionContract(
        corpora=(CorpusContract(source=source, samples=(train, validation), token_budget=128),)
    )
    audited = AuditedCorpus(config={"domain_id": "alice"}, source=source, documents=())
    parent_arrays = {name: np.zeros((4, 288), dtype=np.float32) for name in ROUTER_NAMES}
    parent_arrays["model.embed_tokens.weight"] = np.arange(8, dtype=np.float32)
    overlay_arrays = {
        name: np.full((4, 288), layer + 1, dtype=np.float32)
        for layer, name in enumerate(ROUTER_NAMES)
    }
    reloaded_arrays = {**parent_arrays, **overlay_arrays}
    resolved_config_sha256 = "f" * 64

    lineage = _build_typed_lineage(
        config={
            "name": "fixture-router-run",
            "seed": 7,
            "corpora": [{"domain_id": "alice", "target_expert_pair": [0, 1]}],
            "data": {"sequence_length": 64},
            "training": {
                "steps": 10,
                "batch_size": 1,
                "optimizer": "adamw",
                "learning_rate": 0.01,
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "bias_correction": False,
                "objective_name": "domain_pair_soft_target_cross_entropy",
                "objective_version": "top2-domain-pair-ce-v1",
            },
        },
        collection=collection,
        audited_corpora=(audited,),
        parent_arrays=parent_arrays,
        overlay_arrays=overlay_arrays,
        reloaded_arrays=reloaded_arrays,
        overlay_bundle_sha256="1" * 64,
        resolved_config_sha256=resolved_config_sha256,
        source_context={
            "repository_revision": None,
            "repository_dirty": None,
            "entrypoint": "scripts/run_mlx_router_preflight.py",
            "entrypoint_sha256": "2" * 64,
            "source_bundle_sha256": "3" * 64,
            "dependency_lock_sha256": "4" * 64,
        },
        environment_sha256="5" * 64,
    )

    assert isinstance(lineage, RouterOverlayLineage)
    assert lineage.corpus_contract_sha256 == preflight_script._canonical_sha256(
        collection.model_dump(mode="json")
    )
    assert lineage.training.training_config_sha256 == resolved_config_sha256
    assert len(lineage.router_tensors) == 6
    assert lineage.unchanged_nonrouters.parent_nonrouter_sha256 == (
        lineage.unchanged_nonrouters.candidate_nonrouter_sha256
    )
    assert lineage.reload.candidate_state_sha256 == lineage.reload.reloaded_state_sha256
