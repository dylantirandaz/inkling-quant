from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_matched_control import (
    MATCHED_ATTEMPT_REGISTRY_NAME,
    MATCHED_EVIDENCE_VOLUME_NAME,
    MATCHED_FUNCTION_NAME,
    MatchedAttemptAcknowledgement,
    MatchedAttemptClaim,
    MatchedControlPlaneProvenance,
    MatchedDeployConfirmationChallenge,
    MatchedDeploymentIdentity,
    MatchedExecutionResources,
    MatchedLaunchConfirmationChallenge,
    MatchedLaunchIntent,
    MatchedPostSpawnAcceptance,
    MatchedPublicationSnapshot,
    MatchedReviewedInputs,
    build_matched_control_plane_provenance,
    build_matched_launch_intent,
    build_matched_terminal_receipt_reference,
    claim_matched_attempt,
    matched_attempt_acknowledgement_path,
    matched_attempt_claim_path,
    matched_attempt_registry_key,
    matched_launch_intent_path,
    matched_post_spawn_acceptance_path,
    matched_publication_state_path,
    matched_terminal_receipt_content_sha256,
    matched_terminal_receipt_path,
    validate_matched_attempt_acknowledgement,
    validate_matched_attempt_claim,
    validate_matched_control_plane_provenance,
    validate_matched_launch_intent,
    validate_matched_post_spawn_acceptance,
    validate_matched_publication_state,
    validate_matched_publication_transition,
    validate_matched_terminal_receipt_reference,
)

RUN_ID = "inkling-matched-test"
COMMIT = "a" * 40
TREE = "b" * 40
CONFIG_HASH = "c" * 64
PLAN_HASH = "d" * 64
BF16_REFERENCE_HASH = "e" * 64
Q3_REFERENCE_HASH = "f" * 64
SOURCE_REFERENCE_HASH = "1" * 64
REGISTRY_ID = "di-MatchedAttempt123"
REGISTRY_CREATED_AT = "2026-07-28T12:00:00.000000Z"
VOLUME_ID = "vo-MatchedEvidence123"
CONTROL_FILES = {
    "configs/experiments/matched.yaml": b"schema_version: test\n",
    "patches/inkling.patch": b"diff --git a/a b/a\n",
    "scripts/run_matched.py": b"def run() -> None: ...\n",
    "src/inkling_quant_lab/gguf/matched.py": b'"""matched"""\n',
}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _terminal_payload(outcome: Literal["success", "failure"]) -> bytes:
    schema_version, status = {
        "success": ("inkling-matched-rollup-v1", "passed"),
        "failure": ("inkling-matched-failure-v1", "failed"),
    }[outcome]
    return json.dumps(
        {
            "schema_version": schema_version,
            "status": status,
            "stage": "matched_smoke",
            "run_id": RUN_ID,
            "prompt_text_recorded": False,
            "output_text_recorded": False,
            "receipt_sha256": "6" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _provenance(**overrides: Any) -> MatchedControlPlaneProvenance:
    values: dict[str, Any] = {
        "reviewed_commit_sha": COMMIT,
        "reviewed_tree_sha": TREE,
        "files": CONTROL_FILES,
        "required_paths": tuple(CONTROL_FILES),
    }
    values.update(overrides)
    return build_matched_control_plane_provenance(**values)


def _resources(**overrides: Any) -> MatchedExecutionResources:
    values: dict[str, Any] = {
        "function_timeout_seconds": 14_400,
    }
    values.update(overrides)
    return MatchedExecutionResources(**values)


def _reviewed_inputs(**overrides: Any) -> MatchedReviewedInputs:
    provenance = _provenance()
    values: dict[str, Any] = {
        "reviewed_commit_sha": COMMIT,
        "reviewed_tree_sha": TREE,
        "matched_config_sha256": CONFIG_HASH,
        "matched_plan_sha256": PLAN_HASH,
        "bf16_subject_reference_sha256": BF16_REFERENCE_HASH,
        "q3_verified_export_reference_sha256": Q3_REFERENCE_HASH,
        "source_adoption_reference_sha256": SOURCE_REFERENCE_HASH,
        "control_plane_sha256": provenance.control_plane_sha256,
        "subject_order": ("bf16", "q3"),
        "resources": _resources(),
    }
    values.update(overrides)
    return MatchedReviewedInputs(**values)


def _deployment(**overrides: Any) -> MatchedDeploymentIdentity:
    inputs = _reviewed_inputs()
    values: dict[str, Any] = {
        "control_plane_sha256": inputs.control_plane_sha256,
        "app_name": f"inkling-matched-smoke-{inputs.control_plane_sha256[:12]}",
        "deployment_version": 4,
        "deployment_tag": f"iql-matched-{inputs.control_plane_sha256[:40]}",
        "function_id": "fu-Matched123",
        "attempt_registry_id": REGISTRY_ID,
        "attempt_registry_created_at_utc": REGISTRY_CREATED_AT,
        "evidence_volume_id": VOLUME_ID,
    }
    values.update(overrides)
    return MatchedDeploymentIdentity(**values)


def _launch_challenge(**overrides: Any) -> MatchedLaunchConfirmationChallenge:
    values: dict[str, Any] = {
        "created_at_utc": "2026-07-28T12:30:00.000000Z",
        "authorization_nonce": "2" * 64,
        "billing_cycle_end_utc": "2026-08-01T00:00:00Z",
        "run_id": RUN_ID,
        "reviewed_inputs": _reviewed_inputs(),
        "deployment": _deployment(),
    }
    values.update(overrides)
    return MatchedLaunchConfirmationChallenge(**values)


def _launch_intent() -> MatchedLaunchIntent:
    challenge = _launch_challenge()
    return build_matched_launch_intent(
        challenge,
        confirmation=challenge.confirmation_text(),
        authorized_at_utc="2026-07-28T12:31:00.000000Z",
    )


def _acceptance(**overrides: Any) -> MatchedPostSpawnAcceptance:
    intent = _launch_intent()
    values: dict[str, Any] = {
        "accepted_at_utc": "2026-07-28T12:32:00.000000Z",
        "run_id": RUN_ID,
        "launch_intent_sha256": intent.intent_sha256(),
        "call_id": "fc-Matched123",
        "deployment": intent.deployment,
        "matched_config_sha256": CONFIG_HASH,
        "control_plane_sha256": intent.reviewed_inputs.control_plane_sha256,
    }
    values.update(overrides)
    return MatchedPostSpawnAcceptance(**values)


def _claim(**overrides: Any) -> MatchedAttemptClaim:
    intent = _launch_intent()
    acceptance = _acceptance()
    values: dict[str, Any] = {
        "registry_id": REGISTRY_ID,
        "registry_created_at_utc": REGISTRY_CREATED_AT,
        "registry_key": matched_attempt_registry_key(RUN_ID),
        "run_id": RUN_ID,
        "call_id": acceptance.call_id,
        "input_id": "in-Matched123:0-0",
        "task_id": "ta-Matched123",
        "launch_intent_sha256": intent.intent_sha256(),
        "post_spawn_acceptance_path": matched_post_spawn_acceptance_path(
            RUN_ID,
            intent.intent_sha256(),
        ),
        "post_spawn_acceptance_sha256": acceptance.acceptance_sha256(),
        "matched_config_sha256": CONFIG_HASH,
        "control_plane_sha256": intent.reviewed_inputs.control_plane_sha256,
    }
    values.update(overrides)
    return MatchedAttemptClaim(**values)


def _acknowledgement(**overrides: Any) -> MatchedAttemptAcknowledgement:
    claim = _claim()
    values: dict[str, Any] = {
        "acknowledged_at_utc": "2026-07-28T12:33:00.000000Z",
        "run_id": RUN_ID,
        "registry_key": claim.registry_key,
        "attempt_claim_path": matched_attempt_claim_path(
            RUN_ID,
            claim.claim_sha256(),
        ),
        "attempt_claim_sha256": claim.claim_sha256(),
        "call_id": claim.call_id,
        "input_id": claim.input_id,
        "task_id": claim.task_id,
        "launch_intent_sha256": claim.launch_intent_sha256,
        "matched_config_sha256": claim.matched_config_sha256,
        "control_plane_sha256": claim.control_plane_sha256,
    }
    values.update(overrides)
    return MatchedAttemptAcknowledgement(**values)


def test_control_plane_provenance_is_exact_deterministic_and_commit_bound() -> None:
    first = _provenance()
    second = _provenance(
        files=dict(reversed(tuple(CONTROL_FILES.items()))),
        required_paths=tuple(reversed(tuple(CONTROL_FILES))),
    )

    assert first == second
    assert first.file_count == len(CONTROL_FILES)
    assert tuple(item.path for item in first.files) == tuple(sorted(CONTROL_FILES))
    assert first.control_plane_sha256 == second.control_plane_sha256
    assert first.canonical_bytes().endswith(b"\n")
    assert (
        validate_matched_control_plane_provenance(
            first,
            reviewed_commit_sha=COMMIT,
            reviewed_tree_sha=TREE,
            files=CONTROL_FILES,
            required_paths=tuple(CONTROL_FILES),
        )
        == first
    )

    changed = dict(CONTROL_FILES)
    changed["scripts/run_matched.py"] += b"# changed\n"
    assert _provenance(files=changed).control_plane_sha256 != first.control_plane_sha256
    assert _provenance(reviewed_commit_sha="9" * 40).control_plane_sha256 != (
        first.control_plane_sha256
    )


@pytest.mark.parametrize(
    ("files", "required_paths", "message"),
    (
        ({"../escape.py": b""}, ("../escape.py",), "path"),
        ({"a.py": b""}, ("a.py", "missing.py"), "exact"),
        ({"a.py": b""}, ("a.py", "a.py"), "unique"),
    ),
)
def test_control_plane_provenance_rejects_unsafe_or_incomplete_file_sets(
    files: dict[str, bytes],
    required_paths: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_matched_control_plane_provenance(
            reviewed_commit_sha=COMMIT,
            reviewed_tree_sha=TREE,
            files=files,
            required_paths=required_paths,
        )


def test_resources_and_reviewed_inputs_are_the_exact_matched_cell() -> None:
    resources = _resources()
    assert resources.model_dump(mode="json") == {
        "provider": "modal",
        "gpu_type": "B300",
        "gpu_count": 8,
        "compute_capability": "10.3",
        "cpu_cores": 16,
        "memory_mib": 65_536,
        "ephemeral_disk_mib": 524_288,
        "startup_timeout_seconds": 1_800,
        "function_timeout_seconds": 14_400,
        "max_containers": 1,
        "max_recovery_attempts": 0,
    }
    assert _reviewed_inputs().subject_order == ("bf16", "q3")

    with pytest.raises(ValidationError):
        MatchedExecutionResources(gpu_count=2, function_timeout_seconds=14_400)
    with pytest.raises(ValidationError):
        MatchedReviewedInputs(
            **{
                **_reviewed_inputs().model_dump(mode="json"),
                "subject_order": ("q3", "bf16"),
            }
        )


def test_deployment_identity_is_derived_from_control_plane() -> None:
    deployment = _deployment()

    assert deployment.function_name == MATCHED_FUNCTION_NAME
    assert deployment.attempt_registry_name == MATCHED_ATTEMPT_REGISTRY_NAME
    assert deployment.evidence_volume_name == MATCHED_EVIDENCE_VOLUME_NAME

    with pytest.raises(ValidationError, match="app name"):
        _deployment(app_name="inkling-matched-smoke-" + "0" * 12)
    with pytest.raises(ValidationError, match="deployment tag"):
        _deployment(deployment_tag="iql-matched-" + "0" * 40)


def test_prepare_deploy_requires_the_exact_content_addressed_confirmation() -> None:
    challenge = MatchedDeployConfirmationChallenge(
        created_at_utc="2026-07-28T12:00:00.000000Z",
        confirmation_nonce="3" * 64,
        reviewed_inputs=_reviewed_inputs(),
        app_name=f"inkling-matched-smoke-{_reviewed_inputs().control_plane_sha256[:12]}",
    )

    assert challenge.confirmation_text() == (
        "CONFIRM MATCHED DEPLOY\n" + challenge.challenge_sha256()
    )
    assert challenge.confirm(challenge.confirmation_text()) == challenge
    with pytest.raises(ValueError, match="confirmation"):
        challenge.confirm("CONFIRM MATCHED DEPLOY\n" + "0" * 64)


def test_prepare_launch_builds_one_fully_bound_launch_intent() -> None:
    challenge = _launch_challenge()
    intent = build_matched_launch_intent(
        challenge,
        confirmation=challenge.confirmation_text(),
        authorized_at_utc="2026-07-28T12:31:00.000000Z",
    )

    assert challenge.confirmation_text() == (
        "CONFIRM MATCHED LAUNCH\n" + challenge.challenge_sha256()
    )
    assert intent.reviewed_inputs == challenge.reviewed_inputs
    assert intent.deployment == challenge.deployment
    assert intent.authorization_nonce == challenge.authorization_nonce
    assert intent.billing_cycle_end_utc == challenge.billing_cycle_end_utc
    assert intent.subject_order == ("bf16", "q3")
    assert intent.resources.gpu_count == 8
    assert intent.canonical_bytes().endswith(b"\n")
    assert matched_launch_intent_path(RUN_ID, intent.intent_sha256()).endswith(
        f"/{intent.intent_sha256()}.json"
    )

    with pytest.raises(ValueError, match="confirmation"):
        build_matched_launch_intent(
            challenge,
            confirmation="CONFIRM MATCHED LAUNCH\n" + "0" * 64,
            authorized_at_utc="2026-07-28T12:31:00.000000Z",
        )
    with pytest.raises(ValueError, match="authorization time"):
        build_matched_launch_intent(
            challenge,
            confirmation=challenge.confirmation_text(),
            authorized_at_utc="2026-08-01T00:00:00.000000Z",
        )
    with pytest.raises(ValidationError, match="billing cycle"):
        _launch_challenge(created_at_utc="2026-08-01T00:00:00.000000Z")


def test_launch_intent_validation_rejects_noncanonical_duplicate_or_drifted_bytes() -> None:
    intent = _launch_intent()
    payload = intent.canonical_bytes()

    assert (
        validate_matched_launch_intent(
            payload,
            expected=intent,
            launch_intent_sha256=intent.intent_sha256(),
            evidence_path=matched_launch_intent_path(RUN_ID, intent.intent_sha256()),
        )
        == intent
    )

    with pytest.raises(ValueError, match="canonical"):
        validate_matched_launch_intent(
            json.dumps(intent.model_dump(mode="json"), indent=2).encode("utf-8"),
            expected=intent,
            launch_intent_sha256=intent.intent_sha256(),
            evidence_path=matched_launch_intent_path(RUN_ID, intent.intent_sha256()),
        )
    duplicate = payload[:-2] + b',"run_id":"duplicate"}\n'
    with pytest.raises(ValueError, match="duplicate"):
        validate_matched_launch_intent(
            duplicate,
            expected=intent,
            launch_intent_sha256=intent.intent_sha256(),
            evidence_path=matched_launch_intent_path(RUN_ID, intent.intent_sha256()),
        )
    with pytest.raises(ValueError, match="path"):
        validate_matched_launch_intent(
            payload,
            expected=intent,
            launch_intent_sha256=intent.intent_sha256(),
            evidence_path="runs/wrong.json",
        )


def test_post_spawn_acceptance_and_attempt_claim_are_content_addressed() -> None:
    intent = _launch_intent()
    acceptance = _acceptance()
    claim = _claim()

    assert matched_post_spawn_acceptance_path(
        RUN_ID,
        intent.intent_sha256(),
    ).endswith(f"/{intent.intent_sha256()}.json")
    assert acceptance.launch_intent_sha256 == intent.intent_sha256()
    assert claim.post_spawn_acceptance_sha256 == acceptance.acceptance_sha256()
    assert claim.registry_key == f"{RUN_ID}:matched_smoke"
    assert matched_attempt_claim_path(RUN_ID, claim.claim_sha256()).endswith(
        f"/{claim.claim_sha256()}.json"
    )


def test_durable_control_records_require_canonical_exact_bytes_and_paths() -> None:
    acceptance = _acceptance()
    acceptance_path = matched_post_spawn_acceptance_path(
        RUN_ID,
        acceptance.launch_intent_sha256,
    )
    assert (
        validate_matched_post_spawn_acceptance(
            acceptance.canonical_bytes(),
            expected=acceptance,
            acceptance_sha256=acceptance.acceptance_sha256(),
            evidence_path=acceptance_path,
        )
        == acceptance
    )

    claim = _claim()
    claim_path = matched_attempt_claim_path(RUN_ID, claim.claim_sha256())
    assert (
        validate_matched_attempt_claim(
            claim.canonical_bytes(),
            expected=claim,
            claim_sha256=claim.claim_sha256(),
            evidence_path=claim_path,
        )
        == claim
    )

    acknowledgement = _acknowledgement()
    acknowledgement_path = matched_attempt_acknowledgement_path(
        RUN_ID,
        acknowledgement.acknowledgement_sha256(),
    )
    assert (
        validate_matched_attempt_acknowledgement(
            acknowledgement.canonical_bytes(),
            expected=acknowledgement,
            acknowledgement_sha256=acknowledgement.acknowledgement_sha256(),
            evidence_path=acknowledgement_path,
        )
        == acknowledgement
    )

    drifted_acceptance = _acceptance(
        accepted_at_utc="2026-07-28T12:32:01.000000Z",
    )
    with pytest.raises(ValueError, match="exact expected"):
        validate_matched_post_spawn_acceptance(
            drifted_acceptance.canonical_bytes(),
            expected=acceptance,
            acceptance_sha256=drifted_acceptance.acceptance_sha256(),
            evidence_path=acceptance_path,
        )
    with pytest.raises(ValueError, match="canonical"):
        validate_matched_attempt_claim(
            json.dumps(claim.model_dump(mode="json"), indent=2).encode("utf-8"),
            expected=claim,
            claim_sha256=claim.claim_sha256(),
            evidence_path=claim_path,
        )
    with pytest.raises(ValueError, match="path"):
        validate_matched_attempt_acknowledgement(
            acknowledgement.canonical_bytes(),
            expected=acknowledgement,
            acknowledgement_sha256=acknowledgement.acknowledgement_sha256(),
            evidence_path="runs/wrong.json",
        )


class _FakeRegistry:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool:
        assert isinstance(key, str)
        assert isinstance(value, bytes)
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True


def test_only_the_first_atomic_attempt_claim_wins_and_ack_binds_it() -> None:
    registry = _FakeRegistry()
    claim = _claim()

    assert claim_matched_attempt(registry, claim) == claim.claim_sha256()
    with pytest.raises(RuntimeError, match="consumed"):
        claim_matched_attempt(registry, _claim(task_id="ta-Competing123"))

    acknowledgement = _acknowledgement()
    assert matched_attempt_acknowledgement_path(
        RUN_ID,
        acknowledgement.acknowledgement_sha256(),
    ).endswith(f"/{acknowledgement.acknowledgement_sha256()}.json")


@pytest.mark.parametrize("outcome", ("success", "failure"))
def test_terminal_receipt_hash_and_path_are_canonical_and_outcome_specific(
    outcome: Literal["success", "failure"],
) -> None:
    payload = _terminal_payload(outcome)
    digest = matched_terminal_receipt_content_sha256(
        payload,
        run_id=RUN_ID,
        outcome=outcome,
    )
    path = matched_terminal_receipt_path(
        RUN_ID,
        outcome=outcome,
        content_sha256=digest,
    )
    reference = build_matched_terminal_receipt_reference(
        payload,
        run_id=RUN_ID,
        outcome=outcome,
    )

    assert reference.path == path
    assert reference.content_sha256 == digest
    assert reference.embedded_receipt_sha256 == "6" * 64
    assert (
        validate_matched_terminal_receipt_reference(
            payload,
            expected=reference,
        )
        == reference
    )
    other: Literal["success", "failure"] = "failure" if outcome == "success" else "success"
    other_payload = _terminal_payload(other)
    assert (
        matched_terminal_receipt_content_sha256(
            other_payload,
            run_id=RUN_ID,
            outcome=other,
        )
        != digest
    )
    with pytest.raises(ValueError, match="outcome"):
        matched_terminal_receipt_content_sha256(
            payload,
            run_id=RUN_ID,
            outcome=other,
        )
    with pytest.raises(ValueError, match="run ID"):
        matched_terminal_receipt_content_sha256(
            payload,
            run_id="wrong-run",
            outcome=outcome,
        )
    with pytest.raises(ValueError, match="canonical"):
        matched_terminal_receipt_content_sha256(
            payload + b"\n",
            run_id=RUN_ID,
            outcome=outcome,
        )


def test_publication_state_is_immutable_and_transitions_fail_closed() -> None:
    payload = _terminal_payload("success")
    reference = build_matched_terminal_receipt_reference(
        payload,
        run_id=RUN_ID,
        outcome="success",
    )
    initial = MatchedPublicationSnapshot(
        publication_id="4" * 64,
        run_id=RUN_ID,
        attempt_claim_sha256="5" * 64,
        status="not_started",
        cycle=0,
    )
    installing = MatchedPublicationSnapshot(
        **{
            **initial.model_dump(mode="json"),
            "status": "installing",
            "cycle": 1,
            "terminal_receipt": reference.model_dump(mode="json"),
        }
    )
    confirmed = MatchedPublicationSnapshot(
        **{
            **installing.model_dump(mode="json"),
            "status": "confirmed",
            "mounted_reload_completed": True,
        }
    )

    validate_matched_publication_transition(initial, installing)
    validate_matched_publication_transition(installing, confirmed)
    assert initial.failure_receipt_publication_allowed is True
    assert installing.failure_receipt_publication_allowed is False
    assert confirmed.failure_receipt_publication_allowed is False
    confirmed_path = matched_publication_state_path(
        RUN_ID,
        confirmed.state_sha256(),
    )
    assert confirmed_path.endswith(f"/{confirmed.state_sha256()}.json")
    assert (
        validate_matched_publication_state(
            confirmed.canonical_bytes(),
            expected=confirmed,
            state_sha256=confirmed.state_sha256(),
            evidence_path=confirmed_path,
        )
        == confirmed
    )

    with pytest.raises(ValueError, match="not monotonic"):
        validate_matched_publication_transition(confirmed, installing)
    with pytest.raises(ValueError, match="canonical"):
        validate_matched_publication_state(
            confirmed.canonical_bytes() + b"\n",
            expected=confirmed,
            state_sha256=confirmed.state_sha256(),
            evidence_path=confirmed_path,
        )
    frozen_field = "status"
    with pytest.raises(ValidationError):
        setattr(confirmed, frozen_field, "unknown")


def test_control_records_have_no_prompt_output_or_secret_fields() -> None:
    records = (
        _provenance(),
        _reviewed_inputs(),
        _deployment(),
        _launch_challenge(),
        _launch_intent(),
        _acceptance(),
        _claim(),
    )

    serialized = "\n".join(record.model_dump_json() for record in records)
    for forbidden in (
        "prompt",
        "generated_text",
        "raw_output",
        "authorization_header",
        "access_token",
        "secret_value",
    ):
        assert forbidden not in serialized

    assert hashlib.sha256(b"").hexdigest() not in {
        _launch_intent().intent_sha256(),
        _acceptance().acceptance_sha256(),
        _claim().claim_sha256(),
    }
