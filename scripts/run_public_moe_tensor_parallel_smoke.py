#!/usr/bin/env python3
"""Execute and immutably record the public Stories15M expert tensor-parallel proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from inkling_quant_lab.public_moe_tensor_parallel import (
    run_public_moe_expert_tensor_parallel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = PROJECT_ROOT / "src/inkling_quant_lab/public_moe_tensor_parallel.py"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_empty_label(value: str) -> str:
    if not value or value != value.strip():
        raise argparse.ArgumentTypeError(
            "hardware label must be non-empty without boundary whitespace"
        )
    return value


def _output_path(value: str) -> Path:
    raw = Path(value)
    if ".." in raw.parts:
        raise argparse.ArgumentTypeError("output path cannot contain parent traversal")
    resolved = (PROJECT_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    artifact_root = (PROJECT_ROOT / "artifacts").resolve()
    if not resolved.is_relative_to(artifact_root):
        raise argparse.ArgumentTypeError("output must remain below the repository artifact root")
    if resolved.exists() or resolved.is_symlink():
        raise argparse.ArgumentTypeError("output must be a new immutable path")
    return resolved


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--hardware-label", required=True, type=_non_empty_label)
    parser.add_argument("--output", required=True, type=_output_path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    return parser.parse_args(argv)


def _publish_json(output: Path, value: dict[str, object]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".part", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace immutable output: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    """Run only frozen implementation bytes and publish a path-redacted record."""

    arguments = _arguments(argv)
    runner = Path(__file__).resolve()
    runner_hash_at_start = _file_sha256(runner)
    implementation_hash_at_start = _file_sha256(IMPLEMENTATION)
    result = run_public_moe_expert_tensor_parallel(
        arguments.snapshot,
        hardware_label=arguments.hardware_label,
        timeout_seconds=arguments.timeout_seconds,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
    )
    runner_hash_at_end = _file_sha256(runner)
    implementation_hash_at_end = _file_sha256(IMPLEMENTATION)
    if runner_hash_at_end != runner_hash_at_start:
        raise RuntimeError("runner changed during the measured execution")
    if implementation_hash_at_end != implementation_hash_at_start:
        raise RuntimeError("implementation changed during the measured execution")
    record: dict[str, object] = {
        "schema_version": "public-moe-expert-tensor-parallel-evidence-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_template": (
            "$PYTHON scripts/run_public_moe_tensor_parallel_smoke.py "
            "--snapshot $PINNED_SNAPSHOT --hardware-label $HARDWARE_LABEL "
            "--timeout-seconds 120 --absolute-tolerance 1e-7 "
            "--relative-tolerance 1e-5 --output $NEW_ARTIFACT_PATH"
        ),
        "absolute_local_paths_redacted": True,
        "implementation": {
            "runner": {
                "path": "scripts/run_public_moe_tensor_parallel_smoke.py",
                "sha256_at_start": runner_hash_at_start,
                "sha256_at_end": runner_hash_at_end,
            },
            "module": {
                "path": "src/inkling_quant_lab/public_moe_tensor_parallel.py",
                "sha256_at_start": implementation_hash_at_start,
                "sha256_at_end": implementation_hash_at_end,
            },
            "git_commit": None,
            "git_unavailable_reason": "workspace has no .git metadata",
        },
        "result": result.model_dump(mode="json"),
        "limitations": [
            "The source is a public Stories15M MoE checkpoint, not an Inkling checkpoint.",
            "Every one of its six expert blocks is executed with sharded expert parameters, but "
            "attention, residuals, normalization, embedding lookup, and LM head are not an "
            "end-to-end distributed transformer forward.",
            "This validates numerical execution, exact slice loading, reconstruction, and local "
            "Gloo communication; it is not a latency, memory-peak, energy, training, export, "
            "multi-host, or fault-tolerance result.",
            "The publisher router is randomized and each layer/projection repeats expert weights, "
            "so this does not establish learned specialization preservation.",
        ],
    }
    _publish_json(arguments.output, record)


if __name__ == "__main__":
    main()
