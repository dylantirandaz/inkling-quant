#!/usr/bin/env python3
"""Emit one real local two-rank CPU tensor-parallel evidence record as JSON."""

from __future__ import annotations

import argparse
import json

from inkling_quant_lab.distributed import run_local_cpu_tensor_parallel_smoke


def _non_empty_exact_label(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("hardware label must be non-empty without boundary whitespace")
    return value


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-label", required=True, type=_non_empty_exact_label)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main() -> None:
    """Execute the smoke and write content-complete JSON only after both ranks succeed."""

    arguments = _arguments()
    result = run_local_cpu_tensor_parallel_smoke(
        hardware_label=arguments.hardware_label,
        seed=arguments.seed,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
