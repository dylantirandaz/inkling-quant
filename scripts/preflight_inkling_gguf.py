"""No-Modal preflight for the exact Inkling GGUF plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkling_quant_lab.gguf.inkling import (  # noqa: E402
    audit_pinned_inkling_online,
    compute_cost_ceiling_usd,
    inkling_control_plane_provenance,
    inkling_run_id,
    load_inkling_gguf_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_modal.yaml"),
    )
    args = parser.parse_args()
    config = load_inkling_gguf_config(args.config)
    checked_config = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_modal.yaml"
    if Path(args.config).resolve() != checked_config.resolve():
        raise RuntimeError("Preflight only accepts the checked Inkling Modal configuration")
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    audit = audit_pinned_inkling_online(config, token=os.environ.get("HF_TOKEN"))
    print(
        json.dumps(
            {
                "status": "verified",
                "config_hash": config.config_hash(),
                "control_plane_sha256": control_plane.tree_sha256,
                "control_plane_file_count": control_plane.file_count,
                "run_id": inkling_run_id(config, control_plane.tree_sha256),
                "source": audit.model_dump(mode="json"),
                "toolchain": config.toolchain.model_dump(mode="json"),
                "quantization": config.quantization.model_dump(mode="json"),
                "coverage": config.coverage.model_dump(mode="json"),
                "configured_startup_body_window_usd": str(compute_cost_ceiling_usd(config)),
                "planned_compute_usd": str(config.budget.planned_compute_usd),
                "planned_storage_usd": str(config.budget.planned_storage_usd),
                "workspace_hard_budget_usd": str(config.budget.workspace_hard_budget_usd),
                "max_total_usd": str(config.budget.max_total_usd),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
