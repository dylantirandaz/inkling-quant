"""Contract coverage for the documented release-command config paths."""

from pathlib import Path

import pytest

from inkling_quant_lab.config import load_config

pytestmark = pytest.mark.contract

_EXPERIMENTS = Path(__file__).parents[2] / "configs" / "experiments"


@pytest.mark.parametrize(
    ("release_name", "canonical_name"),
    (
        ("tiny-moe-baseline.yaml", "tiny_moe_baseline.yaml"),
        ("tiny-moe-int8.yaml", "tiny_moe_int8.yaml"),
    ),
)
def test_release_config_alias_resolves_to_canonical_experiment(
    release_name: str,
    canonical_name: str,
) -> None:
    """Documented hyphenated commands retain the canonical config identity."""

    release = load_config(_EXPERIMENTS / release_name)
    canonical = load_config(_EXPERIMENTS / canonical_name)

    assert release == canonical
    assert release.config_hash() == canonical.config_hash()
