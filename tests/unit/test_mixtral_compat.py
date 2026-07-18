from __future__ import annotations

import pytest

from inkling_quant_lab.models.mixtral_compat import (
    capture_defuser_mixtral_bindings,
    restore_defuser_mixtral_bindings,
)

pytestmark = pytest.mark.unit


def test_defuser_mixtral_global_bindings_are_restored_exactly() -> None:
    pytest.importorskip("transformers")
    from transformers import conversion_mapping, modeling_utils
    from transformers.models.mixtral import modeling_mixtral

    bindings = capture_defuser_mixtral_bindings()
    original_class = modeling_mixtral.MixtralSparseMoeBlock
    original_conversion = conversion_mapping.get_checkpoint_conversion_mapping
    original_orig_exists = hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping")
    original_orig = getattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping", None)
    original_utils_exists = hasattr(modeling_utils, "get_checkpoint_conversion_mapping")
    original_utils = getattr(modeling_utils, "get_checkpoint_conversion_mapping", None)

    try:
        modeling_mixtral.MixtralSparseMoeBlock = object
        conversion_mapping.get_checkpoint_conversion_mapping = lambda _model_type: []
        conversion_mapping.orig_get_checkpoint_conversion_mapping = lambda _model_type: []
        modeling_utils.get_checkpoint_conversion_mapping = lambda _model_type: []

        assert restore_defuser_mixtral_bindings(bindings)
        assert modeling_mixtral.MixtralSparseMoeBlock is original_class
        assert conversion_mapping.get_checkpoint_conversion_mapping is original_conversion
        assert (
            hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping")
            is original_orig_exists
        )
        assert (
            getattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping", None)
            is original_orig
        )
        assert hasattr(modeling_utils, "get_checkpoint_conversion_mapping") is original_utils_exists
        assert getattr(modeling_utils, "get_checkpoint_conversion_mapping", None) is original_utils
    finally:
        restore_defuser_mixtral_bindings(bindings)
