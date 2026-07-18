from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from inkling_quant_lab.bootstrap import register_builtins
from inkling_quant_lab.registry import MODEL_ADAPTERS, QUANTIZERS, Registry

pytestmark = pytest.mark.unit


def test_listing_lazy_backend_does_not_import_module():
    registry: Registry[object] = Registry("test backend")
    module_name = "definitely_missing_optional_backend"
    registry.register_lazy(
        "optional",
        module_name,
        "create",
        description="missing optional backend",
        optional_extra="missing",
        available=False,
    )

    descriptors = registry.descriptors()

    assert descriptors[0].name == "optional"
    assert module_name not in sys.modules


def test_builtin_registration_is_idempotent_and_sorted():
    register_builtins()
    register_builtins()
    names = [descriptor.name for descriptor in QUANTIZERS.descriptors()]
    assert names == sorted(names)
    assert {
        "noop",
        "torch_dynamic_int8",
        "torch_weight_only_int4",
        "torch_native_dynamic_int8",
        "torch_native_int4_kleidiai",
        "awq",
        "gptq",
        "fp8",
    } <= set(names)
    assert {
        "hf_causal_lm",
        "hf_causal_lm_linear_mixtral",
        "local_fixture",
        "mlx_lm_mixtral",
    } <= {descriptor.name for descriptor in MODEL_ADAPTERS.descriptors()}


def test_selected_evaluator_and_reporter_are_imported_only_when_created() -> None:
    script = textwrap.dedent(
        """
        import sys

        from inkling_quant_lab.bootstrap import register_builtins
        from inkling_quant_lab.registry import EVALUATORS, REPORTERS

        evaluator_modules = {
            "inkling_quant_lab.evaluation.perplexity",
            "inkling_quant_lab.evaluation.generation",
            "inkling_quant_lab.evaluation.behavioral",
        }
        reporter_module = "inkling_quant_lab.reporting.report"
        register_builtins()
        assert evaluator_modules.isdisjoint(sys.modules)
        assert reporter_module not in sys.modules

        EVALUATORS.create("perplexity")
        assert "inkling_quant_lab.evaluation.perplexity" in sys.modules
        assert "inkling_quant_lab.evaluation.generation" not in sys.modules
        assert "inkling_quant_lab.evaluation.behavioral" not in sys.modules

        REPORTERS.create("markdown")
        assert reporter_module in sys.modules
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
