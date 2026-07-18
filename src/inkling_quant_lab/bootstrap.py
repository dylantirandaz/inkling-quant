"""Idempotent registration of built-in and optional component factories."""

from __future__ import annotations

from inkling_quant_lab.registry import EVALUATORS, MODEL_ADAPTERS, QUANTIZERS, REPORTERS, RUNTIMES


def _lazy(
    registry: object,
    name: str,
    module: str,
    attribute: str,
    description: str,
    *,
    extra: str | None = None,
    available: bool | None = None,
) -> None:
    from inkling_quant_lab.registry import Registry

    typed = registry
    if not isinstance(typed, Registry):
        raise TypeError("registry must be a Registry")
    if name not in typed:
        try:
            typed.register_lazy(
                name,
                module,
                attribute,
                description=description,
                optional_extra=extra,
                available=available,
            )
        except ValueError:
            # Another thread may have installed this same idempotent built-in
            # between the membership check and the locked registration.
            if name not in typed:
                raise


def register_builtins() -> None:
    """Register names without importing PyTorch or optional backend packages."""

    _lazy(
        MODEL_ADAPTERS,
        "local_fixture",
        "inkling_quant_lab.models.local",
        "create_adapter",
        "Deterministic offline dense, MoE, and multimodal fixtures",
        available=True,
    )
    _lazy(
        MODEL_ADAPTERS,
        "hf_causal_lm",
        "inkling_quant_lab.models.hf_causal_lm",
        "create_adapter",
        "Hugging Face causal-LM extension point",
        extra="hf",
        available=None,
    )
    _lazy(
        MODEL_ADAPTERS,
        "hf_causal_lm_linear_mixtral",
        "inkling_quant_lab.models.hf_causal_lm",
        "create_linear_mixtral_adapter",
        "Exact pinned Stories15M adapter with Defuser-linear expert modules",
        extra="gptq",
        available=None,
    )
    _lazy(
        MODEL_ADAPTERS,
        "mlx_lm_mixtral",
        "inkling_quant_lab.models.mlx_lm_mixtral",
        "create_adapter",
        "Exact pinned Stories15M Mixtral adapter for MLX-LM",
        extra="mlx",
        available=None,
    )
    _lazy(
        RUNTIMES,
        "torch_eager_cpu",
        "inkling_quant_lab.runtimes.torch_cpu",
        "create_runtime",
        "Portable eager PyTorch CPU runtime",
        available=True,
    )
    _lazy(
        RUNTIMES,
        "torch_eager_mps",
        "inkling_quant_lab.runtimes.torch_mps",
        "create_runtime",
        "Single-device eager PyTorch runtime for Apple MPS",
        available=None,
    )
    _lazy(
        RUNTIMES,
        "torch_eager_cuda",
        "inkling_quant_lab.runtimes.torch_cuda",
        "create_runtime",
        "Single-device eager PyTorch CUDA runtime",
        available=None,
    )
    _lazy(
        RUNTIMES,
        "mlx_metal",
        "inkling_quant_lab.runtimes.mlx_metal",
        "create_runtime",
        "Single-device MLX runtime for Apple Metal",
        extra="mlx",
        available=None,
    )
    _lazy(
        QUANTIZERS,
        "noop",
        "inkling_quant_lab.quantization.reference",
        "create_quantizer",
        "Exact no-op reference quantizer",
        available=True,
    )
    _lazy(
        QUANTIZERS,
        "torch_dynamic_int8",
        "inkling_quant_lab.quantization.int8",
        "create_quantizer",
        "CPU dynamic INT8 reference quantizer",
        available=True,
    )
    _lazy(
        QUANTIZERS,
        "torch_weight_only_int4",
        "inkling_quant_lab.quantization.weight_only",
        "create_quantizer",
        "CPU packed weight-only INT4 reference quantizer",
        available=True,
    )
    _lazy(
        QUANTIZERS,
        "torch_reference_mixed",
        "inkling_quant_lab.quantization.mixed_precision",
        "create_quantizer",
        "CPU mixed INT8/INT4 reference quantizer",
        available=True,
    )
    _lazy(
        QUANTIZERS,
        "torch_native_dynamic_int8",
        "inkling_quant_lab.quantization.native_cpu",
        "create_native_dynamic_int8_quantizer",
        "Capability-gated prepacked native PyTorch dynamic INT8 CPU quantizer",
        available=None,
    )
    _lazy(
        QUANTIZERS,
        "torch_native_int4_kleidiai",
        "inkling_quant_lab.quantization.native_cpu",
        "create_native_int4_kleidiai_quantizer",
        "Capability-gated prepacked native ATen/KleidiAI W4A8 CPU quantizer",
        available=None,
    )
    _lazy(
        QUANTIZERS,
        "fake_optional_cpu",
        "inkling_quant_lab.quantization.fake_optional",
        "create_quantizer",
        "Test-only configurable CPU optional-backend fixture",
        available=True,
    )
    _lazy(
        QUANTIZERS,
        "mlx_affine",
        "inkling_quant_lab.quantization.mlx",
        "create_quantizer",
        "Exact Stories15M MLX-LM affine q4/q8 full-eligible-leaf quantizer",
        extra="mlx",
        available=None,
    )
    for name, factory, extra, category in (
        ("awq", "create_awq_quantizer", "awq", "GPTQModel AWQ conversion backend"),
        ("gptq", "create_gptq_quantizer", "gptq", "GPTQModel GPTQ conversion backend"),
        (
            "fp8",
            "create_fp8_quantizer",
            "fp8",
            "Transformers fine-grained FP8 conversion backend",
        ),
    ):
        _lazy(
            QUANTIZERS,
            name,
            "inkling_quant_lab.quantization.optional",
            factory,
            category,
            extra=extra,
            available=None,
        )
    evaluator_classes = {
        "forward_loss": ("inkling_quant_lab.evaluation.perplexity", "ForwardLossEvaluator"),
        "perplexity": ("inkling_quant_lab.evaluation.perplexity", "PerplexityEvaluator"),
        "generation_regression": (
            "inkling_quant_lab.evaluation.generation",
            "GenerationRegressionEvaluator",
        ),
        "exact_match": ("inkling_quant_lab.evaluation.generation", "ExactMatchEvaluator"),
        "behavioral_retention": (
            "inkling_quant_lab.evaluation.behavioral",
            "BehavioralEvaluator",
        ),
        "multimodal_contract": (
            "inkling_quant_lab.evaluation.multimodal",
            "MultimodalContractEvaluator",
        ),
    }
    for name, (module, evaluator_class) in evaluator_classes.items():
        _lazy(
            EVALUATORS,
            name,
            module,
            evaluator_class,
            f"Built-in {name} evaluator",
            available=True,
        )
    _lazy(
        REPORTERS,
        "markdown",
        "inkling_quant_lab.reporting.report",
        "MarkdownReporter",
        "Dependency-free Markdown research report",
        available=True,
    )
