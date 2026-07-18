"""Pipeline boundary for safe, dataset-aligned routing capture."""

from __future__ import annotations

from collections.abc import Iterable

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.exceptions import RoutingInstrumentationError
from inkling_quant_lab.models.base import LoadedModel, ModelAdapter, ModelBatch, MoEDescriptor
from inkling_quant_lab.routing.traces import (
    BatchMeta,
    InMemoryRoutingSink,
    RoutingArtifact,
    RoutingMode,
)


class RoutingUnsupportedError(RoutingInstrumentationError):
    """The selected model/adapter exposes no supported routing instrumentation."""

    code = "ROUTING_UNSUPPORTED"


def _effective_config(
    config: ExperimentConfig, *, require_trace_alignment: bool
) -> tuple[ExperimentConfig, RoutingMode]:
    mode: RoutingMode = "full_trace" if require_trace_alignment else config.routing.mode
    if mode == config.routing.mode:
        return config, mode
    routing = config.routing.model_copy(update={"mode": mode})
    return config.model_copy(update={"routing": routing}), mode


def capture_routing_dataset(
    adapter: ModelAdapter,
    model: LoadedModel,
    batches: Iterable[ModelBatch],
    config: ExperimentConfig,
    *,
    require_trace_alignment: bool = False,
) -> RoutingArtifact:
    """Capture routing for local batches through the adapter's loss execution path.

    Aggregate and sampled modes follow the resolved experiment configuration.
    Comparison callers can request full token traces explicitly, without mutating
    the immutable input configuration. Stable sample IDs must be unique across
    the captured dataset so token alignment cannot silently become ambiguous.
    """

    effective_config, mode = _effective_config(
        config, require_trace_alignment=require_trace_alignment
    )
    if mode == "off":
        return InMemoryRoutingSink(mode="off").close()

    descriptor = adapter.discover_moe(model)
    if descriptor is None:
        raise RoutingUnsupportedError(
            f"routing capture is unsupported for model {model.descriptor.model_id}",
            component=adapter.__class__.__name__,
            remediation="disable routing or use an adapter with validated MoE discovery",
            details={"model_id": model.descriptor.model_id, "mode": mode},
        )

    expert_counts = {layer.layer_id: layer.expert_count for layer in descriptor.layers}
    sink = InMemoryRoutingSink(
        mode=mode,
        sampled_token_positions=effective_config.routing.sampled_token_positions,
        expert_counts=expert_counts,
    )
    handle = adapter.attach_routing_hooks(model, sink, effective_config)
    seen_sample_ids: set[str] = set()
    captured_batch_count = 0
    try:
        for batch_index, batch in enumerate(batches):
            batch_meta = BatchMeta(sample_ids=batch.sample_ids, batch_id=str(batch_index))
            duplicates = seen_sample_ids.intersection(batch_meta.sample_ids)
            if duplicates:
                duplicate_list = ", ".join(sorted(duplicates))
                raise RoutingInstrumentationError(
                    f"routing dataset repeats stable sample IDs: {duplicate_list}",
                    component="routing_capture",
                )
            seen_sample_ids.update(batch_meta.sample_ids)
            sink.start_batch(batch_meta)
            try:
                adapter.forward_loss(model, batch)
            finally:
                sink.end_batch()
            captured_batch_count += 1
    finally:
        handle.remove()

    if captured_batch_count == 0:
        raise RoutingInstrumentationError(
            "routing capture dataset produced no batches",
            component="routing_capture",
        )
    return sink.close()


capture_routing = capture_routing_dataset


def expert_usage_by_module(artifact: RoutingArtifact, descriptor: MoEDescriptor) -> dict[str, int]:
    """Map normalized layer/expert counts to exact expert module names.

    Missing or unknown layers are rejected rather than silently manufacturing
    frequency statistics. Explicit zero-traffic experts remain present with a
    count of zero, which lets frequency-tier policies distinguish them from
    missing instrumentation.
    """

    descriptor_layer_ids = [layer.layer_id for layer in descriptor.layers]
    if len(set(descriptor_layer_ids)) != len(descriptor_layer_ids):
        raise RoutingInstrumentationError(
            "MoE descriptor contains duplicate layer IDs",
            component="routing_capture",
        )
    artifact_layer_ids = set(artifact.aggregates)
    expected_layer_ids = set(descriptor_layer_ids)
    missing_layers = expected_layer_ids - artifact_layer_ids
    unknown_layers = artifact_layer_ids - expected_layer_ids
    if missing_layers or unknown_layers:
        details = []
        if missing_layers:
            details.append(f"missing layers: {', '.join(sorted(missing_layers))}")
        if unknown_layers:
            details.append(f"unknown layers: {', '.join(sorted(unknown_layers))}")
        raise RoutingInstrumentationError(
            "routing aggregate/descriptor mismatch (" + "; ".join(details) + ")",
            component="routing_capture",
        )

    usage: dict[str, int] = {}
    for layer in descriptor.layers:
        if len(layer.expert_module_names) != layer.expert_count:
            raise RoutingInstrumentationError(
                f"descriptor layer {layer.layer_id} does not name every expert",
                component="routing_capture",
            )
        aggregate = artifact.aggregates[layer.layer_id]
        if aggregate.expert_count != layer.expert_count:
            raise RoutingInstrumentationError(
                f"routing layer {layer.layer_id} expert count does not match its descriptor",
                component="routing_capture",
            )
        for expert_id, module_name in enumerate(layer.expert_module_names):
            if module_name in usage:
                raise RoutingInstrumentationError(
                    f"duplicate expert module name in descriptor: {module_name}",
                    component="routing_capture",
                )
            usage[module_name] = aggregate.expert_selection_counts.get(expert_id, 0)
    return usage
