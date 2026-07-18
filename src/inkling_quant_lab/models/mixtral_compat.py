"""Scoped Defuser compatibility for legacy per-expert Mixtral checkpoints."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

_DEFUSER_VERSION = "0.0.23"
_LINEAR_MIXTRAL_CLASS = "defuser.modeling.unfused_moe.mixtral.LinearMixtralSparseMoeBlock"


@dataclass(frozen=True, slots=True)
class _AttributeBinding:
    owner: Any
    name: str
    existed: bool
    value: Any


@dataclass(frozen=True, slots=True)
class DefuserMixtralBindings:
    """Exact mutable Transformers bindings touched by Defuser 0.0.23."""

    attributes: tuple[_AttributeBinding, ...]


def _capture_attribute(owner: Any, name: str) -> _AttributeBinding:
    existed = hasattr(owner, name)
    return _AttributeBinding(
        owner=owner,
        name=name,
        existed=existed,
        value=getattr(owner, name, None),
    )


def capture_defuser_mixtral_bindings() -> DefuserMixtralBindings:
    """Capture every global binding changed by ``replace_fused_blocks('mixtral')``."""

    from transformers import conversion_mapping, modeling_utils
    from transformers.models.mixtral import modeling_mixtral

    return DefuserMixtralBindings(
        attributes=(
            _capture_attribute(modeling_mixtral, "MixtralSparseMoeBlock"),
            _capture_attribute(conversion_mapping, "get_checkpoint_conversion_mapping"),
            _capture_attribute(conversion_mapping, "orig_get_checkpoint_conversion_mapping"),
            _capture_attribute(modeling_utils, "get_checkpoint_conversion_mapping"),
        )
    )


def restore_defuser_mixtral_bindings(bindings: DefuserMixtralBindings) -> bool:
    """Restore captured bindings exactly and verify that no global patch remains."""

    changed = False
    for binding in reversed(bindings.attributes):
        current_exists = hasattr(binding.owner, binding.name)
        current = getattr(binding.owner, binding.name, None)
        if current_exists != binding.existed or (current_exists and current is not binding.value):
            changed = True
        if binding.existed:
            setattr(binding.owner, binding.name, binding.value)
        elif current_exists:
            delattr(binding.owner, binding.name)
    for binding in bindings.attributes:
        restored_exists = hasattr(binding.owner, binding.name)
        restored = getattr(binding.owner, binding.name, None)
        if restored_exists != binding.existed or (
            restored_exists and restored is not binding.value
        ):
            raise RuntimeError(f"failed to restore Transformers binding {binding.name}")
    return changed


def apply_defuser_linear_mixtral_patch() -> None:
    """Apply the exact audited Defuser class/checkpoint mapping patch."""

    installed = importlib.metadata.version("defuser")
    if installed != _DEFUSER_VERSION:
        raise RuntimeError(
            f"legacy Mixtral loading requires defuser=={_DEFUSER_VERSION}, found {installed}"
        )
    import defuser
    from transformers.models.mixtral import modeling_mixtral

    if not bool(defuser.replace_fused_blocks("mixtral")):
        raise RuntimeError("Defuser did not apply the registered Mixtral replacement")
    resolved = modeling_mixtral.MixtralSparseMoeBlock
    class_path = f"{resolved.__module__}.{resolved.__qualname__}"
    if class_path != _LINEAR_MIXTRAL_CLASS:
        raise RuntimeError(f"Defuser resolved an unexpected Mixtral class: {class_path}")


@contextmanager
def scoped_defuser_linear_mixtral() -> Iterator[None]:
    """Apply Defuser only while constructing one model, then restore all globals."""

    bindings = capture_defuser_mixtral_bindings()
    try:
        apply_defuser_linear_mixtral_patch()
        yield
    finally:
        restore_defuser_mixtral_bindings(bindings)
