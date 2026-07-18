"""Deterministic persistent-state identity shared by model adapters and quantizers."""

from __future__ import annotations

import hashlib

import torch
from torch import nn


def model_state_sha256(model: nn.Module) -> str:
    """Hash state names, shapes, dtypes, and exact contiguous tensor bytes."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()
