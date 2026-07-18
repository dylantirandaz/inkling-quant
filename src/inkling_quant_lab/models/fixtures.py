"""Deterministic local tokenizer, dense LM, MoE LM, and multimodal stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FixedTokenizer:
    """Small whitespace tokenizer with a checked-in, immutable vocabulary."""

    TOKENS = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "red",
        "blue",
        "green",
        "yellow",
        "one",
        "two",
        "three",
        "four",
        "cat",
        "dog",
        "bird",
        "fish",
        "yes",
        "no",
        "safe",
        "helpful",
        "route",
        "expert",
        "tiny",
        "model",
        "image",
        "sound",
        "answer",
        "unknown",
    )

    def __init__(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(self.TOKENS)}
        self.id_to_token = dict(enumerate(self.TOKENS))
        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.unk_token_id = self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""

        return len(self.TOKENS)

    def encode(self, text: str, *, special_tokens: bool = True) -> list[int]:
        """Encode normalized whitespace tokens deterministically."""

        pieces = text.strip().lower().split()
        tokens = [self.token_to_id.get(piece, self.unk_token_id) for piece in pieces]
        return [self.bos_token_id, *tokens, self.eos_token_id] if special_tokens else tokens

    def decode(self, token_ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        """Decode token IDs without attempting lossy punctuation recovery."""

        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return " ".join(
            self.id_to_token.get(int(token_id), "<unk>")
            for token_id in token_ids
            if not skip_special or int(token_id) not in special
        )

    def batch_encode(self, texts: tuple[str, ...]) -> tuple[Tensor, Tensor]:
        """Pad encoded inputs and return IDs plus an attention mask."""

        if not texts:
            raise ValueError("texts must not be empty")
        encoded = [self.encode(text) for text in texts]
        width = max(len(tokens) for tokens in encoded)
        ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, tokens in enumerate(encoded):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = 1
        return ids, mask


def _initialize_deterministically(module: nn.Module) -> None:
    """Fill parameters from a formula independent of global RNG state."""

    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(0.08 * torch.sin(values * 0.37 + float(index + 1)))


class TinyBlock(nn.Module):
    """Small residual MLP block used by the dense fixture."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear_in: nn.Linear = nn.Linear(hidden_size, hidden_size * 2)
        self.linear_out: nn.Linear = nn.Linear(hidden_size * 2, hidden_size)
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        """Apply a deterministic residual transformation."""

        update = self.linear_out(torch.tanh(self.linear_in(hidden)))
        return cast(Tensor, self.norm(hidden + update))


class TinyDenseCausalLM(nn.Module):
    """Two-block CPU causal language model fixture."""

    architecture_name = "tiny_dense_causal_lm_v1"

    def __init__(self, vocab_size: int = len(FixedTokenizer.TOKENS), hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList((TinyBlock(hidden_size), TinyBlock(hidden_size)))
        self.lm_head: nn.Linear = nn.Linear(hidden_size, vocab_size, bias=False)
        _initialize_deterministically(self)

    def forward(self, input_ids: Tensor, *, multimodal_inputs: Tensor | None = None) -> Tensor:
        """Return next-token logits."""

        del multimodal_inputs
        hidden = self.embedding(input_ids)
        for block_module in self.blocks:
            block = cast(TinyBlock, block_module)
            hidden = block(hidden)
        return cast(Tensor, self.lm_head(hidden))

    def generate(self, input_ids: Tensor, *, max_new_tokens: int) -> Tensor:
        """Greedily append a fixed number of tokens."""

        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated


@dataclass(frozen=True, slots=True)
class RoutingSnapshot:
    """Ephemeral output of a tiny routed layer for instrumentation hooks."""

    layer_id: str
    selected_expert_ids: Tensor
    selected_weights: Tensor
    router_probabilities: Tensor
    router_logits: Tensor


class TinyMoELayer(nn.Module):
    """Top-2 routed layer with four deterministic token-sensitive experts."""

    def __init__(self, hidden_size: int, layer_index: int, expert_count: int = 4) -> None:
        super().__init__()
        self.layer_id = f"moe_{layer_index}"
        self.top_k = 2
        self.router: nn.Linear = nn.Linear(hidden_size, expert_count, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.Tanh(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
            for _ in range(expert_count)
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_size)
        self._snapshot: RoutingSnapshot | None = None

    def forward(self, hidden: Tensor) -> Tensor:
        """Route every token to two experts and retain a detached snapshot."""

        logits = self.router(hidden)
        probabilities = torch.softmax(logits, dim=-1)
        selected_weights, selected_ids = torch.topk(probabilities, self.top_k, dim=-1)
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
        expert_outputs = torch.stack(tuple(expert(hidden) for expert in self.experts), dim=-2)
        gather_index = selected_ids.unsqueeze(-1).expand(*selected_ids.shape, hidden.shape[-1])
        routed = torch.gather(expert_outputs, -2, gather_index)
        update = (routed * selected_weights.unsqueeze(-1)).sum(dim=-2)
        self._snapshot = RoutingSnapshot(
            layer_id=self.layer_id,
            selected_expert_ids=selected_ids.detach(),
            selected_weights=selected_weights.detach(),
            router_probabilities=probabilities.detach(),
            router_logits=logits.detach(),
        )
        return cast(Tensor, self.norm(hidden + update))

    def routing_snapshot(self) -> dict[str, Any]:
        """Expose normalized routing fields for adapter-owned instrumentation."""

        if self._snapshot is None:
            raise RuntimeError("routing snapshot is unavailable before a forward pass")
        return {
            "layer_id": self._snapshot.layer_id,
            "selected_expert_ids": self._snapshot.selected_expert_ids,
            "selected_weights": self._snapshot.selected_weights,
            "router_probabilities": self._snapshot.router_probabilities,
            "router_logits": self._snapshot.router_logits,
        }


class TinyMoECausalLM(nn.Module):
    """Two-layer, four-expert-per-layer, top-2 MoE fixture."""

    architecture_name = "tiny_moe_causal_lm_v1"

    def __init__(self, vocab_size: int = len(FixedTokenizer.TOKENS), hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
        self.moe_layers = nn.ModuleList(
            (TinyMoELayer(hidden_size, 0), TinyMoELayer(hidden_size, 1))
        )
        self.lm_head: nn.Linear = nn.Linear(hidden_size, vocab_size, bias=False)
        _initialize_deterministically(self)
        self._initialize_router_preferences()

    def _initialize_router_preferences(self) -> None:
        with torch.no_grad():
            for layer_index, layer_module in enumerate(self.moe_layers):
                layer = cast(TinyMoELayer, layer_module)
                width = layer.router.in_features
                values = torch.arange(4 * width, dtype=torch.float32).reshape(4, width)
                layer.router.weight.copy_(
                    0.22 * torch.cos(values * 0.31 + float(layer_index) * 0.7)
                )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return logits and update each routed layer's snapshot."""

        hidden = self.embedding(input_ids)
        for layer_module in self.moe_layers:
            layer = cast(TinyMoELayer, layer_module)
            hidden = layer(hidden)
        return cast(Tensor, self.lm_head(hidden))

    def generate(self, input_ids: Tensor, *, max_new_tokens: int) -> Tensor:
        """Greedily generate while exposing routes for every forward pass."""

        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            next_token = self(generated)[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated


class TinyMultimodalCausalLM(TinyDenseCausalLM):
    """Contract-only text model with a tiny image/audio tensor projector."""

    architecture_name = "tiny_multimodal_stub_v1"

    def __init__(self) -> None:
        super().__init__()
        self.multimodal_projector: nn.Linear = nn.Linear(4, self.embedding.embedding_dim)
        _initialize_deterministically(self.multimodal_projector)

    def forward(self, input_ids: Tensor, *, multimodal_inputs: Tensor | None = None) -> Tensor:
        """Condition the first token on a four-value modality vector."""

        hidden = self.embedding(input_ids)
        if multimodal_inputs is not None:
            projected = self.multimodal_projector(multimodal_inputs).unsqueeze(1)
            hidden = hidden.clone()
            hidden[:, :1, :] = hidden[:, :1, :] + projected
        for block_module in self.blocks:
            block = cast(TinyBlock, block_module)
            hidden = block(hidden)
        return cast(Tensor, self.lm_head(hidden))


def causal_loss_per_sample(
    logits: Tensor, input_ids: Tensor, attention_mask: Tensor | None
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Compute per-sample token-mean NLL and valid shifted-token counts."""

    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    if attention_mask is None:
        valid = torch.ones_like(shifted_labels, dtype=torch.bool)
    else:
        valid = attention_mask[:, 1:].bool()
    losses: list[float] = []
    counts: list[int] = []
    for row in range(input_ids.shape[0]):
        row_valid = valid[row]
        count = int(row_valid.sum().item())
        if count <= 0:
            raise ValueError("each loss sample must contain at least two tokens")
        loss = F.cross_entropy(
            shifted_logits[row][row_valid], shifted_labels[row][row_valid], reduction="mean"
        )
        losses.append(float(loss.item()))
        counts.append(count)
    return tuple(losses), tuple(counts)
