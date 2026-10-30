"""Conservative GPU memory estimate used to skip configs that would not fit.

The estimate is deliberately simple and errs on the high side. A config that
slips through and still runs out of memory is caught at run time and recorded
as ENV_ERROR, so the estimate only has to avoid obviously hopeless points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

GIB = 2**30


@dataclass(frozen=True, slots=True)
class ModelShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    weight_bytes: int
    dtype_bytes: int

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, every layer: 2 x layers x kv_heads x head_dim x dtype bytes."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes


def model_shape(model: torch.nn.Module, hf_config: Any) -> ModelShape:
    heads = hf_config.num_attention_heads
    param = next(model.parameters())
    return ModelShape(
        num_layers=hf_config.num_hidden_layers,
        num_kv_heads=getattr(hf_config, "num_key_value_heads", None) or heads,
        head_dim=getattr(hf_config, "head_dim", None) or hf_config.hidden_size // heads,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        weight_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        dtype_bytes=param.element_size(),
    )


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    weights: int
    kv_cache: int
    activations: int
    logits: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.activations + self.logits


def estimate(shape: ModelShape, batch_size: int, input_len: int, output_len: int) -> MemoryEstimate:
    tokens = batch_size * (input_len + output_len)
    # Peak prefill activations: a few live [batch, seq, max(hidden, mlp)] tensors.
    # A factor of 4 is a loose upper bound for one decoder layer with SDPA.
    widest = max(shape.hidden_size, shape.intermediate_size)
    activations = 4 * batch_size * input_len * widest * shape.dtype_bytes
    # logits_to_keep=1 keeps only the last position; fp32 upcast in the head.
    logits = 2 * batch_size * shape.vocab_size * 4
    return MemoryEstimate(
        weights=shape.weight_bytes,
        kv_cache=tokens * shape.kv_bytes_per_token,
        activations=activations,
        logits=logits,
    )


def available_bytes(safety_fraction: float = 0.9) -> int | None:
    """Free device memory now, minus a safety margin; None without CUDA."""
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free * safety_fraction)
