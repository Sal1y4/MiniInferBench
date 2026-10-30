"""A randomly initialized, few-KB Llama so runner tests need no download or GPU."""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def make_tiny_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return LlamaForCausalLM(config).eval()
