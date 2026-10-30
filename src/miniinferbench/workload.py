"""Synthetic, deterministic token workloads.

Prompts are random token ids rather than real text: the benchmark measures
how fast the model processes a given shape (batch x length), not what it
says. Special tokens (BOS, EOS, chat markers) are excluded so that no
sequence accidentally contains control tokens.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch


def make_prompts(
    batch_size: int,
    input_len: int,
    vocab_size: int,
    seed: int,
    excluded_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Return a `[batch_size, input_len]` int64 CPU tensor of token ids.

    The same arguments always produce the same tensor, on any machine,
    because sampling uses a dedicated CPU generator seeded with `seed`.
    """
    excluded = set(excluded_ids)
    allowed = torch.tensor([i for i in range(vocab_size) if i not in excluded], dtype=torch.int64)
    if allowed.numel() == 0:
        raise ValueError("no token ids left after excluding special tokens")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    picks = torch.randint(0, allowed.numel(), (batch_size, input_len), generator=generator)
    return allowed[picks]


def fingerprint(token_ids: torch.Tensor) -> str:
    """Short SHA-256 of the token ids, stored with results to prove the inputs match."""
    data = token_ids.to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
    shape = ",".join(str(d) for d in token_ids.shape).encode()
    return hashlib.sha256(shape + b":" + data).hexdigest()[:16]
