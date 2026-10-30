"""Metrics computed from raw run samples. Pure functions, no GPU needed.

All latency metrics here are *offline, static-batch* numbers: no request
queueing, no tokenization, no network. They are not the TTFT/P99 an online
serving system would report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default), q in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence")
    return sum(values) / len(values)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Sample standard deviation divided by the mean; 0.0 for fewer than two values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) / m if m else 0.0


def tpot_ms(e2e_ms: float, ttft_ms: float, output_len: int) -> float:
    """Time per output token after the first: (E2E - TTFT) / (output_len - 1)."""
    if output_len < 2:
        raise ValueError("TPOT needs at least two output tokens")
    return (e2e_ms - ttft_ms) / (output_len - 1)


def prefill_tokens_per_s(batch_size: int, input_len: int, ttft_ms: float) -> float:
    """Input tokens processed per second during prefill (TTFT used as prefill time)."""
    return batch_size * input_len / (ttft_ms / 1000.0)


def decode_steps_per_s(itl_ms: Sequence[float]) -> float:
    """Decode steps per second over the given steps. Not tokens/s."""
    return len(itl_ms) / (sum(itl_ms) / 1000.0)


def decode_tokens_per_s(itl_ms: Sequence[float], batch_size: int) -> float:
    """steps/s x tokens produced per step. Every sequence stays active (EOS ignored),
    so each step produces exactly `batch_size` tokens."""
    return decode_steps_per_s(itl_ms) * batch_size


def output_tokens_per_s(batch_size: int, output_len: int, e2e_ms: float) -> float:
    """All generated tokens over end-to-end time, prefill included."""
    return batch_size * output_len / (e2e_ms / 1000.0)
