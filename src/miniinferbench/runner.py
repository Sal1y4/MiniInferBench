"""Hand-written greedy prefill + decode loop with per-step timing.

We do not call `model.generate()` for measurement: it hides the step loop, so
there is nowhere to put a timer between tokens. Instead we drive the model one
forward pass at a time and keep the KV cache ourselves. `generate()` is still
used once per configuration, as the reference for a token-level correctness
check.

Measurement boundaries for one run (all inputs are already on the device):

    start ── prefill forward ── first token ── decode step ... ── last token
      |<------------ TTFT ----------->|<- ITL ->|          |<- ITL ->|
      |<------------------------------ E2E ---------------------------->|
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache, GenerationConfig, PreTrainedModel

from miniinferbench.timing import EventTimeline, WallClock


@dataclass(frozen=True, slots=True)
class RunSamples:
    """Raw timings of one prefill + decode run.

    Attributes:
        ttft_ms: Start of prefill to first token, on the device timeline.
        itl_ms: Gap between consecutive tokens, one entry per decode step,
            on the device timeline.
        e2e_wall_ms: Synchronized wall-clock time for the whole run.
        peak_memory_bytes: `torch.cuda.max_memory_allocated()` during the run,
            or None on CPU.
    """

    ttft_ms: float
    itl_ms: list[float]
    e2e_wall_ms: float
    peak_memory_bytes: int | None

    @property
    def e2e_event_ms(self) -> float:
        return self.ttft_ms + sum(self.itl_ms)


def _next_token(logits: torch.Tensor) -> torch.Tensor:
    """Greedy pick for the last position: `[batch, seq, vocab]` -> `[batch, 1]`."""
    return logits[:, -1, :].argmax(dim=-1, keepdim=True)


@torch.inference_mode()
def greedy_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    output_len: int,
    timeline: EventTimeline | None = None,
) -> torch.Tensor:
    """Generate exactly `output_len` tokens per sequence; returns `[batch, output_len]`.

    EOS is ignored on purpose so every run does the same amount of work.
    If `timeline` is given, it is marked before prefill and after every token.
    """
    if output_len < 1:
        raise ValueError(f"output_len must be >= 1, got {output_len}")
    cache = DynamicCache()
    if timeline is not None:
        timeline.mark()
    # logits_to_keep=1: only the last position's logits are needed. Without it
    # prefill materializes batch x input_len x vocab logits, which can cost GBs.
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    token = _next_token(out.logits)
    tokens = [token]
    if timeline is not None:
        timeline.mark()
    for _ in range(output_len - 1):
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        token = _next_token(out.logits)
        tokens.append(token)
        if timeline is not None:
            timeline.mark()
    return torch.cat(tokens, dim=1)


def measure_run(model: PreTrainedModel, input_ids: torch.Tensor, output_len: int) -> RunSamples:
    """Run one timed generation and return its raw samples."""
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timeline = EventTimeline(device)
    wall = WallClock(device)
    wall.start()
    greedy_decode(model, input_ids, output_len, timeline)
    e2e_wall_ms = wall.stop()
    gaps = timeline.finish()
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return RunSamples(
        ttft_ms=gaps[0], itl_ms=gaps[1:], e2e_wall_ms=e2e_wall_ms, peak_memory_bytes=peak
    )


@torch.inference_mode()
def reference_generate(
    model: PreTrainedModel, input_ids: torch.Tensor, output_len: int
) -> torch.Tensor:
    """Hugging Face greedy generation with EOS disabled; returns new tokens only."""
    config = GenerationConfig(
        do_sample=False,
        max_new_tokens=output_len,
        min_new_tokens=output_len,
        eos_token_id=None,
        pad_token_id=0,
    )
    attention_mask = torch.ones_like(input_ids)
    out: torch.Tensor = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, generation_config=config
    )
    return out[:, input_ids.shape[1] :]


@dataclass(frozen=True, slots=True)
class CorrectnessReport:
    matches: bool
    first_mismatch: tuple[int, int] | None  # (sequence, position)


def check_correctness(
    model: PreTrainedModel, input_ids: torch.Tensor, output_len: int
) -> CorrectnessReport:
    """Compare our loop with `generate()` token by token."""
    ours = greedy_decode(model, input_ids, output_len)
    reference = reference_generate(model, input_ids, output_len)
    diff = (ours != reference).nonzero()
    if diff.numel() == 0:
        return CorrectnessReport(matches=True, first_mismatch=None)
    seq, pos = (int(v) for v in diff[0])
    return CorrectnessReport(matches=False, first_mismatch=(seq, pos))
