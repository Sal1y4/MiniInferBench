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

import weakref
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import DynamicCache, GenerationConfig, PreTrainedModel, StaticCache

from miniinferbench.config import DecodeMode, Regression
from miniinferbench.timing import EventTimeline, WallClock

# Deliberate slowdowns for the regression case study. Each one keeps the output
# identical, so only the harness's timing comparison and the trace can catch it.


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


# Optional wrapper around each phase ("prefill", "decode_step"), e.g. the
# profiler's record_function. Defaults to doing nothing.
PhaseScope = Callable[[str], AbstractContextManager[Any]]


def _no_scope(_: str) -> AbstractContextManager[None]:
    return nullcontext()


@dataclass
class _GraphState:
    forward: Callable[..., Any]
    caches: dict[tuple[int, int], StaticCache]


# One compiled forward per model, plus one static cache per (batch, length),
# reused across runs: CUDA graphs replay fixed memory addresses, so the cache
# must stay where it was when the graph was recorded.
_GRAPH_STATE: weakref.WeakKeyDictionary[PreTrainedModel, _GraphState] = weakref.WeakKeyDictionary()


def _graph_state(model: PreTrainedModel, device: torch.device) -> _GraphState:
    state = _GRAPH_STATE.get(model)
    if state is None:
        forward: Callable[..., Any] = model.forward
        if device.type == "cuda":
            # "reduce-overhead" captures each decode step as a CUDA graph and
            # replays it with one launch instead of ~1,300. CPU has no CUDA
            # graphs, so there the static-cache path runs eagerly (used by tests).
            forward = torch.compile(model.forward, mode="reduce-overhead")
        state = _GraphState(forward=forward, caches={})
        _GRAPH_STATE[model] = state
    return state


def _static_cache(
    state: _GraphState, model: PreTrainedModel, batch_size: int, max_len: int
) -> StaticCache:
    key = (batch_size, max_len)
    cache = state.caches.get(key)
    if cache is None:
        cache = StaticCache(config=model.config, max_cache_len=max_len)
        state.caches[key] = cache
    else:
        cache.reset()
    return cache


@torch.inference_mode()
def greedy_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    output_len: int,
    timeline: EventTimeline | None = None,
    regression: Regression = "none",
    decode_mode: DecodeMode = "eager",
    phase: PhaseScope = _no_scope,
) -> torch.Tensor:
    """Generate exactly `output_len` tokens per sequence; returns `[batch, output_len]`.

    EOS is ignored on purpose so every run does the same amount of work.
    If `timeline` is given, it is marked before prefill and after every token.
    `regression` injects a known slowdown (see `Regression`); "none" in normal use.
    `decode_mode="cuda_graph"` uses a static KV cache and replays each decode
    step as a CUDA graph; prefill stays eager in both modes.
    """
    if output_len < 1:
        raise ValueError(f"output_len must be >= 1, got {output_len}")
    if decode_mode == "cuda_graph" and regression == "no_kv_cache":
        raise ValueError("no_kv_cache cannot be combined with cuda_graph decoding")
    batch_size, input_len = input_ids.shape
    device = input_ids.device
    if decode_mode == "cuda_graph":
        state = _graph_state(model, device)
        cache: DynamicCache | StaticCache = _static_cache(
            state, model, batch_size, input_len + output_len
        )
        decode_forward = state.forward
    else:
        cache = DynamicCache()
        decode_forward = model.forward
    if timeline is not None:
        timeline.mark()
    # logits_to_keep=1: only the last position's logits are needed. Without it
    # prefill materializes batch x input_len x vocab logits, which can cost GBs.
    with phase("prefill"):
        out = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            cache_position=torch.arange(input_len, device=device),
        )
        token = _next_token(out.logits)
    tokens = [token]
    if timeline is not None:
        timeline.mark()
    sequence = input_ids
    for step in range(output_len - 1):
        with phase("decode_step"):
            if regression == "no_kv_cache":
                # Recompute attention over the whole sequence every step.
                sequence = torch.cat([sequence, token], dim=1)
                out = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
            else:
                if decode_mode == "cuda_graph" and device.type == "cuda":
                    # The previous step's graph outputs are consumed already.
                    torch.compiler.cudagraph_mark_step_begin()  # type: ignore[no-untyped-call]
                position = torch.tensor([input_len + step], device=device)
                out = decode_forward(
                    input_ids=token,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=position,
                )
            token = _next_token(out.logits)
            if regression == "host_sync_each_step":
                # A common real bug: copying each token to the host (e.g. to check
                # for EOS) forces the CPU to wait for the GPU every step.
                token.cpu()
        tokens.append(token)
        if timeline is not None:
            timeline.mark()
    return torch.cat(tokens, dim=1)


def measure_run(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    output_len: int,
    regression: Regression = "none",
    decode_mode: DecodeMode = "eager",
) -> RunSamples:
    """Run one timed generation and return its raw samples."""
    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timeline = EventTimeline(device)
    wall = WallClock(device)
    wall.start()
    greedy_decode(model, input_ids, output_len, timeline, regression, decode_mode)
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
    # eos_token_id=[] (not None): None falls back to the model's own EOS, and
    # min_new_tokens then masks it to -inf, forcing the runner-up token
    # wherever EOS is the true argmax. Found on SmolLM2, margin ~7 logits.
    config = GenerationConfig(
        do_sample=False,
        max_new_tokens=output_len,
        min_new_tokens=output_len,
        eos_token_id=[],
        pad_token_id=0,
    )
    attention_mask = torch.ones_like(input_ids)
    out: torch.Tensor = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, generation_config=config
    )
    return out[:, input_ids.shape[1] :]


# Default tolerance for a near tie, in logits. bf16 keeps 8 significant bits,
# so a logit around 16 is stored in steps of 0.0625; two kernels that sum in a
# different order can disagree by a few such steps.
DEFAULT_TIE_TOLERANCE = 0.25

MatchKind = Literal["exact", "near_tie", "mismatch"]


@dataclass(frozen=True, slots=True)
class Divergence:
    """Where one sequence first differs from `generate()`, with the evidence.

    `margin` is logit[reference token] - logit[our token] at that position,
    from a fresh forward pass over the shared prefix (logits upcast to float32). A near tie
    means both tokens are the model's top two candidates and `margin` is
    within tolerance: the two paths rounded differently, not computed wrong.
    """

    sequence: int
    position: int
    ours: int
    reference: int
    margin: float
    top_two: bool


@dataclass(frozen=True, slots=True)
class CorrectnessReport:
    kind: MatchKind
    divergences: tuple[Divergence, ...] = ()

    @property
    def matches(self) -> bool:
        """Exact or near tie: the loop is trusted and may be timed."""
        return self.kind != "mismatch"

    @property
    def first_mismatch(self) -> tuple[int, int] | None:  # (sequence, position)
        if not self.divergences:
            return None
        d = self.divergences[0]
        return d.sequence, d.position


@torch.inference_mode()
def _divergence(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    ours: torch.Tensor,
    reference: torch.Tensor,
    seq: int,
    pos: int,
) -> Divergence:
    # Both paths agree up to `pos`, so the shared prefix is well defined.
    prefix = torch.cat([input_ids[seq], reference[seq, :pos]]).unsqueeze(0)
    logits = model(input_ids=prefix, logits_to_keep=1).logits[0, -1].float()
    a, b = int(ours[seq, pos]), int(reference[seq, pos])
    top_two = set(logits.topk(2).indices.tolist())
    return Divergence(
        sequence=seq,
        position=pos,
        ours=a,
        reference=b,
        margin=float(logits[b] - logits[a]),
        top_two={a, b} == top_two,
    )


def check_correctness(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    output_len: int,
    regression: Regression = "none",
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    decode_mode: DecodeMode = "eager",
) -> CorrectnessReport:
    """Compare our loop with `generate()` token by token.

    Greedy decoding in bf16 can legitimately flip between two near-equal
    candidates when kernels differ (e.g. attention with vs without a mask).
    So each diverging sequence is judged by its logit margin: within
    `tie_tolerance` and among the top two it is a near tie, otherwise a
    mismatch. Tokens after the first divergence are not compared, because
    the two continuations no longer share a prefix.
    """
    ours = greedy_decode(
        model, input_ids, output_len, regression=regression, decode_mode=decode_mode
    )
    reference = reference_generate(model, input_ids, output_len)
    divergences = []
    for seq in range(ours.shape[0]):
        diff = (ours[seq] != reference[seq]).nonzero()
        if diff.numel() > 0:
            divergences.append(_divergence(model, input_ids, ours, reference, seq, int(diff[0, 0])))
    if not divergences:
        return CorrectnessReport(kind="exact")
    tie = all(d.top_two and abs(d.margin) <= tie_tolerance for d in divergences)
    return CorrectnessReport(kind="near_tie" if tie else "mismatch", divergences=tuple(divergences))
