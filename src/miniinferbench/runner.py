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
class _StaticState:
    forwards: dict[DecodeMode, Callable[..., Any]]
    caches: dict[tuple[DecodeMode, int, int], StaticCache]


# Per model: one decode forward per mode, plus one static cache per (mode,
# batch, length), reused across runs. CUDA graphs replay fixed memory
# addresses, so the cache must stay where it was when the graph was recorded.
_STATIC_STATE: weakref.WeakKeyDictionary[PreTrainedModel, _StaticState] = (
    weakref.WeakKeyDictionary()
)

# How each static-cache mode runs a decode step. "compiled" fuses ops into
# fewer kernels but still launches them one by one; "cuda_graph" also captures
# the step as a CUDA graph and replays it with one launch. Comparing
# static -> compiled -> cuda_graph separates the three effects.
_COMPILE_MODES: dict[DecodeMode, str | None] = {
    "static": None,
    "compiled": "default",
    "cuda_graph": "reduce-overhead",
}


def _decode_forward(
    model: PreTrainedModel, mode: DecodeMode, device: torch.device
) -> tuple[_StaticState, Callable[..., Any]]:
    state = _STATIC_STATE.get(model)
    if state is None:
        state = _StaticState(forwards={}, caches={})
        _STATIC_STATE[model] = state
    if mode not in state.forwards:
        compile_mode = _COMPILE_MODES[mode]
        forward: Callable[..., Any] = model.forward
        # Compilation is CUDA-only here; on CPU (tests) every static mode runs
        # the same eager static-cache path.
        if compile_mode is not None and device.type == "cuda":
            forward = torch.compile(model.forward, mode=compile_mode)
        state.forwards[mode] = forward
    return state, state.forwards[mode]


def _static_cache(
    state: _StaticState, model: PreTrainedModel, mode: DecodeMode, batch_size: int, max_len: int
) -> StaticCache:
    key = (mode, batch_size, max_len)
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
    `decode_mode` other than "eager" uses a static KV cache; "compiled" and
    "cuda_graph" also compile the decode step (see `_COMPILE_MODES`). Prefill
    is eager in every mode.
    """
    if output_len < 1:
        raise ValueError(f"output_len must be >= 1, got {output_len}")
    if decode_mode != "eager" and regression == "no_kv_cache":
        raise ValueError("no_kv_cache needs decode_mode 'eager'")
    batch_size, input_len = input_ids.shape
    device = input_ids.device
    static = decode_mode != "eager"
    if static:
        state, decode_forward = _decode_forward(model, decode_mode, device)
        cache: DynamicCache | StaticCache = _static_cache(
            state, model, decode_mode, batch_size, input_len + output_len
        )
        # Built once on the device: creating a position tensor from a Python
        # int every step would be a host-to-device copy per step.
        positions = torch.arange(input_len + output_len, device=device)
    else:
        cache = DynamicCache()
        decode_forward = model.forward

    def static_kwargs(start: int, stop: int) -> dict[str, torch.Tensor]:
        # A static cache needs explicit write positions; a dynamic one appends.
        return {"cache_position": positions[start:stop]} if static else {}

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
            **static_kwargs(0, input_len),
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
                out = decode_forward(
                    input_ids=token,
                    past_key_values=cache,
                    use_cache=True,
                    **static_kwargs(input_len + step, input_len + step + 1),
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
    # generate() fills unset fields from model.generation_config, which comes
    # from the checkpoint's generation_config.json. Qwen2.5 ships
    # repetition_penalty=1.1 there, which changed the reference's tokens.
    # Swap in the clean config so no checkpoint default can alter the logits.
    saved = model.generation_config
    model.generation_config = config
    try:
        out: torch.Tensor = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, generation_config=config
        )
    finally:
        model.generation_config = saved
    return out[:, input_ids.shape[1] :]


# Default tolerance for a near tie, in logits. bf16 keeps 8 significant bits,
# so a logit around 16 is stored in steps of 0.0625; two kernels that sum in a
# different order can disagree by a few such steps.
DEFAULT_TIE_TOLERANCE = 0.25

MatchKind = Literal["exact", "near_tie", "mismatch"]


@dataclass(frozen=True, slots=True)
class Divergence:
    """Where one sequence first differs from `generate()`, with the evidence.

    Logits come from a fresh forward pass over the shared prefix (upcast to
    float32). `margin` is logit[reference token] - logit[our token]; each
    `*_gap` is how far that token's logit is below the best logit. A near tie
    means both tokens are within tolerance of the best: argmax over (near-)
    equal values is implementation-defined, so two correct kernels may pick
    differently. Bf16 produces exact ties too, sometimes among more than two
    tokens, which is why the rule is not limited to the top two.
    """

    sequence: int
    position: int
    ours: int
    reference: int
    margin: float
    ours_gap: float
    reference_gap: float


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
    best = logits.max()
    return Divergence(
        sequence=seq,
        position=pos,
        ours=a,
        reference=b,
        margin=float(logits[b] - logits[a]),
        ours_gap=float(best - logits[a]),
        reference_gap=float(best - logits[b]),
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
    So each diverging sequence is re-scored: if both tokens are within
    `tie_tolerance` of the best logit it is a near tie, otherwise a mismatch.
    Tokens after the first divergence are not compared, because the two
    continuations no longer share a prefix.
    """
    ours = greedy_decode(
        model, input_ids, output_len, regression=regression, decode_mode=decode_mode
    )
    return judge_tokens(model, input_ids, ours, tie_tolerance)


def judge_tokens(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    ours: torch.Tensor,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> CorrectnessReport:
    """Compare `[batch, output_len]` tokens from any backend with `generate()`.

    `model` is the Hugging Face model used for the reference and for the
    logit re-scoring of each divergence (see `check_correctness`).
    """
    ours = ours.to(input_ids.device)
    reference = reference_generate(model, input_ids, ours.shape[1])
    divergences = []
    for seq in range(ours.shape[0]):
        diff = (ours[seq] != reference[seq]).nonzero()
        if diff.numel() > 0:
            divergences.append(_divergence(model, input_ids, ours, reference, seq, int(diff[0, 0])))
    if not divergences:
        return CorrectnessReport(kind="exact")
    tie = all(max(d.ours_gap, d.reference_gap) <= tie_tolerance for d in divergences)
    return CorrectnessReport(kind="near_tie" if tie else "mismatch", divergences=tuple(divergences))
